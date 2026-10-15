"""질의 해석 지명을 BE 여행 지역 카탈로그 이름으로 변환

BE는 region_names와 여행 지역 이름이 정확히 같은 여행만 후보로 남김, 카탈로그에 없는 지명
(건물, 상호, 장소 종류)이 그대로 가면 후보가 0개라 검색 결과가 비므로 카탈로그에 맞는 이름만 남기고
나머지는 버려 사진 내용 검색(visual_query)에 맡김
좁게 거르면 상위 지역으로 등록한 여행이 빠지므로 애매하면 넓게 넘김
region_catalog.json은 BE 저장소 data/regions.json의 regionName 사본(BE a1c4ba0),
BE 카탈로그 변경 시 함께 갱신
"""

import json
from collections import defaultdict
from functools import cache
from pathlib import Path

CATALOG_PATH = Path(__file__).with_name("region_catalog.json")

# 카탈로그 도 이름과 앞부분이 다른 약칭과 옛 이름
ALIASES = {
    "경북": ("경상북도",),
    "경남": ("경상남도",),
    "충북": ("충청북도",),
    "충남": ("충청남도",),
    "전북": ("전북특별자치도",),
    "전라북도": ("전북특별자치도",),
    "전라남도": ("전남광주통합특별시",),
    "전남": ("전남광주통합특별시",),
    "광주광역시": ("전남광주통합특별시",),
    "전라도": ("전북특별자치도", "전남광주통합특별시"),
}
# 부산시, 제주도, 강원도, 해운대구처럼 행정 단위만 붙인 표현
UNIT_SUFFIXES = ("시", "도", "군", "구")
# 카탈로그에 없는 특별시와 광역시의 자치구, 상위 시로 넘김
# 광주는 전남광주통합특별시, 인천은 2026-07 개편 전후 이름 모두
DISTRICTS = {
    "서울특별시": (
        "종로", "중", "용산", "성동", "광진", "동대문", "중랑", "성북", "강북", "도봉",
        "노원", "은평", "서대문", "마포", "양천", "강서", "구로", "금천", "영등포", "동작",
        "관악", "서초", "강남", "송파", "강동",
    ),
    "부산광역시": (
        "중", "서", "동", "영도", "부산진", "동래", "남", "북", "해운대", "사하", "금정",
        "강서", "연제", "수영", "사상",
    ),
    "대구광역시": ("중", "동", "서", "남", "북", "수성", "달서"),
    "인천광역시": (
        "중", "동", "서", "미추홀", "연수", "남동", "부평", "계양", "제물포", "영종", "검단",
        "서해",
    ),
    "전남광주통합특별시": ("동", "서", "남", "북", "광산"),
    "대전광역시": ("동", "중", "서", "유성", "대덕"),
    "울산광역시": ("중", "남", "동", "북"),
}  # fmt: skip
# 구 없이 쓰면 다른 뜻과 겹치는 이름, 구를 붙였을 때만 인정
DISTRICT_ONLY_WITH_SUFFIX = {"서해", "수영"}


@cache
def catalog() -> tuple[str, ...]:
    return tuple(json.loads(CATALOG_PATH.read_text(encoding="utf-8")))


@cache
def _districts() -> dict[str, tuple[str, ...]]:
    """자치구 이름(해운대구, 해운대) → 상위 시 목록, 중구처럼 여러 시에 있으면 모두"""
    cities = defaultdict(list)
    for city, bases in DISTRICTS.items():
        for base in bases:
            keys = [base + "구"]
            if len(base) >= 2 and base not in DISTRICT_ONLY_WITH_SUFFIX:
                keys.append(base)
            for key in keys:
                cities[key].append(city)
    return {key: tuple(names) for key, names in cities.items()}


def to_catalog_names(names: list[str]) -> list[str] | None:
    """카탈로그 이름 목록으로 변환, 하나도 맞지 않으면 None(BE 지역 필터 미적용)"""
    matched = []
    for name in names:
        found = _matches(name)
        if not found and " " in name.strip():
            # 부산 기장, 서울 강남처럼 띄어 쓴 이름은 단어별로
            found = [region for part in name.split() for region in _matches(part)]
        matched += found
    # 부산광역시 기장군이 맞으면 부산광역시도, 상위 지역으로만 등록한 여행이 빠지지 않게
    matched += [parent for region in matched if (parent := _parent(region))]
    return list(dict.fromkeys(matched)) or None


def _parent(region: str) -> str | None:
    head, _, rest = region.partition(" ")
    return head if rest and head in catalog() else None


def _matches(name: str) -> list[str]:
    key = name.replace(" ", "")
    if key in ALIASES:
        return [region for region in catalog() if region.split(" ")[0] in ALIASES[key]]
    # 전체 이름이나 단위가 같은 이름이 있으면 그것만, 광주시 → 경기도 광주시
    exact = [
        region for region in catalog() if key == region.replace(" ", "") or key in region.split(" ")
    ]
    if exact:
        return exact
    keys = [k for k in {key, key[:-1] if key.endswith(UNIT_SUFFIXES) else key} if len(k) >= 2]
    found = [region for region in catalog() if any(_contains(region, k) for k in keys)]
    if found:
        return found
    # 해운대구, 영종도처럼 단위를 붙인 자치구 이름도
    for k in (key, *keys):
        if cities := _districts().get(k):
            return [region for region in catalog() if region in cities]
    return []


def _contains(region: str, key: str) -> bool:
    """단위 이름의 앞부분 일치, 또는 가장 작은 단위 안에 포함

    부산 → 부산광역시와 부산광역시 기장군, 성남 → 경기도 성남시
    광주 → 전남광주통합특별시와 경기도 광주시, 통합특별시 아래 목포시 등은 제외
    """
    units = region.split(" ")
    return any(unit.startswith(key) for unit in units) or key in units[-1]
