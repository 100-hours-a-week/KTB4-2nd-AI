"""공공 장소 색인의 생성, 이름 일치 단계, 반경, 내려받기 준비를 작은 원자료로 확인한다."""

import csv
import gzip
import json
import os
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.infra.place_index import (
    PlaceIndex,
    ensure_local_index,
    file_sha256,
    normalize,
    prepare_place_index,
)
from app.jobs.config import JobsSettings
from scripts.build_place_index import build

# 정선 화암동굴 근처
CENTER = (37.3365, 128.7836)


def settings(**fields) -> JobsSettings:
    return JobsSettings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local", **fields)


def write_csv(path: Path, rows: list[dict], *, compress: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def raw(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("raw")
    (root / "tourapi").mkdir()
    (root / "tourapi/tourapi_all.json").write_text(
        json.dumps(
            {
                "items": [
                    {"contentid": "1", "mapy": "37.3366", "mapx": "128.7837", "title": "화암동굴",
                     "contenttypeid": "12"},
                    {"contentid": "2", "mapy": "0", "mapx": "0", "title": "좌표 없음",
                     "contenttypeid": "12"},
                    {"contentid": "1", "mapy": "37.3366", "mapx": "128.7837", "title": "화암동굴",
                     "contenttypeid": "12"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )  # fmt: skip
    write_csv(
        root / "std_tourist/강원특별자치도_정선군_관광지정보.csv",
        [{"관광지명": "화암관광지", "위도": "37.33", "경도": "128.78", "관광지구분": "관광지"}],
    )
    write_csv(
        root / "semas/semas_all.csv.gz",
        [
            {"상가업소번호": "S1", "상호명": "스타벅스", "지점명": "정선점", "위도": "37.3370",
             "경도": "128.7840", "상권업종소분류명": "카페"},
            {"상가업소번호": "S2", "상호명": "카페", "지점명": "", "위도": "37.3368",
             "경도": "128.7838", "상권업종소분류명": "카페"},
            {"상가업소번호": "S3", "상호명": "스타벅스", "지점명": "서울역점", "위도": "37.5547",
             "경도": "126.9707", "상권업종소분류명": "카페"},
        ],
        compress=True,
    )  # fmt: skip
    (root / "wikidata").mkdir()
    (root / "wikidata/wikidata_items.json").write_text(
        json.dumps(
            {"items": [{"qid": "Q1", "lat": 37.34, "lon": 128.79, "label_ko": "정선 아리랑",
                        "label_en": "Jeongseon Arirang", "alias_ko": "아리랑|", "p31_ko": "노래"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )  # fmt: skip
    write_csv(
        root / "overture/overture_kr.csv",
        [{"id": "o1", "lat": "37.3369", "lon": "128.7839", "name_primary": "", "names_common": "",
          "names_rules": "", "brand_primary": "", "basic_category": "cafe"},
         {"id": "o2", "lat": "39.0", "lon": "125.7", "name_primary": "평양역", "names_common": "",
          "names_rules": "", "brand_primary": "", "basic_category": "station"}],
    )  # fmt: skip
    write_csv(
        root / "osm/osm_pois.csv",
        [{"osm_type": "node", "osm_id": "9", "lat": "37.3365", "lon": "128.7836", "tier": "core",
          "main_tag": "tourism=attraction", "name": "화암 동굴", "name_ko": "",
          "name_en": "Hwaam Cave", "name_ko-Latn": "", "alt_name": "", "alt_name_ko": "",
          "old_name": "", "official_name": "",
          "short_name": "", "loc_name": "", "brand": "", "brand_ko": "", "branch": "",
          "other_names": json.dumps({"name:ja": "花岩洞窟"}, ensure_ascii=False)}],
    )  # fmt: skip
    return root


@pytest.fixture(scope="module")
def index_path(raw, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("index") / "places.sqlite"
    meta = build(raw, out)
    assert json.loads(meta["counts"]) == {
        "tourapi": 1, "std_tourist": 1, "semas": 3, "wikidata": 1, "osm": 1,
    }  # fmt: skip
    # 좌표 0, 이름 없음, 대한민국 범위 밖(평양), 같은 원본 ID는 넣지 않는다.
    assert json.loads(meta["dropped"]) == {
        "tourapi:coordinate": 1, "tourapi:duplicate": 1,
        "overture:name": 1, "overture:coordinate": 1,
    }  # fmt: skip
    assert meta["sha256"] == file_sha256(out)
    return out


@pytest.fixture
def index(index_path):
    place_index = PlaceIndex(index_path)
    yield place_index
    place_index.close()


def test_normalize_ignores_spacing_case_and_symbols():
    assert normalize(" 화암 동굴(입구) ") == normalize("화암동굴입구") == "화암동굴입구"
    assert normalize("Ｃａｆé-Ｂ") == "caféb"
    assert normalize(None) == ""


def test_exact_names_rank_before_partial_matches_and_carry_source_ids(index):
    hits = index.search("화암동굴", *CENTER, 1000)
    # 띄어쓴 OSM 이름도 정확 일치다. 같은 점수면 가까운 순이다.
    assert [h.candidate_id for h in hits] == ["kb:osm:node/9", "kb:tourapi:1"]
    assert hits[0].name == "화암 동굴" and hits[0].category == "tourism=attraction"
    assert hits[0].distance_m == 0


def test_two_character_query_matches_names_containing_it_within_radius(index):
    hits = index.search("카페", *CENTER, 1000)
    assert [h.candidate_id for h in hits] == ["kb:semas:S2"]


def test_name_inside_longer_query_is_the_lowest_tier(index):
    hits = index.search("스타벅스정선점앞", *CENTER, 1000)
    assert [h.candidate_id for h in hits] == ["kb:semas:S1"]
    # 질의를 포함한 이름(2단계)이 질의 안의 이름(1단계)보다 앞선다.
    hits = index.search("스타벅스", *CENTER, 1000)
    assert [h.candidate_id for h in hits] == ["kb:semas:S1"]


def test_radius_excludes_far_places_and_is_capped(index):
    assert index.search("스타벅스", *CENTER, 10) == []
    # 서울역 지점은 30 km 상한 밖이라 반경을 크게 줘도 나오지 않는다.
    assert all(h.candidate_id != "kb:semas:S3" for h in index.search("스타벅스", *CENTER, 10**6))


def test_alias_and_other_language_names_are_searchable(index):
    assert [h.candidate_id for h in index.search("Hwaam cave", *CENTER, 1000)] == ["kb:osm:node/9"]
    assert [h.candidate_id for h in index.search("花岩洞窟", *CENTER, 1000)] == ["kb:osm:node/9"]
    assert [h.candidate_id for h in index.search("아리랑", *CENTER, 2000)] == ["kb:wikidata:Q1"]


@pytest.mark.parametrize("query", ["", "  ", "!!", '"OR"'])
def test_empty_or_symbol_queries_do_not_break_full_text_search(index, query):
    assert index.search(query, *CENTER, 1000) == []


def test_three_character_query_inside_longer_names_uses_full_text_index(index):
    # "스타벅"은 정확 일치가 없고 "스타벅스", "스타벅스정선점"에 포함된다(2단계, trigram 색인).
    hits = index.search("스타벅", *CENTER, 1000)
    assert [h.candidate_id for h in hits] == ["kb:semas:S1"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_coordinates_or_radius_are_rejected(index, bad):
    with pytest.raises(ValueError):
        index.search("스타벅스", *CENTER, bad)
    with pytest.raises(ValueError):
        index.search("스타벅스", bad, CENTER[1], 1000)


def test_index_rejects_missing_file_or_other_schema(tmp_path, index_path):
    with pytest.raises(FileNotFoundError):
        PlaceIndex(tmp_path / "none.sqlite")
    other = tmp_path / "old schema.sqlite"
    other.write_bytes(index_path.read_bytes())
    with sqlite3.connect(other) as db:
        db.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    with pytest.raises(ValueError, match="스키마"):
        PlaceIndex(other)


def test_ensure_local_index_downloads_once_and_checks_hash(tmp_path, index_path):
    sha = file_sha256(index_path)
    calls = []

    def download(target):
        calls.append(target)
        target.write_bytes(index_path.read_bytes())

    target = tmp_path / "cache/places.sqlite"
    assert ensure_local_index(target, sha, download) == target
    assert ensure_local_index(target, sha, download) == target
    assert len(calls) == 1 and file_sha256(target) == sha

    # 표지를 남긴 채 파일이 바뀌면 크기나 수정 시각이 달라 다시 받는다.
    target.write_bytes(b"changed")
    os.utime(target, ns=(1, 1))
    ensure_local_index(target, sha, download)
    assert len(calls) == 2 and file_sha256(target) == sha

    def broken(target):
        target.write_bytes(b"not sqlite")

    other = tmp_path / "other/places.sqlite"
    with pytest.raises(ValueError, match="해시"):
        ensure_local_index(other, sha, broken)
    assert not other.exists() and not other.with_name("places.sqlite.partial").exists()


def test_prepare_place_index_uses_settings_and_opens_the_file(tmp_path, index_path):
    assert prepare_place_index(settings()) is None
    # 환경 변수 예시처럼 빈 값으로 둔 항목은 쓰지 않는다.
    empty = settings(PLACE_INDEX_PATH="", PLACE_INDEX_S3_URI="", PLACE_INDEX_SHA256="")
    assert prepare_place_index(empty) is None
    assert prepare_place_index(settings(PLACE_INDEX_PATH=str(index_path))) == index_path
    with pytest.raises(FileNotFoundError):
        prepare_place_index(settings(PLACE_INDEX_PATH=str(tmp_path / "none.sqlite")))
    corrupted = tmp_path / "corrupted.sqlite"
    corrupted.write_bytes(b"not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        prepare_place_index(settings(PLACE_INDEX_PATH=str(corrupted)))


@pytest.mark.parametrize(
    "fields",
    [
        {"PLACE_INDEX_S3_URI": "s3://bucket/places.sqlite"},
        {"PLACE_INDEX_S3_URI": "s3://bucket/places.sqlite", "PLACE_INDEX_PATH": "/tmp/p"},
        {"PLACE_INDEX_SHA256": "a" * 64, "PLACE_INDEX_PATH": "/tmp/p"},
        {"PLACE_INDEX_S3_URI": "https://bucket/places.sqlite", "PLACE_INDEX_SHA256": "a" * 64},
        {"KAKAO_CALLS_PER_JOB": -1},
    ],
)
def test_invalid_place_settings_are_rejected(fields):
    with pytest.raises(ValidationError):
        settings(**fields)


def test_two_character_query_radius_is_capped_at_five_kilometers(index):
    # "카페" 이름을 가진 상가는 중심에서 약 40 m라 5 km 안이다. 반경을 넓혀도 상한까지만 찾는다.
    assert [h.candidate_id for h in index.search("카페", *CENTER, 30_000)] == ["kb:semas:S2"]
    far = (CENTER[0] + 0.06, CENTER[1])  # 약 6.7 km 북쪽
    assert index.search("카페", *far, 30_000) == []
    assert [h.candidate_id for h in index.search("스타벅스", *far, 30_000)] == ["kb:semas:S1"]
