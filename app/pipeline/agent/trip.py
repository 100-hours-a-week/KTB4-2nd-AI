"""정리 에이전트가 보는 여행 상태. 사진은 PhotoState를 그대로 쓰고, 에이전트가 만든 장소와
폴더 묶기 결과만 따로 둔다.

모델에게 보이는 사진 ID는 "S<사진 ID>", 장소 ID는 CLUSTERING의 "p<번호>"와 새 장소 "n<번호>"다.
평가 도구(2026-10-28 state.py)의 Trip을 서비스 상태에 맞춰 옮겼다. 앞뒤 사진은 장소 유무와
관계없이 촬영 시각이 바로 앞과 뒤인 사진이다(결정 148).
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from app.core.errors import PipelineError
from app.infra.place_index import haversine_m
from app.pipeline.state import PhotoState
from app.schemas.process import KST, Issue, Period, ProcessRequest, RegionOrigin


@dataclass(frozen=True)
class NewPlace:
    name: str
    latitude: float
    longitude: float
    candidate_id: str


def region_key(sido: str, sigungu: str) -> tuple[str, str]:
    """시도와 시·군. 일반구가 있는 시의 "성남시 분당구"는 "성남시"로 비교한다.

    BE의 여행 지역 목록(data/regions.json)이 시도와 시·군 단위이기 때문이다.
    """
    return sido.strip(), (sigungu.split() or [""])[0]


def _in_period(photo: PhotoState, period: Period) -> bool:
    if photo.taken_at is None:
        return True
    return period.start_date <= photo.taken_at.astimezone(KST).date() <= period.end_date


def photo_ref(photo: PhotoState) -> str:
    return f"S{photo.source.trip_attachment_id}"


class TripView:
    def __init__(self, photos: list[PhotoState], request: ProcessRequest, regions: list[str]):
        # LOCATING이 기간 밖으로 분리한 사진은 여행 사진으로 보지 않는다.
        self.photos = sorted(
            (photo for photo in photos if _in_period(photo, request.period)),
            key=lambda photo: photo.source.trip_attachment_id,
        )
        self.by_ref = {photo_ref(photo): photo for photo in self.photos}
        self.info = {
            "name": request.trip_name,
            "period": f"{request.period.start_date} ~ {request.period.end_date}",
            "regions": regions,
        }
        first = request.regions[0]
        self.fallback_center = (first.latitude, first.longitude)
        self.new_places: dict[str, NewPlace] = {}
        self.merged: dict[str, str] = {}

    # ---------- 장소
    def canon(self, place: str) -> str:
        while place in self.merged:
            place = self.merged[place]
        return place

    def place_of(self, photo: PhotoState) -> str | None:
        return self.canon(photo.place_id) if photo.place_id else None

    def places(self) -> dict[str, list[PhotoState]]:
        """지금 장소별 사진(묶기 반영)"""
        out: dict[str, list[PhotoState]] = defaultdict(list)
        for photo in self.photos:
            if place := self.place_of(photo):
                out[place].append(photo)
        return dict(out)

    def center(self, place: str) -> tuple[float, float] | None:
        """GPS 사진의 평균 좌표, 없으면 새 장소의 후보 좌표"""
        members = self.places().get(place, [])
        gps = [p for p in members if p.region_origin is RegionOrigin.EXIF]
        if gps:
            return (
                float(np.mean([p.latitude for p in gps])),
                float(np.mean([p.longitude for p in gps])),
            )
        if place in self.new_places:
            new = self.new_places[place]
            return new.latitude, new.longitude
        return None

    def summary(self) -> list[dict]:
        rows = []
        for place, members in sorted(self.places().items(), key=lambda kv: _place_order(kv[0])):
            center = self.center(place)
            times = sorted(p.taken_at for p in members if p.taken_at is not None)
            row = {
                "place_id": place,
                "kind": "새 장소" if place in self.new_places else "기존 장소",
                "photo_count": len(members),
                "gps_photo_count": sum(p.region_origin is RegionOrigin.EXIF for p in members),
                "center": [round(center[0], 5), round(center[1], 5)] if center else None,
                "time_range": [_clock(times[0]), _clock(times[-1])] if times else None,
            }
            if place in self.new_places:
                row["name"] = self.new_places[place].name
            merged = sorted(k for k in self.merged if self.canon(k) == place)
            if merged:
                row["merged_from"] = merged
            rows.append(row)
        return rows

    def relations(self) -> list[dict]:
        """폴더 쌍의 중심 거리와 촬영 시각 범위 사이 간격(분, 겹치면 0). 묶기 단계 입력(결정 128)"""
        spans = {}
        for place, members in self.places().items():
            times = [p.taken_at for p in members if p.taken_at is not None]
            spans[place] = (min(times), max(times)) if times else None
        rows = sorted(self.places(), key=_place_order)
        out = []
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                ca, cb = self.center(a), self.center(b)
                if ca is None or cb is None:
                    continue
                gap = None
                if spans[a] and spans[b]:
                    (a0, a1), (b0, b1) = spans[a], spans[b]
                    if a0 <= b1 and b0 <= a1:
                        gap = 0
                    else:
                        seconds = min(
                            abs((b0 - a1).total_seconds()), abs((a0 - b1).total_seconds())
                        )
                        gap = round(seconds / 60)
                distance = round(haversine_m(*ca, *cb))
                out.append({"a": a, "b": b, "center_distance_m": distance, "time_gap_min": gap})
        return out

    # ---------- 사진
    def targets(self) -> list[PhotoState]:
        """장소가 없는 GPS 없는 사진. 기간 밖 사진은 self.photos에 없다."""
        return [
            photo
            for photo in self.photos
            if photo.place_id is None
            and photo.issue is Issue.UNCLEAR_LOCATION
            and not photo.source.has_gps
        ]

    def neighbors(self, photo: PhotoState) -> tuple[PhotoState | None, PhotoState | None]:
        """촬영 시각이 바로 앞과 바로 뒤인 사진. 장소가 없는 사진도 포함한다(결정 148)."""
        if photo.taken_at is None:
            return None, None
        # 같은 시각이면 사진 ID 순으로 놓아, 한 사진이 앞과 뒤에 함께 오지 않게 한다.
        track = sorted(
            (p for p in self.photos if p.taken_at is not None),
            key=lambda p: (p.taken_at, p.source.trip_attachment_id),
        )
        index = next(i for i, p in enumerate(track) if p is photo)
        before = track[index - 1] if index > 0 else None
        after = track[index + 1] if index + 1 < len(track) else None
        return before, after

    def best_per_place(self, photo: PhotoState) -> list[tuple[str, float, PhotoState]]:
        """장소마다 대상과 코사인 유사도가 가장 높은 사진, 유사도 내림차순"""
        out = []
        for place, members in self.places().items():
            scored = [
                (float(other.embedding @ photo.embedding), other)
                for other in members
                if other is not photo
            ]
            if scored:
                sim, best = max(
                    scored, key=lambda item: (item[0], -item[1].source.trip_attachment_id)
                )
                out.append((place, sim, best))
        return sorted(out, key=lambda item: (-item[1], _place_order(item[0])))

    def trip_center(self) -> tuple[float, float]:
        gps = [p for p in self.photos if p.region_origin is RegionOrigin.EXIF]
        if not gps:
            return self.fallback_center
        return float(np.mean([p.latitude for p in gps])), float(np.mean([p.longitude for p in gps]))

    def anchor(self, near) -> tuple[float, float]:
        """검색 중심: 사진 ID, 장소 ID, 그 밖에는 여행 전체 중심. near는 모델이 준 값 그대로"""
        near = near.strip() if isinstance(near, str) else ""
        photo = self.by_ref.get(near)
        if photo is not None and photo.latitude is not None and photo.longitude is not None:
            return photo.latitude, photo.longitude
        if near in self.places() or near in self.merged:
            if center := self.center(self.canon(near)):
                return center
        return self.trip_center()

    # ---------- 반영
    def assign(self, photo: PhotoState, place: str) -> None:
        """기존 장소는 그 폴더에서 가장 닮은 GPS 사진의 좌표, 새 장소는 후보 좌표(결정 150)"""
        place = self.canon(place)
        anchors = [p for p in self.places().get(place, []) if p.region_origin is RegionOrigin.EXIF]
        if anchors:
            nearest = max(
                anchors,
                key=lambda p: (float(p.embedding @ photo.embedding), -p.source.trip_attachment_id),
            )
            latitude, longitude = nearest.latitude, nearest.longitude
        elif place in self.new_places:
            new = self.new_places[place]
            latitude, longitude = new.latitude, new.longitude
        else:
            raise PipelineError(f"폴더 {place}에 좌표를 줄 사진이 없음")
        photo.place_id = place
        photo.latitude, photo.longitude = latitude, longitude
        photo.region_origin = RegionOrigin.INFERRED
        photo.issue = None

    def add_new_place(self, name: str, latitude: float, longitude: float, candidate_id: str) -> str:
        """같은 후보로 앞 차례에 만든 새 장소가 있으면 그 장소(묶기 반영)를 쓴다."""
        for place, new in self.new_places.items():
            if new.candidate_id == candidate_id:
                return self.canon(place)
        place = f"n{len(self.new_places) + 1}"
        self.new_places[place] = NewPlace(name, latitude, longitude, candidate_id)
        return place

    def finish(self) -> None:
        """묶기를 사진에 반영하고 장소 ID를 p1부터 다시 매긴다.

        기존 장소는 원래 번호 순, 새 장소는 만든 순서로 그 뒤에 둔다. 묶기와 새 장소가 없으면
        CLUSTERING의 번호가 그대로 남는다.
        """
        for photo in self.photos:
            if photo.place_id:
                photo.place_id = self.canon(photo.place_id)
        used = {photo.place_id for photo in self.photos if photo.place_id}
        order = sorted(used, key=lambda place: (place in self.new_places, _place_order(place)[1]))
        mapping = {place: f"p{index + 1}" for index, place in enumerate(order)}
        for photo in self.photos:
            if photo.place_id:
                photo.place_id = mapping[photo.place_id]

    def time_text(self, photo: PhotoState) -> str:
        return _minute(photo.taken_at) if photo.taken_at is not None else "시각 없음"


def release_unshowable_new_places(photos: list[PhotoState]) -> list[int]:
    """흐림·중복 판정 뒤 보일 사진이 하나도 없는 새 장소를 풀고, 그 사진의 ID를 돌려준다(결정 166).

    FILTERING은 폴더마다 흐리지 않은 첫 사진을 보존하므로, 지금 구조에서 풀리는 폴더는 사진이 모두
    흐린 폴더뿐이다. 장소 ID는 다시 매기지 않으므로 번호에 빈칸이 생길 수 있다.

    FINALIZING은 폴더마다 흐림·중복이 아닌 대표 후보가 있어야 하므로, 에이전트가 흐린 사진만으로
    만든 새 장소가 있으면 작업 전체가 실패한다. GPS 사진이 없는 폴더(에이전트가 만든 새 장소)만
    대상이며, 그 사진은 위치 불명의 흐림·중복 사진으로 되돌려 미분류로 남긴다.
    GPS 사진이 있는 폴더는 바꾸지 않는다.
    """
    groups: dict[str, list[PhotoState]] = defaultdict(list)
    for photo in photos:
        if photo.place_id:
            groups[photo.place_id].append(photo)
    released = []
    for members in groups.values():
        if any(p.region_origin is RegionOrigin.EXIF or p.issue is None for p in members):
            continue
        for photo in members:
            photo.place_id = None
            photo.latitude = photo.longitude = None
            photo.region_origin = RegionOrigin.UNKNOWN
            released.append(photo.source.trip_attachment_id)
    return sorted(released)


def _place_order(place: str) -> tuple[str, int]:
    digits = place[1:]
    return place[:1], int(digits) if digits.isdigit() else 0


def _minute(value: datetime) -> str:
    return value.astimezone(KST).strftime("%Y-%m-%d %H:%M")


def _clock(value: datetime) -> str:
    return value.astimezone(KST).strftime("%m-%d %H:%M")
