"""CLUSTERING 뒤: 위치를 못 찾은 GPS 없는 사진을 촬영 시각과 닮은 사진으로 폴더에 넣는다(#152).

두 규칙을 바뀌는 사진이 없을 때까지 번갈아 적용한다. 평가 도구 B1(2026-10-27 b1.py)과 같은 순서다.
시각 끼워 넣기  폴더가 있는 사진을 촬영 시각순으로 놓고, 대상의 바로 앞과 뒤가 같은 폴더이며
               둘 다 interpolate_max_gap_min 이하이면 그 폴더로 넣는다.
닮은 사진 따라가기  코사인이 inherit_min_sim 이상인 폴더 사진(앞서 넣은 사진 포함)의 폴더가
               하나뿐이면 그 폴더로 넣는다.
넣은 사진은 그 폴더 GPS 사진 가운데 가장 닮은 사진의 좌표를 받는다(결정 150).
"""

from datetime import timedelta

import numpy as np

from app.core.errors import PipelineError
from app.pipeline.state import PhotoState
from app.schemas.process import KST, Issue, Period, RegionOrigin


def _in_period(photo: PhotoState, period: Period) -> bool:
    if photo.taken_at is None:
        return True
    return period.start_date <= photo.taken_at.astimezone(KST).date() <= period.end_date


def assign_unlocated(
    photos: list[PhotoState],
    period: Period,
    *,
    inherit_min_sim: float,
    interpolate_max_gap_min: float,
) -> int:
    """LOCATING과 CLUSTERING을 마친 상태를 받아 넣은 사진 수를 돌려준다.

    대상은 GPS가 없고 UNCLEAR_LOCATION인 사진 가운데 여행 기간 안의 사진이다.
    기간 밖 사진은 LOCATING이 분리한 그대로 둔다. 사진 ID 순으로 처리해 결과를 고정한다.
    """
    if not np.isfinite(inherit_min_sim) or not -1 <= inherit_min_sim <= 1:
        raise PipelineError("inherit_min_sim은 -1 이상 1 이하의 유한값이어야 함")
    if not np.isfinite(interpolate_max_gap_min) or interpolate_max_gap_min < 0:
        raise PipelineError("interpolate_max_gap_min은 0 이상의 유한값이어야 함")
    ordered = sorted(photos, key=lambda photo: photo.source.trip_attachment_id)
    targets = [
        photo
        for photo in ordered
        if photo.issue is Issue.UNCLEAR_LOCATION
        and not photo.source.has_gps
        and _in_period(photo, period)
    ]
    if not targets:
        return 0
    target_ids = {photo.source.trip_attachment_id for photo in targets}
    if any(
        photo.embedding is None
        for photo in ordered
        if photo.place_id or photo.source.trip_attachment_id in target_ids
    ):
        raise PipelineError("규칙 배정에는 임베딩이 필요함")
    gap = timedelta(minutes=interpolate_max_gap_min)
    assigned = 0
    changed = True
    while changed:
        changed = False
        # 시각 끼워 넣기: 이번 차례 시작 때 폴더가 있는 사진으로 시각 순서를 만든다.
        track = sorted(
            ((photo.taken_at, photo.source.trip_attachment_id, photo.place_id)
             for photo in ordered if photo.place_id and photo.taken_at is not None),
        )  # fmt: skip
        for photo in targets:
            if photo.place_id or photo.taken_at is None:
                continue
            t = photo.taken_at
            before = [entry for entry in track if entry[0] <= t]
            after = [entry for entry in track if entry[0] >= t]
            if (
                before
                and after
                and before[-1][2] == after[0][2]
                and t - before[-1][0] <= gap
                and after[0][0] - t <= gap
            ):
                _place(photo, before[-1][2], ordered)
                assigned += 1
                changed = True
        # 닮은 사진 따라가기: 앞에서 넣은 사진도 바로 기준이 된다.
        for photo in targets:
            if photo.place_id:
                continue
            folders = {
                other.place_id
                for other in ordered
                if other is not photo
                and other.place_id
                and float(other.embedding @ photo.embedding) >= inherit_min_sim
            }
            if len(folders) == 1:
                _place(photo, folders.pop(), ordered)
                assigned += 1
                changed = True
    return assigned


def _place(photo: PhotoState, place_id: str, photos: list[PhotoState]) -> None:
    anchors = [
        other
        for other in photos
        if other.place_id == place_id and other.region_origin is RegionOrigin.EXIF
    ]
    if not anchors:
        raise PipelineError(f"폴더 {place_id}에 GPS 사진이 없음")
    # 동점이면 작은 사진 ID(photos가 사진 ID 순)를 고른다.
    nearest = max(anchors, key=lambda other: float(other.embedding @ photo.embedding))
    photo.place_id = place_id
    photo.latitude, photo.longitude = nearest.latitude, nearest.longitude
    photo.region_origin = RegionOrigin.INFERRED
    photo.issue = None
