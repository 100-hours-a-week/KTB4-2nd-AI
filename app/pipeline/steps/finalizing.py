"""FINALIZING: 처리 상태를 검증하고 외부 응답으로 조립한다."""

from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime

import numpy as np
from pydantic import ValidationError

from app.core.errors import PipelineError
from app.pipeline.state import PhotoState, TimeStatus
from app.pipeline.steps.represent import quality_score, select_representative
from app.schemas.process import (
    ClockOffset,
    Issue,
    Place,
    PlaceAttachment,
    ProcessResult,
    RegionOrigin,
    UnclassifiedAttachment,
)


def _order(photo: PhotoState) -> tuple[bool, datetime, int]:
    time = photo.taken_at
    return (
        time is None,
        time.astimezone(UTC) if time is not None else datetime.max.replace(tzinfo=UTC),
        photo.source.trip_attachment_id,
    )


def _validate(photo: PhotoState) -> None:
    time = photo.taken_at
    if (
        photo.time_status is TimeStatus.PENDING
        or (photo.time_status is TimeStatus.UNKNOWN) != (time is None)
        or (time is not None and time.utcoffset() is None)
    ):
        raise PipelineError("FINALIZING 처리 시각이 일관되지 않음")
    for value, limit in ((photo.latitude, 90), (photo.longitude, 180)):
        if value is not None and (not np.isfinite(value) or not -limit <= value <= limit):
            raise PipelineError("FINALIZING 대상 좌표가 유효하지 않음")
    if photo.issue is Issue.UNCLEAR_LOCATION:
        if photo.place_id is not None:
            raise PipelineError("위치 미분류 사진에 장소가 지정됨")
    elif (
        not photo.place_id
        or photo.latitude is None
        or photo.longitude is None
        or photo.region_origin is RegionOrigin.UNKNOWN
    ):
        raise PipelineError("장소 사진의 좌표와 출처 및 place_id가 필요함")
    if photo.issue is not Issue.DUPLICATED and photo.duplicate_of_attachment_id is not None:
        raise PipelineError("중복이 아닌 사진에 중복 원본 ID가 지정됨")


def _center(photos: list[PhotoState]) -> tuple[float, float]:
    # 제외 전 장소 구성원으로 구면 평균을 구해 날짜 변경선과 필터링의 영향을 피한다.
    latitudes = np.radians([photo.latitude for photo in photos])
    longitudes = np.radians([photo.longitude for photo in photos])
    x = np.mean(np.cos(latitudes) * np.cos(longitudes))
    y = np.mean(np.cos(latitudes) * np.sin(longitudes))
    z = np.mean(np.sin(latitudes))
    if np.linalg.norm([x, y, z]) < 1e-12:
        raise PipelineError("장소 기준 좌표를 계산할 수 없음")
    return float(np.degrees(np.arctan2(z, np.hypot(x, y)))), float(np.degrees(np.arctan2(y, x)))


def finalize_photos(
    photos: list[PhotoState],
    variances: Mapping[int, float],
    clock_offsets: list[ClockOffset],
) -> ProcessResult:
    """원본과 처리 상태를 수정하지 않고 ProcessResult를 반환한다.

    장소에는 대표 후보가 최소 한 장 있어야 한다. 모두 제외됐다면 입력 가정 위반으로
    실패하며 흐림 사진을 자동 복원하지 않는다. 장소 좌표는 제외 전 구성원의 구면 평균,
    시작과 종료 시각은 폴더에 남는 사진의 확정 시각 범위다. v1 매칭 ID는 항상 None이다.
    """
    by_id = {photo.source.trip_attachment_id: photo for photo in photos}
    if len(by_id) != len(photos):
        raise PipelineError("FINALIZING 사진 ID가 중복됨")
    groups: dict[str, list[PhotoState]] = defaultdict(list)
    fields: dict[int, dict] = {}
    for photo in photos:
        _validate(photo)
        photo_id = photo.source.trip_attachment_id
        if photo.issue is not Issue.UNCLEAR_LOCATION and photo_id not in variances:
            raise PipelineError(f"품질 점수 계산에 필요한 분산이 없음: {photo_id}")
        score = quality_score(variances[photo_id]) if photo_id in variances else None
        fields[photo_id] = dict(
            trip_attachment_id=photo_id,
            taken_at=photo.taken_at,
            latitude=photo.latitude,
            longitude=photo.longitude,
            region_origin=photo.region_origin,
            evaluation=score,
        )
        if photo.issue is Issue.DUPLICATED:
            original = by_id.get(photo.duplicate_of_attachment_id)
            if (
                original is None
                or original.issue is not None
                or original.place_id != photo.place_id
            ):
                raise PipelineError("중복 원본은 같은 장소에 보존된 사진이어야 함")
        if photo.place_id is not None:
            groups[photo.place_id].append(photo)

    places = []
    try:
        for place_id, members in sorted(groups.items()):
            members.sort(key=_order)
            candidates = [photo for photo in members if photo.issue is None]
            representative_id = select_representative(candidates)
            latitude, longitude = _center(members)
            times = [photo.taken_at for photo in candidates if photo.taken_at is not None]
            places.append(
                Place(
                    place_id=place_id,
                    latitude=latitude,
                    longitude=longitude,
                    first_taken_at=min(times) if times else None,
                    last_taken_at=max(times) if times else None,
                    representative_attachment_id=representative_id,
                    attachments=[
                        PlaceAttachment(**fields[p.source.trip_attachment_id]) for p in candidates
                    ],
                )
            )
        unclassified = [
            UnclassifiedAttachment(
                **fields[photo.source.trip_attachment_id],
                issue=photo.issue,
                place_id=photo.place_id,
                duplicate_of_attachment_id=photo.duplicate_of_attachment_id,
            )
            for photo in sorted(photos, key=lambda p: p.source.trip_attachment_id)
            if photo.issue is not None
        ]
        return ProcessResult(
            places=places, unclassified=unclassified, failed=[], clock_offsets=clock_offsets
        )
    except ValidationError as exc:
        raise PipelineError("사진 정리 결과가 응답 계약과 일치하지 않음") from exc
