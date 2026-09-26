"""CLOCK: 원본 시각으로만 기기별 날짜 보정을 추정한다."""

from collections import defaultdict
from datetime import UTC, timedelta

import numpy as np

from app.core.errors import PipelineError
from app.pipeline.state import PhotoState, TimeStatus
from app.schemas.process import ClockOffset

SECONDS_PER_DAY = 86_400


def _vectors(photos: list[PhotoState]) -> np.ndarray:
    vectors = [photo.embedding for photo in photos]
    if any(
        not isinstance(vector, np.ndarray)
        or vector.ndim != 1
        or vector.size == 0
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-5, atol=1e-6)
        for vector in vectors
    ):
        raise PipelineError("CLOCK 대응 사진에 유효한 단위 임베딩이 필요함")
    if len({vector.shape for vector in vectors}) != 1:
        raise PipelineError("CLOCK 대응 사진의 임베딩 차원이 다름")
    return np.stack(vectors)


def _pair_offsets(
    targets: list[PhotoState], anchors: list[PhotoState], min_sim: float
) -> list[float]:
    vectors = _vectors(targets + anchors)
    similarities = np.clip(vectors[: len(targets)] @ vectors[len(targets) :].T, -1.0, 1.0)
    offsets = []
    for index, row in enumerate(similarities):
        best = int(np.argmax(row))
        if row[best] < min_sim or np.count_nonzero(row == row[best]) != 1:
            continue
        column = similarities[:, best]
        if int(np.argmax(column)) != index or np.count_nonzero(column == column[index]) != 1:
            continue
        # 서로의 유일한 최고 후보인 쌍만 사용하므로 사진을 중복해서 세지 않는다.
        target_time = targets[index].source.taken_at.astimezone(UTC)
        reference_time = anchors[best].source.taken_at.astimezone(UTC)
        offsets.append((reference_time - target_time).total_seconds())
    return offsets


def _consistent_days(
    offsets: list[float], min_pairs: int, tolerance_sec: float
) -> tuple[int, int] | None:
    groups: dict[int, list[float]] = defaultdict(list)
    for seconds in offsets:
        days = round(seconds / SECONDS_PER_DAY)
        # 반일 차이는 두 날짜 보정값에 똑같이 가까우므로 근거로 쓰지 않는다.
        if abs(seconds - days * SECONDS_PER_DAY) == SECONDS_PER_DAY / 2:
            continue
        groups[days].append(seconds)
    candidates = []
    for days, values in groups.items():
        center = np.median(values)
        count = sum(abs(value - center) <= tolerance_sec for value in values)
        if count >= min_pairs:
            candidates.append((days, int(count)))
    # 두 날짜 보정값이 각각 최소 근거 수를 충족하면 다수 쪽도 자동 채택하지 않는다.
    return candidates[0] if len(candidates) == 1 else None


def correct_clocks(
    photos: list[PhotoState],
    *,
    clock_pair_min_sim: float,
    clock_min_pairs: int,
    clock_offset_tolerance_sec: float,
) -> list[ClockOffset]:
    """처리 시각과 상태를 갱신하고, 대응 쌍으로 확인한 대상 기기의 오프셋을 반환한다.

    원본 GPS와 시각이 있는 사진을 가진 기기는 원본 시각을 신뢰한다.
    기기명 없는 GPS 사진은 해당 사진의 시각만 신뢰하며 다른 무명 사진으로 확장하지 않는다.
    참조는 원본 GPS와 시각을 함께 가진 사진으로 고정해 보정 결과를 재사용하지 않는다.
    날짜 후보별 초 단위 차이의 중앙값 ± 허용 오차 안의 쌍을 세고 정수 일수만 적용한다.
    기기명 또는 시각 누락과 근거 부족은 UNKNOWN이다. 임베딩 등 내부 오류는 실패한다.
    """
    if not np.isfinite(clock_pair_min_sim) or not -1 <= clock_pair_min_sim <= 1:
        raise PipelineError("clock_pair_min_sim은 -1 이상 1 이하의 유한값이어야 함")
    if (
        isinstance(clock_min_pairs, bool)
        or not isinstance(clock_min_pairs, int)
        or clock_min_pairs < 1
    ):
        raise PipelineError("clock_min_pairs는 1 이상의 정수여야 함")
    if (
        not np.isfinite(clock_offset_tolerance_sec)
        or not 0 <= clock_offset_tolerance_sec < SECONDS_PER_DAY / 2
    ):
        raise PipelineError("CLOCK 허용 오차는 0 이상 반일 미만의 초 단위 값이어야 함")
    for photo in photos:
        if photo.source.taken_at is not None and photo.source.taken_at.utcoffset() is None:
            raise PipelineError("CLOCK 원본 시각에는 시간대가 필요함")

    anchors = sorted(
        (photo for photo in photos if photo.source.has_gps and photo.source.taken_at is not None),
        key=lambda photo: photo.source.trip_attachment_id,
    )
    reference_devices = {
        photo.source.device_model for photo in anchors if photo.source.device_model
    }
    targets: dict[str, list[PhotoState]] = defaultdict(list)
    for photo in photos:
        photo.taken_at = None
        photo.time_status = TimeStatus.UNKNOWN
        source = photo.source
        if source.taken_at is None:
            continue
        if source.has_gps or source.device_model in reference_devices:
            photo.taken_at = source.taken_at
            photo.time_status = TimeStatus.ORIGINAL
        elif source.device_model:
            targets[source.device_model].append(photo)
    if not anchors:
        return []

    results = []
    for device, group in sorted(targets.items()):
        group.sort(key=lambda photo: photo.source.trip_attachment_id)
        offsets = _pair_offsets(group, anchors, clock_pair_min_sim)
        estimate = _consistent_days(offsets, clock_min_pairs, clock_offset_tolerance_sec)
        if estimate is None:
            continue
        days, matched_pairs = estimate
        for photo in group:
            try:
                photo.taken_at = photo.source.taken_at + timedelta(days=days)
            except OverflowError as exc:
                raise PipelineError("CLOCK 보정 결과가 표현 가능한 날짜 범위를 벗어남") from exc
            photo.time_status = TimeStatus.CORRECTED if days else TimeStatus.ORIGINAL
        results.append(ClockOffset(device_model=device, days=days, matched_pairs=matched_pairs))
    return results
