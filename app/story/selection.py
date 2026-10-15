"""스토리 기준선: 품질 필터, 장소별 비례 배분과 평균 임베딩 기반 선별."""

from collections.abc import Callable

import numpy as np

from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.jobs.models import StoryAttachment, StoryMessage
from app.pipeline.steps.download import load_images
from app.pipeline.steps.filtering import _measure


def failure(code: ErrorCode, message: str, **detail) -> PipelineError:
    error = PipelineError(message, detail=detail)
    error.code = code
    return error


def filter_candidates(
    request: StoryMessage, images, *, blur_min: float, phash_max: int, cancelled: Callable[[], bool]
) -> dict[int, list[StoryAttachment]]:
    groups = {}
    for place in request.places:
        retained, hashes = [], []
        photos = sorted(
            place.attachments, key=lambda photo: (photo.taken_at, photo.trip_attachment_id)
        )
        for start in range(0, len(photos), 16):
            if cancelled():
                raise PipelineCancelled()
            chunk = photos[start : start + 16]
            _, decoded = load_images(chunk, images)
            try:
                for photo in chunk:
                    if cancelled():
                        raise PipelineCancelled()
                    variance, phash = _measure(decoded[photo.trip_attachment_id])
                    if variance < blur_min or any(phash - other <= phash_max for other in hashes):
                        continue
                    retained.append(photo)
                    hashes.append(phash)
            finally:
                for image in decoded.values():
                    image.close()
        if not retained:
            raise failure(
                ErrorCode.INVALID_REQUEST,
                "선별 후 사용할 사진이 없는 장소가 있습니다.",
                reason="NO_USABLE_PHOTOS",
                trip_place_id=place.trip_place_id,
            )
        groups[place.trip_place_id] = retained
    return groups


def allocate(counts: list[int], budget: int) -> list[int]:
    """장소당 한 장을 확보한 뒤 남은 사진 수에 비례해 최대 나머지 방식으로 배분한다."""
    if not counts or any(count < 1 for count in counts) or budget < len(counts):
        raise ValueError("사진 예산은 모든 장소에 한 장씩 제공할 수 있어야 합니다.")
    remaining = min(budget, sum(counts)) - len(counts)
    capacity = [count - 1 for count in counts]
    if not remaining:
        return [1] * len(counts)
    total = sum(capacity)
    shares = [remaining * count // total for count in capacity]
    rest = remaining - sum(shares)
    order = sorted(range(len(counts)), key=lambda i: (-(remaining * capacity[i] % total), i))
    for i in order[:rest]:
        shares[i] += 1
    return [share + 1 for share in shares]


def select(groups, vectors: dict[int, np.ndarray], *, budget: int):
    missing = [
        photo.trip_attachment_id
        for group in groups.values()
        for photo in group
        if photo.trip_attachment_id not in vectors
    ]
    if missing:
        raise failure(
            ErrorCode.MODEL_NOT_READY,
            "스토리 선별에 필요한 사진 벡터가 없습니다.",
            reason="MISSING_VECTORS",
            trip_attachment_ids=missing,
        )
    allocations = allocate([len(group) for group in groups.values()], budget)
    selected = []
    for (place_id, group), count in zip(groups.items(), allocations, strict=True):
        matrix = np.asarray(
            [vectors[photo.trip_attachment_id] for photo in group], dtype=np.float64
        )
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise failure(ErrorCode.MODEL_NOT_READY, "사진 벡터 형식이 올바르지 않습니다.")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise failure(ErrorCode.MODEL_NOT_READY, "사진 벡터 크기가 올바르지 않습니다.")
        matrix = matrix / norms[:, None]
        # 평균의 길이는 모든 비교에 같으므로 내적으로 순위를 정한다.
        scores = matrix @ matrix.mean(axis=0)
        order = sorted(
            range(len(group)),
            key=lambda i: (-scores[i], group[i].taken_at, group[i].trip_attachment_id),
        )
        selected.extend((place_id, group[i]) for i in order[:count])
    return sorted(selected, key=lambda item: (item[1].taken_at, item[1].trip_attachment_id))
