"""스토리 기준선: 품질 필터, 장소별 비례 배분, 평균 임베딩 기반 선별과 같은 장면 중복 제외."""

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


def unit_vectors(groups, vectors: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """장소별 후보 벡터를 확인하고, 행마다 L2 정규화한 행렬로 돌려준다."""
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
    units = {}
    for place_id, group in groups.items():
        matrix = np.asarray(
            [vectors[photo.trip_attachment_id] for photo in group], dtype=np.float64
        )
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise failure(ErrorCode.MODEL_NOT_READY, "사진 벡터 형식이 올바르지 않습니다.")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise failure(ErrorCode.MODEL_NOT_READY, "사진 벡터 크기가 올바르지 않습니다.")
        units[place_id] = matrix / norms[:, None]
    return units


def similarities(units: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """장소마다 사진 쌍의 코사인 유사도 행렬.

    기준 계산과 선택이 같은 값을 비교하도록 한 번만 구해 함께 쓴다. 행렬곱과 벡터 내적은
    마지막 자리의 반올림이 달라 기준과 같은 유사도의 판정이 뒤바뀔 수 있고, 행렬곱도
    정확히 대칭이 아닐 수 있어 대칭으로 맞춘다.
    """
    result = {}
    for place_id, matrix in units.items():
        gram = matrix @ matrix.T
        result[place_id] = (gram + gram.T) / 2
    return result


def duplicate_threshold(sims: dict[int, np.ndarray], z: float) -> float | None:
    """같은 장소 사진 쌍의 코사인 유사도 평균 + z × 표준편차. 같은 장소 쌍이 없으면 None.

    서로 비슷한 사진이 많은 여행일수록 기준이 높아지도록 여행마다 정한다(v2 QA Q6, z 1.0).
    쌍이 아주 적으면 기준이 불안정하다. 쌍이 하나면 기준이 그 쌍의 유사도와 같아져 두 장을
    중복으로 본다.
    """
    pairs = [s[np.triu_indices(len(s), k=1)] for s in sims.values() if len(s) > 1]
    if not pairs:
        return None
    values = np.concatenate(pairs)
    return float(values.mean() + z * values.std())


def select(
    groups,
    units: dict[int, np.ndarray],
    sims: dict[int, np.ndarray],
    *,
    budget: int,
    threshold: float | None,
):
    allocations = allocate([len(group) for group in groups.values()], budget)
    selected = []
    for (place_id, group), count in zip(groups.items(), allocations, strict=True):
        matrix, similar = units[place_id], sims[place_id]
        # 평균의 길이는 모든 비교에 같으므로 내적으로 순위를 정한다.
        scores = matrix @ matrix.mean(axis=0)
        order = sorted(
            range(len(group)),
            key=lambda i: (-scores[i], group[i].taken_at, group[i].trip_attachment_id),
        )
        chosen = []
        for i in order:
            if len(chosen) == count:
                break
            # 이미 고른 사진과 기준 이상으로 닮으면 같은 장면으로 보고 건너뛴다.
            # 그래서 못 채운 자리는 비운다(사진 수를 예산에 맞추지 않음).
            if threshold is not None and any(similar[i, j] >= threshold for j in chosen):
                continue
            chosen.append(i)
        selected.extend((place_id, group[i]) for i in chosen)
    return sorted(selected, key=lambda item: (item[1].taken_at, item[1].trip_attachment_id))
