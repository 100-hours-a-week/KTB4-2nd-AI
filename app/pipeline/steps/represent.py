"""FINALIZING에서 사용하는 대표 사진 선정과 선명도 점수."""

import numpy as np

from app.core.errors import PipelineError
from app.pipeline.state import PhotoState


def quality_score(variance: float) -> int:
    """선명도를 0~100 정수로 변환한다. 대표성 점수가 아니다.

    100 * v / (v + 100)으로 포화시켜 여행 내 최솟값과 최댓값에 의존하지 않는다.
    분산 0은 0점, 100은 50점이다. 분모의 100은 초기 정규화 척도이며 흐림 판정
    임계값과 별개다. 실제 사진으로 보정되지 않은 점수로, 절대 품질 확률이 아니다.
    """
    if not np.isfinite(variance) or variance < 0:
        raise PipelineError("품질 점수에는 0 이상의 유한한 라플라시안 분산이 필요함")
    return round(100.0 * (variance / (variance + 100.0)))


def select_representative(photos: list[PhotoState]) -> int:
    """호출자가 전달한 보존 후보 중 평균 임베딩 최근접 사진 ID를 반환한다."""
    if not photos:
        raise PipelineError("대표 사진 후보가 없음")
    ordered = sorted(photos, key=lambda photo: photo.source.trip_attachment_id)
    vectors = [photo.embedding for photo in ordered]
    if (
        any(
            not isinstance(vector, np.ndarray)
            or vector.ndim != 1
            or vector.size == 0
            or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-5, atol=1e-6)
            for vector in vectors
        )
        or len({vector.shape for vector in vectors}) != 1
    ):
        raise PipelineError("대표 사진 후보에 같은 차원의 유효한 단위 임베딩이 필요함")
    matrix = np.stack(vectors).astype(np.float64)
    # 후보는 단위 벡터이므로 평균 벡터를 정규화하지 않아도 코사인 순위가 같다.
    # 평균이 영벡터면 방향을 정할 수 없어 전부 동점으로 보고 가장 작은 ID를 고른다.
    scores = matrix @ matrix.mean(axis=0)
    return ordered[int(np.argmax(scores))].source.trip_attachment_id
