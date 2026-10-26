"""EMBEDDING과 업로드 묶음 임베딩이 공유하는 엔진 호출과 출력 계약 검사"""

import numpy as np

from app.core.errors import PipelineError
from app.embedding.contracts import EmbeddingEngine


def encode_checked(engine: EmbeddingEngine, blobs: list[bytes]) -> np.ndarray:
    """사본 bytes 목록의 벡터. 출력이 (N, dim) float32 단위 벡터가 아니면 PipelineError

    저장 전에 검사해 계약과 다른 벡터가 Qdrant에 들어가지 않게 함. upsert는 호출자가 담당
    """
    vectors = engine.encode_images(blobs)
    if (
        not isinstance(vectors, np.ndarray)
        or vectors.shape != (len(blobs), engine.dim)
        or vectors.dtype != np.float32
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=1e-5, atol=1e-6)
    ):
        raise PipelineError("임베딩 엔진 출력이 float32 단위 벡터 계약과 다름")
    return vectors
