"""벡터 저장소, PipelineContext.qdrant

계약 값을 이 파일 안에서만 다룸. 컬렉션 vectors_{model_version}, point id는 trip_attachment_id,
payload는 비움. 모델이 바뀌면 벡터를 전부 다시 만들므로 컬렉션을 모델별로 분리
"""

import re

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.logging import get_logger

logger = get_logger(__name__)

MEMORY = ":memory:"


def collection_name(model_version: str) -> str:
    """vectors_{model_version}. 이름에 못 쓰는 문자는 _로"""
    return "vectors_" + re.sub(r"[^A-Za-z0-9_-]", "_", model_version)


class QdrantStore:
    """워커는 QDRANT_URL로 서버에, 테스트와 로컬 개발은 ":memory:"로 프로세스 안에서"""

    def __init__(self, location: str, model_version: str, dim: int) -> None:
        self.dim = dim
        self.collection = collection_name(model_version)
        self._client = QdrantClient(MEMORY) if location == MEMORY else QdrantClient(url=location)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """없으면 생성, 있으면 그대로. 재기동해도 /data/qdrant에 남아 있으면 재사용"""
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            logger.info(
                "collection created", extra={"collection": self.collection, "dim": self.dim}
            )

    def upsert(self, ids: list[int], vectors: np.ndarray) -> None:
        """ids[i]의 벡터가 vectors[i]. 같은 id면 덮어씀"""
        if vectors.shape != (len(ids), self.dim):
            raise ValueError(
                f"vectors 모양은 ({len(ids)}, {self.dim})이어야 함, 받은 것 {vectors.shape}"
            )
        points = [
            PointStruct(id=point_id, vector=vector.tolist(), payload={})
            for point_id, vector in zip(ids, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    def fetch(self, ids: list[int]) -> dict[int, np.ndarray]:
        """저장된 벡터. 없는 id는 결과에서 빠짐"""
        records = self._client.retrieve(collection_name=self.collection, ids=ids, with_vectors=True)
        return {int(record.id): np.asarray(record.vector, dtype=np.float32) for record in records}
