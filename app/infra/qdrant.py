"""벡터 저장소. 정리 실행기는 QdrantStore(동기, PipelineContext.qdrant),
ai-api의 검색과 스토리는 AsyncQdrantVectors(비동기, 읽기 전용)

계약 값을 이 파일 안에서만 다룸. 컬렉션 vectors_{model_version}, point id는 trip_attachment_id,
payload는 비움. 모델이 바뀌면 벡터를 전부 다시 만들므로 컬렉션을 모델별로 분리
"""

import re

import httpx
import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, Filter, HasIdCondition, PointStruct, VectorParams

from app.core.logging import get_logger
from app.schemas.query import SearchHit
from app.search.context import ModelUnavailable

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
        try:
            self._ensure_collection()
        except BaseException:
            self._client.close()
            raise

    def close(self) -> None:
        self._client.close()

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


class AsyncQdrantVectors:
    """ai-api의 검색과 스토리용 비동기 읽기. 저장과 컬렉션 생성은 정리 실행기의 QdrantStore만 담당

    컬렉션이 없으면 빈 결과, 시간 초과는 TimeoutError, 그 외 연결과 응답 오류는 ModelUnavailable
    """

    def __init__(self, location: str, model_version: str, dim: int, timeout_sec: int = 5) -> None:
        self.dim = dim
        self.collection = collection_name(model_version)
        # 원격 클라이언트는 생성자에서 서버 버전을 동기로 확인하므로 끔, 이벤트 루프 차단 방지
        self._client = (
            AsyncQdrantClient(location=MEMORY)
            if location == MEMORY
            else AsyncQdrantClient(url=location, timeout=timeout_sec, check_compatibility=False)
        )
        self._collection_ready = False

    async def search(
        self, vector: list[float], *, candidate_ids: list[int], limit: int
    ) -> list[SearchHit]:
        """CandidateSearcher 구현. 후보 ID 필터를 질의에 걸어 후보 밖 사진은 검색 대상에서 제외"""
        if len(vector) != self.dim:
            raise ValueError(f"질의 벡터는 {self.dim}차원이어야 함, 받은 것 {len(vector)}")
        if not candidate_ids:
            return []
        try:
            if not await self._has_collection():
                return []
            response = await self._client.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=Filter(must=[HasIdCondition(has_id=candidate_ids)]),
                limit=limit,
                with_payload=False,
            )
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            if isinstance(exc, ResponseHandlingException) and isinstance(
                exc.source, httpx.TimeoutException
            ):
                raise TimeoutError("벡터 검색 시간이 초과되었습니다.") from None
            # 원인 구분용으로 예외 종류만 기록, 질의 벡터와 후보 ID는 남기지 않음
            logger.warning("qdrant unavailable", extra={"error_type": type(exc).__name__})
            raise ModelUnavailable("벡터 검색을 사용할 수 없습니다.") from None
        return [SearchHit(trip_attachment_id=int(p.id), score=p.score) for p in response.points]

    async def fetch(self, ids: list[int]) -> dict[int, np.ndarray]:
        """스토리 사진 선별용. QdrantStore.fetch와 같은 형식, 없는 id는 결과에서 빠짐"""
        if not ids:
            return {}
        try:
            if not await self._has_collection():
                return {}
            records = await self._client.retrieve(
                collection_name=self.collection, ids=ids, with_vectors=True, with_payload=False
            )
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            if isinstance(exc, ResponseHandlingException) and isinstance(
                exc.source, httpx.TimeoutException
            ):
                raise TimeoutError("벡터 조회 시간이 초과되었습니다.") from None
            logger.warning("qdrant unavailable", extra={"error_type": type(exc).__name__})
            raise ModelUnavailable("벡터 조회를 사용할 수 없습니다.") from None
        return {int(record.id): np.asarray(record.vector, dtype=np.float32) for record in records}

    async def _has_collection(self) -> bool:
        """한 번 확인되면 다시 묻지 않음. 컬렉션은 운영 중 삭제하지 않음"""
        if not self._collection_ready:
            self._collection_ready = await self._client.collection_exists(self.collection)
            if not self._collection_ready:
                logger.warning("collection missing", extra={"collection": self.collection})
        return self._collection_ready

    async def aclose(self) -> None:
        await self._client.close()
