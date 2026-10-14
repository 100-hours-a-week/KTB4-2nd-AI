"""ai-api용 비동기 Qdrant 읽기. 후보 범위, 점수 순서, 컬렉션 없음, 오류 변환을 확인한다."""

from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.infra.qdrant import MEMORY, AsyncQdrantVectors
from app.schemas.query import SearchCandidate, SearchRequest
from app.search import pipeline
from app.search.context import ModelUnavailable, SearchContext, SearchOptions
from app.search.mock import MockImages, MockVLM

pytestmark = pytest.mark.anyio
QUERY = [1.0, 0.0, 0.0]
POINTS = {
    1: [1.0, 0.0, 0.0],
    2: [0.0, 1.0, 0.0],
    3: [0.9, 0.1, 0.0],
    # 질의와 가장 가깝지만 후보가 아닌 사진
    4: [1.0, 0.01, 0.0],
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def vectors():
    store = AsyncQdrantVectors(MEMORY, "test-model", 3)
    yield store
    await store.aclose()


async def seed(store: AsyncQdrantVectors, points: dict[int, list[float]] = POINTS) -> None:
    """정리 실행기가 만든 컬렉션을 흉내. ai-api 클래스는 컬렉션을 만들지 않음"""
    await store._client.create_collection(
        store.collection, vectors_config=VectorParams(size=3, distance=Distance.COSINE)
    )
    await store._client.upsert(
        store.collection,
        points=[PointStruct(id=i, vector=v, payload={}) for i, v in points.items()],
    )


async def test_search_stays_within_candidates_sorted_and_limited(vectors):
    await seed(vectors)

    hits = await vectors.search(QUERY, candidate_ids=[1, 2, 3], limit=2)

    assert [hit.trip_attachment_id for hit in hits] == [1, 3]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score > hits[1].score


async def test_candidates_without_vectors_are_skipped(vectors):
    await seed(vectors)

    hits = await vectors.search(QUERY, candidate_ids=[2, 99], limit=10)

    assert [hit.trip_attachment_id for hit in hits] == [2]


async def test_empty_candidates_return_empty_without_query(vectors):
    vectors._client.query_points = AsyncMock(side_effect=AssertionError("질의 금지"))

    assert await vectors.search(QUERY, candidate_ids=[], limit=5) == []


async def test_missing_collection_returns_empty_until_created(vectors):
    assert await vectors.search(QUERY, candidate_ids=[1], limit=5) == []
    assert await vectors.fetch([1]) == {}

    # 정리 실행기가 첫 사진을 저장해 컬렉션이 생기면 다음 요청부터 검색됨
    await seed(vectors)
    hits = await vectors.search(QUERY, candidate_ids=[1], limit=5)
    assert [hit.trip_attachment_id for hit in hits] == [1]


async def test_vector_dimension_mismatch_is_a_bug(vectors):
    with pytest.raises(ValueError):
        await vectors.search([1.0, 0.0], candidate_ids=[1], limit=5)


async def test_fetch_returns_float32_vectors_and_skips_missing(vectors):
    await seed(vectors)

    got = await vectors.fetch([1, 2, 99])

    assert sorted(got) == [1, 2]
    assert got[1].dtype == np.float32
    np.testing.assert_allclose(got[2], [0.0, 1.0, 0.0])
    assert await vectors.fetch([]) == {}


@pytest.mark.parametrize(
    "error",
    [
        ResponseHandlingException(OSError("connection refused")),
        UnexpectedResponse(500, "Internal Server Error", b"private", httpx.Headers()),
    ],
    ids=["connection", "server_error"],
)
async def test_qdrant_errors_become_model_unavailable(vectors, monkeypatch, error):
    from app.infra import qdrant

    warnings = []
    monkeypatch.setattr(qdrant.logger, "warning", lambda msg, *, extra: warnings.append(extra))
    await seed(vectors)
    vectors._client.query_points = AsyncMock(side_effect=error)
    vectors._client.retrieve = AsyncMock(side_effect=error)

    with pytest.raises(ModelUnavailable) as searched:
        await vectors.search(QUERY, candidate_ids=[1], limit=5)
    with pytest.raises(ModelUnavailable):
        await vectors.fetch([1])
    assert searched.value.__cause__ is None
    # 원인은 예외 종류로만 남고 응답 본문은 로그에 들어가지 않음
    assert warnings == [{"error_type": type(error).__name__}] * 2
    assert "private" not in str(warnings)


async def test_search_pipeline_uses_async_vectors_as_candidate_searcher(vectors):
    await seed(vectors)

    class Embedder:
        async def encode_text(self, text):
            return QUERY

    context = SearchContext(
        vlm=MockVLM(), embedder=Embedder(), vectors=vectors, images=MockImages()
    )
    request = SearchRequest(
        query="바다 사진",
        intent="SEARCH",
        visual_query="바다",
        candidates=[
            SearchCandidate(
                trip_attachment_id=i,
                analyze_storage_key=f"trips/1/{i}.jpg",
                taken_at=None,
                place_name=None,
                trip_name=None,
            )
            for i in (1, 2, 3)
        ],
        limit=2,
    )

    response = await pipeline.search(request, context, SearchOptions())

    assert [hit.trip_attachment_id for hit in response.attachments] == [1, 3]
