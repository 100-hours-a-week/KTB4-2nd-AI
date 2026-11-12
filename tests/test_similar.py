"""유사 사진 후보 API. 후보 범위, 기준 사진 제외, 점수 순서, 입력 검증, 의존성 실패를 확인한다."""

import asyncio
from dataclasses import replace

import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.api.services.search_tracing import TracedVectors
from app.infra.qdrant import MEMORY, AsyncQdrantVectors
from app.search.context import ModelUnavailable, SearchOptions
from app.search.mock import create_mock_context
from app.v2_main import create_app as create_v2_app
from tests.test_query import client_for

pytestmark = pytest.mark.anyio
URL = "/trips/7/attachments/similar"
POINTS = {
    501: [1.0, 0.0, 0.0],
    502: [0.0, 1.0, 0.0],
    503: [0.9, 0.1, 0.0],
    504: [0.7, 0.7, 0.0],
    # 기준 사진과 가장 닮았지만 후보가 아닌 사진(다른 여행)
    509: [1.0, 0.01, 0.0],
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def vectors():
    store = AsyncQdrantVectors(MEMORY, "test-model", 3)
    await store._client.create_collection(
        store.collection, vectors_config=VectorParams(size=3, distance=Distance.COSINE)
    )
    await store._client.upsert(
        store.collection,
        points=[PointStruct(id=i, vector=v, payload={}) for i, v in POINTS.items()],
    )
    yield store
    await store.aclose()


def body(source=501, candidates=(501, 502, 503, 504), **extra):
    return {"trip_attachment_id": source, "candidate_attachment_ids": list(candidates), **extra}


def context_with(vectors):
    return replace(create_mock_context(), vectors=TracedVectors(vectors))


async def test_similar_sorted_within_candidates_without_source(vectors):
    async with client_for(context_with(vectors)) as client:
        response = await client.post(URL, json=body())

    assert response.status_code == 200
    data = response.json()
    assert data["source_attachment_id"] == 501
    assert [item["trip_attachment_id"] for item in data["items"]] == [503, 504, 502]
    scores = [item["score"] for item in data["items"]]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(0.9 / (0.82**0.5))


async def test_limit_bounds_items(vectors):
    async with client_for(context_with(vectors)) as client:
        response = await client.post(URL, json=body(limit=1))

    assert [item["trip_attachment_id"] for item in response.json()["items"]] == [503]


@pytest.mark.parametrize(
    "request_body",
    [
        # 기준 사진 벡터가 아직 없음(임베딩 전이거나 실패)
        body(source=600),
        # 후보가 기준 사진뿐이거나 비어 있음
        body(candidates=[501]),
        body(candidates=[]),
    ],
)
async def test_nothing_to_compare_returns_empty_items(vectors, request_body):
    async with client_for(context_with(vectors)) as client:
        response = await client.post(URL, json=request_body)

    assert response.status_code == 200
    assert response.json() == {
        "source_attachment_id": request_body["trip_attachment_id"],
        "items": [],
    }


async def test_empty_candidates_do_not_touch_vectors():
    class Unexpected:
        async def fetch(self, ids):
            raise AssertionError("벡터를 조회하면 안 됨")

        async def search(self, vector, *, candidate_ids, limit):
            raise AssertionError("벡터를 검색하면 안 됨")

    async with client_for(replace(create_mock_context(), vectors=Unexpected())) as client:
        response = await client.post(URL, json=body(candidates=[]))

    assert response.status_code == 200


@pytest.mark.parametrize(
    "updates",
    [
        {"limit": 0},
        {"limit": 51},
        {"limit": "10"},
        {"trip_attachment_id": 0},
        {"candidate_attachment_ids": [502, 502]},
        {"candidate_attachment_ids": [0]},
        {"candidate_attachment_ids": None},
    ],
)
async def test_invalid_input(vectors, updates):
    async with client_for(context_with(vectors)) as client:
        response = await client.post(URL, json={**body(), **updates})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


async def test_invalid_trip_id_and_missing_auth(vectors):
    async with client_for(context_with(vectors)) as client:
        assert (await client.post("/trips/0/attachments/similar", json=body())).status_code == 422
        response = await client.post(URL, json=body(), headers={"Authorization": ""})

    assert response.status_code == 401


async def test_vector_store_unavailable_returns_503():
    class Down:
        async def fetch(self, ids):
            raise ModelUnavailable("벡터 조회를 사용할 수 없습니다.")

    async with client_for(replace(create_mock_context(), vectors=Down())) as client:
        response = await client.post(URL, json=body())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"


async def test_slow_vector_store_returns_504():
    class Slow:
        async def fetch(self, ids):
            await asyncio.sleep(1)

    options = SearchOptions(retrieval_timeout_sec=0.01)
    async with client_for(replace(create_mock_context(), vectors=Slow()), options) as client:
        response = await client.post(URL, json=body())

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "TIMEOUT"


async def test_result_outside_candidates_is_internal_error():
    class Leaky:
        async def fetch(self, ids):
            return {501: [1.0, 0.0, 0.0]}

        async def search(self, vector, *, candidate_ids, limit):
            from app.schemas.query import SearchHit

            return [SearchHit(trip_attachment_id=509, score=0.99)]

    async with client_for(replace(create_mock_context(), vectors=Leaky())) as client:
        response = await client.post(URL, json=body())

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"


def test_v2_app_exposes_similar_route():
    paths = create_v2_app().openapi()["paths"]

    assert list(paths["/trips/{trip_id}/attachments/similar"]) == ["post"]
