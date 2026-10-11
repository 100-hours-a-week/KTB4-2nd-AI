"""모델 품질이 아닌 검색 HTTP 계약과 외부 의존성 실패 경계를 검증한다."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import pytest

from app.api.services.query_service import QueryService
from app.core.config import Settings
from app.main import create_app
from app.schemas.query import SearchHit
from app.search.context import InvalidModelOutput, ModelUnavailable, SearchOptions, VLMUnavailable
from app.search.mock import MockEmbedder, MockImages, MockVLM, create_mock_context

pytestmark = pytest.mark.anyio
AUTH = {"Authorization": "Bearer test-key", "X-Request-ID": "search-test"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def client_for(context=None, options=None):
    service = QueryService(context, options) if context else None
    app = create_app(
        settings=Settings(API_KEY="test-key", IMAGE_SOURCE="local"), query_service=service
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=AUTH
        ) as client:
            yield client


def search_body(intent="SEARCH", count=2):
    return {
        "query": "바다 사진 보여줘",
        "intent": intent,
        "visual_query": None,
        "candidates": [
            {
                "trip_attachment_id": i + 1,
                "analyze_storage_key": f"trips/1/{i}.jpg",
                "taken_at": None,
                "place_name": None,
                "trip_name": None,
            }
            for i in range(count)
        ],
    }


class FailingVLM(MockVLM):
    def __init__(self, error):
        self.error = error

    async def interpret(self, request):
        raise self.error

    async def answer(self, query, evidence):
        raise self.error


class SlowVLM(MockVLM):
    async def answer(self, query, evidence):
        await asyncio.sleep(60)
        raise AssertionError("시간 제한으로 취소되어야 함")


class FixedVectors:
    def __init__(self, hits):
        self.hits = hits
        self.candidate_ids = None

    async def search(self, vector, *, candidate_ids, limit):
        self.candidate_ids = candidate_ids
        return self.hits


class SpyImages(MockImages):
    def __init__(self):
        self.keys = []

    async def resolve(self, storage_key):
        self.keys.append(storage_key)
        return await super().resolve(storage_key)


async def test_mock_parse_search_question_without_network(monkeypatch):
    import socket

    def forbid_network(*args, **kwargs):
        raise AssertionError("mock 경로는 외부에 접속하면 안 됨")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    async with client_for(create_mock_context()) as client:
        parsed = await client.post(
            "/query/parse", json={"query": "바다 사진?", "reference_date": "2026-10-11"}
        )
        assert parsed.status_code == 200
        assert parsed.json()["intent"] == "QUESTION"
        assert parsed.headers["X-Request-ID"] == "search-test"
        payload = search_body(parsed.json()["intent"])
        payload["visual_query"] = parsed.json()["visual_query"]
        response = await client.post("/query/search", json=payload)
        assert response.status_code == 200
        assert len(response.json()["attachments"]) == 2
        assert response.json()["answer"].startswith("[MOCK]")
        assert response.json()["answer_error"] is None


async def test_unconfigured_server_does_not_use_mock():
    async with client_for() as client:
        response = await client.post("/query/search", json=search_body())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_NOT_READY"


@pytest.mark.parametrize(
    "route,body",
    [
        ("parse", {"query": "사진", "reference_date": "2026-10-11"}),
        ("search", search_body()),
    ],
)
async def test_authentication(route, body):
    async with client_for(create_mock_context()) as client:
        for auth in ["", "Bearer wrong"]:
            response = await client.post(
                f"/query/{route}", json=body, headers={"Authorization": auth}
            )
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "UNAUTHORIZED"


@pytest.mark.parametrize(
    "updates",
    [
        {"limit": None},
        {"limit": 0},
        {"limit": 51},
        {"limit": 1.5},
        {"limit": True},
        {"query": "x"},
        {"intent": "OTHER"},
        {"candidate_attachment_ids": [1]},
    ],
)
async def test_invalid_search_input(updates):
    async with client_for(create_mock_context()) as client:
        response = await client.post("/query/search", json=search_body() | updates)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize(
    "body",
    [
        {"query": "사진", "reference_date": "invalid"},
        {"query": "사진"},
        {"query": "사진", "reference_date": "2026-10-11", "trip_id": 1},
    ],
)
async def test_invalid_parse_input(body):
    async with client_for(create_mock_context()) as client:
        assert (await client.post("/query/parse", json=body)).status_code == 422


async def test_duplicate_candidates_and_naive_time_rejected():
    payload = search_body()
    payload["candidates"][1] = payload["candidates"][0].copy()
    async with client_for(create_mock_context()) as client:
        assert (await client.post("/query/search", json=payload)).status_code == 422
        payload = search_body()
        payload["candidates"][0]["taken_at"] = "2026-10-11T12:00:00"
        assert (await client.post("/query/search", json=payload)).status_code == 422


@pytest.mark.parametrize("intent,error", [("SEARCH", None), ("QUESTION", "NO_EVIDENCE")])
async def test_empty_candidates_skip_dependencies(intent, error):
    class UnexpectedEmbedder:
        async def encode_text(self, text):
            raise AssertionError("빈 후보에서는 호출 금지")

    context = replace(create_mock_context(), embedder=UnexpectedEmbedder())
    async with client_for(context) as client:
        response = await client.post("/query/search", json=search_body(intent, 0))
        assert response.status_code == 200
        assert response.json() == {"attachments": [], "answer": None, "answer_error": error}


@pytest.mark.parametrize("error", [VLMUnavailable(), InvalidModelOutput(), TimeoutError()])
async def test_parse_fallback(error):
    context = replace(create_mock_context(), vlm=FailingVLM(error))
    async with client_for(context) as client:
        response = await client.post(
            "/query/parse", json={"query": "작년 바다", "reference_date": "2026-10-11"}
        )
        assert response.status_code == 200
        assert response.json() == {
            "intent": "SEARCH",
            "date_from": None,
            "date_to": None,
            "region_names": None,
            "visual_query": "작년 바다",
            "parsed": False,
        }


async def test_search_sorted_deduplicated_and_bounded():
    vectors = FixedVectors(
        [
            SearchHit(trip_attachment_id=1, score=0.1),
            SearchHit(trip_attachment_id=2, score=0.8),
            SearchHit(trip_attachment_id=1, score=0.9),
        ]
    )
    context = replace(create_mock_context(), vectors=vectors)
    async with client_for(context) as client:
        response = await client.post("/query/search", json=search_body() | {"limit": 1})
        assert response.status_code == 200
        assert response.json() == {
            "attachments": [{"trip_attachment_id": 1, "score": 0.9}],
            "answer": None,
            "answer_error": None,
        }
        assert vectors.candidate_ids == [1, 2]


async def test_candidate_boundary_before_image_or_vlm_access():
    images = SpyImages()
    context = replace(
        create_mock_context(),
        images=images,
        vectors=FixedVectors([SearchHit(trip_attachment_id=999, score=1)]),
    )
    async with client_for(context) as client:
        response = await client.post("/query/search", json=search_body("QUESTION"))
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "INTERNAL_ERROR"
        assert images.keys == []


async def test_question_uses_k_and_only_resolves_evidence():
    images = SpyImages()
    context = replace(create_mock_context(), images=images)
    async with client_for(context, SearchOptions(question_k=3)) as client:
        response = await client.post(
            "/query/search", json=search_body("QUESTION", 8) | {"limit": 1}
        )
        assert response.status_code == 200
        assert len(response.json()["attachments"]) == 3
        assert len(images.keys) == 3


@pytest.mark.parametrize(
    "vlm,expected",
    [
        (FailingVLM(VLMUnavailable("private URL")), "VLM_UNAVAILABLE"),
        (FailingVLM(InvalidModelOutput()), "VLM_UNAVAILABLE"),
        (SlowVLM(), "TIMEOUT"),
    ],
)
async def test_answer_failure_preserves_photos(vlm, expected):
    context = replace(create_mock_context(), vlm=vlm)
    async with client_for(context, SearchOptions(answer_timeout_sec=0.01)) as client:
        response = await client.post("/query/search", json=search_body("QUESTION"))
        assert response.status_code == 200
        assert len(response.json()["attachments"]) == 2
        assert response.json()["answer"] is None
        assert response.json()["answer_error"] == expected
        assert "private URL" not in response.text


async def test_embedding_unavailable_and_query_fallback():
    class Embedder(MockEmbedder):
        async def encode_text(self, text):
            assert text == "바다 사진 보여줘"
            raise ModelUnavailable("private endpoint")

    context = replace(create_mock_context(), embedder=Embedder())
    async with client_for(context) as client:
        response = await client.post("/query/search", json=search_body())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_NOT_READY"
        assert "private endpoint" not in response.text


async def test_retrieval_deadline_cancels_dependency():
    canceled = asyncio.Event()

    class Embedder:
        async def encode_text(self, text):
            try:
                await asyncio.sleep(60)
            finally:
                canceled.set()

    context = replace(create_mock_context(), embedder=Embedder())
    async with client_for(context, SearchOptions(retrieval_timeout_sec=0.01)) as client:
        response = await client.post("/query/search", json=search_body())
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "TIMEOUT"
        assert canceled.is_set()


async def test_missing_vectors_question_returns_no_evidence():
    context = replace(create_mock_context(), vectors=FixedVectors([]))
    async with client_for(context) as client:
        response = await client.post("/query/search", json=search_body("QUESTION"))
        assert response.status_code == 200
        assert response.json()["answer_error"] == "NO_EVIDENCE"
