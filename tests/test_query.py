"""모델 품질이 아닌 검색 HTTP 계약과 외부 의존성 실패 경계를 검증한다."""

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest
from pydantic import ValidationError

from app.api.services.query_service import QueryService
from app.core.config import Settings
from app.infra.vlm import VLMUnavailable
from app.main import create_app
from app.schemas.query import ParseRequest, ParseResponse, SearchHit, SearchResponse
from app.search import prompts
from app.search.context import InvalidModelOutput, ModelUnavailable, SearchOptions
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

    async def chat(self, messages, image_urls, max_tokens, response_format=None):
        raise self.error


class SlowVLM(MockVLM):
    async def chat(self, messages, image_urls, max_tokens, response_format=None):
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


@pytest.mark.parametrize("image_urls", [[], ["https://example.invalid/photo.jpg"]])
async def test_mock_chat_accepts_optional_response_format_without_changing_output(image_urls):
    vlm = MockVLM()
    messages = [{"role": "user", "content": json.dumps({"query": "바다 사진?"})}]
    response_format = deepcopy(prompts.PARSE_RESPONSE_FORMAT)
    original_format = deepcopy(response_format)

    original = await vlm.chat(messages, image_urls, max_tokens=512)
    without_format = await vlm.chat(messages, image_urls, max_tokens=512, response_format=None)
    with_format = await vlm.chat(
        messages, image_urls, max_tokens=512, response_format=response_format
    )

    assert original == without_format == with_format
    assert response_format == original_format
    if image_urls:
        assert original.startswith("[MOCK]")
    else:
        assert prompts.ParseOutput.model_validate_json(original).visual_query == "바다 사진?"


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


async def test_request_extra_fields_are_ignored_without_changing_scope():
    async with client_for(create_mock_context()) as client:
        response = await client.post(
            "/query/parse",
            json={"query": "사진", "reference_date": "2026-10-11", "trip_id": 1},
        )
        assert response.status_code == 200
        assert "trip_id" not in response.json()
        payload = search_body() | {"candidate_attachment_ids": [999], "new_field": "unused"}
        payload["candidates"][0]["new_field"] = "unused"
        response = await client.post("/query/search", json=payload)
        assert response.status_code == 200
        assert [item["trip_attachment_id"] for item in response.json()["attachments"]] == [1, 2]
        del payload["candidates"]
        assert (await client.post("/query/search", json=payload)).status_code == 422


@pytest.mark.parametrize(
    "model,payload",
    [
        (
            ParseResponse,
            {
                "intent": "SEARCH",
                "date_from": None,
                "date_to": None,
                "region_names": None,
                "visual_query": None,
                "parsed": True,
                "unexpected": 1,
            },
        ),
        (SearchResponse, {"attachments": [], "unexpected": 1}),
        (
            SearchResponse,
            {"attachments": [{"trip_attachment_id": 1, "score": 0.9, "unexpected": 1}]},
        ),
    ],
)
def test_response_extra_fields_remain_forbidden(model, payload):
    with pytest.raises(ValidationError) as error:
        model.model_validate(payload)
    assert error.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    "output",
    [
        "not json",
        '{"intent":"OTHER"}',
        json.dumps(
            {
                "intent": "SEARCH",
                "date_from": None,
                "date_to": None,
                "region_names": None,
                "visual_query": None,
                "parsed": True,
            }
        ),
        json.dumps(
            {
                "intent": "SEARCH",
                "date_from": "2026/09/01",
                "date_to": None,
                "region_names": None,
                "visual_query": None,
            }
        ),
    ],
)
async def test_invalid_vlm_text_uses_parse_fallback(output):
    class VLM:
        async def chat(self, messages, image_urls, max_tokens, response_format=None):
            return output

    context = replace(create_mock_context(), vlm=VLM())
    async with client_for(context) as client:
        response = await client.post(
            "/query/parse", json={"query": "지난달 바다", "reference_date": "2026-10-11"}
        )
        assert response.status_code == 200
        assert response.json()["parsed"] is False
        assert response.json()["visual_query"] == "지난달 바다"


PARSE_REQUEST = ParseRequest(query="지난달 속초 바다", reference_date="2026-10-11")


def model_output(**updates):
    output = {
        "intent": "SEARCH",
        "date_from": None,
        "date_to": None,
        "region_names": None,
        "visual_query": "바다",
    }
    return json.dumps(output | updates, ensure_ascii=False)


def test_response_format_matches_output_model_without_parsed():
    schema = prompts.PARSE_RESPONSE_FORMAT["json_schema"]["schema"]
    fields = set(prompts.ParseOutput.model_fields)
    assert set(schema["required"]) == set(schema["properties"]) == fields
    assert "parsed" not in fields


def test_parse_output_normalizes_empty_values_and_sets_parsed():
    result = prompts.parse_output(
        model_output(region_names=[" 속초 ", "", "속초", "강릉"], visual_query="  "),
        PARSE_REQUEST,
    )
    assert result.region_names == ["속초", "강릉"]
    assert result.visual_query is None
    assert result.parsed is True
    empty = prompts.parse_output(model_output(region_names=[" "]), PARSE_REQUEST)
    assert empty.region_names is None


@pytest.mark.parametrize(
    "date_from,date_to,expected",
    [
        ("2025-09-01", "2025-11-30", ("2025-09-01", "2025-11-30")),
        (None, "2026-09-30", (None, "2026-09-30")),
        # 오늘 뒤의 상한은 오늘로
        ("2026-10-01", "2026-10-31", ("2026-10-01", "2026-10-11")),
        # 자른 뒤에도 역순, 상한 없는 미래 시작, 역순, 달력에 없는 날짜
        ("2026-12-01", "2027-02-28", (None, None)),
        ("2026-10-20", None, (None, None)),
        ("2026-09-30", "2026-09-01", (None, None)),
        ("2025-12-01", "2026-02-29", (None, None)),
    ],
)
def test_parse_output_clears_only_invalid_dates(date_from, date_to, expected):
    result = prompts.parse_output(
        model_output(date_from=date_from, date_to=date_to, region_names=["속초"]),
        PARSE_REQUEST,
    )
    dumped = result.model_dump(mode="json")
    assert (dumped["date_from"], dumped["date_to"]) == expected
    assert dumped["region_names"] == ["속초"] and dumped["visual_query"] == "바다"
    assert dumped["parsed"] is True


async def test_vlm_implementation_bug_is_not_treated_as_invalid_model_text():
    class VLM:
        async def chat(self, messages, image_urls, max_tokens, response_format=None):
            # 전송 클라이언트 내부의 오류이며 반환된 모델 텍스트의 검증 실패가 아니다.
            return ParseResponse.model_validate({})

    async with client_for(replace(create_mock_context(), vlm=VLM())) as client:
        response = await client.post(
            "/query/parse", json={"query": "바다 사진", "reference_date": "2026-10-11"}
        )
        assert response.status_code == 500


async def test_shared_chat_receives_search_prompts_and_only_selected_images():
    calls = []

    class VLM(MockVLM):
        async def chat(self, messages, image_urls, max_tokens, response_format=None):
            calls.append((messages, image_urls, max_tokens, response_format))
            return await super().chat(
                messages, image_urls, max_tokens, response_format=response_format
            )

    class Images(MockImages):
        async def resolve(self, storage_key):
            return f"https://example.invalid/{storage_key}"

    context = replace(
        create_mock_context(),
        vlm=VLM(),
        images=Images(),
        vectors=FixedVectors([SearchHit(trip_attachment_id=2, score=0.9)]),
    )
    async with client_for(context) as client:
        response = await client.post(
            "/query/parse", json={"query": "바다 사진?", "reference_date": "2026-10-11"}
        )
        assert response.status_code == 200
        messages, urls, budget, response_format = calls[0]
        assert response_format == prompts.PARSE_RESPONSE_FORMAT
        assert urls == [] and budget > 0
        assert [item["role"] for item in messages] == ["system", "user"]
        assert json.loads(messages[-1]["content"])["reference_date"] == "2026-10-11"
        payload = search_body("QUESTION")
        payload["candidates"][1]["place_name"] = "협재해수욕장"
        response = await client.post("/query/search", json=payload)
        assert response.status_code == 200
        messages, urls, budget, response_format = calls[1]
        assert response_format is None
        data = json.loads(messages[-1]["content"])
        assert data["query"] == payload["query"]
        assert [item["trip_attachment_id"] for item in data["photos"]] == [2]
        assert data["photos"][0]["image_number"] == 1
        assert data["photos"][0]["place_name"] == "협재해수욕장"
        assert "analyze_storage_key" not in data["photos"][0]
        assert urls == ["https://example.invalid/trips/1/1.jpg"] and budget > 0


async def test_query_logs_distinguish_empty_results_failures_and_parse_outcomes(monkeypatch):
    from app.api.services import query_service

    events = []
    monkeypatch.setattr(query_service.log, "info", lambda message, *, extra: events.append(extra))
    async with client_for(create_mock_context()) as client:
        await client.post("/query/parse", json={"query": "사진?", "reference_date": "2026-10-11"})
        await client.post("/query/search", json=search_body(count=0))
    failing = replace(create_mock_context(), vlm=FailingVLM(VLMUnavailable("private URL")))
    async with client_for(failing) as client:
        await client.post("/query/parse", json={"query": "사진?", "reference_date": "2026-10-11"})
        await client.post("/query/search", json=search_body("QUESTION"))
    invalid = replace(
        create_mock_context(), vectors=FixedVectors([SearchHit(trip_attachment_id=999, score=1)])
    )
    async with client_for(invalid) as client:
        await client.post("/query/search", json=search_body())
    observed = [
        (e["result"], e["intent"], e["candidate_count"], e["returned_count"], e["parsed"])
        for e in events
    ]
    assert observed == [
        ("success", "QUESTION", None, None, True),
        ("success", "SEARCH", 0, 0, None),
        ("fallback", "SEARCH", None, None, False),
        ("partial", "QUESTION", 2, 2, None),
        ("error", "SEARCH", 2, None, None),
    ]
    assert all(e["event"] == "ai_query" and e["duration_ms"] >= 0 for e in events)
    serialized = json.dumps(events, ensure_ascii=False)
    assert "사진?" not in serialized and "private URL" not in serialized
    assert "trips/1/" not in serialized
