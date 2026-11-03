"""실제 GPU 없이 VLM 전송 계약과 검색에서의 실패 처리를 확인한다."""

import asyncio
import json
import logging
import math
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest
from pydantic import ValidationError

from app.api.services.query_service import QueryService
from app.core.config import Settings
from app.infra.vllm_client import VllmClient
from app.infra.vlm import ChatTurn, VLMUnavailable
from app.schemas.query import ParseRequest, SearchRequest
from app.search.mock import create_mock_context

pytestmark = pytest.mark.anyio
MESSAGES = [{"role": "user", "content": "사진을 설명해 주세요."}]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def completion(content="응답", finish_reason="stop"):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ]
    }


def client_for(handler, **kwargs):
    return VllmClient(
        "https://vlm.example/proxy/v1",
        "test-model",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.parametrize(
    "response_format",
    [
        None,
        {},
        {"type": "json_schema", "json_schema": {"name": "result", "schema": {"type": "object"}}},
    ],
)
async def test_text_request_preserves_output_format_and_does_not_parse_model_text(response_format):
    before = deepcopy(response_format)

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/proxy/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-secret"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"] == MESSAGES
        assert body["max_tokens"] == 256
        assert body["stream"] is False
        if response_format is None:
            assert "response_format" not in body
        else:
            assert body["response_format"] == response_format
        # JSON 스키마가 있어도 출력 텍스트의 기능별 검증은 호출자가 담당한다.
        return httpx.Response(200, json=completion("not JSON"))

    async with client_for(handler, api_key="test-secret") as client:
        assert await client.chat(MESSAGES, [], 256, response_format) == ChatTurn("not JSON")
    assert response_format == before


@pytest.mark.parametrize("max_tokens", [-1, 0, True, False, 1.5, "256", None])
async def test_invalid_max_tokens_fails_before_http_request(max_tokens):
    def handler(request):
        pytest.fail("유효하지 않은 max_tokens로 HTTP 요청을 보내면 안 됩니다.")

    async with client_for(handler) as client:
        with pytest.raises(ValueError, match="양의 정수 max_tokens"):
            await client.chat(MESSAGES, [], max_tokens)


@pytest.mark.parametrize("temperature", [None, 0, 0.7, 2])
async def test_temperature_is_sent_only_when_given_and_logged(temperature, caplog):
    caplog.set_level(logging.INFO, logger="app.observability.calls")

    def handler(request):
        body = json.loads(request.content)
        if temperature is None:
            # 미지정이면 생성 서버의 모델 권장값을 쓰도록 본문에서 뺀다.
            assert "temperature" not in body
        else:
            assert body["temperature"] == temperature
        return httpx.Response(200, json=completion())

    options = {} if temperature is None else {"temperature": temperature}
    async with client_for(handler) as client:
        assert await client.chat(MESSAGES, [], 64, **options) == ChatTurn("응답")
    calls = [r for r in caplog.records if getattr(r, "event", None) == "ai_model_call"]
    assert [r.temperature for r in calls] == [temperature]


@pytest.mark.parametrize("temperature", [-0.1, 2.1, math.nan, math.inf, True, "0"])
async def test_invalid_temperature_fails_before_http_request(temperature):
    def handler(request):
        pytest.fail("유효하지 않은 temperature로 HTTP 요청을 보내면 안 됩니다.")

    async with client_for(handler) as client:
        with pytest.raises(ValueError, match="temperature"):
            await client.chat(MESSAGES, [], 64, temperature=temperature)


async def test_images_attach_to_last_user_in_order_without_mutating_messages():
    messages = [
        {"role": "system", "content": "지침"},
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 답"},
        {"role": "user", "content": "현재 질문"},
        {"role": "assistant", "content": "접두사"},
    ]
    before = deepcopy(messages)
    urls = ["https://s3.example/b.jpg?signature=secret", "https://s3.example/a.jpg"]

    def handler(request):
        assert "authorization" not in request.headers
        wire = json.loads(request.content)["messages"]
        assert wire[:3] == messages[:3] and wire[4] == messages[4]
        assert wire[3]["content"] == [
            {"type": "text", "text": "현재 질문"},
            *({"type": "image_url", "image_url": {"url": url}} for url in urls),
        ]
        return httpx.Response(200, json=completion())

    async with client_for(handler) as client:
        assert await client.chat(messages, urls, 256) == ChatTurn("응답")
    assert messages == before


async def test_image_slots_are_filled_in_order_between_caller_text():
    messages = [
        {"role": "system", "content": "지침"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "사진 1:"},
                {"type": "image"},
                {"type": "text", "text": "사진 2:"},
                {"type": "image"},
                {"type": "text", "text": "질문"},
            ],
        },
    ]
    before = deepcopy(messages)
    urls = ["https://s3.example/b.jpg", "https://s3.example/a.jpg"]

    def handler(request):
        wire = json.loads(request.content)["messages"]
        assert wire[0] == messages[0]
        assert wire[1]["content"] == [
            {"type": "text", "text": "사진 1:"},
            {"type": "image_url", "image_url": {"url": urls[0]}},
            {"type": "text", "text": "사진 2:"},
            {"type": "image_url", "image_url": {"url": urls[1]}},
            {"type": "text", "text": "질문"},
        ]
        return httpx.Response(200, json=completion())

    async with client_for(handler) as client:
        assert await client.chat(messages, urls, 256) == ChatTurn("응답")
    assert messages == before


async def test_image_slots_in_earlier_user_messages_are_filled_in_message_order():
    # 정리 에이전트는 도구로 읽은 사진을 다음 user 메시지에 붙이며 대화를 이어 간다.
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "대상"}, {"type": "image"}]},
        {"role": "assistant", "content": "확인"},
        {"role": "user", "content": [{"type": "image"}, {"type": "image"}]},
    ]
    before = deepcopy(messages)
    urls = ["https://s3.example/1.jpg", "https://s3.example/2.jpg", "https://s3.example/3.jpg"]

    def handler(request):
        wire = json.loads(request.content)["messages"]
        assert wire[0]["content"][1] == {"type": "image_url", "image_url": {"url": urls[0]}}
        assert wire[1] == messages[1]
        assert [part["image_url"]["url"] for part in wire[2]["content"]] == urls[1:]
        return httpx.Response(200, json=completion())

    async with client_for(handler) as client:
        assert await client.chat(messages, urls, 256) == ChatTurn("응답")
    assert messages == before


@pytest.mark.parametrize(
    "messages,urls",
    [
        # 자리 수와 URL 수가 다르다.
        ([{"role": "user", "content": [{"type": "image"}]}], ["https://a", "https://b"]),
        ([{"role": "user", "content": [{"type": "image"}, {"type": "image"}]}], ["https://a"]),
        ([{"role": "user", "content": [{"type": "image"}]}], []),
        # user가 아닌 메시지에 자리가 있다.
        (
            [
                {"role": "assistant", "content": [{"type": "image"}]},
                {"role": "user", "content": "현재 질문"},
            ],
            ["https://a"],
        ),
        # 목록 content에 자리 없이 URL만 넘긴다.
        ([{"role": "user", "content": [{"type": "text", "text": "질문"}]}], ["https://a"]),
    ],
)
async def test_mismatched_image_slots_are_rejected_before_network(messages, urls):
    def handler(request):
        pytest.fail("잘못된 입력으로 HTTP 요청하면 안 됨")

    async with client_for(handler) as client:
        with pytest.raises(ValueError, match="이미지 자리"):
            await client.chat(messages, urls, 128)


@pytest.mark.parametrize("status", [302, 400, 401, 429, 500, 503])
async def test_http_errors_are_not_retried_or_exposed(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status, text="private model input", headers={"Location": "https://other.example"}
        )

    async with client_for(handler) as client:
        with pytest.raises(VLMUnavailable) as error:
            await client.chat(MESSAGES, [], 256)
    assert len(calls) == 1
    assert "private" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    "failure, expected",
    [
        (httpx.ConnectError, VLMUnavailable),
        (httpx.ReadError, VLMUnavailable),
        (httpx.ConnectTimeout, TimeoutError),
        (httpx.ReadTimeout, TimeoutError),
        (httpx.WriteTimeout, TimeoutError),
        (httpx.PoolTimeout, TimeoutError),
    ],
)
async def test_transport_errors_map_to_common_contract(failure, expected):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise failure("private endpoint", request=request)

    async with client_for(handler) as client:
        with pytest.raises(expected) as error:
            await client.chat(MESSAGES, [], 256)
    assert calls == 1
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": None},
        None,
        {"choices": [{"message": {"content": "text"}}]},
        completion(None),
        completion([]),
        completion(" "),
        completion("partial", "length"),
        completion("text", "tool_calls"),
    ],
)
async def test_malformed_or_incomplete_completion_is_unavailable(payload):
    async with client_for(lambda _: httpx.Response(200, json=payload)) as client:
        with pytest.raises(VLMUnavailable):
            await client.chat(MESSAGES, [], 256)


async def test_non_json_http_response_is_unavailable():
    async with client_for(lambda _: httpx.Response(200, text="<html>proxy error</html>")) as client:
        with pytest.raises(VLMUnavailable):
            await client.chat(MESSAGES, [], 256)


async def test_total_deadline_cancels_inflight_request():
    canceled = asyncio.Event()

    async def handler(request):
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    async with client_for(handler, timeout_sec=0.02) as client:
        with pytest.raises(TimeoutError):
            await client.chat(MESSAGES, [], 256)
    assert canceled.is_set()


async def test_caller_cancellation_is_not_converted_to_vlm_error():
    started = asyncio.Event()
    canceled = asyncio.Event()

    async def handler(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    async with client_for(handler) as client:
        task = asyncio.create_task(client.chat(MESSAGES, [], 256))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert canceled.is_set()


async def test_connection_transport_is_reused_and_closed():
    class Transport(httpx.AsyncBaseTransport):
        calls = 0
        closed = False

        async def handle_async_request(self, request):
            self.calls += 1
            return httpx.Response(200, json=completion())

        async def aclose(self):
            self.closed = True

    transport = Transport()
    async with VllmClient("http://local/v1/", "model", transport=transport) as client:
        await client.chat(MESSAGES, [], 128)
        await client.chat(MESSAGES, [], 128)
        assert not transport.closed
    assert transport.calls == 2 and transport.closed


async def test_settings_factory_uses_separate_vlm_credentials_and_timeout():
    settings = Settings(
        _env_file=None,
        API_KEY="be-secret",
        IMAGE_SOURCE="local",
        VLM_BASE_URL="http://local/v1",
        VLM_MODEL="served-name",
        VLM_API_KEY="vlm-secret",
        VLM_TIMEOUT_SEC=12,
        VLM_CONNECT_TIMEOUT_SEC=2,
    )
    assert "vlm-secret" not in repr(settings)

    def handler(request):
        assert request.headers["Authorization"] == "Bearer vlm-secret"
        assert json.loads(request.content)["model"] == "served-name"
        assert request.extensions["timeout"]["connect"] == 2
        assert request.extensions["timeout"]["read"] == 12
        return httpx.Response(200, json=completion())

    async with VllmClient.from_settings(settings, transport=httpx.MockTransport(handler)) as client:
        await client.chat(MESSAGES, [], 128)


def test_v1_settings_can_omit_vlm_but_factory_requires_connection_info(monkeypatch):
    for name in ("VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, API_KEY="be-key", IMAGE_SOURCE="local")
    with pytest.raises(ValueError, match="VLM_BASE_URL"):
        VllmClient.from_settings(settings)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_time_budget_is_rejected(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, API_KEY="key", IMAGE_SOURCE="local", VLM_TIMEOUT_SEC=value)
    with pytest.raises(ValueError):
        VllmClient("http://local/v1", "model", timeout_sec=value)


async def test_missing_user_message_is_rejected_before_network():
    def handler(request):
        pytest.fail("잘못된 입력으로 HTTP 요청하면 안 됨")

    async with client_for(handler) as client:
        with pytest.raises(ValueError, match="user"):
            await client.chat([{"role": "system", "content": "지침"}], ["https://image"], 128)


@pytest.mark.parametrize("timeout", [False, True])
async def test_search_pipeline_keeps_fallback_and_evidence_when_real_client_fails(timeout):
    def handler(request):
        if timeout:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(503, text="unavailable")

    async with client_for(handler) as client:
        context = replace(create_mock_context(), vlm=client)
        service = QueryService(context)
        parsed = await service.parse(ParseRequest(query="바다 사진", reference_date="2026-10-12"))
        assert parsed.parsed is False and parsed.visual_query == "바다 사진"
        response = await service.search(
            SearchRequest(
                query="어디에 갔지?",
                intent="QUESTION",
                visual_query=None,
                candidates=[
                    {
                        "trip_attachment_id": 1,
                        "analyze_storage_key": "trips/1/a.jpg",
                        "taken_at": None,
                        "place_name": None,
                        "trip_name": None,
                    }
                ],
            )
        )
        assert response.answer_error == ("TIMEOUT" if timeout else "VLM_UNAVAILABLE")
        assert [photo.trip_attachment_id for photo in response.attachments] == [1]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_photo",
            "description": "사진 읽기",
            "parameters": {"type": "object", "properties": {"ids": {"type": "array"}}},
        },
    }
]


def tool_completion(calls, content=None, finish_reason="tool_calls"):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "tool_calls": calls},
                "finish_reason": finish_reason,
            }
        ]
    }


def wire_call(call_id="call-1", name="read_photo", arguments='{"ids": ["S1"]}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


async def test_tool_request_sends_history_and_returns_tool_calls(caplog):
    caplog.set_level(logging.INFO, logger="app.observability.calls")
    messages = [
        {"role": "system", "content": "지침"},
        {"role": "user", "content": "사진 S1의 장소는?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-0", "name": "read_photo", "arguments": "{}"}],
        },
        {"role": "tool", "tool_call_id": "call-0", "content": "다음 메시지에 첨부"},
        {"role": "user", "content": [{"type": "text", "text": "S1:"}, {"type": "image"}]},
    ]
    before = deepcopy(messages)

    def handler(request):
        body = json.loads(request.content)
        assert body["tools"] == TOOLS and body["tool_choice"] == "required"
        assert "response_format" not in body
        wire = body["messages"]
        assert wire[2]["tool_calls"] == [wire_call("call-0", arguments="{}")]
        assert wire[3] == messages[3]
        assert wire[4]["content"][1] == {"type": "image_url", "image_url": {"url": "https://a"}}
        return httpx.Response(200, json=tool_completion([wire_call(), wire_call("call-2")]))

    async with client_for(handler) as client:
        turn = await client.chat(messages, ["https://a"], 256, tools=TOOLS, tool_choice="required")
    assert turn == ChatTurn(
        content="",
        tool_calls=[
            {"id": "call-1", "name": "read_photo", "arguments": '{"ids": ["S1"]}'},
            {"id": "call-2", "name": "read_photo", "arguments": '{"ids": ["S1"]}'},
        ],
        finish_reason="tool_calls",
    )
    assert messages == before
    calls = [r for r in caplog.records if getattr(r, "event", None) == "ai_model_call"]
    assert [(r.result, r.tool_count, r.tool_call_count) for r in calls] == [("success", 1, 2)]


async def test_tool_turn_without_calls_returns_final_text():
    payload = tool_completion(None, content="최종 답", finish_reason="stop")
    async with client_for(lambda _: httpx.Response(200, json=payload)) as client:
        turn = await client.chat(MESSAGES, [], 64, tools=TOOLS)
    assert turn == ChatTurn(content="최종 답")


@pytest.mark.parametrize(
    "payload",
    [
        tool_completion(None, content="잘림", finish_reason="length"),
        tool_completion([wire_call()], finish_reason="length"),
        tool_completion(None, finish_reason="tool_calls"),
        tool_completion([], content=" ", finish_reason="stop"),
        tool_completion([{"function": {"name": "read_photo", "arguments": "{}"}}]),
        tool_completion([wire_call(name="")]),
        tool_completion([wire_call(arguments={"ids": []})]),
        tool_completion([wire_call()], content=["목록"]),
        {"choices": []},
    ],
)
async def test_incomplete_tool_turn_is_unavailable(payload):
    async with client_for(lambda _: httpx.Response(200, json=payload)) as client:
        with pytest.raises(VLMUnavailable):
            await client.chat(MESSAGES, [], 64, tools=TOOLS)


@pytest.mark.parametrize("tools", [[], {"type": "function"}])
async def test_tool_request_without_tools_fails_before_network(tools):
    def handler(request):
        pytest.fail("도구 없이 HTTP 요청하면 안 됨")

    async with client_for(handler) as client:
        with pytest.raises(ValueError, match="도구"):
            await client.chat(MESSAGES, [], 64, tools=tools)


async def test_tool_calls_with_stop_reason_are_accepted():
    # tool_choice가 required이거나 함수를 지정하면 vLLM은 stop으로 끝내며 도구 호출을 준다.
    payload = tool_completion([wire_call()], finish_reason="stop")
    async with client_for(lambda _: httpx.Response(200, json=payload)) as client:
        turn = await client.chat(MESSAGES, [], 64, tools=TOOLS, tool_choice="required")
    assert [call["id"] for call in turn.tool_calls] == ["call-1"]
    assert turn.finish_reason == "stop"


async def test_image_slot_in_tool_message_is_rejected_before_network():
    messages = [
        {"role": "user", "content": "질문"},
        {"role": "tool", "tool_call_id": "call-0", "content": [{"type": "image"}]},
    ]

    def handler(request):
        pytest.fail("tool 메시지의 이미지로 HTTP 요청하면 안 됨")

    async with client_for(handler) as client:
        with pytest.raises(ValueError, match="이미지 자리"):
            await client.chat(messages, ["https://a"], 64, tools=TOOLS)


async def test_tool_request_timeout_keeps_common_error_and_logs_tool_count(caplog):
    caplog.set_level(logging.INFO, logger="app.observability.calls")

    def handler(request):
        raise httpx.ReadTimeout("느림", request=request)

    async with client_for(handler) as client:
        with pytest.raises(TimeoutError):
            await client.chat(MESSAGES, [], 64, tools=TOOLS)
    calls = [r for r in caplog.records if getattr(r, "event", None) == "ai_model_call"]
    assert [(r.result, r.tool_count) for r in calls] == [("timeout", 1)]


async def test_tool_choice_without_tools_fails_before_network():
    def handler(request):
        pytest.fail("도구 없이 tool_choice로 HTTP 요청하면 안 됨")

    async with client_for(handler) as client:
        with pytest.raises(ValueError, match="tool_choice"):
            await client.chat(MESSAGES, [], 64, tool_choice="required")


async def test_tool_choice_defaults_to_auto_when_tools_are_given():
    def handler(request):
        assert json.loads(request.content)["tool_choice"] == "auto"
        return httpx.Response(200, json=tool_completion([wire_call()]))

    async with client_for(handler) as client:
        turn = await client.chat(MESSAGES, [], 64, tools=TOOLS)
    assert turn.finish_reason == "tool_calls"


async def test_incomplete_tool_turn_logs_reason_and_tool_count(caplog):
    caplog.set_level(logging.INFO, logger="app.observability.calls")
    payload = tool_completion([wire_call()], finish_reason="length")
    async with client_for(lambda _: httpx.Response(200, json=payload)) as client:
        with pytest.raises(VLMUnavailable):
            await client.chat(MESSAGES, [], 64, tools=TOOLS)
    calls = [r for r in caplog.records if getattr(r, "event", None) == "ai_model_call"]
    assert [(r.result, r.finish_reason, r.tool_count) for r in calls] == [
        ("incomplete_output", "length", 1)
    ]


async def test_empty_tool_call_history_is_not_sent():
    messages = [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "답", "tool_calls": []},
        {"role": "user", "content": "다음 질문"},
    ]

    def handler(request):
        assert "tool_calls" not in json.loads(request.content)["messages"][1]
        return httpx.Response(200, json=completion())

    async with client_for(handler) as client:
        assert await client.chat(messages, [], 64) == ChatTurn("응답")
