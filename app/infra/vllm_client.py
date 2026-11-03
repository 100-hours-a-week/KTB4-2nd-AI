"""vLLM의 비스트리밍 chat 전송. 프롬프트와 기능별 출력 검증은 호출자가 담당한다."""

import asyncio
import math
import time
from typing import Any, Self

import httpx

from app.core.config import Settings
from app.infra.vlm import ChatMessage, ChatTurn, ToolCall, VLMUnavailable
from app.observability.calls import log_model_call, usage_fields


class VllmClient:
    """프로세스별로 재사용하고 앱 종료 시 aclose한다. base_url은 /v1까지 포함한다.

    생성 요청은 자동 재시도하지 않는다. 응답을 못 받은 경우에도 서버에서는
    생성 중일 수 있으므로 재시도 여부와 전체 처리 예산은 호출자가 결정한다.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_sec: float = 60.0,
        connect_timeout_sec: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        url = httpx.URL(base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
        ):
            raise ValueError("VLM 주소는 인증 정보와 쿼리가 없는 HTTP(S) URL이어야 합니다.")
        if not model.strip():
            raise ValueError("VLM 모델명이 필요합니다.")
        if any(
            not math.isfinite(value) or value <= 0 for value in (timeout_sec, connect_timeout_sec)
        ):
            raise ValueError("VLM 시간 제한은 양의 유한 값이어야 합니다.")
        self._model = model
        self._timeout_sec = timeout_sec
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(timeout_sec, connect=connect_timeout_sec),
            follow_redirects=False,
            transport=transport,
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> Self:
        if not settings.VLM_BASE_URL or not settings.VLM_MODEL:
            raise ValueError("VLM_BASE_URL과 VLM_MODEL을 설정해야 합니다.")
        return cls(
            settings.VLM_BASE_URL,
            settings.VLM_MODEL,
            api_key=settings.VLM_API_KEY.get_secret_value() if settings.VLM_API_KEY else None,
            timeout_sec=settings.VLM_TIMEOUT_SEC,
            connect_timeout_sec=settings.VLM_CONNECT_TIMEOUT_SEC,
            transport=transport,
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        image_urls: list[str],
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> ChatTurn:
        if tools is None and tool_choice is not None:
            raise ValueError("도구 없이 tool_choice를 지정할 수 없습니다.")
        if tools is not None and (not isinstance(tools, list) or not tools):
            raise ValueError("도구 목록이 비어 있지 않은 목록이어야 합니다.")
        body, call = self._request(messages, image_urls, max_tokens, temperature)
        if response_format is not None:
            body["response_format"] = response_format
        call["structured"] = response_format is not None
        if tools is not None:
            body["tools"] = tools
            body["tool_choice"] = "auto" if tool_choice is None else tool_choice
            call["tool_count"] = len(tools)
        started = time.monotonic()
        response = await self._post(body, call, started)
        finish_reason, usage = None, None
        try:
            data = response.json()
            usage = data.get("usage") if isinstance(data, dict) else None
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            message = choice["message"]
            content = message["content"] if tools is None else message.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("텍스트가 아님")
            tool_calls = (
                []
                if tools is None
                else [
                    ToolCall(
                        id=_text(item["id"]),
                        name=_text(item["function"]["name"]),
                        arguments=_text(item["function"]["arguments"], allow_empty=True),
                    )
                    for item in message.get("tool_calls") or []
                ]
            )
            # 도구 호출이 있으면 vLLM은 tool_calls나 stop으로 끝낸다. 없으면 텍스트가 답이다.
            if finish_reason not in {"stop", "tool_calls"} or (
                not tool_calls and (finish_reason != "stop" or not content.strip())
            ):
                raise ValueError("완성된 응답이 아님")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            # finish_reason=length는 max_tokens에 도달해 출력이 잘린 경우다.
            log_model_call(
                "vlm",
                started,
                "incomplete_output",
                status_code=response.status_code,
                finish_reason=finish_reason if isinstance(finish_reason, str) else None,
                **usage_fields(usage),
                **call,
            )
            raise VLMUnavailable("VLM이 완성된 응답을 반환하지 않았습니다.") from None
        if tools is not None:
            call["tool_call_count"] = len(tool_calls)
        log_model_call(
            "vlm",
            started,
            "success",
            status_code=response.status_code,
            finish_reason=finish_reason,
            output_chars=len(content),
            **usage_fields(usage),
            **call,
        )
        return ChatTurn(content=content, tool_calls=tool_calls, finish_reason=finish_reason)

    def _request(
        self,
        messages: list[ChatMessage],
        image_urls: list[str],
        max_tokens: int,
        temperature: float | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            not messages
            or not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 1
        ):
            raise ValueError("메시지와 양의 정수 max_tokens가 필요합니다.")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature는 0 이상 2 이하의 유한한 수여야 합니다.")
        body = {
            "model": self._model,
            "messages": _wire_messages(messages, image_urls),
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            # 미지정이면 생성 서버가 모델 generation_config의 권장값으로 샘플링한다.
            body["temperature"] = temperature
        call = {
            "image_count": len(image_urls),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return body, call

    async def _post(
        self, body: dict[str, Any], call: dict[str, Any], started: float
    ) -> httpx.Response:
        try:
            # HTTPX의 I/O별 제한 외에 응답 본문을 모두 받기까지의 전체 시간도 제한한다.
            async with asyncio.timeout(self._timeout_sec):
                response = await self._client.post("chat/completions", json=body)
                response.raise_for_status()
        except (httpx.TimeoutException, TimeoutError):
            log_model_call("vlm", started, "timeout", **call)
            raise TimeoutError("VLM 요청 시간이 초과되었습니다.") from None
        except asyncio.CancelledError:
            # 호출자의 더 짧은 시간 제한(검색 단계별 제한 등)이나 작업 취소로 중단된 경우다.
            log_model_call("vlm", started, "cancelled", **call)
            raise
        except httpx.HTTPStatusError as exc:
            log_model_call(
                "vlm", started, "http_error", status_code=exc.response.status_code, **call
            )
            raise VLMUnavailable("VLM 요청에 실패했습니다.") from None
        except httpx.HTTPError as exc:
            # 인증키, presigned URL과 모델 응답 본문을 오류 메시지에 포함하지 않는다.
            log_model_call("vlm", started, "connect_error", error_type=type(exc).__name__, **call)
            raise VLMUnavailable("VLM 요청에 실패했습니다.") from None
        return response

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _text(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or not (allow_empty or value):
        raise ValueError("문자열이 아님")
    return value


def _wire_messages(messages: list[ChatMessage], image_urls: list[str]) -> list[dict[str, Any]]:
    """호출자의 메시지를 바꾸지 않고 전송 형식으로 옮긴다. 이미지는 user 메시지에만 붙인다."""
    wire: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message["role"], "content": message["content"]}
        # OpenAI 규격은 빈 tool_calls 배열을 받지 않으므로 호출이 있을 때만 보낸다.
        if message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in message["tool_calls"]
            ]
        if "tool_call_id" in message:
            item["tool_call_id"] = message["tool_call_id"]
        wire.append(item)
    users = [i for i, message in enumerate(messages) if message["role"] == "user"]
    slots = [
        i
        for i, message in enumerate(messages)
        if not isinstance(message["content"], str)
        for part in message["content"]
        if part["type"] == "image"
    ]
    if slots:
        # 호출자가 이미지 자리를 지정했다. 텍스트 라벨과 이미지의 배치는 호출자가 정한다.
        if any(i not in users for i in slots) or len(slots) != len(image_urls):
            raise ValueError("이미지 자리는 user 메시지에 이미지 수만큼 있어야 합니다.")
        urls = iter(image_urls)
        for i in sorted(set(slots)):
            wire[i]["content"] = [
                {"type": "image_url", "image_url": {"url": next(urls)}}
                if part["type"] == "image"
                else dict(part)
                for part in messages[i]["content"]
            ]
    elif image_urls:
        if not users:
            raise ValueError("이미지를 첨부할 user 메시지가 없습니다.")
        index = users[-1]
        if not isinstance(messages[index]["content"], str):
            raise ValueError("이미지 자리는 user 메시지에 이미지 수만큼 있어야 합니다.")
        wire[index]["content"] = [
            {"type": "text", "text": messages[index]["content"]},
            *({"type": "image_url", "image_url": {"url": url}} for url in image_urls),
        ]
    return wire
