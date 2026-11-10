"""vLLM의 chat 전송. 프롬프트와 기능별 출력 검증은 호출자가 담당한다.

기본은 비스트리밍이다. stream=True는 응답 조각을 모아 같은 결과로 돌려준다. RunPod 프록시는
응답 데이터가 오가지 않는 요청을 약 100초 뒤에 끊으므로, 오래 걸리는 호출만 스트리밍으로 보낸다.
"""

import asyncio
import json
import math
import time
from typing import Any, Self

import httpx

from app.core.config import Settings
from app.infra.vlm import ChatMessage, ChatTurn, ToolCall, VLMTransientError, VLMUnavailable
from app.observability.calls import log_model_call, usage_fields


class VllmClient:
    """프로세스별로 재사용하고 앱 종료 시 aclose한다. base_url은 /v1까지 포함한다.

    생성 요청은 자동 재시도하지 않는다. 응답을 못 받은 경우에도 서버에서는
    생성 중일 수 있으므로 재시도 여부와 전체 처리 예산은 호출자가 결정한다.
    전송 오류와 HTTP 408, 429, 5xx는 VLMTransientError로 구분해 호출자가 고를 수 있게 한다.
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
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatTurn:
        """extra_body는 요청 본문에 더할 vLLM 전용 값(예: mm_processor_kwargs)이다.

        본문이 이미 가진 키(model, messages, max_tokens, stream 등)는 덮어쓰지 않는다.
        """
        if tools is None and tool_choice is not None:
            raise ValueError("도구 없이 tool_choice를 지정할 수 없습니다.")
        if tools is not None and (not isinstance(tools, list) or not tools):
            raise ValueError("도구 목록이 비어 있지 않은 목록이어야 합니다.")
        if stream and tools is not None:
            raise ValueError("도구 호출은 스트리밍으로 보내지 않습니다.")
        body, call = self._request(messages, image_urls, max_tokens, temperature)
        if extra_body:
            if clash := set(extra_body) & set(body):
                raise ValueError(f"extra_body가 본문의 키를 덮어씀: {sorted(clash)}")
            body.update(extra_body)
        if response_format is not None:
            body["response_format"] = response_format
        call["structured"] = response_format is not None
        if tools is not None:
            body["tools"] = tools
            body["tool_choice"] = "auto" if tool_choice is None else tool_choice
            call["tool_count"] = len(tools)
        started = time.monotonic()
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            call["stream"] = True
            status_code, data = await self._post_stream(body, call, started)
        else:
            response = await self._post(body, call, started)
            status_code, data = response.status_code, None
        finish_reason, usage = None, None
        try:
            if data is None:
                if stream:
                    raise ValueError("해석할 수 없는 조각")
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
                status_code=status_code,
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
            status_code=status_code,
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
            status = exc.response.status_code
            log_model_call("vlm", started, "http_error", status_code=status, **call)
            error = VLMTransientError if status in (408, 429) or status >= 500 else VLMUnavailable
            raise error("VLM 요청에 실패했습니다.") from None
        except httpx.HTTPError as exc:
            # 인증키, presigned URL과 모델 응답 본문을 오류 메시지에 포함하지 않는다.
            log_model_call("vlm", started, "connect_error", error_type=type(exc).__name__, **call)
            error = VLMTransientError if isinstance(exc, httpx.TransportError) else VLMUnavailable
            raise error("VLM 요청에 실패했습니다.") from None
        return response

    async def _post_stream(
        self, body: dict[str, Any], call: dict[str, Any], started: float
    ) -> tuple[int, dict[str, Any] | None]:
        """SSE 조각의 content를 이어 붙여 비스트리밍 응답과 같은 모양으로 만든다.

        해석할 수 없는 조각이 있으면 None을 돌려주어 호출자가 불완전한 응답으로 처리한다.
        """
        text: list[str] = []
        finish_reason, usage, broken = None, None, False
        try:
            # 조각이 계속 와도 응답을 모두 받기까지의 전체 시간은 비스트리밍과 같이 제한한다.
            async with asyncio.timeout(self._timeout_sec):
                async with self._client.stream("POST", "chat/completions", json=body) as response:
                    response.raise_for_status()
                    status_code = response.status_code
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            usage = chunk.get("usage") or usage
                            for choice in chunk.get("choices") or []:
                                piece = (choice.get("delta") or {}).get("content")
                                if piece is not None and not isinstance(piece, str):
                                    raise TypeError("텍스트가 아닌 조각")
                                if piece:
                                    text.append(piece)
                                finish_reason = choice.get("finish_reason") or finish_reason
                        except (ValueError, AttributeError, TypeError):
                            broken = True
        except (httpx.TimeoutException, TimeoutError):
            log_model_call("vlm", started, "timeout", **call)
            raise TimeoutError("VLM 요청 시간이 초과되었습니다.") from None
        except asyncio.CancelledError:
            log_model_call("vlm", started, "cancelled", **call)
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            log_model_call("vlm", started, "http_error", status_code=status, **call)
            error = VLMTransientError if status in (408, 429) or status >= 500 else VLMUnavailable
            raise error("VLM 요청에 실패했습니다.") from None
        except httpx.HTTPError as exc:
            log_model_call("vlm", started, "connect_error", error_type=type(exc).__name__, **call)
            error = VLMTransientError if isinstance(exc, httpx.TransportError) else VLMUnavailable
            raise error("VLM 요청에 실패했습니다.") from None
        if broken:
            return status_code, None
        message = {"role": "assistant", "content": "".join(text)}
        return status_code, {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }

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
