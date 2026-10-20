"""vLLM의 비스트리밍 chat 전송. 프롬프트와 기능별 출력 검증은 호출자가 담당한다."""

import asyncio
import math
import time
from typing import Any, Self

import httpx

from app.core.config import Settings
from app.infra.vlm import ChatMessage, VLMUnavailable
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
        temperature: float | None = None,
    ) -> str:
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
        # 호출자의 메시지를 바꾸지 않는다. 이미지는 마지막 user 메시지에만 붙인다.
        wire_messages: list[dict[str, Any]] = [dict(message) for message in messages]
        if image_urls:
            index = next(
                (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"),
                None,
            )
            if index is None:
                raise ValueError("이미지를 첨부할 user 메시지가 없습니다.")
            wire_messages[index]["content"] = [
                {"type": "text", "text": messages[index]["content"]},
                *({"type": "image_url", "image_url": {"url": url}} for url in image_urls),
            ]
        body = {
            "model": self._model,
            "messages": wire_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if temperature is not None:
            # 미지정이면 생성 서버가 모델 generation_config의 권장값으로 샘플링한다.
            body["temperature"] = temperature
        call = {
            "image_count": len(image_urls),
            "max_tokens": max_tokens,
            "structured": response_format is not None,
            "temperature": temperature,
        }
        started = time.monotonic()
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
        finish_reason, usage = None, None
        try:
            data = response.json()
            usage = data.get("usage") if isinstance(data, dict) else None
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            content = choice["message"]["content"]
            if finish_reason != "stop" or not isinstance(content, str) or not content.strip():
                raise ValueError("완성된 텍스트가 아님")
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
            raise VLMUnavailable("VLM이 완성된 텍스트 응답을 반환하지 않았습니다.") from None
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
        return content

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
