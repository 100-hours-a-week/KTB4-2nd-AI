"""Langfuse 추적. LANGFUSE_PUBLIC_KEY와 LANGFUSE_SECRET_KEY가 모두 있을 때만 켜진다.

- 트레이스: 정리와 스토리 작업 1건, 질의 해석·검색·분위기 변경 요청 1건(start_root)
- span: stage()로 잰 단계
- generation: VllmClient.chat 호출 1건의 입력 메시지와 사진, 출력, 토큰, 슬롯 대기 시간

단계와 생성 호출은 기록 중인 트레이스 안에서만 남긴다. 트레이스 밖의 호출(사진 묶음 임베딩 등)이
각자 새 트레이스를 만들지 않게 하기 위해서다. 기록 실패는 경고 로그만 남기고 처리 결과를 바꾸지
않는다. OpenTelemetry 전역 provider를 쓰지 않고 별도 provider를 두어, 나중에 다른 계측이
들어와도 그 span이 Langfuse로 섞여 나가지 않게 한다.
"""

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Any

import httpx
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("app.observability.langfuse")

# 모델에 URL로 넘긴 사진을 Langfuse에 올릴 때의 상한. 넘으면 주소만 남긴다.
IMAGE_MAX_BYTES = 10 * 1024 * 1024
IMAGE_FETCH_TIMEOUT_SEC = 10.0
IMAGE_FETCH_CONCURRENCY = 8
# 실행기는 결과 저장 뒤 약 5~10초 안에 작업 프로세스를 끝낸다(app/jobs/runner.py의 supervise).
FLUSH_TIMEOUT_SEC = 5.0
# SDK 사진 업로드 큐에 이만큼 쌓이면 새 호출의 사진은 붙이지 않는다.
MEDIA_BACKLOG_LIMIT = 50


class LangfuseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        hide_input_in_errors=True,
    )

    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None
    # 미설정이면 SDK 기본값(https://cloud.langfuse.com)
    LANGFUSE_BASE_URL: str | None = None
    # Langfuse 화면의 환경 구분. 예: production, local
    LANGFUSE_TRACING_ENVIRONMENT: str | None = None
    RELEASE: str = "unknown"


_lock = threading.Lock()
_client: Any = None
_ready = False

# GPU 호출 슬롯을 얻기까지 기다린 시간. RequestGate가 호출 직전에 정하고 generation이 읽는다.
_slot_wait: ContextVar[float | None] = ContextVar("langfuse_slot_wait", default=None)
_call_name: ContextVar[str | None] = ContextVar("langfuse_call_name", default=None)


def client() -> Any:
    """프로세스당 한 번 만든 Langfuse 클라이언트. 꺼져 있거나 초기화에 실패하면 None"""
    global _client, _ready
    if _ready:
        return _client
    with _lock:
        if _ready:
            return _client
        try:
            cfg = LangfuseSettings()
            if cfg.LANGFUSE_PUBLIC_KEY and cfg.LANGFUSE_SECRET_KEY:
                from langfuse import Langfuse
                from opentelemetry.sdk.trace import TracerProvider

                _client = Langfuse(
                    public_key=cfg.LANGFUSE_PUBLIC_KEY,
                    secret_key=cfg.LANGFUSE_SECRET_KEY.get_secret_value(),
                    base_url=cfg.LANGFUSE_BASE_URL,
                    environment=cfg.LANGFUSE_TRACING_ENVIRONMENT,
                    release=cfg.RELEASE,
                    # 종료는 ashutdown()이 시간 제한을 두고 처리한다. OTel provider의 atexit 훅은
                    # 내보내기를 시간 제한 없이 기다리므로 등록하지 않는다.
                    tracer_provider=TracerProvider(shutdown_on_exit=False),
                    # 기본 1개로는 사진이 많은 정리 작업의 업로드가 밀린다.
                    media_upload_thread_count=4,
                )
        except Exception as exc:
            _client = None
            _warn("init", exc)
        _ready = True
    return _client


def reset() -> None:
    """테스트용. 다음 client() 호출이 설정을 다시 읽게 한다."""
    global _client, _ready
    with _lock:
        _client, _ready = None, False


def _warn(action: str, exc: BaseException) -> None:
    log.warning(
        "Langfuse 기록 실패",
        extra={"event": "ai_langfuse_failed", "action": action, "error_type": type(exc).__name__},
    )


def active() -> bool:
    """기록 중인 트레이스 안인지. 호출자가 기록할 값을 만들기 전에 확인한다."""
    try:
        return client() is not None and _recording()
    except Exception as exc:
        _warn("active", exc)
        return False


def _recording() -> bool:
    """현재 span이 이 모듈이 연 Langfuse span인지. 다른 계측의 span 아래에서는 기록하지 않는다."""
    from opentelemetry import trace as otel_trace

    span = otel_trace.get_current_span()
    scope = getattr(span, "instrumentation_scope", None)
    return span.is_recording() and getattr(scope, "name", "") == "langfuse-sdk"


@contextmanager
def call_name(name: str) -> Iterator[None]:
    """이 안의 생성 호출에 붙일 이름. 예: 워크플로우의 describe, vote1"""
    token = _call_name.set(name)
    try:
        yield
    finally:
        _call_name.reset(token)


def generation_name(stage: str | None) -> str:
    """단계 이름과 호출 이름을 이어 붙인다. 예: clustering.describe"""
    label = _call_name.get()
    return ".".join(part for part in (stage, label) if part) or "vlm"


@contextmanager
def slot_wait(seconds: float) -> Iterator[None]:
    """GPU 호출 슬롯 대기 시간을 이 호출의 generation에만 남긴다."""
    token = _slot_wait.set(seconds)
    try:
        yield
    finally:
        _slot_wait.reset(token)


class _Noop:
    trace_id = None

    def end(self, **_: Any) -> None:
        pass

    def update(self, **_: Any) -> None:
        pass


NOOP = _Noop()


# 작업 상태(COMPLETED 등)와 요청 결과(success 등). 나머지(FAILED, error, 오류 코드)는 ERROR다.
OK_STATUSES = {"COMPLETED", "success", "fallback"}
PARTIAL_STATUSES = {"partial", "cancelled", "CANCELED", "ABORTED"}


class Root:
    """트레이스 하나의 최상위 span. 만든 맥락에서 end()까지 현재 span으로 둔다."""

    def __init__(self, stack: ExitStack, span: Any) -> None:
        self._stack, self._span = stack, span
        self.trace_id: str | None = getattr(span, "trace_id", None)

    def update(self, **fields: Any) -> None:
        try:
            self._span.update(**fields)
        except Exception as exc:
            _warn("root_update", exc)

    def end(
        self,
        *,
        output: Any = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._stack is None:
            return
        try:
            fields: dict[str, Any] = {}
            if output is not None:
                fields["output"] = _jsonable(output)
            if metadata:
                fields["metadata"] = metadata
            if status is not None:
                fields["metadata"] = {**fields.get("metadata", {}), "status": status}
                if status not in OK_STATUSES:
                    fields["level"] = "WARNING" if status in PARTIAL_STATUSES else "ERROR"
                    fields["status_message"] = status
            if fields:
                self._span.update(**fields)
        except Exception as exc:
            _warn("root_end", exc)
        finally:
            stack, self._stack = self._stack, None
            try:
                stack.close()
            except Exception as exc:
                _warn("root_close", exc)


def start_root(
    name: str,
    *,
    trip_id: int | None = None,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Root | _Noop:
    """요청이나 작업 1건의 트레이스를 연다. 같은 여행의 트레이스는 Langfuse 세션으로 묶는다.

    반환값의 end()를 같은 맥락(같은 asyncio 태스크)에서 반드시 호출한다.
    """
    lf = client()
    if lf is None:
        return NOOP
    stack = ExitStack()
    try:
        from langfuse import propagate_attributes
        from opentelemetry import context as otel_context

        # 다른 계측의 span 아래에 Langfuse 트레이스가 붙지 않게 빈 맥락에서 시작한다.
        token = otel_context.attach(otel_context.Context())
        stack.callback(otel_context.detach, token)
        stack.enter_context(
            propagate_attributes(
                trace_name=name,
                session_id=f"trip-{trip_id}" if trip_id is not None else None,
            )
        )
        span = stack.enter_context(
            lf.start_as_current_observation(
                name=name,
                as_type="span",
                input=_jsonable(input),
                metadata={k: v for k, v in (metadata or {}).items() if v is not None},
            )
        )
        return Root(stack, span)
    except Exception as exc:
        try:
            stack.close()
        except Exception:
            pass
        _warn("root_start", exc)
        return NOOP


class Span:
    """stage() 단계 하나. 기록 중인 트레이스가 없으면 아무것도 하지 않는다."""

    def __init__(self, name: str) -> None:
        self._stack: ExitStack | None = None
        self._span: Any = None
        lf = client()
        if lf is None:
            return
        try:
            if not _recording():
                return
            stack = ExitStack()
            self._span = stack.enter_context(
                lf.start_as_current_observation(name=name, as_type="span")
            )
            self._stack = stack
        except Exception as exc:
            _warn("span_start", exc)

    def end(self, result: str, fields: dict[str, Any]) -> None:
        if self._stack is None:
            return
        try:
            update: dict[str, Any] = {
                "metadata": {"result": result, **{k: v for k, v in fields.items() if v is not None}}
            }
            if result != "success":
                update["level"] = "ERROR" if result == "error" else "WARNING"
                update["status_message"] = result
            self._span.update(**update)
        except Exception as exc:
            _warn("span_end", exc)
        finally:
            stack, self._stack = self._stack, None
            try:
                stack.close()
            except Exception as exc:
                _warn("span_close", exc)


class Generation:
    """VllmClient.chat 호출 1건. 기록 중인 트레이스 밖이면 None을 돌려주는 start()로 만든다.

    URL로 넘긴 사진은 호출과 동시에 내려받기 시작한다. 호출이 끝나면 종료 시각만 정해 두고,
    사진을 붙여 마무리하는 일은 별도 태스크가 한다. 호출자의 시간 제한과 GPU 슬롯이 사진
    내려받기를 기다리지 않게 하기 위해서다. presigned URL의 서명은 남기지 않는다.
    """

    def __init__(self, lf: Any, name: str, model: str, wire: list, params: dict, meta: dict):
        self._lf = lf
        # VllmClient가 응답의 usage를 채운다.
        self.usage: dict[str, Any] = {}
        wait = _slot_wait.get()
        if wait is not None:
            meta = {**meta, "gpu_wait_ms": round(wait * 1000, 2)}
        # 업로드가 밀려 있으면 사진을 붙이지 않는다. 밀린 사진 바이트가 메모리에 쌓이지 않게 한다.
        self._skip_images = _media_backlogged(lf)
        if self._skip_images:
            meta = {**meta, "images_omitted": "upload_backlog"}
        self._wire = _without_inline_images(wire) if self._skip_images else wire
        self._obs = lf.start_observation(
            name=name,
            as_type="generation",
            model=model,
            model_parameters=params,
            metadata=meta,
        )
        urls = sorted(
            {url for url in _image_urls(self._wire) if url.startswith(("http://", "https://"))}
        )
        fetch = urls and not self._skip_images
        self._fetch = asyncio.create_task(_fetch_images(urls)) if fetch else None

    @classmethod
    def start(
        cls,
        name: str,
        model: str,
        wire: list,
        params: dict[str, Any],
        meta: dict[str, Any],
    ) -> "Generation | None":
        lf = client()
        if lf is None:
            return None
        try:
            if not _recording():
                return None
            return cls(lf, name, model, wire, params, meta)
        except Exception as exc:
            _warn("generation_start", exc)
            return None

    def end(self, *, output: Any = None, error: BaseException | None = None) -> None:
        """기다리지 않고 바로 돌아온다. 예외를 올리지 않는다."""
        try:
            end_time = time.time_ns()
            update: dict[str, Any] = {}
            if output is not None:
                update["output"] = output
            if self.usage:
                update["usage_details"] = {
                    key: value
                    for key, value in {
                        "input": self.usage.get("prompt_tokens"),
                        "output": self.usage.get("completion_tokens"),
                        "total": self.usage.get("total_tokens"),
                    }.items()
                    if isinstance(value, int)
                }
            if error is not None:
                cancelled = isinstance(error, asyncio.CancelledError)
                update["level"] = "WARNING" if cancelled else "ERROR"
                update["status_message"] = type(error).__name__
            fetch = self._fetch
            if fetch is not None and isinstance(error, asyncio.CancelledError):
                fetch.cancel()
                fetch = None
            if fetch is None or fetch.done():
                self._finish(update, end_time, _result(fetch))
                return
            task = asyncio.get_running_loop().create_task(self._later(fetch, update, end_time))
            _pending.add(task)
            task.add_done_callback(_pending.discard)
        except Exception as exc:
            _warn("generation_end", exc)
            self._close()

    async def _later(self, fetch: asyncio.Task, update: dict, end_time: int) -> None:
        media: dict[str, Any] = {}
        try:
            media = await fetch
        except asyncio.CancelledError:
            fetch.cancel()
            raise
        except Exception as exc:
            _warn("image_fetch", exc)
        finally:
            self._finish(update, end_time, media)

    def _finish(self, update: dict, end_time: int, media: dict[str, Any]) -> None:
        try:
            self._obs.update(input=_with_media(self._wire, media), **update)
        except Exception as exc:
            _warn("generation_end", exc)
        self._close(end_time)

    def _close(self, end_time: int | None = None) -> None:
        try:
            self._obs.end(end_time=end_time)
        except Exception as exc:
            _warn("generation_close", exc)


def _result(fetch: asyncio.Task | None) -> dict[str, Any]:
    if fetch is None or fetch.cancelled():
        return {}
    if fetch.exception() is not None:
        _warn("image_fetch", fetch.exception())
        return {}
    return fetch.result()


# 사진을 붙여 마무리하는 중인 generation. 프로세스가 끝나기 전에 기다린다.
_pending: set[asyncio.Task] = set()


def _media_backlogged(lf: Any) -> bool:
    """SDK의 사진 업로드 큐가 밀렸는지. 비공개 속성이라 읽지 못하면 밀리지 않은 것으로 본다."""
    try:
        return lf._resources._media_upload_queue.qsize() >= MEDIA_BACKLOG_LIMIT
    except Exception as exc:
        global _backlog_unknown
        if not _backlog_unknown:
            # SDK가 바뀌어 큐를 읽지 못하면 보호 장치가 꺼진 것이므로 한 번 알린다.
            _backlog_unknown = True
            _warn("media_backlog_check", exc)
        return False


_backlog_unknown = False


def _without_inline_images(wire: list) -> list:
    """data URI 사진은 SDK가 업로드 큐에 넣으므로, 큐가 밀렸을 때는 자리 표시만 남긴다."""
    out = []
    for message in wire:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = [
            {"type": "image_url", "image_url": {"url": "[image omitted: upload backlog]"}}
            if isinstance(part, dict)
            and part.get("type") == "image_url"
            and str(part["image_url"]["url"]).startswith("data:")
            else part
            for part in content
        ]
        out.append({**message, "content": parts})
    return out


def chat_output(content: str, tool_calls: list, structured: bool) -> Any:
    """구조화 출력은 JSON으로 펼쳐 화면에서 필드별로 보이게 한다. 변환 실패는 원문으로 남긴다."""
    try:
        if tool_calls:
            return {
                "content": content,
                "tool_calls": [
                    {"name": call["name"], "arguments": call["arguments"]} for call in tool_calls
                ],
            }
        if structured:
            return json.loads(content)
    except Exception:
        pass
    return content


def score_evaluations(trace_id: str | None, reports: list[dict]) -> None:
    """정답 기반 자동 평가 결과를 작업 트레이스의 점수로 붙인다."""
    lf = client()
    if lf is None or not trace_id:
        return
    for report in reports:
        try:
            meta = {
                key: report.get(key)
                for key in ("dataset", "model", "samples", "total_samples", "unclassified_policy")
            }
            for name, value in (report.get("metrics") or {}).items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    lf.create_score(
                        trace_id=trace_id,
                        name=f"{report.get('feature')}.{name}",
                        value=float(value),
                        data_type="NUMERIC",
                        comment=report.get("dataset"),
                        metadata=meta,
                    )
        except Exception as exc:
            _warn("score", exc)


def flush() -> None:
    """작업 프로세스가 끝나기 전에 남은 기록을 보낸다."""
    lf = client() if _ready else None
    if lf is None:
        return
    try:
        lf.flush()
    except Exception as exc:
        _warn("flush", exc)


async def ashutdown() -> None:
    """작업 프로세스와 API가 끝날 때 부른다. 마무리 중인 generation을 기다린 뒤 남은 기록을
    보내고 전송 스레드를 멈춘다. 전체를 FLUSH_TIMEOUT_SEC 안에 끝내고, 넘긴 기록은 버린다.

    SDK의 shutdown은 처음에 자신의 atexit 등록을 해제한다. 여기서 부르지 않으면 프로세스 종료 때
    atexit 훅이 전송 큐가 빌 때까지 시간 제한 없이 기다린다.
    """
    lf = client() if _ready else None
    if lf is None:
        return
    loop = asyncio.get_running_loop()
    deadline = loop.time() + FLUSH_TIMEOUT_SEC
    global _client
    if _pending:
        _, late = await asyncio.wait(set(_pending), timeout=FLUSH_TIMEOUT_SEC / 2)
        for task in late:
            task.cancel()
        if late:
            # 취소된 태스크도 사진 없이 generation을 끝내므로, 그 기록까지 보내도록 잠시 기다린다.
            await asyncio.wait(late, timeout=0.5)
    _client = None
    await _in_daemon(lf.shutdown, max(0.1, deadline - loop.time()))


async def _in_daemon(action, timeout: float) -> None:
    """SDK의 flush와 shutdown은 전송 큐가 빌 때까지 시간 제한 없이 기다린다.

    asyncio.to_thread로 부르면 이벤트 루프 종료가 그 스레드를 기다리므로, 데몬 스레드에서 부르고
    정해진 시간만 기다린다.
    """
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def run() -> None:
        try:
            action()
        except Exception as exc:
            _warn("flush", exc)
        finally:
            try:
                loop.call_soon_threadsafe(_resolve, done)
            except RuntimeError:
                # 시간 제한이 지나 이벤트 루프가 이미 닫혔다.
                pass

    threading.Thread(target=run, name="langfuse-flush", daemon=True).start()
    try:
        await asyncio.wait_for(asyncio.shield(done), timeout)
    except TimeoutError as exc:
        _warn("flush_timeout", exc)


def _resolve(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


def _image_urls(wire: list) -> list[str]:
    return [
        part["image_url"]["url"]
        for message in wire
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


async def _fetch_images(urls: list[str]) -> dict[str, Any]:
    from langfuse import LangfuseMedia

    async def one(http: httpx.AsyncClient, url: str) -> Any:
        async with http.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if not content_type.startswith("image/"):
                return None
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > IMAGE_MAX_BYTES:
                    return None
        return LangfuseMedia(content_bytes=bytes(body), content_type=content_type)

    gate = asyncio.Semaphore(IMAGE_FETCH_CONCURRENCY)

    async def limited(http: httpx.AsyncClient, url: str) -> Any:
        async with gate:
            return await one(http, url)

    timeout = httpx.Timeout(IMAGE_FETCH_TIMEOUT_SEC, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as http:
        try:
            # httpx의 시간 제한은 조각마다 걸리므로 사진 전체를 받는 시간도 제한한다.
            async with asyncio.timeout(IMAGE_FETCH_TIMEOUT_SEC):
                results = await asyncio.gather(
                    *(limited(http, url) for url in urls), return_exceptions=True
                )
        except TimeoutError as exc:
            _warn("image_fetch_timeout", exc)
            return {}
    failed = [r for r in results if isinstance(r, BaseException)]
    if failed:
        _warn("image_fetch", failed[0])
    return {
        url: result
        for url, result in zip(urls, results, strict=True)
        if result is not None and not isinstance(result, BaseException)
    }


def _with_media(wire: list, media: dict[str, Any]) -> list:
    """사진 자리를 내려받은 사진으로 바꾼다. 못 받은 URL은 서명을 뺀 주소만 남긴다."""
    out = []
    for message in wire:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:"):
                    parts.append(part)
                elif url in media:
                    parts.append({"type": "image_url", "image_url": {"url": media[url]}})
                else:
                    parts.append({"type": "image_url", "image_url": {"url": url.split("?", 1)[0]}})
            else:
                parts.append(part)
        out.append({**message, "content": parts})
    return out


def _jsonable(value: Any) -> Any:
    """pydantic 모델은 JSON 모양으로 바꿔 넘긴다."""
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            return None
    return value
