"""Kafka 접수, Redis 대기, 실행 자식, 결과 Kafka 사이에 OpenTelemetry Context를 이음

- 전달 형식은 W3C Trace Context version 00의 traceparent와 선택적 tracestate만 쓴다.
  baggage 등 다른 키는 읽지도 쓰지도 않는다.
- Provider는 Cloud가 설치한 전역 Provider를 그대로 쓴다. set_tracer_provider를 호출하지 않고,
  SDK나 Provider가 없으면 모든 함수가 아무것도 하지 않는다.
- 잘못된 Header나 저장 값은 버리고 업무 처리를 계속한다. 유효한 부모가 없으면 빈 Context에서
  시작해 Provider의 샘플링 정책을 따르는 새 Root가 된다.
- Span 속성은 고정값인 yeodam.operation, yeodam.stage, yeodam.result만 남긴다.
  사진 경로, 좌표, 프롬프트, 사용자 입력과 예외 메시지는 남기지 않는다.
- Langfuse는 별도 Provider를 쓰며 이 모듈과 Context를 주고받지 않는다.
"""

import asyncio
import logging
import re
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.context import Context
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    _PROPAGATOR = TraceContextTextMapPropagator()
except ImportError:  # pragma: no cover - SDK 없는 환경
    otel_trace = None

log = logging.getLogger("app.observability.otel")

TRACEPARENT = "traceparent"
TRACESTATE = "tracestate"
TRACESTATE_MAX_CHARS = 512
FLUSH_TIMEOUT_SEC = 1.5
# Span 상태를 ERROR로 표시할 결과. 취소와 중복 접수는 정상 흐름으로 본다.
ERROR_RESULTS = frozenset({"error", "timeout", "lease_lost", "aborted", "too_large", "retry"})
TRACER_NAME = "yeodam.ai.jobs"

# version 00만 받는다. 55자, 소문자 hex
_TRACEPARENT = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")


def enabled() -> bool:
    return otel_trace is not None


def valid_carrier(carrier: Any) -> dict[str, str] | None:
    """유효한 traceparent와 선택적 tracestate만 남긴 carrier. 아니면 None

    저장 값과 Header 값을 모두 이 함수로 거른다. 예외를 던지지 않는다.
    """
    if otel_trace is None or not isinstance(carrier, Mapping):
        return None
    parent = carrier.get(TRACEPARENT)
    if not isinstance(parent, str) or not _TRACEPARENT.fullmatch(parent):
        return None
    state = carrier.get(TRACESTATE)
    raw = {TRACEPARENT: parent}
    if isinstance(state, str) and state and len(state) <= TRACESTATE_MAX_CHARS:
        raw[TRACESTATE] = state
    try:
        context = _PROPAGATOR.extract(raw, context=Context())
        span_context = otel_trace.get_current_span(context).get_span_context()
    except Exception:
        return None
    if not span_context.is_valid:
        return None
    # 원래 문자열을 그대로 보존한다. tracestate가 형식에 맞지 않으면 propagator가 비워 둔다.
    result = {TRACEPARENT: parent}
    if TRACESTATE in raw and len(span_context.trace_state) > 0:
        result[TRACESTATE] = raw[TRACESTATE]
    return result


def carrier_from_headers(headers: Iterable[tuple[str, bytes | None]] | None) -> dict | None:
    """Kafka Header에서 소문자 traceparent와 tracestate만 읽음. 같은 키가 여러 번이면 무시"""
    if not headers:
        return None
    found: dict[str, list[bytes | None]] = {}
    try:
        for key, value in headers:
            if key in (TRACEPARENT, TRACESTATE):
                found.setdefault(key, []).append(value)
    except (TypeError, ValueError):
        return None
    carrier = {}
    for key, values in found.items():
        if len(values) != 1 or not isinstance(values[0], bytes | bytearray):
            continue
        try:
            carrier[key] = bytes(values[0]).decode("utf-8")
        except UnicodeDecodeError:
            continue
    return valid_carrier(carrier)


def kafka_headers(carrier: Mapping[str, str] | None) -> list[tuple[str, bytes]] | None:
    carrier = valid_carrier(carrier)
    if carrier is None:
        return None
    return [(key, value.encode("utf-8")) for key, value in carrier.items()]


def current_carrier() -> dict[str, str] | None:
    """현재 활성 Span의 Context를 carrier로. 유효한 Span이 없으면 None"""
    if otel_trace is None:
        return None
    carrier: dict[str, str] = {}
    try:
        _PROPAGATOR.inject(carrier)
    except Exception:
        return None
    return valid_carrier(carrier)


def parent_context(carrier: Mapping[str, str] | None):
    """저장된 carrier의 Context. 없거나 잘못됐으면 빈 Context(새 Root)

    호출한 쪽의 현재 Context를 이어받지 않아 다른 작업의 Context가 섞이지 않는다.
    """
    carrier = valid_carrier(carrier)
    if carrier is None:
        return Context()
    return _PROPAGATOR.extract(carrier, context=Context())


def result_of(exc: BaseException) -> str:
    """core.tracing.stage()와 같은 분류"""
    if type(exc).__name__ in {"CancelledError", "PipelineCancelled"}:
        return "cancelled"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "error"


@dataclass
class SpanHandle:
    """with 블록 안에서 result를 정한다. 정하지 않고 예외가 나면 예외 종류로 정한다."""

    result: str | None = None


@contextmanager
def span(
    name: str,
    *,
    parent: Mapping[str, str] | None,
    kind: str = "internal",
    operation: str | None = None,
) -> Iterator[SpanHandle]:
    """작업 경계의 Span 하나를 열고 활성화. parent가 없으면 새 Root

    계측 오류는 업무 처리로 전파하지 않는다. with 본문의 예외는 그대로 전파한다.
    """
    handle = SpanHandle()
    if otel_trace is None:
        yield handle
        return
    try:
        tracer = otel_trace.get_tracer(TRACER_NAME)
        current = tracer.start_span(
            name,
            context=parent_context(parent),
            kind=getattr(SpanKind, kind.upper()),
            attributes={"yeodam.operation": operation} if operation else None,
            record_exception=False,
            set_status_on_exception=False,
        )
    except Exception:
        log.warning("OTel Span 시작 실패", extra={"event": "ai_otel_span_failed", "span": name})
        yield handle
        return
    try:
        with otel_trace.use_span(
            current, end_on_exit=False, record_exception=False, set_status_on_exception=False
        ):
            yield handle
    except BaseException as exc:
        if handle.result is None:
            handle.result = result_of(exc)
        raise
    finally:
        _finish(current, handle.result or "success")


def _finish(current, result: str, end_time: int | None = None) -> None:
    try:
        current.set_attribute("yeodam.result", result)
        if result in ERROR_RESULTS:
            current.set_status(Status(StatusCode.ERROR))
        current.end(end_time=end_time)
    except Exception:
        pass


def stage_span(operation: str, stage: str, result: str, seconds: float) -> None:
    """이미 잰 단계 시간으로 INTERNAL Span 하나를 남김. 부모는 현재 활성 Span(ai.execute)

    시간은 core.tracing이 잰 값을 그대로 쓰고 따로 재지 않는다. 활성 Span이 없으면 남기지 않는다.
    """
    if otel_trace is None:
        return
    try:
        if not otel_trace.get_current_span().get_span_context().is_valid:
            return
        end = time.time_ns()
        start = end - int(max(0.0, seconds) * 1e9)
        current = otel_trace.get_tracer(TRACER_NAME).start_span(
            f"ai.stage.{stage}",
            start_time=start,
            attributes={"yeodam.operation": operation, "yeodam.stage": stage},
        )
        _finish(current, result, end_time=end)
    except Exception:
        pass


async def flush(timeout_sec: float = FLUSH_TIMEOUT_SEC) -> None:
    """짧게 끝나는 자식이 종료 전에 남은 Span을 보냄. 기존 전역 Provider만, 시간 상한 있음"""
    if otel_trace is None:
        return
    force_flush = getattr(otel_trace.get_tracer_provider(), "force_flush", None)
    if force_flush is None:
        return
    try:
        await asyncio.wait_for(
            asyncio.to_thread(force_flush, int(timeout_sec * 1000)), timeout=timeout_sec + 0.5
        )
    except Exception:
        log.warning("OTel Span 전송 마무리 실패", extra={"event": "ai_otel_flush_failed"})
