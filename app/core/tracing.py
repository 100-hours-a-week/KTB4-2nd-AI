"""요청이나 작업 하나의 단계별 시간과 외부 호출 결과를 모음

Trace는 ContextVar로 전달함. asyncio 태스크와 asyncio.to_thread는 맥락을 복사하므로
동기 파이프라인 스레드도 같은 Trace 객체에 기록함. 객체 자체를 공유하므로 스레드에서
추가한 값도 상위 태스크가 읽을 수 있음

emit_stage_logs가 참이면 단계가 끝날 때마다 ai_stage 한 줄을 남김. 정리와 스토리 작업이 사용.
검색은 요청 수가 많을 수 있어 단계별 줄을 남기지 않고 ai_query 한 줄에 요약함
"""

import logging
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("app.core.tracing")


@dataclass
class StageRecord:
    name: str
    # success, error, timeout, cancelled
    result: str
    seconds: float
    parent: str | None = None

    @property
    def full_name(self) -> str:
        """로그의 stage, stages_ms, 지표 phase에 같은 이름을 쓴다. 예: selecting.filter"""
        return f"{self.parent}.{self.name}" if self.parent else self.name


@dataclass
class Trace:
    operation: str
    emit_stage_logs: bool = False
    stages: list[StageRecord] = field(default_factory=list)
    # target|result별 외부 호출 수. 예: vlm|success, embedding|http_error
    calls: Counter = field(default_factory=Counter)
    # 단계와 무관하게 요청 단위로 남길 값. 예: 컷오프 전 검색 결과 수
    fields: dict[str, Any] = field(default_factory=dict)
    gpu_wait_seconds: float = 0.0
    current_stage: str | None = None
    # 작업 요약 로그(ai_job_summary)를 이미 남겼는지
    summarized: bool = False
    # 결과를 저장한 뒤 기록할 정답 기반 평가 보고서
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_stage(self, record: StageRecord, **fields: Any) -> None:
        with self._lock:
            self.stages.append(record)
        if self.emit_stage_logs:
            log.info(
                "처리 단계 종료",
                extra={
                    "event": "ai_stage",
                    "operation": self.operation,
                    "stage": record.full_name,
                    "parent_stage": record.parent,
                    "result": record.result,
                    "duration_ms": round(record.seconds * 1000, 2),
                    **fields,
                },
            )

    def add_call(self, target: str, result: str) -> None:
        with self._lock:
            self.calls[f"{target}|{result}"] += 1

    def add_gpu_wait(self, seconds: float) -> None:
        with self._lock:
            self.gpu_wait_seconds += max(0.0, seconds)

    def stage_ms(self) -> dict[str, float]:
        """같은 이름의 단계는 합산. 부모가 있는 세부 단계는 부모.이름으로"""
        totals: dict[str, float] = {}
        with self._lock:
            for record in self.stages:
                name = record.full_name
                totals[name] = round(totals.get(name, 0.0) + record.seconds * 1000, 2)
        return totals

    def call_counts(self) -> dict[str, int] | None:
        with self._lock:
            return dict(self.calls) or None


_trace: ContextVar[Trace | None] = ContextVar("trace", default=None)


def start_trace(operation: str, *, emit_stage_logs: bool = False) -> tuple[Trace, Token]:
    trace = Trace(operation, emit_stage_logs)
    return trace, _trace.set(trace)


def end_trace(token: Token) -> None:
    _trace.reset(token)


def current_trace() -> Trace | None:
    return _trace.get()


@contextmanager
def stage(name: str, *, parent: str | None = None, **fields: Any) -> Iterator[dict[str, Any]]:
    """단계 하나의 시간 측정. 넘겨받은 dict에 넣은 값은 ai_stage 로그에 함께 남음

    Trace가 없으면 측정만 하고 기록하지 않음. 예외는 그대로 전파함
    """
    trace = current_trace()
    extra: dict[str, Any] = dict(fields)
    previous = trace.current_stage if trace else None
    if trace:
        trace.current_stage = f"{parent}.{name}" if parent else name
    started = time.monotonic()
    result = "success"
    try:
        yield extra
    except BaseException as exc:
        # 상위의 시간 제한에 걸려도 안쪽에서는 CancelledError로 보인다. 검색은 요청이 끝난 뒤
        # 이런 단계를 timeout으로 바꾼다(QueryService). 직접 받은 TimeoutError는 바로 timeout이다.
        if type(exc).__name__ in {"CancelledError", "PipelineCancelled"}:
            result = "cancelled"
        elif isinstance(exc, TimeoutError):
            result = "timeout"
        else:
            result = "error"
        raise
    finally:
        if trace:
            trace.current_stage = previous
            trace.add_stage(StageRecord(name, result, time.monotonic() - started, parent), **extra)
