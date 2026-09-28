"""파이프라인 실행. 스레드 하나에서 run_fn을 돌리고 결과와 실패를 콜백으로

api가 "스레드 하나, 전부 루프 위"라면 워커는 "루프(라우터) + 러너 스레드 하나".
둘이 공유하는 건 _current와 _cancel_flags뿐이라 threading.Lock 하나로 보호.
스레드 풀 크기 1이 v1 MAX_CONCURRENT=1의 실체

관측은 두 층. 실행 전체는 ai_worker_execution, 단계 전환은 ai_pipeline_step.
contextvars는 새 스레드에 상속되지 않으므로 라우터 맥락을 copy_context로 명시 전달
"""

import contextvars
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from app.core.errors import DEFAULT_MESSAGE, ErrorCode, PipelineError
from app.core.logging import (
    FAILURE_INTERNAL,
    STEP_NAMES,
    bind_trip,
    clear_context,
    get_logger,
)
from app.pipeline.bootstrap import RunFn
from app.pipeline.context import PipelineContext
from app.schemas.process import ProcessRequest, ProcessResult
from app.schemas.task import ErrorBody, ProcessStep
from app.worker.callback import CallbackProtocol

log = get_logger(__name__)


def _summarize(result: ProcessResult) -> dict[str, object]:
    """결과를 집계만. 좌표와 사본 키 같은 원본 값은 담지 않음"""
    classified = [a for place in result.places for a in place.attachments]
    origins = Counter(a.region_origin.value for a in classified)
    origins.update(u.region_origin.value for u in result.unclassified)
    return {
        "places": len(result.places),
        "classified": len(classified),
        "unclassified": len(result.unclassified),
        "failed": len(result.failed),
        "origin": dict(origins),
        "issues": dict(Counter(u.issue.value for u in result.unclassified)),
        "clock_offsets": len(result.clock_offsets),
    }


class Runner:
    def __init__(self, ctx: PipelineContext, run_fn: RunFn, callback: CallbackProtocol) -> None:
        self._ctx = ctx
        self._run = run_fn
        self._callback = callback
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self._lock = threading.Lock()
        self._current: int | None = None
        self._cancel_flags: dict[int, threading.Event] = {}

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None

    def submit(self, trip_id: int, request: ProcessRequest) -> bool:
        """실행 중이면 False(라우터가 409). 아니면 스레드에 넘기고 True

        라우터가 심은 request_id와 job_id를 스레드로 옮기기 위해 현재 맥락을 복사
        """
        with self._lock:
            if self._current is not None:
                return False
            self._current = trip_id
            self._cancel_flags[trip_id] = threading.Event()
        context = contextvars.copy_context()
        self._executor.submit(context.run, self._execute, trip_id, request)
        return True

    def cancel(self, trip_id: int) -> bool:
        """플래그만 set. run_fn이 다음 경계에서 PipelineCancelled를 던져 failed{CANCELED}로 끝남"""
        with self._lock:
            flag = self._cancel_flags.get(trip_id)
        if flag is None:
            return False
        flag.set()
        return True

    def shutdown(self) -> None:
        """종료 중인 작업은 프로세스와 함께 사라짐, 드레이닝 없음"""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _execute(self, trip_id: int, request: ProcessRequest) -> None:
        """러너 스레드. 실패는 전부 콜백으로, 스레드 밖으로 예외를 내보내지 않음

        슬롯을 먼저 반환하고 콜백을 보냄. 콜백을 받은 api가 그 자리에서 다음 작업을 보내는데,
        그때 슬롯이 차 있으면 409로 튕겨 watchdog 5초를 기다리게 됨
        """
        bind_trip(trip_id)
        flag = self._cancel_flags[trip_id]
        total = len(request.attachments)
        started = time.monotonic()
        # 단계 전환을 진행률 콜백에서 감지. run_fn은 단계 시작에 done=0으로 한 번 보고함
        current_step: ProcessStep | None = None
        step_started = started

        def close_step(result: str) -> None:
            nonlocal current_step
            if current_step is None:
                return
            log.info(
                "파이프라인 단계를 마쳤습니다.",
                extra={
                    "event": "ai_pipeline_step",
                    "result": result,
                    "step": STEP_NAMES[current_step.value],
                    "total": total,
                    "duration_ms": round((time.monotonic() - step_started) * 1000),
                },
            )
            current_step = None

        def on_progress(step: ProcessStep, done: int, total_count: int) -> None:
            nonlocal current_step, step_started
            if step is not current_step:
                close_step("success")
                current_step = step
                step_started = time.monotonic()
                log.info(
                    "파이프라인 단계를 시작합니다.",
                    extra={
                        "event": "ai_pipeline_step",
                        "result": "started",
                        "step": STEP_NAMES[step.value],
                        "total": total_count,
                    },
                )
            log.debug(
                "파이프라인 진행률입니다.",
                extra={
                    "event": "ai_pipeline_step",
                    "result": "started",
                    "step": STEP_NAMES[step.value],
                    "done": done,
                    "total": total_count,
                },
            )
            self._callback.progress(trip_id, step, done, total_count)

        result: ProcessResult | None = None
        error: ErrorBody | None = None
        log.info(
            "워커가 파이프라인 실행을 시작합니다.",
            extra={
                "event": "ai_worker_execution",
                "result": "started",
                "attachment_count": total,
            },
        )
        try:
            result = self._run(request, self._ctx, on_progress, flag.is_set)
        except PipelineError as e:
            # PipelineCancelled도 여기. code가 CANCELED라 api가 취소로 처리
            failure_stage = STEP_NAMES[current_step.value] if current_step else FAILURE_INTERNAL
            canceled = e.code is ErrorCode.CANCELED
            close_step("canceled" if canceled else "failure")
            log.info(
                "파이프라인 실행이 중단되었습니다.",
                extra={
                    "event": "ai_worker_execution",
                    "result": "canceled" if canceled else "failure",
                    "duration_ms": round((time.monotonic() - started) * 1000),
                    "failure_stage": failure_stage,
                    "error_code": e.code.value,
                    "detail": e.detail,
                },
            )
            error = ErrorBody(code=e.code, message=e.message, detail=e.detail)
        except Exception as e:
            # run.py 규칙 밖의 예외. ProcessResult 검증 실패도 여기. 안 잡으면 스레드가 조용히 죽음
            failure_stage = STEP_NAMES[current_step.value] if current_step else FAILURE_INTERNAL
            close_step("failure")
            log.exception(
                "파이프라인 실행에 실패했습니다.",
                extra={
                    "event": "ai_worker_execution",
                    "result": "failure",
                    "duration_ms": round((time.monotonic() - started) * 1000),
                    "failure_stage": failure_stage,
                    "error_code": ErrorCode.INTERNAL_ERROR.value,
                },
            )
            error = ErrorBody(
                code=ErrorCode.INTERNAL_ERROR,
                message=DEFAULT_MESSAGE[ErrorCode.INTERNAL_ERROR],
                detail={"reason": f"{type(e).__name__}: {e}"},
            )
        finally:
            with self._lock:
                self._current = None
                self._cancel_flags.pop(trip_id, None)

        if error is not None:
            self._callback.failed(trip_id, error)
        else:
            assert result is not None
            close_step("success")
            self._callback.result(trip_id, result)
            log.info(
                "워커가 파이프라인 실행을 마쳤습니다.",
                extra={
                    "event": "ai_worker_execution",
                    "result": "success",
                    "attachment_count": total,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                    **_summarize(result),
                },
            )
            self._log_detail(result)
        clear_context()

    def _log_detail(self, result: ProcessResult) -> None:
        """폴더와 사진 단위 상세. 사진 수만큼 줄이 늘어나므로 DEBUG에서만"""
        if not log.isEnabledFor(logging.DEBUG):
            return
        for place in result.places:
            log.debug(
                "장소 폴더입니다.",
                extra={
                    "event": "ai_place",
                    "place_id": place.place_id,
                    "count": len(place.attachments),
                    "representative": place.representative_attachment_id,
                },
            )
        for place in result.places:
            for attachment in place.attachments:
                log.debug(
                    "사진 분류 결과입니다.",
                    extra={
                        "event": "ai_photo",
                        "attachment_id": attachment.trip_attachment_id,
                        "place_id": place.place_id,
                        "origin": attachment.region_origin.value,
                        "issue": None,
                        "evaluation": attachment.evaluation,
                    },
                )
        for unclassified in result.unclassified:
            log.debug(
                "사진 분류 결과입니다.",
                extra={
                    "event": "ai_photo",
                    "attachment_id": unclassified.trip_attachment_id,
                    "place_id": unclassified.place_id,
                    "origin": unclassified.region_origin.value,
                    "issue": unclassified.issue.value,
                    "evaluation": unclassified.evaluation,
                },
            )
