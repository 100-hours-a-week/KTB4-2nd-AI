"""파이프라인 실행. 스레드 하나에서 run_fn을 돌리고 결과와 실패를 콜백으로

api가 "스레드 하나, 전부 루프 위"라면 워커는 "루프(라우터) + 러너 스레드 하나".
둘이 공유하는 건 _current와 _cancel_flags뿐이라 threading.Lock 하나로 보호.
스레드 풀 크기 1이 v1 MAX_CONCURRENT=1의 실체
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from app.core.errors import DEFAULT_MESSAGE, ErrorCode, PipelineError
from app.core.logging import bind_trip, get_logger
from app.pipeline.bootstrap import RunFn
from app.pipeline.context import PipelineContext
from app.schemas.process import ProcessRequest, ProcessResult
from app.schemas.task import ErrorBody, ProcessStep
from app.worker.callback import CallbackProtocol

log = get_logger(__name__)


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
        """실행 중이면 False(라우터가 409). 아니면 스레드에 넘기고 True"""
        with self._lock:
            if self._current is not None:
                return False
            self._current = trip_id
            self._cancel_flags[trip_id] = threading.Event()
        self._executor.submit(self._execute, trip_id, request)
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

        def on_progress(step: ProcessStep, done: int, total: int) -> None:
            self._callback.progress(trip_id, step, done, total)

        result: ProcessResult | None = None
        error: ErrorBody | None = None
        try:
            log.info("실행 시작", extra={"total": len(request.attachments)})
            result = self._run(request, self._ctx, on_progress, flag.is_set)
        except PipelineError as e:
            # PipelineCancelled도 여기. code가 CANCELED라 api가 취소로 처리
            log.info("실행 중단", extra={"code": e.code.value})
            error = ErrorBody(code=e.code, message=e.message, detail=e.detail)
        except Exception as e:
            # run.py 규칙 밖의 예외. ProcessResult 검증 실패도 여기. 안 잡으면 스레드가 조용히 죽음
            log.exception("실행 실패")
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
            self._callback.result(trip_id, result)
            log.info("실행 완료")
        bind_trip(None)
