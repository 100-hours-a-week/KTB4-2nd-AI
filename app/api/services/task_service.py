"""작업 상태 전이와 도메인 규칙. 상태를 바꾸는 코드는 전부 여기

입구가 셋(라우터, 워커 콜백, watchdog)이라 규칙이 흩어지면 어긋남.
종료 상태 쓰기와 완료 이벤트 set()은 _finish 한 곳에서만.
모든 메서드는 이벤트 루프 스레드에서 불러야 함. asyncio.Event가 그 전제이므로
라우터는 전부 async def, 워커 호출은 전부 await

상태 전이
    (없음) → QUEUED           submit
    QUEUED → PROCESSING       _dispatch_next, 워커가 202
    QUEUED → CANCELED         cancel
    QUEUED → FAILED           _dispatch_next, 워커 연결 실패(WORKER_DEAD)
    PROCESSING → COMPLETED    complete, 워커 result 콜백
    PROCESSING → FAILED       fail(워커 failed 콜백), tick(WORKER_DEAD, TIMEOUT)
    PROCESSING → CANCELED     fail, 워커 failed 콜백의 code가 CANCELED
"""

import asyncio
import time
from collections.abc import Callable

from app.api.services.task_store import Task, TaskStore
from app.api.services.worker_client import SubmitResult, WorkerClientProtocol, WorkerUnavailable
from app.core.config import Settings
from app.core.errors import DEFAULT_MESSAGE, AppError, ErrorCode
from app.core.logging import get_logger
from app.schemas.process import ProcessRequest, ProcessResult
from app.schemas.task import TERMINAL_STATUSES, ErrorBody, ProcessStep, TaskStatus

log = get_logger(__name__)


def _error(code: ErrorCode, detail: dict | None = None) -> ErrorBody:
    """api가 스스로 만드는 실패 본문. 워커가 보낸 실패는 콜백 본문을 그대로 씀"""
    return ErrorBody(code=code, message=DEFAULT_MESSAGE[code], detail=detail or {})


class TaskService:
    def __init__(
        self,
        store: TaskStore,
        worker: WorkerClientProtocol,
        settings: Settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._worker = worker
        self._settings = settings
        self._clock = clock
        # 테스트가 FakeClock을 넣어 120초, 30분 경과를 흉내
        # 완료 이벤트. Task 필드가 아닌 이유는 asyncio.Event가 프로세스 로컬이라
        # store를 Redis로 바꿔도 못 넣기 때문. 이벤트는 상태가 아니라 알림 수단
        self._waiters: dict[int, asyncio.Event] = {}
        # _dispatch_next는 중간에 await가 있어 두 호출이 겹치면 같은 QUEUED를 고를 수 있음.
        # threading.Lock이 아니라 코루틴 사이의 순서를 잡는 락
        self._dispatch_lock = asyncio.Lock()

    # ---------- 접수와 대기, routers/process가 부름 ----------

    async def submit(self, trip_id: int, request: ProcessRequest) -> Task:
        """거절(409, 429, 503) / 대기(QUEUED) / 시작(PROCESSING) 셋 중 하나"""
        existing = self._store.get(trip_id)
        if existing is not None and existing.status not in TERMINAL_STATUSES:
            raise AppError(ErrorCode.DUPLICATE_TASK)

        slot_free = self._store.count(TaskStatus.PROCESSING) < self._settings.MAX_CONCURRENT
        if not slot_free and self._store.count(TaskStatus.QUEUED) >= self._settings.MAX_QUEUE:
            raise AppError(ErrorCode.RATE_LIMITED)
        if slot_free:
            # 당장 시작할 거면 모델이 떠 있어야 함. 슬롯이 차 있으면 이미 떠 있는 것이라 묻지 않음
            ready = await self._worker.ready()
            if ready is None or not ready.model_loaded:
                raise AppError(ErrorCode.MODEL_NOT_READY)

        task = Task(
            trip_id=trip_id,
            request=request,
            total=len(request.attachments),
            created_at=self._clock(),
        )
        self._store.put(task)
        self._waiters[trip_id] = asyncio.Event()  # 새 시도마다 새 Event, 이전 것은 이미 set됨
        log.info("접수", extra={"trip_id": trip_id, "total": task.total})
        await self._dispatch_next()
        return task

    async def wait_done(self, trip_id: int) -> Task:
        """종료 상태(COMPLETED, FAILED, CANCELED)가 될 때까지 대기. 대기열에서 기다린 시간 포함

        별도 타임아웃 없음. tick()이 TIMEOUT과 WORKER_DEAD로 _finish를 부르므로
        30분 넘게 매달리지 않음
        """
        waiter = self._waiters.get(trip_id)
        if waiter is not None:
            await waiter.wait()
        return self._require(trip_id)

    def get(self, trip_id: int) -> Task:
        return self._require(trip_id)

    # ---------- 워커 콜백, routers/internal이 부름 ----------

    def progress(self, trip_id: int, step: ProcessStep, done: int, total: int) -> None:
        task = self._require(trip_id)
        if task.status is not TaskStatus.PROCESSING:
            # 취소나 타임아웃 뒤에 도착한 늦은 콜백
            log.warning("종료된 작업의 진행률 콜백 무시", extra={"trip_id": trip_id})
            return
        task.done = done
        task.current_step = step
        task.last_callback = self._clock()

    async def complete(self, trip_id: int, result: ProcessResult) -> None:
        task = self._require(trip_id)
        if task.status is not TaskStatus.PROCESSING:
            log.warning("종료된 작업의 결과 콜백 무시", extra={"trip_id": trip_id})
            return
        self._finish(task, TaskStatus.COMPLETED, result=result)
        await self._dispatch_next()

    async def fail(self, trip_id: int, error: ErrorBody) -> None:
        """워커 실패 콜백. code가 CANCELED면 취소 완료, 아니면 실패"""
        task = self._require(trip_id)
        if task.status is not TaskStatus.PROCESSING:
            log.warning("종료된 작업의 실패 콜백 무시", extra={"trip_id": trip_id})
            return
        if error.code is ErrorCode.CANCELED:
            self._finish(task, TaskStatus.CANCELED)
        else:
            self._finish(task, TaskStatus.FAILED, error=error)
        await self._dispatch_next()

    # ---------- 취소, routers/process가 부름 ----------

    async def cancel(self, trip_id: int) -> Task:
        """QUEUED는 워커가 모르는 작업이라 여기서 끝. PROCESSING은 워커에 요청만 하고
        실제 CANCELED는 워커가 다음 단계 경계에서 멈춘 뒤 failed 콜백으로 옴"""
        task = self._require(trip_id)
        if task.status in TERMINAL_STATUSES:
            raise AppError(ErrorCode.TASK_NOT_CANCELABLE)
        if task.status is TaskStatus.QUEUED:
            self._finish(task, TaskStatus.CANCELED)
            return task
        await self._worker.cancel(trip_id)
        return task

    # ---------- 감시, watchdog이 부름 ----------

    async def tick(self) -> None:
        """콜백 120초 끊김 → WORKER_DEAD, 시작 30분 초과 → TIMEOUT. 그리고 대기열 진행 안전망"""
        now = self._clock()
        for task in self._store.all():
            if task.status is not TaskStatus.PROCESSING:
                continue
            assert task.last_callback is not None and task.started_at is not None
            if now - task.last_callback > self._settings.WORKER_DEAD_SEC:
                self._finish(task, TaskStatus.FAILED, error=_error(ErrorCode.WORKER_DEAD))
            elif now - task.started_at > self._settings.TASK_TIMEOUT_SEC:
                self._finish(task, TaskStatus.FAILED, error=_error(ErrorCode.TIMEOUT))
        await self._dispatch_next()

    # ---------- 헬스, routers/health가 부름 ----------

    def active_tasks(self) -> int:
        return self._store.count(TaskStatus.PROCESSING)

    def queued(self) -> int:
        return self._store.count(TaskStatus.QUEUED)

    def idle_seconds(self) -> int:
        """작업이 있으면 0. 없으면 마지막 종료 후 경과, 종료된 적 없으면 기동 후 경과"""
        if self.active_tasks() or self.queued():
            return 0
        since = self._store.last_activity
        if since is None:
            since = self._store.boot_time
        return int(self._clock() - since)

    # ---------- 내부 ----------

    def _require(self, trip_id: int) -> Task:
        task = self._store.get(trip_id)
        if task is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND)
        return task

    def _finish(
        self,
        task: Task,
        status: TaskStatus,
        result: ProcessResult | None = None,
        error: ErrorBody | None = None,
    ) -> None:
        """종료 상태를 쓰는 유일한 곳

        다섯 종료 경로(성공, 워커 실패, 취소, 사망, 타임아웃)가 전부 여기로
        """
        task.status = status
        task.result = result
        task.error = error
        task.current_step = None
        task.finished_at = self._clock()
        self._store.last_activity = task.finished_at
        log.info(
            "종료",
            extra={
                "trip_id": task.trip_id,
                "status": status.value,
                "code": error.code.value if error else None,
            },
        )
        waiter = self._waiters.get(task.trip_id)
        if waiter is not None:
            waiter.set()  # wait_done에서 멈춰 있던 POST, DELETE 핸들러가 깨어남

    async def _dispatch_next(self) -> None:
        """슬롯이 빌 때까지 가장 오래된 QUEUED를 워커에 위임

        부르는 곳은 셋. 접수 직후, 종료 콜백 직후, watchdog(앞 둘이 놓친 것).
        워커가 BUSY나 NOT_READY면 QUEUED로 두고 다음 tick에 재시도.
        연결 실패는 기다리지 않고 그 작업을 WORKER_DEAD로 끝냄.
        조용히 매달리는 것보다 빨리 실패하고 백엔드가 재POST
        """
        async with self._dispatch_lock:
            while self._store.count(TaskStatus.PROCESSING) < self._settings.MAX_CONCURRENT:
                queued = [t for t in self._store.all() if t.status is TaskStatus.QUEUED]
                if not queued:
                    return
                task = min(queued, key=lambda t: t.created_at)
                try:
                    result = await self._worker.submit(task.trip_id, task.request)
                except WorkerUnavailable as e:
                    log.error("워커 위임 실패", extra={"trip_id": task.trip_id, "reason": str(e)})
                    self._finish(
                        task,
                        TaskStatus.FAILED,
                        error=_error(ErrorCode.WORKER_DEAD, detail={"reason": str(e)}),
                    )
                    continue
                if result is not SubmitResult.ACCEPTED:
                    log.info(
                        "워커 미수락, 대기 유지", extra={"trip_id": task.trip_id, "reason": result}
                    )
                    return
                now = self._clock()
                task.status = TaskStatus.PROCESSING
                task.started_at = now
                task.last_callback = now
                log.info("위임", extra={"trip_id": task.trip_id})
