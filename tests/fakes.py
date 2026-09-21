"""테스트 대역

FakeClock, FakeWorkerClient는 api 테스트, FakeCallback은 워커 테스트.
FakeEngine과 fake_run은 컨테이너에서도 써야 해서 app/engines/fake.py,
app/pipeline/fake_run.py에 있음
"""

import asyncio
import threading

from app.api.services.task_service import TaskService
from app.api.services.worker_client import SubmitResult, WorkerUnavailable
from app.core.errors import DEFAULT_MESSAGE, ErrorCode
from app.schemas.internal import ReadyResponse
from app.schemas.process import (
    Issue,
    ProcessRequest,
    ProcessResult,
    RegionOrigin,
    UnclassifiedAttachment,
)
from app.schemas.task import ErrorBody, ProcessStep


class FakeClock:
    """time.monotonic 대신. advance로 120초, 30분 경과를 즉시 만듦"""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_result(request: ProcessRequest) -> ProcessResult:
    """검증기를 통과하는 최소 결과. 사진 전부를 위치 불명 미분류로"""
    return ProcessResult(
        places=[],
        unclassified=[
            UnclassifiedAttachment(
                trip_attachment_id=a.trip_attachment_id,
                issue=Issue.UNCLEAR_LOCATION,
                taken_at=a.taken_at,
                latitude=a.latitude,
                longitude=a.longitude,
                region_origin=RegionOrigin.UNKNOWN,
            )
            for a in request.attachments
        ],
    )


class FakeWorkerClient:
    """워커 프로세스 대신 콜백을 직접 부름

    진짜 워커는 다른 프로세스라 콜백이 submit이 돌아온 뒤에 도착함.
    asyncio.create_task로 같은 순서를 만듦.
    submit 안에서 바로 complete를 부르면 _dispatch_next 안에서 _dispatch_next가 다시 도는 재귀가 됨.
    hold=True면 finish(trip_id)를 부를 때까지 콜백을 보류, 대기 중 GET과 DELETE 테스트용.
    service는 픽스처가 TaskService를 만든 뒤 연결
    """

    def __init__(
        self, hold: bool = False, model_loaded: bool = True, unreachable: bool = False
    ) -> None:
        self.hold = hold
        self.model_loaded = model_loaded
        self.unreachable = unreachable
        self.service: TaskService | None = None
        self.submitted: list[int] = []
        self.cancelled: list[int] = []
        self.busy = False
        self._gates: dict[int, asyncio.Event] = {}
        self._jobs: dict[int, asyncio.Task[None]] = {}

    async def submit(self, trip_id: int, request: ProcessRequest) -> SubmitResult:
        if self.unreachable:
            raise WorkerUnavailable("connection refused")
        if not self.model_loaded:
            return SubmitResult.NOT_READY
        if self.busy:
            return SubmitResult.BUSY
        self.busy = True
        self.submitted.append(trip_id)
        self._gates[trip_id] = asyncio.Event()
        self._jobs[trip_id] = asyncio.create_task(self._run(trip_id, request))
        return SubmitResult.ACCEPTED

    async def _run(self, trip_id: int, request: ProcessRequest) -> None:
        assert self.service is not None, "픽스처가 service를 연결해야 함"
        if self.hold:
            await self._gates[trip_id].wait()
        self.busy = False
        if trip_id in self.cancelled:
            error = ErrorBody(code=ErrorCode.CANCELED, message=DEFAULT_MESSAGE[ErrorCode.CANCELED])
            await self.service.fail(trip_id, error)
            return
        total = len(request.attachments)
        self.service.progress(trip_id, ProcessStep.EMBEDDING, total, total)
        await self.service.complete(trip_id, make_result(request))

    def finish(self, trip_id: int) -> None:
        """보류 중인 작업을 끝냄. 진짜 워커의 '다음 단계 경계 도달'에 해당"""
        self._gates[trip_id].set()

    async def cancel(self, trip_id: int) -> None:
        """취소 표시 뒤 바로 경계에 도달한 것으로. 진짜 워커는 단계 사이나 배치 사이에서"""
        self.cancelled.append(trip_id)
        if trip_id in self._gates:
            self.finish(trip_id)

    async def ready(self) -> ReadyResponse | None:
        if self.unreachable:
            return None
        return ReadyResponse(model_loaded=self.model_loaded, model_name="fake", busy=self.busy)

    async def aclose(self) -> None:
        for job in self._jobs.values():
            job.cancel()


class FakeCallback:
    """CallbackProtocol 대역. 호출을 기록하고 result나 failed가 오면 done을 set

    테스트가 wait()로 러너 스레드가 끝나기를 기다림
    """

    def __init__(self) -> None:
        self.progress_calls: list[tuple[int, ProcessStep, int, int]] = []
        self.results: list[tuple[int, ProcessResult]] = []
        self.failures: list[tuple[int, ErrorBody]] = []
        self.done = threading.Event()

    def progress(self, trip_id: int, step: ProcessStep, done: int, total: int) -> None:
        self.progress_calls.append((trip_id, step, done, total))

    def result(self, trip_id: int, result: ProcessResult) -> None:
        self.results.append((trip_id, result))
        self.done.set()

    def failed(self, trip_id: int, error: ErrorBody) -> None:
        self.failures.append((trip_id, error))
        self.done.set()

    def wait(self, timeout: float = 5.0) -> None:
        assert self.done.wait(timeout), "러너가 제한 시간 안에 끝나지 않음"
        self.done.clear()
