"""task_service의 상태 전이 규칙. HTTP 없이 서비스만, 시간은 FakeClock으로"""

import asyncio

import pytest

from app.api.services.task_service import TaskService
from app.api.services.task_store import TaskStore
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.schemas.process import ProcessRequest
from app.schemas.task import ProcessStep, TaskStatus
from tests.fakes import FakeClock, FakeWorkerClient

pytestmark = pytest.mark.anyio


def make_request(n: int = 2) -> ProcessRequest:
    return ProcessRequest.model_validate(
        {
            "trip_name": "제주",
            "period": {"start_date": "2026-10-12", "end_date": "2026-10-14"},
            "regions": [{"latitude": 33.4996, "longitude": 126.5312}],
            "attachments": [
                {
                    "trip_attachment_id": 100 + i,
                    "analyze_storage_key": f"trips/77/analyze/{i}.jpg",
                    "taken_at": "2026-10-12T06:14:00+09:00",
                    "latitude": 33.458,
                    "longitude": 126.9423,
                    "device_model": "Apple iPhone 15",
                }
                for i in range(n)
            ],
        }
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(API_KEY="test", IMAGE_SOURCE="local", MAX_QUEUE=2)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake() -> FakeWorkerClient:
    return FakeWorkerClient(hold=True)


@pytest.fixture
def service(settings: Settings, clock: FakeClock, fake: FakeWorkerClient) -> TaskService:
    svc = TaskService(TaskStore(boot_time=clock()), fake, settings, clock=clock)
    fake.service = svc
    return svc


async def let_callbacks_run() -> None:
    """FakeWorkerClient가 create_task로 띄운 콜백이 돌 틈"""
    for _ in range(5):
        await asyncio.sleep(0)


def assert_error(exc: pytest.ExceptionInfo[AppError], code: ErrorCode) -> None:
    assert exc.value.code is code


# ---------- 접수와 완료 ----------


async def test_submit_dispatches_and_completes(service: TaskService, fake: FakeWorkerClient):
    task = await service.submit(77, make_request(3))
    assert task.status is TaskStatus.PROCESSING
    assert fake.submitted == [77]

    fake.finish(77)
    done = await service.wait_done(77)
    assert done.status is TaskStatus.COMPLETED
    assert done.result is not None and len(done.result.unclassified) == 3
    assert done.to_response().progress.done == 3
    assert done.current_step is None


async def test_duplicate_while_processing(service: TaskService):
    await service.submit(77, make_request())
    with pytest.raises(AppError) as exc:
        await service.submit(77, make_request())
    assert_error(exc, ErrorCode.DUPLICATE_TASK)


async def test_resubmit_after_completed_is_new_attempt(
    service: TaskService, fake: FakeWorkerClient
):
    await service.submit(77, make_request())
    fake.finish(77)
    await service.wait_done(77)

    task = await service.submit(77, make_request(5))
    assert task.status is TaskStatus.PROCESSING
    assert task.total == 5
    assert fake.submitted == [77, 77]


# ---------- 슬롯과 대기열 ----------


async def test_queue_and_rate_limit(service: TaskService, fake: FakeWorkerClient):
    await service.submit(77, make_request())  # 슬롯 차지
    assert (await service.submit(78, make_request())).status is TaskStatus.QUEUED
    assert (await service.submit(79, make_request())).status is TaskStatus.QUEUED
    with pytest.raises(AppError) as exc:
        await service.submit(80, make_request())  # MAX_QUEUE=2
    assert_error(exc, ErrorCode.RATE_LIMITED)
    assert service.queued() == 2

    fake.finish(77)
    await service.wait_done(77)
    await let_callbacks_run()
    # 종료 콜백 직후 가장 오래된 QUEUED가 위임됨
    assert service.get(78).status is TaskStatus.PROCESSING
    assert service.get(79).status is TaskStatus.QUEUED
    assert fake.submitted == [77, 78]


async def test_queued_task_waits_through_queue(service: TaskService, fake: FakeWorkerClient):
    await service.submit(77, make_request())
    await service.submit(78, make_request())
    waiter = asyncio.create_task(service.wait_done(78))
    await let_callbacks_run()
    assert not waiter.done()

    fake.finish(77)
    await let_callbacks_run()
    fake.finish(78)
    done = await waiter
    assert done.status is TaskStatus.COMPLETED


# ---------- 취소 ----------


async def test_cancel_queued(service: TaskService, fake: FakeWorkerClient):
    await service.submit(77, make_request())
    await service.submit(78, make_request())
    task = await service.cancel(78)
    assert task.status is TaskStatus.CANCELED
    assert (await service.wait_done(78)).status is TaskStatus.CANCELED
    assert fake.cancelled == []  # 워커는 모르는 작업


async def test_cancel_processing(service: TaskService, fake: FakeWorkerClient):
    await service.submit(77, make_request())
    await service.submit(78, make_request())
    task = await service.cancel(77)
    assert task.status is TaskStatus.PROCESSING  # 워커가 멈추기 전
    assert fake.cancelled == [77]

    done = await service.wait_done(77)
    assert done.status is TaskStatus.CANCELED
    assert done.error is None
    await let_callbacks_run()
    assert service.get(78).status is TaskStatus.PROCESSING  # 슬롯 반환 뒤 다음 건


async def test_cancel_terminal_is_409(service: TaskService, fake: FakeWorkerClient):
    await service.submit(77, make_request())
    fake.finish(77)
    await service.wait_done(77)
    with pytest.raises(AppError) as exc:
        await service.cancel(77)
    assert_error(exc, ErrorCode.TASK_NOT_CANCELABLE)


# ---------- 감시 ----------


async def test_tick_worker_dead(service: TaskService, clock: FakeClock, settings: Settings):
    await service.submit(77, make_request())
    waiter = asyncio.create_task(service.wait_done(77))

    clock.advance(settings.WORKER_DEAD_SEC - 1)
    await service.tick()
    assert service.get(77).status is TaskStatus.PROCESSING

    clock.advance(2)
    await service.tick()
    done = await waiter
    assert done.status is TaskStatus.FAILED
    assert done.error is not None and done.error.code is ErrorCode.WORKER_DEAD


async def test_tick_timeout_with_live_callbacks(
    service: TaskService, clock: FakeClock, settings: Settings
):
    await service.submit(77, make_request())
    elapsed = 0
    while elapsed <= settings.TASK_TIMEOUT_SEC:
        clock.advance(60)
        elapsed += 60
        service.progress(77, ProcessStep.EMBEDDING, 1, 2)  # 콜백은 계속 옴
        await service.tick()
    done = await service.wait_done(77)
    assert done.status is TaskStatus.FAILED
    assert done.error is not None and done.error.code is ErrorCode.TIMEOUT


# ---------- 워커 상태 ----------


async def test_model_not_ready_is_503(settings: Settings, clock: FakeClock):
    fake = FakeWorkerClient(hold=True, model_loaded=False)
    service = TaskService(TaskStore(boot_time=clock()), fake, settings, clock=clock)
    fake.service = service
    with pytest.raises(AppError) as exc:
        await service.submit(77, make_request())
    assert_error(exc, ErrorCode.MODEL_NOT_READY)
    with pytest.raises(AppError):
        service.get(77)  # 저장되지 않음


async def test_worker_unreachable_at_dispatch_fails_fast(settings: Settings, clock: FakeClock):
    fake = FakeWorkerClient(hold=True)
    service = TaskService(TaskStore(boot_time=clock()), fake, settings, clock=clock)
    fake.service = service
    await service.submit(77, make_request())  # 슬롯 차지, ready는 통과
    await service.submit(78, make_request())  # QUEUED
    fake.unreachable = True

    fake.finish(77)
    await service.wait_done(77)
    await let_callbacks_run()
    done = service.get(78)
    assert done.status is TaskStatus.FAILED
    assert done.error is not None and done.error.code is ErrorCode.WORKER_DEAD


# ---------- 콜백 규칙 ----------


async def test_late_callbacks_ignored_and_unknown_404(service: TaskService, fake: FakeWorkerClient):
    await service.submit(77, make_request())
    fake.finish(77)
    done = await service.wait_done(77)
    service.progress(77, ProcessStep.CLOCK, 0, 2)  # 늦은 콜백
    assert done.current_step is None
    assert done.status is TaskStatus.COMPLETED

    with pytest.raises(AppError) as exc:
        service.progress(999, ProcessStep.CLOCK, 0, 2)
    assert_error(exc, ErrorCode.TASK_NOT_FOUND)


# ---------- 헬스 ----------


async def test_idle_seconds(service: TaskService, fake: FakeWorkerClient, clock: FakeClock):
    clock.advance(30)
    assert service.idle_seconds() == 30  # 종료된 적 없으면 기동 후

    await service.submit(77, make_request())
    clock.advance(100)
    assert service.idle_seconds() == 0  # 작업 중
    assert service.active_tasks() == 1

    fake.finish(77)
    await service.wait_done(77)
    clock.advance(10)
    assert service.idle_seconds() == 10  # 마지막 종료 후
