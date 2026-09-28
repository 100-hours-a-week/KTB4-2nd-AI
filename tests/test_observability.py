"""관측 로그. 공통 JSON 계약, 서드파티 하한, 식별자 전파, 이벤트 기록

Formatter는 전역 상태를 건드리지 않게 직접 만들어 확인하고,
setup_logging을 부르는 테스트는 루트 핸들러를 되돌림
"""

import json
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app.api.services.task_service import TaskService
from app.api.services.task_store import TaskStore
from app.core.config import Settings
from app.core.errors import DEFAULT_MESSAGE, AppError, ErrorCode
from app.core.logging import (
    COMMON_FIELDS,
    NOISY_LOGGERS,
    JsonFormatter,
    bind_job,
    bind_request,
    bind_trip,
    clear_context,
    current_job,
    current_request,
    current_trip,
    setup_logging,
)
from app.engines.fake import FakeEngine
from app.infra.qdrant import MEMORY
from app.main import create_app
from app.pipeline.bootstrap import build_context
from app.pipeline.context import PipelineContext
from app.pipeline.fake_run import fake_run
from app.schemas.task import ErrorBody
from app.worker.runner import Runner
from tests.fakes import FakeCallback, FakeClock, FakeWorkerClient
from tests.test_fake_pipeline import make_request, write_jpeg
from tests.test_task_service import make_request as make_task_request

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def clean_context() -> Iterator[None]:
    """ContextVar가 테스트 사이에 새지 않게"""
    clear_context()
    yield
    clear_context()


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """setup_logging이 루트 핸들러를 갈아치우므로 되돌림. caplog가 계속 동작해야 함"""
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    levels = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    for name, saved in levels.items():
        logging.getLogger(name).setLevel(saved)


def make_record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("app.test", logging.INFO, "f.py", 1, "메시지", None, None)
    record.__dict__.update(extra)
    return record


def format_record(**extra: object) -> dict:
    return json.loads(JsonFormatter("sha-abc").format(make_record(**extra)))


# ---------- 공통 JSON 계약 ----------


def test_common_fields_are_always_present():
    payload = format_record()

    assert payload["service"] == "ai"
    assert payload["release"] == "sha-abc"
    assert payload["message"] == "메시지"
    assert payload["logger"] == "app.test"
    for field in COMMON_FIELDS:
        assert field in payload, f"{field}가 빠짐"
        assert payload[field] is None


def test_timestamp_is_utc_with_milliseconds():
    timestamp = format_record()["timestamp"]

    assert timestamp.endswith("Z")
    assert "+00:00" not in timestamp
    assert len(timestamp.split(".")[1]) == 4  # 밀리초 세 자리와 Z


def test_context_identifiers_are_attached():
    bind_request("req-1")
    bind_trip(42)
    bind_job("exec-1")

    payload = format_record()

    assert payload["request_id"] == "req-1"
    assert payload["trip_id"] == 42
    assert payload["job_id"] == "exec-1"


def test_clear_context_removes_identifiers():
    bind_request("req-1")
    bind_trip(42)
    bind_job("exec-1")
    clear_context()

    payload = format_record()

    assert payload["request_id"] is None
    assert payload["trip_id"] is None
    assert payload["job_id"] is None


def test_extra_overrides_context():
    """api는 콜백이 다른 요청으로 오므로 trip_id와 job_id를 extra로 명시"""
    bind_trip(42)

    payload = format_record(trip_id=7, event="ai_job", result="success", duration_ms=1200)

    assert payload["trip_id"] == 7
    assert payload["event"] == "ai_job"
    assert payload["result"] == "success"
    assert payload["duration_ms"] == 1200


def test_process_role_is_emitted_as_process():
    """process는 LogRecord가 PID로 쓰는 예약어라 extra에는 process_role로 넘김"""
    payload = format_record(process_role="worker")

    assert payload["process"] == "worker"
    assert "process_role" not in payload


def test_process_role_passes_through_logging_api(caplog: pytest.LogCaptureFixture):
    log = logging.getLogger("app.test.alias")

    with caplog.at_level(logging.INFO, logger="app.test.alias"):
        log.info("기동", extra={"process_role": "api"})

    assert caplog.records[0].process_role == "api"


def test_exception_is_recorded_as_stack_trace():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "app.test", logging.ERROR, "f.py", 1, "실패", None, sys.exc_info()
        )

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError" in payload["stack_trace"]
    assert payload["release"] == "unknown"


# ---------- 서드파티 로거 하한 ----------


def test_noisy_loggers_stay_at_warning_on_debug(restore_logging: None):
    setup_logging("DEBUG", "sha-abc")

    assert logging.getLogger("app.worker.runner").getEffectiveLevel() == logging.DEBUG
    for name in NOISY_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING, name


# ---------- 러너의 스레드 전파 ----------


@pytest.fixture
def ctx(tmp_path: Path) -> PipelineContext:
    write_jpeg(tmp_path / "a.jpg", (200, 30, 30))
    write_jpeg(tmp_path / "b.jpg", (30, 200, 30))
    settings = Settings(
        API_KEY="t", IMAGE_SOURCE="local", IMAGE_DIR=tmp_path, QDRANT_URL=MEMORY, FAKE_PIPELINE=True
    )
    return build_context(settings, FakeEngine())


def test_runner_carries_context_into_thread(ctx: PipelineContext):
    """contextvars는 새 스레드에 상속되지 않으므로 copy_context로 넘김"""
    callback = FakeCallback()
    seen: dict[str, object] = {}

    def capture_run(request, pipeline_ctx, on_progress, is_cancelled):  # type: ignore[no-untyped-def]
        seen["request_id"] = current_request()
        seen["job_id"] = current_job()
        seen["trip_id"] = current_trip()
        return fake_run(request, pipeline_ctx, on_progress, is_cancelled)

    bind_request("req-1")
    bind_job("exec-1")
    runner = Runner(ctx, capture_run, callback)
    runner.submit(77, make_request(["a.jpg", "b.jpg"], [True, False]))
    callback.wait()
    runner.shutdown()

    assert seen == {"request_id": "req-1", "job_id": "exec-1", "trip_id": 77}


def test_runner_records_step_and_execution_events(
    ctx: PipelineContext, caplog: pytest.LogCaptureFixture
):
    callback = FakeCallback()
    runner = Runner(ctx, fake_run, callback)

    with caplog.at_level(logging.INFO, logger="app.worker.runner"):
        runner.submit(77, make_request(["a.jpg", "b.jpg"], [True, False]))
        callback.wait()
    runner.shutdown()

    executions = [r for r in caplog.records if getattr(r, "event", None) == "ai_worker_execution"]
    steps = [r for r in caplog.records if getattr(r, "event", None) == "ai_pipeline_step"]

    assert [r.result for r in executions] == ["started", "success"]
    assert executions[-1].duration_ms >= 0
    assert executions[-1].places >= 0
    # 단계는 시작과 완료가 짝. 이름은 클라우드 계약값
    assert {r.step for r in steps} <= {
        "download",
        "embedding",
        "clock_correction",
        "locating",
        "clustering",
        "filtering",
        "finalizing",
    }
    assert [r.result for r in steps].count("started") == [r.result for r in steps].count("success")


def test_runner_records_failure_stage(ctx: PipelineContext, caplog: pytest.LogCaptureFixture):
    callback = FakeCallback()

    def broken_run(request, pipeline_ctx, on_progress, is_cancelled):  # type: ignore[no-untyped-def]
        raise ValueError("boom")

    runner = Runner(ctx, broken_run, callback)
    with caplog.at_level(logging.INFO, logger="app.worker.runner"):
        runner.submit(77, make_request(["a.jpg"], [True]))
        callback.wait()
    runner.shutdown()

    final = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "ai_worker_execution" and r.result == "failure"
    ]
    assert len(final) == 1
    assert final[0].failure_stage == "internal"  # 단계에 들어가기 전
    assert final[0].error_code == ErrorCode.INTERNAL_ERROR.value


# ---------- api 미들웨어 ----------


async def qdrant_ok() -> bool:
    return True


@pytest.fixture
async def api_app(restore_logging: None) -> AsyncIterator[FastAPI]:
    fake = FakeWorkerClient(hold=True)
    application = create_app(
        settings=Settings(API_KEY="k", IMAGE_SOURCE="local"),
        worker=fake,
        qdrant_probe=qdrant_ok,
    )
    async with application.router.lifespan_context(application):
        fake.service = application.state.task_service
        yield application


@pytest.fixture
async def api_client(api_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_middleware_echoes_given_request_id(api_client: httpx.AsyncClient):
    res = await api_client.get("/health", headers={"X-Request-ID": "given-1"})

    assert res.headers["X-Request-ID"] == "given-1"


async def test_middleware_generates_request_id_when_missing(api_client: httpx.AsyncClient):
    """백엔드가 헤더를 붙이기 전까지는 AI가 만들어 씀"""
    res = await api_client.get("/health")

    assert res.headers["X-Request-ID"]


# ---------- ai_job 이벤트 ----------


def build_service(fake: FakeWorkerClient) -> TaskService:
    clock = FakeClock()
    settings = Settings(API_KEY="t", IMAGE_SOURCE="local")
    service = TaskService(TaskStore(boot_time=clock()), fake, settings, clock=clock)
    fake.service = service
    return service


def job_events(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.result for r in caplog.records if getattr(r, "event", None) == "ai_job"]


async def test_ai_job_records_started_and_success_once(caplog: pytest.LogCaptureFixture):
    fake = FakeWorkerClient(hold=True)
    service = build_service(fake)

    with caplog.at_level(logging.INFO, logger="app.api.services.task_service"):
        await service.submit(77, make_task_request())
        fake.finish(77)
        await service.wait_done(77)

    assert job_events(caplog) == ["started", "success"]


async def test_ai_job_records_failure_once(caplog: pytest.LogCaptureFixture):
    fake = FakeWorkerClient(hold=True)
    service = build_service(fake)

    with caplog.at_level(logging.INFO, logger="app.api.services.task_service"):
        await service.submit(77, make_task_request())
        await service.fail(
            77,
            ErrorBody(
                code=ErrorCode.INTERNAL_ERROR, message=DEFAULT_MESSAGE[ErrorCode.INTERNAL_ERROR]
            ),
        )

    assert job_events(caplog) == ["started", "failure"]
    final = [r for r in caplog.records if getattr(r, "event", None) == "ai_job"][-1]
    assert final.error_code == ErrorCode.INTERNAL_ERROR.value
    assert final.job_id is None  # execution_id를 안 보낸 요청


async def test_ai_job_records_canceled_once(caplog: pytest.LogCaptureFixture):
    fake = FakeWorkerClient(hold=True)
    service = build_service(fake)

    with caplog.at_level(logging.INFO, logger="app.api.services.task_service"):
        await service.submit(77, make_task_request())
        await service.fail(
            77, ErrorBody(code=ErrorCode.CANCELED, message=DEFAULT_MESSAGE[ErrorCode.CANCELED])
        )

    assert job_events(caplog) == ["started", "canceled"]


async def test_ai_job_carries_input_summary(caplog: pytest.LogCaptureFixture):
    fake = FakeWorkerClient(hold=True)
    service = build_service(fake)

    with caplog.at_level(logging.INFO, logger="app.api.services.task_service"):
        await service.submit(77, make_task_request(3))

    started = [r for r in caplog.records if getattr(r, "event", None) == "ai_job"][0]
    assert started.attachment_count == 3
    assert started.with_gps == 3
    assert started.with_taken_at == 3
    assert started.devices == 1


async def test_ai_job_dispatch_records_queue_depth(caplog: pytest.LogCaptureFixture):
    fake = FakeWorkerClient(hold=True)
    service = build_service(fake)

    with caplog.at_level(logging.INFO, logger="app.api.services.task_service"):
        await service.submit(77, make_task_request())

    dispatch = [r for r in caplog.records if getattr(r, "event", None) == "ai_job_dispatch"]
    assert len(dispatch) == 1
    assert dispatch[0].result == "success"
    assert dispatch[0].active_tasks == 1
    assert dispatch[0].queue_depth == 0


async def test_rejected_submit_does_not_record_ai_job(caplog: pytest.LogCaptureFixture):
    """모델 미적재로 접수 자체가 거절되면 작업이 생기지 않으므로 ai_job도 없음"""
    fake = FakeWorkerClient(unreachable=True)
    service = build_service(fake)

    with caplog.at_level(logging.INFO, logger="app.api.services.task_service"):
        with pytest.raises(AppError) as exc:
            await service.submit(77, make_task_request())

    assert exc.value.code is ErrorCode.MODEL_NOT_READY
    assert job_events(caplog) == []
