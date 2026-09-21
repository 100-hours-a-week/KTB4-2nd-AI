"""워커. 러너의 예외 매핑과 취소, 콜백 재시도, jobs 라우터의 상태 코드"""

import threading
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app.core.config import Settings
from app.core.errors import DownloadFailed, ErrorCode, PipelineCancelled
from app.engines.fake import FakeEngine
from app.infra.qdrant import MEMORY
from app.pipeline.bootstrap import build_context
from app.pipeline.context import PipelineContext
from app.pipeline.fake_run import fake_run
from app.schemas.internal import JobRequest
from app.schemas.process import ProcessRequest, ProcessResult
from app.schemas.task import ErrorBody, ProcessStep
from app.worker.callback import CallbackClient
from app.worker.runner import Runner
from app.worker_main import create_app
from tests.fakes import FakeCallback
from tests.test_fake_pipeline import make_request, write_jpeg

pytestmark = pytest.mark.anyio


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        API_KEY="t", IMAGE_SOURCE="local", IMAGE_DIR=tmp_path, QDRANT_URL=MEMORY, FAKE_PIPELINE=True
    )


@pytest.fixture
def ctx(settings: Settings, tmp_path: Path) -> PipelineContext:
    write_jpeg(tmp_path / "a.jpg", (200, 30, 30))
    write_jpeg(tmp_path / "b.jpg", (30, 200, 30))
    return build_context(settings, FakeEngine())


@pytest.fixture
def callback() -> FakeCallback:
    return FakeCallback()


def request_two() -> ProcessRequest:
    return make_request(["a.jpg", "b.jpg"], [True, False])


# ---------- 러너 ----------


def test_runner_completes_and_calls_back(ctx: PipelineContext, callback: FakeCallback):
    runner = Runner(ctx, fake_run, callback)
    assert runner.submit(77, request_two())
    assert runner.busy
    callback.wait()

    assert len(callback.results) == 1 and callback.results[0][0] == 77
    assert isinstance(callback.results[0][1], ProcessResult)
    assert callback.failures == []
    steps = [call[1] for call in callback.progress_calls]
    assert steps[0] is ProcessStep.DOWNLOADING and steps[-1] is ProcessStep.FINALIZING
    assert not runner.busy
    runner.shutdown()


def test_runner_rejects_while_busy(ctx: PipelineContext, callback: FakeCallback):
    gate = threading.Event()

    def blocking_run(request, ctx, on_progress, is_cancelled):
        gate.wait(5)
        return fake_run(request, ctx, on_progress, is_cancelled)

    runner = Runner(ctx, blocking_run, callback)
    assert runner.submit(77, request_two())
    assert not runner.submit(78, request_two())  # 라우터가 409
    gate.set()
    callback.wait()
    assert runner.submit(78, request_two())  # 슬롯 반환 뒤
    callback.wait()
    runner.shutdown()


def test_runner_cancel_maps_to_canceled(ctx: PipelineContext, callback: FakeCallback):
    started = threading.Event()

    def slow_run(request, ctx, on_progress, is_cancelled):
        started.set()
        while not is_cancelled():
            threading.Event().wait(0.01)
        raise PipelineCancelled()

    runner = Runner(ctx, slow_run, callback)
    runner.submit(77, request_two())
    assert started.wait(5)
    assert runner.cancel(77)
    callback.wait()

    assert callback.results == []
    assert callback.failures[0][1].code is ErrorCode.CANCELED
    assert not runner.cancel(77)  # 끝난 작업, 라우터가 404
    runner.shutdown()


def test_runner_maps_pipeline_error_code(ctx: PipelineContext, callback: FakeCallback):
    def failing_run(request, ctx, on_progress, is_cancelled):
        raise DownloadFailed(["k1"])

    runner = Runner(ctx, failing_run, callback)
    runner.submit(77, request_two())
    callback.wait()
    error = callback.failures[0][1]
    assert error.code is ErrorCode.DOWNLOAD_FAILED
    assert error.detail == {"keys": ["k1"]}
    runner.shutdown()


def test_runner_maps_unexpected_exception(ctx: PipelineContext, callback: FakeCallback):
    def broken_run(request, ctx, on_progress, is_cancelled):
        raise ValueError("boom")

    runner = Runner(ctx, broken_run, callback)
    runner.submit(77, request_two())
    callback.wait()
    error = callback.failures[0][1]
    assert error.code is ErrorCode.INTERNAL_ERROR
    assert "ValueError" in error.detail["reason"]
    assert not runner.busy  # finally가 슬롯을 돌려줌
    runner.shutdown()


# ---------- 콜백 클라이언트 ----------


def make_client(statuses: list[int | Exception]) -> tuple[CallbackClient, list[str]]:
    """응답 순서를 정해 둔 가짜 api. 연결 실패는 예외로"""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        outcome = statuses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome)

    client = CallbackClient(
        "http://api", retries=3, backoff_seconds=0, transport=httpx.MockTransport(handler)
    )
    return client, seen


def test_callback_retries_then_succeeds():
    client, seen = make_client([503, httpx.ConnectError("refused"), 204])
    client.progress(77, ProcessStep.EMBEDDING, 1, 2)
    assert seen == ["/internal/trips/77/progress"] * 3


def test_callback_stops_on_404():
    client, seen = make_client([404, 204])
    client.failed(77, ErrorBody(code=ErrorCode.TIMEOUT, message="x"))
    assert seen == ["/internal/trips/77/failed"]


def test_callback_gives_up_without_raising():
    client, seen = make_client([500, 500, 500])
    client.progress(77, ProcessStep.CLOCK, 0, 2)  # 예외 없이 돌아옴
    assert len(seen) == 3


# ---------- jobs 라우터 ----------


@pytest.fixture
async def worker_app(settings: Settings, callback: FakeCallback) -> AsyncIterator[FastAPI]:
    gate = threading.Event()

    def gated_run(request, ctx, on_progress, is_cancelled):
        gate.wait(5)
        return fake_run(request, ctx, on_progress, is_cancelled)

    application = create_app(
        settings=settings, engine=FakeEngine(), run_fn=gated_run, callback=callback
    )
    application.state.gate = gate
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(worker_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=worker_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as c:
        yield c


def job_body(trip_id: int) -> dict:
    return JobRequest(trip_id=trip_id, request=request_two()).model_dump(mode="json")


async def test_ready_after_startup(client: httpx.AsyncClient):
    res = await client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"model_loaded": True, "model_name": "fake", "busy": False}


async def test_jobs_accept_busy_and_cancel(
    client: httpx.AsyncClient, worker_app: FastAPI, callback: FakeCallback
):
    res = await client.post("/jobs", json=job_body(77))
    assert res.status_code == 202 and res.json() == {"trip_id": 77}
    assert (await client.get("/ready")).json()["busy"] is True

    res = await client.post("/jobs", json=job_body(78))
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "WORKER_BUSY"

    res = await client.post("/jobs/78/cancel")
    assert res.status_code == 404
    res = await client.post("/jobs/77/cancel")
    assert res.status_code == 204

    worker_app.state.gate.set()
    callback.wait()
    assert (await client.get("/ready")).json()["busy"] is False


async def test_jobs_503_before_model_loaded(settings: Settings, callback: FakeCallback):
    application = create_app(
        settings=settings, engine=FakeEngine(), run_fn=fake_run, callback=callback
    )
    application.state.model_loaded = False  # lifespan을 안 돌린 상태
    application.state.model_name = "fake"
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as c:
        res = await c.post("/jobs", json=job_body(77))
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "MODEL_NOT_READY"
        assert (await c.get("/ready")).json()["model_loaded"] is False


async def test_jobs_422_in_spec_format(client: httpx.AsyncClient):
    res = await client.post("/jobs", json={"trip_id": "x"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_REQUEST"
