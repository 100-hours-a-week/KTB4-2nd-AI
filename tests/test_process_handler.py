"""실제 정리 파이프라인과 작업 상태 저장소의 연결을 검증한다."""

import asyncio
import threading
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from PIL import Image

from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.embedding.client import MODEL_VERSION, EmbeddingUnavailable
from app.engines.fake import FakeEngine
from app.jobs import process_handler
from app.jobs.execution import execute
from app.jobs.kafka import RequestReceiver, publish_results
from app.jobs.sync_bridge import run_sync
from tests.test_v2_jobs import environment, message, process_request, scenario


class Engine(FakeEngine):
    model_version = MODEL_VERSION
    closed = False
    batches = None

    def __init__(self):
        self.batches = []

    def encode_images(self, images):
        self.batches.append(len(images))
        return super().encode_images(images)

    def close(self):
        self.closed = True


@pytest.fixture
def configured(tmp_path, monkeypatch):
    photo = tmp_path / "trips/77/a.jpg"
    photo.parent.mkdir(parents=True)
    noise = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(noise).save(photo)
    from app.jobs.config import JobsSettings

    cfg = JobsSettings(
        _env_file=None,
        API_KEY="test",
        IMAGE_SOURCE="local",
        IMAGE_DIR=tmp_path,
        QDRANT_URL=":memory:",
        EMBED_MODEL=MODEL_VERSION,
    )
    engine = Engine()
    monkeypatch.setattr(process_handler, "JobsSettings", lambda: cfg)
    monkeypatch.setattr(process_handler, "build_engine", lambda _: engine)
    return cfg, engine


def reporter():
    return AsyncMock()


@scenario
async def test_real_pipeline_progress_result_and_cleanup(configured):
    _, engine = configured
    r = reporter()
    result = await process_handler.run(process_request(), r)
    assert result["unclassified"][0]["trip_attachment_id"] == 101
    assert engine.closed
    assert engine.batches == [1]
    steps = [c.args[0] for c in r.report.call_args_list]
    assert steps[0] == "DOWNLOADING" and steps[-1] == "FINALIZING"
    assert r.check_cancel.await_count > 1


@scenario
async def test_pipeline_batches_are_sequential_and_at_most_16(configured):
    _, engine = configured
    request = process_request()
    request["attachments"] = [
        dict(request["attachments"][0], trip_attachment_id=i + 1) for i in range(33)
    ]
    result = await process_handler.run(request, reporter())
    assert len(result["unclassified"]) == 33
    assert engine.batches == [16, 16, 1]


@scenario
async def test_redis_job_completion_and_result_publication(configured):
    async with environment() as (cfg, store, _, producer):
        await RequestReceiver(store, producer).accept(message(cfg, process_request()))
        record = await store.claim("process", "owner")
        await execute(store, record.identity, record.owner, process_handler.run)
        done = await store.get(record.identity)
        assert done.status == "COMPLETED" and done.event.result["unclassified"]
        task = asyncio.create_task(publish_results(store, producer))
        try:
            async with asyncio.timeout(2):
                while not producer.sent:
                    await asyncio.sleep(0.005)
            import json

            event = json.loads(producer.sent[0][1])
            assert event["type"] == "process_result"
            assert event["status"] == "COMPLETED"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("enabled", [False, True])
@scenario
async def test_photos_ready_setting_controls_vector_reuse(configured, monkeypatch, enabled):
    cfg, _ = configured
    cfg.EMBED_ON_PHOTOS_READY = enabled
    seen = []
    build = process_handler.build_context

    def capture(*args, **kwargs):
        ctx = build(*args, **kwargs)
        seen.append(ctx.reuse_vectors)
        return ctx

    monkeypatch.setattr(process_handler, "build_context", capture)
    await process_handler.run(process_request(), reporter())
    assert seen == [enabled]


@pytest.mark.parametrize("mutation", ["duplicate", "period", "missing"])
@scenario
async def test_invalid_input_precedes_external_io(configured, mutation):
    _, engine = configured
    request = process_request()
    if mutation == "duplicate":
        request["attachments"] *= 2
    if mutation == "period":
        request["period"]["start_date"] = "2026-12-01"
    if mutation == "missing":
        del request["type"]
    with pytest.raises(PipelineError) as error:
        await process_handler.run(request, reporter())
    assert error.value.code == ErrorCode.INVALID_REQUEST
    assert not engine.batches


@pytest.mark.parametrize(
    "fault,code",
    [
        (EmbeddingUnavailable("bad"), ErrorCode.MODEL_NOT_READY),
        (httpx.ReadTimeout("slow"), ErrorCode.TIMEOUT),
    ],
)
@scenario
async def test_embedding_failures_are_safe_and_close_resources(configured, fault, code):
    _, engine = configured

    def fail(_):
        raise fault

    engine.encode_images = fail
    with pytest.raises(PipelineError) as error:
        await process_handler.run(process_request(), reporter())
    assert error.value.code == code
    assert engine.closed


@scenario
async def test_setup_failure_closes_engine(configured, monkeypatch):
    _, engine = configured

    def fail(_):
        raise RuntimeError("Qdrant unavailable")

    monkeypatch.setattr(process_handler, "wait_for_qdrant", fail)
    with pytest.raises(RuntimeError):
        await process_handler.run(process_request(), reporter())
    assert engine.closed


@scenario
async def test_cancel_at_pipeline_boundary(configured):
    _, engine = configured
    r = reporter()

    async def progress(step, *_):
        if step == "EMBEDDING":
            r.check_cancel.side_effect = PipelineCancelled()

    r.report.side_effect = progress
    with pytest.raises(PipelineCancelled):
        await process_handler.run(process_request(), r)
    assert engine.closed and engine.batches == []


@scenario
async def test_async_cancel_waits_for_sync_cleanup_without_blocking_loop():
    started, stopped = threading.Event(), threading.Event()

    def operation(progress, cancelled):
        started.set()
        try:
            while not cancelled():
                stopped.wait(0.001)
        finally:
            stopped.set()

    task = asyncio.create_task(run_sync(operation, reporter()))
    while not started.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert stopped.is_set()


@scenario
async def test_real_handler_rejects_fake_or_local_engine(configured):
    cfg, engine = configured
    cfg.FAKE_PIPELINE = True
    with pytest.raises(PipelineError) as error:
        await process_handler.run(process_request(), reporter())
    assert error.value.code == ErrorCode.MODEL_NOT_READY and not engine.batches


@scenario
async def test_qdrant_sdk_wrapped_timeout_is_timeout(configured, monkeypatch):
    from qdrant_client.http.exceptions import ResponseHandlingException

    from app.infra.qdrant import QdrantStore

    def fail(*args, **kwargs):
        raise ResponseHandlingException(httpx.ReadTimeout("secret host"))

    monkeypatch.setattr(QdrantStore, "upsert", fail)
    with pytest.raises(PipelineError) as error:
        await process_handler.run(process_request(), reporter())
    assert error.value.code == ErrorCode.TIMEOUT
    assert "secret" not in error.value.message
    assert configured[1].closed
