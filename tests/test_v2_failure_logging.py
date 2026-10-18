"""실패 위치 기록. 실패 단계와 사유, 예외 위치, 외부 호출 결과와 민감 정보 제외를 확인한다."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError

from app.api.services.query_service import QueryService
from app.core.errors import ErrorCode, PipelineError
from app.core.http_errors import register_error_handlers
from app.core.logging import app_relative, error_fields, validation_fields
from app.infra.vllm_client import VllmClient
from app.jobs import runner as runner_module
from app.jobs.execution import execute
from app.jobs.gpu_requests import GateUnavailable, RequestGate
from app.jobs.kafka import InvalidMessage, RequestReceiver, publish_results
from app.jobs.mock import run as mock_run
from app.jobs.models import StoryMessage
from app.jobs.runner import run_claimed
from app.schemas.query import ParseRequest
from app.search.context import SearchOptions
from app.search.mock import create_mock_context
from app.story.pipeline import finalize, validate_request
from tests.observability_helpers import (  # noqa: F401
    SECRET,
    completion,
    events,
    info_logs,
    rendered,
)
from tests.test_story import draft, request_two_places
from tests.test_v2_jobs import environment, message, process_request, scenario, story_request


def test_error_fields_keep_location_but_not_message():
    with pytest.raises(PipelineError) as error:
        validate_request({"type": "story", "trip_name": SECRET})
    fields = error_fields(error.value)
    assert fields["error_type"] == "PipelineError"
    assert any(frame.startswith("app/story/pipeline.py:") for frame in fields["error_frames"])
    assert SECRET not in json.dumps(fields, ensure_ascii=False)

    try:
        try:
            raise ValueError(SECRET)
        except ValueError as inner:
            raise RuntimeError(SECRET) from inner
    except RuntimeError as exc:
        chained = error_fields(exc)
    assert chained["error_type"] == "RuntimeError" and chained["error_chain"] == ["ValueError"]
    assert SECRET not in json.dumps(chained, ensure_ascii=False)


def test_library_frames_under_app_directory_are_not_app_frames():
    # 컨테이너에서는 코드가 /app/app, 가상환경이 /app/.venv에 있다.
    assert app_relative("/app/app/jobs/runner.py", app_dir="/app/app") == "app/jobs/runner.py"
    assert (
        app_relative("/app/.venv/lib/python3.12/site-packages/redis/client.py", "/app/app") is None
    )
    assert app_relative("/srv/app/x.py", app_dir="/app/app") is None


def test_validation_fields_omit_input_values():
    with pytest.raises(ValidationError) as error:
        StoryMessage.model_validate({**story_request(), "mood": SECRET})
    fields = validation_fields(error.value.errors())
    assert "mood:literal_error" in fields
    assert SECRET not in json.dumps(fields)


@scenario
async def test_unexpected_failure_logs_stage_and_location_without_secret(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")

        async def fail(request, reporter):
            await reporter.report("DOWNLOADING", 1, 1)
            await reporter.report("EMBEDDING", 0, 1)
            raise RuntimeError(SECRET)

        await execute(a, job.identity, job.owner, fail)
        failed = events(caplog, "ai_job_failed")[0]
        assert failed.error_code == "INTERNAL_ERROR" and failed.failure_stage == "embedding"
        assert failed.error_type == "RuntimeError"
        finished = events(caplog, "ai_job_finished")[-1]
        assert finished.failure_stage == "embedding" and finished.progress_done == 0
        assert SECRET not in rendered(caplog)


@scenario
async def test_pipeline_error_logs_fixed_reason_and_detail_names_only(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, story_request()))
        job = await a.claim("story", "worker")

        async def fail(request, reporter):
            error = PipelineError(
                "벡터 없음", detail={"reason": "MISSING_VECTORS", "keys": [SECRET]}
            )
            error.code = ErrorCode.MODEL_NOT_READY
            raise error

        await execute(a, job.identity, job.owner, fail)
        failed = events(caplog, "ai_job_failed")[0]
        assert failed.error_reason == "MISSING_VECTORS"
        assert failed.detail_keys == ["keys", "reason"]
        assert failed.failure_stage == "startup"
        assert events(caplog, "ai_job_finished")[-1].error_reason == "MISSING_VECTORS"
        assert SECRET not in rendered(caplog)


@scenario
async def test_odd_detail_does_not_block_saving_the_failure(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")

        class Odd(Exception):
            detail = "문자열 detail"

        async def odd(request, reporter):
            raise Odd()

        await execute(a, job.identity, job.owner, odd)
        stored = await b.get(job.identity)
        assert stored.status == "FAILED" and stored.event.error.code == "INTERNAL_ERROR"
        assert events(caplog, "ai_job_failed")[0].error_type == "Odd"


@scenario
async def test_completed_job_finish_log_splits_queue_and_execution(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, story_request()))
        job = await a.claim("story", "worker")
        await execute(a, job.identity, job.owner, mock_run)
        finished = events(caplog, "ai_job_finished")[-1]
        assert finished.result == "COMPLETED" and finished.failure_stage is None
        assert finished.execution_ms >= 0 and finished.queue_ms >= 0


@scenario
async def test_queued_cancel_is_attributed_to_queue(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        await a.cancel(77, "process")
        finished = events(caplog, "ai_job_finished")[-1]
        assert finished.result == "CANCELED" and finished.failure_stage == "queue"


@scenario
async def test_received_log_has_kind_and_job_id_and_invalid_retry_is_logged(caplog):
    async with environment() as (cfg, a, b, p):
        receiver = RequestReceiver(a, p)
        await receiver.accept(message(cfg, process_request()))
        received = events(caplog, "ai_job_received")[0]
        assert received.operation == "process" and received.job_id == "execution-1"
        assert received.attachment_count == 1 and received.invalid_fields is None

        broken = message(cfg, process_request("execution-2"), offset=9)
        broken = broken.__class__(broken.topic, 0, 9, broken.key, b"{not json")
        with pytest.raises(InvalidMessage):
            await receiver.accept(broken)
        retry = events(caplog, "ai_job_invalid_retry")[0]
        assert retry.result == "invalid_json" and retry.attempts == 1 and retry.offset == 9

        invalid = process_request("execution-3", trip=78)
        invalid["attachments"][0]["analyze_storage_key"] = ""
        await receiver.accept(message(cfg, invalid))
        rejected = events(caplog, "ai_job_received")[-1]
        assert rejected.invalid_fields and all(":" in f for f in rejected.invalid_fields)


@scenario
async def test_intake_failure_and_publish_result_fields(caplog):
    async with environment() as (cfg, a, b, p):
        invalid = process_request()
        invalid["attachments"][0]["analyze_storage_key"] = ""
        await RequestReceiver(a, p).accept(message(cfg, invalid))
        assert events(caplog, "ai_job_finished")[-1].failure_stage == "intake"

        await RequestReceiver(a, p).accept(message(cfg, process_request("execution-2", trip=78)))
        job = await a.claim("process", "worker")
        await execute(a, job.identity, job.owner, mock_run)
        loop = asyncio.create_task(publish_results(a, p))
        try:
            async with asyncio.timeout(2):
                while len(events(caplog, "ai_job_result_published")) < 2:
                    await asyncio.sleep(0.005)
        finally:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)
        published = events(caplog, "ai_job_result_published")
        assert {r.result for r in published} == {"success"}
        assert {r.job_status for r in published} == {"FAILED", "COMPLETED"}


@scenario
async def test_child_exit_code_is_logged_when_result_was_not_saved(caplog, monkeypatch):
    class Killed:
        returncode = None

        async def wait(self):
            self.returncode = -9
            return -9

        def terminate(self):
            pass

        def kill(self):
            pass

    async def spawn(*args, **kwargs):
        return Killed()

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", spawn)
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")
        await run_claimed(a, job)
        exited = events(caplog, "ai_worker_exit")[0]
        assert exited.exit_code == -9 and exited.error_code == "WORKER_DEAD"
        assert exited.trip_id == 77 and exited.job_id == "execution-1"


@scenario
async def test_finished_child_is_not_reported_as_lease_lost(caplog, monkeypatch):
    async with environment(JOBS_HEARTBEAT_SEC=0.01) as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")

        class Exiting:
            """결과를 저장한 뒤 정리 작업 때문에 heartbeat 주기보다 늦게 끝나는 자식"""

            returncode = None

            async def wait(self):
                await asyncio.sleep(0.05)
                self.returncode = 0
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        async def spawn(*args, **kwargs):
            result = await mock_run(job.request, AsyncMock())
            await a.finish(job.identity, job.owner, "COMPLETED", result=result)
            return Exiting()

        monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", spawn)
        await run_claimed(a, job)
        assert not events(caplog, "ai_job_lease_lost")
        assert (await b.get(job.identity)).status == "COMPLETED"


def test_story_output_violation_names_the_rule(caplog):
    message_ = validate_request(request_two_places())
    selected = [(p.trip_place_id, p.attachments[0]) for p in message_.places]
    for raw, rule in [
        ("not json", "draft_schema"),
        (draft((101,)), "photo_set_mismatch"),
        (draft((101, 101, 102)), "duplicate_photo"),
    ]:
        with pytest.raises(PipelineError):
            finalize(raw, message_, selected)
        assert events(caplog, "ai_story_output_invalid")[-1].rule == rule
    assert "사진 속 풍경" not in rendered(caplog)


@pytest.mark.parametrize(
    "status,body,result,finish_reason",
    [
        (401, {"error": SECRET}, "http_error", None),
        (
            200,
            completion("잘린 글", "length", {"completion_tokens": 2048}),
            "incomplete_output",
            "length",
        ),
        (
            200,
            completion("완성", "stop", {"prompt_tokens": 10, "completion_tokens": 5}),
            "success",
            "stop",
        ),
    ],
)
@scenario
async def test_vlm_call_logs_status_finish_reason_and_tokens(
    caplog, status, body, result, finish_reason
):
    client = VllmClient(
        "https://vlm.example/v1",
        "test-model",
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body)),
    )
    async with client:
        try:
            await client.chat([{"role": "user", "content": "질문"}], [SECRET], 64)
        except Exception:
            pass
    call = events(caplog, "ai_model_call")[-1]
    assert call.target == "vlm" and call.result == result and call.image_count == 1
    assert getattr(call, "finish_reason", None) == finish_reason
    if status == 401:
        assert call.status_code == 401
    if result == "success":
        assert call.completion_tokens == 5 and call.tokens_per_sec is not None
    assert SECRET not in rendered(caplog)


@scenario
async def test_vlm_call_stopped_by_outer_time_limit_is_logged(caplog):
    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json=completion("{}"))

    client = VllmClient("https://vlm.example/v1", "test-model", transport=httpx.MockTransport(slow))
    service = QueryService(
        replace(create_mock_context(), vlm=client), SearchOptions(parse_timeout_sec=0.05)
    )
    async with client:
        parsed = await service.parse(ParseRequest(query="바다", reference_date="2026-10-12"))
    assert parsed.parsed is False
    call = events(caplog, "ai_model_call")[-1]
    assert call.target == "vlm" and call.result == "cancelled"
    assert events(caplog, "ai_query")[-1].result == "fallback"


@scenario
async def test_gpu_gate_timeout_is_logged(caplog):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    gate = RequestGate(redis, "test:gpu:generation", 1, 0.2)
    seconds, _ = await redis.time()
    await redis.zadd("test:gpu:generation", {"other": seconds + 60})
    with pytest.raises(GateUnavailable):
        await gate.run(lambda: asyncio.sleep(0), 1)
    logged = events(caplog, "ai_gpu_gate_unavailable")[0]
    assert logged.result == "timeout" and logged.gate == "generation" and logged.limit == 1
    assert logged.wait_ms >= 200
    await redis.aclose()


@scenario
async def test_gpu_gate_leaves_room_for_client_timeout(caplog):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    gate = RequestGate(redis, "test:gpu:generation", 1, 1)

    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json=completion())

    client = VllmClient(
        "https://vlm.example/v1",
        "test-model",
        timeout_sec=0.05,
        transport=httpx.MockTransport(slow),
    )
    async with client:
        with pytest.raises(TimeoutError):
            await gate.run(lambda: client.chat([{"role": "user", "content": "x"}], [], 8), 0.05)
    assert events(caplog, "ai_model_call")[-1].result == "timeout"
    await redis.aclose()


def test_request_validation_logs_fields_without_values(caplog):
    from fastapi.testclient import TestClient

    class Body(BaseModel):
        count: int

    app = FastAPI()
    register_error_handlers(app)

    @app.post("/check")
    async def check(body: Body):
        return {}

    response = TestClient(app).post("/check", json={"count": SECRET})
    assert response.status_code == 422
    logged = events(caplog, "ai_request_invalid")[0]
    assert logged.invalid_fields == ["body.count:int_parsing"] and logged.path == "/check"
    assert SECRET not in rendered(caplog)
