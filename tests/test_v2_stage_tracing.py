"""세부 단계 기록. 단계별 시간 로그, 작업과 검색 요약, 단계별 지연 지표를 확인한다."""

import asyncio
import json
import logging
from dataclasses import replace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import httpx
import pytest

from app.api.services.query_service import QueryService
from app.api.services.search_tracing import (
    TracedEmbedder,
    TracedImages,
    TracedVectors,
    TracedVlm,
)
from app.core.logging import JsonFormatter
from app.core.tracing import end_trace, stage, start_trace
from app.infra.vllm_client import VllmClient
from app.jobs.execution import execute
from app.jobs.gpu_requests import GateUnavailable, RequestGate
from app.jobs.kafka import RequestReceiver
from app.jobs.mock import run as mock_run
from app.observability.metrics import Metrics
from app.schemas.query import ParseRequest, SearchRequest
from app.search.context import SearchOptions
from app.search.mock import create_mock_context
from tests.observability_helpers import (  # noqa: F401
    SECRET,
    completion,
    events,
    info_logs,
    rendered,
)
from tests.test_v2_jobs import environment, message, process_request, scenario, story_request


def test_stage_records_nested_names_results_and_current_stage(caplog):
    trace, token = start_trace("story", emit_stage_logs=True)
    try:
        with stage("filter", parent="selecting", candidates=3) as fields:
            fields["retained"] = 2
            # 운영 핸들러는 로그를 남기는 시점에 바로 직렬화하므로 단계 안에서 형식을 확인한다.
            inner = json.loads(JsonFormatter().format(logging.makeLogRecord({"msg": "x"})))
        with pytest.raises(ValueError), stage("generate", parent="writing"):
            raise ValueError("boom")
        with pytest.raises(TimeoutError), stage("answer"):
            raise TimeoutError()
    finally:
        end_trace(token)
    stage_logs = events(caplog, "ai_stage")
    assert [(r.stage, r.parent_stage, r.result) for r in stage_logs] == [
        ("selecting.filter", "selecting", "success"),
        ("writing.generate", "writing", "error"),
        ("answer", None, "timeout"),
    ]
    assert stage_logs[0].retained == 2 and stage_logs[0].candidates == 3
    assert inner["stage"] == "selecting.filter"
    assert set(trace.stage_ms()) == {"selecting.filter", "writing.generate", "answer"}
    assert trace.current_stage is None


@scenario
async def test_job_steps_are_logged_summarized_and_measured(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")

        async def fail(request, reporter):
            await reporter.report("DOWNLOADING", 1, 1)
            await reporter.report("EMBEDDING", 0, 1)
            raise RuntimeError(SECRET)

        await execute(a, job.identity, job.owner, fail)
        stage_logs = events(caplog, "ai_stage")
        assert [(r.stage, r.result) for r in stage_logs] == [
            ("download", "success"),
            ("embedding", "error"),
        ]
        summary = events(caplog, "ai_job_summary")[0]
        assert summary.result == "FAILED"
        assert set(summary.stages_ms) == {"download", "embedding"}
        assert SECRET not in rendered(caplog)
        metrics = await a.redis.hgetall(a.key("metrics:operations"))
        assert metrics["process|success|download|count"] == "1"
        assert metrics["process|error|embedding|count"] == "1"
        text = (await Metrics(a.redis, a.prefix).render()).decode()
        assert 'phase="embedding"' in text


@scenario
async def test_completed_job_summary_has_steps(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, story_request()))
        job = await a.claim("story", "worker")
        await execute(a, job.identity, job.owner, mock_run)
        summary = events(caplog, "ai_job_summary")[0]
        assert summary.result == "COMPLETED" and summary.stages_ms


@scenario
async def test_summary_uses_stored_status_after_cancel_request(caplog):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")

        async def cancelled_then_done(request, reporter):
            await a.cancel(77, "process")
            return await mock_run(request, AsyncMock())

        await execute(a, job.identity, job.owner, cancelled_then_done)
        assert (await b.get(job.identity)).status == "CANCELED"
        assert events(caplog, "ai_job_summary")[-1].result == "CANCELED"


@scenario
async def test_search_log_summarizes_stages_scores_and_stage_metrics(caplog):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    mock = create_mock_context()
    context = mock.__class__(
        vlm=TracedVlm(mock.vlm),
        embedder=TracedEmbedder(mock.embedder),
        vectors=TracedVectors(mock.vectors),
        images=TracedImages(mock.images),
    )
    service = QueryService(context, metrics=Metrics(redis, "test:"))
    request = SearchRequest.model_validate(
        {
            "query": "바다에서 뭐 했어?",
            "intent": "QUESTION",
            "visual_query": "바다",
            "candidates": [
                {
                    "trip_attachment_id": i,
                    "analyze_storage_key": f"k/{i}.jpg",
                    "taken_at": None,
                    "place_name": None,
                    "trip_name": None,
                }
                for i in range(1, 6)
            ],
        }
    )
    await service.search(request)
    query = events(caplog, "ai_query")[-1]
    assert {"embed", "retrieve"} <= set(query.stages_ms)
    assert query.hit_count is not None
    fields = await redis.hgetall("test:metrics:operations")
    assert any(key.startswith("search|success|retrieve|") for key in fields)
    await redis.aclose()


@scenario
async def test_traced_vlm_forwards_temperature_only_when_given():
    inner = AsyncMock()
    inner.chat.return_value = "ok"
    vlm = TracedVlm(inner)
    await vlm.chat([], [], 20, temperature=0)
    await vlm.chat([], [], 20)
    assert inner.chat.await_args_list[0].kwargs == {"temperature": 0}
    assert inner.chat.await_args_list[1].kwargs == {}


@scenario
async def test_search_stage_stopped_by_time_limit_is_marked_timeout(caplog):
    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json=completion("{}"))

    client = VllmClient("https://vlm.example/v1", "test-model", transport=httpx.MockTransport(slow))
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    service = QueryService(
        replace(create_mock_context(), vlm=TracedVlm(client)),
        SearchOptions(parse_timeout_sec=0.05),
        metrics=Metrics(redis, "test:"),
    )
    async with client:
        await service.parse(ParseRequest(query="바다", reference_date="2026-10-12"))
    query = events(caplog, "ai_query")[-1]
    assert query.result == "fallback" and query.model_calls == {"vlm|timeout": 1}
    assert query.stages_ms["interpret"] > 0
    fields = await redis.hgetall("test:metrics:operations")
    assert "parse|timeout|interpret|count" in fields
    await redis.aclose()


@scenario
async def test_gpu_slot_wait_is_added_to_trace():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    gate = RequestGate(redis, "test:gpu:generation", 1, 0.2)
    seconds, _ = await redis.time()
    await redis.zadd("test:gpu:generation", {"other": seconds + 60})
    trace, token = start_trace("story")
    try:
        with pytest.raises(GateUnavailable):
            await gate.run(lambda: asyncio.sleep(0), 1)
    finally:
        end_trace(token)
    assert trace.gpu_wait_seconds >= 0.2
    await redis.aclose()
