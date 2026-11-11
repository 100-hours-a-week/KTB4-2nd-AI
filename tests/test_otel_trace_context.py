"""Kafka Header -> Redis -> 실행 자식 -> 결과 Header의 OTel Context 연결과 저장 형식 호환"""

import asyncio
import json
import logging
import os
import textwrap
from dataclasses import replace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import create_model

from app.core.logging import JsonFormatter, otel_ids
from app.core.tracing import stage
from app.jobs import execution
from app.jobs.kafka import RequestReceiver, publish_results
from app.jobs.mock import run as mock_run
from app.jobs.models import TRACE_FIELDS, JobRecord, StrictModel, identity
from app.jobs.runner import StandbyChildren, run_claimed
from app.observability import langfuse_tracing, otel_context
from app.schemas.task import Progress
from tests.test_langfuse_tracing import exporter as langfuse_spans  # noqa: F401
from tests.test_langfuse_tracing import finished as langfuse_finished
from tests.test_v2_jobs import environment, message, process_request, scenario, story_request

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
PARENT_ID = "b7ad6b7169203331"
SAMPLED = f"00-{TRACE_ID}-{PARENT_ID}-01"
NOT_SAMPLED = f"00-{TRACE_ID}-{PARENT_ID}-00"
EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def sdk_provider():
    """Cloud가 설치하는 전역 Provider 대신 메모리 Exporter를 붙인 SDK Provider"""
    current = trace.get_tracer_provider()
    if not isinstance(current, TracerProvider):
        current = TracerProvider()
        trace.set_tracer_provider(current)
    current.add_span_processor(SimpleSpanProcessor(EXPORTER))
    yield


@pytest.fixture(autouse=True)
def clear_spans():
    EXPORTER.clear()
    yield


def spans(name):
    return [span for span in EXPORTER.get_finished_spans() if span.name == name]


def only(name):
    found = spans(name)
    assert len(found) == 1, [span.name for span in EXPORTER.get_finished_spans()]
    return found[0]


def hex_span(span) -> str:
    return format(span.context.span_id, "016x")


def headers(value=SAMPLED, **extra):
    return ((otel_context.TRACEPARENT, value.encode()),) + tuple(
        (key, item.encode()) for key, item in extra.items()
    )


def fake_redis_for(store, monkeypatch):
    """run_job이 만드는 Redis 연결을 같은 fakeredis 서버로. 실제 Redis면 REDIS_URL을 그대로 씀"""
    server = store.redis.connection_pool.connection_kwargs.get("server")
    if server is None:
        return
    monkeypatch.setattr(
        execution,
        "redis_client",
        lambda cfg: fakeredis.aioredis.FakeRedis(server=server, decode_responses=True),
    )


async def publish_once(store, producer):
    task = asyncio.create_task(publish_results(store, producer))
    for _ in range(200):
        if producer.sent:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert producer.sent, "결과가 발행되지 않음"
    return producer.sent[-1]


def record_without_trace() -> JobRecord:
    return JobRecord(
        trip_id=1,
        kind="process",
        execution_id="e",
        request={},
        request_topic="a",
        result_topic="b",
        created_at=0,
        progress=Progress(done=0, total=1),
    )


# 이 변경 전 이미지의 JobRecord. 같은 필드에 extra=forbid
OldJobRecord = create_model(
    "OldJobRecord",
    __base__=StrictModel,
    **{
        name: (field.annotation, field)
        for name, field in JobRecord.model_fields.items()
        if name not in TRACE_FIELDS
    },
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (headers(), {"traceparent": SAMPLED}),
        (headers(tracestate="vendor=abc"), {"traceparent": SAMPLED, "tracestate": "vendor=abc"}),
        # baggage와 다른 Header는 버린다.
        (headers(baggage="user=1", cookie="x"), {"traceparent": SAMPLED}),
        # 512자를 넘는 tracestate만 버리고 부모는 유지
        (headers(tracestate="a=" + "b" * 511), {"traceparent": SAMPLED}),
        (headers(SAMPLED.upper()), None),
        (headers(f"ff-{TRACE_ID}-{PARENT_ID}-01"), None),
        (headers(f"00-{'0' * 32}-{PARENT_ID}-01"), None),
        (headers(SAMPLED + "-extra"), None),
        ((("traceparent", b"\xff"),), None),
        (headers() + headers(), None),
        ((("TraceParent", SAMPLED.encode()),), None),
        ((), None),
        (None, None),
    ],
)
def test_header_carrier_accepts_only_valid_w3c_version_00(raw, expected):
    assert otel_context.carrier_from_headers(raw) == expected


def test_empty_trace_fields_are_not_written_and_old_reader_still_reads():
    record = record_without_trace()
    data = json.loads(record.model_dump_json())
    assert not set(TRACE_FIELDS) & set(data)
    assert not set(TRACE_FIELDS) & set(record.model_dump())
    # 쓰기를 끈 기록은 이전 이미지가 읽을 수 있다.
    OldJobRecord.model_validate_json(record.model_dump_json())
    # 기존 업무 필드의 null은 그대로 남긴다.
    assert data["owner"] is None and data["event"] is None


def test_written_trace_fields_round_trip_and_bad_values_do_not_break_reading():
    record = record_without_trace()
    record.trace_context = {"traceparent": SAMPLED}
    stored = record.model_dump_json()
    assert JobRecord.model_validate_json(stored).trace_context == {"traceparent": SAMPLED}
    # 켜진 뒤의 기록은 이전 strict 이미지가 거부한다. 롤백 하한은 이 호환 이미지다.
    with pytest.raises(ValueError):
        OldJobRecord.model_validate_json(stored)
    broken = json.loads(stored) | {"trace_context": {"traceparent": "bad"}}
    broken["result_trace_context"] = "not-a-carrier"
    loaded = JobRecord.model_validate(broken)
    assert loaded.trace_context is None and loaded.result_trace_context is None


@scenario
async def test_header_reaches_redis_child_and_result_header(monkeypatch):
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        fake_redis_for(a, monkeypatch)
        incoming = replace(
            message(cfg, process_request()),
            headers=headers(tracestate="vendor=abc", baggage="user=1"),
        )
        assert await RequestReceiver(a, producer).accept(incoming) == "accepted"
        receive = only("ai.receive")
        assert receive.kind == trace.SpanKind.CONSUMER
        assert receive.parent.span_id == int(PARENT_ID, 16)
        assert receive.context.trace_id == int(TRACE_ID, 16)

        job = await a.claim("process", "worker")
        assert job.trace_context == {
            "traceparent": f"00-{TRACE_ID}-{hex_span(receive)}-01",
            "tracestate": "vendor=abc",
        }
        assert "trace" not in json.dumps(job.request)

        lines = []
        handler = logging.Handler()
        handler.emit = lambda record: lines.append(json.loads(JsonFormatter().format(record)))
        root = logging.getLogger()
        level = root.level
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            # handoff_wait_ms가 있으면 대기 자식(--standby) 경로와 같은 run_job 호출이다.
            await execution.run_job(cfg, job.identity, job.owner, handoff_wait_ms=1.0)
        finally:
            root.removeHandler(handler)
            root.setLevel(level)

        run = only("ai.execute")
        assert run.parent.span_id == receive.context.span_id
        assert dict(run.attributes) == {"yeodam.operation": "process", "yeodam.result": "success"}
        stages = [
            span for span in EXPORTER.get_finished_spans() if span.name.startswith("ai.stage.")
        ]
        assert [span.attributes["yeodam.stage"] for span in stages] == [
            "download",
            "embedding",
            "clock_correction",
            "locating",
            "clustering",
            "filtering",
            "finalizing",
        ]
        assert all(span.parent.span_id == run.context.span_id for span in stages)
        assert all(span.attributes["yeodam.result"] == "success" for span in stages)

        # 단계 로그와 요약 로그의 trace_id가 실제 OTel Trace와 같다.
        logged = [line for line in lines if line.get("event") in {"ai_stage", "ai_job_summary"}]
        assert {line["trace_id"] for line in logged} == {TRACE_ID}
        assert {line["span_id"] for line in logged} == {hex_span(run)}

        stored = await b.get(job.identity)
        assert stored.status == "COMPLETED"
        assert stored.result_trace_context["traceparent"] == f"00-{TRACE_ID}-{hex_span(run)}-01"

        topic, value, key, sent_headers = await publish_once(a, producer)
        publish = only("ai.publish")
        assert publish.kind == trace.SpanKind.PRODUCER
        assert publish.parent.span_id == run.context.span_id
        sent = dict(sent_headers)
        assert set(sent) == {"traceparent", "tracestate"}
        assert sent["traceparent"].decode() == f"00-{TRACE_ID}-{hex_span(publish)}-01"
        assert sent["tracestate"] == b"vendor=abc"
        # 결과 body, key, topic은 그대로
        assert topic == job.result_topic and key == b"77"
        assert json.loads(value) == stored.event.model_dump(mode="json")
        # 실행이 끝난 뒤 전역 Context에 작업 Span이 남지 않는다.
        assert not trace.get_current_span().get_span_context().is_valid


@scenario
async def test_redelivery_keeps_first_context_and_uses_new_span(monkeypatch):
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        receiver = RequestReceiver(a, producer)
        first = replace(message(cfg, process_request()), headers=headers())
        other = f"00-{'1' * 32}-{'2' * 16}-01"
        again = replace(message(cfg, process_request(), offset=1), headers=headers(other))
        assert await receiver.accept(first) == "accepted"
        assert await receiver.accept(again) == "duplicate"
        received = spans("ai.receive")
        assert len(received) == 2
        assert received[0].context.span_id != received[1].context.span_id
        assert received[1].attributes["yeodam.result"] == "duplicate"
        job = await a.claim("process", "worker")
        assert job.trace_context["traceparent"] == f"00-{TRACE_ID}-{hex_span(received[0])}-01"


@scenario
async def test_publish_retry_uses_new_span_with_same_job(monkeypatch):
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        fake_redis_for(a, monkeypatch)
        await RequestReceiver(a, producer).accept(
            replace(message(cfg, process_request()), headers=headers())
        )
        job = await a.claim("process", "worker")
        await execution.run_job(cfg, job.identity, job.owner)
        producer.fail = True
        # 발행 lease를 짧게 해 같은 결과를 다시 시도하게 한다.
        a.cfg.JOBS_PUBLISH_LEASE_SEC = 0.01
        task = asyncio.create_task(publish_results(a, producer))
        while len(spans("ai.publish")) < 2:
            await asyncio.sleep(0.01)
        producer.fail = False
        while not producer.sent:
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        published = spans("ai.publish")
        assert len({span.context.span_id for span in published}) == len(published)
        assert published[0].attributes["yeodam.result"] == "retry"
        assert published[-1].attributes["yeodam.result"] == "success"
        assert len(producer.sent) == 1
        assert (await b.get(job.identity)).published


@scenario
async def test_watchdog_failure_without_child_context_publishes_under_receive():
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        await RequestReceiver(a, producer).accept(
            replace(message(cfg, process_request()), headers=headers())
        )
        job = await a.claim("process", "worker")
        # 실행 자식 없이 lease가 지나 감시 루프가 실패로 회수한 경우
        later = await a.now() + cfg.JOBS_LEASE_SEC + 1
        a.now = AsyncMock(return_value=later)
        await a.recover()
        stored = await b.get(job.identity)
        assert stored.status == "FAILED" and stored.result_trace_context is None
        _, _, _, sent_headers = await publish_once(a, producer)
        publish = only("ai.publish")
        assert publish.parent.span_id == only("ai.receive").context.span_id
        assert dict(sent_headers)["traceparent"].decode().startswith(f"00-{TRACE_ID}-")


@scenario
async def test_not_sampled_parent_stays_not_sampled(monkeypatch):
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        fake_redis_for(a, monkeypatch)
        await RequestReceiver(a, producer).accept(
            replace(message(cfg, story_request()), headers=headers(NOT_SAMPLED))
        )
        job = await a.claim("story", "worker")
        assert job.trace_context["traceparent"].endswith("-00")
        await execution.run_job(cfg, job.identity, job.owner)
        _, _, _, sent_headers = await publish_once(a, producer)
        assert dict(sent_headers)["traceparent"].decode().endswith("-00")
        # ParentBased 샘플러는 기록하지 않는다. 강제로 sampled로 바꾸지 않는다.
        assert EXPORTER.get_finished_spans() == ()


@pytest.mark.parametrize("raw", [None, headers("garbage")])
@scenario
async def test_missing_or_invalid_header_still_processes_as_new_root(raw, monkeypatch):
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        fake_redis_for(a, monkeypatch)
        incoming = message(cfg, process_request())
        if raw:
            incoming = replace(incoming, headers=raw)
        assert await RequestReceiver(a, producer).accept(incoming) == "accepted"
        receive = only("ai.receive")
        assert receive.parent is None
        job = await a.claim("process", "worker")
        await execution.run_job(cfg, job.identity, job.owner)
        assert (await b.get(job.identity)).status == "COMPLETED"
        assert only("ai.execute").parent.span_id == receive.context.span_id


@scenario
async def test_write_off_keeps_old_format_and_still_processes(monkeypatch):
    async with environment() as (cfg, a, b, producer):
        assert cfg.JOBS_TRACE_CONTEXT_WRITE is False
        fake_redis_for(a, monkeypatch)
        await RequestReceiver(a, producer).accept(
            replace(message(cfg, process_request()), headers=headers())
        )
        job = await a.claim("process", "worker")
        await execution.run_job(cfg, job.identity, job.owner)
        await publish_once(a, producer)
        raw = await a.redis.get(a.record_key(job.identity))
        assert not set(TRACE_FIELDS) & set(json.loads(raw))
        old = OldJobRecord.model_validate_json(raw)
        assert old.status == "COMPLETED" and old.published
        # Redis로 잇지 않으므로 실행은 새 Root, 수신 Span은 Header의 부모를 따른다.
        assert only("ai.receive").parent.span_id == int(PARENT_ID, 16)
        assert only("ai.execute").parent is None


@scenario
async def test_existing_fields_survive_after_write_is_turned_off(monkeypatch):
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        fake_redis_for(a, monkeypatch)
        await RequestReceiver(a, producer).accept(
            replace(message(cfg, process_request()), headers=headers())
        )
        job = await a.claim("process", "worker")
        cfg.JOBS_TRACE_CONTEXT_WRITE = False
        await execution.run_job(cfg, job.identity, job.owner)
        stored = await b.get(job.identity)
        assert stored.status == "COMPLETED"
        # 끄면 새 쓰기만 멈추고 이미 쓴 필드는 지우지 않는다.
        assert stored.trace_context is not None and stored.result_trace_context is None


@scenario
async def test_photos_ready_context_does_not_replace_process_context():
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        receiver = RequestReceiver(a, producer)
        ready = {
            "type": "photos_ready",
            "trip_id": 77,
            "execution_id": "execution-1",
            "batch_no": 1,
            "attachments": [{"trip_attachment_id": 101, "analyze_storage_key": "trips/77/a.jpg"}],
        }
        other = f"00-{'3' * 32}-{'4' * 16}-01"
        assert await receiver.accept(replace(message(cfg, ready), headers=headers(other))) == (
            "staged"
        )
        buffered = await a.redis.get(a.photo_batches_key(identity(77, "process", "execution-1")))
        assert "trace" not in buffered
        await receiver.accept(replace(message(cfg, process_request(), offset=1), headers=headers()))
        job = await a.claim("process", "worker")
        assert job.trace_context["traceparent"].startswith(f"00-{TRACE_ID}-")


@scenario
async def test_jobs_in_one_process_do_not_share_context(monkeypatch):
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        fake_redis_for(a, monkeypatch)
        receiver = RequestReceiver(a, producer)
        second = f"00-{'5' * 32}-{'6' * 16}-01"
        await receiver.accept(replace(message(cfg, process_request("one", 1)), headers=headers()))
        await receiver.accept(
            replace(message(cfg, process_request("two", 2)), headers=headers(second))
        )
        for _ in range(2):
            job = await a.claim("process", "worker")
            await execution.run_job(cfg, job.identity, job.owner)
        runs = spans("ai.execute")
        assert [format(span.context.trace_id, "032x") for span in runs] == [TRACE_ID, "5" * 32]
        assert not trace.get_current_span().get_span_context().is_valid


@scenario
async def test_without_otel_api_everything_still_works(monkeypatch):
    monkeypatch.setattr(otel_context, "otel_trace", None)
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        fake_redis_for(a, monkeypatch)
        await RequestReceiver(a, producer).accept(
            replace(message(cfg, process_request()), headers=headers())
        )
        job = await a.claim("process", "worker")
        assert job.trace_context is None
        await execution.run_job(cfg, job.identity, job.owner)
        _, _, _, sent_headers = await publish_once(a, producer)
        assert sent_headers is None
        assert (await b.get(job.identity)).published


SITECUSTOMIZE = """
import json, os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

class FileExporter(SpanExporter):
    def export(self, spans):
        with open(os.environ["YEODAM_TEST_SPAN_FILE"], "a") as out:
            for span in spans:
                out.write(json.dumps({
                    "name": span.name,
                    "pid": os.getpid(),
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "parent": format(span.parent.span_id, "016x") if span.parent else None,
                    "result": span.attributes.get("yeodam.result"),
                }) + "\\n")
        return SpanExportResult.SUCCESS

provider = TracerProvider()
# 실제 Cloud 설정처럼 배치 처리. 자식 종료 전 force_flush가 없으면 Span이 남지 않는다.
provider.add_span_processor(BatchSpanProcessor(FileExporter(), schedule_delay_millis=60000))
trace.set_tracer_provider(provider)
"""


@pytest.mark.skipif(
    not os.environ.get("JOBS_TEST_REDIS_URL"), reason="실제 Redis의 subprocess 검증"
)
@pytest.mark.parametrize("path", ["spawn", "standby"])
@scenario
async def test_real_child_process_restores_context_from_redis(path, tmp_path, monkeypatch):
    (tmp_path / "sitecustomize.py").write_text(textwrap.dedent(SITECUSTOMIZE))
    span_file = tmp_path / "spans.jsonl"
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}{os.getcwd()}")
    monkeypatch.setenv("YEODAM_TEST_SPAN_FILE", str(span_file))
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True, JOBS_HEARTBEAT_SEC=0.05) as (
        cfg,
        a,
        b,
        producer,
    ):
        await RequestReceiver(a, producer).accept(
            replace(message(cfg, process_request()), headers=headers())
        )
        receive = only("ai.receive")
        job = await a.claim("process", "worker")
        if path == "standby":
            standby = StandbyChildren(a, 1)
            await standby.fill()
            try:
                await asyncio.wait_for(standby.run(job), timeout=20)
            finally:
                await standby.close()
        else:
            await asyncio.wait_for(run_claimed(a, job), timeout=20)
        assert (await b.get(job.identity)).status == "COMPLETED"
        child = [json.loads(line) for line in span_file.read_text().splitlines()]
        run = [span for span in child if span["name"] == "ai.execute"]
        assert len(run) == 1 and run[0]["pid"] != os.getpid()
        assert run[0]["trace_id"] == TRACE_ID
        assert run[0]["parent"] == hex_span(receive)
        assert run[0]["result"] == "success"
        assert sum(span["parent"] == run[0]["span_id"] for span in child) == 7
        _, _, _, sent_headers = await publish_once(a, producer)
        publish = only("ai.publish")
        assert publish.parent.span_id == int(run[0]["span_id"], 16)
        assert dict(sent_headers)["traceparent"].decode() == (
            f"00-{TRACE_ID}-{hex_span(publish)}-01"
        )


def capture_logs():
    """운영 형식으로 바로 직렬화해 로그 당시의 trace_id를 남긴다."""
    lines = []
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(json.loads(JsonFormatter().format(record)))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    def stop():
        root.removeHandler(handler)
        root.setLevel(level)

    return lines, stop


@scenario
async def test_langfuse_on_does_not_take_over_cloud_context(langfuse_spans, monkeypatch):  # noqa: F811
    async with environment(JOBS_TRACE_CONTEXT_WRITE=True) as (cfg, a, b, producer):
        fake_redis_for(a, monkeypatch)
        await RequestReceiver(a, producer).accept(
            replace(message(cfg, process_request()), headers=headers())
        )
        job = await a.claim("process", "worker")
        seen = {}

        async def handler(request, reporter):
            await reporter.report("DOWNLOADING", 0, 1)
            with stage("probe", parent="download"):
                assert langfuse_tracing.active()
                # HTTPX 자동 계측처럼 다른 계측이 현재 Context 아래에 만드는 Span
                with trace.get_tracer("httpx").start_as_current_span("GET") as http:
                    seen["http"] = http
                seen["ids"] = otel_ids()
            return await mock_run(request, reporter)

        lines, stop = capture_logs()
        try:
            await execution.run_job(cfg, job.identity, job.owner, handler)
        finally:
            stop()

        run = only("ai.execute")
        assert format(run.context.trace_id, "032x") == TRACE_ID
        cloud = [span for span in EXPORTER.get_finished_spans() if span.name != "ai.receive"]
        # Cloud가 받는 Span은 모두 같은 Trace이고 부모가 Cloud Span이다.
        assert {format(span.context.trace_id, "032x") for span in cloud} == {TRACE_ID}
        assert only("GET").parent.span_id == run.context.span_id
        stages = [span for span in cloud if span.name.startswith("ai.stage.")]
        assert stages and all(span.parent.span_id == run.context.span_id for span in stages)
        assert "ai.stage.download.probe" in {span.name for span in stages}
        assert seen["ids"] == (TRACE_ID, hex_span(run))
        logged = [line for line in lines if line.get("event") in {"ai_stage", "ai_job_summary"}]
        assert logged and {line["trace_id"] for line in logged} == {TRACE_ID}

        # Langfuse 트레이스는 별도 Trace로 작업, 단계 구조를 그대로 유지한다.
        recorded = langfuse_finished(langfuse_spans)
        root = recorded["process"]
        assert root.parent is None
        assert format(root.context.trace_id, "032x") != TRACE_ID
        assert recorded["download.probe"].parent.span_id == root.context.span_id
        assert root.attributes["session.id"] == "trip-77"
        assert not trace.get_current_span().get_span_context().is_valid


def test_langfuse_root_and_stage_leave_global_context_alone(langfuse_spans):  # noqa: F811
    """API 요청(FastAPI Span) 안에서 Langfuse 트레이스를 열어도 다른 계측의 부모는 그대로"""

    async def run():
        tracer = trace.get_tracer("fastapi")
        with tracer.start_as_current_span("POST /search") as request:
            root = langfuse_tracing.start_root("search", trip_id=3)
            with stage("embed"):
                inner = trace.get_current_span()
                with tracer.start_as_current_span("GET embedding") as call:
                    pass
            root.end(status="success")
            after = trace.get_current_span()
        return request, inner, call, after

    request, inner, call, after = asyncio.run(run())
    assert inner is request and after is request
    assert call.parent.span_id == request.get_span_context().span_id
    recorded = langfuse_finished(langfuse_spans)
    assert recorded["embed"].parent.span_id == recorded["search"].context.span_id
    assert recorded["search"].parent is None
