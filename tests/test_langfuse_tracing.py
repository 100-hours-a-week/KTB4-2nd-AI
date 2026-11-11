"""Langfuse 기록. 네트워크 없이 메모리 exporter로 span 구조와 속성을 확인한다."""

import asyncio
import json
import logging
import uuid
from types import SimpleNamespace

import httpx
import pytest
from langfuse import Langfuse, LangfuseMedia
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.tracing import end_trace, stage, start_trace
from app.infra.vllm_client import VllmClient
from app.infra.vlm import VLMTransientError
from app.observability import langfuse_tracing as lt

SIGNED = "https://bucket.s3.amazonaws.com/trips/1/a.jpg?X-Amz-Signature=secret"
ATTR = "langfuse.observation."
JPEG = b"\xff\xd8\xff"


@pytest.fixture
def exporter(monkeypatch):
    spans = InMemorySpanExporter()
    client = Langfuse(
        # SDK는 공개 키마다 전송 자원을 하나만 만든다. 테스트마다 새 키로 앞 테스트와 분리한다.
        public_key=f"pk-test-{uuid.uuid4().hex}",
        secret_key="sk-test",
        # 점수와 사진 업로드는 연결이 거부되어 버려진다. span은 메모리로만 내보낸다.
        base_url="http://127.0.0.1:9",
        tracer_provider=TracerProvider(),
        span_exporter=spans,
    )
    monkeypatch.setattr(lt, "_client", client)
    monkeypatch.setattr(lt, "_ready", True)

    async def fetched(urls):
        return {
            url: LangfuseMedia(content_bytes=b"\xff\xd8\xff", content_type="image/jpeg")
            for url in urls
        }

    monkeypatch.setattr(lt, "_fetch_images", fetched)
    yield spans
    client.shutdown()


def finished(spans: InMemorySpanExporter) -> dict:
    """이름이 같은 단계와 생성 호출을 구분하려고 생성 호출은 이름 앞에 gen:을 붙인다."""
    lt.flush()
    return {
        ("gen:" if span.attributes.get(ATTR + "type") == "generation" else "") + span.name: span
        for span in spans.get_finished_spans()
    }


def reply(content='{"place": "성산"}', status=200, usage=True):
    def handler(request):
        if status != 200:
            return httpx.Response(status)
        body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        if usage:
            body["usage"] = {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128}
        return httpx.Response(200, json=body)

    return VllmClient("http://vlm/v1", "gemma", transport=httpx.MockTransport(handler))


def ask(client, **kwargs):
    return client.chat(
        [{"role": "user", "content": "어디인가요?"}],
        [SIGNED],
        32,
        response_format={"type": "json_object"},
        **kwargs,
    )


def test_job_trace_nests_stage_and_generation(exporter):
    async def run():
        trace, token = start_trace("process")
        root = lt.start_root("process", trip_id=7, input={"photos": 1}, metadata={"job_id": "j1"})
        async with reply() as client:
            with stage("grouping"):
                with lt.slot_wait(1.5):
                    await ask(client)
        end_trace(token)
        root.end(output={"groups": 1}, status="COMPLETED", metadata={"stages_ms": {"a": 1.0}})
        return root.trace_id

    trace_id = asyncio.run(run())
    spans = finished(exporter)
    root, step, call = spans["process"], spans["grouping"], spans["gen:grouping"]
    assert format(root.context.trace_id, "032x") == trace_id
    assert step.parent.span_id == root.context.span_id
    assert call.parent.span_id == step.context.span_id
    assert root.attributes["session.id"] == "trip-7"
    assert root.attributes["langfuse.trace.name"] == "process"
    assert json.loads(root.attributes[ATTR + "metadata.status"]) == "COMPLETED"
    assert call.attributes[ATTR + "type"] == "generation"
    assert call.attributes[ATTR + "model.name"] == "gemma"
    assert json.loads(call.attributes[ATTR + "usage_details"]) == {
        "input": 120,
        "output": 8,
        "total": 128,
    }
    assert json.loads(call.attributes[ATTR + "output"]) == {"place": "성산"}
    assert json.loads(call.attributes[ATTR + "metadata.gpu_wait_ms"]) == 1500.0
    sent = call.attributes[ATTR + "input"]
    assert "@@@langfuseMedia:type=image/jpeg" in sent and "Signature" not in sent


def test_calls_outside_a_trace_are_not_recorded(exporter):
    async def run():
        async with reply() as client:
            with stage("embed"):
                await ask(client)

    asyncio.run(run())
    assert finished(exporter) == {}


def test_unfetched_image_keeps_only_unsigned_address(exporter, monkeypatch):
    async def nothing(urls):
        return {}

    monkeypatch.setattr(lt, "_fetch_images", nothing)

    async def run():
        root = lt.start_root("search")
        async with reply() as client:
            await ask(client)
        root.end(status="success")

    asyncio.run(run())
    sent = finished(exporter)["gen:vlm"].attributes[ATTR + "input"]
    assert "trips/1/a.jpg" in sent and "Signature" not in sent


def test_failed_call_is_marked_and_still_raises(exporter):
    async def run():
        root = lt.start_root("story", trip_id=3)
        async with reply(status=503) as client:
            with pytest.raises(VLMTransientError):
                await ask(client)
        root.end(status="FAILED")

    asyncio.run(run())
    spans = finished(exporter)
    assert spans["gen:vlm"].attributes[ATTR + "level"] == "ERROR"
    assert spans["gen:vlm"].attributes[ATTR + "status_message"] == "VLMTransientError"
    assert spans["story"].attributes[ATTR + "level"] == "ERROR"


def test_tool_calls_are_recorded_as_output(exporter):
    def handler(request):
        message = {
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "look", "arguments": "{}"}}
            ],
        }
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "tool_calls"}]}
        )

    tools = [{"type": "function", "function": {"name": "look", "parameters": {}}}]

    async def run():
        root = lt.start_root("process")
        async with VllmClient(
            "http://vlm/v1", "gemma", transport=httpx.MockTransport(handler)
        ) as c:
            await c.chat([{"role": "user", "content": "보기"}], [], 32, tools=tools)
        root.end(status="COMPLETED")

    asyncio.run(run())
    call = finished(exporter)["gen:vlm"]
    assert json.loads(call.attributes[ATTR + "output"])["tool_calls"] == [
        {"name": "look", "arguments": "{}"}
    ]
    assert json.loads(call.attributes[ATTR + "metadata.tools"]) == ["look"]


def test_langfuse_errors_do_not_change_the_result(monkeypatch, caplog):
    class Broken:
        def start_observation(self, **_):
            raise RuntimeError("down")

        def start_as_current_observation(self, **_):
            raise RuntimeError("down")

    monkeypatch.setattr(lt, "_client", Broken())
    monkeypatch.setattr(lt, "_ready", True)
    monkeypatch.setattr(lt, "_recording", lambda: True)
    caplog.set_level(logging.WARNING)

    async def run():
        root = lt.start_root("search")
        async with reply() as client:
            with stage("answer"):
                turn = await ask(client)
        root.end(status="success")
        return turn

    assert json.loads(asyncio.run(run()).content) == {"place": "성산"}
    actions = {r.action for r in caplog.records if getattr(r, "event", "") == "ai_langfuse_failed"}
    assert {"root_start", "span_start", "generation_start"} <= actions


def test_disabled_without_keys(monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        lt.LangfuseSettings, "model_config", {**lt.LangfuseSettings.model_config, "env_file": None}
    )
    lt.reset()
    try:
        assert lt.client() is None and not lt.active()
        assert lt.start_root("search") is lt.NOOP
    finally:
        lt.reset()


def test_evaluation_metrics_become_trace_scores(exporter, monkeypatch):
    scores = []
    monkeypatch.setattr(lt._client, "create_score", lambda **kw: scores.append(kw))
    lt.score_evaluations(
        "abc",
        [
            {
                "feature": "process",
                "dataset": "eval200",
                "samples": 200,
                "metrics": {"bcubed_precision": 0.9, "bcubed_recall": 0.8, "note": "x"},
            }
        ],
    )
    assert [(s["name"], s["value"], s["trace_id"]) for s in scores] == [
        ("process.bcubed_precision", 0.9, "abc"),
        ("process.bcubed_recall", 0.8, "abc"),
    ]
    assert scores[0]["metadata"]["dataset"] == "eval200"


def test_shutdown_gives_up_after_the_time_limit(monkeypatch, caplog):
    import threading

    stuck = threading.Event()

    class Stuck:
        def shutdown(self):
            stuck.wait(5)

    monkeypatch.setattr(lt, "_client", Stuck())
    monkeypatch.setattr(lt, "_ready", True)
    monkeypatch.setattr(lt, "FLUSH_TIMEOUT_SEC", 0.2)
    caplog.set_level(logging.WARNING)
    try:
        asyncio.run(lt.ashutdown())
    finally:
        stuck.set()
    assert any(getattr(r, "action", None) == "flush_timeout" for r in caplog.records)


def slow_fetch(monkeypatch, seconds):
    async def fetched(urls):
        await asyncio.sleep(seconds)
        return {
            u: LangfuseMedia(content_bytes=b"\xff\xd8\xff", content_type="image/jpeg") for u in urls
        }

    monkeypatch.setattr(lt, "_fetch_images", fetched)


def test_slow_image_download_does_not_delay_or_time_out_the_call(exporter, monkeypatch):
    slow_fetch(monkeypatch, 0.5)

    async def run():
        root = lt.start_root("search")
        async with reply() as client:
            started = asyncio.get_running_loop().time()
            async with asyncio.timeout(0.3):
                turn = await ask(client)
            took = asyncio.get_running_loop().time() - started
        root.end(status="success")
        await lt.ashutdown()
        return turn, took

    turn, took = asyncio.run(run())
    assert json.loads(turn.content) == {"place": "성산"} and took < 0.3
    call = finished(exporter)["gen:vlm"]
    # 사진은 호출이 끝난 뒤 붙고, generation의 종료 시각은 응답 시각이다.
    assert "@@@langfuseMedia" in call.attributes[ATTR + "input"]
    assert (call.end_time - call.start_time) / 1e9 < 0.3


def test_failed_call_raises_without_waiting_for_images(exporter, monkeypatch):
    slow_fetch(monkeypatch, 0.5)

    async def run():
        root = lt.start_root("story")
        async with reply(status=503) as client:
            started = asyncio.get_running_loop().time()
            with pytest.raises(VLMTransientError):
                await ask(client)
            took = asyncio.get_running_loop().time() - started
        root.end(status="FAILED")
        await lt.ashutdown()
        return took

    assert asyncio.run(run()) < 0.3
    assert finished(exporter)["gen:vlm"].attributes[ATTR + "level"] == "ERROR"


def test_backlogged_uploads_skip_images(exporter, monkeypatch):
    fetched = []

    async def fetch(urls):
        fetched.extend(urls)
        return {}

    monkeypatch.setattr(lt, "_fetch_images", fetch)
    monkeypatch.setattr(lt, "_media_backlogged", lambda _: True)

    async def run():
        root = lt.start_root("process")
        async with reply() as client:
            await client.chat(
                [{"role": "user", "content": [{"type": "image"}, {"type": "image"}]}],
                [SIGNED, "data:image/jpeg;base64,/9j/"],
                32,
            )
        root.end(status="COMPLETED")

    asyncio.run(run())
    call = finished(exporter)["gen:vlm"]
    sent = call.attributes[ATTR + "input"]
    assert "base64" not in sent and "upload backlog" in sent and "Signature" not in sent
    assert json.loads(call.attributes[ATTR + "metadata.images_omitted"]) == "upload_backlog"
    assert fetched == []


def test_spans_of_other_instrumentation_do_not_start_langfuse_records(exporter):
    other = TracerProvider().get_tracer("apm")

    async def run():
        with other.start_as_current_span("http request"):
            async with reply() as client:
                with stage("embed"):
                    await ask(client)

    asyncio.run(run())
    assert finished(exporter) == {}


def test_generation_names_join_stage_and_call(exporter):
    async def run():
        trace, token = start_trace("process")
        root = lt.start_root("process")
        async with reply() as client:
            with stage("clustering"), lt.call_name("describe"):
                await ask(client)
        end_trace(token)
        root.end(status="COMPLETED")

    asyncio.run(run())
    assert "gen:clustering.describe" in finished(exporter)


def test_completed_job_trace_carries_evaluation_scores(exporter, monkeypatch, tmp_path):
    """정리 작업 1건이 트레이스 하나가 되고, 정답 기반 자동 평가가 그 트레이스의 점수가 된다."""
    from app.jobs.execution import execute
    from app.jobs.kafka import RequestReceiver
    from app.observability import datasets
    from tests.test_v2_auto_evaluation import eval_trip, handler_settings, reference
    from tests.test_v2_jobs import environment, message

    scores = []
    monkeypatch.setattr(lt._client, "create_score", lambda **kw: scores.append(kw))
    monkeypatch.setattr(datasets, "reference_sets", lambda: (reference(["a"] * 12),))

    async def run():
        async with environment() as (cfg, a, b, p):
            handler = handler_settings(monkeypatch, tmp_path, cfg)
            await RequestReceiver(a, p).accept(message(cfg, eval_trip(tmp_path)))
            job = await a.claim("process", "worker")
            await execute(a, job.identity, job.owner, handler.run)
            return job

    job = asyncio.run(run())
    spans = finished(exporter)
    root = spans["process"]
    assert root.attributes["session.id"].startswith("trip-")
    assert json.loads(root.attributes[ATTR + "metadata.status"]) == "COMPLETED"
    assert json.loads(root.attributes[ATTR + "metadata.job_id"]) == job.execution_id
    assert spans["finalizing.evaluate"].parent.span_id == root.context.span_id
    trace_id = format(root.context.trace_id, "032x")
    assert scores and {s["trace_id"] for s in scores} == {trace_id}
    assert "process.bcubed_precision" in {s["name"] for s in scores}


def test_root_starts_a_new_trace_under_other_instrumentation(exporter):
    other = TracerProvider().get_tracer("apm")

    async def run():
        with other.start_as_current_span("http request") as server:
            root = lt.start_root("search")
            root.end(status="success")
            return server.get_span_context().trace_id

    apm_trace = asyncio.run(run())
    root = finished(exporter)["search"]
    assert root.parent is None and root.context.trace_id != apm_trace


@pytest.mark.parametrize(
    ("status", "level"),
    [("COMPLETED", None), ("partial", "WARNING"), ("CANCELED", "WARNING"), ("TIMEOUT", "ERROR")],
)
def test_root_level_follows_status(exporter, status, level):
    lt.start_root("story_rewrite").end(status=status)
    assert finished(exporter)["story_rewrite"].attributes.get(ATTR + "level") == level


class FakeQueues:
    def __init__(self, queues, fail=None, hang=False):
        self.queues, self.fail, self.hang = queues, fail, hang
        self.items = []
        self.listed = 0

    async def list_queues(self, **_):
        self.listed += 1
        if self.fail:
            raise self.fail
        if self.hang:
            await asyncio.sleep(10)
        return SimpleNamespace(data=self.queues)

    async def create_queue_item(self, queue_id, *, object_id, object_type):
        self.items.append((queue_id, object_id, object_type))


def review_client(monkeypatch, **kwargs):
    queues = FakeQueues(**kwargs)
    client = SimpleNamespace(async_api=SimpleNamespace(annotation_queues=queues))
    monkeypatch.setattr(lt, "_client", client)
    monkeypatch.setattr(lt, "_ready", True)
    monkeypatch.setattr(lt, "_review_queue_id", None)
    return queues


def enqueue(*trace_ids):
    async def run():
        for trace_id in trace_ids:
            lt.enqueue_review(trace_id)
        # 기다리지 않고 돌아오므로 남은 태스크는 종료 처리처럼 따로 기다린다.
        if lt._pending:
            await asyncio.wait(set(lt._pending))

    asyncio.run(run())


def test_review_queue_receives_the_trace(monkeypatch):
    queues = review_client(
        monkeypatch,
        queues=[
            SimpleNamespace(id="q-other", name="다른"),
            SimpleNamespace(id="q1", name=lt.REVIEW_QUEUE),
        ],
    )
    enqueue("trace-a")
    enqueue("trace-b")
    assert queues.items == [("q1", "trace-a", "TRACE"), ("q1", "trace-b", "TRACE")]
    # 대기열 ID는 처음 한 번만 찾는다.
    assert queues.listed == 1


def test_missing_review_queue_only_warns(monkeypatch, caplog):
    queues = review_client(monkeypatch, queues=[])
    caplog.set_level(logging.WARNING)
    enqueue("trace-a")
    assert queues.items == []
    assert any(getattr(r, "action", None) == "review_queue_missing" for r in caplog.records)


@pytest.mark.parametrize("kwargs", [{"fail": httpx.ConnectError("x")}, {"hang": True}])
def test_review_queue_failure_does_not_raise(monkeypatch, caplog, kwargs):
    review_client(monkeypatch, queues=[], **kwargs)
    monkeypatch.setattr(lt, "REVIEW_TIMEOUT_SEC", 0.1)
    caplog.set_level(logging.WARNING)
    enqueue("trace-a")
    assert any(getattr(r, "action", None) == "review_queue" for r in caplog.records)


def test_review_queue_is_skipped_without_client_or_trace(monkeypatch):
    queues = review_client(monkeypatch, queues=[SimpleNamespace(id="q1", name=lt.REVIEW_QUEUE)])
    enqueue(None)
    monkeypatch.setattr(lt, "_client", None)
    enqueue("trace-a")
    assert queues.items == []


def story_result():
    return {
        "story_summary": "제주 해변",
        "blocks": [
            {
                "order_number": 1,
                "trip_place_id": 5001,
                "trip_attachment_id": 101,
                "day_label": "1일차, 해변",
                "detail_summary": "아침 바다",
                "memo": "아침에 바다를 봤다.",
            }
        ],
        "verification": {"passed": True, "retry_count": 0},
    }


@pytest.mark.parametrize("succeeds", [True, False])
def test_completed_story_trace_goes_to_review_queue(exporter, monkeypatch, succeeds):
    """끝난 스토리 작업만 그 트레이스를 사람 평가 대기열에 넣는다."""
    from app.core.errors import PipelineError
    from app.jobs.execution import execute
    from app.jobs.kafka import RequestReceiver
    from tests.test_v2_jobs import environment, message, story_request

    queued = []
    monkeypatch.setattr(lt, "enqueue_review", queued.append)

    async def handler(request, reporter):
        if not succeeds:
            raise PipelineError("실패")
        return story_result()

    async def run():
        async with environment() as (cfg, a, b, p):
            await RequestReceiver(a, p).accept(message(cfg, story_request()))
            job = await a.claim("story", "worker")
            await execute(a, job.identity, job.owner, handler)

    asyncio.run(run())
    root = finished(exporter)["story"]
    assert queued == ([format(root.context.trace_id, "032x")] if succeeds else [])


def test_review_queue_cancelled_at_shutdown_is_logged(monkeypatch, caplog):
    review_client(monkeypatch, queues=[], hang=True)
    caplog.set_level(logging.WARNING)

    async def run():
        lt.enqueue_review("trace-a")
        await asyncio.sleep(0)
        for task in set(lt._pending):
            task.cancel()
        await asyncio.wait(set(lt._pending))

    asyncio.run(run())
    assert any(getattr(r, "action", None) == "review_queue" for r in caplog.records)
