"""동일 검사를 fakeredis와 JOBS_TEST_REDIS_URL의 실제 Redis에 적용한다."""

import asyncio
import functools
import json
import os
import signal
import sys
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import fakeredis
import fakeredis.aioredis
import httpx
import pytest
from aiokafka import TopicPartition
from aiokafka.errors import (
    CommitFailedError,
    IllegalStateError,
    KafkaError,
    MessageSizeTooLargeError,
)
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.errors import AppError, PipelineCancelled
from app.jobs.config import JobsSettings
from app.jobs.execution import JobReporter, execute, handler_for
from app.jobs.kafka import (
    Incoming,
    InvalidMessage,
    RequestReceiver,
    consume_requests,
    publish_results,
)
from app.jobs.mock import run as mock_run
from app.jobs.models import MESSAGE, RESULT_MAX_BYTES, JobResultEvent
from app.jobs.redis_store import RedisJobStore
from app.jobs.runner import run_claimed
from app.schemas.task import ErrorBody
from app.v2_main import create_app


def scenario(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return run


class Producer:
    def __init__(self):
        self.sent = []
        self.fail = False

    async def send_and_wait(self, topic, value, *, key, headers=None):
        if self.fail:
            raise KafkaError()
        self.sent.append((topic, value, key, headers))


def process_request(execution="execution-1", trip=77):
    return {
        "type": "process",
        "trip_id": trip,
        "execution_id": execution,
        "trigger": "TRIP_CREATED",
        "trip_name": "제주 여행",
        "period": {"start_date": "2026-10-12", "end_date": "2026-10-14"},
        "regions": [{"latitude": 33.4, "longitude": 126.5}],
        "attachments": [{"trip_attachment_id": 101, "analyze_storage_key": "trips/77/a.jpg"}],
    }


def story_request(execution="story-1", trip=77):
    return {
        "type": "story",
        "trip_id": trip,
        "execution_id": execution,
        "trip_name": "제주 여행",
        "period": {"start_date": "2026-10-12", "end_date": "2026-10-14"},
        "mood": "PLAIN",
        "places": [
            {
                "trip_place_id": 5001,
                "place_name": "해변",
                "attachments": [
                    {
                        "trip_attachment_id": 101,
                        "analyze_storage_key": "a.jpg",
                        "taken_at": "2026-10-12T06:00:00+09:00",
                        "evaluation": 80,
                    }
                ],
            }
        ],
    }


def message(cfg, data, work=None, offset=0):
    work = work or ("story-create" if data.get("type") == "story" else "trip-create")
    return Incoming(
        getattr(cfg, f"KAFKA_{work.upper().replace(chr(45), chr(95))}_REQUEST_TOPIC"),
        0,
        offset,
        str(data.get("trip_id", 77)).encode(),
        json.dumps(data).encode(),
    )


@asynccontextmanager
async def environment(**changes):
    cfg = JobsSettings(
        _env_file=None,
        API_KEY="test",
        IMAGE_SOURCE="local",
        JOBS_ENV="test-" + uuid.uuid4().hex,
        JOBS_FAKE_PIPELINE=True,
        JOBS_POLL_SEC=0.001,
        **changes,
    )
    for name in type(cfg).model_fields:
        if name.endswith("_TOPIC"):
            work, direction = name.removeprefix("KAFKA_").removesuffix("_TOPIC").rsplit("_", 1)
            setattr(
                cfg,
                name,
                f"{cfg.JOBS_ENV}.yeodam.{work.lower().replace('_', '-')}.{direction.lower()}",
            )
    real_url = os.environ.get("JOBS_TEST_REDIS_URL")
    if real_url:
        clients = [Redis.from_url(real_url, decode_responses=True) for _ in range(2)]
        from pydantic import SecretStr

        cfg.REDIS_URL = SecretStr(real_url)
    else:
        server = fakeredis.FakeServer()
        clients = [
            fakeredis.aioredis.FakeRedis(server=server, decode_responses=True) for _ in range(2)
        ]
    stores = [RedisJobStore(client, cfg) for client in clients]
    try:
        yield cfg, *stores, Producer()
    finally:
        keys = [key async for key in clients[0].scan_iter(match=cfg.redis_prefix + "*")]
        if keys:
            await clients[0].delete(*keys)
        for client in clients:
            await client.aclose()


@scenario
async def test_atomic_registration_and_cross_topic_exclusion():
    async with environment() as (cfg, a, b, producer):
        receiver_a, receiver_b = RequestReceiver(a, producer), RequestReceiver(b, producer)
        outcomes = await asyncio.gather(
            *[receiver_a.accept(message(cfg, process_request())) for _ in range(20)]
        )
        assert outcomes.count("accepted") == 1 and outcomes.count("duplicate") == 19
        assert await a.redis.zcard(a.key("queue:process")) == 1
        assert (
            await receiver_b.accept(message(cfg, process_request("new"), "trip-update")) == "busy"
        )
        assert (await b.latest(77, "process")).execution_id == "execution-1"
        await a.cancel(77, "process")
        assert (
            await receiver_b.accept(message(cfg, process_request("new"), "trip-update"))
            == "duplicate"
        )
        assert (
            await receiver_b.accept(message(cfg, process_request("retry"), "trip-update"))
            == "accepted"
        )
        assert (await a.latest(77, "process")).result_topic.endswith("trip-update.result")


@scenario
async def test_only_one_runner_per_kind_but_story_is_independent():
    async with environment() as (cfg, a, b, p):
        receiver = RequestReceiver(a, p)
        for trip in (77, 78):
            await receiver.accept(message(cfg, process_request(trip=trip)))
        await receiver.accept(message(cfg, story_request()))
        claims = await asyncio.gather(a.claim("process", "a"), b.claim("process", "b"))
        assert sum(c is not None for c in claims) == 1
        assert await b.claim("process", "c") is None
        assert await b.claim("story", "d") is not None


@scenario
async def test_queued_cancel_records_event_and_terminal_cancel_is_conflict():
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        record = await b.cancel(77, "process")
        assert record.status == "CANCELED"
        assert record.event.status == "CANCELED" and record.event.result is None
        assert await a.claim("process", "worker") is None
        with pytest.raises(AppError) as error:
            await a.cancel(77, "process")
        assert error.value.http_status == 409
        assert record.identity in await b.pending()


@scenario
async def test_running_cancel_boundary_and_late_success_rejected():
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")
        canceled = await b.cancel(77, "process")
        assert canceled.status == "PROCESSING" and canceled.cancel_requested
        with pytest.raises(PipelineCancelled):
            await JobReporter(a, job.identity, job.owner).report("EMBEDDING", 0, 1)
        await execute(a, job.identity, job.owner, mock_run)
        assert (await b.latest(77, "process")).status == "CANCELED"
        assert not await a.finish(
            job.identity,
            job.owner,
            "FAILED",
            error=ErrorBody(code="INTERNAL_ERROR", message="late"),
        )


@scenario
async def test_expired_worker_cannot_renew_or_finish_and_new_execution_is_isolated():
    async with environment() as (cfg, a, b, p):
        clock = [100.0]
        a.now = b.now = AsyncMock(side_effect=lambda: clock[0])
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        old = await a.claim("process", "worker")
        clock[0] += cfg.JOBS_LEASE_SEC + 1
        assert not await b.heartbeat(old.identity, old.owner)
        await b.recover()
        assert (await a.get(old.identity)).event.error.code == "WORKER_DEAD"
        await RequestReceiver(b, p).accept(message(cfg, process_request("retry")))
        new = await a.claim("process", "worker")
        assert not await a.progress(old.identity, old.owner, "FINALIZING", 1, 1)
        assert not await a.finish(old.identity, old.owner, "CANCELED")
        assert (await b.latest(77, "process")).owner == new.owner


@scenario
async def test_timeout_even_if_heartbeat_keeps_running():
    async with environment(JOBS_TIMEOUT_SEC=10) as (cfg, a, b, p):
        clock = [100.0]
        a.now = b.now = AsyncMock(side_effect=lambda: clock[0])
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")
        clock[0] = 105.0
        assert await b.heartbeat(job.identity, job.owner)
        clock[0] = 111.0
        await b.recover()
        assert (await a.get(job.identity)).event.error.code == "TIMEOUT"


@pytest.mark.parametrize("kind", ["process", "story"])
@scenario
async def test_mock_execution_and_result_survive_new_clients(kind):
    async with environment() as (cfg, a, b, p):
        body = process_request() if kind == "process" else story_request()
        await RequestReceiver(a, p).accept(message(cfg, body))
        job = await a.claim(kind, "worker")
        await execute(a, job.identity, job.owner, mock_run)
        complete = await b.latest(77, kind)
        assert complete.status == "COMPLETED" and complete.progress.done == complete.progress.total
        assert "result" not in complete.response().model_dump()
        assert complete.event.result is not None
        assert not await b.finish(job.identity, "wrong-owner", "CANCELED")
        results = await asyncio.gather(
            a.claim_result(job.identity, "a"), b.claim_result(job.identity, "b")
        )
        assert sum(result is not None for result in results) == 1
        claimed = next(result for result in results if result)
        await b.acknowledge_result(job.identity, "wrong-owner")
        assert not (await a.get(job.identity)).published
        await b.acknowledge_result(job.identity, claimed.publish_owner)
        assert job.identity not in await a.pending()


@scenario
async def test_publication_failure_does_not_rerun_analysis():
    async with environment(JOBS_PUBLISH_LEASE_SEC=0.02) as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request(), "trip-update"))
        job = await a.claim("process", "worker")
        await execute(a, job.identity, job.owner, mock_run)
        p.fail = True
        loop = asyncio.create_task(publish_results(a, p))
        await asyncio.sleep(0.015)
        assert not p.sent and (await b.get(job.identity)).status == "COMPLETED"
        assert await b.claim("process", "other") is None
        p.fail = False
        try:
            async with asyncio.timeout(2):
                while not (await b.get(job.identity)).published:
                    await asyncio.sleep(0.005)
        finally:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)
        assert len(p.sent) == 1
        topic, value, key, _ = p.sent[0]
        assert topic.endswith("trip-update.result") and key == b"77"
        assert json.loads(value)["status"] == "COMPLETED"


@scenario
async def test_crash_after_broker_ack_can_redeliver_same_result():
    async with environment() as (cfg, a, b, p):
        clock = [100.0]
        a.now = b.now = AsyncMock(side_effect=lambda: clock[0])
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        await a.cancel(77, "process")
        job = await a.latest(77, "process")
        first = await a.claim_result(job.identity, "dead-api")
        await p.send_and_wait(first.result_topic, first.event.model_dump_json().encode(), key=b"77")
        clock[0] += cfg.JOBS_PUBLISH_LEASE_SEC + 1
        second = await b.claim_result(job.identity, "new-api")
        assert first.event == second.event
        await b.acknowledge_result(job.identity, "dead-api")
        assert not (await a.get(job.identity)).published
        await b.acknowledge_result(job.identity, "new-api")
        assert (await a.get(job.identity)).published


@pytest.mark.parametrize("mutation", ["json", "identity", "type", "key"])
@scenario
async def test_invalid_envelope_is_quarantined_before_acceptance(mutation):
    async with environment() as (cfg, a, b, p):
        body = process_request()
        if mutation == "identity":
            del body["execution_id"]
        if mutation == "type":
            body["type"] = "unknown"
        incoming = message(cfg, body)
        if mutation == "json":
            incoming = Incoming(incoming.topic, 1, 4, b"77", b"{broken")
        if mutation == "key":
            incoming = Incoming(incoming.topic, 1, 4, b"78", incoming.value)
        for replica in (a, b):
            with pytest.raises(InvalidMessage):
                await RequestReceiver(replica, p).accept(incoming)
            assert not p.sent
        assert await RequestReceiver(a, p).accept(incoming) == "quarantined"
        assert dict(p.sent[0][3])["attempts"] == b"3"
        assert p.sent[0][0].endswith("request-dlt")
        assert p.sent[0][1] == incoming.value
        assert await a.claim("process", "w") is None


@scenario
async def test_identifiable_invalid_body_returns_failed_in_result_topic():
    async with environment() as (cfg, a, b, p):
        body = process_request()
        del body["attachments"]
        assert await RequestReceiver(a, p).accept(message(cfg, body)) == "accepted"
        job = await b.latest(77, "process")
        assert job.status == "FAILED" and job.event.error.code == "INVALID_REQUEST"
        assert not p.sent


@scenario
async def test_dlt_publish_failure_is_not_acknowledged():
    async with environment() as (cfg, a, b, p):
        p.fail = True
        body = process_request()
        body.pop("execution_id")
        for _ in range(2):
            with pytest.raises(InvalidMessage):
                await RequestReceiver(a, p).accept(message(cfg, body))
        with pytest.raises(KafkaError):
            await RequestReceiver(a, p).accept(message(cfg, body))
        assert await a.claim("process", "worker") is None


class Consumer:
    def __init__(self, records, commit_error=False):
        self.records = iter(records)
        self.commits = []
        self.commit_error = commit_error

    async def getone(self):
        try:
            return next(self.records)
        except StopIteration:
            raise asyncio.CancelledError() from None

    async def commit(self, offsets):
        if self.commit_error:
            raise CommitFailedError()
        self.commits.append(offsets)


@scenario
async def test_offset_not_advanced_when_redis_registration_fails(monkeypatch):
    receiver = AsyncMock()
    receiver.accept.side_effect = RedisConnectionError("offline")
    consumer = Consumer([Incoming("requests", 1, 5, b"77", b"{}")])

    async def interrupt(_):
        raise asyncio.CancelledError()

    monkeypatch.setattr("app.jobs.kafka.asyncio.sleep", interrupt)
    with pytest.raises(asyncio.CancelledError):
        await consume_requests(consumer, receiver)
    assert consumer.commits == []


@scenario
async def test_commits_only_registered_record_not_other_partition():
    receiver = AsyncMock()
    consumer = Consumer([Incoming("requests", 1, 5, b"77", b"{}")])
    with pytest.raises(asyncio.CancelledError):
        await consume_requests(consumer, receiver)
    receiver.accept.assert_awaited_once()
    assert consumer.commits == [{TopicPartition("requests", 1): 6}]


@scenario
async def test_rebalance_redelivery_uses_existing_execution():
    async with environment() as (cfg, a, b, p):
        incoming = message(cfg, process_request())
        first = Consumer([incoming], commit_error=True)
        with pytest.raises(asyncio.CancelledError):
            await consume_requests(first, RequestReceiver(a, p))
        second = Consumer([incoming])
        with pytest.raises(asyncio.CancelledError):
            await consume_requests(second, RequestReceiver(b, p))
        assert await a.redis.zcard(a.key("queue:process")) == 1
        assert len(second.commits) == 1


@scenario
async def test_revoked_partition_commit_moves_to_next_record():
    first = Incoming("requests", 1, 5, b"77", b"{}")
    second = Incoming("requests", 0, 9, b"78", b"{}")
    receiver = AsyncMock()
    consumer = Consumer([first, second])
    consumer.commit = AsyncMock(side_effect=[IllegalStateError("revoked"), None])
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(consume_requests(consumer, receiver), timeout=1)
    assert receiver.accept.await_args_list[0].args == (first,)
    assert receiver.accept.await_args_list[1].args == (second,)
    assert consumer.commit.await_count == 2


@scenario
async def test_missing_story_handler_fails_before_kafka_starts(monkeypatch):
    async with environment() as (cfg, a, b, p):
        cfg.JOBS_FAKE_PIPELINE = False
        cfg.JOBS_STORY_HANDLER = None
        runtime = AsyncMock()
        monkeypatch.setattr("app.v2_main.KafkaRuntime", lambda _: runtime)
        app = create_app(cfg, a)
        with pytest.raises(ValueError, match="story"):
            async with app.router.lifespan_context(app):
                pytest.fail("실행 함수 없이 요청을 받아서는 안 됨")
        runtime.start.assert_not_awaited()


@scenario
async def test_blocked_child_exits_when_parent_is_killed():
    child_code = (
        "import sys, time; from app.jobs.process_guard import parent_guard\n"
        "with parent_guard(int(sys.argv[1]), interval=0.01):\n"
        " print('ready', flush=True)\n"
        " time.sleep(30)\n"
        " print('unexpected-side-effect', flush=True)\n"
    )
    parent_code = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], str(os.getpid())])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    parent = await asyncio.create_subprocess_exec(
        sys.executable, "-c", parent_code, child_code, stdout=asyncio.subprocess.PIPE
    )
    child_pid = None
    try:
        child_pid = int(await asyncio.wait_for(parent.stdout.readline(), timeout=10))
        assert await asyncio.wait_for(parent.stdout.readline(), timeout=10) == b"ready\n"
        parent.kill()
        # 자식이 stdout을 상속했으므로 EOF는 자식까지 종료된 뒤에만 도착한다.
        remaining = await asyncio.wait_for(parent.stdout.read(), timeout=5)
        assert remaining == b""
        await parent.wait()
    finally:
        for pid in (parent.pid, child_pid):
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        await parent.wait()


@scenario
async def test_two_api_instances_share_status_cancel_and_auth():
    async with environment() as (cfg, a, b, p):
        apps = [create_app(cfg, store, start_kafka=False) for store in (a, b)]
        async with (
            apps[0].router.lifespan_context(apps[0]),
            apps[1].router.lifespan_context(apps[1]),
        ):
            async with (
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=apps[0]), base_url="http://api-a"
                ) as ca,
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=apps[1]), base_url="http://api-b"
                ) as cb,
            ):
                headers = {"Authorization": "Bearer test", "X-Request-ID": "request-123"}
                assert (await ca.get("/trips/77/process")).status_code == 401
                assert (await ca.get("/trips/77/process", headers=headers)).status_code == 404
                await RequestReceiver(a, p).accept(message(cfg, process_request()))
                ra = await ca.get("/trips/77/process", headers=headers)
                rb = await cb.get("/trips/77/process", headers=headers)
                assert ra.json() == rb.json()
                assert ra.headers["X-Request-ID"] == "request-123"
                assert "result" not in ra.json() and "error" not in ra.json()
                assert (await cb.delete("/trips/77/process", headers=headers)).json()[
                    "status"
                ] == "CANCELED"
                assert (await ca.get("/trips/77/process", headers=headers)).json()[
                    "status"
                ] == "CANCELED"
                assert (
                    await ca.post("/trips/77/process", headers=headers, json={})
                ).status_code == 405
                assert (
                    await ca.post(
                        "/query/parse",
                        headers=headers,
                        json={"query": "바다 사진", "reference_date": "2026-10-14"},
                    )
                ).status_code == 503


@pytest.mark.parametrize("error", [ValueError("bad result"), RuntimeError("secret-url")])
@scenario
async def test_handler_failure_is_failed_without_leaking_exception(error):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")

        async def fail(request, reporter):
            raise error

        await execute(a, job.identity, job.owner, fail)
        result = (await b.latest(77, "process")).event
        assert result.status == "FAILED" and result.error.code == "INTERNAL_ERROR"
        assert "secret-url" not in result.model_dump_json()


def test_requests_ignore_extra_and_result_envelope_rejects_extra():
    MESSAGE.validate_python({**process_request(), "future_field": "allowed"})
    with pytest.raises(ValueError):
        JobResultEvent(
            type="process_result",
            trip_id=77,
            execution_id="a",
            status="CANCELED",
            future_field="invalid",
        )


def test_real_handler_not_silently_replaced_by_mock():
    cfg = JobsSettings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local")
    with pytest.raises(ValueError):
        handler_for(cfg, "process")


@pytest.mark.skipif(
    not os.environ.get("JOBS_TEST_REDIS_URL"), reason="실제 Redis의 subprocess 검증"
)
@pytest.mark.parametrize("kind", ["process", "story"])
@scenario
async def test_real_subprocess_execution(kind):
    from pydantic import SecretStr

    async with environment(JOBS_HEARTBEAT_SEC=0.05) as (cfg, a, b, p):
        cfg.REDIS_URL = SecretStr(os.environ["JOBS_TEST_REDIS_URL"])
        request = process_request() if kind == "process" else story_request()
        await RequestReceiver(a, p).accept(message(cfg, request))
        job = await a.claim(kind, "worker")
        await asyncio.wait_for(run_claimed(a, job), timeout=10)
        assert (await b.get(job.identity)).status == "COMPLETED"


@scenario
async def test_pending_leased_results_do_not_starve_other_results():
    async with environment() as (cfg, a, b, p):
        clock = [100.0]
        a.now = b.now = AsyncMock(side_effect=lambda: clock[0])
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.cancel(77, "process")
        await a.claim_result(job.identity, "publisher")
        assert job.identity not in await b.pending()
        clock[0] += cfg.JOBS_PUBLISH_LEASE_SEC + 1
        assert job.identity in await b.pending()


@scenario
async def test_running_cancel_http_ack_does_not_publish_before_stopping():
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")
        app = create_app(cfg, b, start_kafka=False)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://api"
            ) as client:
                headers = {"Authorization": "Bearer test"}
                assert (await client.delete("/trips/77/process", headers=headers)).json()[
                    "status"
                ] == "CANCELED"
                assert (await client.get("/trips/77/process", headers=headers)).json()[
                    "status"
                ] == "PROCESSING"
                assert not await a.pending()
        await execute(a, job.identity, job.owner, mock_run)
        assert (await b.get(job.identity)).event.status == "CANCELED"


@scenario
async def test_invalid_handler_result_is_failed():
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, story_request()))
        job = await a.claim("story", "worker")

        async def invalid(request, reporter):
            return {"blocks": [], "unexpected": True}

        await execute(a, job.identity, job.owner, invalid)
        assert (await b.get(job.identity)).event.error.code == "INTERNAL_ERROR"


@scenario
async def test_redis_loss_is_reported_as_missing_not_fake_completion():
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        keys = [key async for key in a.redis.scan_iter(match=cfg.redis_prefix + "*")]
        await a.redis.delete(*keys)
        with pytest.raises(AppError) as error:
            await b.latest(77, "process")
        assert error.value.code == "TASK_NOT_FOUND"
        assert await b.claim("process", "w") is None
        assert not await b.pending()


@scenario
async def test_api_store_outage_returns_503(monkeypatch):
    async with environment() as (cfg, a, b, p):
        app = create_app(cfg, a, start_kafka=False)
        async with app.router.lifespan_context(app):
            monkeypatch.setattr(
                a.redis, "get", AsyncMock(side_effect=RedisConnectionError("offline"))
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://api"
            ) as client:
                response = await client.get(
                    "/trips/77/process", headers={"Authorization": "Bearer test"}
                )
                assert response.status_code == 503
                assert "offline" not in response.text


@scenario
async def test_fifo_registration_does_not_depend_on_timestamp():
    async with environment() as (cfg, a, b, p):
        a.now = AsyncMock(side_effect=[200.0, 100.0, 100.0])
        for trip in (99, 98, 97):
            await RequestReceiver(a, p).accept(message(cfg, process_request(str(trip), trip)))
        for trip in (99, 98, 97):
            job = await b.claim("process", "worker")
            assert job.trip_id == trip
            await b.finish(job.identity, job.owner, "CANCELED")


@scenario
async def test_dlt_three_failures_ack_before_commit_and_cloud_log(monkeypatch, caplog):
    async with environment() as (cfg, a, b, p):
        body = process_request()
        body.pop("execution_id")
        incoming = message(cfg, body, offset=8)
        consumer = Consumer([incoming])
        p.fail = True
        sleeps = []

        async def retry(_):
            sleeps.append(True)
            assert consumer.commits == []
            if len(sleeps) == 3:
                # 세 번째 검증 뒤 첫 DLT 발행은 실패한다. 다음 발행만 성공시킨다.
                assert not p.sent
                assert "kafka_dlt_published" not in [
                    getattr(r, "event", "") for r in caplog.records
                ]
                p.fail = False

        monkeypatch.setattr("app.jobs.kafka.asyncio.sleep", retry)
        with pytest.raises(asyncio.CancelledError):
            await consume_requests(consumer, RequestReceiver(a, p))
        assert len(sleeps) == 3
        assert consumer.commits == [{TopicPartition(incoming.topic, 0): 9}]
        assert len(p.sent) == 1 and p.sent[0][2] == incoming.key
        headers = dict(p.sent[0][3])
        assert headers["original-topic"] == incoming.topic.encode()
        assert headers["original-partition"] == b"0" and headers["original-offset"] == b"8"
        assert headers["attempts"] == b"3" and headers["failure"] == b"ValueError"
        events = [r for r in caplog.records if getattr(r, "event", "") == "kafka_dlt_published"]
        assert len(events) == 1
        assert (events[0].service, events[0].trip_id, events[0].attempts) == ("ai", "77", 3)


@scenario
async def test_redis_outage_does_not_count_as_invalid_request(monkeypatch):
    async with environment() as (cfg, a, b, p):
        incoming = message(cfg, process_request())
        a.register = AsyncMock(side_effect=RedisConnectionError("offline"))
        receiver = RequestReceiver(a, p)
        consumer = Consumer([incoming])
        sleeps = 0

        async def retry(_):
            nonlocal sleeps
            sleeps += 1
            if sleeps == 4:
                raise asyncio.CancelledError()

        monkeypatch.setattr("app.jobs.kafka.asyncio.sleep", retry)
        with pytest.raises(asyncio.CancelledError):
            await consume_requests(consumer, receiver)
        assert not p.sent and not consumer.commits
        assert not [key async for key in a.redis.scan_iter(match=a.key("invalid:*"))]


@scenario
async def test_kafka_startup_outage_keeps_http_and_recovery_alive(monkeypatch):
    from app.api.services.query_service import QueryService
    from app.search.mock import create_mock_context

    created = []
    available = False

    class Client:
        def __init__(self, *topics, **options):
            self.topics, self.options = topics, options
            self.stopped = False
            created.append(self)

        async def start(self):
            if not available:
                raise KafkaError("offline")

        async def getone(self):
            await asyncio.Future()

        async def stop(self):
            self.stopped = True

        async def send_and_wait(self, *args, **kwargs):
            pass

    monkeypatch.setattr("app.jobs.kafka.AIOKafkaProducer", Client)
    monkeypatch.setattr("app.jobs.kafka.AIOKafkaConsumer", Client)
    async with environment(JOBS_PUBLISH_LEASE_SEC=0.1, WATCHDOG_INTERVAL_SEC=1) as (
        cfg,
        a,
        b,
        p,
    ):
        clock = [100.0]
        a.now = b.now = AsyncMock(side_effect=lambda: clock[0])
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "dead-worker")
        clock[0] += cfg.JOBS_LEASE_SEC + 1
        app = create_app(cfg, a, QueryService(create_mock_context()))
        async with asyncio.timeout(5):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test",
                    headers={"Authorization": "Bearer test"},
                ) as client:
                    while (await a.get(job.identity)).status != "FAILED":
                        await asyncio.sleep(0.01)
                    assert (await client.get("/health")).status_code == 200
                    status = await client.get("/trips/77/process")
                    assert status.status_code == 200 and status.json()["status"] == "FAILED"
                    response = await client.post(
                        "/query/parse",
                        json={
                            "query": "바다 사진",
                            "reference_date": "2026-10-14",
                        },
                    )
                    assert response.status_code == 200
                    assert not (await a.get(job.identity)).published
                    assert app.state.kafka.healthy()
                    available = True
                    clock[0] += 1
                    while not (await a.get(job.identity)).published:
                        clock[0] += 0.1
                        await asyncio.sleep(0.01)
                    assert (await client.get("/health")).status_code == 200
        assert all(c.stopped for c in created)
        consumers = [c for c in created if c.topics]
        assert {c.options["group_id"] for c in consumers} == set(cfg.consumer_groups.values())
        assert all(len(c.topics) == 1 and not c.options["enable_auto_commit"] for c in consumers)
        assert all(c.options["auto_offset_reset"] == "earliest" for c in consumers)
        publishers = [c for c in created if not c.topics]
        assert all(
            c.options["acks"] == "all" and c.options["enable_idempotence"] for c in publishers
        )
        assert all(c.options["compression_type"] == "lz4" for c in publishers)


def test_topic_environment_values_are_used_without_deriving_from_jobs_env(monkeypatch):
    expected = {}
    for index, work in enumerate(("TRIP_CREATE", "TRIP_UPDATE", "STORY_CREATE")):
        request, result = f"cloud.{index}.request", f"cloud.{index}.result"
        monkeypatch.setenv(f"KAFKA_{work}_REQUEST_TOPIC", request)
        monkeypatch.setenv(f"KAFKA_{work}_RESULT_TOPIC", result)
        expected[request] = ("story" if work == "STORY_CREATE" else "process", result)
    cfg = JobsSettings(
        _env_file=None, API_KEY="test", IMAGE_SOURCE="local", JOBS_ENV="independent-redis-prefix"
    )
    assert cfg.topics == expected


def test_lz4_record_batch_round_trip():
    from aiokafka.codec import has_lz4
    from aiokafka.record.default_records import DefaultRecordBatch, DefaultRecordBatchBuilder

    assert has_lz4()
    body = json.dumps({"photos": [process_request()] * 200}).encode()
    builder = DefaultRecordBatchBuilder(
        magic=2,
        compression_type=3,
        is_transactional=False,
        producer_id=-1,
        producer_epoch=-1,
        base_sequence=0,
        batch_size=len(body) * 2,
    )
    builder.append(offset=0, timestamp=0, key=b"77", value=body, headers=[])
    encoded = builder.build()
    batch = DefaultRecordBatch(encoded)
    assert batch.compression_type == 3 and batch.validate_crc()
    assert len(encoded) < len(body)
    records = list(batch)
    assert len(records) == 1 and records[0].key == b"77" and records[0].value == body


@scenario
async def test_oversized_result_is_retained_with_size_in_log(caplog):
    from aiokafka.errors import MessageSizeTooLargeError

    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.cancel(77, "process")
        producer = AsyncMock()
        producer.send_and_wait.side_effect = MessageSizeTooLargeError()
        loop = asyncio.create_task(publish_results(a, producer))
        try:
            async with asyncio.timeout(2):
                while not any(
                    getattr(r, "event", "") == "ai_job_result_too_large" for r in caplog.records
                ):
                    await asyncio.sleep(0.01)
            saved = await b.get(job.identity)
            assert not saved.published and saved.event == job.event
            assert await b.claim("process", "worker") is None
            log = next(
                r for r in caplog.records if getattr(r, "event", "") == "ai_job_result_too_large"
            )
            assert log.payload_bytes == len(job.event.model_dump_json().encode())
        finally:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)


@pytest.mark.parametrize(
    "broken", ["deep_json", "identity_surrogate", "body_surrogate", "nonfinite"]
)
@scenario
async def test_unserializable_json_reaches_dlt_and_next_request_is_consumed(broken, monkeypatch):
    async with environment() as (cfg, a, b, p):
        body = process_request()
        if broken == "identity_surrogate":
            body["execution_id"] = "bad\ud800"
        elif broken == "body_surrogate":
            body["trip_name"] = "bad\ud800"
        elif broken == "nonfinite":
            body["attachments"][0]["latitude"] = float("inf")
        raw = (
            b"[" * 10000 + b"0" + b"]" * 10000
            if broken == "deep_json"
            else json.dumps(body).encode()
        )
        incoming = Incoming(cfg.KAFKA_TRIP_CREATE_REQUEST_TOPIC, 0, 0, b"77", raw)
        following = message(cfg, process_request("good", 78), offset=1)
        consumer = Consumer([incoming, following])
        monkeypatch.setattr("app.jobs.kafka.asyncio.sleep", AsyncMock())
        with pytest.raises(asyncio.CancelledError):
            await consume_requests(consumer, RequestReceiver(a, p))
        assert len(p.sent) == 1 and p.sent[0][1] == raw
        assert dict(p.sent[0][3])["attempts"] == b"3"
        assert len(consumer.commits) == 2
        assert consumer.commits[-1] == {TopicPartition(incoming.topic, 0): 2}
        assert (await b.latest(78, "process")).execution_id == "good"


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
@scenario
async def test_nonfinite_process_result_fails_without_corrupting_redis(coordinate):
    async with environment() as (cfg, a, b, p):
        await RequestReceiver(a, p).accept(message(cfg, process_request()))
        job = await a.claim("process", "worker")

        async def bad_result(request, reporter):
            return {
                "places": [
                    {
                        "place_id": "p1",
                        "latitude": coordinate,
                        "longitude": 126.5,
                        "representative_attachment_id": 101,
                        "attachments": [
                            {
                                "trip_attachment_id": 101,
                                "latitude": 33.4,
                                "longitude": 126.5,
                                "region_origin": "EXIF",
                                "evaluation": 90,
                            }
                        ],
                    }
                ],
                "unclassified": [],
                "failed": [],
                "clock_offsets": [],
            }

        await execute(a, job.identity, job.owner, bad_result)
        saved = await b.get(job.identity)
        assert saved.status == "FAILED" and saved.event.error.code == "INTERNAL_ERROR"
        assert saved.event.result is None
        assert saved.identity in await b.pending()
        assert (await b.latest(77, "process")).response().status == "FAILED"
        await RequestReceiver(a, p).accept(message(cfg, process_request("next", 78)))
        assert (await a.claim("process", "worker")).trip_id == 78


@pytest.mark.parametrize("still_assigned", [True, False])
@scenario
async def test_oversized_dlt_preserves_offset_and_other_partition_progress(
    still_assigned, monkeypatch, caplog
):
    async with environment() as (cfg, a, b, _):
        topic = cfg.KAFKA_TRIP_CREATE_REQUEST_TOPIC
        bad = Incoming(topic, 0, 7, b"77", b"{broken")
        good = message(cfg, process_request("good", 78), offset=20)
        good = Incoming(topic, 1, good.offset, good.key, good.value)
        blocked = TopicPartition(topic, 0)

        class PartitionConsumer(Consumer):
            def __init__(self):
                super().__init__([bad, good])
                self.actions = []

            def assignment(self):
                return {blocked} if still_assigned else set()

            def seek(self, partition, offset):
                self.actions.append(("seek", partition, offset))

            def pause(self, partition):
                self.actions.append(("pause", partition))

        consumer = PartitionConsumer()
        producer = AsyncMock()
        producer.send_and_wait.side_effect = MessageSizeTooLargeError()
        monkeypatch.setattr("app.jobs.kafka.asyncio.sleep", AsyncMock())
        with pytest.raises(asyncio.CancelledError):
            await consume_requests(consumer, RequestReceiver(a, producer))
        assert consumer.actions == (
            [("seek", blocked, 7), ("pause", blocked)] if still_assigned else []
        )
        assert consumer.commits == [{TopicPartition(topic, 1): 21}]
        assert (await b.latest(78, "process")).execution_id == "good"
        producer.send_and_wait.assert_awaited_once()
        entry = next(
            r for r in caplog.records if getattr(r, "event", "") == "ai_kafka_dlt_too_large"
        )
        assert (entry.source_topic, entry.partition, entry.offset) == (topic, 0, 7)
        assert entry.payload_bytes == len(bad.value)
        assert entry.partition_paused is still_assigned


@pytest.mark.parametrize("kind", ["process", "story"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_result_size_limit_counts_utf8_envelope_and_includes_boundary(kind, delta):
    event = JobResultEvent(
        type=f"{kind}_result",
        trip_id=77,
        execution_id="e" * 128,
        status="FAILED",
        error=ErrorBody(code="DOWNLOAD_FAILED", message="실패", detail={"keys": ["한😀\n"]}),
    )
    prefix_size = len(event.model_dump_json().encode("utf-8"))
    event.error.detail["keys"][0] += "x" * (RESULT_MAX_BYTES + delta - prefix_size)
    assert len(event.model_dump_json().encode("utf-8")) == RESULT_MAX_BYTES + delta
    bounded = event.enforce_size_limit()
    if delta <= 0:
        assert bounded is event
    else:
        assert bounded.status == "FAILED" and bounded.error.code == "RESULT_TOO_LARGE"
        assert bounded.result is None
        assert bounded.type == event.type and bounded.execution_id == event.execution_id
        assert bounded.error.detail == {
            "max_bytes": 524288,
            "actual_bytes": 524289,
        }
        assert len(bounded.model_dump_json().encode("utf-8")) < 1024


@pytest.mark.parametrize(
    "kind,status", [("process", "COMPLETED"), ("process", "FAILED"), ("story", "FAILED")]
)
@scenario
async def test_oversized_result_is_failed_before_storage_and_publishes_small_event(kind, status):
    async with environment() as (cfg, a, b, producer):
        body = process_request() if kind == "process" else story_request()
        await RequestReceiver(a, producer).accept(message(cfg, body))
        job = await a.claim(kind, "worker")
        options = (
            {
                "result": {
                    "places": [],
                    "unclassified": [],
                    "failed": [],
                    "clock_offsets": [
                        {"device_model": "기" * RESULT_MAX_BYTES, "days": 0, "matched_pairs": 1}
                    ],
                }
            }
            if status == "COMPLETED"
            else {
                "error": ErrorBody(
                    code="DOWNLOAD_FAILED",
                    message="실패",
                    detail={"keys": ["x" * RESULT_MAX_BYTES]},
                )
            }
        )
        assert await a.finish(job.identity, job.owner, status, **options)
        saved = await b.get(job.identity)
        assert saved.status == "FAILED" and saved.event.error.code == "RESULT_TOO_LARGE"
        assert saved.event.error.detail["actual_bytes"] > RESULT_MAX_BYTES
        assert saved.event.result is None
        assert (await b.latest(77, kind)).response().status == "FAILED"
        assert saved.identity in await b.pending()
        assert not await a.finish(
            job.identity, job.owner, "COMPLETED", result=options.get("result")
        )
        publisher = asyncio.create_task(publish_results(a, producer))
        try:
            async with asyncio.timeout(3):
                while not (await b.get(job.identity)).published:
                    await asyncio.sleep(0.01)
            assert len(producer.sent) == 1
            topic, payload, key, _ = producer.sent[0]
            assert topic == saved.result_topic and key == b"77"
            assert len(payload) <= RESULT_MAX_BYTES
            assert json.loads(payload) == saved.event.model_dump(mode="json")
        finally:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
