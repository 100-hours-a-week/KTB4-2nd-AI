"""photos_ready는 메타데이터만 보관하고 최종 process만 실행 대기열에 넣는다."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from aiokafka import TopicPartition
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.errors import AppError
from app.jobs.execution import execute
from app.jobs.kafka import InvalidMessage, RequestReceiver, consume_requests
from app.jobs.mock import run as mock_run
from app.jobs.models import PHOTOS_READY_MAX_BYTES, PhotoBatchBuffer, PhotosReadyMessage
from app.v2_main import create_app
from tests.test_v2_jobs import Consumer, environment, message, process_request, scenario


def ready_request(batch_no=1, ids=(101,), execution="execution-1", trip=77):
    return {
        "type": "photos_ready",
        "trip_id": trip,
        "execution_id": execution,
        "batch_no": batch_no,
        "attachments": [
            {"trip_attachment_id": photo, "analyze_storage_key": f"trips/{trip}/{photo}.jpg"}
            for photo in ids
        ],
    }


def buffer_key(store, body):
    return store.photo_batches_key(PhotosReadyMessage.model_validate(body).identity)


@scenario
async def test_all_200_photos_stay_idle_until_process_and_are_shared_between_apis():
    async with environment() as (cfg, a, b, producer):
        for start in range(0, 200, 10):
            body = ready_request(start // 10 + 1, range(start + 1, start + 11))
            assert await RequestReceiver(a, producer).accept(message(cfg, body)) == "staged"
        buffer = PhotoBatchBuffer.model_validate_json(await b.redis.get(buffer_key(b, body)))
        assert len(buffer.batches) == 20
        assert sum(map(len, buffer.batches.values())) == 200
        assert 0 < await b.redis.ttl(buffer_key(b, body)) <= cfg.JOBS_PHOTOS_READY_TTL_SEC
        assert await b.claim("process", "runner") is None
        assert await b.pending() == [] and producer.sent == []
        app = create_app(cfg, b, start_kafka=False)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://api-b"
            ) as client:
                response = await client.get(
                    "/trips/77/process", headers={"Authorization": "Bearer test"}
                )
                assert response.status_code == 404
                assert response.json()["error"]["code"] == "TASK_NOT_FOUND"
                assert (await client.get("/health")).json()["queued"] == 0


@pytest.mark.parametrize("count", [1, 16, 200])
@pytest.mark.parametrize("work", ["trip-create", "trip-update"])
@scenario
async def test_only_final_manifest_runs_without_waiting_for_missing_batches(count, work):
    async with environment() as (cfg, a, b, producer):
        # 최종 목록에 없는 사진을 실행 대상에 섞지 않는다.
        batch = ready_request(ids=(999,))
        await RequestReceiver(a, producer).accept(message(cfg, batch, work))
        final = process_request()
        final["trigger"] = "PHOTOS_ADDED" if work == "trip-update" else "TRIP_CREATED"
        final["attachments"] = ready_request(ids=range(1, count + 1))["attachments"]
        assert await RequestReceiver(b, producer).accept(message(cfg, final, work)) == "accepted"
        assert not await a.redis.exists(buffer_key(a, batch))
        job = await a.claim("process", "runner")
        assert job.progress.total == count
        assert job.result_topic.endswith(work + ".result")
        assert [p["trip_attachment_id"] for p in job.request["attachments"]] == list(
            range(1, count + 1)
        )
        await execute(a, job.identity, job.owner, mock_run)
        completed = await b.latest(77, "process")
        assert completed.status == "COMPLETED"
        assert len(completed.event.result["unclassified"]) == count
        assert await a.pending() == [job.identity]
        assert await b.claim("process", "runner") is None


@scenario
async def test_duplicate_batches_are_idempotent_across_replicas():
    async with environment() as (cfg, a, b, producer):
        body = ready_request(ids=(102, 101))
        outcomes = await asyncio.gather(
            *[RequestReceiver(store, producer).accept(message(cfg, body)) for store in (a, b) * 10]
        )
        assert outcomes.count("staged") == 1 and outcomes.count("duplicate") == 19
        body["attachments"].reverse()
        body["future_field"] = "ignored"
        assert await RequestReceiver(b, producer).accept(message(cfg, body)) == "duplicate"
        buffer = PhotoBatchBuffer.model_validate_json(await b.redis.get(buffer_key(b, body)))
        assert len(buffer.batches) == 1 and len(buffer.batches["1"]) == 2


@pytest.mark.parametrize("timing", ["queued", "running", "completed", "canceled", "busy"])
@scenario
async def test_late_batch_never_reopens_closed_execution(timing):
    async with environment() as (cfg, a, b, producer):
        batch = ready_request()
        await RequestReceiver(a, producer).accept(message(cfg, batch))
        if timing == "busy":
            await RequestReceiver(a, producer).accept(message(cfg, process_request("other")))
        await RequestReceiver(b, producer).accept(message(cfg, process_request()))
        if timing in {"running", "completed"}:
            job = await a.claim("process", "runner")
            if timing == "completed":
                await execute(a, job.identity, job.owner, mock_run)
        if timing == "canceled":
            await b.cancel(77, "process")
        before = (await b.latest(77, "process")).model_dump()
        assert await RequestReceiver(b, producer).accept(message(cfg, batch)) == "closed"
        assert not await b.redis.exists(buffer_key(b, batch))
        assert (await a.latest(77, "process")).model_dump() == before


@scenario
async def test_process_racing_with_batches_leaves_no_orphan_buffer():
    async with environment() as (cfg, a, b, producer):
        for trip in range(1, 21):
            batch = ready_request(trip=trip)
            outcomes = await asyncio.gather(
                RequestReceiver(a, producer).accept(message(cfg, batch)),
                RequestReceiver(b, producer).accept(message(cfg, process_request(trip=trip))),
            )
            assert outcomes[0] in {"staged", "closed"} and outcomes[1] == "accepted"
            assert not await a.redis.exists(buffer_key(a, batch))
        assert await b.redis.zcard(b.key("queue:process")) == 20


@scenario
async def test_buffer_expiry_does_not_block_final_process():
    async with environment(JOBS_PHOTOS_READY_TTL_SEC=60) as (cfg, a, b, producer):
        body = ready_request()
        await RequestReceiver(a, producer).accept(message(cfg, body))
        assert 0 < await a.redis.ttl(buffer_key(a, body)) <= 60
        await a.redis.expire(buffer_key(a, body), 0)
        assert (
            await RequestReceiver(b, producer).accept(message(cfg, process_request())) == "accepted"
        )
        assert (await b.claim("process", "runner")).progress.total == 1


@pytest.mark.parametrize(
    "mutation", ["empty", "too_many", "batch_no", "duplicate_id", "topic", "type_list", "type_dict"]
)
@scenario
async def test_invalid_batch_uses_dlt_without_failing_final_process(mutation):
    async with environment() as (cfg, a, b, producer):
        body = ready_request()
        if mutation == "empty":
            body["attachments"] = []
        elif mutation == "too_many":
            body = ready_request(ids=range(1, 18))
        elif mutation == "batch_no":
            body["batch_no"] = 0
        elif mutation == "duplicate_id":
            body = ready_request(ids=(101, 101))
        elif mutation == "type_list":
            body["type"] = ["photos_ready"]
        elif mutation == "type_dict":
            body["type"] = {"name": "photos_ready"}
        work = "story-create" if mutation == "topic" else "trip-create"
        incoming = message(cfg, body, work)
        for store in (a, b):
            with pytest.raises(InvalidMessage):
                await RequestReceiver(store, producer).accept(incoming)
        assert await RequestReceiver(a, producer).accept(incoming) == "quarantined"
        assert producer.sent[0][:3] == (incoming.topic + "-dlt", incoming.value, b"77")
        with pytest.raises(AppError):
            await b.latest(77, "process")
        assert await a.pending() == []
        assert (
            await RequestReceiver(b, producer).accept(message(cfg, process_request())) == "accepted"
        )


@pytest.mark.parametrize("conflict", ["batch", "photo_key", "topic"])
@scenario
async def test_conflicting_batch_keeps_original_metadata(conflict):
    async with environment() as (cfg, a, b, producer):
        body = ready_request()
        await RequestReceiver(a, producer).accept(message(cfg, body))
        before = await a.redis.get(buffer_key(a, body))
        if conflict == "photo_key":
            body["batch_no"] = 2
        if conflict != "topic":
            body["attachments"][0]["analyze_storage_key"] = "changed.jpg"
        work = "trip-update" if conflict == "topic" else "trip-create"
        with pytest.raises(InvalidMessage):
            await RequestReceiver(b, producer).accept(message(cfg, body, work, offset=1))
        assert await a.redis.get(buffer_key(a, body)) == before


@pytest.mark.parametrize("unique_photos", [True, False])
@scenario
async def test_buffer_is_bounded_even_with_repeated_photos(unique_photos):
    async with environment() as (cfg, a, b, producer):
        for number in range(1, 201):
            body = ready_request(number, (number if unique_photos else 101,))
            await RequestReceiver(a, producer).accept(message(cfg, body, offset=number))
        excess = ready_request(201, (201 if unique_photos else 101,))
        with pytest.raises(InvalidMessage):
            await RequestReceiver(b, producer).accept(message(cfg, excess, offset=201))
        buffer = PhotoBatchBuffer.model_validate_json(await a.redis.get(buffer_key(a, excess)))
        assert len(buffer.batches) == 200


@scenario
async def test_batch_commit_waits_for_redis_and_replay_does_not_duplicate(monkeypatch):
    async with environment() as (cfg, a, b, producer):
        body = ready_request()
        incoming = message(cfg, body, offset=5)
        consumer = Consumer([incoming])
        original_stage = a.stage_photos
        a.stage_photos = AsyncMock(side_effect=RedisConnectionError("offline"))

        async def reconnect(_):
            assert not consumer.commits
            assert not await b.redis.exists(buffer_key(b, body))
            a.stage_photos = original_stage

        monkeypatch.setattr("app.jobs.kafka.asyncio.sleep", reconnect)
        with pytest.raises(asyncio.CancelledError):
            await consume_requests(consumer, RequestReceiver(a, producer))
        assert consumer.commits == [{TopicPartition(incoming.topic, 0): 6}]
        saved = json.loads(await b.redis.get(buffer_key(b, body)))
        assert len(saved["batches"]) == 1
        replay = Consumer([incoming])
        with pytest.raises(asyncio.CancelledError):
            await consume_requests(replay, RequestReceiver(b, producer))
        assert len(replay.commits) == 1
        assert json.loads(await b.redis.get(buffer_key(b, body))) == saved


@pytest.mark.parametrize("key", ["a" * 1024, "가" * 341 + "a", "😀" * 256])
@scenario
async def test_s3_key_1024_byte_boundary_is_accepted(key):
    async with environment() as (cfg, store, _, producer):
        body = ready_request()
        body["attachments"][0]["analyze_storage_key"] = key
        assert len(key.encode()) == 1024
        assert await RequestReceiver(store, producer).accept(message(cfg, body)) == "staged"


@pytest.mark.parametrize("key", ["a" * 1025, "가" * 342, "😀" * 257, "a" * 60000])
@scenario
async def test_oversized_s3_key_goes_to_dlt_without_buffer(key):
    async with environment() as (cfg, store, replica, producer):
        body = ready_request()
        body["attachments"][0]["analyze_storage_key"] = key
        incoming = message(cfg, body)
        for _ in range(2):
            with pytest.raises(InvalidMessage):
                await RequestReceiver(store, producer).accept(incoming)
        assert await RequestReceiver(replica, producer).accept(incoming) == "quarantined"
        assert not await store.redis.exists(buffer_key(store, ready_request()))
        assert await store.pending() == []
        assert (
            await RequestReceiver(store, producer).accept(message(cfg, process_request()))
            == "accepted"
        )


@scenario
async def test_buffer_byte_limit_counts_json_escapes_and_preserves_prior_batches(monkeypatch):
    async with environment() as (cfg, store, replica, producer):
        first = ready_request()
        # JSON 직렬화 시 제어 문자 하나가 여섯 바이트로 늘어나는 경우도 검사한다.
        first["attachments"][0]["analyze_storage_key"] = "\x00" * 1024
        topic = message(cfg, first).topic
        one_batch = PhotoBatchBuffer(
            request_topic=topic, batches={"1": PhotosReadyMessage.model_validate(first).attachments}
        ).model_dump_json()
        monkeypatch.setattr(
            "app.jobs.redis_store.PHOTOS_READY_MAX_BYTES", len(one_batch.encode("utf-8"))
        )
        assert await RequestReceiver(store, producer).accept(message(cfg, first)) == "staged"
        saved = await store.redis.get(buffer_key(store, first))
        assert saved == one_batch  # 정확히 상한인 버퍼는 허용한다.
        second = {**first, "batch_no": 2}
        incoming = message(cfg, second, offset=1)
        for _ in range(2):
            with pytest.raises(InvalidMessage):
                await RequestReceiver(replica, producer).accept(incoming)
        assert await RequestReceiver(store, producer).accept(incoming) == "quarantined"
        assert await replica.redis.get(buffer_key(replica, first)) == saved
        assert (
            await RequestReceiver(store, producer).accept(message(cfg, process_request()))
            == "accepted"
        )
        assert not await replica.redis.exists(buffer_key(replica, first))


@scenario
async def test_200_unique_photos_fit_buffer_limit_with_maximum_escaped_keys():
    async with environment() as (cfg, store, replica, producer):
        for start in range(0, 200, 10):
            body = ready_request(start // 10 + 1, range(start + 1, start + 11))
            for photo in body["attachments"]:
                photo["analyze_storage_key"] = "\x00" * 1024
            assert await RequestReceiver(store, producer).accept(message(cfg, body)) == "staged"
        saved = await replica.redis.get(buffer_key(replica, body))
        assert 1024 * 1024 < len(saved.encode("utf-8")) < PHOTOS_READY_MAX_BYTES
        assert await replica.claim("process", "runner") is None
