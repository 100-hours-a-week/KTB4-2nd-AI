"""복제본을 합친 스토리 실행 상한과 종료 중인 API의 작업 처리를 검증한다."""

import asyncio

import pytest

from app.jobs.kafka import RequestReceiver
from app.jobs.redis_store import RedisJobStore
from app.jobs.runner import drain_story_runners, story_runners
from app.schemas.task import ErrorBody
from tests.test_v2_jobs import environment, message, scenario, story_request

STORY_RESULT = {
    "story_summary": "해변 사진",
    "blocks": [
        {
            "order_number": 1,
            "trip_place_id": 5001,
            "trip_attachment_id": 101,
            "day_label": "1일차, 해변",
            "detail_summary": "해변 사진",
            "memo": "해변의 풍경을 담았다.",
        }
    ],
    "verification": {"passed": True, "retry_count": 0},
}


async def enqueue(cfg, store, producer, count):
    receiver = RequestReceiver(store, producer)
    for trip in range(1, count + 1):
        await receiver.accept(message(cfg, story_request(trip=trip)))


@scenario
async def test_four_story_claims_across_replicas_and_safe_slot_reuse():
    async with environment() as (cfg, a, b, producer):
        await enqueue(cfg, a, producer, 6)
        claims = await asyncio.gather(
            *[(a if index % 2 else b).claim("story", str(index)) for index in range(16)]
        )
        active = [job for job in claims if job]
        assert len(active) == len({job.identity for job in active}) == 4
        assert await a.redis.zcard(a.key("queue:story")) == 2
        old = active[2]
        await b.cancel(old.trip_id, "story")
        await a.finish(old.identity, old.owner, "COMPLETED", result={})
        assert (await b.get(old.identity)).status == "CANCELED"
        replacement = await b.claim("story", "next")
        assert replacement is not None
        assert not await a.finish(old.identity, old.owner, "COMPLETED", result={})
        assert await a.claim("story", "extra") is None
        assert replacement.identity in await a.redis.mget(a.slot_keys("story"))


@scenario
async def test_old_image_terminal_record_does_not_keep_added_slot_occupied():
    async with environment() as (cfg, a, b, producer):
        await enqueue(cfg, a, producer, 5)
        active = [await a.claim("story", str(index)) for index in range(4)]
        old = active[-1]
        # 구 이미지의 복구 루프가 추가 슬롯을 모른 채 종료 기록을 저장한 경우.
        RedisJobStore.terminate(
            old, "FAILED", error=ErrorBody(code="WORKER_DEAD", message="실행 중단")
        )
        await a.redis.set(a.record_key(old.identity), old.model_dump_json())
        next_job = await b.claim("story", "new")
        assert next_job.trip_id == 5
        assert old.identity not in await a.redis.mget(a.slot_keys("story"))
        assert all([await a.get(job.identity) for job in active[:-1]])


@scenario
async def test_lower_limit_counts_running_jobs_in_all_slots():
    async with environment() as (cfg, a, b, producer):
        await enqueue(cfg, a, producer, 6)
        active = [await a.claim("story", str(index)) for index in range(4)]
        cfg.JOBS_STORY_CONCURRENCY = 1
        for old in active[:-1]:
            await b.finish(old.identity, old.owner, "COMPLETED", result=STORY_RESULT)
            assert await a.claim("story", "lower-limit") is None
        await b.finish(active[-1].identity, active[-1].owner, "COMPLETED", result=STORY_RESULT)
        assert await a.claim("story", "lower-limit") is not None
        assert await b.claim("story", "over-limit") is None


@pytest.mark.parametrize("cancel", [False, True])
@scenario
async def test_stop_after_claim_commit_returns_original_order_or_cancels(monkeypatch, cancel):
    async with environment() as (cfg, a, b, producer):
        await enqueue(cfg, a, producer, 3)
        queue = a.key("queue:story")
        original_queue = await a.redis.zrange(queue, 0, -1, withscores=True)
        stop = asyncio.Event()
        pipeline = b.redis.pipeline
        interrupted = False

        def delayed_pipeline(*args, **kwargs):
            pipe = pipeline(*args, **kwargs)
            execute = pipe.execute

            async def delayed_execute(*args, **kwargs):
                nonlocal interrupted
                claiming = not interrupted and any(
                    command[0:2] == ("ZREM", queue) for command, _ in pipe.command_stack
                )
                result = await execute(*args, **kwargs)
                if claiming:
                    interrupted = True
                    stop.set()
                    if cancel:
                        await a.cancel(1, "story")
                return result

            pipe.execute = delayed_execute
            return pipe

        monkeypatch.setattr(b.redis, "pipeline", delayed_pipeline)
        assert await b.claim("story", "shutting-down", should_stop=stop.is_set) is None
        record = await a.latest(1, "story")
        assert record.status == ("CANCELED" if cancel else "QUEUED")
        assert not any(await a.redis.mget(a.slot_keys("story")))
        expected = original_queue[1:] if cancel else original_queue
        assert await a.redis.zrange(queue, 0, -1, withscores=True) == expected
        if not cancel:
            assert record.owner is record.started_at is record.heartbeat_at is None
            assert (await a.claim("story", "alive")).trip_id == 1


@pytest.mark.parametrize("replicas", [1, 2, 3])
@scenario
async def test_runner_pools_drain_active_jobs_without_taking_next_job(monkeypatch, replicas):
    async with environment() as (cfg, a, b, producer):
        await enqueue(cfg, a, producer, 6)
        started, release = asyncio.Event(), asyncio.Event()
        active = set()
        seen = set()

        async def run_claimed(store, record):
            active.add(record.identity)
            seen.add(record.identity)
            if len(active) == 4:
                started.set()
            try:
                await release.wait()
                await store.finish(record.identity, record.owner, "COMPLETED", result=STORY_RESULT)
            finally:
                active.remove(record.identity)

        monkeypatch.setattr("app.jobs.runner.run_claimed", run_claimed)
        stops = [asyncio.Event() for _ in range(replicas)]
        tasks = [
            asyncio.create_task(story_runners(a if index % 2 else b, stop))
            for index, stop in enumerate(stops)
        ]
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            assert len(active) == 4
            for stop in stops:
                stop.set()
            release.set()
            await asyncio.gather(
                *[
                    drain_story_runners(task, stop, 1)
                    for task, stop in zip(tasks, stops, strict=True)
                ]
            )
            assert not active and len(seen) == 4
            assert await a.redis.zcard(a.key("queue:story")) == 2
            assert len(await a.pending()) == 4
            assert all([(await a.get(identity)).status == "COMPLETED" for identity in seen])
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@scenario
async def test_shutdown_deadline_records_failure_and_releases_all_slots(monkeypatch):
    async with environment() as (cfg, store, _, producer):
        await enqueue(cfg, store, producer, 5)
        started = asyncio.Event()
        running = set()

        async def run_claimed(store, record):
            running.add(record.identity)
            if len(running) == 4:
                started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr("app.jobs.runner.run_claimed", run_claimed)
        stop = asyncio.Event()
        task = asyncio.create_task(story_runners(store, stop))
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            await asyncio.wait_for(drain_story_runners(task, stop, 0.01), timeout=1)
            for identity in running:
                record = await store.get(identity)
                assert record.status == "FAILED"
                assert record.event.error.code == "WORKER_DEAD"
                assert identity in await store.pending()
            assert not any(await store.redis.mget(store.slot_keys("story")))
            assert await store.redis.zcard(store.key("queue:story")) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
