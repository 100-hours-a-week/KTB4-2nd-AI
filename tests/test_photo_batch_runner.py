"""정리 실행기의 사진 묶음 실행. process 우선 선점, 처리 시간 상한, 묶음 자식의 처리와 로그"""

import asyncio
import json
import logging
import os

import pytest

from app.core.errors import DownloadFailed
from app.jobs.execution import run_batch
from app.jobs.kafka import RequestReceiver
from app.jobs.runner import StandbyChildren, run_photo_batch, runner_loop
from tests.test_photos_ready import ready_request
from tests.test_v2_jobs import FakeChild, environment, message, process_request, scenario


class BatchChild(FakeChild):
    """FakeChild에 사진 묶음 처리를 더함. 묶음은 받은 내용만 기록, hang이면 종료 신호까지 대기"""

    def __init__(self, store, args, piped: bool, order: list, hang: bool, exit_code: int) -> None:
        super().__init__(store, args, piped)
        self.order, self.hang, self.exit_code = order, hang, exit_code
        self.terminated = asyncio.Event()

    def job(self) -> dict:
        if self.stdin:
            return json.loads(self.stdin.data)
        if self.args[3] == "--photo-batch":
            return {"kind": "photo_batch", **json.loads(self.args[4])}
        return {"kind": "process"}

    async def wait(self) -> int:
        if self.returncode is None and self.stdin:
            await self.stdin.closed.wait()
        if self.returncode is not None:
            return self.returncode
        job = self.job()
        if job["kind"] != "photo_batch":
            self.order.append("process")
            return await super().wait()
        self.order.append(("photo_batch", job["trip_id"], job["batch_no"]))
        if self.hang:
            await self.terminated.wait()
        else:
            self.returncode = self.exit_code
        return self.returncode

    def terminate(self) -> None:
        super().terminate()
        self.terminated.set()


def batch_children(monkeypatch, store, *, hang=False, exit_code=0):
    spawned, order = [], []

    async def spawn(*args, **kwargs):
        spawned.append(BatchChild(store, args, "stdin" in kwargs, order, hang, exit_code))
        return spawned[-1]

    monkeypatch.setattr("app.jobs.runner.asyncio.create_subprocess_exec", spawn)
    return spawned, order


async def accept(cfg, store, producer, body):
    return await RequestReceiver(store, producer).accept(message(cfg, body))


async def run_until(store, condition, timeout=3):
    stop = asyncio.Event()
    task = asyncio.create_task(runner_loop(store, "process", stop))

    async def reached() -> None:
        while not await condition():
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(reached(), timeout=timeout)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=3)


@pytest.mark.parametrize("size", [0, 2])
@scenario
async def test_process_runs_before_queued_batches(monkeypatch, size):
    async with environment(JOBS_PROCESS_STANDBY=size, EMBED_ON_PHOTOS_READY=True) as (
        cfg,
        a,
        b,
        p,
    ):
        spawned, order = batch_children(monkeypatch, a)
        for no in (1, 2):
            await accept(cfg, a, p, ready_request(batch_no=no, ids=(no,), trip=1))
        await accept(cfg, a, p, process_request(trip=2))

        async def drained() -> bool:
            return len(order) == 3

        await run_until(a, drained)
        assert order == ["process", ("photo_batch", 1, 1), ("photo_batch", 1, 2)]
        assert (await b.latest(2, "process")).status == "COMPLETED"
        assert await b.redis.zcard(b.photo_batch_queue_key()) == 0
        if size:
            # 묶음도 대기 자식이 stdin으로 받는다.
            assert all(child.args[3] == "--standby" for child in spawned)
        else:
            assert [child.args[3] for child in spawned][1:] == ["--photo-batch"] * 2


@scenario
async def test_batches_stay_queued_when_setting_disabled(monkeypatch):
    async with environment(EMBED_ON_PHOTOS_READY=False) as (cfg, a, b, p):
        _, order = batch_children(monkeypatch, a)
        await accept(cfg, a, p, ready_request(trip=1))
        await accept(cfg, a, p, process_request(trip=2))

        async def completed() -> bool:
            return (await b.latest(2, "process")).status == "COMPLETED"

        await run_until(a, completed)
        assert order == ["process"]
        assert await b.redis.zcard(b.photo_batch_queue_key()) == 1


@pytest.mark.parametrize("failure", ["timeout", "exit"])
@scenario
async def test_batch_child_timeout_or_exit_is_logged_and_stopped(monkeypatch, caplog, failure):
    async with environment(JOBS_PROCESS_STANDBY=0, JOBS_PHOTO_BATCH_TIMEOUT_SEC=0.05) as (
        cfg,
        a,
        b,
        p,
    ):
        spawned, _ = batch_children(monkeypatch, a, hang=failure == "timeout", exit_code=1)
        await accept(cfg, a, p, ready_request())
        claim = await a.claim_batch()
        with caplog.at_level(logging.WARNING, logger="app.jobs.runner"):
            await asyncio.wait_for(run_photo_batch(a, claim, None), timeout=2)
        payload = json.loads(spawned[0].args[4])
        assert (payload["identity"], payload["trip_id"], payload["batch_no"]) == (
            claim.identity,
            77,
            1,
        )
        assert payload["wait_ms"] >= 0
        event = "ai_photo_batch_timeout" if failure == "timeout" else "ai_photo_batch_exit"
        (record,) = [r for r in caplog.records if getattr(r, "event", None) == event]
        assert (record.trip_id, record.batch_no) == (77, 1)
        assert spawned[0].returncode == (-15 if failure == "timeout" else 1)


@pytest.mark.parametrize("case", ["completed", "skipped", "failed"])
@scenario
async def test_batch_child_embeds_buffered_photos_and_logs(caplog, case):
    async with environment() as (cfg, a, b, p):
        await accept(cfg, a, p, ready_request(ids=(102, 101)))
        claim = await a.claim_batch()
        if case == "skipped":
            await accept(cfg, a, p, process_request())
        payload = {"identity": claim.identity, "trip_id": 77, "batch_no": 1, "wait_ms": 1.0}
        seen = []

        def handler(photos):
            seen.append(photos)
            if case == "failed":
                raise DownloadFailed(["trips/77/101.jpg"])
            return list(photos)

        with caplog.at_level(logging.INFO, logger="app.jobs.execution"):
            await run_batch(b, payload, handler, handoff_wait_ms=0.5)
        (record,) = [r for r in caplog.records if getattr(r, "event", None) == "ai_photo_batch"]
        assert (record.result, record.batch_no, record.wait_ms) == (case, 1, 1.0)
        if case == "skipped":
            assert seen == []
            return
        assert seen == [{101: "trips/77/101.jpg", 102: "trips/77/102.jpg"}]
        assert record.attachment_count == 2 and record.duration_ms >= 0
        if case == "completed":
            assert record.embedded_count == 2
        else:
            assert record.error_type == "DownloadFailed"


@pytest.mark.skipif(
    not os.environ.get("JOBS_TEST_REDIS_URL"), reason="실제 Redis의 subprocess 검증"
)
@pytest.mark.parametrize("size", [0, 1])
@scenario
async def test_real_child_runs_photo_batch(capfd, size):
    async with environment(JOBS_PROCESS_STANDBY=size, EMBED_ON_PHOTOS_READY=True) as (
        cfg,
        a,
        b,
        p,
    ):
        await accept(cfg, a, p, ready_request(ids=(101, 102)))
        claim = await a.claim_batch()
        standby = StandbyChildren(a, size) if size else None
        try:
            if standby:
                await standby.fill()
            await asyncio.wait_for(run_photo_batch(a, claim, standby), timeout=15)
        finally:
            if standby:
                await standby.close()
        records = []
        for line in capfd.readouterr().out.splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        (record,) = [r for r in records if r.get("event") == "ai_photo_batch"]
        assert (record["result"], record["embedded_count"], record["trip_id"]) == (
            "completed",
            2,
            77,
        )
        assert (record["handoff_wait_ms"] is None) == (size == 0)
