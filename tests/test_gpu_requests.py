import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.infra.vlm import VLMTransientError, VLMUnavailable
from app.jobs.gpu_requests import (
    GateUnavailable,
    RequestGate,
    embedding_client,
    generation_client,
    generation_gate,
)
from app.search.context import ModelUnavailable
from tests.test_v2_jobs import environment, scenario


@pytest.mark.parametrize("limit", [2, 8])
@scenario
async def test_limit_is_shared_across_api_and_story_clients(limit):
    async with environment() as (cfg, a, b, _):
        cfg.GPU_GENERATION_INFLIGHT = limit
        active = maximum = 0
        full, release = asyncio.Event(), asyncio.Event()

        async def operation(*args):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            if active == limit:
                full.set()
            await release.wait()
            active -= 1
            return "ok"

        model = AsyncMock()
        model.chat.side_effect = operation
        clients = [generation_client(model, s.redis, cfg, kind="search") for s in (a, b, a)]
        tasks = [asyncio.create_task(clients[i % 3].chat([], [], 20)) for i in range(9)]
        try:
            await asyncio.wait_for(full.wait(), timeout=1.5)
            release.set()
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert maximum == limit
        assert await a.redis.zcard(a.key("gpu:generation")) == 0


@scenario
async def test_generation_client_forwards_temperature_only_when_given():
    async with environment() as (cfg, a, _, _):
        model = AsyncMock()
        model.chat.return_value = "ok"
        client = generation_client(model, a.redis, cfg, kind="search")
        await client.chat([], [], 20, temperature=0)
        await client.chat([], [], 20)
        assert model.chat.await_args_list[0].kwargs == {"temperature": 0}
        # 미지정이면 넘기지 않아, temperature를 받지 않는 구현도 감쌀 수 있다.
        assert model.chat.await_args_list[1].kwargs == {}


@scenario
async def test_generation_client_forwards_tool_calls_through_the_same_gate():
    async with environment() as (cfg, a, _, _):
        model = AsyncMock()
        model.chat.return_value = "turn"
        client = generation_client(model, a.redis, cfg, kind="search")
        assert await client.chat([], [], 20, tools=["tool"], temperature=0) == "turn"
        await client.chat([], [], 20, tools=["tool"], tool_choice="required")
        first, second = model.chat.await_args_list
        assert first.args == ([], [], 20, None)
        assert first.kwargs == {"tools": ["tool"], "temperature": 0}
        assert second.kwargs == {"tools": ["tool"], "tool_choice": "required"}
        assert await a.redis.zcard(a.key("gpu:generation")) == 0


@scenario
async def test_generation_client_forwards_stream_and_uses_given_call_limit():
    async with environment() as (cfg, a, _, _):
        model = AsyncMock()
        model.chat.return_value = "turn"
        client = generation_client(model, a.redis, cfg, kind="story", timeout_sec=300)
        await client.chat([], [], 20, stream=True)
        await client.chat([], [], 20)
        first, second = model.chat.await_args_list
        assert first.kwargs == {"stream": True}
        assert second.kwargs == {}
        assert client.timeout_sec == 300
        assert (
            generation_client(model, a.redis, cfg, kind="search").timeout_sec == cfg.VLM_TIMEOUT_SEC
        )


@scenario
async def test_busy_and_redis_failure_never_start_http_call():
    async with environment() as (cfg, a, _, _):
        cfg.GPU_GENERATION_INFLIGHT = 2
        cfg.GPU_WAIT_SEC = 0.01
        client = AsyncMock()
        gate = generation_client(client, a.redis, cfg, kind="search")
        await a.redis.zadd(gate.gate.key, {"a": await a.now() + 60, "b": await a.now() + 60})
        with pytest.raises(VLMUnavailable) as error:
            await gate.chat([], [], 20)
        client.chat.assert_not_called()
        # 정리 에이전트는 슬롯 대기 초과를 다시 보내지 않는다(결정 159).
        assert not isinstance(error.value, VLMTransientError)
        broken = AsyncMock()
        broken.pipeline = Mock(side_effect=RedisConnectionError())
        gate = embedding_client(client, broken, cfg)
        with pytest.raises(ModelUnavailable):
            await gate.encode_text("query")
        client.encode_text.assert_not_called()


@pytest.mark.parametrize("cause", ["cancel", "timeout", "error"])
@scenario
async def test_inflight_slot_released_on_every_exit(cause):
    async with environment() as (cfg, a, _, _):
        started = asyncio.Event()

        async def operation():
            started.set()
            if cause == "error":
                raise ValueError("upstream error")
            await asyncio.Event().wait()

        gate = RequestGate(a.redis, a.key("test-gate"), 1, 0.1)
        task = asyncio.create_task(gate.run(operation, 0.03))
        await started.wait()
        if cause == "cancel":
            task.cancel()
        with pytest.raises(
            {"cancel": asyncio.CancelledError, "timeout": TimeoutError, "error": ValueError}[cause]
        ):
            await task
        assert await a.redis.zcard(gate.key) == 0


@scenario
async def test_expired_token_is_reclaimed_and_fresh_token_is_preserved():
    async with environment() as (cfg, a, _, _):
        gate = RequestGate(a.redis, a.key("test-gate"), 2, 0.1)
        now = await a.now()
        await a.redis.zadd(gate.key, {"dead": now - 1, "alive": now + 100})
        assert await gate.run(AsyncMock(return_value="done"), 0.02) == "done"
        assert await a.redis.zrange(gate.key, 0, -1) == ["alive"]


@scenario
async def test_claim_response_loss_releases_own_token(monkeypatch):
    async with environment() as (cfg, a, _, _):
        gate = RequestGate(a.redis, a.key("test-gate"), 1, 0.1)
        claim = gate._claim

        async def lost(token, lease):
            await claim(token, lease)
            raise RedisConnectionError()

        monkeypatch.setattr(gate, "_claim", lost)
        operation = AsyncMock()
        with pytest.raises(GateUnavailable):
            await gate.run(operation, 0.1)
        operation.assert_not_called()
        assert await a.redis.zcard(gate.key) == 0


# ---------- 결정 222: 정리·스토리 상한과 검색 우선
async def _hold(gate, started, release, log=None, name=None):
    async def operation():
        if log is not None:
            log.append(name)
        started.release()
        await release.wait()
        return name

    return await gate.run(operation, 5)


@scenario
async def test_process_and_story_each_use_at_most_their_share_across_replicas():
    async with environment() as (cfg, a, b, _):
        started, release = asyncio.Semaphore(0), asyncio.Event()
        tasks = [
            asyncio.create_task(_hold(generation_gate(s.redis, cfg, kind, 0.3), started, release))
            for kind in ("process", "story")
            for s in (a, b, a)
            for _ in range(2)
        ]
        try:
            for _ in range(8):
                await asyncio.wait_for(started.acquire(), 1)
            key = cfg.redis_prefix + "gpu:generation"
            assert await a.redis.zcard(key) == 8
            assert await a.redis.zcard(key + ":process") == 4
            assert await a.redis.zcard(key + ":story") == 4
            # 상한에 걸린 나머지 4건은 대기 상한(0.3초) 안에 슬롯을 잡지 못하고 실패한다
            await asyncio.sleep(0.5)
            failed = [t for t in tasks if t.done() and isinstance(t.exception(), GateUnavailable)]
            assert len(failed) == 4
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert await a.redis.zcard(key) == await a.redis.zcard(key + ":process") == 0


@scenario
async def test_waiting_search_gets_freed_slot_before_process_and_story():
    async with environment() as (cfg, a, b, _):
        cfg.GPU_GENERATION_INFLIGHT = 1
        cfg.GPU_PROCESS_GENERATION_INFLIGHT = cfg.GPU_STORY_GENERATION_INFLIGHT = 1
        key = cfg.redis_prefix + "gpu:generation"
        order, started, first, rest = [], asyncio.Semaphore(0), asyncio.Event(), asyncio.Event()
        holder = asyncio.create_task(
            _hold(generation_gate(a.redis, cfg, "process", 2), started, first, order, "holder")
        )
        await asyncio.wait_for(started.acquire(), 1)
        waiting = [
            asyncio.create_task(
                _hold(generation_gate(b.redis, cfg, kind, 2), started, rest, order, kind)
            )
            for kind in ("story", "process")
        ]
        await asyncio.sleep(0.1)
        search = asyncio.create_task(
            _hold(generation_gate(b.redis, cfg, "search", 2), started, rest, order, "search")
        )
        for _ in range(40):
            if await a.redis.zcard(key + ":search-waiting"):
                break
            await asyncio.sleep(0.01)
        assert await a.redis.zcard(key + ":search-waiting") == 1
        first.set()
        await asyncio.wait_for(started.acquire(), 1)
        assert order == ["holder", "search"]
        # 슬롯을 얻은 검색은 표지를 지운다
        assert await a.redis.zcard(key + ":search-waiting") == 0
        rest.set()
        await asyncio.gather(holder, search, *waiting)
        assert order[2:] and set(order[2:]) == {"story", "process"}


@scenario
async def test_search_that_gives_up_clears_its_mark_and_expired_marks_are_ignored():
    async with environment() as (cfg, a, _, _):
        cfg.GPU_GENERATION_INFLIGHT = 1
        key = cfg.redis_prefix + "gpu:generation"
        await a.redis.zadd(key, {"busy": await a.now() + 60})
        with pytest.raises(GateUnavailable):
            await generation_gate(a.redis, cfg, "search", 0.1).run(AsyncMock(), 1)
        assert await a.redis.zcard(key + ":search-waiting") == 0
        await a.redis.zrem(key, "busy")
        # 프로세스가 죽어 지우지 못한 표지는 만료 시각이 지나면 무시한다
        await a.redis.zadd(key + ":search-waiting", {"dead": await a.now() - 1})
        done = AsyncMock(return_value="ok")
        assert await generation_gate(a.redis, cfg, "process", 0.2).run(done, 1) == "ok"
        # 살아 있는 표지가 있으면 정리는 빈 슬롯이 있어도 잡지 않는다
        await a.redis.zadd(key + ":search-waiting", {"alive": await a.now() + 60})
        with pytest.raises(GateUnavailable):
            await generation_gate(a.redis, cfg, "story", 0.1).run(done, 1)
        assert done.await_count == 1
        # 검색은 표지와 관계없이 바로 잡는다
        assert await generation_gate(a.redis, cfg, "search", 0.2).run(done, 1) == "ok"


def test_generation_share_defaults_and_kind_is_checked():
    from app.jobs.config import JobsSettings

    # 전체 한도를 낮춘 배포(예: 1)도 그대로 시작한다. 이때는 전체 한도가 먼저 걸린다.
    assert (
        JobsSettings(
            _env_file=None,
            API_KEY="t",
            IMAGE_SOURCE="local",
            GPU_GENERATION_INFLIGHT=1,
            PROCESS_AGENT_CONCURRENCY=1,
            PROCESS_WORKFLOW_CONCURRENCY=1,
        ).GPU_PROCESS_GENERATION_INFLIGHT
        == 4
    )
    cfg = JobsSettings(_env_file=None, API_KEY="t", IMAGE_SOURCE="local")
    assert (cfg.GPU_PROCESS_GENERATION_INFLIGHT, cfg.GPU_STORY_GENERATION_INFLIGHT) == (4, 4)
    with pytest.raises(ValueError, match="생성 호출 종류"):
        generation_gate(None, cfg, "embedding")
