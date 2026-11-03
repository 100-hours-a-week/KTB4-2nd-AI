import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.infra.vlm import VLMUnavailable
from app.jobs.gpu_requests import GateUnavailable, RequestGate, embedding_client, generation_client
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
        clients = [generation_client(model, s.redis, cfg) for s in (a, b, a)]
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
        client = generation_client(model, a.redis, cfg)
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
        client = generation_client(model, a.redis, cfg)
        assert await client.chat([], [], 20, tools=["tool"], temperature=0) == "turn"
        await client.chat([], [], 20, tools=["tool"], tool_choice="required")
        first, second = model.chat.await_args_list
        assert first.args == ([], [], 20, None)
        assert first.kwargs == {"tools": ["tool"], "temperature": 0}
        assert second.kwargs == {"tools": ["tool"], "tool_choice": "required"}
        assert await a.redis.zcard(a.key("gpu:generation")) == 0


@scenario
async def test_busy_and_redis_failure_never_start_http_call():
    async with environment() as (cfg, a, _, _):
        cfg.GPU_GENERATION_INFLIGHT = 2
        cfg.GPU_WAIT_SEC = 0.01
        client = AsyncMock()
        gate = generation_client(client, a.redis, cfg)
        await a.redis.zadd(gate.gate.key, {"a": await a.now() + 60, "b": await a.now() + 60})
        with pytest.raises(VLMUnavailable):
            await gate.chat([], [], 20)
        client.chat.assert_not_called()
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
