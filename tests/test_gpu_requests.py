import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.infra.vlm import VLMUnavailable
from app.jobs.gpu_requests import GateUnavailable, RequestGate, embedding_client, generation_client
from app.search.context import ModelUnavailable
from tests.test_v2_jobs import environment, scenario


@scenario
async def test_limit_is_shared_across_api_and_story_clients():
    async with environment() as (cfg, a, b, _):
        cfg.GPU_GENERATION_INFLIGHT = 2
        active = maximum = 0

        async def operation(*args):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return "ok"

        model = AsyncMock()
        model.chat.side_effect = operation
        clients = [generation_client(model, s.redis, cfg) for s in (a, b, a)]
        await asyncio.gather(*[clients[i % 3].chat([], [], 20) for i in range(9)])
        assert maximum == 2
        assert await a.redis.zcard(a.key("gpu:generation")) == 0


@scenario
async def test_busy_and_redis_failure_never_start_http_call():
    async with environment() as (cfg, a, _, _):
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
