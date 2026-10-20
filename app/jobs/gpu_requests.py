"""복제본들이 공유하는 외부 GPU HTTP 호출 한도. 작업 대기열과는 별개다."""

import asyncio
import time
import uuid

from redis.exceptions import RedisError, WatchError

from app.core.logging import get_logger
from app.core.tracing import current_trace
from app.infra.vlm import VLMUnavailable
from app.search.context import ModelUnavailable

log = get_logger(__name__)
# 호출 전체 제한에 더하는 여유. 클라이언트 자체의 시간 제한이 먼저 걸려야 원인이 호출 로그에 남는다.
CALL_TIMEOUT_GRACE_SEC = 5


class GateUnavailable(Exception):
    pass


class RequestGate:
    def __init__(self, redis, key: str, limit: int, wait_sec: float):
        self.redis, self.key, self.limit, self.wait_sec = redis, key, limit, wait_sec
        if limit < 1 or wait_sec <= 0:
            raise ValueError("GPU 호출 한도와 대기 시간은 양수여야 합니다.")

    async def _claim(self, token, lease_sec):
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(self.key)
                seconds, micros = await self.redis.time()
                now = seconds + micros / 1_000_000
                if await pipe.zcount(self.key, f"({now}", "+inf") >= self.limit:
                    return False
                pipe.multi()
                pipe.zremrangebyscore(self.key, "-inf", now)
                pipe.zadd(self.key, {token: now + lease_sec})
                await pipe.execute()
                return True
            except WatchError:
                return False

    async def run(self, operation, timeout_sec):
        token = uuid.uuid4().hex
        started = time.monotonic()
        try:
            try:
                async with asyncio.timeout(self.wait_sec):
                    while not await self._claim(token, timeout_sec + 10):
                        await asyncio.sleep(0.05)
            except (TimeoutError, RedisError) as exc:
                waited = time.monotonic() - started
                if trace := current_trace():
                    trace.add_gpu_wait(waited)
                # 슬롯 부족과 Redis 장애를 구분한다. 둘 다 호출 전에 실패한 경우다.
                log.warning(
                    "GPU 호출 슬롯 확보 실패",
                    extra={
                        "event": "ai_gpu_gate_unavailable",
                        "result": "timeout" if isinstance(exc, TimeoutError) else "redis_error",
                        "gate": self.key.rsplit(":", 1)[-1],
                        "limit": self.limit,
                        "wait_ms": round(waited * 1000, 2),
                    },
                )
                raise GateUnavailable("GPU 호출 슬롯을 확보하지 못했습니다.") from None
            if trace := current_trace():
                trace.add_gpu_wait(time.monotonic() - started)
            # client의 I/O timeout 외에 호출 전체 시간을 제한한다.
            async with asyncio.timeout(timeout_sec + CALL_TIMEOUT_GRACE_SEC):
                return await operation()
        finally:
            # claim 중 응답을 잃어도 같은 토큰을 정리한다. 실패 시 lease가 회수한다.
            try:
                async with asyncio.timeout(5):
                    await self.redis.zrem(self.key, token)
            except (TimeoutError, RedisError):
                log.warning("GPU 호출 슬롯 해제 지연", extra={"event": "gpu_slot_release_failed"})


class GatedVlm:
    def __init__(self, client, gate: RequestGate, timeout_sec: float):
        self.client, self.gate, self.timeout_sec = client, gate, timeout_sec

    async def chat(
        self, messages, image_urls, max_tokens, response_format=None, *, temperature=None
    ):
        # 값이 있을 때만 넘겨, temperature를 받지 않는 구현도 그대로 감쌀 수 있게 한다.
        options = {} if temperature is None else {"temperature": temperature}
        try:
            return await self.gate.run(
                lambda: self.client.chat(
                    messages, image_urls, max_tokens, response_format, **options
                ),
                self.timeout_sec,
            )
        except GateUnavailable:
            raise VLMUnavailable("생성 서버의 호출 대기 한도를 초과했습니다.") from None


class GatedEmbedder:
    def __init__(self, client, gate: RequestGate, timeout_sec: float):
        self.client, self.gate, self.timeout_sec = client, gate, timeout_sec

    async def encode_text(self, text):
        try:
            return await self.gate.run(lambda: self.client.encode_text(text), self.timeout_sec)
        except GateUnavailable:
            raise ModelUnavailable("임베딩 서버의 호출 대기 한도를 초과했습니다.") from None


def generation_client(client, redis, cfg):
    return GatedVlm(
        client,
        RequestGate(
            redis,
            cfg.redis_prefix + "gpu:generation",
            cfg.GPU_GENERATION_INFLIGHT,
            cfg.GPU_WAIT_SEC,
        ),
        cfg.VLM_TIMEOUT_SEC,
    )


def embedding_client(client, redis, cfg):
    return GatedEmbedder(
        client,
        RequestGate(
            redis,
            cfg.redis_prefix + "gpu:search-embedding",
            cfg.GPU_SEARCH_EMBEDDING_INFLIGHT,
            cfg.GPU_WAIT_SEC,
        ),
        cfg.EMBEDDING_TIMEOUT_SEC,
    )
