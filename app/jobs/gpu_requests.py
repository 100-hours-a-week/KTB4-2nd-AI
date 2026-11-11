"""복제본들이 공유하는 외부 GPU HTTP 호출 한도. 작업 대기열과는 별개다.

생성 호출(결정 222): 전체 GPU_GENERATION_INFLIGHT 슬롯을 검색, 정리, 스토리(분위기 변경 포함)가
나눠 쓴다. 정리와 스토리는 각자의 상한(GPU_PROCESS_GENERATION_INFLIGHT,
GPU_STORY_GENERATION_INFLIGHT) 안에서만 슬롯을 잡고, 검색은 상한 없이 쓰되 먼저 잡는다.
슬롯을 바로 얻지 못한 검색은 Redis의 대기 표지(ZSET, 점수는 표지의 만료 시각)에 토큰을 올리고,
정리와 스토리는 살아 있는 표지가 하나라도 있으면 슬롯을 잡지 않는다. 판정은 WATCH한 키들 위의
트랜잭션이라 여러 복제본에서도 같다. 표지는 검색이 슬롯을 얻거나 포기하면 지우고, 지우지 못해도
만료 시각이 지나면 무시된다.
"""

import asyncio
import time
import uuid

from redis.exceptions import RedisError, WatchError

from app.core.logging import get_logger
from app.core.tracing import current_trace
from app.infra.vlm import VLMUnavailable
from app.observability.langfuse_tracing import slot_wait
from app.search.context import ModelUnavailable

log = get_logger(__name__)
# 호출 전체 제한에 더하는 여유. 클라이언트 자체의 시간 제한이 먼저 걸려야 원인이 호출 로그에 남는다.
CALL_TIMEOUT_GRACE_SEC = 5


class GateUnavailable(Exception):
    pass


# 검색 대기 표지의 만료 여유. 검색은 wait_sec 안에 슬롯을 얻거나 포기하므로 그보다 조금 길게 둔다.
PRIORITY_MARK_GRACE_SEC = 5


class RequestGate:
    """키 하나(전체 한도)를 여러 프로세스가 lease 토큰으로 나눠 쓴다.

    group과 group_limit을 주면 같은 기능끼리의 상한을 따로 둔다(키 "<key>:<group>").
    priority_key를 주면 우선순위를 따른다. priority=True(검색)는 바로 슬롯을 얻지 못할 때 이 키에
    대기 표지를 올리고, priority=False(정리, 스토리)는 살아 있는 표지가 있으면 슬롯을 잡지 않는다.
    셋 다 주지 않으면 예전과 같이 전체 한도만 본다.
    """

    def __init__(
        self,
        redis,
        key: str,
        limit: int,
        wait_sec: float,
        *,
        group: str | None = None,
        group_limit: int | None = None,
        priority_key: str | None = None,
        priority: bool = False,
    ):
        self.redis, self.key, self.limit, self.wait_sec = redis, key, limit, wait_sec
        if limit < 1 or wait_sec <= 0:
            raise ValueError("GPU 호출 한도와 대기 시간은 양수여야 합니다.")
        if (group is None) != (group_limit is None) or (
            group_limit is not None and group_limit < 1
        ):
            raise ValueError("기능별 GPU 호출 한도는 이름과 양수 한도를 함께 줘야 합니다.")
        if priority and priority_key is None:
            raise ValueError("우선 호출에는 대기 표지 키가 필요합니다.")
        self.group_key = f"{key}:{group}" if group else None
        self.group_limit = group_limit
        self.priority_key, self.priority = priority_key, priority

    async def _claim(self, token, lease_sec):
        yields = self.priority_key is not None and not self.priority
        watched = [self.key]
        if self.group_key:
            watched.append(self.group_key)
        if yields:
            watched.append(self.priority_key)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(*watched)
                seconds, micros = await self.redis.time()
                now = seconds + micros / 1_000_000
                if await pipe.zcount(self.key, f"({now}", "+inf") >= self.limit:
                    return False
                if (
                    self.group_key
                    and await pipe.zcount(self.group_key, f"({now}", "+inf") >= self.group_limit
                ):
                    return False
                # 기다리는 검색이 있으면 정리와 스토리는 빈 슬롯을 양보한다.
                if yields and await pipe.zcount(self.priority_key, f"({now}", "+inf") > 0:
                    return False
                pipe.multi()
                pipe.zremrangebyscore(self.key, "-inf", now)
                pipe.zadd(self.key, {token: now + lease_sec})
                if self.group_key:
                    pipe.zremrangebyscore(self.group_key, "-inf", now)
                    pipe.zadd(self.group_key, {token: now + lease_sec})
                await pipe.execute()
                return True
            except WatchError:
                return False

    async def _mark_waiting(self, token):
        """검색이 기다리는 동안 대기 표지를 올린다. 만료 시각은 대기 상한에 여유를 더한 때다."""
        seconds, micros = await self.redis.time()
        now = seconds + micros / 1_000_000
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(self.priority_key, "-inf", now)
            pipe.zadd(self.priority_key, {token: now + self.wait_sec + PRIORITY_MARK_GRACE_SEC})
            await pipe.execute()

    async def run(self, operation, timeout_sec):
        token = uuid.uuid4().hex
        started = time.monotonic()
        marked = False
        try:
            try:
                async with asyncio.timeout(self.wait_sec):
                    if not await self._claim(token, timeout_sec + 10):
                        if self.priority:
                            marked = True
                            await self._mark_waiting(token)
                        while not await self._claim(token, timeout_sec + 10):
                            await asyncio.sleep(0.05)
                    if marked:
                        # 슬롯을 얻었으면 바로 표지를 지워 정리와 스토리가 다시 잡을 수 있게 한다.
                        marked = False
                        await self.redis.zrem(self.priority_key, token)
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
            waited = time.monotonic() - started
            if trace := current_trace():
                trace.add_gpu_wait(waited)
            # client의 I/O timeout 외에 호출 전체 시간을 제한한다.
            with slot_wait(waited):
                async with asyncio.timeout(timeout_sec + CALL_TIMEOUT_GRACE_SEC):
                    return await operation()
        finally:
            # claim 중 응답을 잃어도 같은 토큰을 정리한다. 실패 시 lease가 회수한다.
            try:
                async with asyncio.timeout(5):
                    async with self.redis.pipeline(transaction=False) as pipe:
                        pipe.zrem(self.key, token)
                        if self.group_key:
                            pipe.zrem(self.group_key, token)
                        if marked:
                            pipe.zrem(self.priority_key, token)
                        await pipe.execute()
            except (TimeoutError, RedisError):
                log.warning("GPU 호출 슬롯 해제 지연", extra={"event": "gpu_slot_release_failed"})


class GatedVlm:
    def __init__(self, client, gate: RequestGate, timeout_sec: float):
        self.client, self.gate, self.timeout_sec = client, gate, timeout_sec

    async def chat(
        self,
        messages,
        image_urls,
        max_tokens,
        response_format=None,
        *,
        tools=None,
        tool_choice=None,
        temperature=None,
        stream=False,
        extra_body=None,
    ):
        # 값이 있을 때만 넘겨, 도구나 temperature, 스트리밍을 받지 않는 구현도
        # 그대로 감쌀 수 있게 한다.
        options = _given(
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            stream=stream or None,
            extra_body=extra_body or None,
        )
        try:
            return await self.gate.run(
                lambda: self.client.chat(
                    messages, image_urls, max_tokens, response_format, **options
                ),
                self.timeout_sec,
            )
        except GateUnavailable:
            raise VLMUnavailable("생성 서버의 호출 대기 한도를 초과했습니다.") from None


def _given(**options):
    return {name: value for name, value in options.items() if value is not None}


class GatedEmbedder:
    def __init__(self, client, gate: RequestGate, timeout_sec: float):
        self.client, self.gate, self.timeout_sec = client, gate, timeout_sec

    async def encode_text(self, text):
        try:
            return await self.gate.run(lambda: self.client.encode_text(text), self.timeout_sec)
        except GateUnavailable:
            raise ModelUnavailable("임베딩 서버의 호출 대기 한도를 초과했습니다.") from None


GENERATION_KINDS = ("search", "story", "process")


def generation_gate(redis, cfg, kind: str, wait_sec: float | None = None) -> RequestGate:
    """생성 호출 슬롯. kind는 search(우선, 기능별 상한 없음), story(분위기 변경 포함), process"""
    if kind not in GENERATION_KINDS:
        raise ValueError(f"생성 호출 종류는 {GENERATION_KINDS} 가운데 하나여야 합니다.")
    key = cfg.redis_prefix + "gpu:generation"
    limits = {
        "story": cfg.GPU_STORY_GENERATION_INFLIGHT,
        "process": cfg.GPU_PROCESS_GENERATION_INFLIGHT,
    }
    return RequestGate(
        redis,
        key,
        cfg.GPU_GENERATION_INFLIGHT,
        cfg.GPU_WAIT_SEC if wait_sec is None else wait_sec,
        group=None if kind == "search" else kind,
        group_limit=limits.get(kind),
        priority_key=key + ":search-waiting",
        priority=kind == "search",
    )


def generation_client(client, redis, cfg, *, kind: str, timeout_sec=None):
    # 분위기 변경처럼 호출 상한이 다른 기능은 같은 슬롯을 쓰되 상한만 따로 준다.
    return GatedVlm(
        client,
        generation_gate(redis, cfg, kind),
        cfg.VLM_TIMEOUT_SEC if timeout_sec is None else timeout_sec,
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
