"""Redis 단일 인스턴스의 WATCH/MULTI로 등록, 실행 선점, 결과 대기를 원자적으로 변경한다.

진행 중 기록과 미발행 결과에는 TTL을 두지 않는다. Redis 전체 유실은 v2의 알려진 한계다.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.jobs.config import MAX_STORY_CONCURRENCY, JobsSettings
from app.jobs.models import (
    PHOTOS_READY_MAX_BYTES,
    JobKind,
    JobRecord,
    JobResultEvent,
    PhotoBatchBuffer,
    PhotosReadyMessage,
    ReadyAttachment,
    TerminalStatus,
)
from app.observability.metrics import log_terminal, observe_commands, terminal_commands
from app.schemas.task import TERMINAL_STATUSES, ErrorBody, Progress, TaskStatus

log = get_logger(__name__)


@dataclass(frozen=True)
class PhotoBatchClaim:
    """실행기가 꺼낸 사진 묶음. enqueued_at은 적재 시각(Redis 시계), 묶음 대기 시간 관측용"""

    identity: str
    batch_no: int
    enqueued_at: float


def photo_batch_member(identity: str, batch_no: int | str) -> str:
    return f"{identity}#{batch_no}"


class RedisJobStore:
    def __init__(self, redis: Redis, settings: JobsSettings) -> None:
        self.redis = redis
        self.cfg = settings
        self.prefix = settings.redis_prefix

    def key(self, name: str) -> str:
        return self.prefix + name

    def record_key(self, identity: str) -> str:
        return self.key("job:" + identity)

    def photo_batches_key(self, identity: str) -> str:
        return self.key("photo_batches:" + identity)

    def photo_batch_queue_key(self) -> str:
        return self.key("queue:photo_batch")

    def slot_keys(self, kind: JobKind) -> list[str]:
        # 첫 키는 기존 단일 실행기와 공유한다. 나머지는 스토리 전용이다.
        base = self.key(f"slot:{kind}")
        return [base] + (
            [f"{base}:{index}" for index in range(1, MAX_STORY_CONCURRENCY)]
            if kind == "story"
            else []
        )

    async def stage_photos(
        self, message: PhotosReadyMessage, request_topic: str
    ) -> Literal["staged", "duplicate", "closed"]:
        """사본 경로를 공유 보관하고 묶음 대기열에 적재한다. 실행 대기열과 상태는 만들지 않는다.

        묶음 대기열은 EMBED_ON_PHOTOS_READY와 무관하게 적재한다. 설정은 정리 실행기만 읽는다.
        """
        key = self.photo_batches_key(message.identity)
        batch_queue = self.photo_batch_queue_key()
        job = self.record_key(message.identity)
        ignored = self.key("ignored:" + message.identity)
        batch_no = str(message.batch_no)
        photos = sorted(message.attachments, key=lambda photo: photo.trip_attachment_id)
        while True:
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    # process 등록과 경합해도 닫힌 실행의 임시 자료를 되살리지 않는다.
                    await pipe.watch(key, job, ignored)
                    if await pipe.exists(job, ignored):
                        return "closed"
                    raw = await pipe.get(key)
                    buffer = (
                        PhotoBatchBuffer.model_validate_json(raw)
                        if raw
                        else PhotoBatchBuffer(request_topic=request_topic)
                    )
                    if buffer.request_topic != request_topic:
                        raise ValueError("같은 실행의 묶음은 같은 요청 토픽을 사용해야 함")
                    previous = buffer.batches.get(batch_no)
                    if previous is not None:
                        if previous != photos:
                            raise ValueError("같은 묶음 번호의 사진 목록 또는 경로가 변경됨")
                        return "duplicate"
                    known = {
                        photo.trip_attachment_id: photo.analyze_storage_key
                        for batch in buffer.batches.values()
                        for photo in batch
                    }
                    for photo in photos:
                        old_key = known.get(photo.trip_attachment_id)
                        if old_key is not None and old_key != photo.analyze_storage_key:
                            raise ValueError("같은 실행의 사진 사본 경로가 변경됨")
                        known[photo.trip_attachment_id] = photo.analyze_storage_key
                    if len(known) > 200 or len(buffer.batches) >= 200:
                        raise ValueError("실행당 사진 또는 묶음 보관 상한 200개 초과")
                    buffer.batches[batch_no] = photos
                    payload = buffer.model_dump_json()
                    if len(payload.encode("utf-8")) > PHOTOS_READY_MAX_BYTES:
                        raise ValueError("실행당 사진 묶음 보관 크기 상한 2MiB 초과")
                    now = await self.now()
                    pipe.multi()
                    pipe.set(key, payload, ex=self.cfg.JOBS_PHOTOS_READY_TTL_SEC)
                    pipe.zadd(batch_queue, {photo_batch_member(message.identity, batch_no): now})
                    # process가 오지 않은 실행의 항목은 버퍼와 같은 기한으로 정리한다.
                    pipe.zremrangebyscore(
                        batch_queue, "-inf", now - self.cfg.JOBS_PHOTOS_READY_TTL_SEC
                    )
                    await pipe.execute()
                    return "staged"
                except WatchError:
                    continue

    async def now(self) -> float:
        seconds, micros = await self.redis.time()
        return seconds + micros / 1_000_000

    async def get(self, identity: str) -> JobRecord | None:
        raw = await self.redis.get(self.record_key(identity))
        return JobRecord.model_validate_json(raw) if raw else None

    async def latest(self, trip_id: int, kind: JobKind) -> JobRecord:
        identity = await self.redis.get(self.key(f"latest:{kind}:{trip_id}"))
        record = await self.get(identity) if identity else None
        if record is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND)
        return record

    async def register(self, record: JobRecord) -> Literal["accepted", "duplicate", "busy"]:
        key = self.record_key(record.identity)
        ignored = self.key("ignored:" + record.identity)
        latest = self.key(f"latest:{record.kind}:{record.trip_id}")
        sequence = self.key("enqueue_sequence")
        photo_batches = self.photo_batches_key(record.identity)
        batch_queue = self.photo_batch_queue_key()
        while True:
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key, ignored, latest, sequence, photo_batches)
                    if await pipe.exists(key, ignored):
                        return "duplicate"
                    # 버퍼를 지우면서 아직 꺼내지 않은 묶음 대기열 항목도 함께 지운다.
                    raw_batches = await pipe.get(photo_batches)
                    staged = (
                        list(PhotoBatchBuffer.model_validate_json(raw_batches).batches)
                        if raw_batches
                        else []
                    )
                    members = [photo_batch_member(record.identity, no) for no in staged]
                    previous_id = await pipe.get(latest)
                    previous = None
                    if previous_id:
                        previous_key = self.record_key(previous_id)
                        await pipe.watch(previous_key)
                        raw = await pipe.get(previous_key)
                        previous = JobRecord.model_validate_json(raw) if raw else None
                    if previous and previous.status not in TERMINAL_STATUSES:
                        # 무시한 실행도 기록한다. Kafka 재전달 때 뒤늦게 실행하지 않는다.
                        pipe.multi()
                        if members:
                            pipe.zrem(batch_queue, *members)
                        pipe.set(ignored, "1")
                        pipe.delete(photo_batches)
                        results = await pipe.execute()
                        self.log_unprocessed(record, "busy", len(staged), results)
                        return "busy"
                    record.created_at = await self.now()
                    # 같은 시각과 시계 보정에도 등록 순서를 유지한다.
                    order = int(await pipe.get(sequence) or 0) + 1
                    pipe.multi()
                    if members:
                        pipe.zrem(batch_queue, *members)
                    pipe.set(sequence, order)
                    pipe.set(key, record.model_dump_json())
                    pipe.set(latest, record.identity)
                    # 최종 process가 전체 목록의 기준이다. 묶음의 누락과 만료를 기다리지 않는다.
                    pipe.delete(photo_batches)
                    destination = "pending" if record.event else f"queue:{record.kind}"
                    score = record.created_at if record.event else order
                    pipe.zadd(self.key(destination), {record.identity: score})
                    if record.event:
                        terminal_commands(
                            pipe, self.key("metrics:operations"), record, record.created_at
                        )
                    results = await pipe.execute()
                    self.log_unprocessed(record, "accepted", len(staged), results)
                    if record.event:
                        log_terminal(record, record.created_at, intake=True)
                    return "accepted"
                except WatchError:
                    continue

    def log_unprocessed(
        self, record: JobRecord, outcome: str, staged_count: int, results: list
    ) -> None:
        """process 접수 시 실행기가 꺼내지 않은 묶음 수. zrem은 트랜잭션의 첫 명령"""
        if not staged_count:
            return
        log.info(
            "process 접수 시 미처리 사진 묶음",
            extra={
                "event": "ai_photo_batch_unprocessed",
                "result": outcome,
                "trip_id": record.trip_id,
                "job_id": record.execution_id,
                "execution_id": record.execution_id,
                "staged_count": staged_count,
                "unprocessed_count": int(results[0]),
            },
        )

    async def claim_batch(self) -> PhotoBatchClaim | None:
        """가장 먼저 적재한 사진 묶음을 꺼냄. 버퍼가 만료된 묶음은 버리고 다음 항목 확인

        process 등록이 대기열 항목을 함께 지우므로 꺼낸 묶음은 아직 process가 없는 실행
        """
        queue = self.photo_batch_queue_key()
        while True:
            popped = await self.redis.zpopmin(queue)
            if not popped:
                return None
            member, enqueued_at = popped[0]
            identity, batch_no = member.rsplit("#", 1)
            if await self.redis.exists(self.photo_batches_key(identity)):
                return PhotoBatchClaim(identity, int(batch_no), enqueued_at)

    async def photo_batch(self, identity: str, batch_no: int) -> list[ReadyAttachment] | None:
        """버퍼에 보관한 묶음의 사진 목록. process 등록이나 만료로 버퍼가 없으면 None"""
        raw = await self.redis.get(self.photo_batches_key(identity))
        if raw is None:
            return None
        return PhotoBatchBuffer.model_validate_json(raw).batches.get(str(batch_no))

    async def claim(
        self, kind: JobKind, worker: str, *, should_stop: Callable[[], bool] | None = None
    ) -> JobRecord | None:
        queue = self.key(f"queue:{kind}")
        slots = self.slot_keys(kind)
        limit = self.cfg.JOBS_STORY_CONCURRENCY if kind == "story" else 1
        while True:
            if should_stop and should_stop():
                return None
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(queue, *slots)
                    occupants = await pipe.mget(slots)
                    # 이전 이미지의 복구 루프는 추가 슬롯을 모른다. 종료된 소유자만 정리한다.
                    stale = []
                    for index, occupant in enumerate(occupants):
                        if occupant is None:
                            continue
                        occupant_key = self.record_key(occupant)
                        await pipe.watch(occupant_key)
                        raw = await pipe.get(occupant_key)
                        if raw and JobRecord.model_validate_json(raw).status in TERMINAL_STATUSES:
                            stale.append(slots[index])
                    if stale:
                        pipe.multi()
                        pipe.delete(*stale)
                        await pipe.execute()
                        continue
                    if sum(owner is not None for owner in occupants) >= limit:
                        return None
                    slot = slots[occupants.index(None)]
                    waiting = await pipe.zrange(queue, 0, 0, withscores=True)
                    if not waiting:
                        return None
                    identity, queue_score = waiting[0]
                    key = self.record_key(identity)
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    record = JobRecord.model_validate_json(raw) if raw else None
                    if record is None or record.status != TaskStatus.QUEUED:
                        pipe.multi()
                        pipe.zrem(queue, identity)
                        await pipe.execute()
                        continue
                    now = await self.now()
                    record.status = TaskStatus.PROCESSING
                    record.owner = worker + ":" + uuid.uuid4().hex
                    record.started_at = record.heartbeat_at = now
                    if should_stop and should_stop():
                        return None
                    pipe.multi()
                    pipe.zrem(queue, identity)
                    pipe.set(slot, identity)
                    pipe.set(key, record.model_dump_json())
                    pipe.zadd(
                        self.key("running"),
                        {identity: now + min(self.cfg.JOBS_LEASE_SEC, self.cfg.JOBS_TIMEOUT_SEC)},
                    )
                    await pipe.execute()
                    if should_stop and should_stop():
                        await self.return_unstarted(record, queue_score)
                        return None
                    return record
                except WatchError:
                    continue

    async def change(
        self,
        identity: str,
        update: Callable[[JobRecord, float], bool],
        *,
        queue_score: float | None = None,
    ) -> JobRecord | None:
        """조건 확인과 상태 저장을 같은 트랜잭션으로 처리. update는 재실행될 수 있다."""
        key = self.record_key(identity)
        while True:
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if raw is None:
                        return None
                    record = JobRecord.model_validate_json(raw)
                    was_terminal = record.status in TERMINAL_STATUSES
                    # 종료 처리가 current_step을 비우므로 실패 단계 로그용으로 먼저 보관한다.
                    previous_step = record.current_step
                    now = await self.now()
                    if not update(record, now):
                        return None
                    slots = self.slot_keys(record.kind)
                    await pipe.watch(*slots)
                    slot_owners = await pipe.mget(slots)
                    pipe.multi()
                    pipe.set(key, record.model_dump_json())
                    if record.status in TERMINAL_STATUSES:
                        if not was_terminal:
                            terminal_commands(pipe, self.key("metrics:operations"), record, now)
                        pipe.zrem(self.key(f"queue:{record.kind}"), identity)
                        pipe.zrem(self.key("running"), identity)
                        for slot, slot_owner in zip(slots, slot_owners, strict=True):
                            if slot_owner == identity:
                                pipe.delete(slot)
                        if not record.published:
                            pipe.zadd(
                                self.key("pending"), {identity: max(now, record.publish_until)}
                            )
                    elif record.status == TaskStatus.PROCESSING:
                        deadline = min(
                            record.heartbeat_at + self.cfg.JOBS_LEASE_SEC,
                            record.started_at + self.cfg.JOBS_TIMEOUT_SEC,
                        )
                        pipe.zadd(self.key("running"), {identity: deadline})
                    elif record.status == TaskStatus.QUEUED and queue_score is not None:
                        pipe.zrem(self.key("running"), identity)
                        for slot, slot_owner in zip(slots, slot_owners, strict=True):
                            if slot_owner == identity:
                                pipe.delete(slot)
                        pipe.zadd(self.key(f"queue:{record.kind}"), {identity: queue_score})
                    await pipe.execute()
                    if not was_terminal and record.status in TERMINAL_STATUSES:
                        log_terminal(record, now, previous_step)
                    return record
                except WatchError:
                    continue

    async def return_unstarted(self, claimed: JobRecord, queue_score: float) -> None:
        """종료 신호와 선점 응답이 겹치면 아직 실행하지 않은 작업을 원래 순서로 돌린다."""

        def update(record: JobRecord, now: float) -> bool:
            if not self.owns(record, claimed.owner, now):
                return False
            if record.cancel_requested:
                self.terminate(record, "CANCELED")
            else:
                record.status = TaskStatus.QUEUED
                record.owner = record.started_at = record.heartbeat_at = None
            return True

        await self.change(claimed.identity, update, queue_score=queue_score)

    def owns(self, record: JobRecord, owner: str, now: float) -> bool:
        return (
            record.status == TaskStatus.PROCESSING
            and record.owner == owner
            and now < record.heartbeat_at + self.cfg.JOBS_LEASE_SEC
            and now < record.started_at + self.cfg.JOBS_TIMEOUT_SEC
        )

    async def heartbeat(self, identity: str, owner: str) -> bool:
        def update(record: JobRecord, now: float) -> bool:
            if not self.owns(record, owner, now):
                return False
            record.heartbeat_at = now
            return True

        return await self.change(identity, update) is not None

    async def progress(self, identity: str, owner: str, step: str, done: int, total: int) -> bool:
        def update(record: JobRecord, now: float) -> bool:
            if not self.owns(record, owner, now) or record.cancel_requested:
                return False
            allowed = (
                {
                    "DOWNLOADING",
                    "EMBEDDING",
                    "CLOCK",
                    "LOCATING",
                    "CLUSTERING",
                    "FILTERING",
                    "FINALIZING",
                }
                if record.kind == "process"
                else {"SELECTING", "CAPTIONING", "WRITING", "VERIFYING"}
            )
            if step not in allowed or total != record.progress.total or not 0 <= done <= total:
                raise ValueError("작업 종류와 진행률 계약 불일치")
            record.current_step = step
            record.progress = Progress(done=done, total=total)
            return True

        return await self.change(identity, update) is not None

    @staticmethod
    def terminate(
        record: JobRecord,
        status: TerminalStatus,
        result: dict | None = None,
        error: ErrorBody | None = None,
    ) -> None:
        record.event = JobResultEvent(
            type=f"{record.kind}_result",
            trip_id=record.trip_id,
            execution_id=record.execution_id,
            status=status,
            result=result,
            error=error,
        ).enforce_size_limit()
        record.status = TaskStatus(record.event.status)
        record.current_step = None
        if record.status == TaskStatus.COMPLETED:
            record.progress.done = record.progress.total

    async def finish(
        self,
        identity: str,
        owner: str,
        status: TerminalStatus,
        result: dict | None = None,
        error: ErrorBody | None = None,
    ) -> bool:
        def update(record: JobRecord, now: float) -> bool:
            if not self.owns(record, owner, now):
                return False
            if record.cancel_requested:
                self.terminate(record, "CANCELED")
            else:
                self.terminate(record, status, result, error)
            return True

        return await self.change(identity, update) is not None

    async def cancel(self, trip_id: int, kind: JobKind) -> JobRecord:
        target = await self.latest(trip_id, kind)

        def update(record: JobRecord, now: float) -> bool:
            if record.status in TERMINAL_STATUSES:
                raise AppError(ErrorCode.TASK_NOT_CANCELABLE)
            record.cancel_requested = True
            if record.status == TaskStatus.QUEUED:
                self.terminate(record, "CANCELED")
            return True

        changed = await self.change(target.identity, update)
        if changed is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND)
        return changed

    async def observe_stages(self, kind: JobKind, stages) -> None:
        """작업의 단계별 시간을 지연 히스토그램에 더한다. phase는 단계 이름, outcome은 단계 결과"""
        if not stages:
            return
        async with self.redis.pipeline(transaction=True) as pipe:
            for record in stages:
                observe_commands(
                    pipe,
                    self.key("metrics:operations"),
                    kind,
                    record.result,
                    record.full_name,
                    record.seconds,
                )
            await pipe.execute()

    async def record_peak_memory(self, kind: JobKind, peak_bytes: int) -> None:
        """가장 최근 작업의 실행 프로세스 최대 메모리

        정리 실행기는 /metrics가 없어 Redis로 공유한다.
        """
        await self.redis.hset(
            self.key("metrics:resources"), f"{kind}|peak_memory_bytes", peak_bytes
        )

    async def recover(self) -> None:
        now = await self.now()
        for identity in await self.redis.zrangebyscore(
            self.key("running"), "-inf", now, start=0, num=100
        ):

            def update(record: JobRecord, current: float) -> bool:
                if record.status != TaskStatus.PROCESSING:
                    return False
                timed_out = current >= record.started_at + self.cfg.JOBS_TIMEOUT_SEC
                dead = current >= record.heartbeat_at + self.cfg.JOBS_LEASE_SEC
                if not timed_out and not dead:
                    return False
                code = ErrorCode.TIMEOUT if timed_out else ErrorCode.WORKER_DEAD
                self.terminate(
                    record, "FAILED", error=ErrorBody(code=code, message="실행이 중단되었습니다.")
                )
                return True

            await self.change(identity, update)

    async def claim_result(self, identity: str, publisher: str) -> JobRecord | None:
        def update(record: JobRecord, now: float) -> bool:
            if record.event is None or record.published or record.publish_until > now:
                return False
            record.publish_owner = publisher
            record.publish_until = now + self.cfg.JOBS_PUBLISH_LEASE_SEC
            return True

        return await self.change(identity, update)

    async def acknowledge_result(self, identity: str, publisher: str) -> None:
        key = self.record_key(identity)
        while True:
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if not raw:
                        return
                    record = JobRecord.model_validate_json(raw)
                    if record.publish_owner != publisher:
                        return
                    record.published = True
                    pipe.multi()
                    pipe.set(key, record.model_dump_json())
                    pipe.zrem(self.key("pending"), identity)
                    await pipe.execute()
                    return
                except WatchError:
                    continue

    async def pending(self) -> list[str]:
        return await self.redis.zrangebyscore(
            self.key("pending"), "-inf", await self.now(), start=0, num=100
        )
