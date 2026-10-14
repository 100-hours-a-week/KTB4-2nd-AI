"""Redis 단일 인스턴스의 WATCH/MULTI로 등록, 실행 선점, 결과 대기를 원자적으로 변경한다.

진행 중 기록과 미발행 결과에는 TTL을 두지 않는다. Redis 전체 유실은 v2의 알려진 한계다.
"""

import uuid
from collections.abc import Callable
from typing import Literal

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.core.errors import AppError, ErrorCode
from app.jobs.config import JobsSettings
from app.jobs.models import (
    PHOTOS_READY_MAX_BYTES,
    JobKind,
    JobRecord,
    JobResultEvent,
    PhotoBatchBuffer,
    PhotosReadyMessage,
    TerminalStatus,
)
from app.schemas.task import TERMINAL_STATUSES, ErrorBody, Progress, TaskStatus


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

    async def stage_photos(
        self, message: PhotosReadyMessage, request_topic: str
    ) -> Literal["staged", "duplicate", "closed"]:
        """사본 경로만 공유 보관한다. process 전에는 실행 대기열과 상태를 만들지 않는다."""
        key = self.photo_batches_key(message.identity)
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
                    pipe.multi()
                    pipe.set(key, payload, ex=self.cfg.JOBS_PHOTOS_READY_TTL_SEC)
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
        while True:
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key, ignored, latest, sequence)
                    if await pipe.exists(key, ignored):
                        return "duplicate"
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
                        pipe.set(ignored, "1")
                        pipe.delete(photo_batches)
                        await pipe.execute()
                        return "busy"
                    record.created_at = await self.now()
                    # 같은 시각과 시계 보정에도 등록 순서를 유지한다.
                    order = int(await pipe.get(sequence) or 0) + 1
                    pipe.multi()
                    pipe.set(sequence, order)
                    pipe.set(key, record.model_dump_json())
                    pipe.set(latest, record.identity)
                    # 최종 process가 전체 목록의 기준이다. 묶음의 누락과 만료를 기다리지 않는다.
                    pipe.delete(photo_batches)
                    destination = "pending" if record.event else f"queue:{record.kind}"
                    score = record.created_at if record.event else order
                    pipe.zadd(self.key(destination), {record.identity: score})
                    await pipe.execute()
                    return "accepted"
                except WatchError:
                    continue

    async def claim(self, kind: JobKind, worker: str) -> JobRecord | None:
        queue = self.key(f"queue:{kind}")
        slot = self.key(f"slot:{kind}")
        while True:
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(queue, slot)
                    if await pipe.get(slot):
                        return None
                    waiting = await pipe.zrange(queue, 0, 0)
                    if not waiting:
                        return None
                    identity = waiting[0]
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
                    pipe.multi()
                    pipe.zrem(queue, identity)
                    pipe.set(slot, identity)
                    pipe.set(key, record.model_dump_json())
                    pipe.zadd(
                        self.key("running"),
                        {identity: now + min(self.cfg.JOBS_LEASE_SEC, self.cfg.JOBS_TIMEOUT_SEC)},
                    )
                    await pipe.execute()
                    return record
                except WatchError:
                    continue

    async def change(
        self, identity: str, update: Callable[[JobRecord, float], bool]
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
                    now = await self.now()
                    if not update(record, now):
                        return None
                    slot = self.key(f"slot:{record.kind}")
                    await pipe.watch(slot)
                    slot_owner = await pipe.get(slot)
                    pipe.multi()
                    pipe.set(key, record.model_dump_json())
                    if record.status in TERMINAL_STATUSES:
                        pipe.zrem(self.key(f"queue:{record.kind}"), identity)
                        pipe.zrem(self.key("running"), identity)
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
                    await pipe.execute()
                    return record
                except WatchError:
                    continue

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
