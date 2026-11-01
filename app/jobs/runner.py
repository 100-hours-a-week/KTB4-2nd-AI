"""Redis 실행 루프. 작업은 subprocess에서 실행해 heartbeat와 API를 막지 않는다."""

import asyncio
import json
import os
import sys
import uuid
from asyncio.subprocess import Process
from collections.abc import Awaitable, Callable

from pydantic import SecretStr
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.errors import ErrorCode
from app.core.logging import error_fields, get_logger, setup_logging
from app.infra.place_index import prepare_place_index
from app.jobs.config import JobsSettings
from app.jobs.execution import handler_for, handoff_line
from app.jobs.models import JobKind, JobRecord
from app.jobs.redis_store import PhotoBatchClaim, RedisJobStore
from app.schemas.task import TERMINAL_STATUSES, ErrorBody

log = get_logger(__name__)


def job_fields(record: JobRecord) -> dict:
    """실행기 프로세스는 작업 맥락을 bind하지 않으므로 식별자를 직접 붙인다."""
    return {"operation": record.kind, "trip_id": record.trip_id, "job_id": record.execution_id}


async def stop_child(process) -> None:
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()


def child_env(cfg: JobsSettings) -> dict[str, str]:
    env = os.environ.copy()
    for name, value in cfg:
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
    return env


async def spawn_child(store: RedisJobStore, record: JobRecord) -> Process:
    """작업 정보를 실행 인자로 넘겨 자식을 새로 기동"""
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.jobs.execution",
        record.identity,
        record.owner,
        str(os.getpid()),
        env=child_env(store.cfg),
    )


async def spawn_batch_child(store: RedisJobStore, payload: dict) -> Process:
    """사진 묶음 정보를 실행 인자로 넘겨 자식을 새로 기동"""
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.jobs.execution",
        "--photo-batch",
        json.dumps(payload),
        str(os.getpid()),
        env=child_env(store.cfg),
    )


async def handoff(process: Process, line: bytes) -> bool:
    """대기 자식의 stdin에 작업 1건 전달. 자식이 이미 종료됐으면 False"""
    try:
        process.stdin.write(line)
        await process.stdin.drain()
    except ConnectionError:
        return False
    process.stdin.close()
    return True


class StandbyChildren:
    """정리 작업용 대기 자식. import와 처리기 적재를 마치고 stdin으로 작업 1건을 기다림"""

    def __init__(self, store: RedisJobStore, size: int) -> None:
        self.store, self.size = store, size
        self.children: list[Process] = []

    def alive(self, process: Process) -> bool:
        if process.returncode is None:
            return True
        log.warning(
            "대기 자식이 작업을 받기 전에 종료",
            extra={"event": "ai_job_standby_exit", "exit_code": process.returncode},
        )
        return False

    async def fill(self) -> None:
        """종료된 대기 자식을 버리고 설정 수까지 기동. 실패하면 다음 작업 때 다시 시도"""
        self.children = [process for process in self.children if self.alive(process)]
        try:
            while len(self.children) < self.size:
                self.children.append(
                    await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "app.jobs.execution",
                        "--standby",
                        str(os.getpid()),
                        env=child_env(self.store.cfg),
                        stdin=asyncio.subprocess.PIPE,
                    )
                )
        except OSError as exc:
            log.warning(
                "대기 자식 기동 실패",
                extra={"event": "ai_job_standby_spawn_failed", **error_fields(exc)},
            )

    def take(self) -> Process | None:
        """먼저 기동한 자식부터 사용, 준비가 끝났을 가능성이 가장 높음"""
        while self.children:
            process = self.children.pop(0)
            if self.alive(process):
                return process
        return None

    async def start(self, line: bytes, spawn: Callable[[], Awaitable[Process]]) -> Process:
        """대기 자식에 작업을 넘기고 바로 보충. 넘길 자식이 없으면 spawn으로 새로 기동"""
        process = self.take()
        if process is not None and not await handoff(process, line):
            await stop_child(process)
            process = None
        if process is None:
            process = await spawn()
        try:
            await self.fill()
        except asyncio.CancelledError:
            # 보충 중 실행기가 종료되면 작업을 받은 자식도 함께 종료
            await stop_child(process)
            raise
        return process

    async def run(self, record: JobRecord) -> None:
        line = handoff_line("process", identity=record.identity, owner=record.owner)
        process = await self.start(line, lambda: spawn_child(self.store, record))
        await supervise(self.store, record, process)

    async def close(self) -> None:
        while self.children:
            await stop_child(self.children.pop())


async def run_claimed(store: RedisJobStore, record: JobRecord) -> None:
    await supervise(store, record, await spawn_child(store, record))


async def run_photo_batch(
    store: RedisJobStore, claim: PhotoBatchClaim, standby: StandbyChildren | None
) -> None:
    """사진 묶음 1건을 자식에서 실행. lease 없이 처리 시간 상한만 두고, 넘기면 강제 종료

    묶음 결과는 Redis 상태가 아니라 Qdrant 벡터뿐이라 실패나 종료 시 회수할 것이 없음
    """
    payload = {
        "identity": claim.identity,
        "trip_id": claim.trip_id,
        "batch_no": claim.batch_no,
        "wait_ms": round((await store.now() - claim.enqueued_at) * 1000, 2),
    }

    def spawn() -> Awaitable[Process]:
        return spawn_batch_child(store, payload)

    if standby:
        process = await standby.start(handoff_line("photo_batch", **payload), spawn)
    else:
        process = await spawn()
    fields = {"operation": "photo_batch", "trip_id": claim.trip_id, "batch_no": claim.batch_no}
    try:
        exit_code = await asyncio.wait_for(
            process.wait(), timeout=store.cfg.JOBS_PHOTO_BATCH_TIMEOUT_SEC
        )
    except TimeoutError:
        log.warning(
            "사진 묶음 처리 시간 초과로 실행 프로세스 종료",
            extra={
                **fields,
                "event": "ai_photo_batch_timeout",
                "timeout_sec": store.cfg.JOBS_PHOTO_BATCH_TIMEOUT_SEC,
            },
        )
    else:
        if exit_code != 0:
            log.warning(
                "사진 묶음 실행 프로세스 비정상 종료",
                extra={**fields, "event": "ai_photo_batch_exit", "exit_code": exit_code},
            )
    finally:
        await stop_child(process)


async def supervise(store: RedisJobStore, record: JobRecord, process: Process) -> None:
    """자식이 끝날 때까지 heartbeat 갱신, 결과 저장 없이 종료되면 WORKER_DEAD로 회수"""
    try:
        while process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=store.cfg.JOBS_HEARTBEAT_SEC)
            except TimeoutError:
                if not await store.heartbeat(record.identity, record.owner):
                    current = await store.get(record.identity)
                    if current is not None and current.status in TERMINAL_STATUSES:
                        # 자식이 결과를 저장하고 지표 기록을 마무리하는 중이며 소유권 상실이 아니다.
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5)
                        except TimeoutError:
                            pass
                    else:
                        log.warning(
                            "heartbeat 갱신 실패로 실행 프로세스 중단",
                            extra={**job_fields(record), "event": "ai_job_lease_lost"},
                        )
                    await stop_child(process)
                    await store.recover()
                    return
        # 정상 종료라면 자식이 결과를 저장했다. 미저장 상태만 실패로 회수한다.
        recovered = await store.finish(
            record.identity,
            record.owner,
            "FAILED",
            error=ErrorBody(
                code=ErrorCode.WORKER_DEAD, message="실행 프로세스가 결과 저장 전에 종료되었습니다."
            ),
        )
        if recovered:
            # 음수 종료 코드는 신호 번호. -9는 메모리 부족 등으로 강제 종료된 경우가 많다.
            log.error(
                "실행 프로세스가 결과 저장 전에 종료",
                extra={
                    **job_fields(record),
                    "event": "ai_worker_exit",
                    "result": "failed",
                    "error_code": str(ErrorCode.WORKER_DEAD),
                    "exit_code": process.returncode,
                },
            )
    finally:
        await stop_child(process)


async def runner_loop(
    store: RedisJobStore, kind: JobKind, stop: asyncio.Event | None = None
) -> None:
    handler_for(store.cfg, kind)  # 실제 구현이 미설정이면 기동을 거부한다.
    stop = stop or asyncio.Event()
    worker = uuid.uuid4().hex
    standby = None
    if kind == "process" and store.cfg.JOBS_PROCESS_STANDBY:
        standby = StandbyChildren(store, store.cfg.JOBS_PROCESS_STANDBY)
    try:
        if standby:
            await standby.fill()
        while not stop.is_set():
            record = None
            try:
                await store.recover()
                if stop.is_set():
                    return
                record = await store.claim(kind, worker, should_stop=stop.is_set)
                if record and standby:
                    await standby.run(record)
                elif record:
                    await run_claimed(store, record)
                elif kind == "process" and store.cfg.EMBED_ON_PHOTOS_READY and not stop.is_set():
                    # process가 없을 때만 묶음을 꺼낸다. 묶음 실행 중 도착한 process가 다음 순서
                    batch = await store.claim_batch()
                    if batch:
                        await run_photo_batch(store, batch, standby)
                        continue
                if record and kind == "process":
                    continue  # 작업 직후에는 다음 작업이 이미 대기열에 있을 수 있어 바로 확인
            except asyncio.CancelledError:
                if record:
                    await store.finish(
                        record.identity,
                        record.owner,
                        "FAILED",
                        error=ErrorBody(
                            code=ErrorCode.WORKER_DEAD, message="실행기가 종료되었습니다."
                        ),
                    )
                    log.warning(
                        "실행기 종료로 진행 중인 작업 실패 처리",
                        extra={
                            **job_fields(record),
                            "event": "ai_job_runner_stopped",
                            "result": "failed",
                            "error_code": str(ErrorCode.WORKER_DEAD),
                        },
                    )
                raise
            except (RedisError, OSError) as exc:
                log.warning(
                    "실행기 연결 실패, 소유권 만료 후 회수",
                    extra={"event": "ai_job_runner_retry", **error_fields(exc)},
                )
            try:
                await asyncio.wait_for(stop.wait(), timeout=store.cfg.JOBS_POLL_SEC)
            except TimeoutError:
                pass
    finally:
        if standby:
            await standby.close()


async def story_runners(store: RedisJobStore, stop: asyncio.Event) -> None:
    """API가 하나만 남아도 실행 한도를 사용하고, 전체 한도는 Redis에서 제한한다."""
    async with asyncio.TaskGroup() as group:
        for _ in range(store.cfg.JOBS_STORY_CONCURRENCY):
            group.create_task(runner_loop(store, "story", stop))


async def drain_story_runners(task: asyncio.Task, stop: asyncio.Event, timeout: float) -> None:
    stop.set()
    done, _ = await asyncio.wait([task], timeout=timeout)
    if not done:
        # 유예가 끝나면 각 루프가 자식을 종료하고 FAILED 결과를 Redis에 남긴다.
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def main(kind: JobKind = "process") -> None:
    cfg = JobsSettings()
    setup_logging(cfg.LOG_LEVEL, cfg.RELEASE)
    redis = Redis.from_url(
        cfg.REDIS_URL.get_secret_value(),
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    try:
        if kind == "process":
            # 대기 자식을 띄우기 전에 받아 둬서, 자식들은 같은 파일을 열기만 한다.
            await asyncio.to_thread(prepare_place_index, cfg)
        await runner_loop(RedisJobStore(redis, cfg), kind)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    import signal

    async def serve() -> None:
        task = asyncio.create_task(main())
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(serve())
