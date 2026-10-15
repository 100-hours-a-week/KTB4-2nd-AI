"""Redis 단일 소비 루프. CPU 작업은 subprocess에서 실행해 heartbeat와 API를 막지 않는다."""

import asyncio
import os
import sys
import uuid

from pydantic import SecretStr
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.errors import ErrorCode
from app.core.logging import get_logger, setup_logging
from app.jobs.config import JobsSettings
from app.jobs.execution import handler_for
from app.jobs.models import JobKind, JobRecord
from app.jobs.redis_store import RedisJobStore
from app.schemas.task import ErrorBody

log = get_logger(__name__)


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


async def run_claimed(store: RedisJobStore, record: JobRecord) -> None:
    child_env = os.environ.copy()
    for name, value in store.cfg:
        if value is None:
            child_env.pop(name, None)
        else:
            child_env[name] = (
                value.get_secret_value() if isinstance(value, SecretStr) else str(value)
            )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.jobs.execution",
        record.identity,
        record.owner,
        str(os.getpid()),
        env=child_env,
    )
    try:
        while process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=store.cfg.JOBS_HEARTBEAT_SEC)
            except TimeoutError:
                if not await store.heartbeat(record.identity, record.owner):
                    await stop_child(process)
                    await store.recover()
                    return
        # 정상 종료라면 자식이 결과를 저장했다. 미저장 상태만 실패로 회수한다.
        await store.finish(
            record.identity,
            record.owner,
            "FAILED",
            error=ErrorBody(
                code=ErrorCode.WORKER_DEAD, message="실행 프로세스가 결과 저장 전에 종료되었습니다."
            ),
        )
    finally:
        await stop_child(process)


async def runner_loop(store: RedisJobStore, kind: JobKind) -> None:
    handler_for(store.cfg, kind)  # 실제 구현이 미설정이면 기동을 거부한다.
    worker = uuid.uuid4().hex
    while True:
        record = None
        try:
            await store.recover()
            record = await store.claim(kind, worker)
            if record:
                await run_claimed(store, record)
        except asyncio.CancelledError:
            if record:
                await store.finish(
                    record.identity,
                    record.owner,
                    "FAILED",
                    error=ErrorBody(code=ErrorCode.WORKER_DEAD, message="실행기가 종료되었습니다."),
                )
            raise
        except (RedisError, OSError):
            log.warning(
                "실행기 연결 실패, 소유권 만료 후 회수", extra={"event": "ai_job_runner_retry"}
            )
        await asyncio.sleep(store.cfg.JOBS_POLL_SEC)


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
