"""작업별 subprocess 진입점. 기능 담당자는 async handler(request, reporter)를 연결한다."""

import asyncio
import importlib
import inspect
import sys

from redis.asyncio import Redis

from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.jobs.config import JobsSettings
from app.jobs.models import JobKind
from app.jobs.process_guard import parent_guard
from app.jobs.redis_store import RedisJobStore
from app.schemas.task import ErrorBody


class JobReporter:
    def __init__(self, store: RedisJobStore, identity: str, owner: str) -> None:
        self.store, self.identity, self.owner = store, identity, owner

    async def report(self, step: str, done: int, total: int) -> None:
        if not await self.store.progress(self.identity, self.owner, step, done, total):
            raise PipelineCancelled()

    async def check_cancel(self) -> None:
        record = await self.store.get(self.identity)
        if (
            record is None
            or record.cancel_requested
            or not self.store.owns(record, self.owner, await self.store.now())
        ):
            raise PipelineCancelled()


def handler_for(cfg: JobsSettings, kind: JobKind):
    if cfg.JOBS_FAKE_PIPELINE:
        from app.jobs.mock import run

        return run
    path = cfg.JOBS_PROCESS_HANDLER if kind == "process" else cfg.JOBS_STORY_HANDLER
    if not path or ":" not in path:
        raise ValueError(f"{kind} 실행 함수를 module:function 형식으로 설정해야 함")
    module, name = path.split(":", 1)
    handler = getattr(importlib.import_module(module), name)
    if not inspect.iscoroutinefunction(handler):
        raise ValueError("작업 실행 함수는 async 함수여야 함")
    return handler


async def execute(store: RedisJobStore, identity: str, owner: str, handler) -> None:
    record = await store.get(identity)
    if record is None or not store.owns(record, owner, await store.now()):
        return
    reporter = JobReporter(store, identity, owner)
    try:
        await reporter.check_cancel()
        result = await handler(record.request, reporter)
        await store.finish(identity, owner, "COMPLETED", result=result)
    except PipelineCancelled:
        await store.finish(identity, owner, "CANCELED")
    except PipelineError as exc:
        await store.finish(
            identity,
            owner,
            "FAILED",
            error=ErrorBody(code=exc.code, message=exc.message, detail=exc.detail),
        )
    except Exception:
        # 입력 사진이나 인증값을 오류 문자열로 외부에 노출하지 않는다.
        await store.finish(
            identity,
            owner,
            "FAILED",
            error=ErrorBody(code=ErrorCode.INTERNAL_ERROR, message="작업 실행에 실패했습니다."),
        )


async def main(identity: str, owner: str) -> None:
    cfg = JobsSettings()
    redis = Redis.from_url(
        cfg.REDIS_URL.get_secret_value(),
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    store = RedisJobStore(redis, cfg)
    record = await store.get(identity)
    if record is None:
        await redis.aclose()
        return

    async def guard() -> None:
        while True:
            await asyncio.sleep(cfg.JOBS_HEARTBEAT_SEC)
            current = await store.get(identity)
            if current is None or not store.owns(current, owner, await store.now()):
                raise RuntimeError("실행 소유권 만료")

    task = asyncio.create_task(execute(store, identity, owner, handler_for(cfg, record.kind)))
    monitor = asyncio.create_task(guard())
    try:
        await asyncio.wait([task, monitor], return_when=asyncio.FIRST_COMPLETED)
    finally:
        task.cancel()
        monitor.cancel()
        await asyncio.gather(task, monitor, return_exceptions=True)
        await redis.aclose()


if __name__ == "__main__":
    with parent_guard(int(sys.argv[3])):
        asyncio.run(main(sys.argv[1], sys.argv[2]))
