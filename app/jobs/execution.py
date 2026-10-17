"""작업별 subprocess 진입점. 기능 담당자는 async handler(request, reporter)를 연결한다."""

import asyncio
import importlib
import inspect
import sys

from redis.asyncio import Redis

from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.core.logging import (
    FAILURE_STARTUP,
    bind_job,
    bind_trip,
    error_fields,
    get_logger,
    setup_logging,
    stage_name,
)
from app.jobs.config import JobsSettings
from app.jobs.models import JobKind
from app.jobs.process_guard import parent_guard
from app.jobs.redis_store import RedisJobStore
from app.schemas.task import TERMINAL_STATUSES, ErrorBody

log = get_logger(__name__)


class JobReporter:
    def __init__(self, store: RedisJobStore, identity: str, owner: str) -> None:
        self.store, self.identity, self.owner = store, identity, owner
        # 실패 로그에 남길 마지막 진행 단계. 저장 성공 여부와 무관하게 시도한 단계
        self.last_step: str | None = None

    async def report(self, step: str, done: int, total: int) -> None:
        self.last_step = step
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

    def failure(exc: BaseException, code: ErrorCode) -> dict:
        fields = {
            "event": "ai_job_failed",
            "operation": record.kind,
            "result": "failed",
            "error_code": str(code),
            "failure_stage": stage_name(reporter.last_step) or FAILURE_STARTUP,
        }
        # 관측 필드를 만들다 실패해도 결과 저장은 이어서 진행한다.
        try:
            detail = getattr(exc, "detail", None)
            detail = detail if isinstance(detail, dict) else {}
            reason = detail.get("reason")
            # reason은 고정된 분류값만 담는다. 나머지 detail은 사진 키를 담을 수 있어 이름만
            fields["error_reason"] = reason if isinstance(reason, str) else None
            fields["detail_keys"] = sorted(str(key) for key in detail) or None
            fields.update(error_fields(exc))
        except Exception:
            fields["error_type"] = type(exc).__name__
        return fields

    try:
        await reporter.check_cancel()
        result = await handler(record.request, reporter)
        await store.finish(identity, owner, "COMPLETED", result=result)
    except PipelineCancelled:
        await store.finish(identity, owner, "CANCELED")
    except PipelineError as exc:
        log.warning("작업 실패", extra=failure(exc, exc.code))
        await store.finish(
            identity,
            owner,
            "FAILED",
            error=ErrorBody(code=exc.code, message=exc.message, detail=exc.detail),
        )
    except Exception as exc:
        log.error("작업 내부 오류", extra=failure(exc, ErrorCode.INTERNAL_ERROR))
        # 입력 사진이나 인증값을 오류 문자열로 외부에 노출하지 않는다.
        await store.finish(
            identity,
            owner,
            "FAILED",
            error=ErrorBody(code=ErrorCode.INTERNAL_ERROR, message="작업 실행에 실패했습니다."),
        )


async def main(identity: str, owner: str) -> None:
    cfg = JobsSettings()
    setup_logging(cfg.LOG_LEVEL, cfg.RELEASE)
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

    bind_trip(record.trip_id)
    bind_job(record.execution_id)

    async def guard() -> None:
        while True:
            await asyncio.sleep(cfg.JOBS_HEARTBEAT_SEC)
            current = await store.get(identity)
            if current is not None and current.status in TERMINAL_STATUSES:
                # 결과는 이미 저장됐다. 남은 마무리 작업을 소유권 상실로 오인해 끊지 않는다.
                continue
            if current is None or not store.owns(current, owner, await store.now()):
                log.warning(
                    "실행 소유권 만료로 작업 중단",
                    extra={
                        "event": "ai_job_lease_lost",
                        "operation": record.kind,
                        "result": "aborted",
                        "status": str(current.status) if current else None,
                    },
                )
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
