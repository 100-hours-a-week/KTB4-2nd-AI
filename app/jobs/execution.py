"""작업별 subprocess 진입점. 기능 담당자는 async handler(request, reporter)를 연결한다.

정리는 대기 자식(--standby)으로 미리 띄워 두고 작업은 stdin으로 전달
업로드 사진 묶음(photo_batch)도 같은 대기 자식이 받음, 대기 자식이 없으면 --photo-batch로 새로 기동
"""

import asyncio
import importlib
import inspect
import json
import resource
import sys
import time
from collections.abc import Callable
from typing import Literal

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
from app.core.tracing import StageRecord, current_trace, end_trace, start_trace
from app.jobs.config import JobsSettings
from app.jobs.models import JobKind, PhotoMetricsRecord
from app.jobs.process_guard import parent_guard
from app.jobs.redis_store import RedisJobStore
from app.schemas.task import TERMINAL_STATUSES, ErrorBody

log = get_logger(__name__)


class JobReporter:
    def __init__(
        self, store: RedisJobStore, identity: str, owner: str, kind: JobKind | None = None
    ) -> None:
        self.store, self.identity, self.owner = store, identity, owner
        # 처리량(사진/초)은 진행률이 사진 수인 정리에서만 의미가 있다.
        self.kind = kind
        # 실패 로그에 남길 마지막 진행 단계. 저장 성공 여부와 무관하게 시도한 단계
        self.last_step: str | None = None
        self._step_started: float | None = None
        self._step_progress = (0, 0)

    def close_step(self, result: str) -> None:
        """진행 중인 단계의 시간을 Trace에 기록. 단계가 바뀔 때와 작업이 끝날 때 호출"""
        if self.last_step is None or self._step_started is None:
            return
        trace = current_trace()
        if trace:
            done, total = self._step_progress
            seconds = time.monotonic() - self._step_started
            trace.add_stage(
                StageRecord(stage_name(self.last_step), result, seconds),
                progress_done=done,
                progress_total=total,
                # 정리의 다운로드와 임베딩 처리량(사진/초). 스토리의 진행률은 단계 번호라 제외
                items_per_sec=round(done / seconds, 2)
                if self.kind == "process" and done and seconds > 0
                else None,
            )
        self._step_started = None

    async def report(self, step: str, done: int, total: int) -> None:
        if step != self.last_step:
            self.close_step("success")
            self.last_step = step
            self._step_started = time.monotonic()
            if trace := current_trace():
                trace.current_stage = stage_name(step)
        self._step_progress = (done, total)
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
    reporter = JobReporter(store, identity, owner, record.kind)

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

    trace, token = start_trace(record.kind, emit_stage_logs=True)
    status = "ABORTED"
    try:
        await reporter.check_cancel()
        result = await handler(record.request, reporter)
        status = "COMPLETED"
        reporter.close_step("success")
        await store.finish(identity, owner, "COMPLETED", result=result)
    except PipelineCancelled:
        status = "CANCELED"
        reporter.close_step("cancelled")
        await store.finish(identity, owner, "CANCELED")
    except PipelineError as exc:
        status = "FAILED"
        log.warning("작업 실패", extra=failure(exc, exc.code))
        reporter.close_step("error")
        await store.finish(
            identity,
            owner,
            "FAILED",
            error=ErrorBody(code=exc.code, message=exc.message, detail=exc.detail),
        )
    except Exception as exc:
        status = "FAILED"
        log.error("작업 내부 오류", extra=failure(exc, ErrorCode.INTERNAL_ERROR))
        reporter.close_step("error")
        # 입력 사진이나 인증값을 오류 문자열로 외부에 노출하지 않는다.
        await store.finish(
            identity,
            owner,
            "FAILED",
            error=ErrorBody(code=ErrorCode.INTERNAL_ERROR, message="작업 실행에 실패했습니다."),
        )
    finally:
        # 소유권 상실로 태스크가 취소되면 status는 ABORTED로 남고 아래 지표 기록은 실행되지 않는다.
        reporter.close_step("cancelled")
        if status == "ABORTED":
            log_summary(record.kind, status, trace)
        end_trace(token)
    try:
        async with asyncio.timeout(2):
            # 취소 요청이 있으면 저장소가 CANCELED로 바꿔 저장하므로 실제 저장된 상태를 요약에 쓴다.
            stored = await store.get(identity)
            if stored is not None and stored.event is not None:
                status = str(stored.status)
            log_summary(record.kind, status, trace)
            await store.observe_stages(record.kind, trace.stages)
            await store.record_peak_memory(record.kind, max_rss_bytes())
            await record_evaluations(store, status, trace.evaluations)
    except Exception as exc:
        # 관측 실패가 이미 저장한 작업 결과를 바꾸지 않는다. 취소(CancelledError)는 그대로 전파된다.
        log.warning(
            "단계 지표 기록 실패", extra={"event": "ai_metrics_write_failed", **error_fields(exc)}
        )
        if not trace.summarized:
            log_summary(record.kind, status, trace)


async def record_evaluations(store: RedisJobStore, status: str, reports: list[dict]) -> None:
    """정답 기반 자동 평가는 결과가 COMPLETED로 저장된 작업만 로그와 지표로 남긴다."""
    if not reports:
        return
    if status != "COMPLETED":
        log.info(
            "작업이 완료되지 않아 자동 평가를 기록하지 않음",
            extra={"event": "ai_evaluation_skipped", "result": status},
        )
        return
    from app.observability.quality import publish

    for report in reports:
        log.info("정답 기반 자동 평가", extra=report)
        # 데이터셋 일부만 올린 여행은 로그로만 남기고 대시보드 지표는 덮어쓰지 않는다.
        if report.get("dashboard", True):
            await publish(store.redis, store.prefix, report)


def max_rss_bytes() -> int:
    """이 실행 프로세스의 최대 상주 메모리. 작업마다 새 프로세스라 작업 하나의 최댓값이다.

    getrusage는 Linux에서 KiB, macOS에서 바이트 단위로 반환한다.
    """
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def log_summary(kind: str, status: str, trace) -> None:
    """작업 하나를 한 줄로 요약. 단계별 시간, 외부 호출 결과, 자원 사용량"""
    trace.summarized = True
    usage = resource.getrusage(resource.RUSAGE_SELF)
    log.info(
        "작업 실행 요약",
        extra={
            "event": "ai_job_summary",
            "operation": kind,
            "result": status,
            "stages_ms": trace.stage_ms() or None,
            "model_calls": trace.call_counts(),
            "gpu_wait_ms": round(trace.gpu_wait_seconds * 1000, 2),
            "max_rss_mb": round(max_rss_bytes() / (1024 * 1024), 1),
            "cpu_user_sec": round(usage.ru_utime, 2),
            "cpu_system_sec": round(usage.ru_stime, 2),
            **trace.fields,
        },
    )


def handoff_line(kind: Literal["process", "photo_batch"], **fields) -> bytes:
    """대기 자식의 stdin에 쓰는 작업 1건. 보낸 시각은 자식의 준비 대기 시간 측정용

    process는 identity와 owner, photo_batch는 identity, trip_id, batch_no, wait_ms
    """
    message = {"kind": kind, **fields, "sent_at": time.time()}
    return (json.dumps(message) + "\n").encode()


def redis_client(cfg: JobsSettings) -> Redis:
    return Redis.from_url(
        cfg.REDIS_URL.get_secret_value(),
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


BatchHandler = Callable[[dict[int, str]], dict[int, PhotoMetricsRecord]]


def batch_handler_for(cfg: JobsSettings) -> BatchHandler:
    """사진 묶음 처리 함수 embed_photos({사진 ID: 사본 키}) -> {저장한 사진 ID: 사진 지표}"""
    if cfg.JOBS_FAKE_PIPELINE:
        from app.jobs.mock import embed_photos
    else:
        from app.jobs.process_handler import embed_photos
    return embed_photos


async def run_batch(
    store: RedisJobStore,
    payload: dict,
    handler: BatchHandler,
    handoff_wait_ms: float | None = None,
) -> None:
    """사진 묶음 1건 임베딩과 사진 지표 저장. 실패해도 작업 상태를 바꾸지 않고 로그만 남김

    결과는 Qdrant 벡터와 Redis 사진 지표뿐이라 빠진 사진은 process가 임베딩하고 지표를 계산
    """
    bind_trip(payload["trip_id"])
    batch_no = payload["batch_no"]
    fields = {
        "event": "ai_photo_batch",
        "operation": "photo_batch",
        "batch_no": batch_no,
        "wait_ms": payload["wait_ms"],
        "handoff_wait_ms": round(handoff_wait_ms, 2) if handoff_wait_ms is not None else None,
    }
    photos = await store.photo_batch(payload["identity"], batch_no)
    if photos is None:
        log.info("process 등록으로 지워진 사진 묶음 건너뜀", extra={**fields, "result": "skipped"})
        return
    fields["attachment_count"] = len(photos)
    started = time.monotonic()
    try:
        embedded = await asyncio.to_thread(
            handler, {photo.trip_attachment_id: photo.analyze_storage_key for photo in photos}
        )
        await store.save_photo_metrics(embedded)
    except Exception as exc:
        log.warning(
            "사진 묶음 임베딩 실패, process에서 다시 임베딩",
            extra={
                **fields,
                "result": "failed",
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                **error_fields(exc),
            },
        )
        return
    log.info(
        "사진 묶음 임베딩",
        extra={
            **fields,
            "result": "completed",
            "embedded_count": len(embedded),
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
        },
    )


async def run_batch_job(
    cfg: JobsSettings,
    payload: dict,
    handler: BatchHandler,
    handoff_wait_ms: float | None = None,
) -> None:
    redis = redis_client(cfg)
    try:
        await run_batch(RedisJobStore(redis, cfg), payload, handler, handoff_wait_ms)
    finally:
        await redis.aclose()


async def run_job(
    cfg: JobsSettings,
    identity: str,
    owner: str,
    handler: Callable | None = None,
    handoff_wait_ms: float | None = None,
) -> None:
    """작업 1건 실행. handler가 없으면 레코드의 작업 종류로 적재"""
    redis = redis_client(cfg)
    store = RedisJobStore(redis, cfg)
    record = await store.get(identity)
    if record is None:
        await redis.aclose()
        return

    bind_trip(record.trip_id)
    bind_job(record.execution_id)
    if handoff_wait_ms is not None:
        # 0에 가까우면 인계 시점에 준비 완료, 크면 그만큼 남은 준비를 기다린 것
        log.info(
            "대기 자식에 작업 인계",
            extra={
                "event": "ai_job_handoff",
                "operation": record.kind,
                "handoff_wait_ms": round(handoff_wait_ms, 2),
            },
        )

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

    task = asyncio.create_task(
        execute(store, identity, owner, handler or handler_for(cfg, record.kind))
    )
    monitor = asyncio.create_task(guard())
    try:
        await asyncio.wait([task, monitor], return_when=asyncio.FIRST_COMPLETED)
    finally:
        task.cancel()
        monitor.cancel()
        await asyncio.gather(task, monitor, return_exceptions=True)
        await redis.aclose()


async def main(identity: str, owner: str) -> None:
    cfg = JobsSettings()
    setup_logging(cfg.LOG_LEVEL, cfg.RELEASE)
    await run_job(cfg, identity, owner)


def standby() -> None:
    """정리 대기 자식. import와 처리기 적재를 마친 뒤 stdin으로 작업 1건을 받아 실행

    받는 작업은 process 또는 photo_batch. 묶음 처리기는 EMBED_ON_PHOTOS_READY일 때 미리 적재
    """
    cfg = JobsSettings()
    setup_logging(cfg.LOG_LEVEL, cfg.RELEASE)
    handler = handler_for(cfg, "process")
    batch_handler = batch_handler_for(cfg) if cfg.EMBED_ON_PHOTOS_READY else None
    line = sys.stdin.readline()
    if not line:
        # 실행기가 작업을 넘기지 않고 stdin을 닫은 경우
        return
    handoff = json.loads(line)
    handoff_wait_ms = max(0.0, (time.time() - handoff["sent_at"]) * 1000)
    if handoff["kind"] == "photo_batch":
        batch_handler = batch_handler or batch_handler_for(cfg)
        asyncio.run(run_batch_job(cfg, handoff, batch_handler, handoff_wait_ms))
    else:
        asyncio.run(run_job(cfg, handoff["identity"], handoff["owner"], handler, handoff_wait_ms))


def photo_batch_main(payload: dict) -> None:
    """대기 자식 없이 새로 기동한 사진 묶음 자식"""
    cfg = JobsSettings()
    setup_logging(cfg.LOG_LEVEL, cfg.RELEASE)
    asyncio.run(run_batch_job(cfg, payload, batch_handler_for(cfg)))


if __name__ == "__main__":
    if sys.argv[1] == "--standby":
        with parent_guard(int(sys.argv[2])):
            standby()
    elif sys.argv[1] == "--photo-batch":
        with parent_guard(int(sys.argv[3])):
            photo_batch_main(json.loads(sys.argv[2]))
    else:
        with parent_guard(int(sys.argv[3])):
            asyncio.run(main(sys.argv[1], sys.argv[2]))
