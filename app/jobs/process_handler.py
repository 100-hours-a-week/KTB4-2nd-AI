"""v2 정리 작업을 기존 Qwen 파이프라인에 연결한다."""

from contextlib import ExitStack

import httpx
from pydantic import ValidationError

from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.embedding.client import MODEL_VERSION, EmbeddingUnavailable
from app.infra.qdrant import QdrantStore
from app.jobs.config import JobsSettings
from app.jobs.execution import JobReporter
from app.jobs.models import ProcessMessage
from app.jobs.sync_bridge import run_sync
from app.pipeline.bootstrap import build_context, build_engine, build_images, wait_for_qdrant
from app.pipeline.run import run as run_pipeline


def _failure(code: ErrorCode, message: str) -> PipelineError:
    error = PipelineError(message)
    error.code = code
    return error


def _run(request: ProcessMessage, cfg: JobsSettings, progress, cancelled) -> dict:
    def check():
        if cancelled():
            raise PipelineCancelled()

    try:
        check()
        with ExitStack() as resources:
            engine = build_engine(cfg)
            resources.callback(engine.close)
            check()
            images = build_images(cfg)
            if close := getattr(images, "close", None):
                resources.callback(close)
            wait_for_qdrant(cfg.QDRANT_URL)
            check()
            vectors = QdrantStore(cfg.QDRANT_URL, engine.model_version, engine.dim)
            resources.callback(vectors.close)
            ctx = build_context(cfg, engine, images=images, qdrant=vectors)
            return run_pipeline(request, ctx, progress, cancelled).model_dump(mode="json")
    except Exception as exc:
        # run()의 안전한 PipelineError가 감싼 외부 호출 오류까지 확인한다.
        pending, seen, causes = [exc], set(), []
        while pending:
            cause = pending.pop()
            if id(cause) in seen:
                continue
            seen.add(id(cause))
            causes.append(cause)
            for nested in (cause.__cause__, cause.__context__, getattr(cause, "source", None)):
                if isinstance(nested, BaseException):
                    pending.append(nested)
        if any(isinstance(cause, (httpx.TimeoutException, TimeoutError)) for cause in causes):
            raise _failure(
                ErrorCode.TIMEOUT, "정리 작업의 외부 호출 시간이 초과되었습니다."
            ) from exc
        if any(isinstance(cause, EmbeddingUnavailable) for cause in causes):
            raise _failure(ErrorCode.MODEL_NOT_READY, "임베딩 서버를 사용할 수 없습니다.") from exc
        raise


async def run(request: dict, reporter: JobReporter) -> dict:
    try:
        message = ProcessMessage.model_validate(request)
        if message.period.start_date > message.period.end_date:
            raise ValueError("여행 기간 역전")
        ids = [photo.trip_attachment_id for photo in message.attachments]
        if len(set(ids)) != len(ids) or any(photo_id <= 0 for photo_id in ids):
            raise ValueError("사진 ID 중복 또는 범위 오류")
    except (ValidationError, ValueError):
        raise _failure(ErrorCode.INVALID_REQUEST, "정리 요청 형식이 올바르지 않습니다.") from None
    cfg = JobsSettings()
    if cfg.FAKE_PIPELINE or cfg.EMBED_MODEL != MODEL_VERSION:
        raise _failure(
            ErrorCode.MODEL_NOT_READY, "v2 정리 실행기는 Qwen HTTP 임베딩 설정이 필요합니다."
        )
    await reporter.check_cancel()
    return await run_sync(lambda progress, cancel: _run(message, cfg, progress, cancel), reporter)
