"""v2 정리 작업을 기존 Qwen 파이프라인에 연결한다."""

from contextlib import ExitStack

import httpx
from pydantic import ValidationError

from app.core.errors import ErrorCode, PipelineCancelled, PipelineError
from app.core.logging import error_fields, get_logger
from app.core.tracing import current_trace, stage
from app.embedding.client import MODEL_VERSION, EmbeddingUnavailable
from app.infra.qdrant import QdrantStore
from app.jobs.config import JobsSettings
from app.jobs.execution import JobReporter
from app.jobs.models import ProcessMessage
from app.jobs.sync_bridge import run_sync
from app.observability import datasets
from app.pipeline.bootstrap import build_context, build_engine, build_images, wait_for_qdrant
from app.pipeline.run import run as run_pipeline

log = get_logger(__name__)


def _failure(code: ErrorCode, message: str) -> PipelineError:
    error = PipelineError(message)
    error.code = code
    return error


class FingerprintingImages:
    """내려받은 사본의 지문을 함께 계산한다. 사진 bytes와 실패 처리는 그대로 전달한다."""

    def __init__(self, images, request: ProcessMessage, hashes: dict) -> None:
        self.images, self.hashes = images, hashes
        self.ids = {
            photo.analyze_storage_key: photo.trip_attachment_id for photo in request.attachments
        }

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        blobs = self.images.get_many(keys)
        for key, data in blobs.items():
            if key in self.ids:
                try:
                    self.hashes[self.ids[key]] = datasets.candidates(data)
                except Exception:
                    # 지문 계산은 관측 기능이다. 어떤 실패도 다운로드 결과를 바꾸지 않는다.
                    self.hashes[self.ids[key]] = None
        return blobs

    def close(self) -> None:
        if close := getattr(self.images, "close", None):
            close()


def auto_evaluate(result: dict, hashes: dict, cfg: JobsSettings) -> list[dict]:
    """평가 데이터셋이면 두 미분류 정책의 보고서를 만든다. 실패해도 정리 결과는 바꾸지 않는다.

    보고서의 로그와 지표 기록은 결과를 저장한 뒤 실행기가 한다(app.jobs.execution).
    """
    try:
        with stage("evaluate", parent="finalizing") as fields:
            match = datasets.closest(hashes)
            # 평가 여행이 아닌 경우와 평가 여행인데 덜 맞은 경우를 구분할 수 있게 남긴다.
            fields["closest_dataset"] = match.dataset if match else None
            fields["match_ratio"] = round(match.ratio, 3) if match else None
            fields["dataset"] = match.dataset if match and match.accepted else None
            if match is None or not match.accepted:
                return []
            return datasets.score(match, result, model=cfg.EMBED_MODEL, release=cfg.RELEASE)
    except Exception as exc:
        log.warning(
            "정답 기반 자동 평가 실패", extra={"event": "ai_evaluation_failed", **error_fields(exc)}
        )
        return []


def evaluation_enabled(cfg: JobsSettings) -> bool:
    """기준 지문 파일을 읽지 못하면 자동 평가만 끄고 정리 작업은 그대로 진행한다."""
    if not cfg.EVAL_AUTO_DETECT:
        return False
    try:
        return bool(datasets.reference_sets())
    except Exception as exc:
        log.warning(
            "평가 기준 지문을 읽지 못해 자동 평가를 끔",
            extra={"event": "ai_evaluation_failed", **error_fields(exc)},
        )
        return False


def _run(request: ProcessMessage, cfg: JobsSettings, progress, cancelled, reports=None) -> dict:
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
            hashes: dict = {}
            if reports is not None and evaluation_enabled(cfg):
                images = FingerprintingImages(images, request, hashes)
            wait_for_qdrant(cfg.QDRANT_URL)
            check()
            vectors = QdrantStore(cfg.QDRANT_URL, engine.model_version, engine.dim)
            resources.callback(vectors.close)
            ctx = build_context(cfg, engine, images=images, qdrant=vectors)
            result = run_pipeline(request, ctx, progress, cancelled).model_dump(mode="json")
            if hashes:
                reports.extend(auto_evaluate(result, hashes, cfg))
            return result
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
        # 임베딩 클라이언트는 상태 코드를 오류 메시지에 담지 않으므로 원인 사슬에서 꺼낸다.
        status = next(
            (c.response.status_code for c in causes if isinstance(c, httpx.HTTPStatusError)), None
        )
        embedding = any(isinstance(c, EmbeddingUnavailable) for c in causes)
        if status is not None or embedding:
            log.warning(
                "정리 작업의 외부 호출 실패",
                extra={
                    "event": "ai_process_dependency_failed",
                    "target": "embedding" if embedding else "unknown",
                    "status_code": status,
                    **error_fields(exc),
                },
            )
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
    reports: list[dict] = []
    result = await run_sync(
        lambda progress, cancel: _run(message, cfg, progress, cancel, reports), reporter
    )
    if trace := current_trace():
        # 결과 저장이 늦어지지 않도록, 그리고 취소된 작업을 기록하지 않도록 저장 뒤로 미룬다.
        trace.evaluations.extend(reports)
    return result
