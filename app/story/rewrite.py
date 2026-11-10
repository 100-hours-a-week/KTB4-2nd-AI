"""분위기 변경 미리보기. VLM 호출 한 번으로 요약, 캡션, 서술 문장을 새 분위기로 다시 쓴다.

요약이 20자를 넘으면 다시 쓰지 않고 출력 규칙 위반으로 실패 처리한다(결정 47번).

작업 큐와 작업 상태 저장소를 거치지 않는 동기 처리이며 결과는 저장하지 않는다.
같은 여행의 요청은 복제본들이 공유하는 Redis 잠금으로 하나만 처리한다.
"""

import asyncio
import time
import uuid

from pydantic import ValidationError
from redis.exceptions import RedisError

from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger, validation_fields
from app.infra.vlm import VLMUnavailable
from app.schemas.story_rewrite import RewriteBlockResponse, RewriteRequest, RewriteResponse
from app.story.pipeline import day_label
from app.story.rewrite_prompts import SUMMARY_LIMIT, RewriteDraft, messages, response_format

log = get_logger(__name__)
# 잠금은 처리 상한이 지나면 스스로 풀린다. 처리가 끝나면 바로 지운다.
LOCK_GRACE_SEC = 30
# 종료할 때 진행 중인 처리를 기다리는 상한. 컨테이너 종료 유예(120초) 안에서 잠금을 풀어야 한다.
SHUTDOWN_WAIT_SEC = 30


class InvalidRewrite(Exception):
    def __init__(self, rule: str, **fields):
        self.rule, self.fields = rule, fields


def parse(model, raw: str, rule: str):
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        raise InvalidRewrite(
            rule, validation_errors=validation_fields(exc.errors()), output_chars=len(raw)
        ) from None


def finalize(raw: str, request: RewriteRequest) -> RewriteResponse:
    draft = parse(RewriteDraft, raw, "draft_schema")
    blocks = written_blocks(draft, request)
    summary = draft.story_summary
    # 스키마는 40자까지 받고 계약의 20자는 여기서 검사한다. 넘으면 다시 쓰지 않는다(결정 47번).
    if len(summary) > SUMMARY_LIMIT:
        raise InvalidRewrite("summary_too_long", summary_chars=len(summary))
    return RewriteResponse(story_summary=summary, blocks=blocks)


def written_blocks(draft: RewriteDraft, request: RewriteRequest) -> list[RewriteBlockResponse]:
    ids = [block.trip_attachment_id for block in draft.blocks]
    if len(set(ids)) != len(ids):
        raise InvalidRewrite("duplicate_photo", blocks=len(ids), unique_photos=len(set(ids)))
    wanted = [block.trip_attachment_id for block in request.blocks]
    if set(ids) != set(wanted):
        raise InvalidRewrite(
            "photo_set_mismatch",
            missing_photos=len(set(wanted) - set(ids)),
            unexpected_photos=len(set(ids) - set(wanted)),
        )
    drafts = {block.trip_attachment_id: block for block in draft.blocks}
    blocks = []
    # 응답은 요청과 같은 순서다. 모델이 늘어놓은 순서는 쓰지 않는다.
    for block in request.blocks:
        written = drafts[block.trip_attachment_id]
        if block.memo is not None and written.memo is None:
            raise InvalidRewrite("memo_missing")
        blocks.append(
            RewriteBlockResponse(
                order_number=block.order_number,
                trip_attachment_id=block.trip_attachment_id,
                day_label=block.day_label
                if block.day_label is not None
                else day_label(block.taken_at, request.period.start_date, block.place_name),
                detail_summary=written.detail_summary,
                # 비어 있던 서술 문장은 채우지 않는다.
                memo=None if block.memo is None else written.memo,
            )
        )
    return blocks


class RewriteService:
    def __init__(self, vlm, urls, redis, prefix: str, *, timeout_sec: float, max_tokens: int):
        self.vlm, self.urls, self.redis, self.prefix = vlm, urls, redis, prefix
        self.timeout_sec, self.max_tokens = timeout_sec, max_tokens
        self._tasks: set[asyncio.Task] = set()

    async def rewrite(self, trip_id: int, request: RewriteRequest) -> RewriteResponse:
        key = f"{self.prefix}story-rewrite:{trip_id}"
        token = uuid.uuid4().hex
        if not await self.redis.set(key, token, nx=True, ex=int(self.timeout_sec) + LOCK_GRACE_SEC):
            raise AppError(ErrorCode.DUPLICATE_TASK)
        # BE가 연결을 끊어도 처리와 잠금 해제는 끝까지 이어진다(계약).
        task = asyncio.create_task(self._locked(key, token, trip_id, request))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return await asyncio.shield(task)

    async def _locked(self, key, token, trip_id, request) -> RewriteResponse:
        started = time.monotonic()
        error_code, rule = None, None
        try:
            async with asyncio.timeout(self.timeout_sec):
                return await self._run(trip_id, request)
        except TimeoutError:
            error_code = ErrorCode.TIMEOUT
            raise AppError(ErrorCode.TIMEOUT, http_status=504) from None
        except VLMUnavailable:
            # 호출 한도 초과, 사진 URL 발급 실패, 생성 결과 잘림을 포함한다(계약).
            error_code = ErrorCode.VLM_UNAVAILABLE
            raise AppError(ErrorCode.VLM_UNAVAILABLE, http_status=503) from None
        except InvalidRewrite as exc:
            error_code, rule = ErrorCode.INTERNAL_ERROR, exc.rule
            log.warning(
                "분위기 변경 결과가 출력 규칙을 만족하지 못했습니다.",
                extra={
                    "event": "ai_story_rewrite_output_invalid",
                    "result": "failed",
                    "rule": exc.rule,
                    **exc.fields,
                },
            )
            raise AppError(
                ErrorCode.INTERNAL_ERROR,
                message="분위기 변경 결과가 출력 규칙을 만족하지 못했습니다.",
            ) from None
        except AppError as exc:
            error_code = exc.code
            raise
        except Exception as exc:
            error_code = ErrorCode.INTERNAL_ERROR
            log.error(
                "분위기 변경 처리 실패",
                extra={"event": "ai_story_rewrite_failed", "error_type": type(exc).__name__},
            )
            raise AppError(ErrorCode.INTERNAL_ERROR) from None
        finally:
            # 생성된 글 본문과 사진 키는 남기지 않는다.
            log.info(
                "분위기 변경 처리",
                extra={
                    "event": "ai_story_rewrite",
                    "result": "failed" if error_code else "success",
                    "error_code": str(error_code) if error_code else None,
                    "rule": rule,
                    "trip_id": trip_id,
                    "mood": request.mood,
                    "blocks": len(request.blocks),
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                },
            )
            await self._release(key, token)

    async def _run(self, trip_id: int, request: RewriteRequest) -> RewriteResponse:
        urls = [await self.urls.resolve(block.analyze_storage_key) for block in request.blocks]
        turn = await self.vlm.chat(
            messages(request, trip_id),
            urls,
            self.max_tokens,
            response_format=response_format([b.trip_attachment_id for b in request.blocks]),
            # 블록 30개는 RunPod 프록시의 무응답 상한(약 100초)을 넘을 수 있다.
            stream=True,
        )
        return finalize(turn.content, request)

    async def _release(self, key, token) -> None:
        try:
            async with asyncio.timeout(5):
                if await self.redis.get(key) == token:
                    await self.redis.delete(key)
        except (TimeoutError, RedisError):
            log.warning(
                "분위기 변경 잠금 해제 지연", extra={"event": "ai_story_rewrite_unlock_failed"}
            )

    async def aclose(self, wait_sec: float = SHUTDOWN_WAIT_SEC) -> None:
        # 기다려도 끝나지 않으면 취소해, 강제 종료 전에 finally에서 잠금을 푼다.
        tasks = list(self._tasks)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=wait_sec)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
