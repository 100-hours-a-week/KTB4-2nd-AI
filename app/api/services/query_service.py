"""HTTP 공통 책임. 검색 알고리즘은 app.search.pipeline에 둔다."""

import asyncio
import time

from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.schemas.query import ParseRequest, ParseResponse, SearchRequest, SearchResponse
from app.search import pipeline
from app.search.context import ModelUnavailable, SearchContext, SearchOptions

log = get_logger(__name__)


class QueryService:
    def __init__(self, context: SearchContext, options: SearchOptions | None = None) -> None:
        self.context = context
        self.options = options or SearchOptions()

    async def parse(self, request: ParseRequest) -> ParseResponse:
        return await self._run("parse", request)

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await self._run("search", request)

    async def _run(
        self, operation: str, request: ParseRequest | SearchRequest
    ) -> ParseResponse | SearchResponse:
        started = time.monotonic()
        outcome = "error"
        intent = request.intent if isinstance(request, SearchRequest) else None
        candidate_count = len(request.candidates) if isinstance(request, SearchRequest) else None
        returned_count = None
        parsed = None
        try:
            async with asyncio.timeout(self.options.request_timeout_sec):
                if isinstance(request, ParseRequest):
                    result = ParseResponse.model_validate(
                        await pipeline.parse(request, self.context, self.options)
                    )
                else:
                    result = SearchResponse.model_validate(
                        await pipeline.search(request, self.context, self.options)
                    )
                    self._validate_result(request, result)
                outcome = "success"
                if isinstance(result, ParseResponse):
                    intent = result.intent
                    parsed = result.parsed
                    if not parsed:
                        outcome = "fallback"
                else:
                    returned_count = len(result.attachments)
                    if result.answer_error:
                        outcome = "partial"
                return result
        except AppError:
            raise
        except ModelUnavailable as exc:
            raise AppError(ErrorCode.MODEL_NOT_READY) from exc
        except TimeoutError as exc:
            raise AppError(ErrorCode.TIMEOUT, http_status=504) from exc
        except Exception as exc:
            # 원질문·사진 키·외부 예외 본문은 로그와 응답에 남기지 않는다.
            log.error("검색 처리 실패", extra={"error_type": type(exc).__name__})
            raise AppError(ErrorCode.INTERNAL_ERROR) from exc
        finally:
            log.info(
                "검색 요청 완료",
                extra={
                    "event": "ai_query",
                    "operation": operation,
                    "result": outcome,
                    "intent": intent,
                    "candidate_count": candidate_count,
                    "returned_count": returned_count,
                    "parsed": parsed,
                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                },
            )

    def _validate_result(self, request: SearchRequest, response: SearchResponse) -> None:
        candidates = {item.trip_attachment_id for item in request.candidates}
        ids = [item.trip_attachment_id for item in response.attachments]
        limit = request.limit if request.intent == "SEARCH" else self.options.question_k
        if not set(ids) <= candidates or len(set(ids)) != len(ids) or len(ids) > limit:
            raise ValueError("검색 결과의 후보 범위·중복·개수 규약 위반")
        if request.intent == "SEARCH" and (response.answer or response.answer_error):
            raise ValueError("개발 초안에서 SEARCH는 답변을 반환하지 않습니다.")
