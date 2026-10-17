"""HTTP 공통 책임. 검색 알고리즘은 app.search.pipeline에 둔다."""

import asyncio
import time

from app.core.errors import AppError, ErrorCode
from app.core.logging import error_fields, get_logger
from app.core.tracing import end_trace, start_trace
from app.schemas.query import ParseRequest, ParseResponse, SearchRequest, SearchResponse
from app.search import pipeline
from app.search.context import ModelUnavailable, SearchContext, SearchOptions

log = get_logger(__name__)


class QueryService:
    def __init__(
        self, context: SearchContext, options: SearchOptions | None = None, metrics=None
    ) -> None:
        self.context = context
        self.options = options or SearchOptions()
        self.metrics = metrics

    async def parse(self, request: ParseRequest) -> ParseResponse:
        return await self._run("parse", request)

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await self._run("search", request)

    async def _run(
        self, operation: str, request: ParseRequest | SearchRequest
    ) -> ParseResponse | SearchResponse:
        started = time.monotonic()
        trace, token = start_trace(operation)
        outcome = "error"
        error_code = None
        answer_error = None
        failure: dict = {}
        cancelled = False
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
                    answer_error = result.answer_error
                    if result.answer_error:
                        outcome = "partial"
                return result
        except asyncio.CancelledError:
            # 요청 자체가 취소된 경우. 아래의 단계 결과 보정을 하지 않는다.
            cancelled = True
            raise
        except AppError as exc:
            error_code = str(exc.code)
            raise
        except ModelUnavailable as exc:
            error_code = str(ErrorCode.MODEL_NOT_READY)
            raise AppError(ErrorCode.MODEL_NOT_READY) from exc
        except TimeoutError as exc:
            error_code = str(ErrorCode.TIMEOUT)
            raise AppError(ErrorCode.TIMEOUT, http_status=504) from exc
        except Exception as exc:
            # 원질문·사진 키·외부 예외 본문은 로그와 응답에 남기지 않는다.
            error_code = str(ErrorCode.INTERNAL_ERROR)
            failure = error_fields(exc)
            log.error(
                "검색 처리 실패",
                extra={"event": "ai_query_failed", "operation": operation, **failure},
            )
            raise AppError(ErrorCode.INTERNAL_ERROR) from exc
        finally:
            end_trace(token)
            if not cancelled:
                # 요청이 끝났는데 중단된 단계는 검색 파이프라인의 단계별 시간 제한에 걸린 것이다.
                for record in trace.stages:
                    if record.result == "cancelled":
                        record.result = "timeout"
                for key in [key for key in trace.calls if key.endswith("|cancelled")]:
                    trace.calls[key.removesuffix("cancelled") + "timeout"] += trace.calls.pop(key)
            duration = time.monotonic() - started
            if self.metrics is not None:
                await self.metrics.query(operation, outcome, duration, stages=trace.stages)
            # 대체 처리(fallback)에서 interpret 호출이 성공했다면 모델 출력 형식 오류다.
            log.info(
                "검색 요청 완료",
                extra={
                    "event": "ai_query",
                    "operation": operation,
                    "result": outcome,
                    "error_code": error_code,
                    "intent": intent,
                    "candidate_count": candidate_count,
                    "returned_count": returned_count,
                    "parsed": parsed,
                    "answer_error": answer_error,
                    "duration_ms": round(duration * 1000, 2),
                    "stages_ms": trace.stage_ms() or None,
                    "model_calls": trace.call_counts(),
                    "gpu_wait_ms": round(trace.gpu_wait_seconds * 1000, 2),
                    **trace.fields,
                    **failure,
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
