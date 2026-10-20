"""검색 담당자의 진입점. 프롬프트·검색 컷오프·근거 선택을 이 경계 안에서 구현한다."""

import asyncio

from app.core.tracing import current_trace
from app.infra.vlm import VLMUnavailable
from app.schemas.query import (
    ParseRequest,
    ParseResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.search import prompts
from app.search.context import (
    Evidence,
    InvalidModelOutput,
    SearchContext,
    SearchOptions,
)


def parse_fallback(request: ParseRequest) -> ParseResponse:
    return ParseResponse(
        intent="SEARCH",
        date_from=None,
        date_to=None,
        region_names=None,
        visual_query=request.query,
        parsed=False,
    )


async def parse(
    request: ParseRequest, context: SearchContext, options: SearchOptions
) -> ParseResponse:
    try:
        async with asyncio.timeout(options.parse_timeout_sec):
            output = await context.vlm.chat(
                prompts.parse_messages(request),
                image_urls=[],
                max_tokens=prompts.PARSE_MAX_TOKENS,
                response_format=prompts.PARSE_RESPONSE_FORMAT,
                temperature=prompts.PARSE_TEMPERATURE,
            )
            return prompts.parse_output(output, request)
    except (VLMUnavailable, InvalidModelOutput, TimeoutError):
        return parse_fallback(request)


def apply_cutoff(ranked: list[SearchHit], options: SearchOptions) -> list[SearchHit]:
    """점수순 결과에서 절대 하한과 1위 점수 대비 비율을 모두 넘는 사진만

    절대 하한은 맞는 사진이 없는 질문을, 비율은 1위와 크게 벌어진 뒤쪽 사진을 거름
    """
    if not ranked:
        return []
    threshold = max(options.min_score, ranked[0].score * options.score_ratio)
    return [hit for hit in ranked if hit.score >= threshold]


async def search(
    request: SearchRequest, context: SearchContext, options: SearchOptions
) -> SearchResponse:
    if not request.candidates:
        return SearchResponse(
            attachments=[], answer_error="NO_EVIDENCE" if request.intent == "QUESTION" else None
        )

    candidates = {item.trip_attachment_id: item for item in request.candidates}
    limit = request.limit if request.intent == "SEARCH" else options.question_k
    async with asyncio.timeout(options.retrieval_timeout_sec):
        vector = await context.embedder.encode_text(
            request.visual_query if request.visual_query is not None else request.query
        )
        hits = await context.vectors.search(vector, candidate_ids=list(candidates), limit=limit)

    # 후보 밖 사진은 이미지 접근 및 VLM 전달 전에 차단한다. 실제 질의 필터도 필수다.
    if any(hit.trip_attachment_id not in candidates for hit in hits):
        raise ValueError("벡터 검색에서 후보 밖 사진이 반환되었습니다.")
    ranked = sorted(hits, key=lambda hit: (-hit.score, hit.trip_attachment_id))
    unique = {hit.trip_attachment_id: hit for hit in reversed(ranked)}
    ordered = sorted(unique.values(), key=lambda hit: (-hit.score, hit.trip_attachment_id))
    selected = apply_cutoff(ordered, options)[:limit]
    if request.intent == "SEARCH":
        return SearchResponse(attachments=selected)
    # 컷오프를 통과한 근거가 없으면 VLM을 부르지 않음, 관련 없는 사진으로 답을 지어내지 않게
    if not selected:
        return SearchResponse(attachments=[], answer_error="NO_EVIDENCE")

    try:
        async with asyncio.timeout(options.answer_timeout_sec):
            evidence = [
                Evidence(
                    candidate=candidates[hit.trip_attachment_id],
                    image_url=await context.images.resolve(
                        candidates[hit.trip_attachment_id].analyze_storage_key
                    ),
                )
                for hit in selected
            ]
            output = await context.vlm.chat(
                prompts.answer_messages(request.query, evidence),
                image_urls=[item.image_url for item in evidence],
                max_tokens=prompts.ANSWER_MAX_TOKENS,
                response_format=prompts.ANSWER_RESPONSE_FORMAT,
                temperature=prompts.ANSWER_TEMPERATURE,
            )
            answer = prompts.parse_answer(output)
        # 컷오프로 근거가 없을 때와 같은 NO_EVIDENCE라 로그에서는 answerable로 구분한다.
        if trace := current_trace():
            trace.fields["answerable"] = answer is not None
        if answer is None:
            return SearchResponse(attachments=selected, answer_error="NO_EVIDENCE")
        return SearchResponse(attachments=selected, answer=answer)
    except TimeoutError:
        return SearchResponse(attachments=selected, answer_error="TIMEOUT")
    except (VLMUnavailable, InvalidModelOutput):
        return SearchResponse(attachments=selected, answer_error="VLM_UNAVAILABLE")
