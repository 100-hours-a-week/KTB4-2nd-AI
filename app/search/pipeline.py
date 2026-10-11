"""검색 담당자의 진입점. 프롬프트·검색 컷오프·근거 선택을 이 경계 안에서 구현한다."""

import asyncio

from app.schemas.query import ParseRequest, ParseResponse, SearchRequest, SearchResponse
from app.search.context import (
    Evidence,
    InvalidModelOutput,
    SearchContext,
    SearchOptions,
    VLMUnavailable,
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
            return await context.vlm.interpret(request)
    except (VLMUnavailable, InvalidModelOutput, TimeoutError):
        return parse_fallback(request)


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
    selected = sorted(unique.values(), key=lambda hit: (-hit.score, hit.trip_attachment_id))[:limit]
    if request.intent == "SEARCH":
        return SearchResponse(attachments=selected)
    if not selected:
        return SearchResponse(attachments=[], answer_error="NO_EVIDENCE")

    # TODO(검색): 실제 검색 컷오프와 근거 선택은 평가 후 구현. 현재는 반환 상한만 적용.
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
            answer = await context.vlm.answer(request.query, evidence)
            if not answer.strip():
                raise InvalidModelOutput("빈 답변")
            return SearchResponse(attachments=selected, answer=answer)
    except TimeoutError:
        return SearchResponse(attachments=selected, answer_error="TIMEOUT")
    except (VLMUnavailable, InvalidModelOutput):
        return SearchResponse(attachments=selected, answer_error="VLM_UNAVAILABLE")
