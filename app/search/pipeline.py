"""검색 담당자의 진입점. 프롬프트·검색 컷오프·근거 선택을 이 경계 안에서 구현한다."""

import asyncio

from app.core.tracing import current_trace
from app.infra.vlm import VLMUnavailable
from app.schemas.query import (
    ParseRequest,
    ParseResponse,
    SearchCandidate,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.search import prompts, times
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
            return prompts.parse_output(output.content, request)
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


def has_visual_query(request: SearchRequest) -> bool:
    """사진 내용 조건 여부

    BE는 해석 결과에 사진 내용이 없으면 원문 query로 바꿔 보내고, 채점 스크립트는 null을 그대로 보냄
    질의 해석이 시간대 단어만 남긴 경우("저녁")도 내용 없음, 11/4 live T02, T06 대응
    """
    visual_query = (request.visual_query or "").strip()
    if not times.strip_time_words(visual_query).strip():
        return False
    return visual_query != request.query.strip()


def order_by_taken_at(candidates: list[SearchCandidate], limit: int) -> list[SearchHit]:
    """사진 내용 조건이 없는 질문의 촬영 시각순 결과

    BE가 score 내림차순으로 다시 정렬하므로 score는 순위에 따라 감소하는 값
    """
    ordered = sorted(candidates, key=lambda item: (item.taken_at, item.trip_attachment_id))[:limit]
    return [
        SearchHit(
            trip_attachment_id=item.trip_attachment_id, score=(len(ordered) - rank) / len(ordered)
        )
        for rank, item in enumerate(ordered)
    ]


async def search(
    request: SearchRequest, context: SearchContext, options: SearchOptions
) -> SearchResponse:
    candidates = {item.trip_attachment_id: item for item in request.candidates}
    words = times.time_words(request.query)
    if words:
        candidates = {
            key: item
            for key, item in candidates.items()
            if times.in_time_windows(item.taken_at, words)
        }
        if trace := current_trace():
            trace.fields["time_words"] = list(words)
            trace.fields["time_candidate_count"] = len(candidates)
    if not candidates:
        return SearchResponse(
            attachments=[], answer_error="NO_EVIDENCE" if request.intent == "QUESTION" else None
        )

    limit = request.limit if request.intent == "SEARCH" else options.question_k
    # 시간대만 있는 질문은 원문 임베딩 점수가 정답 여부와 무관해 컷오프가 정답을 버림
    if words and request.intent == "SEARCH" and not has_visual_query(request):
        return SearchResponse(attachments=order_by_taken_at(list(candidates.values()), limit))

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
            answer = prompts.parse_answer(output.content)
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
