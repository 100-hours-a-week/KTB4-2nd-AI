"""검색 담당자의 진입점. 프롬프트·검색 컷오프·근거 선택을 이 경계 안에서 구현한다."""

import asyncio

import numpy as np

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
    ModelUnavailable,
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


# 형식 오류일 때 ai_query에 남길 모델 원출력 길이. 해석 출력은 수백 자라 원인 확인에 충분
MODEL_OUTPUT_LOG_CHARS = 500


def record_fields(**fields) -> None:
    """ai_query 한 줄에 붙일 값. 검색 요청 밖에서 부르면 기록하지 않음"""
    if trace := current_trace():
        trace.fields.update(fields)


def record_parse_result(response: ParseResponse) -> None:
    """LLM 해석 결과. BE가 search에 보내는 visual_query와 비교하는 기준

    intent는 query_service가 ai_query의 intent로 이미 남김
    """
    record_fields(
        date_from=response.date_from.isoformat() if response.date_from else None,
        date_to=response.date_to.isoformat() if response.date_to else None,
        region_names=response.region_names,
        visual_query=response.visual_query,
    )


async def parse(
    request: ParseRequest, context: SearchContext, options: SearchOptions
) -> ParseResponse:
    record_fields(query=request.query, reference_date=request.reference_date.isoformat())
    content = None
    try:
        async with asyncio.timeout(options.parse_timeout_sec):
            output = await context.vlm.chat(
                prompts.parse_messages(request),
                image_urls=[],
                max_tokens=prompts.PARSE_MAX_TOKENS,
                response_format=prompts.PARSE_RESPONSE_FORMAT,
                temperature=prompts.PARSE_TEMPERATURE,
            )
            content = output.content
            response = prompts.parse_output(content, request)
    except VLMUnavailable:
        record_fields(parse_fallback_reason="vlm_unavailable")
        return parse_fallback(request)
    except InvalidModelOutput:
        record_fields(
            parse_fallback_reason="invalid_output",
            model_output=(content or "")[:MODEL_OUTPUT_LOG_CHARS],
        )
        return parse_fallback(request)
    except TimeoutError:
        record_fields(parse_fallback_reason="timeout")
        return parse_fallback(request)
    record_parse_result(response)
    return response


def cutoff_threshold(ranked: list[SearchHit], options: SearchOptions) -> float:
    """절대 하한과 1위 점수 대비 비율 가운데 큰 값. ranked는 점수 내림차순이고 비어 있지 않음"""
    return max(options.min_score, ranked[0].score * options.score_ratio)


def apply_cutoff(ranked: list[SearchHit], options: SearchOptions) -> list[SearchHit]:
    """점수순 결과에서 절대 하한과 1위 점수 대비 비율을 모두 넘는 사진만

    절대 하한은 맞는 사진이 없는 질문을, 비율은 1위와 크게 벌어진 뒤쪽 사진을 거름
    """
    if not ranked:
        return []
    threshold = cutoff_threshold(ranked, options)
    return [hit for hit in ranked if hit.score >= threshold]


# 촬영 시각순 경로는 중복으로 빠진 자리를 다음 사진으로 채우려고 limit의 이 배수까지 비교
DEDUP_FILL_FACTOR = 2


def drop_duplicates(
    ids: list[int], vectors: dict[int, np.ndarray], threshold: float
) -> tuple[list[int], list[list]]:
    """앞 순서부터 보며 이미 남긴 사진과 코사인이 threshold 이상인 사진 제외

    ids는 점수순이나 촬영 시각순이라 같은 장면이면 앞 사진이 남음
    벡터가 없거나 노름이 0인 사진은 비교 없이 남김(못 믿으면 지우지 않음)
    removed는 [제외한 사진, 비슷한 남긴 사진, 코사인]
    """
    kept: list[int] = []
    units: list[tuple[int, np.ndarray]] = []
    removed: list[list] = []
    for photo_id in ids:
        vector = vectors.get(photo_id)
        norm = float(np.linalg.norm(vector)) if vector is not None else 0.0
        if norm == 0.0:
            kept.append(photo_id)
            continue
        unit = vector / norm
        similar = next(
            (
                (other, cosine)
                for other, other_unit in units
                if (cosine := float(unit @ other_unit)) >= threshold
            ),
            None,
        )
        if similar is None:
            kept.append(photo_id)
            units.append((photo_id, unit))
        else:
            removed.append([photo_id, similar[0], round(similar[1], 4)])
    return kept, removed


async def remove_duplicates(
    ids: list[int], context: SearchContext, options: SearchOptions
) -> list[int]:
    """반환 후보 안 비슷한 사진 제거(#192). 벡터 조회 실패는 제거 없이 그대로 반환

    중복 제거는 결과를 보기 좋게 하는 단계라 실패해도 검색 실패로 처리하지 않음
    """
    if len(ids) < 2:
        return ids
    try:
        async with asyncio.timeout(options.retrieval_timeout_sec):
            vectors = await context.vectors.fetch(ids)
    except (ModelUnavailable, TimeoutError):
        record_fields(dedup_result="fallback")
        return ids
    kept, removed = drop_duplicates(ids, vectors, options.dedup_threshold)
    record_fields(
        dedup_result="applied",
        dedup_threshold=options.dedup_threshold,
        dedup_removed=removed,
    )
    return kept


def record_result(order_mode: str, selected: list[SearchHit], **fields) -> None:
    """ai_query 로그에 남길 정렬 경로와 반환 사진 ID

    BE DB와 trip_attachment_id로 조인해 어떤 사진이 나왔는지 확인하는 용도
    QUESTION은 반환 사진이 곧 답변 근거 사진이라 근거 ID를 따로 남기지 않음
    """
    record_fields(
        order_mode=order_mode,
        result_ids=[hit.trip_attachment_id for hit in selected],
        **fields,
    )


def has_visual_query(request: SearchRequest) -> bool:
    """사진 내용 조건 여부

    BE는 해석 결과에 사진 내용이 없고 날짜와 지역도 없으면 원문 query로 바꿔 보냄
    질의 해석이 시간대 단어만 남긴 경우("저녁")도 내용 없음, 11/4 live T02, T06 대응
    """
    visual_query = (request.visual_query or "").strip()
    if not times.strip_time_words(visual_query).strip():
        return False
    return visual_query != request.query.strip()


def orders_by_taken_at(request: SearchRequest, words: tuple[str, ...]) -> bool:
    """임베딩과 컷오프 없이 촬영 시각순으로 반환할 질문 여부, SEARCH만

    visual_query null은 BE가 날짜나 지역 조건만 있는 해석 결과를 그대로 보낸 경우(#173)
    원문과 같은 visual_query는 해석 실패와 구분되지 않아 시간대 단어가 있을 때만 시각순
    QUESTION은 근거 사진을 골라야 해서 임베딩 순위 유지
    """
    if request.intent != "SEARCH":
        return False
    if not (request.visual_query or "").strip():
        return True
    return bool(words) and not has_visual_query(request)


def order_by_taken_at(candidates: list[SearchCandidate], limit: int) -> list[SearchHit]:
    """사진 내용 조건이 없는 질문의 촬영 시각순 결과

    BE가 score 내림차순으로 다시 정렬하므로 score는 순위에 따라 감소하는 값
    촬영 시각이 없는 사진은 맨 뒤, 지역만 있는 질문은 BE 후보에 섞일 수 있음
    """
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.taken_at is None,
            item.taken_at.timestamp() if item.taken_at else 0.0,
            item.trip_attachment_id,
        ),
    )[:limit]
    return rank_scores([item.trip_attachment_id for item in ordered])


def rank_scores(ids: list[int]) -> list[SearchHit]:
    """순위에 따라 감소하는 score, BE가 score 내림차순으로 다시 정렬해도 순서 유지"""
    return [
        SearchHit(trip_attachment_id=photo_id, score=(len(ids) - rank) / len(ids))
        for rank, photo_id in enumerate(ids)
    ]


async def search(
    request: SearchRequest, context: SearchContext, options: SearchOptions
) -> SearchResponse:
    """질문 원문, BE가 보낸 visual_query, 답변 문장을 ai_query에 남김

    answer_error는 query_service가 응답에서 따로 기록함
    """
    record_fields(query=request.query, visual_query=request.visual_query, limit=request.limit)
    response = await _search(request, context, options)
    if response.answer is not None:
        record_fields(answer=response.answer)
    return response


async def _search(
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
    # 날짜, 지역, 시간대만 있는 질문은 BE와 시간대 필터로 좁힌 후보 전체가 정답
    # 원문 임베딩 점수는 정답 여부와 무관해 컷오프가 정답을 버림
    if orders_by_taken_at(request, words):
        ordered_ids = [
            hit.trip_attachment_id
            for hit in order_by_taken_at(list(candidates.values()), limit * DEDUP_FILL_FACTOR)
        ]
        kept = (await remove_duplicates(ordered_ids, context, options))[:limit]
        ordered_by_time = rank_scores(kept)
        record_result("taken_at", ordered_by_time)
        return SearchResponse(attachments=ordered_by_time)

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
    selected_ids = [hit.trip_attachment_id for hit in selected]
    kept = set(await remove_duplicates(selected_ids, context, options))
    selected = [hit for hit in selected if hit.trip_attachment_id in kept]
    # 컷오프 전 순위와 점수, 적용된 기준값. 정답 사진이 기준값 바로 아래에서 잘렸는지 확인
    record_result(
        "vector",
        selected,
        ranked_scores=[[hit.trip_attachment_id, round(hit.score, 4)] for hit in ordered],
        cutoff_threshold=round(cutoff_threshold(ordered, options), 4) if ordered else None,
    )
    if request.intent == "SEARCH":
        return SearchResponse(attachments=selected)
    # 컷오프를 통과한 근거가 없으면 VLM을 부르지 않음, 관련 없는 사진으로 답을 지어내지 않게
    if not selected:
        return SearchResponse(attachments=[], answer_error="NO_EVIDENCE")

    content = None
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
            content = output.content
            answer = prompts.parse_answer(content)
        # 컷오프로 근거가 없을 때와 같은 NO_EVIDENCE라 로그에서는 answerable로 구분한다.
        if trace := current_trace():
            trace.fields["answerable"] = answer is not None
        if answer is None:
            return SearchResponse(attachments=selected, answer_error="NO_EVIDENCE")
        return SearchResponse(attachments=selected, answer=answer)
    except TimeoutError:
        return SearchResponse(attachments=selected, answer_error="TIMEOUT")
    except InvalidModelOutput:
        record_fields(model_output=(content or "")[:MODEL_OUTPUT_LOG_CHARS])
        return SearchResponse(attachments=selected, answer_error="VLM_UNAVAILABLE")
    except VLMUnavailable:
        return SearchResponse(attachments=selected, answer_error="VLM_UNAVAILABLE")
