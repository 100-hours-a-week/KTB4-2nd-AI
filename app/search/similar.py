"""기준 사진과 닮은 후보 사진. 저장된 이미지 벡터끼리 비교하므로 임베딩과 생성 호출이 없다."""

from app.schemas.similar import SimilarRequest, SimilarResponse
from app.search.context import CandidateSearcher


async def similar(request: SimilarRequest, vectors: CandidateSearcher) -> SimilarResponse:
    """후보 안에서 기준 사진과 코사인 유사도가 높은 순서로 limit개까지 돌려준다.

    기준 사진은 후보에 들어 있어도 결과에서 뺀다. 기준 사진의 벡터가 없으면
    (임베딩 전이거나 실패한 사진) 비교할 수 없으므로 빈 목록을 돌려준다.
    """
    source_id = request.trip_attachment_id
    empty = SimilarResponse(source_attachment_id=source_id, items=[])
    candidates = [i for i in request.candidate_attachment_ids if i != source_id]
    if not candidates:
        return empty
    source = (await vectors.fetch([source_id])).get(source_id)
    if source is None:
        return empty
    hits = await vectors.search(
        [float(value) for value in source], candidate_ids=candidates, limit=request.limit
    )
    return SimilarResponse(source_attachment_id=source_id, items=hits)
