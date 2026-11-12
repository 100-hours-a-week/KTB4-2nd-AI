"""유사 사진 후보 동기 HTTP. 검색 API와 같은 벡터 저장소를 읽는다."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path

from app.api.deps import require_api_key
from app.api.routers.query import QueryServiceDep
from app.schemas.similar import SimilarRequest, SimilarResponse

router = APIRouter(prefix="/trips", tags=["query"], dependencies=[Depends(require_api_key)])


@router.post("/{trip_id}/attachments/similar", response_model=SimilarResponse)
async def similar(
    trip_id: Annotated[int, Path(gt=0)], body: SimilarRequest, service: QueryServiceDep
) -> SimilarResponse:
    return await service.similar(trip_id, body)
