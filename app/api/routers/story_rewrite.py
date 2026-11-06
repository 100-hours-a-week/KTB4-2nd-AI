"""분위기 변경 미리보기 동기 HTTP. 작업 큐와 작업 상태 저장소를 거치지 않는다."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request

from app.api.deps import require_api_key
from app.core.errors import AppError, ErrorCode
from app.schemas.story_rewrite import RewriteRequest, RewriteResponse
from app.story.rewrite import RewriteService

router = APIRouter(prefix="/trips", tags=["story"], dependencies=[Depends(require_api_key)])


def get_rewrite_service(request: Request) -> RewriteService:
    service = getattr(request.app.state, "rewrite_service", None)
    if service is None:
        raise AppError(
            ErrorCode.VLM_UNAVAILABLE,
            message="분위기 변경 의존성이 연결되지 않았습니다.",
            http_status=503,
        )
    return service


RewriteServiceDep = Annotated[RewriteService, Depends(get_rewrite_service)]


@router.post("/{trip_id}/story/rewrite", response_model=RewriteResponse)
async def rewrite(
    trip_id: Annotated[int, Path(gt=0)], body: RewriteRequest, service: RewriteServiceDep
) -> RewriteResponse:
    return await service.rewrite(trip_id, body)
