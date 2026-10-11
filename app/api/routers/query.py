"""동기 검색 HTTP. Kafka와 작업 상태 저장소를 거치지 않는다."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_api_key
from app.api.services.query_service import QueryService
from app.core.errors import AppError, ErrorCode
from app.schemas.query import ParseRequest, ParseResponse, SearchRequest, SearchResponse

router = APIRouter(prefix="/query", tags=["query"], dependencies=[Depends(require_api_key)])


def get_query_service(request: Request) -> QueryService:
    service = request.app.state.query_service
    if service is None:
        raise AppError(ErrorCode.MODEL_NOT_READY, message="검색 의존성이 연결되지 않았습니다.")
    return service


QueryServiceDep = Annotated[QueryService, Depends(get_query_service)]


@router.post("/parse", response_model=ParseResponse)
async def parse(request: ParseRequest, service: QueryServiceDep) -> ParseResponse:
    return await service.parse(request)


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, service: QueryServiceDep) -> SearchResponse:
    return await service.search(request)
