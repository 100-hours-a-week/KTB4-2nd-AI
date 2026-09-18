"""워커 진입점, POST /internal/trips/{trip_id}/progress|result|failed. 같은 컨테이너의 워커만

응답은 전부 204. 워커 callback.py는 상태 코드만 보고 재시도 여부를 정함
"""

from fastapi import APIRouter, Depends

from app.api.deps import ServiceDep, localhost_only
from app.schemas.internal import FailedIn, ProgressIn, ResultIn
from app.schemas.task import ErrorBody

router = APIRouter(
    prefix="/internal/trips", tags=["internal"], dependencies=[Depends(localhost_only)]
)


@router.post("/{trip_id}/progress", status_code=204)
async def post_progress(trip_id: int, body: ProgressIn, service: ServiceDep) -> None:
    service.progress(trip_id, body.step, body.done, body.total)


@router.post("/{trip_id}/result", status_code=204)
async def post_result(trip_id: int, body: ResultIn, service: ServiceDep) -> None:
    await service.complete(trip_id, body.result)


@router.post("/{trip_id}/failed", status_code=204)
async def post_failed(trip_id: int, body: FailedIn, service: ServiceDep) -> None:
    # FailedIn은 ErrorBody의 자식이지만 저장은 부모 타입으로, 응답 직렬화가 선언 타입을 따르게
    error = ErrorBody(code=body.code, message=body.message, detail=body.detail)
    await service.fail(trip_id, error)
