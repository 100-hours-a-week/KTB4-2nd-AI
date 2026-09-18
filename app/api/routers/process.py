"""백엔드 진입점, POST GET DELETE /trips/{trip_id}/process

요청 검증, 서비스 호출, 응답 직렬화만. 도메인 규칙은 task_service.
전부 async def. def면 FastAPI가 스레드풀에서 돌려 task_service의 루프 스레드 전제가 깨짐
"""

from fastapi import APIRouter, Depends

from app.api.deps import ServiceDep, require_api_key
from app.schemas.process import ProcessRequest
from app.schemas.task import CancelResponse, TaskStatus, TaskStatusResponse

router = APIRouter(prefix="/trips", tags=["process"], dependencies=[Depends(require_api_key)])


@router.post("/{trip_id}/process", response_model=TaskStatusResponse)
async def submit_process(
    trip_id: int, body: ProcessRequest, service: ServiceDep
) -> TaskStatusResponse:
    """v1은 동기. 접수 뒤 종료 상태까지 연결을 유지, 최대 TASK_TIMEOUT_SEC. 202 즉시 응답은 v2

    submit은 워커의 접수 응답(202)까지 수 ms, wait_done은 결과 콜백까지 수 분.
    백엔드가 연결을 끊어도 이 핸들러는 취소되지 않고 결과는 store에 남아 GET으로 회수
    """
    await service.submit(trip_id, body)
    task = await service.wait_done(trip_id)
    return task.to_response()


@router.get("/{trip_id}/process", response_model=TaskStatusResponse)
async def get_process(trip_id: int, service: ServiceDep) -> TaskStatusResponse:
    """v1은 선택. 진행률 표시와 연결이 끊긴 뒤 결과 회수용"""
    return service.get(trip_id).to_response()


@router.delete("/{trip_id}/process", response_model=CancelResponse)
async def cancel_process(trip_id: int, service: ServiceDep) -> CancelResponse:
    """PROCESSING 취소는 워커가 다음 단계 경계에서 멈춘 뒤에야 CANCELED가 되므로
    즉시 응답하지 않고 진짜 종료 상태를 기다림. 취소 전에 끝나면 COMPLETED가 돌아감"""
    task = await service.cancel(trip_id)
    if task.status is TaskStatus.PROCESSING:
        task = await service.wait_done(trip_id)
    return CancelResponse(trip_id=trip_id, status=task.status)
