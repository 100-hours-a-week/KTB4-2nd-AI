"""api가 부르는 진입점. POST /jobs, POST /jobs/{trip_id}/cancel, GET /ready

상태 코드가 api WorkerClient와의 계약. 202 접수, 409 실행 중, 503 모델 미적재
"""

from fastapi import APIRouter, Request

from app.core.errors import AppError, ErrorCode
from app.schemas.internal import JobAccepted, JobRequest, ReadyResponse

router = APIRouter(tags=["jobs"])


@router.post("/jobs", status_code=202, response_model=JobAccepted)
async def submit_job(body: JobRequest, request: Request) -> JobAccepted:
    state = request.app.state
    if not state.model_loaded:
        raise AppError(ErrorCode.MODEL_NOT_READY)
    if not state.runner.submit(body.trip_id, body.request):
        raise AppError(ErrorCode.WORKER_BUSY)
    return JobAccepted(trip_id=body.trip_id)


@router.post("/jobs/{trip_id}/cancel", status_code=204)
async def cancel_job(trip_id: int, request: Request) -> None:
    """모르는 trip이면 404. api WorkerClient는 무시함, 이미 끝난 작업"""
    if not request.app.state.runner.cancel(trip_id):
        raise AppError(ErrorCode.TASK_NOT_FOUND)


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    state = request.app.state
    return ReadyResponse(
        model_loaded=state.model_loaded,
        model_name=state.model_name,
        busy=state.runner.busy if state.model_loaded else False,
    )
