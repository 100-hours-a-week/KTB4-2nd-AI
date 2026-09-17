"""api ↔ worker 내부 계약. 백엔드는 보지 않음

api → worker: JobRequest(POST /jobs), worker의 응답 JobAccepted와 ReadyResponse
worker → api: ProgressIn, ResultIn, FailedIn(POST /internal/trips/{trip_id}/progress|result|failed)
"""

from pydantic import BaseModel, Field

from app.schemas.process import ProcessRequest, ProcessResult
from app.schemas.task import ErrorBody, ProcessStep


class JobRequest(BaseModel):
    """api가 워커에 위임할 때. 요청 본문을 그대로 실어 워커가 task_store 없이 실행"""

    trip_id: int
    request: ProcessRequest


class JobAccepted(BaseModel):
    """워커 POST /jobs 202 응답. 바쁘면 202 대신 409"""

    trip_id: int


class ReadyResponse(BaseModel):
    """워커 GET /ready. api의 /health가 합산"""

    model_loaded: bool
    model_name: str
    busy: bool


class ProgressIn(BaseModel):
    step: ProcessStep
    done: int = Field(ge=0)
    total: int = Field(ge=0)


class ResultIn(BaseModel):
    result: ProcessResult


class FailedIn(ErrorBody):
    """실패 콜백 본문, 오류 본문과 같은 모양. code가 CANCELED면 api가 상태를 CANCELED로"""
