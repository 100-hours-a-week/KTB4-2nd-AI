"""헬스체크 응답, API 명세 "헬스체크"에 1:1. ok는 200, starting과 degraded는 503"""

from enum import StrEnum

from pydantic import BaseModel, Field


class HealthStatus(StrEnum):
    OK = "ok"
    STARTING = "starting"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    status: HealthStatus
    model_loaded: bool
    model_name: str
    active_tasks: int = Field(ge=0)
    queued: int = Field(ge=0)
    idle_seconds: int = Field(
        ge=0, description="마지막 작업 종료 후 경과, 작업이 없었으면 기동 후 경과"
    )
