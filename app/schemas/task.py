"""작업 상태 응답, API 명세 "정리 작업 상태 조회"와 "취소"에 1:1"""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from app.core.errors import ErrorCode
from app.schemas.process import ProcessResult


class TaskStatus(StrEnum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


TERMINAL_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED})


class ProcessStep(StrEnum):
    """파이프라인 7단계, 진행률 콜백과 current_step에 사용"""

    DOWNLOADING = "DOWNLOADING"
    EMBEDDING = "EMBEDDING"
    CLOCK = "CLOCK"
    LOCATING = "LOCATING"
    CLUSTERING = "CLUSTERING"
    FILTERING = "FILTERING"
    FINALIZING = "FINALIZING"


class Progress(BaseModel):
    """total은 항상 사진 수"""

    done: int = Field(ge=0)
    total: int = Field(ge=0)


class ErrorBody(BaseModel):
    """명세의 오류 본문 안쪽 {code, message, detail}"""

    code: ErrorCode
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ProcessAccepted(BaseModel):
    """POST 202 응답"""

    trip_id: int
    status: TaskStatus
    total_attachments: int


class TaskStatusResponse(BaseModel):
    """GET 200 응답. result는 COMPLETED일 때만, error는 FAILED일 때만"""

    trip_id: int
    status: TaskStatus
    progress: Progress
    current_step: ProcessStep | None = None
    result: ProcessResult | None = None
    error: ErrorBody | None = None

    @model_validator(mode="after")
    def _status_payload_match(self) -> Self:
        if (self.status is TaskStatus.COMPLETED) != (self.result is not None):
            raise ValueError("result는 COMPLETED일 때만, COMPLETED면 필수")
        if (self.status is TaskStatus.FAILED) != (self.error is not None):
            raise ValueError("error는 FAILED일 때만, FAILED면 필수")
        return self


class CancelResponse(BaseModel):
    """DELETE 200 응답"""

    trip_id: int
    status: TaskStatus
