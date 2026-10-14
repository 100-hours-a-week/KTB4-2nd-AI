"""v2 메시지와 공유 작업 기록. v1 HTTP 계약은 기존 schemas에 보존한다."""

import hashlib
import json
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from app.schemas.process import Period, ProcessRequest, ProcessResult, RequestModel
from app.schemas.task import ErrorBody, Progress, TaskStatus

JobKind = Literal["process", "story"]
TerminalStatus = Literal["COMPLETED", "FAILED", "CANCELED"]


class ProcessMessage(ProcessRequest):
    type: Literal["process"]
    trip_id: int = Field(gt=0)
    execution_id: str = Field(min_length=1, max_length=128)
    trigger: Literal["TRIP_CREATED", "PHOTOS_ADDED"]

    @model_validator(mode="after")
    def original_input_only(self) -> Self:
        # v2는 기존 폴더 대응 대신 전체 사진을 원본 메타데이터로 다시 판정한다.
        self.existing_places = []
        return self


class StoryAttachment(RequestModel):
    trip_attachment_id: int = Field(gt=0)
    analyze_storage_key: str = Field(min_length=1)
    taken_at: AwareDatetime
    evaluation: int | None = Field(default=None, ge=0, le=100)


class StoryPlace(RequestModel):
    trip_place_id: int = Field(gt=0)
    place_name: str
    attachments: list[StoryAttachment] = Field(min_length=1, max_length=200)


class StoryMessage(RequestModel):
    type: Literal["story"]
    trip_id: int = Field(gt=0)
    execution_id: str = Field(min_length=1, max_length=128)
    trip_name: str = Field(min_length=1, max_length=10)
    period: Period
    mood: Literal["PLAIN", "EMOTIONAL", "HUMOROUS", "CALM", "LITERARY"]
    places: list[StoryPlace] = Field(min_length=1, max_length=10)


JobMessage = Annotated[ProcessMessage | StoryMessage, Field(discriminator="type")]
MESSAGE = TypeAdapter(JobMessage)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class StoryBlock(StrictModel):
    order_number: int = Field(ge=1)
    trip_place_id: int
    trip_attachment_id: int
    day_label: str = Field(max_length=40)
    detail_summary: str = Field(max_length=40)
    memo: str = Field(max_length=125)


class Verification(StrictModel):
    passed: bool
    retry_count: int = Field(ge=0)


class StoryResult(StrictModel):
    story_summary: str = Field(max_length=40)
    blocks: list[StoryBlock] = Field(min_length=1, max_length=30)
    verification: Verification


class JobStatusResponse(StrictModel):
    trip_id: int
    execution_id: str
    status: TaskStatus
    progress: Progress
    current_step: str | None


class JobResultEvent(StrictModel):
    type: Literal["process_result", "story_result"]
    trip_id: int
    execution_id: str
    status: TerminalStatus
    result: dict[str, Any] | None = None
    error: ErrorBody | None = None

    @model_validator(mode="after")
    def check_payload(self) -> Self:
        if (self.status == "COMPLETED") != (self.result is not None):
            raise ValueError("COMPLETED만 result를 가져야 함")
        if (self.status == "FAILED") != (self.error is not None):
            raise ValueError("FAILED만 error를 가져야 함")
        if self.result is not None:
            schema = ProcessResult if self.type == "process_result" else StoryResult
            self.result = schema.model_validate(self.result).model_dump(mode="json")
            # v1 ProcessResult에는 바깥 StrictModel의 유한수 검증이 전파되지 않는다.
            # NaN/Inf가 저장 시 null로 변해 다시 읽을 수 없는 기록을 만들지 않게 한다.
            json.dumps(self.result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return self


class JobRecord(StrictModel):
    trip_id: int
    kind: JobKind
    execution_id: str
    request: dict[str, Any]
    request_topic: str
    result_topic: str
    created_at: float
    status: TaskStatus = TaskStatus.QUEUED
    progress: Progress
    current_step: str | None = None
    owner: str | None = None
    heartbeat_at: float | None = None
    started_at: float | None = None
    cancel_requested: bool = False
    event: JobResultEvent | None = None
    publish_owner: str | None = None
    publish_until: float = 0
    published: bool = False

    @property
    def identity(self) -> str:
        return identity(self.trip_id, self.kind, self.execution_id)

    def response(self) -> JobStatusResponse:
        return JobStatusResponse(
            trip_id=self.trip_id,
            execution_id=self.execution_id,
            status=self.status,
            progress=self.progress,
            current_step=self.current_step,
        )


def identity(trip_id: int, kind: JobKind, execution_id: str) -> str:
    digest = hashlib.sha256(execution_id.encode()).hexdigest()
    return f"{kind}:{trip_id}:{digest}"
