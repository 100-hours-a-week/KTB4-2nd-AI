"""v2 검색 API 스키마. BE 요청의 추가 필드는 무시하고 응답은 엄격하게 검증한다."""

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Intent = Literal["SEARCH", "QUESTION"]
AttachmentId = Annotated[int, Field(strict=True, gt=0)]
QueryText = Annotated[str, Field(min_length=2, max_length=100)]


class QueryRequestModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class QueryResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParseRequest(QueryRequestModel):
    query: QueryText
    reference_date: date


class ParseResponse(QueryResponseModel):
    intent: Intent
    date_from: date | None
    date_to: date | None
    region_names: list[str] | None
    visual_query: str | None
    parsed: bool

    @model_validator(mode="after")
    def ordered_dates(self) -> Self:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from은 date_to 이후일 수 없습니다.")
        return self


class SearchCandidate(QueryRequestModel):
    trip_attachment_id: AttachmentId
    analyze_storage_key: str = Field(min_length=1)
    taken_at: AwareDatetime | None
    place_name: str | None
    trip_name: str | None


class SearchRequest(QueryRequestModel):
    query: QueryText
    intent: Intent
    visual_query: str | None
    candidates: list[SearchCandidate]
    limit: int = Field(default=30, ge=1, le=50, strict=True)

    @model_validator(mode="after")
    def unique_candidates(self) -> Self:
        ids = [item.trip_attachment_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidates의 사진 ID는 중복될 수 없습니다.")
        return self


class SearchHit(QueryResponseModel):
    trip_attachment_id: AttachmentId
    score: float = Field(allow_inf_nan=False)


class SearchResponse(QueryResponseModel):
    attachments: list[SearchHit]
    answer: str | None = None
    answer_error: Literal["VLM_UNAVAILABLE", "TIMEOUT", "NO_EVIDENCE"] | None = None

    @model_validator(mode="after")
    def consistent_answer(self) -> Self:
        if self.answer is not None and (not self.answer.strip() or self.answer_error is not None):
            raise ValueError("빈 답변 또는 답변과 오류의 동시 반환은 허용하지 않습니다.")
        return self
