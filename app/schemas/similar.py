"""유사 사진 후보 API 스키마. BE 요청의 추가 필드는 무시하고 응답은 엄격하게 검증한다."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.query import AttachmentId, SearchHit


class SimilarRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    trip_attachment_id: AttachmentId
    candidate_attachment_ids: list[AttachmentId]
    limit: int = Field(default=10, ge=1, le=50, strict=True)

    @model_validator(mode="after")
    def unique_candidates(self) -> Self:
        ids = self.candidate_attachment_ids
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_attachment_ids의 사진 ID는 중복될 수 없습니다.")
        return self


class SimilarResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_attachment_id: AttachmentId
    items: list[SearchHit]
