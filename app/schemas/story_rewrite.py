"""분위기 변경 미리보기 API 스키마. BE 요청의 추가 필드는 무시하고 응답은 엄격하게 검증한다."""

from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.schemas.process import Period

# 분위기 변경은 세 값만 받는다(결정 46번). 스토리 생성은 전환 기간에 다섯 값을 받는다.
RewriteMood = Literal["PLAIN", "HUMOROUS", "LITERARY"]


class RewriteRequestModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RewriteBlockRequest(RewriteRequestModel):
    order_number: int = Field(ge=1)
    trip_attachment_id: int = Field(gt=0)
    place_name: str = Field(min_length=1)
    taken_at: AwareDatetime
    analyze_storage_key: str = Field(min_length=1, max_length=1024)
    # 새로 추가한 사진은 셋 다 null이다. memo는 비어 있어도 null이다.
    day_label: str | None = Field(max_length=40)
    detail_summary: str | None = Field(max_length=40)
    memo: str | None = Field(max_length=125)


class RewriteRequest(RewriteRequestModel):
    trip_name: str = Field(min_length=1, max_length=10)
    period: Period
    mood: RewriteMood
    blocks: list[RewriteBlockRequest] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def ordered_blocks(self) -> Self:
        numbers = [block.order_number for block in self.blocks]
        if any(a >= b for a, b in zip(numbers, numbers[1:], strict=False)):
            raise ValueError("blocks는 order_number 순이어야 합니다.")
        ids = [block.trip_attachment_id for block in self.blocks]
        if len(set(ids)) != len(ids):
            raise ValueError("사진 ID가 중복되었습니다.")
        if self.period.start_date > self.period.end_date:
            raise ValueError("여행 기간이 역전되었습니다.")
        return self


class RewriteBlockResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_number: int
    trip_attachment_id: int
    day_label: str = Field(max_length=40)
    detail_summary: str = Field(min_length=1, max_length=40)
    memo: str | None = Field(min_length=1, max_length=125)


class RewriteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    story_summary: str = Field(min_length=1, max_length=20)
    blocks: list[RewriteBlockResponse] = Field(min_length=1, max_length=30)
