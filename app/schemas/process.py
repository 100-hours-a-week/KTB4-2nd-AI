"""사진 정리 요청과 결과, API 명세 "사진 정리"와 "정리 result"에 1:1

요청 쪽 값 검증은 백엔드가 보장하므로 형식만 확인. 단 taken_at은 오프셋 있는 datetime만 받음,
naive datetime은 우리 계산을 깨뜨리기 때문
결과 쪽은 파이프라인 버그를 백엔드로 보내기 전에 잡기 위해 명세의 규칙을 검증하고,
시각은 항상 KST 오프셋으로 직렬화
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Self
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_serializer, model_validator

KST = ZoneInfo("Asia/Seoul")


def to_kst(value: datetime | None) -> datetime | None:
    """응답 시각은 계약대로 KST 오프셋. 파이프라인이 어떤 시간대로 만들어도 밖으로는 +09:00"""
    return value.astimezone(KST) if value is not None else None


# ---------- 요청, 백엔드 → api ----------


class RequestModel(BaseModel):
    """요청 모델 공통. 명세에 없는 필드가 와도 무시, 백엔드 필드 추가에 묶이지 않기 위해"""

    model_config = ConfigDict(extra="ignore")


class Period(RequestModel):
    start_date: date
    end_date: date


class Region(RequestModel):
    """여행 지역 좌표. v1 미사용, v2 이후 지역 검증용으로 계약에 유지"""

    latitude: float
    longitude: float


class ProcessAttachment(RequestModel):
    trip_attachment_id: int
    analyze_storage_key: str = Field(min_length=1, description="1024px JPEG 사본 키")
    taken_at: AwareDatetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    device_model: str | None = Field(default=None, description="EXIF Make + Model, 시계 보정용")


class ExistingPlace(RequestModel):
    """재처리 시에만, v2"""

    trip_place_id: int
    latitude: float
    longitude: float


class ProcessRequest(RequestModel):
    # 백엔드가 실행마다 새로 발급, 늦게 도착한 이전 실행의 응답을 걸러내는 용도. AI는 그대로 돌려줌
    execution_id: str | None = None
    trip_name: str = Field(min_length=1, max_length=10)
    period: Period
    regions: list[Region] = Field(min_length=1, max_length=10)
    # 200장 상한은 프론트와 백엔드가 보장. 여기 max_length는 CPU 30분을 지키는 방어선
    attachments: list[ProcessAttachment] = Field(min_length=1, max_length=200)
    existing_places: list[ExistingPlace] = Field(default_factory=list)


# ---------- 결과, api → 백엔드 ----------


class RegionOrigin(StrEnum):
    EXIF = "EXIF"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class Issue(StrEnum):
    UNCLEAR_LOCATION = "UNCLEAR_LOCATION"
    BLURRY = "BLURRY"
    DUPLICATED = "DUPLICATED"


class FailedReason(StrEnum):
    DECODE_FAILED = "DECODE_FAILED"


class PlaceAttachment(BaseModel):
    """폴더 안 사진. 좌표가 있어야 폴더에 들어가므로 region_origin은 EXIF 또는 INFERRED"""

    trip_attachment_id: int
    taken_at: AwareDatetime | None = None
    latitude: float
    longitude: float
    region_origin: RegionOrigin
    evaluation: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _origin_known(self) -> Self:
        if self.region_origin is RegionOrigin.UNKNOWN:
            raise ValueError("폴더 안 사진의 region_origin은 UNKNOWN일 수 없음")
        return self

    @field_serializer("taken_at", when_used="json")
    def _ser_taken_at(self, value: datetime | None) -> datetime | None:
        return to_kst(value)


class Place(BaseModel):
    place_id: str = Field(description="응답 안에서만 유효, p1 형식")
    matched_trip_place_id: int | None = Field(default=None, description="v2 재처리, v1은 null")
    latitude: float
    longitude: float
    first_taken_at: AwareDatetime | None = None
    last_taken_at: AwareDatetime | None = None
    representative_attachment_id: int
    attachments: list[PlaceAttachment] = Field(min_length=1)

    @model_validator(mode="after")
    def _representative_in_attachments(self) -> Self:
        ids = {a.trip_attachment_id for a in self.attachments}
        if self.representative_attachment_id not in ids:
            raise ValueError("representative_attachment_id가 attachments 안에 없음")
        return self

    @field_serializer("first_taken_at", "last_taken_at", when_used="json")
    def _ser_times(self, value: datetime | None) -> datetime | None:
        return to_kst(value)


class UnclassifiedAttachment(BaseModel):
    """폴더에 넣지 않은 사진. BLURRY와 DUPLICATED는 복구 대상 폴더가 필수"""

    trip_attachment_id: int
    issue: Issue
    place_id: str | None = None
    duplicate_of_attachment_id: int | None = None
    taken_at: AwareDatetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    region_origin: RegionOrigin
    evaluation: int | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def _issue_rules(self) -> Self:
        if self.issue in (Issue.BLURRY, Issue.DUPLICATED) and self.place_id is None:
            raise ValueError(f"{self.issue}는 place_id가 필수")
        if self.issue is Issue.DUPLICATED and self.duplicate_of_attachment_id is None:
            raise ValueError("DUPLICATED는 duplicate_of_attachment_id가 필수")
        return self

    @field_serializer("taken_at", when_used="json")
    def _ser_taken_at(self, value: datetime | None) -> datetime | None:
        return to_kst(value)


class FailedAttachment(BaseModel):
    """정상 경로에서는 빈 배열. 다운로드 실패는 작업 전체 FAILED"""

    trip_attachment_id: int
    reason: FailedReason


class ClockOffset(BaseModel):
    """기기별 적용한 시계 보정"""

    device_model: str
    days: int

    # 오프셋 결정하는데 근거가 된 사진 쌍의 수
    # (몇 쌍 이상일 때 보정을 적용하는지 thresholds.yaml에 정의)
    matched_pairs: int = Field(ge=0)


class ProcessResult(BaseModel):
    places: list[Place]
    unclassified: list[UnclassifiedAttachment]
    failed: list[FailedAttachment] = Field(default_factory=list)
    clock_offsets: list[ClockOffset] = Field(default_factory=list)

    @model_validator(mode="after")
    def _attachment_ids_unique(self) -> Self:
        """사진 하나는 폴더, 미분류, 실패 중 한 곳에만. 겹치면 파이프라인 버그"""
        ids = [a.trip_attachment_id for p in self.places for a in p.attachments]
        ids += [u.trip_attachment_id for u in self.unclassified]
        ids += [f.trip_attachment_id for f in self.failed]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"trip_attachment_id가 두 곳 이상에 있음: {duplicates}")
        return self
