"""요청의 원본 정보와 사진별 처리 결과. 외부 API 모델과 분리한 내부 상태."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from app.schemas.process import Issue, ProcessRequest, RegionOrigin


class TimeStatus(StrEnum):
    PENDING = "PENDING"  # CLOCK 단계에서 아직 판정하지 않음
    ORIGINAL = "ORIGINAL"  # 원본 시각을 신뢰해 사용
    CORRECTED = "CORRECTED"  # 보정한 시각을 사용
    UNKNOWN = "UNKNOWN"  # 판정했지만 사용할 시각을 확보하지 못함


@dataclass(frozen=True)
class PhotoSource:
    """요청에서 복사한 원본 값. 처리 과정에서 필드를 덮어쓰지 않는다."""

    trip_attachment_id: int
    analyze_storage_key: str
    taken_at: datetime | None
    latitude: float | None
    longitude: float | None
    device_model: str | None

    @property
    def has_gps(self) -> bool:
        """위도와 경도를 모두 받았는지 확인한다. 좌표 0도 유효하다."""
        return self.latitude is not None and self.longitude is not None


@dataclass
class PhotoState:
    """단계별 결과를 채우는 객체. source는 보존하고 처리 필드만 갱신한다.

    None은 아직 결과가 없다는 뜻이다. issue가 None이어도 분류 성공은 아니다.
    필드 사이의 일관성은 값을 채우는 단계에서 책임진다.
    """

    source: PhotoSource
    taken_at: datetime | None = None
    time_status: TimeStatus = TimeStatus.PENDING
    latitude: float | None = None
    longitude: float | None = None
    region_origin: RegionOrigin = RegionOrigin.UNKNOWN
    embedding: NDArray[np.float32] | None = field(default=None, repr=False, compare=False)
    place_id: str | None = None
    issue: Issue | None = None
    duplicate_of_attachment_id: int | None = None
    evaluation: int | None = None


def build_photo_states(request: ProcessRequest) -> list[PhotoState]:
    """입력 순서대로 독립적인 상태를 만든다. 요청을 수정하거나 분류하지 않는다.

    원본 시각이 있어도 사용할 시각은 CLOCK 단계가 판정할 때까지 비워 둔다.
    GPS의 한쪽만 있으면 원본에는 보존하고 사용할 좌표 쌍은 비워 둔다.
    """
    photos = []
    for attachment in request.attachments:
        source = PhotoSource(
            trip_attachment_id=attachment.trip_attachment_id,
            analyze_storage_key=attachment.analyze_storage_key,
            taken_at=attachment.taken_at,
            latitude=attachment.latitude,
            longitude=attachment.longitude,
            device_model=attachment.device_model,
        )
        photo = PhotoState(source=source)
        if source.has_gps:
            photo.latitude = source.latitude
            photo.longitude = source.longitude
            photo.region_origin = RegionOrigin.EXIF
        photos.append(photo)
    return photos
