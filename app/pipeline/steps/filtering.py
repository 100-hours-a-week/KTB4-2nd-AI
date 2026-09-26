"""FILTERING: 장소별 흐림과 중복 판정. 원본과 장소 연결은 보존한다."""

from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime

import cv2
import imagehash
import numpy as np
from PIL import Image

from app.core.errors import PipelineError
from app.pipeline.state import PhotoState, TimeStatus
from app.schemas.process import Issue


def _measure(image: Image.Image) -> tuple[float, imagehash.ImageHash]:
    """긴 변 최대 300px에서 분산을 측정하고, 원래 이미지에서 64비트 pHash를 구한다."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if max(width, height) > 300:
        scale = 300 / max(width, height)
        rgb = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())
    return variance, imagehash.phash(image, hash_size=8, highfreq_factor=4)


def _order(photo: PhotoState) -> tuple[bool, datetime, int]:
    # CLOCK의 처리 시각만 사용한다. 미확정 시각은 뒤에 두고 ID로 순서를 고정한다.
    taken_at = photo.taken_at
    if (
        photo.time_status is TimeStatus.PENDING
        or (photo.time_status is TimeStatus.UNKNOWN) != (taken_at is None)
        or (taken_at is not None and taken_at.utcoffset() is None)
    ):
        raise PipelineError("FILTERING에는 CLOCK이 판정한 일관된 처리 시각이 필요함")
    return (
        taken_at is None,
        taken_at.astimezone(UTC) if taken_at is not None else datetime.max.replace(tzinfo=UTC),
        photo.source.trip_attachment_id,
    )


def filter_photos(
    photos: list[PhotoState],
    images: Mapping[int, Image.Image],
    *,
    blur_var_min: float,
    phash_max_dist: int,
) -> dict[int, float]:
    """분류 사유와 중복 원본 ID를 갱신하고, 처리한 사진의 라플라시안 분산을 반환한다.

    BLURRY를 먼저 판정하며, 남은 사진 중 촬영 시각순으로 앞선 보존 사진을 중복 원본으로
    삼는다. 동시각과 미확정 시각은 ID로 정렬한다. 중복 사진을 다시 원본으로 쓰지 않는다.
    기존 UNCLEAR_LOCATION은 건드리지 않는다. 재실행 시 품질 판정만 다시 계산한다.
    분산은 후속 evaluation 산출에 재사용하며, 여기서는 정규화나 대표 사진 선정을 하지 않는다.
    """
    if not np.isfinite(blur_var_min) or blur_var_min < 0:
        raise PipelineError("blur_var_min은 0 이상의 유한값이어야 함")
    if (
        isinstance(phash_max_dist, bool)
        or not isinstance(phash_max_dist, int)
        or not 0 <= phash_max_dist <= 64
    ):
        raise PipelineError("phash_max_dist는 0 이상 64 이하의 정수여야 함")

    groups: dict[str, list[PhotoState]] = defaultdict(list)
    measurements: dict[int, tuple[float, imagehash.ImageHash]] = {}
    for photo in photos:
        if photo.issue is Issue.UNCLEAR_LOCATION:
            continue
        if not photo.place_id:
            raise PipelineError("FILTERING 대상 사진에 place_id가 필요함")
        photo_id = photo.source.trip_attachment_id
        if photo_id in measurements:
            raise PipelineError("FILTERING 대상 사진 ID가 중복됨")
        _order(photo)
        image = images.get(photo_id)
        if not isinstance(image, Image.Image) or min(image.size) < 1:
            raise PipelineError(f"FILTERING 대상 이미지가 없거나 유효하지 않음: {photo_id}")
        try:
            measurements[photo_id] = _measure(image)
        except (OSError, ValueError, cv2.error) as exc:
            raise PipelineError(f"FILTERING 이미지 측정 실패: {photo_id}") from exc
        groups[photo.place_id].append(photo)

    # 모든 입력의 측정이 성공한 뒤 분류를 갱신한다.
    for group in groups.values():
        retained: list[PhotoState] = []
        for photo in sorted(group, key=_order):
            photo_id = photo.source.trip_attachment_id
            variance, phash = measurements[photo_id]
            photo.issue = None
            photo.duplicate_of_attachment_id = None
            if variance < blur_var_min:
                photo.issue = Issue.BLURRY
                continue
            for original in retained:
                original_id = original.source.trip_attachment_id
                if phash - measurements[original_id][1] <= phash_max_dist:
                    photo.issue = Issue.DUPLICATED
                    photo.duplicate_of_attachment_id = original_id
                    break
            else:
                retained.append(photo)

    return {photo_id: value[0] for photo_id, value in measurements.items()}
