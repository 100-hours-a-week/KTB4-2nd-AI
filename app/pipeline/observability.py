"""단계별 사진 상태 관측. PhotoState를 로그로 옮기는 변환만 둠

api와 worker를 import하지 않음. 서버 없이 scripts/run_pipeline.py로 돌 때도 같은 로그가 남음
좌표와 사본 키는 개인 위치정보라 요약에 담지 않고 사진 단위 기록에만 최소로 둠
embedding은 1152차원이라 값 대신 존재 여부만
"""

import logging
from collections import Counter
from typing import Any

from app.core.logging import STEP_NAMES
from app.pipeline.state import PhotoState
from app.schemas.task import ProcessStep


def summarize(photos: list[PhotoState]) -> dict[str, Any]:
    """단계가 끝난 시점의 분포. 사진 수와 무관하게 줄 하나"""
    return {
        "origin": dict(Counter(photo.region_origin.value for photo in photos)),
        "issue": dict(Counter(photo.issue.value if photo.issue else "none" for photo in photos)),
        "time_status": dict(Counter(photo.time_status.value for photo in photos)),
        "place_count": len({photo.place_id for photo in photos if photo.place_id}),
        "with_coords": sum(
            1 for photo in photos if photo.latitude is not None and photo.longitude is not None
        ),
        "with_embedding": sum(1 for photo in photos if photo.embedding is not None),
    }


def serialize(photo: PhotoState) -> dict[str, Any]:
    """사진 하나의 현재 상태. 단계 전후를 줄 단위로 대조하기 위한 항목만"""
    return {
        "attachment_id": photo.source.trip_attachment_id,
        "time_status": photo.time_status.value,
        "taken_at": photo.taken_at.isoformat() if photo.taken_at else None,
        "origin": photo.region_origin.value,
        "has_coords": photo.latitude is not None and photo.longitude is not None,
        "place_id": photo.place_id,
        "issue": photo.issue.value if photo.issue else None,
        "duplicate_of": photo.duplicate_of_attachment_id,
        "evaluation": photo.evaluation,
        "has_embedding": photo.embedding is not None,
    }


def log_snapshot(log: logging.Logger, step: ProcessStep, photos: list[PhotoState]) -> None:
    """단계 종료 직후 호출. 요약은 INFO, 사진 단위는 DEBUG

    사진 단위는 장수만큼 줄이 늘어나므로 레벨을 먼저 확인해 조립 비용을 아낌
    """
    step_name = STEP_NAMES[step.value]
    log.info(
        "파이프라인 단계 결과를 집계했습니다.",
        extra={
            "event": "ai_pipeline_snapshot",
            "result": "success",
            "step": step_name,
            "total": len(photos),
            **summarize(photos),
        },
    )
    if not log.isEnabledFor(logging.DEBUG):
        return
    for photo in photos:
        log.debug(
            "사진 상태입니다.",
            extra={
                "event": "ai_photo_state",
                "step": step_name,
                **serialize(photo),
            },
        )
