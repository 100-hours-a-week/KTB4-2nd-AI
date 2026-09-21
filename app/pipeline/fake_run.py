"""가짜 파이프라인. run()과 같은 시그니처와 규칙, FAKE_PIPELINE=1이면 러너가 run() 대신 부름

다운로드는 진짜(ctx.images), 임베딩은 ctx.engine(가짜 엔진)과 ctx.qdrant, 나머지 단계는 진행률만.
다운로드가 진짜라 EC2에서 S3 권한과 키 경로가 검증되고, 좌표 있는 사진을 폴더 하나에 넣어
백엔드가 places와 unclassified 저장 경로를 둘 다 탐.
진짜 파이프라인이 붙은 뒤에도 서버 구조만 떼어 확인할 때 씀

run.py의 규칙을 그대로 지킴. 진행률은 DOWNLOADING과 EMBEDDING은 배치마다, 나머지는 시작과 끝.
취소는 단계 사이와 배치 사이. 실패는 예외로만. 디코딩은 EMBEDDING 배치마다 하고 버림,
200장을 한꺼번에 PIL로 풀면 약 500MB라 bytes만 들고 있음
"""

import io
from datetime import datetime

from PIL import Image

from app.core.errors import DecodeFailed, PipelineCancelled
from app.pipeline.context import PipelineContext
from app.pipeline.run import CancelCheck, ProgressCallback
from app.schemas.process import (
    Issue,
    Place,
    PlaceAttachment,
    ProcessAttachment,
    ProcessRequest,
    ProcessResult,
    RegionOrigin,
    UnclassifiedAttachment,
)
from app.schemas.task import ProcessStep

BATCH = 16
FAKE_EVALUATION = 80


def fake_run(
    request: ProcessRequest,
    ctx: PipelineContext,
    on_progress: ProgressCallback,
    is_cancelled: CancelCheck,
) -> ProcessResult:
    attachments = request.attachments
    total = len(attachments)

    def check_cancel() -> None:
        if is_cancelled():
            raise PipelineCancelled()

    # DOWNLOADING, 진짜. bytes만 보관
    blobs: dict[str, bytes] = {}
    for start in range(0, total, BATCH):
        check_cancel()
        chunk = attachments[start : start + BATCH]
        blobs.update(ctx.images.get_many([a.analyze_storage_key for a in chunk]))
        on_progress(ProcessStep.DOWNLOADING, min(start + BATCH, total), total)

    # EMBEDDING, 배치마다 디코딩 → 엔진 → upsert → 버림
    for start in range(0, total, BATCH):
        check_cancel()
        chunk = attachments[start : start + BATCH]
        images = _decode(chunk, blobs)
        vectors = ctx.engine.encode_images(images)
        ctx.qdrant.upsert([a.trip_attachment_id for a in chunk], vectors)
        for image in images:
            image.close()
        on_progress(ProcessStep.EMBEDDING, min(start + BATCH, total), total)

    # 판정 단계는 진행률만
    for step in (
        ProcessStep.CLOCK,
        ProcessStep.LOCATING,
        ProcessStep.CLUSTERING,
        ProcessStep.FILTERING,
    ):
        check_cancel()
        on_progress(step, 0, total)
        on_progress(step, total, total)

    check_cancel()
    on_progress(ProcessStep.FINALIZING, 0, total)
    result = _assemble(attachments)
    on_progress(ProcessStep.FINALIZING, total, total)
    return result


def _decode(chunk: list[ProcessAttachment], blobs: dict[str, bytes]) -> list[Image.Image]:
    """안 열리는 사진은 모아서 DecodeFailed(ids). 첫 실패에서 멈추면 문제 사진을 다 못 알려줌"""
    images: list[Image.Image] = []
    failed: list[int] = []
    for attachment in chunk:
        try:
            image = Image.open(io.BytesIO(blobs[attachment.analyze_storage_key]))
            image.load()
            images.append(image)
        except OSError:
            failed.append(attachment.trip_attachment_id)
    if failed:
        raise DecodeFailed(failed)
    return images


def _assemble(attachments: list[ProcessAttachment]) -> ProcessResult:
    """좌표 있는 사진은 p1 폴더 하나, 대표컷은 첫 장. 없는 사진은 위치 불명 미분류"""
    located = [a for a in attachments if a.latitude is not None and a.longitude is not None]
    unlocated = [a for a in attachments if a not in located]

    places: list[Place] = []
    if located:
        times: list[datetime] = [a.taken_at for a in located if a.taken_at is not None]
        places.append(
            Place(
                place_id="p1",
                latitude=sum(a.latitude for a in located if a.latitude) / len(located),
                longitude=sum(a.longitude for a in located if a.longitude) / len(located),
                first_taken_at=min(times) if times else None,
                last_taken_at=max(times) if times else None,
                representative_attachment_id=located[0].trip_attachment_id,
                attachments=[
                    PlaceAttachment(
                        trip_attachment_id=a.trip_attachment_id,
                        taken_at=a.taken_at,
                        latitude=a.latitude,  # type: ignore[arg-type]
                        longitude=a.longitude,  # type: ignore[arg-type]
                        region_origin=RegionOrigin.EXIF,
                        evaluation=FAKE_EVALUATION,
                    )
                    for a in located
                ],
            )
        )

    unclassified = [
        UnclassifiedAttachment(
            trip_attachment_id=a.trip_attachment_id,
            issue=Issue.UNCLEAR_LOCATION,
            taken_at=a.taken_at,
            latitude=None,
            longitude=None,
            region_origin=RegionOrigin.UNKNOWN,
            evaluation=None,
        )
        for a in unlocated
    ]
    return ProcessResult(places=places, unclassified=unclassified, failed=[], clock_offsets=[])
