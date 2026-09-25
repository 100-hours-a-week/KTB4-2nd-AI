"""LOCATING: 기간 판정 후 이번 요청의 원본 GPS에서만 좌표를 상속한다."""

import numpy as np

from app.core.errors import PipelineError
from app.pipeline.state import PhotoState, TimeStatus
from app.schemas.process import KST, Issue, Period, RegionOrigin


def _embedding(photo: PhotoState) -> np.ndarray:
    vector = photo.embedding
    if (
        not isinstance(vector, np.ndarray)
        or vector.ndim != 1
        or vector.size == 0
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-5, atol=1e-6)
    ):
        raise PipelineError(f"사진 {photo.source.trip_attachment_id}의 단위 임베딩이 필요함")
    return vector


def locate_photos(photos: list[PhotoState], period: Period, *, inherit_min_sim: float) -> None:
    """CLOCK 완료 상태를 받아 좌표와 출처 및 issue를 갱신한다.

    기간은 KST 날짜로 양 끝을 포함한다. UNKNOWN 시각은 기간 밖으로 보지 않는다.
    기간 밖 사진은 좌표가 있어도 UNCLEAR_LOCATION이며 참조에서 제외한다.
    v1의 regions와 외부 저장소는 사용하지 않는다. 동점은 작은 사진 ID를 우선한다.
    """
    if not np.isfinite(inherit_min_sim) or not -1 <= inherit_min_sim <= 1:
        raise PipelineError("inherit_min_sim은 -1 이상 1 이하의 유한값이어야 함")
    if period.start_date > period.end_date:
        raise PipelineError("여행 시작일은 종료일보다 늦을 수 없음")
    for photo in photos:
        if photo.time_status is TimeStatus.PENDING:
            raise PipelineError("LOCATING 전에 CLOCK 판정이 필요함")
        if photo.time_status is TimeStatus.UNKNOWN:
            if photo.taken_at is not None:
                raise PipelineError("UNKNOWN 시각은 None이어야 함")
        elif photo.taken_at is None or photo.taken_at.utcoffset() is None:
            raise PipelineError("확정 시각에는 시간대가 있는 taken_at이 필요함")

    anchors = []
    targets = []
    for photo in photos:
        if photo.issue is not None:
            continue
        if photo.taken_at is not None:
            day = photo.taken_at.astimezone(KST).date()
            if not period.start_date <= day <= period.end_date:
                photo.issue = Issue.UNCLEAR_LOCATION
                continue
        if photo.source.has_gps:
            photo.latitude = photo.source.latitude
            photo.longitude = photo.source.longitude
            photo.region_origin = RegionOrigin.EXIF
            anchors.append(photo)
        else:
            photo.latitude = photo.longitude = None
            photo.region_origin = RegionOrigin.UNKNOWN
            targets.append(photo)

    if not targets:
        return
    if not anchors:
        for photo in targets:
            photo.issue = Issue.UNCLEAR_LOCATION
        return

    # 상속한 좌표를 참조에 추가하지 않고 원본 GPS 집합을 고정한다.
    anchors.sort(key=lambda photo: photo.source.trip_attachment_id)
    vectors = [_embedding(photo) for photo in anchors + targets]
    if len({vector.shape for vector in vectors}) != 1:
        raise PipelineError("위치 상속에 사용하는 임베딩의 차원이 다름")
    anchor_vectors = np.stack(vectors[: len(anchors)])
    for photo, vector in zip(targets, vectors[len(anchors) :], strict=True):
        similarities = np.clip(anchor_vectors @ vector, -1.0, 1.0)
        best = int(np.argmax(similarities))
        if similarities[best] < inherit_min_sim:
            photo.issue = Issue.UNCLEAR_LOCATION
            continue
        source = anchors[best].source
        photo.latitude, photo.longitude = source.latitude, source.longitude
        photo.region_origin = RegionOrigin.INFERRED
