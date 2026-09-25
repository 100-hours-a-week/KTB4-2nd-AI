"""CLUSTERING: 확보한 좌표를 구면 거리 기반 DBSCAN으로 묶는다."""

import numpy as np
from sklearn.cluster import DBSCAN

from app.core.errors import PipelineError
from app.pipeline.state import PhotoState
from app.schemas.process import Issue, RegionOrigin

EARTH_RADIUS_M = 6_371_000


def cluster_photos(photos: list[PhotoState], *, dbscan_eps_m: float, min_samples: int) -> None:
    """미분류를 제외한 사진에 p1 형식의 place_id를 부여한다.

    날짜로 분할하지 않으며 min_samples=1이면 단독 사진도 장소가 된다.
    더 큰 min_samples에서 생기는 노이즈는 UNCLEAR_LOCATION으로 분리한다.
    사진 ID 순으로 처리해 입력 순서와 무관하게 장소 ID와 경계점 배정을 고정한다.
    """
    if not np.isfinite(dbscan_eps_m) or dbscan_eps_m <= 0:
        raise PipelineError("dbscan_eps_m은 양의 유한값이어야 함")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise PipelineError("min_samples는 1 이상의 정수여야 함")
    candidates = sorted(
        (photo for photo in photos if photo.issue is None),
        key=lambda photo: photo.source.trip_attachment_id,
    )
    if not candidates:
        return
    if any(
        photo.latitude is None
        or photo.longitude is None
        or photo.region_origin is RegionOrigin.UNKNOWN
        for photo in candidates
    ):
        raise PipelineError("CLUSTERING 전에 좌표와 출처가 확보되어야 함")
    coordinates = np.array([(photo.latitude, photo.longitude) for photo in candidates])
    if (
        not np.isfinite(coordinates).all()
        or (np.abs(coordinates[:, 0]) > 90).any()
        or (np.abs(coordinates[:, 1]) > 180).any()
    ):
        raise PipelineError("유효한 위도와 경도가 필요함")

    labels = DBSCAN(
        eps=dbscan_eps_m / EARTH_RADIUS_M,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(np.radians(coordinates))
    place_ids: dict[int, str] = {}
    for photo, label in zip(candidates, labels, strict=True):
        if label == -1:
            photo.place_id = None
            photo.issue = Issue.UNCLEAR_LOCATION
        else:
            if label not in place_ids:
                place_ids[label] = f"p{len(place_ids) + 1}"
            photo.place_id = place_ids[label]
