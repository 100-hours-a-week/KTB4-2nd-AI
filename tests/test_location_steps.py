from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from app.core.errors import PipelineError
from app.pipeline.state import PhotoSource, PhotoState, TimeStatus, build_photo_states
from app.pipeline.steps.clustering import cluster_photos
from app.pipeline.steps.locating import locate_photos
from app.schemas.process import Issue, Period, ProcessRequest, RegionOrigin

PERIOD = Period(start_date="2026-09-10", end_date="2026-09-12")


def photo(photo_id, gps=None, vector=(1.0, 0.0), taken_at=None):
    latitude, longitude = gps if gps is not None else (None, None)
    taken_at = datetime.fromisoformat(taken_at) if taken_at else None
    return PhotoState(
        source=PhotoSource(photo_id, f"{photo_id}.jpg", taken_at, latitude, longitude, "camera"),
        taken_at=taken_at,
        time_status=TimeStatus.ORIGINAL if taken_at else TimeStatus.UNKNOWN,
        latitude=latitude,
        longitude=longitude,
        region_origin=RegionOrigin.EXIF if gps is not None else RegionOrigin.UNKNOWN,
        embedding=np.array(vector, dtype=np.float32) if vector is not None else None,
    )


@pytest.mark.parametrize(
    ("taken_at", "excluded"),
    [
        ("2026-09-09T14:59:59+00:00", True),
        ("2026-09-09T15:00:00+00:00", False),
        ("2026-09-12T14:59:59+00:00", False),
        ("2026-09-12T15:00:00+00:00", True),
    ],
)
def test_period_uses_kst_and_includes_both_dates(taken_at, excluded):
    item = photo(1, gps=(0.0, 0.0), vector=None, taken_at=taken_at)
    locate_photos([item], PERIOD, inherit_min_sim=0.8)
    assert item.issue is (Issue.UNCLEAR_LOCATION if excluded else None)
    assert (item.latitude, item.longitude) == (0.0, 0.0)
    assert item.region_origin is RegionOrigin.EXIF


@pytest.mark.parametrize("status", [TimeStatus.UNKNOWN, TimeStatus.CORRECTED])
def test_does_not_use_untrusted_original_date(status):
    item = photo(1, gps=(37.0, 127.0), taken_at="2026-09-10T12:00:00+09:00")
    item.source = replace(item.source, taken_at=datetime.fromisoformat("2010-01-01T12:00:00+09:00"))
    item.time_status = status
    if status is TimeStatus.UNKNOWN:
        item.taken_at = None
    locate_photos([item], PERIOD, inherit_min_sim=0.8)
    assert item.issue is None
    assert item.source.taken_at.year == 2010


def test_highest_similarity_inherits_at_threshold_but_low_similarity_is_unclear():
    first = photo(10, gps=(37.0, 127.0), vector=(1.0, 0.0))
    best = photo(20, gps=(38.0, 128.0), vector=(0.0, 1.0))
    target = photo(3, vector=(0.6, 0.8))
    rejected = photo(4, vector=(-1.0, 0.0))
    photos = [target, first, rejected, best]
    originals = [p.source for p in photos]

    locate_photos(photos, PERIOD, inherit_min_sim=0.8)

    assert (target.latitude, target.longitude) == (38.0, 128.0)
    assert target.region_origin is RegionOrigin.INFERRED
    assert target.issue is None
    assert rejected.issue is Issue.UNCLEAR_LOCATION
    assert rejected.latitude is None
    assert rejected.region_origin is RegionOrigin.UNKNOWN
    assert [p.source for p in photos] == originals
    assert [p.source.trip_attachment_id for p in photos] == [3, 10, 4, 20]
    assert (first.latitude, first.longitude) == (37.0, 127.0)
    assert first.region_origin is RegionOrigin.EXIF


@pytest.mark.parametrize("reverse", [False, True])
def test_similarity_tie_uses_smallest_original_photo_id(reverse):
    anchors = [photo(20, gps=(38.0, 128.0)), photo(10, gps=(37.0, 127.0))]
    if reverse:
        anchors.reverse()
    target = photo(3)
    locate_photos([target, *anchors], PERIOD, inherit_min_sim=1.0)
    assert (target.latitude, target.longitude) == (37.0, 127.0)


def test_inferred_coordinates_never_become_anchors():
    def vector(degrees):
        angle = np.radians(degrees)
        return (np.cos(angle), np.sin(angle))

    anchor = photo(1, gps=(37.0, 127.0), vector=vector(0))
    first = photo(2, vector=vector(40))
    second = photo(3, vector=vector(80))
    locate_photos([anchor, first, second], PERIOD, inherit_min_sim=0.7)
    assert first.region_origin is RegionOrigin.INFERRED
    assert second.issue is Issue.UNCLEAR_LOCATION
    assert second.latitude is None


@pytest.mark.parametrize("outside_anchor", [False, True])
def test_no_eligible_anchor_is_unclear_without_embedding_lookup(outside_anchor):
    target = photo(2, vector=None)
    photos = [target]
    if outside_anchor:
        photos.append(photo(1, gps=(37.0, 127.0), taken_at="2026-09-13T00:00:00+09:00"))
    locate_photos(photos, PERIOD, inherit_min_sim=0.8)
    assert target.issue is Issue.UNCLEAR_LOCATION


@pytest.mark.parametrize("vector", [None, (0.0, 0.0), (2.0, 0.0), (np.nan, 0.0), (1.0,)])
def test_broken_embedding_is_pipeline_error_not_unclear(vector):
    anchor = photo(1, gps=(37.0, 127.0))
    target = photo(2, vector=vector)
    with pytest.raises(PipelineError):
        locate_photos([anchor, target], PERIOD, inherit_min_sim=0.8)
    assert target.issue is None


@pytest.mark.parametrize(
    "invalid_state", ["pending", "unknown_with_time", "known_without_time", "naive_time"]
)
def test_requires_completed_clock_state(invalid_state):
    item = photo(1, gps=(37.0, 127.0))
    if invalid_state == "pending":
        item.time_status = TimeStatus.PENDING
    elif invalid_state == "unknown_with_time":
        item.taken_at = datetime.fromisoformat("2026-09-10T12:00:00+09:00")
    elif invalid_state == "known_without_time":
        item.time_status = TimeStatus.ORIGINAL
    else:
        item.time_status = TimeStatus.ORIGINAL
        item.taken_at = datetime(2026, 9, 10, 12)
    with pytest.raises(PipelineError):
        locate_photos([item], PERIOD, inherit_min_sim=0.8)


def test_nan_similarity_threshold_fails_instead_of_accepting_every_target():
    photos = [photo(1, gps=(37.0, 127.0)), photo(2, vector=(-1.0, 0.0))]
    with pytest.raises(PipelineError):
        locate_photos(photos, PERIOD, inherit_min_sim=np.nan)
    assert photos[1].region_origin is RegionOrigin.UNKNOWN


def test_clustering_connects_neighbors_keeps_revisits_and_singletons():
    # 적도에서 경도 0.001도는 약 111m. 양 끝 약 222m도 중간 사진을 통해 연결된다.
    a = photo(10, gps=(0.0, 0.0), taken_at="2026-09-10T12:00:00+09:00")
    b = photo(20, gps=(0.0, 0.001))
    c = photo(30, gps=(0.0, 0.002), taken_at="2026-09-12T12:00:00+09:00")
    alone = photo(40, gps=(1.0, 1.0))
    excluded = photo(50, gps=(0.0, 0.0))
    excluded.issue = Issue.UNCLEAR_LOCATION
    photos = [alone, c, excluded, a, b]
    cluster_photos(photos, dbscan_eps_m=150, min_samples=1)
    assert [p.place_id for p in photos] == ["p2", "p1", None, "p1", "p1"]
    cluster_photos(list(reversed(photos)), dbscan_eps_m=150, min_samples=1)
    assert [p.place_id for p in photos] == ["p2", "p1", None, "p1", "p1"]


def test_haversine_wraps_longitude_at_date_line():
    photos = [photo(1, gps=(0.0, 179.9995)), photo(2, gps=(0.0, -179.9995))]
    cluster_photos(photos, dbscan_eps_m=150, min_samples=1)
    assert photos[0].place_id == photos[1].place_id == "p1"


def test_noise_with_larger_min_samples_is_unclear():
    photos = [photo(1, gps=(0.0, 0.0)), photo(2, gps=(0.0, 0.001)), photo(3, gps=(1.0, 1.0))]
    cluster_photos(photos, dbscan_eps_m=150, min_samples=2)
    assert [p.place_id for p in photos] == ["p1", "p1", None]
    assert photos[2].issue is Issue.UNCLEAR_LOCATION
    assert photos[2].region_origin is RegionOrigin.EXIF


@pytest.mark.parametrize("gps", [None, (91.0, 127.0), (37.0, np.nan)])
def test_unresolved_or_invalid_coordinates_fail_clustering(gps):
    with pytest.raises(PipelineError):
        cluster_photos([photo(1, gps=gps)], dbscan_eps_m=150, min_samples=1)


def test_empty_and_all_unclassified_are_supported():
    locate_photos([], PERIOD, inherit_min_sim=0.8)
    cluster_photos([], dbscan_eps_m=150, min_samples=1)
    item = photo(1)
    locate_photos([item], PERIOD, inherit_min_sim=0.8)
    cluster_photos([item], dbscan_eps_m=150, min_samples=1)
    assert item.issue is Issue.UNCLEAR_LOCATION
    assert item.place_id is None


def test_request_to_location_and_clusters_preserves_originals_and_ignores_regions():
    request = ProcessRequest(
        trip_name="여행",
        period=PERIOD,
        regions=[{"latitude": -30.0, "longitude": -100.0}],
        attachments=[
            {"trip_attachment_id": 20, "analyze_storage_key": "20.jpg"},
            {
                "trip_attachment_id": 10,
                "analyze_storage_key": "10.jpg",
                "latitude": 37.0,
                "longitude": 127.0,
            },
        ],
    )
    before = request.model_dump()
    photos = build_photo_states(request)
    for item in photos:
        # CLOCK과 EMBEDDING의 완료 출력을 제공하며 해당 알고리즘을 실행한 것은 아니다.
        item.time_status = TimeStatus.UNKNOWN
        item.embedding = np.array([1.0, 0.0], dtype=np.float32)
    locate_photos(photos, request.period, inherit_min_sim=0.8)
    cluster_photos(photos, dbscan_eps_m=200, min_samples=1)
    assert [p.place_id for p in photos] == ["p1", "p1"]
    assert photos[0].region_origin is RegionOrigin.INFERRED
    assert photos[0].source.latitude is None
    assert photos[1].region_origin is RegionOrigin.EXIF
    assert request.model_dump() == before
