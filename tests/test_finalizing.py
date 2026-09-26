from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.core.errors import PipelineError
from app.pipeline.state import PhotoSource, PhotoState, TimeStatus
from app.pipeline.steps.finalizing import finalize_photos
from app.pipeline.steps.represent import quality_score, select_representative
from app.schemas.process import ClockOffset, Issue, RegionOrigin

BASE = datetime(2026, 9, 10, 12, tzinfo=UTC)


def photo(photo_id, *, angle=0, place="p1", issue=None, minute=0):
    radians = np.radians(angle)
    return PhotoState(
        source=PhotoSource(photo_id, f"{photo_id}.jpg", BASE, 37.5, 127.0, "camera"),
        taken_at=BASE + timedelta(minutes=minute) if minute is not None else None,
        time_status=TimeStatus.CORRECTED if minute is not None else TimeStatus.UNKNOWN,
        latitude=37.5,
        longitude=127.0,
        region_origin=RegionOrigin.EXIF,
        embedding=np.array([np.cos(radians), np.sin(radians)], dtype=np.float32),
        place_id=place,
        issue=issue,
    )


@pytest.mark.parametrize("variance,score", [(0, 0), (25, 20), (100, 50), (300, 75), (1e308, 100)])
def test_quality_score_fixed_scale(variance, score):
    assert quality_score(variance) == score


@pytest.mark.parametrize("variance", [-1, float("nan"), float("inf")])
def test_invalid_variance_fails(variance):
    with pytest.raises(PipelineError):
        quality_score(variance)


def test_representative_is_mean_nearest_not_quality_or_first_photo():
    photos = [photo(1, angle=0), photo(9, angle=30), photo(2, angle=90)]
    result = finalize_photos(photos, {1: 900, 9: 100, 2: 300}, [])
    assert result.places[0].representative_attachment_id == 9
    assert [p.evaluation for p in result.places[0].attachments] == [90, 75, 50]
    assert select_representative(photos[::-1]) == 9


def test_representative_single_tie_and_zero_mean():
    assert select_representative([photo(8)]) == 8
    assert select_representative([photo(9), photo(2)]) == 2
    photos = [photo(8), photo(3)]
    photos[1].embedding = np.array([-1, 0], dtype=np.float32)
    assert select_representative(photos) == 3


@pytest.mark.parametrize(
    "vector",
    [
        None,
        np.zeros(2),
        np.array([np.nan, 0]),
        np.array([2.0, 0]),
        np.ones((1, 2)),
        np.array([]),
        np.array([1.0, 0, 0]),
    ],
)
def test_bad_candidate_embedding_fails(vector):
    photos = [photo(1), photo(2)]
    photos[1].embedding = vector
    with pytest.raises(PipelineError):
        select_representative(photos)


def test_assembles_every_photo_once_preserving_exclusions_offsets_and_inputs():
    photos = [
        photo(3, issue=Issue.DUPLICATED),
        photo(2, issue=Issue.BLURRY),
        photo(1),
        photo(4, place="p2", minute=None),
        photo(5, place=None, issue=Issue.UNCLEAR_LOCATION, minute=None),
    ]
    photos[0].duplicate_of_attachment_id = 1
    photos[0].embedding = photos[1].embedding = photos[4].embedding = None
    photos[4].latitude = photos[4].longitude = None
    photos[4].region_origin = RegionOrigin.UNKNOWN
    sources = [vars(p.source).copy() for p in photos]
    states = [(p.issue, p.place_id, p.taken_at, p.evaluation) for p in photos]
    offsets = [ClockOffset(device_model="camera", days=2, matched_pairs=3)]

    result = finalize_photos(photos, {1: 300, 2: 10, 3: 200, 4: 100}, offsets)

    assert [p.place_id for p in result.places] == ["p1", "p2"]
    assert [p.representative_attachment_id for p in result.places] == [1, 4]
    assert all(p.matched_trip_place_id is None for p in result.places)
    assert [p.trip_attachment_id for p in result.unclassified] == [2, 3, 5]
    assert result.unclassified[1].duplicate_of_attachment_id == 1
    assert result.unclassified[0].place_id == result.unclassified[1].place_id == "p1"
    assert result.unclassified[2].evaluation is None
    assert result.places[1].first_taken_at is result.places[1].last_taken_at is None
    assert result.failed == [] and result.clock_offsets == offsets
    assert [vars(p.source) for p in photos] == sources
    assert [(p.issue, p.place_id, p.taken_at, p.evaluation) for p in photos] == states
    assert [p.source.trip_attachment_id for p in photos] == [3, 2, 1, 4, 5]
    assert result == finalize_photos(photos[::-1], {1: 300, 2: 10, 3: 200, 4: 100}, offsets)


def test_times_use_retained_photos_and_serialize_kst():
    photos = [
        photo(1, minute=60),
        photo(2),
        photo(3, minute=None),
        photo(4, issue=Issue.BLURRY, minute=-120),
    ]
    photos[0].taken_at = datetime.fromisoformat("2026-09-10T22:00:00+09:00")
    result = finalize_photos(photos, dict.fromkeys(range(1, 5), 100), [])
    place = result.places[0]
    assert place.first_taken_at == BASE
    assert place.last_taken_at == BASE + timedelta(hours=1)
    assert [p.trip_attachment_id for p in place.attachments] == [2, 1, 3]
    data = result.model_dump(mode="json")
    assert data["places"][0]["first_taken_at"] == "2026-09-10T21:00:00+09:00"


def test_center_handles_dateline_and_keeps_excluded_members():
    photos = [photo(1), photo(2, issue=Issue.BLURRY)]
    for p, longitude in zip(photos, [179.9, -179.9], strict=True):
        p.latitude, p.longitude = 0, longitude
    result = finalize_photos(photos, {1: 100, 2: 0}, [])
    assert result.places[0].latitude == pytest.approx(0)
    assert abs(result.places[0].longitude) == pytest.approx(180)
    photos[1].longitude = -0.1
    with pytest.raises(PipelineError, match="기준 좌표"):
        finalize_photos(photos, {1: 100, 2: 0}, [])


def test_empty_and_all_unclear_results_are_valid():
    assert finalize_photos([], {}, []).places == []
    unclear = photo(1, place=None, issue=Issue.UNCLEAR_LOCATION)
    result = finalize_photos([unclear], {}, [])
    assert not result.places and len(result.unclassified) == 1


def test_no_candidate_is_assumption_violation_without_restoring_blurry_photo():
    blurry = photo(1, issue=Issue.BLURRY)
    with pytest.raises(PipelineError, match="후보가 없음"):
        finalize_photos([blurry], {1: 10}, [])
    assert blurry.issue is Issue.BLURRY


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_id",
        "missing_variance",
        "missing_place",
        "missing_coords",
        "unknown_origin",
        "pending",
        "unknown_with_time",
        "known_without_time",
        "naive_time",
        "invalid_coords",
        "nan_coords",
        "unclear_with_place",
        "unexpected_duplicate_id",
        "missing_original",
        "different_place_original",
        "excluded_original",
        "invalid_response",
    ],
)
def test_invalid_internal_state_fails(problem):
    photos = [photo(1), photo(2)]
    variances = {1: 100, 2: 100}
    offsets = []
    target = photos[1]
    if problem == "duplicate_id":
        photos[1] = photo(1)
    elif problem == "missing_variance":
        del variances[2]
    elif problem == "missing_place":
        target.place_id = None
    elif problem == "missing_coords":
        target.latitude = None
    elif problem == "unknown_origin":
        target.region_origin = RegionOrigin.UNKNOWN
    elif problem == "pending":
        target.time_status = TimeStatus.PENDING
    elif problem == "unknown_with_time":
        target.time_status = TimeStatus.UNKNOWN
    elif problem == "known_without_time":
        target.taken_at = None
    elif problem == "naive_time":
        target.taken_at = BASE.replace(tzinfo=None)
    elif problem == "invalid_coords":
        target.latitude = 91
    elif problem == "nan_coords":
        target.longitude = float("nan")
    elif problem == "unclear_with_place":
        target.issue = Issue.UNCLEAR_LOCATION
    elif problem == "unexpected_duplicate_id":
        target.duplicate_of_attachment_id = 1
    elif problem == "invalid_response":
        offsets = [{"device_model": "camera", "days": 1, "matched_pairs": -1}]
    else:
        target.issue, target.duplicate_of_attachment_id = Issue.DUPLICATED, 1
        if problem == "missing_original":
            target.duplicate_of_attachment_id = 99
        elif problem == "different_place_original":
            target.place_id = "p2"
        elif problem == "excluded_original":
            photos[0].issue = Issue.BLURRY
    with pytest.raises(PipelineError):
        finalize_photos(photos, variances, offsets)
