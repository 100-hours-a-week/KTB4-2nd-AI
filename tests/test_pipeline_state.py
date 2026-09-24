from dataclasses import FrozenInstanceError
from datetime import datetime

import numpy as np
import pytest

from app.pipeline.state import TimeStatus, build_photo_states
from app.schemas.process import Issue, ProcessRequest, RegionOrigin


@pytest.fixture
def request_data():
    return {
        "trip_name": "여행",
        "period": {"start_date": "2026-09-23", "end_date": "2026-09-24"},
        "regions": [{"latitude": 37.5, "longitude": 127.0}],
        "attachments": [
            {
                "trip_attachment_id": 20,
                "analyze_storage_key": "photos/20.jpg",
                "taken_at": "2010-01-01T12:34:56+09:00",
                "latitude": 37.5,
                "longitude": 127.0,
                "device_model": "camera",
            },
            {"trip_attachment_id": 3, "analyze_storage_key": "photos/3.jpg"},
        ],
    }


def test_preserves_input_order_and_source_values(request_data):
    request = ProcessRequest.model_validate(request_data)
    before = request.model_dump()

    photos = build_photo_states(request)

    assert [p.source.trip_attachment_id for p in photos] == [20, 3]
    for photo, attachment in zip(photos, request.attachments, strict=True):
        assert vars(photo.source) == attachment.model_dump()
    assert request.model_dump() == before


def test_starts_pending_even_with_original_time_outside_period(request_data):
    photos = build_photo_states(ProcessRequest.model_validate(request_data))

    assert photos[0].source.taken_at == datetime.fromisoformat("2010-01-01T12:34:56+09:00")
    assert photos[1].source.taken_at is None
    for photo in photos:
        assert photo.taken_at is None
        assert photo.time_status is TimeStatus.PENDING
        assert photo.embedding is None
        assert photo.place_id is None
        assert photo.issue is None
        assert photo.duplicate_of_attachment_id is None
        assert photo.evaluation is None


@pytest.mark.parametrize(
    ("latitude", "longitude"), [(37.5, 127.0), (0.0, 127.0), (37.5, 0.0), (0.0, 0.0)]
)
def test_copies_complete_original_gps(request_data, latitude, longitude):
    request_data["attachments"][0].update(latitude=latitude, longitude=longitude)

    photo = build_photo_states(ProcessRequest.model_validate(request_data))[0]

    assert photo.source.has_gps
    assert (photo.latitude, photo.longitude) == (latitude, longitude)
    assert photo.region_origin is RegionOrigin.EXIF


@pytest.mark.parametrize(("latitude", "longitude"), [(None, None), (37.5, None), (None, 127.0)])
def test_preserves_incomplete_gps_only_in_source(request_data, latitude, longitude):
    request_data["attachments"][0].update(latitude=latitude, longitude=longitude)

    photo = build_photo_states(ProcessRequest.model_validate(request_data))[0]

    assert (photo.source.latitude, photo.source.longitude) == (latitude, longitude)
    assert not photo.source.has_gps
    assert photo.latitude is None
    assert photo.longitude is None
    assert photo.region_origin is RegionOrigin.UNKNOWN
    assert photo.issue is None  # 위치 추론 전이므로 미분류로 확정하지 않는다.


def test_source_fields_cannot_be_overwritten(request_data):
    source = build_photo_states(ProcessRequest.model_validate(request_data))[0].source

    with pytest.raises(FrozenInstanceError):
        source.taken_at = None
    with pytest.raises(FrozenInstanceError):
        source.latitude = 0.0


def test_processing_changes_do_not_modify_sources_or_other_states(request_data):
    request = ProcessRequest.model_validate(request_data)
    before = request.model_dump()
    photos = build_photo_states(request)
    another_run = build_photo_states(request)
    photo = photos[1]

    photo.taken_at = datetime.fromisoformat("2026-09-23T12:34:56+09:00")
    photo.time_status = TimeStatus.CORRECTED
    photo.latitude, photo.longitude = 37.5, 127.0
    photo.region_origin = RegionOrigin.INFERRED
    photo.embedding = np.array([0.6, 0.8], dtype=np.float32)
    photo.place_id = "p1"
    photo.issue = Issue.DUPLICATED
    photo.duplicate_of_attachment_id = 20
    photo.evaluation = 80

    assert request.model_dump() == before
    assert photo.source.taken_at is None
    assert not photo.source.has_gps
    assert photo.source.latitude is None
    assert photo.source.longitude is None
    assert photos[0].time_status is TimeStatus.PENDING
    assert photos[0].embedding is None
    assert photos[0].issue is None
    assert another_run[1].taken_at is None
    assert another_run[1].region_origin is RegionOrigin.UNKNOWN
    assert another_run[1].embedding is None
    assert another_run[1].place_id is None


def test_later_request_changes_do_not_modify_source_snapshot(request_data):
    request = ProcessRequest.model_validate(request_data)
    source = build_photo_states(request)[0].source
    before = vars(source).copy()

    request.attachments[0].taken_at = None
    request.attachments[0].latitude = 0.0
    request.attachments[0].device_model = "changed"

    assert vars(source) == before
