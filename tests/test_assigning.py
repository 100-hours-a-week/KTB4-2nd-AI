"""CLUSTERING 뒤 규칙 배정(시각 끼워 넣기, 닮은 사진 따라가기)을 작은 사진 상태로 확인한다."""

from datetime import datetime

import numpy as np
import pytest

from app.core.errors import PipelineError
from app.pipeline.state import PhotoSource, PhotoState, TimeStatus
from app.pipeline.steps.assigning import assign_unlocated
from app.schemas.process import Issue, Period, RegionOrigin

PERIOD = Period(start_date="2026-09-10", end_date="2026-09-12")
SIM = 0.88
GAP = 30


def unit(*values):
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def photo(photo_id, *, gps=None, place=None, at=None, vector=(1.0, 0.0, 0.0), inherited=None):
    """gps가 있으면 원본 GPS 사진, inherited가 있으면 LOCATING이 좌표를 물려준 사진이다."""
    taken_at = datetime.fromisoformat(f"2026-09-10T{at}:00+09:00") if at else None
    latitude, longitude = gps or (None, None)
    state = PhotoState(
        source=PhotoSource(photo_id, f"{photo_id}.jpg", taken_at, latitude, longitude, "phone"),
        taken_at=taken_at,
        time_status=TimeStatus.ORIGINAL if taken_at else TimeStatus.UNKNOWN,
        embedding=unit(*vector),
        place_id=place,
    )
    if gps:
        state.latitude, state.longitude, state.region_origin = *gps, RegionOrigin.EXIF
    elif inherited:
        state.latitude, state.longitude, state.region_origin = *inherited, RegionOrigin.INFERRED
    elif place is None:
        state.issue = Issue.UNCLEAR_LOCATION
    return state


def assign(photos, sim=SIM, gap=GAP):
    return assign_unlocated(photos, PERIOD, inherit_min_sim=sim, interpolate_max_gap_min=gap)


def test_photo_between_two_same_folder_photos_within_gap_is_interpolated():
    a = photo(1, gps=(37.0, 127.0), place="p1", at="10:00", vector=(1, 0, 0))
    b = photo(2, gps=(37.1, 127.1), place="p1", at="10:30", vector=(0, 1, 0))
    target = photo(3, at="10:15", vector=(0.2, 1, 0))
    assert assign([a, b, target]) == 1
    assert target.place_id == "p1" and target.issue is None
    assert target.region_origin is RegionOrigin.INFERRED
    # 폴더 GPS 사진 가운데 가장 닮은 사진(2)의 좌표를 받는다.
    assert (target.latitude, target.longitude) == (37.1, 127.1)


@pytest.mark.parametrize(
    ("before", "after", "assigned"),
    [
        ("09:45", "10:45", True),  # 양쪽 모두 정확히 30분
        ("09:44", "10:45", False),  # 앞쪽이 31분
        ("09:45", "10:46", False),  # 뒤쪽이 31분
    ],
)
def test_interpolation_requires_both_sides_within_gap(before, after, assigned):
    target = photo(3, at="10:15", vector=(0, 0, 1))
    photos = [
        photo(1, gps=(37.0, 127.0), place="p1", at=before, vector=(1, 0, 0)),
        photo(2, gps=(37.0, 127.0), place="p1", at=after, vector=(1, 0, 0)),
        target,
    ]
    assign(photos)
    assert (target.place_id == "p1") is assigned


def test_neighbors_in_different_folders_do_not_interpolate():
    photos = [
        photo(1, gps=(37.0, 127.0), place="p1", at="10:00", vector=(1, 0, 0)),
        photo(2, gps=(37.5, 127.5), place="p2", at="10:20", vector=(0, 1, 0)),
        photo(3, at="10:10", vector=(0, 0, 1)),
    ]
    assert assign(photos) == 0 and photos[2].issue is Issue.UNCLEAR_LOCATION


def test_similar_photos_in_exactly_one_folder_are_followed():
    photos = [
        photo(1, gps=(37.0, 127.0), place="p1", vector=(1, 0, 0)),
        photo(2, gps=(37.5, 127.5), place="p2", vector=(0, 1, 0)),
        photo(3, vector=(1, 0.1, 0)),
    ]
    assert assign(photos) == 1 and photos[2].place_id == "p1"


def test_similar_photos_in_two_folders_stay_unclassified():
    photos = [
        photo(1, gps=(37.0, 127.0), place="p1", vector=(1, 0.05, 0)),
        photo(2, gps=(37.5, 127.5), place="p2", vector=(1, -0.05, 0)),
        photo(3, vector=(1, 0, 0)),
    ]
    assert assign(photos) == 0 and photos[2].place_id is None


def test_assigned_photos_become_references_for_later_photos():
    # 4는 GPS 사진과 닮지 않았지만 앞에서 넣은 3과 닮아 같은 폴더로 따라간다.
    photos = [
        photo(1, gps=(37.0, 127.0), place="p1", vector=(1, 0, 0)),
        photo(3, vector=(1, 0.4, 0)),
        photo(4, vector=(1, 1, 0)),
    ]
    assert assign(photos) == 2
    assert [p.place_id for p in photos[1:]] == ["p1", "p1"]
    # 좌표는 같은 폴더의 GPS 사진(1)에서 받는다.
    assert (photos[2].latitude, photos[2].longitude) == (37.0, 127.0)


def test_inherited_photos_count_as_folder_members_but_not_as_coordinate_sources():
    photos = [
        photo(1, gps=(37.0, 127.0), place="p1", vector=(1, 0, 0)),
        photo(2, place="p1", inherited=(37.0, 127.0), vector=(0, 1, 0)),
        photo(3, vector=(0, 1, 0.1)),
    ]
    assert assign(photos) == 1 and photos[2].place_id == "p1"
    assert photos[2].latitude == 37.0


def test_out_of_period_and_gps_photos_are_left_as_located():
    outside = photo(3, vector=(1, 0, 0))
    outside.taken_at = datetime.fromisoformat("2026-09-13T10:00:00+09:00")
    gps_unclear = photo(4, gps=(36.0, 128.0), vector=(1, 0, 0))
    gps_unclear.issue = Issue.UNCLEAR_LOCATION
    photos = [photo(1, gps=(37.0, 127.0), place="p1", vector=(1, 0, 0)), outside, gps_unclear]
    assert assign(photos) == 0
    assert outside.issue is gps_unclear.issue is Issue.UNCLEAR_LOCATION
    assert outside.place_id is gps_unclear.place_id is None


def test_no_target_or_no_folder_is_a_no_op():
    assert assign([]) == 0
    lone = photo(1, vector=(1, 0, 0))
    assert assign([lone]) == 0 and lone.issue is Issue.UNCLEAR_LOCATION


@pytest.mark.parametrize(
    ("sim", "gap"), [(float("nan"), 30), (1.5, 30), (0.88, -1), (0.88, float("inf"))]
)
def test_invalid_thresholds_fail(sim, gap):
    with pytest.raises(PipelineError):
        assign([photo(1, vector=(1, 0, 0))], sim=sim, gap=gap)


def test_photo_followed_by_similarity_enables_interpolation_in_the_next_round():
    # 1차례: 10:10 사진은 뒤쪽 폴더 사진이 없어 끼울 수 없고, 10:20 사진은 닮은 사진으로 들어간다.
    # 2차례: 10:20 사진이 시각 순서에 들어와 10:10 사진을 끼워 넣는다.
    a = photo(1, gps=(37.0, 127.0), place="p1", at="10:00", vector=(1, 0, 0))
    later = photo(2, at="10:20", vector=(1, 0.1, 0))
    middle = photo(3, at="10:10", vector=(0, 0, 1))
    assert assign([a, later, middle]) == 2
    assert later.place_id == middle.place_id == "p1"


def test_result_does_not_depend_on_input_order():
    def build():
        return [
            photo(5, gps=(37.0, 127.0), place="p1", at="10:00", vector=(1, 0, 0)),
            photo(1, gps=(37.5, 127.5), place="p2", at="11:00", vector=(0, 1, 0)),
            photo(4, at="10:20", vector=(1, 0.2, 0)),
            photo(2, at="10:40", vector=(0, 0, 1)),
            photo(3, vector=(0.1, 1, 0)),
        ]

    forward, backward = build(), build()[::-1]
    assign(forward)
    assign(backward)
    by_id = lambda photos: {p.source.trip_attachment_id: p.place_id for p in photos}  # noqa: E731
    assert by_id(forward) == by_id(backward)
