from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from app.core.errors import PipelineError
from app.pipeline.state import PhotoSource, PhotoState, TimeStatus
from app.pipeline.steps.clock import correct_clocks

BASE = datetime(2026, 9, 10, 12, 34, 56, tzinfo=timezone(timedelta(hours=9)))
SETTINGS = {
    "clock_pair_min_sim": 0.9,
    "clock_min_pairs": 3,
    "clock_offset_tolerance_sec": 60,
}


def photo(photo_id, device, *, at=BASE, gps=False, vector=None):
    source = PhotoSource(
        photo_id, f"{photo_id}.jpg", at, 37.0 if gps else None, 127.0 if gps else None, device
    )
    return PhotoState(
        source=source,
        embedding=np.asarray(vector, dtype=np.float32) if vector is not None else None,
    )


def pairs(day_offsets=(3, 3, 3), residuals=None):
    residuals = residuals if residuals is not None else [0] * len(day_offsets)
    vectors = np.eye(len(day_offsets), dtype=np.float32)
    anchors, targets = [], []
    for i, (days, seconds) in enumerate(zip(day_offsets, residuals, strict=True)):
        reference_time = BASE + timedelta(hours=i)
        anchors.append(photo(i + 1, "phone", at=reference_time, gps=True, vector=vectors[i]))
        targets.append(
            photo(
                i + 11,
                "camera",
                at=reference_time - timedelta(days=days, seconds=seconds),
                vector=vectors[i],
            )
        )
    return anchors, targets


@pytest.mark.parametrize("days", [3, -2, 0])
def test_corrects_entire_device_preserving_source_and_time_of_day(days):
    anchors, targets = pairs((days,) * 3, (-30, 0, 30))
    unmatched = photo(30, "camera", at=BASE - timedelta(days=days), vector=np.ones(3) / np.sqrt(3))
    missing_time = photo(31, "camera", at=None)
    same_reference_device = photo(32, "phone")
    photos = [*targets, unmatched, missing_time, *anchors, same_reference_device]
    sources = [item.source for item in photos]

    offsets = correct_clocks(photos, **SETTINGS)

    assert [offset.model_dump() for offset in offsets] == [
        {"device_model": "camera", "days": days, "matched_pairs": 3}
    ]
    for item in [*targets, unmatched]:
        assert item.taken_at == item.source.taken_at + timedelta(days=days)
        assert item.taken_at.timetz() == item.source.taken_at.timetz()
        assert item.time_status is (TimeStatus.CORRECTED if days else TimeStatus.ORIGINAL)
    for item in [*anchors, same_reference_device]:
        assert item.time_status is TimeStatus.ORIGINAL
        assert item.taken_at == item.source.taken_at
    assert missing_time.taken_at is None
    assert missing_time.time_status is TimeStatus.UNKNOWN
    assert [item.source for item in photos] == sources
    # 같은 객체로 다시 호출해도 처리 시각에 보정을 중복 적용하지 않는다.
    assert correct_clocks(list(reversed(photos)), **SETTINGS) == offsets
    assert unmatched.taken_at == unmatched.source.taken_at + timedelta(days=days)


def test_compares_instants_across_time_zones():
    anchors, targets = pairs((0, 0, 0))
    targets = [
        photo(i + 11, "camera", at=item.source.taken_at.astimezone(UTC), vector=item.embedding)
        for i, item in enumerate(targets)
    ]
    result = correct_clocks(anchors + targets, **SETTINGS)
    assert result[0].days == 0
    assert targets[0].taken_at.hour == 3  # UTC 표현을 KST 시각으로 덮어쓰지 않는다.
    assert targets[0].taken_at == anchors[0].taken_at


@pytest.mark.parametrize(
    ("day_offsets", "residuals", "expected"),
    [
        ((3, 3), (0, 0), None),  # 최소 대응 수 미달
        ((3, 3, 3), (-60, 0, 60), (3, 3)),  # 허용 오차 경계 포함
        ((3, 3, 3), (-61, 0, 61), None),
        ((3, 3, 3, 3), (-20, 0, 20, 600), (3, 3)),  # 중앙값 밖 이상치 제외
        ((3, 3, 3, 4), (0, 0, 0, 0), (3, 3)),
        ((3, 3, 3, 4, 4, 4), (0,) * 6, None),  # 다른 날짜 보정값도 충분한 근거
        ((3, 3, 3, 3, 4, 4, 4), (0,) * 7, None),  # 다수 쪽도 임의 선택하지 않음
        ((3, 3, 3), (43_200,) * 3, None),  # 반일 차이는 날짜 후보가 모호함
        ((3, 3, 3), (7200,) * 3, (3, 3)),  # 초 차이는 일관성만 확인, 날짜만 적용
    ],
)
def test_requires_one_supported_day_offset(day_offsets, residuals, expected):
    anchors, targets = pairs(day_offsets, residuals)
    result = correct_clocks(anchors + targets, **SETTINGS)
    if expected is None:
        assert result == []
        assert all(
            item.time_status is TimeStatus.UNKNOWN and item.taken_at is None for item in targets
        )
    else:
        assert (result[0].days, result[0].matched_pairs) == expected


def test_three_targets_cannot_reuse_one_reference_to_reach_min_pairs():
    anchors, targets = pairs()
    for item in targets:
        item.embedding = anchors[0].embedding.copy()
    assert correct_clocks(anchors[:1] + targets, **SETTINGS) == []
    assert all(item.time_status is TimeStatus.UNKNOWN for item in targets)


def test_tied_reference_candidates_are_not_resolved_by_photo_id():
    anchors, targets = pairs()
    duplicate_anchors = [
        photo(i + 40, "other-phone", gps=True, at=item.source.taken_at, vector=item.embedding)
        for i, item in enumerate(anchors)
    ]
    assert correct_clocks(anchors + duplicate_anchors + targets, **SETTINGS) == []


@pytest.mark.parametrize(("min_sim", "matched"), [(0.8, True), (0.81, False)])
def test_similarity_threshold_boundary(min_sim, matched):
    anchors, targets = pairs()
    for i, item in enumerate(anchors):
        item.embedding = np.pad(item.embedding, (0, 1))
        targets[i].embedding = np.append(item.embedding[:3] * 0.8, np.float32(0.6))
    result = correct_clocks(anchors + targets, **{**SETTINGS, "clock_pair_min_sim": min_sim})
    assert bool(result) is matched


def test_corrected_photos_are_not_new_clock_references():
    anchors, targets = pairs()
    # 다른 대상 기기와만 유사하고 원본 참조와의 유사도는 임계값에 미달한다.
    others = []
    for i, (anchor, target) in enumerate(zip(anchors, targets, strict=True)):
        anchor.embedding = np.pad(anchor.embedding, (0, 3))
        direction = np.eye(6, dtype=np.float32)[i + 3]
        target.embedding = 0.8 * anchor.embedding + 0.6 * direction
        other_vector = 0.28 * anchor.embedding + 0.96 * direction
        others.append(photo(50 + i, "z-camera", at=BASE - timedelta(days=5), vector=other_vector))
    result = correct_clocks(anchors + targets + others, **{**SETTINGS, "clock_pair_min_sim": 0.75})
    assert [offset.device_model for offset in result] == ["camera"]
    assert all(item.time_status is TimeStatus.UNKNOWN for item in others)


def test_missing_device_names_are_not_treated_as_one_device():
    anchor = photo(1, None, gps=True)
    unknown = photo(2, None)
    named = photo(3, "camera", at=None)
    assert correct_clocks([anchor, unknown, named], **SETTINGS) == []
    assert anchor.time_status is TimeStatus.ORIGINAL
    assert unknown.time_status is TimeStatus.UNKNOWN
    assert named.time_status is TimeStatus.UNKNOWN


def test_no_references_and_empty_input_leave_no_pending_states():
    assert correct_clocks([], **SETTINGS) == []
    photos = [photo(1, "camera"), photo(2, "phone", at=None, gps=True)]
    assert correct_clocks(photos, **SETTINGS) == []
    assert all(item.time_status is TimeStatus.UNKNOWN and item.taken_at is None for item in photos)


@pytest.mark.parametrize("bad_vector", [None, [0, 0, 0], [np.nan, 0, 0], [1, 0]])
def test_needed_invalid_embedding_is_error_not_failed_clock_estimate(bad_vector):
    anchors, targets = pairs()
    targets[0].embedding = None if bad_vector is None else np.array(bad_vector, dtype=np.float32)
    with pytest.raises(PipelineError):
        correct_clocks(anchors + targets, **SETTINGS)


@pytest.mark.parametrize(
    "override",
    [
        {"clock_pair_min_sim": np.nan},
        {"clock_min_pairs": 0},
        {"clock_min_pairs": True},
        {"clock_offset_tolerance_sec": -1},
        {"clock_offset_tolerance_sec": 43_200},
    ],
)
def test_invalid_clock_configuration_fails(override):
    with pytest.raises(PipelineError):
        correct_clocks([], **{**SETTINGS, **override})


def test_naive_source_time_fails_instead_of_using_host_time_zone():
    with pytest.raises(PipelineError):
        correct_clocks([photo(1, "phone", gps=True, at=BASE.replace(tzinfo=None))], **SETTINGS)
