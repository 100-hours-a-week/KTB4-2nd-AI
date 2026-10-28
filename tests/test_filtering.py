from datetime import UTC, datetime, timedelta

import imagehash
import numpy as np
import pytest
from PIL import Image, ImageFilter

from app.core.errors import PipelineError
from app.pipeline.state import PhotoSource, PhotoState, TimeStatus
from app.pipeline.steps import filtering
from app.schemas.process import Issue, RegionOrigin

BASE = datetime(2026, 9, 10, 12, tzinfo=UTC)


def photo(photo_id, *, minute=0, place="p1"):
    taken_at = BASE + timedelta(minutes=minute) if minute is not None else None
    return PhotoState(
        source=PhotoSource(photo_id, f"{photo_id}.jpg", BASE, 37.5, 127.0, "camera"),
        taken_at=taken_at,
        time_status=TimeStatus.CORRECTED if taken_at is not None else TimeStatus.UNKNOWN,
        latitude=37.5,
        longitude=127.0,
        region_origin=RegionOrigin.EXIF,
        place_id=place,
    )


def run_fixed(photos, values, *, blur=100.0, distance=5):
    """ID별 (분산, 64비트 정수) 지표로 판정 규칙을 검증한다."""
    metrics = {
        photo_id: filtering.PhotoMetrics(variance, f"{bits:016x}")
        for photo_id, (variance, bits) in values.items()
    }
    return filtering.filter_photos(photos, metrics, blur_var_min=blur, phash_max_dist=distance)


def test_real_pixels_detect_blur_and_exact_duplicate_without_mutation():
    rng = np.random.default_rng(42)
    sharp = Image.fromarray(rng.integers(0, 256, (200, 300, 3), dtype=np.uint8))
    blurred = sharp.filter(ImageFilter.GaussianBlur(8))
    photos = [photo(3, minute=2), photo(1), photo(2, minute=1)]
    images = {1: sharp, 2: sharp.copy(), 3: blurred}
    sources = [vars(p.source).copy() for p in photos]
    before = {key: value.tobytes() for key, value in images.items()}

    metrics = filtering.measure_images(images)
    variances = filtering.filter_photos(photos, metrics, blur_var_min=100, phash_max_dist=0)

    assert variances[1] == variances[2] > 100 > variances[3]
    assert [p.issue for p in photos] == [Issue.BLURRY, None, Issue.DUPLICATED]
    assert photos[2].duplicate_of_attachment_id == 1
    assert photos[0].duplicate_of_attachment_id is None
    assert [p.source.trip_attachment_id for p in photos] == [3, 1, 2]
    assert [vars(p.source) for p in photos] == sources
    assert all(p.place_id == "p1" and p.latitude == 37.5 for p in photos)
    assert {key: value.tobytes() for key, value in images.items()} == before


@pytest.mark.parametrize(
    "size,expected",
    [
        ((1600, 900), (576, 1024)),
        ((900, 1600), (1024, 576)),
        ((1024, 768), (768, 1024)),
        ((80, 40), (40, 80)),
        ((2000, 1), (1, 1024)),
    ],
)
def test_measure_resizes_without_upscaling_or_changing_input(monkeypatch, size, expected):
    seen = []
    real_laplacian = filtering.cv2.Laplacian

    def capture(array, *args, **kwargs):
        seen.append(array.shape)
        return real_laplacian(array, *args, **kwargs)

    monkeypatch.setattr(filtering.cv2, "Laplacian", capture)
    image = Image.new("RGB", size, "gray")
    variance, phash = filtering._measure(image)

    assert seen == [expected]
    assert variance == 0
    assert phash.hash.shape == (8, 8)
    assert image.size == size


def test_measure_returns_storable_metrics_with_same_values():
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8))
    variance, phash = filtering._measure(image)

    metrics = filtering.measure(image)

    assert metrics == filtering.PhotoMetrics(variance, str(phash))
    assert len(metrics.phash) == 16
    assert imagehash.hex_to_hash(metrics.phash) == phash


@pytest.mark.parametrize("problem", ["not_image", "closed_image", "empty_image"])
def test_measure_images_fails_for_unusable_image(problem):
    images = {1: Image.new("RGB", (10, 10)), 2: Image.new("RGB", (10, 10))}
    if problem == "not_image":
        images[2] = b"not decoded"
    elif problem == "closed_image":
        images[2].close()
    elif problem == "empty_image":
        images[2] = Image.new("RGB", (0, 0))
    with pytest.raises(PipelineError):
        filtering.measure_images(images)


@pytest.mark.parametrize("variance,expected", [(99.9, Issue.BLURRY), (100, None), (100.1, None)])
def test_blur_boundary(variance, expected):
    photos = [photo(1)]
    result = run_fixed(photos, {1: (variance, 0)})
    assert photos[0].issue is expected
    assert result == {1: variance}


@pytest.mark.parametrize(
    "distance,expected", [(4, None), (5, Issue.DUPLICATED), (6, Issue.DUPLICATED)]
)
def test_hamming_boundary(distance, expected):
    photos = [photo(1), photo(2, minute=1)]
    run_fixed(photos, {1: (200, 0), 2: (200, 31)}, distance=distance)
    assert photos[1].issue is expected
    assert photos[1].duplicate_of_attachment_id == (1 if expected else None)


def test_blurry_earlier_photo_is_not_a_duplicate_original():
    photos = [photo(1), photo(2, minute=1), photo(3, minute=2)]
    run_fixed(photos, {1: (10, 0), 2: (200, 0), 3: (200, 0)})
    assert [p.issue for p in photos] == [Issue.BLURRY, None, Issue.DUPLICATED]
    assert photos[2].duplicate_of_attachment_id == 2


def test_no_transitive_duplicate_chain_or_cross_place_comparison():
    # A와 B 거리 2, B와 C 거리 2, A와 C 거리 4. C는 보존된 A와 직접 비교한다.
    photos = [photo(3, minute=2), photo(2, minute=1), photo(1), photo(4, place="p2")]
    values = {1: (200, 0), 2: (200, 3), 3: (200, 15), 4: (200, 0)}
    run_fixed(photos, values, distance=2)
    assert [p.issue for p in photos] == [None, Issue.DUPLICATED, None, None]
    assert photos[1].duplicate_of_attachment_id == 1


def test_uses_corrected_instants_then_id_with_unknown_times_last():
    early = photo(9)
    early.taken_at = datetime.fromisoformat("2026-09-10T20:00:00+09:00")
    photos = [photo(1, minute=None), photo(2), early, photo(3)]
    run_fixed(photos, {p.source.trip_attachment_id: (200, 0) for p in photos})
    assert early.issue is None
    assert [p.duplicate_of_attachment_id for p in photos] == [9, 9, None, 9]

    tied = [photo(4), photo(2)]
    run_fixed(tied, {4: (200, 0), 2: (200, 0)})
    assert tied[0].duplicate_of_attachment_id == 2
    unknown = [photo(4, minute=None), photo(2, minute=None)]
    run_fixed(unknown, {4: (200, 0), 2: (200, 0)})
    assert unknown[0].duplicate_of_attachment_id == 2


def test_earliest_retained_match_wins_and_rerun_recomputes():
    photos = [photo(3, minute=2), photo(2, minute=1), photo(1)]
    values = {1: (200, 0), 2: (200, 15), 3: (200, 3)}
    run_fixed(photos, values, distance=2)
    assert photos[0].duplicate_of_attachment_id == 1
    run_fixed(photos[::-1], values, distance=2)
    assert photos[0].duplicate_of_attachment_id == 1
    run_fixed(photos, values, distance=0)
    assert all(p.issue is None and p.duplicate_of_attachment_id is None for p in photos)


def unclear(photo_id, *, minute=0):
    state = photo(photo_id, minute=minute, place=None)
    state.issue = Issue.UNCLEAR_LOCATION
    state.latitude = state.longitude = None
    state.region_origin = RegionOrigin.UNKNOWN
    return state


def test_location_unknown_photos_get_blur_and_duplicate_without_place():
    # 위치 불명 8과 장소 사진 1은 pHash가 같아도 서로 비교하지 않는다.
    photos = [unclear(9, minute=2), unclear(8, minute=1), unclear(7), photo(1)]
    values = {7: (10, 0), 8: (200, 0), 9: (200, 0), 1: (200, 0)}
    result = run_fixed(photos, values)
    assert [p.issue for p in photos] == [
        Issue.DUPLICATED,
        Issue.UNCLEAR_LOCATION,
        Issue.BLURRY,
        None,
    ]
    assert [p.duplicate_of_attachment_id for p in photos] == [8, None, None, None]
    assert [p.place_id for p in photos] == [None, None, None, "p1"]
    assert all(p.latitude is None for p in photos[:3])
    assert result == {7: 10, 8: 200, 9: 200, 1: 200}


def test_location_unknown_rerun_restores_unclear_location():
    photos = [unclear(9, minute=1), unclear(8)]
    values = {8: (10, 0), 9: (200, 1)}
    run_fixed(photos, values)
    assert [p.issue for p in photos] == [Issue.UNCLEAR_LOCATION, Issue.BLURRY]
    run_fixed(photos, values, blur=0)
    assert [p.issue for p in photos] == [Issue.DUPLICATED, Issue.UNCLEAR_LOCATION]
    assert photos[0].duplicate_of_attachment_id == 8
    run_fixed(photos, values, blur=0, distance=0)
    assert all(p.issue is Issue.UNCLEAR_LOCATION for p in photos)
    assert all(p.duplicate_of_attachment_id is None and p.place_id is None for p in photos)


def test_all_blurry_keep_place():
    photos = [photo(1), photo(2)]
    result = run_fixed(photos, {1: (0, 0), 2: (0, 0)})
    assert result == {1: 0, 2: 0}
    assert all(p.issue is Issue.BLURRY and p.place_id == "p1" for p in photos)
    assert run_fixed([], {}) == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blur_var_min": -1},
        {"blur_var_min": float("nan")},
        {"blur_var_min": float("inf")},
        {"phash_max_dist": -1},
        {"phash_max_dist": 65},
        {"phash_max_dist": 1.5},
        {"phash_max_dist": True},
    ],
)
def test_invalid_thresholds_fail(kwargs):
    settings = {"blur_var_min": 100, "phash_max_dist": 5} | kwargs
    with pytest.raises(PipelineError):
        filtering.filter_photos([], {}, **settings)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_metrics",
        "not_metrics",
        "nan_variance",
        "negative_variance",
        "short_phash",
        "prefixed_phash",
        "missing_place",
        "empty_place",
        "unclear_with_place",
        "missing_unclear_metrics",
        "pending",
        "unknown_with_time",
        "known_without_time",
        "naive_time",
        "duplicate_id",
    ],
)
def test_invalid_inputs_fail_without_partial_classification(problem):
    photos = [photo(1), photo(2)]
    metrics = {
        1: filtering.PhotoMetrics(200.0, "0" * 16),
        2: filtering.PhotoMetrics(200.0, "f" * 16),
    }
    if problem == "missing_metrics":
        del metrics[2]
    elif problem == "not_metrics":
        metrics[2] = (200.0, "f" * 16)
    elif problem == "nan_variance":
        metrics[2] = filtering.PhotoMetrics(float("nan"), "f" * 16)
    elif problem == "negative_variance":
        metrics[2] = filtering.PhotoMetrics(-1.0, "f" * 16)
    elif problem == "short_phash":
        metrics[2] = filtering.PhotoMetrics(200.0, "f" * 4)
    elif problem == "prefixed_phash":
        metrics[2] = filtering.PhotoMetrics(200.0, "0x" + "f" * 14)
    elif problem == "missing_place":
        photos[1].place_id = None
    elif problem == "empty_place":
        photos[1].place_id = ""
    elif problem == "unclear_with_place":
        photos[1].issue = Issue.UNCLEAR_LOCATION
    elif problem == "missing_unclear_metrics":
        photos.append(unclear(3))
    elif problem == "pending":
        photos[1].time_status = TimeStatus.PENDING
    elif problem == "unknown_with_time":
        photos[1].time_status = TimeStatus.UNKNOWN
    elif problem == "known_without_time":
        photos[1].taken_at = None
    elif problem == "naive_time":
        photos[1].taken_at = BASE.replace(tzinfo=None)
    elif problem == "duplicate_id":
        photos[1] = photo(1)
    before = [(p.issue, p.duplicate_of_attachment_id) for p in photos]
    with pytest.raises(PipelineError):
        filtering.filter_photos(photos, metrics, blur_var_min=100, phash_max_dist=5)
    assert [(p.issue, p.duplicate_of_attachment_id) for p in photos] == before
