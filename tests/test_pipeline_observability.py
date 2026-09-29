"""단계별 사진 상태 관측. 집계와 직렬화, run이 단계마다 남기는 스냅샷"""

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.pipeline.bootstrap import build_context
from app.pipeline.observability import serialize, summarize
from app.pipeline.run import run
from app.pipeline.state import PhotoState, TimeStatus, build_photo_states
from app.schemas.process import Issue, ProcessRequest, RegionOrigin
from app.schemas.task import ProcessStep

PIPELINE_LOGGER = "app.pipeline"


class FixedEngine:
    """같은 벡터를 돌려주는 엔진. 판정이 임베딩 난수에 흔들리지 않게"""

    model_version = "siglip2-so400m-naflex"
    dim = 3

    def __init__(self, count: int) -> None:
        self.vectors = np.tile([1, 0, 0], (count, 1)).astype(np.float32)
        self.position = 0

    def encode_images(self, images):  # type: ignore[no-untyped-def]
        start = self.position
        self.position += len(images)
        return self.vectors[start : self.position].copy()


def build_request(count: int, gps: list[bool] | None = None) -> ProcessRequest:
    flags = gps or [True] * count
    attachments = [
        {
            "trip_attachment_id": 100 + i,
            "analyze_storage_key": f"{i}.png",
            "taken_at": (datetime(2026, 9, 10, tzinfo=UTC) + timedelta(minutes=i)).isoformat(),
            "latitude": 37.5 if flags[i] else None,
            "longitude": 127 if flags[i] else None,
            "device_model": "reference",
        }
        for i in range(count)
    ]
    return ProcessRequest.model_validate(
        {
            "trip_name": "관측",
            "period": {"start_date": "2026-09-10", "end_date": "2026-09-12"},
            "regions": [{"latitude": 37.5, "longitude": 127}],
            "attachments": attachments,
        }
    )


@pytest.fixture
def setup(tmp_path):  # type: ignore[no-untyped-def]
    def build(count: int = 3, gps: list[bool] | None = None):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(41)
        for i in range(count):
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                tmp_path / f"{i}.png"
            )
        settings = Settings(
            _env_file=None,
            API_KEY="test",
            IMAGE_SOURCE="local",
            IMAGE_DIR=tmp_path,
            QDRANT_URL=":memory:",
            FAKE_PIPELINE=False,
        )
        ctx = build_context(settings, FixedEngine(count))
        return build_request(count, gps), ctx

    return build


def photos_with_gps(gps: list[bool]) -> list[PhotoState]:
    return build_photo_states(build_request(len(gps), gps))


# ---------- 집계 ----------


def test_summarize_counts_origin_and_issue():
    photos = photos_with_gps([True, False])
    photos[1].issue = Issue.UNCLEAR_LOCATION

    summary = summarize(photos)

    assert summary["origin"] == {"EXIF": 1, "UNKNOWN": 1}
    assert summary["issue"] == {"none": 1, "UNCLEAR_LOCATION": 1}
    assert summary["with_coords"] == 1
    assert summary["with_embedding"] == 0


def test_summarize_counts_places_without_duplicates():
    photos = photos_with_gps([True, True, True])
    photos[0].place_id = "p1"
    photos[1].place_id = "p1"
    photos[2].place_id = "p2"

    assert summarize(photos)["place_count"] == 2


def test_summarize_counts_time_status():
    photos = photos_with_gps([True, True])
    photos[0].time_status = TimeStatus.CORRECTED

    assert summarize(photos)["time_status"] == {"CORRECTED": 1, "PENDING": 1}


# ---------- 직렬화 ----------


def test_serialize_omits_embedding_values():
    photos = photos_with_gps([True])
    photos[0].embedding = np.ones(4, dtype=np.float32)

    line = serialize(photos[0])

    assert line["has_embedding"] is True
    assert "embedding" not in line


def test_serialize_omits_coordinates_and_storage_key():
    """좌표는 개인 위치정보라 값 대신 보유 여부만, 사본 키는 담지 않음"""
    photos = photos_with_gps([True])

    line = serialize(photos[0])

    assert line["has_coords"] is True
    assert "latitude" not in line and "longitude" not in line
    assert "analyze_storage_key" not in line


def test_serialize_carries_classification_fields():
    photos = photos_with_gps([True])
    photos[0].place_id = "p1"
    photos[0].issue = Issue.DUPLICATED
    photos[0].duplicate_of_attachment_id = 100
    photos[0].region_origin = RegionOrigin.INFERRED

    line = serialize(photos[0])

    assert line["place_id"] == "p1"
    assert line["issue"] == "DUPLICATED"
    assert line["duplicate_of"] == 100
    assert line["origin"] == "INFERRED"


# ---------- run이 남기는 스냅샷 ----------


def snapshots(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == "ai_pipeline_snapshot"]


def test_run_records_snapshot_for_every_step(setup, caplog: pytest.LogCaptureFixture):  # type: ignore[no-untyped-def]
    request, ctx = setup(3)

    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER):
        run(request, ctx, lambda step, done, total: None, lambda: False)

    steps = [r.step for r in snapshots(caplog)]
    assert steps == [
        "download",
        "embedding",
        "clock_correction",
        "locating",
        "clustering",
        "filtering",
        "finalizing",
    ]


def test_snapshot_shows_embedding_filled_after_embedding_step(setup, caplog):  # type: ignore[no-untyped-def]
    request, ctx = setup(2)

    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER):
        run(request, ctx, lambda step, done, total: None, lambda: False)

    by_step = {r.step: r for r in snapshots(caplog)}
    assert by_step["download"].with_embedding == 0
    assert by_step["embedding"].with_embedding == 2


def test_photo_state_lines_only_at_debug(setup, caplog: pytest.LogCaptureFixture):  # type: ignore[no-untyped-def]
    request, ctx = setup(2)

    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER):
        run(request, ctx, lambda step, done, total: None, lambda: False)
    assert [r for r in caplog.records if getattr(r, "event", None) == "ai_photo_state"] == []

    caplog.clear()
    request, ctx = setup(2)
    with caplog.at_level(logging.DEBUG, logger=PIPELINE_LOGGER):
        run(request, ctx, lambda step, done, total: None, lambda: False)
    lines = [r for r in caplog.records if getattr(r, "event", None) == "ai_photo_state"]

    assert len(lines) == 2 * 7  # 사진 수 곱하기 단계 수
    assert {r.step for r in lines} == {
        "download",
        "embedding",
        "clock_correction",
        "locating",
        "clustering",
        "filtering",
        "finalizing",
    }


def test_snapshot_tracks_place_assignment(setup, caplog: pytest.LogCaptureFixture):  # type: ignore[no-untyped-def]
    """CLUSTERING 전에는 폴더가 없고 그 뒤에 생김"""
    request, ctx = setup(2)

    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER):
        run(request, ctx, lambda step, done, total: None, lambda: False)

    by_step = {r.step: r for r in snapshots(caplog)}
    assert by_step["locating"].place_count == 0
    assert by_step["clustering"].place_count >= 1


def test_snapshot_uses_process_step_names():
    """로그 단계 이름은 클라우드 계약값이며 ProcessStep 일곱 개와 1대1"""
    from app.core.logging import STEP_NAMES

    assert set(STEP_NAMES) == {step.value for step in ProcessStep}
