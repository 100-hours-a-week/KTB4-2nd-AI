"""가짜 파이프라인과 조립. fake_run이 run.py 규칙과 ProcessResult 검증기를 지키는지"""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import DecodeFailed, DownloadFailed, PipelineCancelled
from app.engines.fake import FakeEngine
from app.infra.qdrant import MEMORY
from app.pipeline.bootstrap import build_context, build_engine, select_run
from app.pipeline.context import PipelineContext
from app.pipeline.fake_run import fake_run
from app.schemas.process import Issue, ProcessRequest, RegionOrigin
from app.schemas.task import ProcessStep

STEPS = list(ProcessStep)


def write_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 48), color).save(path, "JPEG")


def make_request(keys: list[str], located: list[bool]) -> ProcessRequest:
    attachments = []
    for i, (key, has_gps) in enumerate(zip(keys, located, strict=True)):
        attachments.append(
            {
                "trip_attachment_id": 100 + i,
                "analyze_storage_key": key,
                "taken_at": f"2026-10-12T0{6 + i}:00:00+09:00",
                "latitude": 33.45 + i * 0.001 if has_gps else None,
                "longitude": 126.94 if has_gps else None,
                "device_model": "Apple iPhone 15",
            }
        )
    return ProcessRequest.model_validate(
        {
            "trip_name": "제주",
            "period": {"start_date": "2026-10-12", "end_date": "2026-10-14"},
            "regions": [{"latitude": 33.4996, "longitude": 126.5312}],
            "attachments": attachments,
        }
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        API_KEY="t", IMAGE_SOURCE="local", IMAGE_DIR=tmp_path, QDRANT_URL=MEMORY, FAKE_PIPELINE=True
    )


@pytest.fixture
def ctx(settings: Settings, tmp_path: Path) -> PipelineContext:
    write_jpeg(tmp_path / "a.jpg", (200, 30, 30))
    write_jpeg(tmp_path / "b.jpg", (30, 200, 30))
    write_jpeg(tmp_path / "c.jpg", (30, 30, 200))
    return build_context(settings, build_engine(settings))


def no_cancel() -> bool:
    return False


# ---------- 조립 ----------


def test_bootstrap_fake_mode(settings: Settings, ctx: PipelineContext):
    assert isinstance(ctx.engine, FakeEngine)
    assert ctx.qdrant.collection == "vectors_fake"
    assert ctx.thresholds == {}
    assert select_run(settings) is fake_run


def test_select_run_real_mode(settings: Settings):
    from app.pipeline.run import run

    real = settings.model_copy(update={"FAKE_PIPELINE": False})
    assert select_run(real) is run


# ---------- 결과 ----------


def test_result_shape_and_vectors(ctx: PipelineContext):
    request = make_request(["a.jpg", "b.jpg", "c.jpg"], [True, True, False])
    result = fake_run(request, ctx, lambda *_: None, no_cancel)

    assert len(result.places) == 1
    place = result.places[0]
    assert place.place_id == "p1"
    assert place.representative_attachment_id == 100
    assert [a.trip_attachment_id for a in place.attachments] == [100, 101]
    assert all(a.region_origin is RegionOrigin.EXIF for a in place.attachments)
    assert place.first_taken_at == datetime(2026, 10, 11, 21, 0, tzinfo=UTC)
    assert place.last_taken_at == datetime(2026, 10, 11, 22, 0, tzinfo=UTC)

    assert [u.trip_attachment_id for u in result.unclassified] == [102]
    assert result.unclassified[0].issue is Issue.UNCLEAR_LOCATION
    assert result.unclassified[0].region_origin is RegionOrigin.UNKNOWN
    assert result.failed == [] and result.clock_offsets == []

    vectors = ctx.qdrant.fetch([100, 101, 102])
    assert set(vectors) == {100, 101, 102}
    assert all(v.shape == (1152,) for v in vectors.values())


def test_all_unlocated_gives_no_place(ctx: PipelineContext):
    request = make_request(["a.jpg"], [False])
    result = fake_run(request, ctx, lambda *_: None, no_cancel)
    assert result.places == []
    assert len(result.unclassified) == 1


# ---------- 진행률과 취소 ----------


def test_progress_order(ctx: PipelineContext):
    seen: list[tuple[ProcessStep, int, int]] = []
    request = make_request(["a.jpg", "b.jpg", "c.jpg"], [True, True, True])
    fake_run(request, ctx, lambda s, d, t: seen.append((s, d, t)), no_cancel)

    steps_in_order = []
    for step, _, _ in seen:
        if not steps_in_order or steps_in_order[-1] is not step:
            steps_in_order.append(step)
    assert steps_in_order == STEPS
    assert seen[0] == (ProcessStep.DOWNLOADING, 3, 3)
    assert seen[-1] == (ProcessStep.FINALIZING, 3, 3)
    assert all(t == 3 for _, _, t in seen)


def test_cancel_between_steps(ctx: PipelineContext):
    calls = 0

    def cancel_after_first() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    request = make_request(["a.jpg"], [True])
    with pytest.raises(PipelineCancelled):
        fake_run(request, ctx, lambda *_: None, cancel_after_first)


# ---------- 실패 ----------


def test_download_failed_passes_through(ctx: PipelineContext):
    request = make_request(["a.jpg", "missing.jpg"], [True, True])
    with pytest.raises(DownloadFailed) as exc:
        fake_run(request, ctx, lambda *_: None, no_cancel)
    assert exc.value.keys == ["missing.jpg"]


def test_decode_failed_collects_ids(ctx: PipelineContext, tmp_path: Path):
    (tmp_path / "bad.jpg").write_bytes(b"not a jpeg")
    request = make_request(["a.jpg", "bad.jpg"], [True, True])
    with pytest.raises(DecodeFailed) as exc:
        fake_run(request, ctx, lambda *_: None, no_cancel)
    assert exc.value.ids == [101]


# ---------- 엔진 ----------


def test_fake_engine_deterministic_and_normalized():
    engine = FakeEngine()
    red = Image.new("RGB", (64, 48), (200, 30, 30))
    blue = Image.new("RGB", (64, 48), (30, 30, 200))

    first = engine.encode_images([red, blue])
    second = engine.encode_images([red, blue])
    assert first.shape == (2, 1152) and first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert not np.allclose(first[0], first[1])
    assert engine.encode_images([]).shape == (0, 1152)
