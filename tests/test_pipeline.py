import threading
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import DecodeFailed, DownloadFailed, PipelineCancelled, PipelineError
from app.pipeline.bootstrap import build_context, select_run
from app.pipeline.run import run
from app.schemas.process import Issue, ProcessRequest, RegionOrigin
from app.schemas.task import ProcessStep


class FixedEngine:
    model_version = "siglip2-so400m-naflex"
    dim = 3

    def __init__(self, count):
        self.vectors = np.tile([1, 0, 0], (count, 1)).astype(np.float32)
        self.position = 0
        self.batches = []
        self.images = []

    def encode_images(self, images):
        self.batches.append(len(images))
        self.images.extend(images)
        start = self.position
        self.position += len(images)
        return self.vectors[start : self.position].copy()


@pytest.fixture
def setup(tmp_path):
    def build(count=3):
        rng = np.random.default_rng(41)
        attachments = []
        for i in range(count):
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                tmp_path / f"{i}.png"
            )
            attachments.append(
                dict(
                    trip_attachment_id=100 + i,
                    analyze_storage_key=f"{i}.png",
                    taken_at=(datetime(2026, 9, 10, tzinfo=UTC) + timedelta(minutes=i)).isoformat(),
                    latitude=37.5,
                    longitude=127,
                    device_model="reference",
                )
            )
        request = ProcessRequest.model_validate(
            dict(
                trip_name="통합",
                period={"start_date": "2026-09-10", "end_date": "2026-09-12"},
                regions=[{"latitude": 37.5, "longitude": 127}],
                attachments=attachments,
            )
        )
        settings = Settings(
            _env_file=None,
            API_KEY="test",
            IMAGE_SOURCE="local",
            IMAGE_DIR=tmp_path,
            QDRANT_URL=":memory:",
            FAKE_PIPELINE=False,
        )
        engine = FixedEngine(count)
        ctx = build_context(settings, engine)
        ctx.thresholds.update(blur_var_min=100, phash_max_dist=0, inherit_min_sim=0.8)
        return request, ctx, engine

    return build


def test_real_stages_results_storage_and_progress(setup, tmp_path):
    request, ctx, engine = setup(6)
    ids = [50, 10, 70, 20, 90, 30]
    for attachment, photo_id in zip(request.attachments, ids, strict=True):
        attachment.trip_attachment_id = photo_id
    (tmp_path / "1.png").write_bytes((tmp_path / "0.png").read_bytes())
    Image.new("RGB", (32, 32), "gray").save(tmp_path / "2.png")
    for index in (1, 4):
        request.attachments[index].latitude = request.attachments[index].longitude = None
    request.attachments[3].latitude = 35
    request.attachments[5].taken_at = datetime(2026, 8, 1, tzinfo=UTC)
    engine.vectors[3] = [0, 1, 0]
    engine.vectors[4] = [0, 0, 1]
    before = request.model_dump()
    progress = []

    result = select_run(ctx.settings)(
        request, ctx, lambda *args: progress.append(args), lambda: False
    )

    assert len(result.places) == 2
    assert {a.trip_attachment_id for p in result.places for a in p.attachments} == {50, 20}
    excluded = {p.trip_attachment_id: p for p in result.unclassified}
    assert excluded[10].issue is Issue.DUPLICATED and excluded[10].duplicate_of_attachment_id == 50
    assert excluded[10].region_origin is RegionOrigin.INFERRED
    assert excluded[70].issue is Issue.BLURRY
    assert excluded[90].issue is excluded[30].issue is Issue.UNCLEAR_LOCATION
    assert excluded[70].place_id == excluded[10].place_id
    assert result.failed == [] and request.model_dump() == before
    stored = ctx.qdrant.fetch(ids)
    for photo_id, vector in zip(ids, engine.vectors, strict=True):
        np.testing.assert_allclose(stored[photo_id], vector)
    assert list(dict.fromkeys(step for step, _, _ in progress)) == list(ProcessStep)
    for step in ProcessStep:
        done = [d for s, d, _ in progress if s is step]
        assert done[0] == 0 and done[-1] == 6 and done == sorted(done)
    assert all(total == 6 for _, _, total in progress)
    with pytest.raises(ValueError):
        engine.images[0].getpixel((0, 0))  # 성공 후에도 디코딩 자원을 닫는다.


def test_clock_correction_precedes_period_check_and_gps_inheritance(setup):
    request, ctx, engine = setup(6)
    engine.vectors = np.vstack([np.eye(3), np.eye(3)]).astype(np.float32)
    for i in range(3, 6):
        target = request.attachments[i]
        target.device_model = "wrong-date-camera"
        target.latitude = target.longitude = None
        target.taken_at = request.attachments[i - 3].taken_at - timedelta(days=3)
    result = run(request, ctx, lambda *_: None, lambda: False)
    assert len(result.clock_offsets) == 1
    offset = result.clock_offsets[0]
    assert (offset.device_model, offset.days, offset.matched_pairs) == ("wrong-date-camera", 3, 3)
    assert not result.unclassified
    by_id = {a.trip_attachment_id: a for a in result.places[0].attachments}
    for i in range(3, 6):
        assert by_id[100 + i].taken_at == request.attachments[i - 3].taken_at
        assert by_id[100 + i].region_origin is RegionOrigin.INFERRED


def test_batch_boundary_and_all_unlocated(setup):
    request, ctx, engine = setup(17)
    for attachment in request.attachments:
        attachment.latitude = attachment.longitude = None
    progress = []
    result = run(request, ctx, lambda *args: progress.append(args), lambda: False)
    assert engine.batches == [16, 1]
    assert [done for step, done, _ in progress if step is ProcessStep.EMBEDDING] == [0, 16, 17]
    assert not result.places and len(result.unclassified) == 17
    assert len(ctx.qdrant.fetch([a.trip_attachment_id for a in request.attachments])) == 17


@pytest.mark.parametrize("step", list(ProcessStep))
@pytest.mark.parametrize("at_start", [True, False])
def test_cancel_at_step_boundary_never_returns_result(setup, step, at_start):
    request, ctx, _ = setup()
    cancel = threading.Event()

    def callback(current, done, total):
        if current is step and done == (0 if at_start else total):
            cancel.set()

    with pytest.raises(PipelineCancelled):
        run(request, ctx, callback, cancel.is_set)


def test_cancel_between_embedding_batches_retains_only_completed_upserts(setup):
    request, ctx, engine = setup(17)
    cancel = threading.Event()

    def callback(step, done, total):
        if step is ProcessStep.EMBEDDING and done == 16:
            cancel.set()

    with pytest.raises(PipelineCancelled):
        run(request, ctx, callback, cancel.is_set)
    assert engine.batches == [16]
    assert len(ctx.qdrant.fetch(list(range(100, 117)))) == 16


def test_cancel_before_start_and_after_model_before_upsert(setup, monkeypatch):
    request, ctx, engine = setup()
    with pytest.raises(PipelineCancelled):
        run(request, ctx, lambda *_: None, lambda: True)
    assert not engine.batches
    cancel = threading.Event()
    encode = engine.encode_images

    def cancel_encode(images):
        vectors = encode(images)
        cancel.set()
        return vectors

    monkeypatch.setattr(engine, "encode_images", cancel_encode)
    with pytest.raises(PipelineCancelled):
        run(request, ctx, lambda *_: None, cancel.is_set)
    assert ctx.qdrant.fetch([100, 101, 102]) == {}


def test_download_error_and_decode_ids_across_batches(setup, tmp_path):
    request, ctx, engine = setup(17)
    (tmp_path / "0.png").unlink()
    with pytest.raises(DownloadFailed) as error:
        run(request, ctx, lambda *_: None, lambda: False)
    assert error.value.keys == ["0.png"]
    (tmp_path / "0.png").write_bytes(b"broken")
    (tmp_path / "16.png").write_bytes(b"broken")
    with pytest.raises(DecodeFailed) as error:
        run(request, ctx, lambda *_: None, lambda: False)
    assert error.value.ids == [100, 116]
    assert not engine.batches


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "zero", "not_array", "exception"])
def test_invalid_engine_output_fails_before_storage(setup, monkeypatch, kind):
    request, ctx, engine = setup()
    bad = {
        "shape": np.ones((2, 3), dtype=np.float32),
        "dtype": np.eye(3),
        "nan": np.full((3, 3), np.nan, dtype=np.float32),
        "zero": np.zeros((3, 3), dtype=np.float32),
        "not_array": [[1, 0, 0]] * 3,
    }

    def encode(images):
        if kind == "exception":
            raise RuntimeError("model error")
        return bad[kind]

    monkeypatch.setattr(engine, "encode_images", encode)
    with pytest.raises(PipelineError):
        run(request, ctx, lambda *_: None, lambda: False)
    assert ctx.qdrant.fetch([100, 101, 102]) == {}


@pytest.mark.parametrize(
    "kind", ["duplicate_ids", "period", "thresholds", "qdrant", "callback", "all_blurry"]
)
def test_internal_failures(setup, monkeypatch, tmp_path, kind):
    request, ctx, engine = setup()

    def callback(*args):
        pass

    def broken(*args):
        raise OSError("dependency failure")

    if kind == "duplicate_ids":
        request.attachments[1].trip_attachment_id = 100
    elif kind == "period":
        request.period.end_date = request.period.start_date - timedelta(days=1)
    elif kind == "thresholds":
        ctx.thresholds.clear()
    elif kind == "qdrant":
        monkeypatch.setattr(ctx.qdrant, "upsert", broken)
    elif kind == "callback":
        callback = broken
    elif kind == "all_blurry":
        for i in range(3):
            Image.new("RGB", (32, 32), "gray").save(tmp_path / f"{i}.png")
    with pytest.raises(PipelineError):
        run(request, ctx, callback, lambda: False)
    for image in engine.images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_heartbeat_during_blocked_model_without_fabricating_progress(setup, monkeypatch):
    request, ctx, engine = setup(1)
    ctx.settings.WORKER_DEAD_SEC = 1
    heartbeat = threading.Event()
    progress = []
    encode = engine.encode_images

    def slow_encode(images):
        assert heartbeat.wait(5), "모델 호출 중 진행 보고가 없음"
        return encode(images)

    def callback(step, done, total):
        progress.append((step, done, total))
        if progress.count((ProcessStep.EMBEDDING, 0, 1)) >= 2:
            heartbeat.set()

    monkeypatch.setattr(engine, "encode_images", slow_encode)
    result = run(request, ctx, callback, lambda: False)
    assert len(result.places) == 1 and heartbeat.is_set()


def test_local_script_forces_real_pipeline_and_writes_json(setup, tmp_path, monkeypatch, capsys):
    from app.pipeline import bootstrap
    from scripts.run_pipeline import main

    request, _, engine = setup(1)
    path = tmp_path / "request.json"
    output = tmp_path / "result.json"
    path.write_text(request.model_dump_json(), encoding="utf-8")
    seen = []

    def build(settings):
        seen.append(settings)
        return engine

    monkeypatch.setenv("FAKE_PIPELINE", "true")
    monkeypatch.setattr(bootstrap, "build_engine", build)
    assert main([str(path), "--image-dir", str(tmp_path), "--output", str(output)]) == 0
    from app.schemas.process import ProcessResult

    result = ProcessResult.model_validate_json(output.read_text())
    assert len(result.places) == 1 and result.places[0].attachments[0].evaluation != 80
    assert seen[0].FAKE_PIPELINE is False and seen[0].QDRANT_URL == ":memory:"
    assert "FINALIZING 1/1" in capsys.readouterr().err


def test_local_script_failure_returns_nonzero_and_preserves_output(tmp_path, capsys):
    from scripts.run_pipeline import main

    path = tmp_path / "request.json"
    output = tmp_path / "result.json"
    path.write_text("{}")
    output.write_text("previous result")
    assert main([str(path), "--image-dir", str(tmp_path), "--output", str(output)]) == 1
    assert output.read_text() == "previous result"
    assert "실패:" in capsys.readouterr().err
