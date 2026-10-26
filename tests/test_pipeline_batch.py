"""업로드 묶음 임베딩(embed_batch). 저장 범위와 실패 처리"""

import logging

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import DownloadFailed, PipelineError
from app.pipeline.batch import embed_batch
from app.pipeline.bootstrap import build_context
from tests.test_pipeline import FixedEngine


@pytest.fixture
def batch(tmp_path):
    def build(count=3):
        rng = np.random.default_rng(7)
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
        engine = FixedEngine(count)
        photos = {100 + i: f"{i}.png" for i in range(count)}
        return photos, build_context(settings, engine), engine

    return build


def test_embeds_and_stores_every_photo(batch, tmp_path):
    photos, ctx, engine = batch()
    assert embed_batch(photos, ctx) == [100, 101, 102]
    assert engine.batches == [3]
    # 엔진은 사본 bytes를 재인코딩 없이 받는다.
    assert engine.images == [(tmp_path / f"{i}.png").read_bytes() for i in range(3)]
    assert sorted(ctx.qdrant.fetch([100, 101, 102])) == [100, 101, 102]


def test_decode_failure_skips_only_that_photo(batch, tmp_path, caplog):
    photos, ctx, engine = batch()
    (tmp_path / "1.png").write_bytes(b"not an image")
    with caplog.at_level(logging.WARNING, logger="app.pipeline"):
        assert embed_batch(photos, ctx) == [100, 102]
    assert engine.batches == [2]
    assert sorted(ctx.qdrant.fetch([100, 101, 102])) == [100, 102]
    (record,) = [
        r for r in caplog.records if getattr(r, "event", None) == "ai_photo_batch_decode_failed"
    ]
    assert (record.failed_count, record.failed_ids) == (1, [101])


def test_all_decode_failures_skip_engine(batch, tmp_path):
    photos, ctx, engine = batch(2)
    for i in range(2):
        (tmp_path / f"{i}.png").write_bytes(b"broken")
    assert embed_batch(photos, ctx) == []
    assert engine.batches == []
    assert ctx.qdrant.fetch([100, 101]) == {}


def test_download_failure_propagates_without_storage(batch, tmp_path):
    photos, ctx, engine = batch()
    (tmp_path / "2.png").unlink()
    with pytest.raises(DownloadFailed):
        embed_batch(photos, ctx)
    assert engine.batches == []
    assert ctx.qdrant.fetch([100, 101, 102]) == {}


def test_invalid_engine_output_is_not_stored(batch, monkeypatch):
    photos, ctx, engine = batch()
    monkeypatch.setattr(
        engine, "encode_images", lambda images: np.zeros((len(images), 3), dtype=np.float32)
    )
    with pytest.raises(PipelineError):
        embed_batch(photos, ctx)
    assert ctx.qdrant.fetch([100, 101, 102]) == {}
