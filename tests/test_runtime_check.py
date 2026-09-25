"""점검이 실제 엔진 생성 경로와 반환 계약을 사용하고 실패를 알리는지 확인한다."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.pipeline import bootstrap
from scripts import runtime_check


@pytest.fixture
def runtime_engine(monkeypatch: pytest.MonkeyPatch):
    output = np.zeros((1, runtime_check.DIM), dtype=np.float32)
    output[0, 0] = 1.0
    seen = SimpleNamespace(settings=None, images=None, output=output)

    def encode_images(images):
        seen.images = images
        return seen.output

    def build_engine(settings):
        seen.settings = settings
        return SimpleNamespace(encode_images=encode_images)

    monkeypatch.setattr(bootstrap, "build_engine", build_engine)
    return seen


def test_model_check_uses_real_mode_and_requested_settings(monkeypatch, runtime_engine, tmp_path):
    monkeypatch.setenv("FAKE_PIPELINE", "true")
    monkeypatch.setenv("EMBED_DEVICE", "cuda")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)

    runtime_check.check_model(str(tmp_path))

    settings = runtime_engine.settings
    assert settings.FAKE_PIPELINE is False
    assert settings.MODEL_PATH == tmp_path
    assert settings.EMBED_DEVICE == "cuda"
    assert settings.IMAGE_SOURCE == "local"
    assert len(runtime_engine.images) == 1
    assert runtime_engine.images[0].mode == "RGB"


@pytest.mark.parametrize(
    ("kind", "message"),
    [("shape", "모양"), ("dtype", "dtype"), ("nan", "유한"), ("zero", "norm"), ("scaled", "norm")],
)
def test_model_check_rejects_invalid_output(runtime_engine, kind, message):
    if kind == "shape":
        runtime_engine.output = runtime_engine.output[:, :-1]
    elif kind == "dtype":
        runtime_engine.output = runtime_engine.output.astype(np.float64)
    elif kind == "nan":
        runtime_engine.output[0, 0] = np.nan
    elif kind == "zero":
        runtime_engine.output[:] = 0
    else:
        runtime_engine.output *= 2

    with pytest.raises(RuntimeError, match=message):
        runtime_check.check_model("unused")


def test_main_returns_failure_for_invalid_model_output(monkeypatch, runtime_engine, capsys):
    runtime_engine.output *= 2
    monkeypatch.setattr(runtime_check, "check_qdrant", lambda _: pytest.fail("모델 실패 후 호출됨"))
    monkeypatch.setenv("QDRANT_URL", "http://q:6333")

    assert runtime_check.main([]) == 1
    assert "실패:" in capsys.readouterr().err


@pytest.mark.parametrize("skip", [False, True])
def test_main_checks_qdrant_unless_skipped(monkeypatch, runtime_engine, tmp_path, skip):
    calls = []
    monkeypatch.setattr(runtime_check, "check_qdrant", calls.append)
    monkeypatch.setenv("QDRANT_URL", "http://q:6333")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))

    assert runtime_check.main(["--skip-qdrant"] if skip else []) == 0
    assert calls == ([] if skip else ["http://q:6333"])
    assert runtime_engine.settings.MODEL_PATH == Path(tmp_path)


def test_main_reports_engine_load_failure(monkeypatch, capsys):
    def fail(settings):
        raise OSError("model missing")

    monkeypatch.setattr(bootstrap, "build_engine", fail)

    assert runtime_check.main(["--skip-qdrant"]) == 1
    assert "model missing" in capsys.readouterr().err
