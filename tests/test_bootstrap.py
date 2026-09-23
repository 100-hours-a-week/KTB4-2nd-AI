"""엔진 선택과 설정 전달, Qdrant 준비 대기를 검증한다."""

import httpx
import pytest

from app.core.config import Settings
from app.engines.fake import FakeEngine
from app.infra.qdrant import MEMORY
from app.pipeline import bootstrap


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_build_engine_real_mode_forwards_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path, device
):
    from app.engines.siglip_naflex import SiglipNaflexEngine

    received = {}

    def fake_init(self, model_path, device):
        received.update(model_path=model_path, device=device)

    monkeypatch.setattr(SiglipNaflexEngine, "__init__", fake_init)
    settings = Settings(
        _env_file=None,
        API_KEY="test",
        IMAGE_SOURCE="local",
        FAKE_PIPELINE=False,
        MODEL_PATH=tmp_path / "model",
        EMBED_DEVICE=device,
    )

    engine = bootstrap.build_engine(settings)

    assert isinstance(engine, SiglipNaflexEngine)
    assert received == {"model_path": tmp_path / "model", "device": device}


def test_build_engine_fake_mode_does_not_load_real_engine(monkeypatch: pytest.MonkeyPatch):
    from app.engines.siglip_naflex import SiglipNaflexEngine

    def never(*args, **kwargs):
        raise AssertionError("가짜 모드에서 실제 모델을 적재하면 안 됨")

    monkeypatch.setattr(SiglipNaflexEngine, "__init__", never)
    settings = Settings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local", FAKE_PIPELINE=True)

    assert isinstance(bootstrap.build_engine(settings), FakeEngine)


def test_wait_for_qdrant_skips_memory(monkeypatch: pytest.MonkeyPatch):
    def never(*_, **__):
        raise AssertionError(":memory:는 HTTP를 부르면 안 됨")

    monkeypatch.setattr(bootstrap.httpx, "get", never)
    bootstrap.wait_for_qdrant(MEMORY)


def test_wait_for_qdrant_retries_until_ready(monkeypatch: pytest.MonkeyPatch):
    answers = [httpx.ConnectError("refused"), httpx.Response(503), httpx.Response(200)]
    calls = 0

    def fake_get(url: str, timeout: float) -> httpx.Response:
        nonlocal calls
        calls += 1
        outcome = answers.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(bootstrap.httpx, "get", fake_get)
    monkeypatch.setattr(bootstrap.time, "sleep", lambda _: None)
    bootstrap.wait_for_qdrant("http://q:6333", timeout=10, interval=0)
    assert calls == 3


def test_wait_for_qdrant_times_out(monkeypatch: pytest.MonkeyPatch):
    def refused(url: str, timeout: float) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(bootstrap.httpx, "get", refused)
    monkeypatch.setattr(bootstrap.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="준비되지 않음"):
        bootstrap.wait_for_qdrant("http://q:6333", timeout=0, interval=0)
