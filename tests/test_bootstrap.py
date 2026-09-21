"""조립 함수 중 컨테이너 환경에서만 드러나는 것. Qdrant 준비 대기"""

import httpx
import pytest

from app.infra.qdrant import MEMORY
from app.pipeline import bootstrap


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
