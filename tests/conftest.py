"""모든 테스트에서 Langfuse 기록을 끈다.

개발자의 .env나 CI 환경 변수에 키가 있어도 외부로 보내지 않는다. Langfuse를 확인하는 테스트는
tests/test_langfuse_tracing.py의 exporter 픽스처로 메모리 exporter를 쓴다.
"""

import pytest

from app.observability import langfuse_tracing


@pytest.fixture(autouse=True)
def _langfuse_off(monkeypatch):
    monkeypatch.setattr(langfuse_tracing, "_client", None)
    monkeypatch.setattr(langfuse_tracing, "_ready", True)
