"""V3 Kafka SASL_SSL·SCRAM과 Qdrant API Key 설정이 실제 클라이언트 생성자까지 전달되는지 확인한다.

실제 브로커와 Qdrant 서버는 부르지 않는다. 인증과 인증서 거절은 Cloud가 V3 환경에서 검증한다.
"""

import logging
import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock

import certifi
import pytest
from aiokafka.errors import KafkaError
from pydantic import SecretStr, ValidationError

from app.jobs.config import JobsSettings
from app.jobs.kafka import KafkaRuntime, connection_options

pytestmark = pytest.mark.anyio

PASSWORD = "kafka-secret-value"
QDRANT_KEY = "qdrant-secret-value"
SASL = {
    "KAFKA_SECURITY_PROTOCOL": "SASL_SSL",
    "KAFKA_SASL_USERNAME": "v3-ai",
    "KAFKA_SASL_PASSWORD": PASSWORD,
    "KAFKA_SSL_CAFILE": certifi.where(),
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def settings(**changes) -> JobsSettings:
    return JobsSettings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local", **changes)


class Stop(BaseException):
    """연결 재시도 루프를 한 번만 돌리고 빠져나오기 위한 신호"""


def test_plaintext_default_keeps_previous_options():
    assert connection_options(settings()) == {
        "bootstrap_servers": "127.0.0.1:9092",
        "security_protocol": "PLAINTEXT",
    }


def test_sasl_ssl_options_verify_server_and_send_scram_credentials():
    options = connection_options(settings(**SASL))
    context = options.pop("ssl_context")
    assert options == {
        "bootstrap_servers": "127.0.0.1:9092",
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "SCRAM-SHA-512",
        "sasl_plain_username": "v3-ai",
        "sasl_plain_password": PASSWORD,
    }
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
    assert context.cert_store_stats()["x509_ca"] > 0


@pytest.mark.parametrize("name", ["KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD", "KAFKA_SSL_CAFILE"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_sasl_ssl_rejects_missing_or_blank_values(name, value):
    changes = {**SASL, name: value}
    if value is None:
        del changes[name]
    with pytest.raises(ValidationError, match=name):
        settings(**changes)


def test_unknown_protocol_and_mechanism_are_rejected():
    with pytest.raises(ValidationError):
        settings(KAFKA_SECURITY_PROTOCOL="SSL")
    with pytest.raises(ValidationError):
        settings(**SASL, KAFKA_SASL_MECHANISM="PLAIN")


def test_missing_ca_file_fails_instead_of_skipping_verification(tmp_path):
    cfg = settings(**{**SASL, "KAFKA_SSL_CAFILE": str(tmp_path / "missing.crt")})
    with pytest.raises(FileNotFoundError):
        connection_options(cfg)


def test_validation_errors_hide_secret_inputs():
    # 입력 dict 끝의 값은 오류 메시지의 input_value 요약에 그대로 남을 수 있다.
    changes = {
        k: v for k, v in SASL.items() if k not in ("KAFKA_SSL_CAFILE", "KAFKA_SASL_PASSWORD")
    }
    with pytest.raises(ValidationError, match="KAFKA_SSL_CAFILE") as error:
        settings(**changes, KAFKA_SASL_PASSWORD=PASSWORD)
    assert PASSWORD not in str(error.value)


def test_password_is_hidden_from_settings_repr():
    cfg = settings(**SASL, QDRANT_API_KEY=QDRANT_KEY)
    assert PASSWORD not in repr(cfg) and QDRANT_KEY not in repr(cfg)


async def test_producer_and_consumer_receive_same_security_options(monkeypatch, caplog):
    cfg = settings(**SASL)
    created = []
    # 각 루프의 첫 시도는 인증 실패로 재시도하고, 두 번째 시도에서 루프를 끝낸다.
    outcomes = iter([KafkaError("authentication failed"), Stop()] * 2)

    class Client:
        def __init__(self, *topics, **options):
            self.options = options
            created.append(self)

        async def start(self):
            raise next(outcomes)

        async def stop(self):
            pass

    monkeypatch.setattr("app.jobs.kafka.AIOKafkaProducer", Client)
    monkeypatch.setattr("app.jobs.kafka.AIOKafkaConsumer", Client)
    monkeypatch.setattr("app.jobs.kafka.asyncio.sleep", AsyncMock())
    runtime = KafkaRuntime(SimpleNamespace(cfg=cfg))
    caplog.set_level(logging.DEBUG)
    with pytest.raises(Stop):
        await runtime.connect_producer()
    with pytest.raises(Stop):
        await runtime.connect_consumer(cfg.KAFKA_TRIP_CREATE_REQUEST_TOPIC, "yeodam-ai-trip-create")
    assert len(created) == 4
    for client in created:
        assert client.options["security_protocol"] == "SASL_SSL"
        assert client.options["sasl_mechanism"] == "SCRAM-SHA-512"
        assert client.options["sasl_plain_username"] == "v3-ai"
        assert client.options["sasl_plain_password"] == PASSWORD
        assert isinstance(client.options["ssl_context"], ssl.SSLContext)
    # 메시지와 extra 필드를 모두 포함해 비밀번호가 남지 않는지 본다.
    logged = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert [r.event for r in caplog.records] == ["ai_kafka_connect_retry"] * 2
    assert PASSWORD not in logged


@pytest.mark.parametrize("value", [None, "", "  "])
def test_v3_foundation_requires_qdrant_api_key(value):
    changes = {} if value is None else {"QDRANT_API_KEY": value}
    with pytest.raises(ValidationError, match="QDRANT_API_KEY"):
        settings(JOBS_ENV="v3-foundation", **changes)
    assert settings(JOBS_ENV="v3-foundation", QDRANT_API_KEY=QDRANT_KEY).QDRANT_API_KEY


@pytest.mark.parametrize("env", ["local", "staging", "production"])
def test_other_environments_keep_unauthenticated_qdrant(env):
    assert settings(JOBS_ENV=env).QDRANT_API_KEY is None


class Recorder:
    """원격 Qdrant 클라이언트 대역. 생성자 인자만 기록한다."""

    def __init__(self, calls, *args, **kwargs):
        calls.append(kwargs)

    def collection_exists(self, name):
        return True

    def close(self):
        pass


@pytest.mark.parametrize(
    ("key", "sent"), [(SecretStr(QDRANT_KEY), QDRANT_KEY), (SecretStr(""), None), (None, None)]
)
def test_remote_clients_receive_api_key(monkeypatch, key, sent):
    from app.infra import qdrant

    calls = []
    monkeypatch.setattr(qdrant, "QdrantClient", lambda *a, **k: Recorder(calls, *a, **k))
    monkeypatch.setattr(qdrant, "AsyncQdrantClient", lambda *a, **k: Recorder(calls, *a, **k))
    url = "https://qdrant.qdrant.svc.cluster.local:6333"
    qdrant.QdrantStore(url, "m", 3, api_key=key)
    qdrant.AsyncQdrantVectors(url, "m", 3, 5, api_key=key)
    assert [call["api_key"] for call in calls] == [sent, sent]
    assert all(call["url"] == url for call in calls)
    assert "verify" not in calls[0] and "verify" not in calls[1]


async def test_memory_clients_ignore_api_key():
    import numpy as np

    from app.infra.qdrant import MEMORY, AsyncQdrantVectors, QdrantStore

    store = QdrantStore(MEMORY, "m", 3, api_key=SecretStr(QDRANT_KEY))
    store.upsert([1], np.ones((1, 3), dtype=np.float32))
    assert list(store.fetch([1])) == [1]
    store.close()
    vectors = AsyncQdrantVectors(MEMORY, "m", 3, api_key=SecretStr(QDRANT_KEY))
    assert await vectors.fetch([1]) == {}
    await vectors.aclose()


def capture(calls):
    def factory(*args, **kwargs):
        calls.append(kwargs)
        raise Stop

    return factory


def test_process_runner_passes_qdrant_api_key(monkeypatch):
    from app.jobs import process_handler

    cfg = settings(QDRANT_API_KEY=QDRANT_KEY, PROCESS_AGENT_ENABLED=False)
    calls = []
    monkeypatch.setattr(
        process_handler,
        "build_engine",
        lambda _: SimpleNamespace(model_version="m", dim=3, close=lambda: None),
    )
    monkeypatch.setattr(process_handler, "build_images", lambda _: object())
    monkeypatch.setattr(process_handler, "wait_for_qdrant", lambda _: None)
    monkeypatch.setattr(process_handler, "QdrantStore", capture(calls))
    with pytest.raises(Stop), process_handler.open_context(cfg):
        pass
    assert calls == [{"api_key": cfg.QDRANT_API_KEY}]


def test_pipeline_bootstrap_passes_qdrant_api_key(monkeypatch):
    from app.core.config import Settings
    from app.pipeline import bootstrap

    cfg = Settings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local", QDRANT_API_KEY=QDRANT_KEY)
    calls = []
    monkeypatch.setattr(bootstrap, "wait_for_qdrant", lambda _: None)
    monkeypatch.setattr(bootstrap, "QdrantStore", capture(calls))
    with pytest.raises(Stop):
        bootstrap.build_context(cfg, SimpleNamespace(model_version="m", dim=3), images=object())
    assert calls == [{"api_key": cfg.QDRANT_API_KEY}]


async def test_search_service_passes_qdrant_api_key(monkeypatch):
    from app.api import search_bootstrap
    from app.search.mock import MockEmbedder

    cfg = settings(QDRANT_API_KEY=QDRANT_KEY)
    calls = []
    monkeypatch.setattr(search_bootstrap.VllmClient, "from_settings", lambda _: AsyncMock())
    monkeypatch.setattr(search_bootstrap, "AsyncQdrantVectors", capture(calls))
    with pytest.raises(Stop):
        async with search_bootstrap.search_service(cfg, embedder=MockEmbedder()):
            pass
    assert calls == [{"api_key": cfg.QDRANT_API_KEY}]


async def test_story_handler_passes_qdrant_api_key(monkeypatch):
    from app.story import handler
    from tests.test_story import story_request

    cfg = settings(QDRANT_API_KEY=QDRANT_KEY)
    calls = []
    monkeypatch.setattr(handler, "JobsSettings", lambda: cfg)
    monkeypatch.setattr(handler.VllmClient, "from_settings", lambda _: AsyncMock())
    monkeypatch.setattr(handler, "AsyncQdrantVectors", capture(calls))
    with pytest.raises(Stop):
        await handler.run(story_request(), AsyncMock())
    assert calls == [{"api_key": cfg.QDRANT_API_KEY}]
