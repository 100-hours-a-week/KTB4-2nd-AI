"""실제 S3 서명과 VLM HTTP 조립을 검증한다. AWS와 실제 모델은 호출하지 않는다."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from functools import partial
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, unquote, urlsplit

import boto3
import httpx
import pytest
from botocore.config import Config
from botocore.exceptions import NoCredentialsError

from app.api.search_bootstrap import search_service
from app.core.config import Settings
from app.embedding.client import DIM, MODEL_VERSION, QUERY_INSTRUCTION, QwenTextEmbedder
from app.infra.image_resolver import S3ImageResolver
from app.infra.vllm_client import VllmClient
from app.infra.vlm import VLMUnavailable
from app.search.context import SearchOptions
from app.search.mock import MockEmbedder, MockVectors
from app.v2_main import create_app, create_search_app
from tests.test_query import search_body
from tests.test_v2_jobs import environment

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def signed_client():
    return boto3.client(
        "s3",
        region_name="ap-northeast-2",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        config=Config(signature_version="s3v4"),
    )


async def test_s3_get_signing_preserves_key_and_uses_configured_bucket():
    client = signed_client()
    resolver = S3ImageResolver(client, "test-photos", expires_sec=120)
    key = "trips/77/한글 사진+&?.jpg"
    try:
        url = urlsplit(await resolver.resolve(key))
        assert url.scheme == "https" and url.hostname == "test-photos.s3.amazonaws.com"
        assert unquote(url.path) == "/" + key
        query = parse_qs(url.query)
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert query["X-Amz-Expires"] == ["120"]
    finally:
        await resolver.aclose()


@pytest.mark.parametrize("region", ["ap-northeast-2", "eu-central-1"])
async def test_s3_settings_sign_for_bucket_region_without_global_endpoint(monkeypatch, region):
    create_client = boto3.client

    def with_test_credentials(*args, **kwargs):
        return create_client(
            *args, **kwargs, aws_access_key_id="testing", aws_secret_access_key="testing"
        )

    monkeypatch.setattr("app.infra.image_resolver.boto3.client", with_test_credentials)
    cfg = Settings(_env_file=None, API_KEY="test", S3_BUCKET="test-photos", AWS_REGION=region)
    resolver = S3ImageResolver.from_settings(cfg)
    try:
        url = urlsplit(await resolver.resolve("trips/77/a.jpg"))
        assert url.hostname == f"test-photos.s3.{region}.amazonaws.com"
        assert f"/{region}/s3/aws4_request" in parse_qs(url.query)["X-Amz-Credential"][0]
    finally:
        await resolver.aclose()


@pytest.mark.parametrize("key", ["", "가" * 342, "bad\ud800"])
async def test_invalid_key_fails_before_signing(key):
    client = Mock()
    with pytest.raises(VLMUnavailable):
        await S3ImageResolver(client, "test-photos").resolve(key)
    client.generate_presigned_url.assert_not_called()


async def test_sdk_signing_does_not_block_event_loop():
    entered, release = threading.Event(), threading.Event()

    def sign(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return "https://example.invalid/signed"

    client = Mock(generate_presigned_url=sign)
    task = asyncio.create_task(S3ImageResolver(client, "test-photos").resolve("a.jpg"))
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        release.set()
    assert await task == "https://example.invalid/signed"


@pytest.mark.parametrize("fail_signing", [False, True])
async def test_search_factory_signs_only_selected_question_photos_and_closes(
    monkeypatch, fail_signing, caplog
):
    requests, keys = [], []
    s3 = signed_client()
    original_sign = s3.generate_presigned_url
    closed = []
    original_close = s3.close

    def sign(*args, **kwargs):
        keys.append(kwargs["Params"]["Key"])
        if fail_signing:
            raise NoCredentialsError()
        return original_sign(*args, **kwargs)

    def close():
        closed.append("s3")
        original_close()

    s3.generate_presigned_url = sign
    s3.close = close
    monkeypatch.setattr("app.infra.image_resolver.boto3.client", lambda *a, **k: s3)

    def response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "바다입니다."}}]},
        )

    vlm = VllmClient(
        "https://vlm.example.invalid/v1", "test", transport=httpx.MockTransport(response)
    )
    monkeypatch.setattr(VllmClient, "from_settings", lambda *a: vlm)
    async with environment() as (cfg, store, _, _producer):
        cfg.IMAGE_SOURCE, cfg.S3_BUCKET = "s3", "test-photos"
        factory = partial(
            search_service,
            embedder=MockEmbedder(),
            vectors=MockVectors(),
            options=SearchOptions(question_k=1),
        )
        app = create_app(cfg, store, start_kafka=False, query_service_factory=factory)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer test"},
            ) as http:
                search = await http.post("/query/search", json=search_body("SEARCH", 20))
                assert search.status_code == 200 and len(search.json()["attachments"]) == 20
                assert keys == [] and requests == []
                question = await http.post("/query/search", json=search_body("QUESTION", 20))
                assert question.status_code == 200
                result = question.json()
                assert len(result["attachments"]) == 1 and keys == ["trips/1/0.jpg"]
                if fail_signing:
                    assert result["answer"] is None and result["answer_error"] == "VLM_UNAVAILABLE"
                    assert not requests
                else:
                    assert result["answer"] == "바다입니다."
                    content = requests[0]["messages"][-1]["content"]
                    urls = [
                        item["image_url"]["url"] for item in content if item["type"] == "image_url"
                    ]
                    assert len(urls) == 1 and "X-Amz-Signature" in urls[0]
                assert "X-Amz-Signature" not in caplog.text
        assert closed == ["s3"] and vlm._client.is_closed


async def test_factory_startup_failure_closes_vlm(monkeypatch):
    cfg = Settings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local")
    vlm = VllmClient("https://vlm.example.invalid/v1", "test")
    monkeypatch.setattr(VllmClient, "from_settings", lambda *a: vlm)
    with pytest.raises(ValueError, match="S3_BUCKET"):
        async with search_service(cfg, embedder=MockEmbedder(), vectors=MockVectors()):
            raise AssertionError("구성 실패 시 서비스를 제공하면 안 됨")
    assert vlm._client.is_closed


async def test_blank_bucket_fails_before_creating_s3_client(monkeypatch):
    create_client = Mock()
    monkeypatch.setattr("app.infra.image_resolver.boto3.client", create_client)
    cfg = Settings(_env_file=None, API_KEY="test", S3_BUCKET="   ")
    with pytest.raises(ValueError, match="S3_BUCKET"):
        await S3ImageResolver.create(cfg)
    create_client.assert_not_called()


async def test_cancelled_startup_closes_s3_when_creation_finishes_later(monkeypatch):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    client = Mock(close=Mock(side_effect=closed.set))

    def build(cfg):
        entered.set()
        assert release.wait(3)
        return S3ImageResolver(client, "test-photos")

    monkeypatch.setattr(S3ImageResolver, "from_settings", build)
    vlm = VllmClient("https://vlm.example.invalid/v1", "test")
    monkeypatch.setattr(VllmClient, "from_settings", lambda cfg: vlm)
    cfg = Settings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local")

    async def start():
        async with search_service(cfg, vectors=MockVectors(), embedder=MockEmbedder()):
            pytest.fail("취소한 초기화가 완료되면 안 됨")

    task = asyncio.create_task(start())
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert vlm._client.is_closed
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    async with asyncio.timeout(2):
        while not closed.is_set():
            await asyncio.sleep(0.001)
    client.close.assert_called_once()


async def test_creation_before_cancellation_and_repeated_cancel_still_close_s3(monkeypatch):
    delivered = asyncio.Event()
    closing, release, closed = threading.Event(), threading.Event(), threading.Event()

    def close():
        closing.set()
        assert release.wait(3)
        closed.set()

    client = Mock(close=Mock(side_effect=close))
    monkeypatch.setattr(
        S3ImageResolver, "from_settings", lambda cfg: S3ImageResolver(client, "test-photos")
    )
    original_to_thread = asyncio.to_thread

    async def delay_delivery(func, *args, **kwargs):
        result = await original_to_thread(func, *args, **kwargs)
        delivered.set()
        await asyncio.Event().wait()
        return result

    monkeypatch.setattr("app.infra.image_resolver.asyncio.to_thread", delay_delivery)
    cfg = Settings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local")
    task = asyncio.create_task(S3ImageResolver.create(cfg))
    try:
        async with asyncio.timeout(2):
            await delivered.wait()
            task.cancel()
            while not closing.is_set():
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    async with asyncio.timeout(2):
        while not closed.is_set():
            await asyncio.sleep(0.001)
    client.close.assert_called_once()


async def test_factory_lifecycle_runs_once_per_api_and_cleanup_on_error():
    entered, exited = [], []

    @asynccontextmanager
    async def factory(settings):
        entered.append(settings)
        try:
            yield None
        finally:
            exited.append(settings)

    async with environment() as (cfg, store, _, _producer):
        app = create_app(cfg, store, start_kafka=False, query_service_factory=factory)
        with pytest.raises(RuntimeError, match="after startup"):
            async with app.router.lifespan_context(app):
                assert len(entered) == 1 and exited == []
                raise RuntimeError("after startup")
        assert entered == exited == [cfg]


@pytest.mark.parametrize("outcome", ["success", "unavailable", "timeout"])
async def test_owned_qwen_client_search_http_and_failure_contract(monkeypatch, outcome):
    requests = []

    def embed(request):
        requests.append(request)
        if outcome == "timeout":
            raise httpx.ReadTimeout("private upstream details", request=request)
        if outcome == "unavailable":
            return httpx.Response(503, text="private upstream details")
        return httpx.Response(200, json={"data": [{"embedding": [1.0] + [0.0] * (DIM - 1)}]})

    embedder = QwenTextEmbedder(
        "https://embedding.example.invalid/v1", "qwen-test", transport=httpx.MockTransport(embed)
    )
    monkeypatch.setattr(QwenTextEmbedder, "from_settings", lambda cfg: embedder)
    vlm = VllmClient("https://vlm.example.invalid/v1", "test")
    monkeypatch.setattr(VllmClient, "from_settings", lambda cfg: vlm)
    images = Mock(resolve=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr(S3ImageResolver, "from_settings", lambda cfg: images)
    vectors = Mock(search=AsyncMock(side_effect=MockVectors().search))

    async with environment() as (cfg, store, _, _producer):
        app = create_app(
            cfg,
            store,
            start_kafka=False,
            query_service_factory=partial(search_service, vectors=vectors),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer test"},
            ) as http:
                result = await http.post("/query/search", json=search_body())
                assert "private upstream details" not in result.text
                if outcome == "success":
                    assert result.status_code == 200 and len(result.json()["attachments"]) == 2
                    vector = vectors.search.call_args.args[0]
                    assert len(vector) == DIM and vector == [1.0] + [0.0] * (DIM - 1)
                    assert vectors.search.call_args.kwargs == {
                        "candidate_ids": [1, 2],
                        "limit": 30,
                    }
                else:
                    assert result.status_code == (503 if outcome == "unavailable" else 504)
                    expected = "MODEL_NOT_READY" if outcome == "unavailable" else "TIMEOUT"
                    assert result.json()["error"]["code"] == expected
                    vectors.search.assert_not_called()
                assert (await http.get("/health")).status_code == 200
        assert embedder._client.is_closed and vlm._client.is_closed
        images.aclose.assert_awaited_once()
        images.resolve.assert_not_called()

    assert len(requests) == 1 and requests[0].url.path == "/v1/embeddings"
    payload = json.loads(requests[0].content)
    assert payload["messages"][0]["content"][0]["text"] == QUERY_INSTRUCTION
    assert payload["messages"][1]["content"][0]["text"] == search_body()["query"]
    assert payload["model"] == "qwen-test" and payload["encoding_format"] == "float"


async def test_s3_startup_failure_closes_owned_embedding_and_vlm(monkeypatch):
    embedder = QwenTextEmbedder("https://embedding.example.invalid/v1", "test")
    vlm = VllmClient("https://vlm.example.invalid/v1", "test")
    monkeypatch.setattr(QwenTextEmbedder, "from_settings", lambda cfg: embedder)
    monkeypatch.setattr(VllmClient, "from_settings", lambda cfg: vlm)
    cfg = Settings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local")
    with pytest.raises(ValueError, match="S3_BUCKET"):
        async with search_service(cfg, vectors=MockVectors()):
            raise AssertionError("구성 실패 시 서비스를 제공하면 안 됨")
    assert embedder._client.is_closed and vlm._client.is_closed


async def test_borrowed_dependencies_are_not_closed_even_if_image_cleanup_fails(monkeypatch):
    embedder = Mock(aclose=AsyncMock())
    vectors = Mock(aclose=AsyncMock())
    vlm = VllmClient("https://vlm.example.invalid/v1", "test")
    monkeypatch.setattr(VllmClient, "from_settings", lambda cfg: vlm)
    images = Mock(aclose=AsyncMock(side_effect=RuntimeError("image close")))
    monkeypatch.setattr(S3ImageResolver, "from_settings", lambda cfg: images)
    cfg = Settings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local")
    with pytest.raises(RuntimeError, match="image close"):
        async with search_service(cfg, vectors=vectors, embedder=embedder):
            pass
    embedder.aclose.assert_not_called()
    vectors.aclose.assert_not_called()
    assert vlm._client.is_closed


async def test_search_cleanup_runs_even_if_kafka_shutdown_fails(monkeypatch):
    exited = []

    @asynccontextmanager
    async def factory(settings):
        try:
            yield None
        finally:
            exited.append(True)

    runtime = Mock(start=AsyncMock(), stop=AsyncMock(side_effect=RuntimeError("kafka stop")))
    monkeypatch.setattr("app.v2_main.KafkaRuntime", lambda store: runtime)
    async with environment() as (cfg, store, _, _producer):
        app = create_app(cfg, store, query_service_factory=factory)
        with pytest.raises(RuntimeError, match="kafka stop"):
            async with app.router.lifespan_context(app):
                pass
    assert exited == [True]


async def test_two_api_replicas_reuse_distinct_clients_and_keep_polling_during_search(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    embedders, vlms = [], []

    async def embed(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"data": [{"embedding": [1.0] + [0.0] * (DIM - 1)}]})

    def build_embedder(cfg):
        client = QwenTextEmbedder(
            "https://embedding.example.invalid/v1", "test", transport=httpx.MockTransport(embed)
        )
        embedders.append(client)
        return client

    def build_vlm(cfg):
        client = VllmClient("https://vlm.example.invalid/v1", "test")
        vlms.append(client)
        return client

    monkeypatch.setattr(QwenTextEmbedder, "from_settings", build_embedder)
    monkeypatch.setattr(VllmClient, "from_settings", build_vlm)
    monkeypatch.setattr(S3ImageResolver, "from_settings", lambda cfg: Mock(aclose=AsyncMock()))
    from app.jobs.kafka import RequestReceiver
    from tests.test_v2_jobs import message, process_request

    async with environment() as (cfg, store_a, store_b, _producer):
        await RequestReceiver(store_a, _producer).accept(message(cfg, process_request()))
        factory = partial(search_service, vectors=MockVectors())
        a = create_app(cfg, store_a, start_kafka=False, query_service_factory=factory)
        b = create_app(cfg, store_b, start_kafka=False, query_service_factory=factory)
        async with a.router.lifespan_context(a), b.router.lifespan_context(b):
            assert len(embedders) == len(vlms) == 2 and embedders[0] is not embedders[1]
            async with (
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=a),
                    base_url="http://a",
                    headers={"Authorization": "Bearer test"},
                ) as http_a,
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=b),
                    base_url="http://b",
                    headers={"Authorization": "Bearer test"},
                ) as http_b,
            ):
                running = asyncio.create_task(http_a.post("/query/search", json=search_body()))
                try:
                    async with asyncio.timeout(2):
                        await entered.wait()
                        for http in (http_a, http_b):
                            status = await http.get("/trips/77/process")
                            assert status.status_code == 200 and status.json()["status"] == "QUEUED"
                        assert not running.done()
                finally:
                    release.set()
                    response = await running
                assert response.status_code == 200
                assert (await http_b.post("/query/search", json=search_body())).status_code == 200
                assert len(embedders) == 2
    assert all(client._client.is_closed for client in embedders + vlms)


@pytest.mark.parametrize("image_failure", [False, True])
async def test_default_qdrant_uses_qwen_collection_and_is_closed(monkeypatch, image_failure):
    from app.api import search_bootstrap

    vectors = Mock(search=AsyncMock(side_effect=MockVectors().search), aclose=AsyncMock())
    factory = Mock(return_value=vectors)
    monkeypatch.setattr(search_bootstrap, "AsyncQdrantVectors", factory)
    vlm = VllmClient("https://vlm.example.invalid/v1", "test")
    embedder = QwenTextEmbedder("https://embedding.example.invalid/v1", "test")
    monkeypatch.setattr(VllmClient, "from_settings", lambda cfg: vlm)
    monkeypatch.setattr(QwenTextEmbedder, "from_settings", lambda cfg: embedder)
    images = Mock(aclose=AsyncMock())

    async def image_factory(cfg):
        if image_failure:
            raise ValueError("invalid bucket")
        return images

    monkeypatch.setattr(S3ImageResolver, "create", image_factory)
    cfg = Settings(
        _env_file=None,
        API_KEY="test",
        IMAGE_SOURCE="local",
        QDRANT_URL="http://qdrant.test:6333",
        QDRANT_TIMEOUT_SEC=2,
    )
    if image_failure:
        with pytest.raises(ValueError, match="invalid bucket"):
            async with search_service(cfg):
                pytest.fail("invalid image configuration must fail startup")
    else:
        async with search_service(cfg) as service:
            assert service.context.vectors is vectors
        images.aclose.assert_awaited_once()
    factory.assert_called_once_with(cfg.QDRANT_URL, MODEL_VERSION, DIM, 2)
    vectors.aclose.assert_awaited_once()
    assert vlm._client.is_closed and embedder._client.is_closed


@pytest.mark.parametrize("entrypoint", ["enabled", "disabled", "search_only"])
async def test_deployment_entrypoints_select_search_without_implicit_mock(monkeypatch, entrypoint):
    from app.api.services.query_service import QueryService
    from app.search.mock import create_mock_context

    entered, closed = [], []

    @asynccontextmanager
    async def factory(cfg):
        entered.append(cfg)
        try:
            yield QueryService(create_mock_context())
        finally:
            closed.append(cfg)

    def forbidden(*args, **kwargs):
        raise AssertionError("검색 전용 기동에서는 Kafka와 스토리 실행을 시작하면 안 됨")

    monkeypatch.setattr("app.v2_main.search_service", factory)
    monkeypatch.setattr("app.v2_main.KafkaRuntime", forbidden)
    monkeypatch.setattr("app.v2_main.handler_for", forbidden)
    async with environment() as (cfg, store, _, _producer):
        cfg.SEARCH_ENABLED = entrypoint == "enabled"
        cfg.JOBS_FAKE_PIPELINE = False
        cfg.JOBS_STORY_HANDLER = None
        app = (
            create_search_app(cfg, store)
            if entrypoint == "search_only"
            else create_app(cfg, store, start_kafka=False)
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer test"},
            ) as client:
                response = await client.post("/query/search", json=search_body())
                assert response.status_code == (503 if entrypoint == "disabled" else 200)
                assert app.state.kafka is None
        assert len(entered) == (0 if entrypoint == "disabled" else 1)
        assert entered == closed


async def test_enabled_search_missing_settings_fails_startup(monkeypatch):
    async with environment() as (cfg, store, _, _producer):
        cfg.SEARCH_ENABLED = True
        cfg.VLM_BASE_URL = None
        with pytest.raises(ValueError, match="VLM_BASE_URL"):
            app = create_app(cfg, store, start_kafka=False)
            async with app.router.lifespan_context(app):
                pytest.fail("실제 검색 설정이 없으면 검색을 활성화한 채 기동하면 안 됨")
