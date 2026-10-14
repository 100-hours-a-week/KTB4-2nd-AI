"""임베딩 vLLM 서버 없이 요청 형식, 순서, 동시 요청, 응답 검증, 오류 변환을 확인한다."""

import base64
import json
import threading
import time

import httpx
import numpy as np
import pytest

from app.core.config import Settings
from app.embedding import client as embedding
from app.embedding.client import EmbeddingUnavailable, QwenEmbeddingEngine, QwenTextEmbedder
from app.search.context import ModelUnavailable

BASE_URL = "http://embedding.test/v1"
MODEL = "qwen3-vl-embedding-2b"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def unit(index: int, scale: float = 1.0) -> list[float]:
    vector = [0.0] * embedding.DIM
    vector[index] = scale
    return vector


def ok(vector: list[float]) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"index": 0, "embedding": vector}]})


def image_index(request: httpx.Request) -> int:
    """테스트 사진 bytes b"photo-{i}"에서 i를 꺼냄"""
    url = json.loads(request.content)["messages"][1]["content"][0]["image_url"]["url"]
    return int(base64.b64decode(url.removeprefix("data:image/jpeg;base64,")).split(b"-")[1])


@pytest.fixture
def make_engine():
    engines = []

    def build(handler) -> QwenEmbeddingEngine:
        engine = QwenEmbeddingEngine(
            BASE_URL, MODEL, api_key="secret", transport=httpx.MockTransport(handler)
        )
        engines.append(engine)
        return engine

    yield build
    for engine in engines:
        engine.close()


def test_image_request_matches_verified_format(make_engine):
    requests = []

    def handler(request):
        requests.append(request)
        return ok(unit(0))

    vectors = make_engine(handler).encode_images([b"\xff\xd8jpeg"])

    assert vectors.shape == (1, embedding.DIM) and vectors.dtype == np.float32
    request = requests[0]
    assert str(request.url) == f"{BASE_URL}/embeddings"
    assert request.headers["Authorization"] == "Bearer secret"
    assert json.loads(request.content) == {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Represent the user's input."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(b"\xff\xd8jpeg").decode()
                        },
                    }
                ],
            },
        ],
        "encoding_format": "float",
    }


def test_images_keep_input_order_with_bounded_concurrency(make_engine):
    lock = threading.Lock()
    state = {"running": 0, "peak": 0}
    count = 20

    def handler(request):
        index = image_index(request)
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        # 앞 사진일수록 늦게 응답해 도착 순서를 뒤집음
        time.sleep((count - index) * 0.005)
        with lock:
            state["running"] -= 1
        return ok(unit(index))

    images = [f"photo-{i}".encode() for i in range(count)]
    vectors = make_engine(handler).encode_images(images)

    np.testing.assert_array_equal(vectors, np.eye(count, embedding.DIM, dtype=np.float32))
    assert 1 < state["peak"] <= embedding.MAX_CONCURRENCY


def test_empty_images_skip_requests(make_engine):
    def handler(request):
        raise AssertionError("빈 목록에서는 요청 금지")

    vectors = make_engine(handler).encode_images([])

    assert vectors.shape == (0, embedding.DIM) and vectors.dtype == np.float32


def test_vectors_are_normalized_to_unit_length(make_engine):
    vectors = make_engine(lambda request: ok(unit(3, scale=5.0))).encode_images([b"photo-0"])

    assert vectors[0, 3] == 1.0
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


INVALID_RESPONSES = {
    "short": lambda request: ok(unit(0)[:-1]),
    # httpx의 json 인자는 NaN을 막으므로 서버가 보낼 수 있는 JSON 텍스트를 직접 생성
    "nan": lambda request: httpx.Response(
        200, text=json.dumps({"data": [{"embedding": [float("nan")] * embedding.DIM}]})
    ),
    "zero": lambda request: ok([0.0] * embedding.DIM),
    "no_data": lambda request: httpx.Response(200, json={"data": []}),
    "not_json": lambda request: httpx.Response(200, text="not json"),
    "server_error": lambda request: httpx.Response(500, json={"error": "boom"}),
    "unauthorized": lambda request: httpx.Response(401),
}


def connect_error(request):
    raise httpx.ConnectError("refused", request=request)


@pytest.mark.parametrize(
    "handler",
    [*INVALID_RESPONSES.values(), connect_error],
    ids=[*INVALID_RESPONSES, "connect_error"],
)
def test_image_failures_raise_embedding_unavailable(make_engine, handler):
    with pytest.raises(EmbeddingUnavailable):
        make_engine(handler).encode_images([b"photo-0", b"photo-1"])


def test_settings_are_required_and_secret_is_sent():
    settings = Settings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local")
    with pytest.raises(ValueError):
        QwenEmbeddingEngine.from_settings(settings)

    requests = []
    configured = Settings(
        _env_file=None,
        API_KEY="test",
        IMAGE_SOURCE="local",
        EMBEDDING_BASE_URL=BASE_URL,
        EMBEDDING_SERVED_MODEL_NAME=MODEL,
        EMBEDDING_API_KEY="embedding-secret",
    )
    engine = QwenEmbeddingEngine.from_settings(
        configured, transport=httpx.MockTransport(lambda r: requests.append(r) or ok(unit(0)))
    )
    try:
        engine.encode_images([b"photo-0"])
    finally:
        engine.close()
    assert requests[0].headers["Authorization"] == "Bearer embedding-secret"
    assert json.loads(requests[0].content)["model"] == MODEL


@pytest.mark.anyio
async def test_text_request_uses_query_instruction_and_returns_unit_list():
    requests = []

    def handler(request):
        requests.append(request)
        return ok(unit(1, scale=2.0))

    embedder = QwenTextEmbedder(BASE_URL, MODEL, transport=httpx.MockTransport(handler))
    try:
        vector = await embedder.encode_text("바다 사진")
    finally:
        await embedder.aclose()

    assert isinstance(vector, list) and len(vector) == embedding.DIM
    assert vector[1] == 1.0
    assert "Authorization" not in requests[0].headers
    assert json.loads(requests[0].content)["messages"] == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Retrieve images relevant to the user's query."}],
        },
        {"role": "user", "content": [{"type": "text", "text": "바다 사진"}]},
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "handler",
    [*INVALID_RESPONSES.values(), connect_error],
    ids=[*INVALID_RESPONSES, "connect_error"],
)
async def test_text_failures_raise_model_unavailable_without_query(handler):
    embedder = QwenTextEmbedder(BASE_URL, MODEL, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ModelUnavailable) as error:
            await embedder.encode_text("비밀 질문")
    finally:
        await embedder.aclose()
    assert "비밀 질문" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.anyio
async def test_text_timeout_raises_timeout_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    embedder = QwenTextEmbedder(BASE_URL, MODEL, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(TimeoutError):
            await embedder.encode_text("바다 사진")
    finally:
        await embedder.aclose()
