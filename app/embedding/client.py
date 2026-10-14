"""임베딩 vLLM HTTP 클라이언트. 요청 형식은 docker/embedding/README.md의 요청 규칙(10/13 검증)

사진은 정리 실행기의 동기 파이프라인, 텍스트는 ai-api 검색의 이벤트 루프에서 사용
"""

import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Self

import httpx
import numpy as np

from app.core.config import Settings
from app.embedding.contracts import EmbeddingEngine
from app.search.context import ModelUnavailable

# 벡터를 결정하는 값이라 환경 변수가 아닌 코드에 고정. 바꾸면 Qdrant 컬렉션도 새로 만듦
MODEL_VERSION = "qwen3-vl-embedding-2b"
DIM = 2048
IMAGE_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = "Retrieve images relevant to the user's query."
# 정리 묶음 하나(16장)를 한 번에 보내는 동시 요청 수, 운영 규약의 묶음 크기와 같음
MAX_CONCURRENCY = 16


class EmbeddingUnavailable(Exception):
    """임베딩 서버 요청 실패 또는 응답이 2048차원 유한 벡터가 아님"""


def image_request(model: str, image: bytes) -> dict[str, Any]:
    url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
    return _request(model, IMAGE_INSTRUCTION, {"type": "image_url", "image_url": {"url": url}})


def text_request(model: str, text: str) -> dict[str, Any]:
    return _request(model, QUERY_INSTRUCTION, {"type": "text", "text": text})


def _request(model: str, instruction: str, content: dict[str, Any]) -> dict[str, Any]:
    # input 형식은 같은 문장도 다른 벡터가 되어 사진과 비교 불가, 텍스트도 messages로 보냄
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": [content]},
        ],
        "encoding_format": "float",
    }


def parse_vector(response: httpx.Response) -> np.ndarray:
    """응답의 첫 벡터를 float32로 꺼내 L2 정규화, 서버 정규화 값의 반올림 오차까지 계약에 맞춤"""
    try:
        vector = np.asarray(response.json()["data"][0]["embedding"], dtype=np.float32)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise EmbeddingUnavailable("임베딩 응답 형식이 올바르지 않음") from exc
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise EmbeddingUnavailable(f"임베딩 벡터가 {DIM}차원 유한값이 아님")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise EmbeddingUnavailable("임베딩 벡터의 크기가 0")
    return vector / norm


def _client_options(
    base_url: str, api_key: str | None, timeout_sec: float, connect_timeout_sec: float
) -> dict[str, Any]:
    return {
        "base_url": base_url.rstrip("/") + "/",
        "headers": {"Authorization": f"Bearer {api_key}"} if api_key else {},
        "timeout": httpx.Timeout(timeout_sec, connect=connect_timeout_sec),
        "follow_redirects": False,
    }


def _settings_options(settings: Settings) -> dict[str, Any]:
    if not settings.EMBEDDING_BASE_URL or not settings.EMBEDDING_SERVED_MODEL_NAME:
        raise ValueError("EMBEDDING_BASE_URL과 EMBEDDING_SERVED_MODEL_NAME을 설정해야 합니다.")
    api_key = settings.EMBEDDING_API_KEY
    return {
        "base_url": settings.EMBEDDING_BASE_URL,
        "model": settings.EMBEDDING_SERVED_MODEL_NAME,
        "api_key": api_key.get_secret_value() if api_key else None,
        "timeout_sec": settings.EMBEDDING_TIMEOUT_SEC,
        "connect_timeout_sec": settings.EMBEDDING_CONNECT_TIMEOUT_SEC,
    }


class QwenEmbeddingEngine(EmbeddingEngine):
    """정리 실행기의 사진 임베딩. 사진 1장당 요청 1개, 최대 16개 동시

    실패는 EmbeddingUnavailable, run()이 로그를 남기고 PipelineError로 바꿈.
    연결 실패만 1회 재시도, 서버 재기동(약 95초)은 재시도로 덮지 않음
    """

    model_version = MODEL_VERSION
    dim = DIM

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_sec: float = 30.0,
        connect_timeout_sec: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._client = httpx.Client(
            **_client_options(base_url, api_key, timeout_sec, connect_timeout_sec),
            limits=httpx.Limits(max_connections=MAX_CONCURRENCY),
            transport=transport or httpx.HTTPTransport(retries=1),
        )
        self._pool = ThreadPoolExecutor(MAX_CONCURRENCY, thread_name_prefix="embedding")

    @classmethod
    def from_settings(
        cls, settings: Settings, *, transport: httpx.BaseTransport | None = None
    ) -> Self:
        return cls(**_settings_options(settings), transport=transport)

    def encode_images(self, images: list[bytes]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        # map은 응답이 도착한 순서와 무관하게 입력 순서대로 결과를 돌려줌
        return np.stack(list(self._pool.map(self._encode_one, images)))

    def _encode_one(self, image: bytes) -> np.ndarray:
        try:
            response = self._client.post("embeddings", json=image_request(self._model, image))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailable("임베딩 서버 요청 실패") from exc
        return parse_vector(response)

    def close(self) -> None:
        self._pool.shutdown(cancel_futures=True)
        self._client.close()


class QwenTextEmbedder:
    """검색 질의 임베딩. ai-api 프로세스에서 재사용하고 앱 종료 시 aclose

    실패는 ModelUnavailable(검색 503), 시간 초과는 TimeoutError(검색 504)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_sec: float = 30.0,
        connect_timeout_sec: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            **_client_options(base_url, api_key, timeout_sec, connect_timeout_sec),
            transport=transport or httpx.AsyncHTTPTransport(retries=1),
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> Self:
        return cls(**_settings_options(settings), transport=transport)

    async def encode_text(self, text: str) -> list[float]:
        try:
            response = await self._client.post("embeddings", json=text_request(self._model, text))
            response.raise_for_status()
            return parse_vector(response).tolist()
        except httpx.TimeoutException:
            raise TimeoutError("임베딩 요청 시간이 초과되었습니다.") from None
        except (httpx.HTTPError, EmbeddingUnavailable):
            # 질문 원문과 외부 예외 본문을 오류에 남기지 않음
            raise ModelUnavailable("임베딩 서버를 사용할 수 없습니다.") from None

    async def aclose(self) -> None:
        await self._client.aclose()
