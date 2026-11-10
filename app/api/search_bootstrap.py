"""API별 검색 의존성을 구성하고 직접 생성한 클라이언트를 닫는다."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from app.api.services.query_service import QueryService
from app.api.services.search_tracing import (
    TracedEmbedder,
    TracedImages,
    TracedVectors,
    TracedVlm,
)
from app.core.config import Settings
from app.embedding.client import DIM, MODEL_VERSION, QwenTextEmbedder
from app.embedding.contracts import TextEmbedder
from app.infra.image_resolver import S3ImageResolver
from app.infra.qdrant import AsyncQdrantVectors
from app.infra.vllm_client import VllmClient
from app.observability.metrics import Metrics
from app.search.context import CandidateSearcher, SearchContext, SearchOptions


@asynccontextmanager
async def search_service(
    settings: Settings,
    *,
    vectors: CandidateSearcher | None = None,
    embedder: TextEmbedder | None = None,
    options: SearchOptions | None = None,
    redis=None,
) -> AsyncIterator[QueryService]:
    """API별로 한 번 구성한다. 직접 만든 자원만 닫고 주입된 자원은 빌려 쓴다.

    생략한 임베딩과 후보 검색은 Qwen HTTP 클라이언트와 같은 모델 버전의
    Qdrant 읽기 클라이언트로 만든다. 프로세스 안에 모델을 적재하지 않는다.
    """
    async with AsyncExitStack() as resources:
        vlm = await resources.enter_async_context(VllmClient.from_settings(settings))
        if embedder is None:
            embedder = QwenTextEmbedder.from_settings(settings)
            resources.push_async_callback(embedder.aclose)
        if vectors is None:
            vectors = AsyncQdrantVectors(
                settings.QDRANT_URL,
                MODEL_VERSION,
                DIM,
                settings.QDRANT_TIMEOUT_SEC,
                api_key=settings.QDRANT_API_KEY,
            )
            resources.push_async_callback(vectors.aclose)
        images = await S3ImageResolver.create(settings)
        resources.push_async_callback(images.aclose)
        if redis is not None:
            from app.jobs.gpu_requests import embedding_client, generation_client

            vlm = generation_client(vlm, redis, settings, kind="search")
            embedder = embedding_client(embedder, redis, settings)
        # 감싸기는 GPU 호출 한도 밖에 둔다. 단계 시간에는 슬롯 대기 시간도 포함된다.
        yield QueryService(
            SearchContext(
                vlm=TracedVlm(vlm),
                embedder=TracedEmbedder(embedder),
                vectors=TracedVectors(vectors),
                images=TracedImages(images),
            ),
            options,
            metrics=Metrics(redis, settings.redis_prefix) if redis is not None else None,
        )


CALL_TIMEOUT_MARGIN_SEC = 10


@asynccontextmanager
async def rewrite_service(settings, redis):
    """분위기 변경용 VLM 클라이언트와 S3 URL 발급기를 API별로 한 번 만든다.

    호출 상한이 검색(VLM_TIMEOUT_SEC)과 스토리(STORY_VLM_TIMEOUT_SEC)보다 길어 클라이언트를 따로
    두고, GPU 호출 슬롯은 함께 쓴다.
    """
    from app.jobs.gpu_requests import generation_client
    from app.story.rewrite import RewriteService

    timeout = settings.STORY_REWRITE_TIMEOUT_SEC
    # 호출 상한을 처리 상한보다 조금 짧게 두어, 생성이 늦을 때 원인이
    # 호출 로그에 timeout으로 남게 한다.
    call_timeout = max(timeout - CALL_TIMEOUT_MARGIN_SEC, timeout / 2)
    async with AsyncExitStack() as resources:
        vlm = await resources.enter_async_context(
            VllmClient.from_settings(settings.model_copy(update={"VLM_TIMEOUT_SEC": call_timeout}))
        )
        images = await S3ImageResolver.create(settings)
        resources.push_async_callback(images.aclose)
        service = RewriteService(
            # 분위기 변경은 스토리 기능이므로 스토리 상한을 함께 쓴다(결정 222).
            generation_client(vlm, redis, settings, kind="story", timeout_sec=call_timeout),
            images,
            redis,
            settings.redis_prefix,
            timeout_sec=timeout,
            max_tokens=settings.STORY_REWRITE_MAX_TOKENS,
        )
        try:
            yield service
        finally:
            await service.aclose()
