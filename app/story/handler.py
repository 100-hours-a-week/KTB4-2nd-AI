"""Kafka 스토리 작업의 실제 의존성과 기준선 파이프라인을 구성한다."""

from contextlib import AsyncExitStack, contextmanager

import httpx

from app.embedding.client import DIM, MODEL_VERSION
from app.infra.image_resolver import S3ImageResolver
from app.infra.qdrant import AsyncQdrantVectors
from app.infra.vllm_client import VllmClient
from app.jobs.config import JobsSettings
from app.jobs.gpu_requests import generation_client
from app.pipeline.bootstrap import build_images
from app.story.notes import BlogPlaceNotes
from app.story.pipeline import StoryContext, StoryOptions, validate_request
from app.story.pipeline import run as run_pipeline


@contextmanager
def image_source(cfg):
    images = build_images(cfg)
    try:
        yield images
    finally:
        if close := getattr(images, "close", None):
            close()


def place_searches(cfg) -> list[tuple[str, str]]:
    """장소 검색 제공자와 키를 부를 순서대로 돌려준다. 주 검색이 앞이고 키가 없는 검색은 뺀다."""
    keys = {"tavily": cfg.TAVILY_API_KEY, "firecrawl": cfg.FIRECRAWL_API_KEY}
    order = sorted(keys, key=lambda provider: provider != cfg.STORY_SEARCH_PRIMARY)
    # 예시 파일처럼 빈 값으로 둔 키는 없는 것으로 본다.
    found = [
        (provider, keys[provider].get_secret_value().strip())
        for provider in order
        if keys[provider]
    ]
    return [(provider, key) for provider, key in found if key]


async def run(request: dict, reporter) -> dict:
    validate_request(request)
    cfg = JobsSettings()
    options = StoryOptions(
        image_budget=cfg.STORY_IMAGE_BUDGET,
        blur_min=cfg.STORY_BLUR_VAR_MIN,
        phash_max=cfg.STORY_PHASH_MAX_DIST,
        dedupe_z=cfg.STORY_DEDUPE_Z,
        max_tokens=cfg.STORY_MAX_TOKENS,
    )
    await reporter.check_cancel()
    async with AsyncExitStack() as resources:
        # 스토리 생성 호출은 검색보다 오래 걸려 호출 제한 시간을 따로 둔다. 호출 슬롯은 함께 쓴다.
        call_timeout = cfg.STORY_VLM_TIMEOUT_SEC
        vlm = await resources.enter_async_context(
            VllmClient.from_settings(cfg.model_copy(update={"VLM_TIMEOUT_SEC": call_timeout}))
        )
        vectors = AsyncQdrantVectors(
            cfg.QDRANT_URL, MODEL_VERSION, DIM, cfg.QDRANT_TIMEOUT_SEC, api_key=cfg.QDRANT_API_KEY
        )
        resources.push_async_callback(vectors.aclose)
        urls = await S3ImageResolver.create(cfg)
        resources.push_async_callback(urls.aclose)
        gated = generation_client(vlm, reporter.store.redis, cfg, timeout_sec=call_timeout)
        notes = None
        searches = place_searches(cfg)
        if cfg.STORY_PLACE_NOTES_ENABLED and searches:
            # 블로그 글을 받는 클라이언트는 리디렉션을 따르지 않고, 환경 변수의 프록시 설정도
            # 쓰지 않는다.
            http = await resources.enter_async_context(
                httpx.AsyncClient(follow_redirects=False, trust_env=False)
            )
            notes = BlogPlaceNotes(
                http,
                gated,
                reporter.store.redis,
                searches,
                key_prefix=cfg.redis_prefix,
                timeout_sec=cfg.STORY_PLACE_NOTES_TIMEOUT_SEC,
                ttl_sec=cfg.STORY_PLACE_NOTES_TTL_SEC,
            )
        ctx = StoryContext(
            images=lambda: image_source(cfg),
            vectors=vectors,
            urls=urls,
            vlm=gated,
            notes=notes,
        )
        return await run_pipeline(request, reporter, ctx, options)
