"""Kafka 스토리 작업의 실제 의존성과 기준선 파이프라인을 구성한다."""

from contextlib import AsyncExitStack, contextmanager

from app.embedding.client import DIM, MODEL_VERSION
from app.infra.image_resolver import S3ImageResolver
from app.infra.qdrant import AsyncQdrantVectors
from app.infra.vllm_client import VllmClient
from app.jobs.config import JobsSettings
from app.pipeline.bootstrap import build_images
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


async def run(request: dict, reporter) -> dict:
    validate_request(request)
    cfg = JobsSettings()
    options = StoryOptions(
        image_budget=cfg.STORY_IMAGE_BUDGET,
        blur_min=cfg.STORY_BLUR_VAR_MIN,
        phash_max=cfg.STORY_PHASH_MAX_DIST,
        max_tokens=cfg.STORY_MAX_TOKENS,
    )
    await reporter.check_cancel()
    async with AsyncExitStack() as resources:
        vlm = await resources.enter_async_context(VllmClient.from_settings(cfg))
        vectors = AsyncQdrantVectors(cfg.QDRANT_URL, MODEL_VERSION, DIM, cfg.QDRANT_TIMEOUT_SEC)
        resources.push_async_callback(vectors.aclose)
        urls = await S3ImageResolver.create(cfg)
        resources.push_async_callback(urls.aclose)
        ctx = StoryContext(images=lambda: image_source(cfg), vectors=vectors, urls=urls, vlm=vlm)
        return await run_pipeline(request, reporter, ctx, options)
