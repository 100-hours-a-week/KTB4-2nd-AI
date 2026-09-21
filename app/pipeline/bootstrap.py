"""엔진과 PipelineContext 조립. worker_main과 scripts/run_pipeline.py가 같은 코드로 조립

각자 조립하면 컬렉션 이름이나 이미지 소스가 어긋남. 조립 순서는 엔진 → 컨텍스트,
Qdrant 컬렉션 이름과 차원이 엔진에서 나오기 때문
"""

import time
from collections.abc import Callable

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.engines.base import EmbeddingEngine
from app.engines.fake import FakeEngine
from app.infra.images import ImageSource, LocalDirImageSource, S3ImageSource
from app.infra.qdrant import MEMORY, QdrantStore
from app.pipeline.context import PipelineContext, load_thresholds
from app.pipeline.fake_run import fake_run
from app.pipeline.run import CancelCheck, ProgressCallback
from app.schemas.process import ProcessRequest, ProcessResult

RunFn = Callable[[ProcessRequest, PipelineContext, ProgressCallback, CancelCheck], ProcessResult]

log = get_logger(__name__)


def wait_for_qdrant(url: str, timeout: float = 60.0, interval: float = 1.0) -> None:
    """Qdrant /readyz가 200이 될 때까지 대기. 워커 기동 때 한 번

    supervisord가 qdrant와 worker를 같이 띄우므로 워커가 먼저 붙으면 연결 실패로 죽음.
    :memory:는 대기 없음
    """
    if url == MEMORY:
        return
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        try:
            if httpx.get(f"{url}/readyz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        attempt += 1
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Qdrant {url}가 {timeout:.0f}초 안에 준비되지 않음")
        if attempt % 10 == 0:
            log.info("Qdrant 대기", extra={"url": url, "attempt": attempt})
        time.sleep(interval)


def build_engine(settings: Settings) -> EmbeddingEngine:
    """FAKE_PIPELINE이면 FakeEngine, 아니면 파이프라인 담당의 엔진

    진짜 엔진은 함수 안에서 import. 모듈 최상단에 두면 그 파일이 없는 동안 가짜 모드도 못 뜸
    """
    if settings.FAKE_PIPELINE:
        return FakeEngine()
    from app.engines.siglip_naflex import SiglipNaflexEngine  # 파이프라인 담당 파일

    return SiglipNaflexEngine(model_path=settings.MODEL_PATH, device=settings.EMBED_DEVICE)


def build_images(settings: Settings) -> ImageSource:
    if settings.IMAGE_SOURCE == "local":
        return LocalDirImageSource(settings.IMAGE_DIR)
    assert settings.S3_BUCKET is not None  # Settings 검증기가 s3면 필수로 잡음
    return S3ImageSource(bucket=settings.S3_BUCKET, region=settings.AWS_REGION)


def build_context(
    settings: Settings,
    engine: EmbeddingEngine,
    images: ImageSource | None = None,
    qdrant: QdrantStore | None = None,
) -> PipelineContext:
    """images와 qdrant는 테스트가 LocalDirImageSource와 :memory:를 넣기 위한 자리

    FAKE_PIPELINE이면 thresholds를 읽지 않음. 파일이 없거나 값이 틀려도 가짜 모드는 영향 없음
    """
    if images is None:
        images = build_images(settings)
    if qdrant is None:
        wait_for_qdrant(settings.QDRANT_URL)
        qdrant = QdrantStore(settings.QDRANT_URL, engine.model_version, engine.dim)
    thresholds = (
        {}
        if settings.FAKE_PIPELINE
        else load_thresholds(settings.THRESHOLDS_PATH, settings.EMBED_MODEL)
    )
    return PipelineContext(
        images=images,
        engine=engine,
        qdrant=qdrant,
        thresholds=thresholds,
        settings=settings,
        log=get_logger("app.pipeline"),
    )


def select_run(settings: Settings) -> RunFn:
    """러너가 부를 함수. FAKE_PIPELINE이면 fake_run, 아니면 run"""
    if settings.FAKE_PIPELINE:
        return fake_run
    from app.pipeline.run import run

    return run
