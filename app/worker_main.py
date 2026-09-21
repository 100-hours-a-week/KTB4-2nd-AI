"""worker 프로세스 엔트리, uvicorn app.worker_main:app

create_app이 팩토리. 운영은 인자 없이, 테스트는 FakeEngine, run_fn, FakeCallback을 주입.
모델 적재는 lifespan에서 동기로. 적재가 끝나야 요청을 받고, 실패하면 프로세스가 죽어
supervisord가 재시작.
백그라운드 적재는 실패가 "영원히 starting"이라는 조용한 실패가 되어 택하지 않음
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.core.http_errors import register_error_handlers
from app.core.logging import get_logger, setup_logging
from app.engines.base import EmbeddingEngine
from app.pipeline.bootstrap import RunFn, build_context, build_engine, select_run
from app.worker.callback import CallbackClient, CallbackProtocol
from app.worker.routers import jobs
from app.worker.runner import Runner

log = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    engine: EmbeddingEngine | None = None,
    run_fn: RunFn | None = None,
    callback: CallbackProtocol | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        setup_logging(cfg.LOG_LEVEL)
        app.state.model_loaded = False
        app.state.model_name = cfg.EMBED_MODEL

        loaded_engine = engine or build_engine(cfg)
        ctx = build_context(cfg, loaded_engine)
        cb = callback or CallbackClient(cfg.API_URL, retries=cfg.CALLBACK_RETRIES)
        runner = Runner(ctx, run_fn or select_run(cfg), cb)

        app.state.runner = runner
        app.state.model_name = loaded_engine.model_version
        app.state.model_loaded = True
        log.info(
            "worker 기동",
            extra={"model": loaded_engine.model_version, "fake": cfg.FAKE_PIPELINE},
        )
        try:
            yield
        finally:
            runner.shutdown()
            if isinstance(cb, CallbackClient):
                cb.close()

    app = FastAPI(title="여담 AI worker", lifespan=lifespan)
    app.include_router(jobs.router)
    register_error_handlers(app)
    return app


app = create_app()
