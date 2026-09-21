"""api 프로세스 엔트리, uvicorn app.main:app

create_app이 팩토리. 운영은 인자 없이, 테스트는 Settings와 FakeWorkerClient를 주입.
설정과 서비스는 import 시점이 아니라 lifespan(기동)에서 만들어 환경 변수 없이 import 가능
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.routers import health, internal, process
from app.api.routers.health import qdrant_ready
from app.api.services.task_service import TaskService
from app.api.services.task_store import TaskStore
from app.api.services.watchdog import watchdog_loop
from app.api.services.worker_client import WorkerClient, WorkerClientProtocol
from app.core.config import Settings, get_settings
from app.core.http_errors import register_error_handlers
from app.core.logging import get_logger, setup_logging

log = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    worker: WorkerClientProtocol | None = None,
    qdrant_probe: Callable[[], Awaitable[bool]] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 기동. yield 앞은 uvicorn이 요청을 받기 전, 뒤는 종료 신호 뒤
        cfg = settings or get_settings()
        setup_logging(cfg.LOG_LEVEL)
        http = httpx.AsyncClient(timeout=2.0)
        worker_client = worker or WorkerClient(cfg.WORKER_URL)
        service = TaskService(TaskStore(boot_time=time.monotonic()), worker_client, cfg)

        app.state.settings = cfg
        app.state.worker = worker_client
        app.state.task_service = service
        app.state.qdrant_ready = qdrant_probe or (lambda: qdrant_ready(http, cfg.QDRANT_URL))

        watchdog = asyncio.create_task(watchdog_loop(service, cfg.WATCHDOG_INTERVAL_SEC))
        log.info(
            "api 기동", extra={"max_concurrent": cfg.MAX_CONCURRENT, "max_queue": cfg.MAX_QUEUE}
        )
        try:
            yield
        finally:
            # 종료. 처리 중이던 작업은 프로세스와 함께 사라짐, 드레이닝 없음
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)
            await worker_client.aclose()
            await http.aclose()

    app = FastAPI(title="여담 AI", lifespan=lifespan)
    app.include_router(process.router)
    app.include_router(internal.router)
    app.include_router(health.router)

    register_error_handlers(app)
    return app


app = create_app()
