"""v2 API: uvicorn app.v2_main:create_app --factory. 두 복제본이 같은 Redis를 사용한다."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Request
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.api.deps import require_api_key
from app.api.routers import query
from app.api.services.query_service import QueryService
from app.core.http_errors import register_error_handlers
from app.core.logging import bind_request, setup_logging
from app.jobs.config import JobsSettings
from app.jobs.execution import handler_for
from app.jobs.kafka import KafkaRuntime
from app.jobs.models import JobStatusResponse
from app.jobs.redis_store import RedisJobStore
from app.jobs.runner import runner_loop
from app.schemas.task import CancelResponse, TaskStatus


def create_app(
    settings: JobsSettings | None = None,
    store: RedisJobStore | None = None,
    query_service: QueryService | None = None,
    *,
    start_kafka: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = settings or JobsSettings()
        if start_kafka:
            # story 요청을 받는 API는 실행 함수도 준비된 상태여야 한다.
            handler_for(cfg, "story")
        setup_logging(cfg.LOG_LEVEL, cfg.RELEASE)
        redis = (
            None
            if store
            else Redis.from_url(
                cfg.REDIS_URL.get_secret_value(),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
        )
        jobs = store or RedisJobStore(redis, cfg)
        runtime = KafkaRuntime(jobs) if start_kafka else None
        app.state.settings = cfg
        app.state.jobs = jobs
        app.state.query_service = query_service
        app.state.kafka = runtime
        story_runner = None
        try:
            await jobs.redis.ping()
            if runtime:
                await runtime.start()
                story_runner = asyncio.create_task(runner_loop(jobs, "story"))
            app.state.story_runner = story_runner
            yield
        finally:
            if story_runner:
                story_runner.cancel()
                await asyncio.gather(story_runner, return_exceptions=True)
            if runtime:
                await runtime.stop()
            if redis:
                await redis.aclose()

    app = FastAPI(title="여담 AI v2", lifespan=lifespan)
    register_error_handlers(app)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        bind_request(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RedisError)
    async def redis_unavailable(request: Request, exc: RedisError):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "작업 저장소에 연결할 수 없습니다.",
                    "detail": {},
                }
            },
        )

    @app.get(
        "/trips/{trip_id}/{kind}",
        response_model=JobStatusResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def status(trip_id: int, kind: Literal["process", "story"], request: Request):
        return (await request.app.state.jobs.latest(trip_id, kind)).response()

    @app.delete(
        "/trips/{trip_id}/{kind}",
        response_model=CancelResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def cancel(trip_id: int, kind: Literal["process", "story"], request: Request):
        record = await request.app.state.jobs.cancel(trip_id, kind)
        # DELETE의 CANCELED는 명세의 취소 접수 응답이다. 실제 중단은 GET과 결과로 확인한다.
        return CancelResponse(
            trip_id=trip_id, execution_id=record.execution_id, status=TaskStatus.CANCELED
        )

    @app.get("/health")
    async def health(request: Request):
        from fastapi.responses import JSONResponse

        jobs = request.app.state.jobs
        ok = bool(await jobs.redis.ping())
        runtime = request.app.state.kafka
        ok = ok and (runtime is None or runtime.healthy())
        story_task = request.app.state.story_runner
        ok = ok and (story_task is None or not story_task.done())
        return JSONResponse(
            status_code=200 if ok else 503,
            content={
                "status": "ok" if ok else "degraded",
                "model_loaded": False,
                "model_name": request.app.state.settings.EMBED_MODEL,
                "active_tasks": await jobs.redis.zcard(jobs.key("running")),
                "queued": sum(
                    [
                        await jobs.redis.zcard(jobs.key(f"queue:{kind}"))
                        for kind in ("process", "story")
                    ]
                ),
                "idle_seconds": 0,
            },
        )

    app.include_router(query.router)
    return app
