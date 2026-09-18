"""GET /health. 호출자가 인스턴스 정지를 판단하는 근거

ok 200, starting 503(모델 적재 중), degraded 503(워커나 qdrant 응답 없음).
워커와 qdrant 확인은 전부 await. 동기로 부르면 늦을 때 루프가 멈춤
"""

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.deps import ServiceDep, SettingsDep
from app.schemas.health import HealthResponse, HealthStatus

router = APIRouter(tags=["health"])


async def qdrant_ready(http: httpx.AsyncClient, base_url: str) -> bool:
    """qdrant GET /readyz. main.py가 app.state.qdrant_ready로 묶고 테스트는 가짜로 교체"""
    try:
        res = await http.get(f"{base_url}/readyz")
    except httpx.HTTPError:
        return False
    return res.status_code == 200


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def health(request: Request, service: ServiceDep, settings: SettingsDep) -> JSONResponse:
    ready = await request.app.state.worker.ready()
    qdrant_ok = await request.app.state.qdrant_ready()

    if ready is None or not qdrant_ok:
        status = HealthStatus.DEGRADED
    elif not ready.model_loaded:
        status = HealthStatus.STARTING
    else:
        status = HealthStatus.OK

    body = HealthResponse(
        status=status,
        model_loaded=ready.model_loaded if ready else False,
        model_name=ready.model_name if ready else settings.EMBED_MODEL,
        active_tasks=service.active_tasks(),
        queued=service.queued(),
        idle_seconds=service.idle_seconds(),
    )
    code = 200 if status is HealthStatus.OK else 503
    return JSONResponse(status_code=code, content=body.model_dump(mode="json"))
