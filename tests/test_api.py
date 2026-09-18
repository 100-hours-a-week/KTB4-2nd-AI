"""HTTP 경로. 인증, 검증, 동기 POST, 대기 중 GET과 DELETE, 콜백 출처, 헬스

ASGITransport는 lifespan을 돌리지 않으므로 app.router.lifespan_context로 직접 감쌈.
TestClient는 동기라 POST에서 30분 멈추므로 못 씀
"""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeWorkerClient
from tests.test_task_service import make_request

pytestmark = pytest.mark.anyio

AUTH = {"Authorization": "Bearer test-key"}


async def qdrant_ok() -> bool:
    return True


@pytest.fixture
def fake() -> FakeWorkerClient:
    return FakeWorkerClient(hold=True)


@pytest.fixture
def settings() -> Settings:
    return Settings(API_KEY="test-key", IMAGE_SOURCE="local")


@pytest.fixture
async def app(settings: Settings, fake: FakeWorkerClient) -> AsyncIterator[FastAPI]:
    application = create_app(settings=settings, worker=fake, qdrant_probe=qdrant_ok)
    async with application.router.lifespan_context(application):
        fake.service = application.state.task_service
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def body(n: int = 2) -> dict:
    return make_request(n).model_dump(mode="json")


async def start_post(
    client: httpx.AsyncClient, fake: FakeWorkerClient, trip_id: int
) -> asyncio.Task[httpx.Response]:
    """POST를 별도 태스크로 띄우고 워커가 받을 때까지 기다림. 그동안 GET, DELETE를 보낼 수 있음"""
    task = asyncio.create_task(client.post(f"/trips/{trip_id}/process", json=body(), headers=AUTH))
    while trip_id not in fake.submitted:
        await asyncio.sleep(0)
    return task


# ---------- 인증과 검증 ----------


async def test_missing_or_wrong_key_is_401(client: httpx.AsyncClient):
    res = await client.get("/trips/77/process")
    assert res.status_code == 401
    assert res.json()["error"]["code"] == "UNAUTHORIZED"

    res = await client.get("/trips/77/process", headers={"Authorization": "Bearer wrong"})
    assert res.status_code == 401


async def test_invalid_body_is_422_in_spec_format(client: httpx.AsyncClient):
    res = await client.post("/trips/77/process", json={"trip_name": "x"}, headers=AUTH)
    assert res.status_code == 422
    err = res.json()["error"]
    assert err["code"] == "INVALID_REQUEST"
    assert "errors" in err["detail"]


# ---------- 동기 POST ----------


async def test_post_waits_and_returns_completed(client: httpx.AsyncClient, fake: FakeWorkerClient):
    post = await start_post(client, fake, 77)
    assert not post.done()

    fake.finish(77)
    res = await post
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["progress"] == {"done": 2, "total": 2}
    assert data["current_step"] is None
    assert data["error"] is None
    assert len(data["result"]["unclassified"]) == 2
    assert data["result"]["unclassified"][0]["taken_at"].endswith("+09:00")


async def test_get_during_post_shows_processing(client: httpx.AsyncClient, fake: FakeWorkerClient):
    post = await start_post(client, fake, 77)

    res = await client.get("/trips/77/process", headers=AUTH)
    assert res.status_code == 200
    assert res.json()["status"] == "PROCESSING"
    assert res.json()["result"] is None

    fake.finish(77)
    assert (await post).json()["status"] == "COMPLETED"
    # 완료 뒤에도 GET으로 회수 가능
    res = await client.get("/trips/77/process", headers=AUTH)
    assert res.json()["status"] == "COMPLETED"


async def test_delete_during_post_releases_post_as_canceled(
    client: httpx.AsyncClient, fake: FakeWorkerClient
):
    post = await start_post(client, fake, 77)

    res = await client.delete("/trips/77/process", headers=AUTH)
    assert res.status_code == 200
    assert res.json() == {"trip_id": 77, "status": "CANCELED"}

    data = (await post).json()
    assert data["status"] == "CANCELED"
    assert data["result"] is None and data["error"] is None


async def test_duplicate_post_is_409(client: httpx.AsyncClient, fake: FakeWorkerClient):
    post = await start_post(client, fake, 77)
    res = await client.post("/trips/77/process", json=body(), headers=AUTH)
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "DUPLICATE_TASK"
    fake.finish(77)
    await post


async def test_unknown_trip_is_404_and_finished_cancel_is_409(
    client: httpx.AsyncClient, fake: FakeWorkerClient
):
    res = await client.get("/trips/999/process", headers=AUTH)
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "TASK_NOT_FOUND"

    post = await start_post(client, fake, 77)
    fake.finish(77)
    await post
    res = await client.delete("/trips/77/process", headers=AUTH)
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "TASK_NOT_CANCELABLE"


# ---------- 워커 콜백 ----------


async def test_internal_rejects_non_localhost(app: FastAPI):
    transport = httpx.ASGITransport(app=app, client=("10.0.0.5", 4321))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as remote:
        res = await remote.post(
            "/internal/trips/77/progress", json={"step": "EMBEDDING", "done": 1, "total": 2}
        )
    assert res.status_code == 401


async def test_internal_progress_updates_get(client: httpx.AsyncClient, fake: FakeWorkerClient):
    post = await start_post(client, fake, 77)
    res = await client.post(
        "/internal/trips/77/progress", json={"step": "EMBEDDING", "done": 1, "total": 2}
    )
    assert res.status_code == 204

    data = (await client.get("/trips/77/process", headers=AUTH)).json()
    assert data["progress"] == {"done": 1, "total": 2}
    assert data["current_step"] == "EMBEDDING"

    fake.finish(77)
    await post


async def test_internal_failed_callback(client: httpx.AsyncClient, fake: FakeWorkerClient):
    post = await start_post(client, fake, 77)
    res = await client.post(
        "/internal/trips/77/failed",
        json={"code": "DOWNLOAD_FAILED", "message": "x", "detail": {"keys": ["k"]}},
    )
    assert res.status_code == 204

    data = (await post).json()
    assert data["status"] == "FAILED"
    assert data["error"] == {"code": "DOWNLOAD_FAILED", "message": "x", "detail": {"keys": ["k"]}}


# ---------- 헬스 ----------


async def test_health_ok(client: httpx.AsyncClient):
    res = await client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["active_tasks"] == 0 and data["queued"] == 0


async def test_health_starting_and_degraded(client: httpx.AsyncClient, fake: FakeWorkerClient):
    fake.model_loaded = False
    res = await client.get("/health")
    assert res.status_code == 503
    assert res.json()["status"] == "starting"

    fake.unreachable = True
    res = await client.get("/health")
    assert res.status_code == 503
    assert res.json()["status"] == "degraded"
    assert res.json()["model_name"] == "siglip2-so400m-naflex"
