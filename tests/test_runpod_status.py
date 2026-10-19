"""RunPod 상태 API. Pod 상태와 모델 준비 여부의 판정, 결과 재사용, 로그, 인증을 확인한다."""

import asyncio

import httpx
import pytest

from app.api.services.runpod_status import RunpodStatus, pod_id_from
from app.core.logging import JsonFormatter
from app.v2_main import create_app
from tests.observability_helpers import events, info_logs  # noqa: F401
from tests.test_v2_jobs import environment, scenario

VLM_URL = "https://abc123xyz-8000.proxy.runpod.net/v1"
READY = {"object": "list", "data": [{"id": "gemma-4", "object": "model"}]}


class Upstream:
    """RunPod API와 vLLM을 흉내 낸다. 값이 dict면 200 본문, int면 상태 코드, 예외면 발생"""

    def __init__(self, pod=None, models=None):
        self.pod = {"desiredStatus": "RUNNING"} if pod is None else pod
        self.models = READY if models is None else models
        self.calls = {"runpod": 0, "vlm": 0}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "rest.runpod.io":
            assert request.url.path == "/v1/pods/abc123xyz"
            assert request.headers["authorization"] == "Bearer rp-key"
            self.calls["runpod"] += 1
            return self.reply(self.pod, request)
        assert request.url.host == "abc123xyz-8000.proxy.runpod.net"
        assert request.url.path == "/v1/models"
        assert request.headers["authorization"] == "Bearer vlm-key"
        self.calls["vlm"] += 1
        return self.reply(self.models, request)

    @staticmethod
    def reply(value, request):
        if isinstance(value, Exception):
            raise value
        if isinstance(value, int):
            return httpx.Response(value, request=request)
        if isinstance(value, bytes):
            return httpx.Response(200, content=value, request=request)
        return httpx.Response(200, json=value, request=request)

    def checker(self, **changes) -> RunpodStatus:
        options = {
            "vlm_base_url": VLM_URL,
            "vlm_model": "gemma-4",
            "vlm_api_key": "vlm-key",
            "runpod_api_key": "rp-key",
            "transport": httpx.MockTransport(self.handler),
        } | changes
        return RunpodStatus(**options)


async def check(upstream: Upstream, **changes):
    async with upstream.checker(**changes) as checker:
        return (await checker.status()).model_dump()


def test_pod_id_is_read_from_runpod_proxy_url():
    assert pod_id_from(VLM_URL) == "abc123xyz"
    assert pod_id_from("https://ABC123XYZ-8000.proxy.runpod.net/v1/") == "abc123xyz"
    assert pod_id_from("http://127.0.0.1:8000/v1") is None
    assert pod_id_from("https://abc123xyz-8000.proxy.runpod.net.example.com/v1") is None
    assert pod_id_from(None) is None


@pytest.mark.parametrize(
    ("pod", "models", "expected"),
    [
        ({"desiredStatus": "RUNNING"}, READY, {"running": True, "ready": True}),
        # Pod는 켜졌지만 모델 적재가 끝나지 않아 프록시가 오류를 준다.
        ({"desiredStatus": "RUNNING"}, 502, {"running": True, "ready": False}),
        (
            {"desiredStatus": "EXITED"},
            httpx.ConnectError("down"),
            {"running": False, "ready": False},
        ),
        ({"desiredStatus": "TERMINATED"}, 404, {"running": False, "ready": False}),
        (
            {"desiredStatus": "RUNNING"},
            {"data": [{"id": "other-model"}]},
            {"running": True, "ready": False},
        ),
        ({"desiredStatus": "RUNNING"}, {"data": "broken"}, {"running": True, "ready": False}),
    ],
)
@scenario
async def test_status_reports_pod_state_and_model_readiness(pod, models, expected):
    assert await check(Upstream(pod, models)) == expected


@pytest.mark.parametrize(
    "pod",
    [401, 404, httpx.ReadTimeout("slow"), {"desiredStatus": "PAUSED"}, ["RUNNING"], b"not json"],
)
@scenario
async def test_unreadable_runpod_api_makes_running_null(pod):
    assert await check(Upstream(pod)) == {"running": None, "ready": True}


@scenario
async def test_missing_settings_skip_the_affected_check():
    upstream = Upstream()
    assert await check(upstream, runpod_api_key=None) == {"running": None, "ready": True}
    assert upstream.calls == {"runpod": 0, "vlm": 1}

    # RunPod 프록시가 아닌 주소에서는 Pod ID를 알 수 없다.
    async with RunpodStatus(
        vlm_base_url="http://127.0.0.1:8000/v1",
        vlm_model="gemma-4",
        runpod_api_key="rp-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=READY)),
    ) as checker:
        assert (await checker.status()).model_dump() == {"running": None, "ready": True}

    upstream = Upstream()
    assert await check(upstream, vlm_base_url=None) == {"running": None, "ready": False}
    assert upstream.calls == {"runpod": 0, "vlm": 0}


@scenario
async def test_slow_upstream_is_cut_at_timeout():
    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json=READY, request=request)

    async with RunpodStatus(
        vlm_base_url=VLM_URL,
        vlm_model="gemma-4",
        runpod_api_key="rp-key",
        timeout_sec=0.05,
        transport=httpx.MockTransport(slow),
    ) as checker:
        started = asyncio.get_running_loop().time()
        result = await checker.status()
    assert result.model_dump() == {"running": None, "ready": False}
    assert asyncio.get_running_loop().time() - started < 0.5


@scenario
async def test_results_are_reused_within_cache_window():
    upstream = Upstream()
    async with upstream.checker(cache_sec=10) as checker:
        await asyncio.gather(*[checker.status() for _ in range(5)])
        await checker.status()
    assert upstream.calls == {"runpod": 1, "vlm": 1}

    upstream = Upstream()
    async with upstream.checker(cache_sec=0) as checker:
        await checker.status()
        await checker.status()
    assert upstream.calls == {"runpod": 2, "vlm": 2}


@scenario
async def test_status_logs_only_changes_without_keys_or_addresses(caplog):
    upstream = Upstream(models=502)
    async with upstream.checker(cache_sec=0) as checker:
        await checker.status()
        await checker.status()
        upstream.models = READY
        await checker.status()
    logs = events(caplog, "ai_runpod_status")
    assert [(r.running, r.ready, r.pod_status, r.vlm_error) for r in logs] == [
        (True, False, "RUNNING", "http_502"),
        (True, True, "RUNNING", None),
    ]
    # httpx의 요청 로그는 운영 설정(setup_logging)에서 WARNING으로 올려 남지 않는다.
    text = "\n".join(JsonFormatter().format(r) for r in caplog.records if r.name.startswith("app."))
    assert "rp-key" not in text and "vlm-key" not in text and "proxy.runpod.net" not in text


@scenario
async def test_endpoint_requires_api_key_and_returns_both_values(monkeypatch):
    upstream = Upstream(models=502)
    original = RunpodStatus.from_settings.__func__
    monkeypatch.setattr(
        RunpodStatus,
        "from_settings",
        classmethod(
            lambda cls, settings: original(
                cls, settings, transport=httpx.MockTransport(upstream.handler)
            )
        ),
    )
    async with environment(
        VLM_BASE_URL=VLM_URL, VLM_MODEL="gemma-4", VLM_API_KEY="vlm-key", RUNPOD_API_KEY="rp-key"
    ) as (cfg, a, _, _):
        app = create_app(cfg, a, start_kafka=False)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://api.test"
            ) as client,
        ):
            assert (await client.get("/runpod/status")).status_code == 401
            response = await client.get("/runpod/status", headers={"Authorization": "Bearer test"})
    assert response.status_code == 200
    assert response.json() == {"running": True, "ready": False}


@scenario
async def test_endpoint_without_runpod_settings_reports_unknown_pod():
    async with environment() as (cfg, a, _, _):
        app = create_app(cfg, a, start_kafka=False)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://api.test"
            ) as client,
        ):
            response = await client.get("/runpod/status", headers={"Authorization": "Bearer test"})
    assert response.status_code == 200
    assert response.json() == {"running": None, "ready": False}
