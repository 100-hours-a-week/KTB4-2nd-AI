"""RunPod 생성 서버의 상태. Pod 설정 상태는 RunPod API, 생성 모델 준비 여부는 vLLM으로 확인한다.

Pod가 RUNNING이어도 모델 적재가 끝나기 전(재시작 약 3분)에는 vLLM이 요청을 받지 못해
두 값을 함께 준다.
"""

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Self

import httpx

from app.core.logging import get_logger
from app.jobs.config import JobsSettings
from app.schemas.runpod import RunpodStatusResponse

log = get_logger(__name__)

RUNPOD_API_URL = "https://rest.runpod.io/v1"
# RunPod HTTP 프록시 주소 https://{pod_id}-{port}.proxy.runpod.net
PROXY_HOST = re.compile(r"^(?P<pod>[a-z0-9]+)-\d+\.proxy\.runpod\.net$")
POD_RUNNING = {"RUNNING": True, "EXITED": False, "TERMINATED": False}


def pod_id_from(base_url: str | None) -> str | None:
    """VLM 주소에서 Pod ID를 읽는다. Pod를 새로 만들어 주소가 바뀌면 함께 바뀐다."""
    if not base_url:
        return None
    try:
        match = PROXY_HOST.match(httpx.URL(base_url).host)
    except httpx.InvalidURL:
        return None
    return match["pod"] if match else None


@dataclass(frozen=True)
class Probe:
    value: bool | None
    # 로그용 분류값. 성공하면 Pod 상태(RUNNING 등), 실패하면 timeout, http_401 같은 사유
    detail: str | None


class RunpodStatus:
    """결과를 cache_sec 동안 재사용한다. 동시에 들어온 요청은 진행 중인 확인을 함께 기다린다."""

    def __init__(
        self,
        *,
        vlm_base_url: str | None,
        vlm_model: str | None,
        vlm_api_key: str | None = None,
        runpod_api_key: str | None = None,
        runpod_api_url: str = RUNPOD_API_URL,
        cache_sec: float = 10,
        timeout_sec: float = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._vlm_url = vlm_base_url.rstrip("/") if vlm_base_url else None
        self._vlm_model = vlm_model
        self._vlm_key = vlm_api_key
        self._pod_id = pod_id_from(vlm_base_url)
        self._runpod_url = runpod_api_url.rstrip("/")
        self._runpod_key = runpod_api_key
        self._cache_sec = cache_sec
        self._timeout_sec = timeout_sec
        self._client = httpx.AsyncClient(
            timeout=timeout_sec, follow_redirects=False, transport=transport
        )
        self._lock = asyncio.Lock()
        self._cached: tuple[float, RunpodStatusResponse] | None = None
        self._last_logged: dict | None = None

    @classmethod
    def from_settings(
        cls, settings: JobsSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> Self:
        return cls(
            vlm_base_url=settings.VLM_BASE_URL,
            vlm_model=settings.VLM_MODEL,
            vlm_api_key=settings.VLM_API_KEY.get_secret_value() if settings.VLM_API_KEY else None,
            runpod_api_key=(
                settings.RUNPOD_API_KEY.get_secret_value() if settings.RUNPOD_API_KEY else None
            ),
            cache_sec=settings.RUNPOD_STATUS_CACHE_SEC,
            timeout_sec=settings.RUNPOD_STATUS_TIMEOUT_SEC,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def status(self) -> RunpodStatusResponse:
        async with self._lock:
            if self._cached and time.monotonic() - self._cached[0] < self._cache_sec:
                return self._cached[1]
            pod, model = await asyncio.gather(self._pod(), self._model())
            result = RunpodStatusResponse(running=pod.value, ready=bool(model.value))
            self._cached = (time.monotonic(), result)
            self._log_change(pod, model)
            return result

    async def _pod(self) -> Probe:
        if not self._runpod_key:
            return Probe(None, "not_configured")
        if not self._pod_id:
            return Probe(None, "pod_id_unknown")
        body = await self._get(f"{self._runpod_url}/pods/{self._pod_id}", self._runpod_key)
        if isinstance(body, Probe):
            return body
        state = body.get("desiredStatus") if isinstance(body, dict) else None
        if state not in POD_RUNNING:
            return Probe(None, "unknown_state")
        return Probe(POD_RUNNING[state], state)

    async def _model(self) -> Probe:
        if not self._vlm_url or not self._vlm_model:
            return Probe(False, "not_configured")
        body = await self._get(f"{self._vlm_url}/models", self._vlm_key)
        if isinstance(body, Probe):
            return Probe(False, body.detail)
        data = body.get("data") if isinstance(body, dict) else None
        items = data if isinstance(data, list) else []
        served = {item.get("id") for item in items if isinstance(item, dict)}
        if self._vlm_model not in served:
            return Probe(False, "model_missing")
        return Probe(True, None)

    async def _get(self, url: str, api_key: str | None):
        """성공하면 JSON 본문, 실패하면 사유를 담은 Probe. 주소와 키는 사유에 담지 않는다."""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            # httpx의 시간 제한은 연결과 읽기에 따로 적용되므로 호출 전체에도 같은 상한을 둔다.
            async with asyncio.timeout(self._timeout_sec):
                response = await self._client.get(url, headers=headers)
        except (TimeoutError, httpx.TimeoutException):
            return Probe(None, "timeout")
        except (httpx.HTTPError, httpx.InvalidURL):
            return Probe(None, "connect_error")
        if response.status_code != 200:
            return Probe(None, f"http_{response.status_code}")
        try:
            return response.json()
        except ValueError:
            return Probe(None, "invalid_response")

    def _log_change(self, pod: Probe, model: Probe) -> None:
        """폴링마다 남기지 않고 상태가 바뀔 때만 남긴다."""
        fields = {
            "running": pod.value,
            "ready": bool(model.value),
            "pod_status": pod.detail if pod.value is not None else None,
            "runpod_error": pod.detail if pod.value is None else None,
            "vlm_error": model.detail,
        }
        if fields != self._last_logged:
            self._last_logged = fields
            log.info("RunPod 상태 변경", extra={"event": "ai_runpod_status", **fields})
