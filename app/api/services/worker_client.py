"""워커 프로세스 호출, api → worker

api 안에서 워커의 URL과 HTTP를 아는 유일한 파일. task_service는 SubmitResult 값만 봄.
전부 httpx.AsyncClient로 await. 동기로 부르면 워커가 늦을 때 루프 전체가 멈춰
대기 중인 POST뿐 아니라 GET 진행률과 콜백까지 밀림
"""

from enum import StrEnum
from typing import Protocol

import httpx

from app.schemas.internal import JobRequest, ReadyResponse
from app.schemas.process import ProcessRequest


class SubmitResult(StrEnum):
    """워커 POST /jobs의 응답을 세 값으로. 나머지 상태 코드와 연결 실패는 WorkerUnavailable"""

    ACCEPTED = "ACCEPTED"  # 202, 실행 시작
    BUSY = "BUSY"  # 409, 이미 실행 중. api는 QUEUED 유지
    NOT_READY = "NOT_READY"  # 503, 모델 적재 중. api는 QUEUED 유지


class WorkerUnavailable(Exception):
    """연결 실패, 타임아웃, 예상 밖 상태 코드. task_service가 WORKER_DEAD로 바꿈"""


class WorkerClientProtocol(Protocol):
    """task_service가 의존하는 모양. 테스트는 tests/fakes.py의 FakeWorkerClient"""

    async def submit(self, trip_id: int, request: ProcessRequest) -> SubmitResult: ...

    async def cancel(self, trip_id: int) -> None: ...

    async def ready(self) -> ReadyResponse | None: ...

    async def aclose(self) -> None: ...


class WorkerClient:
    """같은 컨테이너의 워커(127.0.0.1:8002). 타임아웃이 짧은 이유는 localhost라 늦으면 죽은 것"""

    def __init__(self, base_url: str, timeout: float = 2.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def submit(self, trip_id: int, request: ProcessRequest) -> SubmitResult:
        body = JobRequest(trip_id=trip_id, request=request).model_dump(mode="json")
        try:
            res = await self._client.post("/jobs", json=body)
        except httpx.HTTPError as e:
            raise WorkerUnavailable(str(e)) from e
        if res.status_code == 202:
            return SubmitResult.ACCEPTED
        if res.status_code == 409:
            return SubmitResult.BUSY
        if res.status_code == 503:
            return SubmitResult.NOT_READY
        raise WorkerUnavailable(f"워커 응답 {res.status_code}")

    async def cancel(self, trip_id: int) -> None:
        """워커가 404(이미 끝남)를 주거나 연결이 안 돼도 무시. 죽은 워커는 watchdog이 정리"""
        try:
            await self._client.post(f"/jobs/{trip_id}/cancel")
        except httpx.HTTPError:
            pass

    async def ready(self) -> ReadyResponse | None:
        try:
            res = await self._client.get("/ready")
            res.raise_for_status()
            return ReadyResponse.model_validate(res.json())
        except (httpx.HTTPError, ValueError):
            return None

    async def aclose(self) -> None:
        await self._client.aclose()
