"""워커 → api 콜백. POST /internal/trips/{trip_id}/progress, result, failed

러너 스레드에서 부르므로 동기 httpx. api 쪽이 루프 위 AsyncClient인 것과 반대.
실패해도 예외를 던지지 않음. progress 실패로 6분짜리 파이프라인을 죽일 이유가 없고,
result 실패는 할 수 있는 게 없음. api는 콜백이 120초 끊기면 WORKER_DEAD로 정리하고 백엔드가 재POST
"""

import time
from typing import Protocol

import httpx
from pydantic import BaseModel

from app.core.logging import get_logger
from app.schemas.internal import FailedIn, ProgressIn, ResultIn
from app.schemas.process import ProcessResult
from app.schemas.task import ErrorBody, ProcessStep

log = get_logger(__name__)


class CallbackProtocol(Protocol):
    """러너가 의존하는 모양. 테스트는 tests/fakes.py의 FakeCallback"""

    def progress(self, trip_id: int, step: ProcessStep, done: int, total: int) -> None: ...

    def result(self, trip_id: int, result: ProcessResult) -> None: ...

    def failed(self, trip_id: int, error: ErrorBody) -> None: ...


class CallbackClient:
    def __init__(
        self,
        api_url: str,
        retries: int = 3,
        timeout: float = 5.0,
        backoff_seconds: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._retries = max(retries, 1)
        self._backoff = backoff_seconds
        self._client = httpx.Client(base_url=api_url, timeout=timeout, transport=transport)

    def progress(self, trip_id: int, step: ProcessStep, done: int, total: int) -> None:
        self._post(trip_id, "progress", ProgressIn(step=step, done=done, total=total))

    def result(self, trip_id: int, result: ProcessResult) -> None:
        self._post(trip_id, "result", ResultIn(result=result))

    def failed(self, trip_id: int, error: ErrorBody) -> None:
        self._post(trip_id, "failed", FailedIn.model_validate(error.model_dump()))

    def _post(self, trip_id: int, kind: str, body: BaseModel) -> None:
        """204면 끝. 404는 api가 재기동돼 task가 없는 것이라 중단. 5xx와 연결 실패는 재시도"""
        path = f"/internal/trips/{trip_id}/{kind}"
        payload = body.model_dump(mode="json")
        for attempt in range(self._retries):
            try:
                response = self._client.post(path, json=payload)
            except httpx.HTTPError as e:
                log.warning(
                    "콜백 연결 실패", extra={"trip_id": trip_id, "kind": kind, "reason": str(e)}
                )
            else:
                if response.status_code < 500:
                    if response.status_code != 204:
                        log.warning(
                            "콜백 거절",
                            extra={
                                "trip_id": trip_id,
                                "kind": kind,
                                "status": response.status_code,
                            },
                        )
                    return
                log.warning(
                    "콜백 서버 오류",
                    extra={"trip_id": trip_id, "kind": kind, "status": response.status_code},
                )
            if attempt < self._retries - 1:
                time.sleep(self._backoff * (2**attempt))
        log.error("콜백 포기", extra={"trip_id": trip_id, "kind": kind})

    def close(self) -> None:
        self._client.close()
