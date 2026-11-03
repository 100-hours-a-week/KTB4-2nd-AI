"""정리 에이전트의 외부 자원과 실패. 정리 실행기가 작업마다 만들어 PipelineContext.agent로 넘긴다.

파이프라인은 동기 스레드에서 돌고 VLM과 카카오 클라이언트는 작업의 이벤트 루프에 묶여 있다.
submit은 그 루프에 코루틴을 넘기는 함수(asyncio.run_coroutine_threadsafe)다.
"""

from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.errors import ErrorCode, PipelineError
from app.infra.kakao_local import KakaoLocalClient
from app.infra.place_index import PlaceIndex
from app.infra.vlm import VlmClient


class ImageUrls(Protocol):
    async def resolve(self, storage_key: str) -> str:
        """분석용 사본 키를 생성 서버가 읽을 수 있는 URL로 바꾼다. 실패는 VLMUnavailable"""
        ...


@dataclass(frozen=True)
class AgentResources:
    vlm: VlmClient
    urls: ImageUrls
    places: PlaceIndex
    kakao: KakaoLocalClient
    submit: Callable[[Coroutine[Any, Any, Any]], Future]
    # time.monotonic() 기준 에이전트 단계의 마감. 작업 자원을 만든 때(다운로드 전)부터 잰다
    deadline: float
    concurrency: int = 4  # 동시에 보내는 생성 호출 수
    kakao_calls: int = 200  # 작업당 카카오 호출 상한
    vlm_retries: int = 3  # 생성 호출의 전송 오류, 408, 429, 5xx, 시간 초과를 다시 보내는 횟수


class AgentUnavailable(PipelineError):
    """생성 서버 호출 실패. 규칙 결과로 대신하지 않고 작업을 실패시킨다(결정 147).

    전송 오류와 HTTP 408, 429, 5xx는 vlm_retries번 다시 보낸 뒤에도 실패할 때만 온다(결정 159).
    """

    code = ErrorCode.MODEL_NOT_READY


class AgentTimeout(PipelineError):
    """에이전트 단계의 시간 예산 초과나 생성 호출 시간 초과(결정 147).

    호출 시간 초과는 vlm_retries번 다시 보낸 뒤에도 실패할 때만 온다(결정 159).
    """

    code = ErrorCode.TIMEOUT
