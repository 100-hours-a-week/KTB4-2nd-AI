"""워크플로우 정리 파이프라인의 외부 자원과 실패. 정리 실행기가 작업마다 만들어
PipelineContext.workflow로 넘긴다.

파이프라인은 동기 스레드에서 돌고, VLM·카카오·이미지 클라이언트는 작업의 이벤트 루프에 묶여 있다.
submit은 그 루프에 코루틴을 넘기는 함수(asyncio.run_coroutine_threadsafe)다.
"""

from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from app.core.errors import ErrorCode, PipelineError
from app.infra.kakao_local import KakaoLocalClient
from app.infra.place_index import PlaceIndex, ensure_local_index, s3_downloader


class RemoteImages(Protocol):
    async def get(self, url: str) -> bytes | None:
        """TourAPI 대표 이미지 URL의 bytes. 실패하면 None(실험의 fetch_image와 같음)"""
        ...


@dataclass(frozen=True)
class WorkflowResources:
    vlm: Any  # VlmClient. chat(..., extra_body=...)로 mm_processor_kwargs를 넘길 수 있어야 함
    kakao: KakaoLocalClient
    places: PlaceIndex
    tour: list[dict]  # TourAPI 원자료(areaBasedList 전체) 레코드 목록
    remote_images: RemoteImages
    submit: Callable[[Coroutine[Any, Any, Any]], Future]
    # time.monotonic() 기준 워크플로우 단계의 마감. 작업 자원을 만든 때(다운로드 전)부터 잰다
    deadline: float
    concurrency: int = 4  # 결정 222(실험 Pod는 8)
    kakao_calls: int = 1000  # 작업당 카카오 호출 상한. 실험은 상한이 없었다(관찰 최대 604회)
    photo_seed: int = 61106  # 실험 기본 사진 순서 시드
    font_path: str | None = None
    vlm_retries: int = 3  # 다시 보낼 만한 실패를 다시 보내는 횟수(PROCESS_VLM_RETRIES, 결정 159)


class HttpRemoteImages:
    """실험 fetch_image와 같다. 상태 200이 아니거나 오류가 나면 None, 시간 제한 15초"""

    def __init__(self, timeout_sec: float = 15.0, transport=None) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout_sec, headers={"User-Agent": "yeodam/1.0"}, transport=transport
        )

    async def get(self, url: str) -> bytes | None:
        try:
            response = await self._client.get(url)
        except httpx.HTTPError:
            return None
        return response.content if response.status_code == 200 else None

    async def aclose(self) -> None:
        await self._client.aclose()


def prepare_tour_raw(cfg) -> Path | None:
    """실행기 시작 때 한 번 부른다. 공공 장소 색인과 같은 방식으로 S3에서 받고 SHA-256을 확인한다.

    설정이 없으면 None. 워크플로우를 끈 실행기는 받지 않는다.
    """
    if not cfg.PROCESS_WORKFLOW_ENABLED or not cfg.TOURAPI_RAW_PATH:
        return None
    path = Path(cfg.TOURAPI_RAW_PATH)
    if cfg.TOURAPI_RAW_S3_URI:
        ensure_local_index(
            path, cfg.TOURAPI_RAW_SHA256, s3_downloader(cfg.TOURAPI_RAW_S3_URI, cfg.AWS_REGION)
        )
    # 시작할 때 한 번 읽어 파일 손상을 작업 전에 드러낸다.
    load_tour_raw(str(path))
    return path


def load_tour_raw(path: str) -> list[dict]:
    """TourAPI 원자료 파일에서 레코드 목록을 꺼낸다(실험 naming.tour_raw와 같음)."""
    import json

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return next(v for v in data.values() if isinstance(v, list))


class WorkflowTimeout(PipelineError):
    """워크플로우 단계의 시간 예산 초과나, 다시 보낸 뒤에도 생성 호출 시간 초과"""

    code = ErrorCode.TIMEOUT


class WorkflowUnavailable(PipelineError):
    """생성 서버 호출 실패(다시 보낸 뒤에도 실패, 불완전 응답, 슬롯 대기 초과). 작업을 실패시킨다.

    출력 토큰 상한에 걸려 잘린 응답은 불완전 응답에 넣지 않는다(결정 224).
    """

    code = ErrorCode.MODEL_NOT_READY
