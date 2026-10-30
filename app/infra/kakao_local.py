"""카카오 로컬 API의 키워드 장소 검색과 좌표의 행정구역 변환. 정리 에이전트가 쓴다(#151, #153).

REST 키는 실행기 비밀값(KAKAO_REST_API_KEY)으로만 받는다. 하루 무료 한도는 모든 작업이 함께 쓰므로
작업마다 KakaoBudget을 새로 만들어 호출 수를 제한한다. 한도를 넘기면 호출하지 않고 None을 돌려준다.
"""

import math
import time
from dataclasses import dataclass
from typing import Self

import httpx

from app.observability.calls import log_model_call

BASE_URL = "https://dapi.kakao.com/v2/local/"
MAX_KEYWORD_RADIUS_M = 20_000
MAX_KEYWORD_SIZE = 15


class KakaoUnavailable(Exception):
    """카카오 호출 실패. 키, 인증키, 응답 본문은 메시지에 담지 않는다."""


class KakaoBudget:
    """작업 하나의 카카오 호출 상한. 자식 프로세스 안에서만 쓰므로 잠금 없이 센다."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("카카오 호출 상한은 0 이상이어야 합니다.")
        self.limit, self.used = limit, 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


@dataclass(frozen=True)
class KakaoPlace:
    candidate_id: str
    name: str
    category: str
    address: str
    latitude: float
    longitude: float
    distance_m: int | None


@dataclass(frozen=True)
class KakaoRegion:
    sido: str
    sigungu: str


class KakaoLocalClient:
    def __init__(
        self,
        rest_api_key: str,
        *,
        timeout_sec: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not rest_api_key.strip():
            raise ValueError("카카오 REST 키가 필요합니다.")
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("카카오 시간 제한은 양의 유한 값이어야 합니다.")
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"KakaoAK {rest_api_key.strip()}"},
            timeout=timeout_sec,
            follow_redirects=False,
            transport=transport,
        )

    async def keyword(
        self,
        query: str,
        latitude: float,
        longitude: float,
        radius_m: int,
        budget: KakaoBudget,
        *,
        size: int = 10,
    ) -> list[KakaoPlace] | None:
        """반경 안에서 키워드로 장소를 찾는다. 상한을 넘기면 None, 결과가 없으면 빈 목록이다."""
        if not query.strip():
            return []
        if not budget.take():
            return None
        params = {
            "query": query,
            "x": f"{longitude:.6f}",
            "y": f"{latitude:.6f}",
            "radius": max(0, min(int(radius_m), MAX_KEYWORD_RADIUS_M)),
            "size": max(1, min(size, MAX_KEYWORD_SIZE)),
            "sort": "accuracy",
        }
        data = await self._get("search/keyword.json", params, "keyword")
        try:
            return [
                KakaoPlace(
                    candidate_id=f"kakao:{d['id']}",
                    name=d["place_name"],
                    category=d.get("category_name") or "",
                    address=d.get("road_address_name") or d.get("address_name") or "",
                    latitude=float(d["y"]),
                    longitude=float(d["x"]),
                    distance_m=int(d["distance"]) if d.get("distance") else None,
                )
                for d in data["documents"]
            ]
        except (KeyError, TypeError, ValueError):
            raise KakaoUnavailable("카카오 키워드 검색 응답 형식이 다릅니다.") from None

    async def region(
        self, latitude: float, longitude: float, budget: KakaoBudget
    ) -> KakaoRegion | None:
        """좌표가 속한 법정동 기준 시도와 시군구. 상한을 넘기거나 행정구역이 없으면 None이다."""
        if not budget.take():
            return None
        params = {"x": f"{longitude:.6f}", "y": f"{latitude:.6f}"}
        data = await self._get("geo/coord2regioncode.json", params, "region")
        try:
            for d in data["documents"]:
                if d.get("region_type") == "B":
                    return KakaoRegion(
                        sido=d["region_1depth_name"], sigungu=d["region_2depth_name"]
                    )
        except (KeyError, TypeError):
            raise KakaoUnavailable("카카오 행정구역 변환 응답 형식이 다릅니다.") from None
        return None

    async def _get(self, path: str, params: dict, operation: str) -> dict:
        started = time.monotonic()
        try:
            response = await self._client.get(path, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException:
            log_model_call("kakao", started, "timeout", operation=operation)
            raise TimeoutError("카카오 요청 시간이 초과되었습니다.") from None
        except httpx.HTTPStatusError as exc:
            log_model_call(
                "kakao",
                started,
                "http_error",
                operation=operation,
                status_code=exc.response.status_code,
            )
            raise KakaoUnavailable("카카오 요청에 실패했습니다.") from None
        except (httpx.HTTPError, ValueError) as exc:
            log_model_call(
                "kakao", started, "error", operation=operation, error_type=type(exc).__name__
            )
            raise KakaoUnavailable("카카오 요청에 실패했습니다.") from None
        log_model_call(
            "kakao", started, "success", operation=operation, status_code=response.status_code
        )
        if not isinstance(data, dict):
            raise KakaoUnavailable("카카오 응답 형식이 다릅니다.")
        return data

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
