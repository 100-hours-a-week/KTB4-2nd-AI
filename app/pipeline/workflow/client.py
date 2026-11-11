"""워크플로우의 모델 호출(실험의 Pod)과 장소 검색 도구(카카오, 공공 장소 색인, TourAPI 원자료).

- 모델 호출: temperature 0, SSE 스트리밍, 이미지 예산은 mm_processor_kwargs.max_soft_tokens(실험과
  같음). 실패 처리는 실험과 달리 정리 에이전트와 같은 규칙으로 작업을 실패시킨다(Pod 참고).
- 카카오와 대표 이미지 내려받기 실패는 빈 결과로 다룬다(실험 Near._get, Searcher._get, fetch_image와
  같음).
"""

import asyncio
import json
import re
from collections import Counter

from app.core.errors import PipelineError
from app.infra.kakao_local import KakaoBudget, KakaoUnavailable
from app.infra.vlm import ImagePart, TextPart, VLMTransientError, VLMUnavailable
from app.observability.langfuse_tracing import call_name
from app.pipeline.workflow.resources import WorkflowResources, WorkflowTimeout, WorkflowUnavailable

# n번째 재시도 전에 RETRY_WAIT_SEC × n초 기다린다(정리 에이전트와 같은 5, 10, 15초).
RETRY_WAIT_SEC = 5.0


def parse_json(text: str | None):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


class Pod:
    """모델 호출. content는 OpenAI 형식의 text와 image_url 목록이다.

    실패 처리는 정리 에이전트(결정 147·159)와 같다. 전송 오류, HTTP 408·429·5xx, 호출 시간 초과만
    retries번 다시 보내고(n번째는 RETRY_WAIT_SEC×n초 뒤), 그래도 실패하거나 다시 보낼 만하지 않은
    실패(불완전 응답, GPU 슬롯 대기 초과)면 작업을 실패시킨다(결정 221). 한 호출이 실패하면 그 뒤의
    호출은 보내지 않고 같은 실패를 올린다. 실험 코드는 실패한 호출을 답이 없는 것으로 다뤘다.
    """

    def __init__(self, vlm, concurrency: int, retries: int) -> None:
        self.vlm = vlm
        self.sem = asyncio.Semaphore(concurrency)
        self.retries = max(0, retries)
        self.done = 0  # 끝난 호출 수. 진행률 보고에 쓴다
        self.stats: Counter = Counter()
        self.failed: PipelineError | None = None

    async def chat(
        self, content: list[dict], max_tokens: int, budget: int | None = None, stage: str = ""
    ) -> dict:
        parts: list[TextPart | ImagePart] = []
        urls: list[str] = []
        for c in content:
            if c["type"] == "text":
                parts.append({"type": "text", "text": c["text"]})
            else:
                parts.append({"type": "image"})
                urls.append(c["image_url"]["url"])
        messages = [{"role": "user", "content": parts}]
        extra = {"mm_processor_kwargs": {"max_soft_tokens": budget}} if budget else None
        try:
            async with self.sem:
                for attempt in range(self.retries + 1):
                    self._raise_if_failed()
                    if attempt:
                        await asyncio.sleep(RETRY_WAIT_SEC * attempt)
                        self.stats["retries"] += 1
                        self._raise_if_failed()
                    self.stats["requests"] += 1
                    try:
                        with call_name(stage or "workflow"):
                            turn = await self.vlm.chat(
                                messages,
                                urls,
                                max_tokens,
                                temperature=0,
                                stream=True,
                                extra_body=extra,
                            )
                    except (VLMTransientError, TimeoutError) as exc:
                        if attempt == self.retries:
                            self._fail(exc, stage)
                        continue
                    except VLMUnavailable as exc:
                        self._fail(exc, stage)
                    self.stats["calls"] += 1
                    return {"text": turn.content, "finish": turn.finish_reason}
        finally:
            self.done += 1
        raise AssertionError("도달하지 않음")

    def _raise_if_failed(self) -> None:
        if self.failed is not None:
            raise type(self.failed)(str(self.failed))

    def _fail(self, exc: Exception, stage: str):
        self.stats["errors"] += 1
        self.stats[f"errors:{stage.split('-')[0]}"] += 1
        if isinstance(exc, TimeoutError):
            error: PipelineError = WorkflowTimeout("생성 서버의 응답 시간이 초과되었습니다.")
        else:
            error = WorkflowUnavailable("생성 서버를 사용할 수 없습니다.")
        if self.failed is None:
            self.failed = error
        raise error from exc


class Places:
    """카카오(작업당 호출 상한), 공공 장소 색인, TourAPI 원자료. 실패는 빈 결과로 다룬다."""

    def __init__(self, resources: WorkflowResources) -> None:
        self.kakao = resources.kakao
        self.index = resources.places
        self.tour = resources.tour
        self.remote = resources.remote_images
        self.budget = KakaoBudget(resources.kakao_calls)
        self.stats: Counter = Counter()

    async def keyword(self, query: str, **options) -> list:
        try:
            found = await self.kakao.search_keyword(query, self.budget, **options)
        except (KakaoUnavailable, TimeoutError):
            self.stats["kakao_errors"] += 1
            return []
        if found is None:
            self.stats["kakao_over_budget"] += 1
            return []
        return found

    async def category(self, code: str, lat: float, lon: float, radius_m: int, **options):
        """(장소 목록, 마지막 쪽인지). 실패와 상한 초과는 빈 마지막 쪽으로 다룬다."""
        try:
            found = await self.kakao.search_category(
                code, lat, lon, radius_m, self.budget, **options
            )
        except (KakaoUnavailable, TimeoutError):
            self.stats["kakao_errors"] += 1
            return [], True
        if found is None:
            self.stats["kakao_over_budget"] += 1
            return [], True
        return found

    async def region(self, lat: float, lon: float):
        try:
            return await self.kakao.region(lat, lon, self.budget)
        except (KakaoUnavailable, TimeoutError):
            self.stats["kakao_errors"] += 1
            return None

    async def image(self, url: str):
        """TourAPI 대표 이미지. 실패하면 None"""
        import io

        from PIL import Image

        try:
            data = await self.remote.get(url)
            if not data:
                return None
            return await asyncio.to_thread(lambda: Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:  # noqa: BLE001, 실험 fetch_image처럼 어떤 실패도 이미지 없음으로 다룬다
            return None
