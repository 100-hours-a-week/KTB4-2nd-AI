"""연결 확인 전용. 네트워크·모델·파일 접근 없이 실행하며 의미 검색을 하지 않는다."""

import json
from typing import Any

from app.infra.vlm import ChatMessage
from app.schemas.query import SearchHit
from app.search.context import SearchContext


class MockVLM:
    async def chat(
        self,
        messages: list[ChatMessage],
        image_urls: list[str],
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        # response_format은 받지만 모의 응답의 생성이나 검증에 사용하지 않는다.
        if image_urls:
            return f"[MOCK] 근거 사진 {len(image_urls)}장으로 답변 경로를 확인했습니다."
        query = json.loads(messages[-1]["content"])["query"]
        return json.dumps(
            {
                "intent": "QUESTION" if "?" in query else "SEARCH",
                "date_from": None,
                "date_to": None,
                "region_names": None,
                "visual_query": query,
                "parsed": True,
            },
            ensure_ascii=False,
        )


class MockEmbedder:
    async def encode_text(self, text: str) -> list[float]:
        return [1.0, 0.0]


class MockVectors:
    async def search(
        self, vector: list[float], *, candidate_ids: list[int], limit: int
    ) -> list[SearchHit]:
        return [
            SearchHit(trip_attachment_id=photo_id, score=1 / (index + 1))
            for index, photo_id in enumerate(candidate_ids[:limit])
        ]


class MockImages:
    async def resolve(self, storage_key: str) -> str:
        # URL을 실제로 요청하지 않는다. 사본 키를 외부 호스트에 전달하지 않는다.
        return "https://example.invalid/mock-photo.jpg"


def create_mock_context() -> SearchContext:
    return SearchContext(
        vlm=MockVLM(), embedder=MockEmbedder(), vectors=MockVectors(), images=MockImages()
    )
