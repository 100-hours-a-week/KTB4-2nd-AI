"""연결 확인 전용. 네트워크·모델·파일 접근 없이 실행하며 의미 검색을 하지 않는다."""

from app.schemas.query import ParseRequest, ParseResponse, SearchHit
from app.search.context import Evidence, SearchContext


class MockVLM:
    async def interpret(self, request: ParseRequest) -> ParseResponse:
        return ParseResponse(
            intent="QUESTION" if "?" in request.query else "SEARCH",
            date_from=None,
            date_to=None,
            region_names=None,
            visual_query=request.query,
            parsed=True,
        )

    async def answer(self, query: str, evidence: list[Evidence]) -> str:
        return f"[MOCK] 근거 사진 {len(evidence)}장으로 답변 경로를 확인했습니다."


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
