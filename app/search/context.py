"""외부 자원의 비동기 인터페이스. 실제 연결과 자원 종료는 앱 조립부가 담당한다."""

from dataclasses import dataclass
from typing import Protocol

from app.embedding.contracts import TextEmbedder as TextEmbedder
from app.infra.vlm import VlmClient
from app.schemas.query import SearchCandidate, SearchHit


class ModelUnavailable(Exception):
    """임베딩 또는 벡터 검색 의존성을 사용할 수 없음."""


class InvalidModelOutput(Exception):
    """VLM 출력 형식 검증 실패. 프롬프트와 응답 검증은 검색 담당자 소유."""


@dataclass(frozen=True)
class Evidence:
    candidate: SearchCandidate
    image_url: str


class CandidateSearcher(Protocol):
    async def search(
        self, vector: list[float], *, candidate_ids: list[int], limit: int
    ) -> list[SearchHit]:
        """DB 질의 자체를 candidate_ids로 제한. 모델/차원 일치는 어댑터 책임."""
        ...


class ImageResolver(Protocol):
    async def resolve(self, storage_key: str) -> str:
        """선별된 사본의 VLM 접근 URL. 실제 구현은 신뢰된 버킷에서만 발급."""
        ...


@dataclass(frozen=True)
class SearchContext:
    vlm: VlmClient
    embedder: TextEmbedder
    vectors: CandidateSearcher
    images: ImageResolver


@dataclass(frozen=True)
class SearchOptions:
    """개발용 시간 상한과 k. 서비스 성능 측정으로 확정한 값이 아니다."""

    question_k: int = 5
    parse_timeout_sec: float = 5.0
    retrieval_timeout_sec: float = 5.0
    answer_timeout_sec: float = 10.0
    request_timeout_sec: float = 30.0

    def __post_init__(self) -> None:
        if self.question_k < 1:
            raise ValueError("question_k는 양수여야 합니다.")
        times = (
            self.parse_timeout_sec,
            self.retrieval_timeout_sec,
            self.answer_timeout_sec,
            self.request_timeout_sec,
        )
        if any(value <= 0 for value in times):
            raise ValueError("시간 상한은 양수여야 합니다.")
