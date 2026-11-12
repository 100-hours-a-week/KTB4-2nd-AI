"""외부 자원의 비동기 인터페이스. 실제 연결과 자원 종료는 앱 조립부가 담당한다."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.embedding.contracts import TextEmbedder as TextEmbedder
from app.infra.vlm import VlmClient
from app.schemas.query import SearchCandidate, SearchHit

# 생성 vLLM 실행 인자 --limit-mm-per-prompt.image와 같은 값, 넘으면 답변 요청 자체가 실패
MAX_EVIDENCE_IMAGES = 5


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

    async def fetch(self, ids: list[int]) -> dict[int, np.ndarray]:
        """반환 후보의 사진 벡터, 결과 안 중복 제거용. 없는 id는 결과에서 빠짐"""
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
    """개발용 시간 상한과 k, 컷오프. 서비스 성능 측정으로 확정한 값이 아니다."""

    question_k: int = 5
    # 11/5 M1 확정값, 잘못된 거절 0건 안에서 Reject-F1 최대(개발셋 격자 탐색, 이웃 ±0.02에서도 유지)
    min_score: float = 0.25
    score_ratio: float = 0.80
    # 11/11 #192 확정값, 반환 사진끼리 코사인이 이 값 이상이면 점수가 낮은 쪽 제외
    # 개발셋 판정 165쌍에서 다른 사진의 잘못 제거 0장 안에서 같은 장면 쌍 최소
    dedup_threshold: float = 0.97
    # 생성 모델 31B(dense) 교체 뒤 11/4 스테이징 해석 3건이 모두 5초 초과, 스테이징 실측 뒤 재조정
    # 호출 하나가 BE의 AI 호출 읽기 상한 30초 안에 끝나야 BE 시간 초과 대신 대체 응답
    parse_timeout_sec: float = 15.0
    retrieval_timeout_sec: float = 5.0
    answer_timeout_sec: float = 20.0
    request_timeout_sec: float = 30.0

    def __post_init__(self) -> None:
        if not 1 <= self.question_k <= MAX_EVIDENCE_IMAGES:
            raise ValueError(f"question_k는 1 이상 {MAX_EVIDENCE_IMAGES} 이하여야 합니다.")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score는 0 이상 1 이하여야 합니다.")
        if not 0 < self.score_ratio <= 1:
            raise ValueError("score_ratio는 0 초과 1 이하여야 합니다.")
        if not 0 < self.dedup_threshold <= 1:
            raise ValueError("dedup_threshold는 0 초과 1 이하여야 합니다.")
        times = (
            self.parse_timeout_sec,
            self.retrieval_timeout_sec,
            self.answer_timeout_sec,
            self.request_timeout_sec,
        )
        if any(value <= 0 for value in times):
            raise ValueError("시간 상한은 양수여야 합니다.")
