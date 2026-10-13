"""검색과 사진 정리가 공유하는 임베딩 계약. 모델 로딩과 서버 실행은 포함하지 않는다."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from PIL import Image


class EmbeddingEngine(ABC):
    """사진 정리에서 사용할 이미지 임베딩 계약. 구현체가 모델 버전과 차원을 제공한다"""

    model_version: str  # Qdrant 컬렉션 이름 vectors_{model_version}에 들어감
    dim: int  # 벡터 차원, 컬렉션 생성 시 size

    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """디코딩된 PIL 이미지 목록 → (N, dim) float32, 행마다 L2 정규화

        배치 크기와 스레드 수는 구현이 정함. 빈 목록이면 (0, dim)
        """


class TextEmbedder(Protocol):
    async def encode_text(self, text: str) -> list[float]:
        """사진 벡터와 비교할 수 있도록 같은 모델과 차원 및 정규화 설정을 사용한다."""
        ...
