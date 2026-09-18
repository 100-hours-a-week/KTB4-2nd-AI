"""임베딩 엔진 인터페이스, 인터페이스 2

파이프라인은 "이미지 목록을 넣으면 L2 정규화된 (N, dim) 벡터가 나온다"만 알면 됨.
구현(siglip_naflex.py)과 생성 방법은 파이프라인 담당.
워커는 기동 때 한 번 만들어 PipelineContext에 넣음
"""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class EmbeddingEngine(ABC):
    """구현 클래스는 model_version과 dim을 클래스 속성으로 두고 encode_images를 구현"""

    model_version: str  # Qdrant 컬렉션 이름 vectors_{model_version}에 들어감
    dim: int  # 벡터 차원, 컬렉션 생성 시 size

    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """디코딩된 PIL 이미지 목록 → (N, dim) float32, 행마다 L2 정규화

        배치 크기와 스레드 수는 구현이 정함. 빈 목록이면 (0, dim)
        """
