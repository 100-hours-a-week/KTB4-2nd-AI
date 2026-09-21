"""가짜 임베딩 엔진. FAKE_PIPELINE=1과 테스트에서 진짜 모델 자리에

tests가 아니라 app에 있는 이유는 .dockerignore가 tests를 제외해 컨테이너에서도 돌아야 하기 때문.
같은 사진은 항상 같은 벡터. 테스트가 단언할 수 있고, 벡터를 비교하는 코드가 붙어도 가짜로 검증 가능
"""

import hashlib

import numpy as np
from PIL import Image

from app.engines.base import EmbeddingEngine

# 진짜 엔진과 같은 차원. Qdrant 컬렉션 크기가 여기서 나옴
DIM = 1152


class FakeEngine(EmbeddingEngine):
    """이미지를 8×8로 줄인 바이트를 시드로 정규분포 벡터를 만들어 L2 정규화

    model_version이 "fake"라 컬렉션이 vectors_fake로 분리되어 진짜 벡터와 섞이지 않음
    """

    model_version = "fake"
    dim = DIM

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.stack([self._one(image) for image in images])
        return vectors.astype(np.float32)

    def _one(self, image: Image.Image) -> np.ndarray:
        thumb = image.convert("RGB").resize((8, 8))
        digest = hashlib.blake2b(thumb.tobytes(), digest_size=8).digest()
        seed = int.from_bytes(digest, "big")
        vector = np.random.default_rng(seed).standard_normal(self.dim)
        return vector / np.linalg.norm(vector)
