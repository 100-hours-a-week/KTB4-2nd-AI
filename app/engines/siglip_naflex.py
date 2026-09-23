from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Siglip2VisionModel

from app.engines.base import EmbeddingEngine


class SiglipNaflexEngine(EmbeddingEngine):
    model_version = "siglip2-so400m-naflex"
    dim = 1152

    def __init__(self, model_path: Path, device: str = "cpu", batch_size: int = 16):
        self.batch_size = batch_size
        self.device = device
        self.model = (
            Siglip2VisionModel.from_pretrained(
                model_path, local_files_only=True, dtype=torch.bfloat16
            )
            .to(device)
            .eval()
        )
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                vector = self.model(**inputs).pooler_output  # (B, D)

            # numpy는 bf16 지원하지 않으므로, ndarray 변환 전에 fp32로 바꿔 반환 계약을 맞춘다.
            chunks.append(vector.float().cpu().numpy())

        vectors = np.concatenate(chunks).astype(np.float32)  # (N, 1152), N은 전체 사진 수
        norm_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # 사진마다 정규화
        return norm_vectors
