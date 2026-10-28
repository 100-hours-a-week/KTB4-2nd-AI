"""파이프라인 컨텍스트, 인터페이스 4

파이프라인이 외부 자원에 접근하는 의존성을 한 객체로 묶어 run()에 주입.
워커는 기동 때 한 번 만들어 모든 작업이 공유.
로컬 개발과 테스트는 폴더와 :memory:와 가짜 엔진으로 만듦
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.core.config import Settings
from app.embedding.contracts import EmbeddingEngine
from app.infra.images import ImageSource
from app.infra.qdrant import QdrantStore
from app.pipeline.steps.filtering import PhotoMetrics

Thresholds = dict[str, float | int]


@dataclass(frozen=True)
class PipelineContext:
    """작업 중 바뀌지 않으므로 frozen"""

    images: ImageSource
    engine: EmbeddingEngine
    qdrant: QdrantStore
    thresholds: Thresholds  # 현재 모델의 임계값, thresholds.yaml의 model_version 아래
    settings: Settings
    log: logging.Logger
    # EMBEDDING에서 qdrant에 저장된 벡터를 먼저 쓰고 없는 사진만 임베딩. v2 EMBED_ON_PHOTOS_READY
    reuse_vectors: bool = False
    # 업로드 묶음에서 계산한 {사진 ID: 지표}. reuse_vectors이고 벡터도 저장된 사진은 다운로드 생략
    photo_metrics: Mapping[int, PhotoMetrics] = field(default_factory=dict)


def load_thresholds(path: Path, model_version: str) -> Thresholds:
    """thresholds.yaml에서 model_version 항목만 읽음. 없으면 기동 시점에 실패"""
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if model_version not in data:
        raise KeyError(f"{path}에 {model_version} 항목이 없음")
    return dict(data[model_version])
