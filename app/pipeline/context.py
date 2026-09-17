"""파이프라인 컨텍스트, 인터페이스 4

파이프라인이 외부 자원에 접근하는 의존성을 한 객체로 묶어 run()에 주입.
워커는 기동 때 한 번 만들어 모든 작업이 공유.
로컬 개발과 테스트는 폴더와 :memory:와 가짜 엔진으로 만듦
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.core.config import Settings
from app.engines.base import EmbeddingEngine
from app.infra.images import ImageSource
from app.infra.qdrant import QdrantStore

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


def load_thresholds(path: Path, model_version: str) -> Thresholds:
    """thresholds.yaml에서 model_version 항목만 읽음. 없으면 기동 시점에 실패"""
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if model_version not in data:
        raise KeyError(f"{path}에 {model_version} 항목이 없음")
    return dict(data[model_version])
