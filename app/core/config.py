"""환경 변수 설정

모든 프로세스가 설정값을 여기서만 읽음. 변수 목록은 저장소 루트의 `.env.example`이 기준,
읽는 순서는 환경 변수 → `.env` 파일. 값이 없거나 타입이 틀리면 기동 시점에 실패
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 프로파일과 모델
    AI_PROFILE: Literal["cpu", "gpu"] = "cpu"
    EMBED_MODEL: str = "siglip2-so400m-naflex"
    EMBED_DEVICE: Literal["cpu", "cuda"] = "cpu"
    MODEL_PATH: Path = Path("models/siglip2")

    # 프로세스 주소, 컨테이너 안에서는 전부 127.0.0.1
    WORKER_URL: str = "http://127.0.0.1:8002"
    API_URL: str = "http://127.0.0.1:8000"
    QDRANT_URL: str = "http://127.0.0.1:6333"

    # 인증, 기본값 없음, 비어 있으면 기동 실패
    API_KEY: str

    # 이미지 소스
    IMAGE_SOURCE: Literal["s3", "local"] = "s3"
    S3_BUCKET: str | None = None
    AWS_REGION: str = "ap-northeast-2"
    IMAGE_DIR: Path = Path("data/photos")

    # 상한과 감시
    MAX_CONCURRENT: int = Field(default=1, ge=1)
    MAX_QUEUE: int = Field(default=20, ge=0)
    WORKER_DEAD_SEC: int = Field(default=120, ge=1)
    TASK_TIMEOUT_SEC: int = Field(default=1800, ge=1)
    WATCHDOG_INTERVAL_SEC: int = Field(default=5, ge=1)
    CALLBACK_RETRIES: int = Field(default=3, ge=0)

    # 파이프라인과 개발
    THRESHOLDS_PATH: Path = Path("app/pipeline/thresholds.yaml")
    FAKE_PIPELINE: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _require_bucket_for_s3(self) -> Self:
        if self.IMAGE_SOURCE == "s3" and not self.S3_BUCKET:
            raise ValueError("IMAGE_SOURCE=s3 이면 S3_BUCKET 이 필요함")
        return self


@lru_cache
def get_settings() -> Settings:
    """프로세스당 한 번만 읽어 캐시"""
    return Settings()
