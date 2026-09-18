"""이미지 바이트 소스, PipelineContext.images

파이프라인은 get_many(keys)만 호출하고 S3인지 폴더인지 모름
디코딩은 파이프라인 몫이라 여기서는 bytes까지
하나라도 못 받으면 부분 실패 없이 DownloadFailed(실패한 키 전부)
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import boto3
from botocore.config import Config

from app.core.errors import DownloadFailed
from app.core.logging import get_logger

logger = get_logger(__name__)


class ImageSource(Protocol):
    """규격. 상속 없이 get_many만 맞으면 이 타입으로 인정"""

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        """키 목록 → {키: 바이트}. 실패한 키가 있으면 DownloadFailed"""
        ...


class LocalDirImageSource:
    """폴더에서 읽음. 파이프라인 담당의 로컬 개발, 테스트, 평가셋"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        failed: list[str] = []
        for key in keys:
            path = self.root / key
            try:
                result[key] = path.read_bytes()
            except OSError:
                failed.append(key)
        if failed:
            raise DownloadFailed(failed)
        return result


class S3ImageSource:
    """S3에서 병렬로 받음. 자격 증명은 인스턴스 역할, 워커 운영용"""

    def __init__(
        self,
        bucket: str,
        region: str,
        max_workers: int = 8,
        retries: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.bucket = bucket
        self.max_workers = max_workers
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        # 클라이언트는 스레드 안전, 커넥션 풀은 스레드 수에 맞춤. SDK 자체 재시도는 끄고 여기서 제어
        self._client = boto3.client(
            "s3",
            region_name=region,
            config=Config(max_pool_connections=max_workers, retries={"max_attempts": 1}),
        )

    def _get_one(self, key: str) -> bytes | None:
        """retries회까지 시도, 간격은 backoff x 2^n. 끝내 실패하면 None"""
        for attempt in range(self.retries):
            try:
                body = self._client.get_object(Bucket=self.bucket, Key=key)["Body"]
                return body.read()
            except Exception as exc:  # noqa: BLE001, 스로틀링과 네트워크 오류 모두 재시도 대상
                if attempt < self.retries - 1:
                    delay = self.backoff_seconds * (2**attempt)
                    logger.warning(
                        "download retry",
                        extra={"key": key, "attempt": attempt + 1, "error": str(exc)},
                    )
                    time.sleep(delay)
                else:
                    logger.error("download failed", extra={"key": key, "error": str(exc)})
        return None

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for key, data in zip(keys, pool.map(self._get_one, keys), strict=True):
                if data is None:
                    failed.append(key)
                else:
                    result[key] = data
        if failed:
            raise DownloadFailed(failed)
        return result
