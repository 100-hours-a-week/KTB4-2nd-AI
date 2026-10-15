"""선별된 분석용 사진에만 S3 GET URL을 발급한다. 이미지 바이트는 내려받지 않는다."""

import asyncio
from threading import Lock
from typing import Self

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings
from app.infra.vlm import VLMUnavailable


class S3ImageResolver:
    def __init__(self, client, bucket: str, *, expires_sec: int = 300) -> None:
        if not bucket.strip() or not 1 <= expires_sec <= 604800:
            raise ValueError("S3 버킷과 1~604800초 사이의 URL 유효기간이 필요합니다.")
        self._client, self._bucket, self._expires_sec = client, bucket, expires_sec

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        if (
            settings.IMAGE_SOURCE != "s3"
            or not settings.S3_BUCKET
            or not settings.S3_BUCKET.strip()
        ):
            raise ValueError(
                "실제 검색 사진 URL 발급에는 IMAGE_SOURCE=s3와 S3_BUCKET이 필요합니다."
            )
        client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
                connect_timeout=5,
                read_timeout=5,
                retries={"max_attempts": 1},
            ),
        )
        return cls(client, settings.S3_BUCKET, expires_sec=settings.S3_PRESIGN_EXPIRES_SEC)

    @classmethod
    async def create(cls, settings: Settings) -> Self:
        """자격 증명 초기화를 스레드에서 수행하고 취소 시 생성 결과도 닫는다."""
        lock = Lock()
        cancelled = False
        created = None

        def build():
            nonlocal created
            resolver = cls.from_settings(settings)
            with lock:
                if not cancelled:
                    created = resolver
                    return resolver
            # 취소가 먼저 일어나면 이 스레드가 뒤늦게 생성된 클라이언트를 닫는다.
            resolver._client.close()
            return None

        try:
            return await asyncio.to_thread(build)
        except asyncio.CancelledError:
            with lock:
                cancelled = True
                resolver = created
            if resolver is not None:
                # 생성이 먼저 끝난 경우다. 추가 취소가 와도 close 실행은 유지한다.
                closing = asyncio.get_running_loop().run_in_executor(None, resolver._client.close)
                await asyncio.shield(closing)
            raise

    async def resolve(self, storage_key: str) -> str:
        try:
            if not storage_key or len(storage_key.encode("utf-8")) > 1024:
                raise ValueError("분석용 사진 키가 올바르지 않습니다.")
            # 자격 증명 갱신이 네트워크를 사용할 수 있어 이벤트 루프 밖에서 서명한다.
            return await asyncio.to_thread(
                self._client.generate_presigned_url,
                "get_object",
                Params={"Bucket": self._bucket, "Key": storage_key},
                ExpiresIn=self._expires_sec,
                HttpMethod="GET",
            )
        except (BotoCoreError, ClientError, ValueError):
            # 기존 검색의 답변 실패 계약에 맞춰 사진 목록은 유지한다.
            # 저장소 키나 자격 증명이 포함된 외부 예외 본문은 노출하지 않는다.
            raise VLMUnavailable("답변용 사진의 접근 URL을 발급하지 못했습니다.") from None

    async def aclose(self) -> None:
        await asyncio.to_thread(self._client.close)
