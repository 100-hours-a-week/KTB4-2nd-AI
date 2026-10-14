"""v2 전용 설정. v1의 기동 및 기본 제한을 변경하지 않는다."""

from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator

from app.core.config import Settings


class JobsSettings(Settings):
    JOBS_ENV: str = Field(default="local", pattern=r"^[a-z][a-z0-9-]*$")
    REDIS_URL: SecretStr = SecretStr("redis://127.0.0.1:6379/0")
    KAFKA_BOOTSTRAP_SERVERS: str = "127.0.0.1:9092"
    KAFKA_SECURITY_PROTOCOL: Literal["PLAINTEXT"] = "PLAINTEXT"
    KAFKA_TRIP_CREATE_REQUEST_TOPIC: str = "local.yeodam.trip-create.request"
    KAFKA_TRIP_CREATE_RESULT_TOPIC: str = "local.yeodam.trip-create.result"
    KAFKA_TRIP_UPDATE_REQUEST_TOPIC: str = "local.yeodam.trip-update.request"
    KAFKA_TRIP_UPDATE_RESULT_TOPIC: str = "local.yeodam.trip-update.result"
    KAFKA_STORY_CREATE_REQUEST_TOPIC: str = "local.yeodam.story-create.request"
    KAFKA_STORY_CREATE_RESULT_TOPIC: str = "local.yeodam.story-create.result"
    # 삭제 주기는 BE 재처리 운영 정책 합의 전까지 정하지 않는다. 완료 기록도 유지한다.
    JOBS_HEARTBEAT_SEC: float = Field(default=5, gt=0)
    JOBS_LEASE_SEC: float = Field(default=120, gt=0)
    JOBS_TIMEOUT_SEC: float = Field(default=300, gt=0)
    JOBS_PUBLISH_LEASE_SEC: float = Field(default=60, gt=0)
    JOBS_POLL_SEC: float = Field(default=0.5, gt=0)
    JOBS_FAKE_PIPELINE: bool = False
    # async def handler(request: dict, reporter: JobReporter) -> dict
    JOBS_PROCESS_HANDLER: str | None = None
    JOBS_STORY_HANDLER: str | None = None

    @model_validator(mode="after")
    def check_jobs_settings(self) -> Self:
        if self.JOBS_HEARTBEAT_SEC * 3 >= self.JOBS_LEASE_SEC:
            raise ValueError("작업 lease는 heartbeat 간격의 3배보다 길어야 함")
        topics = [
            getattr(self, name) for name in type(self).model_fields if name.endswith("_TOPIC")
        ]
        if len(set(topics)) != 6 or any(not topic.strip() for topic in topics):
            raise ValueError("요청과 결과 토픽 6개는 서로 다른 이름이어야 함")
        return self

    @property
    def redis_prefix(self) -> str:
        return f"yeodam:{self.JOBS_ENV}:ai:"

    @property
    def topics(self) -> dict[str, tuple[str, str]]:
        return {
            getattr(self, f"KAFKA_{work}_REQUEST_TOPIC"): (
                kind,
                getattr(self, f"KAFKA_{work}_RESULT_TOPIC"),
            )
            for work, kind in (
                ("TRIP_CREATE", "process"),
                ("TRIP_UPDATE", "process"),
                ("STORY_CREATE", "story"),
            )
        }

    @property
    def consumer_groups(self) -> dict[str, str]:
        return {
            self.KAFKA_TRIP_CREATE_REQUEST_TOPIC: "yeodam-ai-trip-create",
            self.KAFKA_TRIP_UPDATE_REQUEST_TOPIC: "yeodam-ai-trip-update",
            self.KAFKA_STORY_CREATE_REQUEST_TOPIC: "yeodam-ai-story-create",
        }
