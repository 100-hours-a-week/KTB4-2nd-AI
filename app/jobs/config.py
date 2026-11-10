"""v2 전용 설정. v1의 기동 및 기본 제한을 변경하지 않는다."""

import re
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator

from app.core.config import Settings

MAX_STORY_CONCURRENCY = 4


class JobsSettings(Settings):
    # v2 API에서 실제 검색 의존성을 조립한다. 로컬 작업 큐 검증은 비활성화해 사용한다.
    SEARCH_ENABLED: bool = False
    JOBS_ENV: str = Field(default="local", pattern=r"^[a-z][a-z0-9-]*$")
    REDIS_URL: SecretStr = SecretStr("redis://127.0.0.1:6379/0")
    KAFKA_BOOTSTRAP_SERVERS: str = "127.0.0.1:9092"
    KAFKA_SECURITY_PROTOCOL: Literal["PLAINTEXT", "SASL_SSL"] = "PLAINTEXT"
    # SASL_SSL에서만 사용. TLS로 서버 인증서와 호스트 이름을 검증하고 SCRAM으로 계정을 인증
    KAFKA_SASL_MECHANISM: Literal["SCRAM-SHA-256", "SCRAM-SHA-512"] = "SCRAM-SHA-512"
    KAFKA_SASL_USERNAME: str | None = None
    KAFKA_SASL_PASSWORD: SecretStr | None = None
    KAFKA_SSL_CAFILE: str | None = None
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
    # 정리 작업만의 제한 시간. 워크플로우 예산 3,600초(결정 222)에 결과 저장 여유 300초를 더한다.
    # 이전 값 3,000초는 BE의 결과 대기 한도 1시간보다 짧게 둔 값이었다(결정 142, 148).
    # 작업을 회수하는 API와 실행기가 같은 값을 써야 함
    JOBS_PROCESS_TIMEOUT_SEC: float = Field(default=3900, gt=0, allow_inf_nan=False)
    JOBS_PUBLISH_LEASE_SEC: float = Field(default=60, gt=0)
    JOBS_POLL_SEC: float = Field(default=0.5, gt=0)
    # 정리 실행기가 import와 처리기 적재까지 마쳐 두는 대기 자식 수, 0이면 작업마다 새로 기동
    JOBS_PROCESS_STANDBY: int = Field(default=2, ge=0)
    # true면 업로드 묶음을 미리 임베딩하고 정리에서 저장 벡터를 재사용
    # false면 정리 때 전부 임베딩하는 이전 동작, 되돌리기는 정리 실행기 환경 변수만 변경
    EMBED_ON_PHOTOS_READY: bool = True
    # 묶음 자식은 lease가 없어 이 시간을 넘기면 실행기가 강제 종료, 빠진 사진은 정리가 임베딩
    JOBS_PHOTO_BATCH_TIMEOUT_SEC: float = Field(default=60, gt=0, allow_inf_nan=False)
    # 실행 전의 선택적 묶음 정보만 만료한다. 최종 process는 전체 목록을 제공한다.
    JOBS_PHOTOS_READY_TTL_SEC: int = Field(default=7 * 86400, gt=0)
    # 업로드 묶음에서 계산한 사진 지표의 보관 기간, 만료된 사진은 정리가 다시 다운로드해 계산
    JOBS_PHOTO_METRICS_TTL_SEC: int = Field(default=7 * 86400, gt=0)
    JOBS_FAKE_PIPELINE: bool = False
    # async def handler(request: dict, reporter: JobReporter) -> dict
    JOBS_PROCESS_HANDLER: str | None = None
    JOBS_STORY_HANDLER: str | None = None
    # 각 API의 실행 루프 수와 Redis에서 허용하는 전체 스토리 실행 수.
    JOBS_STORY_CONCURRENCY: int = Field(default=4, ge=1, le=MAX_STORY_CONCURRENCY)
    JOBS_SHUTDOWN_GRACE_SEC: float = Field(default=60, ge=0, le=60, allow_inf_nan=False)
    # 실제 모델에서 품질과 메모리를 확인해 조정할 초기 설정값이다.
    STORY_IMAGE_BUDGET: int = Field(default=10, ge=1, le=30)
    STORY_BLUR_VAR_MIN: float = Field(default=80, ge=0, allow_inf_nan=False)
    STORY_PHASH_MAX_DIST: int = Field(default=4, ge=0, le=64)
    STORY_DEDUPE_Z: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    STORY_MAX_TOKENS: int = Field(default=2048, ge=1)
    # 분위기 변경 미리보기(동기 HTTP). 검색처럼 실제 VLM과 S3 URL 발급기를 연결한다.
    # 처리 상한 300초는 위키 계약 값이고, BE 읽기 상한은 320초 이상이다.
    STORY_REWRITE_ENABLED: bool = False
    STORY_REWRITE_TIMEOUT_SEC: float = Field(default=300, gt=0, allow_inf_nan=False)
    # 블록 30개 출력의 초기 상한(11/3 R1 조사). Pod 측정 뒤 조정한다.
    STORY_REWRITE_MAX_TOKENS: int = Field(default=6144, ge=1)
    # API 복제본과 스토리 프로세스가 공유하는 HTTP 호출 한도. GPU 성능 보장값은 아니다.
    GPU_GENERATION_INFLIGHT: int = Field(default=8, ge=1)
    GPU_SEARCH_EMBEDDING_INFLIGHT: int = Field(default=2, ge=1)
    GPU_WAIT_SEC: float = Field(default=2, gt=0, allow_inf_nan=False)
    # 정리 결과를 알려진 평가 데이터셋과 사진 지문으로 대조해 정확도를 자동 기록한다.
    EVAL_AUTO_DETECT: bool = True
    # /runpod/status의 Pod 상태 조회. Read Only 키 권장, 미설정이면 running은 null이다.
    RUNPOD_API_KEY: SecretStr | None = None
    RUNPOD_STATUS_CACHE_SEC: float = Field(default=10, ge=0, allow_inf_nan=False)
    RUNPOD_STATUS_TIMEOUT_SEC: float = Field(default=3, gt=0, allow_inf_nan=False)

    # 정리 에이전트의 공공 장소 색인(#151). S3 URI가 있으면 실행기 시작 때 해시 확인 후 한 번 받음
    PLACE_INDEX_PATH: str | None = None
    PLACE_INDEX_S3_URI: str | None = None
    PLACE_INDEX_SHA256: str | None = None
    # 카카오 로컬 API. 하루 무료 한도를 모든 작업이 함께 쓰므로 작업마다 호출 수를 제한한다.
    KAKAO_REST_API_KEY: SecretStr | None = None
    KAKAO_CALLS_PER_JOB: int = Field(default=200, ge=0)
    KAKAO_TIMEOUT_SEC: float = Field(default=10, gt=0, allow_inf_nan=False)
    # 정리 에이전트(#153). 끄면 규칙 단계 결과로 완료(결정 150). 켜면 VLM, 색인, 카카오 설정이 필요
    PROCESS_AGENT_ENABLED: bool = True
    # 에이전트 단계의 시간 예산. 작업 자원을 만든 때(다운로드 전)부터 재고, 넘기면 작업 실패.
    # FILTERING과 결과 저장을 위해 작업 제한보다 짧게
    PROCESS_AGENT_BUDGET_SEC: float = Field(default=2700, gt=0, allow_inf_nan=False)
    # 정리 작업이 동시에 보내는 생성 호출 수. GPU_GENERATION_INFLIGHT 슬롯을 검색, 스토리와 나눠 씀
    PROCESS_AGENT_CONCURRENCY: int = Field(default=4, ge=1)
    # 정리 작업의 생성 호출 제한 시간과 슬롯 대기
    # 검색과 스토리의 VLM_TIMEOUT_SEC, GPU_WAIT_SEC와 별개
    PROCESS_VLM_TIMEOUT_SEC: float = Field(default=180, gt=0, allow_inf_nan=False)
    PROCESS_VLM_WAIT_SEC: float = Field(default=60, gt=0, allow_inf_nan=False)
    # 전송 오류, HTTP 408, 429, 5xx, 호출 시간 초과를 다시 보내는 횟수(n번째는 5×n초 뒤).
    # 넘으면 작업 실패. 불완전 응답과 슬롯 대기 초과는 다시 보내지 않음(결정 159)
    PROCESS_VLM_RETRIES: int = Field(default=3, ge=0, le=5)
    # 워크플로우 정리 파이프라인(실험 코드 2026-11-05 step3-agent를 옮김). 켜면 CLUSTERING에서
    # 규칙 단계와 정리 에이전트 대신 돈다(에이전트 설정은 무시). 기본은 켜짐(결정 221),
    # 끄면 PROCESS_AGENT_ENABLED에 따라 정리 에이전트나 규칙 단계로 완료한다.
    # 켜면 VLM, 공공 장소 색인, 카카오 키, TourAPI 원자료 파일이 필요하다.
    PROCESS_WORKFLOW_ENABLED: bool = True
    # TourAPI 국문 관광정보 원자료(JSON). 실험은 2026-10-30에 받은 tourapi_all.json을 썼다.
    # 색인과 같이 S3 URI가 있으면 실행기 시작 때 SHA-256을 확인하고 한 번 받는다(결정 221)
    TOURAPI_RAW_PATH: str | None = None
    TOURAPI_RAW_S3_URI: str | None = None
    TOURAPI_RAW_SHA256: str | None = None
    # 워크플로우 단계의 시간 예산(결정 222). 작업 자원을 만든 때부터 재고, 넘기면 작업 실패
    PROCESS_WORKFLOW_BUDGET_SEC: float = Field(default=3600, gt=0, allow_inf_nan=False)
    # 동시 생성 호출 수(결정 222: 슬롯 8개 가운데 정리 4개, 실험은 8). 검색, 스토리와 슬롯을 나눠 씀
    PROCESS_WORKFLOW_CONCURRENCY: int = Field(default=4, ge=1)
    # 생성 호출 한 번의 제한 시간(실험 300초)
    PROCESS_WORKFLOW_VLM_TIMEOUT_SEC: float = Field(default=300, gt=0, allow_inf_nan=False)
    # 작업당 카카오 호출 상한. 실험은 상한이 없었고 200장 평가에서 많게는 604회를 불렀다
    PROCESS_WORKFLOW_KAKAO_CALLS: int = Field(default=1000, ge=0)
    # 사진 순서를 섞는 시드(실험 기본값). 사진 ID와 저장 키 순서를 쓰지 않으려는 규칙
    PROCESS_WORKFLOW_PHOTO_SEED: int = 61106
    # 격자 칸 번호 글꼴. 비우면 실험 글꼴(Arial Bold), DejaVu Sans Bold, Pillow 기본 글꼴 순
    PROCESS_WORKFLOW_FONT_PATH: str | None = None

    @model_validator(mode="after")
    def check_jobs_settings(self) -> Self:
        if self.JOBS_HEARTBEAT_SEC * 3 >= self.JOBS_LEASE_SEC:
            raise ValueError("작업 lease는 heartbeat 간격의 3배보다 길어야 함")
        topics = [
            getattr(self, name) for name in type(self).model_fields if name.endswith("_TOPIC")
        ]
        if len(set(topics)) != 6 or any(not topic.strip() for topic in topics):
            raise ValueError("요청과 결과 토픽 6개는 서로 다른 이름이어야 함")
        if self.KAFKA_SECURITY_PROTOCOL == "SASL_SSL":
            password = self.KAFKA_SASL_PASSWORD
            if not (self.KAFKA_SASL_USERNAME or "").strip():
                raise ValueError("SASL_SSL에는 KAFKA_SASL_USERNAME이 필요함")
            if password is None or not password.get_secret_value().strip():
                raise ValueError("SASL_SSL에는 KAFKA_SASL_PASSWORD가 필요함")
            if not (self.KAFKA_SSL_CAFILE or "").strip():
                raise ValueError("SASL_SSL에는 KAFKA_SSL_CAFILE이 필요함")
        # V3 Qdrant는 API Key 인증을 켠 상태로만 운영. V2와 로컬은 무인증 유지
        if self.JOBS_ENV == "v3-foundation":
            key = self.QDRANT_API_KEY
            if key is None or not key.get_secret_value().strip():
                raise ValueError("JOBS_ENV=v3-foundation에는 QDRANT_API_KEY가 필요함")
        # 환경 변수 예시처럼 빈 값으로 둔 항목은 쓰지 않는 것으로 본다.
        if self.PLACE_INDEX_S3_URI or self.PLACE_INDEX_SHA256:
            if not re.fullmatch(r"s3://[^/]+/.+", self.PLACE_INDEX_S3_URI or ""):
                raise ValueError("PLACE_INDEX_S3_URI는 s3://버킷/키 형식이어야 함")
            if not re.fullmatch(r"[0-9a-f]{64}", self.PLACE_INDEX_SHA256 or ""):
                raise ValueError("PLACE_INDEX_SHA256은 소문자 16진수 64자여야 함")
            if not self.PLACE_INDEX_PATH:
                raise ValueError("PLACE_INDEX_S3_URI를 쓰려면 PLACE_INDEX_PATH가 필요함")
        if self.TOURAPI_RAW_S3_URI or self.TOURAPI_RAW_SHA256:
            if not re.fullmatch(r"s3://[^/]+/.+", self.TOURAPI_RAW_S3_URI or ""):
                raise ValueError("TOURAPI_RAW_S3_URI는 s3://버킷/키 형식이어야 함")
            if not re.fullmatch(r"[0-9a-f]{64}", self.TOURAPI_RAW_SHA256 or ""):
                raise ValueError("TOURAPI_RAW_SHA256은 소문자 16진수 64자여야 함")
            if not self.TOURAPI_RAW_PATH:
                raise ValueError("TOURAPI_RAW_S3_URI를 쓰려면 TOURAPI_RAW_PATH가 필요함")
        # 에이전트를 끈 배포는 예전 제한 시간과 슬롯 수로도 시작할 수 있게 한다.
        if self.PROCESS_AGENT_ENABLED:
            if self.PROCESS_AGENT_BUDGET_SEC >= self.JOBS_PROCESS_TIMEOUT_SEC:
                raise ValueError(
                    "PROCESS_AGENT_BUDGET_SEC는 JOBS_PROCESS_TIMEOUT_SEC보다 짧아야 함"
                )
            if self.PROCESS_AGENT_CONCURRENCY > self.GPU_GENERATION_INFLIGHT:
                raise ValueError("PROCESS_AGENT_CONCURRENCY는 GPU_GENERATION_INFLIGHT 이하여야 함")
        if self.PROCESS_WORKFLOW_ENABLED:
            if self.PROCESS_WORKFLOW_BUDGET_SEC >= self.JOBS_PROCESS_TIMEOUT_SEC:
                raise ValueError(
                    "PROCESS_WORKFLOW_BUDGET_SEC는 JOBS_PROCESS_TIMEOUT_SEC보다 짧아야 함"
                )
            if self.PROCESS_WORKFLOW_CONCURRENCY > self.GPU_GENERATION_INFLIGHT:
                raise ValueError(
                    "PROCESS_WORKFLOW_CONCURRENCY는 GPU_GENERATION_INFLIGHT 이하여야 함"
                )
        return self

    def job_timeout(self, kind: str) -> float:
        """작업 시작부터 강제 실패까지의 시간. 정리만 따로 둔다."""
        return self.JOBS_PROCESS_TIMEOUT_SEC if kind == "process" else self.JOBS_TIMEOUT_SEC

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
