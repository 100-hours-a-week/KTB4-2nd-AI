"""JSON 로깅, 클라우드 관측 계약

세 프로세스의 로그가 컨테이너 stdout 하나로 합쳐지므로 한 줄 JSON으로 통일
공통 필드는 해당하지 않아도 null로 남겨 CloudWatch 쿼리가 필드 존재를 전제할 수 있게 함
요청 단위 식별자는 ContextVar에 두어 깊은 호출에서도 인자 없이 붙음
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

SERVICE = "ai"

# 클라우드 관측 계약의 공통 필드. 값이 없어도 null로 남김
COMMON_FIELDS = (
    "event",
    "result",
    "request_id",
    "trip_id",
    "job_id",
    "duration_ms",
    "failure_stage",
    "error_code",
)

# ProcessStep 이름과 클라우드가 정한 단계 이름의 매핑
# core는 schemas를 import하지 않으므로 키를 문자열로 둠
STEP_NAMES = {
    "DOWNLOADING": "download",
    "EMBEDDING": "embedding",
    "CLOCK": "clock_correction",
    "LOCATING": "locating",
    "CLUSTERING": "clustering",
    "FILTERING": "filtering",
    "FINALIZING": "finalizing",
}

# LogRecord 예약 속성과 이름이 겹치는 공통 필드. extra에는 왼쪽 이름으로 넘기고 출력은 오른쪽
# process는 LogRecord가 이미 PID로 쓰고 있어 extra로 넘기면 KeyError
FIELD_ALIASES = {"process_role": "process"}

# 단계 밖에서 끊긴 경우의 실패 구간 이름. 단계 안에서 끊기면 STEP_NAMES의 값을 씀
FAILURE_QUEUE = "queue"
FAILURE_WORKER_DISPATCH = "worker_dispatch"
FAILURE_MODEL_LOAD = "model_load"
FAILURE_CALLBACK = "callback"
FAILURE_INTERNAL = "internal"

# LOG_LEVEL을 DEBUG로 올리면 루트를 물려받아 이것들도 DEBUG가 됨
# botocore는 S3 요청마다 수십 줄이라 사진 200장이면 우리 로그가 묻힘
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "botocore",
    "boto3",
    "urllib3",
    "s3transfer",
    "transformers",
    "qdrant_client",
    "PIL",
)

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_trip_id: ContextVar[int | None] = ContextVar("trip_id", default=None)
_job_id: ContextVar[str | None] = ContextVar("job_id", default=None)

# LogRecord가 기본으로 갖는 속성. extra=로 넘긴 키를 골라내는 데 사용
_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


def bind_request(request_id: str | None) -> None:
    """현재 맥락에 request_id를 심음. api 미들웨어와 워커 라우터가 부름"""
    _request_id.set(request_id)


def bind_trip(trip_id: int | None) -> None:
    """현재 맥락에 trip_id를 심음

    contextvars는 asyncio 태스크마다 복사되므로 요청 하나에만 붙고,
    새 스레드에는 상속되지 않으므로 러너는 copy_context로 명시 전달
    """
    _trip_id.set(trip_id)


def bind_job(job_id: str | None) -> None:
    """현재 맥락에 job_id를 심음. 값은 ProcessRequest.execution_id, 없으면 None"""
    _job_id.set(job_id)


def clear_context() -> None:
    """작업 종료 때 세 식별자를 해제해 다음 작업에 값이 섞이지 않게 함"""
    _request_id.set(None)
    _trip_id.set(None)
    _job_id.set(None)


def current_request() -> str | None:
    return _request_id.get()


def current_trip() -> int | None:
    return _trip_id.get()


def current_job() -> str | None:
    return _job_id.get()


class JsonFormatter(logging.Formatter):
    """한 줄 JSON. 공통 필드는 해당하지 않아도 null로 남김

    extra=로 같은 이름을 넘기면 ContextVar보다 우선함.
    api는 콜백이 다른 요청으로 오므로 trip_id와 job_id를 extra로 명시
    """

    def __init__(self, release: str = "unknown") -> None:
        super().__init__()
        self.release = release

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_ATTRS and not key.startswith("_")
        }
        for source, target in FIELD_ALIASES.items():
            if source in extras:
                extras[target] = extras.pop(source)
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "service": SERVICE,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context: dict[str, Any] = {
            "request_id": current_request(),
            "trip_id": current_trip(),
            "job_id": current_job(),
        }
        for key in COMMON_FIELDS:
            payload[key] = extras.pop(key, context.get(key))
        payload["release"] = self.release
        payload.update(extras)
        if record.exc_info:
            payload["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", release: str = "unknown") -> None:
    """루트 로거를 JSON 핸들러 하나로 구성. 프로세스 엔트리에서 한 번 호출

    uvicorn 로거의 자체 핸들러를 떼어 루트로 올려 보내 형식을 맞춤
    서드파티 로거는 WARNING으로 고정해 LOG_LEVEL=DEBUG일 때 우리 로그만 남게 함
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(release))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers[:] = []
        uv_logger.propagate = True
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """모듈에서 logger = get_logger(__name__)로 사용"""
    return logging.getLogger(name)
