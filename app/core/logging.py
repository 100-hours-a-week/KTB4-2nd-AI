"""JSON 로깅

세 프로세스의 로그가 컨테이너 stdout 하나로 합쳐지므로 한 줄 JSON으로 통일
작업 시작 때 bind_trip(trip_id)를 부르면 같은 컨텍스트(요청 또는 워커 스레드의 작업)에서
찍히는 로그에 trip_id가 자동으로 붙음
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any

_trip_id: ContextVar[int | None] = ContextVar("trip_id", default=None)

# LogRecord가 기본으로 갖는 속성. extra=로 넘긴 키를 골라내는 데 사용
_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


def bind_trip(trip_id: int | None) -> None:
    """현재 컨텍스트에 trip_id를 심음, None이면 해제

    contextvars는 asyncio 태스크마다 복사되므로 요청 하나에만 붙고,
    새 스레드에는 상속되지 않으므로 워커 러너는 작업 함수 안에서 직접 부름
    """
    _trip_id.set(trip_id)


def current_trip() -> int | None:
    """현재 컨텍스트의 trip_id, 없으면 None"""
    return _trip_id.get()


class JsonFormatter(logging.Formatter):
    """한 줄 JSON. ts, level, logger, msg, trip_id, extra 키, 예외"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        trip_id = current_trip()
        if trip_id is not None:
            payload["trip_id"] = trip_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    """루트 로거를 JSON 핸들러 하나로 구성. 프로세스 엔트리(main.py, worker_main.py)에서 한 번 호출

    uvicorn 로거의 자체 핸들러를 떼어 루트로 올려 보내 형식을 맞춤
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers[:] = []
        uv_logger.propagate = True
    # httpx는 요청마다 INFO 한 줄. 워커 콜백이 작업당 수십 번이라 WARNING부터만
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """모듈에서 logger = get_logger(__name__)로 사용"""
    return logging.getLogger(name)
