"""외부 모델과 저장소 호출의 결과 로그. 호출마다 한 줄 ai_model_call

요청 본문, 응답 본문, URL과 인증값은 남기지 않음. 상태 코드와 소요 시간, 토큰 수 같은 값만
작업 식별자와 현재 단계는 ContextVar에서 자동으로 붙음
"""

import logging
import time
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import current_trace

log = get_logger("app.observability.calls")


def usage_fields(usage: Any) -> dict[str, int | None]:
    """OpenAI 호환 usage에서 정수 토큰 수만. 형식이 다르면 None"""
    fields: dict[str, int | None] = {"prompt_tokens": None, "completion_tokens": None}
    if isinstance(usage, dict):
        for name in fields:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                fields[name] = value
    return fields


def log_model_call(target: str, started: float, result: str, **fields: Any) -> None:
    """target은 vlm, embedding, s3, kakao. result는 success 또는 실패 분류값"""
    seconds = max(0.0, time.monotonic() - started)
    payload = {
        "event": "ai_model_call",
        "target": target,
        "result": result,
        "duration_ms": round(seconds * 1000, 2),
        **fields,
    }
    completion = fields.get("completion_tokens")
    if isinstance(completion, int) and seconds > 0:
        payload["tokens_per_sec"] = round(completion / seconds, 2)
    if trace := current_trace():
        trace.add_call(target, result)
    level = logging.INFO if result == "success" else logging.WARNING
    log.log(level, "외부 호출 결과", extra=payload)
