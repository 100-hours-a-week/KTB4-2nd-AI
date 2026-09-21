"""오류 코드와 예외

오류 코드는 API 명세의 표와 1:1. HTTP 응답으로 나가는 오류는 AppError,
파이프라인 안에서 던지는 실패는 PipelineError 계열
러너가 예외의 code를 읽어 실패 콜백의 오류 코드로 씀
"""

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """API 명세의 오류 코드. UNAUTHORIZED는 명세 표에 추가 예정, 마지막 둘은 내부 코드"""

    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    DUPLICATE_TASK = "DUPLICATE_TASK"
    TASK_NOT_CANCELABLE = "TASK_NOT_CANCELABLE"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    WORKER_DEAD = "WORKER_DEAD"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # 워커 → api 실패 콜백에서 취소를 구분하는 내부 코드, 백엔드 응답에는 status CANCELED로만 나감
    CANCELED = "CANCELED"
    # 명세에는 failed[].reason 값으로만 있음. v1은 디코딩 실패도 작업 전체 FAILED라 오류 코드로도 씀
    DECODE_FAILED = "DECODE_FAILED"
    # 워커 POST /jobs가 실행 중일 때 409. api → worker 내부 계약, 백엔드 응답에는 안 나감
    WORKER_BUSY = "WORKER_BUSY"


# HTTP 응답으로 나갈 수 있는 코드와 상태. 없는 코드는 작업 FAILED 전용이며 AppError로 쓰면 500
HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: 422,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.DUPLICATE_TASK: 409,
    ErrorCode.TASK_NOT_CANCELABLE: 409,
    ErrorCode.TASK_NOT_FOUND: 404,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.MODEL_NOT_READY: 503,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.WORKER_BUSY: 409,
}

# 기본 메시지. 백엔드에 그대로 전달되는 문장이라 명세의 예시 표기를 따름
DEFAULT_MESSAGE: dict[ErrorCode, str] = {
    ErrorCode.INVALID_REQUEST: "요청 형식이 올바르지 않습니다.",
    ErrorCode.UNAUTHORIZED: "인증 키가 없거나 올바르지 않습니다.",
    ErrorCode.DUPLICATE_TASK: "같은 여행의 작업이 진행 중입니다.",
    ErrorCode.TASK_NOT_CANCELABLE: "이미 종료된 작업은 취소할 수 없습니다.",
    ErrorCode.TASK_NOT_FOUND: "작업 기록이 없습니다. 다시 요청해 주세요.",
    ErrorCode.RATE_LIMITED: "대기열이 가득 찼습니다. 잠시 후 다시 요청해 주세요.",
    ErrorCode.MODEL_NOT_READY: "모델 적재가 끝나지 않았습니다.",
    ErrorCode.DOWNLOAD_FAILED: "사본 다운로드에 실패했습니다.",
    ErrorCode.WORKER_DEAD: "워커 응답이 끊겼습니다.",
    ErrorCode.TIMEOUT: "처리 시간 상한을 초과했습니다.",
    ErrorCode.INTERNAL_ERROR: "내부 오류가 발생했습니다.",
    ErrorCode.CANCELED: "작업이 취소되었습니다.",
    ErrorCode.DECODE_FAILED: "이미지 디코딩에 실패했습니다.",
    ErrorCode.WORKER_BUSY: "워커가 다른 작업을 실행 중입니다.",
}


class AppError(Exception):
    """HTTP 응답으로 나가는 오류. main.py의 예외 핸들러가 to_body()를 JSON으로 직렬화"""

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        detail: dict[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.message = message or DEFAULT_MESSAGE[code]
        self.detail = detail or {}
        self.http_status = http_status or HTTP_STATUS.get(code, 500)
        super().__init__(self.message)

    def to_body(self) -> dict[str, Any]:
        """명세의 오류 본문 {error: {code, message, detail}}"""
        return {"error": {"code": self.code.value, "message": self.message, "detail": self.detail}}


class PipelineError(Exception):
    """파이프라인 실패의 기반 클래스. 파이프라인은 실패를 반환값이 아니라 이 계열의 예외로만 표현"""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(self, message: str | None = None, detail: dict[str, Any] | None = None) -> None:
        self.message = message or DEFAULT_MESSAGE[self.code]
        self.detail = detail or {}
        super().__init__(self.message)


class PipelineCancelled(PipelineError):
    """is_cancelled()가 True일 때 파이프라인이 raise. 러너가 CANCELED로 매핑"""

    code = ErrorCode.CANCELED


class DownloadFailed(PipelineError):
    """사본 다운로드 3회 재시도 실패. detail.keys에 실패한 키 목록"""

    code = ErrorCode.DOWNLOAD_FAILED

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        super().__init__(detail={"keys": keys})


class DecodeFailed(PipelineError):
    """이미지 디코딩 실패. detail.ids에 실패한 trip_attachment_id 목록"""

    code = ErrorCode.DECODE_FAILED

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        super().__init__(detail={"ids": ids})
