"""FastAPI 예외 핸들러. api와 worker가 같은 오류 본문 형식을 쓰도록 한 곳에

AppError는 deps, 서비스, 라우터 어디서 던져도 명세의 {error: {code, message, detail}}로.
RequestValidationError는 FastAPI 기본 422 본문 {detail: [...]}을 같은 형식으로 바꿈
"""

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger, validation_fields

log = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_body())

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # 입력값은 질문 원문이나 사진 키를 담을 수 있어 위치와 오류 종류만 남긴다.
        log.warning(
            "요청 형식 오류",
            extra={
                "event": "ai_request_invalid",
                "result": "rejected",
                "error_code": str(ErrorCode.INVALID_REQUEST),
                "method": request.method,
                "path": request.url.path,
                "invalid_fields": validation_fields(exc.errors()),
            },
        )
        # errors()에 JSON으로 못 바꾸는 값(예외 객체 등)이 섞일 수 있어 jsonable_encoder
        err = AppError(ErrorCode.INVALID_REQUEST, detail={"errors": jsonable_encoder(exc.errors())})
        return JSONResponse(status_code=err.http_status, content=err.to_body())
