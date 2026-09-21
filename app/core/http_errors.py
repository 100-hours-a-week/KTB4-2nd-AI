"""FastAPI 예외 핸들러. api와 worker가 같은 오류 본문 형식을 쓰도록 한 곳에

AppError는 deps, 서비스, 라우터 어디서 던져도 명세의 {error: {code, message, detail}}로.
RequestValidationError는 FastAPI 기본 422 본문 {detail: [...]}을 같은 형식으로 바꿈
"""

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import AppError, ErrorCode


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_body())

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        # errors()에 JSON으로 못 바꾸는 값(예외 객체 등)이 섞일 수 있어 jsonable_encoder
        err = AppError(ErrorCode.INVALID_REQUEST, detail={"errors": jsonable_encoder(exc.errors())})
        return JSONResponse(status_code=err.http_status, content=err.to_body())
