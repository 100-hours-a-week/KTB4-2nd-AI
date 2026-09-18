"""라우터 공통 의존성, FastAPI Depends로 주입

서비스와 설정은 전역 변수가 아니라 request.app.state에서 꺼냄.
테스트가 앱을 여러 개 만들어도(가짜 워커, 다른 설정) 서로 섞이지 않게
"""

import hmac
from typing import Annotated

from fastapi import Depends, Header, Request

from app.api.services.task_service import TaskService
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode


def get_settings_dep(request: Request) -> Settings:
    """lifespan이 app.state에 올려 둔 Settings. core.config.get_settings(.env 읽기)와 이름을 구분"""
    return request.app.state.settings


def get_task_service(request: Request) -> TaskService:
    return request.app.state.task_service


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ServiceDep = Annotated[TaskService, Depends(get_task_service)]


def require_api_key(
    settings: SettingsDep,
    authorization: Annotated[str, Header()] = "",
) -> None:
    """Authorization: Bearer {API_KEY}. 없거나 다르면 401

    compare_digest는 비교 시간이 일치 길이에 따라 달라지지 않게 함
    """
    expected = f"Bearer {settings.API_KEY}"
    if not hmac.compare_digest(authorization.encode(), expected.encode()):
        raise AppError(ErrorCode.UNAUTHORIZED)


def localhost_only(request: Request) -> None:
    """워커 콜백은 같은 컨테이너의 127.0.0.1에서만. 명세에 403 코드가 없어 401"""
    host = request.client.host if request.client else None
    if host not in ("127.0.0.1", "::1"):
        raise AppError(ErrorCode.UNAUTHORIZED)
