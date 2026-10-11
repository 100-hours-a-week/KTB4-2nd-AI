"""개발 전용: API_KEY=local-demo uv run uvicorn app.search_dev:create_app --factory."""

from fastapi import FastAPI

from app.api.services.query_service import QueryService
from app.core.config import Settings
from app.main import create_app as create_api
from app.search.mock import create_mock_context


def create_app() -> FastAPI:
    app = create_api(
        settings=Settings(IMAGE_SOURCE="local"),
        query_service=QueryService(create_mock_context()),
    )
    app.title = "여담 검색 MOCK — 외부 모델 미연결"
    return app
