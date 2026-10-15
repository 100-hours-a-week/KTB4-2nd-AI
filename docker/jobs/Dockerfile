# syntax=docker/dockerfile:1.7
# API와 CPU 정리 실행기가 공유하는 이미지. 모델 서버와 Qdrant를 포함하지 않는다.
FROM ghcr.io/astral-sh/uv:0.12.15 AS uv
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:${PATH}"
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev && python -c "from aiokafka.codec import has_lz4; assert has_lz4()"
COPY app ./app
RUN groupadd --system ai && useradd --system --gid ai ai && chown -R ai:ai /app
USER ai
EXPOSE 8000
CMD ["uvicorn", "app.v2_main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "35"]
