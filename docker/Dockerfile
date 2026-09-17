# syntax=docker/dockerfile:1.7
# Build from the AI repository root:
#   uv run scripts/fetch_model.py
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile -t yeodam-ai-worker:local .

ARG UV_VERSION=0.12.15
ARG QDRANT_VERSION=v1.19.1
ARG QDRANT_MANIFEST_DIGEST=sha256:12364fe851b9f17356fc88189fc06d1b521262e04659ec7345975b00c9246a10

FROM qdrant/qdrant:${QDRANT_VERSION}@${QDRANT_MANIFEST_DIGEST} AS qdrant
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    AI_PROFILE=cpu \
    EMBED_MODEL=siglip2-so400m-naflex \
    EMBED_DEVICE=cpu \
    MODEL_PATH=/opt/models/siglip2 \
    WORKER_URL=http://127.0.0.1:8002 \
    API_URL=http://127.0.0.1:8000 \
    QDRANT_URL=http://127.0.0.1:6333 \
    QDRANT__STORAGE__STORAGE_PATH=/data/qdrant/storage \
    QDRANT__STORAGE__SNAPSHOTS_PATH=/data/qdrant/snapshots \
    QDRANT__STORAGE__TEMP_PATH=/data/qdrant/tmp \
    QDRANT__SERVICE__HOST=127.0.0.1 \
    QDRANT__TELEMETRY_DISABLED=true \
    IMAGE_SOURCE=s3 \
    AWS_REGION=ap-northeast-2 \
    MAX_CONCURRENT=1 \
    MAX_QUEUE=20 \
    WORKER_DEAD_SEC=120 \
    TASK_TIMEOUT_SEC=1800 \
    THRESHOLDS_PATH=/app/app/pipeline/thresholds.yaml

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl supervisor libunwind8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant

COPY models/siglip2 /opt/models/siglip2

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY app ./app
COPY scripts ./scripts
COPY docker/supervisord.conf /etc/supervisor/conf.d/ai.conf

RUN groupadd --system --gid 1001 ai \
    && useradd --system --uid 1001 --gid ai ai \
    && mkdir -p /data/qdrant /tmp/supervisor \
    && chown -R ai:ai /app /opt/models/siglip2 /data /tmp/supervisor

USER ai

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/ai.conf"]
