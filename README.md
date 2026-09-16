# 여담 AI 서버

여행 사진을 장소별 폴더로 정리하는 파이프라인과 그 API. 규칙은 `CLAUDE.md`, 계약은 팀 위키 "모델 API 설계"

## 로컬 실행

```
uv sync
cp .env.example .env
docker run -p 6333:6333 -v ./data/qdrant:/qdrant/storage qdrant/qdrant:v1.19.1
uv run uvicorn app.worker_main:app --port 8002
uv run uvicorn app.main:app --port 8000
```

## 검사

```
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
