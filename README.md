# 여담 AI 서버

여행 사진을 장소별 폴더로 정리하는 파이프라인과 그 API. 규칙은 `CLAUDE.md`, 계약은 팀 위키 "모델 API 설계"

## 로컬 실행

```
uv sync
cp .env.example .env
uv run scripts/fetch_model.py
docker run -p 6333:6333 -v ./data/qdrant:/qdrant/storage qdrant/qdrant:v1.19.1
uv run uvicorn app.worker_main:app --port 8002
uv run uvicorn app.main:app --port 8000
```

`fetch_model.py`는 임베딩 가중치를 `models/.raw/`에 받아 vision 타워만 bf16으로 변환한 0.9GB를 `models/siglip2`에 둠. git에는 없음

## 검사

```
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## 표준 명령

CI 워크플로와 사람이 같은 명령을 씀. 워크플로는 이 표의 명령을 그대로 실행

| 단계 | 명령 | PR | main |
| --- | --- | --- | --- |
| 의존성 | `uv sync --frozen` | ✓ | ✓ |
| 린트 | `uv run ruff check .` | ✓ | |
| 포맷 | `uv run ruff format --check .` | ✓ | |
| 테스트 | `uv run pytest` | ✓ | |
| 가중치 캐시 키 | `uv run scripts/fetch_model.py --cache-key` | ✓ | ✓ |
| 가중치 다운로드 | `uv run scripts/fetch_model.py` | ✓ 캐시 미스 시 | ✓ 캐시 미스 시 |
| 모델 실행 검증 | `uv run scripts/runtime_check.py` | ✓ | |
| 이미지 빌드 | `docker buildx build --platform linux/amd64 -f docker/Dockerfile -t ghcr.io/100-hours-a-week/yeodam-ai-worker:sha-$GITHUB_SHA .` | ✓ push 없음 | ✓ |
| 이미지 push | `docker push ghcr.io/100-hours-a-week/yeodam-ai-worker:sha-$GITHUB_SHA` | | ✓ |

- 가중치 캐시는 `models/siglip2` 디렉터리, 키는 `--cache-key` 출력(`siglip2-so400m-naflex-<revision 12자>-bf16-vision`). revision은 `scripts/fetch_model.py`의 `MODELS`에 고정, dtype이나 `--full`이 바뀌면 키도 바뀜
- `runtime_check.py`는 엔진 적재, 1장 임베딩, Qdrant 연결을 확인. 워커 엔트리와 엔진 클래스가 머지된 뒤 추가
- 이미지 태그는 `sha-{전체 커밋 해시}`. 클라우드 CD는 이 태그를 pull만 하고 다시 빌드하지 않음

## 컨테이너 기동 확인

```
docker run --rm -p 8000:8000 -v ./data:/data \
  -e API_KEY=local -e S3_BUCKET=<버킷> -e FAKE_PIPELINE=1 \
  ghcr.io/100-hours-a-week/yeodam-ai-worker:sha-<해시>
curl -s localhost:8000/health
```

기동 약 60초 뒤 `/health`가 `{"status": "ok", ...}` 200. 그전에는 503 `starting`(모델 적재 중)이나 `degraded`(워커나 Qdrant 미응답). `FAKE_PIPELINE=1`이면 진짜 모델 대신 가짜 엔진으로 접수부터 완료까지 확인 가능

## 로그

컨테이너 stdout에 한 줄 JSON. 수집과 CloudWatch 전송은 클라우드가 구성

공통 필드는 `timestamp`, `level`, `service`, `logger`, `message`, `event`, `result`, `request_id`, `trip_id`, `job_id`, `duration_ms`, `failure_stage`, `error_code`, `release`이며 해당하지 않으면 `null`

| 환경 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | 운영과 부하 테스트는 `INFO`. 오류를 재현하는 짧은 구간에만 `DEBUG`로 올리고 확인 뒤 되돌림 |
| `RELEASE` | `unknown` | 배포 식별자. 클라우드가 이미지 SHA를 런타임 환경 변수로 주입 |

관측 이벤트

| `event` | 기록 위치 |
| --- | --- |
| `ai_service_startup` | api와 worker 기동 |
| `ai_job` | api, 작업 접수와 종료. 작업당 `started` 한 번과 최종 한 번 |
| `ai_job_dispatch` | api, 워커 위임 결과 |
| `ai_worker_execution` | worker, 파이프라인 실행 전체 |
| `ai_pipeline_step` | worker, 단계 전환 |
| `ai_callback` | worker, 콜백 최종 결과 |

단계 이름은 `download`, `embedding`, `clock_correction`, `locating`, `clustering`, `filtering`, `finalizing`

요청 식별자는 `X-Request-ID` 헤더로 받고 없으면 생성. api 응답 헤더로 되돌려주며 워커와 콜백까지 같은 값으로 전파
