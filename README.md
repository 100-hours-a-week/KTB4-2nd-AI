# 여담 AI 서버

여행 사진 정리와 검색 및 스토리 생성을 위한 AI 서버입니다. BE와의 계약은 팀 위키의 "모델 API 설계"를 따릅니다.

## 개발 담당과 코드 경계

Grant는 공통 서버와 스토리, 전체 서비스 연결을 담당합니다. Kael은 검색과 사진 정리 알고리즘, 임베딩 및 모델 서버를 담당합니다. 파일별 리뷰 담당자는 [.github/CODEOWNERS](.github/CODEOWNERS)에 정의합니다. 공통 설정과 스키마 및 의존성 변경은 함께 검토합니다.

```text
app/
├── api/          외부 HTTP 요청 처리
├── jobs/         작업 상태와 실행 관리
├── worker/       사진 정리 파이프라인 실행
├── pipeline/     사진 정리 알고리즘
├── search/       검색 파이프라인
├── story/        스토리 파이프라인
├── embedding/    공용 임베딩 계약과 HTTP 클라이언트 구현 위치
└── engines/      기존 로컬 SigLIP2와 테스트용 엔진
docker/
├── Dockerfile    기존 통합 이미지
└── embedding/    독립 임베딩 vLLM 서버의 실행 설정
```

`app/worker/`는 작업을 실행하는 Python 코드이고 `docker/embedding/`은 벡터를 생성하는 별도 서버 구성입니다. 검색과 사진 정리는 같은 임베딩 서버를 호출합니다. 임베딩 서버는 여행의 작업 상태나 Qdrant 저장 및 클러스터링을 담당하지 않습니다.

| 위치 | 담당자 | 수정할 내용 |
| --- | --- | --- |
| `app/api/` | Grant | HTTP 인증, 요청과 응답, 검색 서비스 연결 |
| `app/jobs/` | Grant | 작업 상태와 실행 제어, 워커 호출과 감시. Kafka와 공유 상태 저장소의 연결도 이 경계에서 구현 |
| `app/worker/`, `app/worker_main.py` | Grant | 파이프라인 실행, 진행률 보고와 취소 연결, 워커 기동 |
| `app/story/` | Grant | 스토리 사진 선별, 사실 정리, 생성과 검증 |
| `app/search/` | Kael | 질의 해석, 후보 검색, 근거 선택과 답변 생성 |
| `app/embedding/` | Kael | 검색과 사진 정리의 공용 임베딩 계약, HTTP 클라이언트 구현 |
| `docker/embedding/` | Kael | 공유 임베딩 vLLM 서버의 모델과 실행 설정 |
| `app/engines/`, `app/pipeline/` | Kael | 기존 로컬 엔진 유지, 사진 정리와 클러스터링, 모델별 임계값 |
| `app/infra/qdrant.py` | Kael | 컬렉션 관리, 벡터 저장과 조회, 후보 범위 검색 구현 |
| `app/infra/vllm_client.py`, `app/infra/images.py` | Grant | 생성용 VLM 통신과 이미지 접근 |
| `app/main.py`, 배포 연결 설정 | Grant | 각 기능의 실제 의존성 연결과 앱 수명 관리, 통합 검증 |

Kael은 AWS 임베딩 vLLM과 RunPod 생성용 vLLM의 실행 설정 및 단독 호출을 검증합니다. AWS 임베딩 호출은 Kael이 이미지와 텍스트 입력, 벡터 차원 및 정규화까지 함께 관리합니다. Grant는 제공된 임베딩과 벡터 검색 구현을 서비스에 연결합니다.

### 함께 유지할 인터페이스

- 검색은 `SearchContext`의 `VlmClient`, `TextEmbedder`, `CandidateSearcher`, `ImageResolver`를 사용합니다. 검색 프롬프트와 출력 검증은 `app/search/`에서 구현합니다.
- `EmbeddingEngine`과 `TextEmbedder`는 `app/embedding/contracts.py`에 정의합니다. 기존 `app.engines.base.EmbeddingEngine`과 `app.search.context.TextEmbedder` import도 유지합니다. 이미지 계약은 기존 동기 정리 코드와 호환되고 텍스트 계약은 비동기 검색 코드와 호환됩니다.
- 생성용 VLM은 공통 `chat()`을 사용하고 스토리 프롬프트와 검증은 `app/story/`에서 구현합니다. 임베딩 호출은 생성용 `chat()`과 구분합니다.
- 사진 정리는 `run(request, ctx, on_progress, is_cancelled)`를 통해 실행합니다. 정리 알고리즘은 Kafka나 작업 상태 저장소를 직접 다루지 않고 진행률 콜백과 취소 확인 함수를 사용합니다.
- 공통 인터페이스를 변경할 때는 호출하는 쪽과 구현하는 쪽을 함께 확인합니다. 각 담당자는 기능 테스트를 작성하며, BE부터 모델과 저장소까지의 통합 검증은 Grant가 주도합니다.

### 현재 구현과 배포 범위

`app/jobs/`는 기존 메모리 기반 작업 관리를 분리한 패키지입니다. Kafka와 공유 상태 저장소는 후속 구현 대상입니다. `app/story/`에는 패키지만 마련되어 있으며 생성 로직은 아직 없습니다.

`docker/embedding/`에는 독립 임베딩 서버용 Compose 설정이 있습니다. 공용 임베딩 계약은 분리했지만 HTTP 클라이언트와 실제 연결은 아직 구현하지 않았습니다. 현재 Python 워커는 기존 로컬 SigLIP2를 사용합니다. 모델 및 이미지 버전, GPU 성능과 실제 이미지 요청은 Kael이 검증한 뒤 연결합니다.

기존 `docker/Dockerfile`과 아래 로컬 실행 명령은 기존 SigLIP2 및 통합 구성을 유지합니다. API와 작업 실행기의 컨테이너 분리는 후속 작업입니다. main에서 이미지가 만들어져도 운영 배포를 뜻하지 않으며, V2 이미지는 스테이징 통합 검증 후 배포 여부를 결정합니다.

## 독립 임베딩 서버 설정

NVIDIA GPU와 Docker GPU 지원이 준비된 호스트에서 사용합니다. `docker/embedding/.env.example`을 같은 폴더의 `.env`로 복사하고 검증한 이미지, 모델 revision, 모델명과 인증키를 입력합니다. 이 설정은 임베딩 서버 하나만 실행하며 API나 정리 실행기를 시작하지 않습니다.

```sh
docker compose --env-file docker/embedding/.env -f docker/embedding/compose.yaml config --quiet
docker compose --env-file docker/embedding/.env -f docker/embedding/compose.yaml up -d
```

기본 주소는 같은 호스트에서 확인하는 `http://127.0.0.1:8003/v1`입니다. API와 정리 실행기에서 접근할 내부 주소와 네트워크 접근 범위는 클라우드와 조율합니다. 모델별 전처리, 벡터 차원과 정규화 설정을 맞추고, 검색과 사진 배치가 함께 들어올 때의 처리량도 확인해야 합니다.

설정 참고: [vLLM Docker 실행](https://docs.vllm.ai/en/stable/deployment/docker/), [Compose GPU 설정](https://docs.docker.com/compose/how-tos/gpu-support/).

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
