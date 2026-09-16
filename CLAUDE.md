# 여담(旅談) AI 서버

여행 사진을 장소별 폴더로 정리하는 파이프라인과 그 API. 카카오테크 부트캠프 KTB4 2조 2gether, AI 파트 kael과 grant
v1(10/2)은 정리 파이프라인만, v2(10/23)부터 스토리와 검색과 GPU

**설계 내용은 이 문서에 두지 않음.** API 계약은 위키 "모델 API 설계", v1 구조와 분담과 접점은 `v1_구현.md`가 단일 출처. 이 문서에는 지켜야 할 규칙만 둠

## 문서 위치

| 용도 | 위치 |
| --- | --- |
| API 계약 확정본 | 위키 "모델 API 설계" |
| 서비스 아키텍처 | 위키 "서비스 아키텍처 모듈화" |
| 파이프라인 단계 설계 | 위키 "멀티 스텝 AI 파이프라인", grant 작성 |
| v1 범위, 구조, 분담, 접점 7개 | `v1_구현.md`, kael 로컬 `/Users/wooseok/kakao/yeodam_project/` |
| 결정과 번복의 근거 | `결정기록.md`, 같은 폴더 |
| 칸반보드 | https://github.com/orgs/100-hours-a-week/projects/377 |
| 팀 위키 | https://github.com/100-hours-a-week/KTB4-2nd-wiki/wiki |

---

## 이슈, 브랜치, PR 규칙

팀 컨벤션 "Git 및 GitHub 작업 컨벤션"을 따름. 이슈 하나 = 브랜치 하나 = PR 하나

### 흐름

1. 칸반보드 Todo에 카드 생성, 제목은 `[유형] 작업 내용` → Convert to issue, 저장소 `KTB4-2nd-AI`
2. 이슈 화면 Development → Create a branch, 기준은 `main`. GitHub이 만든 이름을 그대로 사용, 직접 짓지 않음
3. `git fetch origin && git switch <브랜치>`, 카드는 In Progress로
4. 커밋 여러 개, push, PR 생성. 본문 첫 줄 `Close #이슈번호`
5. 상대 리뷰 후 머지, 머지되면 이슈와 카드가 자동으로 닫힘
6. `git switch main && git pull`

### 규칙

| 항목 | 규칙 |
| --- | --- |
| 기본 브랜치 | `main`만. `main`에 직접 커밋하지 않음. 빈 저장소의 첫 커밋만 예외 |
| 작업 유형 | `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `hotfix`. 소문자 |
| 이슈명 | `[feat] API 서버 접수, 상태 조회, 취소, 헬스` |
| 커밋 메시지 | `feat: task_store와 task_service 상태 전이`. 한 커밋에 목적 하나 |
| 브랜치명 | GitHub이 이슈에서 생성한 이름, `12-feat-api-서버-접수-상태-조회` 형식 |
| PR 제목 | 이슈명과 동일 |
| PR 본문 | `.github/pull_request_template.md`. `Close #N`, 변경점, 확인한 것 체크리스트 |
| PR 크기 | `v1_구현.md` §9 태스크 하나. 파일 하나는 너무 작고 v1 전체는 너무 큼 |
| 테스트 | 기능 PR에 테스트 동봉. `test` 유형은 테스트만 고칠 때 |
| 머지 조건 | CI 통과, 상대 승인 1개 |
| 접점 변경 | `run()` 시그니처, `EmbeddingEngine`, `PipelineContext`, 예외 클래스, `thresholds.yaml` 키를 바꾸는 PR은 상대를 반드시 리뷰어로 지정 |

칸반 카드를 이슈로 변환하면 이슈 템플릿 본문이 붙지 않음. Progress 체크박스는 직접 적음

---

## 폴더 구조와 의존 방향

```
app/
├── main.py            api 프로세스 엔트리, 8000
├── worker_main.py     worker 프로세스 엔트리, 8002
├── core/              config, logging, errors
├── schemas/           process, task, internal, health. 전부가 참조
├── api/               routers(문), services(규칙)
├── worker/            routers, runner(실행), callback
├── pipeline/          run, context, types, steps/, thresholds.yaml (연산)
├── engines/           base, siglip_naflex
└── infra/             images, qdrant
tests/  docker/  scripts/  evaluation/  .github/
```

의존은 한 방향. `api → (HTTP) → worker → pipeline → engines, infra`

- `pipeline`은 `api`와 `worker`를 import하지 않음. 서버 없이 `scripts/run_pipeline.py`로 돌아야 함
- `engines`와 `infra`는 `pipeline`을 import하지 않음
- 라우터는 검증과 응답만, 규칙은 서비스에, 연산은 파이프라인에

---

## 분담

| kael | grant |
| --- | --- |
| `core`, `schemas`, `api`, `worker`, `infra`, `pipeline/context.py`, `engines/base.py`, `tests/fakes.py`, `test_task_service.py`, `test_api.py`, `docker`, `scripts/runtime_check.py`, CI | `pipeline/run.py`, `types.py`, `steps/`, `thresholds.yaml`, `engines/siglip_naflex.py`, `evaluation/`, `scripts/run_pipeline.py`, `test_pipeline_steps.py` |

상대 영역의 파일을 고쳐야 하면 PR에서 리뷰어로 지정. 접점 7개의 정의는 `v1_구현.md` §8

---

## 코드 규칙

| 항목 | 규칙 |
| --- | --- |
| Python | 3.12, 의존성은 `uv`와 `pyproject.toml`, `uv.lock` 커밋 |
| import | `from app.schemas.process import ProcessRequest` 식 절대 경로. 상대 import 금지 |
| 설정 | 전부 환경 변수, `core/config.py`의 `Settings` 하나로 읽음. 코드에 값 하드코딩 금지, `.env.example`에 목록 유지 |
| 임계값 | 전부 `pipeline/thresholds.yaml`, 모델 이름 아래에 둠. 코드 상수로 두지 않음 |
| 경계 데이터 | 외부와 주고받는 것은 pydantic 모델(`schemas/`). 파이프라인 내부는 dataclass |
| 실패 표현 | 파이프라인은 예외로만. `PipelineCancelled`, `DownloadFailed(keys)`, `DecodeFailed(ids)`, `PipelineError(code, message)`. 반환값에 실패를 섞지 않음 |
| 오류 응답 | `AppError` → `{"error": {"code", "message", "detail"}}`. 예외를 삼키지 않음 |
| 시각 | timezone-aware `datetime`만. naive 금지. 응답은 ISO 8601 KST 오프셋 포함 |
| 로깅 | `core/logging.py`의 로거, JSON 한 줄, `trip_id` 자동 첨부. `print` 금지 |
| 동시성 | api의 상태는 `task_store` 하나, 락으로 보호. 그 외 전역 가변 상태 금지 |
| 포맷과 린트 | `ruff format`, `ruff check`. CI에서 검사 |
| 식별자 | 코드는 영어, 주석과 docstring은 한국어 가능. 주석은 이유가 있을 때만 |
| 타입 힌트 | 공개 함수와 접점은 필수 |

---

## 계약 값

어기면 백엔드 연동이나 저장 데이터가 깨지는 값

| 항목 | 값 | 위반 시 |
| --- | --- | --- |
| 작업 식별 | `trip_id` + 종류. `task_id` 없음 | 백엔드 경로와 불일치 |
| 응답 스키마 | 필드 추가만 허용, 삭제와 의미 변경 금지 | 백엔드 파싱 실패 |
| 시각 표기 | ISO 8601, KST 오프셋 포함 | 백엔드가 UTC로 재해석해 9시간 어긋남 |
| 오류 본문 | `{error: {code, message, detail}}`, 코드는 명세 표의 11종 | 백엔드 분기 실패 |
| Qdrant 컬렉션 | `vectors_{model_version}` | 모델 교체 시 구벡터와 신벡터 혼재 |
| Qdrant point ID | `trip_attachment_id` 단일 값 | 백엔드 DB와 조인 불가 |
| Qdrant payload | 비움 | 메타데이터 이중 소유 |
| 벡터 | 엔진의 `dim`, L2 정규화 | 컬렉션 안 벡터 혼재 |
| 입력 이미지 | 백엔드가 만든 긴 변 1024px JPEG 사본. AI는 EXIF를 읽지 않음 | 메타데이터 이중 소유 |
| 블러 판정 | 긴 변 300px 고정 | 임계값 전량 무효 |
| 진행률 | `total`은 항상 사진 수 | 프론트 진행 화면 오류 |
| 동시 처리 | v1 `MAX_CONCURRENT=1`, 대기열 20, 초과 429 | 워커 사망 오판 |
| 워커 | `uvicorn --workers 1` | 모델 복제로 메모리 초과 |
| 외부 노출 | api 8000만. worker 8002와 qdrant 6333은 127.0.0.1 | 권한 없는 접근 |

---

## 테스트

- `uv run pytest`가 외부 자원 없이 돌아야 함. S3는 `LocalDirImageSource`, Qdrant는 `:memory:`, 모델은 `FakeEngine`, 파이프라인은 `fake_run`
- 서버 테스트는 `TestClient`, 워커 클라이언트는 가짜로 교체해 콜백을 동기로 호출
- 단계 테스트는 고정 입력과 기대 출력. 임계값은 테스트 안에서 명시
- 실제 모델과 S3를 쓰는 검증은 `scripts/runtime_check.py`와 `evaluation/`으로 분리, CI의 pytest에 넣지 않음

---

## 실행

```
uv sync
docker run -p 6333:6333 -v ./data/qdrant:/qdrant/storage qdrant/qdrant:v1.19.1
uv run uvicorn app.worker_main:app --port 8002
uv run uvicorn app.main:app --port 8000
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

배포는 이미지 하나에 프로세스 셋(qdrant, worker, api), supervisord. Dockerfile과 supervisord.conf는 `docker/`에서 관리하고 클라우드는 compose와 볼륨과 환경 변수만

---

## 작업 규약

- 설계가 바뀌면 코드보다 `결정기록.md`에 먼저 남김. 번복이면 이전 근거를 지우지 않음
- 계약(요청과 응답 형식)이 바뀌면 위키 "모델 API 설계"를 갱신하고 백엔드에 알린 뒤 코드를 고침
- 점검이나 검토 요청은 보고까지. 수정은 승인된 항목만
- 산출물은 요청받은 것만 파일로. 기본은 세션 출력
- 위키는 kael이 직접 올림. 로컬 파일까지만 수정

## 문서 문체

README와 docs, PR 본문에 적용

- 명사형으로 종결, 명사형 종결에 마침표 없음
- 가운데점 금지, 쉼표나 슬래시로 대체
- 표는 왼쪽 정렬, 구분행에 `:` 없음
- 도해는 이미지보다 mermaid 코드블록
