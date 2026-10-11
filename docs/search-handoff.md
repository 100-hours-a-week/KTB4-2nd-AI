# v2 검색 파이프라인 개발 안내

2026-10-11, [작업 이슈 #54](https://github.com/100-hours-a-week/KTB4-2nd-AI/issues/54).
HTTP 진입점과 외부 의존성을 교체할 수 있는 개발 뼈대다. 실제 모델·Qdrant 검색·S3·Kafka 연결은 포함하지 않는다.

## 바로 실행

저장소 루트에서 실행한다. Python 3.12와 프로젝트 의존성이 필요하다.

```bash
uv sync --frozen
API_KEY=local-demo uv run uvicorn app.search_dev:create_app --factory --host 127.0.0.1 --port 8000
```

별도 터미널에서 아래 예시를 실행하거나 `http://127.0.0.1:8000/docs`를 연다.

```bash
curl --fail-with-body http://127.0.0.1:8000/query/parse \
  -H 'Authorization: Bearer local-demo' -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-parse' --data-binary @docs/examples/query-parse.json

curl --fail-with-body http://127.0.0.1:8000/query/search \
  -H 'Authorization: Bearer local-demo' -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-search' --data-binary @docs/examples/query-search.json
```

mock은 `?`가 있으면 QUESTION, 없으면 SEARCH로 분류하고 날짜·지역은 해석하지 않는다. 후보 입력 순서대로 `1, 0.5, 0.333…` 점수를 부여하며 답변에는 `[MOCK]`를 붙인다. 이 값은 의미 유사도와 품질 점수가 아니다. 분석용 키의 실제 파일도 읽지 않는다.

일반 `app.main:app`은 검색 의존성을 주입하지 않으면 인증된 검색 요청에 `503 MODEL_NOT_READY`를 반환한다. `FAKE_PIPELINE`은 v1 정리 설정이므로 검색 mock을 켜지 않는다. mock은 위 개발 전용 엔트리에서만 명시적으로 연결한다. 개발 앱도 기존 앱 팩토리를 재사용하므로 v1 라우트가 함께 보이며 `/health`는 기존 worker·Qdrant를 점검한다. 검색 mock만 실행한 환경의 준비 여부는 위 검색 요청으로 확인한다.

## 구현 경계

```text
BE → query 라우터 → QueryService → pipeline.parse / pipeline.search
                                      ↓ SearchContext
                       VLM / TextEmbedder / CandidateSearcher / ImageResolver
```

| 담당 | 파일·진입점 | 채울 내용 |
| --- | --- | --- |
| 검색 담당 | `app/search/pipeline.py`의 `parse(request, context, options)`, `search(...)` | 의도 분류, 날짜·지역 정규화, 검색 컷오프, 답변용 근거 선택, 실패 처리 |
| 검색 담당 | `SearchContext.vlm`의 실제 구현 | `interpret`·`answer` 프롬프트와 모델 출력 검증. 공통 HTTP 전송 클라이언트는 서버 담당과 연결 |
| 서버 담당 | `app/api/routers/query.py`, `app/api/services/query_service.py`, `app/schemas/query.py` | HTTP 계약, 인증, 시간 제한, 외부 오류 변환, 후보 범위와 출력 검증 |
| 서버 담당 | `app/search/context.py`의 어댑터 구현·앱 조립 | VLM 전송, GPU 텍스트 임베딩, Qdrant 후보 필터, S3 사본 URL, 연결 자원 수명 |
| 공동 | `tests/test_query.py` | 경계 회귀 검증. 실제 검색 품질은 별도 질의셋으로 평가 |

검색 담당자는 모듈 전역에서 모델이나 클라이언트를 생성하지 않고 `context`를 사용한다. 실제 연결은 `QueryService(SearchContext(...))`를 구성해 `app.main.create_app(query_service=...)`에 주입한다. HTTP 클라이언트의 재사용·종료는 앱 조립부가 책임진다. 동기 GPU 추론이나 블로킹 I/O를 async 함수에서 직접 실행하지 않는다. 별도 워커의 비동기 호출 등으로 연결해야 시간 제한과 다른 요청 처리가 동작한다.

## 의존성 계약

| 인터페이스 | 입력 → 출력 | 구현 시 지킬 규칙 |
| --- | --- | --- |
| `VLMClient.interpret` | `ParseRequest` → `ParseResponse` | BE가 준 reference_date 사용. 실제 출력의 스키마·날짜 순서 검증 |
| `TextEmbedder.encode_text` | 텍스트 → `list[float]` | 이미지 벡터와 같은 임베딩 모델·차원·정규화 기준 |
| `CandidateSearcher.search` | 벡터, candidate_ids, limit → `list[SearchHit]` | Qdrant 질의 자체를 후보 ID로 제한. 전체 검색 후 사후 필터만 하는 구현은 금지 |
| `ImageResolver.resolve` | 분석용 S3 키 → 이미지 URL | 선별된 사진만 접근. 신뢰된 버킷의 사본 URL을 발급하고 유효시간 관리 |
| `VLMClient.answer` | 원질문, `list[Evidence]` → 문자열 | Evidence는 사진 메타데이터와 URL. 근거 밖 사실을 보완하지 않도록 프롬프트·검증 구현 |

외부 SDK의 예외는 어댑터에서 변환한다. 임베딩·벡터 의존성 사용 불가는 `ModelUnavailable`, VLM·답변 이미지 접근 실패는 `VLMUnavailable`, 모델 출력 검증 실패는 `InvalidModelOutput`, 시간 초과는 `TimeoutError`다. 취소를 삼키거나 무제한 재시도하지 않는다. 예상하지 못한 프로그래밍 오류는 공통 500으로 노출된다.

## 개발 초안 API

기존 AI·BE 제공 설계를 따라 구현했다. 원문에서 제안으로 표시된 **reference_date, parsed, query, candidates는 최종 BE 합의가 확인되지 않았다.** 이 개발 뼈대의 동작을 확정 계약으로 간주하지 않는다. 이전 candidate_attachment_ids 계약과의 자동 호환은 제공하지 않는다.

| 경로 | 요청 | 응답 |
| --- | --- | --- |
| POST `/query/parse` | query 2~100자, reference_date 날짜 | intent, nullable date_from/date_to/region_names/visual_query, parsed |
| POST `/query/search` | query 2~100자, intent, nullable visual_query, candidates, limit | attachments(ID·score), nullable answer/answer_error |

- 검색 범위는 사용자의 전체 여행이다. BE가 권한·날짜·지역으로 후보를 확정하며 AI는 후보 안에서만 검색한다. 폴더 구성은 BE 담당이다.
- 후보는 사진 ID, analyze_storage_key와 nullable taken_at/place_name/trip_name을 담는다. null 시각을 허용하지만 값이 있으면 오프셋이 필요하다. 중복 ID·알 수 없는 필드는 422로 거부한다.
- 후보 개수에 여행당 200장 제한을 적용하지 않았다. 전체 사용자 후보 상한·본문 크기·페이지 전략은 미정이다. 수만 건 성능을 검증하지 않았다.
- limit은 생략 시 30, 정수 1~50이고 null을 허용하지 않는다. SEARCH의 반환 상한이다. QUESTION은 개발 옵션 question_k(기본 5)를 사용한다.
- visual_query가 null이면 원질문으로 임베딩한다. SEARCH는 현재 초안에서 답변을 생성하지 않는다.
- 벡터 검색 결과를 점수 내림차순으로 정렬하고 중복 사진은 최고 점수 하나만 남긴다. 실제 컷오프는 아직 구현하지 않았다.
- 누락 벡터는 어댑터가 반환한 사진만 사용하는 개발 동작이다. 검색 중 재임베딩하지 않으며 누락을 BE에 알릴 방식은 미정이다.

QUESTION 정상 예시(mock):

```json
{"attachments":[{"trip_attachment_id":412,"score":1.0},{"trip_attachment_id":415,"score":0.5}],"answer":"[MOCK] 근거 사진 2장으로 답변 경로를 확인했습니다.","answer_error":null}
```

빈 SEARCH 후보 예시:

```json
{"attachments":[],"answer":null,"answer_error":null}
```

답변 생성만 실패한 예시:

```json
{"attachments":[{"trip_attachment_id":412,"score":1.0}],"answer":null,"answer_error":"VLM_UNAVAILABLE"}
```

## 실패·시간 제한·관측

| 상황 | 개발 뼈대의 동작 |
| --- | --- |
| 인증 실패 / 잘못된 입력 | 401 UNAUTHORIZED / 422 INVALID_REQUEST |
| 해석 의존성 실패·출력 오류·해석 시간 초과 | 200, parsed=false, SEARCH와 원질문 visual_query, 나머지 조건 null |
| 빈 후보 또는 반환 벡터 없음 | SEARCH는 빈 결과, QUESTION은 200 + NO_EVIDENCE |
| 임베딩·벡터 사용 불가 | 503 MODEL_NOT_READY |
| 검색 단계 또는 전체 요청 시간 초과 | 504 TIMEOUT. HTTP 매핑은 개발 처리안이며 BE 확인 필요 |
| 답변 VLM·이미지 실패 / 답변 시간 초과 | 200 + VLM_UNAVAILABLE / TIMEOUT, 검색 사진 유지 |
| 후보 밖 ID 또는 예상하지 못한 내부 오류 | 500 INTERNAL_ERROR. 후보 밖 이미지는 접근 전에 차단 |

오류 본문은 공통 `{ "error": { "code", "message", "detail" } }`를 사용한다. `SearchOptions` 기본값은 해석 5초, 검색 5초, 답변 10초, 전체 요청 30초다. 실측이나 서비스 SLA가 아닌 개발용 중단 기준이다. 실제 어댑터의 연결·읽기 시간 제한과 BE·프록시 제한도 맞춰야 한다.

`X-Request-ID`가 응답과 JSON 로그에 연결된다. `ai_query`에는 operation, result(success/fallback/partial/error), duration_ms가 기록된다. 원질문·후보 키·사진 URL을 로그에 담지 않는다. 서버의 비동기 취소가 원격 GPU 연산 중단까지 보장하지는 않는다.

## 남은 조율과 배포 의존성

1. BE·검색: 제안 필드 채택, SEARCH 답변 여부, 해석 실패 시 조건 없는 검색의 UX, 후보 규모·본문 크기, 누락 벡터 처리, 504와 답변 오류 코드.
2. AI·Cloud: 임베딩 워커의 텍스트 API, 모델·벡터 차원·컬렉션, VLM 주소·인증·모델 이름, S3 버킷·권한·URL 유효시간. 실제 연결용 환경 변수 이름은 이 뼈대에서 임의 확정하지 않았다.
3. 개발 실행에는 API_KEY만 필수다. 실제 검색용 VLM·임베딩·Qdrant·S3 연결 및 자원 종료를 앱 조립부에 구현해야 한다. Kafka와 작업 상태 저장소는 동기 검색의 선행 조건이 아니다.
4. 단기 브랜치에서 CI·리뷰 후 main으로 통합하며 별도 dev 브랜치는 만들지 않는다. main의 이미지 생성은 운영 배포가 아니다. 운영은 마지막 정상 V1 이미지 조합을 유지한다.
5. V2 스테이징 배포 경로는 Cloud 공지 기준 준비 중이다. PR·공유에 필요한 API·환경 변수·이미지 버전·선행 작업을 기재하고, FE·BE·AI 조합 검증 뒤 운영 적용을 판단한다. 이 뼈대만으로 V2 운영 배포를 요청하지 않는다.

## 검증 명령과 한계

```bash
uv run pytest tests/test_query.py tests/test_api.py -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

mock 테스트는 인증, 스키마, 빈 후보, 후보 범위, 답변 부분 실패, 시간 제한과 기존 정리 API 회귀를 다룬다. 실제 모델 해석·검색 정확도, 한국어 질의 품질, 수만 후보 전송, 혼합 GPU 부하, 스테이징 통합·운영 검증은 별도 수행해야 한다.
