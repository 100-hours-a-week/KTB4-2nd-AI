# v2 검색 파이프라인 개발 안내

검색 API와 로컬에서 실행할 수 있는 mock이 준비되어 있습니다. 검색 담당자는 `app/search/pipeline.py`를 시작점으로 질의 해석, 검색 결과 선별, 사진을 근거로 한 답변 생성을 구현하면 됩니다. VLM과 임베딩 모델, Qdrant, S3의 실제 연결은 서버 담당자와 맞춰서 진행합니다.

## 검색 흐름과 수정할 코드

```text
BE → POST /query/parse → 질문에서 의도와 검색 조건 추출
BE → 권한, 날짜, 지역에 따라 전체 여행에서 후보 사진 선별
BE → POST /query/search → 후보 안에서 사진 검색 → QUESTION이면 답변 생성
```

검색 대상은 사용자의 전체 여행입니다. 사진 접근 권한과 DB 필터는 BE에서 처리하고, AI는 전달받은 후보 안에서만 검색합니다. 검색 결과를 폴더로 묶는 작업도 BE에서 담당합니다. 검색은 동기 HTTP로 처리하므로 Kafka나 작업 상태 저장소 없이 개발을 시작할 수 있습니다.

| 파일 | 역할과 작업 내용 |
| --- | --- |
| [app/search/pipeline.py](../app/search/pipeline.py) | `parse()`와 `search()`에 검색 처리 순서를 구현합니다. 검색 임계값과 답변용 사진 선택 로직을 추가할 위치입니다. |
| [app/search/context.py](../app/search/context.py) | 파이프라인에서 호출할 외부 기능의 인터페이스입니다. `SearchContext`로 필요한 구현을 전달받습니다. |
| [app/search/mock.py](../app/search/mock.py) | 실제 모델 없이 API 흐름을 확인하는 구현입니다. 실제 구현을 연결하는 방식의 예제로 참고할 수 있습니다. |
| [app/schemas/query.py](../app/schemas/query.py) | 요청과 응답 형식입니다. 외부 계약이 바뀌는 수정은 서버 담당자와 먼저 맞춥니다. |
| [tests/test_query.py](../tests/test_query.py) | 검색 흐름과 예외 처리를 검증하는 테스트입니다. 로직을 추가할 때 관련 사례를 함께 보강합니다. |

HTTP 라우터와 `QueryService`는 서버 담당자가 관리합니다. 인증, 요청 전체의 시간 제한, HTTP 오류 변환, 로그, 최종 응답 검증이 이 범위에 포함됩니다.

## 구현할 내용

### 1. 질의 해석

`parse()`는 `context.vlm.interpret(request)`를 호출합니다. 검색 담당자는 이 인터페이스에 맞춰 프롬프트와 모델 출력 검증을 구현합니다. VLM에 요청을 전송하는 공통 클라이언트는 서버 담당자와 연결합니다.

- 질문을 사진 검색인 `SEARCH`와 사진에 근거한 질문인 `QUESTION`으로 구분합니다.
- 날짜 범위, 지역명, 사진 내용 조건인 `visual_query`를 추출합니다.
- 상대 날짜는 BE가 보낸 `reference_date`를 기준으로 해석합니다.
- 출력 형식과 날짜 범위를 검증하고 `ParseResponse`를 반환합니다.

현재는 VLM 호출 실패, 잘못된 모델 출력, 해석 시간 초과가 발생하면 `parsed=false`로 반환합니다. 이때 `intent`는 `SEARCH`, `visual_query`는 원질문이며 날짜와 지역 조건은 `null`입니다.

### 2. 사진 검색과 결과 선별

`search()`에는 텍스트 임베딩, 후보 범위를 제한한 벡터 검색, 점수 정렬, 중복 제거, 반환 개수 제한까지 연결되어 있습니다. 여기에 검색 임계값과 답변용 근거 사진을 고르는 로직을 추가합니다.

- `visual_query`를 임베딩하고, 값이 `null`이면 원질문을 사용합니다.
- Qdrant 질의 자체를 `candidate_ids`로 제한합니다. 전체 사진을 검색한 뒤 결과에서 후보만 걸러내는 방식으로 대체하지 않습니다.
- 후보 밖 ID가 반환되면 이미지에 접근하거나 VLM에 전달하기 전에 차단합니다.
- `SEARCH`는 `limit`까지 반환하고 답변을 생성하지 않습니다. `QUESTION`은 `SearchOptions.question_k`를 사용하며 기본값은 5장입니다.

현재는 유사도 임계값 없이 점수순으로 결과를 반환합니다. 실제 임베딩 모델과 한국어 질의셋으로 평가한 뒤 임계값과 근거 사진 수를 조정해야 합니다. 벡터가 없는 사진은 검색 중 다시 임베딩하지 않으며, 누락을 알리는 방식은 추가 협의가 필요합니다.

### 3. 사진을 근거로 한 답변

`QUESTION`에서는 선택한 사진의 URL과 메타데이터를 `Evidence`로 묶어 `context.vlm.answer(query, evidence)`에 전달합니다. 검색 담당자는 답변 프롬프트와 출력 검증을 구현합니다. 사진이나 제공된 메타데이터로 확인할 수 없는 사실을 답변에 추가하지 않도록 처리합니다.

답변 생성에 실패해도 검색한 사진 목록은 유지합니다. 근거가 없으면 `NO_EVIDENCE`, VLM이나 이미지 접근에 실패하면 `VLM_UNAVAILABLE`, 답변 시간이 초과되면 `TIMEOUT`을 `answer_error`로 반환합니다.

## 외부 기능 연결

모델이나 클라이언트를 파이프라인의 전역 변수로 생성하지 않고 `context`를 통해 호출합니다. 서버 담당자가 실제 구현으로 `SearchContext`를 구성하고 `QueryService`에 연결합니다.

| 인터페이스 | 입력과 반환값 | 연결할 때 맞출 내용 |
| --- | --- | --- |
| `VLMClient.interpret` | `ParseRequest` → `ParseResponse` | 질의 해석 프롬프트와 출력 검증 |
| `TextEmbedder.encode_text` | 텍스트 → `list[float]` | 이미지 벡터와 같은 모델, 차원, 정규화 기준 |
| `CandidateSearcher.search` | 벡터, `candidate_ids`, `limit` → `list[SearchHit]` | Qdrant 컬렉션과 후보 ID 필터 |
| `ImageResolver.resolve` | 분석용 S3 키 → 이미지 URL | 선택한 사진의 사본 URL과 유효시간 |
| `VLMClient.answer` | 원질문, `list[Evidence]` → 문자열 | 사진과 메타데이터에 근거한 답변 |

실제 연결은 `QueryService(SearchContext(...))`를 만들어 `app.main.create_app(query_service=...)`에 전달합니다. 클라이언트 생성과 종료는 서버 담당자가 처리합니다. GPU 추론이나 동기 I/O를 비동기 함수 안에서 직접 실행하면 다른 요청과 시간 제한 처리가 지연될 수 있으므로, 비동기 워커 호출 등으로 연결합니다.

외부 라이브러리의 오류는 연결 코드에서 다음 예외로 변환합니다.

| 상황 | 파이프라인에 전달할 예외 | 현재 API 동작 |
| --- | --- | --- |
| 임베딩 또는 벡터 검색 사용 불가 | `ModelUnavailable` | 503 `MODEL_NOT_READY` |
| VLM 또는 답변용 이미지 접근 실패 | `VLMUnavailable` | 질의 해석은 대체 결과, 답변은 사진 목록과 `answer_error` 반환 |
| 모델 출력 검증 실패 | `InvalidModelOutput` | 질의 해석은 대체 결과, 답변은 `VLM_UNAVAILABLE` 반환 |
| 시간 초과 | `TimeoutError` | 해석은 대체 결과, 답변은 `answer_error=TIMEOUT`, 검색 단계나 전체 요청은 504 `TIMEOUT` |

예상하지 못한 내부 오류와 후보 밖 ID 반환은 500으로 처리합니다. 취소 예외를 삼키거나 제한 없이 재시도하지 않습니다. `SearchOptions`의 기본 시간 제한은 해석 5초, 검색 5초, 답변 10초, 전체 요청 30초입니다. 개발용 값이므로 실제 모델 연결 후 BE와 함께 조정해야 합니다. 원격 GPU 작업은 HTTP 요청이 취소되어도 계속 실행될 수 있습니다.

## 로컬 실행

Python 3.12 환경에서 저장소 루트를 기준으로 실행합니다.

```bash
uv sync --frozen
API_KEY=local-demo uv run uvicorn app.search_dev:create_app --factory --host 127.0.0.1 --port 8000
```

`http://127.0.0.1:8000/docs`에서 요청 형식을 확인하거나 별도 터미널에서 예제를 실행합니다.

```bash
curl --fail-with-body http://127.0.0.1:8000/query/parse \
  -H 'Authorization: Bearer local-demo' -H 'Content-Type: application/json' \
  --data-binary @docs/examples/query-parse.json

curl --fail-with-body http://127.0.0.1:8000/query/search \
  -H 'Authorization: Bearer local-demo' -H 'Content-Type: application/json' \
  --data-binary @docs/examples/query-search.json
```

mock은 질문에 `?`가 있으면 `QUESTION`, 없으면 `SEARCH`로 분류합니다. 날짜와 지역은 해석하지 않고, 후보 입력 순서대로 점수를 매기며, 답변에는 `[MOCK]`를 붙입니다. 실제 모델이나 이미지에 접근하지 않으므로 API 흐름을 확인하는 용도로 사용합니다.

일반 `app.main:app`은 검색 구현이 연결되지 않으면 503을 반환합니다. `FAKE_PIPELINE` 설정은 검색 mock을 켜지 않습니다. 위 개발 앱의 `/health`도 기존 worker와 Qdrant를 점검하므로, mock 실행 여부는 검색 API 요청으로 확인합니다.

## API 연결 시 확인할 사항

현재 스키마는 기존 API 설계의 제안 필드를 반영한 상태입니다. **`reference_date`, `parsed`, 검색 요청의 `query`와 `candidates` 형식은 BE와 최종 확인이 필요합니다.** 기존 `candidate_attachment_ids` 요청을 자동으로 변환하지 않습니다.

- 질문은 2~100자입니다. 후보에는 사진 ID, 분석용 S3 키, 촬영 시각, 장소명, 여행명이 들어갑니다. 촬영 시각과 장소명, 여행명은 `null`을 허용하지만 필드는 보내야 합니다. 촬영 시각이 있으면 시간대 오프셋이 필요합니다.
- 중복 후보 ID와 정의하지 않은 필드는 422로 처리합니다. `limit`은 생략하면 30이며 1~50의 정수만 허용하고 `null`은 허용하지 않습니다.
- 후보 수에 여행당 200장 제한을 적용하지 않습니다. 사용자 전체 여행에서 검색하므로 후보 상한과 요청 크기는 BE와 별도로 정해야 합니다.
- 해석 실패 시 조건 없이 검색을 계속할지, 벡터 누락을 어떻게 알릴지, 검색 시간 초과를 504로 반환할지 함께 확인해야 합니다.

실제 연결 전에는 임베딩 워커의 텍스트 API, Qdrant 컬렉션, VLM 주소와 인증 방식, 모델 이름, S3 접근 권한과 URL 유효시간을 서버 담당자와 맞춥니다. 로컬 mock 실행에는 `API_KEY`만 필요하며 실제 연결용 환경 변수는 아직 정해지지 않았습니다.

## 테스트와 인계

```bash
uv run pytest tests/test_query.py tests/test_api.py -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

파이프라인을 수정할 때 빈 후보, 후보 밖 결과, 답변만 실패하는 경우, 시간 초과에 대한 처리를 유지합니다. 실제 모델을 연결한 뒤에는 한국어 질의셋으로 해석과 검색 품질을 평가하고, 후보가 많을 때의 처리 시간도 별도로 확인합니다. mock 테스트만으로 이 품질과 성능을 확인할 수는 없습니다.

PR에는 연결에 필요한 API와 환경 변수, 이미지 버전, 선행 작업을 적습니다. 변경은 단기 브랜치에서 리뷰 후 `main`에 통합하며, 운영 배포는 스테이징에서 FE와 BE를 함께 검증한 뒤 진행합니다.
