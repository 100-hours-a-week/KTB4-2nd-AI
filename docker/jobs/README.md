# v2 공통 작업 서버 실행

API와 CPU 정리 실행기는 이 Dockerfile로 만든 같은 이미지를 사용합니다. Redis와 Kafka, 임베딩 vLLM, Qdrant 및 생성용 vLLM은 외부 서비스입니다. 기존 v1 이미지와 실행 명령은 유지합니다.

## 로컬 연결 확인

```sh
cp docker/jobs/.env.example docker/jobs/.env
docker compose -f docker/jobs/compose.yaml up --build -d
docker compose -f docker/jobs/compose.yaml ps
```

API 두 개는 서로 다른 임시 로컬 포트로 열립니다. `ps`에 표시된 주소로 GET과 DELETE를 호출합니다. 이 Compose는 Kafka와 Redis까지 포함하는 개발용 구성입니다. AWS의 서비스 배치를 대체하지 않습니다. 최초 기동 시 로컬 업무 토픽과 DLT를 생성합니다.

`JOBS_FAKE_PIPELINE=true`는 사진을 읽거나 모델을 호출하지 않습니다. 정리는 입력 사진을 모두 미분류로, 스토리는 연동 확인용 문장으로 반환합니다. 모델 검증에 사용하지 않습니다.

## 실제 환경의 실행 명령

- API: `uvicorn app.v2_main:create_app --factory --host 0.0.0.0 --port 8000`
- 정리 실행기: `python -m app.jobs.runner`

Fargate API는 2개 고정, EC2 정리 실행기는 1개와 CPU 상한 2 vCPU로 배포합니다. 정리 실행기는 한 작업을 끝낸 뒤 다음 작업을 꺼냅니다. 작업 자체는 자식 프로세스에서 실행해 상태 확인과 heartbeat가 CPU 연산에 막히지 않도록 합니다. 스토리 실행은 API의 실행 루프가 담당하며 Redis 선점으로 중복 실행을 차단합니다. 현재는 정리와 스토리를 각각 최대 1개씩 실행합니다.

실제 실행에서는 `JOBS_FAKE_PIPELINE=false`와 `JOBS_PROCESS_HANDLER`, `JOBS_STORY_HANDLER`를 설정합니다. API는 스토리 실행 함수가 없으면 Kafka 소비를 시작하지 않고 기동에 실패합니다. 정리 실행기도 정리 함수가 없으면 기동에 실패합니다. 각 값은 `package.module:function`이며, 함수 계약은 아래와 같습니다.

```python
async def run(request: dict, reporter: JobReporter) -> dict:
    await reporter.report("SELECTING", 0, 4)
    # 외부 호출 전후와 단계 경계에서 취소를 확인한다.
    await reporter.check_cancel()
    return result
```

기존 정리 함수 `run(request, ctx, on_progress, is_cancelled)`는 유지합니다. 위 async 함수는 서버 담당자가 연결할 어댑터이며, 정리 담당자에게 알고리즘의 시그니처 변경을 요구하지 않습니다. 기능별 입력과 결과는 `app/jobs/models.py`에 있습니다. 임베딩 HTTP 연결과 실제 스토리 파이프라인은 각각의 구현이 준비된 뒤 연결합니다. 검색은 기존 QueryService를 주입하며 Kafka와 Redis 대기열을 거치지 않습니다. 미연결 검색은 503을 반환합니다.

## 통신과 실패 처리

- 요청과 결과 토픽은 클라우드가 전달하는 환경변수 6개를 그대로 사용합니다. `JOBS_ENV`는 Redis 키 공간만 구분합니다. `.env.example`의 `local` 값은 로컬 검증용이며 스테이징과 운영에서는 환경별 값을 주입합니다.
- API마다 요청 종류별 소비자를 하나씩 실행합니다. 여행 생성은 `yeodam-ai-trip-create`, 사진 추가는 `yeodam-ai-trip-update`, 스토리는 `yeodam-ai-story-create` 그룹을 사용합니다. 두 API는 해당 그룹 이름과 Redis를 공유합니다. 환경마다 브로커가 분리되므로 그룹에 환경명을 붙이지 않습니다.
- 요청 key는 UTF-8로 인코딩한 문자열 trip_id이며 value는 UTF-8 JSON입니다. JSON 검증과 Redis 등록 및 대기열 추가가 성공한 뒤에만 해당 partition의 offset을 커밋합니다. 자동 커밋을 끄고 `auto_offset_reset="earliest"`와 기본 poll 제한 5분을 사용합니다.
- 각 소비 루프는 받은 순서대로 등록과 커밋을 마치며 모델을 호출하지 않습니다. Redis ZSET에는 시각 대신 원자적으로 증가하는 등록 순번을 넣습니다. 서로 다른 토픽의 생성과 수정 순서는 이 방식만으로 보장되지 않습니다.
- GET은 상태와 진행률만 반환합니다. 완료, 실패, 취소의 최종 결과는 수신 요청 토픽에 대응하는 Kafka 결과 토픽으로 전달합니다.
- DELETE는 대기 작업을 즉시 취소합니다. 실행 중에는 취소 요청을 기록합니다. DELETE는 기존 명세대로 CANCELED를 응답하지만, GET은 실제 중단 전까지 PROCESSING을 유지하고 다음 경계에서 멈춘 뒤 Kafka CANCELED 결과를 발행합니다.
- 실행 ID와 실행 소유권으로 중복 접수와 이전 실행의 늦은 결과를 차단합니다. 다른 실행이 진행 중이면 새 요청은 기존 계약대로 무시하며 재전달 때도 실행하지 않습니다.
- Kafka 재배치로 파티션 배정이 해제되면 기존 메시지의 커밋 반복을 끝내고 새 배정에서 소비를 이어갑니다.
- Linux에서는 부모 종료 신호로 작업 프로세스를 중단합니다. 별도 감시 스레드도 사용하므로 동기 작업이 asyncio 이벤트 루프를 막아도 부모 종료를 감지합니다. 이미 외부 모델 서버로 보낸 연산의 중단까지 보장하지는 않습니다.
- heartbeat가 120초 끊기면 WORKER_DEAD, 실행 시작 후 5분이 지나면 TIMEOUT으로 처리합니다. 결과 전달 실패는 분석 실패로 바꾸지 않고 재전달합니다.
- 결과 발행 성공 후 Redis 기록 전에 장애가 나면 같은 결과가 다시 발행될 수 있습니다. BE도 execution_id로 중복 결과를 제거해야 합니다.
- JSON 해석 불가, 식별자 누락, key 또는 작업 종류 불일치는 동일 메시지 검증이 3번 실패한 뒤 `요청 토픽-dlt`에 보냅니다. 원본 key와 본문을 보존하며 헤더에는 `original-topic`, `original-partition`, `original-offset`, `ai-error`(사유), `failure`(예외 종류), `attempts`를 넣습니다. 원본 key가 없는 경우 임의의 trip_id를 만들지 않고 null을 유지합니다.
- DLT 발행 성공 뒤에만 원래 offset을 커밋하고 `kafka_dlt_published` JSON 로그를 남깁니다. 검증 실패 횟수는 원본 topic/partition/offset별로 Redis에 공유하며 30일 동안 보관합니다. Redis 또는 Kafka의 연결 실패는 DLT 격리 사유로 세지 않습니다.
- DLT 크기 제한을 넘으면 `ai_kafka_dlt_too_large`에 원본 위치와 크기를 기록합니다. 해당 파티션의 읽기 위치를 원본 offset으로 되돌리고 일시 중지하며, 다른 파티션은 계속 처리합니다. 원본 offset은 커밋하지 않습니다. 요청 최대 크기에 DLT 헤더가 추가되는 공간까지 발행자와 브로커의 제한에 확보한 뒤 소비자를 재시작해야 합니다. 이 오류 로그는 DLT 발행 성공 로그와 다르므로 별도 알림 수집이 필요합니다.
- 식별 가능한 잘못된 본문은 기존 BE 계약대로 `FAILED / INVALID_REQUEST` 결과로 반환합니다. 클라우드 안내의 형식 오류 DLT 범위를 이 경우까지 넓힐지는 BE와 대조할 사항입니다. 정상 접수 후 분석 실패도 결과 토픽의 FAILED로 전달합니다.
- Kafka가 꺼져 있어도 API는 시작하며 발행자와 세 소비자가 각각 백그라운드에서 재연결합니다. 진행률 HTTP, 연결된 검색 서비스, 작업 감시와 실행은 Kafka 초기 연결을 기다리지 않습니다.
- 결과 전송 실패 시 Redis에 결과와 발행 대기 상태를 유지합니다. 브로커 확인 뒤에만 발행 완료로 바꾸므로 재연결 후 전송만 재시도합니다. 원문 크기는 `payload_bytes`로 기록하고 크기 제한 오류는 `ai_job_result_too_large`로 알립니다. 압축이 메시지 크기 제한을 없애지는 않습니다.
- `process_result`와 `story_result`는 압축 전 UTF-8 JSON 전체 본문을 512KiB(524,288바이트)까지 허용합니다. 작업 종료 상태를 저장하기 전에 검사하며 초과하면 `FAILED / RESULT_TOO_LARGE`로 대체합니다. `result`는 null이고 `error.detail`에는 `max_bytes`와 `actual_bytes`만 넣습니다. 긴 실패 상세도 같은 규칙을 적용하므로 BE는 이 오류 코드를 처리해야 합니다. DLT에는 BE 요청 원문에 헤더를 추가할 여유가 별도로 필요합니다.
- Redis 전체 유실 시 작업과 미발행 결과가 유실될 수 있습니다. 이는 v2의 알려진 한계입니다. Redis 등록 실패는 접수 완료로 처리하지 않습니다.
- 완료 및 무시 기록은 자동 삭제하지 않습니다. 재처리 가능 기간에 맞춘 보관 및 정리 정책과 Redis 용량을 배포 전에 확인해야 합니다. Redis는 `noeviction`을 사용합니다.

`/health`는 API의 Redis 연결과 백그라운드 루프 생존을 확인합니다. Kafka 연결 실패 자체로 503을 반환하지 않으며 예상하지 못한 내부 루프 종료는 비정상으로 구분합니다. GPU 모델 준비나 Qdrant 상태는 이 검사만으로 확인되지 않습니다. API 프로세스에는 모델을 적재하지 않으므로 `model_loaded`는 false이며 유휴 종료 판단은 하지 않습니다.

Kafka는 사설 네트워크의 PLAINTEXT로 연결합니다. 접속 주소와 토픽 이름은 이미지에 넣지 않고 클라우드가 환경변수로 전달합니다. SG 접근 제어는 클라우드 설정을 따릅니다.

발행자는 결과와 DLT 모두 `acks="all"`, `enable_idempotence=True`, `compression_type="lz4"`를 사용합니다. 소비자는 별도 압축 설정 없이 해제합니다. 의존성은 `aiokafka[lz4]==0.14.0`과 uv.lock으로 고정했습니다. 이 버전의 LZ4 코덱은 cramjam으로 제공됩니다. Docker 빌드에서도 다음 검사를 실행합니다.

```sh
python -c "from aiokafka.codec import has_lz4; assert has_lz4()"
```

## 검증

```sh
uv run pytest tests/test_v2_jobs.py
JOBS_TEST_REDIS_URL=redis://127.0.0.1:6379/0 uv run pytest tests/test_v2_jobs.py
JOBS_TEST_REDIS_URL=redis://127.0.0.1:6379/0 JOBS_TEST_KAFKA_URL=localhost:9092 uv run pytest tests/test_v2_jobs_integration.py
```

두 번째 명령은 실제 Redis와 별도 실행 프로세스를 사용합니다. 세 번째 명령에는 토픽 생성과 삭제가 가능한 테스트 전용 Kafka가 필요합니다. 테스트마다 고유한 토픽과 Redis 키 공간을 만들고 해당 자원만 정리합니다. 소비 그룹 이름은 배포 계약과 같으므로 같은 브로커에서 이 통합 테스트를 동시에 여러 개 실행하지 않습니다. API 두 개와 정리 실행기를 기동해 요청, 취소와 결과 전달을 검증하고, API 강제 종료와 재기동으로 소비 그룹이 2개에서 1개를 거쳐 다시 2개가 되는지 확인합니다.

이 통합 테스트의 BE는 Python 발행자와 결과 소비자로 대체하며 모델도 mock을 사용합니다. 실제 BE 저장, Docker 기동, AWS 배포와 모델 품질은 별도로 확인해야 합니다. Kafka 주소를 지정하지 않으면 통합 테스트는 건너뜁니다.
