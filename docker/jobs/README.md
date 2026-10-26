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

Fargate API는 두 개 이상의 복제본과 오토스케일링을 전제로 합니다. EC2 정리 실행기는 1개와 CPU 상한 2 vCPU로 배포하며 한 작업을 끝낸 뒤 다음 작업을 꺼냅니다. 작업 자체는 자식 프로세스에서 실행해 상태 확인과 heartbeat가 CPU 연산에 막히지 않도록 합니다. 정리 실행기는 import와 처리기 적재를 마친 대기 자식을 `JOBS_PROCESS_STANDBY`개(기본 2) 띄워 두고 작업을 stdin으로 넘기며, 하나를 쓰면 바로 보충합니다. 대기 자식도 작업 1건만 처리하고 종료하므로 작업별 격리는 같고, 대기 자식 1개당 메모리가 약 220MB 늘어납니다. 0이면 작업마다 새로 띄웁니다. 스토리 실행은 API의 실행 루프가 담당합니다. `JOBS_STORY_CONCURRENCY=4`는 API 전체를 합친 실행 한도이며 Redis 선점으로 중복과 초과 실행을 차단합니다. API 한 개가 일시적으로 남아도 4건을 처리할 수 있습니다. 모든 복제본의 설정을 같게 유지해야 합니다.

API 종료 시에는 새 스토리 선점을 중단하고 실행 중인 작업을 `JOBS_SHUTDOWN_GRACE_SEC` 동안 기다립니다. 기본 유예는 60초이며 완료 결과는 Redis에 저장합니다. 유예를 넘긴 작업은 자식 프로세스를 종료하고 `FAILED / WORKER_DEAD`로 기록합니다. 취소 요청을 이미 받은 작업은 `CANCELED`로 기록합니다. 결과를 아직 발행하지 못했으면 다른 API가 Redis의 대기 결과를 Kafka로 전달합니다. 강제 종료로 실패 기록도 남기지 못한 경우에는 기존 heartbeat 만료 감시가 회수합니다. BE는 실패 결과를 받은 뒤 새 `execution_id`로 재시도합니다. 5분짜리 작업 전체를 종료 유예 안에 마치는 것은 보장하지 않습니다.

실제 실행에서는 `JOBS_FAKE_PIPELINE=false`와 `JOBS_PROCESS_HANDLER`, `JOBS_STORY_HANDLER`를 설정합니다. API는 스토리 실행 함수가 없으면 Kafka 소비를 시작하지 않고 기동에 실패합니다. 정리 실행기도 정리 함수가 없으면 기동에 실패합니다. 각 값은 `package.module:function`이며, 함수 계약은 아래와 같습니다.

```python
async def run(request: dict, reporter: JobReporter) -> dict:
    await reporter.report("SELECTING", 0, 4)
    # 외부 호출 전후와 단계 경계에서 취소를 확인한다.
    await reporter.check_cancel()
    return result
```

기존 정리 함수 `run(request, ctx, on_progress, is_cancelled)`는 유지합니다. 위 async 함수는 서버 담당자가 연결할 어댑터이며, 정리 담당자에게 알고리즘의 시그니처 변경을 요구하지 않습니다. 기능별 입력과 결과는 `app/jobs/models.py`에 있습니다. 정리 연결은 `JOBS_PROCESS_HANDLER=app.jobs.process_handler:run`으로 활성화합니다. `EMBED_MODEL=qwen3-vl-embedding-2b`, `EMBEDDING_BASE_URL`, `EMBEDDING_SERVED_MODEL_NAME`, `QDRANT_URL`을 설정하고 S3 접근 권한과 `S3_BUCKET`, `AWS_REGION`을 전달합니다. `FAKE_PIPELINE`과 `JOBS_FAKE_PIPELINE`은 모두 false여야 합니다. 로컬 사진 검증에는 `IMAGE_SOURCE=local`, `IMAGE_DIR`을 사용할 수 있습니다. 정리는 동기 파이프라인을 별도 스레드에서 실행하며 단계별 진행률과 취소 여부를 Redis와 교환합니다. 임베딩은 최대 16장씩 순서대로 호출하고 종료 시 HTTP 연결과 Qdrant 및 S3 클라이언트를 닫습니다. 스토리는 별도 실행 함수를 연결합니다.

## 실제 검색 연결

기본 API 실행 명령에서 `SEARCH_ENABLED=true`로 설정하면 Qwen 텍스트 임베딩, Qdrant 후보 검색, 생성용 VLM과 S3 URL 발급기를 연결합니다. 검색은 Kafka와 Redis 대기열을 거치지 않으며 API마다 비동기 클라이언트를 한 번 만들고 종료할 때 닫습니다. `SEARCH_ENABLED=false`는 작업 서버만 검증할 때 사용하는 기본값이며, 이때 검색 HTTP는 503을 반환합니다. 검색 설정이 잘못되었을 때 mock으로 전환하지 않습니다.

루트 `.env.example`을 참고해 다음 값을 환경변수로 주입합니다. 컨테이너 안에 설정 파일을 복사할 필요는 없습니다.

| 설정 | 용도 |
| --- | --- |
| `API_KEY`, `REDIS_URL` | BE의 HTTP 인증과 API가 공유할 작업 저장소 |
| `SEARCH_ENABLED=true` | 기본 API에서 실제 검색 활성화 |
| `EMBEDDING_BASE_URL`, `EMBEDDING_SERVED_MODEL_NAME`, `EMBEDDING_API_KEY` | AWS Qwen 임베딩 서버의 주소와 모델 이름, 서버가 인증을 사용할 때의 키 |
| `QDRANT_URL`, `QDRANT_TIMEOUT_SEC` | Qdrant 주소와 개별 HTTP 요청의 시간 상한. 기본 5초 |
| `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY` | RunPod 생성 서버의 주소와 모델 이름, 서버가 인증을 사용할 때의 키 |
| `RUNPOD_API_KEY`, `RUNPOD_STATUS_CACHE_SEC`, `RUNPOD_STATUS_TIMEOUT_SEC` | `/runpod/status`의 Pod 상태 조회용 RunPod 키(Read Only 권장), 결과 재사용 시간과 호출 상한. 기본 10초와 3초 |
| `IMAGE_SOURCE=s3`, `S3_BUCKET`, `AWS_REGION`, `S3_PRESIGN_EXPIRES_SEC` | 분석용 사진 위치와 URL 유효기간. 기본 300초 |

검색은 `vectors_qwen3-vl-embedding-2b` 컬렉션의 2048차원 벡터를 읽습니다. 정리 실행기가 같은 Qwen 모델로 사진을 임베딩하고 BE의 `trip_attachment_id`를 point ID로 저장해야 합니다. v1 벡터를 대신 검색하지 않습니다. 컬렉션이나 후보의 벡터가 없으면 해당 사진은 결과에서 빠지므로, 검색 검증 전에 사진 적재 여부를 확인합니다.

API의 AWS 역할에는 분석용 사본의 `s3:GetObject` 권한이 필요하며, 생성용 VLM 서버가 서명된 URL로 S3에 접근할 수 있어야 합니다. URL은 QUESTION에서 선별된 근거 사진에만 발급하고 이미지를 내려받거나 다시 인코딩하지 않습니다. 실제 URL 만료는 서명에 사용한 자격 증명의 만료에도 제한됩니다.

스토리 실행 함수가 준비되기 전에 검색만 연결하려면 같은 이미지에서 다음 명령을 사용합니다. 이 진입점은 검색을 활성화하고 Kafka 소비와 스토리 실행을 시작하지 않습니다. Redis와 검색용 설정은 필요하며, 정리와 스토리 메시지의 통합 검증에는 기본 API 진입점을 사용합니다.

```sh
uvicorn app.v2_main:create_search_app --factory --host 0.0.0.0 --port 8000
```

테스트나 별도 조립에서는 `create_app(query_service_factory=...)` 또는 `query_service=...`를 사용할 수 있습니다. 둘을 동시에 지정할 수는 없습니다. `search_service(settings, vectors=..., embedder=...)`에 주입한 자원은 호출자가 닫으며, 생략해서 구성 함수가 만든 자원은 구성 함수가 닫습니다.

임베딩과 후보 검색의 시간 초과는 HTTP 504의 `TIMEOUT`, 연결 실패는 HTTP 503의 `MODEL_NOT_READY`로 반환합니다. 사진 검색 뒤 답변만 실패하면 사진 목록을 유지하고 `answer_error`를 반환합니다. `/health` 성공만으로 모델 호출, Qdrant 데이터나 S3 접근까지 확인되지는 않습니다.

## 업로드 묶음 수신

초기 v2에서는 `photos_ready`를 받으면 사진 ID와 `analyze_storage_key`만 Redis에 보관합니다. 이미지를 다운로드하거나 임베딩하지 않으며 정리 실행 대기열에도 넣지 않습니다. 같은 여행과 실행의 묶음은 `batch_no`로 구분하고, 같은 메시지가 다시 와도 중복 저장하지 않습니다. 사진은 S3에 그대로 둡니다.

BE가 모든 사진의 업로드와 분석용 사본 생성을 마친 뒤 보내는 `process`가 정리 시작 기준입니다. 200장은 여행당 상한이며, 200장이 모이는 조건으로 시작하지 않습니다. `process.attachments`의 전체 목록으로 작업을 등록하면서 임시 묶음을 삭제합니다. 묶음 알림이 누락되거나 만료되어도 진행하며, 보관된 목록에만 있는 사진은 처리하지 않습니다. 정리 알고리즘과 모델의 실제 실행은 위의 핸들러 연결이 필요합니다.

- `photos_ready`는 여행 생성 또는 수정 요청 토픽에 같은 실행의 `process`와 동일한 `trip_id` key 및 `execution_id`로 보냅니다. 한 묶음은 1~16장입니다.
- Redis 보관이 성공한 뒤 Kafka offset을 커밋합니다. `process`가 이미 등록되거나 진행 중 작업 때문에 무시된 실행의 늦은 묶음은 다시 보관하지 않습니다.
- 같은 묶음 번호의 내용이나 같은 사진 ID의 사본 경로가 달라지면 원래 정보를 유지하고 잘못된 알림을 DLT로 격리합니다. 초기 보관은 실행당 고유 사진 200장과 묶음 200개까지 허용합니다. S3 키는 UTF-8 1024바이트까지, 실행별 임시 JSON 전체는 2MiB까지 허용합니다. 초과한 묶음도 기존 정보를 유지한 채 DLT로 격리합니다.
- 잘못된 묶음은 검증 실패 3회 후 요청 DLT에 보냅니다. 보조 알림 자체로 `process_result`나 작업 FAILED를 만들지 않으므로 뒤에 오는 정상 `process`를 차단하지 않습니다.
- `process` 전에는 새 정리 상태를 만들지 않습니다. 첫 실행의 GET은 404이며, 이전 실행이 있으면 기존 상태가 보입니다. BE는 `execution_id`를 비교하고 최종 요청 접수 후의 상태를 확인해야 합니다. 업로드 중 취소는 BE에서 업로드와 최종 `process` 발행을 중단하는 범위이며 AI DELETE는 등록된 정리 작업에 적용합니다.
- 최종 요청이 오지 않은 임시 묶음은 기본 7일 후 만료합니다. `JOBS_PHOTOS_READY_TTL_SEC`로 조정하며 새 묶음을 저장할 때만 갱신합니다. 작업 상태와 미발행 결과의 보관 규칙은 바뀌지 않습니다.

묶음을 받는 즉시 임베딩하고 최종 정리에서 벡터를 재사용하는 기능은 후속 개선입니다. 현재 구현은 초기 수신과 대기만 제공합니다.

## 통신과 실패 처리

- 요청과 결과 토픽은 클라우드가 전달하는 환경변수 6개를 그대로 사용합니다. `JOBS_ENV`는 Redis 키 공간만 구분합니다. `.env.example`의 `local` 값은 로컬 검증용이며 스테이징과 운영에서는 환경별 값을 주입합니다.
- API마다 요청 종류별 소비자를 하나씩 실행합니다. 여행 생성은 `yeodam-ai-trip-create`, 사진 추가는 `yeodam-ai-trip-update`, 스토리는 `yeodam-ai-story-create` 그룹을 사용합니다. 두 API는 해당 그룹 이름과 Redis를 공유합니다. 환경마다 브로커가 분리되므로 그룹에 환경명을 붙이지 않습니다.
- 요청 key는 UTF-8로 인코딩한 문자열 trip_id이며 value는 UTF-8 JSON입니다. process와 story는 Redis 작업 등록 및 대기열 추가 후, photos_ready는 임시 묶음 보관 후 해당 partition의 offset을 커밋합니다. 자동 커밋을 끄고 `auto_offset_reset="earliest"`와 기본 poll 제한 5분을 사용합니다.
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
- process와 story의 식별 가능한 잘못된 본문은 기존 BE 계약대로 `FAILED / INVALID_REQUEST` 결과로 반환합니다. 클라우드 안내의 형식 오류 DLT 범위를 이 경우까지 넓힐지는 BE와 대조할 사항입니다. photos_ready의 잘못된 본문은 위 규칙에 따라 DLT로 보냅니다. 정상 접수 후 분석 실패도 결과 토픽의 FAILED로 전달합니다.
- Kafka가 꺼져 있어도 API는 시작하며 발행자와 세 소비자가 각각 백그라운드에서 재연결합니다. 진행률 HTTP, 연결된 검색 서비스, 작업 감시와 실행은 Kafka 초기 연결을 기다리지 않습니다.
- 결과 전송 실패 시 Redis에 결과와 발행 대기 상태를 유지합니다. 브로커 확인 뒤에만 발행 완료로 바꾸므로 재연결 후 전송만 재시도합니다. 원문 크기는 `payload_bytes`로 기록하고 크기 제한 오류는 `ai_job_result_too_large`로 알립니다. 압축이 메시지 크기 제한을 없애지는 않습니다.
- `process_result`와 `story_result`는 압축 전 UTF-8 JSON 전체 본문을 512KiB(524,288바이트)까지 허용합니다. 작업 종료 상태를 저장하기 전에 검사하며 초과하면 `FAILED / RESULT_TOO_LARGE`로 대체합니다. `result`는 null이고 `error.detail`에는 `max_bytes`와 `actual_bytes`만 넣습니다. 긴 실패 상세도 같은 규칙을 적용하므로 BE는 이 오류 코드를 처리해야 합니다. DLT에는 BE 요청 원문에 헤더를 추가할 여유가 별도로 필요합니다.
- Redis 전체 유실 시 작업과 미발행 결과가 유실될 수 있습니다. 이는 v2의 알려진 한계입니다. Redis 등록 실패는 접수 완료로 처리하지 않습니다.
- 완료 및 무시 기록은 자동 삭제하지 않습니다. 재처리 가능 기간에 맞춘 보관 및 정리 정책과 Redis 용량을 배포 전에 확인해야 합니다. Redis는 `noeviction`을 사용합니다.

`/health`는 API의 Redis 연결과 백그라운드 루프 생존을 확인합니다. Kafka 연결 실패 자체로 503을 반환하지 않으며 예상하지 못한 내부 루프 종료는 비정상으로 구분합니다. GPU 모델 준비나 Qdrant 상태는 이 검사만으로 확인되지 않습니다. API 프로세스에는 모델을 적재하지 않으므로 `model_loaded`는 false이며 유휴 종료 판단은 하지 않습니다.

`GET /runpod/status`는 BE가 RunPod 생성 서버의 상태를 확인하는 경로이며 `API_KEY` 인증을 사용합니다. 응답은 `{"running": true, "ready": false}`처럼 두 값입니다. `running`은 RunPod API에서 읽은 Pod의 `desiredStatus`가 `RUNNING`이면 true, `EXITED`나 `TERMINATED`이면 false이고, 키가 없거나 RunPod API를 읽지 못하면 null입니다. Pod ID는 `VLM_BASE_URL`의 `https://<pod-id>-8000.proxy.runpod.net` 주소에서 읽습니다. `ready`는 vLLM `/v1/models`에 `VLM_MODEL`이 있으면 true이고 그 밖에는 false입니다. Pod가 켜진 뒤 모델 적재가 끝날 때까지(재시작 약 3분) `running=true`, `ready=false`가 됩니다. RunPod나 vLLM이 응답하지 않아도 200으로 응답하고, API마다 결과를 `RUNPOD_STATUS_CACHE_SEC` 동안 재사용합니다. 상태가 바뀔 때만 `ai_runpod_status` 로그를 남깁니다.

Kafka는 사설 네트워크의 PLAINTEXT로 연결합니다. 접속 주소와 토픽 이름은 이미지에 넣지 않고 클라우드가 환경변수로 전달합니다. SG 접근 제어는 클라우드 설정을 따릅니다.

발행자는 결과와 DLT 모두 `acks="all"`, `enable_idempotence=True`, `compression_type="lz4"`를 사용합니다. 소비자는 별도 압축 설정 없이 해제합니다. 의존성은 `aiokafka[lz4]==0.14.0`과 uv.lock으로 고정했습니다. 이 버전의 LZ4 코덱은 cramjam으로 제공됩니다. Docker 빌드에서도 다음 검사를 실행합니다.

```sh
python -c "from aiokafka.codec import has_lz4; assert has_lz4()"
```

## 검증

```sh
uv run pytest tests/test_v2_jobs.py tests/test_photos_ready.py
JOBS_TEST_REDIS_URL=redis://127.0.0.1:6379/0 uv run pytest tests/test_v2_jobs.py tests/test_photos_ready.py
JOBS_TEST_REDIS_URL=redis://127.0.0.1:6379/0 JOBS_TEST_KAFKA_URL=localhost:9092 uv run pytest tests/test_v2_jobs_integration.py
```

두 번째 명령은 실제 Redis와 별도 실행 프로세스를 사용합니다. 세 번째 명령에는 토픽 생성과 삭제가 가능한 테스트 전용 Kafka가 필요합니다. 테스트마다 고유한 토픽과 Redis 키 공간을 만들고 해당 자원만 정리합니다. 소비 그룹 이름은 배포 계약과 같으므로 같은 브로커에서 이 통합 테스트를 동시에 여러 개 실행하지 않습니다. API 두 개와 정리 실행기를 기동해 요청, 취소와 결과 전달을 검증하고, API 강제 종료와 재기동으로 소비 그룹이 2개에서 1개를 거쳐 다시 2개가 되는지 확인합니다.

이 통합 테스트의 BE는 Python 발행자와 결과 소비자로 대체하며 모델도 mock을 사용합니다. 실제 BE 저장, Docker 기동, AWS 배포와 모델 품질은 별도로 확인해야 합니다. Kafka 주소를 지정하지 않으면 통합 테스트는 건너뜁니다.


## 스토리 기준선 실행

`JOBS_STORY_HANDLER=app.story.handler:run`으로 실제 스토리를 활성화합니다. Qwen 사진 벡터가 Qdrant에 먼저 저장되어 있어야 합니다. `IMAGE_SOURCE=s3`, `S3_BUCKET`, `AWS_REGION`, `QDRANT_URL`, `VLM_BASE_URL`, `VLM_MODEL`과 필요한 인증값을 전달합니다.

스토리는 S3 사본의 흐림과 중복을 제외하고, 장소마다 한 장을 확보한 뒤 나머지 사진 예산을 남은 사진 수에 비례해 배분합니다. 소수점은 최대 나머지 방식으로 처리하며 동점에서는 입력 장소 순서를 사용합니다. 장소별 평균 임베딩에 가까운 사진부터 고르되, 이미 고른 사진과 코사인 유사도가 기준 이상이면 같은 장면으로 보고 건너뜁니다. 기준은 여행 안 같은 장소 사진 쌍 유사도의 평균 + `STORY_DEDUPE_Z` × 표준편차이며, 건너뛰어 못 채운 자리는 비워 둡니다. 고른 사진은 VLM에 한 번 전달합니다. 모델이 반환한 사진 ID와 형식 및 모든 선택 장소의 포함 여부를 검사하며 결과는 촬영 시각순으로 정렬합니다.

초기 설정은 `STORY_IMAGE_BUDGET=10`, `STORY_BLUR_VAR_MIN=80`, `STORY_PHASH_MAX_DIST=4`, `STORY_DEDUPE_Z=1.0`, `STORY_MAX_TOKENS=2048`입니다. RunPod의 이미지 입력 상한은 사진 예산 이상이어야 하며, 선택한 장소 수보다 예산이 작으면 요청이 실패합니다. 사진이 남지 않은 장소는 `INVALID_REQUEST`의 `NO_USABLE_PHOTOS`, 필요한 벡터 누락은 `MODEL_NOT_READY`의 `MISSING_VECTORS`로 알립니다. 필터 임계값과 실제 호출 품질은 스테이징에서 확인합니다.

프롬프트는 요약 두 필드를 각각 40자 이내의 짧은 구절로, `memo`를 125자 이내로 요청합니다. 응답 스키마는 선별한 사진 수만큼 블록을 요구하고 사진 ID도 해당 요청의 선별 ID로 제한합니다. 최종 검증에서는 모든 선별 사진이 정확히 한 번씩 포함되었는지 검사합니다. 출력 한도에 도달하거나 사진이 누락된 응답을 완료 결과로 전달하지 않습니다. 서버의 `disable_any_whitespace` 옵션은 사용하지 않습니다.

기존 결과 계약의 `verification.passed=true`는 코드가 ID, 형식과 장소 포함 여부를 검사했다는 뜻입니다. 문장의 사실성을 별도 모델로 검증했다는 뜻은 아닙니다. 한 번 작성하는 기준선이므로 재작성 횟수는 0입니다. 입력 `taken_at`은 기존 계약대로 오프셋을 포함한 필수 값이며, `day_label`은 제공된 촬영 시각과 여행 시작일로 계산합니다.


## Redis와 GPU 요청 제어

Redis는 작업 상태와 실행 대기열을 저장하며 GPU를 직접 호출하지 않습니다. 정리 실행기가 한 작업씩 꺼내 AWS 임베딩 vLLM에 최대 16장 묶음을 순차 전달합니다. 검색 HTTP 요청은 Qwen 텍스트 임베딩을 호출하며, 질의 해석과 검색 답변 및 스토리 집필은 RunPod 생성 vLLM을 호출합니다.

API 복제본과 스토리 자식 프로세스는 Redis의 호출 슬롯을 공유합니다. `GPU_GENERATION_INFLIGHT=8`은 검색과 스토리의 생성 호출을 합친 한도이며, 스토리 작업 한도 4건과 구분합니다. `GPU_SEARCH_EMBEDDING_INFLIGHT=2`는 별도의 검색 텍스트 임베딩 한도입니다. `GPU_WAIT_SEC=2` 동안 빈 슬롯을 기다리고 확보하지 못하면 기존 검색 및 스토리 오류 계약으로 반환합니다. 모든 복제본에 같은 설정을 적용하고 RunPod의 동시 처리 설정 및 실제 혼합 부하를 함께 확인합니다. 호출 한도는 검색 응답 시간이나 검색 전용 슬롯을 보장하지 않습니다.

정리의 이미지 임베딩은 기존 단일 실행기와 16장 순차 묶음으로 제한하므로 텍스트 슬롯을 사용하지 않습니다. AWS 임베딩 서버에는 이미지 요청 최대 16개와 텍스트 요청 최대 2개가 함께 도달할 수 있습니다. 이 설정은 검색 우선순위나 지연 시간 상한을 보장하지 않습니다.

슬롯에는 무작위 소유 토큰과 Redis 시각을 기준으로 한 만료 시각을 둡니다. 정상 종료와 오류, 취소 및 전체 호출 시간 초과 시 해당 토큰을 제거하며, 프로세스 강제 종료 후 남은 토큰은 호출 제한 시간과 10초의 여유가 지나면 재사용할 수 있습니다. Redis 장애 시 새 GPU 호출을 시작하지 않습니다. HTTP 연결을 취소해도 서버 내부 GPU 연산의 즉시 종료를 보장하지는 않습니다. 원격 서버의 요청 중단과 복구는 실제 환경에서 확인해야 합니다. vLLM 우선순위 스케줄링은 이번 범위에 포함하지 않습니다.
