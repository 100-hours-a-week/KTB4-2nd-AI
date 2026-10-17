# AI 운영 지표와 품질 평가

Cloud는 Prometheus에 두 API 인스턴스의 사설 주소를 등록하고 Grafana에서 `grafana-dashboard.json`을 가져옵니다. Prometheus 데이터 소스를 선택하고 환경 변수를 실제 환경명으로 바꿉니다. 별도 Prometheus나 Grafana 서버를 이 애플리케이션 안에서 실행하지 않습니다.

`prometheus.yml`은 `yeodam-rules.yml`, `yeodam-targets.json`과 인증키 파일을 참조합니다. `targets.example.json`의 주소를 실제 Fargate task 주소나 서비스 디스커버리 주소로 바꾸고 해당 경로에 마운트합니다. 인스턴스를 교체하면 대상 목록도 갱신해야 합니다. 인증키는 `/etc/prometheus/secrets/ai_api_key` 파일로 전달하며 저장소에 추가하지 않습니다. `/metrics`는 BE API와 같은 Bearer 인증을 요구합니다.

## 집계 방식

정리와 스토리는 Redis 상태가 종료 상태로 처음 바뀌는 트랜잭션에서만 집계합니다. 같은 결과의 발행 재시도나 다른 API의 상태 조회는 처리 횟수를 늘리지 않습니다. `total`은 접수 후 완료까지, `queue`는 실행 대기, `execution`은 실행 시작 후 완료까지의 시간입니다. 실행 전 취소한 작업은 execution 시간이 0입니다. 검색의 parse와 search는 각각 요청 시간과 success, fallback, partial, error 결과를 집계합니다. 검색 지표 저장 실패는 응답을 실패로 바꾸지 않고 `ai_metrics_write_failed` 로그를 남깁니다.

모든 API가 같은 Redis 집계값을 노출하므로 **복제본의 값을 합산하지 않습니다.** 제공한 recording rule은 인스턴스별 값의 최댓값을 취한 뒤 rate와 p95를 계산합니다. 환경 label은 유지합니다. API별로 한 번씩 스크레이프하되 같은 작업을 두 번 센다고 해석하지 않습니다. Redis 유실 시 집계도 초기화되며 장기간 이력은 Prometheus 저장소가 보관합니다.

사진 분류 수와 미분류 수, 스토리 블록 수는 처리량과 출력 상태입니다. 정확도를 나타내지 않습니다. HTTP 로그는 `ai_query`, 작업 종료 로그는 `ai_job_finished`로 찾을 수 있습니다. 모델 응답 본문과 질문, 사진 URL은 메트릭 label에 넣지 않습니다. 작업 식별자는 로그에서 추적하고 운영 메트릭에는 고정된 기능과 단계 및 결과 label만 사용합니다. 평가 메트릭에는 비교 조건인 데이터셋과 모델, k 및 미분류 정책을 함께 표시합니다. 이 값에는 질문, 파일 경로, 사용자 식별자나 실행별 임의 이름을 사용하지 않습니다.

## 정답 기반 평가

`python -m scripts.evaluate_v2 INPUT.json --output REPORT.json`으로 정답과 예측을 비교합니다. 결과를 대시보드에 반영하려면 테스트 환경의 Redis 설정을 준비하고 `--publish`를 추가합니다. 이 명령은 `ai_evaluation` JSON을 출력하고 원본 입력 파일은 수정하지 않습니다. 데이터셋 이름, 모델, 표본 수와 평가 시각을 결과 파일과 로그에 남깁니다. 수치와 조건은 Redis에서 함께 저장하고 읽습니다. Prometheus에서는 같은 API에서 수집한 수치와 조건을 먼저 결합한 뒤 복제본을 합칩니다. 최근 평가는 기능, 데이터셋, 미분류 정책의 조합마다 따로 보관하므로 서로 덮어쓰지 않습니다. 서로 다른 데이터셋과 모델의 숫자를 같은 실험처럼 비교하지 않습니다.

| feature | items의 평가 입력 | 계산하는 지표 |
| --- | --- | --- |
| process | `expected_place`, `predicted_place` | B-cubed precision, recall, F1, ARI와 coverage |
| search | `relevant_ids`, `retrieved_ids`, 동일한 `k` | precision@k, recall@k, MRR@k |
| story | 사람이 확인한 `factual`, `all_places_present`, `mood_matched` boolean | 각 조건을 만족한 비율 |

공통 입력은 `feature`, `dataset`, `model`, `items`입니다. 정리의 미분류 예측은 null로 입력하고 `unclassified_policy`를 `singleton` 또는 `exclude`로 명시합니다. 어느 채점 정책을 서비스 평가에 채택할지는 별도 결정이며, 누락하면 임의로 선택하지 않고 실패합니다. 제외 방식은 분류된 사진에 대한 점수이며 coverage를 함께 봐야 합니다. 스토리의 `verification.passed`는 이 평가의 사실성 정답으로 사용할 수 없습니다. 실제 평가 입력이 없으면 품질 지표를 0이나 1로 만들지 않고 미수집 상태로 둡니다.

### 평가 데이터셋 자동 채점

정리 작업이 끝나면 내려받은 사본의 지문을 `app/observability/eval_sets/`의 기준 지문과 대조합니다. 여행 사진의 90% 이상(최소 10장)이 한 데이터셋과 맞으면 그 데이터셋의 정답으로 자동 채점합니다. 지문은 회색조 이미지의 pHash 64비트이며, 거리 8 이하를 같은 사진으로 봅니다. 기준 지문은 원본을 방향 태그대로 바로 세워 만들고, 사본은 픽셀 그대로와 방향 태그대로 바로 세운 것을 모두 대조합니다. BE가 사진을 회전했는지, 방향 태그를 남겼는지와 무관하게 인식하기 위해서입니다. 현재 기준 지문은 정선 가족여행(`jeongseon-family-trip`, 200장)과 eval(`eval-travel-photos`, 198장)입니다.

인식 결과는 `finalizing.evaluate` 단계의 `ai_stage` 로그에 남습니다. `closest_dataset`과 `match_ratio`는 가장 많이 맞은 데이터셋과 그 비율이며, 기준을 넘었을 때만 `dataset`이 채워집니다. 평가 여행인데 기준에 못 미친 경우를 여기서 찾습니다.

- 흐림이나 중복으로 미분류된 사진은 결과의 원래 장소로 채점합니다.
- 정답이 미분류인 사진은 사진마다 단독 묶음으로 채점합니다.
- 미분류 예측은 `singleton`과 `exclude` 두 정책을 모두 계산해 `ai_evaluation` 로그 두 줄과 지표로 남깁니다.
- 로그에는 대조된 사진 수(`matched_photos`), 대조되지 않은 사진 수(`unmatched_photos`), 최대 지문 거리와 배포 커밋(`release`)이 함께 남습니다. 지표에도 `matched_photos`와 `dataset_photos`가 남고, 조건 정보의 `source`로 자동 채점(`auto`)과 수동 게시(`manual`)를 구분합니다.
- 채점 결과는 작업 결과를 저장한 뒤, 결과가 COMPLETED일 때만 로그와 지표로 남깁니다. 실행 중 취소된 작업은 `ai_evaluation_skipped`만 남깁니다.

지문 계산, 기준 지문 파일 읽기, 채점에 실패해도 정리 결과는 바뀌지 않고 `ai_evaluation_failed` 로그만 남습니다. `EVAL_AUTO_DETECT=false`로 끌 수 있습니다. 라벨이 바뀌면 `scripts/build_eval_fingerprints.py`로 기준 지문을 다시 만듭니다.

평가 지표는 `metrics:quality_v2`와 `metrics:quality_context_v2`에 저장합니다. 이전 버전(`22d2bd6`)의 렌더러가 읽는 `metrics:quality`와 형식이 달라, 같은 키를 쓰면 롤백하거나 버전이 섞일 때 `/metrics` 전체가 실패하기 때문입니다. 이전 키에 남은 수동 평가는 새 대시보드에 나타나지 않으므로 필요하면 다시 게시합니다.

## 로그 이벤트

모든 로그는 한 줄 JSON이며 `trip_id`와 `job_id`로 같은 작업을 찾습니다. `release`는 이미지 빌드 시점의 커밋입니다.

| 이벤트 | 남는 시점 | 문제 위치를 찾는 데 쓰는 필드 |
| --- | --- | --- |
| `ai_job_received` | Kafka 요청을 접수하거나 거절할 때 | `operation`, `result`, `attachment_count`, 형식 오류의 `invalid_fields` |
| `ai_job_invalid_retry` | 처리할 수 없는 요청을 다시 시도할 때(3회째에는 `kafka_dlt_published`로 DLT에 보관) | `result`(오류 분류), `attempts`, `partition`, `offset` |
| `ai_stage` | 정리와 스토리의 단계 하나가 끝날 때 | `stage`, `parent_stage`, `result`, `duration_ms`, 처리 수와 `items_per_sec` |
| `ai_model_call` | VLM을 호출하거나 S3 URL을 발급할 때 | `target`, `result`, `status_code`, `finish_reason`, 토큰 수, `tokens_per_sec`. `result=cancelled`는 호출자의 더 짧은 시간 제한(검색 단계별 제한 등)이나 작업 취소로 중단된 경우다 |
| `ai_process_dependency_failed` | 정리 작업의 사진 임베딩 호출이 실패했을 때 | `target`, `status_code`, 예외 종류 |
| `ai_gpu_gate_unavailable` | GPU 호출 슬롯을 확보하지 못했을 때 | `gate`, `limit`, `wait_ms` |
| `ai_story_output_invalid` | 생성된 스토리가 출력 규칙을 어겼을 때 | `rule`, 검증 오류 위치 |
| `ai_job_failed` | 작업이 실패했을 때 | `failure_stage`, `error_code`, `error_reason`, 예외 종류와 발생 위치 |
| `ai_worker_exit`, `ai_job_lease_lost` | 실행 프로세스가 결과 없이 끝나거나 소유권을 잃었을 때 | `exit_code`(-9는 강제 종료), `status` |
| `ai_job_finished` | 작업이 종료 상태가 될 때 | `result`, `failure_stage`, `queue_ms`, `execution_ms`, `duration_ms` |
| `ai_job_summary` | 작업 실행 프로세스가 끝날 때 | `stages_ms`, `model_calls`, `gpu_wait_ms`, `max_rss_mb`, CPU 시간 |
| `ai_query` | 검색 요청이 끝날 때 | `result`, `error_code`, `answer_error`, `stages_ms`, `hit_count`, `top_score` |
| `ai_evaluation` | 평가 데이터셋을 자동 채점했을 때 | `dataset`, `unclassified_policy`, `metrics`, `matched_photos` |

질문 원문, 사진 키, presigned URL, 인증값, 모델 응답 본문과 예외 메시지는 남기지 않습니다. 예외는 종류(`error_type`, `error_chain`)와 `app/` 안의 발생 위치(`error_frames`)만 남깁니다.

## 세부 단계 지표

`yeodam_ai_operation_duration_seconds`의 `phase`에는 `total`, `queue`, `execution`, `request` 외에 세부 단계 이름이 들어가고, `outcome`에는 단계 결과(`success`, `error`, `timeout`, `cancelled`)가 들어갑니다. 검색은 요청이 끝났는데 중단된 단계를 단계별 시간 제한에 걸린 것으로 보고 `timeout`으로 기록합니다. 로그의 `stage`, `stages_ms`와 지표의 `phase`는 같은 이름(`selecting.filter` 등)을 씁니다. 정리는 `download`, `embedding`, `clock_correction`, `locating`, `clustering`, `filtering`, `finalizing`이고, 스토리는 `selecting`, `writing`, `verifying`과 그 아래 세부 단계(`selecting.filter` 등)입니다. 검색은 `interpret`, `embed`, `retrieve`, `sign_url`, `answer`입니다.

작업 실행 프로세스의 최대 메모리는 `yeodam_ai_job_peak_memory_bytes`(기능별 최근 작업), API 인스턴스의 메모리와 CPU는 `process_resident_memory_bytes`와 `process_cpu_seconds_total`로 봅니다. 앞의 값은 Redis 공유값이라 복제본마다 같고, 뒤의 값은 인스턴스마다 다릅니다.

## 모델 서버와 GPU 지표

`prometheus.yml`의 `yeodam-vllm`과 `yeodam-gpu` job은 각각 `/etc/prometheus/yeodam-vllm-targets.json`, `/etc/prometheus/yeodam-gpu-targets.json`을 읽습니다. `vllm-targets.example.json`과 `gpu-targets.example.json`을 참고해 실제 사설 주소로 바꾼 파일을 기존 `yeodam-targets.json`과 같은 위치에 마운트합니다. 파일이 없으면 해당 job은 대상 없이 비어 있습니다.

GPU 수집기의 포트는 기본으로 `127.0.0.1`에만 열립니다. Prometheus가 다른 호스트에서 수집한다면 `DCGM_EXPORTER_BIND_HOST`나 `NVIDIA_GPU_EXPORTER_BIND_HOST`를 사설 주소로 지정하고, 보안 그룹에서 Prometheus만 허용합니다.

| 대상 | 수집 경로 | 대표 지표 | 배포 담당 |
| --- | --- | --- | --- |
| 임베딩과 생성 vLLM | 각 서버의 `/metrics` | `vllm:num_requests_waiting`, `vllm:num_requests_running`, `vllm:kv_cache_usage_perc`, `vllm:generation_tokens_total` | 수집 대상 등록은 Cloud |
| AWS 임베딩 GPU | `gpu-exporters.compose.yaml`의 `dcgm` 프로필(9400) | `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`(MiB) | 실행과 보안 그룹은 Cloud |
| RunPod 생성 GPU | 같은 파일의 `nvidia-smi` 프로필(9835) | `nvidia_smi_utilization_gpu_ratio`, `nvidia_smi_memory_used_bytes` | Pod 구성은 Kael |

vLLM의 API 키는 `/v1` 경로에만 적용되고 GPU 수집기에는 인증이 없습니다. 세 대상 모두 공개 주소로 열지 말고 Prometheus만 접근하게 합니다. RunPod는 AWS의 Prometheus와 사설망으로 연결되어 있지 않으므로, 생성 서버의 vLLM과 GPU 지표를 어떻게 수집할지는 아직 정하지 않았습니다. 공개 프록시를 쓰려면 인증 방안을 먼저 정해야 하며, Cloud와 Kael의 협의가 필요합니다. 지표 이름은 vLLM `v0.31.0`, dcgm-exporter `4.6.0-4.8.3`, nvidia_gpu_exporter `1.15.1` 기준이며, 버전을 바꾸면 실제 `/metrics` 출력으로 확인합니다.

Prometheus 형식은 [공식 Python 클라이언트](https://prometheus.github.io/client_python/collector/custom/)로 생성하고 [공식 스크레이프 설정](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)을 사용합니다. 실제 서버 스크레이프와 Grafana 화면 확인은 Cloud 환경에서 수행해야 합니다.
