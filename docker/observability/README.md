# AI 운영 지표와 품질 평가

Cloud는 Prometheus에 두 API 인스턴스의 사설 주소를 등록하고 Grafana에서 `grafana-dashboard.json`을 가져옵니다. Prometheus 데이터 소스를 선택하고 환경 변수를 실제 환경명으로 바꿉니다. 별도 Prometheus나 Grafana 서버를 이 애플리케이션 안에서 실행하지 않습니다.

`prometheus.yml`은 `yeodam-rules.yml`, `yeodam-targets.json`과 인증키 파일을 참조합니다. `targets.example.json`의 주소를 실제 Fargate task 주소나 서비스 디스커버리 주소로 바꾸고 해당 경로에 마운트합니다. 인스턴스를 교체하면 대상 목록도 갱신해야 합니다. 인증키는 `/etc/prometheus/secrets/ai_api_key` 파일로 전달하며 저장소에 추가하지 않습니다. `/metrics`는 BE API와 같은 Bearer 인증을 요구합니다.

## 집계 방식

정리와 스토리는 Redis 상태가 종료 상태로 처음 바뀌는 트랜잭션에서만 집계합니다. 같은 결과의 발행 재시도나 다른 API의 상태 조회는 처리 횟수를 늘리지 않습니다. `total`은 접수 후 완료까지, `queue`는 실행 대기, `execution`은 실행 시작 후 완료까지의 시간입니다. 실행 전 취소한 작업은 execution 시간이 0입니다. 검색의 parse와 search는 각각 요청 시간과 success, fallback, partial, error 결과를 집계합니다. 검색 지표 저장 실패는 응답을 실패로 바꾸지 않고 `ai_metrics_write_failed` 로그를 남깁니다.

모든 API가 같은 Redis 집계값을 노출하므로 **복제본의 값을 합산하지 않습니다.** 제공한 recording rule은 인스턴스별 값의 최댓값을 취한 뒤 rate와 p95를 계산합니다. 환경 label은 유지합니다. API별로 한 번씩 스크레이프하되 같은 작업을 두 번 센다고 해석하지 않습니다. Redis 유실 시 집계도 초기화되며 장기간 이력은 Prometheus 저장소가 보관합니다.

사진 분류 수와 미분류 수, 스토리 블록 수는 처리량과 출력 상태입니다. 정확도를 나타내지 않습니다. HTTP 로그는 `ai_query`, 작업 종료 로그는 `ai_job_finished`로 찾을 수 있습니다. 모델 응답 본문과 질문, 사진 URL은 메트릭 label에 넣지 않습니다. 작업 식별자는 로그에서 추적하고 운영 메트릭에는 고정된 기능과 단계 및 결과 label만 사용합니다. 평가 메트릭에는 비교 조건인 데이터셋과 모델, k 및 미분류 정책을 함께 표시합니다. 이 값에는 질문, 파일 경로, 사용자 식별자나 실행별 임의 이름을 사용하지 않습니다.

## 정답 기반 평가

`python -m scripts.evaluate_v2 INPUT.json --output REPORT.json`으로 정답과 예측을 비교합니다. 결과를 대시보드에 반영하려면 테스트 환경의 Redis 설정을 준비하고 `--publish`를 추가합니다. 이 명령은 `ai_evaluation` JSON을 출력하고 원본 입력 파일은 수정하지 않습니다. 데이터셋 이름, 모델, 표본 수와 평가 시각을 결과 파일과 로그에 남깁니다. 수치와 조건은 Redis에서 함께 저장하고 읽습니다. Prometheus에서는 같은 API에서 수집한 수치와 조건을 먼저 결합한 뒤 복제본을 합칩니다. 대시보드는 기능별 최근 평가를 표시하므로 서로 다른 데이터셋과 모델의 숫자를 같은 실험처럼 비교하지 않습니다.

| feature | items의 평가 입력 | 계산하는 지표 |
| --- | --- | --- |
| process | `expected_place`, `predicted_place` | B-cubed precision, recall, F1, ARI와 coverage |
| search | `relevant_ids`, `retrieved_ids`, 동일한 `k` | precision@k, recall@k, MRR@k |
| story | 사람이 확인한 `factual`, `all_places_present`, `mood_matched` boolean | 각 조건을 만족한 비율 |

공통 입력은 `feature`, `dataset`, `model`, `items`입니다. 정리의 미분류 예측은 null로 입력하고 `unclassified_policy`를 `singleton` 또는 `exclude`로 명시합니다. 어느 채점 정책을 서비스 평가에 채택할지는 별도 결정이며, 누락하면 임의로 선택하지 않고 실패합니다. 제외 방식은 분류된 사진에 대한 점수이며 coverage를 함께 봐야 합니다. 스토리의 `verification.passed`는 이 평가의 사실성 정답으로 사용할 수 없습니다. 실제 평가 입력이 없으면 품질 지표를 0이나 1로 만들지 않고 미수집 상태로 둡니다.

Prometheus 형식은 [공식 Python 클라이언트](https://prometheus.github.io/client_python/collector/custom/)로 생성하고 [공식 스크레이프 설정](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)을 사용합니다. 실제 서버 스크레이프와 Grafana 화면 확인은 Cloud 환경에서 수행해야 합니다.
