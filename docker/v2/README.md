# v2 배포 구성

Cloud는 아래 다섯 종류의 컨테이너를 배치합니다. API는 Fargate에서 두 개 이상의 복제본과 오토스케일링을 사용합니다. 정리 실행기 1개와 임베딩 vLLM 및 Qdrant는 g6.xlarge 한 대를 공유합니다. Redis는 별도 컨테이너이고 생성용 vLLM은 RunPod 1 Pod입니다. API와 정리 실행기는 같은 애플리케이션 이미지를 사용하므로 AWS에서 사용하는 이미지 종류는 네 가지입니다.

| 역할 | 이미지 또는 실행 구성 | 실행 명령과 저장소 |
| --- | --- | --- |
| API | `docker/jobs/Dockerfile` | 기본 CMD, Fargate 오토스케일링, 복제본 수와 정책은 Cloud에서 설정 |
| 정리 실행기 | API와 같은 image digest | `python -m app.jobs.runner`, 1개, CPU 상한 2 vCPU |
| 임베딩 vLLM | `docker/embedding/compose.yaml`의 검증 이미지 | 독립 pooling 서버, 모델 캐시 볼륨 |
| Qdrant | 기존 저장소에서 사용하는 v1.19.1 image digest | `/qdrant/storage`를 호스트 EBS에 연결 |
| Redis | `redis:7.4.2` | `noeviction`, 영속화는 Cloud의 후속 결정 |
| RunPod 생성 vLLM | `docker/generation/compose.yaml` | Kael이 확인한 이미지와 모델 revision을 사용 |

## 이미지 준비와 전달

```sh
docker buildx build --platform linux/amd64 --load -f docker/jobs/Dockerfile -t yeodam-ai-app:local .
```

`AI v2 App Image` 워크플로는 main 병합 후 `ghcr.io/100-hours-a-week/yeodam-ai-app:sha-<commit>`을 생성하고 요약에 digest를 기록합니다. PR Gate는 같은 Dockerfile을 push 없이 빌드합니다. 이 워크플로에는 운영 배포나 Cloud 저장소 변경 요청이 없습니다. 기존 v1 이미지 워크플로는 별도로 유지합니다.

스토리와 정리 실행 함수 및 실제 검색 연결 변경을 포함한 이미지여야 합니다. `GPU_*` 설정으로 복제본 전체의 요청 수를 제한하려면 #82의 공통 실행 제어도 포함해야 합니다. 이미지 빌드 성공은 모델 기동이나 스테이징 통합 성공을 뜻하지 않습니다. 공식 Redis, Qdrant와 검증된 vLLM 이미지는 실행 설정과 버전을 전달하며 불필요하게 재포장하지 않습니다. RunPod 모델과 이미지가 아직 미정이면 필수 값을 임의의 태그로 채우지 않습니다.

## 통합 검증

```sh
cp docker/v2/.env.example docker/v2/.env
# .env에 실제 테스트 환경 설정을 입력한다.
docker compose --env-file docker/v2/.env -f docker/v2/compose.yaml config --quiet
docker compose --env-file docker/v2/.env -f docker/v2/compose.yaml up -d
docker compose --env-file docker/v2/.env -f docker/v2/compose.yaml ps
```

이 Compose는 한 Docker 네트워크에서 연결을 확인하는 용도입니다. Fargate와 EC2를 나누어 배포할 때는 내부 DNS와 사설 주소로 `EMBEDDING_BASE_URL`, `QDRANT_URL`, `REDIS_URL`을 전달하고 이 Compose의 내부 hostname을 사용하지 않습니다. Kafka는 Cloud가 제공하는 별도 브로커이며, 여기에 브로커를 추가하거나 운영 토픽을 자동 생성하지 않습니다.

Cloud는 환경별 Kafka 주소와 여섯 토픽, API 인증값, S3 버킷과 읽기 권한, AWS 임베딩 주소 및 인증, RunPod의 `/v1` 주소와 모델 이름 및 인증을 전달합니다. Fargate task role과 EC2 IAM role은 S3 사본 읽기를 허용해야 합니다. RunPod가 접근할 수 있는 presigned URL인지 실제 이미지로 확인합니다. EBS 인스턴스 교체 시 볼륨 유지 및 재연결, Redis 영속화와 백업은 배포 전에 Cloud에서 결정할 항목입니다.

API 복제본마다 같은 설정을 사용하고 정리 실행기는 한 개만 유지합니다. `JOBS_STORY_CONCURRENCY=4`는 복제본 전체의 스토리 실행 한도, `GPU_GENERATION_INFLIGHT=8`은 검색과 스토리가 공유하는 생성 호출 한도입니다. API 복제본 수가 늘어나도 이 한도는 늘어나지 않습니다. 각 API에 최대 네 개의 스토리 자식 프로세스가 뜰 수 있으므로 CPU와 메모리를 배정할 때 고려합니다.

Cloud의 API 컨테이너 작업 정의에는 `stopTimeout=120`을 적용합니다. 이미지의 HTTP 종료 대기는 최대 35초이고, 이후 실행 중인 스토리를 기본 60초 동안 기다립니다. 스토리 종료 유예를 넘기면 작업을 중단하고 실패 결과를 Redis에 남깁니다. 종료 유예는 정상적인 축소와 배포 중 실패를 줄이는 장치이며, 강제 종료 방지나 작업 자동 재개를 보장하지 않습니다. 실패 결과는 다른 API가 발행하고 BE가 재시도를 결정합니다. 정리 실행기는 기존 20초 종료 유예를 유지합니다.

모델 서버의 healthcheck 유예와 GPU 메모리 설정은 실제 GPU에서 확인합니다. 모델 서버 준비 실패를 API liveness와 혼동하지 않습니다. Kafka 단절만으로 API를 재시작하지 않으며 미발행 결과는 Redis에 남습니다.

실행 중인 API의 `/health`, 진행률 GET과 검색 HTTP를 먼저 확인한 뒤 실제 정리와 스토리 요청 및 결과를 검증합니다. 스토리 사진 예산은 생성 서버의 `GENERATION_MAX_IMAGES` 이하여야 합니다. Redis 데이터 유실에 따른 작업 유실은 현재 v2의 알려진 한계입니다.
