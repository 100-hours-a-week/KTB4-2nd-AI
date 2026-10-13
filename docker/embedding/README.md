# 임베딩 vLLM 서버

검색과 사진 정리가 공유하는 임베딩 서버의 실행 설정. 앱 코드는 없고 vLLM이 서버 역할

- 모델: `Qwen/Qwen3-VL-Embedding-2B`, 2048차원
- 실행 값의 기준은 `.env.example`과 `compose.yaml`, AWS 작업 정의는 이 값을 옮겨 작성
- 생성용 서버(RunPod)는 `docker/generation/`에서 별도 관리

## 실행 값

| 항목 | 값 |
| --- | --- |
| 이미지 | `vllm/vllm-openai:v0.31.0` |
| 모델 revision | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` |
| 서빙 이름 | `qwen3-vl-embedding-2b`, 요청 본문의 `model` 필드 값 |
| 실행 인자 | `--runner pooling --trust-remote-code --dtype bfloat16 --max-model-len 4096` |
| 컨테이너 포트 | 8000 |

```sh
cp docker/embedding/.env.example docker/embedding/.env   # EMBEDDING_API_KEY 입력
docker compose --env-file docker/embedding/.env -f docker/embedding/compose.yaml up -d
```

## 배포 요구 사항

| 항목 | 값 |
| --- | --- |
| GPU | 1장, L4 24GB(g6.xlarge) |
| 메모리 | 모델 4.24 GiB, 나머지 VRAM은 vLLM이 KV 캐시로 예약(기본 90%) |
| 기동 시간 | 약 95초(재기동도 같음, 대부분 가중치 적재와 엔진 준비) |
| 헬스 체크 | `GET /health`, 기동 유예 180초 이상. 짧으면 기동 중인 컨테이너를 실패로 보고 재시작 반복 |
| 환경 변수 | `VLLM_API_KEY` 필수, `HF_TOKEN` 선택 |
| 볼륨 | `/root/.cache/huggingface` 유지. 없으면 재기동마다 모델 다운로드 |
| 접근 | 정리 실행기(같은 EC2)와 ai-api(Fargate)만, 외부 비공개 |

## 요청 규칙

클라이언트(`app/embedding/`)가 지킬 규칙

| 규칙 | 이유 |
| --- | --- |
| 텍스트와 사진 모두 `POST /v1/embeddings`에 `messages` 형식, system instruction 포함 | 모델 카드의 대화 템플릿과 같은 입력 |
| `input` 형식 사용 금지 | 같은 문장도 `messages`와 코사인 0.21로 다른 벡터, 사진과 비교 불가 |
| 사진은 base64 data URI로 요청당 1장 | URL 전달과 속도 차이 없음, 임베딩 서버에 S3 권한 불필요 |
| 사진 instruction `Represent the user's input.` | 유무에 따른 차이 미미(코사인 0.9995), 기본값 고정 |
| 질의 instruction `Retrieve images relevant to the user's query.` | 1차 값, 검색 평가에서 확정 |
| 정리는 16장 묶음 단위로 순차, 묶음 안 16장은 동시 요청 | 검색 텍스트 앞에 쌓이는 사진 수를 묶음 하나로 제한 |

```json
{
  "model": "qwen3-vl-embedding-2b",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}
  ],
  "encoding_format": "float"
}
```

응답 `data[0].embedding`은 2048차원, L2 정규화 완료

## 검증 결과

2026-10-13, Colab L4, `pip`로 설치한 vLLM 0.31.0, 평가셋 200장(긴 변 1024px JPEG)

| 항목 | 결과 |
| --- | --- |
| 형식 | 2048차원, L2 노름 오차 최대 1.19e-07 |
| 200장 | base64 20.9초, URL 21.8초. 16장 묶음 평균 1.60초, 최대 2.05초. 기존 transformers 구현 버스트 27.7초 |
| 상식 점검 | 질의 5개의 상위 5장 적중 4, 3, 4, 0, 5. 같은 장소 사진 유사도 평균 0.497, 다른 장소 0.365 |
| 흔들림 | 같은 사진도 묶음 구성에 따라 코사인 최소 0.994, 임계값에 여유 필요 |

- Docker 이미지 자체의 기동은 AWS 배포 때 확인
- 클럭 미고정 버스트 측정이라 지속 부하에서는 더 느릴 수 있음
