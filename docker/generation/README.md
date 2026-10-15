# RunPod 생성용 vLLM 설정

검색(질의 해석, 질의응답)과 스토리가 쓰는 RunPod 생성용 vLLM의 실행 설정. RunPod Pod 1개 기준, 서버 설정과 단독 호출 검증은 Kael, `app/infra/vllm_client.py`와 스토리 호출 연결은 Grant

## RunPod 구성(2026-10-15 검증)

| 항목 | 값 |
| --- | --- |
| 템플릿 | `yeodam-vllm-gemma4`(`la07dfg9qc`) |
| GPU | RTX A5000 24GB 1장, Secure Cloud |
| 호스트 CUDA | 13.0 이상, 이미지가 `cuda>=13.0` 요구, Pod 생성 시 CUDA 필터 필수 |
| 이미지 | `vllm/vllm-openai:v0.31.0` |
| 모델 | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, 모델과 토크나이저 revision `180b2d35e35e4e48f6245367f3b41036f7cdabd6` |
| 저장소 | `/workspace` 볼륨 50GB, `HF_HOME=/workspace/huggingface`, `VLLM_CACHE_ROOT=/workspace/vllm`, 가중치와 torch.compile 캐시 유지 |
| 인증 | `VLLM_API_KEY`, `HF_TOKEN`은 RunPod Secret `vllm_api_key`, `hf_token` 참조, 실제 키는 저장소에 두지 않음 |
| 주소 | `8000/http`, `https://<pod-id>-8000.proxy.runpod.net/v1`, ai-api의 `VLM_BASE_URL` |

Container Start Command, `.env.example`과 `compose.yaml`의 값과 동일

```
cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit --revision 180b2d35e35e4e48f6245367f3b41036f7cdabd6 --tokenizer-revision 180b2d35e35e4e48f6245367f3b41036f7cdabd6 --host 0.0.0.0 --port 8000 --served-model-name gemma-4 --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 8 --limit-mm-per-prompt.image 10 --limit-mm-per-prompt.audio 0 --limit-mm-per-prompt.video 0 --scheduling-policy priority --safetensors-load-strategy prefetch
```

## 인자 근거

| 인자 | 값 | 근거 |
| --- | --- | --- |
| `--max-model-len` | 16384 | 사진 10장 스토리 입력 약 3,600에서 4,300토큰과 출력 예산, 8192면 장소 10개 요청이 입력과 `max_tokens` 합계 초과로 거부 |
| `--limit-mm-per-prompt.image` | 10 | `STORY_IMAGE_BUDGET` 10 이상, 검색 질의응답은 5장 |
| `--limit-mm-per-prompt.audio`, `video` | 0 | 미사용, 기동 시 메모리 측정에서 제외 |
| `--gpu-memory-utilization` | 0.90 | 가중치 16.64 GiB, KV 캐시 3.85 GiB(43,116토큰) |
| `--max-num-seqs` | 8 | 스토리 최대 요청 약 5,800토큰 기준 KV에 7에서 8건, 검색 요청은 약 2,000토큰 |
| `--scheduling-policy` | priority | 요청별 `priority` 사용 대비, 미전달 시 도착순과 동일 |
| `--safetensors-load-strategy` | prefetch | 볼륨(FUSE)에서 가중치를 읽는 속도가 느려 미지정 시 재시작 약 11분, 지정 시 약 3분 |

미적용 인자

- `--structured-outputs-config.disable_any_whitespace`: 스토리 블록이 1개에서 끝나 장소가 빠지는 실패 7/16(10/15 실측), 백엔드를 `xgrammar`로 명시하지 않으면 기동 실패

## 실측(2026-10-15, A5000)

| 항목 | 결과 |
| --- | --- |
| 첫 기동 | 약 6분(가중치 다운로드 19초, 적재 114초, torch.compile 103초) |
| 재시작 | 약 3분(prefetch, 컴파일 캐시 사용) |
| 스토리 사진 10장 | 1건 17.5초, 서로 다른 요청 동시 4건 건당 19.5에서 21.5초, 출력 766에서 941토큰 |
| 질의 해석 | 0.9에서 1.8초 |
| 질의응답 사진 5장 | 2.2에서 3.8초 |

사진은 base64로 전송한 값, 운영의 presigned URL 다운로드 시간은 미포함

## 호출 시 주의

- RunPod 프록시 요청당 100초 제한
- Python urllib 기본 User-Agent는 프록시가 403 처리, httpx 기본값은 통과
- 서버 기본 샘플링은 모델 `generation_config.json`(temperature 1.0, top_k 64, top_p 0.95)
- 프롬프트와 출력 검증은 검색, 스토리 코드에서 관리, 서버 설정에 넣지 않음

## 로컬 compose

GPU 장비에서 같은 설정을 확인할 때 사용, Pod 생성 자동화 아님

```
cp docker/generation/.env.example docker/generation/.env  # GENERATION_API_KEY 입력
docker compose --env-file docker/generation/.env -f docker/generation/compose.yaml up -d
```

AWS 임베딩 서버 설정은 `docker/embedding/`에서 별도 관리, 생성용 서버의 모델과 인증을 임베딩 설정과 혼용하지 않음
