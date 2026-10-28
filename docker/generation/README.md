# RunPod 생성용 vLLM 설정

검색(질의 해석, 질의응답)과 스토리가 쓰는 RunPod 생성용 vLLM의 실행 설정. RunPod Pod 1개 기준, 서버 설정과 단독 호출 검증은 Kael, `app/infra/vllm_client.py`와 스토리 호출 연결은 Grant

## RunPod 구성(2026-10-28 검증)

| 항목 | 값 |
| --- | --- |
| 템플릿 | `yeodam-vllm-gemma4`(`la07dfg9qc`) |
| GPU | A40 48GB 1장, Secure Cloud |
| 호스트 CUDA | 13.0 이상, 이미지가 `cuda>=13.0` 요구, Pod 생성 시 CUDA 필터 필수 |
| 이미지 | `vllm/vllm-openai:v0.31.0` |
| 모델 | `google/gemma-4-31B-it-qat-w4a16-ct`, 모델과 토크나이저 revision `52f3f65bc7a02d555763bc923bd1d9094898219d` |
| 저장소 | 가중치는 컨테이너 디스크 50GB의 `HF_HOME=/root/.cache/huggingface`, 재시작마다 다시 받음(12에서 18초). torch.compile 캐시는 `/workspace` 볼륨의 `VLLM_CACHE_ROOT=/workspace/vllm`에 유지 |
| 인증 | `VLLM_API_KEY`, `HF_TOKEN`은 RunPod Secret `vllm_api_key`, `hf_token` 참조, 실제 키는 저장소에 두지 않음 |
| 주소 | `8000/http`, `https://<pod-id>-8000.proxy.runpod.net/v1`, ai-api의 `VLM_BASE_URL` |

가중치를 볼륨 대신 컨테이너 디스크에 두는 이유는 30GB 네트워크 볼륨에 기존 26B 가중치(16.01 GiB)와 31B 가중치(21.67 GiB)가 함께 들어가지 않기 때문

Container Start Command, `.env.example`과 `compose.yaml`의 값과 동일

```
google/gemma-4-31B-it-qat-w4a16-ct --revision 52f3f65bc7a02d555763bc923bd1d9094898219d --tokenizer-revision 52f3f65bc7a02d555763bc923bd1d9094898219d --host 0.0.0.0 --port 8000 --served-model-name gemma-4 --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 8 --limit-mm-per-prompt.image 30 --limit-mm-per-prompt.audio 0 --limit-mm-per-prompt.video 0 --scheduling-policy priority --safetensors-load-strategy prefetch --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --trust-request-mm-kwargs
```

## 인자 근거

| 인자 | 값 | 근거 |
| --- | --- | --- |
| `--max-model-len` | 32768 | 사진 30장 입력(장당 약 390토큰, 약 12,000토큰)과 여러 턴 도구 호출 대화, 출력 예산 |
| `--limit-mm-per-prompt.image` | 30 | 미분류 사진 재배치 작업의 요청당 사진 수, `STORY_IMAGE_BUDGET` 기본 10, 검색 질의응답은 5장 |
| `--limit-mm-per-prompt.audio`, `video` | 0 | 미사용, 기동 시 메모리 측정에서 제외 |
| `--gpu-memory-utilization` | 0.90 | 가중치 19.78 GiB, KV 캐시 19.3 GiB(89,055토큰) |
| `--max-num-seqs` | 8 | 스토리 최대 요청 약 5,800토큰 기준 KV에 8건 이상, 32768토큰 요청은 동시 2.72건 |
| `--scheduling-policy` | priority | 요청별 `priority` 사용 대비, 미전달 시 도착순과 동일 |
| `--safetensors-load-strategy` | prefetch | 가중치를 페이지 캐시에 미리 올려 적재 시간 단축, 볼륨(FUSE)에서 읽던 26B 기준 재시작 약 11분에서 약 3분 |
| `--enable-auto-tool-choice`, `--tool-call-parser` | gemma4 | 요청에 `tools`가 있을 때만 응답의 `tool_calls`로 분리 |
| `--reasoning-parser` | gemma4 | 요청에 `chat_template_kwargs.enable_thinking`이 true일 때만 사고 과정을 응답의 `reasoning`으로 분리, 미전달 요청은 기존과 동일 |
| `--trust-request-mm-kwargs` | 사용 | 요청별 `mm_processor_kwargs`(이미지 해상도 등) 허용, 호출 주체는 API 키를 가진 ai-api뿐 |

prefetch 사용 시 체크포인트(21.67 GiB)가 Pod RAM(20.86 GiB)의 90%를 넘는다는 경고가 나오나 10/28 기동은 3초 안에 정상 완료, 메모리 부족으로 기동이 실패하면 이 인자 제거

미적용 인자

- `--structured-outputs-config.disable_any_whitespace`: 스토리 블록이 1개에서 끝나 장소가 빠지는 실패 7/16(10/15 실측), 백엔드를 `xgrammar`로 명시하지 않으면 기동 실패
- `--chat-template examples/tool_chat_template_gemma4.jinja`: 서버의 모든 요청에 적용, 모델 기본 템플릿(Google 공식판, 7/9 공개)의 이전 판이라 도구 호출 반복과 턴 종료 수정 미반영, 기본 템플릿이 도구 호출 지원

## 실측(2026-10-28, A40)

| 항목 | 결과 |
| --- | --- |
| 첫 기동 | 3분 43초(가중치 다운로드 17.7초, 적재 4초, torch.compile 96.6초) |
| 재시작 | 약 2분(컴파일 캐시 사용), 인자 변경으로 컴파일 캐시를 새로 만들면 약 3분 30초 |
| 생성 속도 | 18.7에서 20.8 tok/s |
| 스토리 사진 9장 | 생성 29.1초(입력 3,518토큰, 출력 544토큰), 작업 전체 39.5초 |
| 질의 해석 | 4.0에서 4.3초(출력 83에서 84토큰), 시간 제한 5초 |
| 질의응답 사진 4장 | 5.9초(출력 45토큰), 시간 제한 10초 |
| 도구 호출, 사고 과정 분리 | 정상, 기존 구조화 출력도 `content`에 그대로 |

스토리와 검색은 스테이징 운영 경로의 값, presigned URL 다운로드 시간 포함

## 이전 구성 실측(2026-10-15, A5000, 26B)

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
- 도구 호출은 요청에 `tools`와 `tool_choice: "auto"`, 사고 과정은 `chat_template_kwargs: {"enable_thinking": true}`
- 31B 생성 속도 기준 검색 시간 제한 여유가 작음(질의 해석 약 1초), 동시 요청이 몰리면 질의 해석 대체 응답 발생 가능
- 프롬프트와 출력 검증은 검색, 스토리 코드에서 관리, 서버 설정에 넣지 않음

## 로컬 compose

GPU 장비에서 같은 설정을 확인할 때 사용, Pod 생성 자동화 아님

```
cp docker/generation/.env.example docker/generation/.env  # GENERATION_API_KEY 입력
docker compose --env-file docker/generation/.env -f docker/generation/compose.yaml up -d
```

AWS 임베딩 서버 설정은 `docker/embedding/`에서 별도 관리, 생성용 서버의 모델과 인증을 임베딩 설정과 혼용하지 않음
