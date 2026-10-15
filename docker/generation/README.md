# RunPod 생성용 vLLM 설정

검색과 스토리가 사용하는 RunPod 생성용 vLLM의 실행 설정을 관리할 경로입니다. Kael이 서버 실행 설정과 단독 호출을 검증하고, Grant는 `app/infra/vllm_client.py`와 스토리 호출을 연결합니다.

검증할 모델과 이미지 및 인증을 `.env`에 입력한 뒤 `docker compose --env-file docker/generation/.env -f docker/generation/compose.yaml up -d`로 실행합니다. RunPod의 Pod 생성 화면에는 같은 이미지, 명령과 환경 변수를 전달합니다. 이 파일은 Pod 생성 자동화가 아니며 RunPod는 v2에서 1 Pod를 사용하는 기준입니다.

Kael은 `.env.example`의 항목을 참고해 검증한 이미지와 모델 revision, served model name 및 인증을 실행 설정에 반영합니다. Compose가 변수들을 vLLM 실행 인자로 전달합니다. `GENERATION_MAX_IMAGES`는 `STORY_IMAGE_BUDGET` 이상으로 설정하며 초기 10장은 실제 모델과 GPU에서 메모리를 확인할 값입니다. `GENERATION_MAX_MODEL_LEN`도 사진 입력과 출력 토큰 예산을 포함해 검증합니다. 실제 인증키는 저장소에 추가하지 않습니다.

서버 준비 후 Grant에게 `/v1`까지 포함한 호출 주소와 served model name, 인증키를 설정한 위치를 전달합니다. 텍스트, 이미지와 `response_format` 요청을 실제 모델에서 확인해야 합니다. 프롬프트와 출력 검증은 검색 및 스토리 코드에서 관리하며 서버 설정에 넣지 않습니다.

AWS의 임베딩 서버 설정은 `docker/embedding/`에서 별도로 관리합니다. 생성용 서버의 모델과 인증을 임베딩 설정과 혼용하지 않습니다.

실행 인자는 [vLLM serve 문서](https://docs.vllm.ai/en/latest/cli/serve/)를 기준으로 작성했으며 선택한 이미지 버전에서 인자 지원 여부와 구조화 출력을 확인해야 합니다.
