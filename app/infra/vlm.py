"""검색과 스토리가 공유하는 VLM 전송 계약. 프롬프트와 출력 해석은 호출자가 담당한다."""

from typing import Any, Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class VLMUnavailable(Exception):
    """VLM 또는 답변용 이미지 접근 실패. 연결 코드에서 외부 예외를 변환한다."""


class VlmClient(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        image_urls: list[str],
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        *,
        temperature: float | None = None,
    ) -> str:
        """모델이 생성한 텍스트를 반환한다. JSON 파싱이나 기능별 검증은 수행하지 않는다.

        image_urls는 마지막 user 메시지에 첨부할 이미지이며 전달 순서를 유지한다.
        텍스트 전용 호출은 빈 목록을 전달한다. max_tokens는 출력 토큰 상한이다.
        response_format은 호출자가 구성한 OpenAI 호환 출력 형식이다.
        값이 있으면 /v1/chat/completions 요청 본문에 그대로 전달하고 None이면 생략한다.
        temperature는 0 이상 2 이하의 샘플링 온도다. None이면 요청에서 생략해 생성 서버의
        모델 권장값을 쓴다. 0이어도 vLLM은 배치 구성에 따라 결과가 달라질 수 있다.
        연결 실패는 VLMUnavailable, 시간 초과는 TimeoutError로 전달하고 취소는 전파한다.
        """
        ...
