"""검색, 스토리와 정리 에이전트가 공유하는 VLM 전송 계약. 프롬프트와 출력 해석은 호출자 담당"""

from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    """image_urls의 다음 URL이 들어갈 자리. URL은 클라이언트가 전송할 때 채운다."""

    type: Literal["image"]


class ToolCall(TypedDict):
    """모델이 요청한 도구 호출. arguments는 모델이 만든 JSON 문자열이며 해석은 호출자가 한다."""

    id: str
    name: str
    arguments: str


class ChatMessage(TypedDict):
    """tool_calls는 도구를 호출한 assistant 메시지, tool_call_id는 tool 메시지에만 쓴다."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[TextPart | ImagePart]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]


@dataclass(frozen=True)
class ChatTurn:
    """도구를 줄 수 있는 대화의 한 차례 응답. 도구 호출이 없으면 content가 최종 텍스트다."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> ChatTurn:
        """모델 응답 한 차례를 ChatTurn으로 반환한다. JSON 파싱이나 기능별 검증은 수행하지 않는다.

        image_urls는 user 메시지에 첨부할 이미지이며 전달 순서를 유지한다.
        content 목록에 image 자리가 있으면 user 메시지들의 자리를 앞에서부터 차례로 채우고,
        자리 수는 image_urls 수와 같아야 한다. 자리가 없으면 마지막 user 메시지에 붙이며,
        그 메시지의 content는 문자열이어야 하고 이미지는 텍스트 뒤에 붙는다.
        텍스트 전용 호출은 빈 목록을 전달한다. max_tokens는 출력 토큰 상한이다.
        response_format은 호출자가 구성한 OpenAI 호환 출력 형식이다.
        값이 있으면 /v1/chat/completions 요청 본문에 그대로 전달하고 None이면 생략한다.
        tools는 OpenAI 호환 function 도구 정의 목록이며 그대로 전달한다. None이면 도구 없이
        요청하고, 응답은 stop으로 끝난 비어 있지 않은 텍스트여야 한다(tool_calls는 빈 목록).
        tool_choice는 auto, none, required 또는 특정 function 지정이며 tools가 있을 때만 쓰고
        None이면 auto다. 도구 실행과 arguments 해석, 대화 반복은 호출자가 담당한다.
        temperature는 0 이상 2 이하의 샘플링 온도다. None이면 요청에서 생략해 생성 서버의
        모델 권장값을 쓴다. 0이어도 vLLM은 배치 구성에 따라 결과가 달라질 수 있다.
        연결 실패는 VLMUnavailable, 시간 초과는 TimeoutError로 전달하고 취소는 전파한다.
        """
        ...
