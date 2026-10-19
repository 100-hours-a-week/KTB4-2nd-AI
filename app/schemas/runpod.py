"""RunPod 생성 서버 상태 응답, GET /runpod/status"""

from pydantic import BaseModel, Field


class RunpodStatusResponse(BaseModel):
    running: bool | None = Field(
        description="RunPod API의 Pod desiredStatus가 RUNNING이면 true, 읽지 못하면 null"
    )
    ready: bool = Field(description="생성 vLLM이 설정한 모델로 요청을 받을 수 있으면 true")
