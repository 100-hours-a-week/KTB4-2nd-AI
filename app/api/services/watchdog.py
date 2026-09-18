"""감시 태스크. 요청이 없어도 시간이 지나면 해야 하는 일

lifespan에서 asyncio.create_task로 띄움. 스레드가 아닌 이유는 task_service의 asyncio.Event가
루프 스레드에서만 안전해서. 5초마다 dict를 훑는 밀리초짜리라 루프를 막지 않음
"""

import asyncio

from app.api.services.task_service import TaskService
from app.core.logging import get_logger

log = get_logger(__name__)


async def watchdog_loop(service: TaskService, interval: float) -> None:
    """interval마다 tick(). 예외는 로그만 남기고 계속, 한 번 터졌다고 감시가 죽으면 안 됨"""
    while True:
        await asyncio.sleep(interval)
        try:
            await service.tick()
        except Exception:
            log.exception("watchdog tick 실패")
