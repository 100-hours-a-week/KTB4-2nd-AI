"""동기 파이프라인과 비동기 작업 상태 저장소를 연결한다."""

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeout

from app.core.errors import PipelineCancelled
from app.jobs.execution import JobReporter


async def run_sync(operation: Callable, reporter: JobReporter):
    loop = asyncio.get_running_loop()
    stopped = threading.Event()

    def call(method, *args):
        if stopped.is_set():
            raise PipelineCancelled()
        future = asyncio.run_coroutine_threadsafe(method(*args), loop)
        try:
            # Redis의 socket timeout보다 길게 기다리되 영구 대기는 피한다.
            return future.result(timeout=15)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError("작업 상태 저장소 응답 시간이 초과되었습니다.") from None

    def progress(step, done, total):
        call(reporter.report, str(step), done, total)

    def cancelled():
        if stopped.is_set():
            return True
        try:
            call(reporter.check_cancel)
        except PipelineCancelled:
            return True
        return False

    task = asyncio.create_task(asyncio.to_thread(operation, progress, cancelled))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        stopped.set()
        # 동기 코드가 자원을 닫을 때까지 이벤트 루프를 유지한다.
        # 프로세스 종료 기한은 상위 runner가 SIGTERM/SIGKILL로 보장한다.
        await asyncio.gather(task, return_exceptions=True)
        raise
