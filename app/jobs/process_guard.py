"""작업의 이벤트 루프가 멈춰 있어도 부모 종료 시 자식 프로세스를 종료한다."""

import ctypes
import os
import signal
import sys
import threading
from contextlib import contextmanager


@contextmanager
def parent_guard(parent: int, interval: float = 0.2):
    if sys.platform == "linux":
        # 배포 환경에서는 GIL을 놓지 않는 native 연산 중에도 커널이 종료한다.
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        prctl.restype = ctypes.c_int
        if prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "부모 종료 신호 설정 실패")
    # 신호 등록 전에 부모가 종료된 경우도 실행을 시작하지 않는다.
    if os.getppid() != parent:
        raise ChildProcessError("실행기가 이미 종료되었습니다.")
    stopped = threading.Event()

    def watch_parent():
        while not stopped.wait(interval):
            if os.getppid() != parent:
                # asyncio 취소에 의존하면 블로킹 작업이 끝날 때까지 계속 실행된다.
                os.kill(os.getpid(), signal.SIGKILL)
                return

    monitor = threading.Thread(target=watch_parent, name="job-parent-guard", daemon=True)
    monitor.start()
    try:
        yield
    finally:
        stopped.set()
        monitor.join(timeout=1)
