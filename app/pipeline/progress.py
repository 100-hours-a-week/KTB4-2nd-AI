"""긴 모델 호출 중에도 마지막 진행 상태를 보고하는 요청별 보조 스레드."""

import threading
import time
from collections.abc import Callable

from app.schemas.task import ProcessStep


class ProgressReporter:
    def __init__(
        self, callback: Callable[[ProcessStep, int, int], None], total: int, interval: float
    ) -> None:
        self.callback = callback
        self.total = total
        self.interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._current: tuple[ProcessStep, int] | None = None
        self._last_sent = 0.0
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._heartbeat, name="pipeline-progress", daemon=True
        )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        if exc_type is None and self._error is not None:
            raise self._error

    def report(self, step: ProcessStep, done: int) -> None:
        with self._lock:
            if self._error is not None:
                raise self._error
            self._current = (step, done)
            self.callback(step, done, self.total)
            self._last_sent = time.monotonic()

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._stop.is_set():
                    return
                if self._current is None or time.monotonic() - self._last_sent < self.interval:
                    continue
                try:
                    # 완료 수를 임의로 늘리지 않고 현재 단계를 그대로 보고한다.
                    self.callback(*self._current, self.total)
                    self._last_sent = time.monotonic()
                except Exception as exc:
                    self._error = exc
                    self._stop.set()
