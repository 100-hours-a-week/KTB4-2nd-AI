import threading

import pytest

from app.pipeline.progress import ProgressReporter
from app.schemas.task import ProcessStep


def test_heartbeat_repeats_latest_count_and_stops_on_exit():
    repeated = threading.Event()
    events = []

    def callback(*args):
        events.append(args)
        if len(events) >= 2:
            repeated.set()

    with ProgressReporter(callback, 17, interval=0.01) as progress:
        progress.report(ProcessStep.EMBEDDING, 16)
        assert repeated.wait(3)
        progress.report(ProcessStep.CLOCK, 0)
    assert events[0] == events[1] == (ProcessStep.EMBEDDING, 16, 17)
    assert events[-1] == (ProcessStep.CLOCK, 0, 17)
    assert not progress._thread.is_alive()


def test_heartbeat_failure_reaches_caller_and_thread_stops():
    raised = threading.Event()
    calls = 0

    def callback(*args):
        nonlocal calls
        calls += 1
        if calls > 1:
            raised.set()
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        with ProgressReporter(callback, 1, interval=0.01) as progress:
            progress.report(ProcessStep.EMBEDDING, 0)
            assert raised.wait(3)
    assert not progress._thread.is_alive()


def test_stage_exception_is_preserved_and_thread_stops():
    with pytest.raises(ValueError, match="stage failed"):
        with ProgressReporter(lambda *_: None, 1, interval=0.01) as progress:
            progress.report(ProcessStep.CLOCK, 0)
            raise ValueError("stage failed")
    assert not progress._thread.is_alive()
