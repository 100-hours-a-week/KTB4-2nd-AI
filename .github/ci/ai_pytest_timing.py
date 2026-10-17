"""CI-only pytest timing hooks; never retain parameters, fixture values or exception bodies."""

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import pytest

STATE = {}


def identity(nodeid):
    return hashlib.sha256(nodeid.encode()).hexdigest()


def public_name(nodeid):
    return nodeid.split("[", 1)[0]


def pytest_configure(config):
    STATE.clear()
    STATE.update(
        configuredAt=time.perf_counter(),
        inventory=[],
        cases={},
        collectors=[],
        fixtures=[],
        fixtureStack=[],
        collectionComplete=False,
        collectionFailures=0,
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection(session):
    start = time.perf_counter()
    try:
        return (yield)
    finally:
        STATE["collectionSeconds"] = time.perf_counter() - start
        STATE["collectionComplete"] = True


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_make_collect_report(collector):
    start = time.perf_counter()
    report = yield
    STATE["collectors"].append(
        {
            "id": identity(collector.nodeid),
            "name": public_name(collector.nodeid),
            "kind": type(collector).__name__,
            "seconds": time.perf_counter() - start,
            "outcome": report.outcome,
        }
    )
    return report


def pytest_collectreport(report):
    if report.failed:
        STATE["collectionFailures"] += 1


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    result = yield
    STATE["inventory"] = [identity(item.nodeid) for item in items]
    return result


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    frame = {"start": time.perf_counter(), "children": 0.0}
    stack = STATE["fixtureStack"]
    stack.append(frame)
    raised = True
    try:
        result = yield
        raised = False
        return result
    finally:
        elapsed = time.perf_counter() - frame["start"]
        stack.pop()
        if stack:
            stack[-1]["children"] += elapsed
        STATE["fixtures"].append(
            {
                "name": fixturedef.argname,
                "scope": fixturedef.scope,
                "base": public_name(fixturedef.baseid),
                "module": getattr(fixturedef.func, "__module__", "unknown"),
                "requestedBy": identity(request.node.nodeid),
                "setupSeconds": elapsed,
                "exclusiveSetupSeconds": max(0.0, elapsed - frame["children"]),
                "raised": raised,
            }
        )


def pytest_runtest_logreport(report):
    case = STATE["cases"].setdefault(
        identity(report.nodeid), {"name": public_name(report.nodeid), "phases": []}
    )
    case["phases"].append(
        {
            "phase": report.when,
            "outcome": report.outcome,
            "seconds": report.duration,
            "wasXfail": hasattr(report, "wasxfail"),
        }
    )


def case_outcome(case):
    phases = case["phases"]
    if any(row["outcome"] == "failed" for row in phases):
        return "failed"
    if any(row["wasXfail"] for row in phases):
        return "xfailed" if any(row["outcome"] == "skipped" for row in phases) else "xpassed"
    if any(row["outcome"] == "skipped" for row in phases):
        return "skipped"
    if any(row["phase"] == "call" and row["outcome"] == "passed" for row in phases):
        return "passed"
    return "incomplete"


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    result = yield
    inventory = Counter(STATE["inventory"])
    cases = STATE["cases"]
    issues = []
    if not STATE["collectionComplete"] or STATE["collectionFailures"]:
        issues.append("collection incomplete or failed")
    if set(inventory) != set(cases) or any(count != 1 for count in inventory.values()):
        issues.append("selected items and reported cases differ or contain duplicates")
    if session.testscollected != len(STATE["inventory"]):
        issues.append("collected count and final selected inventory differ")
    if any(not any(p["phase"] == "teardown" for p in c["phases"]) for c in cases.values()):
        issues.append("missing case teardown report")
    if STATE["fixtureStack"]:
        issues.append("unfinished fixture timing frame")
    data = {
        "complete": not issues,
        "issues": issues,
        "exitCode": int(session.exitstatus),
        "pytestVersion": pytest.__version__,
        "sessionSeconds": time.perf_counter() - STATE["configuredAt"],
        "collectionSeconds": STATE.get("collectionSeconds"),
        "collectionFailures": STATE["collectionFailures"],
        "collected": session.testscollected,
        "selectedIdentities": dict(sorted(inventory.items())),
        "outcomes": dict(Counter(case_outcome(case) for case in cases.values())),
        "cases": cases,
        "collectors": STATE["collectors"],
        "fixtures": STATE["fixtures"],
        "limits": [
            "Nested collector durations overlap; do not sum them as independent costs.",
            "Fixture setup hooks record actual instantiations, not cache hits or use counts.",
            "Exclusive setup removes dynamically nested fixture setup intervals only.",
            "Fixture factories called by a test are timed in call, not fixture setup.",
            "Teardown is measured per test phase, including final session cleanup; "
            "individual fixture teardown is not timed.",
            "Hook/import/report overhead is included in the wrapped command; "
            "its magnitude is not separately measured here.",
        ],
    }
    dest = Path(os.environ["AI_PYTEST_TIMING_OUTPUT"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2) + "\n")
    return result
