"""Exercise real pytest hooks without touching the service's tests or external services."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CI = Path(__file__).resolve().parents[1] / "ci"
sys.path.insert(0, str(CI))
import test_timing as timing  # noqa: E402


def execute(root, source, instrumented):
    root.mkdir()
    (root / "test_fixture.py").write_text(source)
    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(CI),
        "AI_PYTEST_TIMING_OUTPUT": str(root / "timing.json"),
    }
    command = [sys.executable, "-m", "pytest", "-q", "--junitxml=result.xml"]
    if instrumented:
        command += ["-p", "ai_pytest_timing"]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, check=False)
    data = json.loads((root / "timing.json").read_text()) if instrumented else None
    suites = ET.parse(root / "result.xml").getroot()
    identities = sorted(
        (
            case.get("classname"),
            case.get("name"),
            case.find("skipped") is not None,
            case.find("failure") is not None,
            case.find("error") is not None,
        )
        for case in suites.iter("testcase")
    )
    return result.returncode, identities, data


class PytestTimingTest(unittest.TestCase):
    def test_scope_outcomes_and_fixture_lifetimes_match_the_control(self):
        source = """
import time
import pytest
@pytest.fixture(scope="session")
def shared():
    time.sleep(0.002)
    yield object()
    time.sleep(0.002)
@pytest.fixture
def leaf():
    time.sleep(0.002)
    return 1
@pytest.fixture
def dynamic(request):
    return request.getfixturevalue("leaf")
@pytest.fixture
def factory():
    def build():
        time.sleep(0.004)
        return 2
    return build
@pytest.mark.parametrize("n", [1, 2], ids=["PRIVATE_PARAM_ONE", "PRIVATE_PARAM_TWO"])
def test_dynamic(n, shared, dynamic):
    assert n > 0 and dynamic == 1
@pytest.mark.skip(reason="PRIVATE_SKIP_REASON")
def test_skip():
    pass
@pytest.mark.xfail(reason="PRIVATE_XFAIL_REASON")
def test_expected_failure():
    assert False
def test_factory(factory):
    assert factory() == 2
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = execute(root / "control", source, False)
            observed = execute(root / "observed", source, True)
        self.assertEqual(control[:2], observed[:2])
        self.assertEqual(observed[0], 0)
        data = observed[2]
        self.assertTrue(data["complete"], data["issues"])
        self.assertEqual(data["collected"], 5)
        self.assertEqual(data["outcomes"], {"passed": 3, "skipped": 1, "xfailed": 1})
        self.assertEqual(len(data["selectedIdentities"]), 5)
        self.assertEqual(sum(f["name"] == "shared" for f in data["fixtures"]), 1)
        self.assertEqual(sum(f["name"] == "dynamic" for f in data["fixtures"]), 2)
        for frame in data["fixtures"]:
            self.assertGreaterEqual(frame["setupSeconds"], frame["exclusiveSetupSeconds"])
            self.assertGreaterEqual(frame["exclusiveSetupSeconds"], 0)
        dynamic = [f for f in data["fixtures"] if f["name"] == "dynamic"]
        self.assertTrue(all(f["setupSeconds"] > f["exclusiveSetupSeconds"] for f in dynamic))
        self.assertNotIn("PRIVATE_", json.dumps(data))
        self.assertGreater(data["collectionSeconds"], 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "command.json").write_text('{"exitCode": 0, "elapsedSeconds": 1.0}')
            (output / "pytest.json").write_text(json.dumps(data))
            with patch.object(timing, "OUTPUT", output), redirect_stdout(io.StringIO()):
                self.assertEqual(timing.report(), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["outcomes"], data["outcomes"])
            self.assertEqual(len(summary["slowCases"]), 5)
            self.assertGreater(summary["phaseSeconds"]["call"], 0)

    def test_setup_and_teardown_failures_keep_original_outcomes(self):
        source = """
import pytest
@pytest.fixture
def broken_setup():
    raise RuntimeError("PRIVATE_EXCEPTION")
@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("PRIVATE_EXCEPTION")
def test_one(broken_setup):
    pass
def test_two(broken_teardown):
    pass
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = execute(root / "control", source, False)
            observed = execute(root / "observed", source, True)
        self.assertEqual(control[:2], observed[:2])
        self.assertEqual(observed[0], 1)
        self.assertEqual(observed[2]["outcomes"], {"failed": 2})
        self.assertTrue(observed[2]["complete"])
        self.assertNotIn("PRIVATE_EXCEPTION", json.dumps(observed[2]))

    def test_collection_failure_is_not_reported_as_complete(self):
        source = 'raise RuntimeError("PRIVATE_COLLECTION_ERROR")\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = execute(root / "control", source, False)
            observed = execute(root / "observed", source, True)
        self.assertEqual(control[:2], observed[:2])
        self.assertEqual(observed[0], 2)
        self.assertFalse(observed[2]["complete"])
        self.assertGreater(observed[2]["collectionFailures"], 0)
        self.assertNotIn("PRIVATE_COLLECTION_ERROR", json.dumps(observed[2]))

    def test_wrapper_preserves_the_test_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.object(timing, "OUTPUT", output),
                patch.object(timing, "execution_context", return_value={}),
                patch.object(timing.subprocess, "run", return_value=SimpleNamespace(returncode=7)),
            ):
                self.assertEqual(timing.run(), 7)
                self.assertEqual(json.loads((output / "command.json").read_text())["exitCode"], 7)

    def test_missing_measurement_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "command.json").write_text('{"exitCode": 1}')
            with patch.object(timing, "OUTPUT", output):
                self.assertEqual(timing.report(), 1)
            self.assertFalse(json.loads((output / "summary.json").read_text())["complete"])

    def test_tests_not_started_does_not_invent_a_successful_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(timing, "OUTPUT", output):
                self.assertEqual(timing.report(), 1)
            self.assertFalse(json.loads((output / "summary.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
