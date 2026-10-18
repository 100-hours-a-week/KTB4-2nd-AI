"""Compare real pytest selection/outcomes and reject incomplete shard evidence."""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

CI = Path(__file__).resolve().parents[1] / "ci"
sys.path.insert(0, str(CI))
import test_shards as shards  # noqa: E402
import test_timing as timing  # noqa: E402


class TestShardsTest(unittest.TestCase):
    def test_modules_stay_together_and_unknown_tests_are_included(self):
        records = [
            {"id": shards.digest(name), "module": module}
            for name, module in [("a1", "a.py"), ("a2", "a.py"), ("b", "b.py"), ("new", "new.py")]
        ]
        plan = shards.make_plan(records, 2, {"a.py": 8, "b.py": 3})
        self.assertEqual(plan, shards.make_plan(list(reversed(records)), 2, {"b.py": 3, "a.py": 8}))
        self.assertEqual(set(plan["assignments"]), {row["id"] for row in records})
        self.assertEqual(
            plan["assignments"][records[0]["id"]], plan["assignments"][records[1]["id"]]
        )
        self.assertEqual(set(plan["assignments"].values()), {1, 2})
        self.assertEqual(shards.make_plan(records, 1, {})["inventory"], plan["inventory"])

    def test_invalid_partition_inputs_fail(self):
        item = {"id": "one", "module": "test_one.py"}
        for records, count, hints in [
            ([item, item], 2, {}),
            ([item], 3, {}),
            ([item], 1, {"test_one.py": float("nan")}),
            ([item], 1, {"test_one.py": -1}),
            ([{"id": "one", "module": "../outside.py"}], 1, {}),
        ]:
            with (
                self.subTest(records=records, count=count, hints=hints),
                self.assertRaises(ValueError),
            ):
                shards.make_plan(records, count, hints)
        for count, shard in [(0, 1), (2, 0), (2, 3)]:
            with patch.dict(os.environ, AI_TEST_SHARDS=str(count), AI_TEST_SHARD=str(shard)):
                with self.assertRaises(ValueError):
                    shards.settings()

    def test_real_pytest_union_matches_control_and_bad_reports_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / "tests"
            tests.mkdir()
            (root / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
            (tests / "conftest.py").write_text(
                'import pytest\n@pytest.fixture(scope="module")\ndef state():\n    return []\n'
            )
            (tests / "test_a.py").write_text(
                """
import pytest
@pytest.mark.parametrize("n", [1, 2], ids=["PRIVATE_FIRST", "PRIVATE_SECOND"])
def test_order(n, state):
    assert len(state) == n - 1
    state.append(n)
"""
            )
            (tests / "test_b.py").write_text(
                """
import pytest
@pytest.mark.skip(reason="PRIVATE_SKIP")
def test_skip():
    pass
@pytest.mark.xfail(reason="PRIVATE_XFAIL")
def test_expected_failure():
    assert False
"""
            )
            (tests / "test_new.py").write_text(
                """
import pytest
class TestNested:
    @pytest.mark.parametrize("n", [1, 2])
    def test_inside(self, n):
        assert n > 0
def test_new_module_without_hint():
    assert True
"""
            )
            hints = {"tests/test_a.py": 8, "tests/test_b.py": 4}
            hint_path = root / "hints.json"
            hint_path.write_text(json.dumps({"modules": hints}))

            def execute(output, count, shard, instrumented=True):
                output.mkdir(parents=True)
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    f"--junitxml={output / 'result.xml'}",
                ]
                env = {
                    **os.environ,
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "PYTHONPATH": str(CI),
                    "AI_TEST_SHARDS": str(count),
                    "AI_TEST_SHARD": str(shard),
                    "AI_TEST_HINTS": str(hint_path),
                    "AI_PYTEST_TIMING_OUTPUT": str(output / "pytest.json"),
                }
                if instrumented:
                    command += ["-p", "ai_pytest_timing"]
                run = subprocess.run(command, cwd=root, env=env, capture_output=True, check=False)
                self.assertEqual(run.returncode, 0, run.stdout.decode() + run.stderr.decode())
                xml = ET.parse(output / "result.xml")
                identities = [
                    (case.get("classname"), case.get("name"), case.find("skipped") is not None)
                    for case in xml.iter("testcase")
                ]
                if instrumented:
                    metadata = {
                        "checkoutSha": "fixed-source",
                        "PR_HEAD_SHA": "fixed-head",
                        "GITHUB_RUN_ID": "fixture",
                        "GITHUB_RUN_ATTEMPT": "1",
                        "fileHashes": {"fixture": "fixed"},
                        "AI_TEST_SHARDS": str(count),
                        "AI_TEST_SHARD": str(shard),
                        "pythonVersion": "fixture",
                        "elapsedSeconds": 1.0,
                        "exitCode": 0,
                    }
                    (output / "command.json").write_text(json.dumps(metadata))
                    with patch.object(timing, "OUTPUT", output), redirect_stdout(io.StringIO()):
                        self.assertEqual(timing.report(), 0)
                return identities

            control = execute(root / "control", 1, 1, False)
            single = execute(root / "single/1", 1, 1)
            combined = execute(root / "sharded/1", 2, 1) + execute(root / "sharded/2", 2, 2)
            self.assertEqual(sorted(control), sorted(single))
            self.assertEqual(sorted(control), sorted(combined))
            self.assertEqual(len(control), 7)
            full = shards.aggregate(root / "single", 1, hints)
            observed = shards.aggregate(root / "sharded", 2, hints)
            self.assertEqual(full["selectedIdentities"], observed["selectedIdentities"])
            self.assertEqual(full["caseOutcomes"], observed["caseOutcomes"])
            self.assertEqual(observed["outcomes"], {"passed": 5, "skipped": 1, "xfailed": 1})
            self.assertNotIn("PRIVATE_", json.dumps(observed))

            def change(path, field, value):
                data = json.loads(path.read_text())
                data[field] = value
                path.write_text(json.dumps(data))

            mutations = {
                "missing": lambda p: (p / "2/pytest.json").unlink(),
                "extra": lambda p: shutil.copytree(p / "1", p / "extra"),
                "foreign_source": lambda p: change(p / "2/command.json", "checkoutSha", "other"),
                "wrong_shard": lambda p: change(p / "2/command.json", "AI_TEST_SHARD", "1"),
                "failed": lambda p: change(p / "2/command.json", "exitCode", 1),
                "lost_test": lambda p: change(p / "2/pytest.json", "selectedIdentities", {}),
                "bad_count": lambda p: change(p / "2/pytest.json", "collected", 99),
            }
            for name, mutate in mutations.items():
                with self.subTest(rejected=name):
                    candidate = root / ("bad-" + name)
                    shutil.copytree(root / "sharded", candidate)
                    mutate(candidate)
                    with self.assertRaises(ValueError):
                        shards.aggregate(candidate, 2, hints)


if __name__ == "__main__":
    unittest.main()
