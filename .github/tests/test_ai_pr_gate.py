"""Protect the required PR gate when individual checks run in parallel."""

import os
import subprocess
import unittest
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "ai-pr-gate.yml"


class AIPRGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # BaseLoader preserves literal strings such as 'on', 'false', and expressions.
        cls.workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
        cls.jobs = cls.workflow["jobs"]

    def test_required_gate_waits_for_every_independent_check(self):
        gate = self.jobs["ai-ci-gate"]
        checks = set(self.jobs) - {"ai-ci-gate"}
        self.assertEqual(set(gate["needs"]), checks)
        self.assertEqual(gate["if"], "always()")
        for name in checks:
            self.assertNotIn("needs", self.jobs[name])

    def test_only_all_success_passes_the_actual_gate_script(self):
        steps = self.jobs["ai-ci-gate"]["steps"]
        require = next(step for step in steps if step["name"] == "Require tests and both images")
        base = {"TESTS_RESULT": "success", "V1_RESULT": "success", "V2_RESULT": "success"}

        def execute(results):
            return subprocess.run(
                ["bash", "-euo", "pipefail", "-c", require["run"]],
                env={**os.environ, **results},
                capture_output=True,
                check=False,
            ).returncode

        self.assertEqual(execute(base), 0)
        for key in base:
            for outcome in ("failure", "cancelled", "skipped", ""):
                with self.subTest(check=key, outcome=outcome):
                    self.assertNotEqual(execute({**base, key: outcome}), 0)

    def test_two_test_runners_require_complete_coverage_and_one_model_check(self):
        tests = self.jobs["tests"]
        self.assertEqual(tests["strategy"]["matrix"]["shard"], ["1", "2"])
        self.assertEqual(tests["strategy"]["fail-fast"], "false")
        steps = {step["name"]: step for step in tests["steps"]}
        self.assertEqual(steps["Test"]["env"]["AI_TEST_SHARDS"], "2")
        self.assertEqual(steps["Test"]["env"]["AI_TEST_SHARD"], "${{ matrix.shard }}")
        for name in (
            "Compute model cache key",
            "Cache model weights",
            "Fetch model if cache missed",
            "Check model load and one-image inference",
        ):
            self.assertEqual(
                steps[name]["if"],
                "steps.changes.outputs.model_runtime == 'true' && matrix.shard == 1",
            )
        gate = {step["name"]: step for step in self.jobs["ai-ci-gate"]["steps"]}
        self.assertIn("--count 2", gate["Verify complete test coverage"]["run"])
        self.assertNotIn("if", gate["Verify complete test coverage"])
        self.assertEqual(
            gate["Upload verified test coverage"]["with"]["if-no-files-found"], "error"
        )
        for name, job in self.jobs.items():
            if name != "ai-ci-gate":
                self.assertNotIn("if", job)

    def test_image_checks_cannot_publish_or_dispatch(self):
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        for job in self.jobs.values():
            for step in job.get("steps", []):
                if "docker/build-push-action@" in step.get("uses", ""):
                    self.assertEqual(step["with"]["push"], "false")
                    self.assertNotIn("tags", step["with"])
                self.assertNotIn("gh api", step.get("run", ""))
                self.assertNotIn("CLOUD_RELEASE_TOKEN", str(step))


if __name__ == "__main__":
    unittest.main()
