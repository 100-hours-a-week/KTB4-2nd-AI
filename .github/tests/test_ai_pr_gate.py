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
