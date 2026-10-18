"""Run the original pytest scope with numeric timing evidence and preserve its exit code."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

OUTPUT = Path("build/ci-timing")


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def execution_context():
    files = [
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".github/workflows/ai-pr-gate.yml",
        ".github/ci/ai_pytest_timing.py",
        ".github/ci/test_timing.py",
    ]
    data = {
        key: os.environ.get(key)
        for key in (
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "PR_HEAD_SHA",
            "RUNNER_OS",
            "RUNNER_ARCH",
            "ImageOS",
            "ImageVersion",
            "UV_VERSION",
            "UV_CACHE_HIT",
        )
    }
    data.update(
        checkoutSha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        pythonVersion=platform.python_version(),
        logicalCpu=os.cpu_count(),
        fileHashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},
    )
    if Path("/proc/meminfo").exists():
        data["memoryTotalKiB"] = int(
            re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)
        )
    if Path("/proc/cpuinfo").exists():
        model = re.search(r"model name\s*:\s*([^\n]+)", Path("/proc/cpuinfo").read_text())
        data["cpuModel"] = model.group(1) if model else None
    return data


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT / "pytest.json"
    report_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env["AI_PYTEST_TIMING_OUTPUT"] = str(report_path.resolve())
    plugin_path = str(Path(".github/ci").resolve())
    env["PYTHONPATH"] = plugin_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    command = [sys.executable, "-m", "pytest", "-p", "ai_pytest_timing"]
    metadata = execution_context()
    metadata.update(command=command, startedAt=datetime.now(UTC).isoformat())
    save(OUTPUT / "command.json", metadata)
    start = time.perf_counter()
    result = subprocess.run(command, env=env, check=False)
    metadata.update(
        elapsedSeconds=time.perf_counter() - start,
        exitCode=result.returncode,
        endedAt=datetime.now(UTC).isoformat(),
    )
    save(OUTPUT / "command.json", metadata)
    return result.returncode


def report():
    if not (OUTPUT / "command.json").exists():
        save(OUTPUT / "summary.json", {"complete": False, "issues": ["missing test command"]})
        return 1
    command = json.loads((OUTPUT / "command.json").read_text())
    path = OUTPUT / "pytest.json"
    if not path.exists():
        save(OUTPUT / "summary.json", {"complete": False, "issues": ["missing pytest report"]})
        return 1
    data = json.loads(path.read_text())
    phases = Counter()
    slow = []
    for key, case in data["cases"].items():
        duration = sum(row["seconds"] for row in case["phases"])
        slow.append({"id": key, "name": case["name"], "seconds": duration})
        for row in case["phases"]:
            phases[row["phase"]] += row["seconds"]
    fixtures = defaultdict(lambda: {"calls": 0, "setupSeconds": 0.0, "exclusiveSetupSeconds": 0.0})
    for row in data["fixtures"]:
        group = fixtures[(row["base"], row["name"], row["scope"], row["module"])]
        group["calls"] += 1
        group["setupSeconds"] += row["setupSeconds"]
        group["exclusiveSetupSeconds"] += row["exclusiveSetupSeconds"]
    summary = {
        "complete": data["complete"] and data["exitCode"] == command.get("exitCode"),
        "issues": data["issues"],
        "exitCode": command.get("exitCode"),
        "commandSeconds": command.get("elapsedSeconds"),
        "sessionSeconds": data["sessionSeconds"],
        "collectionSeconds": data["collectionSeconds"],
        "collected": data["collected"],
        "outcomes": data["outcomes"],
        "phaseSeconds": dict(phases),
        "slowCases": sorted(slow, key=lambda row: row["seconds"], reverse=True),
        "fixtures": sorted(
            [
                {"base": key[0], "name": key[1], "scope": key[2], "module": key[3], **value}
                for key, value in fixtures.items()
            ],
            key=lambda row: row["exclusiveSetupSeconds"],
            reverse=True,
        ),
        "limits": data["limits"],
    }
    save(OUTPUT / "summary.json", summary)
    lines = [
        "## AI pytest timing",
        "",
        f"- Command: {summary['commandSeconds']} seconds; exit: {summary['exitCode']}",
        f"- Collected: {summary['collected']}; outcomes: {summary['outcomes']}; "
        f"complete: {summary['complete']}",
        f"- Collection: {summary['collectionSeconds']} seconds; phases: {summary['phaseSeconds']}",
        "",
        "| Test (parameter values omitted) | Setup + call + teardown seconds |",
        "| --- | ---: |",
    ]
    lines += [f"| {row['name']} | {row['seconds']:.3f} |" for row in summary["slowCases"][:20]]
    lines += [
        "",
        "| Fixture | Scope | Instantiations | Exclusive setup seconds |",
        "| --- | --- | ---: | ---: |",
    ]
    lines += [
        f"| {row['base']}::{row['name']} | {row['scope']} | {row['calls']} | "
        f"{row['exclusiveSetupSeconds']:.3f} |"
        for row in summary["fixtures"][:20]
    ]
    lines += ["", *["- " + value for value in summary["limits"]]]
    markdown = "\n".join(lines) + "\n"
    (OUTPUT / "summary.md").write_text(markdown)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(markdown)
    print(markdown)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "report"])
    args = parser.parse_args()
    sys.exit(run() if args.action == "run" else report())
