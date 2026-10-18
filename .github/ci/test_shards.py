"""Keep pytest modules together and verify every discovered test across CI shards."""

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from statistics import median

HINTS = Path(__file__).with_name("test-duration-hints.json")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def settings():
    count = int(os.environ.get("AI_TEST_SHARDS", "1"))
    shard = int(os.environ.get("AI_TEST_SHARD", "1"))
    if count not in (1, 2) or shard not in range(1, count + 1):
        raise ValueError("Expected one or two shards and an in-range shard number")
    return count, shard


def load_hints():
    path = Path(os.environ.get("AI_TEST_HINTS", HINTS))
    return json.loads(path.read_text())["modules"]


def make_plan(records, count, hints):
    if count not in (1, 2):
        raise ValueError("Expected one or two shards")
    records = sorted(records, key=lambda row: row["id"])
    identities = [row["id"] for row in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate test identity")
    for row in records:
        if not row["id"] or not row["module"]:
            raise ValueError("Missing test identity or module")
        path = PurePosixPath(row["module"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Expected repository-relative module")
    weights = {name: float(value) for name, value in hints.items()}
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("Duration hints must be positive finite numbers")
    default = median(weights.values()) if weights else 1.0
    modules = {row["module"] for row in records}
    loads = [0.0] * count
    owners = {}
    for module in sorted(modules, key=lambda name: (-weights.get(name, default), name)):
        index = min(range(count), key=lambda candidate: (loads[candidate], candidate))
        owners[module] = index + 1
        loads[index] += weights.get(module, default)
    plan = {
        "schema": 1,
        "count": count,
        "inventory": records,
        "inventoryHash": digest(records),
        "hintsHash": digest(weights),
        "assignments": {row["id"]: owners[row["module"]] for row in records},
        "estimatedSeconds": [round(value, 6) for value in loads],
    }
    plan["planHash"] = digest(plan)
    return plan


def phase_outcomes(data):
    return {
        key: [[p["phase"], p["outcome"], p["wasXfail"]] for p in case["phases"]]
        for key, case in data["cases"].items()
    }


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


def aggregate(root, count, hints):
    if count not in (1, 2):
        raise ValueError("Expected one or two shards")
    paths = sorted(Path(root).rglob("pytest.json"))
    if len(paths) != count:
        raise ValueError("Missing or extra shard reports")
    expected_plan = fixed = None
    observed_shards = set()
    identities = Counter()
    outcomes = Counter()
    cases = {}
    shards = []
    for path in paths:
        data = json.loads(path.read_text())
        command = json.loads(path.with_name("command.json").read_text())
        summary = json.loads(path.with_name("summary.json").read_text())
        if (
            not data["complete"]
            or not summary["complete"]
            or command["exitCode"] != 0
            or data["exitCode"] != 0
        ):
            raise ValueError("Incomplete or failed test shard")
        partition = data["partition"]
        plan, shard = partition["plan"], partition["shard"]
        if shard in observed_shards or shard not in range(1, count + 1):
            raise ValueError("Duplicate or invalid shard number")
        if int(command["AI_TEST_SHARDS"]) != count or int(command["AI_TEST_SHARD"]) != shard:
            raise ValueError("Command and partition settings differ")
        recomputed = make_plan(plan["inventory"], count, hints)
        if plan != recomputed:
            raise ValueError("Partition does not match discovered tests and current hints")
        source = {
            key: command[key]
            for key in (
                "checkoutSha",
                "PR_HEAD_SHA",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "fileHashes",
            )
        }
        if any(not value for value in source.values()):
            raise ValueError("Missing execution identity")
        if expected_plan is None:
            expected_plan, fixed = plan, source
        elif expected_plan != plan or fixed != source:
            raise ValueError("Shard discovery, assignment or execution source differs")
        selected = {key: 1 for key, owner in plan["assignments"].items() if owner == shard}
        if not selected or data["selectedIdentities"] != selected:
            raise ValueError("Selected tests differ from the assigned partition")
        if set(data["cases"]) != set(selected) or data["collected"] != len(selected):
            raise ValueError("Assigned tests and actual cases differ")
        if summary["outcomes"] != data["outcomes"] or summary["collected"] != len(selected):
            raise ValueError("Summary and raw test inventory differ")
        actual_outcomes = dict(Counter(case_outcome(case) for case in data["cases"].values()))
        if actual_outcomes != data["outcomes"]:
            raise ValueError("Case phases and reported outcomes differ")
        if data["outcomes"].get("failed", 0) or data["outcomes"].get("incomplete", 0):
            raise ValueError("Failed or incomplete test outcomes")
        identities.update(selected)
        outcomes.update(data["outcomes"])
        cases.update(phase_outcomes(data))
        observed_shards.add(shard)
        shards.append(
            {
                "shard": shard,
                "tests": len(selected),
                "commandSeconds": command["elapsedSeconds"],
                "collectionSeconds": data["collectionSeconds"],
                "pythonVersion": command["pythonVersion"],
                "pytestVersion": data["pytestVersion"],
                "uvCacheHit": command.get("UV_CACHE_HIT"),
                "runnerImage": command.get("ImageVersion"),
                "cpuModel": command.get("cpuModel"),
            }
        )
    expected = {row["id"]: 1 for row in expected_plan["inventory"]}
    if dict(identities) != expected or observed_shards != set(range(1, count + 1)):
        raise ValueError("Missing or duplicate tests in combined shards")
    return {
        "complete": True,
        "source": fixed,
        "count": count,
        "planHash": expected_plan["planHash"],
        "inventoryHash": expected_plan["inventoryHash"],
        "selectedIdentities": dict(sorted(identities.items())),
        "caseOutcomes": dict(sorted(cases.items())),
        "collected": len(identities),
        "outcomes": dict(outcomes),
        "shards": sorted(shards, key=lambda row: row["shard"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["aggregate"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--count", type=int, choices=[1, 2], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate(args.root, args.count, load_hints())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {result['collected']} tests across {result['count']} shards")
