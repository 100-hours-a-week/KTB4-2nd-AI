#!/usr/bin/env bash
set -Eeuo pipefail

# Deliberately print only reproducibility metadata, never the environment or secrets.
{
  echo '### CI execution context'
  echo "- Checkout SHA: $(git rev-parse HEAD)"
  echo "- PR head SHA: ${PR_HEAD_SHA}"
  echo "- Workflow blob: $(git hash-object .github/workflows/ai-pr-gate.yml)"
  echo "- Runner: ${RUNNER_OS}/${RUNNER_ARCH}"
  echo "- Logical CPUs: $(nproc)"
  awk '/^MemTotal:/ {printf "- Memory KiB: %s\n", $2}' /proc/meminfo
  echo "- Run/attempt: ${GITHUB_RUN_ID}/${GITHUB_RUN_ATTEMPT}"
} | tee -a "${GITHUB_STEP_SUMMARY}"
