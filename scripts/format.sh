#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

usage() {
  cat <<'USAGE'
Usage: scripts/format.sh [--check]

Run Black and isort with the repository configuration across the core TNFR packages.

Options:
  --check        Verify formatting without modifying files.
  -h, --help     Show this help message and exit.
USAGE
}

CHECK_MODE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK_MODE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

declare -a TARGETS=("src" "tests" "scripts" "benchmarks")
declare -a EXISTING_TARGETS=()
for target in "${TARGETS[@]}"; do
  if [[ -d "$target" ]]; then
    EXISTING_TARGETS+=("$target")
  fi
done

if [[ ${#EXISTING_TARGETS[@]} -eq 0 ]]; then
  echo "No formatting targets found." >&2
  exit 0
fi

USE_POETRY=0
if command -v poetry >/dev/null 2>&1; then
  if [[ -f "poetry.lock" ]]; then
    USE_POETRY=1
  elif [[ -f "pyproject.toml" ]]; then
    if grep -q "^\\[tool\\.poetry\\]" pyproject.toml; then
      USE_POETRY=1
    fi
  fi
fi

if [[ $USE_POETRY -eq 1 ]]; then
  BLACK_CMD=(poetry run black)
  ISORT_CMD=(poetry run isort)
else
  BLACK_CMD=(python -m black)
  ISORT_CMD=(python -m isort)
fi

if [[ $CHECK_MODE -eq 1 ]]; then
  BLACK_CMD+=("--check")
  ISORT_CMD+=("--check")
fi

printf 'Running %s on %s\n' "${BLACK_CMD[*]}" "${EXISTING_TARGETS[*]}"
"${BLACK_CMD[@]}" "${EXISTING_TARGETS[@]}"

printf 'Running %s on %s\n' "${ISORT_CMD[*]}" "${EXISTING_TARGETS[*]}"
"${ISORT_CMD[@]}" "${EXISTING_TARGETS[@]}"

