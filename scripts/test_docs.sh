#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK_EXECUTION_TIMEOUT=120

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install --upgrade pip
python -m pip install -r docs/requirements.txt
python -m pip install -e .[numpy]

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m doctest -o ELLIPSIS docs/index.md docs/foundations.md

for notebook_dir in docs/theory docs/notebooks; do
  if [ -d "$notebook_dir" ]; then
    while IFS= read -r notebook; do
      PYTHONWARNINGS="ignore:MissingIDFieldWarning" python -m jupyter nbconvert \
        --to notebook --execute "$notebook" --stdout \
        --ExecutePreprocessor.timeout="$NOTEBOOK_EXECUTION_TIMEOUT" > /dev/null
    done < <(find "$notebook_dir" -type f -name '*.ipynb' | sort)
  fi
done
