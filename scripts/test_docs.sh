#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install --upgrade pip
python -m pip install -r docs/requirements.txt
python -m pip install -e .[numpy]

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m doctest -o ELLIPSIS docs/index.md docs/foundations.md

if [ -d "docs/notebooks" ]; then
  while IFS= read -r notebook; do
    PYTHONWARNINGS="ignore:MissingIDFieldWarning" python -m jupyter nbconvert \
      --to notebook --execute "$notebook" --stdout > /dev/null
  done < <(find docs/notebooks -type f -name '*.ipynb' | sort)
fi
