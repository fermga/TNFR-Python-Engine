#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"

python -m pip install --quiet ".[test,typecheck]"
python -m pydocstyle --add-ignore=D202 src/tnfr/selector.py src/tnfr/utils/data.py src/tnfr/utils/graph.py
python -m mypy src/tnfr
python -m coverage run --source=src -m pytest "$@"
python -m coverage report -m
python -m vulture --min-confidence 80 src tests
