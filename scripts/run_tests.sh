#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
pydocstyle src/tnfr/selector.py src/tnfr/utils/data.py src/tnfr/utils/graph.py
coverage run --source=src -m pytest "$@"
coverage report -m
vulture --min-confidence 80 src tests
