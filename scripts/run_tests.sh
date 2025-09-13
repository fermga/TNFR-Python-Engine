#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
pydocstyle src/tnfr/selector.py src/tnfr/value_utils.py src/tnfr/graph_utils.py
python -m pytest "$@"
