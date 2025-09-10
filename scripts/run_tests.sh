#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
python -m pytest "$@"
