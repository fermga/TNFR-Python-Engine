#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install --upgrade pip
python -m pip install -r docs/requirements.txt
python -m pip install -e .[docs,numpy]

SPHINX_SOURCE="docs/source"
SPHINX_BUILD="docs/_build"

sphinx-build -b doctest "$SPHINX_SOURCE" "$SPHINX_BUILD/doctest"
sphinx-build -b html "$SPHINX_SOURCE" "$SPHINX_BUILD/html"
