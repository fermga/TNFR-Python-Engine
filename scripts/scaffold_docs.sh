#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
THEORY_DIR="$DOCS_DIR/theory"

mkdir -p "$THEORY_DIR"

cat > "$DOCS_DIR/index.md" <<'DOC'
# TNFR Documentation Index (Phase 3 scaffold)

TODO: Populate the landing page with orientation content for the TNFR documentation set.

DOC

cat > "$DOCS_DIR/foundations.md" <<'DOC'
# Foundations — Mathematics scaffold

TODO: Capture the mathematics quick-start, canonical operator setup, and telemetry primers.

DOC

NOTEBOOK_DIR="$THEORY_DIR" python <<'PY'
from pathlib import Path
import json
import os
import uuid

notebooks = [
    ("01-structural-frequency.ipynb", "Structural frequency primer"),
    ("02-phase-synchrony.ipynb", "Phase synchrony lattices"),
    ("03-delta-nfr-gradients.ipynb", "ΔNFR gradient fields"),
    ("04-coherence-metrics.ipynb", "Coherence metrics walkthrough"),
    ("05-sense-index.ipynb", "Sense index calibration"),
    ("06-recursivity.ipynb", "Recursivity cascades"),
]

output_dir = Path(os.environ["NOTEBOOK_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

language_info = {
    "name": "python",
    "version": "3.11",
    "mimetype": "text/x-python",
    "codemirror_mode": {"name": "ipython", "version": 3},
    "pygments_lexer": "ipython3",
}

for filename, title in notebooks:
    path = output_dir / filename
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "id": uuid.uuid4().hex,
                "source": [
                    f"# {title}\n",
                    "\n",
                    "TODO: Outline the notebook narrative, learning goals, and TNFR invariants to cover.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "id": uuid.uuid4().hex,
                "outputs": [],
                "source": ["# TODO: implement walkthrough cells\n"],
            },
        ],
        "metadata": {
            "kernelspec": kernelspec,
            "language_info": language_info,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=2)
        handle.write("\n")
PY
