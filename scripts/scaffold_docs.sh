#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
NOTEBOOK_DIR="$DOCS_DIR/notebooks/mathematical-foundations"

mkdir -p "$NOTEBOOK_DIR"

cat > "$DOCS_DIR/foundations.md" <<'DOC'
# Foundations — Quick Start (Mathematics)

This guide introduces the mathematical APIs that keep TNFR simulations canonical. It focuses on
Hilbert-space primitives, ΔNFR orchestration, and telemetry so you can launch reproducible
experiments without leaving the mathematical layer.

## 1. Setup checklist

Install the package and optional notebook extras:

```bash
pip install tnfr[notebook]
```

Verify imports succeed inside your environment:

    >>> from tnfr.mathematics import HilbertSpace, make_coherence_operator

Keep documentation examples doctest-friendly by running them through `python -m doctest` or
`pytest --doctest-glob='*.md'` before publishing updates.

## 2. Create a coherent state

Use the Hilbert space helpers to define canonical basis vectors and validate norm preservation.

    >>> from tnfr.mathematics import HilbertSpace
    >>> space = HilbertSpace(dimension=3)
    >>> vector = [1.0, 0.0, 0.0]
    >>> bool(space.is_normalized(vector))
    True

The `HilbertSpace` dataclass enforces positive dimensions and returns complex identity bases that
match TNFR's discrete spectral component. Norm checks rely on `numpy.vdot`, giving you immediate
feedback if a vector drifts away from unit coherence.

## 3. Activate protective math flags

Mathematical feature flags gate optional checks and logging. You can activate them via environment
variables or the scoped context manager.

```bash
export TNFR_ENABLE_MATH_VALIDATION=1
export TNFR_ENABLE_MATH_DYNAMICS=1
export TNFR_LOG_PERF=1
```

or programmatically:

    >>> from tnfr.config.feature_flags import context_flags, get_flags
    >>> get_flags().enable_math_validation
    False
    >>> with context_flags(enable_math_validation=True, log_performance=True) as active:
    ...     active.enable_math_validation, active.log_performance
    (True, True)
    >>> get_flags().log_performance
    False

The context manager preserves flag stacks so nested overrides unwind cleanly after each experiment.

## 4. Compose operators and track ΔNFR

Operators created in `tnfr.mathematics` keep structural semantics explicit. The factory validates
spectral inputs and enforces positive semidefinite behaviour, ensuring coherence remains monotonic.

    >>> operator = make_coherence_operator(dim=2, c_min=0.3)
    >>> round(operator.c_min, 2)
    0.3

To run operator sequences on graph nodes, couple the mathematics layer with the structural facade:

    >>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY, VF_PRIMARY  # doctest: +SKIP
    >>> from tnfr.dynamics import set_delta_nfr_hook  # doctest: +SKIP
    >>> from tnfr.structural import Coupling, create_nfr, run_sequence  # doctest: +SKIP
    >>> G, node = create_nfr("math", vf=1.0, theta=0.2)  # doctest: +SKIP
    >>> def sync(graph):  # doctest: +SKIP
    ...     graph.nodes[node][DNFR_PRIMARY] = 0.01  # doctest: +SKIP
    ...     graph.nodes[node][THETA_PRIMARY] += 0.02  # doctest: +SKIP
    ...     graph.nodes[node][VF_PRIMARY] += 0.05  # doctest: +SKIP
    >>> set_delta_nfr_hook(G, sync)  # doctest: +SKIP
    >>> run_sequence(G, node, [Coupling()])  # doctest: +SKIP
    >>> round(G.nodes[node][THETA_PRIMARY], 2)  # doctest: +SKIP

The hook keeps ΔNFR, νf, and phase aligned after each operator application, satisfying the nodal
equation `∂EPI/∂t = νf · ΔNFR(t)`.

## 5. Simulate unitary evolution

Stable unitary flows prevent coherence loss and make ΔNFR modulation auditable. The helper returns
both the pass/fail flag and the resulting norm so doctests can assert stability.

    >>> space_unitary = HilbertSpace(dimension=2)
    >>> operator_unitary = make_coherence_operator(dim=2, c_min=0.4)
    >>> state = [1.0, 0.0]
    >>> from tnfr.mathematics import stable_unitary
    >>> stable_unitary(state, operator_unitary, space_unitary)
    (True, 1.0)

Pair the call with active logging flags to capture structural frequency, ΔNFR, and phase telemetry.

## 6. Cost reference

| Operation | Primary cost driver | ΔNFR impact | Notes |
| --- | --- | --- | --- |
| `make_coherence_operator` | Eigenvalue validation (`O(n)`) | Keeps thresholds ≥ `c_min` | Supply diagonal spectra for cheaper instantiation. |
| `run_sequence` | Graph traversal (`O(|E|)`) | Applies ΔNFR hook once per operator | Pre-validate sequences with `validate_sequence` to catch grammar drift. |
| `stable_unitary` | Eigen decomposition (`O(n^3)`) | Preserves Hilbert norm | Cache operators when iterating multiple steps. |
| `coherence_expectation` | Matrix-vector multiply (`O(n^2)`) | Reports scalar coherence | Toggle `normalise=False` when the state is already unit norm. |

Treat these costs as guidance for smoke tests and notebook demonstrations; production workflows
should profile real ΔNFR sequences and telemetry loads.
DOC

NOTEBOOK_DIR="$NOTEBOOK_DIR" python <<'PY'
from pathlib import Path
import json
import os
import uuid

notebooks = [
    ("01-structural-frequency.ipynb", "Structural frequency primer", "Set up ΔNFR-friendly spectral scaffolds."),
    ("02-phase-synchrony.ipynb", "Phase synchrony lattices", "Calibrate phase locks and coherence windows."),
    ("03-delta-nfr-gradients.ipynb", "ΔNFR gradient fields", "Track ΔNFR modulation under canonical hooks."),
    ("04-coherence-metrics.ipynb", "Coherence metrics walkthrough", "Compute C(t) and auxiliary metrics."),
    ("05-sense-index.ipynb", "Sense index calibration", "Stabilise Si measurements across nodes."),
    ("06-recursivity.ipynb", "Recursivity cascades", "Map nested EPIs without breaking fractality."),
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

for filename, title, summary in notebooks:
    path = output_dir / filename
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "id": uuid.uuid4().hex,
                "source": [f"# {title}\n", "\n", f"{summary}\n"],
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
