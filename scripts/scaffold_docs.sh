#!/usr/bin/env bash
set -euo pipefail

# This scaffold regenerates the Phase-3 documentation templates that ship with the
# repository.  By default it only creates missing files so that authored content
# is preserved.  Pass --refresh when you intentionally want to overwrite the
# rendered quick-start guides or notebook shells.

usage() {
  cat <<'USAGE'
Usage: scripts/scaffold_docs.sh [--refresh]

Without flags the script creates any missing quick-start pages and notebooks
using the canonical Phase-3 templates.  Existing files are left untouched so you
cannot accidentally clobber hand-edited documentation.  Invoke the script with
--refresh to rebuild every template from scratch (useful after upstream updates
to the canonical wording).  Always review the git diff after running with
--refresh to make sure you are not discarding new work.
USAGE
}

REFRESH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh)
      REFRESH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
THEORY_DIR="$DOCS_DIR/theory"
GETTING_STARTED_DIR="$DOCS_DIR/getting-started"

relpath() {
  python - "$REPO_ROOT" "$1" <<'PY'
import os
import sys
root, target = sys.argv[1:]
print(os.path.relpath(target, root))
PY
}

should_write() {
  local path="$1"
  local rel
  rel=$(relpath "$path")
  if [[ -f "$path" && "$REFRESH" -eq 0 ]]; then
    printf 'Skipping existing %s (use --refresh to overwrite)\n' "$rel" >&2
    return 1
  fi
  mkdir -p "$(dirname "$path")"
  if [[ -f "$path" ]]; then
    printf 'Refreshing %s\n' "$rel"
  else
    printf 'Writing %s\n' "$rel"
  fi
  return 0
}

if should_write "$DOCS_DIR/index.md"; then
  cat <<'DOC' > "$DOCS_DIR/index.md"
# TNFR Documentation Index

Welcome to the canonical reference for the TNFR Python Engine. This page orients you to the
major documentation areas so you can quickly find the right level of detail—whether you are
bootstrapping an environment, validating operator semantics, or diving into the underlying
theory.

## Documentation map

- **Getting started** – begin with the practical [Quickstart](getting-started/quickstart.md) to
  spin up a TNFR node, then review the [migrating guide](getting-started/migrating-remesh-window.md)
  if you are coming from Remesh Window.
- **API reference** – consult the [overview](api/overview.md) plus the focused guides on
  [structural operators](api/operators.md) and [telemetry utilities](api/telemetry.md) when you need
  concrete call signatures or examples.
- **Mathematical Foundations** – the notebooks under `theory/` connect the canonical equations with
  implementation choices. Use them when you must align derivations with code paths.
- **Examples** – cloneable scenarios in [examples/README.md](examples/README.md) that demonstrate
  cross-scale coherence checks.
- **Security** – operational guidance for monitoring and supply-chain hygiene in
  [security/](security/monitoring.md).
- **Releases** – version-by-version summaries in the [release notes](releases.md).

!!! important "Mathematical Foundations"
    The [Mathematical Foundations overview](theory/00_tnfr_overview.ipynb) anchors the canonical
    nodal equation and structural operators. Each primer (structural frequency, phase synchrony,
    ΔNFR gradient fields, coherence metrics, sense index, and recursivity cascades) expands the
    derivations used by the engine. Refer back here whenever you need to validate analytical
    assumptions or reproduce the derivations behind telemetry outputs.

!!! tip "Quick-start pathways"
    * For implementers: follow the [Quickstart](getting-started/quickstart.md) to configure
      dependencies, initialize a seed, and run your first coherence sweep.
    * For theorists: the [Mathematical Quick Start](foundations.md) bridges the primer notebooks with
      the code-level abstractions.

## Release cadence

Stable builds, bug fixes, and structural operator updates are catalogued in the
[Release notes](releases.md). Use that page to confirm which operators, telemetry fields, and
notebook revisions shipped in a given version before you align experiments or migrations.

## Need a different entry point?

Use the navigation sidebar (Material theme) to jump directly into operators, notebooks, or example
bundles. Each section cross-links back to this index so you can maintain orientation while
exploring deeper content.
DOC
fi

if should_write "$DOCS_DIR/foundations.md"; then
  cat <<'DOC' > "$DOCS_DIR/foundations.md"
# Foundations — Mathematics quick start

The mathematics layer exposes the canonical spaces, ΔNFR generators, and runtime diagnostics that
keep the nodal equation faithful to `∂EPI/∂t = νf · ΔNFR(t)`. This quick start walks through the
minimal scaffolding required to stand up a reproducible spectral experiment, turn on validation
guards, and observe unitary stability before coupling into higher level operators.

## 1. Canonical quick-start

1. **Select a space** – use :class:`tnfr.mathematics.HilbertSpace` for discrete spectral experiments
   or :class:`tnfr.mathematics.BanachSpaceEPI` when mixing the continuous EPI tail.
2. **Construct ΔNFR** – call :func:`tnfr.mathematics.build_delta_nfr` with a topology (`"laplacian"`
   or `"adjacency"`) and νf scaling. The helper guarantees a Hermitian generator so downstream
   coherence checks remain meaningful.
3. **Wrap operators** – initialise
   :class:`tnfr.mathematics.CoherenceOperator`/:class:`~tnfr.mathematics.FrequencyOperator` to
   project coherence and νf expectations.
4. **Collect metrics** – invoke :func:`tnfr.mathematics.normalized`, :func:`~tnfr.mathematics.coherence`,
   :func:`~tnfr.mathematics.frequency_positive`, and :func:`~tnfr.mathematics.stable_unitary` to ensure
   ΔNFR preserves Hilbert norms while sustaining positive structural frequency.

The notebooks [`theory/00_tnfr_overview.ipynb`](theory/00_tnfr_overview.ipynb) and
[`theory/02_phase_synchrony_lattices.ipynb`](theory/02_phase_synchrony_lattices.ipynb) replay these
steps with expanded derivations and visual telemetry overlays.

## 2. Environment feature flags

Mathematics diagnostics respect three environment variables. They are read via
:func:`tnfr.config.get_flags` and can be temporarily overridden with
:func:`tnfr.config.context_flags`.

* `TNFR_ENABLE_MATH_VALIDATION` – enables strict ΔNFR/Hilbert assertions inside runtime validators.
* `TNFR_ENABLE_MATH_DYNAMICS` – unlocks experimental spectral integrators in
  :mod:`tnfr.mathematics.dynamics`.
* `TNFR_LOG_PERF` – activates debug logging for normalization, coherence, and unitary metrics.

The snippet below demonstrates the override stack; the state before and after `context_flags`
confirms that overrides remain scoped to the `with` block.

```pycon
>>> from tnfr.config.feature_flags import get_flags, context_flags
>>> get_flags().enable_math_validation
False
>>> with context_flags(enable_math_validation=True, log_performance=True) as scoped:
...     (scoped.enable_math_validation, scoped.log_performance)
(True, True)
>>> get_flags().log_performance
False

```

When running shell commands, export the variables directly, e.g.
`TNFR_ENABLE_MATH_VALIDATION=1 TNFR_LOG_PERF=1 python -m doctest docs/foundations.md`.

## 3. Executable ΔNFR and unitary validation

The following session builds a Laplacian ΔNFR generator, evaluates unitary stability, and asserts νf
positivity. All routines are deterministic when a NumPy generator seed is supplied to
:func:`build_delta_nfr`, making the snippet safe for doctest execution.

```pycon
>>> import numpy as np
>>> from tnfr.mathematics import (
...     HilbertSpace,
...     build_delta_nfr,
...     CoherenceOperator,
...     FrequencyOperator,
...     stable_unitary,
...     coherence,
...     frequency_positive,
... )
>>> space = HilbertSpace(dimension=3)
>>> delta = build_delta_nfr(3, topology="laplacian", nu_f=0.8, scale=0.25)
>>> delta.shape
(3, 3)
>>> operator = CoherenceOperator(delta)
>>> state = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
>>> unitary_passed, unitary_norm = stable_unitary(state, operator, space)
>>> unitary_passed
True
>>> round(unitary_norm, 12)
1.0
>>> frequency_positive(state, FrequencyOperator(np.eye(3)))['passed']
True
>>> coherence(state, operator, threshold=operator.c_min)
(True, 0.4)

```

To integrate ΔNFR outputs into networkx graphs, see the migration recipe in
[`getting-started/quickstart.md`](getting-started/quickstart.md) and the operator catalogue under
[`api/operators.md`](api/operators.md).

## 4. Telemetry cost and logging budget

| Metric guard | Flag dependency | Dominant cost | Logging channel |
| --- | --- | --- | --- |
| `normalized` | `TNFR_LOG_PERF` | `O(n)` vector norm | `tnfr.mathematics.runtime` debug record |
| `coherence` / `coherence_expectation` | `TNFR_LOG_PERF` | `O(n²)` due to matrix-vector multiply | Same channel with payload `{\"threshold\": …}` |
| `frequency_positive` | `TNFR_LOG_PERF` | `O(n²)` spectrum check plus projection | Debug message includes `"projection_passed"` and spectrum extrema |
| `stable_unitary` | `TNFR_LOG_PERF` | `O(n³)` eigendecomposition per step | Debug payload logs `"norm_after"` for ΔNFR unitary audits |

The runtime helpers defer to Python's :mod:`logging` package. Configure it once at process start
(`logging.basicConfig(level=logging.DEBUG)`) and then enable `TNFR_LOG_PERF` to stream the tabled
payloads without instrumenting call sites. The Phase 3 guideline is to sample the `stable_unitary`
log at each integration step while only periodically recording the cheaper `normalized` metric to
control storage costs.

## 5. Next steps

* Load the lattice notebooks listed above to inspect full ΔNFR evolution traces.
* Refer to [`api/telemetry.md`](api/telemetry.md) for downstream aggregation and to
  [`theory/00_tnfr_overview.ipynb`](theory/00_tnfr_overview.ipynb) for the derivation that ties the Hilbert
  norms back to ΔNFR coherence envelopes.
DOC
fi

if should_write "$GETTING_STARTED_DIR/quickstart.md"; then
  cat <<'DOC' > "$GETTING_STARTED_DIR/quickstart.md"
# Quickstart

Follow this guide to install the TNFR Python Engine, warm optional dependencies, and execute the
first structural workflows from Python and the CLI.

## Installation

Install the engine from PyPI. Python 3.9 or newer is required.

```bash
pip install tnfr
```

Optional extras:

- NumPy: `pip install tnfr[numpy]`
- YAML: `pip install tnfr[yaml]`
- orjson (faster JSON serialization): `pip install tnfr[orjson]`
- All extras: `pip install tnfr[numpy,yaml,orjson]`

When `orjson` is unavailable the engine falls back to Python's built-in `json` module.

### Optional imports with cache helpers

Use `tnfr.utils.cached_import` to load optional dependencies lazily and keep a shared cache of
successes and failures. Failures are memoised and logged once per module. Set `lazy=True` to obtain a
lightweight proxy that postpones the real import until the object is first used. When optional
packages are installed at runtime, call `tnfr.utils.prune_failed_imports` to clear the consolidated
failure registry before retrying.

```python
from tnfr.utils import cached_import, prune_failed_imports, warm_cached_import

np = cached_import("numpy")
safe_load = cached_import("yaml", "safe_load")

# Postpone work until the symbol is first accessed.
safe_lazy = cached_import("yaml", "safe_load", lazy=True)

# Warm optional dependencies during application bootstrap.
warm_cached_import("numpy", ("yaml", "safe_load"))

# Provide a shared cache with an explicit lock.
from cachetools import TTLCache
import threading

cache = TTLCache(32, 60)
lock = threading.Lock()
cached_import("numpy", cache=cache, lock=lock)

# Clear caches after installing a dependency at runtime.
cached_import.cache_clear()
prune_failed_imports()
```

### Persistent cache layers

`tnfr.utils.cache.build_cache_manager` now hydrates multi-layer caches from a global configuration or
per-graph overrides. Use `tnfr.utils.cache.configure_global_cache_layers` to point the shared cache
manager to a Shelve file (filesystem persistence) and/or a Redis namespace for distributed
hydration. Calling `tnfr.utils.cache.reset_global_cache_manager` after updating the configuration
rebuilds the shared manager with the new layers:

```python
from tnfr.utils.cache import configure_global_cache_layers, reset_global_cache_manager

configure_global_cache_layers(
    shelve={"path": "/tmp/tnfr-cache.db", "flag": "c", "writeback": False},
    redis={"namespace": "tnfr:cache"},  # provide ``client`` or ``client_factory`` when needed
    replace=True,
)
reset_global_cache_manager()
```

Graphs can override the global settings by storing a mapping under
`tnfr.utils.cache._GRAPH_CACHE_LAYERS_KEY`. Supported keys match the global configuration (`"shelve"`
and `"redis"`). Whenever the configuration is present, `build_cache_manager` automatically wires the
extra layers for edge caches, jitter state, and RNG seeds while preserving cache hit/miss telemetry.

## Python quickstart

Create a resonant node, apply structural operators, and read coherence metrics. The sequence
preserves the nodal equation because `create_nfr` seeds the node with its νf and phase while
`run_sequence` validates the canonical grammar.

```python
from tnfr import create_nfr, run_sequence
from tnfr.structural import (
    Emission,
    Reception,
    Coherence,
    Resonance,
    Silence,
)
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si

G, node = create_nfr("A", epi=0.2, vf=1.0, theta=0.0)
ops = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
run_sequence(G, node, ops)

C, mean_delta_nfr, mean_depi = compute_coherence(G, return_means=True)
si_per_node = compute_Si(G)
print(
    f"C(t)={C:.3f}, ΔNFR̄={mean_delta_nfr:.3f}, dEPI/dt̄={mean_depi:.3f}, "
    f"Si={si_per_node[node]:.3f}"
)
```

Both `tnfr.dynamics.step` and `tnfr.dynamics.run` accept an optional `n_jobs` dictionary to pin
process/thread counts for ΔNFR, Si, integrators, phase coordination, and νf adaptation without
mutating `G.graph`.

### Preparing existing graphs

When you build a NetworkX graph outside of `create_nfr`, normalise its configuration with
`tnfr.prepare_network` before stepping the dynamics. The helper attaches the default configuration,
telemetry history, ΔNFR hook, and optional observer wiring. Versions prior to **TNFR 5.0** exposed a
legacy alias for the same helper. The alias has now been removed; update existing code to call
`prepare_network` directly before upgrading.

```python
import networkx as nx
from tnfr import prepare_network

G = nx.path_graph(4)
G.graph["ATTACH_STD_OBSERVER"] = True
prepare_network(G)
```

## CLI quickstart

The CLI mirrors the Python API while enforcing the canonical operator tokens. Create a sequence file
matching the Emission → Reception → Coherence → Resonance → Silence order:

```json
[
  "emission",
  "reception",
  "coherence",
  "resonance",
  "silence"
]
```

Starting with **TNFR 2.0** the CLI accepts **only** the English operator tokens. Rewrite existing
automation to match the canonical identifiers before upgrading.

Run the sequence on a single node and persist telemetry to `history.json`:

```bash
tnfr sequence --nodes 1 --sequence-file sequence.json --save-history history.json
```

Use `--summary-limit` to bound the number of samples per series in CLI summaries. Pass `0` or a
negative value to disable trimming altogether when exporting metrics.

| Canonical token | Operator role        |
| --------------- | -------------------- |
| `emission`      | Initiates resonance  |
| `reception`     | Captures information |
| `coherence`     | Stabilises the form  |
| `resonance`     | Propagates coherence |
| `silence`       | Freezes evolution    |

The command updates νf, ΔNFR, and phase using the same hooks as the Python API. Inspect the saved
history for the series of C(t), mean ΔNFR, and Si.

### Presets

Use the English preset identifiers when invoking `--preset` from the CLI:

| Preset identifier      | Description (summary)      |
| ---------------------- | -------------------------- |
| `resonant_bootstrap`   | Balanced start-up profile  |
| `contained_mutation`   | Mutation with guard rails  |
| `coupling_exploration` | Coupling sweep for studies |
| `canonical_example`    | Minimal tutorial sequence  |

## Next steps

- Explore the [examples](../examples/README.md) for multi-node scenarios and CLI workflows.
- Review the [API overview](../api/overview.md) before extending operator pipelines.
- Consult the [telemetry and utilities](../api/telemetry.md) guide to instrument your experiments
  with trace capture and reproducible caches.
DOC
fi

export NOTEBOOK_DIR="$THEORY_DIR"
export NOTEBOOK_REFRESH="$REFRESH"

python <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

repo_root = Path(os.environ["NOTEBOOK_DIR"]).parents[1]
notebook_dir = Path(os.environ["NOTEBOOK_DIR"])
refresh = os.environ["NOTEBOOK_REFRESH"] == "1"

notebooks = {
    "00_tnfr_overview.ipynb": {
        "title": "TNFR overview",
        "summary": "Map the documentation surface and connect the nodal equation to Phase-3 notebooks.",
        "sections": [
            (
                "Orientation",
                [
                    "# TNFR overview\n",
                    "\n",
                    "This notebook summarises the canonical moving parts of the Transcendent Nodal Fractal Resonance (TNFR) paradigm.\n",
                    "Use it as a landing zone before diving into the specialised primers.\n",
                ],
            ),
            (
                "Structural invariants",
                [
                    "## Structural invariants\n",
                    "\n",
                    "- **Nodal equation**: `∂EPI/∂t = νf · ΔNFR(t)` guides every operator sequence.\n",
                    "- **Phase discipline**: preserve synchrony checks before coupling nodes.\n",
                    "- **Operational fractality**: nested EPIs must conserve identity and telemetry coverage.\n",
                ],
            ),
            (
                "Notebook map",
                [
                    "## Notebook map\n",
                    "\n",
                    "| Notebook | Purpose |\n",
                    "| --- | --- |\n",
                    "| 01_hilbert_space_h_nfr | Construct ℋ_NFR with reproducible spectra. |\n",
                    "| 02_coherence_operator_hatC | Track coherence envelopes and ΔNFR expectations. |\n",
                    "| 03_frequency_operator_hatJ | Maintain positive νf across operator chains. |\n",
                    "| 04_validator_and_metrics_suite | Execute Phase-3 validator stack with logging budgets. |\n",
                    "| 05_unitary_dynamics_delta_nfr | Study ΔNFR unitary propagation with controlled perturbations. |\n",
                ],
            ),
        ],
        "code": [
            "from tnfr import create_nfr, run_sequence\n",
            "from tnfr.structural import Emission, Reception, Coherence, Resonance, Silence\n",
            "\n",
            "G, node = create_nfr('A', epi=0.25, vf=1.0, theta=0.0)\n",
            "run_sequence(G, node, [Emission(), Reception(), Coherence(), Resonance(), Silence()])\n",
            "G.nodes[node]['history'][-1]\n",
        ],
    },
    "01_hilbert_space_h_nfr.ipynb": {
        "title": "Hilbert space ℋ_NFR",
        "summary": "Build the canonical Hilbert space and confirm ΔNFR hermiticity before experiments.",
        "sections": [
            (
                "Construction",
                [
                    "# Hilbert space ℋ_NFR\n",
                    "\n",
                    "The ℋ_NFR space provides the orthonormal basis used to express EPI states.\n",
                    "This template highlights deterministic construction and ΔNFR compatibility checks.\n",
                ],
            ),
            (
                "Diagnostics",
                [
                    "## Diagnostics\n",
                    "\n",
                    "- Verify Hermitian ΔNFR generators before applying operator sequences.\n",
                    "- Sample C(t) and νf expectations to confirm coherence envelopes.\n",
                    "- Persist feature flags alongside experimental metadata for reproducibility.\n",
                ],
            ),
        ],
        "code": [
            "import numpy as np\n",
            "from tnfr.mathematics import HilbertSpace, build_delta_nfr\n",
            "\n",
            "space = HilbertSpace(dimension=4)\n",
            "rng = np.random.default_rng(42)\n",
            "delta = build_delta_nfr(4, topology='laplacian', nu_f=0.9, scale=0.2, rng=rng)\n",
            "np.allclose(delta, delta.conj().T)\n",
        ],
    },
    "02_coherence_operator_hatC.ipynb": {
        "title": "Coherence operator Ĉ",
        "summary": "Track coherence budgets and thresholds derived from ΔNFR spectra.",
        "sections": [
            (
                "Operator overview",
                [
                    "# Coherence operator Ĉ\n",
                    "\n",
                    "Ĉ projects EPI states onto coherence envelopes that respect ℋ_NFR topology.\n",
                    "Use the snippets below to validate thresholds before integrating dynamics.\n",
                ],
            ),
            (
                "Telemetry",
                [
                    "## Telemetry hooks\n",
                    "\n",
                    "- Record `coherence_expectation` outputs for each operator application.\n",
                    "- Cross-check with `stable_unitary` when ΔNFR perturbations occur.\n",
                    "- Archive the operator metadata alongside experiment seeds.\n",
                ],
            ),
        ],
        "code": [
            "import numpy as np\n",
            "from tnfr.mathematics import CoherenceOperator, build_delta_nfr\n",
            "\n",
            "rng = np.random.default_rng(12)\n",
            "delta = build_delta_nfr(3, topology='adjacency', nu_f=0.75, scale=0.15, rng=rng)\n",
            "operator = CoherenceOperator(delta)\n",
            "state = np.array([1.0, 0.0, 0.0], dtype=np.complex128)\n",
            "operator.coherence_expectation(state)\n",
        ],
    },
    "03_frequency_operator_hatJ.ipynb": {
        "title": "Frequency operator Ĵ",
        "summary": "Guard positive structural frequency during resonant propagation.",
        "sections": [
            (
                "Operator role",
                [
                    "# Frequency operator Ĵ\n",
                    "\n",
                    "Ĵ evaluates νf trajectories and clamps drift that would collapse coherence.\n",
                    "Embed these checks inside longer operator chains to trace stability.\n",
                ],
            ),
            (
                "Phase discipline",
                [
                    "## Phase discipline\n",
                    "\n",
                    "- Couple νf checks with explicit phase synchrony logging.\n",
                    "- Maintain deterministic seeds for ΔNFR perturbations during studies.\n",
                    "- Tie results back to the ℋ_NFR basis to preserve structural traceability.\n",
                ],
            ),
        ],
        "code": [
            "import numpy as np\n",
            "from tnfr.mathematics import FrequencyOperator\n",
            "\n",
            "state = np.array([0.6, 0.4j, 0.0])\n",
            "operator = FrequencyOperator(np.diag([1.0, 0.95, 0.85]))\n",
            "operator.frequency_positive(state)\n",
        ],
    },
    "04_validator_and_metrics_suite.ipynb": {
        "title": "Validator and metrics suite",
        "summary": "Execute the validator stack with logging budgets tuned for Phase-3 fieldwork.",
        "sections": [
            (
                "Validator stack",
                [
                    "# Validator and metrics suite\n",
                    "\n",
                    "Phase-3 emphasises explicit logging of coherence, νf, and ΔNFR stability.\n",
                    "Use the suite to script repeatable validation sweeps across experiments.\n",
                ],
            ),
            (
                "Logging budget",
                [
                    "## Logging budget\n",
                    "\n",
                    "- Sample `stable_unitary` each integration step.\n",
                    "- Down-sample cheaper metrics (e.g. `normalized`) to control storage.\n",
                    "- Persist flag states (`TNFR_ENABLE_MATH_VALIDATION`, etc.) with the logs.\n",
                ],
            ),
        ],
        "code": [
            "from tnfr.mathematics import (\n",
            "    coherence,\n",
            "    frequency_positive,\n",
            "    stable_unitary,\n",
            ")\n",
            "\n",
            "def validate_state(state, coherence_op, frequency_op, space):\n",
            "    results = {\n",
            "        'coherence': coherence(state, coherence_op),\n",
            "        'frequency': frequency_positive(state, frequency_op),\n",
            "        'unitary': stable_unitary(state, coherence_op, space),\n",
            "    }\n",
            "    return results\n",
        ],
    },

    "05_unitary_dynamics_delta_nfr.ipynb": {
        "title": "Unitary dynamics & ΔNFR",
        "summary": "Trace ΔNFR propagation under controlled perturbations and confirm coherence recovery.",
        "sections": [
            (
                "Dynamics overview",
                [
                    "# Unitary dynamics & ΔNFR\n",
                    "\n",
                    "This notebook captures unitary propagation experiments with ΔNFR perturbations.\n",
                    "Phase-3 workflows demand explicit evidence that coherence envelopes recover.\n",
                ],
            ),
            (
                "Experiment sketch",
                [
                    "## Experiment sketch\n",
                    "\n",
                    "1. Build ΔNFR with reproducible seeds.\n",
                    "2. Apply a bounded perturbation and observe `stable_unitary`.\n",
                    "3. Record νf adjustments and Si shifts after recovery.\n",
                ],
            ),
        ],
        "code": [
            "import numpy as np\n",
            "from tnfr.mathematics import (\n",
            "    HilbertSpace,\n",
            "    build_delta_nfr,\n",
            "    CoherenceOperator,\n",
            "    stable_unitary,\n",
            ")\n",
            "\n",
            "space = HilbertSpace(dimension=3)\n",
            "rng = np.random.default_rng(7)\n",
            "delta = build_delta_nfr(3, topology='laplacian', nu_f=0.8, scale=0.25, rng=rng)\n",
            "operator = CoherenceOperator(delta)\n",
            "state = np.array([1.0, 0.0, 0.0], dtype=np.complex128)\n",
            "stable_unitary(state, operator, space)\n",
        ],
    },

}

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

for filename, data in notebooks.items():
    path = notebook_dir / filename
    rel = path.relative_to(repo_root)
    was_existing = path.exists()
    if was_existing and not refresh:
        print(f"Skipping existing {rel} (use --refresh to overwrite)")
        continue

    cells = []
    cell_id = 1
    for _, source in data["sections"]:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "id": f"cell-{cell_id:04d}",
                "source": source,
            }
        )
        cell_id += 1

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "id": f"cell-{cell_id:04d}",
            "outputs": [],
            "source": data["code"],
        }
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": kernelspec,
            "language_info": language_info,
            "tnfr": {
                "title": data["title"],
                "summary": data["summary"],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    action = "Refreshed" if was_existing else "Created"
    print(f"{action} {rel}")
PY

# Remind contributors how to run the script safely when the notebook generator executes via Python.
if [[ "$REFRESH" -eq 0 ]]; then
  echo "Documentation scaffolding complete. Existing content was preserved." >&2
else
  echo "Documentation scaffolding refreshed. Review git diff before committing." >&2
fi
