# Quickstart

Follow this guide to install the TNFR Python Engine, warm optional dependencies, and
execute the first structural workflows from Python and the CLI.

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

Use `tnfr.utils.cached_import` to load optional dependencies lazily and keep a shared cache
of successes and failures. Failures are memoised and logged once per module. Set
`lazy=True` to obtain a lightweight proxy that postpones the real import until the object is
first used. When optional packages are installed at runtime, call
`tnfr.utils.prune_failed_imports` to clear the consolidated failure registry before
retrying.

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

Both `tnfr.dynamics.step` and `tnfr.dynamics.run` accept an optional `n_jobs` dictionary to
pin process/thread counts for ΔNFR, Si, integrators, phase coordination, and νf adaptation
without mutating `G.graph`.

### Preparing existing graphs

When you build a NetworkX graph outside of `create_nfr`, normalise its configuration with
`tnfr.prepare_network` before stepping the dynamics. The helper attaches the default
configuration, telemetry history, ΔNFR hook, and optional observer wiring. The legacy name
`tnfr.preparar_red` now emits a :class:`DeprecationWarning`, remains available only as a
bridge, and is scheduled for removal on **2025-06-01**.

```python
import networkx as nx
from tnfr import prepare_network

G = nx.path_graph(4)
G.graph["ATTACH_STD_OBSERVER"] = True
prepare_network(G)
```

## CLI quickstart

The CLI mirrors the Python API while enforcing the canonical operator tokens. Create a
sequence file matching the Emission → Reception → Coherence → Resonance → Silence order:

```json
[
  "emission",
  "reception",
  "coherence",
  "resonance",
  "silence"
]
```

> Legacy Spanish tokens (``"emision"``, ``"recepcion"``, …) are still accepted
> by the parser but now emit :class:`DeprecationWarning` and will be removed in a
> future release. Migrate automation scripts to the English identifiers to stay
> within the supported contract.

Run the sequence on a single node and persist telemetry to `history.json`:

```bash
tnfr sequence --nodes 1 --sequence-file sequence.json --save-history history.json
```

| Canonical token | English operator name |
| --------------- | --------------------- |
| `emision`       | Emission              |
| `recepcion`     | Reception             |
| `coherencia`    | Coherence             |
| `resonancia`    | Resonance             |
| `silencio`      | Silence               |

The command updates νf, ΔNFR, and phase using the same hooks as the Python API. Inspect the
saved history for the series of C(t), mean ΔNFR, and Si.

### Presets

Use the English preset identifiers when invoking `--preset` from the CLI:

| Preferred name         | Legacy alias             |
| ---------------------- | ------------------------ |
| `resonant_bootstrap`   | `arranque_resonante`     |
| `contained_mutation`   | `mutacion_contenida`     |
| `coupling_exploration` | `exploracion_acople`     |
| `canonical_example`    | `ejemplo_canonico`       |

The legacy Spanish aliases remain available for backward compatibility throughout the 1.x
series, but scripts and documentation should migrate to the English identifiers.

## Next steps

- Explore the [examples](../examples/README.md) for multi-node scenarios and CLI workflows.
- Review the [API overview](../api/overview.md) before extending operator pipelines.
- Consult the [telemetry and utilities](../api/telemetry.md) guide to instrument your
  experiments with trace capture and reproducible caches.
