# TNFR Python Engine

Engine for **modeling, simulation and measurement** of multiscale structural coherence through **structural operators** (emission, reception, coherence, dissonance, coupling, resonance, silence, expansion, contraction, self‑organization, mutation, transition, recursivity).

---

## What is `tnfr`?

`tnfr` is a Python library to **operate with form**: build nodes, couple them into networks, and **modulate their coherence** over time using structural operators. It does not describe “things”; it **activates processes**. Its theoretical basis is the *Teoria de la Naturaleza Fractal Resonante (TNFR)*, which understands reality as **networks of coherence** that persist because they **resonate**.

In practical terms, `tnfr` lets you:

* Model **Resonant Fractal Nodes (NFR)** with parameters for **frequency** (νf), **phase** (θ), and **form** (EPI). Use the ASCII constants `VF_KEY` and `THETA_KEY` to reference these attributes programmatically; the Unicode names remain available as aliases.
* Apply **structural operators** to start, stabilize, propagate, or reconfigure coherence.
* **Simulate** nodal dynamics with discrete/continuous integrators.
* **Measure** global coherence C(t), nodal gradient ΔNFR, and the **Sense Index** (Si).
* **Visualize** states and trajectories (coupling matrices, C(t) curves, graphs).

A form emerges and persists when **internal reorganization** (ΔNFR) **resonates** with the node’s **frequency** (νf).

---

## Installation

```bash
pip install tnfr
```
* https://pypi.org/project/tnfr/
* Requires **Python ≥ 3.9**.
* Install extras:
  * NumPy: `pip install tnfr[numpy]`
  * YAML: `pip install tnfr[yaml]`
  * orjson (faster JSON serialization): `pip install tnfr[orjson]`
  * All: `pip install tnfr[numpy,yaml,orjson]`
* When `orjson` is unavailable the engine falls back to Python's built-in
  `json` module.

### Optional imports with cache

Use ``tnfr.cached_import`` to load optional dependencies and cache the result.
It returns ``None`` when the module (or attribute) is missing, avoiding
repeated import attempts:

```python
from tnfr import cached_import

np = cached_import("numpy")
safe_load = cached_import("yaml", "safe_load")

# provide a shared cache with an explicit lock
from cachetools import TTLCache
import threading

cache = TTLCache(32, 60)
lock = threading.Lock()
cached_import("numpy", cache=cache, lock=lock)

# clear the cache (e.g. after installing a dependency at runtime)
cached_import.cache_clear()
```

``tnfr.import_utils.optional_import`` is deprecated; use
``cached_import`` instead.

For optional JavaScript tooling, install the Node.js dependencies:

```bash
npm install
```

## Tests

Run the test suite from the project root using the helper script, which sets
the necessary `PYTHONPATH`:

```bash
./scripts/run_tests.sh
```

Avoid running `pytest` directly or executing the script from other directories,
as the environment may be misconfigured and imports will fail.

## Locking policy

The engine centralises reusable process-wide locks in
`tnfr.locking`. Modules obtain named locks via `locking.get_lock()` or
use `locking.locked()` as a context manager. This avoids scattering
`threading.Lock` instances across the codebase and ensures that shared
resources are synchronised consistently. Module-level caches or global
state should always use these named locks; only short-lived objects may
instantiate ad-hoc locks directly when they are not shared.

---

## Callback error handling

Callback errors are stored in a ring buffer attached to the graph.  The
buffer retains at most the last 100 errors by default, but the limit can be
adjusted at runtime via ``tnfr.callback_utils.set_callback_error_limit`` and
inspected with ``tnfr.callback_utils.get_callback_error_limit``.

---

## Why TNFR (in 60 seconds)

* **From objects to coherences:** you model **processes** that hold, not fixed entities.
* **Operators instead of rules:** you compose **structural operators** (e.g., *emission*, *coherence*, *dissonance*) to **build trajectories**.
* **Operational fractality:** the same pattern works for **ideas, teams, tissues, narratives**; the scales change, **the logic doesn’t**.

---

## Key concepts (operational summary)

* **Node (NFR):** a unit that persists because it **resonates**. Parameterized by **νf** (frequency), **θ** (phase), and **EPI** (coherent form).
* **Structural operators** - functions that reorganize the network:

  * **Emission** (start), **Reception** (open), **Coherence** (stabilize), **Dissonance** (creative tension), **Coupling** (synchrony), **Resonance** (propagate), **Silence** (latency), **Expansion**, **Contraction**, **Self‑organization**, **Mutation**, **Transition**, **Recursivity**.
* **Magnitudes:**

  * **C(t):** global coherence.
  * **ΔNFR:** nodal gradient (need for reorganization).
  * **νf:** structural frequency (Hz\_str).
  * **Si:** sense index (ability to generate stable shared coherence).

---

## Typical workflow

1. **Model** your system as a network: nodes (agents, ideas, tissues, modules) and couplings.
2. **Select** a **trajectory of operators** aligned with your goal (e.g., *start → couple → stabilize*).
3. **Simulate** the dynamics: number of steps, step size, tolerances.
4. **Measure**: C(t), ΔNFR, Si; identify bifurcations and collapses.
5. **Iterate** with controlled **dissonance** to open mutations without losing form.

---

## Main metrics

* `coherence(traj) → C(t)`: global stability; higher values indicate sustained form.
* `gradient(state) → ΔNFR`: local demand for reorganization (high = risk of collapse/bifurcation).
* `sense_index(traj) → Si`: proxy for **structural sense** (capacity to generate shared coherence) combining **νf**, phase, and topology.

## Topological remeshing

Use ``tnfr.operators.apply_topological_remesh`` (``from tnfr.operators import apply_topological_remesh``)
to reorganize connectivity based on nodal EPI similarity while preserving
graph connectivity. Modes:

- ``"knn"`` – connect each node to its ``k`` nearest neighbours (with optional
  rewiring).
- ``"mst"`` – retain only a minimum spanning tree.
- ``"community"`` – collapse modular communities and reconnect them by
  similarity.

All modes ensure connectivity by adding a base MST.

---

## History configuration

Recorded series are stored under `G.graph['history']`. Set `HISTORY_MAXLEN` in
the graph (or override the default) to keep only the most recent entries. The
value must be non‑negative; negative values raise ``ValueError``. When the
limit is positive the library uses bounded `deque` objects and removes the
least populated series when the number of history keys grows beyond the limit.
Compaction of internal usage counters happens every
``HISTORY_COMPACT_EVERY`` accesses, which can also be adjusted through
``G.graph``.

### Random node sampling

To reduce costly comparisons the engine stores a per‑step random subset of
node ids under `G.graph['_node_sample']`. Operators may use this to avoid
scanning the whole network. Sampling is skipped automatically when the graph
has fewer than **50 nodes**, in which case all nodes are included.

### Jitter RNG cache

`random_jitter` uses an LRU cache of `random.Random` instances keyed by `(seed, node)`.
`JITTER_CACHE_SIZE` controls the maximum number of cached generators (default: `256`);
when the limit is exceeded the least‑recently used entry is discarded. Increase it for
large graphs or heavy jitter usage, or lower it to save memory.

To adjust the number of cached jitter sequences used for deterministic noise,
configure `JITTER_MANAGER` before calling `setup`:

```python
from tnfr.operators import JITTER_MANAGER

# Resize cache to keep only 512 entries
JITTER_MANAGER.max_entries = 512
JITTER_MANAGER.setup(force=True)

# or in a single call
JITTER_MANAGER.setup(max_entries=512)
```

`setup` preserves the current size unless a new `max_entries` value is supplied.
Custom sizes persist across subsequent `setup` calls.

### Edge version tracking

Wrap sequences of edge mutations with `edge_version_update(G)` so the edge
version increments on entry and exit. This keeps caches and structural logs
aligned with the network's evolution.

### Defaults injection performance

`inject_defaults` evita copias profundas cuando los valores son inmutables (números,
cadenas, tuplas). Solo se usa `copy.deepcopy` para estructuras mutables, reduciendo
el costo de inicializar grafos con parámetros por defecto.

---

## Trained GPT

https://chatgpt.com/g/g-67abc78885a88191b2d67f94fd60dc97-tnfr-teoria-de-la-naturaleza-fractal-resonante

---

## Changelog

* Removed deprecated alias `sigma_vector_global`; use `sigma_vector_from_graph` instead.
* Cleaned up `tnfr.program.__all__` to exclude private helpers.
* Stopped re-exporting ``CallbackSpec`` and ``apply_topological_remesh`` at the
  package root; import them via ``tnfr.trace`` and ``tnfr.operators``.

---

## MIT License

Copyright (c) 2025 TNFR - Teoría de la naturaleza fractral resonante

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

If you use `tnfr` in research or projects, please cite the TNFR conceptual framework and link to the PyPI package.
