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

## Quick start

### Desde Python

```python
from tnfr import create_nfr, run_sequence
from tnfr.structural import (
    Emision,
    Recepcion,
    Coherencia,
    Resonancia,
    Silencio,
)
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si

G, nodo = create_nfr("A", epi=0.2, vf=1.0, theta=0.0)
ops = [Emision(), Recepcion(), Coherencia(), Resonancia(), Silencio()]
run_sequence(G, nodo, ops)

C, delta_nfr_medio, depi_medio = compute_coherence(G, return_means=True)
si_por_nodo = compute_Si(G)
print(f"C(t)={C:.3f}, ΔNFR̄={delta_nfr_medio:.3f}, dEPI/dt̄={depi_medio:.3f}, Si={si_por_nodo[nodo]:.3f}")
```

La secuencia respeta la ecuación nodal porque `create_nfr` inicializa el nodo con su **νf** y fase, `run_sequence` aplica operadores canónicos en el orden emisión → recepción → coherencia → resonancia → silencio y, después de cada paso, el gancho dinámico recalcula **ΔNFR** y la fase activa del nodo. Esa telemetría permite medir **C(t)** y **Si**, adelantando lo que se desarrolla en [Key concepts (operational summary)](#key-concepts-operational-summary) y [Main metrics](#main-metrics).

### Desde la línea de comandos

Archivo `secuencia.json`:

```json
[
  "emision",
  "recepcion",
  "coherencia",
  "resonancia",
  "silencio"
]
```

```bash
tnfr sequence --nodes 1 --sequence-file secuencia.json --save-history historia.json
```

El subcomando `sequence` carga la trayectoria canónica del archivo JSON, ejecuta los operadores con la gramática oficial y actualiza **νf**, **ΔNFR** y fase usando los mismos ganchos que la API de Python. Al finalizar se vuelcan en `historia.json` las series de **C(t)**, **ΔNFR** medio y **Si**, que amplían las secciones sobre [operadores estructurales](#key-concepts-operational-summary) y [métricas](#main-metrics).

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

Use ``tnfr.cached_import`` to load optional dependencies and cache the result
via a process-wide LRU cache. Missing modules (or attributes) yield ``None``
without triggering repeated imports. The helper records failures and emits a
single warning per module to keep logs tidy. When optional packages are
installed at runtime call ``prune_failed_imports`` to clear the consolidated
failure/warning registry before retrying:

```python
from tnfr import cached_import, prune_failed_imports

np = cached_import("numpy")
safe_load = cached_import("yaml", "safe_load")

# provide a shared cache with an explicit lock
from cachetools import TTLCache
import threading

cache = TTLCache(32, 60)
lock = threading.Lock()
cached_import("numpy", cache=cache, lock=lock)

# clear caches after installing a dependency at runtime
cached_import.cache_clear()
prune_failed_imports()
```

All project tooling is now Python-native; no Node.js or npm workflows are
required to build, test, or document the engine. The repository therefore
does not vendor any JavaScript dependencies and keeps the `node_modules/`
tree out of version control.

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
`tnfr.locking`. Modules obtain named locks via `locking.get_lock()` and
use the returned `threading.Lock` in their own critical sections. This
avoids scattering `threading.Lock` instances across the codebase and
ensures that shared resources are synchronised consistently.
Module-level caches or global state should always use these named
locks; only short-lived objects may instantiate ad-hoc locks directly
when they are not shared.

---

## Callback error handling

Callback errors are stored in a ring buffer attached to the graph.  The
buffer retains at most the last 100 errors by default, but the limit can be
adjusted at runtime via ``tnfr.callback_utils.callback_manager.set_callback_error_limit``
and inspected with ``tnfr.callback_utils.callback_manager.get_callback_error_limit``.

---

## Helper utilities API

`tnfr.helpers` bundles a compact set of public helpers that stay stable across
releases. They provide ergonomic access to the most common preparation steps
when orchestrating TNFR experiments.

### Collections and numeric helpers

* ``ensure_collection(it, *, max_materialize=...)`` — materialize potentially
  lazy iterables once, enforcing a configurable limit to keep simulations
  bounded.
* ``clamp(x, a, b)`` and ``clamp01(x)`` — restrict scalars to safe ranges for
  operator parameters.
* ``kahan_sum_nd(values, dims)`` — numerically stable accumulators used to
  track coherence magnitudes across long trajectories (use ``dims=1`` for
  scalars, ``dims=2`` for paired values, etc.).
* ``angle_diff(a, b)`` — compute minimal angular differences (radians) to
  compare structural phases.

### Historial de operadores estructurales

* ``push_glyph(nd, glyph, window)`` — registra la aplicación de un operador en el
  historial del nodo respetando la ventana configurada.
* ``recent_glyph(nd, glyph, window)`` — comprueba si un operador específico
  aparece en el historial reciente de un nodo.
* ``ensure_history(G)`` — prepare the graph-level history container with the
  appropriate bounds.
* ``last_glyph(nd)`` — inspecciona el último operador emitido por un nodo.
* ``count_glyphs(G, window=None, *, last_only=False)`` — agrega el uso de
  operadores estructurales en la red ya sea desde todo el historial o una
  ventana limitada.

### Graph caches and ΔNFR invalidation

* ``cached_node_list(G)`` — lazily cache a stable tuple of node identifiers,
  respecting opt-in sorted ordering.
* ``ensure_node_index_map(G)`` / ``ensure_node_offset_map(G)`` — expose cached
  index and offset mappings for graphs that need to project nodes to arrays.
* ``node_set_checksum(G, nodes=None, *, presorted=False, store=True)`` —
  produce deterministic BLAKE2b hashes to detect topology changes.
* ``stable_json(obj)`` — render deterministic JSON strings suited for hashing
  and reproducible logs.
* ``get_graph(obj)`` / ``get_graph_mapping(G, key, warn_msg)`` — normalise
  access to graph-level metadata regardless of wrappers.
* ``EdgeCacheManager`` together with ``edge_version_cache``,
  ``cached_nodes_and_A`` and ``edge_version_update`` encapsulate the edge
  version cache. ``increment_edge_version`` bumps the version manually for
  imperative workflows.
* ``mark_dnfr_prep_dirty(G)`` — invalidate precomputed ΔNFR preparation when
  mutating edges outside the cache helpers.

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
obtain the manager with ``get_jitter_manager`` before calling ``setup``:

```python
from tnfr.operators import get_jitter_manager

manager = get_jitter_manager()
# Resize cache to keep only 512 entries
manager.max_entries = 512

# or in a single call that also clears previous counters
manager.setup(max_entries=512)
```

``setup`` preserves the current size unless a new ``max_entries`` value is
supplied. Custom sizes persist across subsequent ``setup`` calls, and
``max_entries`` assignments take effect immediately.

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
* Removed legacy `tnfr.program` alias; import programming helpers from `tnfr.execution`.
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
