# Remaining nodal lookup cost: explicit stable traversal

Date: 2026-09-05. Scope: `utils/cache.py`, its public stub, and
`operators/jitter.py`; dedicated tests and benchmark. Prior audit changes are
preserved. No SDK execution, canonical operator, EPI update, seed derivation,
grammar rule, dependency, or `manual/` content changed in this track.

## Result and structural contract

The previous audit removed repeated serialization and checksum thrashing.
Public `NodeNX.offset()` still validates an ordered node snapshot on every
lookup because arbitrary NetworkX mutations and filtered views do not have a
universal mutation version. A full jitter sweep consequently retained O(V²)
ordered-comparison work. This default remains correct and unchanged.

The new public `tnfr.utils.cache.stable_node_offsets(graph)` context permits a
caller-owned synchronous traversal to validate node order once at entry and
once at exit. `random_jitter` uses its private offset snapshot for exact
`NodeNX` adapters. Custom node implementations retain their own `offset()`
semantics, and public `NodeNX.offset()` remains fully checked inside a scope.
The scope yields an insertion-ordered tuple for traversal; jitter offsets
continue to honor `SORT_NODES` and the existing stable representation order.

This changes lookup work only. The random draw remains determined by recorded
graph seed, canonical offset, and the node's persistent draw count. No
process identity enters RNG derivation. Exact trajectory comparisons below
cover nodal form, capacity, pressure, phase, C(t), Si, and executed histories;
the scope does not alter the nodal equation or U1–U6 enforcement.

## Ownership and limitations

The caller promises exclusive access and stable node membership, node order,
node attribute dictionary identities, label representations, and SORT_NODES
through the whole scope, including nested callbacks. Node attribute values
and edges may change. The context is not a graph lock or a transaction.

- Only ordinary NetworkX Graph, DiGraph, MultiGraph, and MultiDiGraph instances
  are accepted. Views and graph subclasses retain the existing unscoped path.
- Membership, actual node-object identity/order, node storage, node attribute
  dictionaries, and sorting mode are checked at the final boundary. Cheap
  storage/size/sorting/target checks reject detectable changes before a draw.
  A change elsewhere that preserves size can be detected only at the boundary.
- A violation raises; earlier valid operations and already-consumed draws are
  not rolled back. Transient mutations restored before the boundary are not
  tracked. Mutable label representations remain prohibited by the contract.
- Scopes nest and expire on success or exception. An invalid outer scope cannot
  be silently rebased by entering another scope for the same graph. The
  original body exception is preserved; Python 3.11+ adds a violation note.
- Graph identity separates ownership only. Graph copies do not inherit a
  snapshot. Copied execution contexts cannot revive an expired scope, and
  other threads or async tasks use the ordinary lookup path. Callers must not
  yield/await while holding this synchronous scope.
- No snapshot is persisted on the graph. A warm scope uses O(V) temporary
  storage and boundary work, then O(1) lookup for fixed nesting depth.
  Arbitrarily nested scope selection is O(depth). Cold sorting/checksum work
  keeps its previous cost.

Automatic SDK scoping was deliberately not introduced: Dissonance execution
can reach integrity hooks, glyph dispatch, propagation, and warning/logging
handlers. A complete exclusion policy for user mutation would be brittle.
The explicit context works around the existing canonical execution primitive
without changing its default behavior or claiming atomic capture against
noncooperating NetworkX writers. Readout capture has the same exclusive-access
requirement unless all writers participate in a shared lock.

Minimal standalone example, run from the repository root:

```python
import sys
sys.path.insert(0, "src")
import networkx as nx
from tnfr.node import NodeNX
from tnfr.operators.jitter import random_jitter
from tnfr.utils.cache import stable_node_offsets

graph = nx.cycle_graph(12)
graph.graph["RANDOM_SEED"] = 7
with stable_node_offsets(graph) as nodes:
    draws = [random_jitter(NodeNX.from_graph(graph, node), 0.1) for node in nodes]
```

For canonical word execution the same context can surround
`_run_network_sequence(graph, word, validate=True)`; the benchmark below uses
that existing SDK primitive. Callers are responsible for the same ownership
contract for all operators and callbacks inside the word.

## Controlled timing and operation counts

Runtime: Python 3.12.10, NetworkX 3.5, the repository `.venv312` on the same
Windows host. Immutable baseline:
`tmp/nodal-synergy-dist/tnfr-0.0.3.5-py3-none-any.whl`, SHA-256
`b0bc7f6733c08ea6bc4c95d63a8000cab4fdd745ecc3cd6b29ba0c1f7d8fbdb9`.

`benchmarks/stable_node_offset_scaling.py` inserts `src` explicitly, and puts
the selected wheel ahead of it for baseline runs without installing either.
Each size/mode uses three independently initialized cycle graphs, seed 7,
coherent zero phases, EPI 0, uniform initial frequency in [0.4, 0.7] Hz_str,
OZ noise enabled with sigma 0.1, and history window 20. Adapter creation and
one initial map population are outside timing. Scope entry/exit, instrumented
ordered comparison counts, and actual draw/operator execution are timed.
SDK timing also includes per-layer telemetry capture; JSON serialization is
outside timing. Runs were sequential; timings are finite host measurements,
not a machine-independent speed guarantee.

Median elapsed seconds:

| Workload | Nodes | Immutable wheel | Current, ordinary | Current, scoped |
|---|---:|---:|---:|---:|
| One jitter draw per node | 500 | 0.034410 | 0.035289 | 0.021847 |
| One jitter draw per node | 2000 | 0.281597 | 0.299796 | 0.089137 |
| Complete validated SDK word | 500 | 1.460740 | 1.751593 | 1.570032 |
| Complete validated SDK word | 2000 | 20.041977 | 19.657554 | 18.112817 |

The isolated 2000-node jitter sweep is 3.16 times faster than the immutable
wheel in this measurement. Multiplying graph size by four multiplies scoped
jitter time by 4.08, consistent with the bounded linear traversal work.
Full-word timing has larger variation and other operator work dominates;
these runs do not establish an attributable general SDK speedup.

For both workloads, counted ordered-node comparisons per run were:

| Nodes | Ordinary | Scoped |
|---|---:|---:|
| 500 | 250000 | 1000 |
| 2000 | 4000000 | 4000 |

The counter sums elements passed to `_same_node_snapshot`, demonstrating
V² versus 2V ordered comparisons. Scope construction and final node attribute
identity validation add O(V) dictionary work outside that counter, rather
than additional full scans per draw. The regression fixes this work-count
expectation independently of wall-clock timing.

Raw elapsed samples, in the same mode order as the table:

| Workload/size | Wheel samples | Ordinary samples | Scoped samples |
|---|---|---|---|
| Jitter/500 | 0.0339183, 0.0347937, 0.0344098 | 0.0351220, 0.0352887, 0.0354353 | 0.0218732, 0.0217747, 0.0218466 |
| Jitter/2000 | 0.2839762, 0.2813258, 0.2815973 | 0.3229282, 0.2997959, 0.2879943 | 0.0891367, 0.0892531, 0.0887648 |
| SDK/500 | 1.4607403, 1.4548939, 1.4610667 | 1.6110919, 1.7515934, 1.7676493 | 1.5721082, 1.5439341, 1.5700325 |
| SDK/2000 | 19.6726708, 20.0419770, 24.3308860 | 19.3716754, 19.6575542, 19.7619487 | 19.8863060, 18.1128172, 17.9741641 |

Exact reproducible commands (repeat the three modes with `--workload sdk`):

```powershell
.venv312/Scripts/python.exe benchmarks/stable_node_offset_scaling.py --workload jitter --wheel tmp/nodal-synergy-dist/tnfr-0.0.3.5-py3-none-any.whl
.venv312/Scripts/python.exe benchmarks/stable_node_offset_scaling.py --workload jitter
.venv312/Scripts/python.exe benchmarks/stable_node_offset_scaling.py --workload jitter --scoped
```

## Exact canonical trajectory evidence

The SDK benchmark applies the valid word
`[Emission, Coherence, Dissonance, Coherence, Silence]` to every node in
operator layers, with `validate=True`. It captures EPI, frequency, phase,
Delta NFR, Si, C(t), and each executed history after every complete layer.
All recorded values match exactly across the immutable wheel, ordinary
source, and scoped source, for every repeat at 500 and 2000 nodes. Comparison
uses compact deterministic JSON and SHA-256, without numeric rounding.

| Workload/size | Identical SHA-256 in all three modes |
|---|---|
| Jitter/500 | `3905d26ae16e32cc3ec5dbcc292bcdf30ce2e33535eb782f652ac26613e4b81b` |
| Jitter/2000 | `7c10cecc3b664a144a012390d410310ce8d9722ac27a7b7e53268e96866c3647` |
| SDK/500 | `eeb7caa1b96f1f06d4eb2d309395d85869c66cbd4a5369eda1c88fb56fba150e` |
| SDK/2000 | `c5123974b7706acb86a50defe883e117f817fce499d23ea254c410e1ef2fa4c6` |

The dedicated 12-node regression additionally repeats the same word three
times, comparing the initial state and all 15 subsequent layer checkpoints
between scoped and ordinary source. This is exact equality of 960 nodal
numeric values, 16 global C values, and all history prefixes. It is evidence
for this fixture/word/seed and the additional mixed-label/directed/multigraph
jitter tests, not a universal guarantee for all user callbacks or integrators.

## Verification and files

Before edits: root full baseline 3326 passed, 11 skipped, 97 warnings;
focused node/cache/seed baseline 149 passed. New regression file
`tests/test_stable_node_offsets.py`: 29 passed, covering ordered/cross-type
replacement, sorting changes, copied graphs, target storage replacement,
nested scopes, body exceptions, copied thread/task contexts, rejected views
and subclasses, custom node semantics, canonical telemetry, and callback
mutation rejection without rollback.

Implementation files: `src/tnfr/utils/cache.py`,
`src/tnfr/utils/cache.pyi`, `src/tnfr/operators/jitter.py`.
Evidence files: `tests/test_stable_node_offsets.py`,
`benchmarks/stable_node_offset_scaling.py`, and this report.

Final integrated operator/core-physics/SDK/cache/seed/replay verification:
**1084 passed, 14 warnings in 15.59 seconds**:

```powershell
.venv312/Scripts/python.exe -m pytest tests/operators tests/core_physics tests/sdk tests/test_nodal_lookup_consistency.py tests/test_stable_node_offsets.py tests/test_cache_ownership_and_invalidation.py tests/test_graph_seed_resolution.py tests/test_replay_state_restoration.py -q --tb=short
```

Root owns full-suite and build verification for the combined work.
