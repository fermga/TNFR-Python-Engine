# Nodal lookup: correctness and measured cache reuse

## Structural scope

This work resolves the node-lookup bottleneck isolated in the seed-resolution
report. Node offsets select deterministic noise streams, which feed operators
such as Dissonance in the structural-pressure channel. Node index maps also
associate graph nodes with the arrays used by nodal evolution. Reusing an
incorrect order can therefore assign a random stream or array position to the
wrong node. Preserving those associations supports the nodal equation and
reproducibility invariant without changing any operator formula, physical
parameter, phase gate, grammar rule, or unit.

The implementation scope is `src/tnfr/utils/cache.py` and `src/tnfr/node.py`.
The existing public functions and their arguments are unchanged; their `.pyi`
signatures need no change. All earlier uncommitted work and `manual/` are
preserved. This is a bounded correction of the node-list/checksum path, not
a general replacement of the cache infrastructure.

## Reproduced root cause

The immutable baseline was:

```text
tmp/resolution-dist/tnfr-0.0.3.5-py3-none-any.whl
SHA256 a7e7236a2686f4b452ca680ae86bce0e0385dd3e4e4bdcf196b9372e67e9e719
```

The initial node-list calculation used the explicit-node checksum path, which
stored `(token, checksum)`. Later lookups used the implicit-node path, which
could reuse only a three-field record containing a node-set snapshot. When the
new digest matched the old token, an early return left the two-field record
unchanged. Consequently every offset lookup serialized/sorted/hashed the whole
graph again. Above the 1,024-entry node-digest LRU capacity, sequential graph
sweeps also evicted every digest before its next use.

For one warmed sweep over 1,100 nodes, the baseline wheel profile recorded
1,210,000 node-digest computations and JSON serializations: exactly `N**2`.
It took 14.504 seconds under `cProfile`, with 14.342 seconds inside
`NodeNX.offset`. Profile overhead is excluded from the final controlled timings
below; earlier draft timings do not describe the final identity checks.

## Correctness corrections

The initial 24 regression cases reproduced **18 failures and 6 passing
controls** before source edits. They demonstrated more than unnecessary work:

- Removing and reinserting a node preserved the node set and its digest but
  changed NetworkX iteration order. Cached node lists and offsets stayed stale.
- Distinct node objects with identical serialized representations could replace
  each other without changing a checksum; a digest alone could not establish
  node identity.
- Changing `SORT_NODES` did not rebuild an existing offset map. Index maps and
  offset maps have different ordering requirements and must retain that
  distinction.
- `presorted=True` and `False` checksum calls shared a cache identity even when
  their digest order differed. A cache hit also bypassed `store=False` removal.
- Constructing `NodeNX` directly on a shallow graph copy inserted its adapter
  into the original graph's inherited cache. Separate tests also reproduced
  this contamination with an empty inherited adapter cache in both constructor
  and `from_graph` entry points.

The corrected behavior is:

1. A node-list lookup obtains one ordered tuple of current graph nodes and
   compares actual objects pairwise by identity with the stored tuple.
   Equality alone would conflate replacements such as `1`, `True`, and `1.0`.
   Unchanged structure reuses the derived
   maps directly, with no digest or sorting work. The same snapshot is passed
   to the rebuild path when a change is detected.
2. NodeCache records weak graph ownership. Ordinary graph copies rebuild their
   own mutable maps; filtered views validate their actual visible node order.
   The offset map records its sorting mode, while the index map keeps graph
   iteration order. Lexicographic stable-representation sorting remains the
   existing convention, including mixed-label graphs.
3. Explicit and implicit checksum calls share one implementation. The record
   contains the digest, ordered snapshot, and `presorted` mode. Legacy two- and
   three-field records are rebuilt into the new representation even when their
   digest is unchanged. A subset cannot stand in for a later full-graph request.
4. Both NodeNX construction paths detach foreign or empty adapter-cache
   dictionaries before inserting an adapter.

Independent review exposed an additional identity boundary in the digest LRU.
`typed=True` distinguishes scalar types but does not distinguish equal nested
tuples such as `(True,)` and `(1.0,)`. The final 1,024-entry digest cache therefore
uses a small key holding the actual node, hashing its object identity and
comparing with `is`. Keeping the node alive while its entry exists prevents
identity reuse. Object identity never enters digest bytes or RNG seed material.
The cache retains its bounded capacity and existing diagnostic/clear interfaces.
Tests also cover equal immutable custom labels with different representations.

Weak ownership must not make an otherwise picklable node-list cache fail to
serialize. Self-review caught that consequence and added a reduction method
which preserves cache data while omitting runtime ownership. The first lookup
after restoration binds a fresh cache to the restored graph. A regression covers
this specific formerly supported pickle round trip; it does not claim arbitrary
graph objects or runtime callbacks form a portable checkpoint.

## Controlled scaling and exact trajectories

The tracked experiment is
[benchmarks/nodal_lookup_scaling.py](../../benchmarks/nodal_lookup_scaling.py).
It explicitly prepends the repository `src` directory, or places the selected
immutable wheel ahead of it, and prints the imported module location and wheel
checksum. It does not install either implementation.

```powershell
.venv312/Scripts/python.exe benchmarks/nodal_lookup_scaling.py --wheel tmp/resolution-dist/tnfr-0.0.3.5-py3-none-any.whl
.venv312/Scripts/python.exe benchmarks/nodal_lookup_scaling.py
```

Environment: Python 3.12.10, NetworkX 3.5, this Windows workspace. Each size uses
an empty graph in integer insertion order, seed 7, amplitude 0.1, one warm-up
draw per node, then three timed complete sweeps. Graph/adapter construction,
warm-up, digest-counter reading, and trajectory hashing are outside the timers.

| Nodes | Baseline sweep times (s) | Corrected sweep times (s) | Median baseline / corrected |
|---:|---|---|---|
| 500 | 0.0837114, 0.0902435, 0.0824573 | 0.0357638, 0.0359758, 0.0439360 | 0.0837114 / 0.0359758 s (2.33×) |
| 2,000 | 20.7584645, 21.3305946, 21.3168247 | 0.2909867, 0.3262488, 0.2856199 | 21.3168247 / 0.2909867 s (73.26×) |

At 500 nodes the baseline made 250,000 digest-cache hits per sweep; at 2,000
it recomputed 4,000,000 digests per sweep. The corrected warmed sweeps made
zero digest-cache hits and zero misses at both sizes. This counter evidence
explains the discontinuity above the old LRU capacity and the measured result.

The experiment hashes every timed noise value as a big-endian IEEE-754 double,
in sweep/node order. Baseline and corrected implementations produced identical
SHA256 values and ended with four recorded draws per node:

| Nodes | Both trajectory SHA256 values |
|---:|---|
| 500 | `3d65efd6ece43560ac5a1d308e284227078efbc9c3f9fba58d346148da47f3f3` |
| 2,000 | `580062aee78cb717a0c1a8bfbe56c1219e5d2a30411c4f78396baac488405dfc` |

These are measured improvements for the specified node-lookup/noise workload,
not whole-engine speedup claims. The full ordered tuple comparison remains
`O(N)` per lookup and `O(N**2)` over a sweep. It deliberately preserves detection
of direct NetworkX mutations and dynamic views instead of assuming callers use
TNFR version hooks. The expensive repeated serialization, sorting, and LRU
thrashing are removed. Further asymptotic improvement would need an explicit,
verified mutation/snapshot contract.

Unchanged graph/order/seed trajectories remain identical. Reordering or toggling
the sorting policy now selects offsets matching a freshly constructed graph
with the same current order; this intentionally corrects the stale behavior.
Arbitrary user-defined node labels still need stable equality/hash semantics,
as NetworkX requires, and stable representations while their digest is cached.
Checksums remain representation-based diagnostics; actual object identity and
order, rather than checksum injectivity, protect map reuse.

## Validated canonical nodal execution

A separate comparison exercised actual operators through the shared validated
SDK execution primitive, rather than only calling jitter. The fixture was
`cycle_graph(12)`, seed 7, phase zero at every node, vacuum EPI, initialized
uniform frequencies in `[0.4, 0.7]` Hz_str, `OZ_NOISE_MODE=True`, and
`OZ_SIGMA=0.1`. The trace window was 20 to retain all 15 executed steps.

The word `[Emission, Coherence, Dissonance, Coherence, Silence]` ran three times
through `tnfr.sdk.simple._run_network_sequence(..., validate=True)`, with every
operator applied to all 12 nodes before advancing. The initial state and all
15 complete operator layers were recorded: 180 actual node/operator applications,
16 checkpoints. At each checkpoint, records included every node's EPI, frequency,
phase, pressure, computed Si, and executed glyph history, plus global C(t).
`compute_Si(..., inplace=False)` kept measurement from writing Si into the graph.

The immutable wheel and final source records were **exactly equal**, without
a tolerance: maximum absolute difference was `0.0` across all 960 recorded nodal
numeric values and all 16 global coherence values. Every node's history matched
the requested prefix after every layer; there were no fallback substitutions.
The full compact, sorted-key JSON records had SHA256:

```text
bbcae94217138532629c1a49d2ab547053109031d96166f3ff1f56442c83143d
```

The following summaries are rounded for display; equality was checked on the
unrounded complete records. Phase remained exactly zero at every node.

| Checkpoint | C(t) | Mean EPI | Mean frequency (Hz_str) | Mean absolute pressure | Mean Si |
|---|---:|---:|---:|---:|---:|
| Initial | 1.0000000000 | 0 | 0.5654409856 | 0 | 0.8586802079 |
| Word 1 | 0.9209043928 | 0.0795774715 | 0.5204446217 | 0.0858890541 | 0.8311500069 |
| Word 2 | 0.9281361458 | 0.1591549431 | 0.4790289546 | 0.0774281387 | 0.8314712300 |
| Word 3 | 0.9165020089 | 0.2387324146 | 0.4409090416 | 0.0911050824 | 0.8267194065 |

The final history at every node was `[AL, IL, OZ, IL, SHA]` repeated three times.
This verifies the specified canonical trajectory and telemetry; it does not
establish parity for arbitrary words, graph mutations, or separate time
integrators. No additional pressure-recomputation callback was installed.

Reproduce from the repository root with this script, passing the wheel path as
the optional first argument for the baseline, then running without it for source:

```python
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
if len(sys.argv) > 1:
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import networkx as nx
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR
from tnfr.initialization import init_node_attrs
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si
from tnfr.sdk.simple import _run_network_sequence

graph = nx.cycle_graph(12)
graph.graph.update(RANDOM_SEED=7, INIT_RANDOM_PHASE=False,
    INIT_EPI_VALUE=0.0, INIT_VF_MODE="uniform", INIT_VF_MIN=0.4, INIT_VF_MAX=0.7,
    OZ_NOISE_MODE=True, OZ_SIGMA=0.1, GLYPH_HYSTERESIS_WINDOW=20)
init_node_attrs(graph)
records = []
def record(operator):
    sense = compute_Si(graph, inplace=False)
    records.append({"step": len(records), "operator": operator,
        "C": compute_coherence(graph), "nodes": [
            {"id": node, "EPI": float(get_attr(attrs, ALIAS_EPI, 0.0)),
             "vf": float(get_attr(attrs, ALIAS_VF, 0.0)),
             "phase": float(get_attr(attrs, ALIAS_THETA, 0.0)),
             "dnfr": float(get_attr(attrs, ALIAS_DNFR, 0.0)),
             "Si": float(sense[node]), "history": list(attrs.get("glyph_history", []))}
            for node, attrs in graph.nodes(data=True)]})
record("initial")
word = ["emission", "coherence", "dissonance", "coherence", "silence"]
_run_network_sequence(graph, word, cycles=3, validate=True, on_step=record)
payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
print(hashlib.sha256(payload.encode()).hexdigest())
print(payload)
```

## Validation

Before edits, the focused cache/seed baseline passed 111 tests. Root's full
baseline passed 3,218 tests with 11 skipped and 97 warnings in 106.21 seconds.

[tests/test_nodal_lookup_consistency.py](../../tests/test_nodal_lookup_consistency.py)
now contains **38 passing cases**. Coverage includes repeated-work counters,
legacy checksum records, four NetworkX graph kinds, same-size replacement and
reordering, equal-representation labels, sorting modes, copy ownership,
filtered views, checksum subset/mode isolation, `store=False`, graph lifetime,
specific cache pickle compatibility, reordered-graph jitter parity, scalar and
nested-label identity, equal immutable custom labels, and bounded cache interfaces.

The final bounded integration run passed **977 tests, 14 warnings, 14.41 seconds**:

```powershell
.venv312/Scripts/python.exe -m pytest tests/operators tests/core_physics tests/sdk tests/test_nodal_lookup_consistency.py tests/test_cache_ownership_and_invalidation.py tests/test_graph_seed_resolution.py tests/test_replay_state_restoration.py -q --tb=short
```

Warnings were existing grammar anti-pattern diagnostics and the optional JAX
import warning. The final focused cache/seed/lookup run passed 149 tests in 0.95 seconds.
No dependencies or canonical physics parameters were changed.
