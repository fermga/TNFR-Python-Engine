# Resolution: recorded graph seeds and reproducible jitter

## Scope and contract

This resolves the remaining initialization/runtime seed contradiction from the
fourth audit. The actual configuration key is `RANDOM_SEED`, not `RNG_SEED`.
The affected invariant is reproducibility: identical initial graphs, recorded
seeds, node ordering, and ordered operations must produce identical random
choices. No nodal equation, operator channel, phase gate, or grammar rule changes.

`rng.resolve_graph_seed()` is now the shared graph-seed resolver. An explicit
integer, including zero, retains the existing stream. `None` requests one
64-bit entropy draw; the realized Python integer replaces `RANDOM_SEED` in graph
metadata. A lock protects simultaneous resolution. `base_seed()` remains a
compatibility wrapper. `validate_graph_seed()` performs the same input check
without drawing entropy or changing the graph.

Initialization, runtime candidate sampling, Coupling candidate selection,
topological remeshing, and jitter now share this resolver. Configuration and
public/low-level operator entry points validate supplied seeds before mutations.
Integer-like NumPy scalars are accepted; booleans, strings, floats (including
integral floats), NaN, and infinity are rejected. Configuration validation keeps
its `TNFRConfigError` exception; direct RNG/runtime checks raise `ValueError`.
Signed and arbitrarily large integers preserve the existing 64-bit masking, so
integers differing by a multiple of `2**64` can intentionally share streams.

## Reproduced defects and compatibility evidence

The immutable fourth-audit wheel was used as the pre-resolution implementation:

```text
tmp/fourth-audit-dist/tnfr-0.0.3.5-py3-none-any.whl
SHA256 248f10b2d7398d6d544ccaa4aaef762700a23b14a9c2bc3df2f998afd9cab15f
```

On `path_graph(60)`, initialization with `RANDOM_SEED=None` completed, but
sampling, noise-enabled Dissonance, and remeshing each subsequently failed with
`TypeError` from `int(None)`. All three now succeed and expose a recorded integer.

Two separately allocated `path_graph(12)` graphs with seed zero had identical
initialized node attributes but different jitter. The old seed included
`id(graph)`, and cache eviction/disablement could reset its progress. Its noise
stream therefore could not be reconstructed from the recorded seed alone.

The comparison fixture also exercised seed zero on `path_graph(60)`, sampling
seven nodes at step 3, and `knn` remeshing with `k=2, p_rewire=0.8`. The wheel and
corrected source returned the following identical controls:

| Control | Both implementations |
|---|---|
| Initialized node data JSON SHA256 | `ef78321b6f845ea183d3f9aefb866ec5500df7b5332e3b0dbfa4cffc0719a51b` |
| Sample at step 3 | `[25, 49, 21, 33, 30, 34, 35]` |
| Sorted remeshed edge JSON SHA256 | `e06cb6749270f1e23cc071dbcb64168781e7d912412c98f422b0860838945251` |

Hashes use UTF-8 `json.dumps`: initialized node dictionaries with `sort_keys=True`,
and lexicographically sorted edges with each endpoint pair sorted. Full fixture
and exact commands are in `tmp/resolution-seeds-repro.py`:

```powershell
.venv312/Scripts/python.exe tmp/resolution-seeds-repro.py tmp/fourth-audit-dist/tnfr-0.0.3.5-py3-none-any.whl
.venv312/Scripts/python.exe tmp/resolution-seeds-repro.py
```

The script explicitly prepends `src`; the optional wheel argument then takes
precedence and its actual import location and checksum are printed.

## Jitter recording and copy semantics

Jitter intentionally changes its formerly process-dependent stream. With
`H(seed, key)` denoting the existing 64-bit BLAKE2 seed hash, the new draw is:

```text
node_seed = H(recorded_graph_seed, canonical_node_offset)
draw_rng = Random(H(node_seed, node_draw_count))
noise = draw_rng.uniform(-amplitude, amplitude)
```

Each node records `_rng_jitter_progress = {"seed": ..., "offset": ..., "draws": ...}`.
`draws` is the next draw index. Each update replaces the entire constant-size
record; it never mutates an inherited record. Therefore NetworkX `graph.copy()`
inherits the continuation point, and the next draw on either copy does not
advance the other. This record is persistent node data, not an evictable cache.
Its keys and integer values survive JSON round trips.

The same root seed, offset, and draw index yield the same noise independently
of object identity, cache size, eviction, or cache clearing. A changed root seed
or canonical offset starts that node's new stream at index zero. Zero amplitude
does not resolve `None` or advance progress. Legacy jitter cache classes and
management functions remain callable, but clearing them no longer resets live
node trajectories.

Resolved graph copies inherit the integer seed. Copies made while the seed is
still `None` resolve independently. NetworkX views share their parent's graph
and node metadata and therefore share resolution/progress. This follows view
ownership rather than treating a view as an independent simulation.

Replay from initialization needs the recorded integer, original graph/order,
configuration, and ordered operations. Continuation additionally needs the
recorded node progress and current structural state. An explicit remeshing
`seed=` still overrides the graph seed for that call and must be recorded with
the operation. This is not an arbitrary-object checkpoint or a guarantee that
different concurrent scheduling orders give identical caller-level results.
Independent SDK/CLI topology-generator seed interfaces retain their existing
defaults; this change concerns canonical graph `RANDOM_SEED` consumers.

## Bounded performance check

Review rejected an initial graph-wide counter dictionary because validating and
copying every counter on every draw would add quadratic work to a network sweep.
The final per-node record requires constant-size validation and replacement.
An unrelated malformed node record is not inspected when another node draws.

The benchmark initializes an empty graph, performs one warm-up draw per node,
then times one complete sweep. Python 3.12.10, this Windows workspace, seed 7,
amplitude 0.1; initialization and adapter construction are outside the timer.
These are individual wall-time observations, not statistical speedup claims:

| Adapter / nodes | Fourth-audit wheel | Corrected source |
|---|---:|---:|
| `NodeNX`, 500 | 0.101072 s | 0.081082 s |
| `NodeNX`, 2,000 | 20.534746 s | 21.811411 s |
| Fixed-offset protocol adapter, 500 | 0.025737 s | 0.020723 s |
| Fixed-offset protocol adapter, 2,000 | 0.123373 s | 0.088669 s |

The fixed-offset adapter isolates jitter from the existing node-order/checksum
lookup. The full `NodeNX` path remains expensive at 2,000 nodes in both builds;
this resolution does not claim to optimize that independent cache path.
Reproduce with `tmp/resolution-jitter-benchmark.py`, passing the wheel path for
the old implementation and optionally `--fixed-offset` for the isolated path.

## Validation and changed files

Before source changes, the focused SDK/replay/operator baseline passed 81 tests
(2 warnings). Root's full pre-resolution baseline passed 3,082 tests with
11 skipped and 97 warnings in 104.54 seconds.

The final bounded integration run passed **783 tests, 13 warnings, 3.50 seconds**:

```powershell
.venv312/Scripts/python.exe -m pytest tests/test_graph_seed_resolution.py tests/operators tests/core_physics tests/sdk/test_experiment_reproducibility.py tests/test_replay_state_restoration.py tests/test_configuration_contracts.py -q --tb=short
```

The new seed suite contains 61 cases covering independent legacy hash/initial
draw-order controls, zero/negative/large/NumPy integer seeds, concurrent entropy
resolution, recorded-seed initialization and runtime replay, Coupling candidates,
shallow-copy isolation, JSON progress continuation, disabled/cleared caches,
seed/offset changes, malformed progress, and invalid-seed rejection before
initialization, sampling, operators, or remeshing mutate the graph.
Independent review also caught a removed `typing.cast` import still used by
the legacy cache settings property. The import was restored and a regression
now exercises both `JitterCache.settings` and `get_jitter_manager().settings`.

Implementation changes: `src/tnfr/rng.py` and its `.pyi` surface;
`initialization.py`; `config/defaults_core.py`, `config/tnfr_config.py`;
`dynamics/sampling.py`; `operators/__init__.py`, `operators/definitions_base.py`,
`operators/jitter.py`, and `operators/remesh.py`. Tests:
`tests/test_graph_seed_resolution.py` and the existing None-initialization case
in `tests/test_configuration_contracts.py`. No dependency, security policy,
canonical operator formula, or `manual/` content was changed by this resolution.
