# Fourth cache ownership and invalidation audit — 2026-09-05

## Scope and evidence

This pass examined graph-local workspaces, cached adjacency requests,
hierarchical invalidation, graph-change hooks, persistent snapshots and signed
cache layers. Changes preserve the nodal equation, operator contracts, public
return types and prior uncommitted audits. The final adapter correction in
`node.py` was coordinated with the SDK audit.

The focused pre-edit baseline passed **31 tests in 0.73 seconds**:

```sh
python -m pytest tests/physics/test_field_cache_invalidation.py tests/physics/test_equivariance_cache_isolation.py tests/physics/test_field_readout_consistency.py -q --tb=short
```

The new [50 regression cases](../../tests/test_cache_ownership_and_invalidation.py)
passed against working source in **0.68 seconds**. The 47 original cases produced
**38 failures and nine passing controls in 0.58 seconds** against the immutable
third-audit wheel. Three additional defensive extension-opcode cases cover the
independent review correction described below. These are regression-case counts,
not 38 independent defect families. Controls include signed shelf round trips
for pickle protocols 0–5.

```sh
python -m pytest tests/test_cache_ownership_and_invalidation.py -q --tb=short
python -m pytest tests/test_cache_ownership_and_invalidation.py -q --tb=no -k "not extension_opcodes" -o pythonpath=tmp/third-audit-dist/tnfr-0.0.3.5-py3-none-any.whl
```

The scoped integration run passed **1,701 tests, with one skip and 23 warnings,
in 38.62 seconds**. It excluded two SDK modules whose independent fourth-pass
implementation was still in progress. Warnings concern optional JAX absence
and repeated Coherence. This is not the final repository-wide result.

```sh
python -m pytest tests/test_cache_ownership_and_invalidation.py tests/physics tests/core_physics tests/operators tests/sdk -q --tb=short --ignore=tests/sdk/test_experiment_reproducibility.py --ignore=tests/sdk/test_topology_identity.py
```

An earlier integration run, before those exclusions, had 1,710 passes and 26
failures in the concurrently added SDK modules. The final copied-adapter
dispatch assertion was then strengthened and the complete cache module rerun
successfully. Independent review subsequently added the three extension-opcode
guards. No speedup is inferred from these test runtimes.

After that final correction, the 50 cache cases plus the 31 existing field/cache
baseline cases passed together: **81 passed in 1.17 seconds**.

Environment: Windows, `.venv312/Scripts/python.exe`, Python 3.12.10, NumPy
2.3.3 and NetworkX 3.5. Standalone reproductions must prepend `src` to
`sys.path`; the wheel command overrides pytest's configured source path.
The immutable wheel's SHA-256 is
`1a8821330c7a787d284bc813a37b8e2ee6a08997b05daa4803fbc20d12439d49`.

## C01 — Copies and views reused state owned by another graph

`Graph.copy()` shallow-copies graph metadata. Although `edge_version_cache`
replaced a copied outer `EdgeCacheManager`, it reused the copied inner
`CacheManager`. A buffer request on the copy returned the original array;
writing `2` into the copy's workspace changed the original workspace from
`[1, 1, 1]` to `[2, 2, 2]`. Incrementing the copy's edge version also cleared the
original graph's cached result.

[utils/cache.py](../../src/tnfr/utils/cache.py) now binds the inner manager to
its owning metadata dictionary and creates fresh state when that owner differs.
NetworkX views share the *same* metadata dictionary, so this cache path computes
fresh results for views. A two-node subgraph view of a three-node path formerly
returned the parent's cached size `3`; it now returns `2` while the parent
retains `3`.

The centralized `GRAPH_RUNTIME_CACHE_KEYS` set identifies rebuildable cache and
adapter state for the SDK's graph-state copy helper. It includes `_node_cache`
and `_node_cache_weak`; their `NodeNX` values contain graph references and
runtime locks. Configuration dictionaries are deliberately excluded from the
runtime-key set and remain persistent data.

[NodeNX.from_graph](../../src/tnfr/node.py) now checks the graph bound to a
cached adapter and detaches a copied adapter dictionary before filling it.
Both strong and weak cache modes are covered. The execution regression applies
canonical IL through `apply_glyph_obj(NodeNX.from_graph(copy, 0), Glyph.IL)`:
the copy's pressure decreases from `0.2`, and the original remains exactly
`0.2`. The old adapter instead bound that call to the original graph.

## C02 — Cache keys omitted explicit request parameters

[buffer_cache.py](../../src/tnfr/metrics/buffer_cache.py) keyed workspaces only
by prefix, element count and buffer count. Requesting float32, integer, complex
or structured buffers after a float64 request returned float64 storage. The
key now includes normalized `numpy.dtype`; equivalent descriptions such as
`"f8"` and `numpy.float64` still reuse the same workspace.

`cached_nodes_and_A` omitted explicit node order and the sparse/dense request.
After requesting `(0, 1, 2)`, a request for `(1, 2, 0)` returned the old ordering.
A sparse request could also make a subsequent dense request return `None`.
The key now contains the actual node tuple and construction mode. Tests compare
the dense result with `networkx.to_numpy_array(..., weight=None)` independently.

[cache_utils.py](../../src/tnfr/metrics/cache_utils.py) now copies configuration
before updating it, preventing shallow-copy policy contamination. Successive
buffer and trigonometric capacity updates use the maximum of both stored
requirements: buffer capacity `256` followed by trig capacity `16` retains
shared capacity `256` instead of silently reducing it to `16`.

## C03 — Clearing the unified LRU left occupied capacity

[UnifiedLRUCache.clear](../../src/tnfr/utils/unified_cache.py) emptied the
dictionary but retained `_currsize`. Subsequent inserts could be evicted
prematurely or attempt to evict from an empty dictionary. External per-key
locks and removal callbacks were also left behind.

Clear now resets both entry and weighted-size accounting and dispatches removal
cleanup for former entries. Historical hit/miss counters retain their previous
policy. Tests cover ordinary and size-weighted caches, callback execution,
external-lock cleanup and successful insertion after clearing.

## C04 — Function invalidation and argument binding were incomplete

`invalidate_function_cache` always used the global cache and invalidated every
entry sharing the function's structural dependencies. It did nothing for
dependency-free functions or functions using a custom cache. It now targets
the wrapper's own internal dependency in the correct cache instance. Functions
sharing structural dependencies retain their unrelated entries.

Separate closures with the same module/qualified name formerly shared a result;
each wrapper now has its own cache namespace. Structural dependencies remain
available independently for graph-driven invalidation.

Graph extraction no longer assumes only the first positional argument or the
keyword names `G` and `graph`. The first explicitly supplied graph participates
in dependency hashing regardless of its parameter name. A keyword-only
`network` argument now observes an EPI update from `1` to `2` instead of returning
the previous `1`.

## C05 — Graph hooks missed bulk operations and canonical property names

`GraphChangeTracker` wrapped only four single-node/edge methods. NetworkX bulk
operations therefore retained stale hierarchy entries. The wrappers also
rejected positional multigraph edge keys.

The shared wrapper now covers single and bulk additions/removals, weighted edge
addition, `update`, `clear_edges` and `clear`, forwarding original arguments and
return values. Nested method calls count as one public operation. Invalidation
runs after exceptions as well: a malformed bulk edge list can modify its valid
prefix before NetworkX raises. Clearing metadata retains the installed tracker
reference. Weak graph membership avoids retaining obsolete numeric graph IDs.

Property notifications now match canonical alias dependencies as well as the
existing node-specific and legacy dependency names. For example, notifying a
`theta` update invalidates entries declared against `node_phase`.

## C06 — Persistent entries could survive explicit invalidation

`PersistentTNFRCache.set_persistent(..., persist_to_disk=False)` and writes to
non-persisted levels referenced an unassigned `file_path`, raising
`UnboundLocalError` after the memory write. The disk branch is now conditional.
A memory-only replacement removes its older disk snapshot so that eviction or
restart cannot resurrect the replaced value.

Dependency invalidation previously removed only memory entries. The next read
loaded the stale disk value again; the claimed lazy dependency check did not
exist. Invalidation now preserves selective memory removal and conservatively
discards disk snapshots. The legacy disk format has no authenticated dependency
index, so invalidation does not unpickle files merely to inspect metadata.
Tests verify that the invalidated value remains absent after reopening the
persistent cache.

This is an explicit tradeoff: unrelated disk entries may need recomputation;
unaffected memory entries remain available. The return count still measures
invalidated memory entries, preserving the documented API.

## C07 — Signed cache data was decoded before its interpretation was trusted

[cache_layers.py](../../src/tnfr/utils/cache_layers.py) had two distinct gaps:

1. `ShelveCacheLayer.load` used `Shelf.__getitem__`, which unpickles the outer
   shelf object before the inner signature check. A harmless tampered object
   wrote a marker file before rejection in the regression.
2. The HMAC authenticated payload bytes but omitted their raw/pickle mode.
   Changing a signed raw payload's mode could execute those bytes as a pickle.
   This affected both Shelve and Redis.

Independent review also identified that Python's registered pickle extensions
can bypass `Unpickler.find_class` through a process-wide extension cache. The
final reader therefore validates a strict bytes/memo/framing opcode allowlist
*before constructing any outer unpickler*. Extensions, globals, reducers and
persistent IDs are prohibited. Three defensive tests verify that EXT1, EXT2 and
EXT4 probes never reach that unpickler; these tests register no extensions and
construct no executing payload.

Secure shelves now write a fixed protocol-3 bytes-only outer representation.
The requested pickle protocol still controls the authenticated inner value,
preserving protocols 0–5 without allowing old-protocol global reconstruction in
the outer layer. Version-2 envelopes authenticate the interpretation mode and
payload before inner pickle decoding. Tests alter
the mode of a signed raw, harmless marker payload and verify rejection before
the marker is written. Redis tests use an in-memory fake client, not a server.
Secure Redis also rejects nonbinary, unsigned responses instead of accepting
them when `require_signature=True`.

Unsigned shelves retain their existing explicitly trusted-object behavior.
Legacy version-1 signed cache entries are rejected and must be rebuilt because
their interpretation was not authenticated. Public signer/validator call shapes
and cache-layer return types remain unchanged. These tests establish the
specified decoding boundary, not resistance to resource-exhaustion attacks or
a general-purpose safe-pickle format.

## Remaining limits

- Ordinary graph-local edge caches still follow the explicit edge-version
  protocol. Direct NetworkX mutations must use `increment_edge_version` or
  `edge_version_update`; this pass does not instrument every untracked graph.
- Graph views intentionally bypass this shared metadata cache, trading reuse
  for correct ownership. Cached scratch arrays remain mutable workspaces.
- Dependency hashing covers the recognized structural fields. Unknown custom
  dependency tokens still require explicit invalidation; this is not general
  introspection of arbitrary object state or every graph attribute.
- The runtime-key set covers the inspected cache and node-adapter modules,
  not every third-party runtime object a user might place in graph metadata.
- Persistent pickle snapshots remain trusted local data. Signed layer
  verification is separate from that legacy API. Disk invalidation is
  conservative and no cross-process transaction guarantee is added.
- This pass does not certify cache correctness under concurrent graph mutation,
  test a live Redis deployment, or claim complete coverage of every specialized
  mathematical cache elsewhere in the repository.
