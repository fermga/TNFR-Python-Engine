# Telemetry, metrics, and utilities

TNFR simulations require auditable telemetry, deterministic caches, and reproducible metrics.
This guide consolidates the APIs that expose coherence data, structural histories, and helper
facades.

## Official metrics and telemetry

- **C(t)** — `tnfr.metrics.common.compute_coherence`: global stability with optional means for
  ΔNFR and dEPI/dt.
- **ΔNFR** — computed via graph hooks such as `compute_delta_nfr`, blending phase, EPI, νf, and
  topology.
- **νf** — structural frequency in Hz_str, maintained by dynamics modules.
- **Si** — `tnfr.metrics.sense_index.compute_Si`: ability to produce meaningful reorganisation
  combining νf, phase, and topology.
- **Phase θ** — `tnfr.dynamics.coordinate_global_local_phase` and related helpers.
- **Compatibility** — graphs that still expose the legacy phase alias (see
  ``LEGACY_PHASE_ALIAS`` in ``tests/legacy_tokens.py``) or the ``"θ"`` symbol must
  be rewritten manually before importing TNFR 15.0.0+. Alias helpers operate
  purely on the English ``"theta"``/``"phase"`` keys and reject untranslated
  payloads.
- **Topology** — coupling maps available through operator utilities like
  `tnfr.operators.apply_topological_remesh`.

Register telemetry callbacks before running dynamics:

```python
from tnfr.metrics import register_metrics_callbacks
from tnfr.trace import register_trace

register_metrics_callbacks(G)
register_trace(G)
```

Histories are stored under `G.graph['history']` and can be prepared with the structural history
helpers exposed by the `tnfr.glyph_history` module.

## Trace capture and callback safety

`tnfr.trace.register_trace` attaches before/after callbacks via the shared callback manager.
It records Γ specs, selector state, ΔNFR weights, Kuramoto metrics, and operator counts so every
simulation leaves an auditable trail. Callback errors are stored in a ring buffer attached to
the graph (default length 100). Adjust or inspect the buffer at runtime with
`tnfr.callback_utils.callback_manager.set_callback_error_limit` and
`get_callback_error_limit`.

### Trace verbosity presets

`G.graph["TRACE"]` accepts a `verbosity` knob that determines which field producers execute when
no explicit `capture` list is provided. The CLI mirrors these presets through the
`--trace-verbosity {basic,detailed,debug}` switch so scripted runs can stay in sync with
manual API configuration. The presets are:

- `"basic"` — captures the structural configuration (`gamma`, grammar, selector, ΔNFR/SI weights,
  callback map, THOL state) while skipping the heavier collectors. Use this for smoke tests or
  performance-sensitive runs where topology snapshots are enough.
- `"detailed"` — extends the basic payload with the Kuramoto order parameters and Σ⃗ snapshot while
  omitting glyph counts, avoiding the most expensive history walk. Pick this tier when you need
  coherence metrics without paying the full glyph audit cost.
- `"debug"` — executes the full collector suite, including glyph counts, to preserve the legacy
  trace payload. This remains the default level and is intended for investigations and regression
  hunts where complete operator coverage matters more than runtime.

If you still need a custom field mix, set `TRACE["capture"]` explicitly; the resolver will honour
that list (or mapping) and ignore the verbosity preset. Identifiers are case-sensitive and the
following capture names are recognised:

- `"gamma"` — canonical Γ specification snapshot.
- `"grammar"` — canonical grammar configuration.
- `"selector"` — active glyph selector name.
- `"dnfr_weights"` — ΔNFR mixing weights.
- `"si_weights"` — Si weighting and sensitivity payload.
- `"callbacks"` — registered callback names per phase.
- `"thol_open_nodes"` — count of nodes with an open THOL block.
- `"kuramoto"` — network Kuramoto order parameters.
- `"sigma"` — global sense-plane vector Σ⃗.
- `"glyph_counts"` (alias `"glyphs"`) — per-step glyph/operator count audit.

### Metrics verbosity tiers

The metrics orchestrator follows the same pattern via `G.graph["METRICS"]["verbosity"]`,
which is exposed on the CLI as `--metrics-verbosity {basic,detailed,debug}`:

- `"basic"` keeps the coherence and stability core (C(t), ΔSi, B) while skipping phase sync,
  Σ⃗ statistics, Si aggregates, glyph timing, and the coherence/diagnosis callback hooks. This is
  useful for lightweight runs or smoke tests.
- `"detailed"` enables `_update_phase_sync`, `_update_sigma`, and `_aggregate_si` while attaching
  the coherence observers. It deliberately skips `_compute_advanced_metrics` and the diagnosis
  callbacks so you get richer stability traces without the most expensive glyph timing jobs.
- `"debug"` retains the entire collector suite, including `_compute_advanced_metrics` and the
  diagnosis callbacks, to mirror the legacy payload. This remains the default verbosity for
  investigations that require a full glyph and diagnosis audit trail.

As with traces, an explicit override of `METRICS` parameters (for example `save_by_node` or
`normalize_series`) still applies regardless of the verbosity preset.

## Locking policy

The engine centralises reusable process-wide locks in `tnfr.locking`. Obtain named locks with
`tnfr.locking.get_lock()` and reuse them for caches, RNG seeds, and other shared resources.
Avoid scattering bare `threading.Lock` instances across modules; only short-lived objects may
instantiate ad-hoc locks when they are not shared.

## Helper utilities API (`tnfr.helpers` and `tnfr.utils`)

### Collections and numeric helpers

- `ensure_collection(it, *, max_materialize=...)` — materialise potentially lazy iterables
  once, enforcing a configurable limit to keep simulations bounded.
- `clamp(x, a, b)` and `clamp01(x)` — restrict scalars to safe ranges for operator parameters.
- `kahan_sum_nd(values, dims)` — numerically stable accumulators used to track coherence
  magnitudes across long trajectories.
- `angle_diff(a, b)` — compute minimal angular differences (radians) to compare structural
  phases.

### Structural history helpers

- `push_glyph(nd, glyph, window)` — record operator usage in the node history while honouring
  the configured window.
- `recent_glyph(nd, glyph, window)` — check whether a specific operator appears in a node's
  recent history.
- `ensure_history(G)` — prepare the graph-level history container with appropriate bounds.
- `last_glyph(nd)` — inspect the last operator emitted by a node.
- `count_glyphs(G, window=None, *, last_only=False)` — aggregate operator usage across the
  network using the full history or a bounded window.

### Graph caches and ΔNFR invalidation

- `cached_node_list(G)` — lazily cache a stable tuple of node identifiers, respecting opt-in
  sorted ordering.
- `ensure_node_index_map(G)` / `ensure_node_offset_map(G)` — expose cached index and offset
  mappings for graphs that need to project nodes to arrays.
- `node_set_checksum(G, nodes=None, *, presorted=False, store=True)` — produce deterministic
  BLAKE2b hashes to detect topology changes.
- `stable_json(obj)` — render deterministic JSON strings suited for hashing and reproducible
  logs.
- `get_graph(obj)` / `get_graph_mapping(G, key, warn_msg)` — normalise access to graph-level
  metadata regardless of wrappers.
- `EdgeCacheManager`, `edge_version_cache`, `cached_nodes_and_A`, `edge_version_update`, and
  `increment_edge_version` — encapsulate the edge version cache and bump versions for
  imperative workflows.
- `mark_dnfr_prep_dirty(G)` — invalidate precomputed ΔNFR preparation when mutating edges
  outside the cache helpers.

### Simulation best practices

- Configure histories with `G.graph['history']` and `HISTORY_MAXLEN` to cap series without
  losing traceability.
- Use random sampling (`G.graph['_node_sample']`) and the jitter cache
  (`tnfr.operators.get_jitter_manager`) to accelerate comparisons and deterministic noise.
- Coordinate edge updates with `edge_version_update(G)` to keep topology versions and derived
  caches aligned.

### Topological remeshing

Use `tnfr.operators.apply_topological_remesh` (`from tnfr.operators import
apply_topological_remesh`) to reorganise connectivity based on nodal EPI similarity while
preserving graph connectivity. Pair it with
`tnfr.operators.apply_remesh_if_globally_stable(G, stable_step_window=...)` to gate
remeshing on a minimum window of stable steps. Only the English
`stable_step_window` keyword is accepted.
Modes:

- `"knn"` — connect each node to its `k` nearest neighbours (with optional rewiring).
- `"mst"` — retain only a minimum spanning tree.
- `"community"` — collapse modular communities and reconnect them by similarity.

All modes ensure connectivity by adding a base MST.

## Additional references

- `scripts/run_tests.sh` runs the canonical QA battery (`pydocstyle`, `pytest` with coverage,
  and `vulture`).
- The `tnfr` CLI exposes subcommands such as `sequence`, `metrics`, and more. Inspect `tnfr
  --help` for the full list.
- Explore `tnfr.dynamics`, `tnfr.structural`, `tnfr.metrics`, `tnfr.operators`, `tnfr.helpers`,
  and `tnfr.observers` for domain-specific extensions.
