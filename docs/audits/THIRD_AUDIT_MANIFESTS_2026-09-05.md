# Third audit: manifest production, graph loading, and reporting

Scope: engine and fractal-partition exporters, their SDK/CLI route, and the
promotion consumer. No EPI evolution or new mathematical identity is introduced.
All standalone diagnostics imported the repository's `src` explicitly.

## Reproduced failures and corrections

1. **Exporter/runner schema mismatch.** The two exporters advertised compatibility
   with the self-optimization pipeline, but wrote summary metadata without
   `entries` or reconstructible graph state. With both `src` and the checked-out
   factorization helpers importable, passing an exported fractal manifest to
   the SDK raised `ValueError: Manifest JSON is missing 'entries'`. Adding an
   entries list alone would still fail: the runner required a Paley modulus and
   node indices. Both producers now use
   [the shared manifest module](../../src/tnfr/engines/manifest.py) to publish
   graph payloads and an entries index. The runner consumes these payloads
   alongside its existing Paley format. Paley-only imports are lazy.

2. **Unavailable telemetry hidden as successful export.** Both exporters imported
   coherence and sense-index functions from an incorrect module, caught the
   import failure, and wrote `None` even for a finite two-node state. Their
   potential alias did resolve; its values were not implicated in this import
   failure. A shared readout now calls canonical `metrics.common.compute_coherence`,
   `compute_Si(..., inplace=False)`, and canonical structural potential. For
   per-node pressure `0.5` and velocity `0.25`, the recorded coherence is
   `1/(1+0.5+0.25)`. Metric failures remain visible under an `errors` mapping.

3. **Historical drift mislabeled as improvement.** Archived `C=0.1` and identical
   before/after snapshots `C=0.8` produced `delta_c=+0.7` during a dry run.
   The promotion helper consumed this field as an improvement. In
   [the runner](../../scripts/run_self_optimization.py), `delta_c`, `delta_phi_s`,
   and `delta_si` now mean after minus before. Historical differences are
   `manifest_*_drift`; previously archived delta fields also retain an explicit
   `manifest_` prefix. Existing snapshot-suffixed aliases remain available.
   [Promotion](../../factorization-lab/tnfr_factorization/self_opt_support.py)
   now requires a finite, strictly positive actual snapshot delta and rejects
   dry runs, in addition to its existing validation conditions.

4. **Partition limit, ordering, and zero aliases.** An entire coherent community
   could exceed `max_partition_size`; the packing loop only merged communities.
   Oversized communities are now split in stable graph order and the explicit
   capacity also caps adaptive sizing. Seeds and equal-score choices follow
   insertion order, and compensated member sums remove hash-order reduction
   differences. A three-node phase path `[0,0.8,1.6]`, threshold `0.85`, and
   capacity `2` yields `[[left,middle],[right]]` under Python hash seeds 1 and 42.
   Community node IDs preserve their scalar types, so integer `1` and string
   `"1"` are distinct. Shared alias reading preserves canonical zero frequency
   and phase. Single-node spatial queries handle SciPy's scalar return.

5. **Graph diagnostics and partition identity.** Connectivity checks failed on
   directed and empty graphs. The optimization engine now shares an empty-safe,
   directed-aware graph diagnostic kernel for recommendations and snapshots;
   directed density uses the NetworkX definition. An empty EPI sample no longer
   produces a NaN variance. Partition identity is separate from a graph-node
   validation target and is preserved in payload metadata. A stable digest in
   partition payload filenames distinguishes labels with equal sanitized forms.
   The obsolete datetime mutation and retry path were removed; it could otherwise
   retry an operation after a later snapshot failure.

6. **Entry identity and seed labels.** Duplicate entry IDs are rejected, missing
   Paley nodes are rejected rather than silently omitted, and manifest-relative
   paths are resolved from an absolute manifest directory. A seed label is offset
   by the original manifest entry index, so filtering does not change the label
   of the same partition. The engine currently treats this seed as recorded
   metadata, not an execution RNG configuration. CLI help and result `seed_scope`
   state this explicitly; no seeded-execution guarantee is inferred.

## Supported state and remaining limits

`tnfr-graph-json-v1` preserves Graph, DiGraph, MultiGraph, and MultiDiGraph,
node insertion order, scalar IDs, edge keys, weights, and built-in finite JSON
attributes, including JSON triad/history values. Unsupported Python objects,
tuple IDs, arrays, callbacks, RNG instances, non-string attribute-map keys, and
nonfinite values are rejected instead of stringified. This is a graph-state
round trip, not a checkpoint of arbitrary Python runtime or learned engine
state. JSON output is validated before publishing a bundle and individual
files use the existing atomic writer. A bundle is not a multi-file transaction;
one output directory is intended for one exported bundle.

The SDK wrappers still use the optional factorization-lab orchestration helper,
and that helper still needs the repository CLI/validator. The new graph schema
and generic CLI loader do not themselves require Paley reconstruction. No
standalone installed-wheel orchestration guarantee is added. Operation-specific
validator test mappings for pattern/fractal workflows remain absent, so their
external validation status is pending; successful loading and dry-run output
are not reported as validated optimization. Worker scheduling and persisted
timestamps are also outside the seed-label guarantee.

## Validation

Pre-change exporter suites: 10 passed, 2 skipped. The skips initially cited
missing optional helpers; explicitly loading the repository helper reproduced
the entries failure. Legacy runner baseline: 2 passed.

The permissive integration skips were replaced with actual SDK success,
partition-count, and zero-dry-run-delta assertions. Regression coverage includes
all graph kinds, mixed scalar labels, zero/parallel weights, finite-state
rejection, preserved graph attributes/history, canonical telemetry, size bounds,
single-node indexing, cross-process hash seeds, directed/empty runner inputs,
duplicate IDs, stable filtered seed labels, and promotion rejection for missing,
nonfinite, nonpositive, or dry-run changes. No coherence gain is claimed from
these readout and transport changes.

Final focused run: **55 passed**, with no skips or warnings, across manifest
contracts, both exporter SDK integrations, legacy runner tests, self-optimization
engine tests, and factorization helper tests. Whitespace validation passed.
