# Reuse of Nodal Field Readouts

## Scope and reproduced duplication

This change preserves the existing field definitions and public function
signatures in `physics/unified.py`, `physics/variational.py`, and
`physics/conservation.py`. It changes composition of read-only diagnostics,
without evolving EPI, invoking new operators, or modifying U1–U6.

The full baseline completed with 3218 passed, 11 skipped, and 97 warnings in
106.21 s. The scoped pre-change baseline passed 212 tests in 2.19 s.

Instrumentation counted requests to the five existing base-field functions:
structural potential, phase gradient, phase curvature, phase current, and
pressure flux. Repeated requests still incur cache lookup, state dependency
hashing, and result handling even when the field itself is cached.

| Entry point | Requests before | Requests after | Warm mean before (ms) | Warm mean after (ms) |
|---|---:|---:|---:|---:|
| `compute_unified_field_suite` | 38 | 5 | 14.82 | 2.27 |
| `compute_variational_suite` | 31 | 5 | 12.53 | 2.67 |
| `capture_lagrangian_snapshot` | 14 | 5 | 5.47 | 2.08 |
| `translate_sectors` | 14 | 5 | 5.44 | 2.02 |
| `capture_conservation_snapshot` | 7 | 5 | 2.79 | 2.06 |

The variational suite captured its entire Lagrangian snapshot twice, including
inside grammar-labelled diagnostics, then read fields again for threshold
analysis. The unified suite separately recomputed each derived expression and
its scalar totals. Conservation snapshot capture read the currents again for
divergence. Its snapshot-energy helper also duplicated the quadratic formula
despite documentation claiming one definition.

## Implementation and identities

A private, local readout collects the five required fields once, copying their
maps. It is never retained in graph metadata. Composite functions derive all
their outputs from those owned maps. Standalone functions still request only
the fields they need; for example, kinetic density does not incur structural
potential or coherence-length computation.

`unified.py` retains the formulas for the raw quadratic energy and bilinear
interaction. Private field-map helpers serve both standalone and composite
entry points. Conservation retains the charge and neighbor-mean divergence
formulas. The variational suite passes its captured snapshot to its existing
heuristic checks and uses the same captured fields for threshold comparisons.

The regression checks preserve these numerical identities:

- `H(i) = T(i) + V(i) = raw_energy(i)/2`, within floating-point tolerance.
- `structural_energy = sum(H(i))` and `noether_charge = sum(charge_density)`.
- Snapshot divergence equals the neighbor-mean differences of the snapshot's
  recorded currents, including successors, self-loops, and parallel neighbors.
- Every composite derived map agrees with its standalone public counterpart.

Sector-translation documentation now correctly identifies the five fields in
these expressions. Coherence length is part of the broader field suite but is
absent from this quadratic energy; no extra coherence-length fit is introduced.
The complex-field documentation states its algebraic definition without assuming
a universal correlation range. Conservation-balance documentation distinguishes
a finite-interval residual from grammar verification and identifies numerical,
topological, and source-related causes that require separate investigation.

## Validation and measurement method

The 24 new cases in `tests/physics/test_readout_reuse.py` cover exact request
counts, four NetworkX graph kinds, mixed node labels, disconnected nodes,
self-loops, parallel edges, independent public-entry-point agreement, empty
outputs and zero totals, and ownership of returned mappings. Actual Coherence
operator execution changes pressure between captures: subsequent readouts
refresh, while previously captured snapshots retain their values. A test also
substitutes field readers returning shared dictionaries to verify detachment.

The focused integration command is:

```text
python -m pytest tests/physics/test_readout_reuse.py tests/physics/test_variational.py tests/physics/test_conservation_gauge_unification.py tests/physics/test_spectral_conservation.py tests/physics/test_field_readout_consistency.py -q
```

The final focused run passed 236 tests in 2.43 s. Existing public-function
monkeypatch tests continued to pass. Whitespace validation passed.

Timing used Python 3.12.10, NumPy 2.3, and NetworkX 3.5 on the shared Windows
workspace. The fixture was `watts_strogatz_graph(80, 4, 0.2, seed=17)`; a
`default_rng(17)` assigned phase from `uniform(-1, 1)` followed by pressure
from `uniform(-0.4, 0.4)` for each node in graph order. Each entry point was
warmed, then timed for 12 calls using `perf_counter`, outside instrumentation.
These short timings are indicative measurements, not portable performance
guarantees. The request-count reductions are deterministic regression checks.

## Compatibility and limits

No public signatures, field definitions, normalization factors, result keys,
operator contracts, or cache invalidation policies change. No numerical
contradiction among the tested standalone formulas was found. The improvement
removes repeated reads and an independently maintained copy of the energy
formula.

The local readout is not a concurrency transaction. As before, callers must
hold graph state fixed while synchronously reading fields; no lock or global
rollback is added. Public mappings remain mutable, but mutations do not alter
the graph or later results. Individual calls made separately still perform
their own field requests. Existing field-kernel approximation, precision, and
mathematical-scope limits remain in force. Algebraic consistency is not a
proof that grammar implies a universal conservation or Lyapunov theorem.
