# Nodal dynamics: internal reuse, correctness and scaling

## Scope and starting point

This pass follows the [remaining-contradictions resolution](RESOLUTION_REMAINING_CONTRADICTIONS_2026-09-05.md).
The pre-edit default suite passed **3,218 tests, 11 skipped, 97 warnings**, in
106.21 seconds. Previous uncommitted work is preserved, and `manual/` is outside
the work. The immutable comparison package is
`tmp/resolution-dist/tnfr-0.0.3.5-py3-none-any.whl`.

The organizing equation remains `EPI' = νf · ΔNFR`. Improvements reuse the
same node identity, conductance, field definitions and operator contracts.
They do not replace canonical operators, change U1–U6 thresholds, or claim a
new physical law. Four concrete connections were implemented:

| Internal connection | Implemented reuse | Correctness boundary |
|---|---|---|
| Stable node identity determines offsets and reproducible random streams | Ordered node snapshots validate reusable maps without rehashing every unchanged node. | Direct NetworkX changes, sorting mode, views and copied graph ownership remain checked. |
| Edge current divergence is the Dirichlet gradient | One sparse conductance/flux representation serves current, divergence, energy and weighted read-outs. | Symmetric effective conductance for the variational identity; capacity remains a separate factor. |
| Derived diagnostics contract the same base fields | Composite suites capture five base maps once and reuse canonical algebra. | Captures are local and detached; the caller must hold graph state fixed during a read-out. |
| Diffusion and its stationary measure use the same transition matrix | A shared validated row normalization drives the directed Laplacian and stationary solver. | Absorbing zero rows, finite positive measures and numerical residuals are checked explicitly. |

Details are recorded in the [node-lookup report](NODAL_SYNERGY_NODE_LOOKUP_2026-09-05.md),
[field-readout report](NODAL_SYNERGY_READOUTS_2026-09-05.md), and
[directed-transition report](NODAL_SYNERGY_DIRECTED_2026-09-05.md).

## One constitutive transport calculation

For fixed symmetric effective conductance `W`, let `d_i=Σ_j W_ij`, `x=EPI`,
and `J_ij=W_ij(x_i−x_j)`. The exact restricted relationships are

```text
div(J) = (D−W)x = ∇E_D,
E_D = ¼ Σ_ij W_ij(x_i−x_j)²,
ΔNFR_epi = −D⁻¹ div(J),
x' = −diag(νf_i/d_i) div(J),
E_D' = −Σ_i (νf_i/d_i) div(J)_i² ≤ 0.
```

Zero-strength rows use zero transport. Zero capacity freezes the nodal rate
without requiring zero pressure or current. These are EPI-channel quantities;
the tetrad charge, its currents and its quadratic energy remain different
diagnostics. Their separate algebra is preserved.

The private [conductance snapshot](../../src/tnfr/physics/_conductance.py)
records the current node order, outgoing indices, effective weights and row
strengths in `O(V+E)` storage. Parallel weights are summed in their original
numeric representation before floating-point conversion; validation applies
to their aggregate. Self-loops contribute once to strength and zero to flux.
Effective zero arcs are omitted before subtracting fields. Explicit subsets
retain the matrix API's induced-graph semantics and requested node order.

The [diffusion module](../../src/tnfr/physics/structural_diffusion.py) reuses
this snapshot for adjacency, constitutive flux and degree weights. Vector
read-outs now avoid dense matrices: `current_divergence`,
`compute_diffusion_energy`, `degree_weighted_total` and the symmetric stationary
measure. `structural_current` still returns its documented dense matrix, but
fills only actual weighted positions. Matrix-returning spectral and resistance
APIs remain dense; this is not a new sparse eigensolver or integrator.

Two reproduced numeric contradictions are corrected:

- Two edgeless nodes with finite EPI `[1e308,−1e308]` previously produced `NaN`
  currents/divergence while the Dirichlet gradient was zero. The shared edge
  calculation returns zero for all three read-outs, including a zero-weight
  arc between the nodes.
- A symmetric two-node graph with edge weight `1e308` previously overflowed
  the global degree sum and returned a zero stationary vector. Scaling the
  degree vector before normalization preserves `[1/2,1/2]`.

Independent review also caught a regression in the first snapshot draft:
converting parallel weights before addition erased the exact aggregate of
`[2**53+1,−2**53]` and could conceal a negative aggregate. Addition now precedes
conversion, matching the previous matrix convention; both signs are tested.
Finite EPI, effective conductance, representable row strengths and numerical
range are checked. Positive mobility underflow remains an explicit error.

## Measurements

Node lookup used seed 7, amplitude 0.1, one warm-up draw per node and three
timed jitter sweeps. At 2,000 nodes, median time changed from **21.3168 s to
0.2910 s** (73.26 times faster on this fixture). Warmed digest recomputations changed from **4,000,000 to zero**
per sweep, and the exact random-draw SHA-256 matched the previous wheel.
The cause was a checksum record that never acquired the reusable node snapshot,
combined with digest-cache thrashing. The fix also corrects sort-mode changes,
representation-collision replacement, checksum subsets, and copied adapter/map
ownership. Identity-based cache keys distinguish equal labels whose serialized
representations differ, including `True` versus `1.0` and nested tuples such as
`(True,)` versus `(1.0,)`. Object identity is not included in checksums or random
seeds. Ordered snapshot comparison remains `O(V)` per lookup; the measured
improvement is not a claim of constant-time mutation detection.

Canonical execution also matched the previous wheel exactly: on a 12-node
cycle with seed 7, the validated word `[Emission, Coherence, Dissonance,
Coherence, Silence]` repeated three times produced 180 node-level operator
applications and 16 checkpoints. All 960 recorded EPI, νf, phase, ΔNFR and Si
values, 16 C(t) readings and executed histories matched with maximum numerical
difference zero. The canonical JSON SHA-256 was
`bbcae94217138532629c1a49d2ab547053109031d96166f3ff1f56442c83143d`.
The full initialization, configuration and reproduction script are in the
[node-lookup report](NODAL_SYNERGY_NODE_LOOKUP_2026-09-05.md).

The unified field suite now requests the five base kernels **5 times instead
of 38**, and the variational suite **5 instead of 31**. On the documented
80-node fixture, their short warm means changed from 14.82 to 2.27 ms and
12.53 to 2.67 ms. These deterministic request-count reductions remove repeated
cache dependency checks and repeated algebra; they do not change field values.

Transport measurements used an unweighted path, `default_rng(7)` EPI sampled
from a normal distribution followed by capacities uniform in `[0.1,2]`, and
phase zero. Timing is the median of five warmed calls; peak allocation is a
separate `tracemalloc` call after garbage collection. Both implementations ran
in fresh processes. Gradient norms, energies and dissipation values matched
for these fixtures.

| Nodes | Read-out | Previous median | Current median | Previous peak | Current peak |
|---:|---|---:|---:|---:|---:|
| 500 | Current divergence | 1.590 ms | 2.007 ms | 6,014,040 B | 153,026 B |
| 500 | Diffusion energy | 3.150 ms | 2.486 ms | 2,325,312 B | 153,434 B |
| 2,000 | Current divergence | 39.544 ms | 8.117 ms | 96,050,040 B | 626,414 B |
| 2,000 | Diffusion energy | 40.693 ms | 9.918 ms | 36,099,312 B | 626,822 B |

The small divergence case is slower despite lower allocation; the sparse
representation is not universally faster. The larger fixture demonstrates the
avoided quadratic storage. Timings are local observations, not portable
performance guarantees or evidence of increased physical coherence.
Reproduction scripts:
[node lookup](../../benchmarks/nodal_lookup_scaling.py) and
[transport read-outs](../../benchmarks/nodal_transport_readouts.py).

## Remaining boundaries and useful next investigations

The subsequent [remaining-issues resolution](REMAINING_NODAL_ISSUES_RESOLUTION_2026-09-05.md)
addresses the numerical pressure and transport cases and the avoidable lookup
and spectral work identified below. This section records their state at the
end of the original synergy pass; the follow-up gives their current status.

The four implemented connections strengthen identity and numerical consistency
of existing TNFR components. They do not prove the full tetrad variational
bridge, universal confinement, grammar convergence, or state completeness;
the [shared scope note](../../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) still applies.

The canonical multichannel pressure implementation was not rewritten. Its
mean-then-subtract arithmetic can lose information at very large common EPI
offsets. On P3 with `EPI=[1e16+2,1e16,1e16+4]`, pure EPI pressure from the
current callback is `[-2,4,−4]`, whereas the edge-difference representation gives
`[-2,3,−4]`. A future pressure-kernel change needs coordinated scalar, fused,
dense and backend tests plus integration/trajectory review. The shared flux
read-out preserves the more stable differences; mathematical equality is not
claimed as bitwise equality for every floating-point input.

Likewise, a row of finite conductances can have an unrepresentable float total.
The raw directed matrix solver can still normalize it after scaling, whereas
the graph conductance read-out rejects it because it also exposes row strength.
These explicit numerical domains differ. Graph-state read-outs are synchronous
operations, not concurrency transactions. Dense resistance/spectral analysis
and per-lookup ordered snapshots remain further scaling opportunities.

## Integrated verification

The final default suite passed **3,326 tests, 11 skipped, 97 warnings**, in
115.62 seconds: 108 added regression cases compared with the pre-edit baseline.
The unchanged skips cover unavailable JAX and scikit-learn dependencies and
the Windows platform's lack of an extended `longdouble` mantissa. The default
configuration excludes slow tests. Separately, the factorization laboratory's
snapshot and seed-management suites passed **30 tests** in 1.14 seconds.
Focused counts overlap with the default suite and are not added together.

Isolated builds produced the wheel and source archive in
`tmp/nodal-synergy-dist/`. Verification parsed all **562 Python source/stub
files** (455 `.py`, 107 `.pyi`), checked their exact byte equality in both
archives, and confirmed that package name, version, Python/dependency
requirements and entry points match the immutable comparison wheel. All 134
local links in 24 audit reports resolve. `git diff --check` passes. The
canonical `AGENTS.md` and `.github/agents/my-agent.md` mirror remain unchanged
and byte-identical.

Validation commands and local evidence:

```text
.venv312/Scripts/python.exe -m pytest -q --tb=short -rs
    tmp/nodal-synergy-final-tests.log
.venv312/Scripts/python.exe -m pytest factorization-lab/tests/test_snapshot_system.py factorization-lab/tests/test_seed_management.py -q --tb=short
.venv312/Scripts/python.exe -m build --outdir tmp/nodal-synergy-dist
    tmp/nodal-synergy-build.log
.venv312/Scripts/python.exe tmp/verify_nodal_synergy_artifacts.py
git -c core.safecrlf=false diff --check
```

No commit, release or remote publication is performed.
