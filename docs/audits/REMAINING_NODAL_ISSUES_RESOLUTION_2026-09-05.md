# Resolution of the remaining nodal numerical and scaling issues

## Scope and baseline

This pass follows the [nodal synergy audit](NODAL_SYNERGY_AUDIT_2026-09-05.md).
Its reference is the nodal equation `EPI' = νf · ΔNFR`, the existing operator
contracts and the canonical U1–U6 execution rules. Earlier uncommitted changes
are preserved. No changes are made under `manual/`.

The pre-edit default suite passed **3,326 tests, 11 skipped, 97 warnings**, in
112.23 seconds. The immutable comparison wheel is
`tmp/nodal-synergy-dist/tnfr-0.0.3.5-py3-none-any.whl`, SHA-256
`b0bc7f6733c08ea6bc4c95d63a8000cab4fdd745ecc3cd6b29ba0c1f7d8fbdb9`.
The environment is Python 3.12.10, NumPy 2.3.3 and NetworkX 3.5 on Windows.

The scope distinguishes correctable implementations from necessary numerical
and ownership contracts. A normalized quantity can be representable even when
an intermediate raw sum is not. A detached readout is not an atomic operation
with arbitrary concurrent graph writers. An API returning every node pair has
an intrinsically quadratic output. Neither code changes nor a finite test suite
establish the unrestricted mathematical claims excluded by the
[shared scope note](../../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).

## One pressure law for every backend and graph size

The Torch adapter previously switched at 1,000 nodes from the canonical
dispatcher to a separate tensor implementation. It accumulated incoming rather
than outgoing neighbors, ignored edge conductance, converted structural fields
to float32, used a separate topology expression and wrote a literal pressure
key instead of the shared alias writer. It also multiplied pressure by νf,
placing capacity inside ΔNFR before the nodal equation applies capacity again.

On a directed graph with 1,000 nodes, arcs `0→1` of weight `0.5` and `0→2` of
weight `1.5`, initial EPI `node_id % 3`, zero phase, and a pure EPI channel:

```text
Canonical pressure at nodes [0, 1, 2] = [1.75, 0, 0].
Previous Torch, homogeneous νf = 0:    [0, 0, 0].
Previous Torch, homogeneous νf = 2:    [0, -2, -4].
Corrected Torch, either capacity:     [1.75, 0, 0].
```

Initialization uses `set_attr` and the canonical alias groups. Capacity zero
must freeze the derivative, not erase an independently defined EPI pressure.
The isolated outgoing rows at nodes 1 and 2 have zero pressure. These values
were measured in separate processes against the immutable wheel and source,
using installed PyTorch **2.5.1+cu121** with CUDA available.

[TorchBackend](../../src/tnfr/backends/torch_backend.py) now delegates at every
graph size to `default_compute_delta_nfr`, forwarding cache, worker and
profiling arguments. The unused divergent private tensor kernel is removed.
The adapter reports CPU execution and `supports_gpu=False`; its configured
tensor device remains available for interface compatibility. This deliberately
removes an incorrect accelerated path. It does not claim a replacement GPU
pressure kernel or preserved GPU performance.

The [regressions](../../tests/core_physics/test_torch_pressure_contract.py)
initially produced **13 failures and 9 passing controls**. They cover the
999/1,000 boundary, all four NetworkX graph classes, zero and nonzero capacity,
weighted and parallel edges, loops, sinks, existing pressure aliases, seeded
mixed channels and profiling. After the correction, the combined Torch,
optimized-pressure and mathematical backend run passed **54 tests**, with
three unavailable-JAX skips. Source EPI, phase and capacity remain unchanged
by a pressure readout.

## Stable pressure under offsets and extreme relative weights

The reducer evaluates differences using each channel's existing neighborhood
convention: effective conductance for EPI and unique outgoing neighbors for
the arithmetic frequency mean.

```text
p_epi(i) = w_epi · Σ_j W_ij (EPI_j − EPI_i) / Σ_j W_ij.
p_vf(i)  = w_vf  · Σ_(j in N(i)) (νf_j − νf_i) / |N(i)|.
```

Subtracting before accumulation preserves representable differences at a
large common offset. On P3 with `EPI=[1e16+2, 1e16, 1e16+4]`, the pure EPI
pressure is now `[-2, 3, -4]` rather than `[-2, 4, -4]`. Disabled linear
channels skip their arithmetic; zero-conductance pairs are excluded from the
EPI channel before differences are formed. Phase wrapping and the topology channel keep their
existing definitions. Capacity remains outside the pressure in the evolution
equation.

The [shared difference reducer](../../src/tnfr/mathematics/_neighbor_differences.py)
serves scalar, NumPy fallback and fused paths. Ordinary vector operations use
sparse edge arrays. Rows with mixed-sign contributions, intermediate overflow
or probability/product underflow use exact arithmetic on the supplied floats,
including the channel coefficient, and round the final result. The shared
[integer-ratio accumulator](../../src/tnfr/mathematics/_exact_weighted.py)
also serves raw weighted totals. It aligns power-of-two denominators and
avoids repeated per-term Fraction reductions. This does not recover information
already lost when the initial state was converted to float.

For example, center EPI zero, neighbor EPIs `[1e308, 0]` and weights
`[1e-200, 1e200]` have representable pressure approximately `1e-92`. Rounding
the smaller normalized probability to zero first would falsely give zero
pressure. The exceptional reducer retains the contribution. A float matrix
cannot store that `1e-400` transition probability, so multiplying a separately
rounded matrix is not promised to agree bitwise with this direct evaluation.
Final pressure overflowing float range raises an error; a final pressure
itself below the representable range can still round to zero. The combined
channel result is checked for finiteness before any pressure values are
written, preserving the prior values when assembly overflows.

Independent review found and corrected finite cancellation too. Unweighted
neighbor values `[1e16, 1, -1e16]` at center zero now give `1/3` instead of
the array path's `0.5`. Values `[7e16, 1, -3e16]` with weights `[3, 1, 7]`
now give positive `1/11`; summing already-rounded normalized products had
given approximately `-3.909`, reversing the pressure sign. Exact mixed-sign
handling applies without an approximate cancellation threshold. Six edge
orders and four actual execution paths were independently checked.

The [pressure report](REMAINING_NODAL_PRESSURE_2026-09-05.md) records the
cross-path regressions, timestep-controlled integration and custom-hook scope.
Seed 17 on the specified eight-node graph, with three validated
`[Emission, Coupling, Coherence, Silence]` cycles, repeats exactly in each
version. Compared with the wheel, EPI, capacity and phase match exactly;
the maximum pressure difference is `5.55e-17`.

Warm pressure medians over 15 calls were:

| Nodes | Execution path | Previous | Corrected |
|---:|---|---:|---:|
| 80 | Fused | 0.916 ms | 2.573 ms |
| 80 | NumPy fallback | 1.055 ms | 2.147 ms |
| 1,000 | Fused | 10.269 ms | 21.015 ms |
| 1,000 | NumPy fallback | 12.288 ms | 21.331 ms |

One-sign vector reduction is linear in nodes plus edges. Grouped mixed-sign
and range-loss rows add exact integer-arithmetic cost. At 1,000 nodes the
correction costs approximately 1.7–2.0 times the previous time on this fixture.
The shared integer-ratio implementation reduces the initial per-term Fraction
implementation's approximately 40 ms to approximately 21 ms. No pressure
speedup is claimed. The actual NumPy-free fallback remains scalar.

## Shared normalization and selective spectral work

The [normalization kernel](../../src/tnfr/mathematics/_weight_normalization.py)
represents a positive row strength as `row_max · sum(weight/row_max)` and
normalizes before forming a possibly overflowing raw degree. Graph diffusion
and the directed matrix solver now use this same operation. Zero rows retain
the absorbing-walk convention and zero EPI diffusion pressure.

On a three-node star with edge weights `1e308`, the central raw degree exceeds
float range, but the normalized walk and its stationary measure
`[0.5, 0.25, 0.25]` are representable. For the documented EPI/capacity fixture,
the Dirichlet energy is `6.25e306` and the mobility, gradient and energy rate
remain finite. The [transport report](REMAINING_NODAL_TRANSPORT_2026-09-05.md)
gives the numeric domains and independent comparisons.

Raw outputs are checked at the requested result, rather than rejected solely
because a dispensable intermediate overflows. The degree-weighted total
accumulates `Σ_ij W_ij EPI_i` directly; the overflowing-degree star above has
total zero. Mixed-sign totals use the same exact accumulator as pressure,
preserving residuals and signs lost by rounded products. Nonfinite EPI and
nonzero final totals outside float range remain explicit errors.

Resistance geometry uses each component's largest non-loop conductance to
scale its Laplacian. Commute time cancels this common scale before raw
resistance or volume is materialized. A two-node edge has commute time two
both at conductance `1e308` and at `nextafter(0,1)`; the latter raw resistance
cannot fit in a float. The overflowing star has commute times four from the
center to a leaf and eight between leaves. Unreachable pairs remain infinite,
self-distance remains zero, and loops retain their holding-time contribution.
The dense pseudoinverse still has its documented numerical-rank limitations.

Rhythm and homogeneous relaxation readouts now request eigenvalues alone.
An existing full eigendecomposition can supply them; a value-only cache is
upgraded when actual eigenvectors are requested. Direct weight changes are
validated from effective parallel-edge aggregates, so integer cancellation
cannot hide a topology change behind individually rounded edge weights.

At 1,000 nodes, retained spectrum arrays fell **8,008,000 → 8,000 bytes** and
cold-spectrum peak allocation fell **32.31 → 8.51 MB**. The measured cold time
fell **117.77 → 76.27 ms**; all eigenvalues remain available, with maximum
wheel/source difference `1.78e-15`. Warm pulse reads were slower
(**2.10 → 3.25 ms**) because the stronger conductance signature costs more to
check, despite lower allocation. The full-spectrum solver and its Laplacian
remain dense. All-pairs resistance and commute APIs necessarily return
quadratic-size matrices; no sparse eigensolver is claimed.

## Explicit stable-offset traversal

The opt-in context `tnfr.utils.cache.stable_node_offsets(graph)` amortizes
jitter's node-order validation over an exclusively owned synchronous
traversal. Callers keep node membership, insertion order, label representation,
node dictionaries and sorting policy fixed; attribute values and edges may
evolve through operators. Public `NodeNX.offset` retains its complete checks.

The context checks the complete snapshot at both boundaries and checks the
target during draws. It rejects views/subclasses, separates graph/thread/task
ownership, expires on exceptions and never stores the scope on the graph.
Final contract violations raise without rolling back earlier operations;
transient mutations restored before exit are outside its guarantee. Offset
selection is constant-time for fixed nesting depth, with linear boundary
validation and temporary storage.

At 2,000 nodes, a jitter sweep changed from **0.281597 s** in the wheel to
**0.089137 s** with the scope (unscoped source: **0.299796 s**). Ordered
snapshot elements compared fell **4,000,000 → 4,000**. The
[lookup report](REMAINING_NODAL_LOOKUP_2026-09-05.md) also measures a complete
validated SDK word; its other operator costs dominate, so the jitter result
is not attributed to the entire engine. Every recorded numeric/history
checkpoint matched across wheel, unscoped source and scoped source for the
500- and 2,000-node fixtures.

## Readout ownership and mathematical limits

The [public field guide](../STRUCTURAL_FIELDS_TETRAD.md#readout-ownership-and-checkpoints)
and the public composite/snapshot docstrings now state the execution contract:
hold topology, attributes and configuration fixed for the complete readout.
Capture between completed canonical operator applications. Returned field
maps are detached from later evolution. Shared graphs require an owner lock
used by both readers and writers, or an independently owned graph copied while
writers are stopped. The engine does not claim that a reader-only lock can
protect direct NetworkX writes.

This makes the existing synchronous boundary explicit at the API, without
adding a persistent graph cache or an unverified concurrency guarantee. The
operator catalog, phase checks, grammar policies and Hz_str units are unchanged.

The full tetrad variational bridge, unrestricted grammar convergence and
state reconstruction are not made true by these engineering fixes. Existing
finite-graph counterexamples rule out the unrestricted statements; restricted
theorems require the pressure law, topology, capacity, integrator, step size
and observables to be specified. The exact symmetric EPI diffusion and energy
identities remain the applicable transport reference.

## Disposition of the previous remaining issues

| Previous item | Resolution and remaining contract |
|---|---|
| Large common EPI offsets | Shared difference arithmetic corrects every tested canonical pressure path; exact mixed-sign sums also correct cancellation and sign reversal. |
| Graph/matrix normalization domains | Shared scaled weights remove the avoidable row-sum restriction; requested raw outputs still have explicit float-range limits. |
| Repeated full node scans | An explicit stable-offset context removes the quadratic jitter scan under its ownership contract. Ordinary arbitrary-mutation checks remain enabled. |
| Dense spectral work | Value-only consumers no longer compute or retain an unused basis. A full spectrum still uses a dense solver; all-pairs return types require quadratic output. |
| Concurrent readout capture | The public API now specifies stable graph ownership and caller coordination; no unsupported atomicity guarantee is added. |
| Unrestricted theoretical claims | Existing counterexamples and restricted hypotheses remain authoritative. Universal convergence, reconstruction or variational sufficiency is not asserted. |

Custom gradient callbacks retain their contract of returning a representable
unweighted component before external weighting. Pseudoinverse conditioning,
unrepresentable final float outputs and arbitrary concurrent writers are not
silently treated as solved by the internal arithmetic changes.

## Integrated verification

The final default suite passed **3,499 tests, 11 skipped, 97 warnings**, in
122.99 seconds. This is **173 added regression cases** compared with the
3,326-test baseline. The unchanged skips cover seven unavailable-JAX cases,
three unavailable-scikit-learn modules and one platform limitation of
`longdouble`. The default test configuration excludes slow tests. The separate
factorization laboratory snapshot and seed-management run passed **30 tests**
in 1.03 seconds. Focused counts overlap and are not added to the full suite.

Isolated builds produced a wheel and source archive under
`tmp/remaining-nodal-dist/`. All **565 source/stub files** (458 `.py`, 107 `.pyi`)
parse and match their exact bytes in both archives. Package name, version,
Python/dependency requirements and entry points match the immutable comparison
wheel. All **161 local links in 28 audit reports** resolve. The complete diff
passes whitespace checks. `AGENTS.md` and its `.github/agents/my-agent.md`
mirror retain their pre-pass SHA-256 and remain byte-identical.

```text
.venv312/Scripts/python.exe -m pytest -q --tb=short -rs
    tmp/remaining-nodal-verified-tests.log
.venv312/Scripts/python.exe -m pytest factorization-lab/tests/test_snapshot_system.py factorization-lab/tests/test_seed_management.py -q --tb=short
.venv312/Scripts/python.exe -m build --outdir tmp/remaining-nodal-dist
    tmp/remaining-nodal-build.log
.venv312/Scripts/python.exe tmp/verify_remaining_nodal_artifacts.py
git -c core.safecrlf=false diff --check
```

The changes remain uncommitted. No release, installation into the main
environment or remote publication was performed.
