# Stable linear nodal pressure

This resolves the common-offset pressure defect recorded in the
[nodal synergy audit](NODAL_SYNERGY_AUDIT_2026-09-05.md). It preserves the
nodal equation `EPI' = νf · ΔNFR`, the existing channel coefficients, operator
execution and grammar gates. The pre-edit full baseline was 3,326 passed,
11 skipped and 97 warnings in 112.23 seconds. The immutable comparison is
`tmp/nodal-synergy-dist/tnfr-0.0.3.5-py3-none-any.whl`. Prior uncommitted work
and `manual/` were preserved.

## Reproduced error and numerical resolution

On an unweighted three-node path with EPI
`[1e16+2, 1e16, 1e16+4]`, zero phase and unit frequency, both the previous
default fused path and its fallback produced pure EPI pressure `[-2,4,-4]`.
The exact mean of the representable neighbor differences is `[-2,3,-4]`.
Averaging absolute values first rounded away information under the large
common offset. The same defect affected the linear frequency channel.

The shared private
[`_neighbor_differences.py`](../../src/tnfr/mathematics/_neighbor_differences.py)
now evaluates `Σ_j p_ij · (x_j − x_i)` directly. The EPI channel uses live
effective edge weights; frequency retains its arithmetic mean over unique
outgoing neighbors. Parallel edges are aggregated before float conversion,
self-loops have zero difference, and isolates or zero-strength EPI rows have
zero EPI pressure. Disabled linear channels are skipped before subtraction.

The array reducer uses the shared
[`_weight_normalization.py`](../../src/tnfr/mathematics/_weight_normalization.py)
to normalize finite nonnegative weights without overflowing their row sum.
It operates on actual active edges, with no dense all-pair difference matrix.
Rows whose differences have both signs, and rows with intermediate range
loss, use exact rational arithmetic on the input floats, including the
channel coefficient, and round only the final contribution to float. Selected
edges are grouped once, with no repeated whole-graph scan per row. Canonical
edge arrays are already grouped; arbitrary interleaved arrays may require
sorting. The one-sign ordinary array reduction uses `O(V+E)` work and storage;
a global linear-time claim does not apply to grouping and arbitrary-precision
arithmetic. Mixed signs are common in diffusion, so this is not merely an
extreme-input path.

The integer-ratio implementation is shared with raw transport totals in
[`_exact_weighted.py`](../../src/tnfr/mathematics/_exact_weighted.py).
Binary float denominators are powers of two. Aligning those denominators
permits exact integer multiplication and addition, followed by a single
`Fraction` conversion for pressure. This preserves the rational formula while
avoiding repeated intermediate Fraction reductions.

The exceptional branch is needed for both overflow and underflow:

- A directed row with center `0.9e308`, neighbors `[-1.7e308,1.7e308]` and
  weights `[0.7,0.3]` has representable pressure approximately `-1.58e308`,
  although one unweighted subtraction overflows.
- For EPI `[1e308,-1e308]`, an EPI coefficient `1e-300` produces pressure
  approximately `[-2e8,2e8]`; the unweighted component need not fit float.
- With center `0`, neighbors `[1e308,0]` and weights `[1e-200,1e200]`, one
  transition probability rounds to zero, but its pressure contribution is
  approximately `1e-92`. Returning zero would falsely indicate equilibrium.
  The reducer detects the lost probability/product and retains that term.
- Overflow of an active weighted linear contribution or of the final channel
  sum raises `ValueError` before any DNFR values are written. Nonfinite active nodal values and
  invalid effective weights are rejected explicitly. This is not a general
  transaction guarantee for preparation metadata or user callbacks.
- Final values below the smallest representable float can round to zero.
  This preserves ordinary final-underflow behavior; the explicit rejection
  policy concerns overflow/nonfinite pressure, not every nonzero rational
  outside the float range.

Independent review found that compensated addition of already-rounded products
was insufficient. With center zero and neighbors `[1e16,1,-1e16]`, the initial
array implementation returned `0.5` instead of `1/3`. With neighbors
`[7e16,1,-3e16]` and weights `[3,1,7]`, even scalar compensated addition of
normalized products returned approximately `-3.90909` instead of `+1/11`.
The final implementation selects exact arithmetic for every mixed-sign row;
it uses no approximate cancellation threshold. Both original values and
weights remain available, preserving the residual before product rounding.

Review also verified overflow during channel assembly: on a two-node path,
EPI `[-1.7e308,1.7e308]`, frequency `[0,1.7e308]`, and equal EPI/frequency
coefficients produce finite individual contributions but an overflowing sum.
All paths now validate the complete result before writing any node, including
nodes preceding the invalid row. They preserve the prior pressure values on
that rejection.

## Shared paths and compatibility

[`dynamics/dnfr.py`](../../src/tnfr/dynamics/dnfr.py) routes scalar and NumPy
fallback calculations through the shared difference reducers. Its legacy
dense accumulator no longer sums unused absolute EPI/frequency values.
[`dynamics/fused_dnfr.py`](../../src/tnfr/dynamics/fused_dnfr.py) gives the
NumPy and optional Numba phase/topology kernels the same precomputed linear
contributions. The optimized NumPy backend already delegates to this canonical
pipeline and needs no additional formula. Torch dispatch changes are covered
by the [overall resolution report](REMAINING_NODAL_ISSUES_RESOLUTION_2026-09-05.md).

Public signatures, channel normalization, directed outgoing orientation,
aliases and the placement of `νf` in integration are unchanged. NumPy-free
scalar operation and the existing parallel-worker contract remain available.
Optional example hooks also use difference-before-mean arithmetic, while
the generic custom-gradient hook contract still expects each unweighted
callback result to be representable before its external weight is applied.

Floating-point reduction order changes intentionally. Exact real-arithmetic
equivalence does not imply bitwise equality between matrix multiplication,
edge reduction and scalar summation. Matrix transition probabilities can
also underflow independently of a representable pressure term. No new
physical threshold, universal convergence result or equality claim is added.

## Validation and measured cost

The new
[`test_stable_neighbor_pressure.py`](../../tests/core_physics/test_stable_neighbor_pressure.py)
contains 75 cases. They cover large offsets in both linear channels, uniform
extreme fields, overflowing differences, tiny probabilities and coefficients,
disabled channels, zero edges, rejected unrepresentable pressure, all four
NetworkX graph kinds, parallel weights, loops, mixed aliases, isolates,
NumPy/Numba dispatch and nodal integration including zero frequency. Added
cancellation regressions exercise both counterexamples across five paths,
shuffled/interleaved weighted rows, 40 seeded comparisons with an independent
Fraction oracle, and final-channel overflow before any write.

Those cases and the existing computation-path, backend, nodal-equation,
Python-fallback and orientation suites passed: **140 tests in 4.14 seconds**.
The integration witness uses one explicit Euler step (`DT_MIN=0`) so that
subdivision into increments below the EPI unit in the last place does not
confound the pressure comparison. It verifies `dEPI=νf·ΔNFR` and zero-capacity
freezing without changing the integrator.

The isolated before/after script is `tmp/remaining_pressure_evidence.py`;
its JSON outputs are `tmp/remaining-pressure-before.json` and
`tmp/remaining-pressure-after.json`. Timing uses seed 17, Watts–Strogatz
graphs with degree parameter 4 and rewiring 0.2, ordinary finite nodal values,
three warm-ups and 15 calls. These are local median measurements, not a
performance guarantee.

| Nodes / path | Before, ms | After, ms |
|---|---:|---:|
| 80 / fused | 0.916 | 2.573 |
| 80 / fallback with NumPy | 1.055 | 2.147 |
| 1,000 / fused | 10.269 | 21.015 |
| 1,000 / fallback with NumPy | 12.288 | 21.331 |

Fallback reuse avoids a separate scalar traversal when NumPy is available.
Nevertheless, unconditional exact handling of mixed-sign rows increases cost:
the final implementation is approximately 2.0–2.8 times slower at 80 nodes and
1.7–2.0 times slower at 1,000 nodes on this fixture. Naive per-term Fraction
arithmetic measured approximately 40 ms at 1,000 nodes; shared integer-ratio
arithmetic reduces that to approximately 21 ms. The implementation accepts
this measured correctness cost rather than introducing an unverified
approximate cancellation threshold. It does not claim a pressure speedup.

For trajectory compatibility, an eight-node graph with seed 17 and initially
aligned phases executes three validated words
`[Emission, Coupling, Coherence, Silence]`. Both versions repeat exactly for
the same seed. Across all 12 recorded operator steps, EPI, frequency and phase
match the prior wheel exactly; the largest DNFR difference is
`5.551115123125783e-17`. The complete trace agrees with `rtol=atol=1e-12`.
U3 and canonical word validation remain active. This finite fixture does not
prove bitwise compatibility for every trajectory or every optional backend.
