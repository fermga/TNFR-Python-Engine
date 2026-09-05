# Remaining nodal transport: scaled conductance and spectral read-outs

## Scope and baseline

This closes the graph-versus-matrix row-normalization discrepancy recorded in
the [nodal synergy audit](NODAL_SYNERGY_AUDIT_2026-09-05.md), preserves finite
weighted totals and commute times across overflowing intermediates, and removes
unused eigenvector work from existing scalar/vector diagnostics. The pre-edit default
suite passed **3,326 tests, 11 skipped, 97 warnings**, in 112.23 seconds. The
immutable comparison package is
`tmp/nodal-synergy-dist/tnfr-0.0.3.5-py3-none-any.whl`.

The changes preserve the outgoing EPI-channel convention, actual nodal
capacities, effective parallel-edge aggregation, self-loop strength and
absorbing zero-strength rows. They do not change grammar constants, operator
contracts, or the distinction between the Dirichlet energy and the tetrad
energy. The [shared mathematical scope](../../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md)
continues to apply. Pressure-kernel and backend changes are documented
separately in the encompassing resolution pass.

## One normalized conductance calculation

For a positive row of finite effective conductance, define

```text
s_i = max_j W_ij,
t_i = sum_j (W_ij / s_i),
d_i = s_i t_i,
P_ij = (W_ij / s_i) / t_i.
```

The product representing `d_i` need not fit in a float for `P` to be finite.
The private [weight normalization helper](../../src/tnfr/mathematics/_weight_normalization.py)
returns probabilities, row maxima and scaled row totals. Dense matrices,
single rows and sparse outgoing edge vectors use the same normalization.
Zero rows remain zero in this arithmetic helper; the walk reader adds their
absorbing diagonal. The NumPy handle comes from the existing optional numerical
layer, so importing the helper does not disable the scalar fallback.

The [conductance snapshot](../../src/tnfr/physics/_conductance.py) now separates
its finite edge representation from checked raw strength materialization.
Normalized graph operators and the
[directed matrix solver](../../src/tnfr/physics/directed_diffusion.py) share the
same numerical domain for finite effective weights, including overflowing raw
row sums. Matrix-returning graph APIs assemble only their final dense matrix
from edge values. The graph stationary measure uses strengths represented by
binary mantissas and exponents, without overflowing their common normalization.

For symmetric conductance, normalized coefficients are computed through
square-root ratios before multiplication. This matters at extreme scale
separation: `W_ij/d_i` may underflow while `W_ij/sqrt(d_i d_j)` is still
representable. Probabilities or final coefficients smaller than float range
round to zero; this does not assert that their exact mathematical values vanish.

Validation continues to apply **after parallel weights are aggregated in their
original numeric representation**. A finite row total is no longer mandatory
for a normalized output, but each effective edge weight must itself be finite
and nonnegative. An unrepresentable sum of parallel conductances is rejected.
Requested node subsets keep induced-adjacency semantics and node order.

## Exact witness and retained raw-output boundaries

Take a three-node star with two conductances `1e308`, EPI `[0, 1/4, -1/4]`
and unit capacities. The center has mathematical strength `2e308`, which is
not representable as a float. Before this pass, the graph Laplacian,
symmetric Laplacian, stationary measure and energy balance all raised a
row-strength error, although the raw directed matrix solver returned a valid
normalized Laplacian.

After the change, under `np.errstate(all="raise")`, the graph gives

```text
L_rw = [[ 1, -1/2, -1/2],
        [-1,    1,    0],
        [-1,    0,    1]],
pi = [1/2, 1/4, 1/4],
gradient = [0, 2.5e307, -2.5e307],
mobility = [5e-309, 1e-308, 1e-308],
EPI_rate = [0, -1/4, 1/4],
energy = 6.25e306,
energy_rate = -1.2499999999999998e307.
```

The symmetric off-diagonal coefficients are `-1/sqrt(2)`. The variational
identity remains `EPI_rate = -mobility * gradient` and
`energy_rate = gradient @ EPI_rate <= 0`. Mobility uses actual capacities;
tests also set one capacity to zero and another to two. Scaled division is
used only when raw strengths overflow, retaining the ordinary direct-division
path. Positive mobility below float range remains an explicit error.

These normalized results are not substitutes for raw quantities:

| Requested quantity | Numerical boundary |
|---|---|
| Raw row strengths | Checked materialization raises when the sum does not fit. |
| Degree-weighted total | Edge-based summation retains finite cancellation despite overflowing intermediate degrees or products; nonfinite EPI or an out-of-range final scalar raises `ValueError`. |
| Constitutive current and divergence | Finite edge fluxes and finite outgoing sums are required; a valid normalized walk does not imply finite raw flux. |
| Dirichlet balance | Returned mobility, gradient, energy and rates must remain representable; raw degree overflow alone no longer blocks a representable balance. |
| Resistance/commute implementation | Components are scaled before solving; raw volume and raw resistance need not be separately representable for commute time. Out-of-range requested reachable entries raise explicitly; infinity denotes unreachable pairs. |

For the original star, the degree-weighted total is **zero**, despite the
unrepresentable central strength. The implementation sums `W_ij * EPI_i`
directly. One-sign sums use `math.fsum`; mixed-sign contributions and
range-limited intermediates use the shared
[exact weighted-sum reducer](../../src/tnfr/mathematics/_exact_weighted.py)
also used by pressure. The reducer aligns integer ratios of the original
binary floats, then rounds only the final scalar. Raw totals do not multiply
a rounded normalized mean by an unrepresentable degree.

This policy also fixes finite product cancellation: weights `[1.5,1,1]`
and EPI `[2**53+2,-13510798882111492,0.5]` have exact total `-0.5`, whereas
summing the separately rounded products gives `+0.5`. The regression compares
the result with an independent `Fraction` calculation. Products `+2e308`
and `-2e308` can likewise cancel to an exactly representable zero. A genuinely
out-of-range final total, including nonzero values below float range, remains
an explicit error.

For the star with center EPI one and leaf EPIs zero, the normalized walk is
valid but the center's requested raw divergence overflows and is rejected.

An additional scale-separation witness uses a loop of weight `1e308` and an
incident edge of weight `nextafter(0,1)`. Its outgoing edge probability rounds
to zero, but its symmetric coefficient remains a negative representable
subnormal. The regression compares this coefficient with an independent
80-digit decimal square root.

## Resistance and commute time: cancel the common scale before read-out

Each positive-conductance component uses its largest **non-loop** weight `s`
to construct the dimensionless Laplacian `B_tilde = (D-W)/s`. Loops are removed
before this Laplacian is assembled: they contribute holding time to the walk
but do not change resistance. The component pseudoinverse gives
`R_tilde = s * R`, so

```text
R = R_tilde / s,
commute = (volume / s) * R_tilde.
```

The requested output is assembled in these coordinates. A component volume
outside float range is retained internally as an exact rational number until
its scale cancels; no public infinite raw volume is substituted into the
formula. Exceptional scaling products use exact arithmetic before final float
conversion. The normalized pseudoinverse itself remains floating point.

A two-node graph with weight `1e308` now returns resistance approximately
`1e-308` and commute time approximately **two**, despite volume `2e308`.
With weight `nextafter(0,1)`, commute time remains two even though the requested
raw resistance is too large and raises. Adding a self-loop of the same
conductance changes commute time to three at every tested common scale. The
overflowing three-node star returns commute four between center and leaf and
eight between leaves; an added isolate remains unreachable.

The distinction between intermediate and requested output range is explicit.
A loop of `1e308` with an ordinary edge of `1e-308` has representable resistance
approximately `1e308`, but its commute time is approximately `1e616` and raises.
Within-component conductance ratios that vanish during normalization are
rejected as a numerical-domain limitation. Scaling does not remove the
pseudoinverse's rank-resolution and conditioning limits or supply an arbitrary-
precision graph solver. Certificate paths that explicitly assemble raw
combinatorial matrices retain their own numerical limitations.

## Existing spectral APIs: compute values when only values are requested

`relaxation_spectrum` at common capacity, `compute_emergent_pulse` and the
Fiedler stability certificate require eigenvalues but previously requested a
full `eigh` decomposition. The pulse additionally copied and discarded the
entire eigenvector matrix through `structural_eigenmodes`.

The shared topology cache now stores an `eigvalsh` result when only values are
needed. A later mode-shape request upgrades it to a full decomposition; an
already cached full decomposition supplies subsequent value-only requests.
Public mode arrays remain detached. Capacity is read afresh and is not part
of the geometry cache. Empty, singleton and disconnected spectra are tested;
the empty pulse returns zero multiplicity and zero nonuniform modes.

The cache signature uses **effective aggregated conductance**, matching the
operator. This also corrects a concrete pre-existing stale-cache case:
parallel integer weights `[2**53+1, -2**53]` have aggregate one; changing the
first to `2**53` makes the aggregate zero, although separately converting the
first weights to float gives the same value. The two-node spectrum now changes
from `[0,2]` to `[0,0]`; a further change making the aggregate negative raises.

This is not a sparse eigensolver. Full eigenvalue calculations still construct
a dense Laplacian and use a dense numerical solver. Complete mode shapes,
current matrices, resistance and commute-time matrices each explicitly return
`N` by `N` arrays, imposing a quadratic output-size lower bound. No new sparse
public API or mandatory dependency was introduced.

## Measurements

Measurements compare fresh processes importing the saved wheel and current
`src`. Fixtures are unweighted paths with common capacity `0.75`; no random
state is used. Each timing is the median of three calls after one warm-up.
The cold spectrum cache is removed before each call. Peak allocation is a
separate `tracemalloc` call after collection. Warm pulse measurements follow
the spectrum measurement with its cache retained. The reproduction script is
[measure_remaining_nodal_transport.py](../../tmp/measure_remaining_nodal_transport.py).

| Nodes | Existing read-out | Before median | After median | Before peak | After peak |
|---:|---|---:|---:|---:|---:|
| 500 | Symmetric normalized Laplacian | 2.369 ms | 1.724 ms | 8,134,980 B | 2,106,284 B |
| 500 | Cold relaxation spectrum | 33.204 ms | 23.666 ms | 8,186,296 B | 2,241,652 B |
| 500 | Warm emergent pulse | 0.401 ms | 1.597 ms | 2,075,844 B | 236,992 B |
| 1,000 | Symmetric normalized Laplacian | 8.860 ms | 4.187 ms | 32,203,480 B | 8,210,600 B |
| 1,000 | Cold relaxation spectrum | 117.768 ms | 76.268 ms | 32,312,796 B | 8,514,056 B |
| 1,000 | Warm emergent pulse | 2.100 ms | 3.248 ms | 8,148,344 B | 504,272 B |

The value-only cache retains **4,000 instead of 2,004,000 array bytes** at 500
nodes and **8,000 instead of 8,008,000 array bytes** at 1,000 nodes. All 500
and 1,000 decay rates are returned. Maximum absolute differences from the
wheel are `1.5543122344752192e-15` and `1.7763568394002505e-15`, respectively.
Both implementations report 499 and 999 nonuniform pulse modes. The 1,000-node
vibration-energy reading changes from `500.0` to `499.9999999999999`.

Warm pulse timing is **slower** despite lower allocation: the stronger
effective-conductance signature has additional graph-reading overhead.
These local observations establish reduced unnecessary storage and unchanged
spectral content to floating-point precision, not a universal speedup or a
physical increase in coherence. Raw measurements are in
`tmp/remaining-transport-before.json` and `tmp/remaining-transport-after.json`;
the matching `.npz` files retain all compared decay rates.

## Focused verification

The dedicated file contains **47 regression cases**. With the existing
conductance, directed-transition, energy, generator, random-walk and structural
diffusion suites, **238 tests passed in 1.23 seconds**:

```text
.venv312/Scripts/python.exe -m pytest \
  tests/physics/test_remaining_nodal_transport.py \
  tests/physics/test_conductance_readout_synergy.py \
  tests/physics/test_directed_transition_consistency.py \
  tests/physics/test_diffusion_energy_balance.py \
  tests/physics/test_diffusion_generator_consistency.py \
  tests/physics/test_random_walk_consistency.py \
  tests/physics/test_structural_diffusion.py -q --tb=short
```

The one previous regression that required graph normalization to fail on an
overflowing degree now asserts the retained raw-strength error instead.
Dedicated checks cover four graph classes, subnormal/unit/overflowing scales,
loops, absorbing rows, invalid aggregated weights, actual heterogeneous and
zero capacities, absence of dense allocation for stationary/energy vectors,
cache direction changes, capacity reuse, detached mode arrays and direct edge
mutations. Additional cases cover exact finite cancellation, explicit scalar
underflow, finite-product sign reversal, scale-invariant commute times, loop
holding time, disconnected volumes and requested raw-output overflow.
`git diff --check` passes for the changed files. The enclosing
resolution report records integrated verification; focused counts overlap
with that suite and must not be added to it.
