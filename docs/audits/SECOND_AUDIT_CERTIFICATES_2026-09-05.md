# Second audit: truthful geometric certificates

Date: 2026-09-05. Scope: `physics/variational.py`,
`physics/symplectic_substrate.py`, their tests, and the variational theory note.
This addresses T01 and T03 from the [first mathematical audit](THEORY_CONTRADICTIONS_2026-09-05.md)
and removes unsupported certification claims encountered while tracing T02.
No operator, EPI evolution rule, grammar threshold, or `AGENTS.md` was changed.

## Symplectic preservation

The previous diagnostic called `sum(abs(q*p))` a symplectic volume. It rejected
a canonical rotation from `(1,0)` to `(1/sqrt(2),-1/sqrt(2))`, and accepted the
anti-symplectic reflection `(q,p)->(q,-p)`. In the two-sector space this
reflection even has determinant +1: volume preservation alone is insufficient.

The new shared `symplectic_pullback_residual(M, n_nodes)` verifies the actual
tangent identity:

```text
residual = max(abs(M.T @ omega @ M - omega)).
```

It validates real, finite matrices of shape `(4N,4N)` and uses the existing
canonical symplectic form and node-coordinate ordering. The harmonic-flow
and U(2) checks now reuse it instead of duplicating the pullback expression.

`check_symplectic_preservation` retains its original positional arguments.
Optional keyword arguments `jacobian` and `jacobian_tolerance` provide local
tangent evidence, with a tight default tolerance independent of the legacy
snapshot-ratio tolerance. The caller remains responsible for establishing
that the supplied matrix actually is the derivative of the intended map.
One sampled Jacobian cannot establish a global theorem for an engine operator.

Compatibility behavior:

| Input | `is_canonical` | `classification` | Evidence |
| --- | --- | --- | --- |
| Snapshots only | `None` | `inconclusive` | No derivative available |
| Passing supplied Jacobian | `True` | `canonical` | Local pullback residual |
| Failing supplied Jacobian | `False` | `non_symplectic` | Local pullback residual |

Legacy ratio fields retain their numeric values and names. Their previous
classification is available as `heuristic_classification`, explicitly a
product statistic. `verification_method` and `symplectic_residual` identify the
actual evidence. `classify_operator_canonical` accepts the same Jacobian
keywords and forwards them. Energy increase or decrease does not certify
symplecticity; its historical energy-expectation fields are documented as such.

The misleading function names `compute_phase_space_volume` and
`compute_poisson_bracket_estimate` remain callable for compatibility, but their
documentation now identifies the returned snapshot/covariance statistics.
The genuine substrate Poisson bracket already accepts observable gradients.

## Singular and regular reduction levels

For `J(z)=|z|^2/2`, the zero level is `{0}` and its quotient is a point.
Previously a zero-field two-node graph received a six-dimensional
nondegenerate reduction certificate because the matrix depended only on N.

The corrected certificate reports dimension 0 and
`reduction_status="singular_zero_level"`, with `is_regular_level=False` and
`is_valid_reduction=False`. This means the regular-level theorem is
inapplicable, not that the point quotient fails to exist. Relative phases and
the determinant of a regular-level form are undefined there, represented by
`None` and NaN respectively.

At positive energy, the verifier constructs the actual horizontal tangent
space orthogonal to the level normal `z` and orbit direction `omega*z`.
An orthonormal basis B is obtained from that two-row normal matrix, and the
restricted form is `B.T @ omega @ B`. Its determinant checks nondegeneracy. The
resulting determinant is approximately 1 in this orthonormal basis; it is
not the previous `(2N)^2` determinant of a non-orthonormal basis.

Relative phases are checked only between nonzero complex coordinates, using
the largest-amplitude pair as reference and wrapped differences. A zero
action no longer supplies an undefined reference angle. Tiny positive energy
is still a regular level; numerical tolerance does not collapse it to zero.
Nonfinite data are rejected, and nonzero coordinates whose energy underflows
require rescaling instead of receiving a false zero-level certificate.

The legacy `reduced_symplectic_form_matrix(n_nodes)` remains available with its
existing values. Its documentation now states its local, regular-chart
assumptions. The actual positive-level quotient is `CP^(2N-1)`, not a globally
flat linear space; local constant forms do not establish global flatness.

## Additional bounded corrections

For the implemented quadratic potential, `V'(x)=x` and `V''(x)=1`. Nonzero
thresholds cannot be critical points merely because observed values are near
them. `analyze_potential_critical_points` now reports the correct derivatives,
`is_critical=False`, and `critical_type="regular"` at those thresholds. The
new `near_threshold_count` preserves the useful proximity observation.

Grammar-labelled variational records now explicitly carry
`verification_scope="heuristic"`. Their legacy energy/interaction comparisons
do not check operator history, U3 phase admissibility, nested identities, or
U6 drift. They are documented as diagnostics rather than grammar validators.

Two stale `variational.__all__` entries referred to definitions absent from
the repository and made wildcard imports fail. They were removed; a regression
checks that all declared public exports exist.

## Unresolved model bridges

The canonical field formulas and energy readout are preserved. Their exact
algebraic identities do not derive the full nodal equation from the tetrad
potential. On one edge in the pure EPI channel, `EPI=[1,0]` yields
`DeltaNFR=[-1,1]`, while the negative derivative of the implemented potential
is `[-2,2]`. This counterexample remains; no replacement physics was invented.

Likewise, the implemented isotropic substrate has `q''=-q`, while the separate
graph-wave model has `q''=-L_rw*q`. The graph-wave overdamped limit is valid for
that model under its damping assumptions. It does not by itself establish the
coordinate, metric, or dissipation bridge to the isotropic substrate or full
four-channel pressure. The updated module documentation and
[variational theory note](../../theory/TNFR_VARIATIONAL_PRINCIPLE.md) state these
limits explicitly. Global symplecticity of all 13 operators remains unproved.

## Validation

The focused pre-change baseline passed 141 tests. Regression coverage now
includes the canonical rotation, determinant-one anti-symplectic reflection,
a contraction fixing the zero snapshot, strict Jacobian tolerances, malformed
matrices, zero energy, positive energies at scales `1e-12`, `1`, and `1e12`,
zero-action reference pairs, wrapped relative phases, and nonzero-threshold
derivatives. Existing energy/readout and conservation tests are also rerun.
Validation passed 245 tests across the variational, substrate, and core
conservation suites, followed by the additional public-export regression.
The conservation suite emitted 11 existing coherence-antipattern warnings.
Fixtures are explicit or use existing fixed seeds; no operator trajectory was
changed, so no improvement in C(t) is claimed.

## Independent field cross-review corrections

An independent shortest-path oracle checked 16 fixtures covering Graph,
DiGraph, MultiGraph, and MultiDiGraph, both sides of the dense/streamed size
cutoff, zero and positive lengths, parallel-edge minima, and signed pressure.
The exact distance semantics agreed to maximum absolute error `1.42e-14`.
Two additional boundary failures were reproduced and corrected:

- An explicit landmark request on edges of lengths `0` and `1e-155`, with
  unit sources, could return infinite exact potentials while certifying zero
  approximation error. Both exact validation branches now share a finite-output
  guard. Zero-distance exclusion remains unchanged.
- At the center of a 51-node star, sources `1e16`, `1`, and `-1e16` summed to
  `1` in standard mode and `0` in fresh research mode on Windows, where
  longdouble aliases float64. Changing mode without clearing the cache returned
  the old result object, hiding the inconsistency. Dense exact dot products
  also lost signed residuals.

Canonical streamed sums, signed dense exact rows, and the direct dense
fallback now share `physics._helpers.compensated_sum`. It delegates to
`math.fsum` at float64 precision and the existing compensated numerical
accumulator for genuinely extended scalar types, converting only the final
sum to the existing public float result. Positive-source dense matrix
products remain available. A regression checks the analytic residual
`1e30 + 1 - 1e30 = 1` across sizes, modes, direct dense calls, and their fallback.

The central cache accepts the explicit `precision_mode` dependency. Canonical
potential, phase gradient/curvature, autocorrelation length, and aggregate
telemetry declare it. Switching back to a previous mode may reuse its valid
entry; topology-only distance caches remain independent of numerical mode.
This is scoped to these declared consumers, not a claim that every repository
cache incorporates every global setting. Likewise, exact distance evaluation
does not promise exact arithmetic, arbitrary precision, or bitwise-identical
threshold decisions: graph distances and public outputs retain their existing
floating-point representations. Explicit landmark estimates retain their
documented approximation semantics.

The final focused field/cache/variational run passed 127 tests, with one skip:
the extended-intermediate-range regression requires a longdouble mantissa
wider than float64, which this Windows runtime does not provide. The test
that forces research mode to use float64 and all signed-residual regressions
passed. The finite-action return documentation also now states only the
sampled finite trajectory result, without claiming an equivalence to U2.
