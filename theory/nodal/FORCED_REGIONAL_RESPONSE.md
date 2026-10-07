# Forced regional response and environmental readouts

Regional balances, represented phase reduction and environmental-input geometry.

Part of [Forced support balance and model boundaries](../FORCED_SUPPORT_BALANCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 7. A region and its environment on the same nodal support

The full-network balance does not by itself identify which mechanism supports
a particular region. For a fixed proper nonempty node set `R`, retain the
**full-graph** conductance, strengths `d_i`, positive capacities and `e>0`
from section 1. Connectivity is unnecessary for the identities below, but
every full strength must be positive. Removing the environment and
renormalizing an induced graph changes the model and is not this observation.

Let `h_i=d_i/nu_i`, `Z_R=sum_R h_i`,

$$
M_R=\sum_{i\in R}h_i x_i,\qquad m_R=M_R/Z_R,\qquad
z_i=x_i-m_R,\qquad V_R=\frac12\sum_{i\in R}h_i z_i^2.
$$

`M_R` is a weighted EPI total, with no identification as physical mass.
`V_R` measures internal EPI contrast about the region's own weighted mean;
neither quantity alone defines coherence, identity or biological activity.
For an edge oriented from the region to its complement, define the outward
EPI current `J_ij=W_ij*(x_i-x_j)`. Direct substitution of the nodal equation
and cancellation of opposite internal currents give

$$
\dot M_R=-e\sum_{i\in R,j\notin R}J_{ij}
              +\sum_{i\in R}d_iF_i.
$$

The complementary region receives the opposite cut current. Since
`sum_R h_i*z_i=0`, differentiating the regional mean contributes zero to
the fixed-metric variance derivative. Pairing internal edges yields

$$
\dot V_R=
-e\sum_{\{i,j\}\subset R}W_{ij}(x_i-x_j)^2
-e\sum_{i\in R,j\notin R}W_{ij}z_i(x_i-x_j)
+\sum_{i\in R}d_i z_iF_i.
$$

The first sum counts each internal undirected edge once. It is nonpositive.
The boundary and non-EPI source terms are signed: either can supply or remove
regional contrast. Loops contribute to `d_i` but carry zero EPI current.
Zero-weight support edges contribute no EPI current; their influence on the
other canonical channels remains in the independently captured `F`.

For stored pressure `p`, retain
`epsilon_i=p_i-(-e*(Bx)_i/d_i+F_i)`. The stored-rate balances add
`sum_R d_i*epsilon_i` and `sum_R d_i*z_i*epsilon_i`, respectively.
When a forcing observation is available, split this discrepancy into its
fresh-kernel arithmetic defect and stored-minus-fresh pressure residual.
Do not redefine `F` from the observed derivative to remove either term.
Linearity also separates the phase, capacity and topology contributions.

These identities use one fixed region and metric. If support or capacity
varies continuously, differentiation includes metric-variation terms; birth,
membership changes and discrete events require their own endpoint budgets.
An instantaneous equality is not a finite-time restoration or invariance
theorem. Even exactly balanced source and loss at one instant do not prove
that the conditions supplying that balance persist.

The single executable owner is
[`observe_regional_support_balance`](../../src/tnfr/physics/support_transport.py).
It rebuilds snapshot caches, preserves the full source node order and accepts
an explicit ordered region plus an independently supplied source vector.
The observer checks four exact identities: model and stored-pressure rates
for both weighted total and variance. It does not evolve the graph or choose
its region. Public detached records carry no causal execution seal.

The independent
[`regional tests`](../../tests/physics/test_regional_support_balance.py) include
a three-node path with `x=(0,1,5)`, unit edges, capacities `(1,2,1)`, `e=1`,
`F=0` and `R={0,1}`. Here `h_R=(1,1)`, `m_R=1/2`, internal dissipation is `1`,
boundary work is `2` and `dot V_R=1`: internal diffusion can coexist with
growing regional contrast. Complementary total fluxes cancel. This control
demonstrates why whole-network smoothing cannot replace a regional budget.

### Conditional child response is not an autonomous child region

For a selected child set `B` with no internal edges or self-loops, fixed
external EPI and the same held canonical source give, for each child,

$$
\dot x_i=-e\nu_i(x_i-x_i^*),\qquad
x_i^*=\frac{\sum_{j\notin B}W_{ij}x_j}{d_i}+\frac{F_i}{e}.
$$

The conditional relaxation rate is `e*nu_i`. This is the diagonal block of
the existing nodal generator, without an added restoring force. If the
environment changes, this conditional profile changes too and is not a
fixed future target. Dependence on the environment is consistent with a
relational NFR; the open question is how the coupled system generates and
maintains the relevant region and its supporting boundary conditions.

## 9. Finite regional observation with a held nodal rate

On the same full support and positive metric as section 7, hold the **stored**
nodal rate `r_i=nu_i*p_i` for one declared duration `dt>=0`. For an observed
endpoint `x_plus`, let `y=x+dt*r` and `delta=x_plus-y`. Both are detached
algebraic coordinates; constructing them does not execute a nodal step.
Write `mean_R(v)=sum_R h_i*v_i/Z_R` and `v_c=v-mean_R(v)` on the region.
The exact finite identities are

$$
M_R(x_+)-M_R(x)=dt\,\dot M_R(x)+\sum_{i\in R}h_i\delta_i,
$$

$$
V_R(x_+)-V_R(x)=dt\,\dot V_R(x)
+\frac{dt^2}{2}\sum_{i\in R}h_i(r_c)_i^2
+\sum_{i\in R}h_i(y_c)_i(\delta_c)_i
+\frac12\sum_{i\in R}h_i(\delta_c)_i^2.
$$

The dotted terms use the stored-pressure balances of section 7, including
the independently identified canonical source and stored-minus-model
pressure discrepancy. That discrepancy may contain an intentional IL
pressure contraction; it is not automatically floating-point error.
The quadratic rate term is nonnegative. Endpoint discrepancy can contain
rounding, clipping or a different supplied endpoint and remains signed in
its linear cross term. Uniform endpoint shifts change the weighted total
but not centered variance. Changed capacity, support or region invalidates
this fixed-metric identity and requires additional budgets.

This identity supplies finite accounting, not solver authentication or a
future stability theorem. An offline runtime application must separately
bind the starting pressure, actual duration, integration boundary and later
events. In particular, a phase update after integration can change the
future canonical source while leaving the just-completed EPI increment
unchanged; it must not be retroactively substituted into the consumed rate.
The interruption checkpoint and frozen regional observation contract are in
the [single execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md).

The shared executable owner is
[`observe_regional_support_euler`](../../src/tnfr/physics/support_transport.py).
It reuses the instantaneous regional observer, rebuilds both snapshots and
rejects changed node order, full conductance, support or capacity. Its
[`24 focused tests`](../../tests/physics/test_regional_support_euler.py) cover
direct heterogeneous examples, a mean-only uniform shift, clipping, zero
duration, invalid metric/support changes and detached cache tampering.
Together with the 79 existing support/forcing tests, 103 tests pass.

### Regional contrast is not a universal coherence score

`V_R` measures dispersion about a region's current weighted mean. It is not
the canonical `C=1/(1+|DeltaNFR|+|dEPI|)`, nor error relative to a predeclared
nonuniform pattern. Boundary forcing can create a structured difference
between children while diffusion dissipates internal differences. Increasing
`V_R` therefore need not be instability, just as decreasing it need not be
restoration of a particular form. A finite signed budget identifies those
mechanisms without defining an NFR by one favorable scalar trend.

For example, when two independent children have the same fixed capacity,
their conditional equation from section 7 gives
`dot(x_i-x_j)=-e*nu*((x_i-x_j)-(x_i_star-x_j_star))` under held external
inputs. A nonzero target contrast can cause the squared contrast first to
decrease and then increase when the signed difference crosses zero. This
is an algebraic property of the stated stable scalar response, not a proof
that the actual moving environment follows the held-input model. Actual
IL pressure writes and finite-step defects must still be retained.

## 11. Phase-source relevance at a fixed regional state

To distinguish a phase-readout difference from a difference in the nodal
drive, hold `x`, `nu`, `W`, `e`, the channel weights and region fixed. Let two
declared phase arrays produce captured canonical sources `F_a` and `F_b`.
The exact source components from the shared forcing owner use represented
unit phase gradients; their trigonometric evaluation is not replaced by a
linearized phase model. With `Delta F=F_b-F_a`, the model pressure and nodal
rate differences are

$$
\Delta p_{\mathrm{model}}=\Delta F,\qquad
\Delta\dot x_{\mathrm{model}}=\operatorname{diag}(\nu)\Delta F.
$$

Since the state, region and metric have not changed, neither have their
mean, internal dissipation or boundary work. Section 7 immediately gives

$$
\Delta\dot M_R=\sum_{i\in R}d_i\Delta F_i,\qquad
\Delta\dot V_R=\sum_{i\in R}d_i z_i\Delta F_i.
$$

When only phase changes, `Delta F=Delta F_phase`; capacity and topology
source differences vanish. These identities reuse the full-graph regional
owner and the independently captured source. They do not reconstruct a
force from an endpoint derivative or introduce a new evolution law.

The fresh represented kernel also has a discrepancy `epsilon_kernel`
relative to this model, so `Delta p_fresh=Delta F+Delta epsilon_kernel`.
If the recorded stored pressure is retained, its actual nodal rate remains
unchanged in both detached readings. Its stored-minus-fresh residual changes
by `-Delta p_fresh`. Thus a changed candidate fresh pressure is not a changed
historical increment or an observed alternative future. A regional sum can
also vanish despite a nonzero pointwise source difference; retain both.

The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md) fixes two already
archived phase alternatives, the original endpoint and nine existing regions.
The baseline fresh capture must first reproduce the archived capture exactly.
The test concerns the relevance of an established enumeration ambiguity,
not another phase-coordination experiment or a maintenance theorem. An
exact matching baseline validates that retained input on the current numeric
path; it does not make a two-input comparison a uniform robustness bound.

### Retained two-phase comparison

[`thol_phase_source_relevance.py`](../../benchmarks/thol_phase_source_relevance.py)
implements that fixed protocol. It authenticates the native, phase-audit and
regional-identity inputs, binds the complete original coordination boundary
and checks the saved label/enumeration projections. Saved output phases are
already aligned to original node identities. Both fresh captures use the
original endpoint node and neighbor order, with the retained cached channel
weights and admitted default NumPy pressure branch. The full baseline
capture reproduces the archive exactly before the alternative is evaluated.

The alternative phase array changes the phase-source component at seven of
sixteen nodes. Its largest absolute difference is approximately
`0.04914323880773078`; the largest difference in the fresh-kernel arithmetic
discrepancy is approximately `2.610120553818296e-17`. These are separate
observed quantities, not a universal floating-point error bound. Capacity
and topology contributions, regional metrics, boundary work and internal
dissipation remain fixed. All nine source-difference identities close exactly.

| Fixed region | Baseline model variance rate | Alternative model variance rate | Difference |
| --- | ---: | ---: | ---: |
| Pair 0 | `-0.399116639750073` | `-0.399116639750073` | exactly `0` |
| Pair 1 | `+0.007133454707669` | `+0.000771577106453` | `-0.006361877601216` |
| Pair 2 | `-0.400949399259614` | `-0.397283880240531` | `+0.003665519019083` |
| Pair 3 | `+0.000771577106453` | `-0.005590300494763` | `-0.006361877601216` |
| Pair 4 | `-0.399116639750073` | `-0.399116639750073` | `+7.997873694454377e-18` |
| Pair 5 | `+0.000771577106453` | `+0.000771577106453` | exactly `0` |
| Pair 6 | `-0.399116639750073` | `-0.399116639750073` | exactly `0` |
| Pair 7 | `+0.000771577106453` | `+0.000771577106453` | `-2.14527927667545e-17` |
| All actual children | `+0.005813679258035` | `+0.007568908697630` | `+0.001755229439595` |

The pair-3 model rate changes sign. The child cohort's model weighted-total
rate also changes, from approximately `1.0523570656281966` to
`1.1735385705832913`. Tiny nonzero differences are retained rather than
classified by a fitted tolerance. The complete captures preserve the other
weighted-total rates and every exact coefficient.

Consequently, source invariance under these two previously archived phase
outputs is **rejected in scope**. The known enumeration ambiguity can change
the modeled regional drive, not only its tetrad display. This comparison
does not rerun the coordination algorithm: it connects the earlier finite
ordering witness to fresh pressure at one fixed endpoint. All historical
stored nodal rates remain exactly unchanged, including the pair-3 stored
variance rate; the sign change above concerns the candidate fresh model.
It is not a new trajectory, an observed future transition or a verdict on
regional coherence. Source reproducibility remains an unmet prerequisite
for the proposed maintenance interpretation.

Validation passes **151 tests**: 43 new
[`source-relevance tests`](../../tests/physics/test_thol_phase_source_relevance.py)
plus the reused forcing, regional, identity and phase tests. An independent
standard-library recount checks the saved source decomposition and all nine
regional rate pairs directly from primitive retained state and conductance;
175 exact checks pass without another kernel call. Both new Python files
pass flake8. The study itself uses exactly two forcing-capture calls, zero
coordination calls and zero native calls. It changes no engine evolution.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/thol_phase_source_relevance.py
```

The three pinned input identities are retained in the output and the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md). Output:
`artifacts/research/thol_phase_source_relevance_2026_09_18.json`, SHA-256
`62c4fb07068980acebc75b264f63d715e3a0d28bef6fcc5628441937c01ffb86`.
Working-source digest:
`sha256:f3f8609af8ff7033a98c9f0ac449ca4a1343ffece1aa73620340ab98e31e918c`.
Local validation: `artifacts/research/phase_source_relevance_validation_2026_09_18.json`.
The ignored inputs remain required for the retained run; portable tests use
declared synthetic records and do not regenerate the historical studies.

## 12. Exact reduction of represented phase components

The source sensitivity in section 11 motivates a shared arithmetic contract
before changing production phase coordination. Fix a finite nonempty list
of materialized binary64 component pairs `(c_j,s_j)`. These are supplied
numeric values, not claims that either component is an exact transcendental
cosine or sine. Define

$$
a=\sum_j c_j,\qquad b=\sum_j s_j,
$$

with exact dyadic arithmetic. The existing
[`exact_weighted_sum_ratio`](../../src/tnfr/mathematics/_exact_weighted.py)
computes each sum with unit weights: binary64 denominators are powers of
two, so alignment to their largest denominator and integer addition preserve
every bit. Addition of integers is associative and commutative. Hence `a`,
`b` and their exact joint-zero predicate are independent of input permutation.
This is an algebraic argument for every admitted finite list; finite
permutation tests validate its implementation rather than prove it by sampling.

If `a=b=0`, the resultant has no direction. The readout must return an
explicit unavailable angle rather than call `atan2(0,0)`. An empty input is
a separate invalid request, not evidence of cancellation. For a nonzero
pair, set `m=max(|a|,|b|)>0` and retain the exact scaled pair `(a/m,b/m)`.
This common positive scaling preserves the exact direction. Both components
lie in `[-1,1]` and at least one equals `+1` or `-1`. Materializing each once
therefore avoids overflow and joint-zero underflow even when the unscaled
sum is outside the finite floating-point range.

Let `(u,v)` be the resulting binary64 pair and retain the exact defects
`u-a/m` and `v-b/m`. The minor component can underflow to signed zero;
that loss must remain visible. A numerical `atan2(v,u)` is now defined and
uses the same inputs under every permutation of the fixed component list.
That determinism is conditional on the same numeric runtime. It is not a
certificate of correctly rounded transcendental angle, a cross-library
identity, global gauge covariance or continuity near a vanishing resultant.
Canonicalizing an exactly zero component to positive zero also specifies
the numeric display at the negative-real-axis branch; it does not choose
a direction for an exactly zero pair.

There are two distinct zero questions. The sum of supplied rounded
components can vanish even when the exact-real sum
`sum exp(i*represented_theta_j)` does not, or conversely differ from the
ideal phase preparation. The existing rational resultant enclosures address
the represented-angle transcendental question separately. The exact
two-neighbor midpoint contract also keeps its own strict chart hypotheses;
it is not replaced by this finite component reduction.

The reducer is a mathematical readout. It introduces no phase gain,
alignment epsilon, external direction, pressure law or EPI update. Its
production use is a separate caller contract, documented below.
The [single execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md) owns the
implementation checkpoint and the subsequent integration boundary.

The implementation is
[`reduce_phasor_components`](../../src/tnfr/mathematics/phasor_resultant.py),
returning a frozen `RepresentedPhasorResultant`. Its retained `components`
follow input order, so the complete record is not invariant under a
permutation; the derived sums, scale, materialized pair, defects and angle
are. Nonempty finite outer iterables are required, and generators are
consumed once. The shared finite-real materialization rejects booleans,
nonfinite values, overflow and nonzero input underflow; input signed zeros
are canonicalized to positive zero. The binary64 rounding precondition is
checked before consuming the input. No unit-norm assumption or trigonometric
evaluation is introduced. At cancellation, `joint_zero=True` and all
direction/materialization fields are `None`. This is distinct from the
exception for an empty or malformed input.

The [`52 independent reducer tests`](../../tests/test_phasor_resultant.py)
compare with a separate `Fraction.from_float` summation oracle. They cover
all permutations of a cancellation-sensitive four-pair control, exact
cancellation without an `atan2` call, input generators, signed zeros,
axes, subnormals, sums above the finite float range and a minor scaled
component that underflows while the exact pair remains nonzero. They also
check the explicit input and IEEE-rounding preconditions. With the reused
phase-midpoint, unified numerical, circular-semantics and stable-pressure
regressions, **225 tests pass**. Both new Python files pass flake8.
This validated the standalone reducer contract before production integration;
no retained research trajectory was rerun. Local validation is retained at
`artifacts/research/phasor_resultant_validation_2026_09_18.json`.

### Caller policies remain separate from arithmetic

The existing callers do not have one common degenerate-result contract.
Replacing their sums without reviewing these policies would not complete
the phase repair.

| Existing caller | Current reduction | Empty or degenerate behavior |
| --- | --- | --- |
| [Legacy global coordination](../../src/tnfr/dynamics/coordination.py) | NumPy means or scalar `fsum` | Nonempty zero rounded resultant reaches `atan2`; empty graph returns after earlier history/gain bookkeeping |
| [Local phase list mean](../../src/tnfr/metrics/trig.py) | NumPy means or compensated fallback | No usable neighbors return the caller's fallback; nonempty cancellation reaches `atan2` |
| [Bulk local mean](../../src/tnfr/metrics/trig.py) | `bincount` and division by neighbor count | Isolates retain node phase; nonempty cancellation reaches `atan2` |
| [Unified circular mean](../../src/tnfr/mathematics/unified_numerical.py) | NumPy means or scalar `fsum` | Empty input rejects; mean resultant norm at or below the configured numeric tolerance rejects |
| [Default nonfused pressure](../../src/tnfr/dynamics/dnfr.py) | Neighbor component averages | The existing `1e-12` small-resultant policy uses the node's phase, giving zero phase pressure |
| [Fused pressure](../../src/tnfr/dynamics/fused_dnfr.py) | Indexed component addition | Isolates have zero gradient; nonempty cancellation reaches `atan2`, except rows handled by the separate certified midpoint path |

Those thresholds and fallbacks describe existing code, not derivations from
the new reducer. Pairwise U3 phase admissibility uses wrapped separation
and needs no resultant mean. The first repair target is global phase
coordination, where the retained ordering witness was observed; the public
circular mean, local means and pressure branches need their own caller
contracts before integration. The reducer alone changes none of them.

### Versioned global-coordination integration

[`coordinate_global_local_phase`](../../src/tnfr/dynamics/coordination.py) now
accepts `global_reduction="exact_components_v1"`. The default `"legacy"`
retains its previous reduction and returns `None`. Both paths share the
adaptive-gain and local-proposal implementation. The selected version changes
the global reduction and its domain policy; it is not a physical coupling
coefficient or a new pressure law. The native runtime still uses the default.

For the opt-in path, the global target comes from the exact sum of the
materialized cached components. Local means, their neighbor order, phase
displacements and phase writes retain their existing owners. The cases are:

| Condition | Selected behavior |
| --- | --- |
| Nonempty graph, effective `kG != 0`, nonzero exact resultant | Use the reducer's materialized direction in the existing update |
| Nonempty graph, effective `kG != 0`, exact joint-zero resultant | Raise `UndefinedGlobalPhaseError` and restore graph-owned state |
| Effective `kG == 0` | Apply zero global displacement without choosing a global target; local evolution remains active |
| Empty graph | Return a distinct `empty_graph` record after the existing history/gain bookkeeping; no resultant is requested |

An exact resultant can be tiny and nonzero. No alignment tolerance turns it
into cancellation, and this arithmetic decision does not certify its angle
as an approximation to the true transcendental sum of the represented phases.
At inactive coupling the resultant is still retained as a diagnostic; its
available angle, if any, is not selected as the update's global target.

The exact invocation captures the existing
[`GraphTransactionSnapshot`](../../src/tnfr/operators/network_stage.py) before
caller gain/job conversion, history changes or cache writes. Failures during
admission, proposal computation, phase writes or evidence construction trigger
the shared restoration path, retaining the primary exception. The opt-in
reader rejects nonfinite gains, components and proposals; failed job-count
conversion also propagates rather than taking the legacy fallback. Restoration
covers graph-owned state and aliases admitted by that transaction owner, not
external I/O or all independently retained Python references. The coordinator
does not introduce a second rollback implementation.

The returned frozen `GlobalPhaseCoordinationEvidence` records the numerical
version, empty/applied status, node and neighbor order, primitive phases,
materialized-component reducer record, requested/effective gains and their
mode, local targets, raw proposals, realized phases and execution branch.
It is a public completed-call readout; it is not a sealed causal certificate,
and arbitrary hashable node identifiers are not deep-copied into immutable
identities. Its numeric evidence concerns that call, not future execution.

Permutation invariance holds for the derived global resultant with the same
materialized component multiset. Whole-coordinator invariance would additionally
require appropriate invariance of local reductions, adaptive gains and input
materialization. Their policies, and the pressure/U3 paths, remain separate.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md) owns validation and
the next frozen comparison; no historical source output is overwritten.


## 15. Child-cohort distortion and regional mean-to-shape transfer

[`thol_child_distortion_audit.py`](../../benchmarks/thol_child_distortion_audit.py)
reuses the two authenticated saved reset and regional-response reports for
exactly `t=2.25 -> 2.5`, the actual eight children, and the same two archived
branches. It neither advances a graph nor evaluates a reset, pressure or
phase kernel. The fixed metric is the restriction of the original full
`H=diag(d/nu)`; parent coordinates and their boundary remain in the model.

### Staged nodal accounting

Write `delta_0`, `delta_g`, `delta_i`, `delta_f` for the paired EPI difference
before Reception, at integrator entry, after integration and at the endpoint.
The stored pressure was generated at `delta_0`. With admitted common reset
map/offset `(S,c)` and generation coefficients `(A,b)`,

`delta_g = S delta_0 + r_reset`,

`delta_f = (S-hA) delta_0 + r_reset + r_pressure + r_integrator + r_post`.

The common source and affine offset cancel only after exact equality checks.
The map is `S-hA`; replacing it by `(I-hA)S` would change pressure timing.
At the integrator entry the stored-minus-current-model rate splits as

`A(delta_g-delta_0) + diag(nu) (epsilon_kernel + epsilon_stored)`.

Here the first term is the intentional held-input lag, while the other terms
are differences of the originally captured generation defects. No forcing
is inferred from the endpoint derivative. Source components and parent-cut
work remain available through the existing regional observer.

For child-centered restriction `C_B` and increment `q`, the same observer
at zero elapsed time gives the exact reset budget

`E_B(delta+q)-E_B(delta) = <C_B delta,C_B q>_H + ||C_B q||_H^2/2`.

All sixteen paired local writes are retained, followed by the finite Euler
budget at `h=1/4` and the postintegration jump. Signed cross terms are essential:
a per-node positive quadratic term alone cannot explain the energy change.
The later phase update is not assigned to an earlier EPI difference.

### A connection to regional identity

Let `m_B,m_P` be the child and parent H-weighted means of `delta_0`.
The one retained map `T=S-hA` acts on the exact four-part decomposition

`delta_0 = m_P 1 + (m_B-m_P) 1_B + z_B + z_P`,

where the two residuals are separately centered and supported on their own
cohort. The audit retains each `C_B T` image, all centered runtime residuals,
and the complete Gram energy. These are contributions to one recorded map,
not four interventions; their energies are not additive without cross terms.
It also retains `C_B T 1`, so constant preservation is checked rather than
assumed for represented coefficients.

This gives a precise mechanism test: a nonzero `C_B T 1_B` converts a
child-parent mean contrast into within-child spatial contrast, even if
both initial cohorts are internally uniform and the global constant image
vanishes. On a support with no internal child conductance and common child
capacity `nu_B`, `(A 1_B)_B = e nu_B 1_B`, so `C_B A 1_B=0`. Any nonzero
mean-to-child-shape transfer in this held-pressure step then lies in the
admitted Reception map. This is a statement about the combined recorded
support and sequential map; it does not isolate update order as a cause.

The transfer complements the earlier family-mean closure obstruction: it
measures cohort means feeding unresolved shape on a two-cohort partition.
The earlier obstruction measured hidden coordinates feeding eight family
means. Neither direction makes an arbitrary subset an autonomous NFR.


### Retained result and maintenance boundary

The independent represented-rational recount agrees with every stage:

| Paired child state | Centered error energy | H-weighted mean offset |
| --- | ---: | ---: |
| Before Reception | 1.162021485807489e-8 | 0.06635321519 |
| After Reception | 4.174981357577767e-7 | 0.05390852580 |
| After nodal integration | 4.894373864559578e-7 | 0.05132075833 |
| Final endpoint | 4.894373864559578e-7 | 0.05132075833 |

Reception contributes `4.058779208997018e-7`, about **84.9442%** of the
observed increase. Integration contributes `7.193925069818111e-8`, the
remaining **15.0558%**; the postintegration EPI contribution is exactly zero.
These percentages describe signed stage increments in this interval, not
universal operator gains. Parent writes do not immediately change the
child-only score, but change the inputs later consumed by child writes.
Individual child-write increments have both signs and largely cancel;
their complete ordered sum, rather than the largest intermediate score,
is the Reception contribution.

The integration's first-order parent-boundary work is
`+9.735871208336425e-8`. Its held-generation lag contributes
`-2.830200479482354e-8`, and the generation-kernel discrepancy contributes
about `+4.447225349707089e-21`. Internal child dissipation, paired forcing
work and stored-minus-fresh generation work are zero here. The Euler
quadratic term is `+2.882543409599146e-9`; the endpoint-defect linear and
quadratic terms are about `3.68103e-20` and `6.16782e-33`. Thus the held-input
lag partly **attenuates** the increase; neither that lag nor numerical
residuals explains the dominant growth. These are finite Euler accounting
terms, not time-integrated flux measurements.

The mean-to-shape mechanism is present. The initial child-parent mean
contrast is about `0.05994453363`. The eight entries of `C_B S 1_B` are
nonzero, ranging from about `-0.00194634` to `0.00426771`, while
`C_B A 1_B` is **exactly zero**. The isolated self-energy of the mapped mean
contrast is `1.272946951102293e-7`; the mapped parent-centered and
child-centered self-energies are about `7.19564e-8` and `5.97976e-9`.
Their three cross terms are positive and together contribute about
`2.84206e-7`. All other terms remain in the exact Gram ledger, including
the tiny nonzero represented global-constant image. The self-energies alone
must not be presented as additive percentages of the endpoint error.

This accounts for the distortion through the existing nodal transport,
recorded Reception map and actual boundary. It establishes neither a new
physical force nor autonomous regional restoration. Mean relaxation can
feed spatial distortion in a region whose environment is coupled differently
to its members. That is the useful connection to the NFR identity problem:
a maintenance test must retain regional form, mean contrast and environmental
response together instead of relying on one global attenuation score.

The historical follow-up criterion, evaluated in section 16, required a
restoring response after **actual
regional form damage** under a predeclared canonical perturbation and
continuation. Its control must retain measurable nonuniform form, with
control drift reported; reduced error caused only by control flattening is
not sufficient. Keep the mean-to-shape and boundary contributions visible,
and distinguish an existing state-dependent restoring mechanism from supplied
operator timing or an imposed target. A failed damage, control or mechanism
gate is a reported negative/inconclusive result, not a reason to tune the
pressure or select a favorable region. This remains a criterion for that
study, not a current assignment to repeat it; the execution plan owns resumption.

Local output: `artifacts/research/thol_child_distortion_audit_2026_09_18.json`,
SHA-256 `f3ba30169c969e2b39a0958f3437ce0c6822a3bd7c52006942f184398c1b66b3`.
The producer ran once on the two saved reports, with zero kernel, pressure
capture or native calls. Validation includes 27 new portable tests, 40
existing regional/reset tests, and a separate standard-library/Fraction
recount of **277 exact checks**. The existing reset suite also invoked its
optional historical read-only producer test once; that verification is
separate from this producer's budget and did not extend or rewrite evidence.


## 16. Localized regional form damage and finite configured restoration

[`thol_regional_restoration.py`](../../benchmarks/thol_regional_restoration.py)
uses one complete authenticated control replay to `t=1.5`, then two detached
copies of that live source. The shared copy owner rebuilds runtime caches;
the study explicitly preserves local neighbor insertion order and the known
last-operator instance marker, checks scientific source equality, and checks
that execution leaves the opposite branch and retained source unchanged.
This avoids interpreting the archival readout's opaque resource descriptions
as a complete runtime checkpoint. It does not claim independence of arbitrary
external resources or callbacks.

The predeclared intervention is one public default Emission on the first
actual child in the retained birth receipt. This is a supplied preparation
mark, not an endogenous target-selection law. Both strict read-only admission
and the native stage apply. The region stays all eight children, with the
original full metric `H=diag(d/nu)` restricted to them. No factor, target,
pressure law or region is fitted to the response.

For a localized EPI jump `J` at child `j`, subtracting the regional H-weighted
mean gives the exact initial form-damage energy

`E0 = (H_j/2) (1-H_j/H_B) J^2`, where `H_B=sum_B H_i`.

Thus a nonzero local jump on a proper multi-node region changes its centered
form as well as its mean. The control at that instant fixes the reference
independently of the later outcome; the continuation compares the same two
branches at seven fixed times from `1.5` to `3`.

### Shared accounting and decision scope

[`thol_regional_restoration_accounting.py`](../../benchmarks/thol_regional_restoration_accounting.py)
centralizes the pure reader for this test. It reuses the regional Euler,
paired-distance, source-decomposition and record-admission owners. It keeps
pre-generation, pre-integration, integration and endpoint EPI budgets,
per-glyph EPI and pressure writes, and any intervening residual. Captured
pressure generation precedes named glyphs and integration; clocks and
full support/capacity are checked at the observed boundaries.

At integrator entry let `delta_g` be the current paired field, `delta_0`
its pressure-generation value and `delta F_g,delta F_0` the independently
captured source differences. The stored-minus-current-model pressure splits
exactly into generation kernel discrepancy, generation stored-minus-fresh
discrepancy, observed intervening pressure writes, and

`e [gradient(delta_0)-gradient(delta_g)] + delta F_0-delta F_g`.

The last two terms are held-EPI and held-source lags. The source difference
need not vanish in a localized experiment. Each term is evaluated in the
same integration-entry regional energy balance; its work is a signed
finite Euler contribution, not a measured integrated flux.

For `z_i=delta_i-m_B`, parent-boundary work is further resolved as

`e sum_cut w_ij z_i(delta_j-delta_i)`
`= -e sum_cut w_ij z_i^2 + e sum_cut w_ij z_i(delta_j-m_B)`.

The first term is nonpositive ordinary relaxation through the boundary;
the second retains the incoming parent field. This algebraic split adds no
new feedback law. A negative self term alone is not evidence that a regional
entity autonomously maintains itself.

The declared finite criterion requires actual positive initial form damage,
positive control variance at all seven boundaries, decreased endpoint error,
decreased error/control-variance ratio, and nondecreased endpoint control
variance. The last condition is a conservative sufficient policy against
flattening, not a general definition of pattern identity. Control centered
vector drift and mean offsets remain visible. Passing the criterion supports
finite configured tracking of an evolving control; it does not establish
exact original-form return, erasure of operator history, endogenous scheduling
or autonomous long-term maintenance.


### Measured result

The retained invocation completes the twelve continuation calls and 26
post-preparation forcing-capture API calls. The single target is actual
child `0_sub_0`; its native AL jump is
`44798133900177/562949953421312`. All six predefined gates pass. Both
branches execute `IL/IL/IL/EN/IL/AL`, on unchanged support and capacity.

| Time | Centered paired error | Control variance | Error / control variance |
| --- | ---: | ---: | ---: |
| 1.50, after localized AL | 0.006734007748 | 0.002657876774 | 2.533604196 |
| 1.75 | 0.006296948745 | 0.003925631432 | 1.604060099 |
| 2.00 | 0.005890587815 | 0.005124599118 | 1.149472901 |
| 2.25 | 0.005512655227 | 0.006256037468 | 0.881173627 |
| 2.50 | 0.002939941119 | 0.015159875045 | 0.193929113 |
| 2.75 | 0.002766568872 | 0.015512596781 | 0.178343375 |
| 3.00 | 0.002554142631 | 0.016061734388 | 0.159020350 |

The error energy decreases at every sampled boundary and by **62.07099%**
over the whole window. Its ratio to control variance decreases by
**93.72355%**, while control variance grows by a factor **6.04307**. The
normalization therefore benefits from growing control contrast as well as
falling raw error; both are reported separately. The paired mean offset
falls from about `0.01437493` to `0.00937148`, but does not vanish. The
control's centered-vector drift energy at the endpoint is about
`0.00625356`: the control itself evolves substantially.

The exact aggregate ledger is:

| Contribution | Child-error energy change / Euler work |
| --- | ---: |
| Reception EPI writes | -0.002218738764485463 |
| Actual integration, all six steps | -0.001961126353158593 |
| Pre-generation and postintegration EPI changes | 0 exactly |
| Boundary self-relaxation, within integration | -0.002429152538002955 |
| Incoming parent field, within integration | +0.000079302599110432 |
| IL pressure-write correction, within integration | +0.000451285357921915 |
| Held-EPI lag, within integration | -0.000099945741002708 |
| Euler quadratic term, within integration | +0.000037383968814722 |

The first three rows telescope the measured endpoint energy; the later
rows decompose integration and must not be added a second time. Paired
non-EPI source-channel vectors, internal child dissipation, held-source lag
and stored-minus-fresh generation discrepancies are exactly zero. Generation
kernel work is about `5.42265e-19`, and combined integration endpoint-defect
work about `-1.41588e-18`; they remain explicit. Every observed intervening
pressure write is allocated to its glyph receipt, with zero unassigned
remainder. IL and common AL make no immediate paired EPI-energy change.

The positive IL correction is relative to the captured unattenuated-pressure
Euler reference; it is not an executed no-IL experiment or a claim that IL
violates its nodal coherence contract. Decreasing stored pressure and
reducing paired form error are different observables. The negative boundary
self term is ordinary coupled transport, while the prescribed EN event
supplies the large direct EPI-error reduction.

This gives a useful comparison with section 15: Reception increased the
child-cohort error for the earlier all-child perturbation but decreases it
for this localized perturbation. An operator label alone therefore does
not determine the sign of a regional form-error budget. Sections 17-18
characterize that direction and boundary dependence from the existing maps,
without another trajectory sweep.

The scientific outcome is **supported_in_scope** for finite configured
recovery toward the evolving control. It does not establish autonomous NFR
maintenance: the source preparation, marked intervention and operator timing
remain supplied, and control-form drift is nonzero. No physical emergence
claim follows.

Retained output: `artifacts/research/thol_regional_restoration_2026_09_18.json`,
SHA-256 `7ebf3b9cd372f70a5b5d170b16e50fae22ec2a56595c73adb3ce82866c596259`.
The `.completed.json` checkpoint has identical bytes. Validation includes
58 new portable tests (25 producer/CLI and 33 accounting), 51 reused observer
tests, and **289 independent exact checks** using only the standard library.
Two earlier invocations lost their results to output-format failures;
normalization and complete manifest validation now precede execution, and a
durable completed-result checkpoint precedes final provenance/output checks.
Those technical repetitions are recorded separately in the execution plan
and do not count as independent scientific replications.

## 17. Conditional regional response and environmental input

### A condition derived from the admitted nodal map

Fix a positive full-support metric `H=diag(d_i/nu_i)` and a nonempty proper
region `B`. Let `C` restrict to `B` and subtract its H-weighted mean,
`H_B=diag(H_i:i in B)`, and `M=C^T H_B C`. The regional form-error energy is
`E_B(delta)=delta^T M delta/2`. It does not measure the regional mean,
phase, capacity, operator history or the whole identity of an NFR.

For two trajectories sharing admitted reset coefficients and generation
coefficients, with equal affine offsets and non-EPI sources, the held-pressure
interval of section 15 has

`delta_after = T delta_before + r`, with `T=S-hA`.

The matrices come from the retained operator/support coefficients, not a fit
to the endpoint. The residual `r` retains reset realization, pressure
realization, integration and postintegration changes separately. Common-source
cancellation is a checked premise of each pair; the source need not have the
same value in two different experiments. Replacing `T` by `(I-hA)S` would
refresh pressure at a different stage and would change the declared dynamics.

The exact quadratic response form is

`Q=T^T M T-M`,

`E_B(delta_after)-E_B(delta_before)`
`= delta_before^T Q delta_before/2`
`  + <CT delta_before,Cr>_H + ||Cr||_H^2/2`.

For the declared exact map (`r=0`), the sign is determined by the initial
difference and admitted coefficients. With a measured residual it is a
retained-runtime verification. A prospective binary64 guarantee additionally
needs a residual bound supplied independently of the endpoint being predicted.

### Existing shape versus incoming mean and parent differences

Write the initial difference in the same four-part decomposition as section 15:

`delta=z_B + m_P 1 + (m_B-m_P)1_B + z_P = z_B+w`.

Here `z_B` and `z_P` are centered on their respective cohorts, extended by zero
outside them; `Cw=0`. The component `m_P 1` remains explicit even when its
represented image is very small: represented constant preservation must be
checked, not assumed. Define

`v=CTz_B`, `u=CTw+Cr`,

`Z=||C delta||_H^2`, `A_shape=||v||_H^2`, `U=||u||_H^2`.

The regional error does not increase **if and only if**

`2<v,u>_H + U <= Z-A_shape`.

The right side is the attenuation available from the initial regional shape;
the left side is the signed effect of mean contrast, parent shape and the
realization residual. Either side may have an unfavorable sign. Parent
contributions and their cross terms cannot be replaced by an operator label
or treated as nonnegative additive percentages of the final error.

A direction-independent sufficient condition for every arbitrary centered
input with norm at most `sqrt(U)` is

`Z >= A_shape+U` and `(Z-A_shape-U)^2 >= 4 A_shape U`.

This is exactly `sqrt(A_shape)+sqrt(U)<=sqrt(Z)`, expressed using rational
arithmetic. The triangle inequality proves sufficiency; an input aligned with
`v` attains the upper bound. It is not necessary for a particular input
direction, because a negative cross term can produce cancellation. Neither
the norm nor the signed test derives an autonomous bound on the parent input.

### Why a region-only gain can fail

A finite constant `q` with `E_B(T delta)<=q E_B(delta)` for every real `delta`
can exist only if `T(ker C)` is contained in `ker C`. Conversely, preserving
that kernel induces a linear map on the finite-dimensional regional quotient,
which has a finite norm. This converse supplies neither `q<=1` nor strict
contraction. A basis of `ker C` is `1_B` together with the individual outside
coordinate vectors; their centered images provide a finite exact test.

In particular, `CT1_B!=0` maps zero initial regional shape error to positive
shape error. Section 15 already supplies this obstruction for its declared
map: `CA1_B=0` but `CS1_B!=0`. Thus a regional variance alone cannot give an
unconditional response guarantee for that map. The obstruction is algebraic;
it does not assert that every basis vector is a grammar-admitted runtime
intervention. Environmental relations or additional state can restrict the
admissible inputs, but those restrictions must be derived and checked.

The reusable implementation is
[regional_response.py](../../src/tnfr/physics/regional_response.py). It reuses the
exact-coordinate and matrix-product owners, retains both the quadratic form
and the signed input ledger, and performs no evolution or operator selection.

### Matched retained witnesses

The [admission adapter](../../benchmarks/thol_regional_response_admission.py)
authenticates the earlier reset report and the localized-restoration report.
At `t=9/4 -> 5/2` all four records share exactly the same `S`, `A`, zero
affine offset, original full H metric, support/capacities and local EN
configuration/order. The sources cancel within each pair; their values also
agree across the experiments. These are common **declared** coefficients:
matching configuration does not prove identity of historical implementation
sources or a general binary64 kernel theorem. The later record's local reset
defect is retained in full without inventing a rounding/clipping split.

The [criterion study](../../benchmarks/thol_regional_response_criterion.py)
applies the same form to Reception alone and to the complete held-pressure
interval. It first reconstructs every full paired endpoint from the admitted
map and its separate residual components. For the complete interval:

| Read-out | Earlier all-child perturbation | Localized perturbation |
| --- | ---: | ---: |
| Initial regional error energy | 1.162021486e-8 | 0.005512655227 |
| Final regional error energy | 4.894373865e-7 | 0.002939941119 |
| Isolated child-shape image ratio `A_shape/Z` | 0.5145995631 | 0.5275976258 |
| Available squared-norm drop `Z-A_shape` | 1.128091474e-8 | 0.005208382835 |
| Signed input work `2<v,u>+U` | 9.669152579e-7 | 0.000062954619 |
| Actual energy change | +4.778171716e-7 | -0.002572714108 |
| Combined realization correction to ideal energy change | -3.177557986e-20 | -9.007103052e-18 |
| Sufficient input-ball condition | Fails | Passes |

The initial perturbation magnitudes differ. The isolated ratios describe
images of the two actual child-shape directions, not an all-direction gain
bound or separately executed interventions. Both images attenuate, but the
mean/parent input overwhelms the first witness's available margin. The
localized witness has ample margin and also passes the conservative
direction-independent test at its **observed** input radius. The unknown
future environment has not been bounded by that observation. Reception alone
has the same opposing signs, and the measured realization corrections do not
reverse any of these four signs.

For both `S` and `T`, the exact image of `1_B` has centered energy about
`3.542510399e-5`, while `CA1_B=0`. The same-map obstruction therefore excludes
a finite unrestricted regional-shape gain. This identifies an environmental
closure obligation, not a contradiction of conditional recovery or of the
operator's nodal coherence contract. The child cohort still has no internal
conductance and is defined by recorded lineage; this calculation does not
promote it to an independently emerged coherent region.

This completes the conditional-criterion delivery with no new graph,
trajectory, scalar kernel or forcing capture. The retained result is
`artifacts/research/thol_regional_response_criterion_2026_09_18.json`.
There are 82 new portable tests, 84 reused passing observer tests, and 239
independent exact checks using only standard-library arithmetic. The latter
reconstruct local reset composition, the nodal generator, source/pressure
defects, response matrices and both sign criteria from the saved evidence.

## 18. Environmental-input geometry and protected read-outs

### Image and annihilator in the original metric

Keep the same region `B`, centering `C`, positive metric `H_B` and declared
transition `T`. Let the columns of `N` be `1_B` followed by every outside
coordinate vector. They form a basis of `ker C`. The exact input map is

`G=CTN : regional-mean/outside differences -> centered regional shape`.

Its columns belong to the `(size(B)-1)`-dimensional H-centered shape space.
A centered vector `y` represents the scalar read-out `y^T H_B z`. That
read-out is insensitive to every declared environmental input precisely when

`G^T H_B y=0` and `1^T H_B y=0`.

These are arbitrary independent algebraic inputs. Their individual or joint
reachability under the native policy, phase, grammar and source constraints is
not inferred from this map. Likewise, input insensitivity alone would not
make a read-out constant under the region's self-dynamics or immune to the
separately measured binary64 realization residual.

For independent columns `V` of `G`, their Gram matrix `K=V^T H_B V` is
positive definite. The image projector is

`P_image=V K^-1 V^T H_B`.

With local centering `P_B=I-1 h_B^T/sum(h_B)`, the complementary projector is
`P_protected=P_B-P_image`. Both are H-self-adjoint and idempotent. Its image
gives every protected centered read-out, with dimension
`size(B)-1-rank(G)`. Exact column reconstruction `G=V K^-1 V^T H_B G`
provides a certificate of the image, rather than relying on a numerical
singular-value threshold.

The existing [regional response owner](../../src/tnfr/physics/regional_response.py)
now shares one validated domain, centering and input basis between the response
criterion and this geometry. It reuses the [exact Krylov rank](../../src/tnfr/mathematics/krylov.py)
and [matrix inverse](../../src/tnfr/physics/_exact_linear_algebra.py) implementations.
The weighted image projector factors through two
[shared exact matrix products](../../src/tnfr/mathematics/_exact_linear_algebra.py),
retaining a full-sized zero matrix at rank zero. Coordinate order, exact
reconstruction, idempotence and H-self-adjointness checks remain in the observer.
No new elimination algorithm, physical parameter or evolution path is added.

### Exact classification of the retained child cohort

The [retained geometry study](../../benchmarks/thol_regional_input_geometry.py)
authenticates section 17's result, computes each of `S` and `S-hA` once,
and matches the centering, full metric and input images of both retained
perturbations. It does not replay their admission or trajectories.

| Declared step | Full input rank | Parent-only input rank | Child-shape dimension | Protected dimension |
| --- | ---: | ---: | ---: | ---: |
| Reception `S` | 7 | 7 | 7 | 0 |
| Reception plus held-pressure Euler `S-hA` | 7 | 7 | 7 | 0 |

A separate standard-library proof reconstructs `CTN` directly and proves the
rank using a nonzero `7 x 7` minor from child rows 0--6 and parent columns
0--6. The determinant is approximately `5.34828347915e-9` for Reception and
`2.46679963704e-8` for the complete interval; exact rational determinants are
retained in the artifact. H-centering supplies the independent rank upper
bound seven. The weighted annihilator has only the zero vector, and
`P_image=P_B`, `P_protected=0` exactly for both maps.

The parent-only witnesses establish this result without relying on the tiny
represented global-constant image. The change of input geometry between
the stages is exactly `G_T-G_S=-h C A N`; it changes coefficients but does
not reduce the image dimension. The regional-indicator column is unchanged
because `CA1_B=0` in this support/capacity configuration.

Thus every nonzero centered linear child-shape read-out can be affected by
some declared parent input. This excludes absolute input protection for
this particular formed support and lineage-defined region. It neither
excludes the conditional recovery of section 17 nor rules out protection
under a genuinely derived restriction on environmental inputs. It says
nothing by itself about a different geometry, nonlinear native reachability,
repeated stability, spontaneous formation or physical particles.
Full rank also does not measure the magnitude of sensitivity: weak response
and finite conditional robustness are separate from exact input immunity.

Retained output:
`artifacts/research/thol_regional_input_geometry_2026_09_18.json`.
Independent reconstruction, exact minors and annihilator certificate:
`artifacts/research/regional_input_geometry_crosscheck_2026_09_18.json`.
Validation: 76 new portable tests, 82 reused passing tests, and 100 independent
exact checks, including the final report's projectors, Gram inverse, image
reconstruction and source/output hashes. All trajectory, kernel and forcing
capture counts remain zero.

## 19. Support symmetry versus the admitted nodal and reset maps

### Separate mathematical objects

Let `Gamma` be the complete automorphism group of the original weighted
support, including any zero-conductance support edges. Retain its subgroup
that preserves the child set, positive capacity and full metric. For a
permutation with `P[permuted(i),i]=1`, a declared matrix `M` respects the
permutation exactly when `PM=MP`, equivalently
`M[permuted(i),permuted(j)]-M[i,j]=0` at every entry. A vector field is
invariant when `v[permuted(i)]-v[i]=0` at every node.

These are different checks. For an affine map `Mx+a`, both matrix
commutation and `Pa=a` are needed. The paired difference map can cancel a
common `a` even when each individual affine evolution lacks that symmetry.
Here `b` is the captured contribution of the canonical phase, capacity and
topology pressure channels multiplied by capacity; `c` is the Reception
offset. The individual held-pressure interval has offset `c+h b`.

The [exact equivariance owner](../../src/tnfr/physics/equivariance.py) now
centralizes the exact matrix/field checks alongside the existing numerical
diffusion diagnostics. It delegates complete enumeration and orbit extraction
to [symmetry sectors](../../src/tnfr/physics/symmetry_sectors.py). The explicit
cap of 2,000 is an operational limit: exceeding it raises instead of returning
a partial group. The owner creates a detached support graph for enumeration,
not an evolved TNFR state, and uses no numerical commutation tolerance.

For the common matrix subgroup `Gamma_0`, the environmental space justified
by an **assumed invariant input** is `Fix(Gamma_0) intersect ker(C)`. Since
the group preserves the region, a basis consists of `1_B` and the indicator
of each outside orbit. This characterizes a conditional input space; group
commutation does not itself establish that a particular initial state,
source, boundary history or native input belongs to it.

### Complete retained classification

The [retained symmetry study](../../benchmarks/thol_regional_map_symmetry.py)
authenticates the prior response report and preserves its original support,
region, metric and coefficients. Exact checks rederive `H=d/nu`, `A`, the
captured source, and the ordered local-row composition of `S`. The weighted
support has 16 nodes and 24 edges, with no zero-conductance edges in this
witness. Its complete group has eight elements. Every one preserves the
child set, capacities and full metric.

| Object checked within that group | Preserving permutations |
| --- | ---: |
| Weighted support, child set, capacity and H | 8 |
| Nodal generator `A` | 8 |
| Ordered Reception map `S` | 1, identity only |
| Held-pressure map `T=S-hA` | 1, identity only |
| Zero Reception offset `c` | 8 |
| Captured canonical source `b` | 1, identity only |
| Captured generation EPI and phase in each paired record | 1, identity only |

The eight support actions have four orbits: even parents, odd parents, even
children and odd children. The common subgroup for the three declared
matrices is the identity, whose sixteen orbits are singletons. Its fixed
environmental space therefore still has nine independent inputs: the
regional mean and each of the eight parent coordinates. Their image retains
rank seven. No nontrivial restriction on those inputs follows from a common
symmetry of these matrices.

A compact obstruction uses the cyclic shift by two positions on both parent
and child index sets. It preserves the weighted support and commutes with `A`,
but

`S[0,0]-S[2,2]`
`= -630642904502747332437308389707 / 162259276829213363391578010288128`
`~= -0.003886636972791`.

The same entry difference occurs in `T` because `A` commutes. Thus this is
not solely the tiny represented constant-mode leakage of the previous
study. In every retained branch the local EN row family itself satisfies
`row_permuted(i)[permuted(j)]=row_i[j]` for all eight symmetries. Its ordered
composition does not. This identifies fixed sequential composition as the
obstruction at the declared affine-map level. No alternative order or
simultaneous runtime event has been executed by this calculation, and it
does not establish a counterfactual binary64 trajectory.

Source and state asymmetry remain distinct. The four recorded source vectors
are equal across the experiments but invariant only under the identity;
paired cancellation is still valid. Phase invariance here concerns the saved
represented coordinates, not equivalence under a global phase gauge.
Likewise, matrix commutation does not certify complete history or runtime
equivariance. Simultaneous stage semantics cannot be substituted for the
captured sequential map without a separate derivation and admission.

The completed result obstructs a symmetry-based input restriction for this
particular map and cohort. It does not rule out conditional recovery,
correlated inputs derived by another mechanism, or other configurations.
In particular it does not decide whether a Platonic graph forms or persists
under TNFR dynamics. It establishes the check required before transferring
geometric symmetry to a dynamical claim.

Retained report: `artifacts/research/thol_regional_map_symmetry_2026_09_18.json`.
Independent exact enumeration, composition and symmetry witnesses:
`artifacts/research/map_symmetry_crosscheck_2026_09_18.json`.
Validation: 113 new portable tests, 140 reused passing tests and 487
independent exact checks, including final report/source identities, signed
defects, fixed-input images and scope flags. All native, kernel, forcing
capture and historical-admission counts are zero.

## 20. Same-snapshot Reception and the limit of geometric protection

### Comparison using the existing coefficient kernel

For the same retained neighborhood and represented mix, let J contain the
local Reception coefficient row of each target. The sequential map S is
their chronological single-target composition; J reads every row from one
common input. Define U=J-hA with the same pre-generated nodal pressure timing
as T=S-hA. The existing [comparison reader](../../benchmarks/thol_regional_map_symmetry.py)
now exposes this comparison through `--snapshot-comparison`, reusing the
exact symmetry and regional-input owners rather than adding another engine.

All 64 saved rows (16 nodes, two paired experiments) match the
[shared represented Reception row](../../src/tnfr/operators/_neighbor_epi_kernel.py).
Its coefficients depend on mix and input support, not earlier EPI writes.
Reception uses an unweighted neighbor mean; A retains the actual weighted
transport. The comparison checks configured mix, neighbor membership and
the common hard interval. J is a declared unclipped real-affine map:
binary64 mean/blend evaluation, clipping, semantic kinds and auxiliary writes
remain distinct. Sequential defects and outcomes are not transferred to J.

### Exact result and the unmet input condition

| Quantity | Sequential S / T | Same-snapshot J / U |
| --- | ---: | ---: |
| Common map subgroup order, including A | 1 | 8 |
| Fixed environmental input dimension | 9 | 3 |
| Centered image rank for those fixed inputs | 7 | 1 |
| Orthogonal centered read-out dimension for those fixed inputs | 0 | 6 |
| Centered image rank for unrestricted inputs | 7 | 7 |
| Image of regional-mean contrast `CM 1_B` | Nonzero | Zero |

The three conditional inputs are the regional constant and the even-parent
and odd-parent orbit indicators. Their centered output spans one direction,
constant on each child orbit with opposite signs and the ratio required by
the original H metric. Thus six independent centered linear read-outs
annihilate this restricted input image. This is an exact one-step conditional
input-protection result, not a persistence or repeated self-dynamics theorem.

The condition is not met by either recorded experiment. For each actual
paired difference delta, decompose delta=z_B+w as in section 17: w retains
the parent differences and the H-weighted regional mean. Both retained w
vectors preserve only the identity, as do the saved source, EPI and phase.
No averaging is applied. Consequently the six-direction result cannot be
promoted to either retained environment; unrestricted input rank stays seven.
The common-source cancellation still holds, separately from these symmetries.

### Runtime boundary and disposition

The criterion report is a projection of historical evidence, not a complete
graph-owned stage checkpoint. It lacks replayable grammar histories/debt,
EPI kinds and complete graph/source/runtime configuration. Older trace
boundaries additionally contain explicitly unavailable resources and named
callables, not restorable live owners. The existing
[two-phase dispatcher](../../src/tnfr/operators/network_stage.py) requires those
preflight and commit conditions. No synthetic history or defaults are inserted
to present this map calculation as authenticated runtime execution.

This closes the ordering/symmetry comparison. Same-snapshot semantics remove
the exhibited ordering obstruction at the declared-map level, while the
actual input restriction remains unjustified. Neither symmetry nor regular
polyhedral shape is established as necessary for an NFR. The broader open
question concerns a closed endogenous feedback mechanism for differentiated
coherence, using the existing nodal channels and explicit state/history;
additional symmetry sweeps and historical-state restoration are parked.

Retained report:
`artifacts/research/thol_snapshot_reception_comparison_2026_09_18.json`.
Independent arithmetic:
`artifacts/research/snapshot_map_independent_2026_09_18.json`.
This delivery performs 64 coefficient-builder calls, zero scalar evolution
kernel calls, zero native steps and zero live stage calls. No historical
trajectory is rerun or extended.
Validation: 14 new portable tests, 149 distinct reused passing tests and 274
independent exact checks, including the final report and working-source
identities. The previous symmetry/input reports retain their original bytes.
