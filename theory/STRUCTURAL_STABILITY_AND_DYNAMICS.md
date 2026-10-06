# Structural Stability and Dynamics

This document collects implemented stability diagnostics, phase-transition
read-outs, life telemetry, lifecycle classification, an auxiliary Hamiltonian
model, and integrity tools anchored to the nodal equation
$\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$. Exact
restricted results, selected policies, finite observations, and compatibility
models are identified separately.

**Status**: Mixed — exact restricted diffusion results, operational operator
classifications and measured diagnostics are identified separately.

**Ownership.** This is a capability and scope reference, not a research queue.
The [diffusion theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md) and
[REMESH history theorem](REMESH_INFINITY_DERIVATION.md) own their respective
derivations; [operator-event contracts](../docs/contracts/OPERATOR_EVENTS.md)
own execution and evidence admission. Sections 1.4-1.6 retain the applicability
bridges and a held/refreshed-pressure counterexample.
The field-energy balance belongs to
[the conservation note](STRUCTURAL_CONSERVATION_THEOREM.md), and phase/telemetry
interpretations to [parameter foundations](NODAL_PARAMETER_FOUNDATIONS.md).
Current research priorities belong to the
[single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md).

---

## 1. Lyapunov Stability Analysis

### 1.1 Tetrad-plus-current structural energy

`compute_energy_functional` implements the non-negative quadratic diagnostic

$$
E[G] = \frac{1}{2}\sum_i \left[\Phi_s(i)^2 + |\nabla\phi|(i)^2 + K_\phi(i)^2 + J_\phi(i)^2 + J_{\Delta\mathrm{NFR}}(i)^2\right]
$$

This is the **tetrad-plus-current structural energy**: it contains three tetrad
fields $(\Phi_s,|\nabla\phi|,K_\phi)$ and the two currents
$(J_\phi,J_{\Delta\mathrm{NFR}})$. It omits the fourth tetrad field $\xi_C$, so
calling it the “tetrad energy” is inaccurate. It also contains no explicit EPI
or $\nu_f$ term, although recomputing the fields after a state change can alter
the value indirectly.

For general engine trajectories, $E$ is a **Lyapunov candidate**. The finite
difference returned by `compute_lyapunov_derivative` only reports whether two
supplied snapshots show non-increase within its tolerance. Grammar compliance
does not prove $dE/dt\leq0$. A proved Lyapunov result exists for the different,
weighted centered energy of restricted pure EPI diffusion; see
[the heterogeneous diffusion theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md).

The historical `coherence_matrix` is also outside this Lyapunov claim. It is
an auxiliary entrywise-bounded affinity built from phase, EPI, frequency and
Si, and it is distinct from canonical $C(t)$. Entrywise nonnegativity does not
imply a nonnegative spectrum after the graph-support mask: three identical
nodes on a path give $W=I+A_{P_3}$ with eigenvalue $1-\sqrt{2}<0$. The matrix
can supply a Hermitian term to the auxiliary Hamiltonian because it is real
symmetric; it is not a positive-semidefinite coherence observable.

### 1.2 Canonical channels and nominal U2 roles

The [generated canonical contract table](../docs/API_CONTRACTS.md#canonical-contracts)
owns primary nodal channels, tokens and measurement contexts; the separate
[generated grammar role table](UNIFIED_GRAMMAR_RULES.md#1-canonical-operator-roles)
owns U2 composition roles. A primary channel is not a list of every secondary
write or a stability classification: phase-channel ZHIR is a U2 destabilizer,
while pressure-channel NAV is U2-neutral.

[`physics.lyapunov`](../src/tnfr/physics/lyapunov.py) maps those U2 roles to a
legacy compatibility model. Its public objects retain names such as
`OperatorLyapunovBound`, but the multipliers are **nominal policy values**, not
proved bounds on the structural energy above or on $C(t)$. The preferred API
names are `OperatorPolicyMultiplier`, `OPERATOR_POLICY_MULTIPLIERS`,
`U2PolicyRole`, `evaluate_sequence_policy`, and
`compare_operator_energy_to_policy`; the legacy names remain source-compatible
wrappers.

#### Stabilizer policy entries

| Operator | Canonical default input | Policy multiplier |
|----------|-------------------------|-------------------|
| **IL** (Coherence) | pressure retention $\pi/(\pi+1)\approx0.7585$ | $0.7585$ |
| **THOL** (Self-organization) | acceleration $1/(4\pi)\approx0.0796$ | $1-1/(4\pi)\approx0.9204$ |

#### Destabilizer policy entries

| Operator | Canonical default input | Policy multiplier |
|----------|-------------------------|-------------------|
| **OZ** (Dissonance) | pressure scale $(\pi+1)/\pi\approx1.3183$ | $1.3183$ |
| **ZHIR** (Mutation) | phase-shift factor $1/\pi\approx0.3183$ | $1+1/\pi\approx1.3183$ |
| **VAL** (Expansion) | capacity scale $1+1/(4\pi)\approx1.0796$ | $1.0796$ |

#### U2-neutral policy entries

AL, EN, UM, RA, SHA, NUL, NAV, and REMESH receive multiplier one because they
carry no U2 stabilizer/destabilizer debt. This value does not predict zero
observed change in phase-dependent or pressure-dependent energy fields.

The legacy `compute_operator_energy_bound` and `compute_sequence_energy_bound`
functions interpret their numeric input as an abstract policy score. Their
names do not confer energy-bound semantics. The legacy `within_bound` result of
`verify_operator_lyapunov` is a one-sided screen against that model and carries
`is_lyapunov_certificate=False`. Actual field-energy and coherence changes
require before/after telemetry, including the actual glyph executed after
grammar selection. See
[example 39](../examples/02_physics_regimes/39_nodal_equation_decomposition.py).

### 1.3 U2 stability scope

Grammar rule U2 requires destabilizers to be accompanied by stabilizers and
limits uncompensated debt. This is an operator-composition policy. It does not
by itself imply

$$
\sum_{\text{ops}} \Delta E_{\text{op}} \le 0
$$

because the grammar does not supply a common state functional, elapsed times, or
proved gain bounds for arbitrary operator realizations. Even a nominal
multiplier product at most one remains a compatibility-model result; analytic
contractivity requires a separate dynamical proof.

Examples 29 and 38 report the nominal product beside measured energy changes.
Agreement or disagreement in a finite run measures the compatibility model; it
cannot establish a grammar-wide guarantee.

`analyze_operator_policy_context` also reports a U2 multiplier beside the
normalized graph diffusion gap. It deliberately leaves the historical
`effective_convergence_rate` field as `NaN`: an operator-position multiplier and
a continuous-time pure-EPI eigenvalue cannot be combined without an explicit
operator-time model. Its `policy_half_steps` is a score-model statistic; its
`diffusion_relaxation_time` is the separate unit-capacity pure-EPI scale.

### 1.4 Restricted affine EPI gain theorem

The [affine-reset and hybrid-word theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md#affine-reset-gain-and-hybrid-word-theorem)
supplies a conditional gain independently of U2 policy. For one positive common
diffusion metric `H=diag(h)`, let `Q=I-1h^T/(h^T1)` and
`V(x)=||Qx||_H^2/2`. A declared reset `x+=Ax+b` has a finite global
multiplicative `V` gain exactly when `A1` and `b` are uniform. Otherwise a
uniform input gives the decisive `V_before=0<V_after` counterexample. For an
admitted map, the sharp gain is the squared weighted induced norm of `QAQ`.

The owner derives the represented-coefficient rational upper bound and its
composition with certified continuous diffusion. Neither an eigensolver estimate,
a caller tolerance nor a nominal policy multiplier decides contraction. A
finite-horizon bound needs valid reset/flow proofs in one exactly matched
normalized metric and node order. Declared repetition with positive flow
duration lets the contraction bound certify asymptotic disagreement decay.
It identifies the limit with the initial weighted consensus when the reset
and flow proofs also establish exact preservation of that mean.

The [EN/RA realization boundary](TNFR_DIFFUSION_STABILITY_THEOREM.md#reception-and-resonance-runtime-realizations)
keeps four objects separate: an ideal-real convex blend, its represented
coefficient matrix, the two-stage binary64 proposal, and the accepted runtime
snapshot. Its fixed connected undirected support, positive capacities, signed
scalar or uniform-real EPI, convex mixing and inactive hard-clipping premises
must all hold. RA additionally requires admitted circular U3 neighbours and
preserved sign/kind identity. A nontrivial local blend need not preserve the
weighted mean, and a capacity boost can change the diffusion metric. Fixed-map
repetition requires fixed support and, for RA, fixed U3 sets and a
forward-invariant sign/kind domain. Snapshot agreement does not prove global
binary64 affinity.

Atomicity is a separate [operator execution contract](STRUCTURAL_OPERATORS.md#152-base-operator-workflow).
A stage derives all proposals from one immutable snapshot and validates them
before committing graph-owned state; ordered auxiliary streams and opaque
callbacks retain their stated limits. This supplies no affine gain by itself.
The [event-stage composition contract](../docs/contracts/OPERATOR_EVENTS.md#atomic-stage-observations-and-finite-composition)
owns optional represented-map certificates and explicit abstentions. Only
intact maps bound to the observed EPI endpoints and one common metric enter its
finite product. In particular, the REMESH glyph is advisory; delayed EPI mixing
is a separate history-dependent operation.

### 1.5 Operator-event physical-time boundary

The [schedule and transaction contract](../docs/contracts/OPERATOR_EVENTS.md#schedules-and-transaction-boundaries)
assigns zero duration to each named jump and `m+1` declared flow intervals to a
word of `m` events. Exact rationalized binary64 durations and offsets determine
ordering; absolute float timestamps are displays. This clock distinguishes
flow duration from operator counts and supplies no laboratory calibration.
The executor's finite held-input identity, graph rollback and external-effect
limits are specified there; none establishes solver accuracy or order.

Held and refreshed pressure define different models. For fixed pure-EPI
`A=diag(nu_f)L_rw` and `T=sum_k h_k`, the exact-real outer maps are

$$
\widehat{x}_H=(I-TA)x_0,
\qquad
\widehat{x}_R=\left[\prod_k(I-h_kA)\right]x_0.
$$

Internal substeps that retain the initial pressure leave the first map
unchanged; explicit pressure refresh gives the second. On the nonuniform mode
of unit-capacity $K_2$, $\mu=2$, $T=1.25$ gives $g_H=-1.5$, whereas two refreshed
segments of length $0.625$ give $g_R=0.0625$. Thus even modal stability can
change. This counterexample concerns the real-arithmetic maps, not a claim that
an observed binary64 endpoint equals either formula.

[Captured flow evidence](../docs/contracts/OPERATOR_EVENTS.md#captured-flow-and-executor-provenance)
distinguishes a measured nodal residual, trusted held-pressure replay and exact
affine identification. [Pressure-refreshed partitions](../docs/contracts/OPERATOR_EVENTS.md#pressure-refreshed-physical-partitions)
add the actual refresh boundaries. Exact endpoint promotion requires fixed
conductance/capacity, verified pure-EPI pressure and intact per-segment evidence;
modal agreement alone is insufficient. Equal spectra do not establish a common
eigenbasis or commuting generators. Missing or failed evidence remains an
abstention rather than a proof of the exact map.

The [Mutation refinement contract](../docs/contracts/OPERATOR_EVENTS.md#mutation-trigger-and-refinement-observations)
separates an offline coordinate pairing from a jump bound to the same execution.
Only the terminal live segment supplies Mutation's observed secant; a whole-parent
secant is a different observation. The sufficient held-pressure gate certificate
requires strict rate-error margin and observed decision agreement. Physical
refresh can change the gate legitimately. These observations do not prove U4
readiness, adaptive grammar, solver order or future behavior;
[§6.2](#zhir-temporal-evidence-and-execution-boundary)
separates trigger evidence from the nodal-product prediction.

Delayed REMESH requires augmented history. A fixed-history zero coefficient is
not a repeated gain: with lag one and `alpha=1`, EPI `(2,0)` and delayed row
`(0,2)` alternate under successive empty-schedule cycles. The
[recorded/causal sequence contract](../docs/contracts/OPERATOR_EVENTS.md#recorded-and-causal-cycle-sequences)
retains this obstruction and distinguishes caller-ordered endpoint continuity
from same-invocation causal provenance and graph-owned atomicity.

For fixed uniform `alpha in [0,1]`, positive delays, ordered support and a
positive diagonal spatial metric, the separate unclipped
[exact companion-history theorem](REMESH_INFINITY_DERIVATION.md#24-exact-finite-companion-history-stability)
uses the stationary temporal weights in
`V(X)=sum_j pi_j ||Q_H x_j||_H^2/2`. Its Jensen balance proves nonincrease;
interior `alpha` gives temporal convergence to each coordinate's history
barycenter, while `alpha=1` admits periodic histories. This is neither spatial
consensus nor zero pressure. The
[runtime history balances](../docs/contracts/OPERATOR_EVENTS.md#exact-and-runtime-history-balances)
retain signed rounding/clipping defects and the difference between a lifted
companion state and the live history. Hard-clamp nonexpansiveness does not
control preceding rounding; soft clipping can increase disagreement. Finite
adjacent-cycle balances telescope only after their history, metric and endpoint
bindings pass.

A [positive normalized margin on one causal block](../docs/contracts/OPERATOR_EVENTS.md#finite-causal-block-margins)
does not establish a uniform repeated-runtime theorem. Normalization is
undefined at zero initial energy, and amplitude scaling excludes a uniform
positive absolute drop. Promotion through these block bounds requires a
forward-invariant runtime class, a uniform positive normalized margin and an
intrablock prefix bound. The
[exact common-gain and signed-defect theorems](../docs/contracts/OPERATOR_EVENTS.md#exact-remesh-memory-and-schedule-certificates)
supply conditional bounds for declared schedule families: fixed companion and
metric, consensus preservation, common gain `q in [0,1]`, and, for the robust
extension, `E_H(z)-E_H(y)<=eta*J` with `eta>=0` and Jensen input envelope `J`.
This signed defect differs from the energy of `z-y`.
`q_eff=q*(1+eta)<1` gives strict decay; equality only gives nonincrease.
[Finite runtime verification](../docs/contracts/OPERATOR_EVENTS.md#relative-defect-runtime-observations)
of these inequalities does not prove class invariance or future repetition.

The [binary64 REMESH class certificates](../docs/contracts/OPERATOR_EVENTS.md#binary64-remesh-class-certificates)
retain the large-rounding counterexample, exact `alpha=1` class and sharp
antisymmetric P2 half-alpha bound. Their restricted numeric domains are not a
complete graph/event model. [P2 Reception/REMESH binding](../docs/contracts/OPERATOR_EVENTS.md#p2-reception-and-remesh-binding)
separately links the half-Reception `q=0` kernel to accepted finite stages,
causal sequences and individually transactional policy invocations. Each call
is revalidated; finite extinction of active-history disagreement does not
certify auxiliary Reception state, future calls or full TNFR stability.

Finally, [three-mesh and reversible-mode references](../docs/contracts/OPERATOR_EVENTS.md#refinement-and-reversible-diffusion-references)
separate finite comparisons from the
[exact reversible single-eigenmode Euler theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem).
Its fixed reversible generator, positive capacity, nonzero single mode and
`0<mu*h<1` proper-subdivision premises are essential. The
[executor binding](TNFR_DIFFUSION_STABILITY_THEOREM.md#finite-executor-binding-of-the-exact-mode)
propagates represented defects in the full state, since they can leave that
mode. The [effective-P2 REMESH family](REMESH_INFINITY_DERIVATION.md#28-effective-p2-three-mesh-reference-family)
adds its own zero-defect, delay and hard-clipping hypotheses. Neither a finite
error decrease nor endpoint binding proves binary64 asymptotic convergence,
solver order, repeated mixed dynamics or future stability.

The executable
[`165_operator_event_relaxation.py`](../examples/02_physics_regimes/165_operator_event_relaxation.py)
records the schedule-only distinctions with two coincident jumps. Continuous
duration and Euler modal counts do not redefine U2 debt or U4 recency.

### 1.6 Spectral Gap Characterisation

For connected fixed symmetric pure EPI diffusion, the first positive
generalized eigenvalue $\lambda_*$ of $Bv=\lambda Hv$, with
$H=\operatorname{diag}(d_i/\nu_i)$, controls the weighted-energy decay:

| Quantity | Expression | Physical meaning |
|----------|-----------|------------------|
| Energy decay | $V(t) \le e^{-2\lambda_*t}V(0)$ | Exact upper bound |
| Relaxation scale | $1/\lambda_*$ | Heterogeneous structural time |
| Homogeneous reduction | $\lambda_*=\nu_f\lambda_2(L_{sym})$ | Common-capacity case |

The corresponding exact single-mode Euler theorem is now available for every
fixed connected symmetric rational conductance with positive rational
capacity. If `v=x_0-mean_H(x_0)*1` is nonzero and satisfies the exact identity
`A*v=mu*v` for `A=diag(nu_f)L_rw`, then

```text
x(T) = mean_H(x_0)*1 + exp(-mu*T)*v,
x_P(T) = mean_H(x_0)*1 + product_j(1-mu*h_j)*v.
```

For a positive partition with `0 < mu*h_j < 1`, the factor error is bounded by
`(mu^2/2) sum_j h_j^2 <= (mu^2/2) T h_max`. Multiplication by
`||v||_inf` gives the exact `L_inf` endpoint error; multiplication by
`E_H(v)` times the squared factor error gives its `H`-error energy. Proper
positive subdivision strictly improves the factor and quadratic bound, and the `h_max` estimate
proves conditional exact-real convergence for fixed-data admissible families.
The complete derivation and runtime-scope boundary are centralized in the
[`reversible single-eigenmode theorem`](TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem).
The pure certificate does not inspect binary64 execution, glyphs or REMESH.

This table is the exact real-arithmetic theorem. The executable binary64
certificate separately rationalizes its materialized generator and displayed
metric. It first requires exact invariance of the consensus subspace, then
proves positivity of the symmetrized dissipation on the metric-orthogonal
quotient and returns a downward-rounded certified rate. Its generalized
eigenvalue remains an estimate. Exact conservation of the displayed weighted
mean is a further independent Boolean; without it, the snapshot projection
center is not promoted as the final consensus value.

For time-varying capacities satisfying positive finite per-node bounds, the
Dirichlet energy is a common Lyapunov function with decay rate at least
`2 lambda_2(B) min_i(lower_i/d_i)`. The executable theorem treats the effective
binary64 conductances and capacity bounds as exact real coefficients, then forms
degrees, the Laplacian, the quotient gap and the rate in rational arithmetic.
Ordinary binary64 eigengaps, products and energies remain labelled estimates;
whether `diag(strength)-adjacency` happens to annihilate constants after floating
accumulation is a separate diagnostic. A positive exact rate that underflows on
publication still proves the exact-real theorem but supplies no operational
binary64 rate. The routine neither observes the future capacity schedule nor
certifies a numerical integrator. Within the conditional exact-real model the
field converges to consensus, but the consensus value is schedule-dependent
unless the capacity ratios remain fixed.

For a finite family of changing symmetric topologies on fixed node support,
the normalized metric `d_i/nu_i` is a common Lyapunov metric when the raw
represented metric vectors are exactly proportional across every regime.
Caller-tolerance proximity is diagnostic and cannot promote the theorem. The
minimum certified rational quotient bound then controls arbitrary switching.
Every represented regime must fix the uniform EPI field exactly; the weaker
consensus-subspace condition and preservation of the displayed weighted mean are
reported independently and cannot replace that canonical fixed-point identity.
The object's equilibrium, Lyapunov-value and derivative fields are binary64
diagnostics for its first snapshot and displayed normalized metric. Composition
instead uses the rationalized reference metric and certified rate; a sampled
trajectory must recenter that common metric at every state.
This is a restricted
topology-change theorem; it does not cover node creation/removal or nodal-type
transitions.

**Implementation**:
`physics.structural_diffusion.verify_heterogeneous_diffusion_stability` provides
the exact fixed-capacity certificate;
`derive_time_varying_diffusion_stability_bound` supplies the conditional common
bound; `verify_switching_diffusion_stability` checks the common-metric switching
theorem. `diagnose_euler_relaxation_window` resolves the frozen graph's actual
explicit-Euler modal factors and solver-step relaxation count, using a
dimensionless zero-mode tolerance relative to the fastest decay rate. That count is a
numerical-integration quantity, not the U4 operator-position window.
`physics.lyapunov` retains nominal operator-role diagnostics and must not be read
as a universal convergence prover.

---

## 2. Phase Transitions

### 2.1 Order-parameter candidate

The symmetry-breaking field $\mathcal{S}$ is the implemented order-parameter
candidate for controlled phase-transition sweeps:

$$
\mathcal{S}(i) = \left(|\nabla\phi|^2 - K_\phi^2\right) + \left(J_\phi^2 - J_{\Delta\mathrm{NFR}}^2\right)
$$

### 2.2 Phase Classification

The phase is decided by the standardized spatial imbalance of the signed global mean,
$z = |\langle\mathcal{S}\rangle| / \sqrt{\mathrm{Var}(\mathcal{S})/N}$, and likewise for
signed chirality. Because graph nodes are coupled, this ratio is not a
hypothesis-test z-score without an independent sampling or effective-sample-size
model. The engine retains the historical function name `symmetry_zscore`, but
uses the ratio only as a deterministic classifier input. The selected operational
cut is $z = 1$. The local magnitudes $\langle|\mathcal{S}|\rangle$ and
$\langle|\chi|\rangle$ remain separate telemetry and cannot establish global
symmetry breaking because opposite signs may cancel.

| Phase | Condition | Operational meaning |
|-------|-----------|------------------|
| **NON_LIFE** | $z \le 1$ | signed mean below the selected standardized-imbalance cut |
| **LIFE** | $z > 1$ AND $z_\chi > 1$ | both signed-imbalance ratios exceed the selected cut |
| **CRITICAL** | $z > 1$ AND $z_\chi \le 1$ | order imbalance without a chirality-ratio crossing |

The classifier evaluates the ratio in scaled coordinates before dimensional
variance can underflow. Snapshot and series share that capture path and retain
coherence-length availability/provenance; unavailable xi is not rewritten to
zero. Unrepresentable variance or susceptibility rejects explicitly.

A negative control demonstrates the size boundary: on P3 with phases
`[0, 0.2, 0.6]` and stored pressure `[0, 0.3, 0.6]`, the label is `NON_LIFE`.
The disjoint union of two identical copies is labelled `LIFE`: the local
fields and component dynamics are unchanged, but both ratios acquire a
factor `sqrt(2)`. This is the declared normalization, not autonomous formation.
No production selector consumes these labels. The
[phase controls](../tests/physics/test_phase_transition.py) retain this witness.

### 2.3 Effective time-series exponent fit

`fit_critical_exponent` and `detect_phase_transition` fit the diagnostic model

$$
|\langle\mathcal{S}\rangle| \sim |t-t_c|^{p_{\mathrm{fit}}}
$$

by log-log regression. In the time-series detector, $t_c$ is the time of the
largest sampled susceptibility, when that maximum is positive. The fit uses
only positive samples strictly after $t_c$; it never substitutes earlier
samples. It returns `exponent`/`measured_exponent` and $R^2$, or `None` when fewer
than three eligible post-$t_c$ samples exist.

Positivity, rather than a fixed amplitude/time epsilon, determines eligibility.
Centered shared correlation supplies the fit quality; a tiny imperfect response
does not become a perfect fit through an absolute residual cutoff. An initially
above-threshold sample takes precedence over a later recrossing when reporting
the first observed crossing.

The neutral symbol $p_{\mathrm{fit}}$ avoids assigning competing $\beta$ and
$\gamma$ names to the same implemented regression. The result is
protocol-dependent. It becomes a conventional critical exponent only when the
time coordinate is mapped to a declared control-parameter distance and a
finite-size protocol supports that interpretation. No universal value follows
from the nodal equation.

### 2.4 Classification scale

The classifier uses only `Z_SIGNIFICANCE = 1`, a selected standardized-spread
policy. It is neither a p-value cut nor a graph-independent critical constant.
Legacy constants such as $\pi/16$, `0.034` and `0.155` do not enter this phase
classification and must not be presented as universal transition thresholds.

### 2.5 Susceptibility

The finite-sample structural susceptibility diagnostic is

$$
\chi_{\mathcal{S}}(t) = N \cdot \operatorname{Var}(\mathcal{S})
$$

The implementation records its maximum along the supplied sequence. A sampled
peak is not a divergence theorem.

### 2.6 Finite-size protocol

`analyze_phase_finite_size_scaling` accepts one shared control grid at three or
more node counts, with a balanced replicate axis. It reports the sampled
pseudocritical control, replicate standard errors and power-law slopes of peak
susceptibility, order magnitude and coherence length against node count `N`.
The use of `N` is explicit: converting these slopes to conventional exponent
ratios requires an independently justified linear-size or dimension map.
Replicating one graph family does not establish universality. A susceptibility
maximum on the first or last sampled control value is flagged as unbracketed
instead of being presented as a located critical point. Every exact maximizer
is retained, so plateaus are marked ambiguous and any boundary contact makes
the sampled peak unbracketed.

**Implementation**: `src/tnfr/physics/phase_transition.py` provides snapshot and
time-series classification; `src/tnfr/physics/phase_scaling.py` provides the
balanced finite-size diagnostic.

---

## 3. Life-telemetry diagnostics

### 3.1 Implemented formulas

[`physics.life`](../src/tnfr/physics/life.py) consumes supplied time series; it
does not evolve the graph. It is an assumption-explicit logistic diagnostic,
not a biological classifier or an operator implementation. Every channel must
be a finite numeric one-dimensional series; Boolean, multidimensional, NaN and
infinite inputs are rejected. `detect_life_emergence` additionally requires a
nonempty nonnegative EPI series, matching channel shapes and strictly increasing
sample times. It requires $0\leq\varepsilon\leq1$, $\gamma\geq0$, and
$\mathrm{EPI}_{\max}>0$. No input is clipped or broadcast implicitly.

Let $x(t)\geq0$ be the supplied EPI-magnitude series. The declared model computes

$$
G(t)=\gamma x(t)\left(1-\frac{x(t)}{\mathrm{EPI}_{\max}}\right)
$$

and the **time-local** autopoietic coefficient

$$
A(t)=\frac{G(t)\,\dot x(t)}{|\Delta\mathrm{NFR}_{\mathrm{ext}}(t)|^2+\epsilon_{\mathrm{num}}}.
$$

The numerator and denominator are evaluated element by element. The
implementation does not take the ensemble or time averages shown in older
versions of this note, and `dEPI_dt` supplies $\dot x$ rather than the function
estimating it internally. The small $\epsilon_{\mathrm{num}}$ is the shared
safe division guard.

The remaining returned series are

$$
V_i(t)=\frac{|\varepsilon G(t)|}
{|\varepsilon G(t)|+|\Delta\mathrm{NFR}_{\mathrm{ext}}(t)|+\epsilon_{\mathrm{num}}},
$$

$$
S(t)=\frac{\varepsilon\,|\gamma(1-2x(t)/\mathrm{EPI}_{\max})|}
{|\partial_t\Delta\mathrm{NFR}_{\mathrm{ext}}(t)|+\delta+\epsilon_{\mathrm{num}}},
\qquad
M(t)=\frac{x(t)-\mathrm{EPI}_{\max}/2}{\mathrm{EPI}_{\max}}.
$$

`LifeTelemetry.vitality_index` is the internal-versus-total pressure ratio
$V_i$ above. It does not include $C(t)$; callers may combine the two as a
separate analysis.

### 3.2 Operational threshold detection

`detect_life_emergence` uses $A(t)>1$ as a selected operational event. It
linearly interpolates the first transition from $A\leq1$ to $A>1$, returns the
first supplied time when the series already starts above one, and otherwise
returns `None`. This classifier records a telemetry crossing; it does not prove
future self-sustenance or a biological classification.

The returned `LifeTelemetry` contains the supplied times, $V_i$, $A$, $S$, $M$,
and `life_threshold_time`.

---

## 4. Node lifecycle classifier

### 4.1 Implemented state priority

`get_lifecycle_state` is an instantaneous rule-based classifier. It does not
estimate whether $\nu_f$ or $\Delta\mathrm{NFR}$ is increasing. With the default
parameters, it evaluates conditions in this order:

| Returned state | Implemented condition |
|----------------|-----------------------|
| **COLLAPSING** | $\nu_f<0.01$, or $\lvert\Delta\mathrm{NFR}\rvert>10$, or a non-isolated node has coupling $<0.1$ |
| **MUTATION** | $\lvert\Delta\mathrm{NFR}\rvert>5$ and $\nu_f>0.1$ |
| **PROPAGATION** | coupling $>0.7$ and $\nu_f>0.1$ |
| **STABILIZATION** | $\lvert\Delta\mathrm{NFR}\rvert<1$ and local structural coherence $>0.8$ |
| **ACTIVATION** | $\nu_f\geq0.1$ after the earlier checks |
| **DORMANT** | all remaining states above the collapse-frequency cut |

The stabilization test reads the shared local diagnostic
`structural_coherence(DeltaNFR, dEPI_dt) = 1/(1 + |DeltaNFR| + |dEPI_dt|)`.
It uses the stored pressure and rate, not EPI magnitude or a prediction
reconstructed from the nodal product. The threshold is a configured classifier
policy, not a persistence theorem. `LifecycleState.COLLAPSED` exists in the enum but
`get_lifecycle_state` currently returns `COLLAPSING` for every collapse trigger
and never returns `COLLAPSED`.

For a node with neighbors, the coupling diagnostic uses the shared circular
neighbor mean and shortest-arc phase difference:

$$
\bar\theta_i=\operatorname{atan2}\!\left(\sum_{j\in\mathcal N(i)}\sin\theta_j,
\sum_{j\in\mathcal N(i)}\cos\theta_j\right),\qquad
c_i=1-\frac{|\operatorname{wrap}(\theta_i-\bar\theta_i)|}{\pi}.
$$

The numerical mean convention does not establish a uniquely defined direction
when the neighbor resultant vanishes. Isolates receive $c_i=0$, and the
network-decoupling collapse check applies only when neighbors exist.

### 4.2 Collapse-reason check

`check_collapse_conditions` is a separate predicate. It returns the first
matching reason in this order:

| Collapse reason | Default condition |
|-----------------|-------------------|
| **Frequency failure** | $\nu_f<0.01$ |
| **Extreme dissonance** | $\lvert\Delta\mathrm{NFR}\rvert>10$ |
| **Network decoupling** | non-isolated node with $c_i<0.1$ |
| **EPI dissolution** | scalar EPI $<0.01$ |

The EPI-dissolution condition belongs to this separate predicate and is not
consulted by `get_lifecycle_state`. Configuration values override graph values,
which in turn override these operational defaults.

**Implementation**:
[`operators.lifecycle`](../src/tnfr/operators/lifecycle.py) provides
`LifecycleState`, `CollapseReason`, `get_lifecycle_state`,
`check_collapse_conditions`, and `should_collapse`.

---

## 5. Auxiliary internal Hamiltonian

### 5.1 Implemented matrix

[`operators.hamiltonian`](../src/tnfr/operators/hamiltonian.py) constructs the
finite matrix

$$
H_{\mathrm{int}}=H_{\mathrm{coh}}+H_{\mathrm{freq}}+H_{\mathrm{coupling}},
$$

with the literal implementation

$$
H_{\mathrm{coh}}=C_0 W,\qquad
H_{\mathrm{freq}}=\operatorname{diag}(\nu_{f,1},\ldots,\nu_{f,N}),\qquad
H_{\mathrm{coupling}}=J_0 A_{\mathrm{sym}}.
$$

$W$ is the matrix returned by `coherence_matrix`. The constructor default is
$C_0=-1$, so the coherence term is $-W$; writing an additional leading minus
sign reverses the implemented sign. The default coupling is $J_0=0.1$, and the
builder writes both matrix directions for every graph edge. The constructor
requires finite entries in every component and their sum, then checks an
absolute Hermiticity residual tolerance of `1e-10`. The auxiliary scale must
be a finite nonzero real number. Acceptance is not an exact algebraic
Hermiticity certificate.

This matrix supplies an auxiliary linear model. The repository does not derive
the general engine trajectory or the canonical graph $\Delta\mathrm{NFR}$ from
it.

### 5.2 Unitary flow and eigenmodes

For an exactly Hermitian matrix and real nonzero declared
$\hbar_{\mathrm{str}}$, the ideal evolution is unitary:

$$
U(t)=\exp\left(-\frac{iH_{\mathrm{int}}t}{\hbar_{\mathrm{str}}}\right).
$$

`time_evolution_operator` numerically evaluates this exponential and applies
an `allclose` unitarity check. `get_spectrum` uses a Hermitian eigensolver;
its returned ascending eigenvalues and eigenvectors numerically approximate

$$
H_{\mathrm{int}}|\phi_n\rangle=E_n|\phi_n\rangle.
$$

For the exact Hermitian model, an eigenvector evolves only by the phase
$e^{-iE_nt/\hbar_{\mathrm{str}}}$ in this auxiliary unitary flow. That makes it
a stationary ray of this model; it does not make it a maximally stable TNFR
configuration or establish dissipative attraction.

### 5.3 Compatibility helpers and sign scope

Despite its name, `compute_delta_nfr_operator()` does not compute a commutator.
It literally returns

$$
G_+=\frac{i}{\hbar_{\mathrm{str}}}H_{\mathrm{int}},
$$

which is anti-Hermitian and is the negative of the ket-state generator
$-iH_{\mathrm{int}}/\hbar_{\mathrm{str}}$ used by $U(t)$.

`compute_node_delta_nfr(n)` implements the localized-projector read-out
for $\rho_n=|n\rangle\langle n|$, defined by the real part of

$$
\frac{i}{\hbar_{\mathrm{str}}}
\langle n|[H_{\mathrm{int}},\rho_n]|n\rangle.
$$

For this localized projector, the displayed diagonal commutator is exactly
zero: $[H,\rho_n]_{nn}=H_{nn}-H_{nn}=0$. The implementation returns this
identity directly after checking captured node membership, without matrix
products. Under the implemented unitary
$U\rho U^\dagger$, the density-matrix derivative would instead carry the sign
$-i[H,\rho]/\hbar_{\mathrm{str}}$. These helpers therefore do not reconstruct
the engine's node-local structural pressure; canonical $\Delta\mathrm{NFR}$ is
computed by
[`dynamics.dnfr`](../src/tnfr/dynamics/dnfr.py).

Selecting its opt-in `compute_delta_nfr_hamiltonian` hook writes zero stored
pressure at every node and labels that null read-out in pressure metadata.
It supplies no density evolution or independently derived pressure law. The
hook rebuilds its auxiliary matrix on each invocation, so capacity, support,
affinity configuration and scale changes are not hidden by a node-count cache.
`cache_hamiltonian=True` retains only the latest constructed snapshot for
inspection; it does not reuse that snapshot on the next call. Regression
coverage belongs to
[`test_hamiltonian_pressure_scope.py`](../tests/operators/test_hamiltonian_pressure_scope.py).

---

## 6. Optional structural integrity tools

### 6.1 Reactive monitor

The reactive monitor is opt-in. `enable_integrity_monitor(G, mode=...)` creates
a `StructuralIntegrityMonitor` and stores it in
`G.graph["integrity_monitor"]`. Calls through the operator-class pipeline then
invoke `before_operator` and `after_operator`. Without an attached monitor,
ordinary operator calls do not run these diagnostics.

For each monitored call, the implementation attempts to compare conservation
snapshots, the finite change of the structural-energy candidate, heuristic
grammar-violation labels, Noether-charge drift, and an operator-specific
postcondition. These are runtime diagnostics. They do not turn the
five-term energy into a general Lyapunov theorem. `ENFORCE` raises after a
reported unhealthy result; it does not roll back an operator mutation.

An `IntegrityReport` is healthy only when its conservation quality is above
`0.7`, its sampled energy change is classified stable, no grammar diagnostics
are present, and its postcondition check passes. Charge drift is reported but
is not part of that property.

### 6.2 Implemented postcondition registry

`POSTCONDITIONS` has one entry for every canonical operator name, but several
entries check only a measurable proxy and REMESH is explicitly advisory:

| Operator | Check currently performed by the reactive registry |
|----------|----------------------------------------------------|
| **AL** | EPI does not decrease; $\nu_f$, phase and $\Delta\mathrm{NFR}$ do not change |
| **EN** | immediate operator-local $C(t)$ is unchanged before pressure refresh |
| **IL** | $C(t)$ does not decrease and $\lvert\Delta\mathrm{NFR}\rvert$ does not increase |
| **OZ** | $\lvert\Delta\mathrm{NFR}\rvert$ does not decrease |
| **UM** | $\lvert\Delta\mathrm{NFR}\rvert$ does not increase |
| **RA** | nonzero EPI sign is preserved and $\nu_f$ does not decrease |
| **SHA** | EPI is unchanged and $\nu_f$ does not increase |
| **VAL** | $\nu_f$ does not decrease |
| **NUL** | $\nu_f$ does not increase |
| **THOL** | $C(t)$ does not fall by more than 10% |
| **ZHIR** | delegates phase, identity, and bifurcation checks to the mutation postcondition module |
| **NAV** | at least one of $\nu_f$, $\theta$, or $\Delta\mathrm{NFR}$ changes |
| **REMESH** | advisory entry; returns success without a network-remesh check |

The U3 phase-compatibility gate for UM and RA is a hard precondition in their
operator pipeline, separate from the reactive postcondition table. Registry
lookup uses the lower-case English function name (`"coherence"`,
`"self_organization"`, and so on); a glyph string such as `"IL"` does not
select the corresponding registry checker.

#### ZHIR temporal evidence and execution boundary

The [Mutation contract](STRUCTURAL_OPERATORS.md#91-mutation-zhir) owns trigger
admission, history precedence, SDK abstention and identity/provenance semantics.
The nodal product `predicted_depi_dt = nu_f * DeltaNFR` is an instantaneous
prediction; it does not replace the observed signed two-sample secant required
by the strict gate `observed_depi_dt > xi`. Active finite capacity and independent
U4b context are also required.

Timestamped evidence must be finite, have a positive interval and end at the
live EPI state. Invalid authoritative history cannot fall through to a legacy
source. Untimestamped compatibility histories declare operator-step units,
not elapsed physical time. Missing observations remain `None`; `rate_gap` is
available only when observed and predicted rates share the physical-time basis.
The [event Mutation boundary](../docs/contracts/OPERATOR_EVENTS.md#mutation-trigger-and-refinement-observations)
adds timestamp/declared-duration agreement and execution provenance; these
claims are separate from gate crossing.

The three-sample `compute_d2epi_dt2` diagnostic and the branch-selection capacity
threshold are also distinct from the two-sample trigger. Neither a predicted
crossing, an acceleration alert nor a reactive postcondition establishes
bifurcation, grammar readiness or future stability.

Registry coverage is therefore not a proof that every full operator contract
has been verified. For a reproducible catalog-level measurement, use
`audit_operator_contracts`. It builds controlled graphs, places each request in
its intended context, checks the actually appended glyph so a grammar fallback
cannot certify the request, and returns an `OperatorContractAudit`. Its REMESH
case remains an advisory network-level result.

### 6.3 Monitor modes

| Mode | Behaviour |
|------|-----------|
| **OFF** | Hook methods return default data without metric computation |
| **OBSERVE** | Record reports and violations without raising |
| **ENFORCE** | Raise `StructuralIntegrityViolation` on failure |

### 6.4 Suggestions and SDK scope

Corrective suggestions are generated for recognized conservation-derived
grammar labels, an increasing energy-candidate observation, or charge drift
above the internal alert. A postcondition failure alone need not produce a
suggestion.

The SDK exposes two different dictionary-returning conveniences:

- `Network.integrity_check(operator_name)` calls `after_operator` directly for
  at most ten current nodes and returns `operator`, `nodes_checked`, `passed`,
  `failed`, `pass_rate`, and per-node `reports`. It does not execute an
  operator, capture a matching before snapshot, attach the monitor, or audit
  all 13 operators. Use an English function name to activate a registry check.
- `Network.audit_operators()` runs the independent controlled
  `audit_operator_contracts` protocol and returns a dictionary with the 13
  contextual results and its summary. It audits the operator implementation,
  rather than the current `Network` instance's trajectory.

**Implementation**:
[`physics.integrity`](../src/tnfr/physics/integrity.py) provides
`IntegrityReport`, `IntegritySummary`, `MonitorMode`,
`StructuralIntegrityViolation`, `POSTCONDITIONS`, and
`audit_operator_contracts`.

**Tests**:
[`tests/physics/test_structural_integrity.py`](../tests/physics/test_structural_integrity.py)
and
[`tests/sdk/test_simple_advanced.py`](../tests/sdk/test_simple_advanced.py).

---

## Implementation and examples

| Module | Content |
|--------|---------|
| `src/tnfr/physics/lyapunov.py` | Nominal U2-role multipliers and spectral diagnostics |
| `src/tnfr/physics/structural_diffusion.py` | Fixed, time-varying and exact-common-metric pure-EPI flow certificates |
| `src/tnfr/physics/hybrid_operator_stability.py` | Declared affine-reset gains and hybrid flow/reset budgets |
| `src/tnfr/physics/reception_realization.py` | Read-only EN runtime-to-affine-flow boundary |
| `src/tnfr/physics/resonance_realization.py` | Read-only identity-gated RA runtime-to-affine-flow boundary and post-RA metric audit |
| `src/tnfr/physics/network_stage_stability.py` | All-target EN/RA certificates and the validated one-stage positive-duration post-flow bridge |
| `src/tnfr/physics/pointwise_stage_stability.py` | Executor-bound pointwise affine realization and gain levels |
| `src/tnfr/operators/event_timing.py` | Exact finite flow/jump schedules and binary64 clock readiness |
| `src/tnfr/operators/event_runtime.py` | Atomic observed flow/glyph binding and finite represented EPI-map composition |
| `src/tnfr/operators/event_remesh_runtime.py` | Atomic schedule/delayed-REMESH cycle with separate evidence channels |
| `src/tnfr/operators/event_remesh_sequence.py` | Exact continuity across ordered supplied cycle results |
| `src/tnfr/operators/event_remesh_causal_runtime.py` / `src/tnfr/operators/event_remesh_causal_runtime.pyi` | One graph-owned finite causal cycle sequence and exact public interface |
| `src/tnfr/physics/event_refinement.py` | Offline and executor-linked event-local ZHIR evidence |
| `src/tnfr/physics/event_remesh_refinement.py` | Finite strict three-mesh event/REMESH observations |
| `src/tnfr/physics/event_remesh_reference.py` / `src/tnfr/physics/event_remesh_reference.pyi` | Effective-P2 finite reference-family certificate and exact public interface |
| `src/tnfr/physics/reversible_eigenmode_reference.py` / `src/tnfr/physics/reversible_eigenmode_reference.pyi` | General exact-rational reversible single-eigenmode Euler theorem and public interface |
| `src/tnfr/physics/runtime_eigenmode_reference.py` / `src/tnfr/physics/runtime_eigenmode_reference.pyi` | Finite executor binding with exact pressure/execution defects and full-matrix propagation |
| `src/tnfr/physics/remesh_history_stability.py` | Exact uniform finite companion-history Lyapunov certificate |
| `src/tnfr/physics/remesh_schedule_policy_stability.py` / `src/tnfr/physics/remesh_schedule_policy_stability.pyi` | Conditional exact uniform REMESH/schedule spatial-disagreement theorem |
| `src/tnfr/physics/remesh_schedule_relative_defect_stability.py` / `src/tnfr/physics/remesh_schedule_relative_defect_stability.pyi` | Conditional exact `q_eff=q*(1+eta)` robust policy theorem |
| `src/tnfr/physics/binary64_remesh_relative_defect.py` / `src/tnfr/physics/binary64_remesh_relative_defect.pyi` | Exact pairwise REMESH boundary; uniform `alpha=1`, `eta=0` class; and sharp `alpha=1/2` antisymmetric P2 class with `eta=135/124` |
| `src/tnfr/physics/binary64_p2_reception_stability.py` / `src/tnfr/physics/binary64_p2_reception_stability.pyi` | Global `q=0` P2 half-Reception kernel composed with the `alpha=1` REMESH class |
| `src/tnfr/physics/runtime_p2_reception_stage.py` / `src/tnfr/physics/runtime_p2_reception_stage.pyi` | Finite executor binding of one P2 two-phase EN EPI stage to the global `q=0` kernel |
| `src/tnfr/physics/runtime_p2_reception_remesh_sequence.py` / `src/tnfr/physics/runtime_p2_reception_remesh_sequence.pyi` | Finite causal binding of executed P2 EN/REMESH cycles and observed active-suffix extinction |
| `src/tnfr/physics/runtime_p2_reception_remesh_policy.py` / `src/tnfr/physics/runtime_p2_reception_remesh_policy.pyi` | Transactional per-invocation P2 preflight, execution and finite certification |
| `src/tnfr/physics/runtime_remesh_history_stability.py` | One-transition executed binary64 REMESH/companion bridge |
| `src/tnfr/physics/remesh_schedule_stability.py` | Exact REMESH-head/schedule-head augmented-energy balance |
| `src/tnfr/physics/runtime_remesh_schedule_stability.py` | Adjacent-cycle runtime/history energy telescope |
| `src/tnfr/physics/runtime_remesh_schedule_block_margin.py` / `src/tnfr/physics/runtime_remesh_schedule_block_margin.pyi` | Exact normalized margin for a contiguous causally executed finite block |
| `src/tnfr/physics/runtime_remesh_schedule_relative_defect.py` / `src/tnfr/physics/runtime_remesh_schedule_relative_defect.pyi` | Finite causal verification of signed defects and the robust policy envelope |
| `src/tnfr/operators/_delayed_remesh_kernel.py` | Immutable delayed REMESH proposals and one-step evidence |
| `src/tnfr/physics/phase_quotient.py` | Fixed-branch pairwise quotient, restricted canonical phase lift and counterexample |
| `src/tnfr/physics/coherence_geometry.py` | Local, fixed-network and fixed-capacity coherence strata |
| `src/tnfr/physics/phase_transition.py` | Order parameter, operational phase classification, effective exponent fit |
| `src/tnfr/physics/life.py` | Strict supplied-series logistic diagnostics and selected $A(t)>1$ event |
| `src/tnfr/operators/lifecycle.py` | Instantaneous node-state and collapse predicates |
| `src/tnfr/operators/hamiltonian.py` | Auxiliary matrix, unitary flow, spectrum, compatibility helpers |
| `src/tnfr/physics/integrity.py` | Optional reactive monitor and contextual operator audit |

The finite three-mesh contracts are falsified and sealed by
[`test_event_remesh_refinement.py`](../tests/physics/test_event_remesh_refinement.py).
The effective-P2 reference family and public example are checked by
[`test_event_remesh_reference.py`](../tests/physics/test_event_remesh_reference.py)
and
[`test_event_remesh_reference_example.py`](../tests/physics/test_event_remesh_reference_example.py).
The general reversible exact-mode theorem, sealing and public `P3` example are
checked by
[`test_reversible_eigenmode_reference.py`](../tests/physics/test_reversible_eigenmode_reference.py)
and
[`test_reversible_eigenmode_reference_example.py`](../tests/physics/test_reversible_eigenmode_reference_example.py).
The finite executor binding, proof seals and public executed `P3` example are
checked by
[`test_runtime_eigenmode_reference.py`](../tests/physics/test_runtime_eigenmode_reference.py)
and
[`test_runtime_eigenmode_reference_example.py`](../tests/physics/test_runtime_eigenmode_reference_example.py).
The finite companion-history theorem is checked exactly by
[`test_remesh_history_stability.py`](../tests/physics/test_remesh_history_stability.py).
The common-`q` exact policy theorem, public facade and example are checked by
[`test_remesh_schedule_policy_stability.py`](../tests/physics/test_remesh_schedule_policy_stability.py)
and
[`test_remesh_schedule_policy_stability_example.py`](../tests/physics/test_remesh_schedule_policy_stability_example.py).
The robust signed-defect theorem and its finite causal adapter are checked by
[`test_remesh_schedule_relative_defect_stability.py`](../tests/physics/test_remesh_schedule_relative_defect_stability.py),
[`test_runtime_remesh_schedule_relative_defect.py`](../tests/physics/test_runtime_remesh_schedule_relative_defect.py)
and
[`test_runtime_remesh_schedule_relative_defect_example.py`](../tests/physics/test_runtime_remesh_schedule_relative_defect_example.py).
The binary64 REMESH boundary, P2 numeric EPI-kernel composition and public
examples are checked by
[`test_binary64_remesh_relative_defect.py`](../tests/physics/test_binary64_remesh_relative_defect.py),
[`test_half_alpha_antisymmetric_remesh_class.py`](../tests/physics/test_half_alpha_antisymmetric_remesh_class.py),
[`test_binary64_p2_reception_stability.py`](../tests/physics/test_binary64_p2_reception_stability.py),
[`test_binary64_remesh_relative_defect_example.py`](../tests/physics/test_binary64_remesh_relative_defect_example.py),
[`test_half_alpha_antisymmetric_remesh_class_example.py`](../tests/physics/test_half_alpha_antisymmetric_remesh_class_example.py)
and
[`test_binary64_p2_reception_stability_example.py`](../tests/physics/test_binary64_p2_reception_stability_example.py).
The finite executed-stage binding and example are checked by
[`test_runtime_p2_reception_stage.py`](../tests/physics/test_runtime_p2_reception_stage.py)
and
[`test_runtime_p2_reception_stage_example.py`](../tests/physics/test_runtime_p2_reception_stage_example.py).
The finite causal P2 EN/REMESH sequence, active-suffix boundary and public
example are checked by
[`test_runtime_p2_reception_remesh_sequence.py`](../tests/physics/test_runtime_p2_reception_remesh_sequence.py)
and
[`test_runtime_p2_reception_remesh_sequence_example.py`](../tests/physics/test_runtime_p2_reception_remesh_sequence_example.py).
The reusable policy, its rollback boundaries and two-call example are checked by
[`test_runtime_p2_reception_remesh_policy.py`](../tests/physics/test_runtime_p2_reception_remesh_policy.py)
and
[`test_runtime_p2_reception_remesh_policy_example.py`](../tests/physics/test_runtime_p2_reception_remesh_policy_example.py).
The runtime residual and lifted-energy bridge is tested in
[`test_runtime_remesh_history_stability.py`](../tests/physics/test_runtime_remesh_history_stability.py).
The pure schedule-head telescope and gain-based lower bound are tested in
[`test_remesh_schedule_stability.py`](../tests/physics/test_remesh_schedule_stability.py).
The adjacent runtime schedule binding and finite history telescope are tested in
[`test_runtime_remesh_schedule_stability.py`](../tests/physics/test_runtime_remesh_schedule_stability.py).
The finite same-invocation causal wrapper, receipt bindings, graph rollback and
public example are tested in
[`test_event_remesh_causal_runtime.py`](../tests/operators/test_event_remesh_causal_runtime.py)
and
[`test_event_remesh_causal_runtime_example.py`](../tests/operators/test_event_remesh_causal_runtime_example.py).
The exact finite-block identities, normalized diagnostics, public facade and
`139/256` versus zero-margin examples are checked by
[`test_runtime_remesh_schedule_block_margin.py`](../tests/physics/test_runtime_remesh_schedule_block_margin.py)
and
[`test_runtime_remesh_schedule_block_margin_example.py`](../tests/physics/test_runtime_remesh_schedule_block_margin_example.py).

### SDK Entry Points

```python
from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)
monitor = enable_integrity_monitor(net.G, mode=MonitorMode.OBSERVE)
# Subsequent operator-class calls append IntegrityReport objects to monitor.summary.

snapshot = net.integrity_check("coherence")  # dict; up to ten current nodes
catalog = net.audit_operators()               # dict; 13 controlled probes
```

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [29_lyapunov_stability_demo.py](../examples/02_physics_regimes/29_lyapunov_stability_demo.py) | Nominal operator-role multipliers, measured energy diagnostics, spectral read-outs, and life telemetry |
| [161_core_research_trajectory.py](../examples/02_physics_regimes/161_core_research_trajectory.py) | Sampled pure-EPI path, modal limit, common energy budget and two-mesh comparison |
| [162_hybrid_epi_stability.py](../examples/02_physics_regimes/162_hybrid_epi_stability.py) | Affine amplification absorbed by diffusion, consensus drift, and the infinite-gain local-offset witness |
| [163_reception_runtime_bridge.py](../examples/02_physics_regimes/163_reception_runtime_bridge.py) | EN ideal-real, represented, runtime-snapshot and pressure-refresh boundary |
| [164_resonance_runtime_bridge.py](../examples/02_physics_regimes/164_resonance_runtime_bridge.py) | U3-filtered RA identity gate, four realization layers, post-RA flow certificate and switching abstention |
| [166_event_remesh_reference_family.py](../examples/02_physics_regimes/166_event_remesh_reference_family.py) | Effective-P2 `2/4/8`-segment finite Euler/REMESH reference family and explicit false scope |
| [167_reversible_eigenmode_reference.py](../examples/02_physics_regimes/167_reversible_eigenmode_reference.py) | Both exact nonuniform modes of nonregular `P3`, with exact-real Euler refinement bounds and explicit runtime abstention |
| [168_runtime_reversible_eigenmode_reference.py](../examples/02_physics_regimes/168_runtime_reversible_eigenmode_reference.py) | Finite `2/4/8`-segment executed nonregular-`P3` binding with nonzero `rho`, `eta`, `epsilon` and explicit false runtime-convergence scope |
| [169_event_remesh_causal_runtime.py](../examples/02_physics_regimes/169_event_remesh_causal_runtime.py) | One finite same-invocation event/REMESH cycle sequence with causal receipts, graph-owned atomicity, and the lag-one `alpha=1` stability boundary |
| [170_runtime_remesh_block_margin.py](../examples/02_physics_regimes/170_runtime_remesh_block_margin.py) | Exact `kappa=139/256` finite-block lower margin and the causal `alpha=1`, `kappa=0` boundary, with uniform-class and prefix claims withheld |
| [171_remesh_schedule_policy_stability.py](../examples/02_physics_regimes/171_remesh_schedule_policy_stability.py) | Conditional exact common-`q` policy theorem with prefix gain upper bound one, uniform block margin `1-q`, strict pure-delay disagreement decay and the zero-margin `q=1` boundary |
| [172_runtime_remesh_relative_defect.py](../examples/02_physics_regimes/172_runtime_remesh_relative_defect.py) | Exact `q_eff=q*(1+eta)` theorem bound to zero-defect and positive-binary64-defect finite causal blocks, with forward invariance and future stability withheld |
| [173_binary64_remesh_relative_defect.py](../examples/02_physics_regimes/173_binary64_remesh_relative_defect.py) | Exact pairwise `alpha=1/2` obstruction and the forward-invariant REMESH-only `alpha=1`, `eta=0` hard-clip class |
| [174_binary64_p2_reception_remesh_stability.py](../examples/02_physics_regimes/174_binary64_p2_reception_remesh_stability.py) | Global `q=0` P2 half-Reception numeric kernel composed with `alpha=1`, giving active-history extinction after `tau_global+1` restricted cycles |
| [175_runtime_p2_reception_stage.py](../examples/02_physics_regimes/175_runtime_p2_reception_stage.py) | One executed two-phase P2 EN EPI stage bound to the global `q=0` kernel, with REMESH graph binding and repeated runtime withheld |
| [176_runtime_p2_reception_remesh_sequence.py](../examples/02_physics_regimes/176_runtime_p2_reception_remesh_sequence.py) | One completed same-invocation P2 EN/REMESH sequence with `N >= tau_global+1` and finite observed active-history extinction |
| [177_runtime_p2_reception_remesh_policy.py](../examples/02_physics_regimes/177_runtime_p2_reception_remesh_policy.py) | Two successive independently validated finite P2 policy invocations with future and auxiliary-state claims withheld |
| [178_half_alpha_antisymmetric_remesh_class.py](../examples/02_physics_regimes/178_half_alpha_antisymmetric_remesh_class.py) | Sharp `eta=135/124` for the REMESH-only `alpha=1/2` antisymmetric P2 class, exact IEEE tail/core proof, strict and zero-margin `q` boundaries, and excluded generalizations |

## Cross-References

- Structural-energy candidate and conservation diagnostics: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) §8
- Grammar U2 (stabilizer/debt policy): [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Hamiltonian/Lagrangian formulation: [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md)
- Order parameter $\mathcal{S}$: [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) §3.2
- Dissipative extensions: [DISSIPATIVE_AND_OPEN_SYSTEMS.md](DISSIPATIVE_AND_OPEN_SYSTEMS.md)
- Gauge structure: [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md)
