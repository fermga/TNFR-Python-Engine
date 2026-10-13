# Mathematical Scope of Structural Diagnostics and Grammar

**Status:** Exact identities under stated hypotheses, canonical engine policies,
and open sufficiency questions. This note fixes the distinction between these
three categories without changing the operator catalog or numerical policies.

The nodal equation is

    x'(t) = ν_f(t) p(t),    x = EPI,    p = ΔNFR.

It specifies the relation between structural change, capacity, and pressure.
A theorem about its solutions also needs a pressure law, graph assumptions,
initial conditions, and, for a discrete evolution, an integrator and step size.
Operator contracts supply additional constraints; they do not follow from the
scalar identity without those premises.

## 1. Existence, boundedness, and convergence

If f(t) = ν_f(t)p(t) is locally integrable on a finite-dimensional structural
coordinate space, then

    x(t) = x(t₀) + ∫[t₀,t] f(s) ds

defines an absolutely continuous trajectory on each finite interval, with the
nodal equation holding almost everywhere. Continuity of f gives the usual
pointwise derivative. In particular, x = 0 is not a singularity of this equation:
ν_f = 1 and p = 1 give x' = 1 at zero. U1's generator requirement is an
initialization contract for the operator language, including its supported
pre-existing-structure context, rather than a claim that this derivative is
undefined.

On an infinite horizon the following properties are different:

| Property | Exact statement |
|----------|-----------------|
| Bounded trajectory | The partial integrals of f remain bounded. |
| Convergence to a finite limit | The vector improper integral of f converges. |
| Absolute integrability | The integral of the norm of f is finite. This is sufficient for convergence and gives finite total variation. |

Neither boundedness nor p(t) tending to zero establishes the last two properties:

- x(t) = sin(t), ν_f = 1, p(t) = cos(t) is bounded and satisfies the nodal
  equation, but x(t) has no limit.
- p(t) = 1/(1+t), ν_f = 1 gives x(t) = x(0) + log(1+t). Its pressure is bounded
  and tends to zero, while its accumulated displacement diverges.
- On a finite connected undirected graph, fixed homogeneous ν_f > 0 and the
  pure EPI law p = −L_rw x give autonomous negative feedback. Relaxation does
  not mathematically require an additional named operator during that flow.

An example of a sufficient analytic assumption is

    ||ν_f(t)p(t)|| ≤ M exp(−a(t−t₀)),    M < ∞, a > 0.

Then the tail displacement is at most (M/a) exp(−a(t−t₀)). For repeated operator
applications, proving such a bound requires control of their gains, elapsed
times, and intervening feedback. U2's prescribed stabilizers and debt accounting
remain mandatory engine policy; grammar acceptance by itself is not an
infinite-horizon convergence certificate.

## 2. Modal relaxation and the grammar calibration

On a fixed undirected graph with positive edge weights, homogeneous frequency,
and pure EPI diffusion, each nonstationary mode of an explicit Euler step has
multiplier

    q_k = 1 − ν_f dt λ_k.

Its amplitude after n steps is q_kⁿ times the initial amplitude. Decay requires
|q_k| < 1; a negative multiplier alternates sign and need not decay. The
continuous solution instead has multiplier exp(−ν_f λ_k t). These are different
evolution laws at finite dt.

For a loopless graph without isolated vertices, trace(L_rw)/N = 1. This is the
mean of all eigenvalues, including stationary modes, not the relaxation rate
of each mode. Isolates and self-loops require different trace accounting.

The functions derive_bifurcation_window_from_physics and
derive_u2_debt_capacity_from_physics retain their public names and formulas.
They use the fixed surrogate rate ρ = 1 and q = 1 − ν_f dt ρ. At the canonical
ν_f = 1 and dt = 0.5:

- The first n with qⁿ < 1/(π+1) is **3**, the U4b recency policy.
- The floor of 1/(1−q) is **2**, the U2 debt-capacity policy.

The geometric sum describes a scalar recurrence with unit forcing and
0 ≤ q < 1. Its finite steady state is not a maximum physical pressure or a
criterion for integrability of a sustained forcing. Mapping its floor to a
count of destabilizing operators is the engine's calibration choice.

The helpers admit finite positive nonboolean inputs only when their represented
product satisfies `0 < nu_f*dt <= 1`. Rational evaluation then uses that
binary64 product and the represented target band. The exact `q=0` surrogate
crosses in one step; negative or invalid multipliers are rejected. The U4
search remains capped at 64 positions even when the target is not reached.
U2 evaluates the reciprocal floor exactly, including subnormal rates. These
are scalar grammar calibrations, not stability results for graph Euler modes.

**Finite-graph witness.** On the 21-node unweighted path,

    λ₂ = 1 − cos(π/20) = 0.012311659404862...
    q₂³ = (1 − 0.5 λ₂)³ = 0.981645960340198...
    1/(π+1) = 0.241453007005224...

The Fiedler amplitude needs 231 steps to cross that target. The same graph
still has mean eigenvalue 1. Thus the three-operation grammar window is not a
mode-uniform relaxation theorem. A trajectory bound must use the relevant
spectrum, norm, Euler stability region, stationary-mode treatment, and
operator-induced gains. Directed non-normal and heterogeneous-frequency
generators require their own analysis.

The read-only
[`diagnose_euler_relaxation_window`](../src/tnfr/physics/structural_diffusion.py)
now performs that analysis for frozen symmetric pure-EPI diffusion with
heterogeneous frequency. It returns the actual modal multipliers, explicit-Euler
stability limit and first solver step below a declared target. It also reports
the canonical U4 window for comparison while keeping the units distinct:
solver steps are not operator positions. Directed non-normal transport and any
policy mapping remain open.

## 3. Assumption-explicit model thresholds

[`physics.life`](../src/tnfr/physics/life.py) is a supplied-series diagnostic
model, not a consequence of the nodal equation and not a biological
classifier. It rejects Boolean, non-finite, multidimensional and implicitly
broadcast inputs; the combined detector additionally requires nonnegative EPI,
matched lengths and strictly increasing times. Its selected event $A(t)>1$
means only that the declared time-local ratio
$G(t)\dot x(t)/(|\Delta\mathrm{NFR}_{\rm ext}(t)|^2+\epsilon_{\rm num})$
exceeds one on the supplied samples. The reported time is the first sample
already above one or the linear interpolation of the first upward crossing. It
does not establish persistence, autonomy or biological life. Formulas and
parameter domains are centralized in
[Structural Stability and Dynamics](STRUCTURAL_STABILITY_AND_DYNAMICS.md#3-life-telemetry-diagnostics).

## 4. Structural potential and topology-dependent bounds

For a fixed distance convention, define

    B_ij = d(i,j)⁻² for reachable j ≠ i with 0 < d(i,j) < infinity; otherwise 0,
    Φ_s = B p.

The field is linear in pressure. The triangle inequality gives the exact bound

    |Φ_s(i)| ≤ Σ_j B_ij |p_j|,
    ||Φ_s||∞ ≤ b_G ||p||∞,    b_G = max_i Σ_j B_ij.

For two pressure fields on the same graph,

    ||Φ_s(t₁) − Φ_s(t₀)||∞ ≤ b_G ||p(t₁) − p(t₀)||∞.

If the graph changes, an additional term is necessary. Writing B₀ and B₁ for
the two kernels gives

    Φ₁ − Φ₀ = B₁(p₁−p₀) + (B₁−B₀)p₀.

These bounds explicitly depend on pressure, graph geometry, and normalization.
There is no phase angle in Bp. Setting every phase to zero and p_j = 1 on K₄
gives Φ_s(i) = 3 at every vertex. On K_n it gives n−1, and multiplying p by a
multiplies Φ_s by a. A transition from p = 0 to p = 1 has the same n−1 drift.

Consequently the canonical **π/4 per-node** and **π/2 drift** thresholds are
selected safety policies. A state can exceed them and trigger telemetry; phase
wrapping does not enforce them. Establishing them as trajectory bounds would
require explicit pressure and topology hypotheses implying the inequalities
above. U6 remains a read-only monitor, separate from the word grammar.

Its implemented statistic is the mean absolute nodewise drift from a declared
reference, not the maximum drift or an intermediate-time measurement. Equal
endpoint fields can hide an excursion between them. For a fixed reference
and smooth aligned kernels, y=Bp-Phi_reference satisfies
y'=B p'+B' p. A forward-confinement theorem for ||y||_1/N would need a
well-posed law and inward boundary conditions, plus compatible event jumps.
The nodal EPI identity and word labels do not supply those derivatives.
Rescaling every distance by a>0 gives B_new=a^-2 B; a fixed numerical
pi/2 policy therefore depends on the declared distance convention.

The inverse-square exponent α = 2 is the canonical kernel choice. Chain
summability does not uniquely select it: Σ d⁻α converges for every α > 1,
and independent unit-variance pressure gives variance Σ d⁻²α, convergent for
every α > 1/2. On a graph family with shell counts bounded by C r^(d−1),
α > d is sufficient for absolute accumulation and 2α > d for independent
unit-variance accumulation. These are sufficient shell-growth conditions;
they need not hold on every graph family. Particular ζ(2) and ζ(4) values do
not establish universal π-fraction confinement.

In contrast, the phase definitions give exact kinematic bounds:

    |∇φ| ≤ π,    |K_φ| ≤ π.

The curvature warning level 0.9π is a selected margin inside that bound. The
phase-gradient warning level π/16 is also a policy value.

## 5. Diagnostic channels and state reconstruction

The tetrad selects four useful readouts: source accumulation Φ_s, local phase
mismatch |∇φ|, local circular curvature K_φ, and correlation length ξ_C. These
are canonical telemetry channels. Canonical status does not establish that
their values reconstruct all state variables or every independent observable.

On a graph, edge derivatives and Laplacian powers can be generated by composing
a gradient and a Laplacian. Algebraic generation does not imply that successive
operators, or lossy scalar summaries of their outputs, are linearly dependent.
If L has at least four distinct eigenvalues, I, L, L², and L³ are linearly
independent: a dependence would give a polynomial of degree at most three
vanishing at four distinct points. More generally, independence is controlled
by the minimal polynomial, not a universal second-order cutoff.

An independent obstruction comes from the nodal equation. At fixed graph,
phase, and pure-EPI pressure, rescaling homogeneous ν_f leaves the tetrad
unchanged but rescales x' = ν_f p. Therefore the tetrad does not determine the
full dynamical state. Canonical coherence also depends on pressure amplitude:
with dEPI = 0 and uniform |p| = 1 it equals 1/2; doubling pressure gives 1/3.
Only the separate normalized dispersion statistic is scale invariant.

A reconstruction or minimality theorem must specify the state space, gauge
identifications, full fields versus node/global summaries, allowed observables,
and the desired equivalence relation. A diagnostic removal study instead needs
explicit fixtures and detection thresholds. Neither a test count nor an
operator-composition identity supplies those missing hypotheses. Completeness
and minimality in this stronger sense remain open.

For fixed connected undirected pure EPI diffusion, this specification yields a
conditional reconstruction result. The full nodal potential has observation
operator `O=-K L_rw`; if `rank(O)=N-1`, it recovers EPI modulo a uniform shift.
The conserved `d_i/nu_i` mean supplies that last scalar. Capacity is still
required to determine the generator. The repository certificate evaluates the
rank per graph; the condition has been measured on the 142 connected simple
graph-atlas cases through six nodes and is not asserted universally.

## 6. Authoritative entry points

- [Unified grammar](UNIFIED_GRAMMAR_RULES.md): current U1–U6 requirements.
- [Tetrad scope](MINIMAL_STRUCTURAL_DEGREES.md): interpretation and limitations.
- [Field definitions and API](../docs/STRUCTURAL_FIELDS_TETRAD.md).
- [Canonical calibration functions](../src/tnfr/config/physics_derivation.py).
- [Threshold values](../src/tnfr/constants/canonical.py).
- The deterministic finite-graph witnesses used by this scope statement are
  recorded directly in the sections above and covered by the linked tests.

## 7. Derived observables and dynamical closure

A diagnostic can be a function of nodal structure without being a primitive
force or a dynamically sufficient emergent variable. Write the full declared
state as z and an observable as y=f(z). If a closed dynamics z'=F(z) has already
been justified, then y'=Df(z)F(z) on differentiable charts is a derived law.
An autonomous reduced law y'=g(y) additionally requires Df(z)F(z) to agree for
all states with the same y. Defining f alone proves neither statement. Maxima,
clamps, phase branches and discrete events require their own nonsmooth rules.

Conversely, using an observable in F(z)=H(z,f(z)) is not intrinsically invalid:
substitution gives a state-dependent law. The dependence H must be derived
from the declared structural dynamics or identified as an additional
constitutive/control assumption. A diagnostic label neither forbids causal
use nor justifies it. This is the distinction relevant to generative TNFR
research; calling any computed quantity "emergent" does not close the dynamics.

### Sense Index: definition, information loss and consumers

The [shared Si owner](../src/tnfr/metrics/sense_index.py) computes

`Si_i=clip(alpha*|nu_i|/nu_max + beta*(1-disp_i) + gamma*(1-|p_i|/p_max))`,

with clamped normalized components, wrapped phase dispersion, configured
normalized weights and graph-wide absolute maxima. A zero maximum uses
denominator one; both current Si backends refresh maxima from the effective
stored aliases before normalization. Thus Si is a
defined structural read-out. Its formula discards pressure sign, and common
positive capacity rescaling preserves the normalized capacity term. It is
not a complete dynamical coordinate or a proved stability certificate.
Writing default coefficients in terms of pi supplies reproducible values
and a normalization identity, not a unique derivation of their causal role.

Three finite controls make the distinction explicit. They use exact dyadic
values and a chosen threshold to expose an existing policy's dependency;
they are not proposed research parameters or new physical laws.

- On a mutual-neighbor pair with capacity (1/2,1), phase consensus and stored
  pressure zero, weights (1,0,0) give Si=(1/2,1), while (0,1,0) give Si=(1,1).
  At si_hi=3/4, identical nodal fields can therefore produce different
  capacity-gate decisions when only diagnostic configuration changes.
  If the counter condition is met, the existing mu=1/2 averaging proposal
  changes the left capacity to 3/4; this is a conditional proposal, not an
  executed experiment or a new coefficient choice for the primary study.
- Add a disconnected isolated node. Raising its capacity from 1/2 to 2,
  with fresh maxima and weights (1,0,0), changes the unchanged pair's Si
  from (1/2,1) to (1/4,1/2). The right gate now fails at that same threshold.
  This is a graph-global control dependency, not transport through an edge.
  A global law could be postulated explicitly; local emergence has not
  thereby been derived. These stored-pressure examples test the consumer
  contract, rather than asserting fresh full-channel equilibrium.
- With pure EPI pressure on the same unit pair, unit capacities and equal
  phases, x=(0,1) and x=(1,0) have opposite pressures and opposite nodal
  rates. Their nodewise Si values agree for any common diagnostic weights.
  Thus the full Si vector alone cannot determine the direction of EPI change.

Computing Si writes only its diagnostic/cache state, not the nodal triad or
support. Its consumers are separate: the
[capacity gate](../src/tnfr/dynamics/adaptation.py),
[glyph candidate selectors](../src/tnfr/dynamics/selectors.py), and the enabled
functional-link branch of [Coupling](../src/tnfr/operators/_coupling_stage_kernel.py).
That branch includes Si similarity in a configured link compatibility score;
U3 and final admission remain separate. No Si-dependent phase evolution is
inferred from the mere existence of Si sensitivities.

`step(..., use_Si=False)` skips refresh, not the consumers of stored Si.
It is therefore not an implementation of telemetry-independent dynamics.
The present scope clarification changes no runtime behavior. For the primary
research line, retain Si as a read-out; do not use its weights, thresholds or
counter policy to fill an unexplained causal step. Its existing controllers
remain available as explicitly conditional engineering models.

The built-in selectors validate the stored decision inputs and use one shared
snapshot/kernel across execution backends. Each new batch renews its pressure
and acceleration normalizers and score weights; engine-owned snapshots are
released even after failure. This prevents past maxima from silently changing
a later decision, but does not recompute the underlying pressure or Si. Missing
Si retains the explicit baseline 0.5; missing pressure/acceleration retains zero.
Finite Si is clipped to [0,1]. These defaults, thresholds, hysteresis, repetition
and lag rules remain controller choices, not deductions from the nodal equation.
The [decision controls](../tests/test_selector_decision_admission.py) test this
boundary and scalar/worker agreement without treating a candidate as a live
operator-admission certificate.

The corresponding reusable constraint is the
[source-tangency identity](nodal/FORCED_SOURCE_AND_CLOCK.md#22-source-tangency-without-a-telemetry-controller):
it derives what phase/capacity evolution must satisfy to preserve zero
pressure, without selecting that evolution through Si. The finite witnesses
are reproducible with
`artifacts/research/validate_source_tangency_2026_09_18.py`; their successful
checks do not prove a general autonomous Si reduction or an emergent mechanism.

The separate [capacity/phase result](nodal/FORCED_SOURCE_AND_CLOCK.md#23-capacity-exposure-does-not-determine-a-phase-clock)
applies this same distinction to the word frequency: accumulated capacity
follows from the nodal equation, while identifying it with circular phase
requires an additional relation. The configured oscillator and phase-relaxation
models are not uniquely selected by that equation.

### Uniform capacity and default selector reachability

The configured diagnostic and selector have a concrete interaction on the
retained prism. More generally, consider a nonempty graph with valid finite
readings, equal positive capacities, fresh Si normalization and the default
nonnegative normalized `SI_WEIGHTS`. Every node has `nu_norm=1`, so

\[
\mathrm{Si}_i=\operatorname{clip}_{[0,1]}
 \{\alpha+\beta(1-\mathrm{disp}_i)+\gamma(1-|p_i|_{\mathrm{norm}})\}
 \ \ge\alpha.
\]

The configured primary weight represents `pi/(pi+1)>3/4`, while the default
selector's upper cut is `si_hi=1/2`. Hence every node's **default base choice
is IL**, independently of its pressure sign/magnitude and phase alignment.
The represented normalized coefficients also satisfy this strict separation;
the claim is not based on rounding them to displayed decimal values.
Both current Si backends use the actual positive maximum, including positive
subnormal capacities; their denominator-one fallback concerns only an exactly
zero maximum. This theorem excludes that all-zero inactive class.

The complete decision path must still be distinguished from its base choice.
The [native selector](../src/tnfr/dynamics/selectors.py) can force AL or EN
through its configured lag counters. The shared
[grammar fallback](../src/tnfr/operators/grammar_dynamics.py) tries
`IL,THOL,EN,SHA,RA,NAV,AL`, never ZHIR. Thus this ordinary default path cannot
introduce ZHIR while the stated diagnostic condition holds. When IL itself
is admitted and neither lag forces a replacement, the applied glyph is IL.
The parametric selector, custom callables, stale or skipped Si refresh,
heterogeneous capacities and changed diagnostic weights/cuts have different
scope. No conclusion about all future steps follows if those conditions change.

A valid signed growth secant and U4b context can therefore make Mutation
**eligible without making it selected**. A manually requested Mutation or a
custom selector returning it would test a different occurrence rule. The
configured Si formula does not prove that the physical nodal structure forbids
a phase change. Conversely, altering Si weights or disabling its refresh to
obtain the desired event would not derive a missing TNFR mechanism.
This is a control-policy reachability result, not a newly prescribed policy
or a defect warranting an arbitrary threshold change.

Controls: [diagnostic lower bound and selection](../tests/physics/test_uniform_capacity_selector_scope.py).
The single finite causal check and source-work interpretation belong to the
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).

## 8. Grammar derivation: premises, language and trajectories

The grammatical audit distinguishes four claims: an identity following from
the nodal equation, a conditional theorem for a specified realization, an
operator/invariant contract, and a selected engine policy. A contract can be
mandatory without being a theorem of the scalar equation. A classification
predicate that returns membership in a declared set is executable consistency,
not an independent mathematical derivation of that membership.

| Rule | Structural content that can be established | Additional premise or policy | What acceptance does not establish |
|------|-------------------------------------------|------------------------------|------------------------------------|
| U1a | The rate is zero if capacity or pressure is zero; EPI zero is not singular. | A new word uses a generator unless existing form is declared. | Origin of basal capacity, unique birth mechanism or spontaneous selection. |
| U1b | Suppressing capacity suppresses the rate when pressure is bounded. | The supported endpoint set is SHA/NAV/REMESH/OZ. | A stationary endpoint, accumulated-change convergence or equivalence with the older PDF closure set. |
| U2 | Accumulated change obeys the flow/jump balance below. | Named coverage, causal unit debt and capacity two. | A bound on physical pressure, an energy gain or stability under arbitrary repetition. |
| U3 | Circular separation and the pairwise phasor cross term are computable. | A nonnegative cross-term domain motivates the default pi/2 gate; the coupling contract requires it. | All-support compatibility, phase-source rigidity or forward invariance. |
| U4 | On smooth segments, acceleration is the derivative of capacity times pressure plus capacity times pressure derivative. | Trigger/handler roles, prior IL and three-position context. | A mathematical bifurcation, threshold crossing, unique next operator or a physical waiting time. |
| U5 | A declared differentiable hierarchy obeys the chain rule; exact reduction requires closure. | Depth-local stabilization and a separately specified coherence target. | Autonomous macro dynamics, identity preservation or a universal coherence inequality. |
| U6 | Potential is Bp; changes contain pressure and distance-kernel terms. | Mean absolute reference drift below pi/2. | Pointwise bounds, intermediate-time confinement, future safety or metric-scale invariance. |

The [unified specification](UNIFIED_GRAMMAR_RULES.md) owns the supported rules.
This section owns their logical scope. The operator-role registry and
[operator effects](../src/tnfr/operators/operator_contracts.py) must remain
separate: sharing a primary channel does not determine a grammatical role.
For example, IL and OZ both act on pressure but have different roles.
THOL nesting and network-scale REMESH also show why target scale is not the
number of state entries an operator may modify.

**Necessary and sufficient for what?** A grammar predicate can be necessary
for admission to a declared engine API. It need not be mathematically necessary
for every solution of the nodal equation, and it is not thereby sufficient for
stable evolution. Pure diffusion without named stabilizers is the existing
counterexample to the first promotion. Gain, timing, source and history
obligations prevent the second. A grammar-valid word specifies possibilities;
it does not select targets, invocation times or one autonomous successor.

## 9. U2 and U4 in the actual flow/jump model

For locally finite event times and piecewise absolutely continuous EPI, using
right-continuous (post-event) endpoint values, the engine's hybrid
interpretation requires both terms:

    x(t)-x(t0) = integral D_nu(s) p(s) ds
                + sum_{events in (t0,t]} [x(event+)-x(event-)].

This assumes a common coordinate support; births, removals and remeshing need
explicit coordinate/history maps. An EPI-writing zero-duration operator is a
jump, not an ordinary finite pressure integrated over zero time. Between jumps
the nodal equation holds; the shared event executor records the separate maps.
An infinite-horizon sufficient condition is absolute integrability of the
flow rate plus absolute summability of jump magnitudes, with events locally
finite. U2 role counts do not prove either bound. The earlier flow-only
integral statements apply to continuous segments, not silently to all jumps.

A useful reusable restricted energy balance comes from fixed symmetric
nonnegative conductance W with positive degrees d and positive held capacity.
Let B=D-W, H=diag(d_i/nu_i), z=x-1*(1^T Hx)/(1^T H1), and let the declared
pressure be p=-e L_rw x+F with constant e>0. Then

    V = (1/2) z^T H z,
    V' = -e z^T B z + z^T D F.

Thus diffusion dissipates disagreement while the actual canonical source can
add or remove it. At a same-support, same-metric jump j=x_plus-x_minus,
with P the H-centering projection,

    V_plus-V_minus = z_minus^T H Pj + (1/2)(Pj)^T H(Pj).

These identities make a grammar-to-stability proof obligation concrete:
bound the source work and each actual jump, then combine them over the declared
times and history. Changing metric/support and delayed REMESH add their own
terms. Reuse [regional balances](FORCED_SUPPORT_BALANCE.md),
[represented event schedules](STRUCTURAL_OPERATORS.md), and the
[delayed-history results](REMESH_INFINITY_DERIVATION.md), rather than fitting
an energy meaning to the unit debt counter. Existing certificates remain
restricted to their declared maps, norms and realization defects.

For U4, on a smooth segment,

    x'' = D_nu_dot p + D_nu p_dot.

The EPI equation does not by itself specify nu_dot or p_dot, which can depend
on phase, support and other state. A large acceleration or a configured
threshold crossing is an event diagnostic; a bifurcation theorem requires
a declared family of dynamics and a change in its equilibria, stability or
other invariant structure. Event secants, continuous derivatives and
operator-index recency are different quantities. Existing Mutation evidence
and event clocks preserve this separation. No new threshold is needed here.

## 10. U3: exact geometric content and a strict-gate counterexample

For two nonnegative phasor amplitudes a,b,

    |a exp(i theta_i)+b exp(i theta_j)|^2
      = a^2+b^2+2ab cos(theta_i-theta_j).

Demanding a nonnegative pairwise cross term yields circular separation at most
pi/2 when both amplitudes are positive. This gives a geometric interpretation
of the selected gate. It does not derive that design requirement or the full
coupling map from x'=D_nu p. The actual UM/RA gates apply to their participating
relations. Existing support and the full canonical phase-pressure neighborhood
are not globally filtered through U3. Zero-weight support can still participate
in phase pressure. Tightening an operator's gate does not change that owner.

On fixed undirected support with no isolated nodes, strict gaps below pi/2
on every edge imply
Re(exp(-i theta_i) S_i)>0, where S_i=sum_neighbors exp(i theta_j).
Hence all S_i are nonzero and the displacement from each center is inside
its regular open half-pi chart. But the source derivative

    g_i = wrap(Arg(S_i)-theta_i)/pi,
    Dg = (R-I)/pi,
    R_ij = 1[j in N(i)] Re(exp(i theta_j)/S_i)

depends on neighbor-to-resultant angles, not only neighbor-to-center angles.
The previously proved [nonnegative irreducible criterion](nodal/FORCED_SOURCE_AND_CLOCK.md#24-rigidity-and-flexibility-of-a-held-phase-source)
is an additional hypothesis not implied by strict U3.

**Exact six-node witness.** Use a unit double-star: centers u,v joined by an
edge; leaves a,b attached to u; leaves c,d attached to v. Set

    (theta_u,theta_v,theta_a,theta_b,theta_c,theta_d)
      = (0, pi/3, -pi/3, -pi/3, 2pi/3, 2pi/3).

Every edge has separation pi/3, strictly within U3. The resultants are
S_u=3/2-i*sqrt(3)/2, S_v=i*sqrt(3), and each leaf sees its center's phasor.
Their squared magnitudes are (3,3,1,1,1,1), and

    g=(-1/6,1/6,1/3,1/3,-1/3,-1/3).

The two center-to-center entries of R vanish exactly. Each center assigns
1/2 to each of its leaves; each leaf assigns one to its center. Thus R is
nonnegative but reducible into two three-node classes despite the connected
support. The exact shared observer gives rank(R-I)=4 and tangent dimension
two. Independent infinitesimal rotation of either star preserves g to first
order. This refutes the pending implication that strict U3 and connected
support force rank n-1 for nonzero source.

**The apparent freedom is obstructed at second order.** Locally fixing leaf
sources forces theta_a=theta_b=theta_u-pi/3 and
theta_c=theta_d=theta_v+pi/3. Write delta=theta_v-theta_u. Fixing g_u=-1/6
then requires

    Im[exp(i*pi/6)*(exp(i*delta)+2exp(-i*pi/3))]
      = sin(delta+pi/6)-1 = 0.

Near delta=pi/3 this forces delta=pi/3 exactly; the real part remains positive.
The local finite fixed-source set therefore has only common rotation. Along
the extra tangent which rotates v,c,d by eta,

    g_u''(0)=-1/(pi*sqrt(3)),   g_v''(0)=1/(pi*sqrt(3)).

Leaf source derivatives stay zero. A rank-deficient linearization is therefore
not a finite relative-motion mechanism, even strictly inside the gate.
The [exact regression fixture](../tests/physics/test_grammar_phase_geometry.py)
uses rational cosine Gram data and the existing derivative/rank owners; it
does not simulate a trajectory or assert runtime operator admission.

Finally, an instantaneous phase gate is not a forward-invariance theorem.
For a declared differentiable relative phase delta(t), the upper active
boundary requires delta_dot<=0 and the lower boundary delta_dot>=0.
An invariant-set proof additionally needs a well-posed phase law and the
appropriate boundary conditions throughout that set, plus checks on jumps.
The EPI equation alone supplies no such phase law. A merged UM stage may
verify its used relations without certifying all later flows.

## 11. U5: exact scale dynamics and coherence ordering are separate

For a differentiable representation y=f(x), y'=Df(x)x' is an identity along
the declared micro trajectory. An autonomous macro law additionally requires
this value to be identical on every fiber f(x)=y. For a fixed affine EPI
system x'=Tx+b and linear read-out y=Qx, the exact condition is

    QT=A_bar Q,

equivalently invariance of ker(Q) under T; Qb supplies the macro source.
Failure produces hidden-state dependence. Reuse the repository's quotient,
observability and memory derivations; adding a scale stabilizer does not
establish this algebraic condition.

Even exact closure does not order nonlinear coherence. On unit K4 with unit
capacity and pure EPI pressure, partition the vertices into {0,1} and {2,3}.
Take x=(0,-3/4,3/8,3/8)+c*1; then

    p=x'=(0,1,-1/2,-1/2).

This equitable partition closes exactly. Block means y=(-3/8,3/8)+c*1 obey
a two-node diffusion with effective capacity 2/3 and pressure
p_bar=(3/4,-3/4). In the first block, y'=1/2. Using the same canonical
C=1/(1+|p|+|x'|) at both scales gives

    C_parent=4/9 < (C_child_0+C_child_1)/2 = 2/3.

This is a failure of the proposed average-coherence inequality with
alpha=1/2 despite exact dynamical closure. The effective capacity is fixed
by the two external neighbors among three, not fitted to the inequality.
An arbitrary uniform offset can make every EPI positive without changing it.
More generally, at zero pressure/rate, m equilibrium children have C=1 and
their sum is m, so an unnormalized universal target needs alpha<=1/m even
in that simple case. The supplied
[U5 assessment](../src/tnfr/physics/multiscale_coherence.py) correctly requires
an explicit hierarchy and coefficient. No new physical coefficient is derived.

## 12. Audit integration and remaining obligations

The [grammar owner](UNIFIED_GRAMMAR_RULES.md#8-composition-and-implementation)
specifies shared word admission, profiles, live rejection and observation.
Signed nonzero form, causal debt, prior Coherence, actual recursion metadata
and live phase checks are separate inputs. A diagnostic waiver is not a normal
word pass; missing phase/reference observations are unavailable rather than
evidence of safety. Tests establish those implemented contracts, not their
physical necessity.

There is no complete derivation of the thirteen maps, their invocation law,
role catalog or numerical policies from the scalar nodal identity alone.
Nor is there a proof that six grammatical rules are minimal or sufficient for
all admitted TNFR dynamics. These limitations do not justify discarding an
existing contract without a replacement argument.

The strict-U3 witness in §10 disproves the proposed rank implication while its
finite fixed-source set is locally rigid. The U5 witness in §11 separates exact
macro closure from coherence ordering. Both remain reusable counterexamples;
neither selects a phase/capacity/support law or establishes NFR maintenance.
The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone assigns current work.

## 13. Structural grammar refactor and the full nodal system

Physical conclusions require explicit structural and dynamical premises.
Renaming a calibrated constant, differentiating an underdetermined identity
or removing a contract cannot supply the missing law. The
[grammar basis registry](../src/tnfr/operators/grammar_canon.py) records which
parts are identities, conditional theorems, contracts or policies; these are
provenance declarations, not proof-carrying permission to execute.

The [word and execution interface](UNIFIED_GRAMMAR_RULES.md#8-composition-and-implementation)
owns profiles, rejection modes and read-only observations. Its finite typed
execution evidence can bound the represented EPI composition in a common
metric. Gain at most one proves nonincrease for those maps; strict inequality
proves contraction. Gain above one does not prove that the actual observed
trajectory expanded. Missing evidence remains unavailable.

Such a result does not authenticate another graph, supply a next operator,
or extend to a global binary64 runtime, future schedule, full-channel or tetrad
theorem. The following equations identify the additional closure obligations.

### Differentiation exposes missing laws; it does not invent them

Let z=(x,nu,theta,W,h) denote a declared complete state, including relevant
support/weights and history h. On a regular continuous chart with pressure
p=P(z), the EPI component is

    x'=D_nu P(z),
    x''=D_nu_dot p + D_nu p'.

The derivative p' includes the derivatives of every channel on which P
depends. For the fixed-support, fixed-coefficient realization
p=-e L_W x+w_phi g(theta)-v L_U nu with no changing topology source,

    p'=-e L_W D_nu p + (w_phi/pi)(R-I)theta' - v L_U nu'.

If conductance changes smoothly, add -e L_W' x; if any other source or
coefficient changes, its derivative must also be retained. Actual edge
birth/deletion and operator writes are hybrid events, not silently smooth
terms. These identities reuse sections 9-10 and the
[existing source-tangency derivation](nodal/FORCED_SOURCE_AND_CLOCK.md#22-source-tangency-without-a-telemetry-controller).
They constrain proposed laws for nu, theta and support; they do not select
those laws. The [same-initial-state completions](nodal/FORCED_SOURCE_AND_CLOCK.md#23-capacity-exposure-does-not-determine-a-phase-clock)
already prove nonuniqueness of the EPI future without an additional phase
relation. Differentiating the original identity adds no independent relation
capable of removing that ambiguity.

For fixed symmetric conductance, positive capacity and the pure EPI channel,
the closed diffusion and its Dirichlet dissipation are already derived. They
are a reusable positive example of structural dynamics without operator-count
thresholds. Extending that result requires accounting for source work,
changing geometry, auxiliary channels and jumps, not assuming the extension.
The auxiliary symplectic model and its conserved quantities have their own
premises; the [variational bridge](TNFR_VARIATIONAL_PRINCIPLE.md) cannot be
promoted into a derivation of the complete nodal dynamics without proving
the correspondence of their states and trajectories.

### The tetrad participates in the same closure problem

All four canonical fields remain required read-outs, computed by the existing
[field owner](../src/tnfr/physics/fields.py):

| Field | Joint structural dependency | Obligation for a dynamical conclusion |
|-------|-----------------------------|---------------------------------------|
| Phi_s | Pressure and the distance kernel B of the current geometry. | Phi_s'=B p'+B' p on smooth aligned charts; event and reference changes separately. |
| Phase gradient magnitude | Circular phase differences on the declared neighborhood. | Actual phase/support law; wrapping and absolute-value boundaries cannot be treated as globally smooth. |
| K_phi | Circular curvature on that same support. | A regular-chart derivative or an explicit branch/event treatment; no global substitution by the EPI Laplacian. |
| xi_C | The declared correlation field, graph distances, fit and fallback. | State/scale and estimator regime must be specified; the spectral fallback is not an identity for all correlation fits. |

For a differentiable tetrad observation T(z), T'=DT(z)F(z) follows only after
a full law z'=F(z) is specified. An autonomous tetrad law additionally needs
that derivative to agree on every state sharing T(z). The existing capacity
rescaling counterexample in section 5 excludes an unconditional claim that
the tetrad determines the evolution. Singular charts and estimator switches
need separate treatment. Geometry may legitimately enter a derived law, but
its diagnostic definition alone is not that derivation.

A candidate autonomous NFR mechanism must therefore address the triad,
pressure, support, memory and tetrad together. An EPI energy bound, an
instantaneous U3 gate or a U6 threshold alone cannot stand in for this joint
claim. Reuse the existing exact quotient/memory tests to ask whether a proposed
macro observation closes; use the tetrad to report its geometric and
correlation consequences without imposing a diagnostic target as a new force.
These are constitutive admission requirements. The
[single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) owns their
current priority; this scope note does not open another trajectory campaign
or promote a configured selector to emergent physics.

## 14. Constitutive closure audit from the nodal law

The constitutive question starts with structural form and `x'=diag(nu)*p`, not with
the claim that each implemented formula is a necessary TNFR law. The
[foundation audit](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation)
owns EPI types, coordinate changes, units and the original source's conflicting
definitions. Source fidelity preserves the underlying hypothesis; it does not
require keeping an inconsistent dimensional equation or a lossy chart claim.

### Existing laws and independently supplied premises

| Component | Implemented or derived content | Missing implication |
|-----------|--------------------------------|---------------------|
| EPI flow | Shared integrator implements capacity times declared pressure; isolated EPI diffusion has its exact Dirichlet balance. | The nodal identity alone does not choose the state space or pressure functional. |
| Detached smooth-sine research | [Reciprocal form/phase exchange](nodal/RESONANCE_FOUNDATIONS.md#reciprocal-exchange) derives joint balances and response under a complete, globally defined comparison law. | The sine kernel, cosine storage, exchange/loss scales, held capacities/support, clock, preparation and events remain premises. The [storage counterfamily](nodal/SINE_CONSTITUTIVE_INFORMATION.md#phase-storage-selection-boundary) retains balances and organized equilibria without uniquely selecting that law; it is separate from native argument-pressure execution. |
| Phase relaxation | [coordination.py](../src/tnfr/dynamics/coordination.py) blends global/local circular means per invocation with configured, possibly adaptive gains. | Invocation time, gains and their adjustment policy are not derived by the EPI identity. |
| Oscillator proposal | [phase_evolution.py](../src/tnfr/dynamics/phase_evolution.py) adds `dt*nu` and a configured sine coupling over admitted neighbors. | Capacity as angular speed and the coupling coefficient are additional constitutive premises. This is distinct from mean relaxation. |
| Capacity adaptation | [adaptation.py](../src/tnfr/dynamics/adaptation.py) applies clipped neighbor averaging after stored Si/pressure gates and a consecutive-call count. | Averaging gain, thresholds, waiting count and use of that diagnostic as a controller are supplied policies; there is no independently derived capacity rate. |
| Optional extended system | [canonical.py](../src/tnfr/dynamics/canonical.py) supplies explicit nonlinear phase/pressure responses; the [integrator](../src/tnfr/dynamics/integrators.py) retains the live additive Gamma source in the EPI row. | Its coefficients, source and flux correspondence need justification; it provides no capacity law and does not follow from the EPI product. |
| Edge creation | [_coupling_stage_kernel.py](../src/tnfr/operators/_coupling_stage_kernel.py) combines candidate sampling, phase checks and configured link scores. | U3 constrains which proposed links may be accepted; it does not select their occurrence, weight or persistence. |
| Remeshed topology | [remesh.py](../src/tnfr/operators/remesh.py) provides declared MST/neighbor/random rewiring policies. | EPI-distance construction and seeded reproducibility are not a derived support evolution law. |
| Variational completion | [Variational section 13](TNFR_VARIATIONAL_PRINCIPLE.md#13-forced-potential-family-and-reciprocal-closure-constraints) derives the joint potential family compatible with a specified EPI mobility. | Its free non-EPI function, auxiliary mobilities and invocation law remain undetermined. |

The ordinary [runtime](../src/tnfr/dynamics/runtime.py) integrates EPI before
phase coordination and capacity adaptation. Disabling glyph selection or
fresh Si computation does not by itself disable those later steps. Their
retained gain/history state belongs in any declared complete model.

The optional extended pressure response is exactly `-(1+0.135)*div(J_p)`;
it has no restoring dependence on p. Constant nonzero supplied divergence
therefore gives a constant pressure slope, not automatic relaxation. With
capacity one, p=1/2 and zero flux, the configured phase rate is 0.575, not
zero: zero flux removes transport terms, not all auxiliary dynamics. These
documentation errors are corrected without changing the arithmetic.

### Optional pressure composition and source consistency

Composing the actual shared readers sharpens the preceding scalar-input audit.
On fixed unique-neighbor support let `L_U=I-N^-1 U`, with zero rows at
isolates. The pressure-contrast reader gives `J_p=-L_U p`; the integrator's
scalar "divergence" gives `diag(sqrt(k))*L_U J_p`, where `k` counts unique
neighbors. The optional pressure response therefore implements

\[
\dot p=(1+0.135)\operatorname{diag}(\sqrt{k})L_U^2p.
\]

This is not an incidence divergence of oriented edge fluxes. On a regular
undirected graph the nonconstant eigenmodes have positive squared-Laplacian
rates, rather than diffusion decay. The formula is a continuous vector-field
interpretation of the supplied update; finite Euler steps, rounding and
clipping retain separate effects.

The retained unit prism provides an in-band, zero-phase-gap control.
Set `x=1/2*1-P/4`, `P=(1,-1,0)^2`, unit capacity, and pure-EPI pressure.
The shared canonical pressure is `p=P/4` and `L_U P=P`. Differentiating that
same constitutive law using the shared joint-response owner requires
`p_dot=-L_U*x_dot=-p`. The optional block instead gives
`p_dot=(1+0.135)*sqrt(3)*p`, with the opposite sign. Both expressions refer
to the same prepared state; no trajectory, fitted parameter or numerical
step size is needed to detect the incompatibility.

The optional path remains an explicitly configured independent-pressure
model, disabled by default in `update_epi_via_nodal_equation`. Ordinary
runtime dispatches its resolved integrator directly and then coordinates
phase and adapts capacity; this wrapper's flag alone does not replace that
path. Reversing one sign would not derive agreement
with the full multichannel chain rule. Its arithmetic is retained while
unwarranted conservation/invariant claims are removed. Three unused private
synthetic flux helpers are removed; execution already used the shared real
field readers and divergence owner. The [fresh-pressure phase comparison](TNFR_VARIATIONAL_PRINCIPLE.md#1314-existing-optional-feedback-pressure-consistency-before-recurrence)
has a separate exact dissipation identity on a regular prism chart and is
not silently substituted for the optional model. Controls:
[extended-pressure scope](../tests/physics/test_extended_pressure_feedback_scope.py).

### Capacity remains undetermined even under dissipation

On unit P2 with consensus phase and homogeneous positive capacity a(t), both
the capacity-gradient and topology sources vanish. With EPI-channel coefficient
e>0, x(0)=c*1+d*(1,-1), c>d>0, compare two mathematical completions:

```text
a_A(t)=1,     x_A(t)=c*1+d*exp(-2*e*t)*(1,-1),
a_B(t)=1+t,   x_B(t)=c*1+d*exp(-2*e*(t+t^2/2))*(1,-1).
```

Time t>=0 is expressed in one fixed normalized structural unit for this witness.
Both solve the same nodal equation and the same canonical pressure realization,
share the complete initial triad/support/history, initial tetrad and EPI rate,
keep positive EPI and capacity, and strictly dissipate Dirichlet energy. Yet
their initial accelerations differ by `-2*e*d*(1,-1)`. Thus adding positivity,
phase compatibility and dissipation does not determine capacity evolution.
These are logical countermodels, not proposed TNFR mechanisms or two executions
of a single configured deterministic runtime. Their exact controls reuse the
actual pressure and field owners in
[test_constitutive_capacity_scope.py](../tests/physics/test_constitutive_capacity_scope.py).

### Moving conductance: an exact balance and an unobservable scale

On fixed active symmetric support, write `B=D-W`, `g=-D^-1 Bx` and `r=nu*p`.
For supplied differentiable positive conductances and positive row strengths,

```text
g_i' = mean_W(r_j-r_i)
       + [sum_j W_ij'*(x_j-x_i)-d_i'*g_i]/d_i,
E_D' = (Bx)^T r + (1/4) sum_ij W_ij'*(x_i-x_j)^2.
```

The first derivative separates form evolution from changing geometry; the
energy balance separates nodal work from the change of its conductance
weights. There is no explicit capacity derivative in this energy because
E_D depends on x and W. The shared
[support derivative observer](../src/tnfr/physics/support_transport.py)
now evaluates both identities exactly on declared rational/represented input,
including loops and isolates. It reuses the snapshot, Laplacian and energy
owners, validates symmetric edge rates and does not select W', refresh
pressure or execute a graph change. Active-set changes need the reset owner.

For any positive c(t), `W(t)=c(t)*W_0` gives the identical normalized walk
`D(t)^-1 W(t)=D_0^-1 W_0`. At fixed unweighted support all other canonical
gradient channels are likewise unchanged by this weight scaling. Functions
with c(0)=1 but different c'(0) give distinct conductance histories with the
same initial state and the same instantaneous EPI generator. Nevertheless
E_D scales by c. In the derivative balance the geometry term in g' vanishes
while conductance work equals `(c'/c)*E_D`. This is nonidentifiability of
the weight law from EPI transport, not evidence that weights physically change.

The tetrad must retain its metric conventions. With explicit fixed edge
length, structural potential is unchanged at fixed pressure; with weight as
the compatibility length, distances scale by c and Phi_s by c^-2. Phase
gradient and curvature use their declared support/phase. The coherence-length
fit has estimator-specific distance conventions; a small-graph spectral
fallback must not certify every fit or backend. Portable controls in
[test_constitutive_support_scope.py](../tests/physics/test_constitutive_support_scope.py)
state which estimator they use. No complete tetrad-autonomy theorem follows.

### What is now derived and what must come next

This subsection records mathematical dependencies, not a chronological task
queue. The execution plan determines the current next step.

The variational calculation supplies useful positive constraints: the EPI
law fixes the x-dependent part of a compatible potential, requiring reciprocal
couplings when auxiliary gradient-flow premises are chosen. It also gives a
source-compatibility criterion for boundedness below. The strict-U3 star
counterexample rules out only a constant positive diagonal gradient metric
for the actual nonlinear phase source near consensus; it leaves other metrics
and non-gradient dynamics open. These are consequences to reuse, not reasons
to impose a preferred energy minimizer.

Before proposing missing laws, define the intended EPI state/equivalence,
directed pressure action and clock independently of a retrospective fit.
Test a proposed representation with the existing
[closure and minimal-realization owners](../src/tnfr/physics/epi_memory.py),
[observability](../src/tnfr/physics/observability.py) and
[operator quotient](../src/tnfr/physics/operator_quotient.py).
The continuous fibre condition and event intertwining are in the foundation
note. A scalar observable may close without reconstructing the full state;
a rich storage object may still lack any justified evolution. Retain all
four tetrad channels as consequences to account for, and introduce neither
a physical clock nor a selection functional solely to obtain stability.

<a id="justified-law-admission"></a>
### When an additional law is justified

A predictive model needs additional laws or declared input histories whenever
a consumed channel has no specified evolution or justified elimination.
Holding capacity, support or another channel fixed is already an explicit
completion; every such channel need not immediately acquire autonomous
evolution. Missing closure does not by itself invalidate the nodal row or
establish a need for a more general equation. Mathematical closure,
well-posedness and physical justification are separate obligations. A proposed
law may be an independently stated constitutive hypothesis, a consequence of
stronger premises, or an inherited reduction with retained memory and error.
Its status must survive integration into the engine.

Evaluate each proposed law or hypothesis against its existing owner: state
its domain and units, classify its logical status, expose its independent
premises, and derive the consequence that needs it. Test an alternative or
counterexample before claiming necessity or uniqueness. Record whether the
claim is derived under those premises, remains a supplied hypothesis, is
underdetermined, or is rejected in the declared domain. A revised foundational
premise needs the same scrutiny and a dependency review of earlier results;
a failure in one completion does not invalidate every model of the nodal row.

NFR and operator properties are part of this admission, with their logical
status retained. Use the [NFR property requirements](EMERGENT_ONTOLOGY.md#12-can-nfr-properties-justify-a-joint-law)
and [participation in form](EMERGENT_ONTOLOGY.md#nfr-participation-in-form)
to specify identity, form/capacity domains, retained internal and boundary
state, circular phase and any claimed nested description. A law must respect
the admitted domains and symmetries; a reduction must preserve the information
consumed by its claimed response or account for its loss. Name which
requirements are definitions, independent constitutive premises or consequences
of the chosen model.

For claimed operator actions, select the actual execution path and apply its
[event-map and composition contracts](STRUCTURAL_OPERATORS.md#22-operator-as-transformation),
including live admission, target/history dependence, primary and secondary
state changes, ordering and event costs. A collective description also needs
the [operator descent criterion](STRUCTURAL_OPERATORS.md#collective-operator-descent).
The [grammar scope](UNIFIED_GRAMMAR_RULES.md#purpose-and-authority) remains
distinct from continuous-law selection and autonomous event occurrence.
A detached continuous comparison has its own laws and declared interventions;
matching a named operator's apparent effect does not establish that operator's
contract. The canonical catalog does not by itself prove generative completeness
or uniquely determine the continuous pressure/phase law.

The existing [parameter ledger](NODAL_PARAMETER_FOUNDATIONS.md#2-parameter-and-dependency-ledger)
identifies the missing premises. The
[phase/storage alternatives](nodal/SINE_CONSTITUTIVE_INFORMATION.md#phase-storage-selection-boundary)
and [replication counterfamily](nodal/SINE_PAIR_MOBILITY.md#sine-replica-constitutive-nonselection)
show why balance, symmetry, organized equilibria or a shared local response
cannot alone select the complete nonlinear law. An added information or
selection principle needs independent justification; reproducing a desired
pattern does not provide it. Distinguish genuinely different completions from
coordinate or clock changes that transform every row of the same model.

The [matrix-response comparison](nodal/ORGANIZED_EFFECTIVE_LAW.md)
makes that distinction explicit for a changed collective form row: it needs
its own storage and input action, whereas a regular charge-like chart
transforms both rows of that same law. Shared equilibria do not transfer the
old storage bound, fine-source acquisition or finite observation certificate.
Neither a static matrix nor closed collective equations supply an eliminated
hidden-state history or a collective event's occurrence law.

The [organized-target comparison](nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-constitutive-freedom)
retains positive loss, target storage, the complete tangent and the quadratic
field, yet reverses the nominal mixed interaction's leading nonlinear sign.
Its additional phase moments `Z9,Z18` make the remaining information premise
explicit. Matching these local object properties therefore does not replace
constitutive admission. Its separate
[same-source acquisition proof](nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-same-source-acquisition)
retains both formed identities across the continuous constitutive family,
using each law's own force, storage and capture bounds. Formation from this
supplied source therefore does not select one member either. Beyond the nominal
local onset, the [finite carried-state comparison](nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-finite-carried-response)
retains opposite mixed responses under its own-law recovery, contact-work and
observation-error premises. Physical sensor calibration and law selection remain
open.

Use the derived organizations as test cases for those premises. Specify which
generative claim requires more than the present preparation, identity and
interaction results, then revisit the premise that blocks it. For example,
[uniqueness at full equilibrium](FUNDAMENTAL_THEORY.md#environment-substrate-and-pressure)
does not supply spontaneous initial activity, and
[finite positive-loss dynamics](nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
restricts sustained nonstationary motion. Neither restriction excludes static
organized identities or every other admitted completion. Autonomous source
preparation, contact occurrence, scale selection and a physical observation
require their own arguments; they are not consequences of a successful
prepared response. The [bound-organization target](EMERGENT_ONTOLOGY.md#generative-bound-organization)
also needs a stated composite identity and binding criterion; a signed
storage excess alone supplies neither.

Integrate an admitted completion through its declared-law and shared execution
owners, with matched domains, sources, events, balances and observations.
Diagnostics and grammar do not silently provide its missing controller.
Preserve valid conditional objects and counterexamples while testing a
prospective discriminator, equivalence or obstruction between alternatives.
Reusing a proof method or numerical kernel does not transfer the old law's
derivative, regularity, source or trapping bounds. Match every consumed
hypothesis before inheriting a certificate; otherwise derive it for the new
complete law. A local response coefficient alone is not a finite-response
separation with an admitted error budget.
The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
selects the investigation; this admission criterion creates no parallel queue.

<a id="law-selection-and-boundary-information"></a>
### Law distinction, independent premises and retained boundary information

The [complete noisy-record separation](nodal/ORGANIZED_LAW_OBSERVATION.md#organized-law-record-separation-result)
establishes that two supplied laws are distinguishable under one declared
protocol despite independently unknown initialization. It neither chooses a
law from a measured response nor supplies a new constitutive axiom. The
following premise audit separates restrictions already justified by their
owners from restrictions that would need an additional argument.

Circular phase, odd reciprocal currents, the nodal row and an exact work/loss
balance still admit the [storage alternatives](nodal/SINE_CONSTITUTIVE_INFORMATION.md#phase-storage-selection-boundary).
Fixing cosine storage can conditionally select its matching current, but it
would assume the storage whose uniqueness is in question. Likewise,
[first-resultant sufficiency](nodal/SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency)
selects sine in its stated additive information class; universal
[balanced-neighborhood cancellation](nodal/SINE_CONSTITUTIVE_INFORMATION.md#balanced-neighborhood-information-admission)
restates that restriction rather than independently justifying it. Repeating
these conditional uniqueness proofs cannot remove their premises.

The [C9 composition classification](nodal/COMPOSITION_COMPATIBLE_CONSTITUTIVE_LAWS.md#composition-compatible-classification)
admits nonsine profiles with matched target storage, tangent and quadratic
field. Requiring an unchanged profile across the
[oriented interface interval](nodal/ORIENTED_INTERFACE_PROFILE_INHERITANCE.md)
is a stronger conditional selector. The
[NFR requirements](EMERGENT_ONTOLOGY.md#12-can-nfr-properties-justify-a-joint-law)
also permit retained boundary-dependent laws and derived memory; they do
not require that stronger profile premise. Acquisition from one supplied
source and local stability remain useful conditional results, while
[operator descent](STRUCTURAL_OPERATORS.md#collective-operator-descent)
has separate event, target, history, admission and occurrence obligations.
None of these reviewed results supplies an independently justified unique
law selector.

The [physical comparison atlas](PHYSICAL_REGIME_CORRESPONDENCES.md#collective-properties-and-fundamental-phenomena)
contains no completely admitted physical identification. Its retained
[electrical review](research/ORGANIZED_ELECTRICAL_FEASIBILITY.md#organized-electrical-physical-admission)
and [recovery-area review](research/ORGANIZED_RECOVERY_OBSERVATION.md#recovery-operational-admission)
expose preparation, observation and complete-law gaps. They concern different
laws and protocols from the new harmonic record theorem: their apparatus
bounds and uncertainty certificates do not transfer to it. What remains
common is the obligation to admit preparation, loading, clock and measurement
under the actual model. This is a review of the repository's admitted
bridges, not a claim that no physical realization can exist.

An information requirement can be studied without assuming that nature
minimizes it. For a declared additive current and marked boundary, distinguish
the data sufficient for an instantaneous response at every relative phase
from the state needed to evolve those data. A minimality theorem must name
the boundary class, regularity, retained degree, allowed decoding and phase
transformation. An explicit sufficient moment list alone does not prove a
lower bound on all descriptions in that class. Universal finite-boundary
claims also need not give the minimum on a fixed small-degree support or
an invariant prepared family. The
[regular boundary-information theorem](nodal/SINE_CONSTITUTIVE_INFORMATION.md#minimal-regular-boundary-information)
now supplies such a sharp bound for finite odd Fourier currents, including
nonlinear decoders. It measures the information required by a supplied law;
even the minimum two is shared by different single-harmonic laws with the
same consensus slope. Neither finite dimension nor a two-coordinate
preference by itself selects the fundamental sine harmonic.

This distinction connects law admission to the information a collective
interface consumes, while retaining two existing obstructions.
[Repeated composition](nodal/REPEATED_ORGANIZED_COMPOSITION.md)
can require the child-to-port association; aggregate marginal observations
do not authorize its removal. And
[phase-moment motion](nodal/SINE_CONSTITUTIVE_INFORMATION.md#phase-motion-information)
depends on phase/rate association, so an instantaneous response summary
need not be an autonomous dynamical state. Finite information, exact
aggregation or fewer channels must not be promoted to fundamental selection
principles without an independent argument.

<a id="post-contact-foundation-decision"></a>
### What the contact results change in the foundation audit

The bounded review declared at `6fc83d05` compares existing proofs, not new
responses. The [own-law source](nodal/ORGANIZED_ALTERNATIVE_OUTER_ACQUISITION.md)
and [finite contact](nodal/ORGANIZED_ALTERNATIVE_CONTACT.md) certificates now
retain both endpoint laws through the actual preparation and intervention
history. Their common marked identity does not make the complete laws
equivalent. The [ideal contact theorem](nodal/ORGANIZED_CONTACT_DYNAMIC_CANCELLATION.md)
separates complete-field reflection protection from law-dependent response
onset. Its limiting preparation is not the acquired source. The subsequent
[preparation theorem](nodal/ORGANIZED_CONTACT_PREPARATION_SENSITIVITY.md)
shows that all 96 internal directions are invisible at first amplitude
variation along the carrier, while explicit internal disturbances generate
quadratic response. This is neither finite-amplitude transverse stability
nor an all-times retention certificate. Its dynamical reflection is also
not the stationary sector assumed in the older reflection obstruction.

These results support the following dependency assessment. Task activation
and the prospective protocol belong only to the
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-checkpoint).

| Question | What the existing evidence permits | Missing admission |
| --- | --- | --- |
| Independent constitutive justification | The two supplied laws form the retained identities and are mathematically distinguishable; the information minimum constrains a description of a law | An independently justified restriction or a complete physical observation. Matching targets, dissipation, symmetry, onset order or operator labels does not select a law |
| Continued-contact maintenance | The original unit-strength sine law has full-state stationary acceptance and restricted instability theorems. The separately supplied [weak-contact law](nodal/ORGANIZED_WEAK_CONTACT_MAINTENANCE.md#weak-contact-source-basin) now certifies actual-source capture by strict minima on five explicit donor-origin intervals at fixed eta=1e-5, with every original uncertainty retained | Unit-strength nonseparable feasibility remains open. Weak-contact gaps retain saddle-converging origins and otherwise unclassified destinations. Neither selected contact strength nor capture justifies the complete law or its event occurrence |
| Independent physical property | Collective measurement maps remain admissible in principle; existing electrical comparisons expose explicit failures | A joint certificate of independently prepared state, complete loaded law/support, input and clock calibration, and observation errors, obtained without the reserved response |
| Additional collective coordinates | Constituent memory already derives quadratic feedback and nonclosure; compression already supplies a controlled finite prediction | A named missing predictive capability. Another covariance representation alone closes none of the preceding gaps |

The review's stationary target question concerns a **stable organization
in the original complete marked class under the original unit-strength
law**. Its unresolved status does not obstruct every supplied contact law:
the weighted comparison has its own source-to-target obligations. The
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#alternative-weak-contact-source-admission)
selects the next dependency. Identity retention alone need not approach
a stable point; the separate basin theorem establishes that stronger
conclusion for its declared source intervals and strength.
The ontology distinguishes formation, finite interaction and
[maintenance](EMERGENT_ONTOLOGY.md#generative-bound-organization); the
finite-interaction theorems alone do not settle that last obligation.

The [marked-class proof](nodal/ORGANIZED_CONTACT_STATIONARY_SEARCH.md#marked-cycle-information-and-connected-chambers)
retains the mixed cycles omitted by the fifteen reported windings. The
actual source lies in its product class, and uninterrupted cut-free
maintenance cannot leave that class. A search in another class cannot
repair this source dependency. The full problem includes nonacute gaps,
reflection-breaking states and the strips removed by the previous search's
artificial margin. All 53 relative phases remain; the relative donor origin
is included in those coordinates, not an additional degree of freedom.

There are useful reductions but no full-domain stationary stability decision. The
[full-fine curvature witness](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#5-a-full-fine-obstruction-to-stable-retention-near-the-carrier)
retains its original acute outer-gap and internal-gap-defect hypotheses.
The subsequent
[winding-triangle argument](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#full-chamber-phase-curvature-tests)
removes the acute outer-gap requirement from a stronger necessary defect
bound and excludes every common internal phase profile from stable
stationary contact. An exact counterexample proves its nine fixed
phase-pair sums are incomplete. The
[stationarity extension](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#stationary-resultant-and-slot-comparison)
proves nonzero aligned neighbor resultants for every PSD stationary
candidate and requires negative midpoint edge curvature in the cyclic slot
comparison, even for degenerate Hessians. This is not a sign verdict on the
total curvature. Nonseparable targets meeting those necessary conditions
still require a full stationary/curvature decision. The
[exact donor relaxation](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#exact-donor-relaxation)
preserves that decision through nine strictly positive donor directions
and 44 host-relative directions. Its local reduced potential preserves
stationary points and local minima. Separately, the
[global stationary reconstruction](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#global-stationary-donor-reconstruction)
uses all host balance equations to determine at most one donor completion
for compatible host data. Host residuals, unit modulus, donor equations
and marked lifts remain separate feasibility requirements. This does
not extend the local reduced potential globally or close a reduced
evolution law. The
[common-host-profile obstruction](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#common-host-profile-obstruction)
further excludes a common internal profile of the five hosts: stationarity
forces the donor to share it, putting the full state in the already
excluded common-profile sector. The
[stationary compatibility manifold](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#stationary-compatibility-manifold)
then makes the regular local constraint count precise. At an admitted
stationary point with strictly positive joined host Hessian block A,
the host residual and unit-circle guards
have rank 36; nine donor coordinates parameterize host compatibility
locally. Host relaxation preserves eight relative curvature directions
and 45 positive directions. Positivity of A is not a global premise already
proved by donor reconstruction. At PSD singular A with nullity r, the guard
Jacobian has rank `36-r` and r host-only neutral directions persist; the
regular chart cannot be assumed there. These are stationary charts, not
closed temporal laws.

The [finite host-reorganization exclusion](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#finite-host-reorganization-exclusion)
adds an actual stationary restriction beyond those coordinate reductions.
For a cut-free winding-one reference rim, an explicit neighborhood of
common host profiles contains no PSD equilibrium, even with an arbitrary
donor profile. Near the uniform rim, host balance alone excludes every
equilibrium. The distance is a maximum phasor chord error on the fine host
coordinates. The closed uniform-rim neighborhood cannot retain the
unforced flow indefinitely: compactness and the exact storage loss force
equilibrium omega limits, which its residual bound excludes. Arbitrarily
late returns would give a limit in that closed tube, so eventual exit is
permanent. It has no uniform time bound and does not imply winding loss.
It cannot be transferred to a general PSD-only neighborhood that may
contain saddles.
The [actual outer-one corollary](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#actual-outer-one-host-reorganization)
combines the retained acquisition and complete observation-history bounds:
all twelve original allowances, all 96 relative preparation directions and
every donor origin enter the uniform-rim tube of radius `1/25`. Under the
unchanged unforced law, each eventually remains outside that tube. This is
an analytical consequence of the existing source proofs, not a new
numerical source assessment or a change to frozen response flags. It gives
no uniform exit-time bound or winding-loss result.

The [complete-flow convergence theorem](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#full-contact-convergence)
adds a single limiting equilibrium and finite total variation of the
continuous phase lifts for every finite initial state on the fixed law's
zero-form-mean leaf; all forms tend to zero. It derives this from the
analytic potential and complete dissipative equations without assuming
convexity, symmetry or a positive Hessian. The actual outer-one sources
therefore have limiting uniform-reference host distance at least
`1/(9*sqrt(2))`. Convergence does not select a stable target: saddle,
degenerate and marked-cut limits remain possible. Only a cut-free limit
guarantees eventual constancy of the marked windings. The permanent
uniform-tube exit uses compactness and loss alone; the analytic argument
is needed for the single limit and finite phase variation. Sustained
nonstationary motion cannot bypass stationary maintenance in this model.

A separate compactness argument gives a positive uniform distance between
the common-host family and all PSD stationary states in the compact closure
of the original marked phasor class, if any exist. It includes degenerate
host blocks and approach to marked cuts, but computes no global gap or
target. The remaining sector requires finite nonseparable host
reorganization; its existence and stability are still undecided. No
physical identification follows. The
[reflection obstruction](nodal/ORGANIZED_CONTACT_REFLECTION_OBSTRUCTION.md#5-complete-law-consequence-and-scope)
requires its static sector. The
[root and stability certificate](nodal/ORGANIZED_CONTACT_STATIONARY_SEARCH.md#root-and-stability-acceptance)
can accept a constructed target, but supplies neither that target nor a
global exclusion. The frozen 24-start nonfinding adds neither. Carrier
mode separation does not extend those tools to arbitrary stationary states.
The generic root/Hessian certificate is reusable independently of that
search's `1/1024` marked margin; the full-class question retains strict
avoidance of the actual cuts, including the omitted margin strips.

For the fixed sine law, strict positive relative phase curvature at a
certified stationary point gives local asymptotic stability of the complete
two-row law on its fixed-mean leaf. A negative relative phase direction
instead gives an unstable complete tangent mode. A global obstruction must
therefore establish a negative direction at **every** stationary point in
the declared class; merely failing to find a strictly positive Hessian does
not exclude degenerate stable points. Conversely, a certified stable point
would establish an available target, not capture of the actual source or
indefinite maintenance of its fifteen identities. Even a global stationary
instability theorem would leave possible special-source saddle manifolds
and asymptotic approaches to marked cuts to be addressed separately.

Thus the review identifies an analytical certificate mechanism as the
missing admission, rather than authorizing another numerical search.
The original sine law has the existing stationary acceptance tools; choosing
it for this question does not prefer it as a fundamental law or transfer
its conclusions to the alternative. A subsequent comparison needs its own
force derivatives, storage and stability proof.

The [law-selection boundary](#law-selection-and-boundary-information) and
the failed physical-admission reviews remain unchanged. The required physical
certificate is more than improved voltage precision: absent phase or hidden
preparation information, loading and joint sensor coverage remain independent
obligations. Likewise, supplying a support event to preserve an identity
would add an occurrence law; the
[support-law nonselection result](nodal/RELATIONAL_SUPPORT_EVENTS.md)
does not derive that law from passivity or desired retention. No source,
trajectory, stationary search, physical record or frozen producer was
evaluated in this review.

<a id="post-capture-foundation-decision"></a>
### What source-to-attractor capture changes

The [actual-source basin theorem](nodal/ORGANIZED_WEAK_CONTACT_MAINTENANCE.md#weak-contact-source-basin)
closes a specific generative dependency: the acquired and observed host,
joined to its independently prepared donor at fixed eta=1e-5, reaches a
specified strict relative minimum on five declared origin intervals.
The conclusion retains the full fine state and original errors. It is
stronger than stationary existence, finite retention or mean-only
attraction. The [ontology](EMERGENT_ONTOLOGY.md#generative-bound-organization)
still distinguishes this conditional organization from physical binding,
autonomous support formation and identification as a material constituent.

The five target labels describe symmetry-related relative registrations.
Their multiplicity is not a species count. Graph symmetry constrains the
allowed reduced torque, but does not by itself fix its nonzero harmonics,
sign or stable zeros. The sine proof uses its supplied potential in host
shape elimination, the fifth-order restoring coefficient, the uniform
remainder and source transport. Those premises cannot be removed by
calling the resulting registration a property of topology alone.

This makes constitutive robustness the most direct new foundation test.
The [retained harmonic alternative](nodal/ORGANIZED_ALTERNATIVE_CONTACT.md#organized-alternative-contact)
already has own-law preparation and finite interaction results. It is a
declared counterlaw with the same primitive domains, reciprocal support
and loss structure, rather than a newly fitted explanation. The
[own-law weak-contact classification](nodal/ORGANIZED_ALTERNATIVE_WEAK_CONTACT.md)
now changes the attracting registrations themselves: its admitted tube
contains ten minima and ten index-one saddles, and the five sine-attracting
relative origins are saddles under the alternative. Own-law normal
curvature and uniform torque remainders retain all fine directions in
this verdict. Marked identity and support symmetry therefore do not
determine these attracting geometries for the retained pair.

The separate [own-source theorem](nodal/ORGANIZED_ALTERNATIVE_WEAK_CONTACT.md#alternative-weak-contact-source-capture)
now carries the actual alternative-law preparation, its observation
history and every original error through the changed weak contact.
It retains marked identity at every donor origin and assigns an
attracting destination on ten declared intervals. On one common
primitive preparation interval, sine and the alternative have disjoint
asymptotic registrations, each with its own reached state. This supplies
an acquired-outcome distinction without a finite settling deadline or
an independent physical readout.

The separate [joint transition theorem](nodal/ORGANIZED_CONSTITUTIVE_TRANSITION.md)
now quantifies a local loss of stability within the existing constitutive
family. Its critical deformation scales cubically with contact strength,
with explicit joint derivative errors for 0<eta<=1e-12. All 52 normal
phase directions remain positive while the soft registration undergoes
a supercritical pitchfork. The central critical state has quartic
restoring storage and remains asymptotically attracting, without
exponential stability. The marked identity survives the local transition.

This distinguishes robustness at fixed contact from a deformation
tolerance uniform as contact vanishes. It is stronger than qualitative
hyperbolic persistence, but it does not certify the transition at eta=1e-5,
capture intermediate-law sources or select a unique fundamental law.
The comparison coordinate connecting deformation and contact is not a
new autonomous evolution rule. A stationary shape elimination also does
not supply the effective dynamics of a parent interacting with another
organization: retained slow state and causal memory remain separate
obligations. Existing closure and event counterexamples must be checked
before treating either as a new research gap.

The alternative priorities retain their existing boundaries. Narrowing
basin gaps or enlarging eta refines the same conditional construction
without yet distinguishing an independent premise. The original
unit-strength stationary problem concerns a different supplied law.
[Assembly work](nodal/ORGANIZED_ASSEMBLY_WORK.md) and
[event admission](nodal/RELATIONAL_SUPPORT_EVENTS.md#support-event-premise-admission)
already explain why a permitted contact is not an occurrence law; another
proof of that distinction would not classify the maintained organization.
The [physical review](research/ORGANIZED_RECOVERY_OBSERVATION.md#recovery-operational-admission)
still lacks independent preparation and measurement coverage. Capture
does not supply those missing observations, a laboratory clock or a
unique microscopic law.

The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#independent-boundary-premise-admission)
owns the single next gate and its complete-law scope. Endpoint capture
and the local transition distinguish retained identity from law-dependent
registration. They do not select nature's law, globally classify the
intermediate models, admit their sources or supply autonomous occurrence.

<a id="post-transition-dynamics-decision"></a>
### Priority after a soft collective mode appears

The stationary transition leaves a specific dynamical question: what
governs the slow registration, and which part of a complete initial state
sets its subsequent motion? Its answer would connect constitutive
sensitivity to collective relaxation and predictive state without
introducing a new microscopic law. Eliminating stationary host shape
does not prove that the same graph is invariant under the two-row flow.

This is distinct from two already resolved questions. The
[constituent-memory result](nodal/ORGANIZED_CONSTITUENT_MEMORY.md#organized-constituent-memory-result)
excludes general closure by means or instantaneous current corrections
on the earlier five-child model. The
[controlled slow-phase theorem](nodal/SINE_FORM_PHASE_REDUCTION.md#sine-controlled-slow-phase)
retains form initialization and proves finite-horizon errors at fixed
support as a different constitutive ratio decreases. Neither supplies
the new critical registration's mobility, local full-state projection
or relaxation at fixed positive weak contact. Their methods transfer;
their coefficients and error bounds do not.

The [repeated-composition theorem](nodal/REPEATED_ORGANIZED_COMPOSITION.md)
already retains child-to-port association in one finite parent interaction.
That boundary requirement survives a new reduction: a scalar isolated-mode
description cannot inherit another support's interaction certificate merely
because both objects are composed of C9 children.

| Obligation | Existing conclusion and genuinely missing premise |
| --- | --- |
| Independent law justification or observation | Compatibility, information cost and acquired outcomes do not select a unique law. A new independent restriction or jointly admitted preparation, clock and physical readout is needed; the scoped electrical and recovery obstructions remain valid. |
| Autonomous preparation and contact occurrence | Event admissibility, passivity and assembly work do not select occurrence. Progress requires a named access, timing, reset or reservoir mechanism with its own justification; the pitchfork supplies none. |
| Sufficient parent dynamics | The isolated reduction and one admitted marked interaction derive original-clock drift, released means, full-state initialization and local identity retention. Their effective coordinates do not close arbitrary states; a changed boundary must re-admit the surviving mode and its dynamics. |

The [local reduction](nodal/ORGANIZED_CONSTITUTIVE_TRANSITION.md#constitutive-transition-local-dynamics)
closes the selected consequence: inherited mobility and algebraic critical
recovery follow from the complete law, with a strong-stable exception.
Equal initial phases can approach from opposite sides because the center
initialization also consumes form. This rules out resetting that initial
coordinate to the observed registration alone.

The [marked-contact result](nodal/CRITICAL_PARENT_INTERACTION.md)
now re-admits relative form and phase means alongside the critical
registration. One organized neighbor changes the asymptotic recovery
coefficient through the joined tangent geometry. At matched means and
storage, the internal registration still determines the sign of the
initial receiver exchange. A separate hidden-form witness excludes an
exact instantaneous quotient even in those three coordinates.

These results do not authorize arbitrary contacts, operator descent,
intermediate-law acquisition or a uniform reduced solver. The
[boundary-constraint theorem](nodal/COLLECTIVE_BOUNDARY_CONSTRAINTS.md)
now separates the reusable local mechanism from its constitutive input:
positive boundary curvature constrains precisely the old neutral
directions visible at that boundary. Normal compliance changes the
static coefficient; the complete kinetic pencil, memory and hidden
initialization determine the dynamic response. At fixed total weight,
split contact removes the critical mode retained by a single port.

<a id="boundary-criterion-foundation-decision"></a>
<a id="post-resource-foundation-decision"></a>
### Independent premises after structure and observation bounds

The mathematical methods are established kernel and Schur-complement
arguments. Their useful consequence here is the separation of inputs:
connectivity does not specify phase curvature; a positive tangent does
not specify a recovery rate; and a neutral tangent does not specify
nonlinear attraction. Circular nonnegative potentials with the same
Hessian can have different restoring orders or local stability. A
cancelled exported mean signal also need not remove internal restoration.

This closes the named structural dependency without selecting the
potential, kinetic rows, critical tuning or occurrence of contact.
The matched-conductance example is a discriminator for that distinction,
not grounds for further contact variants. Independent law justification
and physical admission remain open for their existing reasons.

The [phase-response classification](nodal/SINE_CONSTITUTIVE_INFORMATION.md#phase-response-faithfulness)
tests one distinct premise: every nontrivial primitive phase rotation
must be visible to some query of the declared current channel. Exact
Fourier support determines its invisible subgroup, but both retained
counterlaws are already faithful. Circle-valued state and full-state
identity do not impose faithfulness on every current or collective
observable. A many-to-one flow quotient can lose native U3 or winding
information even when both continuous rows descend exactly.

The [free probe](nodal/ORGANIZED_FREE_PHASE_PROBE.md) and
[small-phase comparison](nodal/ORGANIZED_SMALL_RESOURCE_INFORMATION.md)
now provide conditional access, including full fine-state errors, preparation
resources and distinguishable or overlapping reading sets. The latter removes
large initial mean forms as a requirement for every observation in this family.
Neither result supplies a new restriction on the current, kinetic rows or
accessible physical preparations. A lower information or attachment cost does
not become a selection principle without independent justification for that
requirement. The [reciprocal resource envelope](nodal/RECIPROCAL_FORM_RESOURCE_ENVELOPE.md)
also bounds what changing contacts can supply; it does not produce an occurrence
law or transfer its weighted comparison model to native phase-pressure support.

The remaining distinction is between an unresolved proof and an absent premise:

- **Complete-law justification.** Held capacity and support already complete
  those channels in the selected models. The
  [NFR requirements](EMERGENT_ONTOLOGY.md#12-can-nfr-properties-justify-a-joint-law)
  allow boundary-dependent inherited laws and memory; they do not require
  cosine storage, first-resultant sufficiency or unchanged scalar profiles
  across every contact. The retained countermodels already satisfy the weaker
  requirements. Selecting among them needs additional independent content,
  not another derivation from the same premises or replacement of the nodal row.
- **Equivalence and observation.** The
  [form chart foundation](FUNDAMENTAL_THEORY.md#foundational-state-admission),
  [complete interface maps](nodal/ORIENTED_INTERFACE_PROFILE_INHERITANCE.md#oriented-interface-selection)
  and [scalar calibration theorem](nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-calibrated-observation)
  have stated transformation classes. Disjoint record sets exclude equivalence
  preserving that preparation, clock, input and observation contract. They do
  not classify arbitrary nonlinear state conjugacies with changed inputs and
  readouts. A broader equivalence question first needs a justified operational
  class; an unspecified class is not a missing theorem to solve by enumeration.
- **Generative occurrence.** The
  [operator activation audit](STRUCTURAL_OPERATORS.md#operator-mechanism-and-activation-audit)
  and [event premises](nodal/RELATIONAL_SUPPORT_EVENTS.md#support-event-premise-admission)
  already separate candidate access, eligibility, action, cost and occurrence.
  A reservoir or source port retains its own preparation and constitutive law.
  Its availability does not derive those inputs. Autonomous capacity, support
  and preparation become necessary for a claim consuming them; they are not
  prerequisites to every conditional formation or physical comparison.
- **Independent physical bridge.** The small-phase theorem consumes a supplied
  relative phase and fine-state budget, a declared clock and an observation law.
  Gain and exposure allowances presume those inputs. The
  [recovery preparation counterexample](research/ORGANIZED_RECOVERY_OBSERVATION.md#recovery-operational-admission)
  shows why voltage precision alone cannot certify unobserved relative phases.
  The [electrical review](research/ORGANIZED_ELECTRICAL_FEASIBILITY.md#organized-electrical-physical-admission)
  separately retains complete-law and preparation mismatches. Those older
  models supply no bridge for the eighteen-node comparison; their numerical
  apparatus limits do not transfer to it either. A collective sensor map is
  allowed, with independent preparation, loading, clock and reduction errors.

There are still conditional mathematical questions, including receiver
acquisition or an all-future exclusion beyond its
[finite nonacquisition result](nodal/CONSTITUTIVE_RECEIVER_COMPARISON.md#constitutive-receiver-comparison-result).
That question requires a new source-valid invariant-domain or capture argument;
extending the finite run does not supply one. It is not evidence that another
law or engine mechanism is already justified. The same distinction applies to
deferred unit-strength stationary classification and capacity/support models.

The reviewed owners therefore identify no new independently justified selector
or ready complete mechanism that resolves the current dependency. This is a
boundary of the admitted repository evidence, not a proof that none can exist.
The [constitutive entry condition](research/FIVE_STAGE_EXECUTION_PLAN.md#new-constitutive-evidence-admission)
requires a named new premise or observation, its independent basis and the
retained alternative it can exclude. For an autonomous generative claim, the
input may instead be a justified source/occurrence mechanism with its own
capability; it need not select the current. Physical comparison need not await
unique law selection or substrate origin once its own bridge is admitted.
The [sole queue](research/FIVE_STAGE_EXECUTION_PLAN.md#post-resource-foundation-review)
records this selection and physical-admission dependency; another internal
observation refinement does not fill the missing input. It does not forbid
investigating a named new conditional capability, such as
[localized identity transport](EMERGENT_ONTOLOGY.md#localized-identity-transport),
under an explicitly supplied complete law.

### Cross-channel foundation audit

The follow-through is centralized in
[NODAL_PARAMETER_FOUNDATIONS.md](NODAL_PARAMETER_FOUNDATIONS.md), not another
constitutive-law inventory. It covers every foundational parameter family,
derives coefficientwise form/time covariance and the conditional local
diffusive generator, and records telemetry, metric, energy and memory units.
The shared coherence-length estimator retains distinct fit and spectral
fallback provenance. Configured gains and phase-period interpretations require
their stated premises; a numeric value alone does not close autonomous
phase/capacity/support dynamics.
