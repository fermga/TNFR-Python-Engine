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

The actual functions also retain their finite fallbacks: a nonpositive q returns
a one-operation window, and the search stops at 64; nonpositive ν_f dt returns
zero debt capacity. These are compatibility policies, not stability results for
the corresponding Euler modes.

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

The corresponding reusable constraint is the
[source-tangency identity](FORCED_SUPPORT_BALANCE.md#22-source-tangency-without-a-telemetry-controller):
it derives what phase/capacity evolution must satisfy to preserve zero
pressure, without selecting that evolution through Si. The finite witnesses
are reproducible with
`artifacts/research/validate_source_tangency_2026_09_18.py`; their successful
checks do not prove a general autonomous Si reduction or an emergent mechanism.

The separate [capacity/phase result](FORCED_SUPPORT_BALANCE.md#23-capacity-exposure-does-not-determine-a-phase-clock)
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
The previously proved [nonnegative irreducible criterion](FORCED_SUPPORT_BALANCE.md#24-rigidity-and-flexibility-of-a-held-phase-source)
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

The review corrected concrete implementation discrepancies:

- Signed finite nonzero EPI now counts as existing form in static, cached
  and incremental initiation checks. Invalid provided values do not silently
  become permission. The legacy missing-live-value default is retained for
  provisional selection; it is not evidence that a physical state was supplied.
- Public name validation now delegates causal U2/U4 checks to the same owner
  as operator validation. Previously it accepted both a three-unit debt prefix
  and Mutation without prior IL. Parsing uses the same decision, removing a
  second implementation that required stabilizers even without destabilizers.
- The exact diagnostic OZ/ZHIR waiver is labeled separately and does not
  supply live execution preconditions. Legacy pair-compatibility and THOL-end
  restrictions remain additional API policies; core acceptance alone does not
  imply those stricter APIs accept a word.
- The grammar observer normalizes operator instances, reports isolated U3
  assessment as unavailable, and separates phase rejection from unrelated debt
  or history rejection. Its sequence preview keeps the current phase snapshot;
  it does not execute a future word.
- Unified U6 reporting reuses the canonical drift/domain validator. Missing
  or mismatched reference fields are unavailable rather than zero drift.
  A passing mean drift is no longer described as trajectory safety.

There is still no complete derivation of the thirteen maps, their invocation
law, role catalog or numerical policies from the scalar nodal identity alone.
Nor is there a proof that six grammatical rules are minimal or sufficient for
all admissible TNFR dynamics. These are explicit research obligations, not
grounds to remove existing contracts without a replacement theorem.

For the main generative question, the audit closes the proposed strict-U3
**rank** implication negatively while the witness's finite fixed-source set
is locally rigid. Finite geometry must be distinguished from tangent freedom
before using either to support autonomous NFR maintenance. A joint
phase/capacity/support law would still need independent structural derivation;
grammar admission or a telemetry score cannot silently supply it. The single
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) retains that queue.

## 13. Structural grammar refactor and the full nodal system

The target is a grammar whose physical conclusions can be traced to nodal
structure and evolution, not to unexplained preferences. This is not achieved
by renaming a calibrated constant, differentiating an underdetermined identity,
or removing a contract without proving a replacement. The implementation now
separates these responsibilities:

- [Grammar basis registry](../src/tnfr/operators/grammar_canon.py): immutable,
  compositional identity/theorem/contract/policy records state their hypotheses,
  configured choices and mathematical owner. A rule can contain several kinds.
  These are declarations of provenance, not proof-carrying admission tokens.
  SDK role views share this owner instead of maintaining incomplete role lists.
- [Word profiles](../src/tnfr/operators/grammar_patterns.py): `legacy` retains
  the extra adjacency and THOL-terminal preferences; `core` excludes those
  preferences while retaining the shared U1-U5 word contracts. The context
  key `compatibility_profile` carries an explicit choice through the existing
  word/event entry points. Neither profile derives the remaining calibrated
  U2/U4 policies or waives live operator checks. Existing anti-pattern warnings
  and the optional semantic validator remain separate policies; `core` is not
  a global diagnostic-disable switch.
- [Rejection mode](../src/tnfr/operators/grammar_dynamics.py): graph setting
  `GRAMMAR_REJECTION_MODE="raise"` rejects blocked requests without computing
  a priority-list substitution. The absent/default `fallback` mode preserves
  compatibility. Validated words always reject blocked steps. This changes
  grammar replacement, not independent selector policies, and derives no
  preferred next operator. The existing `filter_candidates` remains a set
  of policy-admissible candidates, not an autonomous selector.
- [Structural evidence](../src/tnfr/operators/grammar_evidence.py): a typed,
  intact finite executor result can supply the common-metric gain bound of
  its own represented EPI composition. A bound at most one certifies
  nonincrease for those represented maps; a bound below one certifies their
  contraction. The comparison value one is the identity gain, not a fitted
  safety threshold. A bound above one does not prove that the actual observed
  trajectory expanded. Missing evidence is unavailable; a caller Boolean,
  raw dictionary or research claim label cannot supply it.
- [Observation](../src/tnfr/operators/grammar_observations.py): word policy,
  current-snapshot U3, caller-declared contract/U6 flags and typed structural
  execution evidence are distinct. The latter does not authenticate the
  current graph or authorize its next step. A finite represented-map result
  is not a global executable-map, future, full-channel or tetrad theorem.

These changes make the boundaries executable. They do not assert that the
complete grammar or operator invocation law has already emerged from the
nodal equation. Existing strict phase checks, state-domain checks, history
requirements and atomic execution remain in their shared owners.

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
[existing source-tangency derivation](FORCED_SUPPORT_BALANCE.md#22-source-tangency-without-a-telemetry-controller).
They constrain proposed laws for nu, theta and support; they do not select
those laws. The [same-initial-state completions](FORCED_SUPPORT_BALANCE.md#23-capacity-exposure-does-not-determine-a-phase-clock)
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

This gate starts with coherent structural form and `x'=diag(nu)*p`, not with
the claim that each implemented formula is a necessary TNFR law. The
[foundation audit](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation)
owns EPI types, coordinate changes, units and the original source's conflicting
definitions. Source fidelity preserves the underlying hypothesis; it does not
require keeping an inconsistent dimensional equation or a lossy chart claim.

### Existing laws and independently supplied premises

| Component | Implemented or derived content | Missing implication |
|-----------|--------------------------------|---------------------|
| EPI flow | Shared integrator implements capacity times declared pressure; isolated EPI diffusion has its exact Dirichlet balance. | The nodal identity alone does not choose the state space or pressure functional. |
| Phase relaxation | [coordination.py](../src/tnfr/dynamics/coordination.py) blends global/local circular means per invocation with configured, possibly adaptive gains. | Invocation time, gains and their adjustment policy are not derived by the EPI identity. |
| Oscillator proposal | [phase_evolution.py](../src/tnfr/dynamics/phase_evolution.py) adds `dt*nu` and a configured sine coupling over admitted neighbors. | Capacity as angular speed and the coupling coefficient are additional constitutive premises. This is distinct from mean relaxation. |
| Capacity adaptation | [adaptation.py](../src/tnfr/dynamics/adaptation.py) applies clipped neighbor averaging after stored Si/pressure gates and a consecutive-call count. | Averaging gain, thresholds, waiting count and use of that diagnostic as a controller are supplied policies; there is no independently derived capacity rate. |
| Optional extended system | [canonical.py](../src/tnfr/dynamics/canonical.py) supplies explicit nonlinear phase/pressure responses when requested by the integrator. | Its coefficients and flux correspondence need justification; it provides no capacity law and does not follow from the EPI product. |
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

### Cross-channel foundation audit

The follow-through is centralized in
[NODAL_PARAMETER_FOUNDATIONS.md](NODAL_PARAMETER_FOUNDATIONS.md), not another
constitutive-law inventory. It covers every foundational parameter family,
derives coefficientwise form/time covariance and the conditional local
diffusive generator, and records telemetry, metric, energy and memory units.
The earlier scalar/vector coherence-fit distance discrepancy is resolved by
one shared estimator; fit and spectral fallback still have distinct meanings.
Configured gains and phase-period capacity claims are corrected in source
comments without changing their numeric values. This strengthens the evidence
boundary but does not close autonomous phase/capacity/support dynamics.
