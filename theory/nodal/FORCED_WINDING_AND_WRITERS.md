# Winding generation and native writer boundaries

Acute winding retention, chart escape and native Mutation/capacity writer obstructions.

Part of [Forced support balance and model boundaries](../FORCED_SUPPORT_BALANCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 32. Acute-cycle relaxation retains phase winding while form relaxes

### The supplied joint law and the support it actually uses

Fix a simple undirected phase-support cycle with cyclically ordered vertices
`i=0,...,n-1`, `n>=3`, and positive common capacity `kappa`. Let every support
edge remain U3-admitted. The supplied averaged-sine phase law is

```text
theta_dot_i = kappa + (K/2)*(sin(delta_i)-sin(delta_(i-1))),    K>0,
delta_i = wrap(theta_(i+1)-theta_i).
```

Indices are cyclic. The phase denominator is two independently of transport
conductance. Fix symmetric nonnegative conductances W on that support whose
positive part is connected. Write `s_i=sum_j W_ij`, `S=sum_i s_i`,
`B=diag(s)-W` and `L_rw=diag(s)^(-1)*B`. The form law considered here is

```text
x_dot = kappa*(-e*L_rw*x + w*g),             e>0, w>=0,
g_i = Arg(exp(i*delta_i)+exp(-i*delta_(i-1)))/pi.
```

The phase and EPI coefficients `w,e` are supplied effective pressure weights.
The full configured mixture is admissible: common fixed capacity makes its
capacity-gradient channel zero, and every node of the simple cycle has unique
support degree two, making its topology-gradient channel zero. Their
coefficients need not be disabled or renormalized, even with unequal transport
conductances. Gamma, further operator events, support changes and evolving
capacities are absent. Any clipping is inactive under
the chart condition below. This is a conditional continuous model of the
declared channels; the nodal product alone does not derive this phase law
or the selection of its support.

### Acute gaps form an invariant region

Assume all initial oriented gaps lie in `[m,M]` with
`-pi/2<m<=M<pi/2`. On the corresponding continuous lifts, differentiating
the neighboring phase equations gives the exact gap equation

```text
delta_dot_i = (K/2)*(sin(delta_(i+1))-2*sin(delta_i)
                    +sin(delta_(i-1)))
            = -(K/2)*(L_cycle*sin(delta))_i.
```

At a maximal gap its derivative is nonpositive; at a minimal gap it is
nonnegative, since sine is increasing on this interval. Thus `[m,M]^n` is
forward invariant. This also supplies continuation inside the branch-safe
acute chart. If the configured U3 gate contains that interval, no edge loses
admission along this exact trajectory. A gate narrower than the initial gaps
does not meet the premise.

The gap sum has zero derivative. Since it is initially `2*pi*ell` for the
cycle winding integer ell, let

```text
delta_bar = 2*pi*ell/n,
u = delta-delta_bar*1,
rho = max(abs(m),abs(M)) < pi/2,
lambda_C = lambda_2(L_cycle) = 2-2*cos(2*pi/n).
```

Then `sum u_i=0`. Multiplication of the gap equation by u gives

```text
(1/2)*d||u||_2^2/dt
 = -(K/2)*sum_i (u_(i+1)-u_i)
                  *(sin(delta_(i+1))-sin(delta_i))
 <= -(K*cos(rho)/2)*u^T*L_cycle*u
 <= -gamma*||u||_2^2,

gamma = K*cos(rho)*lambda_C/2 > 0.
```

Consequently `||u(t)||_2<=Q*exp(-gamma*t)`, where `Q=||u(0)||_2`.
This is convergence to uniform **oriented gaps**, not to equal phases when
`ell!=0`. The phase-rate error also satisfies
`||theta_dot-kappa*1||_2<=K*||u||_2`: the cycle incidence has norm at most
two and sine is 1-Lipschitz. Its integrability implies that each lifted
`theta_i(t)-kappa*t` has a limit, with a rate-error tail bounded by
`K*Q*exp(-gamma*t)/gamma`. The common rotation is therefore separated from
the convergent relative configuration. No global semicircle containing all
node phases is required.

### The canonical phase source vanishes, but phase geometry remains

Both neighbor displacements lie in the center's open half-pi interval.
Their resultant is

```text
exp(i*delta_i)+exp(-i*delta_(i-1))
 = 2*cos((delta_i+delta_(i-1))/2)
       *exp(i*(delta_i-delta_(i-1))/2).
```

Its displayed cosine is positive. The exact phasor direction is therefore
the midpoint and

```text
g_i = (delta_i-delta_(i-1))/(2*pi),
sum_i g_i = 0,
||g(t)||_2 <= (Q/pi)*exp(-gamma*t).
```

This local identity uses the actual neighbor-phasor pressure, not a proposed
global replacement of that pressure by a Laplacian. It ceases to be justified
outside its chart/resultant premises. At the limiting winding configuration
the pressure source is zero, whereas the tetrad's local absolute phase
gradient is `abs(delta_bar)=2*pi*abs(ell)/n`. Thus vanishing phase pressure
does not erase nonuniform phase geometry. This maintained relative-phase
configuration does not, on its own, identify an autonomous NFR or a particle.

### One existing margin controls availability and conditional stiffness

The same geometry supplies more information without another fitted parameter.
Let `r_i=|sum_(j in N(i)) exp(i*theta_j)|/2`. The preceding exact identity gives

```text
r_i = cos((delta_i+delta_(i-1))/2) >= cos(rho) > 0.
```

Thus the preserved acute interval certifies availability of every local phasor
direction for this ideal trajectory, even when the global resultant is zero.
The existing `cosine_lower_bound` in `CycleRelaxationEnvelope` is an outward
rational lower bound for this same quantity; a separate tunable availability
threshold is unnecessary. The certificate does not assert zero rounding error
or that an arbitrary native phase writer preserves the interval.

It also controls the Hessian of the supplied sine law's alignment potential.
For an oriented cycle incidence matrix A and
`V(theta)=sum_edges(1-cos(delta_e))`,

```text
Hessian(V) = A diag(cos(delta_e)) A^T >= cos(rho)*L_cycle.
```

After removing the common rotation, its smallest eigenvalue is at least
`cos(rho)*lambda_2(L_cycle)`. Multiplication by the declared phase mobility
`K/2` gives the existing decay bound. This is a conditional restoring margin,
not a new primitive or a phase law derived from the nodal product.
The phase-pressure response is a different derivative: within these fixed
two-neighbor branches, `D_theta g=-L_cycle/(2*pi)`, independent of the cosine
weights. Confusing these two Jacobians would silently substitute the supplied
sine evolution for the actual Arg-based pressure. Existing source, response
and cycle owners already contain the necessary quantities.

### The weighted form response and its nonconserved mean

Let `m_s=(sum_i s_i*x_i)/S`, `y=x-m_s*1`, and
`||z||_s^2=sum_i s_i*z_i^2`. Fixed reversible transport is self-adjoint in
this metric. Let

```text
lambda = kappa*e*lambda_2(L_rw) > 0,
D0 = ||y(0)||_s^2,
A = Q/pi,
C = kappa*w*sqrt(max_i s_i)*A.
```

Projection onto the s-centered subspace is an orthogonal contraction.
Variation of constants and the preceding source bound yield

```text
||y(t)||_s <= exp(-lambda*t)*sqrt(D0) + C*J(lambda,gamma,t),

J(lambda,gamma,t)
 = integral_0^t exp(-lambda*(t-v))*exp(-gamma*v) dv
 = (exp(-gamma*t)-exp(-lambda*t))/(lambda-gamma), lambda!=gamma,
 = t*exp(-lambda*t),                                  lambda=gamma.
```

Both terms tend to zero. The equality case is retained instead of dividing
by a zero rate difference. Conservative positive lower bounds on either
rate remain valid because the defining integral decreases as either rate
increases. Squared upper bounds can be formed from these nonnegative terms
without treating rounded square roots as exact coefficients.

The s-weighted mean need not be conserved. Diffusion makes no contribution
to its derivative, but the actual phase source gives

```text
m_s_dot = (kappa*w/S)*sum_i s_i*g_i
        = (kappa*w/(2*pi*S))*sum_i (s_i-s_(i+1))*u_i.
```

Define

```text
M = kappa*w*Q*sqrt(sum_i (s_i-s_(i+1))^2)/(2*pi*S).
```

Then `abs(m_s_dot)<=M*exp(-gamma*t)`. Hence a finite limit `m_infinity`
exists and

```text
abs(m_s(t)-m_s(0)) <= M*(1-exp(-gamma*t))/gamma,
abs(m_infinity-m_s(t)) <= M*exp(-gamma*t)/gamma.
```

Every EPI coordinate converges to this same limiting mean. The theorem does
not assign the limit to the initial mean unless the weighted source actually
vanishes. Equal transport strengths make M zero; unequal strengths require
the retained source integral or its bound. The mean-tail estimate refers to
the exact modeled trajectory at time t, not automatically to a numerically
computed endpoint with an unmeasured solver error.

For a sufficient all-time form-chart check, `J<=1/lambda`. A conservative
global disagreement bound is

```text
D_global = C^2/lambda^2,                    D0=0,
           D0,                             C=0,
           2*D0+2*C^2/lambda^2,             otherwise.
```

Since `abs(y_i)<=sqrt(D_global/min_i s_i)`, the entire exact trajectory lies
inside the interval centered at `m_s(0)` with radius
`M/gamma+sqrt(D_global/min_i s_i)`. If this interval lies in the configured
hard EPI chart, clipping stays inactive and the preceding linear form
equation applies throughout. This sufficient interval can be conservative;
failure to fit is not evidence that clipping actually occurs.

### Apply the theorem to the actual closing-edge conductance

For the ideal arithmetic of Section 31's default Coupling event, put
`f=1/(pi+1)` and retain its five cyclic gaps

```text
delta(0) = (pi/5)*(2-2*f, 2+f, 2, 2, 2+f).
```

Their mean is `2*pi/5`, their winding is one, and
`Q=f*pi*sqrt(6)/5`. Their maximum `(2+f)*pi/5` is strictly below `pi/2`.
The theorem therefore predicts convergence toward uniform winding-one
gaps under the supplied phase law. The predicted nonzero winding is retained,
while the initially nonzero phase pressure decays.

The EPI graph must retain the closing conductance
`a=1-(2+f)/10`, not replace it by one. Its strengths and total are
`s=(1+a,2,2,2,1+a)` and `S=8+2*a`. In particular,

```text
sum_i s_i*g_i = (a-1)*(delta_0-delta_3)/(2*pi),
sum_i s_i*w*g_i at t=0 = w*f*(1-a)/5,
M = kappa*w*(1-a)*Q/(sqrt(2)*pi*S).
```

For `w=1/2` this recovers Section 31's positive initial compatibility
`f*(1-a)/10`. Uniform initial EPI therefore starts with a positive weighted
mean derivative, even though the unweighted phase-source sum is zero.
The asymptotic result is uniform EPI at a finite, potentially shifted value,
alongside nonuniform winding-one phase geometry. It is not a maintained
nonuniform scalar EPI profile.

The ideal expression for f and the captured binary64 event endpoint are
separate initial data. The implemented endpoint certificate starts from
the **actual** materialized phases, capacities, conductances and EPI, rather
than silently replacing them with the ideal formulas above. Exact real
arithmetic then defines the conditional reference model from those inputs.

### Reuse and numerical evidence boundaries

[`cycle_relaxation.py`](../../src/tnfr/physics/cycle_relaxation.py) bounds this
conditional model without advancing a graph. It reuses the shared detached
pressure/transport capture, exact rational-pi branch enclosures and reversible
spectral-gap owners. An actual materialized oriented gap is represented as
`r_i+2*k_i*pi`, with rational r_i and a verified branch integer k_i. Its
winding and deviations from the mean retain mathematical pi; a binary64
gap sum is not rounded to an integer and promoted to a proof.

Rational enclosures bound the initial Q, the acute radius and the phase and
transport gaps. For `0<=rho<pi/2`, the concavity bound
`cos(rho)>=1-2*rho/pi` supplies a conservative positive phase coefficient.
Shared exact exponential enclosures, with outward exponent rounding on a
2^-64 arithmetic grid and the shared exponent work limit 4096, bound the decay, Duhamel response,
mean drift/tail and all-time chart. These are exact conditional inequalities
for the reference model, not floating-point asymptotic convergence results.

Production execution instead refreshes its represented pressure and uses
the shared phase proposal and nodal integrator. Trigonometric reduction,
source realization, represented coefficients, Euler truncation, phase writes
and possible clipping retain their own numerical discrepancies. Finite
production observations can check a declared horizon against the prospective
envelopes; they do not remove those discrepancies or certify arbitrary future
binary64 execution. The preceding Coupling event remains policy-selected,
and this maintenance theorem does not derive autonomous support selection,
formation from a uniform preparation, or physical identification.

### Frozen finite production continuation

The [post-event controls](../../tests/physics/test_cycle_postevent_relaxation.py)
reuse the same default UM event through the
[shared preparation](../../tests/joint_phase_helpers.py). Before continuing,
the evaluator fixes envelopes at `t=0,8,16,32` from the actual captured
endpoint, with `K=1/2`. The finite execution uses `h=1/8` for 256 steps,
with fresh canonical pressure, the shared phase proposal and the existing
nodal Euler integrator. There are no further operators, support/capacity
changes, controller updates or fitted final targets.

The conservative represented-state reference rates are approximately
`gamma_lower=0.03573031566296992` and
`lambda_lower=0.32231115638261676`. Its all-time sufficient EPI interval is
approximately `[-0.09720168,0.34720168]`, strictly inside the hard chart
`[-1,1]`. These are rounded displays of rational model bounds.

The retained finite observations are:

| Read-out | At event endpoint | At t=32 |
| --- | ---: | ---: |
| Gap deviation norm from winding-one twist | 0.3716106 | 0.00342818 |
| Canonical phase-source norm | 0.1079811 | 0.000641409 |
| Strength-metric EPI disagreement norm | 0 | 0.00172119 |
| Weighted EPI mean | 0.125 | 0.1280933944 |

The form disagreement initially rises in response to the source and then
falls; its observed peak norm is approximately `0.05042945`. All four
reserved observations satisfy the prospective envelopes. Each finite step
retains winding one, a positive U3 margin and interior EPI, while the exact
Euler energy accounting has zero identity residual. The maximum retained
state-rounding defect is approximately `1.39e-17`; fresh-pressure assembly
defect is at most approximately `2.00e-18`. These do not bound accumulated
phase or Euler truncation error or prove numerical convergence.

Independent 110-digit comparisons check affine-pi initial gaps, source/mean
identities, the rational spectral lower bound and the exponential convolution.
Reversed orientation, node relabeling, zero source, equal decay rates, unit
transport weights and rejected branch/gate/capacity/support domains delimit
the implementation. Replacing the closing weight by one makes the ideal
mean-drive bound zero, demonstrating why the actual weight cannot be dropped.
The raw operator pressure and refreshed pressure produce the same conditional
reference but retain different reported stored-pressure residuals.

## 33. A common phase semicircle obstructs winding generation by the existing maps

### The question is generation, not maintenance of a prepared sector

Sections 31 and 32 distinguish a support event that exposes a prepared phase
arrangement as a new cycle period from subsequent conditional maintenance.
They do not show that those mechanisms can generate the phase arrangement
from an initially common semicircle. The following obstruction addresses that
specific missing step. It concerns cycle winding, not every possible form of
coherence, nonuniform EPI or NFR identity.

Let a finite set of existing nodes have real phase lifts `q_i` in a common
interval `I=[a,b]` of width `b-a<pi`, with physical phases `q_i mod 2*pi`.
Such a closed interval is contained in an open semicircle. The interval can
cross the displayed zero-phase cut, and need not have width below the U3
gate `pi/2`. In particular, this is a **global chart** hypothesis, not the
weaker condition that every existing support edge has an acute gap.

All angles, circular means and arithmetic in the proof initially mean exact
real operations. The implemented binary64 maps and detached chart observation
have separate boundaries below. Support may change among these same nodes.
No assumption about EPI equality, transport conductance or Sense Index is
needed for the phase-chart statement.

### Actual Coupling proposal and merge formulas preserve that chart

The target kernel first selects its U3-compatible existing neighbors. Its
phase consensus is the circular mean of either those neighbors alone or,
when bidirectional writes are enabled, the target together with those
neighbors. An admitted target has a nonempty compatible-neighbor set.
Every selected input is in I. After rotation by `-(a+b)/2`, each input
phasor has strictly positive real part. Their sum cannot be zero, and its
direction has a unique lift c between the minimum and maximum input lifts.
Thus `c in I` for either bidirectional setting.

For the declared push `f in [0,1]`, each proposed phase write has the form

```text
p_i = q_i + f*wrap(c-q_i) = (1-f)*q_i+f*c in I.
```

The second equality follows from `abs(c-q_i)<=b-a<pi`; the shortest arc is
the ordinary difference in this common lift. Reduction modulo `2*pi` is a
representation change, not motion out of the chart. This argument covers
the target and every optional bidirectional neighbor write. It does not
require the default factor specifically; the kernel's entire admitted
interval `[0,1]` is convex.

The simultaneous stage can give one node several proposals from different
targets. All proposals read the same snapshot and use the same original
`q_i`. If they are `p_i^(1),...,p_i^(r)`, the actual stage's idealized
shortest-arc displacement merge is

```text
q_i^+ = q_i + (1/r)*sum_alpha wrap(p_i^(alpha)-q_i)
      = (1/r)*sum_alpha p_i^(alpha) in I.
```

Each difference has magnitude below pi, so the second equality does not
identify distinct circle branches. Untouched nodes remain in I. Source-rank
ordering is relevant to represented summation but not to this exact convex
identity. The post-merge U3 validation remains a separate requirement: the
chart result neither promises that every requested stage is admitted nor
weakens the configured gate.

Functional-link candidates and their eventual weights are chosen separately
from these phase proposals. Adding edges between existing nodes whose final
phases stay in I cannot invalidate the common lift. This conclusion holds
regardless of whether EPI, Sense Index, candidate sampling or a threshold
selects a particular link. It establishes a phase constraint on the existing
selection policy, not a derivation or endorsement of that policy as an
emergent physical law. The fixed-node premise excludes importing a new node
with an independently prescribed phase outside I.

### Equal-capacity sine evolution preserves the moving chart

Between events consider the supplied averaged-sine law with one common
capacity/free angular rate `kappa>=0`, coupling `K>=0`, and whatever neighbor
subset `A_i(t)` the U3 gate currently admits:

```text
theta_dot_i = kappa + (K/d_i)*sum_(j in A_i) sin(theta_j-theta_i),
d_i = |A_i|,
```

with zero coupling contribution when `d_i=0`. In a frame rotating at kappa,
the same formula acts on lifted `q_i=theta_i-kappa*t` without the free term.
While the range is below pi, a maximal q_i has only nonpositive sine
contributions, and a minimal q_i has only nonnegative contributions.
Consequently the maximum cannot increase and the minimum cannot decrease.
The chart I is invariant in the rotating frame, and its width remains
strictly below pi.

This extremum argument does not require reciprocity, a fixed neighbor count
or a conserved phase mean. It remains valid when the gate changes the
admitted subsets, or when the support changes without a phase write. For
a discontinuous gate it states invariance for every absolutely continuous
trajectory satisfying the supplied equation almost everywhere; it does
not assert a new existence or uniqueness theorem for arbitrary switching
fields. Grammar and operator events retain their own admission conditions.

The shared helper also has a directly applicable exact-Euler statement. For
one simultaneous step with `h>0` and `h*K<=1`, put

```text
a_ij = sin(q_j-q_i)/(q_j-q_i),         q_j!=q_i,
a_ij = 1,                            q_j=q_i.
```

Within the common chart `0<a_ij<=1`. For a nonempty admitted set the ideal
update, after removing the common rotation `h*kappa`, is

```text
q_i^+ = (1-(h*K/d_i)*sum_j a_ij)*q_i
        + (h*K/d_i)*sum_j a_ij*q_j.
```

All coefficients are nonnegative and sum to one. An empty admitted set leaves
q_i unchanged in this rotating frame. Thus the Euler map preserves I for
arbitrary simultaneous admitted subsets under this step bound. The bound
is sufficient, not a declaration that larger steps always lose the chart.
The helper does not enforce `h*K<=1` for all callers; a composition invoking
this result must establish it separately. Sine of the raw displayed phase
difference equals sine of the lifted difference only in this exact-real
identity; represented argument reduction remains numerical evidence.

### Why neither new edges nor finite compositions can create a winding

For any oriented edge between nodes in I, its unique principal gap is
`q_head-q_tail`, since its magnitude is below pi. For every closed support
walk these differences telescope:

```text
sum_(edges in a closed walk) wrap(theta_head-theta_tail) = 0.
```

Every cycle period therefore vanishes, independently of the cycle basis.
This includes a cycle created by the next admissible link addition. Existing
cycles, new cycles and a different choice of spanning tree have the same
zero-period conclusion.

Induction now applies to any finite admitted composition consisting only of
the stated exact Coupling stages, common-capacity sine-flow intervals or
Euler steps satisfying their step condition, and support changes among the
same nodes that leave phases unchanged. There is a common moving phase
chart throughout, so none of these compositions generates a nonzero cycle
winding from the stipulated preparation. This conditional statement does
not require choosing a particular sequence of functional-link decisions.
It is not an unrestricted claim about all 13 operators or complete runtime
execution.

Section 31's initially prepared five-node path has phases
`0,2*pi/5,4*pi/5,6*pi/5,8*pi/5`. Its shortest containing circular arc has
width `8*pi/5`, exceeding pi. It is already outside this invariant chart
class, even though each consecutive path gap is acute. Before closure the
path has no cycle winding to measure. Closure makes that supplied,
non-semicircle phase arrangement observable as a new nonzero period; it
does not demonstrate its generation from a common-chart preparation.

### Common capacity is an ideal identity with an explicit numerical safeguard

The Coupling capacity proposal writes only requested targets. Bidirectional
phase writes do not themselves give every affected neighbor a capacity write.
Its declared blend, with `r in [0,1]`, is

```text
nu_target^+ = nu_target+r*(mean(nu_compatible_neighbors)-nu_target).
```

If every node has the same capacity kappa, this formula preserves kappa
exactly. Disabling capacity synchronization also preserves that supplied
common value. Hence Coupling does not create the heterogeneous free rates
that would escape the preceding **exact-model** premise.

The represented implementation requires an explicit equal-input identity.
Unguarded evaluation of `fsum(neighbors)/len(neighbors)` is not always
idempotent: three binary64 values `0.1` have the represented mean
`0.10000000000000002`. With blend factor one, a target-only write would
create a capacity difference from equal represented inputs. Summation can
also overflow for several equal large finite capacities even when their
mathematical mean is that same finite input.

The shared `coupling_capacity_blend` therefore returns the target value
directly when all compatible-neighbor capacities equal it. This is the
declared blend's equal-input identity, not a new capacity-selection law or
a change to nonuniform-input arithmetic. Both the direct and staged owners
reuse this numerical kernel. A runtime study must still retain the observed
capacities: this narrow safeguard does not certify unrelated capacity writes,
phase arithmetic or future invocations.

### A necessary capacity-contrast budget for leaving the common chart

Allow the supplied capacities in the same continuous sine law to vary by
node and time. Assume locally integrable finite rates and an absolutely
continuous lifted trajectory satisfying that law almost everywhere between
finitely many admitted Coupling events. Use unrotated real lifts for this
estimate. Up to the first chart boundary, let

```text
D(t) = max_i q_i(t)-min_i q_i(t) < pi,
osc(nu(t)) = max_i nu_i(t)-min_i nu_i(t).
```

The coupling contribution still points inward at each phase extremum.
Consequently the derivative of the maximum is at most `max_i nu_i(t)` and
that of the minimum is at least `min_i nu_i(t)` almost everywhere. It follows
that

```text
D'(t) <= osc(nu(t)),
D(t) <= D(0)+integral_0^t osc(nu(v)) dv.
```

Each exact Coupling phase event has `D(t^+)<=D(t^-)`, by the convex proof
above, so the same bound survives their insertion. Support-only changes
among the existing nodes do not change D. These statements concern every
trajectory satisfying the premises; they add no existence or uniqueness
claim for the discontinuous gate.

If `t_*` is the first boundary with `D(t_*)=pi`, a necessary condition is

```text
integral_0^t_* osc(nu(v)) dv >= pi-D(0).
```

If a uniform bound `osc(nu)<=R`, `R>0`, is supplied, this also gives
`t_* >= (pi-D(0))/R`. Zero contrast recovers the invariant common-chart
result. A capacity update can change the subsequent rate contrast without
an instantaneous phase jump; it contributes through the integral, not as
an invented phase displacement.

This budget is necessary, not sufficient. Inward coupling can prevent chart
escape even when the bound allows it. Losing the global common chart is
itself weaker than creating a nonzero cycle period or a maintained pattern;
the first limiting antipodal pair need not be a support edge. The budget
does not derive the supplied capacities or their evolution. It supplies a
checkable requirement for a proposed escape mechanism, not that mechanism.

### Exact chart observation, finite execution and the remaining obligation

[`phase_chart.py`](../../src/tnfr/physics/phase_chart.py) observes whether the
captured phase set admits a common open semicircle. It interprets represented
radians as exact real inputs, retains mathematical pi through the shared
rational enclosures, and uses the equivalent criterion that the largest
empty circular gap exceed pi. Successful admission provides consistent lifts
and zero periods in the shared support-cycle coordinates. Exclusion of the
chart and an unresolved enclosure are separate outcomes; neither is a
positive winding or a successful formation result.

This detached observer does not establish how its state was produced, seal
a trajectory or prove that a future runtime event retains the same chart.
Its connected 2-to-32-node, at-most-50-edge implementation budget is narrower
than the chart theorem. The actual arithmetic owners are
[`_coupling_stage_kernel.py`](../../src/tnfr/operators/_coupling_stage_kernel.py)
and [`phase_evolution.py`](../../src/tnfr/dynamics/phase_evolution.py); the
[formation controls](../../tests/physics/test_phase_chart_formation.py) exercise
their finite behavior separately from the exact composition argument.
Phasor evaluation, shortest-arc subtraction, displacement averaging, modulo
writes and Euler rounding must not be silently identified with exact convex
maps. In particular, the equal-capacity safeguard alone gives no binary64
phase-chart theorem for arbitrary repetition.

Generating nonzero winding from the invariant preparation requires leaving
at least one of its premises. Heterogeneous free rates can remove the common
rotation; a phase-changing map outside the proved convex family can remove
the chart invariant; and an introduced or prepared phase outside the chart
changes the initial-data hypothesis. None of these possibilities is by
itself a sufficient generative mechanism. Topology changes alone among the
same common-chart phases cannot supply the missing period. A large numerical
step or accumulated rounding discrepancy is also not a derived physical
source. Any proposed escape must identify its actual operator/state
preconditions or constitutive law and retain the independent evidence for
formation, maintenance and selection.

### Finite controls for overlapping writes and the excluded preparations

The production control fixes a five-node path with displayed signed phases
`(-0.30,-0.12,0.04,0.16,0.29)` represented modulo tau, common capacity one,
EPI `1/8`, Si `0.8`, seed 17 and declared initial AL histories. One actual
default simultaneous Coupling stage targets `(1,3)` with all eligible
candidates; no factor or compatibility threshold is tuned. Node 2 receives
overlapping phase proposals. The stage adds `(0,3)`, `(1,3)` and `(1,4)`:
cycle rank rises from zero to three while every fundamental period is zero.
The exact endpoint observer proves both common charts; their diameters shrink.

Independent 80-digit circular-mean and convex-merge calculations agree with
the represented writes within `2e-15` radians while retaining a nonzero
arithmetic difference. Reversing target order preserves primary endpoints.
Four subsequent shared joint steps at `h=1/8`, `K=1/2` retain the observed
common chart, common capacity and zero cycle periods. These finite checks
neither seal future execution nor turn the exact model into a binary64 theorem.

A separate three-node path starts with equal zero phase and supplied positive
capacities `(1,2,3)`. One phase proposal at `h=2`, `K=1/2` gives `(2,4,6)`
radians, outside every common open semicircle. This control violates the
common-capacity premise and is a tree: it demonstrates possible chart escape,
not winding birth, an exact continuous trajectory or autonomous contrast.
Section 31's prepared wound path and its executed C5 endpoint are also
correctly excluded from this obstruction's initial class.

Geometry controls retain repeated phases, node relabeling, a chart crossing
zero and transport-weight independence. The represented pair `(0,math.pi)`
is slightly narrower than a mathematical semicircle and is admitted with a
small positive exact margin. A valid deliberately coarse pi enclosure instead
returns `unresolved` where required, with unavailable periods rather than zeros.
[Direct/staged capacity regressions](../../tests/operators/test_coupling_jacobi_stage.py)
confirm that the repaired shared capacity identity retains `0.1` at a
three-neighbor target, plus zero, subnormal and extreme finite kernel inputs.

## 34. Native runtime admission uses relaxation, not the supplied sine clock

### Reuse the native owners before transferring the oscillator result

The preceding sine-law results do not identify the phase evolution used by
`runtime.step`. That entry point calls
[`coordinate_global_local_phase`](../../src/tnfr/dynamics/coordination.py), a
configured relaxation per invocation. It does not call the averaged-sine
proposal, add `dt*nu_f` to phase, or use U3 to filter the coordinator's local
neighbors. Capacity contrast therefore is not, by itself, a native angular
speed contrast. Section 33's accumulated capacity-contrast budget concerns
its supplied oscillator law and cannot be transferred to this native map.

The existing
[native diameter bound](PRIMITIVE_PHASE_CLOSURE.md#native-phase-contrast-budget),
[writer audit](PRIMITIVE_PHASE_CLOSURE.md#native-phase-writer-closure) and
[selector reachability result](../DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#uniform-capacity-and-default-selector-reachability)
remain the mathematical and policy owners. This section connects those
results to the complete step boundary, extends the represented common-capacity
identity beyond the earlier retained unit value, and specifies what an actual
native formation check must retain. It introduces no substitute phase law.

### The actual step order and freshness boundary

For the built-in path, [`runtime.py`](../../src/tnfr/dynamics/runtime.py) performs
the following operations in order:

1. Before-step callbacks; mutation-flow boundary and node-sample bookkeeping.
2. The configured pressure refresh, then Si refresh when `use_Si=True`.
3. Selector decisions, lag overrides, grammar processing and sequential
   primitive glyph execution; another mutation-flow boundary is recorded.
4. The configured integrator, followed by EPI/capacity clamps and phase
   re-expression.
5. Global/local phase coordination, followed by the stored-pressure/Si
   capacity-adaptation gate and another mutation-flow boundary.
6. The optional auxiliary math step, delayed-EPI history, automatic REMESH,
   validators, after-step callbacks and cache telemetry.

There is no intervening pressure/Si refresh after the glyphs, EPI integration
or phase coordination. Under ordinary IL execution, the integrator consumes
the operator-contracted stored pressure. AL/EN can change form without
refreshing that pressure. The later adaptation gate reads the resulting
stored pressure together with Si computed before those state changes.
Automatic REMESH can introduce a further EPI jump after that gate. These
values identify their actual observation times; calling them fresh terminal
state pressure and Si would be incorrect.

`use_Si=False` suppresses the refresh, not consumers of stored Si.
`apply_glyphs=False` suppresses glyph execution, not subsequent coordination
and adaptation. The native integrator owns physical EPI time, whereas the
coordinator's gains act once per invocation without a `dt` argument. Changing
the number of native steps can therefore change the number of relaxation
maps even at a fixed total EPI integration duration.

### Default coordination stays in the common chart and consumes its diameter

In a common open-semicircle lift, the exact global phasor mean `b` and every
nonempty local mean `l_i` lie in the same interval; isolates use their own
phase. The actual coordinator's ideal formula is

```text
q_i^+ = (1-kG-kL)*q_i + kG*b + kL*l_i.
```

The existing diameter proof applies when `kG,kL>=0` and `kG+kL<=1`:
`D^+<=(1-kG)*D`. It is stronger than merely retaining the chart. The default
enabled adaptive policy clamps both gains after its stable, transition or
dissonant branch. Their nominal configured bounds are

```text
1/(8*pi^2) <= kG <= 1/(2*pi),
1/(8*pi)   <= kL <= 1/(2*pi).
```

For executable inequalities the endpoints are their stored binary64 values,
treated as exact coefficients. Their global lower endpoint `m` is positive
and their two upper endpoints sum to less than one. Thus every default
adaptive branch supplies the exact coefficient bound `D^+<=(1-m)*D` for
the ideal map. Kuramoto order and recent disruptive-glyph load choose the
adaptive branch; neither branch supplies an outward phase source in this
domain. The separate fallback defaults for a partially configured policy
must not be substituted for the fully injected default configuration.

These facts are already checked by the
[native diameter controls](../../tests/physics/test_native_phase_diameter_budget.py).
They do not guarantee arbitrary configured overrides, disabled adaptation,
modified gain bounds, undefined represented resultants or all binary64
trajectories. Legacy native coordination uses ordinary reductions; its
optional exact-component evidence version is not silently enabled by
`runtime.step` and does not certify transcendental phase accuracy either.

### Which native glyphs and other writers preserve the premises

With fresh default Si and equal positive capacities, its capacity-normalized
term equals one and the existing selector bound gives a base IL choice.
For initialized nodes, meaning nonzero scalar EPI or valid retained nonempty
glyph history, IL is admitted by the incremental grammar: U1a no longer
requires a generator, and IL supplies stabilization rather than new debt.
The default selector has no parametric soft-repetition filter. Lag overrides
can request AL or EN; if one is rejected in this initialized class, IL is
the first admitted fallback. Thus the relevant reachable primitive set is
`{IL,AL,EN}` under those hypotheses. Outside that class, inspect the actual
applied labels and grammar outcome; a label appearing later in the fallback
list does not prove either its occurrence or its impossibility.

The native dispatcher does not invoke the public `Coherence` class or an
all-target operator stage. `_op_IL` changes only stored pressure; `_op_AL`
changes form. For a graph-bound EN call, the adapter captures a
`ReceptionReadSnapshot` and its prepared writer changes form and its kind.
These paths leave phase, capacity and support unchanged. Their history and
cache writes do not constitute a phase update. A public operator with an
additional configured phase action is a different path.

The built-in default integrator advances scalar EPI, its rate bookkeeping
and the runtime clock, holding phase and capacity fixed. The ordinary clamps
apply common capacity bounds to every node. An equal positive capacity
already inside those bounds therefore remains that same value. Exact phase
wrapping preserves the circular state; its represented implementation is
addressed below. No EPI stability or eventual EPI equilibrium follows just
from these phase/capacity facts.

Automatic network REMESH is the protected delayed-EPI map. It does not call
topological remeshing or the separate structural-memory phase/capacity
interpolation. Its transaction protects phase, capacity and support against
its `ON_REMESH` observers. The standard optional math engine advances a
separate state vector and telemetry without an inverse projection onto the
nodal phases or capacities. Arbitrary custom pressure hooks, integrators,
selectors, mutable-object implementations, validators and before/after
callbacks remain additional writers whose effects require their own audit.
The theorem excludes such unaccounted changes rather than treating a callback
name or a suppressed exception as evidence of a read-only operation.

### Preserve the declared adaptation fixed point in represented arithmetic

The adaptation owner uses a held neighbor-capacity snapshot and eligibility
from stored pressure, stored Si and a consecutive-call counter. In exact
arithmetic, a common capacity kappa is unchanged by

```text
nu_i^+ = (1-mu)*nu_i + mu*mean(nu_neighbors),      0<=mu<=1,
```

regardless of which subset qualifies. This does not require the stored gate
inputs to describe the latest phase/form state: arbitrary eligibility still
leaves that equal-capacity fixed point unchanged.

A represented implementation must preserve this identity explicitly.
For example, with target and neighbor capacities `0.104` and default
`mu=0.1`, the two-product expression can return `0.10400000000000001`.
If only that target is eligible, it creates a numerical contrast between
previously equal capacities. This is not evidence of a structurally generated
capacity mechanism. The shared adaptation proposal retains the current value
when its represented neighbor mean equals that value, avoiding the redundant
multiply/add operations. Section 38 additionally enforces the neighbor and
convex-proposal intervals for nonuniform inputs. Interior arithmetic and the
eligibility policy remain unchanged; neither guard changes a threshold or
event-selection law.

Phase clamps have a corresponding representation boundary. The native
contract re-expresses phases in `[-pi,pi)`, not necessarily `[0,2*pi)`.
Recomputing `(theta+pi) % (2*pi)-pi` for an already valid centered value can
lose small increments unnecessarily. The clamp therefore retains that valid
value directly and uses the existing re-expression for values outside the
centered range. The coordinator can subsequently store a raw unwrapped
proposal. This guard preserves the existing range contract where it applies;
it is neither a phase evolution law nor a guarantee of exact re-expression
at other branches.

The detached common-chart observer currently accepts canonical represented
inputs in `[0,math.tau)`. A centered or raw native endpoint outside that
input domain is not admitted merely because a periodic interpretation exists.
Finite controls that use this observer must state their actual input range;
they may not normalize the live graph just to manufacture an admissible
observation. A separately retained compatible lift and its representation
error are different evidence.

### Conditional whole-step conclusion and its limits

Start with initialized scalar nodes, fixed existing support, equal positive
capacity inside the configured rails and a common open-semicircle phase
chart. Retain fresh default Si, default selection and adaptive gains, the
built-in integration path and no unaccounted phase/capacity/support writer.
For each admitted **exact-model** native step, the primitive glyphs preserve
these phase/capacity conditions, adaptation preserves common capacity and
the coordinator contracts the chart diameter. For any finite number N of
such steps,

```text
D_N <= (1-m)^N*D_0 < pi.
```

Every support-cycle period remains zero. Indefinite continuation satisfying
these same ideal premises would exhaust phase contrast, rather than renew
it. This conditional conclusion is the existing native theorem with its
actual writer/admission requirements; it is not a new autonomous formation
law or an unrestricted claim about every grammar history and runtime option.

The numerical fixed-point and clamp repairs remove identifiable arithmetic
defects from this path. They do not make phasor evaluation, shortest-arc
arithmetic, normalization or complete runtime repetition exact. A finite
native experiment must retain actual glyphs, effective gains, primitive
phase/capacity endpoints, pressure/Si freshness and every intervening writer.
The existing diameter ledger then keeps phase realization and intervening
changes as explicit defects; it is not permission to assign unobserved
changes a zero value. Whether a different admitted native mechanism can
supply and maintain phase contrast remains a separate formation question.

### Finite ordinary-step control

[Native step controls](../../tests/physics/test_native_step_formation.py) execute
two independently initialized unit-conductance C5 graphs with seed 17, actual
injected defaults, common capacity `0.3`, phases
`(0.10,0.13,0.11,0.17,0.14)` and EPI
`(0.125,0.128,0.121,0.127,0.123)`. The cases differ only in supplied stability
counters: all zero versus `VF_ADAPT_TAU-1`. Those initial counters do not
certify preceding stability. The default `dt=0.5`, selector, pressure reader,
Si weights, coordinator, adaptation and REMESH gate are retained. Transparent
wrappers observe the original functions without replacing their decisions.

Both executions refresh pressure and Si, apply native IL at every node and
advance EPI nontrivially through the nodal integrator. The coordinator takes
its adaptive stable branch with represented gains
`kG=0.04461342911494398`, `kL=0.140157221158957`. The observed chart diameter
falls from approximately `0.07` to `0.05566448226923737`; the exact endpoint
observer separately admits both charts and finds zero cycle period. All
observed phases remain in the observer's input range without re-expression.
Support and capacity remain unchanged. Only node 4 passes the pressure/Si
gate; only the counter-ready case reaches the capacity writer, which retains
`0.3` exactly. Its counter advances rather than being reset.

The trace also retains the timing limitations: adaptation reads the IL-scaled
stored pressure and pre-glyph Si. The detached terminal pressure comparison
has a nonzero freshness residual. Mutation-flow timestamps and delayed-EPI
history contain the actual integrated endpoints. Automatic REMESH abstains
because the required stable-fraction history is absent; its gate was executed,
but this control does not claim an applied REMESH map. These are two finite
whole-step observations, not a proof of arbitrary binary64 repetition,
unobserved prior history or autonomous pattern formation.

## 35. Heterogeneous capacity opens a conditional Mutation admission gate

Section 34 closes a uniform-capacity class; it does not establish that fresh
default selection always excludes phase writers. The question addressed here
is whether heterogeneous supplied capacities can simultaneously satisfy the
default selector, live Mutation evidence and incremental grammar. The owners
remain [native selection](../../src/tnfr/dynamics/selectors.py),
[Mutation evidence](../../src/tnfr/physics/mutation_trigger.py),
[its runtime adapter](../../src/tnfr/operators/_mutation_gate.py) and
[incremental grammar](../../src/tnfr/operators/grammar_dynamics.py). Admission
of an endpoint and production of its preceding history are separate results.

### The ideal fresh-state compatibility inequality

Let `V=max_i(nu_i)>0`, `P=max_i(abs(p_i))`, `r_i=nu_i/V`, and
`z_i=abs(p_i)/P` when `P>0`. Let `d_i` be the absolute wrapped displacement
from the target phase to its neighbor-phasor mean, divided by pi. This is
the ideal phase dispersion used in the default Si definition; an isolate
has zero dispersion. Write `a,b,c` for the normalized default Si weights,
approximately `(0.75854699,0.18315345,0.05829955)`. Then

```text
Si_i = a*r_i + b*(1-d_i) + c*(1-z_i).
```

With no lag override, the default selector proposes ZHIR exactly when
`Si_i<=lo` and `z_i<=h`, where the injected thresholds are `lo=0.25` and
`h=0.133`. Thus its ideal low-Si condition is equivalent to

```text
r_i <= [lo-b*(1-d_i)-c*(1-z_i)]/a,       z_i<=h.
```

If `P=0`, both normalization owners use their zero-pressure fallback and
`z_i=0`; no division by zero or positive-pressure premise is required.
Mutation still requires strictly positive target capacity. Its additional
configured floor `ZHIR_MIN_VF` defaults to zero; `0.1` is the signed growth
threshold `ZHIR_THRESHOLD_XI`, not a capacity floor.

In an exact common open-semicircle chart of diameter `D<pi`, every defined
neighbor mean lies in that chart, so `d_i<=D/pi`. A necessary bound is

```text
r_i <= R(D) = [lo-b*(1-D/pi)-c*(1-h)]/a.
```

For the nominal defaults, `R(0)` is approximately `0.02148955`,
`R(pi/4)` approximately `0.08185280`, and its limiting value as `D` tends
to pi approximately `0.26294256`. These are restrictions on relative
capacity, not a proof of selection: current pressure, history, grammar and
represented numerical margins still have to be checked. An additional
capacity floor larger than `V*R(D)` would exclude this chart's gate.
Executable comparisons use captured represented coefficients and thresholds;
the rounded displays above are not directed interval certificates.

For the pressure bound, use the separate effective pressure weights
`A,B,C`, whose nominal defaults coincide with `a,b,c`. On nonnegative
conductance and support with defined phase resultants, the ideal default
pressure is

```text
p_i = A*g_i + B*(weighted_neighbor_mean(x)-x_i)
              + C*(neighbor_mean(nu)-nu_i),
abs(g_i) <= D/pi,
P <= A*D/pi + B*(x_max-x_min) + C*(V-nu_min).
```

The capacity difference is raw capacity, **not** the normalized Si quantity
`r_i`. The EPI mean uses transport conductance; the phase and capacity means
use support neighbors. The default topology coefficient is zero. The bound
therefore uses the actual EPI and capacity ranges, including their units;
replacing the last term by `C` would generally be incorrect. With the default
rails it is at most `A*D/pi+2*B+C*VF_MAX`, where `VF_MAX` is the stored
binary64 value near `2*pi`. This bounds fresh current pressure only.

In particular, a small current product `nu_i*p_i` does not invalidate a
larger previously observed signed EPI secant. Mutation validates two retained
samples ending at the live EPI, with timestamped evidence authoritative.
The preceding interval may have had different pressure, phase or capacity.
Conversely, supplying a valid sample pair does not establish that native
execution generated it. EPI rails alone do not exclude the history gate:
two samples in `[-1,1]` can exceed the `0.1` threshold over an appropriately
short positive interval. No measured derivative is reconstructed from
current pressure to manufacture this evidence.

### Which additional reachable primitives can supply a phase write

For initialized nodes and fresh default selection, the base choices are
`{IL,OZ,ZHIR,NAV,RA}`. Lag overrides add AL and EN. An incremental-grammar
rejection has IL as its first admitted fallback in this class; it does not
make every later fallback label reachable. Successful native primitive
writes have the following scope:

| Reachable glyph | Direct state writes relevant here | Support change |
| --- | --- | --- |
| IL, OZ, NAV | Stored pressure | None |
| AL, EN | EPI, with EN's form-kind contract | None |
| RA | EPI, optional capacity amplification and phase alignment | None |
| ZHIR | Phase, after live evidence and grammar admission | None |

This is the primitive dispatcher audited in section 34, not the public
operator classes, arbitrary hooks or an all-target stage. In particular,
the native selector does not directly choose UM or a child-creating THOL
operation. Optional network REMESH retains its separate protected EPI scope.

The [RA primitive](../../src/tnfr/operators/__init__.py) uses only its actual
U3-compatible neighbor subset. In a common chart, its ideal phase write
has the form `q_i^+=(1-eta)*q_i+eta*b_i`, where `b_i` is that subset's
phasor mean and default `eta=1/(4*pi)` lies in `[0,1]`. It therefore cannot
expand the common lift hull. This remains true for successive admitted
primitive writes; it does not remove represented mean/wrap errors.
Fresh default RA selection also requires `Si_i<0.5`. Nonnegativity of the
other Si terms gives `nu_i/V<0.5/a`. Its optional default multiplier
`1+1/(8*pi)` consequently leaves that target below approximately
`0.68538189*V`, with `V` from the selection snapshot. This limited bound
does not supply a general capacity-convergence theorem.

ZHIR supplies a different write. Its
[shared proposal](../../src/tnfr/operators/_mutation_stage_kernel.py) uses the
current pressure sign and the default configured shift factor, giving a
shift near `+0.25` or `-0.25` radians. The trigger is the observed signed
EPI growth, not the sign or magnitude of this current pressure. Positive
growth, phase-shift direction and subsequent useful source work must not
be identified with one another.

### Declared-history endpoint: the admission intersection is nonempty

Consider connected unit-conductance P4 with nodes `(0,1,2,3)`, injected
defaults and the following supplied state:

```text
capacity = (0.6, 1.2, 6.0, 6.0)
phase    = (2.5, 1.0, 0.2, 2.0)
EPI      = (-0.85, 0.95, -0.9, 0.9).
```

Supply node 0's glyph history `IL,OZ` and timestamped EPI pairs
`(-0.5,-0.95),(0.0,-0.85)`, with zero lag counters before selection.
These histories are declared initial data, not a claimed native prefix.
An independent binary64 endpoint refresh gives approximately

```text
pressure = (0.00247643659, -0.12731743356, 0.50822502874, -0.76429162695)
Si       = (0.22966947381,  0.36304588940, 0.88544357994,  0.83676124714)
base     = (ZHIR, NAV, IL, IL).
```

At target 0, the endpoint pressure can also be checked directly as
`-A*1.5/pi+B*1.8+C*0.6`. The shared history gate reports an observed rate
approximately `0.2`, above `0.1`, while the current nodal product is only
approximately `0.00148586195`. The incremental grammar admits ZHIR for the
declared history. All supplied EPI and capacity values are inside the
default rails. The exact common-chart observer admits diameter `2.3<pi`.
Thus the selector, live history, capacity and grammar conditions are jointly
compatible; the equal-capacity obstruction does not extend to this endpoint.

The admitted default positive shift would move target 0 from `2.5` to
`2.75`, outside the previous lift hull `[0.2,2.5]`. The new diameter is
still `2.55<pi`: outward phase motion is not a chart-crossing claim.
P4 is a tree, so its zero support-cycle winding is vacuous. The maintained
native execution controls below use a separate declared graph; endpoint
arithmetic and read-only admission do not certify an applied P4 batch or
a complete native P4 step.

### A finite native prefix produces its own Mutation evidence

The [heterogeneous Mutation controls](../../tests/physics/test_heterogeneous_mutation_admission.py)
retain one prospective six-node case: a complete graph on nodes `(0,1,2,3,4)`
with pendant node 9 joined only to node 1. All conductances are one. Inject
defaults, seed 17, initial runtime time zero, empty glyph histories and no
supplied Si. Initialize

```text
node 0:    phase=0.0, EPI= 0.6, capacity=0.8
nodes 1-4: phase=2.8, EPI=-1.0, capacity=0.1
node 9:    phase=4.7, EPI= 1.0, capacity=6.25.
```

Three ordinary native steps, each with the default physical interval `0.5`,
produce the target-0 sequence `IL,OZ,ZHIR`. The pressure reader, fresh Si,
selector, grammar, nodal integrator, coordinator and adaptation remain the
original owners; observation wrappers do not replace their decisions.
Before the three target decisions the retained fresh values are approximately

| Native decision | Target pressure | Target Si | Global maximum absolute pressure |
| --- | --- | --- | --- |
| IL | 0.34221320754 | 0.15845231195 | 1.18360987888 |
| OZ | 0.18135858879 | 0.19478282220 | 0.70716117563 |
| ZHIR | 0.04528547527 | 0.23430789739 | 0.63220290369 |

The first two low-Si base choices request OZ; the initial lack of a recent
handler makes grammar supply IL first. The next OZ is then admitted. Before
the third decision, normalized target pressure is below `0.133`, and the
actual timestamped history supplies a positive secant approximately
`0.19126945644>0.1`. Its current nodal product is only approximately
`0.03622838022`; that prediction is not substituted for the observed rate.
The actual preceding `IL,OZ` supplies the grammar context. This is a native
history-producing trace, unlike the P4 endpoint's supplied histories.

The paired control changes only all initial capacities to `0.8`; it yields
target `IL,IL,IL`. This comparison identifies a finite consequence of
supplied capacity heterogeneity under the same configured policy; it does
not establish how that initial heterogeneity arose.

The initial phases `(0,2.8,4.7)` do not admit a common open-semicircle chart.
Therefore this case is not an escape from section 34's common-chart class.
Before the third decision, however, an independent exact-pi lift check
already admits a common chart. The target phase is
`1.063472935200046`; its neighbors lie between `2.751877932766905` and
`2.8637878070029825`. The actual ZHIR jump is `+0.25`, toward every one
of those neighbors. The pendant's raw phase `-2.2705373138792972` requires
adding true `2*pi`, enclosed by the shared rational pi bounds. This check
does not pass an unsupported negative value to the detached common-chart
observer or replace true `2*pi` by represented `math.tau`.

The retained lift's diameter is approximately `2.949175<pi` before ZHIR
and decreases by `0.25` at that phase-only event. This causal occurrence
therefore supplies observed phase alignment, not outward hull renewal.
Capacity remains heterogeneous, so the event does not contradict section
34's equal-capacity selection restriction. A new cycle period, coherent
identity or continued source renewal does not follow. The ensuing
coordinator's contribution and the corresponding source-work balance
are measured in section 36.

### From admission to the retained source budget

The native trace closes finite occurrence with genuine temporal evidence;
section 36 measures whether that actual write renews source work and how the
ensuing relaxation changes it. One local distinction is already exact.
Hold a neighbor-phase center fixed, choose a consistent branch, and write
`u=center-phase` and `delta` for the admitted Mutation shift. Provided the
before/after separations remain on that branch,

```text
(u-delta)^2-u^2 = delta^2-2*u*delta.
```

For the default direction `sign(delta)=sign(p)`, opposite signs of `p` and
`u` increase this squared separation. Matching signs decrease it unless
the shift overshoots sufficiently. This is a fixed-center local identity,
not a global hull, winding, pressure-work or field-energy theorem. The P4
endpoint has `u=-1.5` and positive pressure, but a positive Mutation shift
can instead align a target whose neighbor center lies ahead. Triggering
Mutation therefore does not imply contrast renewal.

The completed budget in section 36 reuses this actual third-step case and its
uniform-capacity control, retaining phase endpoints around the glyph batch
and coordinator. It introduces no additional case or longer horizon. The
[forcing capture](../../src/tnfr/physics/forcing_realization.py) reports a
pressure vector `F`, not `diag(nu)*F`. At fixed EPI, capacity and support,
the prospective refreshed Dirichlet-rate increment is
`(B*x)^T*diag(nu)*Delta_p`; a unit-capacity formula cannot be copied directly
to this heterogeneous case. The instantaneous EPI energy remains unchanged.
Moreover, ZHIR retains stored pressure: the ensuing held-pressure interval
does not automatically use that refreshed-law increment. Keep actual flow
provenance separate from the detached fresh-source comparison.
Do not replace the phase source by its linear comparison
outside an admitted chart or attribute earlier glyphs' changes to ZHIR.
The observed secant, current nodal product, operator jump and subsequent
relaxation remain separate evidence. This bounded continuation does not
authorize an open-ended search for a favorable trace.

A reachable configured mechanism does not derive the Si policy or prove
repeated renewal of a coherent pattern. The formation objective still
requires useful retained source work against relaxation and its maintenance
under the same admitted execution.
This section adds no autonomous selection law, substrate-creation result,
particle identification or new active research branch.

## 36. Native Mutation and coordination in the Dirichlet source budget

The native occurrence in section 35 is now examined through the existing
pressure and energy owners, using the same three-step heterogeneous case
and its uniform-capacity control. No additional initial condition, longer
run, substituted glyph or revised selector is introduced. The distinction
is between a phase event's effect on a prospective fresh nodal rate and
the work of the pressure actually retained by the following integrator.

### One detached channel ledger, with heterogeneous mobility retained

[Forcing realization](../../src/tnfr/physics/forcing_realization.py) owns
`ForcingDirichletBalance` and `observe_forcing_dirichlet_balance`. The
observer consumes an existing `NonEpiForcingObservation`, reuses its shared
channel decomposition and rebuilds the transport snapshot. It performs no
graph evolution, pressure recapture or new phase-kernel call. The primitive
full-pressure vector remains detached data; cached gradient, energy and
pressure-defect fields are recomputed rather than trusted as evidence.

For fixed symmetric nonnegative conductance, write `B=D-W`, `x=EPI`,
`E_D=x^T*B*x/2`, `nu` for the capacity vector and `e` for the EPI pressure
weight. The existing EPI gradient is `g_epi=-D^(-1)*B*x`, with zero rows
at zero strength. Let `F` be the sum of the captured phase, capacity and
topology pressure channels. Define

```text
p_model  = e*g_epi + F
eps_k    = p_fresh - p_model
eps_s    = p_stored - p_fresh
score    = diag(nu)*B*x
R_epi    = score dot (e*g_epi)
R_source = score dot F
R_model  = R_epi + R_source
R_fresh  = R_model + score dot eps_k
R_stored = R_fresh + score dot eps_s.
```

The three source-channel works are reported separately and retain their
signs. The final rate agrees exactly with the rebuilt snapshot's stored
nodal energy rate. These are rational identities for represented input
values, not claims of exact transcendental phase evaluation, authenticated
history or universal runtime stability. The captured kernel discrepancy
is relative to the represented channel reference, not an error enclosure
against an exact-real phasor model. Zero capacity and zero-strength rows
retain their existing degenerate-mobility convention.

The EPI term satisfies

```text
R_epi = -D_loss,
D_loss = e*sum_{d_i>0}(nu_i/d_i)*(B*x)_i^2 >= 0.
```

Thus instantaneous nondecrease for the prospective refreshed nodal row
requires `R_source + score dot eps_k >= D_loss`. A positive change in
phase-source work alone is insufficient; capacity-source work, diffusion
and the declared numerical reference matter too. This criterion concerns
Dirichlet form contrast. It does not identify that contrast with coherent
identity, tetrad energy, a particle or a maintained NFR.

At fixed EPI, capacity and support, a phase-only ZHIR event has zero
instantaneous EPI-energy jump. It can nevertheless change the fresh phase
pressure at its target and at neighbors whose resultants include that
target. The channel ledger retains those nonlocal contributions instead
of inferring source work from the target's phase displacement alone.
Stored pressure remains unchanged by ZHIR, so its held nodal rate does
not automatically become the newly observed fresh rate.

### Reuse the finite Euler endpoint identity

The existing
[`observe_support_transport_euler`](../../src/tnfr/physics/support_transport.py)
owns the finite energy accounting. With fixed conductance and capacity,
held `r=diag(nu)*p_stored`, interval `h`, exact reference `y=x+h*r` and
observed endpoint defect `delta=x_after-y`, it reports

```text
E_D(x_after)-E_D(x)
  = h*R_stored + h^2*r^T*B*r/2
    + (B*y) dot delta + delta^T*B*delta/2.
```

The linear drift, nonnegative Euler quadratic contribution and signed
endpoint-defect contribution remain separate. The last term can contain
rounding, clipping or other deviations from the declared reference. The
algebra alone does not identify their origin. A distinct clipping budget
would require an observed raw pre-clip endpoint; no such intermediate
endpoint is inferred from the final value. Likewise, an instantaneous
rate sampled at an endpoint is not an integral of refreshed rates along
the actual held-pressure interval.

The quadratic contribution is also present under exact continuation with
constant held rate; it is not automatically a numerical error. A
nonpositive initial energy rate therefore does not exclude positive finite
energy change. At consensus, for example, the initial rate is zero while
a nonuniform held forcing can generate positive energy quadratically.
The instantaneous source criterion above must not replace the integrated
or finite endpoint budget when assessing maintenance.

### What the retained third step shows

The enhanced
[native admission controls](../../tests/physics/test_heterogeneous_mutation_admission.py)
retain the actual target event, integrator and coordinator boundaries from
section 35. The third heterogeneous batch applies
`(ZHIR,OZ,IL,IL,IL,IL)` in node order `(0,1,2,3,4,9)`. Thus the pressure
at flow start also includes the later nodes' pressure writes; it is not
attributed solely to the target Mutation. Approximate global rates are:

| Boundary | EPI term | Phase work | Capacity work | Stored-residual work | Fresh rate |
| --- | --- | --- | --- | --- | --- |
| Before target ZHIR | `-1.939783470` | `+2.456864795` | `-0.207571077` | `0` | `+0.309510247` |
| After target ZHIR | `-1.939783470` | `+2.107507269` | `-0.207571077` | `+0.349357525` | `-0.039847278` |
| Subsequent flow start | `-1.939783470` | `+2.107507269` | `-0.207571077` | `+0.326643961` | `-0.039847278` |
| Before coordinator | `-1.974221035` | `+2.138489438` | `-0.192984463` | `+0.340073217` | `-0.028716059` |
| After coordinator | `-1.974221035` | `+1.660508618` | `-0.192984463` | `+0.818054037` | `-0.506696879` |

Topology work is zero at these boundaries. The retained kernel-rate
defects have magnitude below `6e-16`; the table's decimals are rounded
displays, while each underlying rational ledger has zero identity residual.

The target ZHIR write preserves EPI, capacity, conductance and stored
pressure. Its observed phase alignment therefore changes the prospective
fresh rate in the dissipative direction while leaving instantaneous energy
and the stored rate unchanged. The following coordinator also preserves
EPI energy at its phase event and makes its own prospective fresh rate
more negative. These comparisons use each event's actual EPI endpoint;
the intervening integration changes `B*x`, so their work increments must
not be added as if they used one unchanged score.

The heterogeneous run's actual EPI-energy increase is accounted for by
its positive retained-pressure drift, Euler quadratic contribution and
endpoint defect. At flow start, `R_stored` is approximately `+0.286796684`.
The finite third-flow accounting is:

| Contribution | Heterogeneous case | Uniform-capacity control |
| --- | --- | --- |
| Held drift `h*R_stored` | `+0.14339834185` | `-0.12701644047` |
| Held-flow quadratic term | `+1.13459958299` | `+0.01902551467` |
| Combined endpoint-defect term | `-1.16456266791` | approximately `-1.38e-15` |
| Observed energy change | `+0.11343525693` | `-0.10799092580` |

Both exact endpoint identities have zero residual. The heterogeneous
endpoint-defect contribution is substantial; without an actual raw
pre-clip capture it is not assigned entirely to clipping or to rounding.
The observed increase is not evidence that the aligning Mutation or
coordinator supplied a positive fresh-rate increment. The uniform control,
which applies IL at every node in all three steps, instead loses EPI energy
over its corresponding interval. This is a finite
comparison of the supplied capacity configurations and their resulting
trajectories, not a universal claim that heterogeneous capacities increase
energy or that homogeneous capacities always decrease it.

[Detached ledger controls](../../tests/physics/test_forcing_dirichlet_balance.py)
independently check heterogeneous mobility, signed channel contributions,
fresh-versus-stored pressure and reconstruction of altered caches. They
also retain the distinction between exact arithmetic over detached data
and evidence of a live kernel or causal execution.

### Interpretation boundary

This retained case demonstrates native Mutation occurrence, observed phase
alignment and a finite EPI increase under held pressure. The fresh-law and
executed-flow ledgers explain why these observations coexist. Neither
the energy increase nor the operator label establishes sustained renewal
of a coherent identity. The unresolved modeling distinction concerns how
pressure refresh and the interval law contribute to retained source work;
changing that law would be a declared model change, not an explanation
retroactively applied to this trace. The sole task queue remains the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md).

## 37. Separate the frozen hard-rail reference from its represented endpoint

Section 36 leaves a substantial combined endpoint defect in the same
heterogeneous third interval. A detached exact reference can separate the
energy effect of the configured hard rails from the remaining endpoint
difference without inventing an internal clipping trace. This section uses
that interval and its existing uniform-capacity control only.

### Frozen constant-velocity hard clipping has an exact semigroup

Let the common scalar rail be `[L,U]`, with `L<=U`, and define
`C(a)=min(U,max(L,a))` componentwise. Hold the velocity vector `v` fixed,
start with every `x_i` inside the rail, and put

```text
T_t(x) = C(x+t*v),       t>=0.
```

For a coordinate with `v_i>=0`, the lower rail is inactive and

```text
T_s(T_t(x))_i
  = min(U, min(U,x_i+t*v_i)+s*v_i)
  = min(U, x_i+(s+t)*v_i)
  = T_(s+t)(x)_i.
```

If the inner value has reached U, nonnegative `s*v_i` leaves it there;
otherwise both expressions apply the same upper cap to the same sum.
For `v_i<=0`, the corresponding lower-cap calculation is

```text
T_s(T_t(x))_i
  = max(L, max(L,x_i+t*v_i)+s*v_i)
  = max(L, x_i+(s+t)*v_i).
```

Zero velocity satisfies both cases. Thus `T_0` is the identity on the rail
and `T_s o T_t=T_(s+t)` for nonnegative times. By induction, exact repeated
hard clipping of this fixed velocity over any nonnegative partition gives
the same final reference `C(x+h*v)` for the same total duration h.

The in-rail initial state and fixed velocity are essential premises.
Refreshed pressure, changing capacity, forcing, operator jumps, different
rails or a different clipping map require separate laws. The identity is
not binary64 partition invariance: multiplication, addition and time
accumulation can round differently under different partitions, even when
the exact reference agrees. Native phase coordination and adaptation also
act per invocation and are outside this frozen reference.

The hard-rail reference is a constrained map with declared rails. At a
saturated outward-moving coordinate it suppresses further displacement;
it is not an unconstrained solution of `dx/dt=nu*p` there. Its algebra does
not derive the rails or remove their role from the execution provenance.

### Common hard rails cannot increase Dirichlet energy relative to raw input

The common scalar projection is 1-Lipschitz, so for any two coordinates
`abs(C(y_i)-C(y_j))<=abs(y_i-y_j)`. On symmetric nonnegative conductance,
each edge's squared separation therefore decreases or stays fixed. Summing
the weighted terms gives

```text
E_D(C(y)) <= E_D(y).
```

This compares the projected endpoint with its raw input y, not with the
initial state x. It does not imply `E_D(C(x+h*v))<=E_D(x)`: a nonuniform
held velocity can create contrast before projection. The common hard-rail
assumption must not be replaced by different per-node rails or a configured
soft clipping function.

### An exact reference split without an internal clipping claim

For the retained held-pressure interval, set

```text
v = diag(nu)*p_stored
y = x+h*v
z = C(y).
```

Treat the captured scalar state, capacity, pressure, duration and rail
values as exact rationals. With fixed conductance and the actual captured
endpoint `x_actual`, the energy telescope is

```text
E_D(x_actual)-E_D(x)
  = [E_D(y)-E_D(x)]
    + [E_D(z)-E_D(y)]
    + [E_D(x_actual)-E_D(z)].
```

The first bracket is the existing held-flow drift plus quadratic term.
The second is a nonpositive exact hard-rail reference correction. The last
is the signed remaining energy difference between the represented actual
endpoint and that projected reference. The corresponding state difference
`x_actual-z` remains available; a small energy difference alone would not
establish a small state difference.

The [support transport owner](../../src/tnfr/physics/support_transport.py)
exposes `SupportTransportClippedFlow` through
`observe_support_transport_clipped_flow(before, after, dt, lower=..., upper=...)`.
It reuses the existing Euler budget for this decomposition rather than
introducing another integrator or an inferred pressure law. The reference
y is not asserted to be a captured raw runtime array, and z is not asserted
to be an observed internal clipping output. Agreement with this reference
does not authenticate how many substeps or clipping calls occurred. Without
their trace, the last bracket is an endpoint difference, not an exclusively
rounding or exclusively clipping measurement.

### The same retained interval is close to its projected exact reference

The retained cases use the actual third interval `h=0.5` and captured
common hard rails `[-1,1]`. No new trajectory or pressure refresh is
performed. Approximate energies and signed contributions are:

| Quantity | Heterogeneous case | Uniform-capacity control |
| --- | --- | --- |
| Initial energy `E_D(x)` | `6.427820565521181` | `7.184031670285270` |
| Raw reference energy `E_D(y)` | `7.705818490365846` | `7.076040744486429` |
| Projected reference energy `E_D(z)` | `6.541255822452809` | `7.076040744486429` |
| Actual endpoint energy | `6.541255822452811` | `7.076040744486428` |
| Raw reference change | `+1.277997924844665` | `-0.107990925798841` |
| Hard-rail reference correction | `-1.164562667913037` | `0` |
| Actual-minus-projected-reference energy | approximately `+3.16e-15` | approximately `-1.38e-15` |
| Observed total energy change | `+0.113435256931631` | `-0.107990925798842` |

The exact rational telescopes have zero residual; these decimal displays
are rounded separately. In the heterogeneous reference only pendant node 9
crosses a rail: its raw value is approximately `-2.498611286112409` and
the projected value is `-1`, which also equals its observed endpoint.
Every other reference coordinate remains unprojected. The maximum absolute
actual-minus-projected-reference state difference is approximately
`4.16e-16`. In the uniform control no reference coordinate is projected,
and the maximum absolute state difference is approximately `3.51e-16`.

Thus the heterogeneous case's large combined Euler defect from section 36
is resolved, at the endpoint level, into a substantial nonpositive hard-rail
reference correction and a small remaining difference from that reference.
This does not turn the constructed y or z into observations of internal
runtime clipping. It also does not establish binary64 partition invariance.
The [reference controls](../../tests/physics/test_support_transport_clipped_flow.py)
and enhanced [retained native controls](../../tests/physics/test_heterogeneous_mutation_admission.py)
exercise the same shared observer; the existing
[runtime partition boundary](../../tests/physics/test_runtime_flow_refinement.py)
retains the distinction between exact composition and represented execution.

### Consequence for the maintenance claim

The heterogeneous projected exact reference still gains Dirichlet energy.
The positive observed change is therefore not explained away as the small
remaining endpoint discrepancy. It belongs to this configured bounded
held-pressure evolution, whose hard projection removes part of a larger
raw increase. Its source still includes supplied heterogeneous capacity,
retained operator pressure and the declared clock and boundary policy.

Sections 35 and 36 separately show that the actual Mutation and coordinator
align phases and reduce their prospective fresh-rate work in this case.
The surviving finite form-energy increase does not contradict those results
and does not establish a restoring law, persistent differentiated identity
or autonomous NFR maintenance. No internal clipping history, new feedback
law, physical interpretation or additional research queue follows from
this endpoint decomposition.

## 38. Default capacity writers cannot restore an inward-contracted profile

The source and interval audits above establish finite native behavior;
they do not explain regeneration of the capacity contrast that supports
a differentiated profile. This section examines that missing restoration
condition directly. It uses the admitted initialized default writer class
from sections 34 and 35, without extending the prior native prefix or
introducing a new capacity law.

### Writer and timing premises

At a native decision, require freshly computed default Si and the ordinary
default selector, with its injected thresholds and operator factors. Use
initialized scalar nodes, so IL is admitted as the first grammar fallback.
Retain the complete node set and exclude custom selectors, operator hooks,
callbacks or other unaccounted capacity writers. Capacities are nonnegative
and initially inside the fixed common configured rails. The maximum M is
positive. These are execution premises, not properties inferred from a
label saying that a run is canonical.

The selection snapshot precedes the sequential primitive batch. Its base
choices are `{IL,OZ,ZHIR,NAV,RA}`; lag overrides add AL and EN. In this
initialized class a rejected candidate does not reach a later
capacity-changing fallback because IL is already admitted. The actual
IL, OZ, ZHIR, NAV, AL and EN primitives preserve capacity. RA writes only
its own target's capacity. Consequently, before a target's own RA write,
its capacity is still the value used for its selection, even if earlier
targets have already changed their own states. Public operator classes
and deliberately supplied words are different execution paths.

The built-in nodal integrator and phase coordinator preserve capacity.
The capacity adaptation owner uses a held neighbor snapshot, updating
only its eligible subset. Its pressure, Si and counter gates need not be
fresh for the interval argument below: any eligible subset has the same
convex-range property. The protected optional network REMESH operation
does not supply a hidden capacity update. The relevant source owners are
[selection](../../src/tnfr/dynamics/selectors.py),
[primitive operators](../../src/tnfr/operators/__init__.py),
[runtime ordering](../../src/tnfr/dynamics/runtime.py) and
[adaptation](../../src/tnfr/dynamics/adaptation.py).

### Exact interval invariance under those writers

Let `m=min_i(nu_i)` and `M=max_i(nu_i)` at the selection snapshot. Write
`a` for the default normalized Si capacity weight and `hi=0.5` for its
upper selection threshold. The other Si contributions are nonnegative,
so the ideal fresh definition gives `Si_i>=a*nu_i/M`. An ordinary RA
choice requires `Si_i<hi`; hence

```text
nu_i < (hi/a)*M.
```

The optional default RA amplification multiplies the target by
`gamma=1+b`, where the effective default boost b has nominal recipe
`1/(8*pi)`. The effective coefficients satisfy

```text
gamma*hi/a < 1       (approximately 0.68538189),
```

Thus even an amplified RA target remains below the selection snapshot's M.
The multiplier is at least one, so it cannot cross below m. An unamplified
target is unchanged. All other reachable primitives preserve capacity;
therefore the entire admitted sequential batch remains in `[m,M]`.
This argument uses the selection snapshot bound, not the false claim
that RA is itself a convex neighbor average.

For an eligible adaptation target, the ideal update is

```text
nu_i^+ = (1-eta)*nu_i + eta*mean(nu_neighbors),       0<=eta<=1.
```

Every neighbor value and the target lie in the current interval, so both
the mean and the convex update do too. An isolate uses its own value;
an ineligible target is unchanged. The common rails are inactive on that
already in-rail interval. Thus one admitted exact-model native step obeys

```text
min(nu_after) >= min(nu_before),
max(nu_after) <= max(nu_before).
```

Induction gives nested capacity intervals for every finite continuation
that satisfies these premises. No common semicircle, phase lock, fixed
pressure, fixed eligible subset or convergence to consensus is required.
In particular, this argument must not import the phase-chart observer's
phase representation, connected-graph or resource restrictions.

### An arbitrarily small inward perturbation prevents profile recovery

Declare a nonuniform target capacity profile `nu_star`, with extrema
`0<=m<M`, and arithmetic mean `mu` strictly between them. For any
`0<epsilon<=1`, consider the nearby inward perturbation

```text
nu_epsilon = (1-epsilon)*nu_star + epsilon*mu*1,
l = (1-epsilon)*m + epsilon*mu,
u = (1-epsilon)*M + epsilon*mu.
```

Then `m<l<=u<M`. Every future capacity profile w admitted by the nested
interval result lies in `[l,u]^n`. At target nodes attaining m and M,
comparison with the original profile immediately gives

```text
norm_inf(w-nu_star) >= max(l-m, M-u) > 0.
```

Removing a common positive capacity scale does not remove the obstruction.
For any `c>0`, the same two coordinates give

```text
norm_inf(w-c*nu_star) >= max(l-c*m, c*M-u)
                      >= (M*l-m*u)/(M+m)
                       = epsilon*mu*(M-m)/(M+m) > 0.
```

The second inequality follows by averaging the two affine lower bounds
with weights `M/(M+m)` and `m/(M+m)`, which cancels c. The bound is sharp
over this full interval box: use `c=(l+u)/(M+m)` and project `c*nu_star`
onto `[l,u]` coordinatewise. This mathematical construction proves the
bound's sharpness; it is not a proposed engine operation.

For positive m, the equivalent ratio statement is `u/l<M/m`; the
quantitative bound above also handles m equal to zero without dividing by
it. The argument uses extrema and is unchanged by a permitted relabeling
of the target nodes. It does not assume that the original target was a
fixed point: every future state in the interval box remains a positive
distance from its original positive-scaling orbit.

Because epsilon can be arbitrarily small, this rules out local asymptotic
restoration of a nonuniform capacity profile, including recovery modulo
common positive scale, within this admitted exact writer class whenever
such capacity perturbations belong to the restoration claim. It does not
rule out transient organization, a family with different capacity profiles,
or restoration of an identity definition that does not retain this profile.

### The adaptation implementation must preserve its declared convex range

Exact convexity cannot simply be attributed to rounded arithmetic. With
default `eta=0.1`, target capacity `0.104` and a singleton neighbor equal
to the immediately preceding binary64 value, the former two-product blend
can return the immediately following value above `0.104`. It therefore
exceeds both inputs. The equal-input guard from section 34 does not cover
this nonuniform case.

The shared adaptation proposal now bounds its represented stable mean by
the actual neighbor minimum and maximum, and bounds the represented blend
by the target/mean interval. These comparison guards retain the existing
arithmetic result when it is already inside the declared interval. They
change neither eligibility, counters, factors nor the intended exact
convex map. They enforce the adaptation range in represented arithmetic
without treating numerical overshoot as a structural source.

That implementation guarantee concerns adaptation. The full interval
theorem still has its stated exact-model selector and RA hypotheses;
binary64 Si normalization, threshold comparisons and multiplication require
their own evidence before upgrading it to a represented runtime theorem
for this class. No assumed future rounding margin supplies that evidence.

### Reuse and scope of the obstruction

The conditional algebra belongs with
[capacity feedback](../../src/tnfr/physics/capacity_feedback.py), using the
shared exact scalar/vector readers. `derive_capacity_interval_recovery_bound`
returns a `CapacityIntervalRecoveryBound` containing the original and
perturbed intervals, their ratios, both separation bounds and a minimizing
common scale. Its executable domain is nonempty strictly positive capacity
vectors and `contraction` in `(0,1]`; a uniform vector returns zero
separation, not a false nonrecovery claim. The zero-minimum extension of
the proof above is not an additional admitted evaluator domain.
Its existing P2 Coupling envelope
and separate binary64 lattice theorem concern different supplied maps;
neither authenticates the present selector, grammar or runtime class.
The new interval separation likewise remains conditional on the writer
admission established above, rather than a seal on arbitrary executions.

[Recovery-bound controls](../../tests/physics/test_capacity_recovery.py)
exercise the exact algebra and its supported domain;
[native writer controls](../../tests/physics/test_native_capacity_envelope.py)
check the declared selector and primitive paths. The
[adaptation regressions](../../tests/test_structural_stability_adaptation.py)
retain the adjacent-value boundary independently of the conditional
recovery theorem. These checks supply finite implementation evidence,
not admission of unobserved future choices.

The native control uses one default step on unit path4, seed 17, constant
EPI `0.125`, zero phases, capacities `(0.3,0.3,0.3,1)` and empty initial
histories. Its actual word is `(RA,RA,NAV,IL)`; the first two capacities
become approximately `0.31193662073189216`, while the global interval stays
`[0.3,1]`. All four decisions precede the first write. Nodes 1 and 2,
initially equal in capacity, become unequal: the interval theorem does
not assert that every local pairwise contrast decreases.

This result identifies a concrete missing regeneration property for the
default capacity-supported route. It is not a theorem about all thirteen
operators: explicitly chosen capacity-expanding operators, changed
selection laws, new nodes or other capacity writers lie outside its
premises. It does not identify capacity with every possible scalar-form,
phase-sector or support identity, prove physical emergence, or prescribe
a replacement feedback law. The investigation's task queue remains owned
by the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md).

## 39. The same stationary reduced identity constrains capacity differences

Section 38 obstructs recovery of an original capacity profile, but an
identity definition may deliberately discard capacity. The narrower
question here is whether altered capacities can still realize the same
EPI shape and phase geometry with zero fresh pressure. This is an algebraic
stationary-realizability gate, not a theorem about arbitrary convergence,
form derivatives or complete-runtime persistence.

### Fix the reduced identity and the pressure realization

Compare two states on the same connected reciprocal unique-neighbor
support, with the same transport conductances and effective pressure
coefficients. Let their EPI values differ by one common additive constant
and their exact phase configurations by one common rotation. Require a
well-defined phase source under the declared circular conventions. The
weighted EPI gradient and ideal phase source then agree; the fixed topology
channel agrees as well. Let `b>0` be the active capacity-pressure coefficient.

Write the unweighted capacity-support operator as

```text
(G*q)_i = mean(q_neighbors)-q_i.
```

This G uses unique support neighbors, including edges with zero transport
conductance. It is generally different from the weighted EPI operator
`-L_rw`. With `delta_nu=nu_after-nu_before`, the ideal fresh-pressure
difference between the two states is exactly

```text
delta_p = b*G*delta_nu.
```

All capacities must be positive to identify zero unforced nodal velocity
with zero pressure. Gamma is absent or zero, and neither clipping nor an
operator jump is being used to keep EPI at an endpoint. Under those
premises, if both states realize zero fresh pressure, `G*delta_nu=0`.

On connected reciprocal support, the kernel of G consists precisely of
constant vectors. At a maximum of a harmonic vector, its neighbor average
can equal that maximum only if every neighbor has the same value; connectivity
propagates this equality. Conversely, a constant vector has zero neighbor
difference. Therefore the necessary and sufficient capacity condition for
preserving the original zero-pressure realization, with the other declared
identity data fixed, is

```text
nu_after = nu_before + c*1,
```

for a common additive capacity offset c that preserves positive capacities.
This is not common capacity scaling: adding a constant preserves capacity
neighbor differences, whereas scaling a nonuniform profile generally does
not. Unequal transport weights do not alter this conclusion because G
belongs to the capacity channel's support convention.

### The capacity interval obstruction survives this reduced identity

Let a nonuniform original capacity profile have extrema m and M, and
contract it toward its arithmetic mean as in section 38. The perturbed
capacity difference is `epsilon*(mu*1-nu_star)`. Since G annihilates the
constant mean, its prospective ideal pressure change at the original
form/phase identity is

```text
delta_p = -epsilon*b*G*nu_star = -epsilon*F_vf_star.
```

The perturbed interval `[l,u]` has width `(1-epsilon)*(M-m)`.
Conditional on subsequent
capacities w remaining in that interval, for every real additive offset c,

```text
norm_inf(w-nu_star-c*1)
  >= max(l-m-c, M+c-u)
  >= [M-m-(u-l)]/2
   = epsilon*(M-m)/2 > 0.
```

The second inequality averages the two extreme-coordinate bounds. It
therefore holds uniformly over all future profiles in the interval box,
not merely for the initial contracted profile. The unconstrained bound is
sharp at `c=(l+u-m-M)/2` with `w` obtained by projecting `nu_star+c*1`
into `[l,u]`. That minimizing offset can make some shifted target
capacities negative; no sharpness claim is made after imposing positive
capacity on the shifted target. The positive lower bound remains valid
for every model-admitted positive shift because that is a smaller
set of comparisons.

The existing
[`CapacityIntervalRecoveryBound`](../../src/tnfr/physics/capacity_feedback.py)
now also reports `additive_separation` and `minimizing_offset`. These
quantities reuse the same declared perturbation and interval geometry;
they introduce no evolution law or additional run.

Combining the connected-support identity with this interval separation
excludes an exact zero-pressure stationary realization of the original
EPI shape and phase geometry at every subsequent state satisfying the
interval and fixed-model premises. It does not establish a positive lower
bound on a trajectory's pressure, prove nonconvergence of its EPI or phase,
or rule out a different identity. Such conclusions would need additional
compactness, continuity, admission and dynamical hypotheses.

### Retain phase realization and kernel differences in represented captures

[Forcing realization](../../src/tnfr/physics/forcing_realization.py) supplies
`ForcingCapacityDifference` through `observe_forcing_capacity_difference`.
The detached comparison reuses the validated channel decomposition and
rebuilt transport snapshots. It requires matching support, conductance
and coefficients, positive capacities, an exact common EPI offset and
an exact common raw phase offset in the captured coordinates.

The mathematical theorem allows a common circular rotation, whereas this
observer's raw-offset admission is a narrower represented-coordinate
condition. Independently rewrapping or rounding phases must not be treated
as proof of that condition. Even an admitted represented common offset can
change the materialized phase-kernel coefficients slightly; their observed
difference remains explicit rather than being declared zero.

Consequently, the represented pressure ledger separates

```text
fresh_pressure_change
  = capacity_pressure_change
    + phase_realization_change
    + kernel_assembly_defect_change.
```

The stored-pressure difference additionally retains the change in stored
versus fresh residual. Neither a supplied capture nor an exact arithmetic
identity authenticates a kernel call, a stationary baseline or a causal
trajectory. The observer does not infer pressure from an observed EPI
derivative or declare a numerical near-zero value to be exact stationarity.

In particular, the general nodal-rate difference is

```text
delta_rate = diag(nu_after)*delta_p
             + diag(delta_nu)*p_before.
```

The second term vanishes only for a zero-pressure baseline. Actual captured
baseline residuals must remain in that identity; a model's exact stationary
construction does not erase them.

### A prospective stationary construction and its capacity controls

The [stationary identity controls](../../tests/physics/test_stationary_capacity_identity.py)
use a declared three-node path with transport weights 1 and 2, injected
default pressure coefficients, seed 17 and the default NumPy kernel. Before
any kernel observation, prescribe

```text
phase = (0, pi/8, 0),
capacity = (1/2, 1, 1/2),
x_i = 3/4 - (w_phase/w_epi)*(phase_i/pi)
            - (w_vf/w_epi)*capacity_i.
```

At this particular target the two endpoints have equal phase and capacity,
so the weighted EPI and unweighted capacity rows share the relevant
three-coordinate pattern. This special coincidence does not make the
capacity channel conductance-weighted in general. The phase source is
`(1/8,-1/8,1/8)` and the transport strengths `(1,3,2)` give the required
zero weighted sum. The prescribed exact-real target therefore has zero
pressure; no measured pressure is fitted to define its EPI.

Its materialized EPI is approximately
`(0.5908450569081046,-0.08600896788251489,0.5908450569081046)`.
The shared signed phase-pressure realization captures approximately
`(0.12500000000000003,-0.125,0.12500000000000003)`: singleton phasor rows and
the certified two-neighbor midpoint round separately. Their exact represented
degree-weighted sum is `3/2^55`, rather than the ideal zero. Baseline fresh
pressure is approximately `(+2.43e-17,+3.47e-18,+2.43e-17)`. The separately
retained phase-realization, modeled-pressure and kernel residuals account for
this nonzero result. Supplied stored zeros are initialization, not evidence
of an exact runtime stationary state or mean conservation.

Half contraction toward the capacity mean gives
`(7/12,5/6,7/12)` while holding materialized EPI and phase fixed. The shared
observer finds

```text
capacity_pressure_change = w_vf*(-1/4,+1/4,-1/4),
fresh_pressure_change = capacity_pressure_change,
```

with zero phase-realization and kernel-defect changes in this comparison.
Each nonzero component has magnitude approximately `0.014574888647966173`.
The final captured pressure also retains the small baseline residual;
it is not replaced by that difference vector. Stored pressures remain
their supplied zeros, exposing the fresh-versus-stored discrepancy rather
than concealing the loss of stationary realizability.

The complementary controls retain the theorem's boundaries:

| Declared change | Observed pressure-difference consequence |
| --- | --- |
| Add `1/4` to every capacity on the connected path | Exact fresh-pressure change is zero |
| Raise only node 0 capacity by `1/4` | Capacity contribution is `w_vf*(-1/4,1/8,0)`; its middle entry is not the transport-weighted `w_vf/12` |
| Shift capacities independently on two disconnected edges and an isolate | Component offsets `1/4`, `1/2`, `3/4` leave fresh pressure unchanged |
| Set the capacity-channel weight explicitly to zero, then contract capacities | Fresh pressure is unchanged |

The asymmetric control retains its small nonzero kernel-defect change.
The rate comparison also keeps `diag(delta_nu)*p_before` rather than
treating the numerical baseline as exactly zero. These are detached
production-kernel observations at prospectively specified inputs, not an
operator schedule, a trajectory, a perturbation-recovery experiment or an
infinite-time stability claim.

[Difference-observer controls](../../tests/physics/test_forcing_capacity_difference.py)
check the shared arithmetic, admission and represented phase/assembly
contributions; [recovery-bound controls](../../tests/physics/test_capacity_recovery.py)
check the additive separation without assuming that its minimizing shifted
target stays positive.

### Limits of this stationary gate

Disconnected support allows an independent constant capacity shift on
each connected component. A zero capacity-pressure coefficient makes this
channel insensitive to capacity differences. Changed phase geometry,
transport, topology or coefficients can compensate a changed capacity
source and are outside the fixed-identity comparison.

Positive capacities, no Gamma and no clipping are also substantive
conditions. Zero capacity can suppress a nonzero pressure; forcing can
cancel it; a rail can hold an endpoint under outward pressure. None is a
zero-pressure stationary realization of the declared unforced model.
Likewise, a common EPI drift may preserve differences while using nonzero
pressure, and moving or periodic patterns require their own treatment.
The present result addresses stationary pressure realizability only and
does not introduce a second task queue or a replacement constitutive law.

## 40. Static phase compensation and the native direction obstruction

Section 39 fixes both form and phase geometry. Here the phase is allowed
to change while the original EPI shape is retained. The same prospectively
declared weighted P3 supplies the comparison; no new oscillator, selected
target after observation or extended trajectory is introduced. Static
pressure compatibility, the actual native writer and complete-runtime
restoration remain different questions.

### A different phase geometry can compensate the lost capacity source

Write `a=w_phase>0`, `e=w_epi>0` and `b=w_vf>0` for the effective pressure
coefficients. In the P3 family, both endpoints have the same EPI X, phase
and capacity; the middle has EPI Y, a relative phase gap delta and a
capacity excess d. Transport edge weights remain 1 and 2. Because the
endpoint values agree, the weighted EPI row and the unweighted support
rows yield

```text
p = [e*(Y-X)+a*delta/pi+b*d]*(1,-1,1).
```

The topology contribution is zero because the injected topology weight
is zero, not because this irregular support has zero topology gradient.
The original construction has `delta_0=pi/8`, `d=1/2` and zero ideal
pressure. Inward capacity contraction reduces d to `(1-epsilon)*d`.
Keeping the same EPI shape therefore requires

```text
delta_required = delta_0 + pi*(b/a)*epsilon*d.
```

This is a positive phase-gap increase. The effective default ratio
`b/a<1/12` keeps `delta_required<pi/6` for `0<epsilon<=1`, so the required
state remains in a regular acute chart. Static feasibility is not blocked
by a phase-wrap singularity. For the retained half contraction,
`delta_required` is approximately `0.45306233345003016`, compared with
the original gap `0.39269908169872414`.

This construction preserves EPI shape through a different phase geometry;
it does not preserve the full form-and-phase identity of section 39.
Prescribing that compensating phase is initialization of a compatible
state, not evidence that the engine generates its restoring change.

### The native phase update moves in the opposite direction

In this common chart, let g be the shared global phasor target. Each
endpoint's local target is the middle phase; the middle's local target
is the common endpoint phase. The ideal formula implemented by the
[native coordinator](../../src/tnfr/dynamics/coordination.py) therefore gives

```text
delta_after = (1-kG-2*kL)*delta_before.
```

The global target cancels from this difference. Unequal transport
conductances do not change the phase-neighbor convention. The injected
default adaptive gain bounds from section 34 imply
`0<1-kG-2*kL<1`, independently of which admitted adaptive branch supplies
the gains. Thus this writer decreases the positive gap, whereas lost
capacity support requires an increase.

At fixed EPI and capacity, a gap contraction by factor rho contributes

```text
delta_p_phase = -a*(1-rho)*delta_before/pi*(1,-1,1).
```

Starting from the uncompensated capacity contraction, this adds to the
existing residual `-b*epsilon*d*(1,-1,1)`. Starting instead from the
statically compensated ideal zero-pressure state, the same native phase
write produces a nonzero residual. The compatible stationary surface is
therefore not invariant under this phase map.

The default selector does not supply a compensating Mutation in either
retained segment. After half contraction, `min(nu)/max(nu)=0.7`. With
alpha denoting the Si capacity coefficient, fresh default Si is bounded
below by `0.7*alpha>0.5`; every base choice is IL. Initial lag counters
remain below their forcing thresholds, and the nonzero EPI values admit
IL in the incremental grammar. This argument concerns the actual default
selection path, not a deliberately supplied operator word.

### Two isolated native segments confirm the direction

The finite controls reuse the materialized EPI and capacities from section
39. One starts with the original phase gap; the other initializes the
analytically prescribed compensating gap before any observation. Each
segment refreshes pressure and Si, executes the actual default glyph batch
and invokes the native coordinator. No nodal integration, capacity
adaptation, full runtime step or temporal-growth history is performed.

Both batches request and apply IL at all three nodes, pass grammar and
leave the lag counters at one. The coordinator selects its stable adaptive
branch with represented gains
`kG=0.04461342911494398`, `kL=0.140157221158957`, giving
`rho=0.675072128567142`. The retained observations are:

| Initial phase choice | Gap before coordination | Gap after coordination | Maximum fresh pressure magnitude before | Maximum fresh pressure magnitude after |
| --- | --- | --- | --- | --- |
| Original, uncompensated gap | `0.39269908169872414` | `0.2651002049687197` | approximately `0.014574888647966177` | approximately `0.04538402112491463` |
| Prescribed static compensation | `0.45306233345003016` | `0.3058497538157081` | approximately `3.99e-17` | approximately `0.03554492002170308` |

Pressure magnitudes are displayed; after coordination the endpoint
pressures are negative and the middle pressure positive. The statically
compensated capture retains its small nonzero baseline rather than claiming
an exact represented equilibrium. Its pressure increase is not explained
by a large initialization residual.

EPI, capacity and support remain unchanged in both segments. IL contracts
stored pressure, and the coordinator leaves that held pressure unchanged.
The fresh pressures in the table are detached read-outs; they are not
substituted into an executed physical interval. Neither segment is evidence
of complete-runtime recovery or failure of every future trajectory.

### An exact invariant source class closes this native P3 route

No further run is needed to combine the earlier conditional results for
the uncompensated half-contraction case. Retain the initialized, fresh-Si,
default writer class of sections 34 and 38, this fixed support and these
coefficients, with no additional phase/capacity writers. The capacity
interval stays within `[l,u]=[7/12,5/6]`. Its minimum-to-maximum ratio cannot
fall below 0.7, so every future fresh base selection remains IL. Lag AL/EN
and admitted fallback IL also preserve phase and capacity. Adaptation may
change any eligible subset without leaving the interval.

The equal endpoint phases remain equal under the coordinator, and its
positive gap factor keeps `0<=delta_t<=delta_0`. These phase facts do not
require intermediate EPI to remain at its original value. At any subsequent
state whose EPI does equal the original shape modulo a common offset,
capacity differences satisfy

```text
nu_middle-nu_endpoint <= u-l = (1-epsilon)*d,
mean(nu_endpoints)-nu_middle >= -(u-l).
```

Consequently its ideal fresh pressure necessarily obeys

```text
p_endpoint <= -b*epsilon*d < 0,
p_middle   >= +b*epsilon*d > 0.
```

This is a uniform exclusion of zero-pressure stationary realizability of
that original EPI shape within the declared exact source class, including
unequal future endpoint capacities. It is conditional on the stated
writer, chart, source and capacity-envelope premises. It is not an
unrestricted binary64 continuation theorem, a statement about every actual
held pressure, or a proof that hybrid EPI jumps cannot maintain another
kind of pattern.

The separately initialized compensated gap is larger than `delta_0` and
lies outside this source-box hypothesis. Its one-invocation failure above
is retained separately; it is not retroactively included in that invariant
class. Together the results distinguish an available static compensation
from its absence as a restoring native phase action in the audited route.
They supply no replacement phase law or additional execution queue.
