# Coherent pattern contact and formation boundaries

This note owns the conditional two-ring contact result. It reuses the
[supplied phase law](nodal/FORCED_PHASE_LOCKING.md#29-graph-independent-locking-and-phase-source-constraints),
[cycle-period geometry](nodal/FORCED_PHASE_LOCKING.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods)
and [isolated-ring relaxation](nodal/FORCED_WINDING_AND_WRITERS.md#32-acute-cycle-relaxation-retains-phase-winding-while-form-relaxes).
The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns the next research task.

Its [formation boundary](#formation-boundary-under-the-same-gated-law)
separates this prepared survival from formation under the same phase law:
isolated gated C5 cannot enter an acute winding sector from winding zero.
The [one-neighbor extension](#one-external-phase-neighbor-budget-and-entrance-obstruction)
derives the actual contact budget and proves that this entrance obstruction
persists with one fixed external phase neighbor.
The [distributed interface](#distributed-two-port-contact-a-realizable-entrance-witness)
admits recovery from an existing winding. The
[gated potential](#winding-zero-formation-a-dissipation-and-preparation-boundary)
then constrains formation from winding zero, including an exact exclusion
for an initially uniform recipient.

**Model boundary.** These are primitive phase patterns under an explicitly
supplied averaged-sine law. They are not the derived form angles of the
[passive collective model](nodal/DERIVED_FORM_PHASE.md). The result concerns
prepared identity during interaction. It does not derive the phase law,
create support, establish spontaneous pattern formation or identify particles.

The later [relational region interaction](nodal/RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-region-interaction)
reuses this geometry and regional accounting with a different, fully coupled
phase/form law. This note's supplied sine evolution and numerical contact
results must not be transferred to that model.

The [phase-premise review](nodal/PRIMITIVE_PHASE_CLOSURE.md#phase-premise-review)
separates this supplied law from operator admission and derived form phase.
Changing the cutoff changes the preparation bounds; the exclusions below
cannot be transferred unchanged to another phase law. The remaining
gate-itinerary question is open and deferred in the sole execution plan.

## Model and prospective control

Prepare two unit undirected cycles, `A=(0,1,2,3,4)` and `B=(5,6,7,8,9)`.
For local index `k=0,...,4`, set

\[
q=2\pi/5,\qquad
\theta_{A,k}(0)=qk,\qquad
\theta_{B,k}(0)=qk+\beta,\qquad x_i(0)=1/2.
\]

Capacity is held at one. Add a reciprocal unit bridge `(0,5)` for
`0<=t<=T=1/2`, then remove it without changing the nodal state. All transport
weights and diagnostic lengths are one. The support schedule is a supplied
intervention, not a generated operator sequence. On the contact graph,
`d_0=d_5=3` and all other degrees are two.

Choose equal normalized EPI and phase pressure coefficients, with capacity
and topology coefficients zero. The latter must be selected explicitly:
the degree-three ports make the joined graph irregular. No Gamma, capacity
adaptation or glyph event is present. On the fully admitted acute domain,

\[
\dot\theta_i=1+\frac1{d_i}\sum_{j\sim i}\sin(\theta_j-\theta_i),
\qquad
\dot x=-\tfrac12L_{rw}x+\tfrac12g(\theta),
\]

where `L_rw=I-D^{-1}W` and

\[
g_i=\frac1\pi\operatorname{wrap}\left(
\operatorname{Arg}\sum_{j\sim i}e^{i\theta_j}-\theta_i\right).
\]

The executable phase law has a U3 gate of `pi/2`. The proof below ensures
that every edge remains strictly inside it; it does not replace the law
outside admission by an ungated continuation. The structural clock and the
identification of held capacity with free angular rate are supplied model
choices, not consequences of `dx/dt=nu_f*DeltaNFR`.

**Frozen before production execution, 2026-09-26:** use only
`beta=pi/24` and `beta=pi/12`, with `h=1/128`, 64 same-snapshot Euler steps,
and clipping rails `[0,1]`. Initial mathematical turns are `k/5` and
`k/5+1/48` or `k/5+1/24`. Record their binary64 materialization separately.
Use the existing `execute_joint_step` test adapter: NumPy pressure assembly,
the shared U3 phase proposal, scalar `DefaultIntegrator` EPI execution and
refreshed endpoint pressure. The scalar integrator path is selected locally
in the test; this does not relabel the vectorized pressure path.
There is no random preparation, fit, tolerance-selected identity or sweep.

Identity means each **original oriented cycle** retains winding one.
Phase deformation, EPI response, resultant availability and acute admission
are separate observations. Winding at isolated endpoints alone does not
prove path preservation. The control must retain consistent phase lifts and
check all eleven edges, including the bridge.

## A sufficient energy barrier and its limitation

In the co-rotating frame, the cosine phase energy satisfies

\[
V=\sum_{\{i,j\}}(1-\cos(\theta_j-\theta_i)),\qquad
\dot V=-\sum_i d_i(\dot\theta_i-1)^2\le0.
\]

Unequal port degrees change the mobility, not this identity. One acute
winding-one C5 has minimum `V_*=5(1-cos(q))`. At its first acute boundary,
the offending oriented gap must be `+pi/2`: a gap of `-pi/2`, with the other
four at most `pi/2`, cannot have total `2*pi`. Convexity of `1-cos` on the
acute interval gives

\[
V_{\partial}\ge1+4(1-\cos(3\pi/8)),\qquad
\Delta=V_{\partial}-V_*=5\cos(2\pi/5)-4\cos(3\pi/8).
\]

The other ring still costs at least `V_*`; a bridge acute boundary costs
one. Thus `1-cos(beta)<min(Delta,1)` is sufficient to prevent exit.
The low offset satisfies it and the high offset does not. For example,
elementary squared-rational bounds give `0.0142<Delta<0.01825`, whereas
`1-cos(pi/24)<0.00858` and `1-cos(pi/12)>0.03375`.
Failure of this sufficient test is **inconclusive**, not a prediction of
loss. The stronger comparison below decides both frozen cases without
changing their model or preparation after execution.

## A shared rotating reference protects both identities

The **same joined graph** has the exact reference

\[
\theta^*_{A,k}(t)=\theta^*_{B,k}(t)=t+qk.
\]

The two ring sine currents cancel at every node, including the ports;
the reference bridge current is zero. Use continuous lifts and set
`a_i=theta_i-theta_i^*`. Initially `a_A=0`, `a_B=beta`. Reference edge
representatives are `delta^*=+q,-q,0`. Their fixed integer branch shifts
do not change the sine. Subtraction gives

\[
\dot a_i=\frac1{d_i}\sum_{j\sim i}c_{ij}(a)(a_j-a_i),\qquad
c_{ij}(a)=\int_0^1\cos\bigl(\delta^*_{ij}+s(a_j-a_i)\bigr)\,ds.
\]

If `0<beta<pi/10`, every state in the box `0<=a_i<=beta` has

\[
0<\cos(q+\beta)\le c_{ij}\le1.
\]

At a maximal offset the derivative is nonpositive; at a minimal offset it
is nonnegative. The maximum principle preserves the box. Since
`q+beta<pi/2`, the box is strictly inside the acute domain and closes the
bootstrap for all times on this fixed contact graph. Smoothness there and
bounded rates provide continuation of the specified solution.

Both predeclared offsets are below `pi/10`. Every original oriented cycle
gap remains in `[q-beta,q+beta]`; the bridge gap stays in `[-beta,beta]`.
Their lifts therefore cannot encounter an antipodal branch crossing, and
each original winding remains one. This predicts persistence of a geometric
identity while allowing its shape to change. It is not a claim that arbitrary
preparations, capacities, interfaces or primitive laws have this property.

## Phase geometry bounds the form response

In the reference frame at a ring node, the two ring-neighbor phasors sum to

\[
2\cos\left(q+\frac{a_{\rm next}-a_{\rm prev}}2\right)
\exp\left(i\frac{a_{\rm next}+a_{\rm prev}}2\right).
\]

Its real amplitude is positive and its direction lies in `[0,beta]`.
At a port, adding the bridge phasor with direction in the same cone
preserves nonzero resultant and that direction bound. Consequently

\[
|g_i|\le\beta/\pi,\qquad
\|x(t)-\tfrac12\mathbf1\|_\infty\le\frac{t\beta}{2\pi}.
\]

The second inequality follows from positivity and maximum-norm contraction
of the EPI diffusion semigroup, or its extremum principle. At `T=1/2`,
the two radii are exactly `1/96` and `1/48`. Both envelopes are strictly
inside the frozen rails, so ideal clipping is inactive. Phase geometry
controls a genuine form response; it is not an independent sustaining input.

The interaction is already nonzero at the initial state. Only the two ports
have nonzero co-rotating phase rates, respectively `+sin(beta)/3` and its
negative. Their initial form rates are

\[
\dot x_0(0)=-\dot x_5(0)=
\frac1{2\pi}\operatorname{atan2}\bigl(
\sin\beta,\,2\cos q+\cos\beta\bigr)>0.
\]

Every other initial form rate is zero. This cancellation uses the declared
preparation symmetry; the general phase source is not transport-conservative.

## Discrete and represented execution have separate obligations

In exact arithmetic, phase Euler has the offset update

\[
a_i^+=\left(1-\frac h{d_i}\sum_jc_{ij}\right)a_i
       +\frac h{d_i}\sum_jc_{ij}a_j.
\]

For `h<=1`, its coefficients form a convex row. It preserves the same box,
and so does straight interpolation of the consistently lifted offsets.
For EPI Euler, `I-(h/2)L_rw` is stochastic for `h<=2`; the source estimate
gives the same finite form envelope. These are properties of the ideal
discrete method, separately from the continuous theorem.

An actual binary64 trace has materialization, phase proposal, pressure and
integration defects. Retain these separately. An initial offset error `r_0`
and per-step phase proposal error bounds `epsilon_k` would inflate the box
to `[-r_k,beta+r_k]`, with `r_k=r_0+sum epsilon_j`. The sufficient admission
condition becomes `q+beta+2*r_k<pi/2`, and the corresponding source bound is
`(beta+2*r_k)/pi`. A high-precision recomputation is a numerical residual
estimate, not automatically an outward-rounded interval certificate.

The finite control checks the actual lifted segment endpoints on consistent
branches; an affine gap between acute endpoints remains acute throughout
that segment. Small lifted nodal increments and explicit edge consistency
checks rule out an unobserved whole-turn jump in the chosen interpolation.
This establishes the retained discrete path property, not an error bound
between that interpolation and the continuous solution. Source assembly and
EPI Euler residuals reuse the existing forcing/transport observations.

### Finite production evidence

The [bounded control](../tests/physics/test_coherent_pattern_contact.py)
passes five contact tests, sharing the two frozen trajectories through one module
fixture. On both traces, both original windings remain `+1`, every edge
and each retained lifted segment stays acute, and every EPI proposal is
strictly inside the rails. Initial port responses agree with the independent
degree-three formulas above. Removal leaves the entire nodal state unchanged.

| Readout | `beta=pi/24` | `beta=pi/12` |
| --- | ---: | ---: |
| Predicted endpoint form radius | `1/96` | `1/48` |
| Observed maximum endpoint form deviation | `0.00444678019` | `0.00891375558` |
| Maximum endpoint phase deformation, relative to initial co-rotating phase | `0.0177181698` | `0.0352466810` |
| Minimum acute margin over all edges and stored states | `0.297115913` | `0.280229412` |
| Observed cosine-energy change | `-0.00382244056` | `-0.01512052456` |
| Endpoint regional form means | `(0.501148432, 0.498851568)` | `(0.502299569, 0.497700431)` |

Across the two controls the largest local lifted-phase residual estimate is
approximately `4.44e-16`; the largest accumulated phase-plus-initial estimate
is `2.23e-14`. The source residual estimate is at most `3.95e-17`.
The exact represented pressure-assembly and EPI Euler state defects have
displayed maxima `1.16e-18` and `5.55e-17`, respectively. Euler accounting
has zero identity residual. Tests use an explicit absolute comparison
tolerance `1e-12`, with no default relative-tolerance substitution. Numerical
energy comparisons are finite observations, separately from the exact ODE
dissipation identity; no complete-runtime stability conclusion follows.

Execution used Python 3.13.6, NumPy 2.5.3, NetworkX 3.6.1 and mpmath 1.3.0,
with 80-digit residual recomputation against the represented inputs/lifts.
The pressure path, scalar integrator, initialization and step schedule are
fixed above. The JUnit receipt records those versions and per-case metrics
from the same fixture, without additional trajectories:

```sh
python -m pytest tests/physics/test_coherent_pattern_contact.py -k "not formation_control and not contact_budget_control" -q --junitxml=artifacts/research/coherent_pattern_contact_2026_09_26.xml
```

The receipt is a local ignored artifact; this owner and the executable
control retain the result and reproduction recipe. This verification covers
the selected contact model and its five tests, not the full research inventory or
empirical physics.

## Removal and scope of recovery

Removing the bridge changes support and subsequent rates, without a jump
in EPI, capacity or phase. The following recovery statement concerns the
**unclipped continuous model**; the finite runtime control stops at removal
and does not certify all-time clipping inactivity afterwards. Each endpoint
is inside its ring's acute winding-one basin. The existing isolated-ring
result then applies: gaps relax to `q`, with each ring retaining its
own common phase offset, and EPI relaxes to that ring's post-contact mean.
Degree-two acute ring phase sources have zero sum, so that form mean is
conserved **after** removal. Contact does not generally preserve each
regional mean; recovery to the original EPI value `1/2` is not asserted.
No additional numerical long-time continuation is required by this control.

The result supplies conditional interaction, bounded deformation and recovery
of an already prepared identity. Formation from preparations without that
identity, independent justification of the primitive phase/capacity laws and
physical correspondence remain open. The energy-test failure in the second
case is resolved by shared nodal comparison, not by post-evaluation tuning.

## Formation boundary under the same gated law

Prepared contact survival does not imply that the same law can form the
identity on an isolated ring. In fact, the fixed unit C5 with common capacity,
unit coupling and the literal `pi/2` gate has a stronger entrance obstruction
than the existing common-semicircle theorem. This result concerns phase
geometry independently of the form row; pressure still reads full support.

### An exact obstruction to acute winding formation

Let `delta_i=wrap(theta_(i+1)-theta_i)` be principal oriented gaps on an
interval without antipodal edges. For winding one, `sum delta_i=2*pi`.
Write `b=pi/2` and define nonnegative gap excess and negative-gap mass

\[
E=\sum_i(\delta_i-b)_+,\qquad N=\sum_i(-\delta_i)_+.
\]

If `N>0`, at most four of the five gaps are positive. Hence

\[
2\pi+N=\sum_i(\delta_i)_+\le4b+E=2\pi+E,
\qquad N\le E.
\]

This count is specific to C5. It must not be generalized to every cycle,
network or TNFR model.

Define admission `a_i=1` when `abs(delta_i)<=b`, and zero otherwise.
The co-rotating phase rate is

\[
f_i=\frac{a_i\sin\delta_i-a_{i-1}\sin\delta_{i-1}}{a_{i-1}+a_i},
\]

with zero coupling rate when the denominator is zero. In particular, if
`delta_i>b`, the edge itself is excluded. Its endpoints have at most one
other admitted neighbor, so the **actual admitted-count normalization** gives

\[
\dot\delta_i=a_{i-1}\sin\delta_{i-1}+a_{i+1}\sin\delta_{i+1}.
\]

Each admitted negative gap appears at most twice when summing these rates
over positive-excess edges. Positive contributions help, and
`sin(delta)>=delta` for negative principal gaps. The positive-part chain
rule therefore gives, almost everywhere,

\[
\dot E\ge-2N\ge-2E,
\qquad E(t)\ge e^{-2(t-s)}E(s).
\]

This covers simultaneous gate crossings and the inclusive threshold at
equality. The statement applies to every absolutely continuous solution
satisfying the literal gated equation almost everywhere. It does not assert
global existence or uniqueness for a discontinuous vector field or silently
replace it by a differential inclusion.

Suppose there were an entirely acute winding-one endpoint. On the maximal
preceding antipodal-free interval, that endpoint has `E=0`; the estimate
forces `E=0` throughout the interval. Then `N=0` and all gaps belong to
`[0,b]`. This compact interval cannot approach an antipodal boundary, so
the interval extends to the initial time, which must already have winding
one. Reflection gives the winding-minus-one result.

**Conclusion:** a well-defined winding-zero preparation cannot enter, or
converge to, a strictly acute winding-one or winding-minus-one lock under
this fixed C5 law. Convergence to a strictly interior lock would imply
finite entrance. The result permits winding changes, fragmented states and
boundary accumulation; it does not exclude other supports, unequal
capacities, external interactions or different constitutive phase laws.
The isolated positive-formation search is closed under these premises.

### A separate exact-Euler boundary

For ideal simultaneous Euler with `0<h<1/2`, a step without an antipodal
wrap on a winding-one branch obeys

\[
E_{n+1}\ge\sum_{\delta_i>b}(\delta_i+h\dot\delta_i-b)
\ge(1-2h)E_n.
\]

If an edge instead wraps at an antipode, its endpoint magnitude is at least
`pi-2*h>pi/2`: the common free rate cancels and each coupling rate has
magnitude at most one. Such an endpoint cannot be fully acute. Thus this
step bound preserves the same no-entrance property for ideal Euler,
separately from the continuous theorem. Binary64 residuals require their
own accounting. Neither an excessively large step nor rounding-induced
admission establishes a continuous formation mechanism.

### A winding change with no coherent capture

The following family gives an explicit negative control beyond the common
semicircle, using `q=2*pi/5` and `eta` in `[pi/40,pi/20]`:

\[
\theta(0)=(0,q,4q+\eta,2q,3q),\qquad x(0)=\tfrac12\mathbf1.
\]

The initial winding is zero. Only edges `(0,1)` and `(3,4)` are phase
admitted; node 2 has no admitted phase neighbor. With

\[
d(t)=2\arctan\!\left(\tan(q/2)e^{-2t}\right),
\]

the exact co-rotating solution is

\[
\theta(t)-t\mathbf1=
(q/2-d/2,\ q/2+d/2,\ 4q+\eta,\ 5q/2-d/2,\ 5q/2+d/2).
\]

It follows from the independent pair equation `d_dot=-2*sin(d)`.
The other three edges remain excluded for all time: their principal-gap
magnitudes have lower bounds `0.55*pi`, `0.625*pi` and `0.8*pi`.
The raw closing gap is `-2*q-d`. It passes through `-pi` at
`t_*=log(5)/4`, when `d=q/2`, changing the support-cycle winding from zero
to minus one. At the crossing itself the antipodal branch observation
requires its explicit boundary treatment. The active phase support remains
two pairs and an isolated node; no fully acute winding cycle forms.

All local pressure resultants remain nonzero. The positive `eta` also
keeps the phase pressure away from its antipodal displacement boundary.
Full-support pressure, unlike phase evolution, continues to consume all
five edges. Its exact phase source is

\[
\begin{aligned}
g_0&=(q+d)/\pi,&g_4&=-g_0,\\
g_1&=-3(q+d)/(4\pi)+\eta/(2\pi),&
g_3&=3(q+d)/(4\pi)+\eta/(2\pi),\\
g_2&=1-\eta/\pi.
\end{aligned}
\]

Thus `sum g_i=1`, and, while unclipped, the arithmetic form mean is
`mean(x)=1/2+t/10`. The nonzero winding accompanies fragmented phase
coupling and persistent uniform form drive. It is not evidence of
formation of the admitted coherent cycle or bounded stationary form.

**Frozen before production execution, 2026-09-26:** use only `eta=pi/40`,
`T=1/2`, `h=1/128` and 64 steps. Retain the same pressure coefficients,
unit capacities, rails `[0,1]`, absent Gamma, absence of events and
NumPy-pressure/scalar-EPI execution adapter as the contact control.
Use exact initial turns `(0,1/5,13/16,2/5,3/5)` before binary64
materialization. The bound `abs(g_i)<=1` gives
`norm(x-1/2,inf)<=t/2`, keeping the finite trace inside `[1/4,3/4]`.
The independent endpoint mean prediction is `0.55`.

The finite check must retain the actual admitted-neighbor counts, full
pressure support, a lifted closing-edge branch passage, source and nodal
Euler defects, and the absence of coherent capture. The observed discrete
crossing time must not be called the exact ODE crossing time. This one
control checks the implementation's distinction between winding and
formation; the theorem, not a sweep or simulation, closes the isolated
formation question.

### Finite formation-control evidence

Three additional tests in the shared
[control module](../tests/physics/test_coherent_pattern_contact.py)
passed, with the earlier two contact trajectories deselected. A separate
module fixture executes only the newly frozen trajectory. Independent
pressure evaluations at the two analytic endpoints are snapshots, not
additional trajectories.

The phase-admitted neighbors remain `((1,),(0,),(),(4,),(3,))`, while every
pressure row consumes both support neighbors. Winding changes from zero
to minus one, but no retained state is fully acute. The lifted Euler
closing-edge passage occurs in `[51/128,13/32]`; the separate exact ODE
event time is `log(5)/4`, approximately `0.4023594781`.

| Endpoint observation | Prediction or error budget | Measured value |
| --- | --- | ---: |
| Arithmetic EPI mean | `0.55` | `0.5499999999999999` |
| Mean forecast error | Exact represented accumulated defect budget `6.6691e-16` | `1.2213e-16` |
| Maximum form deviation | At most `1/4` | `0.21738514` |
| Pair-gap error against the continuous solution | At most `T*h=0.00390625`, plus arithmetic | `0.00252501` |

For the last row, `d_dot=-2*sin(d)` has `abs(d_ddot)<=2`, giving local
Euler error at most `h^2`. Its Euler map is nonexpansive on `[0,q]` for
the frozen step, hence the accumulated exact-arithmetic error is at most
`t*h`. The independently propagated scalar pair recurrence is checked
against both the analytic solution and the actual retained phase lifts;
it does not replace the production phase owner.

The mean budget accumulates the actual rate-mean defect from `1/10` and
the exact represented Euler endpoint defects. All proposals stay strictly
inside the rails. Maximum local estimated phase and source-sum defects are
approximately `2.22e-16` and `5.00e-16`; the largest represented EPI Euler
defect is `5.51e-17`. Source-sum arithmetic is not a proof of a general
conservation law: the nonzero sum here follows from this preparation's
explicit phasor branches.

Versions, path, clock, mean budget and both kinds of crossing evidence are
retained in the local JUnit receipt, using the same runtime versions and
80-digit residual arithmetic as the contact control:

```sh
python -m pytest tests/physics/test_coherent_pattern_contact.py -k formation_control -q --junitxml=artifacts/research/coherent_pattern_formation_2026_09_26.xml
```

This resolves a potential false formation claim without altering the phase
gate or pressure channels. A coupled environment changes the port equations
used in the isolated proof; any induced-formation claim must retain those
extra rates and the supplied environment explicitly.

## One external phase neighbor: budget and entrance obstruction

Keep the same recipient unit C5, common capacity one and literal gated
averaged-sine law. Connect recipient node 0 to one evolving external node
5. The donor can be the other C5 used above. No donor trajectory is imposed:
its reciprocal evolution remains part of the full graph. This section asks
whether the extra port can induce an acute winding identity in a recipient
that initially has winding zero.

### Exact regional contact balance

Use the preceding winding-one branch variables `delta`, `b`, `E`, `N`
and admitted ring-edge indicators `a_i`. Set

\[
\gamma=\operatorname{wrap}(\theta_5-\theta_0),\quad
c=\mathbf1_{|\gamma|\le b},\quad n=a_4+a_0,\quad
S=a_0\sin\delta_0-a_4\sin\delta_4.
\]

The co-rotating rate with only the two internal ring neighbors is `f_0^0=S/n`
when `n>0`, and zero otherwise. The actual full-graph rate is
`f_0=(S+c*sin(gamma))/(n+c)`, with the same empty-neighborhood convention.
This is a same-snapshot algebraic comparison, not evolution of an isolated
replacement graph. The contact correction is

\[
u=f_0-f_0^0=
\begin{cases}
0,&c=0,\\
(\sin\gamma-f_0^0)/(n+1),&c=1.
\end{cases}
\]

It changes only the recipient gap rates `delta_4_dot` by `+u` and
`delta_0_dot` by `-u`. With `P={i:delta_i>b}`, the exact excess balance is

\[
\dot E=R+u\sigma,\qquad
R=\sum_{i\in P}(a_{i-1}\sin\delta_{i-1}+a_{i+1}\sin\delta_{i+1}),
\qquad \sigma=\mathbf1_{4\in P}-\mathbf1_{0\in P}.
\]

As before, `R>=-2*N>=-2*E`. If both port-adjacent gaps have positive excess,
their contact contributions cancel in the sum. The contact sign depends
on the actual donor phase and internal port rate, not on a coherence score.
All these quantities are derived observations; none is a new driver.

On any such branch interval, finite arrival at `E(T)=0` would require

\[
\int_s^T e^{2(r-s)}[-u(r)\sigma(r)]\,dr\ge E(s).
\]

This is necessary, not sufficient. Since `abs(u)<=1`, it also requires
`T-s>=log(1+2*E(s))/2`. The inequalities retain actual admitted counts;
using the full topological degree in the phase denominator would change
the model. Pressure continues to use its separate full-support contract.

### A local improvement does not establish capture

Let `epsilon` lie in `(0,b)`, take any positive `w_0,...,w_3` with sum one,
and prepare recipient gaps

\[
\delta_i=b+w_i\epsilon\ (i=0,1,2,3),\qquad \delta_4=-\epsilon.
\]

The recipient already has winding one, with excess `E=epsilon`; it is not
an initially winding-zero formation trial. A donor with uniform winding
one and acute bridge gap `gamma` has actual co-rotating port rates

\[
f_0=(\sin\epsilon+\sin\gamma)/2,\qquad
f_5=-\sin\gamma/3.
\]

Recipient rates `f_1=f_2=f_3=0`, `f_4=-sin(epsilon)` give

\[
R=-2\sin\epsilon,\qquad
u=(\sin\gamma-\sin\epsilon)/2,\qquad
\dot E=-\tfrac32\sin\epsilon-\tfrac12\sin\gamma.
\]

For fixed positive `gamma`, this stays negative as `epsilon` tends to zero.
Thus a uniform global estimate `E_dot>=-C*E` is false with the contact.
Nevertheless, the middle excluded gaps have zero instantaneous rate. A
decreasing sum can coexist with individual gaps that prevent capture.

### Why a single external neighbor still cannot create the acute identity

The following argument covers every evolving donor through
`abs(sin(gamma))<=1`; it does not freeze the donor or fit a source.
On a winding-one branch with `0<E<b`, every gap outside P is admitted,
because `N<=E<b` excludes a negative gate-off gap.

Project the gaps onto `bar_delta` in `[0,b]^5` with sum `4*b`, keeping
every P gap at b. Raise negative non-P gaps to zero, then distribute the
remaining `E-N` among the non-P gaps without exceeding b. Such a filling
exists, and its total increase on non-P gaps is exactly E.
This is an algebraic projection: keep the original excluded set P when
evaluating its expression at the projected point, even though the literal
inclusive gate would admit the boundary gaps there. It is not a new
trajectory or a reassigned gate policy.

When the bridge is excluded or `sigma=0`, the isolated bound applies.
Otherwise suppose `0 in P`, `4 not in P`; the other case has reflected
coefficients and the opposite bridge term. Let `m_j` count the two adjacent
edge indices that belong to P. For the admitted bridge, the excess derivative becomes

\[
\dot E=\sum_{j\notin P}v_j\sin\delta_j-\tfrac12\sin\gamma,
\qquad v_j=m_j-\tfrac12\mathbf1_{j=4},\quad 0\le v_j\le2.
\]

The sine function is 1-Lipschitz, so replacing non-P gaps by their projection
costs at most `2*E` in this expression. On the projected polytope, deficits
`b-bar_delta_j` sum to b. Concavity of sine on `[0,b]` puts a minimum at a
vertex with one non-P gap zero and all others equal to b. If B counts the
cyclic P/non-P interfaces, the worst value, including the bridge, is
`B-1-max(v_j)`.

For C5, B is two or four. If B is four, this is nonnegative. If B is two
and the complement of P contains at least two edges, its endpoint weights
are at most one, again giving nonnegativity. The sole exception has
`P={0,1,2,3}`; reflection gives `P={1,2,3,4}`. Consequently
`E_dot>=-2*E` holds near zero except in those two patterns.

Suppose a positive-excess interval ended at finite time with `E=0`.
Unless the endpoint is `(b,b,b,b,0)` or `(0,b,b,b,b)`, continuity excludes
both exceptional patterns nearby and the local exponential bound forbids
arrival. Near the first corner, gaps 0 through 3 remain positive. For
the non-port edge 1, whenever its gap exceeds b its rate is a sum of
nonnegative admitted neighboring sines. Therefore `(delta_1-b)_+` is
nondecreasing almost everywhere. Its zero endpoint value forces it to be
zero throughout a preceding neighborhood, excluding the first exceptional
pattern. Continuity excludes the other pattern there. At the second corner
use edge 2 with positive neighboring gaps 1 and 3. The same exponential
argument then rules out arrival at either corner as well.

Thus no positive-excess interval can end at zero. An acute winding-one
endpoint would force `E=0` throughout its preceding antipodal-free interval;
the resulting gaps in `[0,b]` cannot encounter an antipodal endpoint. The
initial recipient must already have winding one. Reflection proves the
winding-minus-one result, and convergence to a strict acute lock would
entail finite entry.

**Conclusion:** the fixed single-edge interface does not allow an initially
winding-zero recipient to reach a strict acute winding identity under this
common-capacity phase law. This remains true with an actual evolving donor,
bridge admission/removal and simultaneous gate boundaries, provided there
are no phase jumps and the equation holds almost everywhere along an
absolutely continuous solution. It does not establish existence/uniqueness
at every discontinuity. Multiple external neighbors, moving attachment
ports, capacity changes, other supports or other phase laws are outside
this theorem. No claim about arbitrary TNFR identities follows.

### Frozen local balance controls

No induced-formation trajectory is justified after that obstruction.
Instead, three finite, non-mutating snapshot checks test the new balance
against the production owner. **Frozen before production execution,
2026-09-26:** use `epsilon=pi/40`, weights `(1,2,3,4)/10`, recipient turns
`(0,201/800,403/800,606/800,1/80)`, and donor turns
`gamma/(2*pi)+k/5`, reduced modulo one. Use `gamma=pi/6,-pi/6,3*pi/4`
on the same two-C5 graph with bridge `(0,5)`. All capacities and edge
weights/lengths are one; EPI is `1/2`. Keep the previous explicit EPI/phase
pressure weights, both U3 limits `pi/2`, absent Gamma and no glyph events.

Materialize exact turns with 80-digit arithmetic into binary64. Capture
detached forcing/support observations, actual phase-admitted neighbors and
one non-mutating shared phase proposal with `h=1/128` per snapshot. Do not
write that proposal, integrate EPI or execute a trajectory. Check independent
represented-input rates and proposal residuals separately from the ideal
formulas above. The two acute bridge signs must give opposite contact
effects; the excluded bridge has zero phase correction while still
contributing to full-support pressure. The unequal gap excesses avoid
zero-resultant and antipodal pressure-displacement ambiguities. Both
prepared cycles have winding one; these checks cannot be called formation.

### Finite local-budget evidence

The two `contact_budget_control` tests in the shared
[control module](../tests/physics/test_coherent_pattern_contact.py) pass with
the eight earlier tests deselected. They reuse one fixture containing the
three frozen observations. No proposal is applied, no EPI integrator is
invoked, and no trajectory is generated.

The common isolated rate is approximately `R=-0.1569181915`. The actual
contact effect changes sign as predicted:

| Bridge gap | Port correction `u` | Predicted excess rate | Rate inferred from the finite phase proposal |
| --- | ---: | ---: | ---: |
| `pi/6` | `+0.2107704521` | `-0.3676886436` | `-0.3676886436` |
| `-pi/6` | `-0.2892295479` | `+0.1323113564` | `+0.1323113564` |
| `3*pi/4`, excluded | `0` | `-0.1569181915` | `-0.1569181915` |

The comparison retains initial materialization, represented-input rates
and proposal-increment residuals separately. The maximum estimated phase
proposal residual is `3.47e-16`, and the largest full-support pressure-source
residual is `9.02e-17`. The minimum resultant modulus and pressure branch
margin agree with `2*sin(3*pi/800)` and `pi/800`, respectively, within
the declared absolute numerical tolerance. These are finite numerical
comparisons, not validated interval certificates.

Pressure retains three support neighbors at both ports in every case.
In particular, exclusion of the bridge from phase motion does not remove
its pressure contribution. Detached support rows are sorted, while live
phase admission preserves neighbor insertion order; the control retains
both conventions instead of assuming identical array ordering. Nodes,
edges and declared configuration remain unchanged after observation and
proposal. This check makes no equality assertion about private pressure
caches.

The local receipt records exact preparation turns, node/order conventions,
versions, path and residual estimates from those same observations:

```sh
python -m pytest tests/physics/test_coherent_pattern_contact.py -k contact_budget_control -q --junitxml=artifacts/research/one_port_phase_budget_2026_09_26.xml
```

Execution used Python 3.13.6, NumPy 2.5.3, NetworkX 3.6.1 and mpmath 1.3.0,
with binary64 production and 80-digit residual arithmetic. The continuous
entrance theorem remains a separate analytical result. This control
verifies the new balance and its observation path, not the creation or loss
of an identity.

## Distributed two-port contact: a realizable entrance witness

The one-port obstruction does not extend to the fixed interface
`(0,5),(1,6)` between the same two C5 rings. Keep the supplied gated phase
law, held unit capacities, unit conductances, equal EPI/phase pressure
weights and absence of Gamma/events. Only support and preparation change.
The second contact creates a third independent cycle; both external phase
differences must come from one nodal state. They are not independent controls.

### Two-port budget and exact preparation

For recipient port `k=0,1`, let `n_k` count its admitted ring neighbors,
`f_k^iso` be their mean sine (zero if none), and `gamma_k` the phase of its
external neighbor relative to that port. Its actual correction is

```text
u_k = 0                                      if the bridge is excluded,
u_k = (sin(gamma_k)-f_k^iso)/(n_k+1)           if the bridge is admitted.
```

With the earlier excess set P and isolated excess rate R, the regional
balance becomes

```text
E_dot = R + u_0*(1_{4 in P}-1_{0 in P})
          + u_1*(1_{0 in P}-1_{1 in P}).
```

Both corrections act on gap 0, with opposite signs. Their allowed values
are constrained by the square cycle and the evolving donor.

Set `b=pi/2`, `q=2*pi/5`, and prescribe

```text
d = b+epsilon,        r = (2*pi-d)/4,       gamma = (d+q)/2,
recipient lifts = (0, d, d+r, d+2*r, d+3*r),
donor lifts     = (gamma, gamma-q, gamma-2*q, gamma-3*q, gamma-4*q).
```

For sufficiently small positive epsilon, the recipient has winding +1,
one excluded gap d and four acute gaps r. The donor has acute winding -1;
the two bridge gaps are `gamma,-gamma`. The new square `(0,1,6,5)` has
period zero, since `d-gamma+q-gamma=0`. Thus the joint preparation is
realizable. This is not a winding-zero initial recipient.

Subtract the common angular rate 1 and denote the remaining nodal rates
by f. Initially, while gap 0 is excluded,

```text
f_0 = (-sin(r)+sin(gamma))/2,       f_1 = -f_0,
f_5 = -sin(gamma)/3,               f_6 = -f_5,
all other f_i = 0,
d_dot = sin(r)-sin(gamma).
```

At `epsilon=0` the excluded-side limit is `-m`, where
`m=sin(9*pi/20)-sin(3*pi/8)>1/16`. The literal inclusive gate instead
uses three neighbors at each recipient port, giving
`d_dot=(2/3)*(sin(r)-sin(d)-sin(gamma))<0`. Both sides point inward.
For comparison, removing bridge `(1,6)` at the same preparation gives
`d_dot=(3*sin(r)-sin(gamma))/2>0`. This comparison is algebraic, not a
second executed preparation. The donor rates above are nonzero: admission
does not assume a stationary environment or imposed port torques.

The centered bridge phases maximize the available initial inward contrast
within this fixed uniform-donor family. Joint square realizability requires
`gamma_0-gamma_1=d+q`, so

```text
sin(gamma_0)-sin(gamma_1)
    = 2*cos((gamma_0+gamma_1)/2)*sin((d+q)/2)
    <= 2*sin((d+q)/2).
```

Equality occurs at the chosen `gamma_0=-gamma_1=(d+q)/2`, with both
bridges acute. A uniform donor of the same winding instead fixes
`gamma_0-gamma_1=d-q`; at `d=b` its maximum contrast is
`2*sin(pi/20)<2*sin(3*pi/8)`, insufficient for this inward direction.
This comparison concerns the declared uniform donor families. It does
not make opposite winding a universal requirement for induced formation.

### A short continuous entrance theorem

Fix **`epsilon=pi/8000` and `T=1/64`**. In the co-rotating frame,
`abs(f_i)<=1`, so every support gap changes by at most `2*t`. The smallest
initial noncritical acute margin is `pi/20-epsilon/2>2*T`. Consequently
only recipient edge `(0,1)` can change gate before T; no support gap can
reach an antipodal branch during this interval.

On the excluded chamber, the recipient gap derivative differs from its
initial value by at most `4*t`: each nodal mean sine is 2-Lipschitz in the
supremum norm of co-rotating phases. Here `r<=3*pi/8` and
`gamma>=9*pi/20`, both below `pi/2`, hence

```text
d_dot(t) <= -m+4*t < 0                       for 0 <= t <= T,
d(T)-b <= epsilon-m*T+2*T^2 < 0              if the edge stayed excluded.
```

The last inequality follows from `m>1/16` and
`epsilon<1/2048=T/16-2*T^2`; thus exclusion through T is impossible.
One direct bound for m follows from
`cos(z)>=1-z^2/2` and `cos(z)<=1-z^2/2+z^4/24` at these arguments:

```text
m >= pi^2*(1/128-1/800)-pi^4/98304 > 1/16.
```

The final strict inequality already follows from `157/50<pi<22/7`.
On the admitted chamber, extend its smooth field back to the initial
preparation. Its gap derivative there is
`(2/3)*(sin(r)-sin(d)-sin(gamma))<-1/2`; the same `4*t` bound keeps it
negative through T. There can be no immediate return to the excluded
side. Smooth existence on each chamber and transverse inward crossing
therefore construct a local absolutely continuous solution of the literal
law, with a strictly acute recipient at T and the acute donor retained.
The value at the single crossing instant does not change that a.e. law.
This is local existence/admission, not a global well-posedness claim.

Full-support pressure is also regular. Initially its relative resultants
at recipient ports are conjugates of
`exp(i*d)+exp(-i*r)+exp(i*gamma)`; the other recipient rows are `2*cos(r)`.
Donor port resultants are `2*cos(q)+exp(+-i*gamma)` and other donor rows
are `2*cos(q)`. Every real part exceeds `1/2`: at the most restrictive
recipient port use
`cos(3*pi/8)+cos(9*pi/20)-3*epsilon/2>1/2`.
At most three terms each change real part by `2*t`, giving
`Re(z_i(t))>1/2-6*T=13/32`. No resultant vanishes or reaches its relative
Arg cut. The phase pressure source satisfies `abs(g_i)<1/2`; the more
general `abs(g_i)<=1` bound suffices below.

With initial EPI `1/2`, the shared form equation is
`x_dot=(-L_rw*x+g)/2`. Its maximum principle gives
`abs(x_i(t)-1/2)<=t/2<=1/128`. Thus clipping to `[0,1]` is inactive for
this continuous witness. These are conditional phase and form results
for the supplied laws, not consequences of the nodal product alone.

### Frozen shared-engine control

**Declared before execution, 2026-09-26:** execute this single preparation
for 32 joint steps with `h=1/2048`, `T=1/64`. Store exact rational nodal
turns and materialize with 80-digit arithmetic into binary64. Reuse the
topological chord-extension owner to check cycle rank `2 -> 3` and the
additional square. Compute exact periods from these declared nodal turns;
do not weaken the strict-acute reconstruction API to admit the initial
excluded edge.

Reuse the existing U3 phase proposal, NumPy full-support pressure and
scalar nodal Euler through `tests/joint_phase_helpers.py`. Both U3 limits
are `pi/2`; capacity/topology pressure weights are zero, capacities and
all conductances/lengths are one, initial EPI is `1/2`, extended dynamics
is disabled and there are no Gamma/glyph events. Record node/support order,
gate membership, both ring windings, the new square period, full-support
resultants, pressure branches, clipping margins and actual donor motion.
Compare initial rates with the formulas above, and retain represented-input
phase/source/held-Euler residuals throughout the trace.

This control tests local entrance by the actual shared executor. The
continuous theorem and binary64 trace have separate scopes; a discrete
gate bracket is not a continuous crossing-time enclosure. No topology,
offset, horizon or step-size sweep is admitted. Success would demonstrate
recovery into an acute sector from an already wound state, not formation
from winding zero or indefinite maintenance under contact.

### Finite distributed-contact evidence

The three focused tests in
[test_two_port_coherent_contact.py](../tests/physics/test_two_port_coherent_contact.py)
pass using one shared fixture. The fixed control required no parameter
adjustment. Exact fundamental cycle periods are `(1,0,-1)` in the shared
owner's basis; the explicitly oriented recipient, donor and square periods
are `(1,-1,0)`. These periods persist at all 33 retained states.

| Observation | Finite result |
| --- | ---: |
| Initial recipient excess rate | `-0.06387707919015755` |
| First admitted endpoint for edge `(0,1)` | Step 13 |
| Discrete gate bracket | `[3/512,13/2048]` |
| Endpoint minimum all-support acute margin | `0.006582062193005909` |
| Endpoint maximum form displacement from `1/2` | `0.002714587714422123` |
| Endpoint maximum donor deformation after subtracting common rotation | `0.005129347493660036` |
| Minimum retained full-support resultant modulus | `0.6131536621655245` |
| Minimum retained pressure branch margin | `2.039457036751049` |

The recipient starts with only edge `(0,1)` excluded, enters the acute
region and remains there for the rest of the retained trace. The donor
remains acute and moves relative to common rotation. Full pressure keeps
all three neighbors at both recipient ports even before edge `(0,1)` is
phase-admitted; using the gated neighbor set would give a different source.
Form stays within the predicted radius `1/128`, and every held Euler
proposal is strictly inside the clipping interval.

Maximum estimated phase proposal, phase-source and held nodal state
residuals are `4.41e-16`, `9.73e-17` and `5.55e-17`, respectively.
These numerical comparisons are not validated interval bounds or an
error enclosure for the continuous trajectory. In particular, the
discrete gate bracket above does not enclose a proven ODE crossing time.
The continuous entrance theorem independently supplies its own horizon.

```sh
python -m pytest tests/physics/test_two_port_coherent_contact.py -k "not formation_budget and not gate_barrier_control" -q --junitxml=artifacts/research/two_port_entrance_2026_09_26.xml
```

The receipt retains exact preparations, configuration, gate rows, cycle
periods, execution path and arithmetic estimates. Execution used Python
3.13.6, NumPy 2.5.3, NetworkX 3.6.1 and mpmath 1.3.0; production values
are binary64 and reference residual arithmetic uses 80 decimal digits.
This closes local two-port entrance admission. A path from winding zero
to a new acute regional identity remains a separate open obligation.

## Winding-zero formation: a dissipation and preparation boundary

Keep the same fixed two-contact graph, common unit capacity and supplied
phase/form laws. A recipient winding change and the later existence of
two acute wound rings are different conditions. This section derives a
necessary preparation budget before attempting the latter. It does not
exclude every nonuniform winding-zero preparation.

### A potential valid across phase gates and winding changes

The earlier cosine energy assumes all support edges are phase-admitted.
For the literal gated law its global continuation is instead

```text
psi(z) = 1-max(cos(z),0),
V_g(theta) = sum_{undirected support edges {i,j}} psi(theta_j-theta_i).
```

It equals `1-cos(z)` on acute separations and stays at one outside them.
It is periodic, nonnegative and Lipschitz. It is a potential of this
specified phase law, not physical energy, the tetrad energy or a new
decision rule for selecting operators. Its cap follows by integrating the
existing gated sine torque; it does not modify the gate or phase dynamics.

Let `a_ij` be the symmetric inclusive gate, `n_i=sum_j a_ij`, and
`f_i=sum_j a_ij*sin(theta_j-theta_i)/n_i`, with `f_i=0` when `n_i=0`.
The literal row is `theta_dot_i=1+f_i`. For any absolutely continuous
solution satisfying that equation almost everywhere,

```text
dV_g/dt = sum_edges a_ij*sin(theta_j-theta_i)*(f_j-f_i)
        = -sum_i n_i*f_i^2 <= 0,
V_g(t) + integral_0^t sum_i n_i*f_i^2 ds = V_g(0).
```

Away from `cos(z)=0`, this follows by the chain rule and pairing the two
ends of every edge. On a gate boundary level set, the phase difference
has derivative zero almost everywhere on that set, so the same formula
holds with the inclusive gate. At an antipodal crossing psi is locally
constant and has no jump. This proves the integrated identity through
gate changes without asserting global existence or uniqueness of the
discontinuous ODE. Common angular drift cancels. Unequal free rates,
directed interactions, support events or another phase law require their
own balance; Euler/binary64 nonincrease does not follow automatically.

The EPI row remains driven by full-support pressure. It supplies no return
term to this phase row, so changing initial form or its pressure does not
replenish V_g under this declared closure. The ordinary runtime's different
phase map cannot be substituted into this theorem.

### What two acute identities cost in this model

For a strictly acute C5 of winding +1, each of its five oriented gaps
is positive: the other four sum to less than `2*pi`. Reflection gives
the negative-winding case. With `q=2*pi/5`, convexity and the strict chord
bound `1-cos(z)<2*z/pi` for `0<z<pi/2` give

```text
5*(1-cos(q)) <= V_ring < 4.
```

Both strict-acute wound rings, with either relative winding sign, thus
require

```text
V_g >= Q = 10*(1-cos(q)) = (25-5*sqrt(5))/2 > 6.9.
```

Bridge contributions are nonnegative, so this is a necessary lower bound
even when the final bridges are excluded. If the recipient initially has
uniform phase, its ring potential is zero. An acute wound donor contributes
less than four and the two contacts at most two. Therefore
`V_g(0)<6<Q`: **an initially uniform recipient cannot form an acute winding
identity while an acute wound donor is present at the endpoint.** The
exclusion even allows the donor to leave and re-enter its acute sector
between endpoints. No contact duration can fix the missing budget.

For a general nonuniform winding-zero recipient, the total-potential
necessary filter is `V_g(0)>=Q`. The donor/contact bounds also imply the weaker
necessary condition

```text
V_recipient(0) > Q-6 = (13-5*sqrt(5))/2 = approximately 0.90983.
```

Neither inequality is sufficient. A state can have enough potential yet
be blocked by gates, joint cycle geometry, donor loss or pressure
singularities. These exclusions concern this supplied dissipative phase
closure; they are not a no-emergence theorem for every TNFR model.

### A branch-crossing candidate that still cannot form the target

Use a same-winding uniform donor and a nonuniform winding-zero recipient:

```text
epsilon = pi/8000,  d = pi+epsilon,  r = (2*pi-d)/4,
gamma = (d-q)/2,
recipient lifts = (0,d,d+r,d+2*r,d+3*r),
donor lifts     = (gamma,gamma+q,gamma+2*q,gamma+3*q,gamma+4*q).
```

The recipient, donor and square `(0,1,6,5)` periods are `(0,+1,-1)`.
Only recipient edge `(0,1)` is phase-excluded; its principal gap is
`-pi+epsilon`. Both bridges are acute, with gaps `gamma,-gamma`.
The unwrapped recipient port separation obeys
`d_dot=sin(r)-sin(gamma)<0`. Thus a winding change is locally feasible.

More precisely, let `m=sin(3*pi/10)-sin(pi/4)>1/10` and `T=1/64`.
The strict speed bound follows from `sqrt(5)>223/100` and
`sqrt(2)<99/70`, which give `m>281/2800>1/10`.
The common bound `abs(f_i)<=1` retains all other gates and gives
`d_dot(t)<=-m+4*t`. Since `epsilon-m*T+2*T^2<0`, d crosses pi before T;
edge `(0,1)` remains excluded on both sides of this branch crossing.
The actual donor evolves and remains acute. Full-support pressure is
regular: initial relative resultants have real parts above `1/4`, and
their change is at most `6*T`, leaving real parts above `5/32`. The
continuous local path therefore changes recipient winding `0 -> +1`
and square period `-1 -> 0`, with donor winding unchanged. The square
already carried a supplied winding; this is a redistribution of cycle
periods, not creation of the substrate.

Nevertheless, its initial potential is

```text
V_g(0) = 1 + 4*(1-cos(r)) + 5*(1-cos(q)) + 2*(1-cos(gamma)) < Q.
```

At epsilon zero the deficit is exactly
`-2+2*sqrt(2)-5*cos(q)+2*cos(3*pi/10)>3/10`. For example, use
`sqrt(2)>7/5`, `cos(q)<1/3`, `cos(3*pi/10)>7/12`.
The derivative of initial potential with respect to epsilon is
`-sin(r)+sin(gamma)`, of magnitude at most two; the frozen epsilon
cannot exhaust that deficit. The dissipation identity excludes the
two-acute-ring target at every future time, conditional on existence of
the stated phase solution. An extended simulation of this preparation
is therefore not a formation experiment worth pursuing.

### Frozen non-mutating budget check

**Declared before execution, 2026-09-26:** check only the preparation above
as one static snapshot. Use exact rational nodal turns, 80-digit
materialization into binary64, and the same configuration and pressure
owner as the two-port entrance control. Record exact cycle periods,
actual gate membership, full-support pressure/resultant availability,
V_g, Q and the initial potential derivative.

Request one shared phase proposal with `h=1/2048` but do not apply it or
integrate EPI. Compare its represented-input rate against the independent
edge derivative and `-sum_i n_i*f_i^2`. The uncapped cosine sum exceeds
Q in this preparation, whereas V_g is below Q: reusing the acute-only
potential outside its domain would miss the exclusion. Check graph
nodes, edges and declared configuration remain unchanged, excluding
private pressure caches from that equality. Reuse one fixture for the
two budget tests in the existing two-port control module. No formation
trajectory, offset search or duration extension is authorized by this
protocol. Arithmetic residuals are finite estimates, not an all-time
numerical certificate for the continuous theorem.

### Finite budget evidence

The two `formation_budget` tests pass, with the three earlier trajectory
tests deselected. They reuse one non-mutating observation and one unapplied
phase proposal. The exact oriented periods are `(0,+1,-1)`; the shared
fundamental basis reports `(0,1,0)`. The initial gap rate is approximately
`-0.1020950324`, consistent with motion toward the antipodal passage.

| Quantity | Finite observation |
| --- | ---: |
| Gated phase potential V_g | `6.4509574551312285` |
| Two-acute-ring lower bound Q | `6.9098300562505255` |
| Missing potential Q-V_g | `0.45887260111929734` |
| Uncapped cosine sum, inapplicable globally | `7.450957378024945` |
| Gated potential rate from independent admitted rates | `-0.446886878829151` |
| Rate inferred from the shared phase proposal | `-0.4468868788288669` |

The independent edge derivative agrees with the degree-weighted squared
rates within the 80-digit calculation. The maximum estimated proposal
**rate** residual is `3.06e-13` (increment residual divided by h); the
phase-source residual is below `8.61e-17`. The minimum resultant modulus
is `0.3118524913428369`, and the pressure branch margin is
`2.5505192562434775`. Actual pressure still reads the excluded recipient
edge. The retained nodes, edges and declared configuration are unchanged.

```sh
python -m pytest tests/physics/test_two_port_coherent_contact.py -k formation_budget -q --junitxml=artifacts/research/two_port_formation_budget_2026_09_26.xml
```

The receipt retains exact turns, configuration, actual neighbor counts,
potential observations and arithmetic estimates. Execution used Python
3.13.6, NumPy 2.5.3, NetworkX 3.6.1 and mpmath 1.3.0; binary64 production
was compared with 80-digit reference arithmetic. No trajectory was run
for this preparation, and no sampled derivative is promoted to the
all-time theorem. The excluded preparations are closed; general
nonuniform winding-zero reachability remains open.

## Joint port dynamics: sufficient potential does not give a formation path

The two port corrections are not independently adjustable. Their shared
bridge terms cancel in a weighted combination of recipient and donor
gap rates. This supplies a path obstruction beyond the potential filter.
All statements retain fixed support, unit capacity, the same supplied
phase law and an actually evolving donor.

### An excluded-side barrier for a donor of the same winding

Write `d=delta_0` for the positive recipient gap on edge `(0,1)` after
a winding change, and `e=eta_0` for the donor gap on `(5,6)`. Consider
`b<=d<=pi`, an acute donor of winding +1, the other four recipient edges
admitted, and both bridges admitted. The donor's gaps are positive;
the other recipient gaps may have either sign. With bridge separations
`gamma_0,gamma_1`, define

```text
A = sin(delta_1)+sin(delta_4),
B = sin(eta_1)+sin(eta_4),
C = sin(gamma_0)-sin(gamma_1).
```

As long as the recipient shared edge is excluded, the actual admitted
neighbor counts are two at recipient ports and three at donor ports:

```text
d_dot = (A-C)/2,
e_dot = (B-2*sin(e)+C)/3,
Z = 2*d+3*e,
Z_dot = A+B-2*sin(e).
```

The factors two and three are derived from those counts. They are not
fitted coefficients. The cancellation retains both reciprocal donor
responses; replacing the donor by fixed phases would lose it.

Recipient cycle closure gives `delta_1+delta_4>=pi-d`. For
`u,v in [-b,b]` with `s=u+v>=0`, the minimum sine sum at that s is
`1-cos(s)`, achieved when one argument is b. Thus `A>=1+cos(d)`.
Donor closure similarly gives `eta_1+eta_4>=pi-e` and
`B>=1+cos(e)`. At the surface `Z=2*pi`, where
`d=pi-3*e/2` and `0<e<=pi/3`,

```text
Z_dot >= 2-cos(3*e/2)+cos(e)-2*sin(e)
      >= 2-sqrt(3) > 0.
```

Here `cos(e)>=cos(3*e/2)`. Therefore the side `Z>=2*pi` is protected
throughout this chamber. A shared-edge antipodal passage starts with
`d=pi`, hence `Z>2*pi`. At a proposed subsequent entrance `d=b`, the
barrier forces `e>=pi/3`.

At that entrance the joint square has period zero: both bridges are
admitted and `d-e in [0,b]`, so `gamma_0-gamma_1=d-e`. Consequently

```text
C <= 2*sin((b-e)/2) <= 2*sin(pi/12),
A >= 1,
d_dot >= (1-2*sin(pi/12))/2 > 0.
```

This is the incoming, **excluded-side** rate. It prevents arrival from
`d>b`; the different inclusive-gate rate cannot bypass that obstruction.
The argument closes the route from a shared-edge slip to acute recovery
while the other recipient edges and both bridges remain admitted and the
donor remains acute. It is a phase-law theorem, not an energy condition or
an unrestricted statement about all switching itineraries.

### An uninterrupted-admission obstruction for either donor sign

The result does not need a guessed single-slip trajectory. Suppose the
recipient starts with well-defined winding zero and its four edges other
than `(0,1)` remain admitted. Their principal gaps are continuous and
define `S=sum_{i=1}^4 delta_i` and the continuous lift `d=2*pi-S` of the
port separation. Initially `d in (pi,3*pi)`; a strict-acute winding-one
endpoint requires `d in (0,b)`.

For a donor of winding +1, consider the final connected interval in
`b<d<3*b` before the first arrival at b. It starts either at the initial
time with `d>pi`, or at `d=3*b`. Thus `Z>2*pi` initially in this interval.
The shared edge is excluded throughout the open strip, including any
repeated antipodal crossings, so its admitted-count equations hold.
On `Z=2*pi` within this strip, `d=pi-3*e/2` automatically lies in
`[b,pi]`; the preceding barrier calculation applies. The positive
incoming rate at b rules out the proposed endpoint. Earlier admission of
the shared edge or repeated changes of its winding cannot evade this
argument while the other admission hypotheses hold.

For a donor of winding -1, its corresponding gap satisfies `-b<e<0`.
At the unavoidable decreasing crossing `d=pi`, joint circular geometry
with both bridges admitted requires

```text
kappa = gamma_0-gamma_1 = -pi-e in (-pi,-b),
C = sin(gamma_0)-sin(gamma_1) <= -(1+cos(e)) < -1.
```

This follows by writing `C=2*cos(M)*sin(kappa/2)` and using
`abs(M)<=b-abs(kappa)/2`. These phases are geometrically realizable; the
square may carry nonzero winding. It is their **direction of motion**
that blocks formation: recipient closure gives `A>=0` at `d=pi`, and
therefore `d_dot>1/2`. A decreasing passage is impossible. Reflection
in co-rotating phases (equivalently `theta -> 2*t-theta`) covers a
target recipient of winding -1 while preserving the common free drift.

In every excluded-side neighborhood used above, all other phase edges
remain admitted, so the trajectory follows one fixed smooth sine field
almost everywhere. Its integral representation is continuously
differentiable there. The strict outward rates therefore prevent a
downward arrival even for the stated absolutely continuous solution
class; isolated gate or branch instants do not bypass the barrier.

**Conclusion:** a winding-zero recipient cannot reach a strict-acute
nonzero winding while all four non-shared recipient edges and both
bridges remain admitted and the donor remains strictly acute with
nonzero winding. This allows arbitrary admitted asymmetry, adequate
initial potential, and changes of the shared edge's gate or branch.
It is conditional on absolutely continuous solutions of the supplied
law, not a global existence theorem or an unrestricted numerical claim.

Any successful route preserving the donor must consequently involve an
additional bridge or non-shared recipient edge being phase-excluded at
some time. This can already occur in its preparation; the result does
not force a unique first event. Gate switches and simultaneous corners
remain open mechanisms. A counterexample to this uninterrupted route is
not a proof that a route with those events succeeds. The support graph
has not changed; it is the law's active phase-neighbor set that must
change or start outside this class.

### A high-potential winding-zero preparation

The obstruction is relevant even after the earlier budget test passes.
Fix the following exact preparation:

```text
epsilon = pi/8000, d = pi+epsilon,
a = pi/20, s = (2*pi-d-2*a)/2,
e = pi/8, u = (2*pi-e)/4 = 15*pi/32, gamma = (d-e)/2,
recipient lifts = (0,d,d+a,d+a+s,d+a+2*s),
donor lifts = (gamma,gamma+e,gamma+e+u,gamma+e+2*u,gamma+e+3*u).
```

The oriented recipient, donor and square periods are `(0,+1,-1)`.
Only recipient edge `(0,1)` is excluded. The exact initial potential is

```text
V_g = 1+2*(1-cos(a))+2*(1-cos(s))
        +(1-cos(e))+4*(1-cos(u))+2*(1-cos(gamma))
    > 9-7*pi/20-pi/8000 > 7 > Q.
```

For this elementary bound, drop the positive a/e terms and use
`cos(s)<a+epsilon/2`, `cos(u)<pi/32` and `cos(gamma)<pi/16`.
The preparation passes the necessary potential test; it does not
establish a successful trajectory.

Initially `d_dot=sin(a)-sin(gamma)<-3/4`. Over `T=1/128`, the bound
`abs(f_i)<=1` keeps all other gates unchanged and changes this rate by
at most `4*t`. Since `epsilon-(3/4)*T+2*T^2<0`, the recipient crosses
its antipode before T. All initial relative resultants have real parts
greater than `1/8`; their change is at most `6*T`, leaving real parts
above `5/64`. Full-support pressure remains defined and away from its
relative Arg cut. This constructs a local continuous branch passage with
an acute evolving donor and no clipping requirement beyond the earlier
form maximum principle. It does not construct the later gate entrance.

The donor starts at `e=pi/8` but has positive e rate; it is not held
at that small gap. After the slip, the Z barrier prevents completion
unless one of its admission hypotheses changes. Starting with sufficient
potential and favorable local motion has not removed this dynamical
obligation.

### Frozen non-mutating port-balance checks

**Declared before execution, 2026-09-26:** inspect exactly two static
preparations, with no applied proposal and no EPI integration:

1. The high-potential preparation above. Its exact nodal turns have common
   denominator 32000: recipient `(0,16002,16802,24001,31200)` and donor
   `(7001,9001,16501,24001,31501)`.
2. A point on `Z=2*pi`: `d=3*pi/4`, `e=pi/6`, equal remaining recipient
   gaps `5*pi/16`, equal remaining donor gaps `11*pi/24`, and centered
   bridges `gamma=7*pi/24`. Exact nodal turns have denominator 96:
   recipient `(0,36,51,66,81)`, donor `(14,22,44,66,88)`. Its periods
   are `(+1,+1,0)` and only edge `(0,1)` is excluded.

Reuse the existing two-port control owner, configuration, full-support
pressure capture and one non-mutating shared phase proposal per state
with `h=1/2048`. Materialize exact turns with 80-digit arithmetic into
binary64. Check exact cycle realizability, phase admission, independent
initial sine rates and the identity `2*d_dot+3*e_dot=A+B-2*sin(e)` against
the actual proposal. At the second state require positive outward Z rate;
at the first retain both the passed potential filter and the negative
recipient gap rate. Record pressure availability and graph-state
preservation, excluding private pressure caches from that equality.
Keep proposal-increment and recovered-rate rounding estimates separate.
These checks do not supply a formation trajectory, interval enclosure or
an all-time binary64 barrier certificate.

### Finite joint-balance evidence

The two `gate_barrier_control` tests pass with the previous five tests
deselected. Both preparations pass the necessary potential filter, and
the actual shared phase proposal agrees with the independent port rates.
No proposal is applied and no trajectory is executed.

| Observation | High-potential branch preparation | Z-boundary preparation |
| --- | ---: | ---: |
| V_g | `8.00562295359448` | `7.172066037239505` |
| V_g-Q | `1.0957928973439555` | `0.262235980988979` |
| Recipient gap rate from proposal | `-0.8243891023519354` | `0.03811627201135204` |
| Donor gap rate from proposal | `1.0622165744662195` | `0.8565321344433414` |
| Z rate from proposal | `1.5378715186947878` | `2.6458289473527286` |
| Independent Z rate | `1.5378715186946759` | `2.6458289473527112` |
| Minimum relative resultant real part | `0.18258615921863527` | `0.261052384440103` |

The boundary observation has `Z_dot>2-sqrt(3)` as predicted. The first
observation confirms that a rapidly decreasing recipient gap coexists
with a changing donor gap and increasing Z. It does not establish
formation; this is precisely the coupled response omitted by a fixed
donor approximation.

The largest estimated recovered-rate residual is `6.06e-13`, below
the declared finite comparison allowance `2*ulp(2*pi)/h`, approximately
`3.64e-12`. This allowance addresses subtraction followed by division
by h, not physical uncertainty or an all-time arithmetic bound. Full
phase-source residual estimates are below `1.92e-16`. Both snapshots
retain regular full-support pressure and unchanged nodes, edges and
declared configuration.

```sh
python -m pytest tests/physics/test_two_port_coherent_contact.py -k gate_barrier_control -q --junitxml=artifacts/research/two_port_gate_barrier_2026_09_26.xml
```

Execution used Python 3.13.6, NumPy 2.5.3, NetworkX 3.6.1 and mpmath
1.3.0, with binary64 production and 80-digit reference arithmetic. The
receipt retains exact turns, cycles, admitted neighbors, configuration,
execution paths and estimates. The analytical obstruction has broader
scope than these two finite checks. Exclusion/re-admission of another
phase edge remains an unproved route, not a demonstrated mechanism.
