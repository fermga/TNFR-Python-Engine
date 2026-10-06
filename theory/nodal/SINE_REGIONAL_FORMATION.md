# Conservative regional formation and its barriers

Finite winding entry, regional channel balance, distributed source geometry, retention barriers and a robust finite passage under the full conservative environment.

Part of [Sine pattern geometry and dissipative capture](SINE_PATTERN_DYNAMICS.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 35. Conservative exchange can produce finite regional winding

<a id="conservative-regional-winding-entry"></a>

Section 27's positive-loss capture theorem does not extend by setting its
loss coefficient to zero: its preparation scaling divides by that coefficient.
There is instead a direct finite-time mechanism under the conservative
complete law. It concerns a receiver's cycle winding, an observation of part
of the full network. It does not claim entry into a two-sided invariant
full-state trapping set, which remains excluded by the
[conservative formation boundary](RESONANCE_FOUNDATIONS.md#conservative-formation-boundary).

### A global full-law phase enclosure

Let the support be finite, connected, simple, undirected and unit-weighted,
with at least two nodes. Keep unit held capacities, `e=0`, `w=beta=1`, no
forcing and no support events. In the declared clock `tau=t/pi`, the
normalized-sine rows are

\[
x'=KS(\theta),\qquad \theta'=KLx,\qquad
K=\operatorname{diag}(1/d_i),\quad
S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i).
\]

Here and below a prime denotes `d/dtau`. Smooth sine and the linear form
row give a unique global solution in real phase lifts; wrapping remains an
observation, even through an antipodal crossing. Put `v0=KLx(0)`. Since
`|S_i|<=d_i` and `||KL||_infinity=2`, direct integration gives

\[
\boxed{\begin{aligned}
\|x(\tau)-x(0)\|_\infty&\le\tau,\\
\|\theta(\tau)-\theta(0)-\tau v_0\|_\infty&\le\tau^2.
\end{aligned}}
\]

Indeed the phase remainder is the integral of
`KL(x(s)-x(0))`, whose norm is at most `2s`. Consequently every oriented
edge satisfies the full nonlinear bound

\[
\left|\delta_{ij}(\tau)-\delta_{ij}(0)
            -\tau(v_{0j}-v_{0i})\right|\le2\tau^2.
\]

No linearized trajectory replaces either consumed row. On a supplied
interval `[a,b]` with `0<=a<=b`, take the range of the affine edge expression
over that interval and add `[-2b^2,2b^2]`. If each resulting interval lies
strictly inside one branch `((2h_ij-1)pi,(2h_ij+1)pi)` with declared integer
`h_ij`, the entire solution has those edge offsets throughout the interval.
For an oriented cycle, telescoping the actual lifted differences then gives
`W=-sum(h_ij)`. A failed enclosure is inconclusive; it does not prove that
the actual solution leaves the branch. This is a whole-interval result,
not a collection of sampled winding observations.

### One frozen two-port preparation

Reuse the supplied
[return-path support](RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-equilibrium):
the two cycles `(0,1,2,3,4)` and `(5,6,7,8,9)`, mediator `10`, edges
`0-10`, `10-5`, and the return edge `1-6`. This section uses the smooth
conservative sine law above; the older native-law equilibrium and pulse
certificates on this support are not transferred. Declare

\[
R=\{5,6,7,8,9\},\quad Q=V\setminus R,\qquad
x_{10}(0)=24,\quad x_1(0)=-24,
\]
\[
x_i(0)=0\ (i\notin\{1,10\}),\qquad \theta_i(0)=0\quad\text{for all }i.
\]

Thus the receiver starts with uniform form, equal phases and winding zero.
Its initial phase velocities are derived from the full nodal law:

\[
(v_{05},v_{06},v_{07},v_{08},v_{09})=(-8,8,0,0,0).
\]

The two port degrees remain three. These rates are not independently
prescribed receiver inputs: they follow from the retained environmental
forms, supplied support and complete `KLx` row. Freeze the retention
interval `[1/4,1/3]` in `tau`, equivalently `[pi/4,pi/3]` in `t`. Along the
oriented receiver cycle the affine gap slopes are `(16,-8,0,0,-8)`.
The global enclosure gives, simultaneously throughout that interval,

\[
\begin{aligned}
\delta_{56}&\in[34/9,50/9]\subset(\pi,3\pi),\\
\delta_{67},\delta_{95}&\in[-26/9,-16/9]\subset(-\pi,-\pi/2),\\
\delta_{78},\delta_{89}&\in[-2/9,2/9]\subset(-\pi,\pi).
\end{aligned}
\]

The fixed offsets are therefore `(1,0,0,0,0)` and the receiver winding is
**exactly `-1` for every time in the frozen interval**. Its strict branch
margin is at least `pi-26/9>0`; the other faces are farther away. Since it
begins with winding zero, at least one receiver edge crosses an antipodal
branch before or at `tau=1/4`. The smooth dynamics are uninterrupted there.
The proof does not locate a unique crossing or certify the receiver's
winding at every earlier time.

This is an intentionally limited identity claim. In particular the two
outer gaps above are obtuse throughout the interval, so the result is not
an acute-sector capture certificate. No attraction, asymptotic winding,
indefinite maintenance or complete NFR identity follows.

<a id="sine-regional-storage-balance"></a>

### Regional work retains the cross-boundary storage

For any node partition `R,Q`, define internal storage from edges lying
entirely within the region, denoted by `E(R)`:

\[
E_R=\frac12x_R^{\mathsf T}L_Rx_R
       +\sum_{\{i,j\}\in E(R)}(1-\cos(\theta_j-\theta_i)),
\]

and similarly `E_Q`. Let `E_cross` sum the same edge storage over the
cross-boundary edges exactly once. The actual rates still use full-support
`q=Lx`, `S` and `K`, not the isolated regional Laplacians. For `i in R` put

\[
f_i=\sum_{j\in Q:j\sim i}(x_j-x_i),\qquad
s_i=\sum_{j\in Q:j\sim i}\sin(\theta_j-\theta_i).
\]

The regional gradients are `q_i+f_i` and `-S_i+s_i`. Nodewise cancellation
of `q_i x_i'-S_i theta_i'=0` under the conservative rows proves

\[
\boxed{E_R'=P_R:=\sum_{i\in R}(f_i x_i'+s_i\theta_i'),\qquad
E_Q'=P_Q,\qquad E_{\rm cross}'=-P_R-P_Q.}
\]

This is the multi-port specialization of the existing
[receiver work convention](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-receiver-port-passage).
Both form and phase work are needed. The global storage
`E=E_R+E_Q+E_cross` is conserved; incoming work can later return and is not
an irreversible loss term or a reusable event allowance.

For the frozen preparation the initial ledger is exactly

\[
(E_R(0),E_Q(0),E_{\rm cross}(0))=(0,864,576),\qquad E(0)=1440.
\]

The environment supplies all nonzero initial form, but the energy stored
on its contacts must not be counted again as environmental internal storage.
Because both `delta_67` and `delta_95` lie in `(-pi,-pi/2)`, their phase
storage contributions each exceed one. Therefore throughout the retention
interval

\[
\boxed{\int_0^\tau P_R(s)\,ds=E_R(\tau)>2.}
\]

At the initial instant all sine currents vanish, so `x'(0)=0` and
`P_R(0)=0`. The strictly positive integrated work is a conclusion about the
subsequent full-network exchange, not an initial power score relabelled as
stored energy.

<a id="conservative-regional-phase-transport"></a>
### Acute winding requires internal phase redistribution

Keep the same law, support and unit capacities, globally equal initial phases,
and uniform initial receiver form; environmental form amplitudes may be
arbitrary. In a fully acute C5 with winding `+1`, every principal gap is
positive: otherwise the other four, each smaller than `pi/2`, could not sum
to `2pi`. The negative-winding case has the reversed signs. The sum of the
two inner gaps `7->8->9` therefore has magnitude strictly between `pi/2`
and `pi`, because the other three contribute less than `3pi/2`. In particular

\[
\boxed{\cos(\theta_9-\theta_7)<0}
\]

is necessary for fully acute winding at any time. This is a correlation of
existing phases, not a new coordinate or an additional law. The relevant
organization is an oriented phase pattern, not equality of all phases.

The shared inner path yields a stronger time restriction than treating its
edges as independent. Write `z=theta_9-theta_7`. The full rows give

\[
z'=x_9-x_7+\frac{x_6-x_5}{2},\qquad z(0)=z'(0)=0,
\]
\[
|z''|\le3,\qquad |z(\tau)|\le\frac32\tau^2.
\]

The derivative bound follows from `|x_i'|<=1`. Equivalently the exact
pair-current coefficients in `z''=c^T S` are
`c_9=1/2,c_7=-1/2,c_6=1/6,c_5=-1/6`, with all others zero;
`sum_edges |c_i-c_j|=3`. This retains the telescoping correlation instead
of adding two independent edge-error radii.

If `tau^2<=pi/3`, the individual inner raw gaps have magnitude at most
`2tau^2<=2pi/3<pi` by the global enclosure, so their principal sum is
exactly `z`. Acute winding would require `|z|>pi/2`, contradicting the bound.
Thus **`tau^2>pi/3` is necessary**, independently of environmental amplitude.
The existing `[1/4,1/3]` winding interval lies entirely inside this exclusion.
The inequality is not an optimal entry time or an obstruction after its
cutoff. Additional ports, nonuniform receiver form, nonflat preparation or
different capacities require their own calculation.

### The complete rows already transmit phase motion to the inner path

There is an explicit local propagation mechanism, although it does not prove
finite acute entry. Put `A=KL`. At globally flat phase, `x'(0)=0` and
`theta''(0)=0`, while differentiation of the same complete sine rows gives

\[
v_0=Ax(0),\qquad x''(0)=-A^2x(0),\qquad
\boxed{\theta'''(0)=-A^3x(0).}
\]

For the supplied preparation `x_10=H,x_1=-H`, all other forms zero, the
inner phase third derivatives at nodes `(7,8,9)` are
`(5H/9,0,-5H/9)`. For example, the receiver form second derivatives are
`(7H/9,-7H/9,H/6,0,-H/6)`. Hence

\[
z'''(0)=-10H/9,\qquad z(\tau)=-5H\tau^3/27+O(\tau^5).
\]

The odd phase powers follow from the exact time-reversal symmetry about
this flat-phase preparation: after removing its common phase origin,
`theta(-tau)=-theta(tau)` and `x(-tau)=x(tau)`. At `H=24`, the leading
inner spread is `-40tau^3/9`, in the same orientation as the certified
negative winding. This local expansion has no supplied finite-time remainder;
extrapolating it past the necessary-time cutoff would not certify entry.
[`analyze_sine_conservative_phase_transport`](../../src/tnfr/physics/relational_sine_entry.py)
retains the admitted source, full-row derivatives and correlated transport
bounds without replacing the nonlinear law by its Taylor polynomial.

<a id="conservative-regional-work-retention"></a>
### A finite regional work budget can retain an admitted acute state

There is a sufficient continuation condition that retains form as well as
phase. On an acute winding-one C5, convexity of `1-cos(delta)` gives the
minimum phase storage

\[
V_5=5\bigl(1-\cos(2\pi/5)\bigr).
\]

At an acute-sector boundary at least one gap is `pi/2`. The other four
sum to `3pi/2`; convexity minimizes their storage at `3pi/8` each. A
`-pi/2` boundary gap cannot carry winding one with the remaining four
gaps in `[-pi/2,pi/2]`. Reflection gives the negative-winding case. Thus
the exact minimum phase storage on either sector's boundary is

\[
\boxed{B=1+4\bigl(1-\cos(3\pi/8)\bigr)
         =5-4\cos(3\pi/8).}
\]

Suppose an independently admitted full state at time `T` has acute receiver
winding `+1` or `-1`. For a declared finite horizon `h`, suppose a proved
bound on its actual boundary work satisfies

\[
E_R(T)+\sup_{0\le s\le h}\int_T^{T+s}P_R(u)\,du
\le E_{\max}<B.
\]

The exact regional balance then prevents a first acute-face exit before
`T+h`: at such an exit phase storage alone would be at least `B`. Throughout
this interval the internal form storage also satisfies
`F_R<=E_max-V_5<B-V_5`, where `B-V_5` is approximately `0.01435` in the
declared units. A strict budget supplies a positive acute margin and retains
the full regional form differences; it is not just a winding sample. The
actual full-support phase rates and environmental work remain part of the
state and balance. This uses the phase-barrier geometry of the
[sector theorem](SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture), not its positive-loss
capture conclusion. It supplies no entry from a flat receiver by itself.

For the frozen `H=24` preparation, entry into this low-storage region would
require genuine return of storage to the environment. At `tau=1/4`, the
existing phase bound gives

\[
\delta_{56}\in[31/8,33/8]\subset(\pi,4\pi/3),\qquad
\delta_{67},\delta_{95}\in[-17/8,-15/8].
\]

The first edge has phase storage greater than `3/2`. Each outer edge has
storage greater than `5/4`, since `15/8>7pi/12` and
`cos(7pi/12)<-1/4`. Consequently `E_R(1/4)>4`. Since `B<7/2`, any later
entry time `T` with `E_R(T)<B` must satisfy

\[
\boxed{\int_{1/4}^{T}P_R(u)\,du
       =E_R(T)-E_R(1/4)<-1/2.}
\]

This necessary net return is compatible with global conservation and is
not dissipation. It identifies internal redistribution and export, beyond
the already proved incoming transfer, as obligations of a low-storage entry
claim. It does not exclude an acute high-energy passage and does not prove
that the required return or entry occurs. No new trajectory is inferred
from these analytic restrictions.

<a id="conservative-regional-organization-control"></a>

### A bounded full-state organization control

The [prospective declaration](../../docs/assets/conservative_regional_organization/declaration.json)
retains the same support and conservative complete law, with `H=3`, all
initial phases zero and unit capacities. Receiver initial phase velocities
are `(-1,+1,0,0,0)` in `tau`; total initial storage is `45/2`, partitioned as
receiver `0`, environment `27/2` and cross-boundary `9`. This normalizes one
preparation for study. It is not an optimal amplitude or a selected law.

The horizon is `t in [0,12]`, using original structural time, with 256 fixed
steps of `3/64` and Taylor order 10. The target is a contiguous whole-time
acute receiver winding interval of at least `1/4` in `t`, with fixed edge
branches and acute margin at least `1/16` radian, certified by a strict
lower bound greater than that threshold. These numerical and
observation choices are supplied policies. Full regional form, phase rates,
storage and signed boundary work remain observable; the target does not
require the stronger low-storage protection condition above.

The [producer](../../benchmarks/conservative_regional_organization.py) freezes
the [protocol](../../docs/assets/conservative_regional_organization/response-v1.protocol.json),
[source archive](../../docs/assets/conservative_regional_organization/response-v1.source.zip)
and runtime before invoking the shared
Picard/Taylor forecast. The
[regional observer](../../src/tnfr/physics/relational_sine_regional.py) distinguishes
certified finite retention, exclusion on a fully covered horizon and unresolved
or partial evidence. No amplitude, horizon or solver budget is adjusted in
response to the reserved evolution. A numerical enclosure failure cannot
become a mathematical impossibility result.

The [frozen response](../../docs/assets/conservative_regional_organization/response-v1.json)
certifies all 256 whole-time tubes, covering the entire declared horizon.
Its outcome is `acute_winding_excluded_on_horizon`, with no evaluation error
or unresolved numerical step. Every receiver edge remains acute and all
principal-branch offsets remain zero, so receiver winding is exactly zero
throughout. The minimum certified acute margin exceeds `0.348` radian.
Each tube also independently excludes acute unit winding through a
nonnegative two-hop phase cosine. These are full-interval exclusions, not
an inference from endpoint sampling.

The exchange nevertheless changes the receiver. Its endpoint internal
storage is approximately `5.19076`, consisting of form storage `4.17198`
and phase storage `1.01878`. Every tube bounds phase storage below `1.263`,
whereas acute unit winding on this C5 requires at least
`V_5=5*(1-cos(2*pi/5))`, approximately `3.45492`. The signed regional power
is strictly positive on 224 tubes and strictly negative on 23; its interval
contains zero on the remaining 9. Thus incoming and returned work both occur,
but total storage exceeding `V_5` does not supply the required phase geometry.
The endpoint's nonzero relative phase rates also preclude interpreting it as
phase locking or equilibrium. The exact interval record, not these rounded
display values, owns the evidence.

This closes the declared `H=3` control negatively without retuning preparation,
horizon or target. It does not exclude later winding, other preparations or
other definitions of an organized pattern. In particular, nonuniform form
and locally acute phases can be organized without nonzero winding; winding
is the selected identity discriminator here, not a universal NFR definition.
This response does not establish how boundary exchange and reversible
internal form/phase conversion supply *directed phase organization*.
The total-work ledger alone cannot decide it. The separate
[same-orbit construction](SINE_CONSERVATIVE_PREPARATION.md#sine-conservative-formation-retention) proves
a conditional existence result with a different preparation and support;
the failed response here remains retained without replay.

### Controls, implementation and limits

Resetting the two nonzero environmental forms to zero, while retaining the
same support, law, receiver state and phases, gives an exact stationary
control. This is a preparation control, not an equal-energy comparison.
A distinct support control removes `1-6` before initialization and keeps
the supplied forms. Then the receiver reflection `(6 9)(7 8)` fixes the
entire preparation and law. By the existing
[single-port symmetry obstruction](SINE_PATTERN_RECOVERY.md#a-symmetry-obstruction-to-forming-the-receiver-twist),
receiver winding stays zero whenever no receiver edge is antipodal.
Neither control introduces a support event into the evaluated trajectory.

[`certify_sine_conservative_winding_entry`](../../src/tnfr/physics/relational_sine_entry.py)
reuses admitted complete-law state and reports the analytic phase enclosure,
declared cycle offsets and whole-interval branch margins. The proof supplies
its numerical budget before evaluating any response; a trajectory solver is
unnecessary for this witness. Regional work observations retain their own
exact edge ledger and full-support rates rather than replacing them with
the phase enclosure.

`comparison.regional_storage_balance(region=...)` in the
[shared sine owner](../../src/tnfr/physics/relational_sine_comparison.py) also
supports the existing general sine coefficients and nonnegative capacities.
In its original structural clock `t`, the same calculation gives
`dE_R/dt=-e*sum_R(nu_i/d_i)*q_i^2+P_R`, where phase storage and phase
boundary work carry the declared `beta`. The complement has its own full-node
loss; the cross-edge derivative remains `-P_R-P_Q`. This snapshot identity
does not integrate boundary work or substitute for the finite-window theorem.
No inverse capacity is needed at zero capacity.

The [analytic declaration](../../docs/assets/conservative_regional_winding/declaration.json)
precedes the retained
[certificate and controls](../../docs/assets/conservative_regional_winding/certificate-v1.json).
The [producer](../../benchmarks/conservative_regional_winding.py) reuses the
write-once evidence transport and retains its
[source archive](../../docs/assets/conservative_regional_winding/certificate-v1.source.zip).
There is no reserved trajectory: the global inequality certifies every time
in the declared window. Independent tests check the complete nodal jet,
telescoping, storage partitions and both controls without fitting a response.

The analytic winding theorem establishes one conditional mechanism for acquiring
and retaining a regional winding through conservative environmental exchange. The large,
asymmetric environmental preparation and both ports are supplied premises.
Their origin, a mechanism selecting this preparation, a low-budget formation
law and physical identification remain open. This proof neither selects the
sine constitutive law nor derives a connection-occurrence law.

<a id="sine-regional-channel-accessibility"></a>

## 36. Regional channel balance and phase accessibility

### Separate internal conversion from boundary input

Retain the complete smooth sine law on fixed simple connected unit support,
held nonnegative capacities and no forcing or events. Set `K=diag(nu_i/d_i)`
using **full-network** degrees, `a=w/pi` and storage scale `beta>0`. For a
region `R`, let `q_R=L_R*x_R` and `S_R` be its internal form gradient and
internal sine current. External contributions are

\[
f_i=\sum_{j\notin R,\,j\sim i}(x_j-x_i),\qquad
s_i=\sum_{j\notin R,\,j\sim i}\sin(\theta_j-\theta_i).
\]

Thus the restriction of the full form gradient is `q_R-f`, and the full
sine current is `S_R+s`. The actual regional rows in original structural time
are

\[
\dot x_R=K_R[-e(q_R-f)+a(S_R+s)],\qquad
\dot\theta_R=\frac a\beta K_R(q_R-f).
\]

Write internal storage as `E_R=F_R+beta*V_R`, with
`F_R=x_R^T L_R x_R/2` and `V_R=sum_internal(1-cos(delta))`.
Its gradients are `q_R` and `-S_R`. Define

\[
J_R=a q_R^\mathsf T K_R S_R,\quad
B_F=a q_R^\mathsf T K_R s,\quad
B_V=a S_R^\mathsf T K_R f,\quad
D_F=e q_R^\mathsf T K_R(q_R-f).
\]

Direct differentiation yields the exact identities

\[
\boxed{\dot F_R=J_R+B_F-D_F,\qquad
       \frac{d(\beta V_R)}{dt}=-J_R+B_V.}
\]

Positive `J_R` converts phase storage into form storage, while its negative
converts form into phase. `B_F` is the input to internal form storage from
external phase currents; `B_V` is the input to internal phase storage from
external form contrasts. They can have either sign. `D_F` is signed and is
not the nonnegative full-node loss in the
[existing regional ledger](#sine-regional-storage-balance).

Indeed that ledger has
`P_R=f^T*xdot_R+beta*s^T*thetadot_R` and
`D_R=e*(q_R-f)^T K_R(q_R-f)`. Expanding both gives
`P_R-D_R=B_F+B_V-D_F`. The cross term `f^T K_R s` cancels, while
internal conversion cancels between the two storage rows. The old form and
phase **conjugate boundary-work** fields are therefore not the new inputs
to the individual stores and must not be renamed as such. These identities
remain valid at zero capacity and require no inverse capacity.

For `e=0`, the cumulative phase-directed work satisfies

\[
\beta[V_R(t)-V_R(0)]=\int_0^t[-J_R(u)+B_V(u)]\,du.
\]

This is a derived balance under the declared complete law, not an extra
pressure, damping term or controller. Instantaneous rates and their running
integrals are different observations. Retained environmental state supplies
both boundary inputs; eliminating it requires the separate memory contract.

### Equal storage does not determine the initial phase response

If all receiver phases are initially equal, the internal phase gradient
vanishes. The second derivative of its weighted phase storage is then

\[
\boxed{\frac{d^2(\beta V_R)}{dt^2}(0)
 =\beta\sum_{\{i,j\}\in E_R}
       [\dot\theta_j(0)-\dot\theta_i(0)]^2.}
\]

The terms involving phase acceleration vanish because every internal sine
is zero. The rates on the right are those of the full support, including
external form contrasts and full degrees. A common initial phase rotation
does not grow phase storage. Relative velocities, rather than total storage
alone, determine this initial curvature. This is a local-in-time identity;
it cannot certify later branch passage or retention.

An exact control reuses the two-C5/intermediary support and return edge `1-6`.
Take `e=0,w=beta=1`, unit capacity, all phases zero and all forms zero except
`x_10=H` and `x_1=+H` or `-H`. Both preparations have exactly the same
storage partitions: receiver `0`, complement `3H^2/2`, cross-boundary `H^2`.
In `tau=t/pi`, receiver phase rates and initial phase-storage curvature are

| Environmental form | Receiver phase rates in cycle order | `d^2 V_R/dtau^2(0)` |
| --- | --- | --- |
| `x_10=H, x_1=-H` | `(-H/3,+H/3,0,0,0)` | `2H^2/3` |
| `x_10=H, x_1=+H` | `(-H/3,-H/3,0,0,0)` | `2H^2/9` |

For `H!=0`, orientation changes initial phase growth by a factor of three at
identical storage, including each regional partition. Original-time curvature
divides these entries by `pi^2`. The raw conserved form means differ and are
retained; if a common mean leaf is required, independent uniform form shifts
set them equal without changing either field or any storage. No response is
simulated, and neither preparation is thereby proved to form an acute twist.

### A running phase barrier for any acute unit-winding entry

The [pure-phase C5 minimax](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-receiver-port-passage)
is `7/2`: every continuous phase path from consensus to a uniform unit twist
reaches `V_R>=7/2`. This is a geometric result on the phase torus, independent
of the positive-loss assumptions of other results using it.

It also applies when the endpoint is **any** fully acute state of winding
`+1` or `-1`. At that endpoint keep its integer edge offsets and linearly
interpolate its principal gaps to the uniform gaps `+2*pi/5` or `-2*pi/5`.
Their sum remains `+2*pi` or `-2*pi`, so the change lifts to a continuous
nodal phase path. All gaps remain in `(-pi/2,pi/2)`; convexity of `1-cos`
there bounds this appended path's storage by its initial endpoint storage.
If the original path never reached `7/2`, the concatenation would contradict
the minimax. This establishes a necessary **running**, not final, condition:

\[
\boxed{\text{flat receiver reaches an acute unit-winding state by }T
\quad\Longrightarrow\quad
\max_{0\le t\le T}\int_0^t[-J_R(u)+B_V(u)]\,du
\ge \frac72\beta.}
\]

Exceeding this bound does not supply oriented passage or a particular target;
the two signs of winding have identical storage. Remaining below it excludes
acute entry, not arbitrary nonacute winding. Nor does the barrier exclude an
already prepared acute state. The zero initial potential in this formula
uses flat phase; the [larger sector argument](#sine-cycle-sector-barrier)
retains the initial potential and gives a broader exclusion without it.
A signed branch crossing still depends on the actual relative phase velocity,
and protected retention still requires its own full-state and work bounds.

### Shared implementation and evidence boundary

`regional_storage_balance` in the
[comparison owner](../../src/tnfr/physics/relational_sine_comparison.py) exposes
these channels for both region and complement, preserving the original
conjugate work ledger. The same kernel consumes admitted whole-time boxes in
[`assess_sine_regional_channels`](../../src/tnfr/physics/relational_sine_regional.py).
It integrates interval rates over each complete step, retains endpoint
balance residuals and distinguishes complete horizons from validated prefixes.
The `7/2` exclusion is conditional on certified flat initial receiver phase
and applies only to the validated prefix. Taylor endpoint enclosures remain
numerical premises; neither export nor re-reading authenticates their origin.
The [API contract](../../docs/contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-regional-channel-history)
owns admission and projection details. These observers add no evolution law.

### Retrospective channel diagnosis of the frozen control

The [separate analysis record](../../docs/assets/conservative_regional_organization/channel-analysis-v1.json)
binds the original response, protocol and source archive by hash and retains
its own [analysis source archive](../../docs/assets/conservative_regional_organization/channel-analysis-v1.source.zip).
No solver is replayed and no preparation, horizon or original verdict changes.
All 256 original tubes pass primitive/chain admission and recomputed Picard
inclusion. Integrating their signed channel intervals on `t in [0,12]` gives
the following outward-rounded bounds; the record retains exact fractions.

| Cumulative quantity | Certified enclosing interval | Meaning |
| --- | --- | --- |
| `integral J_R dt` | `[2.982,3.205]` | Positive net internal conversion from phase to form |
| `integral B_F dt` | `[0.952,1.203]` | Positive net boundary input to internal form storage |
| `integral B_V dt` | `[4.036,4.187]` | Positive net boundary input to internal phase storage |

Their balances contain the independently rebuilt endpoint stores
`F_R(12)` approximately `4.17197693` and `V_R(12)` approximately `1.01878426`.
They do not imply that each channel stays positive at every instant.
The running phase-work upper bound is less than `1.377`; direct primitive
phase-storage enclosures give the tighter bound `V_R<1.263`. Both stay below
the necessary `7/2` barrier over the entire validated horizon. In particular,
the boundary phase input alone exceeds the barrier, but the net retained
phase-directed work does not. Counting gross input as available phase storage
would give the wrong accessibility diagnosis.

This identifies a realized conversion mechanism under the declared law. It
does not establish that another orientation suppresses conversion long enough
to create the target, nor that phase-to-form conversion prevents every other
kind of pattern. The equal-budget curvature control supplies only an initial
discriminator. The distributed boundary theorem below supplies one sufficient
finite-passage class; it does not reopen this frozen control.

## 37. Distributed source geometry can produce finite acute winding

<a id="sine-conservative-source-geometry"></a>

The conservative full law admits a sufficient preparation-to-passage result
when the boundary supplies enough distinct phase directions. This changes
the supplied support and preparation, not the nodal law. The result is
finite acute winding, not attraction, an indefinitely maintained pattern or
an autonomous choice of its initial source.

### The environmental source image uses full degrees

Keep Section 35's finite connected simple unit support, unit held capacities,
`e=0`, `w=beta=1`, no forcing or events, and `tau=t/pi`. Let the receiver
`R` have uniform initial form `c`, and let the whole network start with one
common phase. Put `Q=V\setminus R` and `y=x_Q(0)-c`. If `A_RQ` is the
receiver-to-environment adjacency block and `D_R` retains the **full**
receiver degrees, then the complete phase row gives

\[
\boxed{v_R(0)=T y,\qquad T=-D_R^{-1}A_{RQ}.}
\]

The internal form gradient is initially zero; the external contrasts give
the displayed row. Columns for environmental nodes without receiver contact
are zero and remain part of the retained state. They can affect later
evolution and are not declared dispensable.

Subtracting one receiver row of `T` from the others removes common phase
velocity. Full rank `|R|-1` of this relative map is sufficient to prescribe
arbitrary initial relative phase velocities. It is not necessary to reach
one particular acute phase cell. The relevant geometric admission is whether
the source image intersects that cell, with declared integer edge offsets.
Rank deficiency alone does not settle this intersection or nonlinear later
accessibility.

For example, the earlier adjacent two-port C5 has initial receiver velocity
of the form `(u,v,0,0,0)`. Its three non-port phases coincide in every affine
initial-velocity profile. An acute unit-winding C5 instead has five distinct
circular phases: its five principal gaps share a sign and their proper
partial sums have magnitude strictly between zero and `2pi`. Consequently
that two-port image cannot contain an acute unit-winding profile. This is
an obstruction to this direct short-time construction, not a proof that
subsequent nonlinear redistribution is impossible.

### A source image plus a global remainder gives actual finite passage

Let `R` induce the declared C5 cycle. Supply a source shape `y` and scale
`H>0`, so the full initial forms are `c` on `R` and `c+H*y` on `Q`.
Write `u=T*y`. Select `0<s_0<s_1`, integer cycle offsets `h_ij`, and a
strict margin `m>0` such that for every `s in [s_0,s_1]`,

\[
-\frac\pi2+m
\le s(u_j-u_i)-2\pi h_{ij}
\le\frac\pi2-m,\qquad
-\sum_{C_5}h_{ij}\in\{-1,1\}.
\]

For fixed offsets these are strict-interior linear phase-cell conditions
on the source profile. At one admitted interior point, continuity supplies
a nonempty surrounding `s` interval. Their feasibility is a geometric
property of `T`; it does not assert that a preparation is selected by the
dynamics.

The [global full-law enclosure](#conservative-regional-winding-entry), which
uses bounded sine currents rather than a linearized solution, gives

\[
\left|\theta_i(\tau)-\theta_i(0)-\tau v_i(0)\right|\le\tau^2.
\]

At `tau=s/H`, every cycle gap therefore differs from the admitted affine
profile by at most `2s_1^2/H^2`. Hence

\[
\boxed{\frac{2s_1^2}{H^2}<m}
\]

is sufficient for the **actual complete nonlinear solution** to have the
declared acute winding throughout `[s_0/H,s_1/H]`, with margin at least
`m-2s_1^2/H^2`. Its initial winding is zero, so an earlier branch passage
has occurred. No prescribed boundary trajectory, extra pressure, damping or
support event has been inserted. All environmental coordinates continue to
evolve under the same nodal rows.

The full regional state remains bounded as well:
`|x_i(tau)-c|<=tau`, and for its five cycle edges
`F_R(tau)<=10tau^2`. Actual phase rates satisfy
`|theta_i'(tau)-v_i(0)|<=2tau`; each relative edge rate has error at most
`4tau`. These are finite-time statements about the consumed form and phase
rows, not just a phase snapshot. The lifetime in the original clock is
`pi*(s_1-s_0)/H`: increasing source scale improves this enclosure while
shortening its certified passage window. It does not establish persistence
on a fixed time scale or approach to equilibrium.

### One supplied distributed preparation

Take receiver cycle `(0,1,2,3,4)` and five private environmental leaves
`(5,6,7,8,9)`, with leaf `5+i` joined only to receiver node `i`. Every
receiver degree is three, each leaf degree is one, and `T=-I/3`. Keep the
same unit-capacity conservative law and zero common phase, and supply

\[
x_i(0)=0,\qquad x_{5+i}(0)=-30(i-2),\qquad i=0,\ldots,4.
\]

Thus `v_R(0)=(-20,-10,0,10,20)`. Freeze

\[
\tau\in[1/8,2/15],\qquad t\in[\pi/8,2\pi/15].
\]

The four forward cycle edges have affine gap `10tau`; the closing edge
`4->0` has affine gap `-40tau`. With the common global edge error
`r=2(2/15)^2=8/225`, the complete solution obeys

\[
\begin{aligned}
\delta_{i,i+1}&\in[5/4-r,\;4/3+r],\quad i=0,1,2,3,\\
\delta_{40}+2\pi&\in[2\pi-16/3-r,\;2\pi-5+r].
\end{aligned}
\]

All these principal gaps are positive and smaller than `pi/2-1/5`.
For the first four the upper endpoint is `308/225`; for the closing edge
the margin is `5-3pi/2-8/225`. The elementary bounds `3.14<pi<22/7`
already prove both strict margins greater than `1/5`. The offsets are
`(0,0,0,0,-1)`, so actual winding is exactly `+1` throughout the window.
Its original-clock duration is `pi/120`. The earlier two-port source's
time restriction does not apply: the non-port velocity degeneracy has been
removed by the separately supplied distributed contacts.

The same enclosure proves that this witness is a transit: at `tau=1/6`,
the four forward principal gaps belong to `[29/18,31/18]`, strictly inside
`(pi/2,pi)`. Thus it has exited the acute sector by `t=pi/6`. This does not
say that its winding has disappeared. It separates the certified finite
acute episode from captured or maintained organization without a new
trajectory calculation.

Here `F_R<=8/45`, and the full relative phase rates retain the bounds just
proved. The initial global storage is exactly `4500`, entirely on the five
cross-boundary edges: receiver and environmental internal stores are zero.
The degree-weighted form mean and initial phase mean are both zero. Global
storage is conserved, while regional work obeys the existing channel and
cross-edge ledgers. This high-budget preparation encodes a desired relative
phase arrangement in environmental form; the theorem does not derive that
information or identify its storage with physical energy.

### The same storage and mean do not select the acquired winding

On exactly the same graph, keep all phases zero and instead supply

\[
x_i(0)=9\quad(i\in R),\qquad
(x_5,x_6,x_7,x_8,x_9)=(9,-21,-51,-51,-21).
\]

Equivalently, begin with receiver form zero and leaf contrasts
`-30*(0,1,2,2,1)`, then shift every form by `9`. This common shift changes
neither field nor storage and sets the degree-weighted form mean to zero.
Both sources thus have the same support, coefficients, capacities, initial
phase, total storage `4500`, all three regional storage partitions and
conserved means. Their receivers satisfy the same uniform-form
preparation condition; their environmental arrangements are different.

The control is fixed by the full-graph reflection `i -> -i mod 5` on
receiver nodes and its corresponding leaf permutation. Equivariance of the
complete sine rows and uniqueness preserve that reflection for all time.
It reverses cycle orientation, so principal increments cancel in reflected
pairs whenever no cycle edge is antipodal. In particular the receiver can
never have acute winding `+1` or `-1`. This is an all-time conditional
symmetry obstruction, independent of the positive source's finite window.
It neither claims consensus nor excludes other form or phase organization.

### Shared admission and scope

[`analyze_sine_conservative_source_geometry`](../../src/tnfr/physics/relational_sine_entry.py)
rebuilds the environmental map from admitted full-support state, retaining
all columns, relative rank, actual initial velocities and identical-row
groups. The same owner supplies
`certify_sine_conservative_winding_entry` and its global enclosure. Source
image geometry, a verified finite phase window, reflection invariance and
conditional regional storage retention are separate contracts; one must not
substitute a rank, a tangent derivative or an approximate midpoint for their
full hypotheses.

The [declared matched control](../../docs/assets/conservative_source_geometry/declaration.json),
[analytic certificate](../../docs/assets/conservative_source_geometry/certificate-v1.json)
and [source archive](../../docs/assets/conservative_source_geometry/certificate-v1.source.zip)
retain this positive preparation and its equal-budget reflection control.
The shared [analytic instrument](../../benchmarks/conservative_regional_winding.py)
records all eight gates as satisfied, with source/declaration hashes and
exact interval projections. This is a reproducible evaluation of analytic
bounds, not a sampled or reserved numerical trajectory. Its duration
`pi/120` is a separately declared target; it does not satisfy or rewrite
the earlier control's quarter-unit duration criterion.

This supplies a genuine sufficient mechanism under the declared model:
initially phase-flat receivers can acquire finite acute winding through
their evolving environment, and equal scalar storage does not determine
which source does so. The support and the structured initial form are
supplied. Pattern selection, a primitive microscopic law, connection birth,
indefinite maintenance and physical identification remain unresolved.

## 38. Low regional storage does not close the retention handoff

<a id="sine-conservative-handoff-obstruction"></a>

The regional work theorem requires a bound on **future accumulated work**,
not just small entry storage. The following class demonstrates why that
premise cannot be removed, even for an identity actually acquired from flat
phase under the unchanged complete law. It is a source-class obstruction to
a proposed implication, not a search for an optimal amplitude or support.

### Admit the actual full-support velocity ramp

Retain Section 37's finite connected simple unit graph, unit capacities,
`e=0`, `w=beta=1`, no forcing or events, and `tau=t/pi`. Let the receiver
induce the oriented C5 `(r_0,r_1,r_2,r_3,r_4)`. Its initial forms are one
constant `c`, and every initial network phase is common. Environmental
forms are supplied and subsequently evolve under the same full law.

Suppose the **actual** initial row `v=KLx(0)`, with full graph degrees,
satisfies

\[
v_{r_i}=v_c+\sigma(i-2)\omega,\qquad
\sigma\in\{-1,1\},\quad\omega>0.
\]

Thus the first four oriented edge velocities equal `sigma*omega`, while
the closing velocity is `-4sigma*omega`. This hypothesis is a test of
admitted state and contact geometry, not a prescribed phase drive. The
private-leaf construction realizes it with leaf forms `-3omega(i-2)`
and receiver form zero, but that particular support is not required.

Put

\[
\alpha=\frac{2\pi}{5},\qquad
\tau_* = \frac{\alpha}{\omega},\qquad
\tau_{\mathrm{out}}=\frac{5}{3\omega}.
\]

The following explicit sufficient conditions define the admitted class:

\[
\begin{aligned}
2\tau_*^2&<\frac\pi{10},\\
10\tau_*^2+10\tau_*^4&<B-V_5,\\
2\tau_{\mathrm{out}}^2
&<\min\left\{\frac53-\frac\pi2,\;\pi-\frac53\right\},
\end{aligned}
\]

where `V_5=5(1-cos(alpha))` and `B=5-4cos(3pi/8)` are the existing
regional minimum and acute-face barrier. They hold, for example, for every
`omega>=64`. Indeed `tau_*<1/50`,

\[
B-V_5
=\frac{5(\sqrt5-1)}4-2\sqrt{2-\sqrt2}
>\frac{13}{1000},
\]

and `2tau_out^2<=50/(9*64^2)<2/21<5/3-pi/2`.
The other exit inequality is weaker. The value `64` is a convenient
sufficient rate, not a dynamical threshold or an optimized preparation.

### Acquired entry near the storage minimum

At `tau_*`, the affine receiver profile is the uniform twist. Use edge
offsets zero on the first four edges and `-sigma` on the closing edge.
The actual principal gaps have the form

\[
\delta_e(\tau_*)=\sigma\alpha+\epsilon_e,\qquad
|\epsilon_e|\le2\tau_*^2,\qquad
\sum_{e\in C_5}\epsilon_e=0.
\]

The global nonlinear enclosure supplies the individual bounds. The last
identity is exact telescoping on the cycle, not an assumption of
independent errors. The first admission inequality keeps every gap acute,
so the actual winding is `sigma`; initially it was zero.

For `U(delta)=1-cos(delta)`, Taylor's inequality `U''<=1` and the
zero error sum cancel the linear phase-storage term. Consequently

\[
V_5\le V_R(\tau_*)\le V_5+10\tau_*^4,\qquad
F_R(\tau_*)\le10\tau_*^2,
\]

and hence

\[
\boxed{E_R(\tau_*)\le V_5+10\tau_*^2+10\tau_*^4<B.}
\]

The form bound uses `|x_i(tau)-c|<=tau` on all five receiver nodes.
Thus these acquired entry states approach the minimum regional storage as
`omega` increases. This controls internal form and phase, but not the
environmental form contrasts consumed by `KLx`.

### Strict finite exit and a necessary positive work transfer

At `tau_out`, each of the first four oriented principal gaps satisfies

\[
\sigma\delta_e(\tau_{\mathrm{out}})
\in\left[\frac53-2\tau_{\mathrm{out}}^2,
          \frac53+2\tau_{\mathrm{out}}^2\right]
\subset\left(\frac\pi2,\pi\right).
\]

Thus a first acute-face exit `tau_exit` after entry exists, with

\[
0<\tau_{\mathrm{exit}}-\tau_*
<\frac{5/3-2\pi/5}{\omega}.
\]

The first exit is strictly before the displayed nonacute endpoint by
continuity. Multiply this duration by `pi` in the original clock. No
conclusion about disappearance of winding follows from an acute-face exit.

At the first exit, all receiver gaps still belong to the closed acute
chart with the same winding. The existing face theorem gives
`E_R(t_exit)>=B`. The exact regional balance in the original clock
therefore forces

\[
\boxed{\int_{t_*}^{t_{\mathrm{exit}}}P_R(t)\,dt
\ge B-E_R(t_*)
\ge B-V_5-10\tau_*^2-10\tau_*^4>0.}
\]

For `omega>=64`, the displayed coarse bounds already make this integral
greater than `1/125`. Actual environmental work has crossed the missing
retention allowance; this is not merely a failed upper-bound calculation.
The full law conserves global storage, and internal form/phase conversion
remains part of the regional channel balance.

This class has arbitrarily small regional excess storage and arbitrarily
short acute lifetime. Accordingly, acute membership and `E_R<B` alone
cannot supply a uniform positive retention duration across admitted
environments. The class does not keep global storage fixed as `omega`
increases. It neither excludes finite maintenance for other sources nor
establishes or excludes another definition of regional identity.

### Zero instantaneous work does not remove the missing information

There is a separate instantaneous counterexample. At an exact receiver
twist `delta_e=sigma*alpha` with uniform receiver form, the internal
quantities satisfy `q_R=S_R=0`. Every regional channel in Section 36 and
`P_R` is therefore zero at that instant, regardless of the retained
environment. Nevertheless differentiation of the internal edge storage
under the complete law gives the exact identity

\[
\boxed{\frac{d^2E_R}{d\tau^2}
=\sum_{e\in C_5}(\Delta_e x')^2
 +\cos\alpha\sum_{e\in C_5}(\Delta_e\theta')^2.}
\]

The phase-acceleration term cancels because the oriented edge
accelerations telescope. Both sums retain actual full-support nodal
rates. On the private-leaf support, match each leaf phase to its receiver
phase, keep receiver form zero and supply leaf forms `-3omega(i-2)`.
Then the first sum is zero and the second gives
`E_R''=20cos(alpha)*omega^2`, although `E_R=V_5` and `P_R=0`.
This constructed snapshot is not claimed to be an exact state attained
by the phase-flat acquisition family.

Likewise, in that acquisition family the actual first-four phase rates
at entry satisfy `|Delta_e theta'(tau_*)-sigma*omega|<=4tau_*`.
They can grow while regional excess storage tends to zero. Regional
storage and instantaneous power discard precisely this environmental
rate information. A sufficient prediction must retain the source state
or its justified [causal memory](SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure);
it cannot replace them by a small local storage observation.

### Relative rhythm means compatible phase rates, not identical phases

The exact kinematic row on every retained edge is

\[
\delta_{ij}'=(KLx)_j-(KLx)_i,\qquad
\delta_{ij}(T+s)=\delta_{ij}(T)
 +\int_T^{T+s}\bigl[(KLx)_j-(KLx)_i\bigr]\,d\tau.
\]

The integral is read on the same continuous phase chart. A fixed relative
shape requires equal phase rates on its connected receiver, and that
equality must be preserved by the complete field. Equality at one instant
does not establish its continuation. An evolving identity instead permits
relative oscillations provided their accumulated differences stay within
the declared sector throughout the interval. Equal phases or constant
relative geometry are not necessary for such finite retention. Capacity
is still the held nodal mobility; it is not identified with these measured
phase rates or a physical oscillation frequency.

For this unforced fixed-support law, `sum_i d_i theta_i'=sum_i(Lx)_i=0`.
Consequently a common nonzero phase velocity across the **whole** connected
network is impossible under these premises. A regional common velocity can
be compensated by the retained environment. Neither this balance nor an
admitted finite relative rhythm establishes a universal permanent pulse.

### Shared admission and boundary of the result

[`assess_sine_conservative_handoff`](../../src/tnfr/physics/relational_sine_entry.py)
and `SineConservativeHandoff` share the admitted full-state source and
global enclosure with the entry owner. The theorem concerns the declared
conservative ramp class on unchanged support. Its sufficient inequalities
may fail without deciding another preparation. The existing future-work
retention theorem remains valid; this result identifies a class whose
actual work violates its premise after a genuinely acquired acute entry.
No damping, contact removal, prescribed feedback, autonomous preparation,
indefinite-stability claim or physical identification is added.

## 42. A larger cycle sector gives a global conservative barrier

<a id="sine-cycle-sector-barrier"></a>

The flat-receiver phase-path bound in Section 36 has a stronger geometric
form. It applies to any source outside a specified unit-winding sector,
including nonuniform initial phases with zero winding. This is a direct
boundary argument for the existing phase potential; it introduces neither
a changed law nor a new initial preparation.

### The boundary minimum is exactly `7/2`

On an oriented C5 let `delta_i` be its principal edge gaps and
`W=(sum_i delta_i)/(2pi)`. For `sigma=+1` or `-1`, define

\[
\Omega_\sigma=
 \{W=\sigma,\quad |\delta_i|<2\pi/3\text{ for every edge}\},
\qquad V_R=\sum_i U(\delta_i),\quad U(s)=1-\cos s.
\]

Every fully acute state of winding `sigma` lies in `Omega_sigma`.
The sector contains no principal-branch seam: a path leaving it first
reaches a gap of magnitude `2pi/3`. Reflection reduces the proof to
`sigma=+1`, with `sum delta_i=2pi`.

The following tangent inequality holds on the **whole** required interval:

\[
U(s)\ge U(\pi/3)+\sin(\pi/3)(s-\pi/3),
\qquad -2\pi/3\le s\le2\pi/3.
\]

Indeed `sin(s)-sin(pi/3)` is nonpositive to the left of `pi/3`
and nonnegative from there to `2pi/3`. Integrating this derivative
difference on either side proves the inequality without assuming that
`U` is convex on the entire interval.

If one boundary gap is `+2pi/3`, the other four sum to `4pi/3`.
Summing their tangent bounds cancels the linear terms and gives

\[
V_R\ge U(2\pi/3)+4U(\pi/3)
       =\frac32+4\cdot\frac12=\frac72.
\]

Equality occurs at the five permutations of
`(2pi/3,pi/3,pi/3,pi/3,pi/3)`. All five edge sine currents agree there,
so these are actual critical cycle configurations, rather than a bound
from independently incompatible gaps. If a boundary gap is `-2pi/3`,
the other four must all equal `+2pi/3`, giving `V_R=15/2` instead.
Consequently the minimum over the complete sector boundary is exactly

\[
\boxed{\min_{\partial\Omega_\sigma}V_R=\frac72.}
\]

This does not replace the smaller **acute-face** minimum
`B=5-4cos(3pi/8)`, or the minimum principal-branch seam cost
`6-2sqrt(2)`. They concern different boundaries. In particular,
`Omega_sigma` contains nonacute states; its protection does not prove
that every regional edge remains acute.

### A conserved full budget separates the sector in both directions

Retain a finite connected unit-support graph containing the receiver C5,
unit held capacities, `e=0,w=beta=1`, no inputs or events, and
`tau=t/pi`. The existing full law conserves

\[
H=\frac12\sum_{\{i,j\}}(x_i-x_j)^2+
  \sum_{\{i,j\}}[1-\cos(\theta_j-\theta_i)]\ge V_R.
\]

The smooth finite-dimensional flow exists in both time directions: bounded
form rates preclude finite-time form blowup, and the linear phase row then
has bounded rates on every finite interval. If `H<7/2`, continuity and the
boundary minimum give two conclusions for all real time:

- A receiver initially in `Omega_sigma` remains there with winding `sigma`.
- A receiver initially outside `Omega_sigma` cannot enter it, and therefore
  cannot become an acute unit-winding state of that sign.

The same finite-time conclusions hold at `H=7/2`. At a sector boundary,
equality in the phase minimum requires one principal gap `sigma*2pi/3`
and the other four `sigma*pi/3`. Indeed the tangent inequality above is
strict away from `pi/3` for each of those four remaining gaps. Equality
in `H>=V_R` also forces every full-edge form difference to zero and every
non-cycle edge phase difference to zero modulo `2pi`. On connected support,
the form is therefore a common constant, which need not be zero. The equal
oriented cycle sine currents cancel at every receiver, and all other sine
currents vanish. Both full rows vanish: such a state is an equilibrium.
If additional edges make these simultaneous equalities impossible, there
is no boundary state at this budget. Otherwise uniqueness of the smooth
flow prevents a distinct orbit from reaching that equilibrium at a finite
time. A boundary equilibrium itself remains there. Thus `H<=7/2` excludes
finite crossing in either direction; an asymptotic approach is a different
question.

An initial `W=0` receiver is outside both sectors. Thus a conservative
source with that winding and `H<=7/2` cannot acquire either acute orientation,
even if its initial phases are not equal. More generally any continuous
phase path from outside to an acute endpoint must have `max V_R>=7/2`.
For an actual finite-time orbit of this complete law, the necessary full
budget is strictly `H>7/2`, not equality. The geometric phase-path bound
alone does not imply that stronger dynamical conclusion.
Using the original-time channel rates of Section 36, the corresponding
running phase-input condition over original time `[0,t_*]` is

\[
V_R(0)+\max_{0\le s\le t_*}\int_0^s[-J_R(v)+B_V(v)]\,dv
 \ge\frac72
\]

when the initial receiver is outside the target sector. Section 36's
flat-phase statement is its `V_R(0)=0` specialization. Full uncertainty boxes must
certify the storage bound and initial sector membership or exclusion,
rather than substituting a midpoint or cached winding label.

### Negative collective feedback can still lie below this barrier

Take Section 41's two matched contact-lag preparations with
`u=1`, `gamma=1/4` and `rho=1/10`. Both start with uniform receiver
form and phase, so `W_R=0`, and their exact full storage is

\[
H=\frac52+5-(1+4\cos(1/10))\cos(1/4).
\]

The alternating Taylor bounds
`C_6(z)=1-z^2/2+z^4/24-z^6/720 <= cos(z) <= C_4(z)` at these
two positive arguments give the rational enclosure
`2674/1000 < H < 2675/1000 < 7/2`. The barrier therefore excludes
acute winding `+1` and `-1` for **both** cyclic arrangements at all times.
The reflected source retains its independent symmetry obstruction; the
asymmetric ordering removes that obstruction but not this conserved-budget
one. Their negative collective transfer and positive initial internal
derivatives remain valid local results.

For the same uniform-form family and general declared `u,gamma,rho`,
a necessary budget condition for acute entry is

\[
\boxed{u^2>\frac25
 \left[\frac72-5+(1+4\cos\rho)\cos\gamma\right]}
\]

when the right side is positive. Equality does not permit finite-time
entry. This neither identifies a successful amplitude nor guarantees
oriented passage or maintenance after the budget obstruction disappears.

### Compatible conserved quantities still do not determine reachability

Section 40's reference with `h=1/50` has
`H_*=V_5+1/10>7/2`; this barrier does not exclude every zero-winding
source with that total storage. There is even a continuous geometric path
preserving those scalar invariants. Take receiver phases `theta_i=i*s`
for `i=0,...,4` and `0<=s<=2*pi/5`. Their potential is
`V_R(s)=4(1-cos(s))+1-cos(4s)`; its derivative
`8*sin(5s/2)*cos(3s/2)` changes sign at `s=pi/3`, where
`V_R=7/2`. Set each private leaf phase equal to its receiver phase,
and along that path choose

\[
u(s)=\sqrt{\frac25[H_*-V_R(s)]},\qquad
x_R(s)=\frac{u(s)}4\mathbf1,\qquad
x_Q(s)=-\frac{3u(s)}4\mathbf1.
\]

Then contact phase storage vanishes, `H=V_R+5u^2/2=H_*`, and the
degree-weighted form mean is zero. A common continuous phase shift fixes
the lifted weighted phase mean as well. The endpoint is the compatible
unit twist with `u=1/5`. This path is a **geometric compatibility witness**,
not a solution of the nodal equations. Its uniform, phase-consensus starting
state in fact has a rotationally invariant zero-winding evolution; scalar
compatibility alone cannot prescribe a different trajectory.

Nor can an outside trajectory enter the exact invariant reference family
at a finite time. A positive-width retention neighborhood has a separate
entry question. Its own correlated storage bounds may exclude a named
source even above `7/2`, as Section 41's `H=14201/4000` control and
Section 40's fixed small-error box demonstrate. When that energy test
passes, the two-sided finite retention estimate still places the earliest
first appearance before its certified backward interval. None of these
constraints supplies the missing directed autonomous handoff.

[`assess_sine_cycle_barrier`](../../src/tnfr/physics/relational_sine_regional.py)
re-admits the full source and computes the strict storage flag and the
closed `H<=7/2` finite-time gate separately
from each sector's membership. Other nodes, contacts and any receiver chords
remain in the law and full storage. An undecided principal branch or energy
bound can leave a conclusion unavailable; no report chooses a missing
preparation or executes a trajectory.

## 43. Fast contact motion can exclude finite receiver organization

<a id="sine-contact-averaging"></a>

The full storage barrier supplies a necessary lower resource bound, not a
monotone relation between supplied form contrast and acquisition. On the
same C5/private-leaf support, the actual contact phases can turn rapidly
enough that their accumulated sine currents stay small over a fixed
horizon. The following estimate retains the nonlinear receiver and every
leaf; it does not prescribe a periodic input or replace the environment
by a fitted average.

### The full rows bound the accumulated contact currents

Retain unit capacities, `e=0,w=beta=1`, no events or inputs, and
`tau=t/pi`. Initially let receiver forms be the constant `a_0`, leaf
forms the constant `b_0`, and receiver phases one common value `theta_0`.
The five initial contact phases may be arbitrary. Write
`u=a_0-b_0`, fix a horizon `T>0`, and define continuous contact lifts
and their integrated currents by

\[
\phi_i=\theta_{l_i}-\theta_{r_i},\qquad
I_i(\tau)=\int_0^\tau\sin\phi_i(s)\,ds.
\]

The full unit-capacity rows give `|x_i'|<=1` and
`|theta_i''|<=2`, hence `|phi_i''|<=4`. The uniform initial block
forms also give `phi_i'(0)=-4u/3`, independently of the initial
contact phases. If

\[
M=\frac43|u|-4T>0,
\]

each contact rate keeps its sign and satisfies `|phi_i'|>=M`
throughout `[0,T]`. Integration by parts using
`sin(phi)=-(cos(phi))'/phi'` then bounds every prefix:

\[
|I_i(\tau)|\le\frac2M+\frac{4\tau}{M^2}
 \le\epsilon_T:=\frac2M+\frac{4T}{M^2},
\qquad 0\le\tau\le T.
\]

This bound concerns the actual moving contacts. Their rates need not be
constant and their phases need not be acute; the smooth sine law supplies
the derivative bound everywhere.

### An exact change of variables retains the nonlinear receiver

Let `L_R` be the combinatorial Laplacian of C5 and `Id` the identity
matrix. Since `x_Q=b_0*1-I` exactly, use the corrected receiver
coordinates

\[
z=x_R-a_0\mathbf1-I/3,\qquad
q=\theta_R-\theta_0\mathbf1-u\tau\mathbf1/3.
\]

Here `I` is the five-vector of integrated contact currents, not a matrix.
Put `V(q)=sum_i[1-cos(q_(i+1)-q_i)]` and `g=grad V`; the receiver
internal sine-current vector is `-g`. Direct substitution into the full
degree-three receiver and degree-one leaf rows gives

\[
z'=-\frac13g(q),\qquad
q'=\frac13(L_R+\mathrm{Id})z+
    \frac19(L_R+4\mathrm{Id})I.
\]

No current has been discarded. Both `z(0)` and `q(0)` are zero.
For the nonnegative auxiliary function

\[
\mathcal E=\frac12z^\mathsf T(L_R+\mathrm{Id})z+V(q),
\]

the two autonomous terms cancel exactly:

\[
\mathcal E'=\frac19g^\mathsf T(L_R+4\mathrm{Id})I.
\]

This is a proof function, not the total storage or a new physical
reservoir. The cycle incidence matrix and
`sin^2(delta)<=2[1-cos(delta)]` imply `||g||^2<=8V`;
the same graph has `||L_R+4Id||_2<=8`. With
`||I||_2<=sqrt(5)*epsilon_T` it follows that

\[
|\mathcal E'|\le\frac{16\sqrt{10}}9
 \epsilon_T\sqrt{\mathcal E},\qquad
\boxed{V_R(\tau)\le\mathcal E(\tau)
 \le\frac{640}{81}\epsilon_T^2\tau^2.}
\]

The square-root integration starts from `E(0)=0`; applying it first
to `sqrt(E+eta)` and then taking `eta` down to zero justifies the
bound at zero without dividing by a vanishing quantity.

### A prospective finite exclusion despite a large total budget

The receiver starts flat. If

\[
\frac{640}{81}\epsilon_T^2T^2<\frac72,
\]

its entire phase path stays below the acute-entry barrier. It cannot
acquire either acute unit-winding orientation anywhere on `[0,T]`,
whatever the five initial contact phases or their cyclic ordering.
The original structural-time horizon is `pi*T`.

For the exact supplied values `u=20`, `T=1`, one has
`M=68/3`, `epsilon_T=111/1156` and
`(640/81)*(111/1156)^2<73/1000<7/2`. Yet its full initial
storage is at least `5u^2/2=1000`, because all additional contact
phase costs are nonnegative. This is an analytic family control,
without a numerical trajectory or a search over contact orderings.

The result is finite-horizon and preparation-specific. It establishes
neither damping nor absence of later acquisition, and it does not exclude
nonacute winding. If `M<=0` or the upper bound is too large, this
sufficient test is unavailable rather than evidence of formation. The
nonuniform environmental form ramps in Section 37 do not have these
uniform block forms and are outside the theorem. Combining this estimate
with Section 42 removes some low-budget and rapid-contact preparations;
it selects neither a successful intermediate source nor its occurrence.

[`assess_sine_contact_averaging`](../../src/tnfr/physics/relational_sine_partition.py)
rebuilds the admitted C5/private-leaf source, verifies its uniform block
forms and flat receiver phase, and evaluates these finite-horizon bounds.
The detached reader preserves the actual source and full storage; it does
not integrate the source or change its form contrast to make the test pass.

## 44. A distributed preparation has a robust full-state passage

<a id="robust-conservative-passage"></a>

The finite acute-entry construction of Section 37 can be joined to an
explicit nonzero-width full-state retention set. This answers an existence
question for the same distributed-form preparation class: initially absent
winding can be acquired and then retained over a declared short interval,
with independent perturbations of every form and phase coordinate. It does
not require the particular moving-reference neighborhood of Section 40.
Nor does it establish prolonged maintenance or solve Section 41's distinct
uniform-form, ordered-contact-phase preparation question.

### Direct full-row bounds compose into an independent target box

Retain the conservative unit law and clock `tau=t/pi`. On the complete
fixed support write `A=D^(-1)L`, so the actual rows are
`x'=D^(-1)S(theta)` and `theta'=Ax`. For every state, the normalized
rows give

\[
|x_i'|\le1,\qquad |\theta_i''|=|(Ax')_i|\le2,
\qquad \|A\|_\infty=2.
\]

Supply nominal initial data `(x^0,theta^0)` and independent coordinate
errors bounded by `epsilon>=0` in both form and continuously lifted
phase. These bounds hold at every node, including the environment.
All members share the stated fixed capacity, support and complete law.
For their actual solutions,

\[
\boxed{
|x_i(\tau)-x_i^0|\le\epsilon+\tau,\qquad
|\theta_i(\tau)-\theta_i^0-\tau(Ax^0)_i|
 \le\epsilon(1+2\tau)+\tau^2,
\quad \tau\ge0.}
\]

The second inequality uses an initial phase-rate error at most
`2epsilon` and the global acceleration bound. It is not a linearized
trajectory or a truncated local jet; its remainder bounds the complete
nonlinear rows for the stated interval. No small-time or exponential
estimate is needed to derive it.

Choose a checkpoint `a>0` and a later endpoint `b>a`, with `h=b-a`.
Every actual checkpoint belongs to the explicitly supplied box centered at

\[
x^*=x^0,\qquad \theta^*=\theta^0+aAx^0,
\]

with per-node form and phase radii

\[
r_x=a+\epsilon,\qquad r_\theta=a^2+\epsilon(1+2a).
\]

Consider now **any** independent member of that target box, not only those
already shown to be reachable from the source. Evolving the same full law
for `0<=s<=h` gives

\[
|\theta_i(a+s)-\theta_i^0-(a+s)(Ax^0)_i|
 \le r_\theta+2sr_x+s^2,
\qquad |x_i(a+s)-x_i^0|\le r_x+s.
\]

This follows by admitting its initial form and phase deviations at the
checkpoint and applying the same acceleration bound afresh. In particular,
the whole-interval edge-gap error is bounded by

\[
\boxed{q=2(r_\theta+2hr_x+h^2)
       =2[b^2+\epsilon(1+2b)].}
\]

Thus the direct source enclosure and the independent target continuation
compose exactly at this level of bounds. If the nominal affine edge gaps
on `[a,b]`, enlarged by `[-q,q]`, lie strictly within one declared acute
winding sector, all source members reach that sector and all target members
retain it over the subsequent interval. The target is a sufficient full-state
set; it does not assert that every one of its points is reached from the
smaller source set.

### An exact distributed source and a fixed short window

Use the unchanged C5/private-leaf graph, with receiver nodes `i=0,...,4`
and corresponding leaves `5+i`. The nominal source is Section 37's ramp

\[
x_i^0=0,\qquad x_{5+i}^0=-3\omega(i-2),\qquad
\theta_i^0=0\text{ at all ten nodes},\qquad \omega=64.
\]

Supply independent source radii and times

\[
\epsilon=\frac1{4096},\qquad
a=\frac{11}{512},\qquad b=\frac3{128},\qquad
h=\frac1{512}.
\]

Every source member initially has receiver winding zero: each initial raw
receiver gap has magnitude at most `2epsilon=1/2048<pi`, and their sum
telescopes to zero. The nominal full-support phase rates are

\[
(Ax^0)_i=\omega(i-2),\qquad
(Ax^0)_{5+i}=-3\omega(i-2).
\]

The target center therefore retains ten explicit rational form values and
ten rational phase values; no response supplies that center. Its independent
coordinate radii are

\[
r_x=\frac{89}{4096},\qquad
r_\theta=\frac{751}{1048576},\qquad
q=\frac{211}{131072}<\frac1{500}.
\]

On `[a,b]` the four forward affine receiver gaps range from `11/8` to
`3/2`; the closing raw gap ranges from `-6` to `-11/2`. After the single
closing-edge turn correction, the whole target-box continuation satisfies

\[
\begin{aligned}
\delta_0,\ldots,\delta_3
 &\in[11/8-q,\ 3/2+q],\\
\delta_4&\in[2\pi-6-q,\ 2\pi-11/2+q].
\end{aligned}
\]

All five intervals lie strictly in `(0,pi/2)`. For example the elementary
bounds `157/50<pi<22/7` give an acute margin greater than `1/16`;
the limiting forward-gap margin is already greater than
`7/100-q=224101/3276800`. The raw gaps telescope, so their principal
sum is exactly `2pi`, and the receiver has winding `+1` throughout.
This is a whole-interval conclusion for every independent target member,
not a collection of sampled windings.

The target also bounds the receiver's internal form. Each cycle form
contrast has magnitude at most `2(r_x+h)`, hence

\[
F_R\le\frac52[2(r_x+h)]^2
     =\frac{47045}{8388608}<\frac6{1000}.
\]

The original structural-time checkpoint is `pi*a` and the guaranteed
subsequent duration is `pi/512`. Full environmental form, phase, storage
and weighted means remain those of each actual trajectory. No controller,
support event, damping or removed environmental coordinate performs the
handoff.

### What this passage establishes and what it does not

This proves robust finite acquisition followed by a separately admitted
full-state retention window. The source and target both have positive
width in all twenty form/phase coordinates. It makes the finite existence
claim quantitative rather than relying on unspecified continuous dependence.

The interval is deliberately short and the distributed environment is
strongly prepared. Its nominal full storage is `45omega^2=184320`,
and the source box retains the corresponding perturbed budgets. The
regional phase storage need not lie below the acute-face barrier `B`;
this is not Section 35's low-regional-storage work certificate. The rapid
relative phase rates have not disappeared, and no extended lifetime,
attraction, autonomous preparation or physical identification is established.

Section 40's small moving-family neighborhood is an optional sufficient
maintenance target, not a definition of every coherent identity. Requiring
entry into that particular neighborhood would be an additional research
question. Conversely this finite passage does not settle a prescribed
longer duration, a lifetime that does not shrink with the preparation's
rapid transport scale, or the separate ordered-contact-phase source class.
Those stronger questions need their own target and acceptance conditions;
they do not invalidate the positive bounded result here.

The existing
[`certify_sine_conservative_winding_entry`](../../src/tnfr/physics/relational_sine_entry.py)
owns this source-error and checkpoint-box extension. Its optional
`source_error_bound` retains a nominal admitted source with full-coordinate
uncertainty and exposes the composed phase enclosure and form bound.
The analytic radii above describe exact real boxes. If interval construction
rounds their endpoints outward, the implementation measures the resulting
effective radii about the exact centers and uses those in its separate
entry-box continuation bounds. It does not promote a rounded display into
an exact-radius proof. The original source-trajectory bounds retain their
own tighter calculation; the explicit dyadic witness needs no such enlargement.
It remains an analytic certificate, not a numerical trajectory or evidence
that a source is selected autonomously.

The full rows identify the information needed for a stronger retention
question. Let `P` subtract the five-node receiver mean, `q_R=P x_R`,
and `r=x_R-x_Q` in matched receiver/leaf order. The actual relative
receiver phase rate is

\[
P=I_5-\frac15\mathbf1\mathbf1^\mathsf T,\qquad
\boxed{v=P\theta_R'=\frac13\bigl(L_Rq_R+Pr\bigr).}
\]

Small internal form or phase storage does not control the environmental
contrast `Pr`. The ramp has `q_R=0` initially but supplies the large
relative rates `v_i=omega(i-2)`. Cancellation between the two terms can
reduce relative motion, but an instantaneous cancellation does not prove
its continuation under the full law. Longer retention therefore requires
bounds on their actual accumulated phase transport and regional work over
the declared interval. A moving compatible family is one sufficient tool
for those bounds; it is not a necessary definition of identity. Any longer
prospective duration is an evaluation requirement, not a new physical
constant or a change to the finite claim proved above.
