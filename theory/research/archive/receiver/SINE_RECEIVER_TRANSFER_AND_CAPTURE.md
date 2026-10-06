# Eleven-node transfer conditions and endpoint capture

**Archived preparation-specific study.** The eleven-node receiver campaign is deferred. Its conditional proofs, negative results and unresolved transfer range remain reusable under their stated assumptions; this document assigns no current task.

Receiver port-work and passage obligations, source-specific exclusion, complete equilibrium classification, and donor-well capture for the same retained eleven-node case study.

Part of [Native pattern reduction and memory](../../../nodal/RELATIONAL_PATTERN_MEMORY.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="sine-receiver-transfer-admission"></a>
### Receiver identity transfer requires donor loss and its own passage proof

Return to the original effective coefficients `e=w=1/2`, `beta=1`,
`delta=1/2` and structural clock. Keep all eleven nodes, their held
capacities, the initial phases, arbitrary donor forms `D_0,...,D_4`,
intermediary form `H`, zero receiver forms and `F<=4` unchanged.
The proposed endpoint is now **transfer**, not the excluded coexistence:
the donor is flat, the receiver has winding `+1`, both bridge phase gaps
vanish, and form is uniform. No support event, input or replacement law
is supplied.

#### Exact endpoint, conserved origins and full-state recovery

In turns, choose the reference

\[
v^{\rm tr}=(0,0,0,0,0,\ 0,1/5,2/5,3/5,4/5,\ 0),
\qquad \theta^{\rm tr}=2\pi v^{\rm tr}.
\]

The donor and bridge sine terms vanish individually. The receiver's two
ring terms cancel at every node, including its port. Thus `S=0` on the
full graph; uniform form gives `q=0`. This is an exact equilibrium of
both complete rows. Its phase Hessian assigns weight one to donor and
bridge edges, and weight `cos(2*pi/5)>0` to receiver edges. It is positive
on the full common-phase quotient, including the intermediary and
relative movement of the two rings. The held capacity asymmetry does
not remove that property.

For this capacity vector, the conserved-coordinate weights are

\[
\omega=(3,2,2,2,2,\ 3,4/3,2,2,4,\ 2),\qquad
M_\omega:=\sum_i\omega_i=\frac{76}{3}.
\]

Section 18 therefore fixes the final uniform form to

\[
x_\infty=M_x=\frac3{76}
\left[3D_0+2(D_1+D_2+D_3+D_4)+2H\right].
\]

The initial weighted phase-turn sum is `4`, whereas the displayed
target representative has weighted turn sum `82/15`. If the final
continuous lifts have the form
`theta_i/(2*pi)=v_i^tr+gamma+m_i`, with integers `m_i`, conservation gives

\[
\gamma=-\frac{11}{190}
       -\frac3{76}\sum_i\omega_i m_i.
\]

In particular, the representative with no additional integer turns
has common shift `-11/190` turns. The actual integer vector is a property
of the lifted history, not prescribed by a snapshot or this endpoint
test. The circular target admits the resulting common origin; no
inconsistent fixed phase mean is imposed. This compatibility does not
produce a trajectory realizing that history.

The existing whole-support recovery theorem in Section 18 applies to
this target. For example, retain a consistent local deviation lift,
remove only the two common origins, and let

\[
Z^2=\|P(x-M_x\mathbf1)\|^2+
\|P(\theta-\theta^{\rm tr})\|^2,\qquad
P=I-\mathbf1\mathbf1^\mathsf T/11.
\]

The same maximum target edge angle `2*pi/5` and full-support gap
`lambda_2(L)>=1/11` permit `r=1/16` and `c_r>1/5`. Consequently the
following strict conditions are sufficient for trapping and recovery:

\[
\boxed{Z<\frac1{16},\qquad E-V_5<\frac1{28160}.}
\]

For an uncertainty set both upper bounds must hold uniformly, with all
eleven forms and phases retained. This is a nonempty basin around the
exact target, not a certificate that the original preparation enters
it. Its small radius must not be replaced by a receiver-only norm.

#### Endpoint budgets permit transfer but do not establish it

At the preparation and transfer target, respectively,

\[
\begin{array}{c|cc}
&\text{initial}&\text{transfer target}\\ \hline
E&F+V_5&V_5\\
\mathcal W&\frac45F+V_5&V_5
\end{array}
\]

Both endpoints have exact `S=0`; the target also has `q=0`. Hence the
coexistence obstruction's positive target gap is absent: the transfer
gap in `W` is `-(4/5)*F`. If convergence occurred, the full accumulated
continuous loss would equal `F`. Storage released while the donor
unwinds remains inside this same balance; it is not an external input
or a reusable loss reservoir. For `F>0`, these two endpoint inequalities
are compatible with transfer, but do not prove it.

#### A phase passage barrier and exact excluded controls

Define the target phase region by **donor strictly acute with winding
zero and receiver strictly acute with winding one**. Bridge phases and
forms are not constrained in this definition. Any convergence to the
transfer target enters this open product region at a finite time.
At the positive infimum `T` of entry times, both rings are closed acute
with those windings, and at least one is on an acute face.

If the receiver is on a face, the previous convexity proof gives phase
storage at least `B_5`. If the donor is on a face, its edge at `pi/2`
or `-pi/2` costs one, while the closed-acute winding-one receiver costs
at least `V_5`. Since `1+V_5>B_5` (`V_5>3` and `B_5<4` suffice), every
such first entry satisfies

\[
E(T)\ge B_5,\qquad
D_{\rm loss}(T)=E(0)-E(T)\le
A_{\rm tr}:=F+V_5-B_5.
\]

There is no added `V_5` from a maintained donor at this boundary.
Reusing the coexistence allowance `F-B_5` would reject this different
endpoint for the wrong reason.

For `F>0`, the connected support and zero receiver forms imply
`q(0)!=0`. Loss is strictly positive on an initial interval, so transfer
requires the strict condition

\[
\boxed{F>B_5-V_5.}
\]

This also follows by considering the first exit from the initial product
region (acute donor winding one, acute receiver winding zero): a donor
face costs at least `B_5`, whereas a receiver face together with the
remaining closed-acute donor costs at least `1+V_5>B_5`. Consequently
`F<=B_5-V_5` cannot leave that initial product region. At equality the
strict initial loss prevents the boundary from being reached. When
`F=0`, both full gradients vanish and the preparation is stationary.

The exact silent subspace proved earlier,
`D_0=H=0`, `D_1=-D_4`, `D_2=-D_3`, remains an independent exclusion,
including its nonzero-storage members: its receiver stays flat forever.
Nonzero loss or satisfaction of the endpoint budgets does not remove
that symmetry obstruction.

#### Two necessary winding changes consume disjoint parts of one loss

Before the target-product entry at `T`, the receiver must cross an
antipodal edge, and the donor must also cross one in leaving winding
one. Let corresponding crossing times be `s_R,s_D` with
`0<s_R,s_D<T`. They need not have any prescribed order and may coincide.
An initially flat receiver edge requires at least `pi` displacement of
its continuous lifted phase difference. A donor edge initially has
increment `alpha=2*pi/5` modulo `2*pi`, so any antipodal value requires
at least `pi-alpha=3*pi/5` displacement. This statement also covers
the closing edge's initial raw lift.

For a ring `C`, allocate only its actual nodal loss

\[
D_C(t)=e\int_0^t\sum_{i\in C}k_iq_i(s)^2\,ds,
\qquad k_i=\nu_i/d_i.
\]

For its edge `i--j`, weighted Cauchy--Schwarz gives

\[
|\theta_j(s)-\theta_i(s)-\theta_j(0)+\theta_i(0)|^2
\le\frac{b^2(k_i+k_j)}e\,sD_C(s).
\]

The actual maximum edge mobilities are `k_D,max=1` on the donor and
`k_R,max=5/4` on the receiver. Their node sets are disjoint, so
`D_D(s_D)+D_R(s_R)<=D_loss(T)`: intermediary loss is omitted, not
double-counted. Using the strict inequalities `s_D,s_R<T` therefore yields

\[
D_{\rm loss}(T)
>\frac{e\pi^2}{b^2T}
  \left(\frac{9}{25}+\frac45\right)
 =\frac{29e\pi^2}{25b^2T},
\]
\[
\boxed{T
>\frac{29e\pi^2}{25b^2 A_{\rm tr}}
 =\frac{58\pi^4}{25(F+V_5-B_5)}.}
\]

The bound requires `A_tr>0` and is expressed in the original clock.
It is a necessary earliest-entry condition, not a predicted transfer
time. It neither assumes that the donor unwinds first nor adds two
estimates of the same dissipated quantity. It uses the full support
through each `q_i`, even when the loss is partitioned by ring.

A certified source-specific loss lower bound `L(tau)` can also be
reused without copying the coexistence verdict. If the admitted horizon
`tau` is strictly below the transfer entry-time lower bound, and
`L(tau)>A_tr`, entry is excluded both before `tau` by the action estimate
and after `tau` by nondecreasing accumulated loss. The deficit must be
recomputed against `A_tr`. Independent lower bounds on the same total
loss combine by their maximum, not by addition; this differs from adding
the disjoint nodal contributions in the proof above.

#### What remains undecided

The target is an exact, locally recoverable collective state compatible
with the conserved coordinates. The low-budget, zero-form and silent
controls exclude specified preparations, and the joint action estimate
constrains every remaining successful path. These results do **not**
classify accessibility for the rest of the six-coordinate class `F<=4`.
The [static donor well](#sine-donor-well-retention) and
[dissipative capture](#sine-donor-dissipative-capture) select the original
donor endpoint on their stated source classes. Outside those certificates,
the [receiver port-work and order conditions](#sine-receiver-port-passage)
remain necessary restrictions, not a classification of accessibility.
In particular, a first visit to the larger acute product region need not
satisfy the full-state norm and excess-storage bounds of the recovery
basin, and local receiver derivatives do not bound that later state.

The missing positive argument is an actual evolution from an admitted
preparation into that whole-support basin. A negative argument instead
needs a valid separating invariant or a loss/action bound ruling out
every necessary passage for the remaining preparations. The endpoint
inequalities and existing finite loss estimates supply neither claim
uniformly. No trajectory, new preparation, parameter search or changed
law is inferred from this gap. The maintained coexistence exclusion
remains intact; transfer cannot be counted as replication of a second
maintained organization.

A static control makes this limitation more precise. First shrink the
initial form deviations continuously toward the conserved uniform form
`M_x*1`, keeping the initial phases fixed. Then flatten the donor along
`theta_j=j*t`, from `t=2*pi/5` to zero, and twist the receiver along the
same family from zero to `2*pi/5`, keeping both ports and the
intermediary aligned. At each phase step a single common phase shift
can preserve the weighted lifted mean. Thus this comparison path respects
both conserved means and the supplied support. Its first leg decreases
`W` from `(4/5)*F+V_5` to `V_5`. On either phase leg, uniform form gives

\[
\mathcal W=V=5-4\cos t-\cos4t,\qquad
\frac{dV}{dt}=8\sin(5t/2)\cos(3t/2),\qquad
0\le t\le2\pi/5.
\]

The maximum along these declared legs is `7/2`, at `t=pi/3`. Therefore
for `F>=(5/4)*(7/2-V_5)` the source and transfer target lie on a continuous
path inside `W<=W(0)`, even with both conserved coordinates retained.
The [critical-set argument below](#sine-donor-well-retention) proves that
`7/2` is the exact static minimax between these two equilibrium wells.
The displayed path is still not a dynamics construction: `W` increases
on parts of it, which cannot occur on the actual flow. In particular,
sufficiently energetic members of the silent
subspace also satisfy this connectivity test while remaining exactly
unable to transfer. Connectivity of this auxiliary sublevel is therefore
insufficient; the missing dynamical direction cannot be supplied by
static endpoint and budget checks.

The existing
[`SineMediatedFormation.receiver_transfer()`](../../../../src/tnfr/physics/relational_sine_formation.py)
reader reuses the admitted preparation and horizon while preserving the
original coexistence report. Its separate `SineReceiverTransferAdmission`
retains exact target geometry, a compatible lifted origin, the
source-minus-target margins `F` and `(4/5)*F`, the entry allowance and
both actual edge mobilities. The necessary time uses the shared outward
interval calculation; a missing positive allowance leaves that time
unavailable. These necessary inequalities do not consume the research
ceiling `F<=4`, so their reader does not impose an artificial budget cap.
A passing necessary-condition status is not a successful
transfer or an initial/evolved recovery-basin certificate. The
[formation controls](../../../../tests/physics/test_relational_sine_formation.py)
check those distinct target and numerical admission boundaries, and the
[independent algebra controls](../../../../tests/physics/test_relational_sine_formation_barrier.py)
verify the target, conserved-origin calculation, disjoint action constants
and static comparison path. None advances a transfer trajectory.

<a id="sine-receiver-port-passage"></a>
### Receiver acquisition requires accumulated port work and a geometric passage

Keep the same supplied two-C5/intermediary support and the six-coordinate
preparation: donor winding `+1`, receiver flat, intermediary aligned,
and receiver form initially zero. The following identities and necessary
conditions hold for the complete sine law with `e,w,beta>0` and
strictly positive held capacities, without a restriction on `w/e`.
The numerical specialization retains the original half weights,
`beta=1`, `delta=1/2` and `F<=4`. No external drive, input law or
event is added by treating the receiver boundary as a port.

#### Use the actual full-node loss and the existing boundary-work convention

Let `R={5,...,9}`, with port `p=5` and intermediary `h=10`.
Let `L_R` be the isolated receiver-cycle Laplacian only for defining
its **internal storage**:

\[
E_R=F_R+\beta V_R,\qquad
F_R=\frac12x_R^\mathsf TL_Rx_R,\qquad
V_R=\sum_{\{i,j\}\in C_5^R}(1-\cos(\theta_j-\theta_i)).
\]

Its consumed rates still use the full eleven-node `q=Lx`, `S` and
`k_i=nu_i/d_i`; in particular the port degree remains three. Define

\[
f=x_h-x_p,\qquad s=\sin(\theta_h-\theta_p),\qquad
P_R=f\dot x_p+\beta s\dot\theta_p,
\]
\[
D_R(t)=e\int_0^t\sum_{i\in R}k_iq_i(u)^2\,du.
\]

The receiver's internal form gradient is `(q_i)_(i in R)+f*e_p`,
and `grad V_R=-(S_i)_(i in R)+s*e_p`, with both restricted vectors
retaining their **full-support** values. Therefore

\[
\begin{aligned}
\dot E_R
&=\sum_{i\in R}(q_i\dot x_i-\beta S_i\dot\theta_i)
  +f\dot x_p+\beta s\dot\theta_p\\
&=-e\sum_{i\in R}k_iq_i^2+P_R,
\end{aligned}
\]

using `a=w/pi=beta*b`. Since `E_R(0)=0`, the exact integrated
balance is

\[
\boxed{J_R(t):=\int_0^tP_R(u)\,du=E_R(t)+D_R(t).}
\]

This uses the existing intermediary-star boundary-work sign convention:
`P_R` is the **negative** of the port-5 work rate into the hidden star
in the [mediated-pressure balance](../../../nodal/SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure).
The form and phase contributions are both necessary. It is not the
weighted-form flux reported by a regional form-transfer observation.
No second regional pressure or dissipation convention is introduced.

Although `J_R(t)>=0` for this zero-storage preparation, neither its
derivative `P_R` nor its increments need be nonnegative. Work can return
through the same port. Likewise a positive instantaneous boundary power
does not prove an increase of internal receiver storage: the simultaneous
full-node loss must be subtracted.
For the actual preparation, intermediary form `H` gives the exact control
`P_R(0)=D_R'(0)=e*H^2/3` and hence `E_R'(0)=0`. Positive incoming
power at the initial port is then spent entirely on simultaneous loss.

#### The exact receiver barrier demands a running maximum, not final work

For the relative phase torus of an isolated C5, the exact minimum
possible maximum of `V_R` along a continuous path from consensus to
either uniform winding-one twist is `7/2`. This reuses the
[critical-set and well-separation proof](#sine-donor-well-retention):
the twist is an isolated minimum of value `V_5`, and there is no
critical value between `V_5` and the first saddle value `7/2`.
Regular sublevel components cannot merge in that interval. The path
`theta_j=j*t`, `0<=t<=2*pi/5`, attains maximum `7/2`; reflection
gives the opposite handedness. This is an exact phase-geometry minimax,
not a sharp dynamical work threshold or a newly selected law.

Any actual trajectory converging to either receiver twist must therefore
have a finite first time `tau_R>0` with `V_R(tau_R)=7/2`.
To make the finite-time implication explicit, take a sufficiently late
phase state near the limiting twist. A short path from it to the exact
twist stays in that minimum's sublevel component below `7/2`.
Appending it to the actual receiver phase path proves that the barrier
was already crossed earlier. An arbitrary transient nonzero winding,
without convergence to this maintained geometry, is a different claim.

At this first barrier,

\[
\boxed{J_R(\tau_R)
=\frac72\beta+F_R(\tau_R)+D_R(\tau_R)
>\frac72\beta.}
\]

Strictness follows from the actual phase row: if `D_R(tau_R)=0`,
positivity and continuity force `q_i(t)=0` at every receiver node
throughout `[0,tau_R]`. Then `dot theta_i=b*k_i*q_i=0` there, so
the receiver phase remains flat and cannot reach the barrier. This
argument also covers preparations with zero intermediary form but a
later donor response; it does not assume positive initial receiver loss.

The condition is on a **running** signed work integral. At a maintained
twisted limit, `F_R` tends to zero and
`J_R(infinity)=beta*V_5+D_R(infinity)` by the same balance. Those
endpoint facts do not imply that this final value exceeds `7*beta/2`:
the receiver can return work after its barrier passage. A final-work
budget cannot replace the required transient maximum.

There is also a quantitative receiver-action cost. Let `lambda_R>0`
be any justified upper bound for the spectrum of
`K_R^(1/2)L_RK_R^(1/2)`. From the initial flat phase, use its
continuous lift `eta_R=theta_R(tau)-theta_R(0)` and `H_R<=L_R`:

\[
V_R(\tau)\le\frac12\eta_R^\mathsf TL_R\eta_R
\le\frac{b^2\lambda_R\tau}{2e}D_R(\tau).
\]

The second inequality follows by inserting
`eta_R=b*K_R^(1/2)*integral_0^tau(K_R^(1/2)*q_R)dt` and applying
Cauchy--Schwarz in time; here `q_R` is the restriction of the full
`q=Lx`, never the internal gradient `L_R*x_R`. Thus at the first barrier

\[
D_R(\tau_R)\ge\frac{7e}{b^2\lambda_R\tau_R},\qquad
J_R(\tau_R)\ge\frac72\beta+
                 \frac{7e}{b^2\lambda_R\tau_R}.
\]

For the original capacities, `K_R^(1/2)L_RK_R^(1/2)` is bounded
above by the receiver principal compression of the full `B`; that
compression additionally retains the bridge contribution on the port
diagonal. Thus `lambda_R<=3`, and at half weights this gives
`D_R(tau_R)>=14*pi^2/(3*tau_R)`. The storage scale enters the
phase rate through `b=w/(beta*pi)` and the barrier through `beta`;
there is no additional factor of `beta` in the unit-phase inequality.
This first potential-barrier action cost is distinct from the earlier
antipodal winding-change estimate. The unknown `tau_R` is not a
forecast, and this additional loss bound tends to zero as `tau_R`
tends to infinity.

#### A necessary order of potential barriers for the original budget range

Let `tau_D` be the donor's first time at `V_D=7/2`, or infinity
if that never occurs. Until that time its continuous phase stays in
the original one-twist component of `{V_D<7/2}`, where `V_D>=V_5`.
For every nonzero preparation, initial `q!=0` and positive held
capacities give strict global loss over an initial time interval.
Consequently `E(t)<E(0)=F+beta*V_5` for all positive `t`.
The zero-form preparation is stationary and has no receiver passage.

If `F<=7*beta/2` and `tau_R` is finite, it is impossible to have
`tau_D>=tau_R`: at the receiver barrier the donor would still have
`V_D>=V_5`, requiring `E>=beta*(7/2+V_5)`, contrary to that
strict energy inequality. Therefore

\[
\boxed{F\le\frac72\beta,\quad\tau_R<\infty
\quad\Longrightarrow\quad \tau_D<\tau_R.}
\]

Indeed energy at the receiver barrier gives the stronger pointwise
restriction `V_D(tau_R)<F/beta+V_5-7/2<=V_5`. The conclusion
concerns potential-well passage. It does **not** say that the donor
has already changed its wrapped winding, entered an acute consensus
chart, or selected its final equilibrium.

At any instant when both ring potentials are at least `7/2`, total
storage is at least `7*beta`. Thus a necessary condition for such a
simultaneous barrier state is

\[
\boxed{F>\beta(7-V_5)=\frac\beta4(3+5\sqrt5).}
\]

For `beta=1`, exclusion of simultaneous barrier states can use exact
rational source arithmetic: put `z=4*F-3`; exclusion holds when
`z<=0` or when `z>0` and `z^2<=125`. This is not exclusion of
sequential passages. In particular it cannot be substituted for a
receiver-transfer verdict on the upper part of the source class.

#### The remaining estimate is a bound on actual accumulated receiver supply

These results constrain a receiver-directed mechanism without proving
that it succeeds. A uniform bound `sup_t J_R(t)<=7*beta/2` would
exclude maintained receiver acquisition; more sharply, a bound on
`sup_t[J_R(t)-D_R(t)]` below the same barrier would do so. A
positive route still needs an actual full state entering the receiver
recovery basin, not only enough boundary work or a barrier crossing.

The current source controls provide neither such a running-supply upper
bound nor a successful basin-entry state for the remaining preparation
class. The instantaneous port owner fixes the correct sign and channels,
but does not integrate or bound their future history. Treating the donor
and intermediary signals as freely adjustable inputs would change this
autonomous problem. The ordering theorem, necessary action and exact
work balance do not fill that gap with a waveform, time cutoff, new
preparation or a replacement law.

The existing
[`comparison.mediated_pressure(mediator=10)`](../../../../src/tnfr/physics/relational_sine_mediation.py)
retains `port_boundary_form_work`, `port_boundary_phase_work` and
`port_boundary_work` in its actual `ports` order. Their orientation is
into the hidden star; the receiver's incoming rate is the negative of
the port-5 entry. These per-port arrays own the corresponding aggregate
sums and use full port rates. They provide a snapshot, not any of the
future accumulated-work quantities in this theorem.

The formation owner's existing
[`SineReceiverTransferAdmission`](../../../../src/tnfr/physics/relational_sine_formation.py)
retains the necessary barrier, running-work lower bound and the
conditional `donor_barrier_first_required` and
`simultaneous_barrier_passage_excluded` fields. Their public admission
keeps that reader's original half-weight, unit-beta, positive-half-contrast
scope, although the mathematical balances and ordering have the broader
positive-coefficient scope stated above. None of these necessary passage
fields changes the transfer verdict or records an observed future order.
Unsupported or missing fields remain unavailable, not zero supplied work.
The [mediated-pressure controls](../../../../tests/physics/test_relational_sine_mediated_pressure.py)
check channel signs and full-rate reuse; the
[formation controls](../../../../tests/physics/test_relational_sine_formation.py)
and [independent algebra controls](../../../../tests/physics/test_relational_sine_formation_barrier.py)
check the receiver balance, action factor and exact conditional thresholds.
No history integral, new solver or forced receiver experiment is evaluated.

<a id="sine-localized-receiver-exclusion"></a>
### A complete nonlinear prefix and dissipative tail can exclude receiver acquisition

Fix the original localized preparation, rather than selecting a new
profile after observing a response:

\[
x_{10}(0)=2,\qquad x_i(0)=0\ (i\ne10),\qquad
\theta_j(0)=2\pi j/5\ (0\le j\le4),
\]
\[
\theta_i(0)=0\ (5\le i\le10),\qquad
e=w=\frac12,\quad\beta=1,\quad\delta=\frac12.
\]

All eleven nodes, twelve edges and the original positive held capacities
remain present. In particular the receiver capacities at nodes 6 and 9
are `3/2` and `1/2`; the full port degrees include the bridges.
Here `F=4`, `N=32/3` and `E(0)=4+V_5`. No input, support event,
phase projection, prescribed mediator signal or replacement pressure
law is supplied. A tangent solution or a reduced reflection-symmetric
model does not certify this full nonlinear preparation.

The following is a **conditional proof rule** for a separately frozen
validated response. Its mathematical conclusion is not established by
the source or by naming a future horizon. Let `T>0` be that declared
horizon, and require a validated enclosure of the entire full-state
trajectory on `[0,T]` under the unchanged law.

#### The two enclosure obligations are different

For every accepted integration cell, its full nonlinear Picard tube must
enclose every intermediate state on that cell, not only its two endpoint
boxes. Evaluate the receiver's five-edge unit phase potential through
the shared circular interval kernels on each such tube, retaining all
edge-difference uncertainty. Let `M_R` be a certified upper bound over
the union of those tubes:

\[
V_R(t)\le M_R\quad\text{for every }0\le t\le T.
\]

Independently, let `U_E` be an outward upper bound on the **total**
storage at the validated endpoint, using every one of the twelve
edges and both fields in the full endpoint box:

\[
E(T)=\frac12\sum_{\{i,j\}}(x_i(T)-x_j(T))^2
 +\beta\sum_{\{i,j\}}[1-\cos(\theta_j(T)-\theta_i(T))]
 \le U_E.
\]

The endpoint expression cannot omit either bridge, replace the hidden
state by a midpoint or equilibrium, change the admitted held capacities,
or treat an independently propagated diagnostic as the actual full-state
storage. Correlated bounds can improve sharpness when justified; an
outward sum of actual edge enclosures already gives a valid sufficient
bound. The certificate requires the full frozen horizon to be reached.
A partial valid prefix is not a completed response.

The sufficient strict comparisons are

\[
\boxed{M_R<\frac72,\qquad U_E<\frac72\beta.}
\]

The fixed preparation has `beta=1`; the factor is displayed to make
clear the distinction between unit phase potential and scaled storage.
Neither equality is silently accepted as a strict safety margin.

#### Why these finite inequalities cover the whole future

The prefix bound supplies `V_R(t)<7/2` for all `0<=t<=T`.
For every later time, the exact full sine balance gives

\[
E(t)\le E(T)\le U_E<\frac72\beta,\qquad
0\le\beta V_R(t)\le E(t),\qquad t\ge T.
\]

The sine flow is globally defined under the retained finite-state,
fixed-support, positive-capacity premises. This tail argument uses its
storage dissipation with positive `e`, not a continuation of a numerical
stepper or an assumed relaxation rate. Therefore `V_R(t)<7/2`
for **every** future time, including the part not numerically integrated.
The continuous receiver phase path cannot leave its initial consensus
component of `{V_R<7/2}`. By the exact phase minimax proved above,
convergence to either maintained receiver twist is impossible.

There is a stronger conditional endpoint conclusion on this particular
support. The [complete asymptotic theorem](#sine-eleven-node-asymptotic-equilibria)
gives convergence to one full relative equilibrium. At that equilibrium
the bridge sine currents vanish and the receiver is a C5 critical
configuration. Its only critical configurations below `7/2` are
consensus and the two uniform twists. The uniform bound
`V_R(t)<=max(M_R,U_E/beta)<7/2` keeps the entire receiver path, and
its limit, in the initial consensus component; the two twist minima
belong to different components. Thus both strict checks also certify
**asymptotic receiver phase consensus**. The full form converges to
its conserved weighted mean by the same asymptotic theorem. This
does not select the donor geometry, bridge branches or common phase
origin, and does not assert zero wrapped receiver winding at every
intermediate time.

An endpoint with `E(T)<7*beta/2` alone would not suffice. The receiver
could already have crossed its barrier and entered a twist component
whose limiting value `beta*V_5` is below that same energy ceiling.
Conversely, a prefix safely below the receiver barrier does not control
its infinite future if the endpoint energy remains too high. The two
obligations cannot be substituted for one another.

#### Availability, failed bounds and actual passages remain distinct

If the complete prefix or its intermediate tubes are unavailable, there
is no all-time receiver exclusion from this rule. If the full prefix
is validated but an upper bound reaches or exceeds a threshold, that
comparison is inconclusive: it does not prove the actual trajectory
crossed the barrier. In particular, `M_R<7/2` together with
`U_E>=7*beta/2` leaves an unresolved late-time estimate. A solver
failure, an inconclusive enclosure and a demonstrated physical or
mathematical passage are different outcomes.

Only a completed response satisfying both strict tests certifies the
global exclusion. The protocol must retain the exact source, support,
law, clock, numerical budget and failed checks as well as successful
ones. Increasing a horizon or changing a preparation after seeing a
response is a separately declared evaluation, not completion of the
same frozen prediction. No finite receiver work record is invented by
this energy-tail route; it gives a sufficient upper bound on receiver
storage without replacing the exact boundary-work identity.

The [research reader](../../../../src/tnfr/research/relational_receiver_barrier.py)
keeps the preparation in `prepare_receiver_barrier`, consumes shared
full-state evidence through `assess_receiver_barrier_forecast`, and
evaluates the separately frozen response through `evaluate_receiver_barrier`.
Its `asymptotic_receiver_consensus_certified` flag requires both complete
prefix and strict endpoint checks, using the preceding conditional
corollary. Retained step coverage includes the exact held intermediary
capacity at every endpoint. A public forecast record still does not
authenticate its production; the frozen source and response retain that
provenance separately.

#### Evaluated result for the frozen localized preparation

The locally retained `artifacts/research/relational_receiver_barrier/`
contains `response-v1.protocol.json`, `response-v1.json` and
`response-v1.source.zip`; these evidence files are not distributed with the
documentation site. The protocol declared `T=32`, step `1/8`, Taylor order 16
and the shared outward dyadic-128 rational interval kernels before evaluation.
The retained response completed all 256 cells under that budget. Its exact rational bounds
give the following rounded displays:

\[
M_R\approx0.01848801719151926<0.018489<\frac72,
\]
\[
U_E\approx3.4768357092831135<3.476836<\frac72,\qquad
\frac72-U_E>0.023164.
\]

`M_R` is an upper bound on the entire validated prefix, not a measured
maximum and not the infinite-time upper bound. The later tail uses
`E(t)<=U_E`. Together these two certificates establish that this exact
localized source never reaches the receiver potential barrier and that
its receiver converges to phase consensus by the preceding corollary.
Neither receiver handedness target is reached asymptotically. The
proof does not select the donor's final geometry, a bridge branch or a
time at which wrapped winding must be zero.

The protocol SHA-256 is
`16f90bbfb534f684e7f80c36ab45dbe37f879ffc703c3160b660232346ceb90c`;
the archived producer source has SHA-256
`e334410abe3ccce14e96d5cd8d4a4a079a05bf3796ae8931d7eff5747c251a94`.
The response retains the actual full-state tubes, endpoint bounds,
declaration, runtime and source manifest. No shortened horizon, changed
preparation, reduced law or repeat response was substituted for the
frozen test. This is one source-specific nonlinear exclusion, not an
exclusion of every six-coordinate preparation with `F<=4` or of TNFR
pattern formation under other complete laws.

A later reader-admission correction rejects Boolean capacity values in
manually supplied forecast records. The actual frozen preparation uses exact
rational capacities and its checked enclosures are unchanged; the archived
producer and evaluated response were not regenerated for that correction.

#### Exact relative coordinates do not replace failed numerical evidence

If independent interval boxes lose useful correlations, a different
coordinate representation can be considered without changing the law.
Only the common form translation and common phase shift are removed
here. With reference node `h=10`, put

\[
u_i=x_i-x_h,\qquad v_i=\theta_i-\theta_h\quad(i\ne h),
\qquad u_h=v_h=0.
\]

For continuous phase lifts, the following twenty-coordinate system is
exact and globally closed:

\[
\begin{aligned}
q_i(u)&=\sum_{j\sim i}(u_i-u_j),&
S_i(v)&=\sum_{j\sim i}\sin(v_j-v_i),\\
\dot u_i&=-e(k_iq_i-k_hq_h)+a(k_iS_i-k_hS_h),&
\dot v_i&=b(k_iq_i-k_hq_h),\qquad i\ne h,
\end{aligned}
\]

where `k_i=nu_i/d_i`, `a=w/pi` and `b=w/(beta*pi)` retain
the actual support and held capacities. These are differences of the
complete fine rows, including loss; they are not a reflection-symmetric
approximation or a closure obtained by removing the intermediary.
They reuse the [reference-node quotient construction](../../../nodal/SINE_PAIR_MOBILITY.md#sine-mobility-relative-geometry),
with the present constant-mobility law's conserved means.

To reconstruct discarded origins, let `rho_i=1/k_i`,
`M=sum_i rho_i`, and retain the two initial invariants
`I_x=sum_i rho_i*x_i`, `I_theta=sum_i rho_i*theta_i` in their
declared continuous lift. Then

\[
x_h=\frac{I_x-\sum_{i\ne h}\rho_i u_i}{M},\qquad
\theta_h=\frac{I_\theta-\sum_{i\ne h}\rho_i v_i}{M},
\]

followed by `x_i=x_h+u_i`, `theta_i=theta_h+v_i`. Arbitrarily
wrapping each `v_i` discards the integer-lift information needed for
this real-phase reconstruction. Total storage and receiver phase
potential themselves depend only on the relative edge differences.
The held capacity is a parameter in these equations, not an additional
evolving coordinate.

This exact quotient can remove uncertainty in common origins, but it
does not guarantee a narrower interval enclosure: its coupled rate
differences can still produce box overestimation. The original
preparation supplies no donor reflection symmetry that removes further
constituent coordinates. A producer using this representation would
need its own declared layout, transformed initial enclosure and frozen
numerical budget. It cannot reinterpret an unavailable twenty-three-
coordinate prefix as a successful evaluation or replace the unchanged
whole-time and endpoint obligations above. The successful frozen result
uses the original full-state representation and does not require this
alternative numerical layout.

### Boundary of this result

The shared formation owner admits explicit donor form, computes
these full-support quadratic forms and derivative evidence,
and retains the supplied preparation and window. Its general
majorant is a sufficient analytic bound, with explicit numerical
availability and a required positive norm polynomial. An
unavailable bound is not a failed trajectory.

Its separate `maintained_target_obstruction` applies the auxiliary
coexistence certificate only under `beta=1`, `delta=1/2` and the exact
stored-coefficient condition `e>0`, `0<w/e<3/2`.
Its `exchange_to_loss_ratio` and
`sufficient_ratio_upper_bound` expose this admission without treating the
upper endpoint as a critical law constant. The mixed coefficient is
`h=w/(2*pi*e)`, not a separately configured physical parameter.
It evaluates the correlated margin `V_5-(4/5)*F`, rather than subtracting
independently widened initial and target intervals. A positive lower margin
certifies maintained-target exclusion; an unavailable or unresolved margin
does not certify formation. The phase-action and early-loss readers retain
their different transient-entry conclusions. Tests of these owners verify
the matrix premises and report boundaries; they do not replace the
global derivative argument. The
[independent whole-support algebra controls](../../../../tests/physics/test_relational_sine_formation_barrier.py)
differentiate the actual fine field, verify the path and spectral premises,
and check the uniform endpoint separation, ratio dependence and complete
constant-clock transformation. The
[formation-report controls](../../../../tests/physics/test_relational_sine_formation.py)
cover action admission, effective-coefficient model guards, unavailable
bounds and correlated maintained-target margins.

The entire six-coordinate class with `F<=4` is now excluded from
the specified maintained two-twist coexistence target throughout the stated
positive coefficient ratio domain and fixed capacities, by the joint
auxiliary-function proof above. The earlier constant-donor, silent-subspace and individual-profile
results remain
valid with their own stronger or different scopes, including their
entry-time statements. Independent ranges of `N/F`, `R` and `gamma`
were not treated as jointly attained or substituted into this proof.

That coexistence certificate does not select the loss law, alter the
preparation budget or transfer to another capacity contrast or constitutive model.
Transient joint acute entry and receiver-only transfer retain the separate
obligations stated above. No successful pattern formation, optimal profile,
autonomous substrate or physical identification follows from this
negative maintained-formation result.

<a id="sine-eleven-node-asymptotic-equilibria"></a>
### Every positive-loss trajectory on this support converges to one relative equilibrium

This conclusion concerns the same connected unit graph consisting of
cycles `(0,1,2,3,4)` and `(5,6,7,8,9)` and bridges `0--10`, `5--10`.
It needs no special preparation, form budget, initial winding or acute
phase premise. Let **every held capacity be strictly positive** and
`e,w,beta>0`, with no input, clipping, event or support change. Use the
same complete normalized-sine rows

\[
\dot x=-eKq+aKS,\qquad \dot\theta=bKq,\qquad
q=Lx,\quad K=\operatorname{diag}(\nu_i/d_i),\quad
a=w/\pi,\quad b=w/(\beta\pi).
\]

The result therefore includes the six-coordinate class above, but is
not restricted to its capacities or half-weight coefficients. Only one
common phase origin is removed when comparing relative geometries.
Node labels, constituent phases and the two bridge gaps are retained.

#### Reuse compactness and approach to the equilibrium set

The
[global sine-law argument](../../../nodal/RESONANCE_FOUNDATIONS.md#the-sine-obstruction-also-controls-nonperiodic-long-time-motion)
already gives global continuation and approach to the equilibrium set.
Its compactness hypotheses hold here: `E` is nonincreasing and
nonnegative, so it bounds all form differences on the connected graph;
the conserved weighted form mean bounds the common form origin, and
circular phase lies on a compact torus. Thus every finite initial state
has a forward orbit with compact closure in form times circular phase.

With strictly positive `K`, the largest invariant zero-loss subset is
exactly `q=S=0`. Indeed, zero loss forces `q=0`. Remaining there requires
`LK S=0`, hence `KS=c*1`; reciprocity `sum_i S_i=0` and positivity of
`K` imply `c=0`. Form is therefore uniform at every equilibrium, fixed
by the source's conserved weighted form mean. What remains to strengthen
the earlier limit-set theorem is the full **relative** phase critical set,
including every nonacute branch.

#### Critical bridge currents vanish, rather than being discarded

For either bridge, remove that edge and sum `S_i=0` over one resulting
connected component. Internal edge sine terms cancel pairwise. The only
remaining term is the bridge sine current, which must consequently be
zero. Its phase difference is therefore either `0` or `pi` modulo
`2*pi`; the antipodal branch must not be removed by an acute admission
rule.

Fix intermediary phase zero. The donor and receiver port phases may
each independently be `0` or `pi`, giving four relative bridge choices.
This does not allow an arbitrary independent rotation of either ring:
its port is fixed by its bridge choice relative to the common intermediary.

#### An odd cycle has finitely many complete sine-critical branches

Orient either C5 and let `eta_j` be the phase increment from its
`j`th node to its next node, modulo `2*pi`. At a nonport ring node,
`S_i=0` equates the two oriented ring sine currents. At the port the
bridge current has already been proved zero, so the same equality holds.
Thus all five `sin(eta_j)` equal one value `s`.

Choose its principal inverse-sine representative
`alpha=arcsin(s)` in `[-pi/2,pi/2]`. Every edge increment is then
either `alpha` or `pi-alpha` modulo `2*pi`. If `k` of the five edges
take the second branch, circular closure gives

\[
(5-2k)\alpha+k\pi=2\pi m,\qquad m\in\mathbb Z.
\]

The two branch endpoints do not supply exceptional continuous families.
If `alpha=pi/2`, both edge choices coincide at `pi/2` modulo `2*pi`,
whose fivefold sum does not close. At `alpha=-pi/2` the same argument
uses `-pi/2`. Therefore every actual critical solution has
`|alpha|<pi/2`. Since `5-2k` never vanishes,

\[
\boxed{\alpha=\pi\frac{2m-k}{5-2k},\qquad
\left|\frac{2m-k}{5-2k}\right|<\frac12.}
\]

For each of the finitely many edge masks, this strict inequality admits
only finitely many integers `m`. In exact turn coordinates, a constructive
classification uses base increment

\[
c=\frac{\alpha}{2\pi}=\frac{2m-k}{2(5-2k)},\qquad |c|<\frac14,
\]

with edge turns `c` off the mask and `1/2-c` on it, reduced modulo one
when reconstructing nodal phases. Starting from port phase zero, their
successive sums determine all ring phases and the integer closure
condition determines the closing edge. Every listed state has equal
oriented sine currents, so the construction is sufficient as well as
necessary.

There is no branch duplication hidden by the mask. Because
`cos(alpha)>0`, the mask is precisely the set of negative-cosine edges,
and the common sine current fixes its principal `alpha` uniquely.
The case `alpha=0` is included: its edges are zero or antipodal, and
closure requires an even number of antipodal edges. Cyclic permutations
of distinct masks remain distinct **labeled** geometries; this classification
does not quotient graph automorphisms.

The exact integer possibilities and resulting counts are:

| Negative-cosine edges `k` | Integers `m` | Principal base turns `c` | Labeled C5 geometries |
| --- | --- | --- | --- |
| 0 | `-1,0,1` | `-1/5,0,1/5` | `3` |
| 1 | `0,1` | `-1/6,1/6` | `2*binomial(5,1)=10` |
| 2 | `1` | `0` | `binomial(5,2)=10` |
| 3 | none | none | `0` |
| 4 | `2` | `0` | `binomial(5,4)=5` |
| 5 | `2,3` | `1/10,-1/10` | `2` |

There are exactly `30` relative phase geometries on each labeled ring.
Combining both independent ring choices with the four bridge choices
gives **`4*30^2=3600` relative phase-critical geometries** on the entire
eleven-node support. One canonical representative fixes intermediary
turn zero and records every other turn in `[0,1)`. Uniform form with
any common value completes an equilibrium. On the conserved form-mean
leaf of a particular trajectory, that value is already fixed.

This finite classification keeps critical states with negative cosines,
antipodal bridges and nonzero winding beyond the acute class. Enumeration
alone does not classify their stability or basins. The separate
[inertia argument below](#sine-eleven-node-equilibrium-stability)
settles local stability using the complete law.

#### Connectedness of the limit set gives one relative endpoint

Pass to real form together with phase relative to the intermediary,
`exp(i*(theta_i-theta_10))`. This is a continuous quotient by the one
common circular origin. The orbit is still precompact. For each `T`,
the closure of its connected forward tail `t>=T` is compact and
connected. These nonempty tail closures are nested, so their intersection,
the omega-limit set, is nonempty, compact and connected.

The earlier dissipation theorem puts that limit set inside `q=S=0`.
On the fixed weighted form-mean leaf, the classification just proved
leaves only `3600` possible relative equilibria. A connected subset of
a finite set consists of one point. Precompactness then implies
convergence to that point: otherwise a sequence remaining a positive
distance away would have a different omega-limit point. Thus

\[
\boxed{x(t)\longrightarrow M_x\mathbf1,\qquad
e^{i(\theta_i(t)-\theta_{10}(t))}
\longrightarrow e^{i\theta_i^*}
\quad\text{for one classified geometry }\theta^*.}
\]

The identity of `theta*` is not selected by this argument. In particular,
convergence to some equilibrium is not convergence to the receiver-transfer
target, nor evidence that the limiting equilibrium attracts an open set.
A source initially at any unstable equilibrium is also covered.

#### Conserved lifted phase reconstructs the final common origin

Relative convergence alone would not rule out continuing common rotation.
Here the independently proved weighted phase-lift invariant supplies the
missing reconstruction. Choose any continuous real lift of the actual
phase trajectory, and let

\[
I_\theta=\sum_i\omega_i\theta_i(t),\qquad
\omega_i=1/k_i,\qquad M_\omega=\sum_i\omega_i>0.
\]

Fix the limiting representative with `theta_10^*=0`. For sufficiently
large time every relative phase error is inside one strict circular
chart about zero. Its unique small real representative `epsilon_i(t)`
is continuous and tends to zero, with `epsilon_10=0`. In that chart

\[
\theta_i(t)=\theta_{10}(t)+\theta_i^*+2\pi n_i+
\epsilon_i(t).
\]

The integer vector `n` is fixed for all these later times: an integer
difference of continuous lifts cannot change while the chart remains
admitted. Choose `n_10=0`. Conservation now gives

\[
\theta_{10}(t)=\frac{I_\theta-
\sum_i\omega_i(\theta_i^*+2\pi n_i+\epsilon_i(t))}{M_\omega}
\longrightarrow
\gamma_\infty:=\frac{I_\theta-
\sum_i\omega_i(\theta_i^*+2\pi n_i)}{M_\omega}.
\]

Consequently the chosen continuous lifts themselves converge to finite
limits `gamma_infinity+theta_i^*+2*pi*n_i`, and the circular phases
converge to the corresponding full equilibrium. The integer vector
retains the actual trajectory's prior turns; the exact classifier does
not predict it. No weighted phase mean has been promoted to a
single-valued function on the entire torus.

#### Scope of the stronger long-time conclusion

This result applies to every finite initial state on this fixed support,
not only the earlier six-coordinate preparation. It proves existence of
one limiting geometry, not its selection, a convergence rate, a finite
arrival time or successful identity transfer. The receiver-transfer
accessibility question remains separate even though unclassified
persistent motion is no longer an alternative long-time outcome in
this positive-loss model.

Strictly positive capacities, positive `e,w,beta` and the unchanged sine
law are essential stated premises. Zero capacities retain additional
frozen data; the conservative `e=0` pulse and recurrence theorems are
different results. The odd-cycle closure argument does not extend by
renaming the graph: on an even cycle the coefficient `n-2k` can vanish,
and continuous critical families can remain. Native argument-pressure
dynamics also retains its separate regularity obligations. For example,
flat donor phases zero, flat receiver phases `pi` and intermediary phase
zero define an exact sine equilibrium with uniform form, but the
intermediary's native neighbor resultant is `1+(-1)=0`. That catalog
member is outside the native argument law's regular domain. No theorem
about arbitrary TNFR runtimes, support birth or physical constituents
follows from this classification.

The shared
[`CircularPhaseState` and `reconstruct_circular_phase_state`](../../../../src/tnfr/physics/phase_cycle_geometry.py)
retain the full circular geometry, including nonacute and antipodal edges,
without weakening the existing acute recovery contract.
`classify_c5_sine_critical_set(geometry, cycles=...)` uses that owner to
construct the exact factored `C5SineCriticalSet`; no numerical root
search or independent phase catalog is required. The separate
[`assess_sine_asymptotic_equilibria`](../../../../src/tnfr/physics/relational_sine_equilibria.py)
reader, also available as a sine comparison's
`asymptotic_equilibria(cycles=...)`, returns `SineAsymptoticEquilibria`
with the complete-law premises checked against its captured source.
An exact phase classification and the availability of this asymptotic
theorem are distinct from identifying the source's future endpoint.
The reader does not evolve the state or choose a basin.

The
[independent exact controls](../../../../tests/physics/test_relational_sine_equilibria.py)
check full-support criticality, branch closure and completeness, including
the nonacute controls. The
[observation controls](../../../../tests/physics/test_relational_sine_equilibria_observation.py)
check circular reconstruction, labeling and the law-admission boundary.
These finite tests exercise the classification and its engine integration;
the infinite-time conclusion follows from the compactness and connected
limit-set argument above, not a sampled convergence trace.

<a id="sine-eleven-node-equilibrium-stability"></a>
### Exact local stability of every classified equilibrium

The [general bridge-tree composition theorem](../../../nodal/SINE_PATTERN_DYNAMICS.md#sine-bridge-tree-composition)
now owns the extension to arbitrary connected components and the full-law
inertia argument on connected support. The calculation below retains the
exact C5 component classification and this support's specialized counts.

Keep the complete eleven-node sine law, fixed unit support, all strictly
positive held capacities and `e,w,beta>0` of the preceding theorem.
Choose any of its exact critical phase geometries and uniform form.
No restriction to acute edges or to either ring's reflection subspace
is imposed. Write `H` for the full cosine-weighted phase Hessian, with
quadratic form

\[
v^\mathsf THv=\sum_{\{i,j\}}\cos(\theta_j^*-\theta_i^*)
                         (v_j-v_i)^2.
\]

Inertia means the numbers of positive, negative and zero directions,
in that order. The geometric calculation first removes the one common
phase direction; the subsequent dynamics also removes the independently
conserved common form coordinate.

#### Ring constraints and bridge directions determine the inertia exactly

For one ring, let `alpha` and its supplementary-edge mask be those of
the critical classification, and let `k` be the mask size. Every edge
cosine has magnitude `c_alpha=cos(alpha)>0`; its sign is positive off
the mask and negative on it. If `z_j` are the five oriented edge
variations of an actual nodal phase perturbation, they satisfy
`sum_j z_j=0`. Conversely every such vector is produced by a ring
perturbation, uniquely modulo its common origin. The ring quadratic form
is therefore the restriction of

\[
D=c_\alpha\operatorname{diag}(\sigma_0,\ldots,\sigma_4),
\qquad \sigma_j\in\{1,-1\},
\]

to `1^perp`. The full edge form has inertia `(5-k,k,0)`. Its
`D`-orthogonal complement to `1^perp` is spanned by `D^(-1)*1`, since
`z^T D(D^(-1)*1)=z^T*1=0`, and that complementary direction has value

\[
\mathbf1^\mathsf TD^{-1}\mathbf1=\frac{5-2k}{c_\alpha}\ne0.
\]

Thus the restriction is nondegenerate: remove one positive direction
when `k<5/2`, or one negative direction when `k>5/2`. This treats the
cycle constraint explicitly; the sign of a single edge alone would not
justify the conclusion on an arbitrary graph.

| Allowed mask size `k` | C5 relative inertia `(positive, negative, zero)` | Number of labeled ring branches |
| --- | --- | --- |
| 0 | `(4,0,0)` | `3` |
| 1 | `(3,1,0)` | `10` |
| 2 | `(2,2,0)` | `10` |
| 4 | `(1,3,0)` | `5` |
| 5 | `(0,4,0)` | `2` |

On the full graph, edge variations satisfy exactly the two ring-sum
constraints. The two bridge variations are independent: the incidence
map from nodal phases modulo one common origin is an isomorphism onto
this ten-dimensional constrained edge space. Consequently the Hessian
form is a direct sum of both constrained ring forms and the two scalar
bridge terms. Each zero-phase bridge contributes one positive direction;
each antipodal bridge contributes one negative direction.

Let `j_D,j_R` be the two negative ring indices in the table, and let
`n_pi` be the number of antipodal bridges. Then

\[
\boxed{j=j_D+j_R+n_\pi,\qquad
\operatorname{inertia}(H_{\rm relative})=(10-j,j,0).}
\]

The unrestricted eleven-dimensional phase Hessian has inertia
`(10-j,j,1)`; its sole kernel is the common phase direction. No other
zero direction occurs in any of the 3,600 branches. In particular, the
two bridge choices cannot be discarded before testing stability.

#### The full reciprocal Jacobian, rather than phase gradient descent

The
[existing complete sine derivative](../../../nodal/RESONANCE_FOUNDATIONS.md#resonance-tangent)
and [stiffness argument](../../../nodal/RELATIONAL_RECOVERY_AND_INTERACTION.md#regular-equilibrium-stiffness)
apply to both consumed rows. To make their use explicit, introduce

\[
\xi=K^{-1/2}\delta x,\qquad \eta=K^{-1/2}\delta\theta,
\qquad h=K^{-1/2}\mathbf1,
\]
\[
B=K^{1/2}LK^{1/2},\qquad C=K^{1/2}HK^{1/2}.
\]

Fixing the two conserved weighted means in a local phase lift is exactly
`xi,eta in h^perp`. Both `B` and `C` preserve this subspace. On it,
`B>0`, and `C` has the relative Hessian inertia just computed: the
invertible capacity rescaling identifies this mean-fixed complement
with the nodal phase quotient. In particular `C` is nonsingular.
The exact full tangent rows on this twenty-dimensional space are

\[
\dot\xi=-eB\xi-aC\eta,\qquad
\dot\eta=bB\xi.
\]

Eliminating `xi` and writing `eta=B^(1/2)*z` gives

\[
\ddot z+eB\dot z+A z=0,\qquad
A=abB^{1/2}CB^{1/2},
\]

whose symmetric stiffness `A` has the same inertia as `C`. This is
an exact rewriting of the linearized two-row law, not a new inertial
constitutive assumption. Its quadratic eigenvalue pencil is

\[
P(s)=s^2I+esB+A.
\]

For a possibly complex eigenvector `z!=0`, multiply `P(s)z=0` by
`z^*`. Writing `s=sigma+i*omega`, the imaginary part is

\[
\omega\left(2\sigma\|z\|^2+e\,z^*Bz\right)=0.
\]

Every nonreal eigenvalue therefore has strictly negative real part.
There are no nonzero imaginary eigenvalues; `s=0` is also excluded
because `A` is nonsingular. For real `s>=0`,
`P'(s)=2*s*I+e*B>0`, so its ordered real eigenvalues increase strictly.
At zero there are exactly `j` negative eigenvalues, and for sufficiently
large `s` the entire matrix is positive. Hence exactly `j` eigenvalues
cross zero at positive real values, counted with multiplicity. At each
crossing the derivative restricted to its kernel is positive definite;
eliminating the invertible complementary block gives a first-order
positive term on that kernel. Thus the zero's determinant multiplicity
equals the kernel dimension, with no uncounted tangential crossings.

The full tangent consequently has exactly `j` positive real eigenvalues,
`20-j` eigenvalues with negative real part, and no relative center modes:

\[
\boxed{\dim E^{\rm unstable}=j,\qquad
\dim E^{\rm stable}=20-j,\qquad
\dim E^{\rm center}_{\rm relative}=0.}
\]

This also rules out an instability hidden in a complex right-half-plane
pair. In unrestricted form/phase coordinates the two common-origin
directions remain neutral. They are the conserved-coordinate freedoms,
not unresolved relative degeneracies. Fixing their weighted means gives
the tangent space just analyzed; perturbing those means changes the
equilibrium's eventual common origins.

#### Nine local attractors, with every other relative branch unstable

The condition `j=0` requires both ring masks to have `k=0` and both
bridge phase gaps to be zero. Each ring can then have exactly one of
the three uniform principal increments `0`, `2*pi/5` or `-2*pi/5`.
There are therefore **nine locally exponentially attracting relative
equilibria**, indexed by the pair of ring windings

\[
(w_D,w_R)\in\{-1,0,1\}^2,
\]

with aligned ports and intermediary. Smoothness and the strictly stable
full relative Jacobian give nonlinear local exponential attraction on
each conserved-mean leaf, or attraction to the corresponding common-origin
orbit when those two means are allowed to vary. The
[whole-support recovery theorem](../../../nodal/SINE_PATTERN_RECOVERY.md#sine-interacting-recovery) provides
explicit sufficient neighborhoods for these acute critical geometries.
It does not certify a distant preparation's entry into them.

Every other branch has `j>=1`, hence a positive real eigenvalue and
nonlinear instability. All **3,591 remaining relative equilibria are
hyperbolic and unstable**; none has an unclassified relative zero mode.
The dimensions follow from the structural inertia formula, without
numerically diagonalizing 3,600 Jacobians.

Strictly positive changes in held capacities or `e,w,beta` alter
eigenvalues, response rates and potentially basins, but not this local
stability classification on the fixed support. They cannot produce a
local Hopf or zero-mode crossing in this equilibrium family while the
stated positivity premises hold. This is not a claim that global basins
cannot change. In particular, the auxiliary-function proof's sufficient
ratio cutoff `w/e<3/2` is not a detected local bifurcation threshold.

#### Stable composition retains causal interaction and its observations

The result admits an independently chosen consensus or either handed
twist on each ring as one stable **full-network** geometry. Its proof
retains perturbations of every node and both bridges; it does not turn
the regions into dynamically independent copies. The actual matrices
`B` and `C` need not commute, and the existing
[mediated response](../../../nodal/RESONANCE_FOUNDATIONS.md#mediated-resonance)
uses this same retained coupling and intermediary. Geometry can preserve
the local identities while permitting a response between them.

That response owner already covers any acute critical target with zero
bridge gaps. The present classification shows this is exactly the nine-member
attracting family on the stated support. At fixed capacities and coefficients,
reversing a ring's twist leaves all its edge cosines, the complete tangent
and every linear transfer unchanged. The nine phase geometries therefore
give four distinct tangent geometries, determined by whether each ring
is flat or twisted. Linear response alone does not recover handedness.
This reuses a proved observation limitation; it does not select a new
controller, onset law or physical identification.

#### The original source budget excludes all four attracting coexistence states

Return only for this corollary to the original six-coordinate source,
`F<=4`, `beta=1`, `delta=1/2` and the proved ratio domain `0<w/e<3/2`.
Each of the four attracting states with both ring windings nonzero has
uniform form, `q=S=0`, and `V=2V_5`, regardless of either handedness.
The [same monotone functional](SINE_RECEIVER_FORMATION_BOUNDS.md#sine-maintained-target-obstruction)
therefore gives the unchanged gap

\[
\mathcal W_{\rm coexistence}-\mathcal W(0)
=V_5-\frac45F>\frac14.
\]

All four attracting two-twist endpoints are excluded for this source
class, not only the original `(+1,+1)` target. Among locally attracting
geometries, exactly five remain compatible with that obstruction:

\[
(0,0),\qquad (+1,0),\quad(-1,0),\qquad
(0,+1),\quad(0,-1).
\]

These are remaining candidates, not five demonstrated outcomes. The
classification does not exclude convergence along a stable manifold to
an unstable equilibrium, and the six-dimensional source slice need not
inherit an ambient almost-everywhere statement. Nor do an initial odd
response or an available receiver-only basin choose a terminal winding.
Receiver-transfer accessibility remains the separate missing dynamical
argument identified above.

#### Shared geometry and law readers retain different claims

The exact catalog's `phase_hessian_inertia(...)` method retains the
selected branch's constrained geometric inertia;
`phase_hessian_index_counts` aggregates the factorized catalog without
an equilibrium eigenvalue scan. These geometry readers belong to
[`phase_cycle_geometry.py`](../../../../src/tnfr/physics/phase_cycle_geometry.py).
The law report `SineAsymptoticEquilibria.classify_equilibrium(...)`
revalidates the consumed source and catalog before returning
`SineEquilibriumStability` from
[`relational_sine_equilibria.py`](../../../../src/tnfr/physics/relational_sine_equilibria.py).
Its relative mode dimensions and nonlinear stability flags require the
strictly positive law and capacity premises proved here. An available
geometric inertia under unsupported dynamics supplies no such flags.
The reader classifies a declared equilibrium; it does not predict which
branch a particular source reaches or install a different evolution law.

[Independent algebra and complete-Jacobian controls](../../../../tests/physics/test_relational_sine_equilibria.py)
check the constrained inertia and its full-law connection, including
noncommuting matrices. The
[observation controls](../../../../tests/physics/test_relational_sine_equilibria_observation.py)
exercise the catalog, source admission and unavailable domains. These
finite controls support the implementation; the proof above supplies
the all-branch and positive-parameter quantifiers.

<a id="sine-donor-well-retention"></a>
### A subcritical auxiliary well selects the original donor endpoint

Retain the same eleven-node support, held capacities with `delta=1/2`,
`beta=1`, donor winding `+1`, flat receiver and aligned bridges. The
initial form remains the six-coordinate donor/intermediary preparation
of Section 22. Use the proved coefficient domain `e>0`,
`0<w/e<3/2`; it includes the original `e=w=1/2` without a parameter
survey or another constitutive law. Let `F` be its actual initial form
storage, and define

\[
F_c=\frac54\left(\frac72-V_5\right)
   =\frac{25\sqrt5-55}{16}>0.
\]

Then

\[
\boxed{0\le F\le F_c
\quad\Longrightarrow\quad
\text{relative equilibrium limit }(w_D,w_R)=(+1,0).}
\]

This determines a terminal relative identity, not just failure to reach
one proposed target. It supplies no finite convergence time or claim
that either ring remains in an acute or fixed-winding chart at every
intermediate time. Its proof uses the full coupled law, including the
intermediary and receiver capacity contrast.

#### Properness and critical points of the existing auxiliary function

Work on the fixed weighted form-mean leaf and quotient only the common
circular phase origin. The form part has dimension ten; the phase part
is a compact ten-dimensional torus. Common lifted-phase means determine
the final representative as in the asymptotic theorem, but are not a
globally defined function on this torus quotient.

Reuse exactly the function already proved to decrease strictly:

\[
\mathcal W=\frac45\mathcal F+V-hq^\mathsf TKS,
\qquad h=\frac{w}{2\pi e},\qquad
\dot{\mathcal W}<0\quad\text{if }(q,S)\ne(0,0).
\]

On the fixed mean leaf, the positive graph gap makes `mathcal F`
coercive in relative form. The sine vector is uniformly bounded, while
`q=Lx` is linear in that form. Thus for some positive constants
`A,C`, independent of phase, `W>=A*||u||^2-C*||u||`, where `u` is
any fixed linear coordinate on this form leaf. Consequently `W` is
bounded below and every sublevel is compact. Keeping the common form
translation free would destroy this properness premise.

A critical point of `W` on this quotient has zero derivative along
the actual flow. Strict decrease therefore forces `q=S=0`.
Conversely, at `q=S=0` every derivative of `W` vanishes: form is
uniform, `V` is critical and both factors of the mixed term vanish.
The critical set is exactly the previously classified equilibrium set,
and its critical values are exactly `V` at those geometries.

The single-C5 critical values supplied by the exact catalog are

\[
0,\quad V_5,\quad \frac72,\quad 4,\quad 8,\quad
\frac{25+5\sqrt5}{4}.
\]

Each bridge adds zero or two. Hence the only full-support critical
values strictly below `7/2` are `0`, `2` and `V_5`. In particular,
there is **no critical value in `(V_5,7/2)`**. The four critical
geometries at `V_5` are the two donor-only twists and the two
receiver-only twists; each is a locally attracting relative equilibrium.

Each is also an isolated strict local minimum of `W`. To see this
without assuming that `W` is the physical energy, take any nearby
nonequilibrium state in its local attracting neighborhood. Its forward
orbit converges to that equilibrium, while `W` decreases strictly
before taking the limiting value `V_5`. Its initial `W` is therefore
strictly greater than `V_5`. The local attraction and strict-decrease
results have compatible coefficients and the same retained state.

#### Different one-twist wells cannot connect below `7/2`

Around the donor-only minimum choose a small closed coordinate ball
that excludes every other critical point. Its boundary has minimum
`W` strictly above `V_5`. For a level `r_0` between `V_5` and that
boundary minimum, also chosen below `7/2`, the donor's component of
`{W<r_0}` stays inside the ball and contains only the donor critical
point.

This component cannot merge with another component as the level rises
to any `r<7/2`. Indeed a closed strip `r_0<=W<=r` is compact and
has no critical point. The auxiliary vector field
`-grad(W)/||grad(W)||^2` lowers `W` at unit rate there; its finite
flow, with a cutoff outside the strip, deforms the higher sublevel to
the lower one without changing its components. This is a topological
proof device, not a replacement for the TNFR dynamics. It shows that
for every `V_5<r<7/2`, the donor component contains no other critical
geometry, including the lower-valued bridge saddles or consensus.

In particular every continuous path from that donor minimum to either
receiver-only minimum must have `max W>=7/2`. The existing path that
first flattens the donor and then twists the receiver, at uniform form,
attains `max W=7/2`. Thus the static minimax is exactly

\[
\inf_{\gamma:D_+\to R_\pm}\ \max_s\mathcal W(\gamma(s))
=\frac72.
\]

The negative receiver orientation uses the reflected second phase leg.
The same reasoning restricted to an isolated donor phase torus gives
its twist-to-flat potential minimax `7/2`; a winding-seam cost alone
does not prove this barrier. The critical-set and component argument
is what supplies the previously missing path statement.

#### The actual preparation enters and remains in the donor component

Initially `S=0`, so `W(0)=V_5+(4/5)*F`. At the fixed initial phases,
contracting form toward its conserved uniform mean gives
`W=V_5+(4/5)*s^2*F` for `0<=s<=1`. If `F<F_c`, this entire path
lies below `7/2`. Choose a regular level `r` strictly between
`W(0)` and `7/2` (and above `V_5`); the actual initial state is
in the donor component of `{W<r}`. Nonincrease of `W` and continuity
keep its actual orbit in that component.

The equality `F=F_c` needs a separate argument; replacing a closed
inequality by a strict numerical tolerance would not suffice. Here
`F>0`, hence `q(0)!=0` on the connected support. Strict decrease
gives `W(t)<7/2` for all sufficiently small positive times.
At those same times `V(theta(t))<7/2`, by continuity from `V_5`.
For each fixed phase, `W` is a convex quadratic in form: its mixed
term is linear in form. The straight form segment from `x(t)` to
the conserved uniform mean consequently lies below `7/2`, since
both endpoint values, `W(t)` and `V(theta(t))`, do. At uniform form,
a short path in a sufficiently small phase-chart neighborhood joins
`theta(t)` to the initial donor phase; throughout it `W=V<7/2`.
These two compact paths place the actual state at time `t` in the
same donor component at some level `r<7/2`. This resolves the
boundary without assigning a new preparation or integrating a response.

Finally the already proved global asymptotic theorem gives convergence
to one relative critical geometry. The forward tail stays in a compact
sublevel of the retained donor component, whose only critical point is
the original donor-only minimum. That point must be its limit. At
`F=0` the original state itself is an equilibrium, consistently with
the conclusion. The conserved means set the terminal uniform form and
lifted phase representative; they do not change its relative geometry.

#### A stronger preparation control, without a claimed dynamical threshold

The theorem excludes convergence to either receiver-only attractor and
entry into any valid full-state recovery basin for either one. It is
stronger than the earlier acute-face budget or the bare energy barrier:
the latter gives only the necessary transfer condition
`F>7/2-V_5`, using strict initial energy loss before donor unwinding.
The auxiliary-well condition gives instead `F>F_c` as a necessary
condition for transfer.

For a rational control, set every donor form to zero and only the
intermediary form to `H=7/30`. Then `F=49/900` and

\[
\frac72-V_5<\frac{49}{900}<F_c.
\]

This nonsilent preparation has `E(0)>7/2`, yet `W(0)<7/2` and
therefore returns to the original donor identity. These are exact
algebraic inequalities, not an evaluated trajectory. For example the
left inequality follows from `sqrt(5)<56/25` and `1/20<49/900`;
the right follows
from `(16*(49/900)+55)^2<3125`.

For any admitted `F>=0`, the exact retention condition is equivalently

\[
\boxed{(16F+55)^2\le3125.}
\]

Both sides of the unsquared inequality `16F+55<=25*sqrt(5)` are
positive, so squaring introduces no extraneous branch. This permits
represented rational inputs to use the shared exact arithmetic rather
than a rounded decimal threshold or a widened difference of intervals.

The threshold is exact for this **static sublevel separation**: above
it the existing comparison path connects the source and target inside
`W<=W(0)`. It is not a demonstrated dynamical transition at `F_c`.
At equality the actual trajectory has already lost auxiliary value
before it could cross; above it, silence, further losses and the actual
flow direction can still prevent transfer. The rest of the original
`F<=4` class retains its unresolved accessibility obligation.

The shared formation report's
[`donor_well_retention()`](../../../../src/tnfr/physics/relational_sine_formation.py)
returns `SineDonorWellRetention` from the same revalidated exact
preparation. Its `exact_retention_polynomial_margin` controls admission;
the displayed critical-storage and escape-margin intervals do not.
`relative_donor_pattern_convergence_certified` and
`receiver_only_targets_excluded` state the two proved consequences.
Unsupported law premises remain `unavailable`; a valid source outside
this sufficient interval is `not_certified`, with no positive transfer
verdict. The half-weight receiver-transfer reader retains this result
separately from its acute-entry/action checks. The
[formation controls](../../../../tests/physics/test_relational_sine_formation.py)
and [independent algebra controls](../../../../tests/physics/test_relational_sine_formation_barrier.py)
exercise the exact threshold and the distinct source-to-basin conclusion;
neither evaluates a trajectory or infers a finite recovery time.

<a id="sine-donor-dissipative-capture"></a>
### Early dissipation captures an above-threshold preparation in the donor well

Return to the original `e=w=1/2`, `beta=1`, `delta=1/2` and structural
clock. Keep the same six initial form coordinates, exact donor twist,
flat receiver, aligned intermediary, complete support and held capacities.
No input, event, clipping or changed law is supplied. Write

\[
F=\mathcal F(0),\qquad N=q(0)^\mathsf TKq(0),\qquad
D=\frac72-V_5=\frac{5\sqrt5-11}{4}>0,
\]
\[
A=\frac45F-\frac{N}{25},\qquad T=\frac14.
\]

The exact source-specific sufficient condition is

\[
\boxed{A<D
\quad\Longrightarrow\quad
\text{relative equilibrium limit }(w_D,w_R)=(+1,0).}
\]

The loss term makes this different from the static well test: a source
with `W(0)>7/2` can satisfy this condition. The conclusion follows from
guaranteed entry into the donor sublevel component by the fixed time
`T`, followed by the existing asymptotic theorem. Neither the state at
`T` nor its numerical trajectory is reconstructed, and `T` is not a
convergence deadline. The constants below are sufficient proof bounds,
not new parameters in the nodal law.

#### Full-state norm bounds from the exact initial sine balance

Reuse `y=K^(1/2)q`, `zeta=K^(1/2)S`, and
`B=K^(1/2)LK^(1/2)`. At the declared preparation `zeta(0)=0` and
`||y(0)||^2=N` for every choice of the six form coordinates. The
full eleven-node equations are

\[
\dot y=-\tfrac12By+aB\zeta,\qquad
\dot\zeta=-aC(\theta)y,\qquad a=\frac1{2\pi}<\frac16.
\]

The actual support and capacities give `||B||<=3`. The edge
representation also gives `-B<=C(theta)<=B`, so
`||C(theta)||<=3` globally, without an acute-phase restriction.
Let `Y(t)=||y(t)||`, `Z(t)=||zeta(t)||`, and `Y_0=sqrt(N)`.
Variation of constants, the contraction of `exp(-Bt/2)` and these
operator bounds yield

\[
Y(t)\le Y_0+3a\int_0^tZ(s)\,ds,\qquad
Z(t)\le3a\int_0^tY(s)\,ds.
\]

Comparison with the corresponding two scalar integral equations gives
`Y(t)<=Y_0*cosh(3*a*t)` and `Z(t)<=Y_0*sinh(3*a*t)`.
On `0<=t<=1/4`, use `3*a*t<1/8` and
`cosh(1/8)<=128/127<11/10`; the first bound follows directly by
majorizing its power series by `sum_k(1/128)^k`. Consequently

\[
Y(t)\le\frac{11}{10}Y_0,\qquad
Z(t)\le\frac{11}{20}tY_0.
\]

The same variation-of-constants formula gives a lower bound, retaining
the initial vector rather than inferring its norm from a derivative:

\[
\begin{aligned}
Y(t)&\ge e^{-3t/2}Y_0-3a\int_0^tZ(s)\,ds\\
&\ge Y_0\left(1-\frac32t-\frac{11}{80}t^2\right)
 \ge(1-2t)Y_0.
\end{aligned}
\]

The final factor is positive throughout the declared window. These
integral inequalities also include the zero-norm source without division
by a norm; when `N=0` the source is the stationary donor equilibrium.

#### A directional initial loss of the same auxiliary function

At half weights `h=a`. The already derived exact derivative and
`C(theta)<=B` give

\[
\dot{\mathcal W}
\le -\frac25Y^2
 +a\,y^\mathsf T(-\tfrac15I+\tfrac12B)\zeta
 -a^2\zeta^\mathsf TB\zeta+a^2y^\mathsf TBy.
\]

Here `||-I/5+B/2||<=13/10`. Dropping only the nonpositive
`zeta` quadratic and using `a<1/6` therefore gives

\[
\dot{\mathcal W}\le-\frac{19}{60}Y^2+\frac{13}{60}YZ.
\]

The norm estimates now supply a lower bound on actual accumulated
auxiliary decrease, not on an independently prescribed pressure:

\[
\begin{aligned}
\mathcal W(0)-\mathcal W(T)
&\ge N\left[
 \frac{19}{60}\int_0^{1/4}(1-2t)^2\,dt
 -\frac{13}{60}\frac{121}{200}\int_0^{1/4}t\,dt
 \right]\\
&=N\left(\frac{133}{2880}-\frac{1573}{384000}\right)
 =\frac{48481}{1152000}N
 \ge\frac{N}{25}.
\end{aligned}
\]

The last comparison is strict when `N>0`; retaining `N/25` gives
a simpler reusable certificate without optimizing its gain. In
particular,

\[
\mathcal W(T)\le V_5+A<\frac72.
\]

This inequality alone would not identify which well contains the state.
The next argument supplies that missing component information.

#### The phase path stays below the barrier as a consequence of the same test

Let `eta=theta(T)-theta(0)` in the actual continuous primitive-phase
lift. The full phase row gives

\[
\eta=aK^{1/2}\int_0^T y(t)\,dt,
\qquad
\eta^\mathsf TL\eta
 \le3a^2NT^2(11/10)^2.
\]

At the exact initial phase, `grad V=0`; globally `H(theta)<=L`.
Taylor's integral formula along the entire straight phase segment
therefore gives, for every `s in [0,1]`,

\[
V(\theta(0)+s\eta)
 \le V_5+\frac{s^2}{2}\eta^\mathsf TL\eta
 \le V_5+P,\qquad P:=\frac{121N}{38400}.
\]

This is a path in the circular state represented through a declared
lift, not an identification of primitive phase with a regional angle.
It does not require the segment to remain acute.

Importantly, `P<D` is **not another independent admission condition**.
The same full-support bound `||B||<=3` implies

\[
N=x(0)^\mathsf TLKLx(0)\le6F,
\qquad A\ge\frac{14}{25}F.
\]

Thus `A<D` already gives

\[
P\le\frac{121}{6400}F
 <\frac{121}{3584}D<D.
\]

At fixed phase `theta(T)`, convexity of `W` in form places the
straight segment from `x(T)` to the conserved uniform form below
`7/2`: its endpoint values are bounded by `V_5+A` and `V_5+P`.
At uniform form, the preceding phase segment joins it to the donor
equilibrium with `W=V<7/2`. This compact concatenation places the
actual state at `T` in the donor component of `{W<r}` for some
`r<7/2`. Subsequent nonincrease and the already proved global
single-equilibrium convergence force the original donor-only limit.
No transient receiver winding or interim current is discarded by this
argument.

#### One fixed above-threshold witness and its open preparation neighborhood

Fix, before evaluating any response, zero donor forms, intermediary
form `H=1/4`, and the analytic window `[0,1/4]`. The source has

\[
F=\frac1{16},\qquad N=\frac16,\qquad
A=\frac{13}{300},\qquad P=\frac{121}{230400}.
\]

The static retention criterion fails strictly, since
`(16*F+55)^2=3136>3125`, and `W(0)=V_5+1/20>7/2`.
Nevertheless `A<D`: equivalently `sqrt(5)>838/375`, whose
squared comparison is `703125>702244`. The source is nonsilent,
and the new theorem certifies its donor-only limit without integrating
its response. Its total initial energy also exceeds `7/2`, so neither
the bare energy barrier nor the earlier static auxiliary test supplies
this conclusion.

Both `F` and `N` are continuous quadratic forms on the six declared
initial form coordinates. At this witness the inequalities `F>F_c`,
`F<4`, and `A<D` are all strict. They consequently persist in a
nonempty open neighborhood in that **six-dimensional preparation
space**. Every source there has the same proved limit. This is not
an assertion about arbitrary phase or capacity perturbations, nor a
profile search or a new source-selection policy.

For represented rational source coordinates, the sole admission test
is exactly

\[
\boxed{125-(4A+11)^2>0.}
\]

There is no sign ambiguity: the actual support already proves
`A>=14F/25>=0`. At the fixed witness this polynomial margin is
`881/5625>0`. Outward display intervals for `V_5` and the analytic
upper-bound expressions do not decide this exact comparison. The certificate
reports a guaranteed state-set property at `T`, not an observed or
reconstructed endpoint, an error-enclosed trajectory, or entry into
a particular finite-radius recovery certificate. Outside this sufficient
criterion, receiver-transfer accessibility remains unresolved.

The existing formation owner exposes
[`donor_dissipative_capture()`](../../../../src/tnfr/physics/relational_sine_formation.py)
as `SineDonorDissipativeCapture`. It revalidates the primitive preparation,
recomputes `N` through every actual graph row, and checks the fixed-law
premises before using `endpoint_exact_polynomial_margin` for admission.
The retained phase polynomial is derived evidence, not a second
independent gate. `endpoint_functional_bounds` and
`phase_path_storage_bounds` enclose the **analytic upper-bound values**
`V_5+A` and `V_5+P`; they are not two-sided enclosures of an actual
future state or observed storage. The fixed `horizon=1/4` belongs to
this proof and does not reuse another report's finite-time verdict.

`donor_component_entry_certified` records the guaranteed sublevel
component membership by that horizon. Its relative-convergence and
receiver-target-exclusion flags retain the separate all-time consequences.
Unsupported premises remain `unavailable`, and a valid source failing
this sufficient criterion is `not_certified`. The
[formation controls](../../../../tests/physics/test_relational_sine_formation.py)
cover these consumed-state and interface boundaries; the
[independent algebra controls](../../../../tests/physics/test_relational_sine_formation_barrier.py)
check the full-field bounds, exact admission and fixed above-threshold
witness. Neither test owner evaluates a transfer trajectory.
