# Geometry and gain from two successive phase inputs

<a id="sine-two-pulse-inference"></a>

## Question and conditional claim

The [calibrated local inverse](SINE_TWO_PORT_INFERENCE.md#sine-two-port-inference)
constrains geometry after a positive sensor-gain interval is supplied.
Its [reserved software result](SINE_TWO_PORT_INFERENCE.md#sine-two-port-inference-result)
also shows that widening that interval can substantially widen the
necessary angle bound. That control is not an exact response collision.

Here two declared phase inputs and three scalar readings constrain an
original source angle and one unknown held positive gain together. Both
inputs act on one uninterrupted full-state trajectory. The second response
retains the form and phase evolution caused by the first input, and the
two recorded increments share the middle reading's error. The result is a
conditional outer enclosure, not exact joint identification, state
reconstruction, independent physical calibration or a new reserved experiment.

The [ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization) permits
collective observations but retains their independently justified measurement
premises. The [strategy](../NODAL_RESEARCH_STRATEGY.md#constitutive-information-justification)
also distinguishes constitutive information from a fitted response. The
complete sine law, clock, phase-input calibration, positive gain prior and
held affine observation model below remain supplied premises.

<a id="sine-two-pulse-source"></a>
## F1: one full source, two events and three readings

Reuse the [realizable affine source family](SINE_TWO_PORT_INFERENCE.md#sine-two-port-inference-domain)
on donor nodes `0,...,8`, receiver nodes `9,...,17`, their unit C9 cycles,
and contacts `(0,9)` and `(1,10)`. Ports `0,1,9,10` have degree three;
the other nodes have degree two. Let \(L\) be the full Laplacian,
\(M\) its degree matrix, \(A=M^{-1}L\), and \(P_M\) the
projection removing one common degree-weighted mean. All norms below
are \(\|z\|_M=(z^TMz)^{1/2}\), unless stated otherwise.

The nominal donor bulk and receiver short angles lie in supplied closed
subintervals of
\[
11/8\le b\le3/2,\qquad 2/3\le c\le1. \tag{1}
\]
Their other angles are the donor short arc \(4\pi-8b\), receiver
bulk arc \((2\pi-c)/8\), and contact gaps
\(\pm(4\pi-8b-c)/2\). The shared source owner reconstructs all
eighteen lifts \(\theta^0(b,c)\), subtracts their full weighted mean,
and retains periods \((2,1,0)\). No equilibrium equation is imposed.

At the original pre-probe time admit
\[
x^-=m_x\mathbf1+u,\qquad
\theta^-=m_\theta\mathbf1+\theta^0(b,c)+v,
\quad P_Mu=u,\quad P_Mv=v,\quad
\|u\|_M\le X,\quad\|v\|_M\le Y. \tag{2}
\]
The common form and lifted phase means are arbitrary; every node may
carry an error. The receiver coordinate and full residual vectors remain
nuisance state. Source geometry requires the same strict acute guard
\(\mu_0-Y>0\), where \(\mu_0\) is a lower bound for the nominal
minimum acute margin over the declared rectangle. The algebraic response
bounds below are global, but the offered inverse certificate retains this
guard to interpret the original geometric statistic on its declared branch.

Set \(q=e_4-e_5\). Supply two cumulative phase amplitudes and equal
elapsed windows satisfying
\[
0<a_1\le a_2\le1,\qquad 0<h\le1/2,\qquad T=2h. \tag{3}
\]
The actual events are
\[
\begin{aligned}
\theta(0^+)&=\theta^-+a_1q,& x(0^+)&=x^-,\\
\theta(h^+)&=\theta(h^-)+(a_2-a_1)q,&
x(h^+)&=x(h^-).
\end{aligned} \tag{4}
\]
Thus the second jump is the difference of the two supplied cumulative
amplitudes. It is not another jump of size \(a_2\), and neither event
resets the state. Here \(\|q\|_M=2\),
\(\|q\|_{M^{-1}}=1\), and \(\mathbf1^TMq=0\).

Read the scalar \(q^Tx\) at \(0,h,2h\). Form continuity makes the
middle reading the same on either side of the second phase event. One
unknown gain \(G\in[G_-,G_+]\), with \(0<G_-\le G_+\), and
one arbitrary constant offset \(O\) apply to all three readings:
\[
r_j=Gq^Tx(jh)+O+\eta_j,\qquad |\eta_j|\le\delta,
\quad j=0,1,2. \tag{5}
\]
The supplied primitive recorded intervals are
\(r_j\in[c_j-\tau_j,c_j+\tau_j]\), with exact midpoint
\(c_j\) and nonnegative half-width \(\tau_j\). No statistical
independence is assumed. The gain interval is prior information, not a
calibration result manufactured from these readings. An unknown drifting
gain, unbounded offset drift or uncertain input amplitude is outside the
present theorem.

Form retains the source model's normalized structural units. Phases and
input amplitudes are in radians, and \(h\) uses its fast structural clock
\(\tau\). Gain has recorded units per local form-readout unit;
\(O,\delta,\tau_j\) and the readings use recorded units. No laboratory
unit or sensor identification is supplied by these declarations.

## F2: complete dynamics and one retained history

Keep unit held capacities, beta one, \(e=1023/1024\), \(w=1/1024\),
the fast structural clock \(\tau=et\), and
\[
\gamma=\frac1{1023\pi},\qquad
f_i(\theta)=\frac1{M_{ii}}
 \sum_{j\sim i}\sin(\theta_j-\theta_i).
\]
On both continuous windows the complete rows remain
\[
x'=-Ax+\gamma f(\theta),\qquad \theta'=\gamma Ax. \tag{6}
\]
There is no other input or support change. Both common means are
conserved, including across (4). No phase coordinate or intermediate form
is reconstructed from a reduced response or replaced by its initial value.

Define the continuous dynamical phase displacement
\[
d(\tau)=\gamma\int_0^\tau Ax(s)\,ds.
\]
It is continuous at the second event. On the two respective windows,
the actual phase is exactly
\[
\theta(\tau)=m_\theta\mathbf1+\theta^0(b,c)+a_jq+v+d(\tau).
\tag{7}
\]
This common original reference is an algebraic description of the actual
history, not a repeated preparation. In particular, the second window
starts with the evolved \(x(h)\) and accumulated \(d(h)\).

## F3: a uniform finite error for both full-flow increments

The degree operator is self-adjoint, \(0\preceq A\preceq2I\), so
its heat semigroup is contractive. The sine map is globally Lipschitz
with constant two. On every nominal affine source, its interior rows
cancel and only the four port rows may be nonzero. Consequently
\(\|f(\theta^0)\|_M\le\sqrt{12}<F=4\).
These facts and their exact graph realization are shared with the
[one-input proof](SINE_TWO_PORT_INFERENCE.md#port-locality-and-a-finite-full-law-error-bound).
The present bound does not assume that any later state has port-supported
forcing or reuse the earlier fourth-order locality cancellation.

Use \(g=1/3069>\gamma\), and define
\[
Q=\frac{X+gT(F+4a_2+2Y)}{1-4g^2T^2}. \tag{8}
\]
Its denominator is positive for \(T\le1\). If \(Q_T\) is the
actual supremum of \(\|P_Mx\|_M\) on \([0,T]\), Duhamel's
formula, (7) and the global Lipschitz bound give
\[
Q_T\le X+gT(F+4a_2+2Y)+4g^2T^2Q_T.
\]
Hence throughout both windows
\[
\|P_Mx(\tau)\|_M\le Q,\qquad
\|d(\tau)\|_M\le2gTQ. \tag{9}
\]
This includes the form created by the first input and the natural
noncritical port currents. No acute trajectory, stability or source-reset
premise enters (8)-(9).

The local nominal phase-input identity is
\[
q^Tf(\theta^0+a_jq)
 =-2\sin(3a_j/2)\cos(b-a_j/2).
\]
Let
\[
y_1=q^T[x(h)-x^-],\qquad y_2=q^T[x(2h)-x(h)],
\qquad K_j=2\gamma h\sin(3a_j/2)>0.
\]
Integrating the actual form row in each window gives
\[
y_j=-K_j\cos(b-a_j/2)+\rho_j,\qquad |\rho_j|\le E,
\tag{10}
\]
where
\[
\boxed{E=2hQ+2ghY+4g^2ThQ.} \tag{11}
\]
Indeed, \(|q^TAx|\le2Q\), while the difference between the actual
phase in (7) and its nominal phase-input reference has norm at most
\(Y+2gTQ\). Multiplying its sine-map error by \(\gamma\) and
integrating for \(h\) proves (11). Both remainders concern the same
trajectory. Bounding them separately by \(E\) is a conservative
relaxation, not permission to choose separate intermediate states.

A sufficient whole-window acute margin is
\[
\mu_0-Y-2a_2-2gTQ>0. \tag{12}
\]
It is optional additional evidence. Its failure neither invalidates
(8)-(11) nor prevents the inverse when the original source guard passes.
No event-work allowance, maintenance, winding retention after the inputs
or eventual recovery is certified by this inverse.

<a id="sine-two-pulse-joint-inverse"></a>
## F4: rank, shared observation errors and outer projections

Set \(\alpha_j=a_j/2\) and transform the two unknowns into
\[
z=(U,V)^T=(G\cos b,G\sin b)^T.
\]
The leading response matrix is
\[
C=\begin{pmatrix}
K_1\cos\alpha_1&K_1\sin\alpha_1\\
K_2\cos\alpha_2&K_2\sin\alpha_2
\end{pmatrix},\qquad
\det C=K_1K_2\sin(\alpha_2-\alpha_1). \tag{13}
\]
It has rank two when \(a_1<a_2\). Equal cumulative amplitudes
give a zero second jump and a rank-deficient leading matrix. This is
an obstruction to the present inversion method, not a theorem that the
full nonlinear two-time response contains no additional information.
An unresolved positive factor is numerical unavailability, not an exact
rank-one proof. Certify positivity of both scales and
\(d=\sin(\alpha_2-\alpha_1)\) separately. Their positive product can
have an outward lower endpoint of zero on a fixed absolute grid; that
diagnostic interval does not invalidate the factor-based rank proof.
Cancel the redundant scale factors before numerical inversion:
\[
C^{-1}=\begin{pmatrix}
(\sin\alpha_2/K_1)/d&-(\sin\alpha_1/K_2)/d\\
-(\cos\alpha_2/K_1)/d&(\cos\alpha_1/K_2)/d
\end{pmatrix}. \tag{13a}
\]
Evaluate the two divisions successively rather than materializing the
products \(K_jd\). This also avoids losing a resolvable denominator
through product rounding. The output retains the full determinant interval
as a diagnostic. A nonzero determinant alone gives no useful resolution
guarantee: conditioning must be compared with the full errors.

Form exact recorded midpoint differences
\[
m=(c_1-c_0,c_2-c_1)^T.
\]
The arbitrary common offset cancels before outward coefficient arithmetic.
Let \(w=(w_1,w_2)\) be either exact row of \(C^{-1}\). The
corresponding transformed coordinate has center \(-w\cdot m\) and
error radius at most
\[
\begin{aligned}
\epsilon_w={}&G_+E(|w_1|+|w_2|)\\
 &+(\tau_0+\delta)|w_1|
 +(\tau_1+\delta)|w_1-w_2|
 +(\tau_2+\delta)|w_2|.
\end{aligned} \tag{14}
\]
To see this, write each midpoint discrepancy from its true recorded
reading together with the additive reading error as one bounded nuisance
of size \(\tau_j+\delta\). The two increments use the incidence
matrix
\[
D=\begin{pmatrix}-1&1&0\\0&-1&1\end{pmatrix}.
\]
After applying \(-w\), their three coefficients are
\((w_1,w_2-w_1,-w_2)\). The last three terms of (14) are precisely
the support bound for that scalar combination of the three reading errors.
The first term retains the two complete-flow remainders and the same
unknown gain's upper bound.

For exact recorded points with per-reading error \(\delta\), the two
increment errors obey
\[
|e_1|\le2\delta,\quad |e_2|\le2\delta,\quad
|e_1+e_2|\le2\delta. \tag{15}
\]
These describe their shared-middle-error hexagon, not two independently
chosen reading pairs. Both inverse rows in (13) have opposite-sign entries.
Thus, for equal reading-error allowances, (14)'s scalar noise radius equals
the bound obtained from separate \(2\delta\) increment allowances.
Correctly retaining the shared reading is not a claim of narrower marginal
intervals in this particular sign pattern. The final coordinate rectangle
still discards correlations between its two coordinates.

Outward arithmetic encloses the inverse rows, centers and radii rather
than replacing them with point estimates. Intersect the resulting
\((U,V)\) rectangle with the necessary prior coordinate ranges
\[
\begin{aligned}
U&\in[G_-\cos b_+,\ G_+\cos b_-],\\
V&\in[G_-\sin b_-,\ G_+\sin b_+].
\end{aligned} \tag{16}
\]
The prior branch lies strictly in the first quadrant. Empty strict
intersections exclude the combined premises; touching endpoints remain
admitted. For a retained rectangle, project outward through
\[
b=\operatorname{Arg}(U+iV),\qquad
G=\sqrt{U^2+V^2}, \tag{17}
\]
and intersect each projection with its corresponding prior. Do not
normalize \((U,V)\) to a unit vector before checking the unknown gain.
Unresolved argument or norm arithmetic cannot certify a result.

Every compatible full state and held gain yields values of \(b,G\)
within these outer projections. Their Cartesian product need not be jointly
realizable, and
nonempty marginals do not prove that any full trajectory yields the supplied
readings. An incompatible result excludes the joint assumptions without
identifying which preparation, observation or model premise failed.

The uniquely defined original geometric statistic is the pre-probe donor
long-arc mean
\[
B_{\mathrm{initial}}
 =\frac{4\pi-(\theta^-_1-\theta^-_0)}8
 =b-\frac{v_1-v_0}{8}. \tag{18}
\]
Since both ports have degree three,
\(|v_1-v_0|\le\sqrt{2/3}\,Y\le Y\). Expand the nominal angle
projection by \([-Y/8,Y/8]\), using the **original** phase radius,
and do not clip it back to the nominal prior. Neither instantaneous
interior event changes the port phases, but their continuous evolution
can change this statistic. The report concerns its original value, not
its value at the middle or final reading.

<a id="sine-two-pulse-conditioning"></a>
## A rational conditioning example without a response campaign

The following constants demonstrate an informative error regime; they are
not a selected hidden source, recorded observation or frozen experiment:
\[
a_1=1/4,\quad a_2=3/4,\quad h=2^{-21},\quad
X=Y=2^{-40},\quad [G_-,G_+]=[1,2],\quad
\delta=2^{-60},\quad \tau_j\le\delta. \tag{19}
\]
Retain the full priors (1). Their original acute guard passes. The example
does not assert that the phase-input states remain acute.

Elementary sine bounds and \(\gamma>1/3216\) give
\[
\sin(\alpha_2-\alpha_1)>1/5,\qquad
\sin(3a_1/2)>1/3,\qquad \sin(3a_2/2)>4/5,
\]
so \(K_1>h/4824\) and \(K_2>h/2010\). If \(S_j\) is the
sum of absolute entries of inverse row \(j\), then
\[
S_1<\frac{41205}{4h},\qquad S_2<\frac{34170}{h},\qquad
\sqrt{S_1^2+S_2^2}<\frac{36000}{h}. \tag{20}
\]
The first bound uses \(\sin\alpha_2<3/8\) and
\(\sin\alpha_1<1/8\); the second uses both cosines at most one.
With \(\tau_j+\delta\le2\delta\), (14) bounds the Euclidean
norm of the two coordinate radii by
\[
R=\frac{72000E}{h}+\frac{144000\delta}{h}
 <\frac1{3072}. \tag{21}
\]
The last inequality is an exact rational substitution into (8) and (11),
so it evaluates no selected reading or trajectory.

For data produced by an admitted state, the true \(z\) belongs to the
ideal rectangle and has norm \(G\ge1\). The rectangle has diameter
at most \(2R\); every point has norm at least \(1-2R\).
It remains in the first quadrant because
\(\cos(3/2)>1/16\), \(\sin(11/8)>9/10\), and
\(2R<1/1536\). The norm is one-Lipschitz, while the argument
gradient has norm at most \(1/(1-2R)\). Consequently
\[
\begin{aligned}
\operatorname{width}(G)&\le2R<1/1536,\\
\operatorname{width}(b)&\le\frac{2R}{1-2R}<1/1535,\\
\operatorname{width}(B_{\mathrm{initial}})
 &\le\frac{2R}{1-2R}+Y/4<1/1024.
\end{aligned} \tag{22}
\]
Prior intersections cannot enlarge these projections. This establishes
feasibility in exact arithmetic. A numerical report still has to retain
outward coefficient and projection errors and pass its own availability
checks. No synthetic chosen reading establishes independent calibration
or a finite reserved-response verdict.

<a id="sine-two-pulse-boundary"></a>
## Implementation and evidence boundaries

The [implementation](../../src/tnfr/physics/relational_sine_two_pulse_inference.py)
reuses the original affine geometry, shared exact/represented scalar
admission and outward trigonometric, argument and square-root arithmetic.
It consumes three primitive recorded intervals, not a hidden source,
incoming forward report or cached verdict. The
[contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-two-pulse-inference)
owns the input and availability rules, and the
[tests](../../tests/physics/test_sine_two_pulse_inference.py) exercise the
independent algebra, scalar boundaries, rank and correlated-error behavior.
These controls establish implementation behavior, not external observations.

The earlier [two-time constitutive comparison](SINE_PAIR_INTERACTION.md#sine-pair-receiver-two-time)
also retains one state and one constant unknown through successive readings,
but uses a different conservative doubled-C5 model and a constitutive
coefficient. Its coefficients and verdicts do not transfer here. The
[hidden-state observation owner](SINE_ENVIRONMENTAL_MEMORY.md#sine-hidden-state-observability)
similarly distinguishes rank, necessary compatibility and joint existence;
it does not supply this protocol's observation or gain model.

Known clock and law scales are essential to interpreting (17) as a gain
bound. If either is left unknown, the leading response can combine it with
gain; this theorem does not identify those scales separately. The receiver,
full nodal residuals and common offset remain unobserved. Acquisition,
event funding, subsequent maintenance and physical identification retain
their own obligations.

The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns any later reserved two-input experiment. It would need one complete
source trajectory, both declared events, a held sensor, three readings,
information exclusion and frozen numerical/error/resolution budgets before
evaluating its response. This conditional theorem starts no such campaign
and changes none of the earlier calibrated theorem, prospective protocol,
source archives or retained responses.


<a id="sine-two-pulse-inference-protocol"></a>
## Prospective reserved software evaluation

This separately admitted protocol applies the preceding conditional theorem;
it supplies no physical observation or independent sensor calibration. The
preceding theorem and conditioning example remain unchanged. The source,
observation and stopping conditions below are fixed before any of these
reserved responses are evaluated. There is no evaluated result in this
prospective section.

### Full source family and hidden cases

Use the same eighteen-node support, degree metric, continuous phase lifts,
complete law and fast structural clock as (1)--(6). The public inputs are
\[
[b_-,b_+]=[11/8,3/2],\quad [c_-,c_+]=[2/3,1],\quad
X=Y=2^{-40},\quad [G_-,G_+]=[1,2],
\]
\[
(a_1,a_2)=(1/4,3/4),\quad h=2^{-21},\quad\delta=2^{-60}. \tag{23}
\]
The gain interval is a supplied prior, not the output of a calibration
experiment. Law, clock and both input amplitudes are held exactly as declared.

The three deterministic cases use the following primitives, withheld from
the inverse worker:

| Case index \(k\) | Donor \(b\) | Receiver \(c\) | Held gain \(G\) |
| --- | --- | --- | --- |
| 0 | \(89/64\) | \(17/24\) | \(9/8\) |
| 1 | \(91/64\) | \(19/24\) | \(11/8\) |
| 2 | \(95/64\) | \(23/24\) | \(15/8\) |

For node \(i=0,\ldots,17\), define exact raw residuals
\[
\widetilde u_i=\frac{(i+2)(k+2)}{2^{54}},\qquad
\widetilde v_i=\frac{((5i+3k)\bmod23)+1}{2^{54}},
\qquad u=P_M\widetilde u,\quad v=P_M\widetilde v. \tag{24}
\]
Use common means \(m_x=(k+2)/11\), \(m_\theta=-(k+2)/13\)
and the full source (2). The nominal eighteen-phase vector is reconstructed
from the affine source family, including its single full degree-weighted
centering. No equilibrium, ideal target or acquired-source flag is substituted.

Every centered form and phase residual is nonzero in each case. Their exact
weighted sums vanish; their squared degree norms are at most \(2^{-80}\).
These preparation checks use only rational primitives, not an evaluated
response. For example, the centered raw coordinate ranges give the
conservative squared bounds \(40\cdot76^2/2^{108}\) for form and
\(40\cdot23^2/2^{108}\) for phase, both strictly below \(2^{-80}\).
The hidden nominal and actual source angles are distinct:
\[
B_{\mathrm{initial}}=b-\frac{v_1-v_0}{8}
=b-\frac5{2^{57}}. \tag{25}
\]
The recorded source arrays and their checks must retain this difference.

### Two full flows and one sensor history

At global time zero supply the phase jump \(q/4\), evolve the full law
for \(h\), supply the second jump \(q/2\), and evolve for another
\(h\). Form is continuous across both events. Use the existing
[full-state readout producer](../../src/tnfr/physics/relational_sine_two_port_readout.py)
with two successive order-four calls. The second call's initial form and
phase boxes must be exactly the respective eighteen-coordinate slices of
the first call's complete validated endpoint. Its phase increment is
\(a_2-a_1=1/2\), not \(a_2\).

Each call uses one fixed source-box Picard/Taylor step: the shared
[smooth-flow kernel](../../src/tnfr/mathematics/_validated_taylor.py), its
sixteen-iteration Picard budget, dyadic-128 outward arithmetic, source-box
coefficients through order four, and a fifth-order whole-tube remainder.
Require strict Picard inclusion and the complete duration in both calls.
The autonomous solver's local time starts at zero in each call; the retained
protocol records their global starts as \(0\) and \(h\). This is a local
clock origin for the same law, not a restart of the physical state.

The first enclosure contains every trajectory from its source box. At the
second event, adding the fixed phase vector to that endpoint box therefore
contains every actual post-event state. The second validated step encloses
their continuations. Cartesian wrapping may discard correlation and widen
bounds; it does not justify shrinking the carried state or resetting any
coordinate. Retain all thirty-six initial, tube and endpoint rows of both
calls, together with their derivative and inclusion evidence. The global
smooth-law domain does not certify whole-window acuteness or identity.

Within case \(k\), one gain from the table and offset \(O=(2k+3)/11\)
remain held for all three scalar readings. Their fixed additive errors are
\[
(\eta_0,\eta_1,\eta_2)
=2^{-61}\bigl(k-1,\ 1-k,\ (-1)^k\bigr). \tag{26}
\]
Each lies within the public per-reading bound \(\delta\). Gains and
offsets may differ between cases; neither may change within a case.
The three true-readout enclosures are the first call's baseline, its
endpoint readout, and the second call's endpoint readout. Applying the
held affine sensor and each fixed error yields three recorded intervals.
The middle reading is constructed once and shared by both increments;
it equals the second call's pre-event form baseline before sensor errors.
Do not draw or synthesize a separate second middle reading.

Require each recorded interval half-width to be at most \(\delta\).
This numerical interval width is additional to the declared additive
sensor-error bound. It encloses one software-generated scalar reading;
it does not represent a second independent observation or sensor noise.

### Public inverse packet and declared controls

A fresh inverse worker receives only the nine primitive inputs accepted by
`infer_sine_two_pulse_geometry_gain`: `bulk_angle_bounds`,
`receiver_short_angle_bounds`, `form_radius`, `phase_radius`,
`phase_increments`, `probe_duration`, `recorded_reading_bounds`,
`readout_error_bound` and `readout_gain_bounds`. Only the three recorded
intervals come from the producing path. The inverse receives no hidden
\(b,c,G,O\), nodal state, error realization, producer report, endpoint
box, actual angle or prior verdict. Retain the exact public JSON packets,
worker code and allowlist check. This checks information exclusion in the
declared software path; it is not cryptographic blindness or independent
execution authentication.

For each unchanged triple of recorded intervals, also evaluate:

- A false donor prior \([11/8,353/256]\), retaining the other primary
  inputs; require `incompatible`.
- A false gain prior \([31/16,2]\), retaining the other primary inputs;
  require `incompatible`.
- Equal cumulative amplitudes \((1/4,1/4)\), retaining the other primary
  inputs; require `unavailable` with the rank-deficiency reason. This is an
  inversion-method control on the same packet, not a generated equal-input
  trajectory or a proof that the full nonlinear law has no information.

Finally compare both observed increments with the complete phase-blind
alternative \(x'=-Ax,\ \theta'=\gamma Ax\), supplied with the same
original source and phase events. Its form evolution ignores those events,
and heat contraction preserves \(\|P_Mx\|_M\le X\) at all times.
Since \(\|q\|_{M^{-1}}=1\) and \(\|A\|_M\le2\), each elapsed
window, including the second, satisfies
\[
|r_j-r_{j-1}|\le 2G_+hX+2\delta,\qquad j=1,2. \tag{27}
\]
Require each retained recorded-increment enclosure to be disjoint from this
symmetric interval. This is a bound from the same original source, with no
new heat trajectory or reset, and excludes only the declared alternative.

### Joint stopping rule and retention

For every case require admitted source priors and full residual norms;
both complete thirty-six-coordinate horizons and strict Picard margins;
exact first-endpoint/second-source association and both declared events;
one held sensor with one middle reading; each recorded half-width within
budget; and the public-only input boundary. The primary inverse must be
available, rank certified and `bounded_candidate`; its nominal angle,
original actual angle and gain marginals must contain the respective
hidden primitives. Require both the actual-angle and gain widths to be
strictly below \(1/1024\). All three inverse controls and both
phase-blind increment exclusions must pass as declared. Optional
whole-window acuteness is not part of this stopping rule.

The source base is immutable revision
`d86753763fcdd65eddf52c87dd3c5c74348129ce`. Before the first response, retain
this prospective proof, the exact protocol, source inventory, producing
and inverse-worker code, numerical budgets and stopping predicates under
`docs/assets/sine_formed_classes/two-pulse-inference-v1`. The source archive
records the base and any actual overlays; unrecorded working-tree changes
cannot supply the evaluated code. Keep the original prior protocols,
archives and responses unchanged.

Retain the first attempt, including any unavailable response or export
failure. Do not change the cases, priors, errors, horizon, arithmetic or
thresholds to obtain a passing record. Save all hidden primitive arrays,
both complete forward reports, the three readings, exact public packets,
all inverse reports and per-condition outcomes, followed by a manifest.
The saved result tests this finite software inference chain under its
supplied premises. It does not identify a physical sensor, derive the
preparation or support, prove joint realization of every retained angle/gain
pair, account for event work, or certify subsequent maintenance.


<a id="sine-two-pulse-inference-result"></a>
## Retained first evaluation and claim boundary

The [saved response](../../docs/assets/sine_formed_classes/two-pulse-inference-v1.json)
has status `certified_reserved_joint_inference`. All sixty-six fixed
conditions passed in the first reserved attempt across the three cases.
Both complete-flow windows, their full-state handoff and phase-only events,
the one-middle-reading association and public-only inverse requests passed
without a retry, changed budget, source reset or earlier producer rerun.

The primary marginals contain every declared nominal angle, its actual
original value \(B_{\mathrm{initial}}=b-5/2^{57}\), and the respective
held gain \(9/8,11/8,15/8\). The following decimal endpoints are rounded
outward for display; widths are approximations to the saved exact rational
differences, rather than differences of those rounded display endpoints.

| Case | Original actual-angle enclosure (radians) | Angle width | Gain enclosure | Gain width |
| --- | --- | --- | --- | --- |
| 1 | `[1.390535815643, 1.390714000311]` | `1.781846672682e-4` | `[1.124785196410, 1.125213380441]` | `4.281840298225e-4` |
| 2 | `[1.421806401222, 1.421943436179]` | `1.370349557123e-4` | `[1.374785991809, 1.375212314611]` | `4.263228007694e-4` |
| 3 | `[1.484331260848, 1.484418628125]` | `8.736727612697e-5` | `[1.874788260847, 1.875209614522]` | `4.213536743246e-4` |

Every exact actual-angle and gain width is strictly below \(1/1024\).
The largest recorded-reading half-width is below \(3.09\times10^{-38}\),
well inside the separate \(2^{-60}\) numerical budget. Both false-prior
requests are `incompatible` in every case; each equal-amplitude request
abstains as `unavailable` for the declared leading-rank limitation. Both
recorded increments exclude the phase-blind interval (27), whose exact
radius here is \(2^{-58}=4\delta\). None of these controls substitutes
a second preparation or an independently generated middle reading.

The optional `whole_window_acute_certified` flag is false in all three
primary reports. The global full-law response bounds still support the
inverse, which refers to the original pre-probe geometric statistic.
The absent acute certificate is not a failed stopping condition and supplies
no post-input identity, recovery or maintenance claim.

The [protocol](../../docs/assets/sine_formed_classes/two-pulse-inference-v1.protocol.json),
[source archive](../../docs/assets/sine_formed_classes/two-pulse-inference-v1.source.zip)
and [manifest](../../docs/assets/sine_formed_classes/two-pulse-inference-v1.manifest.json)
retain the complete source recipe, producing code, first responses and
fixed verdicts. The archived prospective proof has 26,821 bytes and SHA-256
`2cae8b0d9257edf9ce7b78b9ca6d3feb0c11c331341ae517636a22da3ec5b285`;
its content is preserved above, independently of routine checkout newline
conversion. The source archive SHA-256 is
`5abff67d487287ad3d4949d970f5c6a448bd4e62dec8a212e817008ede031e38`.
The [retained evidence audit](../../tests/physics/test_sine_formed_evidence.py)
checks the saved primitives and numerical evidence without regenerating
these reserved responses or calling the inverse again.

This is finite evidence that the declared software observation chain
constrains original geometry and a held gain simultaneously, conditional on
the known complete law, clock, exact inputs, source family and observation
model. It is neither independent physical calibration nor exact joint
identifiability: retained marginal pairs need not all be realizable, and
receiver geometry and fine state remain unresolved. The result does not
justify an unknown clock or law scale, autonomous support or preparation,
event funding or physical constituent identification. Any continuation
requires a separate admission in the
[sole execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).
