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
An unresolved interval determinant is numerical unavailability, not an
exact rank-one proof. A nonzero determinant alone also gives no useful
resolution guarantee: conditioning must be compared with the full errors.

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
