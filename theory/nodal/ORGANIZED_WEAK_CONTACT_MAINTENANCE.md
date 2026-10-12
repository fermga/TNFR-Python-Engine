# Retained organized identity under sufficiently weak external contact

<a id="organized-weak-contact-maintenance"></a>

## 1. A controlled change of contact strength

The [unit-contact problem](ORGANIZED_CONTACT_MAINTENANCE.md) leaves
stationary stability in the original complete marked class unresolved.
This result changes one supplied model parameter: all 45 external spokes
have the same positive weight eta. It proves stationary existence and
all-future identity retention for a declared weak-contact range, including
the original acquired and observed source family. It is not a solution
of the unit-weight stationary problem.

| Item | Declared model and scope |
| --- | --- |
| Coordinates and graph | The same 54 nodes (c,j), five host children c=0,...,4, donor c=5, and slots j=0,...,8; retain all signed forms and circular phases |
| Old support | All 99 old edges have weight one: six internal C9 cycles and nine host rim C5 cycles |
| Changed support parameter | Every same-slot host-donor spoke has weight eta>0; eta is dimensionless and distinct from the constitutive deformation parameter used elsewhere |
| Complete law | Both weighted Laplacian and weighted sine current consume the same eta. Held unit capacities, uniform quarter factors and the smooth full-circle phase domain remain supplied |
| Clock | Prime denotes the existing tau=(1023/1024)t clock; both rows transform together |
| Means | Work on the zero total form leaf and any declared conserved lifted phase mean, including the original donor origin beta |
| Marked identity | Six internal winding-one cycles, nine old rim winding-one cycles, and the original 32 independent zero mixed host periods; new spoke loops receive no prescribed winding |
| Generic entry | The initial old edges lie in their original acute fixed-branch cell and the full weighted storage is below the barrier in section 2 |
| Source interval | 0<eta<=1/100000, with all twelve original source allowances, all 96 relative directions and every supplied donor origin beta |
| Events and execution | The contact event changes weights/support only, with coordinates unchanged. No later forcing or event is supplied. This is a detached complete-law theorem, not native operator admission or an occurrence rule |
| Evidence boundary | Analytical inequalities and static algebraic controls; no stationary search, trajectory, source-producer replay, response fit or new physical record |

Let \(E_{\rm old}\) and \(E_{\rm sp}\) denote the old edges and spokes.
Define

\[
 \begin{split}
 L_\eta&=L_{\rm old}+\eta L_{\rm sp},\\
 (S_\eta)_a&=\sum_{b\sim a}w_{ab}
                 \sin(\theta_b-\theta_a),\\
 V_\eta(\theta)&=\sum_{E_{\rm old}}[1-\cos\delta_e]
                 +\eta\sum_{E_{\rm sp}}[1-\cos\delta_e],\\
 x'&=(-L_\eta x+\gamma S_\eta)/4,\qquad
 \theta'=\gamma L_\eta x/4,\qquad
 \gamma=1/(1023\pi).
 \end{split}
 \tag{1}
\]

Here w=1 on old edges and w=eta on spokes. Changing only the sine
term, only the form exchange or only a recorded storage cost would define
a different model. Reciprocity gives the actual complete storage law

\[
 H_\eta=\tfrac12x^TL_\eta x+V_\eta,\qquad
 H_\eta'=-\tfrac14\|L_\eta x\|_2^2,\qquad
 (\mathbf1^Tx)'=(\mathbf1^T\theta)'=0.
 \tag{2}
\]

Eta is supplied contact strength, not a selected law, inferred physical
constant or control inferred from dissipation. Eta=0 is a disconnected
reference used in the proof; every dynamical conclusion below requires
strictly positive eta.

## 2. The old marked cell retains its complete boundary barrier

Use the original lifted branches: internal closing edges and host rim
closing edges receive their fixed 2*pi addition. Every old edge gap
lies in (-pi/2,pi/2). The resulting cell preserves the complete marked
class, including the zero mixed host periods, not just fifteen winding
labels. Spokes are unrestricted.

Put

\[
 \begin{split}
 V_n&=n[1-\cos(2\pi/n)],\\
 b_n&=1+(n-1)[1-\cos(3\pi/(2(n-1)))],\\
 V_*&=6V_9+9V_5,\qquad
 B=\min\{b_9-V_9,\ b_5-V_5\}>\frac7{500}.
 \end{split}
 \tag{3}
\]

The strict rational lower bound is the existing
[cycle-face inequality](ORGANIZED_OUTER_WINDING.md#5-every-product-cell-boundary-and-all-future-capture).
Its method applies here because the 99 old unit edges are partitioned
into six C9 and nine C5 cycles. Convexity of 1-cos on the closed acute
interval bounds their total phase cost below by \(V_*\).

On a positive face of an n-cycle, one angle equals pi/2 and the remaining
angles sum to 3*pi/2, giving \(b_n\) by Jensen. A negative internal face
instead has remaining mean 5*pi/16 and a larger cost. A negative rim
face is empty: one angle -pi/2 and four at most pi/2 cannot sum to 2*pi.
These cases cover all 198 signed old-edge faces: 54 positive and 54
negative internal faces, 45 positive rim faces and 45 empty negative rim
faces. No joint attainability of separate cycle minima is assumed.

The weighted spoke potential and the full form storage are nonnegative.
Consequently every actual boundary state satisfies

\[
 H_\eta\ge V_*+B.
 \tag{4}
\]

The old graph has two components. After fixing an origin in each, its
closed acute cell is a bounded polytope of relative lifts. Restoring
the relative host-donor origin on the circle makes its closure compact
modulo common rotation. A fixed lifted mean does not require pretending
that this relative origin is fixed.

**Entry and retention theorem.** If the initial old phases belong to
this cell and \(H_\eta(0)<V_*+B\), no future old edge reaches an acute
face under (1). A first exit would contradict (2)--(4). The strict
energy margin separates the retained phase sublevel from all old faces.
All six internal and nine rim windings and all original mixed host
periods therefore persist for every future time.

For every fixed eta>0 the complete graph is connected. Thus the form
part of \(H_\eta\) bounds all forms on the zero-mean leaf. Together with
the compact circular phase description this makes the retained sublevel
precompact. This is an all-future trapping argument for the full state;
no carrier restriction, acute-spoke premise or phase reset is used.

## 3. Positive contact admits stationary minima and a stable retained set

Consider the reference phase circle

\[
 x=0,\qquad
 \theta_{c,j}=\psi_j+\frac{2\pi(c-2)}5\quad(c<5),\qquad
 \theta_{5,j}=\psi_j+\beta,\qquad
 \psi_j=\frac{2\pi(j-4)}9.
 \tag{5}
\]

It lies strictly inside the old cell for every circular beta. The fifth
roots of unity sum to zero, so its unweighted spoke potential is exactly
45, independently of beta. Therefore

\[
 H_\eta\big|_{\text{reference circle}}=V_*+45\eta.
 \tag{6}
\]

For every

\[
 0<\eta<B/45,
 \tag{7}
\]

the reference circle lies below the boundary barrier. Minimizing \(V_\eta\)
on the compact closed old cell, including its relative-origin circle,
then gives an interior minimum: its value is at most (6), whereas every
boundary value is at least \(V_*+B\). At that minimum all full phase
derivatives vanish and the full relative phase Hessian is positive
semidefinite. With x=0 it is a stationary point of both rows in (1).

This proves existence in the original complete marked class. It does
not make the reference circle itself stationary when eta>0, select an
origin, or prove a strictly positive Hessian. The minimum may be
degenerate.

Let \(\mathcal M_\eta\) be the compact set of phase minima just obtained,
modulo common rotation. The equilibrium set with x=0 and phase in
\(\mathcal M_\eta\) is Lyapunov stable in the relative circular phase
space with mean-zero forms. Equivalently one may use its full periodic
preimage on a fixed-mean lift leaf; a single compact choice of lifted
representatives is not asserted to be invariant.
Subtract its minimum storage value from \(H_\eta\). Within the trapping
cell this difference is nonnegative and vanishes exactly on that set:
connectedness makes its form contribution zero only at x=0. Compactness
gives a positive storage separation from the complement of any
neighborhood of the set inside a bounded retained sublevel. Continuity
provides a smaller initial neighborhood below this separation, and
(2) prevents exit. This is set stability, without an assumed isolated
or strictly positive stationary point.

There is a nonempty open set of initial full states below the barrier,
since (6) is strict. Every such state admitted inside the cell retains
the original identity. The actual acquired preparation requires its own
inclusion, supplied in section 5.

## 4. The weighted complete law still converges to one equilibrium

The [complete-contact convergence method](ORGANIZED_CONTACT_MAINTENANCE.md#full-contact-convergence)
must be re-admitted for (1). In the range (7), eta<1: indeed B<5 already
follows from (3). Edgewise comparison gives

\[
 0<L_\eta|_{\mathbf1^\perp},\qquad
 L_\eta\le L_1,\qquad
 \mu_\eta=\|L_\eta\|\le10,\qquad
 \ell_\eta=\lambda_{\min}(L_\eta|_{\mathbf1^\perp})>0.
 \tag{8}
\]

The positive lower bound is for each fixed eta. No uniform positive
gap is asserted as eta approaches zero.

Set \(k=\gamma^2/16\), \(U_\eta=kV_\eta\),
\(g_\eta=\nabla U_\eta\), \(P_\eta=L_\eta^{-1}\) on the relative
space, and \(v=\theta'=\gamma L_\eta x/4\). Differentiating the actual
second row gives

\[
 P_\eta v'+\tfrac14v+g_\eta=0,\qquad
 E_\eta=\tfrac12v^TP_\eta v+U_\eta=kH_\eta,\qquad
 E_\eta'=-\tfrac14\|v\|^2.
 \tag{9}
\]

The Hessian of \(V_\eta\) is the actual weighted cosine Laplacian, between
\(-L_\eta\) and \(L_\eta\) in Loewner order. The same proof function
\(F_\eta=E_\eta+g_\eta^Tv/8\) therefore obeys, with all coefficients
now consuming eta,

\[
 F_\eta'
 \le-\frac{17}{128}\|v\|^2-\frac1{16}g_\eta^TL_\eta g_\eta
 \le-\min\{17/128,\ell_\eta/16\}
                  (\|v\|^2+\|g_\eta\|^2).
 \tag{10}
\]

These estimates use k<1/16 and (8). \(F_\eta\) is an analytic proof
function, not a substituted storage or law.

Bounded form energy, compact circular phase space and the unchanged
autonomous loss identity put every omega-limit point at \(v=g_\eta=0\).
The potential and \(F_\eta\) are real analytic. The finite-chart analytic
gradient inequality and finite-length argument in the linked owner
therefore apply with \(P_\eta\) and \(\ell_\eta\): every finite
zero-form-mean initial state has one limiting equilibrium, x tends to
zero, and the original continuous phase lifts have finite total variation
and converge. This derives lift boundedness rather than assuming it.
The graph metric \(P_\eta\) carries no identification with physical mass.

In particular every state trapped by section 2 converges to an
equilibrium strictly inside the same old acute cell, retaining all its
marked information. The limit is not selected and need not belong to
\(\mathcal M_\eta\); convergence by itself does not imply that the
limiting Hessian is positive semidefinite. No convergence rate uniform
in eta, finite settling deadline or statement at eta=0 follows.

<a id="weak-contact-actual-source-capture"></a>
## 5. The original acquired and observed source enters the weak-contact trap

Keep the complete [outer acquisition](ORGANIZED_OUTER_WINDING.md),
[sequential observation word](ORGANIZED_SEQUENTIAL_OBSERVATION.md) and
[independent donor preparation](ORGANIZED_EXTERNAL_REGISTRATION.md).
The five host sources and donor retain all twelve original allowances
in [0,1e-10], their separate zero-sum errors and all 96 relative
directions. Every beta is supplied at donor primitive preparation.
No reached coordinate, residual, clock, command or waiting interval is
changed. Only the subsequently supplied spoke weight is eta.

The retained acquisition inequalities give, before the observation word,

\[
 H_H(0)<5V_9+9V_5+\frac{11}{1000},\qquad
 \|x_H(0)\|_\infty\le X=\frac1{100}.
 \tag{11}
\]

Use the new analytic upper bound \(\bar\gamma=1/3000>\gamma\), not a
re-evaluated archived interval endpoint. Over the original word of
duration T=1/5000, put

\[
 \begin{split}
 B_o&=X+\bar\gamma T<\frac{101}{10000},\\
 d_x&=(2B_o+\bar\gamma)T<\frac1{200000},\\
 m&=2\bar\gamma B_oT<\frac1{500000000}.
 \end{split}
 \tag{12}
\]

The degree-four host heat contraction bounds every form by \(B_o\),
including all fifteen supplied phase events. Forms never jump. The full
form row then bounds each net form change by \(d_x\). The phase row
bounds each continuous phase drift by m. The final commanded phase
offset is exactly zero, while that drift and the form change remain.

Each of the 90 host edges consequently has net form-gap change at most
\(2d_x\) and phase-gap change at most 2m. Its initial and final form gaps
have magnitude at most \(2B_o\). The square difference identity and the
global Lipschitz constant one of 1-cos give the endpoint bound

\[
 H_H(T)-H_H(0)
 \le360B_od_x+180m<\frac1{50000}.
 \tag{13}
\]

This bound compares actual endpoints. It neither cancels the paid work
of intermediate events nor treats a final inverse command as restoration
of state. All earlier acquisition work, event work and continuous losses
retain their original accounting.

The [recovered donor bounds](ORGANIZED_EXTERNAL_INTERACTION.md#2-actual-source-and-observation-history-transport)
give form and relative-phase norms at most q=1e-70. Its isolated twist
has zero phase gradient and internal Laplacian norm below four, hence

\[
 H_D\le V_9+4q^2.
 \tag{14}
\]

The phases at contact also lie in the required old acute cell. This
follows separately from the original source and observation-history
inclusions. With zeta=1e-6, A=7/5 and \(\alpha=2\pi/9\), the final
host rim gaps have magnitude at most \(A+2\zeta+2m<\pi/2\), and the
host internal gaps lie in
\([\alpha-2\zeta-2m,\alpha+2\zeta+2m]\subset(0,\pi/2)\).
The donor internal gaps lie in
\([\alpha-2q,\alpha+2q]\subset(0,\pi/2)\). The observation word
already preserves the original branches and mixed periods. Energy alone
is not used to infer this initial phase admission.

The new support event leaves coordinates unchanged. Its complete
weighted storage jump is

\[
 W_\eta=\eta\sum_{c,j}
 \left[\tfrac12(x_{D,j}-x_{H,c,j})^2+
             1-\cos(\theta_{D,j}-\theta_{H,c,j})\right].
 \tag{15}
\]

No old edge changes weight. Since \(B_o+q<11/1000\), for every
\(0<\eta\le1/100000\) and every beta,

\[
 W_\eta
 \le45\eta\left[\tfrac12(B_o+q)^2+2\right]
 <\frac{901}{10^6}.
 \tag{16}
\]

This is supplied positive contact work, not an event funded by past
loss. Its bound does not select when the contact occurs.

Combining the actual endpoint estimates gives

\[
 \begin{split}
 H_\eta(0)-V_*
 &<\frac{11}{1000}+\frac1{50000}+4q^2+\frac{901}{10^6}
 <\frac3{250},\\
 B-[H_\eta(0)-V_*]&>\frac1{500}.
 \end{split}
 \tag{17}
\]

Here time zero is the unchanged contacted endpoint, not a state reset.
The full joined form total is zero, and the joined lifted phase total
is the original 9*beta. The interval for eta lies inside (7), because
B>7/500.

Thus the entire original outer-one source family, every original
allowance tuple and every donor origin remains in the original old
acute cell for all future time under this weak-contact law. All fifteen
winding-one identities and all 32 independent zero mixed host periods
persist. The full forms tend to zero and the continuous phase lifts
converge to one, still unspecified, retained-identity equilibrium. A
possible degenerate or saddle limit is not excluded by source convergence.

These are new analytical consequences of the retained source inequalities
under the explicitly changed contact model. No frozen producer was
replayed, no archived response was re-evaluated, and no original source
or response flag is amended. In particular the original unit-contact
response thresholds and horizons do not become weak-contact predictions.

## 6. Meaning and remaining boundary

The positive weak-contact interval proves that the same graph topology
and complete marked identity are compatible with stationary existence and
indefinite maintenance. Quantitative contact strength matters: assigning
the same eta consistently in both evolution rows and storage preserves
enough of the old energy barrier to admit the actual reached sources.
The supplied coupling has not been selected by TNFR's nodal identity or
by a physical observation.

The unit-weight problem at eta=1 remains outside the proved interval.
Its finite common-host exclusions, complete-flow convergence and remaining
stationary feasibility question are unchanged. No critical coupling,
sharp transition, unit-contact stable target or unit-contact winding loss
has been established. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#contact-maintenance-certificate-admission)
owns that unresolved dependency.

The result gives no autonomous contact selector, event reservoir,
laboratory unit bridge or identification with physical matter. Its
reusable mechanisms are the full old-edge barrier, explicit new-edge
work, carried source-history estimate and analytic convergence under the
actual positive weighted law.
