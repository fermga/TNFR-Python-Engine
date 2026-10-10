# Complete-law admission for slow receiver acquisition

<a id="organized-receiver-slow-admission"></a>

The unchanged two-ramp preparation admits an exact second-order phase law
and a uniformly controlled first-order velocity approximation after its
initial form layer. At structural time 400, every noncentral receiver phase
actually moves in the direction of the donor ramp. Its slow-phase rate has magnitude
greater than \(3/20\), with the sign of \(j-4\). This strengthens the earlier
mixed-coordinate current result: the receiver's primitive phase, rather
than only \(\theta+\gamma x\), has the stated instantaneous motion.

For every later time, the complete phase velocity differs from the sine
descent field by less than \(11\times10^{-6}\) in Euclidean norm. This
uniform velocity bound does not identify the full trajectory with a
phase-only trajectory. It supplies a source-aware differential inclusion
and a precise corridor and terminal-capture contract. Eventual receiver
winding, continued donor retention and joined capture remain unresolved;
no reference trajectory or reserved response is evaluated here.

## Retained source, law and identity

The complete state, support and preparation are those of the
[receiver-mechanism owner](ORGANIZED_RECEIVER_MECHANISM.md#organized-receiver-source).
There are 27 labelled nodes, three unit C9 supports and the oriented
contacts \((0,1),(1,2),(2,0)\). Each contact uses offsets zero and one
with weights one half. All support is present at preparation; its weighted
Laplacian is \(L\), and every node has weighted degree four. Capacities
remain one. The structural clocks are
\[
 \tau=(1023/1024)t,\qquad s=\epsilon\tau,\qquad
 \gamma=1/(1023\pi),\qquad \epsilon=\gamma^2.                 \tag{1}
\]
These are fixed structural units, without a laboratory-time identification.
Write
\[
 A=L/4,\qquad S_i(\theta)=\sum_jw_{ij}\sin(\theta_j-\theta_i),
 \qquad F(\theta)=S(\theta)/4 .                              \tag{2}
\]
The actual detached weighted sine law is
\[
 x_\tau=-Ax+\gamma F(\theta),\qquad
 \theta_\tau=\gamma Ax .                                    \tag{3}
\]
No forcing, state reset, support event or evolving conductance is added.
This is neither native unique-neighbor phase pressure nor execution of a
registered operator. The nodal identity alone does not select (3).

Let \(q_j=j-4\), \(\alpha=2\pi/9\), and let \(Q\) contain
\(q\) in regions zero and one and zero in region two. Retain
\[
 m=\frac{2046}{9}(355/113)^2,\quad
 x_0=mQ+\eta_x,\quad \theta_0=\eta_\theta,
 \quad |\eta_{x,i}|\le r_x=10^{-10},\quad
 |\eta_{\theta,i}|\le r_\theta=10^{-10}.                    \tag{4}
\]
Both errors have separate exact zero sums within each original C9.
Every allowed combination is included. Only the global means remain
conserved after evolution begins. The common phase origin and continuous
lifts remain fixed. All 54 fine coordinates, boundary contacts and the
full preparation cost are retained. The nominal storage is \(150m^2\),
including \(78m^2\) on contacts; its error allowance and the regional
boundary-work identity remain in the linked source owner.

The relevant identity is principal-difference winding around each marked
C9. The earlier theorem establishes winding \((1,1,0)\) at \(\tau=200\)
and excludes any receiver branch crossing throughout its proved initial
delay. Neither that exclusion nor a directed phase velocity specifies an
eventual acquired identity. This retained constituent and boundary state
follows the [NFR ontology](../EMERGENT_ONTOLOGY.md#nfr-participation-in-form).

<a id="organized-receiver-inertial-law"></a>
## Exact second-order phase representation

All vectors below lie on the global mean-zero leaf. The actual connected
support makes \(A\) positive definite there, with
\[
 A\succeq gI,\quad g=1/10,\qquad \|A\|_2\le2,
 \qquad \|F\|_2\le M=\sqrt{27},\quad \|DF\|_2\le K=2 .       \tag{5}
\]
The last inequality follows from the symmetric weighted cosine Laplacian:
\(DF=-B\operatorname{diag}(w_e\cos\delta_e)B^T/4\), whose
absolute quadratic form is bounded by that of \(L/4\).
These bounds hold globally in the supplied continuous lifts, including
phase-branch passages. They require no all-time acute assumption.

Put \(v=\theta_s\). Equation (3) gives an invertible representation
on this leaf:
\[
 v=Ax/\gamma,\qquad x=\gamma A^{-1}v,\qquad
 \boxed{\epsilon A^{-1}\theta_{ss}+\theta_s=F(\theta).}       \tag{6}
\]
Indeed, differentiating \(\theta_\tau=\gamma Ax\), substituting
the form row and changing both derivatives to the same clock yields
\(\epsilon\theta_{ss}=A(F-\theta_s)\). Conversely, (6) and
the reconstruction of \(x\) give both original rows. The initial
velocity is fixed by the original source:
\[
 \theta(0)=\eta_\theta,\qquad
 \theta_s(0)=\frac{A(mQ+\eta_x)}{\gamma}.                  \tag{7}
\]
It is not a freely reset slow initial condition. Form memory is retained
as this velocity. If a nonzero global form mean were admitted, its conserved
value would also have to be retained separately; it is zero under (4).

Let
\[
 U(\theta)=\sum_{\{i,j\}}w_{ij}[1-\cos(\theta_j-\theta_i)],
 \qquad H=\tfrac12x^TLx+U(\theta).
\]
Since \(\nabla U=-4F\), the same storage becomes
\[
 \mathcal E:=H/4
   =\frac\epsilon2v^TA^{-1}v+\frac14U(\theta),\qquad
 \boxed{\mathcal E_s=-\|v\|_2^2.}                          \tag{8}
\]
Multiplication of (6) by \(v\) proves (8); it is the original loss
identity in the new coordinates and clock. The matrix \(\epsilon A^{-1}\)
is a derived structural coefficient for this model. Its occurrence does
not identify an inertial mass, mechanical energy or a physical equation.
Dropping the second derivative changes the model and needs an error bound.

<a id="organized-receiver-uniform-defect"></a>
## Uniform defect from the phase-descent field

Define the actual velocity defect
\[
 r=v-F(\theta)
   =\frac{Lx-\gamma S(\theta)}{4\gamma}
   =-\frac{x_\tau}{\gamma}.                                \tag{9}
\]
This is a consequence of both declared rows, not a pressure reconstructed
from the response. Differentiating in structural time gives the exact law
\[
 r_\tau=-[A+\epsilon DF(\theta)]r
                 -\epsilon DF(\theta)F(\theta).            \tag{10}
\]
Both \(r\) and the forcing in (10) are globally mean zero. For
\(\lambda=g-K\epsilon>0\), the Euclidean norm therefore satisfies
\[
 D^+\|r\|_2\le-\lambda\|r\|_2+\epsilon KM.                \tag{11}
\]
For nonzero \(r\), this follows by pairing (10) with
\(r/\|r\|_2\) and using (5); the upper right derivative extends
the inequality through zero.

The source gradient from the actual weighted support satisfies
\(\|LQ\|_2^2=2277/2\). Since \(F(0)=0\), a full-source
initial bound is
\[
 R_0=\frac{m\sqrt{2277/2}}{4\gamma}
       +\frac{2\sqrt{27}\,r_x}{\gamma}
       +2\sqrt{27}\,r_\theta\ \ge\ \|r(0)\|_2 .           \tag{12}
\]
It retains the two independent errors and does not replace the original
source by a recovered or stationary state. Solving (11) gives
\[
 \|r(\tau)\|_2\le R(\tau):=
 R_0e^{-\lambda\tau}
       +\frac{\epsilon KM}{\lambda}(1-e^{-\lambda\tau}).    \tag{13}
\]
In particular, the simpler monotone upper envelope
\(R_0e^{-\lambda\tau}+\epsilon KM/\lambda\) is available
without deciding whether \(R_0\) exceeds the asymptotic bound.

At the analytic release time \(\tau_0=400\), outward rational
enclosures give
\[
 R_0e^{-400\lambda}+\frac{\epsilon KM}{\lambda}
       <\rho:=\frac{11}{10^6}.                              \tag{14}
\]
The left side is approximately \(0.000010061734672\). It decreases
thereafter, so the complete law satisfies, for every \(\tau\ge400\),
\[
 \boxed{\|\theta_s-F(\theta)\|_2<\rho.}                   \tag{15}
\]
No phase trajectory was evaluated to choose (14). Time 400 is an analytic
release for a uniform estimate, not a reserved acquisition deadline.
For other nonnegative primitive error allowances, (12)--(14) must be
recomputed and may be unavailable. The original errors in (4) supply
the frozen conclusion.

The bound is uniform on velocity defect, not on distance to any chosen
first-order trajectory. Small persistent defects can accumulate or act
across a basin boundary. It would be invalid to replace (15) by an
infinite-time shadowing or common-attractor assertion.

The actual phase potential nevertheless obeys the useful inequality
\[
 U_s=-4F(\theta)\mathbin{\cdot}[F(\theta)+r]
 \le-4\|F(\theta)\|_2\,[\|F(\theta)\|_2-\rho].
\]
Thus it strictly decreases wherever \(\|F\|_2>\rho\) after
release. Near the critical set this inequality is inconclusive; the
complete storage (8) remains the exact monotone quantity everywhere.

<a id="organized-receiver-phase-motion"></a>
## Actual receiver phase motion at release

The source-controlled transport estimate in the
[mechanism proof](ORGANIZED_RECEIVER_MECHANISM.md#organized-receiver-memory)
applies at any declared time. Set \(P=\alpha Q\), and define
\[
\begin{split}
 A_0&=\gamma(m\sqrt{120}+\sqrt{27}\,r_x),\\
 B_0&=e^{-40}A_0+10\epsilon\sqrt{27},\\
 b_0&=\sqrt{120}|\gamma m-\alpha|
       +\sqrt{27}(r_\theta+\gamma r_x)
       +B_0+400\epsilon\sqrt{27}.
\end{split}                                                   \tag{16}
\]
Then
\[
 \|\gamma x(400)\|_2\le B_0,\qquad
 \|\theta(400)-P\|_2\le b_0<21/100000 .                    \tag{17}
\]
The phase radius is approximately \(0.000207559164\). The rational
ramp mismatch remains explicit: \(355/113\) is not set equal to
\(\pi\). At release the donor principal internal differences are
\(\alpha\) plus at most \(\sqrt2b_0\), while receiver differences
have magnitude at most \(\sqrt2b_0\). Their respective windings
are therefore still \((1,1,0)\). More strongly, for every
\(200\le\tau\le400\), the phase radius is bounded by
\[
 \sqrt{120}|\gamma m-\alpha|
 +\sqrt{27}(r_\theta+\gamma r_x)
 +e^{-20}A_0+10\epsilon\sqrt{27}+400\epsilon\sqrt{27}.
\]
This also has positive donor branch margin. Thus neither donor loses
its acquired winding between its certified time-200 state and release.
The earlier receiver-delay theorem already covers that whole interval.

The exact reference receiver current is
\[
 F_{2,j}(P)=\tfrac14(1+\cos\alpha)\sin(\alpha(j-4)).        \tag{18}
\]
The degree-four support and the sine Lipschitz bound give, nodewise,
\[
 |F_i(\theta(400))-F_i(P)|\le\sqrt2b_0 .                   \tag{19}
\]
For all eight noncentral receiver nodes, (15), (18) and (19) imply
\[
 \operatorname{sign}(j-4)\,\theta_{s,2,j}(400)
 >\tfrac14(1+\cos\alpha)\sin(\pi/9)-\sqrt2b_0-\rho
 >\frac3{20}.                                               \tag{20}
\]
The estimate keeps every permitted source error, so it is not restricted
to the exactly symmetric nominal source. No sign is claimed for the
central receiver node. In structural time the rate bound is multiplied
by \(\epsilon\), and in original time also by \(1023/1024\).
Positive instantaneous phase motion does not guarantee a branch crossing,
monotone motion at later times or retained donor identity thereafter.

<a id="organized-receiver-capture-contract"></a>
## Source-aware corridor and terminal-capture contract

Use elapsed slow time \(u=\epsilon(\tau-400)\). The actual
post-release phase is an absolutely continuous solution of the inclusion
\[
 \theta_u=F(\theta)+r(u),\qquad \|r(u)\|_2<\rho,
 \qquad \|\theta(0)-P\|_2\le b_0.                           \tag{21}
\]
The error \(r\) is the dynamically constrained defect (9), not a supplied
controller. The inclusion is an outer description: proving a property
for every admissible disturbance proves it for the full law, while an
unsuccessful inclusion bound does not refute acquisition.

For example, let a proposed differentiable phase corridor be described
by \(h_k(\theta)\le0\). A strict outward-boundary estimate
\[
 \nabla h_k(\theta)\mathbin{\cdot}F(\theta)
       +\rho\|\nabla h_k(\theta)\|_2<0                    \tag{22}
\]
on each active boundary, together with source inclusion, prevents a first
exit of the actual phase. A receiver-acquisition argument would also
need progress to a declared terminal set; invariance alone supplies no
entry or finite passage time. A directed phase or winding observation
does not replace this boundary and progress proof.

A separately admitted finite reference can instead provide a tube.
On an interval of slow length \(h\), let \(w(u)\) have global mean
zero, continuous declared lifts, and a rigorously bounded residual
\(\|w_u-F(w)\|_2\le\eta\). Suppose both the reference and its
prospective error ball lie in a convex domain \(D\) within the conserved
global mean-zero phase leaf, and freshly bound
\[
 \mu\ge\sup_{z\in D}
       \lambda_{\max}\bigl(DF(z)|_{\mathbf1^\perp}\bigr).   \tag{23}
\]
The symmetric Jacobian preserves \(\mathbf1^\perp\). The restriction
retains the actual error directions and permits a negative relative
growth bound; the full 27-coordinate Jacobian always has the common-phase
zero mode. Every error ball below is taken within the same mean-zero leaf.
If the incoming phase error is at most \(e_0\), then a sufficient
whole-interval error envelope is
\[
 e(u)=e^{\mu u}e_0+(\rho+\eta)\begin{cases}
  (e^{\mu u}-1)/\mu,&\mu\ne0,\\
  u,&\mu=0.
 \end{cases}                                                \tag{24}
\]
The proof pairs the error equation with its normalized error, uses the
signed logarithmic bound (23), and integrates its scalar inequality.
Strict inclusion of \(w(u)+\overline B_{e(u)}\) in \(D\)
closes the first-exit argument; a presumed future tube cannot certify
its own premise. The global choice \(\mu=K=2\) is always available,
but sharper local bounds may be essential. At the first interval take
\(w(0)=P\), \(e_0=b_0\); later intervals inherit the complete
preceding endpoint error, without resetting it to zero. Any residual,
state, lift, Jacobian or source premise must be re-admitted rather than
copied from a report's cached verdict.

Throughout that finite transport, each donor's fixed registered internal
edge must satisfy
\[
 |\delta_e(w(u))|+\sqrt2e(u)<\pi .                          \tag{25}
\]
The seam integers are the winding-one donor integers admitted at release.
Equation (25) retains each donor identity continuously. The receiver
must be allowed to cross its initial principal branch; an imposed
all-time acute receiver cell would exclude the outcome being tested.

For terminal capture, use the equal-support target \(\Theta=\alpha(q,q,q)\)
and the full-state barrier from the
[capture owner](ACQUIRED_ORGANIZED_NETWORK_CAPTURE.md#organized-network-capture-barrier).
The full form remains bounded for every \(\tau\ge400\) by
\(\|x(\tau)\|_2\le X_0:=B_0/\gamma\). At a proposed terminal
reference \(w_*\), with certified phase error \(e_*\), require
both
\[
 \max_{\text{fine }e}|\delta_e(w_*)|+\sqrt2e_*<2,            \tag{26}
\]
and
\[
 \boxed{
 U(w_*)-U(\Theta)+\|\nabla U(w_*)\|_2e_*
           +4e_*^2+4X_0^2<196/625 .}                       \tag{27}
\]
All edge integers in (26) are the fixed integers registered to
\(\Theta\), including the receiver and contact edges. The potential
and gradient in (27) use the actual complete weighted support.
The global Hessian bound \(\|\nabla^2U\|_2\le8\) proves the
phase remainder \(4e_*^2\); the same Laplacian bound proves the
form contribution \(4X_0^2\). The linear gradient term generally
does not vanish and must be retained. A phase-only energy or separately
passing marginal budgets cannot replace (26)--(27) on the same state.

If a source-admitted corridor or finite tube proves donor retention and
(26)--(27), the actual full state enters the joined trapping sublevel.
The existing barrier then proves winding one in every constituent for
all future time and convergence to the equal registered organization.
No such corridor or endpoint is asserted here. The contract identifies
the remaining nonlinear estimate without changing the source, law or clock.

The [implementation](../../src/tnfr/physics/_organized_receiver_slow_admission.py)
rebuilds the primitive source, support, gradient norm, release transport and
velocity-defect inequalities. It uses the shared preparation kernel and
outward rational arithmetic. Its report certifies the reduction and
release motion only; it accepts no trajectory or capture verdict.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md) remains the
sole queue. A later numerical response must have its horizon, observation,
source and budget frozen before evaluation. Supplied donor preparation,
support and constitutive law remain premises of every result above.
