# Finite directional transit of the two-port C9 pair

<a id="sine-two-port-directional-transit"></a>
<a id="sine-two-port-transit"></a>

## Complete-state question and supplied preparation

The [two-port compatibility theorem](SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-compatibility)
admits a deformed acute equilibrium for winding classes two and one. The
[subsequent handoff obstruction](SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-handoff-obstruction)
shows that undeformed pairs cannot pass a direct full-boundary total-storage
capture test, even after minimizing over their relative origins. Neither
statement decides the direction of their actual motion.

This result proves finite motion toward smaller donor short gaps and larger
receiver short gaps under the same complete positive-loss law. It uses an
auxiliary gradient flow only inside a proved comparison estimate. No
trajectory solver, fitted endpoint, equilibrium reset or pressure change is
part of the certificate. The mathematical result is an F4 conditional theorem;
its rational certificate checks the explicit sufficient bounds below.

Supply two unit C9 cycles with nodes `0,...,8` and `9,...,17`, and unit
contacts `(0,9)` and `(1,10)`. The contact ports have degree three; the other
nodes have degree two. With the actual graph Laplacian \(L\), degree
matrix \(M\), and \(K=M^{-1}\), retain held capacity one, beta one,
\(e=1023/1024\), \(w=1/1024\), no forcing and no events. Set
\[
\gamma=\frac1{1023\pi},\quad \eta=\gamma^2,\quad
\tau=et,\quad \sigma=\eta\tau,\quad
S_i(\theta)=\sum_{j\sim i}\sin(\theta_j-\theta_i).
\]
The actual complete rows in \(\tau\) are
\[
x'=-KLx+\gamma KS(\theta),\qquad
\theta'=\gamma KLx. \tag{1}
\]
All eighteen form and eighteen phase coordinates remain in this model.

The nominal initial form is zero. Declare the continuous phase lifts
\[
\widetilde\theta^0_j=(j-\tfrac12)\frac{4\pi}{9},\qquad
\widetilde\theta^0_{9+j}=(j-\tfrac12)\frac{2\pi}{9},
\qquad 0\le j\le8. \tag{2}
\]
Their full-network weighted mean is \(21\pi/20\). Use the zero-mean
nominal representative
\(\theta^0=\widetilde\theta^0-(21\pi/20)\mathbf1\).
The implementation equivalently starts from turns \(2j/9\) and
\(1/18+j/9\), whose weighted mean is \(229/360\), and subtracts that
single full-network mean. The two constructions differ before centering only
by one common shift and give exactly the same centered phases. Neither
construction adjusts the components' relative origin or resets actual errors.

Both contact midpoints share a phase origin. Their principal gaps are
\(\delta,-\delta\), with \(\delta=\pi/9\); the ring gaps are
\(4\pi/9\) and \(2\pi/9\). This preparation is specified directly
from the two undeformed twists, without the implicit compatible target.
Relative to the earlier `(j-4)` isolated lifts, it supplies receiver-minus-
donor origin \(-7\pi/9\). This is a new preparation premise, not a late
rotation of an already evaluated response.

Admit every initial state of the form
\[
x_i(0)=\epsilon^x_i,\qquad
\theta_i(0)=\theta_i^0+\epsilon^\theta_i,\qquad
|\epsilon^x_i|\le r_x,\quad |\epsilon^\theta_i|\le r_\theta. \tag{3}
\]
The phase radius is in radians and the form radius in the model's signed
form units. They are separate nonnegative primitives. No reflection symmetry,
zero error sum or correlation is required of these thirty-six errors.
Common form and phase means are retained, not reset to nominal values.
The support is supplied from this initial time. Any preceding formation,
support-attachment event or associated work accounting needs its own premises.

Fix the endpoint at
\[
\Sigma=\tfrac14\quad\hbox{in slow time},\qquad
\tau_* =\frac{1023^2\pi^2}{4},\qquad
t_*=261888\pi^2. \tag{4}
\]
These are declared structural clocks, without a laboratory-unit bridge.

## A complete-law comparison using the dissipated storage

<a id="sine-energy-informed-slow-comparison"></a>

The following estimate improves the long-horizon exponential envelope of the
[existing finite slow-phase comparison](SINE_FORM_PHASE_REDUCTION.md#sine-controlled-slow-phase)
when a common acute chart can be certified. It adds that chart premise and
does not replace the earlier globally valid estimate outside it.

Write \(A=KL\), \(f(\theta)=KS(\theta)\), and let \(P_M\)
remove the weighted common mode. Put
\[
z=\gamma P_Mx,\qquad y=\theta+z,
\qquad \|v\|_M^2=v^TMv.
\]
The exact mixed-coordinate equations are
\[
z_\tau=-Az+\eta f(\theta),\qquad y_\sigma=f(\theta). \tag{5}
\]
These are the same-information coordinates of the
[form-phase reduction owner](SINE_FORM_PHASE_REDUCTION.md#sine-form-phase-memory-equivalence).
Form is recovered as \(P_Mx=z/\gamma\); independently wrapping both
mixed phases would discard this information.

Let \(\psi_\sigma=f(\psi)\) be a declared reference. Match its common
phase origin to the actual state's conserved weighted phase mean. Suppose
that \(A\) has weighted quotient gap at least \(\lambda>0\), and
\(f\) is \(\Lambda\)-Lipschitz in this norm. On a common acute period
cell its derivative is \(-K\mathcal H\), where the cosine-weighted
Hessian \(\mathcal H\) is positive semidefinite. Therefore
\[
\langle u-v,f(u)-f(v)\rangle_M\le0 \tag{6}
\]
whenever the segment from \(u\) to \(v\) stays in that cell. This
is a gradient-flow comparison property, not an assertion that the actual
phase alone obeys a gradient law.

Assume initial bounds \(\|z(0)\|_M\le Z_0\) and
\(\|y(0)-\psi(0)\|_M\le v_0\). Let \(U_{\rm floor}\) be a
proved lower bound on phase potential in this cell and choose
\(E_0\ge H(0)-U_{\rm floor}\), where
\[
H=\tfrac12x^TLx+U(\theta),\qquad
U=\sum_{\{i,j\}\in E}[1-\cos(\theta_j-\theta_i)].
\]
The complete-law identity is \(H_\tau=-\|Ax\|_M^2\). Consequently
\[
H_\sigma=-\frac{\|Az\|_M^2}{\eta^2},\qquad
\int_0^\Sigma\|z\|_M^2\,d\sigma
 \le\frac{\eta^2E_0}{\lambda^2}. \tag{7}
\]
The factor is \(\eta^2\): both the slow clock and the retained form
scaling contribute. The inequality applies before any first exit of the
actual phase from the cell. Positivity of the form storage and the phase
floor bound the available storage decrease.

For \(v=y-\psi\), split its derivative as
\(f(y)-f(\psi)+f(\theta)-f(y)\). Equation (6) and Lipschitz continuity
give the upper norm derivative bound
\(d\|v\|_M/d\sigma\le\Lambda\|z\|_M\), including zero norm by
the upper Dini derivative. Cauchy-Schwarz and (7) yield
\[
V:=v_0+\frac{\Lambda\eta\sqrt{\Sigma E_0}}{\lambda},
\qquad \sup_{0\le\sigma\le\Sigma}\|y-\psi\|_M\le V. \tag{8}
\]
On the acute reference trajectory,
\[
\frac d{d\sigma}\|f(\psi)\|_M^2
 =-2f(\psi)^T\mathcal H(\psi)f(\psi)\le0.
\]
Thus any \(F_0\ge\|f(\psi(0))\|_M\) bounds the reference speed.
Variation of constants in the first row of (5), on the zero-weighted-mean
space, now gives the closed bounds
\[
Z:=\frac{Z_0+(\eta/\lambda)(F_0+\Lambda V)}
          {1-\eta\Lambda/\lambda},\qquad
\sup\|z\|_M\le Z,\qquad
\sup\|\theta-\psi\|_M\le Q:=V+Z, \tag{9}
\]
provided \(\eta\Lambda/\lambda<1\). Indeed, on any finite prefix the
form supremum is at most
\(Z_0+(\eta/\lambda)[F_0+\Lambda(V+\sup\|z\|_M)]\).
Solving that inequality proves (9); no derivative of an unknown forcing or
commutation of graph operators is used.

Equations (7)--(9) initially hold up to the first possible chart exit. If
the proved reference edge margins exceed the edge projections of \(Q\),
they preclude that exit for both \(\theta\) and \(y\). The reference
cell is convex, so the segment premise in (6) is retained. This strict
first-exit argument closes the comparison without assuming the conclusion.
The estimate controls a finite horizon; its square-root growth in (8) is
not an all-time capture or convergence theorem.

## Explicit bounds for the fixed two-port preparation

The graph has weighted mass forty and diameter nine. For any real vector,
the weighted variance is at most one quarter of its squared range, while
the squared range is at most nine times \(x^TLx\), by Cauchy-Schwarz on
a path between its extreme nodes. Hence
\[
\|P_Mx\|_M^2\le\frac{40\cdot9}{4}x^TLx,
\qquad \lambda=\frac1{90}. \tag{10}
\]
The weighted variance bound follows from
\((x_i-\min x)(\max x-x_i)\ge0\) and averaging with positive
weights. Equivalently,
\(L-\lambda[M-dd^T/40]\) is positive semidefinite, with
\(d=M\mathbf1\). The normalized Laplacian has norm at most two, and
\(-L\preceq\mathcal H(\theta)\preceq L\); therefore
\(\Lambda=2\) is a global Lipschitz bound.

Initialize the reference at the centered representative of (2). Every
internal sine sum vanishes. Its
four port rates have magnitudes \(\sin\delta/3\), with opposite signs
at the two ports of each component. Consequently
\[
\|f(\psi(0))\|_M^2=\frac43\sin^2\delta
 <\frac{40}{243}<\left(\frac5{12}\right)^2,
\qquad F_0=\frac5{12}. \tag{11}
\]
Before any reference exit, its speed norm does not increase. Each edge
functional has weighted dual norm
\(\sqrt{1/d_i+1/d_j}\le1\). Every edge thus changes by at most
\(F_0\Sigma=5/48\). The smaller initial acute margin is
\(\pi/18>1/6\), so every reference edge retains margin strictly greater
than \(1/6-5/48=1/16\) throughout \([0,\Sigma]\). A first-exit
argument proves that this reference estimate holds on the entire interval.

The two actual ring periods remain two and one on this cell. Jensen's
inequality for \(1-\cos s\) on the acute interval supplies
\[
U_{\rm floor}=9[1-\cos(4\pi/9)]+9[1-\cos(2\pi/9)]. \tag{12}
\]
The contact terms are nonnegative. Initially, their nominal sum is at most
\(\delta^2<10/81\). The global phase-potential Lipschitz estimate and
the twenty-edge form quadratic give
\[
H(0)-U_{\rm floor}
 <\frac{10}{81}+40r_\theta+40r_x^2.
\]
Require the exact rational admission
\[
\frac{10}{81}+40r_\theta+40r_x^2<\frac18. \tag{13}
\]
This retains all form uncertainty, rather than substituting a nominal
zero-form energy. It supplies \(E_0=1/8\) in (7)--(8).

The elementary bounds on pi imply
\[
\frac1{3216}<\gamma<\frac1{3069},\qquad
\eta<\overline\eta:=\frac1{9000000},\qquad
\sqrt{\Sigma E_0}<\frac3{16}.
\]
For each member, center the form and shift the nominal reference by that
member's conserved phase-mean error. Weighted centering is contractive.
Since \(\sqrt{40}<7\), take the following entirely rational quantities:
\[
\begin{aligned}
Z_0&=7r_x/3069,&v_0&=7r_\theta+Z_0,\\
V&=v_0+2\overline\eta(3/16)90,\\
Z&=\frac{Z_0+90\overline\eta(5/12+2V)}
          {1-180\overline\eta},& Q&=V+Z.
\end{aligned} \tag{14}
\]
All substitutions enlarge the earlier upper bounds. Require
\(Q<1/16\). The reference margin from (11) then keeps both actual
phase and mixed coordinate strictly acute throughout the finite interval.
This also preserves the ring periods and the zero period of the four-edge
cycle `(0,9,10,1)`; the three cycles are the integer basis used in the
compatibility owner. The proof retains all complete-state coordinates and
memberwise means. In particular,
\[
\sup\|P_Mx\|_M<3216Z,\qquad
|\overline x_M|\le r_x. \tag{15}
\]
These are bounds on original form, not merely on its scaled surrogate.

## The actual short gaps move by more than the declared amount

Let \(a(\sigma)=\theta_1-\theta_0\) and
\(c(\sigma)=\theta_{10}-\theta_9\) in the retained acute lifts. The
nominal reference has initial derivatives
\[
a_\psi'(0)=-\tfrac23\sin\delta,\qquad
c_\psi'(0)=\tfrac23\sin\delta.
\]
Both short-edge functionals have weighted dual norm
\(\sqrt{2/3}<g:=5/6\). Since
\(\|\psi''\|_M\le\Lambda F_0\) on the interval, Taylor's integral
remainder bounds either signed reference displacement from below by
\[
\tfrac23\sin\delta\,\Sigma
 -\tfrac12g\Lambda F_0\Sigma^2.
\]
Here \(\sin(\pi/9)>\sin(1/3)>1/3-(1/3)^3/6=53/162\).
At \(\Sigma=1/4\), the displayed lower bound therefore exceeds
\(53/972-25/1152\). The actual endpoint gap differs from its
mean-matched reference by at most \(gQ\); each actual initial gap
differs from its nominal value by at most \(2r_\theta\). Define
\[
J=\frac{53}{972}-\frac{25}{1152}-\frac56Q-2r_\theta. \tag{16}
\]
Every admitted complete-state member satisfies
\[
a(0)-a(\Sigma)>J,\qquad c(\Sigma)-c(0)>J. \tag{17}
\]
These are changes relative to each member's own actual initial gaps, not
relative to idealized or reset starting values. They need not be monotone
at every intermediate instant; for nominal zero form the actual phase
velocity initially vanishes. The statement is a strict finite endpoint
change under the full form-phase law.

For the explicit choice
\(r_x=r_\theta=1/65536\), (13) holds and rational evaluation of (14)
gives \(Q<1/8192\). In particular the entire finite acute margin is
positive, and
\[
J>\frac1{32}+\frac{11491}{7962624}>\frac1{32}. \tag{18}
\]
Thus at the declared time the donor short gap has decreased by more than
\(1/32\) radian and the receiver short gap has increased by more than
\(1/32\) radian, for every state in this full error family. Both ring
identities and the interface period persist throughout the interval.

The implemented sufficient certificate requires (13), a strictly positive
bootstrap margin \(1/16-Q\), and a strictly positive directional margin
\(J-1/32\). A nonpositive reported margin leaves the requested certificate
unavailable; it is not an observed reversed motion or an instability claim.
The rational rules are declared explicitly even when a sharper strict
inequality could resolve an equality case.

The shared
[`assess_sine_two_port_transit`](../../src/tnfr/physics/relational_sine_two_port_transit.py)
returns `SineTwoPortTransit` from the mandatory nonnegative primitives
`form_error_radius` and `phase_error_radius`. It rebuilds the actual support,
centered rational source turns, integer cycle periods and normalized spectral
slack matrices before applying the exact rational bounds. Candidate quantities
remain separate from certified actual errors. If the energy or chart premise
fails, actual error and directional fields remain unavailable. A certified
error envelope and the stricter directional threshold retain separate flags.
The SDK projection preserves these distinctions; no supplied endpoint,
cached equilibrium report or prior capture verdict is consumed.

The result supplies the first bounded dynamical step beyond static
compatibility and the failed direct storage handoff. It does not prove entry
into the new equilibrium's basin, eventual convergence, all-time acuteness,
formation of support, passive contact or physical binding. The prescribed
origin, support, law, clock and uncertainty family remain supplied premises.
