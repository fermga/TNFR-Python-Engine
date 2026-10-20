# Collective boundary constraints and the scope of restoration

This owner states a conditional consequence of the supplied two-row
constitutive flow: a positively curved boundary removes exactly the old
zero-curvature directions that its actual phase constraints distinguish.
The full flow, rather than a static stiffness alone, determines relaxation.
The result uses ordinary positive-semidefinite linear algebra and local
dynamical systems arguments. It is not an additional TNFR axiom or a
selection principle for a fundamental constitutive law.

The [form foundation](../FUNDAMENTAL_THEORY.md),
[parameter foundation](../NODAL_PARAMETER_FOUNDATIONS.md) and
[NFR participation contract](../EMERGENT_ONTOLOGY.md#nfr-participation-in-form)
retain their separate responsibilities. The conditional critical parent in
[the interaction owner](CRITICAL_PARENT_INTERACTION.md#critical-parent-interaction-result)
provides one concrete control below; its preparation and physical
identification are not premises supplied by this theorem.

| Required premise | Consequence admitted here | Information still needed |
| --- | --- | --- |
| Complete held two-row law, connected support and clock | Exact storage loss and full linear pencil | Selection of that law, support or clock |
| Actual equilibrium and its old positive-semidefinite Hessian | Old neutral phase subspace | Formation or entry into its local domain |
| Actual positive boundary curvatures | Kernel intersection and restoration of the distinguished directions | Positivity from an independently justified constitutive premise |
| Admitted positive normal curvature | Exact static compliance | Dynamic memory, rates and hidden initialization |
| Positive complete relative Hessian | Local exponential attraction of the full smooth flow | Quantitative basin or finite capture deadline |
| A surviving Hessian kernel | Semisimple linear zero modes and their linear projector | Nonlinear center dynamics and higher-order attraction |

## 1. Actual model and equilibrium

Use an orthonormal chart of the relative phase space after removing the
common phase rotation. Use the supplied nondimensional structural chart
and held unit nodal capacities. Both the relative phase displacement and
signed form have dimension \(n\). The total form is zero and the common
phase origin is fixed. Hold the connected support, its symmetric
positive-definite relative conductance Laplacian \(L\), the positive
constant \(\gamma\), and the clock \(\tau\). The complete unforced model is

\[
 x'=\frac{-Lx-\gamma\nabla U(\theta)}4,\qquad
 \theta'=\frac{\gamma Lx}4,\qquad {}'=\frac{d}{d\tau}.
 \tag{1}
\]

Assume \(U\) is sufficiently smooth on the declared local phase chart. Its
storage and continuous loss are

\[
 H(x,\theta)=\frac12x^\top Lx+U(\theta),\qquad
 H'=-\frac14\|Lx\|^2.
 \tag{2}
\]

The cancellation in (2) follows directly from both rows of (1); it does not
select \(U\). Fix an admitted equilibrium \(x_*=0\),
\(\nabla U(\theta_*)=0\). All subsequent Hessians are evaluated at this
same equilibrium:

\[
 M=\nabla^2U(\theta_*)=M_0+B^\top K B,\qquad
 M_0\succeq0,\qquad K\succ0.
 \tag{3}
\]

For pairwise boundary edges, \(B\) is their oriented phase-difference map
restricted to the relative chart, and
\(K_{ee}=w_e V_e''(\delta_{e,*})\). Thus \(K\) contains the actual
constitutive curvatures at the actual gaps, not merely positive conductances
\(w_e\). The old contribution \(M_0\) must also be re-admitted at
\(\theta_*\); a displaced equilibrium need not retain an old endpoint's
Hessian. More general boundary observations are allowed only when their
complete Hessian really has (3). A nonlinear observation with nonzero
current can add Hessian terms omitted by an incidence-only formula.

The theorem does not assume that a support addition is passive. Adding a
pairwise edge of weight \(w_e\) in both rows changes the storage by
\(w_e[(x_i-x_j)^2/2+V_e(\theta_i-\theta_j)]\), under that edge's declared
potential normalization. That event work is separate from (2).

<a id="collective-boundary-criterion"></a>

## 2. Which collective directions does the boundary restore?

For every relative phase vector \(v\),

\[
 v^\top Mv=v^\top M_0v+\|K^{1/2}Bv\|^2.
 \tag{4}
\]

Both terms are nonnegative. A zero quadratic value of a symmetric
positive-semidefinite matrix is equivalent to membership in its kernel.
Consequently

\[
 \boxed{\ker M=\ker M_0\cap\ker(K^{1/2}B).}
 \tag{5}
\]

Let \(T\) be any full-rank basis of \(\ker M_0\), with \(k_0\) columns.
Under \(K\succ0\), the number of surviving relative phase zero modes is

\[
 k=\dim\ker M=k_0-\operatorname{rank}(BT).
 \tag{6}
\]

The criterion concerns restrictions on the old neutral space. Adding
edges, increasing conductance or preserving a graph symmetry does not
alone determine this restricted rank or the actual curvature signs.

If \(K\succeq0\) includes zero-curvature rows, (5) still holds, but (6)
uses \(\operatorname{rank}(K^{1/2}BT)\). Such rows impose no quadratic
restoring constraint. If \(K\) is signed, neither nonnegativity in (4)
nor the kernel-intersection conclusion follows in general.

## 3. Full-flow modes, initialization and nonlinear scope

Put \(y=\theta-\theta_*\), and order the full linear state as \((y,x)\).
The actual Jacobian and its relative quadratic pencil are

\[
 J=\begin{pmatrix}
 0&\gamma L/4\\
 -\gamma M/4&-L/4
 \end{pmatrix},\qquad
 Q(s)=s^2L^{-1}+\frac s4I+\frac{\gamma^2}{16}M.
 \tag{7}
\]

Eliminating \(x\) gives
\(L^{-1}y''+y'/4+\gamma^2My/16=0\), so no simultaneous
diagonalization of \(L\) and \(M\) has been assumed.
For an eigenpair \(Q(s)v=0\), multiplication by \(v^*\) gives
\(a s^2+b s+c=0\), with \(a>0,b>0,c\ge0\).
Its nonreal roots have negative real part and its nonzero real roots are
negative. All nonzero full-flow eigenvalues therefore lie strictly in
the left half-plane.

For \(k>0\), to count the zeros and admit their initialization, let \(W\) be any
full-rank basis of \(\ker M\), and set \(G_W=W^\top W\). Define

\[
 R=\begin{pmatrix}W\\0\end{pmatrix},\qquad
 C=G_W^{-1}\begin{pmatrix}W^\top&\gamma W^\top\end{pmatrix}.
 \tag{8}
\]

Direct substitution yields \(JR=0\), \(CJ=0\), \(CR=I_k\).
A generalized zero eigenvector satisfying \(Jz=Ra\) would give
\(a=CJz=0\). Hence zero is semisimple. The full \(2n\)-dimensional
flow has exactly \(k\) zero modes and \(2n-k\) strictly stable modes.
Its zero spectral projector and retained linear amplitude are

\[
 P_0=RC=
 \begin{pmatrix}
 WG_W^{-1}W^\top&\gamma WG_W^{-1}W^\top\\
 0&0
 \end{pmatrix},\qquad
 a_0=G_W^{-1}W^\top(y_0+\gamma x_0).
 \tag{9}
\]

The Gram matrix is essential when \(W\) is not orthonormal. In the
linearized system, \(C(y(\tau),x(\tau))\) is constant and the stable
component decays. Formula (9) is a
linear spectral initialization; it is not a nonlinear phase-only
measurement or an exact nonlinear quotient.

If (5) has trivial kernel, then \(M\succ0\) and \(J\) is Hurwitz.
The smooth actual nonlinear flow is locally exponentially attracting.
For example, a positive solution of
\(J^\top P+PJ=-I\) gives a local quadratic Lyapunov function after
bounding the nonlinear remainder. This statement supplies no numerical
radius, uniform contact limit or source-capture certificate.

If a kernel survives, (5)--(9) alone imply no nonlinear attraction.
A separately admitted center manifold can retain nonlinear normal lag,
and its tracking initialization need not equal (9) beyond first order.
The [critical-parent initialization proof](ORGANIZED_CONSTITUTIVE_TRANSITION.md#constitutive-transition-initialization)
is a model-specific example of the additional work.

A sufficient higher-order local admission is a strict local minimum of
the full storage, an isolated stationary point in a small compact
sublevel component, and containment of that component in the identity
chart. Equation (2), \(L\succ0\), and invariance imply that a trajectory
with identically zero loss has \(x=0\) and \(\nabla U=0\).
LaSalle's argument then gives local asymptotic attraction. Neither
strict higher-order storage nor isolation follows from the kernel rank.

<a id="boundary-static-and-dynamic-reduction"></a>

## 4. Static normal compliance and dynamic elimination

### 4.1 Exact quadratic static reduction

Complete the old kernel basis \(T\) by a matrix \(S\) such that
\(\mathcal R=[T\ S]\) is invertible. Orthogonality is unnecessary.
Write \(y=Tq+Sz\), and define

\[
 A=S^\top M_0S\succ0,\qquad D=BT,\qquad E=BS.
 \tag{10}
\]

Because \(M_0T=0\), the exact quadratic phase storage has block matrix

\[
 \mathcal R^\top M\mathcal R=
 \begin{pmatrix}
 D^\top KD&D^\top KE\\
 E^\top KD&A+E^\top KE
 \end{pmatrix}.
 \tag{11}
\]

Minimizing over the admitted normal coordinate gives

\[
 z_*(q)=-(A+E^\top KE)^{-1}E^\top KDq
 \tag{12}
\]

and the exact effective static stiffness

\[
\begin{split}
 M_{\rm stat}
 &=D^\top[K-KE(A+E^\top KE)^{-1}E^\top K]D\\
 &=D^\top(K^{-1}+EA^{-1}E^\top)^{-1}D.
\end{split}
 \tag{13}
\]

The second form follows by the matrix inversion identity; it applies
to strictly positive \(K\). The bracket is positive definite, so

\[
 \ker M_{\rm stat}=\ker D,\qquad
 0\preceq M_{\rm stat}\preceq D^\top KD.
 \tag{14}
\]

The upper value is the result of artificially freezing the normal
coordinates. Equality along a given \(q\) holds precisely when
\(E^\top KDq=0\). Thus normal compliance can reduce the stiffness
without changing which directions are restored under these premises.
For diagonal edge \(K\), zero-curvature rows can be deleted before using
the inverse expression; more generally restrict to its positive
eigenspace. The first Schur expression remains available for \(K\succeq0\).

Normal invertibility also admits a local nonlinear stationary minimizing
graph by the implicit-function theorem. This is static elimination at
fixed retained coordinate. It does not establish that this graph is
invariant under (1).

### 4.2 The actual spectral and initial-state reduction

Transform the full pencil by congruence,
\(\widehat Q(s)=\mathcal R^\top Q(s)\mathcal R\). Its normal block
has positive mass, damping and stiffness, and is invertible for
\(\operatorname{Re}s\ge0\). The retained spectral equation uses

\[
 Q_{\rm dyn}(s)=
 \widehat Q_{TT}(s)
 -\widehat Q_{TS}(s)\widehat Q_{SS}(s)^{-1}
  \widehat Q_{ST}(s),\qquad
 Q_{\rm dyn}(0)=\frac{\gamma^2}{16}M_{\rm stat}.
 \tag{15}
\]

Only its zero-frequency value is fixed by (13).
For the linear initial-value problem, the transformed Laplace
right-hand side is

\[
 \widehat f(s)=
 \mathcal R^\top\left[
 \left(sL^{-1}+\frac14I\right)y_0+\frac{\gamma}{4}x_0
 \right],
 \tag{16}
\]

and elimination retains
\(\widehat f_T-\widehat Q_{TS}\widehat Q_{SS}^{-1}\widehat f_S\).
The normal initialization is not generally removed. The frequency
dependence corresponds to normal dynamics and, in a reduced time-domain
description, memory. Any cancellation or exact Markov closure needs its
own proof.

For a small explicit control, take

\[
 L=\begin{pmatrix}2&1\\1&2\end{pmatrix},\quad
 M=\begin{pmatrix}0&0\\0&1\end{pmatrix},\quad T=e_1,\quad S=e_2.
 \tag{17}
\]

Here \(M_{\rm stat}=0\) and the static cross block vanishes, but
\(\widehat Q_{TS}=-s^2/3\) and
\(\widehat Q_{SS}=2s^2/3+s/4+\gamma^2/16\).
Dynamic coupling and hidden initialization remain. Even a zero static
cross block cannot justify discarding those terms.

The existing [native effective-connection proof](RELATIONAL_EFFECTIVE_CONNECTIONS.md#static-effective-curvature-and-dynamic-recovery-are-complementary)
already distinguishes static curvature from dynamic recovery on its
own admitted acute model. Equations (10)--(17) apply that method to an
old critical kernel with actual boundary constraints; they do not
transfer its coefficients or verdicts between laws.
The shared [derive_coordinate_memory implementation](../../src/tnfr/mathematics/linear_observation.py)
already provides the exact block-memory identity for an admitted linear
generator and declared retained coordinates. It does not authenticate
the graph, equilibrium or nonlinear closure. No additional solver is
required for this theorem.

<a id="boundary-two-port-control"></a>

## 5. One fixed-total-conductance control on the critical parent

### 5.1 Matched model and restricted boundary rank

Use the previously admitted 54-node critical parent and one nine-node
neighbor from the [single-port interaction owner](CRITICAL_PARENT_INTERACTION.md#critical-parent-interaction-state).
Hold its actual \(\gamma,\tau\), a fixed
\(0<\eta\le10^{-12}\), and its own critical constitutive parameter
\(\lambda_c(\eta)\). The parent has hosts \(c=0,\ldots,4\), donor
\(c=5\), and the neighbor is \(c=6\); every child has nine slots.
The aligned equilibrium is unchanged by the following additions because
the new gaps vanish and \(j_{\lambda_c}(0)=0\),
\(j_{\lambda_c}'(0)=1\).

Compare exactly the same total per-slot added conductance
\(\epsilon=\epsilon_1+\epsilon_2>0\):

- A single port places weight \(\epsilon\) on each of the nine
  same-slot host-2-to-neighbor edges.
- A split port places weight \(\epsilon_1>0\) on those edges and
  weight \(\epsilon_2>0\) on each of the nine same-slot
  donor-to-neighbor edges.

These are alternative attachments from the same disconnected
parent-plus-neighbor support with zero external weights, not a
sequential event changing the single attachment into the split one.
The comparison reallocates the host-port weight; it does not add a
second copy of that port at unchanged weight. Both rows use the same weights.
There is no new controller, clock or input.

Let \(t\) be the parent's physical host-minus-donor phase coordinate
and \(d\) the neighbor-minus-parent mean phase. The parent critical
tangent has slot-constant entries at the two ports

\[
 h=w_{2,j}=\frac16+u_{2,t}(0)>\frac17,\qquad
 g=w_{5,j}=-\frac56,\qquad
 a=h-g=1+u_{2,t}(0)>0.
 \tag{18}
\]

The old relative phase kernel is two-dimensional: the parent's
critical direction and the independent relative mean phase.
On this kernel the boundary observations are \(d-ht\) and \(d-gt\).
One port has restricted rank one. Positive weights on both ports
have rank two because \(h\ne g\). The single port retains one
quartic critical direction; the split port removes both old
relative phase zero directions.

The 63-node relative phase space has dimension 62. Its 56 fine phase
directions remain positive: the parent's 48 and the neighbor's eight,
with positive-semidefinite boundary additions. The remaining old
normal directions are also positive. Thus the split model has
\(M_{63}\succ0\), and all 124 full relative flow eigenvalues lie
strictly in the left half-plane for every fixed positive pair
\((\epsilon_1,\epsilon_2)\). Its smooth full flow is locally
exponentially attracting. The single-port model instead has one
zero and 123 stable modes; its quartic attraction needs the
separate proof already given in the interaction owner.

### 5.2 Exact normal compliance at the aligned equilibrium

The distinction survives relaxation of the internal host shape.
Let \(P=I-\mathbf1\mathbf1^\top/5\), \(\ell=Pe_2\),
\(u_0=u(0)\), and let \(D_5\) be the forward rim-difference map.
Write \(b=2\pi/5\) and \(b_c=b(c-2)\).
In the mean-zero host space define the actual own-law Hessian

\[
\begin{split}
 A_H=P\{&
 D_5^\top\operatorname{diag}
 [j_{\lambda_c}'(b+D_5u_0)]D_5\\
 &+\eta\,\operatorname{diag}
 [j_{\lambda_c}'(b_c+u_{0,c})]\}P\big|_{\mathbf1^\perp}
 \succ\frac15I,\qquad
 \chi=\ell^\top A_H^{-1}\ell\in(0,4).
\end{split}
 \tag{19}
\]

These are the admitted parent normal bounds, not a fit to the new
response. The last inequality uses \(\|\ell\|^2=4/5\).
At fixed physical \(t\), let \(y\) denote host normal displacement
from the stationary parent tangent. The exact quadratic phase storage is

\[
 U_2=\frac92\left[
 y^\top A_Hy+
 \epsilon_1(d-ht-\ell^\top y)^2+
 \epsilon_2(d-gt)^2
 \right].
 \tag{20}
\]

The factor nine retains all slots. Fine coordinates minimize at zero:
their gradient vanishes on the carrier and their actual quadratic
form is positive. Thus (20) is the full quadratic static reduction,
not an assumption that the fine state can be omitted dynamically.

Put \(k=\epsilon_1\epsilon_2/(\epsilon_1+\epsilon_2)\).
Minimization first over \(d\) and then over \(y\) gives

\[
 d_*=\frac{\epsilon_1(ht+\ell^\top y)+\epsilon_2gt}{\epsilon},
 \qquad
 y_*=-\frac{ka}{1+k\chi}A_H^{-1}\ell\,t,
 \tag{21}
\]

and

\[
 \boxed{K_{\rm eff}=\frac{9ka^2}{1+k\chi}},\qquad
 \frac{9ka^2}{1+4k}<K_{\rm eff}<9ka^2.
 \tag{22}
\]

The upper numerator alone freezes the internal host normal state.
The denominator is its exact compliance correction. The local
nonlinear static minimizing potential satisfies
\(U_{\rm red}(t)=U_*+K_{\rm eff}t^2/2+O(t^4)\).
Normal invertibility gives this graph and the inherited reflection
makes the potential even. In the single-port limit \(k=0\), the
quadratic stiffness vanishes irrespective of the positive total
conductance. Equation (22) is a static stiffness, not a decay rate.

### 5.3 Coupled dynamic control

The released means provide a direct reason to retain that distinction.
Let \(r\) be neighbor-minus-parent mean form and retain \(d\) as above.
Define

\[
 c=\frac7{24},\quad
 m=\frac{9\gamma^2}{4\|w\|^2},\quad
 S_0=\epsilon_1+\epsilon_2,\quad
 S_1=\epsilon_1h+\epsilon_2g,\quad
 S_2=\epsilon_1h^2+\epsilon_2g^2.
 \tag{23}
\]

Projection of the actual two-row Jacobian onto the three zero modes
at zero added contact gives the leading small-contact matrix on
\((t,r,d)\):

\[
 \begin{pmatrix}
 -mS_2&0&mS_1\\
 c\gamma S_1&-cS_0&-c\gamma S_0\\
 0&c\gamma S_0&0
 \end{pmatrix}.
 \tag{24}
\]

It follows by summing the two marked boundary contributions, with
the same parent critical projection and \(6:1\) mean weights as in
the interaction owner. The two-row cancellation in the critical
left projection removes a direct retained mean-form term from its
first row; it does not remove the other two equations.
Here
\(S_0S_2-S_1^2=\epsilon_1\epsilon_2(h-g)^2>0\).
For a fixed positive split ratio all entries scale with the total
contact, and the three formerly neutral modes must be treated
together. The complete pencil proves stability without inferring
a scalar rate from (22) or assuming a stationary minimizing graph
is invariant. No per-eigenvalue expansion or uniform
\(\eta\)-dependent time approximation is claimed.

Positive split weights can even give \(S_1=0\), while \(S_2>0\).
That cancellation of the leading receiver mean-form response to
the critical coordinate does not eliminate restoration. It does
not assert cancellation of arbitrary raw-state responses.

### 5.4 Identity, event work and the local domain

All conclusions are local to the same marked constituent identities.
The inherited small phase chart also admits the donor port:
its new gap is bounded by
\(1/20+1/24+1/50=67/600<3/25\), where the actual
\(j_{\lambda_c}'\) is greater than \(9/10\).
The old host-port and internal chart bounds remain in force.
The second nine-edge port adds nine independent cycles: eight
donor-neighbor mixed-slot cycles and one same-slot triangle
\(c=2\to6\to5\to2\), using the existing host-2-to-donor spoke.
Their windings are zero at the aligned equilibrium and stay zero
in this cut-free local chart. The previously marked constituent
and parent periods retain their original values.
Each attachment's event admission starts from the common disconnected
parent-plus-neighbor baseline. For the split attachment, the new
contact work is the sum of the two actual edge additions,
including their form-difference terms. Continuous loss cannot
erase the obligation to account for that work.

For each fixed admitted parameter pair, choose a sufficiently small
closed full-state relative ball inside this identity chart.
The positive Hessian makes the post-contact storage excess on its
boundary strictly positive. A primitive pre-contact state inside
the ball whose old excess plus the actual contact work is below
that boundary barrier enters a local invariant sublevel component
and is attracted to the equilibrium. The inside-ball premise
cannot be replaced by a global low-energy statement.
This is a conditional event-and-domain admission, not a claim
that an existing prepared source satisfies it. No source response,
trajectory or finite capture deadline has been evaluated.

<a id="boundary-constitutive-scope"></a>

## 6. Counterexamples identify the missing premises

**A graph and its loss do not select curvature or a relaxation rate.**
For one relative coordinate take \(L=l>0\) and
\(U(q)=a(1-\cos q)\). The origin is stationary and its polynomial is

\[
 \frac{s^2}{l}+\frac s4+\frac{\gamma^2a}{16}=0.
 \tag{25}
\]

Positive, zero and negative \(a\) give, respectively, stable modes,
a zero mode, and a positive mode; distinct positive values give
different rates at the same \(L\). Alternatively, a positive
conductance sine edge has opposite curvature signs at its
stationary gaps zero and \(\pi\). Its graph and continuous loss
identity are the same. The equilibrium and actual law are
therefore indispensable inputs to (3).

**Old semidefiniteness and positive boundary curvature have separate
roles.** With \(M_0=\operatorname{diag}(-1,0)\),
\(B=(0,1)\), \(K=1\), the boundary distinguishes the old kernel
but the total Hessian remains indefinite. With
\(M_0=\operatorname{diag}(1,0)\), \(B=(1,0)\), \(K=-1\),
the total Hessian is zero: signed curvature has created an
additional cancellation kernel. With \(K=0\), any nonzero \(B\)
imposes no quadratic constraint. These controls prevent extending
(6) to unsigned topological rank without its hypotheses.

**A positive Hessian at a nonstationary state is insufficient.**
For \(U(q)=1-\cos(q-a)\) with small nonzero \(a\), the Hessian at
zero is positive but the gradient is nonzero. The point is not
an equilibrium of (1) and has no equilibrium-attraction verdict.

**A surviving kernel does not decide higher-order attraction.**
Consider the smooth periodic nonnegative potentials

\[
 U_4(q)=(1-\cos q)^2=\frac{q^4}{4}+O(q^6),\qquad
 U_6(q)=(1-\cos q)^3=\frac{q^6}{8}+O(q^8).
 \tag{26}
\]

They have the same zero Hessian at zero, while their first restoring
currents are \(q^3\) and \(3q^5/4\). Both have isolated local
storage minima; their higher-order laws are different.
The equally nonnegative periodic potential
\(U_-(q)=4-(1-\cos q)^2\) instead has a quartic local maximum
at zero and the same zero Hessian. Nearby zero-form states have
strictly lower storage than that equilibrium, so (2) prevents
them from converging to it. Nonnegative storage and a
positive-semidefinite tangent thus do not prove attraction.
The additive constant in \(U_-\) changes no current or loss;
this control does not also impose the incompatible normalization
\(U_-(0)=0\). These are fixed-support local constitutive controls;
they do not claim equal support-event work, which depends on the
declared potential normalization.

**Static elimination does not discard dynamical state.**
Equation (17) has zero static cross curvature and nonzero dynamic
coupling. Formula (16) retains the hidden initial state.
The actual [critical-parent state witnesses](CRITICAL_PARENT_INTERACTION.md#critical-parent-interaction-state)
likewise distinguish instantaneous full-state rates from a
slaved center law. Neither matched coarse means nor a static
minimizer is a complete state of the arbitrary nearby flow.

## 7. Reuse and evidence boundary

The reusable conditional result is (5) together with the full-flow
count and projector (7)--(9). Equations (13)--(16) state the distinct
static and dynamic reductions. The concrete split-port control
shows that the same total added conductance can retain or remove
a critical direction, and gives its exact normal compliance under
one already admitted constitutive law.

Static controls in
[test_collective_boundary_constraints.py](../../tests/physics/test_collective_boundary_constraints.py)
exercise independent noncommuting pencils, Gram-normalized zero
projectors, signed and zero curvature controls, hidden memory,
equal-Hessian nonlinear counterexamples, actual 63-node boundary
incidence, and the fixed-total compliance calculation. They are
implementation checks of the stated algebra, not source capture
or evidence for a uniquely selected law.

The result leaves actual boundary curvature, constitutive choice,
equilibrium, event occurrence and physical observation as inputs
requiring their own admission. It does not derive a universal
restoring law from connectivity, dissipation or the nodal identity.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the sole subsequent research queue; the conditional result remains here.
