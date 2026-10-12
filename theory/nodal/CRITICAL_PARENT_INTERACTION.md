# Interaction at a marked boundary of a critical parent

<a id="critical-parent-interaction-result"></a>

A critical parent can transmit its internal registration to an
additional organized neighbor through an explicitly marked boundary.
The inherited local state has three coordinates at zero contact.
At positive contact two of these modes become stable, while one
critical mode remains. The neighbor changes that mode's relaxation
coefficient through the geometry of the joined state.

This is a conditional, local theorem for supplied support and prepared
organizations. It does not establish the occurrence of contact,
acquisition of the additional neighbor, or entry of any original
source into the local domain.

| Premise | Declaration |
| --- | --- |
| Parent | The 54-node critical organization in the [transition owner](ORGANIZED_CONSTITUTIVE_TRANSITION.md#weak-contact-constitutive-transition-result), at fixed \(0<\eta\le10^{-12}\) and \(\lambda=\lambda_c(\eta)\) |
| Neighbor | One additional C9 with the same internal carrier and constitutive law |
| Boundary | Nine same-slot edges from host child c=2 to the neighbor, all of supplied weight epsilon |
| Dynamics | Both rows use all edge weights; unit capacity, gamma, quarter factors and tau remain unchanged |
| Local state | All 63 forms and phases, zero total form, fixed total phase, and all fine coordinates |
| Small-contact description | A local three-dimensional center manifold at epsilon=0; constants and neighborhoods may depend on fixed eta |
| Positive-contact description | Local critical attraction at each fixed epsilon>0, with its own neighborhood and transverse rates |
| Evidence | Exact fields, projection, stationary elimination, center-manifold coefficients and static controls; no trajectories, scans or source replay |

## 1. Complete joined law and the cost of contact

Keep the parent labels c=0,...,5 and add c=6 for the neighbor.
Each child has j=0,...,8, with primitive node order 9*c+j.
The parent's 144 edges retain their weights: 99 old internal/rim
edges of weight one and 45 spokes of weight eta. The neighbor has
nine internal edges of weight one. Add
\(\{(2,j),(6,j)\}\), j=0,...,8, with weight epsilon.
Write \(L_\epsilon=L_0+\epsilon L_{\rm b}\).

All edges use the same potential and current:
\[
 V_\lambda(a)=1-\cos a+\lambda\frac{(1-\cos9a)^2}{6\,9^4},
 \qquad
 j_\lambda(a)=\sin a+
       \lambda\frac{\sin9a-\tfrac12\sin18a}{2187},
 \qquad \lambda=\lambda_c(\eta).
\tag{1}
\]
For the full phase potential
\(U_\epsilon=\sum_e w_eV_\lambda(\theta_b-\theta_i)\),
the exact equations are
\[
 x'=\frac{-L_\epsilon x-\gamma\nabla U_\epsilon}4,\qquad
 \theta'=\frac\gamma4L_\epsilon x,\qquad
 H_\epsilon=\frac12x^TL_\epsilon x+U_\epsilon,\qquad
 H_\epsilon'=-\frac14\|L_\epsilon x\|^2.
\tag{2}
\]
Here \(\gamma=1/(1023\pi)\) and prime means the unchanged tau
clock. Total form and total lifted phase are conserved. The form row
supplies the pressure under held unit capacity. There is no external
input or later event in these equations.
Work on the zero-total-form leaf and a fixed-total-phase local chart.

If the boundary is inserted without changing the primitive state,
its storage jump is exactly
\[
 \Delta H=\epsilon\sum_{j=0}^8
 \left[\frac12(x_{6,j}-x_{2,j})^2+
        V_\lambda(\theta_{6,j}-\theta_{2,j})\right]\ge0 .
\tag{3}
\]
This is supplied event work. Continuous loss is not its reservoir,
and the event is not declared passive or autonomously selected.
The jump is zero at the exactly aligned equilibrium, which does not
by itself explain why the event occurs.

The local identity chart uses alpha=2*pi/9,
\(\theta_{c,j}=\alpha(j-4)+\Phi_c+q_{c,j}\), with
\(\sum_jq_{c,j}=0\), and the parent's original compatible lifts.
For example, the open neighborhood
\[
 \|h\|_2<6\eta,\qquad \|q\|_\infty<1/100,\qquad
 |t|<1/20,\qquad |d|<1/20
\tag{4}
\]
has the same parent marked class, neighbor winding one, and zero
winding on the eight independent new mixed slot cycles.
Here t and d are defined below; the old host shape is
\(h_c=\Phi_c-\Phi_5-t-b_c\), with
\(b_c=(2\pi/5)(c-2)\) and \(\sum_{c=0}^4h_c=0\).
Every new boundary gap has magnitude
less than
\(1/20+1/120+6\eta+1/50<2/25\).
Thus the boundary remains in a single circular chart with
\(j_\lambda'>9/10\), using
\(\cos a-2/243\le j_\lambda'(a)\).
The old internal and rim bounds remain those of the transition
owner. This is a local identity domain, not a source-entry certificate.

## 2. Exact carrier and collective boundary equations

The seven-child carrier \(q=0\), with forms constant within each
child, is exactly invariant: internal currents cancel and every
remaining edge joins equal slots. Its equations use the weighted
seven-vertex graph in both rows, with the same quarter factors.
This is a twelve-dimensional relative system, not a stationary
approximation. The other 112 relative fine form/phase directions
are retained when testing its local stability.

Let \(X_c\) be the child mean form, and define
\[
\begin{split}
 A&=\frac16\sum_{c=0}^5X_c,& B&=X_6,& r&=B-A,\\
 \Omega&=\frac16\sum_{c=0}^5\Phi_c,&\Psi&=\Phi_6,&d&=\Psi-\Omega,\\
 t&=\frac15\sum_{c=0}^4\Phi_c-\Phi_5,&
 \xi&=X_2-A,&s&=\Phi_2-\Omega .
\end{split}
\tag{5}
\]
The mean coordinates r and d are neighbor minus parent.
On fixed global means, \(A=-r/7,\ B=6r/7\), and the phase means
are \(\Omega=-d/7,\ \Psi=6d/7\), up to the fixed common origin.
The actual marked port has form offset xi and phase offset s.

On the carrier, direct summation of the complete rows gives
\[
\begin{split}
 r'&=-\frac{7\epsilon}{24}
           [r-\xi+\gamma j_\lambda(d-s)],\\
 d'&=\frac{7\gamma\epsilon}{24}(r-\xi),\\
 t'&=\frac{3\gamma\eta}{2}
       \left(\frac15\sum_{c=0}^4X_c-X_5\right)
       -\frac{\gamma\epsilon}{20}(r-\xi).
\end{split}
\tag{6}
\]
For a noncarrier state, the first line uses the actual averaged
boundary current
\[
 \overline j_{\rm b}
 =\frac19\sum_jj_\lambda(d-s+q_{6,j}-q_{2,j})
\tag{7}
\]
in place of \(j_\lambda(d-s)\); the other two mean rows are still
exact as written. Thus (6) is not an unqualified closure in t,r,d.
The zero fine means cancel the linear Taylor term, giving the
own-law error bound
\[
 |\overline j_{\rm b}-j_\lambda(d-s)|
 \le\frac{J_2}{18}\|q_6-q_2\|_2^2,\qquad J_2=\frac{10}{9}.
\tag{7a}
\]

The isolated parent's critical phase curve, at fixed parent mean,
has host origins \(t/6+b_c+u_c(t)\) and donor origin \(-5t/6\).
At t=0 the marked port c=2 has \(u_2(0)=0\) by reflection.
Let w be that curve's tangent and put
\[
 N=\|w\|_2^2=\frac{15}{2}+9\|u_t(0)\|_2^2,\qquad
 h_\eta=w_{2,j}=\frac16+u_{2,t}(0),\qquad
 m_\eta=\frac{9\gamma^2}{4N}.
\tag{8}
\]
The own-law bound \(\|u_t(0)\|_2<6\eta\) gives
\(h_\eta>1/6-6\eta>1/7\).
Extend w by zero on the neighbor. The critical spectral coordinate
\[
 \zeta=\frac{w^T(\theta_P-\theta_{P,*}+\gamma x_P)}N
\tag{9}
\]
ignores the parent common means because w has zero sum.
Projecting the actual joined rows gives
\[
 \zeta'=-\frac{\gamma^2}{4N}w^T\nabla U_P(\theta_P)
             +\epsilon m_\eta h_\eta\,\overline j_{\rm b}.
\tag{10}
\]
Every boundary form-diffusion term cancels in this row. The
cancellation requires the same boundary weights in both complete
rows. Physical t instead has the explicit form term in (6);
it is not identical to zeta on arbitrary states.

## 3. Three local coordinates at zero contact

At epsilon=0, fix the parent's critical state and the aligned
neighbor carrier, with all forms zero. The full relative state has
124 dimensions. The parent contributes one center direction and
105 stable directions. The neighbor's eight relative phases and
eight relative forms are stable, since its phase Hessian is
\(\cos\alpha L_9>0\) on its relative space. The two independent
component-mean differences r,d have zero rows.
There are therefore three semisimple zero eigenvalues and 121
strictly stable eigenvalues. On the exact carrier the count is
three center and nine stable directions.

For each fixed eta, the finite-smoothness center construction used
in the [local parent dynamics](ORGANIZED_CONSTITUTIVE_TRANSITION.md#constitutive-transition-local-dynamics)
applies after adjoining \(\epsilon'=0\).
Choose a finite \(C^k\) manifold, k at least five, equivariant under
reflection and with the product of the parent center curve and the
two mean coordinates at epsilon=0. The local spectral gap exists
at fixed eta; no lower bound uniform in eta is asserted.
The analytic extension to negative epsilon needed for the parameter
theorem is only a proof device, not an admitted negative-weight model.

Use physical t,r,d as center coordinates and write z=(t,r,d).
All three reverse under the reflection fixing c=2, while epsilon
is unchanged. The aligned state is an exact equilibrium for every
epsilon. The local rows are therefore odd in z.
At zero contact the actual center port has
\[
 s=h_\eta t+O_\eta(t^3),\qquad
 \xi=O_\eta(t^3),\qquad
 \zeta=t+O_\eta(t^3).
\tag{11}
\]
The equalities concern the invariant center curve. The stationary
shape supplies the same tangent, but not an invariant trajectory.
At nonzero epsilon the graph corrections in (11) are
\(O_\eta(\epsilon\|z\|)\).

Let \(b_\eta>0\) be the parent's critical cubic coefficient,
already derived in its unchanged tau clock:
\[
 b_\eta=m_\eta A_5\eta^5
           B_s(0,\mu_c,\eta),\qquad
 A_5=5\varphi^{10},\qquad 55<B_s(0,\mu_c,\eta)<70 .
\tag{12}
\]
The notation A_5 here is the torque coefficient, not the mean A
in (5). Put c=7/24. Projection of the full field gives
\[
\begin{split}
 t'&=-b_\eta t^3+
       \epsilon m_\eta h_\eta(d-h_\eta t)+R_t,\\
 r'&=-c\epsilon[r+\gamma(d-h_\eta t)]+R_r,\\
 d'&=c\gamma\epsilon r+R_d,\\
 R_t&=O_\eta\!\left(|t|^5+
          \epsilon\|z\|^3+\epsilon^2\|z\|\right),\\
 R_r,R_d&=O_\eta\!\left(
          \epsilon\|z\|^3+\epsilon^2\|z\|\right).
\end{split}
\tag{13}
\]

For the first row, the parent linear field vanishes under its left
center vector. Consequently a first-order correction to the stable
graph cannot add a term linear in epsilon*z to (10). The direct
boundary term is the one displayed in (13), since \(j_\lambda'(0)=1\).
Changing from zeta to physical t changes the derivative only by
\(O_\eta(\epsilon^2\|z\|+\epsilon\|z\|^3+|t|^5)\), using (11).
This explains why the explicit raw form term in (6) is absent from
the leading center t row; it has not been dropped from the full law.
The other rows follow from (6), and reflection excludes quadratic
state remainders.

The leading linear contact matrix, in the order (t,r,d), is
\[
 \epsilon
 \begin{pmatrix}
  -m_\eta h_\eta^2&0&m_\eta h_\eta\\
  c\gamma h_\eta&-c&-c\gamma\\
  0&c\gamma&0
 \end{pmatrix}.
\tag{14}
\]
It has one zero eigenvalue and two with negative real part:
the remaining polynomial is
\[
 s^2+(c+m_\eta h_\eta^2)s+
            c m_\eta h_\eta^2+c^2\gamma^2 .
\tag{15}
\]
The two additional decay rates are of order epsilon at fixed eta.
Eliminating these mean coordinates at epsilon=0 would erase genuine
neutral state and cannot give a uniform instantaneous scalar law.

At epsilon=0 the leading full-state center initialization is
\((\Pi_\eta(\theta_P,x_P),r,d)\), where
\(\Pi_\eta=\zeta+O_\eta(\|(\delta\theta_P,x_P)\|^2)\).
Here the parent map consumes its mean-subtracted phase and form
coordinates; the linear w projection already performs that removal.
For small contact, the three-dimensional slow spectral subspace is
separated from the other 121 directions and changes by O_eta(epsilon).
Its linear initialization, expressed in physical coordinates on that
subspace, is \(P_{3,\epsilon}\delta Z\), with
\(P_{3,\epsilon}=P_{3,0}+O_\eta(\epsilon)\) and
\(P_{3,0}\delta Z=(\zeta,r,d)\). The corresponding nonlinear
tracking initialization retains the discarded-state correction in
all three coordinates. In particular, availability of r and d as
physical coordinates on the invariant graph does not make its
tracking initial means equal to the raw means of an arbitrary
off-manifold state.
No uniform nonlinear initialization error, neighborhood or tracking
rate as eta or epsilon tends to zero is asserted.

<a id="critical-parent-loading"></a>

## 4. A fixed positive contact changes the critical relaxation

There is a stronger exact local conclusion at every fixed
epsilon>0. At the aligned state the new gaps are zero and
\(j_\lambda'(0)=1\). Its phase Hessian is
\[
 M_{63}=\operatorname{diag}(M_{P,*},\cos\alpha L_9)
                   +\epsilon L_{\rm b}.
\tag{16}
\]
All three terms are positive semidefinite. Before adding the
boundary, their kernel consists of the parent critical tangent and
the two component rotations. On the fixed-global-phase space,
the boundary eliminates one linear combination. Its remaining
critical vector is
\[
\begin{split}
 \widetilde w_P&=w-\frac{h_\eta}{7}\mathbf1_P,&
 \widetilde w_N&=\frac{6h_\eta}{7}\mathbf1_N,\\
 \widetilde N&=\|\widetilde w\|^2
                 =N+\frac{54}{7}h_\eta^2.
\end{split}
\tag{17}
\]
Its two endpoints agree on every new edge. Conversely, equality on
those edges imposes d=h_eta*t on the two old relative kernel
directions, leaving only (17). Thus the relative phase Hessian has
61 positive directions and one zero.
The complete 124-dimensional pencil
\[
 s^2L_\epsilon^{-1}+\frac s4I+\frac{\gamma^2}{16}M_{63}
\tag{18}
\]
has one simple zero eigenvalue and 123 eigenvalues with negative
real part, by the scalar quadratic and kernel Schur-complement
arguments applied to this actual joined Hessian.

The stationary normal elimination can also be identified exactly.
In the local chart, the 56 fine phase directions remain positive.
The old 48 directions and the neighbor's eight directions already
have positive internal curvature; new edges add nonnegative
curvature because their gaps satisfy (4). At the carrier their
fine gradients vanish. Integrating the fine Hessian therefore
forces q=0 at any nearby normal stationary state.

At fixed physical t, neighbor stationarity then requires
\(j_\lambda(d-s)=0\). The strict local monotonicity of this current
gives d=s. All boundary currents vanish, and the parent's projected
host equations uniquely recover its unchanged u(t).
Let \(s_*(t)=t/6+u_2(t)\). With global phase fixed, the joined
stationary curve is exactly
\[
 \widetilde\Theta_P(t)=\Theta_P(t)-\frac{s_*(t)}7\mathbf1_P,
 \qquad
 \widetilde\Theta_N(t)=
       \alpha(j-4)+\frac{6s_*(t)}7+\text{fixed common origin}.
\tag{19}
\]
Its boundary energy is zero. Its reduced phase potential is the
same \(9\mathcal W(t)\) as the parent, plus the constant internal
neighbor potential. The positive normal Hessian ensures local
uniqueness of this normal elimination at fixed t.

Apply the invariant-center calculation to this joined system.
The normalized left vector is
\((\widetilde w,\gamma\widetilde w)/\widetilde N\).
As in the parent derivation, the actual invariant curve agrees
with the stationary curve through degree two; its cubic normal
correction is annihilated by
\(\widetilde w^TM_{63}=0\). The resulting cubic coefficient is
\[
 t'=-b_{\eta,\epsilon}^{\rm joined}t^3+O_{\eta,\epsilon}(t^5),
 \qquad
 b_{\eta,\epsilon}^{\rm joined}
   =b_\eta\frac{N}{N+(54/7)h_\eta^2}>0 .
\tag{20}
\]
For each fixed positive epsilon this coefficient has the stated
exact value, independent of epsilon. Neighborhoods and transverse
decay rates do depend on epsilon. Since u_t=O(eta),
\[
 \frac{b_{\eta,\epsilon}^{\rm joined}}{b_\eta}
            \longrightarrow\frac{35}{36}\quad\hbox{as }\eta\to0
\tag{21}
\]
along the proved critical family, without a uniform dynamical-limit
claim. The effect comes from the larger tangent norm in (17);
the stationary quartic potential is unchanged.
This is an inherited change in relaxation, not a physical mass
identification or a change to primitive capacity.

## 5. Joined initialization, identity and asymptotic decay

The positive normal Hessian and the unchanged strictly positive
quartic reduced potential make the aligned joined state a strict
local minimum of H_epsilon on the fixed-mean leaf. At each fixed
positive eta and epsilon, a small compact sublevel is invariant and
lies inside the identity chart. Storage loss and connected support
force x=0 on its zero-loss invariant set; local stationary
classification leaves only the critical equilibrium.
Thus the full joined state is Lyapunov stable and asymptotically
attracting, including all fine perturbations admitted locally.

The contact work can be included in an explicit sufficient admission
condition. Choose a small closed full relative ball \(\mathcal B\)
inside the identity chart and this strict-minimum neighborhood, and set
\[
 \mathcal E_{\eta,\epsilon}
  =\min_{\partial\mathcal B}
        [H_\epsilon(Z)-H_\epsilon(Z_*)]>0 .
\tag{21a}
\]
If the primitive state immediately before the event is already
inside this ball and
\[
 H_0(Z_0)-H_0(Z_*)+\Delta H(Z_0)
                    <\mathcal E_{\eta,\epsilon},
\tag{21b}
\]
the joined trajectory cannot leave the ball.
The reference has zero boundary work, so
\(H_\epsilon(Z_*)=H_0(Z_*)\), and (21b) is exactly the
post-event sublevel condition with the paid jump (3).
This is a local, sufficient admission; neither a numerical ball
radius nor a global low-energy capture criterion is supplied.

Finite-smoothness center reduction and stable tracking apply with
123 stable directions. The same graph-slope and quadratic-defect
argument as in the
[parent initialization proof](ORGANIZED_CONSTITUTIVE_TRANSITION.md#constitutive-transition-initialization)
gives
\[
\begin{split}
 \Pi_{\eta,\epsilon}^{\rm joined}(\delta\theta,x)
 &=\frac{\widetilde w^T(\delta\theta+\gamma x)}
                  {\widetilde N}
              +O_{\eta,\epsilon}(\|(\delta\theta,x)\|^2)\\
 &=\frac{N\zeta+(54h_\eta/7)(d+\gamma r)}
              {N+(54/7)h_\eta^2}
              +O_{\eta,\epsilon}(\|(\delta\theta,x)\|^2).
\end{split}
\tag{22}
\]
The linear identity uses the physical component means, so it
retains the neighbor's form and phase rather than replacing them
with their stationary values.

The zero set of this nonlinear initialization map is the local
123-dimensional strong-stable manifold. Outside it, the physical
parent registration has the asymptotic behavior
\[
 t(\tau)\sim
 \frac{\operatorname{sgn}\Pi_{\eta,\epsilon}^{\rm joined}}
            {\sqrt{2b_{\eta,\epsilon}^{\rm joined}\tau}} .
\tag{23}
\]
The exceptional set approaches exponentially. These conclusions
hold at fixed positive contact and on the corresponding local
invariant neighborhood. They do not give a finite prepared-source
capture time or uniform approximation error.

At epsilon=0, r and d are additional neutral coordinates.
At positive epsilon their decay rates tend to zero with epsilon,
while the eventual critical curve requires d=s_*(t).
Consequently the positive-contact asymptotic coefficient (20)
cannot be substituted into the zero-contact transient problem.
The small-contact and long-time limits have separate obligations.

<a id="critical-parent-interaction-state"></a>

## 6. What the marked boundary observes, and what means omit

A direct field calculation makes the inherited interaction
discriminating. Put all forms to zero, keep the fine carrier, choose
the parent's stationary shape u(t), and set d=0. The parent and
neighbor means then agree, but their marked port gap is
\(-s_*(t)\). The receiver's exact initial form rate is
\[
 X_6'=\frac{\epsilon\gamma}{4}j_\lambda(s_*(t))
      =\frac{\epsilon\gamma h_\eta}{4}\,t
                         +O_\eta(\epsilon t^3).
\tag{24}
\]
Reflection gives opposite rates for sufficiently small nonzero
t and -t, despite identical parent means and identical neighbor
state. The relevant internal registration is therefore consumed
by this actual boundary.
These two preparations also have exactly equal old storage and
equal contact-event work: the same reflection makes the parent
reduced potential even and \(s_*(t)\) odd, while \(V_\lambda\)
is even. Means and storage therefore do not remove this marked-port
distinction.
For this same preparation the raw physical t' is initially zero:
every form is zero in its exact phase row. The preparation lies
on a stationary shape graph, not on the invariant center manifold,
so the center t row in (13) must not be used as its instantaneous
rate.

Even retaining t,r,d does not close the exact field for arbitrary
nearby states. Keep every phase at the aligned equilibrium.
Compare zero forms with a perturbation adding a form delta to
each node in child 2 and subtracting delta from each node in
another host child. All other forms stay zero. The complete
means and t,r,d remain zero, and the phase identity is identical,
but xi=delta in the perturbed state. Its mean rates differ by
\[
 \Delta t'=\frac{\epsilon\gamma}{20}\,\delta,\qquad
 \Delta r'=\frac{7\epsilon}{24}\,\delta,\qquad
 \Delta d'=-\frac{7\gamma\epsilon}{24}\,\delta .
\tag{25}
\]
The old host-versus-donor form contrast is zero, so no omitted
parent term is needed for these differences.

Equations (24)--(25) are exact local preparation-and-field
witnesses, not integrated responses. They establish both the
usefulness of the internal mode and the limits of a mean-state
description. The center manifold supplies a justified local
description with initialized stable transients, not an
instantaneous quotient of every admissible full state.

## 7. Scope and static controls

The boundary, its weight and its occurrence remain supplied.
The local identity and relaxation theorem is conditional on entering
the declared neighborhood. No original acquisition, observation
word or frozen response was rerun; no source certificate was
transferred from another law or support. The additional C9 is
assumed organized here. Neither the supplied contact nor the
change of critical relaxation is a newly identified physical force.

The result adds a complete-state interaction consequence to the
stationary and unforced parent results: the parent exports its
internal registration through a marked port, while a neighbor
changes the inherited relaxation and initialization.
The retained state and boundary have been specified before any
numerical or physical evaluation.

The [boundary-constraint criterion](COLLECTIVE_BOUNDARY_CONSTRAINTS.md)
identifies which parts of this conclusion use positive curvature and
complete kinetics, and which retain the particular constitutive law.
Its fixed-total-conductance split-port control removes the surviving
neutral mode; it does not transfer this single-port cubic coefficient.

[Static interaction tests](../../tests/physics/test_critical_parent_interaction.py)
check the 63-node edge sums, mean orientations, projected
three-coordinate matrix, full kernel and tangent norm, added
storage, and the independent preparation witnesses. They do not
solve a trajectory or search for a root. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns subsequent work.
