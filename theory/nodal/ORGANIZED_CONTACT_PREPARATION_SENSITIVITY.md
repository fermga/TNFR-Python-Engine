# Internal preparation modes and organized contact response

<a id="organized-contact-preparation-sensitivity"></a>

This owner answers the full-fine preparation question declared at `b2801170`
in the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#organized-contact-preparation-sensitivity).
Every internal disturbance with zero mean in each child is invisible to the
donor mean at first variation along the entire evolving ideal carrier.
Opposite internal modes can nevertheless generate a second-order mean
response. Exact reflection protection is a separate, stronger condition.

The result concerns conditional perturbations of the exact limiting family
in the [dynamic cancellation theorem](ORGANIZED_CONTACT_DYNAMIC_CANCELLATION.md).
It does not turn those perturbations into acquired residuals or supply a
finite noisy response, transverse stability theorem or physical readout.

The [constituent-memory theorem](ORGANIZED_CONSTITUENT_MEMORY.md) already
establishes tangent decoupling, quadratic covariance feedback and closure
collisions for the sine \(C_9\mathbin\square C_5\) model. The present
contribution is the full \(C_9\mathbin\square W_6\) reciprocal contact
under both retained laws: all-time internal-mode invisibility of its donor
observable, the distinction between a linear kernel and exact reflection
protection, and explicit phase/form disturbances that break the ideal
onset behavior. Its coefficients and conclusions are derived on this
support rather than transferred from that predecessor.

## 1. Full model and preparation directions

Retain exactly the two endpoint currents

\[
 j_\lambda(a)=\sin a+\frac\lambda{2187}
       (\sin9a-\tfrac12\sin18a),\qquad \lambda=0,1,
 \tag{1}
\]

and their potential \(V_\lambda\) from the prior owner. On all 54 nodes
and 144 unit edges of \(C_9\mathbin\square W_6\), both complete rows are

\[
 x'=\tfrac14(-Lx+\gamma S_\lambda(\theta)),\qquad
 \theta'=\tfrac\gamma4Lx,\qquad
 S_{\lambda,a}(\theta)=\sum_{b\sim a}j_\lambda(\theta_b-\theta_a),
 \quad \gamma=\frac1{1023\pi}.
 \tag{2}
\]

Prime means \(\tau=(1023/1024)t\). Forms are signed, capacities are held
at one, and phases occupy the full circle with declared lifts. There is
no later input or event. The same quarter factor remains in both rows;
the donor and host both react to contact.

Write \(\theta^0(\beta)\) for the ideal initial phases
\(\psi_j+\phi_c\) on host child \(c=0,\ldots,4\), and
\(\psi_j+\beta\) on the donor, where
\(\psi_j=2\pi(j-4)/9\), \(\phi_c=2\pi(c-2)/5\). The perturbed
initial state is

\[
 x(0;\epsilon)=\epsilon u,\qquad
 \theta(0;\epsilon)=\theta^0(\beta)+\epsilon v,
 \qquad \mathbf1^Tu=\mathbf1^Tv=0.
 \tag{3}
\]

Both directions have all 54 coordinates. Their constraints fix total
form and the phase gauge, so the complete histories have total form zero
and lifted phase total \(9\beta\). The amplitude \(\epsilon\) scales
specified form and phase directions in their respective structural units;
no physical identification of these units is supplied.

Let \(\bar y_c=\frac19\sum_j y_{c,j}\). Each row splits uniquely
into its six child means and eight zero-mean modes per child. Thus the
constrained preparation space has ten collective and 96 internal real
directions. The donor observation is the same linear functional
\(d^Tx=\bar x_D=R_D\); it does not consume hidden state directly.

## 2. Variations along the evolving full carrier

Denote the unperturbed full solution by \((x_0,\theta_0)\). It remains
on the twelve-coordinate carrier, but its forms and origins evolve under
both rows of (2). Define the amplitude expansion on every fixed finite
time interval by

\[
 x=x_0+\epsilon x_1+\epsilon^2x_2+o(\epsilon^2),\qquad
 \theta=\theta_0+\epsilon y_1+\epsilon^2y_2+o(\epsilon^2).
 \tag{4}
\]

The second coefficients are one half of the second amplitude derivatives.
They are not additional preparations or frozen source coordinates. Smooth
dependence follows from the smooth complete field. Put
\(J(\tau)=DS_\lambda(\theta_0(\tau))\) and
\(Q(\tau)=D^2S_\lambda(\theta_0(\tau))\). Explicitly,

\[
 \begin{split}
 (Jz)_a&=\sum_{b\sim a}j_\lambda'(\theta_{0,b}-\theta_{0,a})(z_b-z_a),\\
 Q[z,w]_a&=\sum_{b\sim a}j_\lambda''(\theta_{0,b}-\theta_{0,a})
                      (z_b-z_a)(w_b-w_a).
 \end{split}\tag{5}
\]

With \(A=-L/4\), \(b=\gamma/4\) and \(C=\gamma L/4\), the
complete variational equations are

\[
 \begin{array}{ll}
 x_1'=Ax_1+bJy_1,& y_1'=Cx_1,\\
 x_2'=Ax_2+bJy_2+\tfrac b2Q[y_1,y_1],&y_2'=Cx_2,
 \end{array}
 \qquad (x_1,y_1)(0)=(u,v),\quad(x_2,y_2)(0)=(0,0).
 \tag{6}
\]

All four variation rows have zero global mean. In particular, the
second-order forcing retains equal opposite edge reactions; it creates
no net form. The time dependence of \(J,Q\) is retained throughout.

Let \(T\) be the simultaneous cyclic slot shift in all six children.
The combined map \((x,\theta)\mapsto(Tx,T\theta-\alpha\mathbf1)\),
\(\alpha=2\pi/9\), fixes every carrier point circularly. It is a
symmetry of the complete field because slot shift is a support permutation
and common phase translation is a symmetry. Its tangent is \(T\) on
both variation rows. Consequently \(J(\tau)T=TJ(\tau)\), for every
time along the evolving carrier.

In complex slot mode \(k\in\mathbb Z/9\mathbb Z\), write
\(\mu_k=2-2\cos(2\pi k/9)\). The first variation retains the full
six-form/six-phase system in that mode, with

\[
 L_k=K+\mu_k I_6,\qquad
 J_k(\tau)=J_{\rm wheel}(\Phi(\tau))
                   -j_\lambda'(\alpha)\mu_kI_6.
 \tag{7}
\]

Here \(K\) is the wheel Laplacian and \(\Phi\) comprises all six
evolving origins. No mode is fed by another at first order. Real directions
retain conjugate mode pairs. If \(\bar u_c=\bar v_c=0\) in every
child, only \(k\ne0\) is present, and uniqueness in the zero mode gives

\[
 \bar x_{1,c}(\tau)=\bar y_{1,c}(\tau)=0
 \quad\text{for every child and every }\tau\ge0,
 \qquad \partial_\epsilon R_D(\tau;0)=0.
 \tag{8}
\]

This is an all-time first-variation statement. It does not assert a uniform
amplitude remainder over an infinite horizon or stability of internal modes.
Collective directions instead obey the complete wheel tangent system and
can affect the donor mean at first order. Equation (8) supplies a 96-real-
dimensional invisible subspace; it does not classify every possible
additional kernel of the full output history.

## 3. Quadratic selection and exact reflection protection

Slot equivariance gives
\(TQ[z,w]=Q[Tz,Tw]\). Thus modes \(k,l\) force only mode
\(k+l\pmod9\) at second order. A mean response receives the collective
pair \((0,0)\) or internal pairs \((k,-k)\); a collective/internal
pair cannot reach mode zero. In the internal case the mean second-order
system is

\[
 \begin{split}
 \bar x_2'&=-\tfrac14K\bar x_2+
       \tfrac\gamma4J_{\rm wheel}(\Phi)\bar y_2+
       \tfrac\gamma8\overline{Q[y_1,y_1]},\\
 \bar y_2'&=\tfrac\gamma4K\bar x_2,\\
 \overline{Q[y_1,y_1]}_c
   &=\sum_{a\sim_{W_6}c}j_\lambda''(\Phi_a-\Phi_c)
                      \overline{(y_{1,a}-y_{1,c})^2}.
 \end{split}\tag{9}
\]

Internal-cycle edge reactions cancel exactly in the last child average.
They still affect \(y_1\) through (7). With the Fourier convention
\(y_j=\sum_k\widehat y_ke^{2\pi ikj/9}\), the squared gap average
is \(\sum_k\widehat{\Delta y}_k\widehat{\Delta y}_{-k}\).
For real directions it is a sum of squared magnitudes, but its current
coefficient \(j_\lambda''\) can have either sign. Opposite-mode pairing
allows mean forcing; it does not guarantee a nonzero response for every
internal direction.

Exact zero response has a stronger sufficient condition. At
\(\beta=m\pi/5\), let the full-support reflection be
\(P(c,j)=(4+m-c\bmod5,8-j)\), \(P(D,j)=(D,8-j)\). The
[complete-field conjugation](ORGANIZED_CONTACT_DYNAMIC_CANCELLATION.md#2-exact-carrier-and-complete-field-symmetries)
fixes (3) for every amplitude whenever

\[
 Pu=-u,\qquad Pv=-v.
 \tag{10}
\]

Uniqueness then gives \(R_D(\tau;\epsilon)=0\) for all future time,
not merely zero first or second variation. The reflection condition is a
constraint on both rows and the complete fine state. It need not follow
from child means, winding labels or the zero-mode kernel (8). Directions
that satisfy neither this protection nor a nonzero coefficient below
remain unclassified beyond the bounded amplitude and time orders; vanishing
coefficients are not promoted to an invariance theorem.

## 4. Complete contact-time jets through order three

The following formulas retain all directions in (3), including mixtures of
form, phase and collective perturbations. In this section evaluate
\(J,Q\) at \(\theta^0\), let \(T_3=D^3S_\lambda(\theta^0)\),
and put \(a=CbS_\lambda(\theta^0)=\theta_0''(0)\).
Although \(\theta_0'(0)=0\), this acceleration is generally nonzero.

The linear coefficients of the form time derivatives are

\[
 \begin{split}
 x_1(0)&=u,\qquad z_1:=x_1'(0)=Au+bJv,\\
 z_2:=x_1''(0)&=Az_1+bJCu,\\
 x_1'''(0)&=Az_2+bJCz_1+bQ[a,v].
 \end{split}\tag{11}
\]

The quadratic amplitude coefficients are

\[
 \begin{split}
 x_2(0)&=0,\qquad w_1:=x_2'(0)=\tfrac b2Q[v,v],\\
 w_2:=x_2''(0)&=Aw_1+bQ[v,Cu],\\
 x_2'''(0)&=Aw_2+bJCw_1+bQ[Cu,Cu]+bQ[v,Cz_1]
                       +\tfrac b2T_3[a,v,v].
 \end{split}\tag{12}
\]

Apply \(d^T\) only after forming these full vectors. The terms containing
\(a\), the diffusion terms and both reactions prevent a frozen-host or
prescribed-donor interpretation. The factor one half in (12) follows the
coefficient convention (4), not a change in the complete law.

For pure phase disturbances the first quadratic donor derivative is

\[
 d^Tx_2'(0)=-\frac\gamma8\sum_{c=0}^4
    j_\lambda''(\beta-\phi_c)\,
       \overline{(v_D-v_c)^2}.
 \tag{13}
\]

For pure form disturbances \(v=0\), both quadratic time derivatives
of orders one and two vanish. Phase feedback yields at order three

\[
 d^Tx_2'''(0)=-\frac{\gamma^3}{64}\sum_{c=0}^4
   j_\lambda''(\beta-\phi_c)\,
       \overline{((Lu)_D-(Lu)_c)^2}.
 \tag{14}
\]

This term would be lost by suppressing the phase row. These are contact-time
derivatives; the corresponding Taylor terms in time carry the usual
factorials, independently of the amplitude convention.

## 5. Winding-admissible internal leakage witnesses

Set \(\beta=0\), where the unperturbed response is identically zero for
both laws. Choose the rational zero-mean slot vector
\(w=(1,-1,0,0,0,0,0,0,0)\), supported in host child \(c=1\) only.
Use it first as \(v\), with \(u=0\), then as \(u\), with \(v=0\).
Both directions belong to the all-time first-order kernel (8); neither
satisfies the full reflection condition (10).

Write \(s_1=\sin(2\pi/5)>0\), \(s_2=\sin(4\pi/5)>0\), and
\(q_1=j_\lambda''(2\pi/5)\), \(q_0=j_\lambda''(4\pi/5)\).
The actual endpoint laws give

\[
 q_1=-(1-\lambda/27)s_1-(2\lambda/27)s_2<0,\qquad
 q_0=-(1-\lambda/27)s_2+(2\lambda/27)s_1.
 \tag{15}
\]

For the phase witness, \(\overline{w^2}=2/9\), so (13) gives

\[
 d^Tx_2'(0)=-\gamma q_1/36>0.
 \tag{16}
\]

For the form witness, the spoke gap in \(Lu\) is
\(-(L_{C_9}+4I)w\) at child 1, zero at its two rim neighbors,
and \(-w\) at the two remaining host children. Direct integer arithmetic
gives \(\|w\|^2=2\) and \(\|(L_{C_9}+4I)w\|^2=100\).
The full set of five contact gaps in (14), rather than the disturbed child
alone, therefore gives

\[
 \begin{split}
 d^Tx_2'''(0)&=\frac{\gamma^3}{288}(q_0-49q_1)\\
 &=\begin{cases}
   \gamma^3(49s_1-s_2)/288,&\lambda=0,\\
   \gamma^3(1276s_1+72s_2)/7776,&\lambda=1.
 \end{cases}
 \end{split}\tag{17}
\]

Both are strictly positive: \(s_2<s_1\), and the alternative expression
has two positive terms. This is the first nonzero quadratic time derivative
for the form witness. The linear donor variation is zero at all times in
both examples, not just through this jet budget.

An explicit common initial winding allowance is
\(0<|\epsilon|\le1/10\). In the phase witness internal edge deviations
are at most \(2|\epsilon|\le1/5\), and host outer edge deviations at
most \(|\epsilon|\le1/10\). Since
\(\alpha-1/5>0\), \(\alpha+1/5<\pi/2\), and
\(2\pi/5\pm1/10\in(0,\pi/2)\), every marked internal and outer
cycle retains its initial winding-one acute chart. Undisturbed cycles are
unchanged. The form witness leaves all initial phases unchanged. No winding
is assigned to the new contact loops, and initial admission proves no
finite-time maintenance interval.

Nonzero quadratic coefficients imply leakage for sufficiently small nonzero
amplitudes inside this admitted range; the range itself is a winding
allowance, not a uniform truncated-series error certificate. For the pure
form witness the third contact-time derivative is exactly quadratic in
\(\epsilon\): differentiating (2) three times introduces at most two
initial form factors when initial phases are fixed. Its constant and linear
donor terms vanish here. Neither statement identifies an acquired source
or requires a trajectory evaluation.

## 6. Work, storage and the scope of robustness

Attachment preserves the prepared coordinates and adds the actual contact
storage. With \(\delta_c=\beta-\phi_c\),
\(\Delta u_{c,j}=u_{D,j}-u_{c,j}\), and similarly \(\Delta v\),

\[
 \begin{split}
 W(\epsilon)&=\sum_{c,j}\left[
   \tfrac12\epsilon^2(\Delta u_{c,j})^2+
   V_\lambda(\delta_c+\epsilon\Delta v_{c,j})\right],\\
 W_0&=45+5\lambda/2916,\\
 W_1&=\sum_{c,j}j_\lambda(\delta_c)\Delta v_{c,j},\\
 W_2&=\tfrac12\sum_{c,j}
       [ (\Delta u_{c,j})^2+
          j_\lambda'(\delta_c)(\Delta v_{c,j})^2].
 \end{split}\tag{18}
\]

Here \(W_r\) are amplitude coefficients. Internal zero-mean phase
directions have \(W_1=0\). The quadratic correction need not be
nonnegative because the contacts need not be acute. The positive baseline
attachment cost remains supplied work; continuous loss does not fund it
without a separate reservoir and occurrence law.

The full storage remains
\(H=\frac12x^TLx+\sum_{\text{all edges}}V_\lambda\), with
\(H'=-\|Lx\|^2/4\). At preparation its quadratic coefficient is

\[
 H_2(0)=\tfrac12u^TLu+
       \tfrac12\sum_{\{a,b\}}j_\lambda'(\theta_b^0-\theta_a^0)
                                      (v_b-v_a)^2.
 \tag{19}
\]

This is not automatically a coercive perturbation energy. Along the
evolving base the quadratic coefficient of the storage-loss identity is
\(-[\|Lx_1\|^2+2\langle Lx_0,Lx_2\rangle]/4\), which need not
be nonpositive by itself. First-order invisibility therefore implies
neither internal damping nor long-time transverse stability.

The mechanism separates three statements: the complete slot symmetry hides
all internal modes at first order; opposite modes can feed a collective
response at second order; full reflection conditions can protect exact
zero response. The leakage witnesses show that zero response at a protected
ideal origin is not a property of its full open preparation neighborhood.
The ideal law-dependent onset orders consequently require their exact
preparation scope. This does not alter the earlier finite acquired-source
contact bounds, classify every zero quadratic direction, establish a noisy
discriminator, select a unique law or identify a physical property.

## 7. Detached coefficient observer

The [full-fine observer](../../src/tnfr/physics/_organized_contact_preparation_sensitivity.py)
admits both complete primitive direction vectors, the endpoint law and a
supplied registration before forming the derivatives in (11)--(12). It
reuses the shared contact geometry, unit Laplacian, edge law and outward
rational interval arithmetic. The report distinguishes amplitude
coefficients from time derivatives and retains the evolving base's initial
phase acceleration, work/storage coefficients and initial winding allowance.

Exact child-mean and reflection relations justify their corresponding
symmetry flags through (8) and (10). An interval containing zero does not
establish a vanishing derivative or invariant output. The symbolic
registration and mode statements belong to this proof; the observer at a
supplied origin is not a registration scan, source certificate or finite
trajectory. Its fixed limits are preparation amplitude order two and
contact-time derivative order three.
The [contract tests](../../tests/physics/test_organized_contact_preparation_sensitivity.py)
differentiate the complete fine-state time rows independently of the
amplitude recurrence, and check all form/phase coefficients, work, storage,
explicit witnesses, primitive admission and bounded stopping. They evaluate
no acquired source or trajectory.
