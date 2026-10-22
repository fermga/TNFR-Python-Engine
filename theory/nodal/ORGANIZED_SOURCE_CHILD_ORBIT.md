# Exact organized-source state modulo a global child exchange

<a id="organized-source-child-orbit"></a>

**Conditional construction.** The two exchangeable children need not have
equal form or phase. Their averaged coordinates together with one joint
rank-at-most-one Gram matrix retain the complete primary state up to the
single exchange of those children. On the specified organized-source law,
these coordinates close both nonlinear continuous rows, energy and loss,
the fixed energy guard and the admitted marked
Reception–Coherence–Silence maps. Separate pair summaries can lose the
relations required for this prediction.

This extends the represented domain of the
[fixed-sector chart](ORGANIZED_SOURCE_CHILD_QUOTIENT.md#organized-source-child-quotient),
whose 72 primary coordinates require equal children. It does not replace
its separate finite-error results. The present generic primary dimension
is still 90. Removing a redundant global label is different from reducing
the number of continuous degrees of freedom.
[Section 7](#organized-source-child-relaxation) separately proves a
quantitative approach to that sector in the admitted acute energy domain,
with finite tolerance deadlines and decaying instantaneous prediction defects.
[Section 8](#organized-source-child-tracking) bounds the accumulated error
of a separately evolved symmetric core, for unit capacities without events.

The construction applies the existing
[support-symmetry criterion](SINE_PAIR_STATE.md#sine-pair-support-symmetry)
and [joint Gram method](SINE_PAIR_STATE.md#sine-replica-capacity-asymmetry)
to the actual 45-node source, reciprocal phase row and marked ideal word.
Its new obligation is to identify which relations across the nine paired
slots this support preserves. It introduces no new constitutive law.

## 1. Actual symmetry and retained state

Use the [source support and law](ORGANIZED_SOURCE_PORT.md): nodes `9*c+j`,
five nine-cycles and the matching child edges
`(0,1),(2,3),(0,2),(1,3),(0,4)`. All ninety edges have unit conductance.
The global permutation J exchanges whole children one and two and fixes
children zero, three and four. It fixes the marked targets zero and one.
The cycles couple adjacent slots, so swapping just one corresponding
pair of nodes is generally not a support automorphism.

Keep the ordered child orbits `A={0}, B={1,2}, C={3}, D={4}`, with nine
slots each. Their masses are `1,2,1,1`; write `M_mass` for the diagonal
36-coordinate mass matrix. P copies projected coordinates to their fine
nodes and `R=M_mass^-1 P^T` averages the fibers. Thus `Pi=PR=(I+J)/2`.
The inherited weighted Laplacian is L_w, as derived in the fixed-sector
owner; its 63 edges have total weight ninety.

Declare the raw real carrier lift `psi_(9c+j)=alpha*(j-4)`,
`alpha=2*pi/9`. Retain 36 projected forms u and 36 projected real phase
offsets d. Write `vartheta=bar_psi+d`. The unequal children have

\[
 x_{1j}=u_{Bj}+a_j,\quad x_{2j}=u_{Bj}-a_j,\qquad
 \Theta_{1j}=\vartheta_{Bj}+b_j,\quad
 \Theta_{2j}=\vartheta_{Bj}-b_j.
\]

All other children are copied from their singleton orbit. Positive held
capacities must satisfy `nu_(1j)=nu_(2j)`. Their 36 retained coordinates
are separate law parameters, not part of the 90-primary-coordinate count.
They may vary between slots and between the four child orbits.

Set `v=(a_0,...,a_8,b_0,...,b_8)` and retain

\[
 G=vv^T=
 \begin{pmatrix}G_{aa}&G_{ab}\\G_{ab}^T&G_{bb}\end{pmatrix}.
\]

The domain requires one jointly symmetric positive-semidefinite matrix
of rank at most one. Its 171 distinct entries are redundant coordinates.
They are products of one deterministic hidden vector, not an ensemble
covariance, and cannot be replaced by unrelated positive local blocks.
The [joint-realizability owner](DERIVED_FORM_PHASE.md) explains the general
distinction between pairwise admissibility and a common underlying state.

If G is nonzero, choose its first positive diagonal entry G_kk and set
`v_k=sqrt(G_kk)>0`, `v_i=G_ik/v_k`. Rank-one realizability proves that
these values reconstruct G, and its only real lifts are v and -v. At
G=0 the only lift is zero. Hence `(u,d,G,nu)` reconstructs the fine
primary state up to exactly J. Separate signs for different slots or for
the form and phase blocks do not describe the same equivalence relation.

The generic dimension is `72+18=90`, with a singular zero-hidden stratum.
The canonical sign rule is an algebraic representative, not a globally
smooth choice of labeled state. At v=0 the differential of `v -> vv^T`
vanishes; this coordinate singularity does not make the fine flow singular.
Labeled nonsymmetric observations need a choice between the two lifts.
Invariant observations and fixed-target values do not.

These coordinates retain declared real phase lifts. They do not quotient
independent turns, define a global chart on circular phases, or infer a
preferred lift from wrapped observations.

## 2. Closed projected nonlinear law

The complete unforced law in structural time tau is

\[
 \dot x=\tfrac14D_f[-L_f x+\gamma S_f(\Theta)],\qquad
 \dot\Theta=\tfrac\gamma4D_fL_f x,\qquad
 \gamma=(1023\pi)^{-1},
\]

where `S_f(Theta)_i=sum_(k~i) sin(Theta_k-Theta_i)` and Gamma is absent.
Every consumed capacity is held between events; the same capacity
multiplies both rows. Define `q=M_mass^-1 L_w u` and let D be the
diagonal retained capacity matrix. Hidden form cancels in the projected
Laplacian, so `R L_f x=q` even away from the fixed sector.

For nonnegative r, put

\[
 C(r)=\cos\sqrt r,\qquad
 F(r)=\frac{\sin\sqrt r}{\sqrt r},\quad F(0)=1.
\]

Their power series in r remove any division at zero. For a B-cycle edge
jk define

\[
 r_{jk}=G_{bb,jj}+G_{bb,kk}-2G_{bb,jk}=(b_k-b_j)^2,
 \qquad r_j=G_{bb,jj}=b_j^2.
\]

The projected sine current `s_G=R S_f(Theta)` is obtained by these exact
contributions; each displayed term is added to the indicated row:

| Row and neighbor | Contribution |
| --- | --- |
| B slot j, adjacent B slot k | `sin(vartheta_Bk-vartheta_Bj)*C(r_jk)` |
| B slot j, U slot j, U=A or C | `sin(vartheta_Uj-vartheta_Bj)*C(r_j)` |
| U slot j, B slot j, U=A or C | `2*sin(vartheta_Bj-vartheta_Uj)*C(r_j)` |
| Every unaffected cycle or A–D edge | Its original fine sine term |

These follow from `sin(t+b)+sin(t-b)=2*sin(t)*cos(b)` and the
inherited neighbor counts. Consequently both projected rows close:

\[
 \dot u=\tfrac14D(-q+\gamma s_G),\qquad
 \dot d=\tfrac\gamma4Dq.
\]

The phase row does not depend directly on hidden coordinates, but the
form row consumes cross-slot entries of G_bb. Their evolution will also
consume the mixed block G_ab. Keeping just the phase correction at one
instant does not establish future closure.

## 3. Joint Gram evolution and exact balance

Let `H=L_C9+2I`, with diagonal four and adjacent entries minus one, and
`V=diag(nu_Bj)`. Define the symmetric matrix K as a weighted cycle
Laplacian with edge weight

\[
 \cos(\vartheta_{Bk}-\vartheta_{Bj})F(r_{jk}),
\]

plus the diagonal entry

\[
 [\cos(\vartheta_{Aj}-\vartheta_{Bj})+
   \cos(\vartheta_{Cj}-\vartheta_{Bj})]F(r_j).
\]

The half-difference of the two fine sine currents is exactly `-K b`.
Subtracting the complete fine rows therefore gives

\[
 \dot a=-\tfrac14VHa-\tfrac\gamma4VKb,\qquad
 \dot b=\tfrac\gamma4VHa,\qquad
 \dot v=\mathcal A v,\quad
 \mathcal A=\tfrac14
 \begin{pmatrix}-VH&-\gamma VK\\\gamma VH&0\end{pmatrix}.
\]

The entries of K depend on `(d,G_bb)`, not on the lift's overall sign.
The exact joint equation is thus

\[
 \boxed{\dot G=\mathcal A G+G\mathcal A^T.}
\]

These equations are smooth on the realized domain, including G=0;
the power series also define a smooth extension to ambient coordinates.
Starting from a realized G, the fine lifted solution proves preservation
of rank-one realizability and supplies the unique projected solution.
The two signs give the same retained trajectory. This is a constructive
closure proof, not a prescription to integrate redundant unconstrained
entries or repair a matrix that has drifted out of its domain.

Let bar E be the fixed-sector excess at `(u,d)`, and
`H_*=45*(1-cos(alpha))`. Exact edge grouping yields

\[
 \begin{split}
 E={}&\bar E+\operatorname{tr}(H G_{aa})\\
 &+2\sum_{jk\in C_9}\cos(\vartheta_{Bk}-\vartheta_{Bj})[1-C(r_{jk})]\\
 &+2\sum_j[\cos(\vartheta_{Aj}-\vartheta_{Bj})+
             \cos(\vartheta_{Cj}-\vartheta_{Bj})][1-C(r_j)].
 \end{split}
\]

Each unoriented cycle edge appears once. This identity does not require
acute angles; the correction need not be nonnegative outside an acute
projected domain. For arbitrary admitted symmetric positive capacities,

\[
 \rho=\tfrac14q^T M_{\rm mass}Dq+
       \tfrac12\operatorname{tr}(H V H G_{aa}),\qquad
 \dot E=-\rho.
\]

Indeed the two hidden fine gradients are `q_B+Ha` and `q_B-Ha`, whose
weighted squares cancel the cross terms. Slot-dependent V must remain
between the two H factors. The fine inverse-capacity-weighted form and
phase-offset totals reduce to `sum(mass/nu*u)` and `sum(mass/nu*d)`;
hidden terms cancel because the paired capacities agree.

These are exact identities for the supplied law. They neither select
its constitutive coefficients nor identify these quantities with
laboratory energy, time or measured information.

## 4. Guard and marked hybrid maps

Use the same supplied fixed energy level and mandatory first downward
crossing from the [guard owner](ORGANIZED_SOURCE_GUARD_SENSITIVITY.md).
Since the full energy is reconstructed exactly, its crossing set is
the same for both fine lifts and the retained trajectory. Existence,
positive first occurrence and transversality remain the guard owner's
separate premises; this representation does not establish them anew.

For the complete ideal word, intersect with the
[section 23 word domain](ORGANIZED_SOURCE_PORT.md#organized-source-word-reuse):
initial unit capacities, its acute lift cell, `E<1/100`,
`abs(initial form mean)<1/30`, and complete supplied admission context.
The continuous identity above allows more general held capacities; it
does not widen this particular first-word admission theorem.

Let beta, lambda and kappa be the already admitted Reception, Coherence
and Silence factors. Each stage uses one common fine snapshot at the
fixed targets i=0,1. Reception retains the same inherited mean:

\[
 u_i^+=(1-\beta)u_i+\frac\beta5\sum_bN_{ib}u_b.
\]

The hidden forms cancel in the two exchanged neighbors. Only these two
singleton projected forms change; d, G and capacities are held.

For Coherence, write j=i for the target's slot. Its relative resultant is

\[
 z_i=\sum_{k\in\{\text{two A-cycle neighbors, D slot }j\}}
           e^{i(\vartheta_k-\vartheta_i)}
       +2e^{i(\vartheta_{Bj}-\vartheta_i)}C(G_{bb,jj}).
\]

This is the exact five-neighbor fine sum. On the existing regular argument
branch, `d_i^+=d_i+lambda*Arg(z_i)`; every other projected phase, all
forms, G and capacities are held. Silence then sends the two target
capacities to `kappa*nu_i`, holding u, d and G. It preserves equality
between the capacities of children one and two, so the subsequent
held-capacity law remains in this domain.

Each primary stage commutes with J. The represented state therefore
reconstructs the marked ideal word and its changed-capacity continuation
up to that same global exchange. The hidden vector itself is unchanged
by these marked maps; a capacity jump does not reset G.

Histories, ordinals, debt, prior Coherence, policies and any labeled
selector inputs remain separately retained and admitted. Arbitrary
runtime contexts are not functions of G. Full labeled execution needs
the matching fine context and lift; no quotient of arbitrary context or
new occurrence law follows. Stored-pressure jumps and fresh-law
association retain their existing contracts. See the
[operator descent](../STRUCTURAL_OPERATORS.md#collective-operator-descent)
and [grammar descent](../STRUCTURAL_OPERATORS.md#collective-grammar-descent)
conditions. The proof concerns the primary ideal maps under those
premises, not a new runtime executor.

## 5. Frozen information-loss controls

<a id="organized-source-child-orbit-controls"></a>

Both controls were fixed at `h=1/128`, unit capacity and carrier
projected phase before evaluation. They are static state and derivative
identities, not samples from a trajectory or unknown crossing states.
They disprove the named omissions; they do not prove that all 171
entries are a minimal collection of stored numbers.

### Cross-slot phase association is visible immediately

Set projected form `h*(e0-e1)` on child A, every other projected form
zero, and a=0. In the first hidden phase vector all b_j are +h except
b_2=-h; in the second, only b_4=-h. Every local pair Gram is
`(a_j^2,a_j*b_j,b_j^2)=(0,0,h^2)`. Their hidden means agree, so the
means of each labeled child also agree. Both states have

\[
 E=6h^2+4\cos\alpha(1-\cos2h)+36(1-\cos h)
       \le32h^2=1/512<1/100,\qquad \rho=20h^2.
\]

Their marked targets and every incident initial form, phase and capacity
agree. Thus initial fresh target pressure and direct ideal word outputs
at the targets agree as well, conditional on equal admitted context.
The largest cycle gap is at most `alpha+2h<pi/2`, matching gaps have
magnitude h and the global form mean is zero.

Nevertheless their projected B-slot-1 form rates differ, first minus
second, by

\[
 \dot u^{(1)}_{B1}-\dot u^{(2)}_{B1}
   =\frac\gamma4\sin\alpha(\cos2h-1)<0.
\]

The missing entry `G_bb,1,2` is respectively `-h^2` and `+h^2`.
Adjacent phase associations matter even when local pair Grams, child
means, energy, instantaneous loss and initial target data all agree.

### Separate complete form and phase Grams still lose mixed information

Now set every projected form to zero, `a=h*e0`, and compare
`b=+h*e1` with `b=-h*e1`. All local pair Grams agree because the two
supports are disjoint. Even the full separate matrices G_aa and G_bb
agree, as do both projected first derivative rows. Both preparations
are acute, have zero global form mean, and satisfy

\[
 E=4h^2+4(1+\cos\alpha)(1-\cos h)\le8h^2<1/100,
 \qquad \rho=9h^2.
\]

But `G_ab,0,1` is respectively `+h^2` and `-h^2`. Since
`(Ha)_1=-h`, the squared hidden phase has

\[
 \frac{d}{d\tau}b_1^2=\mp\frac\gamma2h^2.
\]

This distinction reaches a projected observable at the next derivative:

\[
 \ddot u^{(+)}_{B0}-\ddot u^{(-)}_{B0}
       =\frac58\gamma^2h\sin\alpha\sin h>0.
\]

To verify the coefficient, the initial projected phase rate is zero and
the two projected form rates coincide, so their q derivatives coincide.
Only the derivative of the cycle phase correction differs. Here
`dot b_1-dot b_0=-5*gamma*h/4`; differentiating the two cosines gives
a projected sine-rate contrast `5*gamma*h*sin(alpha)*sin(h)/2`.
The additional form-row factor gamma/4 yields the stated curvature.
The mixed block records a relative association between hidden form and
phase that neither separate full Gram retains.

## 6. Detached implementation and boundary

The [private orbit reader](../../src/tnfr/physics/_organized_source_child_orbit.py)
encodes fine primitives, decodes admitted invariant coordinates to a
canonical fine lift and observes the same complete law through the shared
fine capacity reader. Projected rows are actual fiber averages; selecting
one representative off the fixed sector would be incorrect. The joint
Gram derivative is built from the fine hidden derivatives. No second
sine kernel, numerical integrator or event executor is introduced.

The [shared exact rank-one lift](../../src/tnfr/mathematics/_exact_linear_algebra.py)
normalizes every matrix entry before checking symmetry and realizability.
It also serves the existing [P5 invariant decoder](../../src/tnfr/physics/p5_reduction.py).
The implementation retains exact rational or exactly represented real
input values and rational canonical lifts. A realizable rational Gram
whose canonical square root is irrational is explicitly unsupported by
this decoder, though it is a valid point of the real mathematical domain.
It is not rounded to a nearby lift or declared physically impossible.

The encoded fine state and every consumed primitive are readmitted;
cached reports, Boolean scalars, nonfinite values, unequal paired
capacities or arbitrary positive-semidefinite higher-rank matrices do not
replace these premises. The fine state in a decoded report is the
canonical representative, not a claim to recover the original labeled
preparation without its missing overall sign.

[Independent controls](../../tests/physics/test_organized_source_child_orbit.py)
compare exact reconstruction, the shared-law observation and the
prospectively fixed state/derivative distinctions. Algebraic closure and
the conditional word proof establish sufficiency under this selected
law. These reconstruction results alone establish neither a minimal state,
faster simulation, an attractor, autonomous hierarchy formation, nor an
empirical mapping.

<a id="organized-source-child-relaxation"></a>
## 7. Quantitative approach to the symmetric sector

The exact joint state explains what unequal children retain. A different
question is when their difference becomes small enough for a declared
approximation. The existing [full-state capture](ORGANIZED_SOURCE_CAPTURE.md)
already implies eventual equality on its admitted source. The result here
adds a uniform decay rate and a finite tolerance deadline while the
projected environment continues to move. It uses the same complete law;
no feedback, noise removal or equality reset is added.

This is a common quadratic argument for a time-varying hidden stiffness.
The [weak-contact residual theorem](ORGANIZED_WEAK_CONTACT_MAINTENANCE.md#weak-contact-fine-relaxation)
provides a related method on a different support. The
[formed-class maintenance theorem](SINE_FORMED_CLASS_MAINTENANCE.md#a-strict-lyapunov-bound-on-each-retained-acute-chart)
uses a fixed-target potential. Neither its constants nor a fixed-potential
derivative can be imported into the present moving hidden subsystem.

### Domain, phase collar and capacities

Admit the actual fixed winding-one acute lift cell and `E<1/100` from
the full fine primitives. Keep positive held capacities, with equal
values on paired children as in section 1. The
[source word-domain argument](ORGANIZED_SOURCE_PORT.md#organized-source-word-reuse)
gives, on this unrestricted acute cell,

\[
 \sum_e u_e^2\le4E<1/25,
\]

where u_e is each edge's deviation from its carrier angle. This follows
from the cycle Taylor bound with cancellation of the carrier's first
variation, together with nonnegative form and matching storage. Thus
every cycle angle has magnitude below `alpha+1/5<pi/3`, and every
matching angle has magnitude below 1/5. These inequalities concern the
complete actual state, rather than only its projected phase.

The straight lifted segment from a state to its global child exchange
has the same bounds: each edge angle interpolates two angles inside
the same interval. Each weight in the secant matrix K from section 3
is an average of cosine along this segment. Therefore, throughout the
admitted flow,

\[
 \tfrac12H\preceq K(\tau)\preceq H,\qquad
 H=L_{C_9}+2I\succeq2I.
\]

The shared acute boundary barrier exceeds 1/100, and the actual
positive-capacity storage balance is nonpositive. Initial membership
therefore retains this domain. Capacities of the other three children
may be nonuniform; they affect the moving projected environment but do
not enter the paired matrix V. This proof does not differentiate K or
require it to commute with H or V.

### A common quadratic metric for both hidden rows

Define the auxiliary quantity

\[
 W=a^TV^{-1}a+b^TV^{-1}b+2\gamma a^TV^{-1}b,
 \qquad N=\|a\|_2^2+\|b\|_2^2=\operatorname{tr}G.
\]

W is a positive proof metric in the declared normalized coordinates,
not the physical storage or a new primitive parameter. It is a linear
functional of the diagonal and same-slot mixed entries of G. Keeping
this scalar supplies an envelope; section 5 still excludes those local
entries as a complete predictive state.

With `nu_min=min_j V_jj` and `nu_max=max_j V_jj`, the metric sandwich is

\[
 \frac{1-\gamma}{\nu_{\max}}N\le W
       \le\frac{1+\gamma}{\nu_{\min}}N.
\]

Differentiate using both actual hidden rows, including the reciprocal
phase row. The capacity factors cancel exactly:

\[
 W'=-\frac{1-\gamma^2}{2}a^THa
       -\frac\gamma2a^TKb-\frac{\gamma^2}{2}b^TKb.
\]

From `0<=K<=H`, congruence gives `K H^-1 K<=K`; no simultaneous
diagonalization is used. Young's inequality then gives

\[
 \frac\gamma2|a^TKb|
 \le\frac14a^THa+\frac{\gamma^2}{4}b^TKH^{-1}Kb
 \le\frac14a^THa+\frac{\gamma^2}{4}b^TKb.
\]

Since `0<gamma<1/2`, the remaining form coefficient
`1/4-gamma^2/2` is at least `gamma^2/4`. Using the two matrix lower
bounds proves

\[
 W'\le-\frac{\gamma^2}{4}N\le-r_*W,
 \qquad r_* =\frac{\nu_{\min}\gamma^2}{4(1+\gamma)}>0.
\]

Consequently, for every elapsed structural time tau in the declared law,

\[
 W(\tau)\le W(0)e^{-r_*\tau},\qquad
 N(\tau)\le\frac{\nu_{\max}}{1-\gamma}W(0)e^{-r_*\tau}.
\]

The rate is not uniform when paired capacity tends to zero; zero paired
capacity is outside this certificate. A common clock rescaling must
rescale the rate and elapsed time consistently. The represented norm
does not identify form and phase with the same laboratory unit.

### Passage through the marked word and prediction scope

The separately admitted ideal Reception–Coherence–Silence word at
targets zero and one holds a, b and V exactly. Thus it has `Delta W=0`
and `Delta N=0`, including Silence's change of the A-target capacities.
The existing passivity theorem retains the same acute energy domain.
Before and after this word, the same rate applies, without restarting
the bound or requiring the environment to have equilibrated first.
Elapsed tau is the sum of the continuous segments; the ideal jumps
have no invented finite duration. Word admission, history, policy,
selection and any represented map error remain separate premises.

This proves acquisition of any positive declared tolerance, not entry
into exact equality at a finite time. A nonzero hidden vector cannot
reach zero under its smooth homogeneous differential equation in finite
time: uniqueness applied backwards from zero would contradict its
nonzero initial state. The marked word also cannot make it zero because
it holds that vector. Exact symmetry remains an asymptotic statement.

At one actual state compare its projected field with the fixed-sector
field evaluated at the same projected coordinates and capacities.
The earlier quadratic sine defect gives

\[
 \|R\dot x-f_{\rm sector}(u,d)\|_\infty
 \le\frac{5\gamma\nu_{\rm all,max}}4\|b\|_\infty^2
 \le\frac{5\gamma\nu_{\rm all,max}}4N,\qquad
 R\dot\Theta-g_{\rm sector}(u,d)=0.
\]

The global capacity maximum is held on a continuous segment; the
admitted Silence factor is at most one, so its initial value also
bounds the segment after this marked word. At a target, the actual
and projected Coherence resultants differ by at most `b_j^2`.
Both have real part greater than four in the admitted energy domain,
so their argument difference is at most `N/4`, and their ideal phase
reset difference is at most `lambda*N/4` for `0<=lambda<=1`.
Reception commutes with this projection, and Silence holds the primary
coordinates. These instantaneous defects inherit the displayed decay.

This section does not follow a separately evolved reduced trajectory.
Its accumulated error has the separate coupled stability proof in
[section 8](#organized-source-child-tracking), under narrower capacity and
event premises.
Nor does a small N give a uniform late-time energy-guard timing error:
that question still needs a strictly positive crossing-loss margin.

### Prospectively fixed static controls

Both controls fix unit capacities, `h=1/128`, and zero projected form
and phase offsets. They evaluate algebra and bounds, not a trajectory
or a performed operator event.

First take `a=h*e0`, `b=h*e1`. The section 5 mixed-information control
already establishes its acute energy admission; its initial metric
is exactly `W(0)=2h^2`. For outward gamma bounds `g_lo<=gamma<=g_hi`,
define the rational certified rate and elapsed time

\[
 r=\frac{g_{\rm lo}^2}{4(1+g_{\rm hi})},\qquad T=10/r.
\]

Since `exp(-10)<2^-10` and `g_hi<1/2`, the certificate implies

\[
 N(T)\le\frac{2h^2}{1-g_{\rm hi}}e^{-10}<h^2/256.
\]

Initially `N(0)=2h^2`; the declared approximation tolerance is therefore
absent initially and guaranteed by this conservative structural deadline.
This finite-tolerance acquisition does not assert formation of a new
NFR identity, spontaneous support selection or a physical timescale.

Second take `a=epsilon*h*e0`, `b=h*e0`, with `epsilon=1/65536`.
Here `K_00=2*(1+cos(alpha))*sinc(h)`, so the complete fine law gives

\[
 N'=h^2[-2\epsilon^2+
           \tfrac{\gamma\epsilon}{2}(4-K_{00})]>0.
\]

Indeed `cos(alpha)<4/5`, `sinc(h)<1` and `gamma>1/4096` imply
`gamma*(4-K_00)>1/10240>4*epsilon`. Its excess is
`4*epsilon^2*h^2+4*(1+cos(alpha))*(1-cos(h))<1/100`, and all fine
edge angles are acute. Meanwhile the proved inequality gives `W'<0`.
Thus exponential control by W does not assert monotonically decreasing
Euclidean hidden distance. Both rows and their mixed term matter.

### Primitive certificate and numerical boundary

The [relaxation reader](../../src/tnfr/physics/_organized_source_child_relaxation.py)
rebuilds the canonical fine lift, shared observation, raw offset angles,
energy and boundary margins from primitives. It supplies a conditional
metric and squared-distance allowance, with explicit unavailability when
the geometric premises or exponential work budget are not certified.
It admits no event context and performs no integration or reset.

The report-free [quadratic kernel](../../src/tnfr/physics/_sine_lyapunov.py)
keeps this common-metric theorem separate from its existing fixed-potential
helpers. The reader uses the existing exact rational negative-exponential
evaluator, whose nonnegative exponent is limited to 4096. That limit is
a numerical work boundary; it does not limit the analytical theorem.
Exact rational scalar bounds preserve nonzero allowances below the
trigonometric interval grid rather than materializing them as zero.

[Independent controls](../../tests/physics/test_organized_source_child_relaxation.py)
check the full fine derivative, the metric sandwich, nonuniform capacities,
the frozen deadline and transient-growth distinction, event-map invariance
and unavailable domains. The result supplies a justified tolerance for
discarding this asymmetry in instantaneous predictions under the selected
law. It neither establishes a unique canonical constitutive law nor makes
the lossy reduced state exactly sufficient at finite time.

For the frozen first control, the rational certificate gives
`r>2.4196e-8` and `T<4.133e8` in structural tau units, with a joint
squared-norm allowance below `5.544e-9`, compared with the declared
`h^2/256>2.384e-7`. The instantaneous form-field defect is bounded by
`2.157e-12` and the target argument defect by `1.386e-9`. These are
conservative analytical allowances at the prescribed deadline, not
observed errors or a laboratory settling time.

<a id="organized-source-child-tracking"></a>
## 8. An independently evolved symmetric approximation

Small instantaneous defects do not by themselves bound an accumulated
prediction error. Here the complete state z evolves under the same fine
law while a symmetric state w evolves independently from `w(0)=Pi z(0)`.
All capacities are exactly one and remain held; Gamma, external input,
events and changes of support are absent. Time is the same structural
tau in both rows. The initial fine state must lie in section 7's actual
winding-one acute cell with `E<1/100`.

The prediction is the error between Pi z(tau) and w(tau), including both
form and lifted phase. The result is uniform for all nonnegative tau and
decays to zero. It strengthens the [earlier finite comparison](ORGANIZED_SOURCE_CHILD_QUOTIENT.md#organized-source-child-first-hit-error)
on this narrower invariant domain; it does not replace that comparison's
global-sine or first-hit scope. Neither calculation evaluates a trajectory.

### Two retained domains and the correct error space

In the acute lifted cell the storage is convex and invariant under the
whole-child exchange J. Consequently

\[
 E(\Pi z)\le\tfrac12[E(z)+E(Jz)]=E(z).
\]

The projection stays in the same convex lift cell. The initial core
therefore satisfies `E(w(0))<=E(z(0))<1/100`. Each independent solution
retains its acute sublevel by the actual storage loss and boundary barrier.
The core remains symmetric because the complete law commutes with J.
At every time, Pi z and w both have the section 7 phase collar. The
segment joining their lifted phases stays in that collar, so its sine
secant Hessian obeys `L/2 <= K_e(tau) <= L` as quadratic forms.

Write `p=(Pi x)-w_x` and `q=(Pi delta)-w_delta` in the fine 45-node lift.
At unit capacity both complete rows conserve their ordinary sums, and
Pi preserves those sums. Matched initialization therefore gives
`sum(p)=sum(q)=0` for every time. The initial source means may be nonzero.
The actual [full-support spectral bound](ORGANIZED_SOURCE_PORT.md#a-gap-for-the-actual-fine-support)
is `p^T Lp >= (4/15)||p||^2` on this mean-zero space, and likewise for q.
The hidden nine-slot gap two is not the gap for these errors.

All norms below use the 45-node lift. Equivalently, the 36-coordinate
norm weights the child orbits by their masses `1,2,1,1`. An unweighted
36-coordinate norm does not give the displayed identities.

### Actual edge forcing and a forced common metric

Orient each edge and let B have +1 at its tail and -1 at its head, so
`L=B B^T` and the sine field is `B sin(theta_head-theta_tail)`.
The difference between the projected sine field and the sine field at
the projected state is `B j`. On each of the two copies of a B-child
cycle edge from slot l to k, put

\[
 j_e=\sin(\bar\theta_e)[\cos(b_k-b_l)-1].
\]

On each of the four matching edges adjacent to the exchanged children
at slot l, use `j_e=sin(bar theta_e)[cos(b_l)-1]`, with its own oriented
projected angle. Other entries vanish. Pairing the two child currents
proves this identity at singleton as well as paired nodes; the linear
form row and reciprocal phase row commute with Pi exactly.

Since `|1-cos v|<=v^2/2`, the actual edge multiplicities give

\[
 \|j\|^2\le\tfrac12\sum_{C_9}(b_k-b_l)^4+\sum_l b_l^4
 \le 5\|b\|^4.
\]

For the last inequality, each squared edge difference is at most
`2||b||^2`, their sum is at most `4||b||^2`, and the fourth-power slot
sum is at most `||b||^4`. No nodal maximum-degree approximation replaces
this fine edge count. The error obeys both complete rows:

\[
 p'=-(Lp+\gamma K_e q)/4+\gamma Bj/4,\qquad
 q'=\gamma Lp/4.
\]

Define `W_e=||p||^2+||q||^2+2 gamma p^T q`. It lies between
`(1-gamma)(||p||^2+||q||^2)` and
`(1+gamma)(||p||^2+||q||^2)`. The common-metric argument of section 7,
now on the full mean-zero space, bounds the homogeneous derivative by
`-p^T Lp/8-gamma^2 q^T Lq/8`. The additional term is
`(gamma/2)(p+gamma q)^T B j`. Apply Young's inequality separately:

\[
 \begin{split}
 \tfrac\gamma2\sqrt{p^TLp}\,\|j\|
   &\le \tfrac1{16}p^TLp+\gamma^2\|j\|^2,\\
 \tfrac{\gamma^2}2\sqrt{q^TLq}\,\|j\|
   &\le \tfrac{\gamma^2}{16}q^TLq+\gamma^2\|j\|^2.
 \end{split}
\]

Using `0<gamma<=1/2`, the full spectral gap and the metric sandwich yields

\[
 W_e'\le-r_eW_e+10\gamma^2\|b\|^4,\qquad
 r_e=\frac{\gamma^2}{60(1+\gamma)}.
\]

Both stiffnesses may change with the moving environment. This proof
neither differentiates them nor assumes simultaneous diagonalization.

### Uniform accumulated error and its limits

For unit capacities let `C_0=W_h(0)/(1-gamma)`, with W_h the hidden
metric of section 7. Then `||b(tau)||^2<=C_0 exp(-r_h tau)` and
`r_h=15 r_e`. Integrating the forced scalar inequality from `W_e(0)=0`
gives

\[
 W_e(\tau)\le
 \frac{600(1+\gamma)}{29}C_0^2
       [e^{-r_e\tau}-e^{-30r_e\tau}],\qquad
 \|p\|^2+\|q\|^2\le\frac{W_e(\tau)}{1-\gamma}.
\]

The bracket is nonnegative, bounded by one and converges to zero.
Since C_0 is quadratic in the initial hidden coordinates, the projected
error norm has a uniform quadratic dependence on their amplitude. This
is a controlled lossy approximation; it does not make the symmetric state
exactly sufficient for a nonzero hidden preparation at finite time.

For comparison with the unprojected fine state, orthogonality gives
`||z-w||^2=2N+||p||^2+||q||^2`. Its hidden contribution has the factor
two from the two copies of each slot. Thus the quadratic projected-norm
claim must not be attributed to the full reconstructed error, whose
leading norm contribution can be linear in hidden amplitude.

Matched initialization is essential: a reference with a different common
phase offset has a constant error even at zero hidden state. This result
does not extend through the marked word, whose Silence changes capacities
and whose resets require a separate error budget. It supplies no guard-time
accuracy without crossing transversality, new NFR identity, physical
identification or unique selection of the constitutive law.

### Detached certificate and prospectively fixed controls

The [tracking reader](../../src/tnfr/physics/_organized_source_child_tracking.py)
admits all primitives and unit capacities, rebuilds the section 7 domain
at time zero, and constructs the matched reference with the existing exact
quotient lift. No supplied report or proposed reference replaces these
premises. From outward `g_lo<=gamma<=g_hi`, use

\[
 r=\frac{g_{\rm lo}^2}{60(1+g_{\rm hi})},\qquad
 C=\frac{W_{h,0}^{\rm upper}}{1-g_{\rm hi}},\qquad
 F=10g_{\rm hi}^2 C^2.
\]

The evaluated metric allowance is
`F integral_0^T exp[-r(T-s)] exp[-30r s] ds`; the uniform allowance is
`F/(29r)`. Divide each by `1-g_hi` for the lifted joint squared error.
The [shared convolution kernel](../../src/tnfr/mathematics/_comparison_flow.py)
uses the smaller of a simple positive bound and the exponential difference
quotient, keeping exact zero at T=0 and avoiding small-time cancellation.
The existing cycle relaxation consumer delegates to that same kernel.
The shared negative-exponential evaluator admits each argument up to
4096; this is a numerical work cap, not a bound on the analytical time
domain. If only this cap fails, the uniform bound remains available and
the requested pointwise evaluation is explicitly unavailable.

The prospective preparation fixes unit capacities, `h=1/128`, zero
projected form and offsets, `a=h e0`, and `b=h e1`. Its reduced reference
is the carrier equilibrium. At time zero the comparison error is exactly
zero, yet its projected form-rate defect at child-one slot zero is
`(gamma/4) sin(alpha) [cos(h)-1]<0`. The full and reduced evolutions
therefore need not agree even from matched projected initialization.
The frozen acceptance thresholds are a uniform joint squared-error
allowance below `h^2/128` and, at `T=10/r`, a pointwise allowance below
`h^2/2^20`. A companion preparation with h/2 must reduce both allowances
by exactly 1/16. The time and thresholds were fixed before evaluation.

[Primitive and numerical controls](../../tests/physics/test_organized_source_child_tracking.py),
[independent edge and forced-metric controls](../../tests/physics/test_organized_source_child_tracking_proof.py)
and [convolution controls](../../tests/mathematics/test_decay_convolution.py)
test these contracts, including means, raw lifts, tiny rational scaling
and unavailable domains. They evaluate static derivatives and analytical
allowances, not independent trajectories, measured errors or solver speed.

The frozen certificate gives a uniform joint squared-error allowance below
`3.087e-7`, against `h^2/128>4.768e-7`. Its certified rate exceeds
`1.6131e-9`, and the prescribed structural time is below `6.200e9`.
At that time the allowance is below `1.402e-11`, against
`h^2/2^20>5.820e-11`. The companion has exactly 1/16 of both allowances;
the initial projected form-rate defect is strictly negative. These numbers
are conservative proof bounds in structural units, not measured settling
times. No historical continuation supplies evidence for this comparison.
