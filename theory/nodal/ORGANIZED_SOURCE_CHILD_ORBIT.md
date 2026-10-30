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

<a id="organized-source-word-phase-memory"></a>
## 9. A structural word converts hidden phase into persistent mean error

The zero-limit raw tracking conclusion of section 8 does not transfer
through the marked word. On the existing off-carrier preparation, the
ideal Reception--Coherence--Silence word creates a nonzero projected phase
error from an initially exact projected match. A component of that error
is conserved by both complete post-word flows. Hidden differences may
relax while this collective memory remains.

This is a conditional obstruction for the supplied mathematical source
and ideal primary maps. It strengthens the
[immediate word comparison](ORGANIZED_SOURCE_CHILD_QUOTIENT.md#organized-source-child-word-error)
by following a conserved error component, without calculating a trajectory
or asserting the full asymptotic error. The earlier no-event tracking and
exact orbit-state results retain their original premises.

### 9.1 Matched incoming states and the admitted action

Use the actual ninety-edge support of section 1. For child `c=0,...,4`
and slot `j=0,...,8`, keep the exact real carrier
`psi_(9*c+j)=alpha*(j-4)`, `alpha=2*pi/9`. The incoming fine preparation is

\[
 x=0,\qquad \nu_i=1,\qquad
 \delta_9=\phi+b,\quad\delta_{18}=\phi-b,\quad
 \delta_i=0\ (i\ne9,18),\qquad
 \phi=2^{-7},\quad b=2^{-20}.
\]

The comparison state is the independently continued symmetric core
`w=Pi z`, with both offsets at nodes 9 and 18 equal to phi. Both initial
form and phase projections match exactly, including their ordinary means.
No unequal common phase offset is inserted. These are the previously
constructed static off-carrier inputs, not claimed states at either
unknown first guard crossing. Their irrational carrier is kept exact;
binary64 carrier midpoints would specify different preparations.

Apply one supplied word at a common time. Every ideal stage uses one
incoming snapshot for both targets 0 and 1; no continuous time elapses
between Reception, Coherence and Silence. Use the registered default
Reception mix beta, Coherence pressure retention, phase coefficient lambda
and Silence factor kappa, interpreting represented coefficients as exact
real parameters. In particular `0<beta<=1`, `0<lambda<1` and `0<kappa<1`.
The inherited five-neighbor counts remain those of the fine support.

The [word domain](ORGANIZED_SOURCE_PORT.md#organized-source-word-reuse)
requires the stipulated `AL,IL` histories, ordinal two, prior Coherence,
zero debt, common kind, initialized core grammar, rejection mode `raise`,
and explicit IL `event_domain` admission. Keep the existing Reception
ceilings, unit rails and inclusive Silence pre-call capacity floor.
These are supplied contextual premises, not information reconstructed
from the physical coordinates. The default signed-positive IL policy
would reject zero target form; it is not the policy used here.

Both exact inputs lie in the fixed winding-one acute cell. Actual edge
accounting gives `E_f=4*(1+cos(alpha))*(1-cos(phi)*cos(b))` and
`E_c=4*(1+cos(alpha))*(1-cos(phi))`. Hence their excess obeys
`0<=E<=4*(phi^2+b^2)<1/100`, and both form means vanish. The shared
source barrier and one-word reuse budget therefore provide the required
mathematical domain and scalar margins. Reception has zero form proposals.
Its target-0 fresh pressure is `gamma*sin(phi)*cos(b)/2` on the fine input
and `gamma*sin(phi)/2` on the core; target-1 fresh pressure is zero. These
obey the retained Reception ceilings, and the unit pre-Silence capacities
exceed the declared floor. A raw-phase reader requiring represented centers
and a positive radius cannot be called on rounded carrier values and used
as proof of this exact preparation's admission. Exact offset geometry,
the barrier and shared factor/budget owners suffice here.

### 9.2 The complete ideal word and its work

Reception changes no physical coordinate on either zero-form input.
At target zero, the common-snapshot Coherence resultants are

\[
 Z_f=r+2e^{i\phi}\cos b,\qquad Z_c=r+2e^{i\phi},\qquad
 r=1+2\cos\alpha>0.
\]

Both are in the right half-plane, with arguments `0<a_f,a_c<phi`.
Moreover

\[
 \Im(Z_f\overline{Z_c})=2r(\cos b-1)\sin\phi<0,
 \qquad
 \eta=\lambda(a_f-a_c)<0.
\]

The argument difference lies in `(-phi,phi)`, so this sign identifies
the actual continuous phase branch. Target one sees neighbors
`{0,2,10,19,37}`, all with zero incoming offsets. Its resultant is the
same positive real `3+2*cos(alpha)` on both inputs, and its phase increment
is exactly zero. Reading target zero after its update would change that
resultant; such serial execution is outside the simultaneous word.

Thus Coherence gives `delta_0,f^+=lambda*a_f` and
`delta_0,c^+=lambda*a_c`, holding every other phase and all forms.
Its stored-pressure contraction does not supply the subsequent constitutive
law. Silence holds these physical coordinates and gives the same capacities
on both branches:

\[
 D=\operatorname{diag}(\kappa,\kappa,1,\ldots,1),\qquad
 \Pi x_f^+-x_c^+=0,\qquad
 \Pi\delta_f^+-\delta_c^+=\eta e_0.
\]

The hidden vector remains `a=0,b=b*e_0` in the nine-slot notation;
none of these marked maps changes it. The core remains exactly symmetric.
This accounts for every primary coordinate, not only the target observation.

For either branch, only target zero changes phase. With `a=Arg Z_0`,
its exact storage jump is

\[
 \Delta H_{\rm EN}=0,\qquad
 \Delta H_{\rm IL}=|Z_0|[\cos a-\cos((1-\lambda)a)]<0,
 \qquad\Delta H_{\rm SHA}=0.
\]

There is no capacity term in H and no target-1 phase increment. The actual
shared edge is included in Z_0, so no joint work is omitted. The established
word passivity argument keeps the whole phase-reset comparison path inside
the acute sublevel. Both successors therefore retain all five windings
and `E<1/100`. The creation of predictive error is compatible with passive
work and structural retention.

### 9.3 A conserved error of the full post-word laws

Independently refresh each branch's complete dissipative law in the same
elapsed structural time `s>=0`, with the displayed D held and no further
events or external input:

\[
 x'=\tfrac14D[-Lx+\gamma S(\psi+\delta)],\qquad
 \delta'=\tfrac\gamma4DLx,\qquad \gamma=(1023\pi)^{-1}.
\]

All forms and phases continue to evolve. These globally Lipschitz rows
give global solutions. Their own balance `H'=-(Lx)^TD(Lx)/4<=0` and the
same barrier retain the acute energy domain. No frozen pressure or
constant projected environment replaces either row.

Let `rho_i=1/nu_i` and `R=sum_i rho_i=43+2/kappa`. Because L has zero
column sums and reciprocal sine edge currents cancel,

\[
 {d\over ds}\sum_i\rho_i x_i=0,\qquad
 {d\over ds}\sum_i\rho_i\delta_i=0.
\]

These are inverse-capacity weights, without a degree factor: this source's
rows use D/4, unlike the separate degree-normalized sine comparison.
The common fixed carrier has zero time derivative; both continuations
retain the same declared real offsets rather than rewrapping or centering
them at each reading.

Whole-child exchange preserves D, so `rho^T Pi=rho^T`. For the actual
error `e_theta(s)=Pi delta_f(s)-delta_c(s)` this proves

\[
 \sum_i\rho_i e_{\theta,i}(s)={\eta\over\kappa},\qquad
 \mu_D(e_\theta(s))
 := {\sum_i\rho_i e_{\theta,i}(s)\over R}
 ={\eta\over43\kappa+2}=:m_*\ne0.
\]

The individual post-word means are
`(2*kappa*phi+lambda*a_f)/(43*kappa+2)` and
`(2*kappa*phi+lambda*a_c)/(43*kappa+2)`. Forms have zero conserved
weighted means in both branches. Coherence creates the phase discrepancy;
Silence changes its averaging weights. No conservation across the reset
is assumed to obtain conservation along the continuations.

A positive weighted mean cannot exceed the maximum absolute component.
Consequently, for every common continuation time,

\[
 \boxed{\displaystyle
 \|\Pi\delta_f(s)-\delta_c(s)\|_\infty
 \ge {|\eta|\over43\kappa+2}>0,\qquad s\ge0.}
\]

In particular the raw projected tracking error cannot converge to zero.
The proof follows the full fine state and the separately evolved core;
it does not assume that projecting the fine nonlinear dynamics closes.
It needs neither a finite-time expansion nor a computed long trajectory.

A simple analytic floor makes the nonzero size explicit. Put
`zeta=a_c-a_f` in `(0,phi)`. The exact cross-resultant identity gives
`sin(zeta)=2*r*(1-cos(b))*sin(phi)/(|Z_f|*|Z_c|)`. Here `r>2`, both
moduli are below five, `1-cos(b)>b^2/4` and `sin(phi)>phi/2`.
Consequently `zeta>phi*b^2/50`. The actual default `lambda>1/4` and
`kappa<1` then yield

\[
 |m_*|>{\phi b^2\over9000}={2^{-47}\over9000}>2^{-61}.
\]

This floor follows from elementary inequalities and the fixed parameters,
not subtraction of binary64 arguments or fitting an observed discrepancy.
It is in declared ideal phase units, without a represented-runtime or
laboratory-resolution claim.

### 9.4 What a predictor must remember, and what remains open

The hidden relaxation theorem of section 7 remains applicable: the word
holds hidden coordinates and their unit capacities, while both successors
retain the required energy domain. Hidden phase differences therefore
decay under that theorem even though the displayed mean error cannot.
A decaying instantaneous projection defect is insufficient to erase
an error already transferred into a conserved collective coordinate.

For raw lifted-phase prediction, a reference must account for this scalar
event-induced mean displacement, either through its state/reset law or
through an explicit offset record. The exact orbit representation already
retains the hidden phase used by the resultant. A symmetric description
that discarded it cannot recover eta from the shared projection alone.
This is a necessity statement, not a proof that adding one scalar makes
all nonlinear or hybrid predictions sufficient.

Writing `e_theta=m_*1+q` with `mu_D(q)=0` separates the conserved part.
An observation invariant under a common phase shift does not see `m_*1`.
The lower bound therefore excludes zero-limit raw tracking, not relative
phase tracking modulo that shift. The mean calculation alone gives no
decay estimate for q. Qualitative asymptotic conclusions must be checked
against the existing positive-capacity convergence theorem in the
[source owner](ORGANIZED_SOURCE_PORT.md#21-silence-suffix-and-positive-capacity-continuation);
they would not by themselves give a finite-time tracking allowance. A
quantitative repaired prediction needs its own post-word capacity, stability,
initial-error and forcing bounds. No loss of winding, autonomous occurrence,
physical validation or
failure of the original unit-capacity no-event theorem follows.

### 9.5 Static controls and evidence boundary

The [phase-memory controls](../../tests/research/test_organized_word_phase_memory.py)
rebuild the actual support, projection, exact offset domain, registered
factors and both complete resultants. They check the simultaneous ideal
successors, work, strict nonzero argument defect, capacity-weighted
conservation and projection identity. Shared outward rational arithmetic
resolves the sign without discarding the small quadratic contribution.
Independent existing static word controls retain their original scope.

The frozen preparation, controls, reviews and preservation records are in
`artifacts/organized-word-phase-memory-20261023/`. No runtime word,
first-hit search, trajectory integration or retained response producer
is executed. Supplied history and event-domain context remain hypotheses
of the ideal action, rather than a claimed runtime receipt.

<a id="organized-source-postword-tracking"></a>
## 10. Quantitative prediction after the word with conserved means retained

After the ideal word, a symmetric reference can predict the fine state
with an explicitly bounded decaying error if its conserved mean offsets
are accounted for. The bound includes both the initial word defect and
the subsequent forcing from the hidden children. It uses the actual
nonuniform Silence capacities. This supplies a finite prediction allowance
beyond qualitative convergence, while retaining section 9's obstruction
to zero-limit raw tracking without the mean correction.

### 10.1 Actual post-word domain and the weighted error space

Keep the support, exact carrier, simultaneous ideal word and supplied
admission context of section 9. More generally, the argument below applies
to two independently admitted ideal successors on that same acute sublevel,
one full state z_f and one symmetric state z_c, with the same capacities

\[
 D=\operatorname{diag}(\kappa,\kappa,1,\ldots,1),\qquad 0<\kappa<1.
\]

Both must have `E<1/100` in the same winding-one lifted cell. Their complete
continuations are the section 9 shared-capacity dissipative rows, in the
same elapsed structural time, with D held and no further events or input.
Word occurrence and represented roundoff are outside these ideal premises.
For the fixed witness, all these properties follow from its exact
successors and passive work, rather than from a rounded phase preparation.

Set `M=D^-1`, `rho=M*1` and `R=1^T M*1=43+2/kappa`. Let

\[
 e_x=\Pi x_f-x_c,\quad e_\theta=\Pi\delta_f-\delta_c,\qquad
 m_x={\rho^Te_x\over R},\quad m_\theta={\rho^Te_\theta\over R},
 \qquad p=e_x-m_x\mathbf1,\quad q=e_\theta-m_\theta\mathbf1.
\]

The two means are constant by the actual full-row cancellation and
`rho^T Pi=rho^T`. Consequently `rho^T p=rho^T q=0`. These are weighted
zero-mean conditions; the ordinary sums of p and q need not vanish.
Adding the constant pair `(m_x*1,m_theta*1)` to the symmetric reference
also gives a solution of the same law: L annihilates a common form shift
and the sine field ignores a common phase shift. Thus p and q compare
the projected fine state with a well-defined corrected reference.

Every norm below uses the fine 45-node lift. In the 36-coordinate chart,
`||u||_M^2` has weights `orbit_mass/nu`; replacing them by unweighted
chart coordinates would change the statement.

### 10.2 Complete error rows and the retained hidden forcing

Each continuation retains its acute sublevel by the shared storage loss
and boundary barrier. Convexity and the global child symmetry place
`Pi z_f` in the same sublevel. The actual edge deviations from the carrier
are less than 1/5 in absolute value. The segment between projected fine
and core phases therefore stays in the section 7 collar, also after the
common-phase correction, which changes no edge angle. The sine secant
matrix obeys `L/2 <= K(t) <= L` as quadratic forms and `K(t)*1=0`.

Orient the ninety actual edges as in section 8 and write `L=B B^T`.
The projected-current defect has the exact form `Pi S_f-S(Pi Theta_f)=B j`.
On the two exchanged child-cycle copies its edge value is
`sin(bar theta_e)*(cos(b_k-b_l)-1)`; on their four matching-edge copies
per slot it is `sin(bar theta_e)*(cos(b_l)-1)`. Other edges contribute zero.
The actual edge multiplicities give

\[
 \|j(t)\|^2\le\tfrac12\sum_{C_9}(b_k-b_l)^4+\sum_l b_l^4
 \le5\|b(t)\|^4.
\]

This identity concerns the moving fine background. The changed target
capacities do not change those edge currents or their multiplicities.
Since D commutes with Pi and L commutes with Pi, the complete centered
error rows are

\[
 p'={D\over4}[-Lp-\gamma K(t)q+\gamma B j(t)],\qquad
 q'={\gamma D\over4}Lp.
\]

No derivative of an evolving projection is introduced, and no lossy
autonomous closure is assumed. Both constant mean components disappear
from these rows by the exact shift symmetries.

### 10.3 Weighted coercivity and the forced common metric

For any `rho^T u=0`, the minimizer of `sum rho_i*(u_i-c)^2` is c=0.
With `bar u` the ordinary mean, the actual full-support spectral gap
therefore gives

\[
 \|u\|_M^2
 =\min_c\sum_i\rho_i(u_i-c)^2
 \le{1\over\kappa}\|u-\bar u\mathbf1\|^2
 \le{15\over4\kappa}u^TLu.
\]

Thus `u^T L u >= (4*kappa/15)*||u||_M^2` on the correct weighted leaf.
The hidden nine-slot gap and the unmodified ordinary-mean estimate cannot
replace this proof.

Use the positive metric

\[
 W=p^TMp+q^TMq+2\gamma p^TMq,\qquad
 (1-\gamma)(\|p\|_M^2+\|q\|_M^2)
 \le W\le(1+\gamma)(\|p\|_M^2+\|q\|_M^2).
\]

D cancels against M when both complete rows are differentiated:

\[
 W'=-{1-\gamma^2\over2}p^TLp
     -{\gamma\over2}p^TKq-{\gamma^2\over2}q^TKq
     +{\gamma\over2}(p+\gamma q)^TBj.
\]

On the ordinary common-phase quotient, `K L^dagger K <= K`; L^dagger
denotes the Moore--Penrose inverse. L and K kill all constant parts, so
this quadratic-form inequality also applies to the weighted-centered
p and q. Young's inequality gives the homogeneous bound
`-p^T Lp/8-gamma^2*q^T Lq/8`, using `0<gamma<1/2` and `K>=L/2`.
For the forcing, separately bound

\[
 {\gamma\over2}|p^TBj|\le{p^TLp\over16}+\gamma^2\|j\|^2,
 \qquad
 {\gamma^2\over2}|q^TBj|\le{\gamma^2q^TLq\over16}
                              +\gamma^2\|j\|^2.
\]

The resulting inequality and weighted coercivity prove

\[
 W'\le-rW+10\gamma^2\|b(t)\|^4,\qquad
 r={\kappa\gamma^2\over60(1+\gamma)}>0.
\]

Neither D with L or K, nor the two stiffnesses with each other, must
commute. No stiffness derivative is needed. The capacity dependence
remains explicit; this rate is not uniform as kappa tends to zero.

### 10.4 Initial word error, finite decay and exact separation

The hidden children still have unit capacities. Their section 7 metric
W_h and the retained post-word domain therefore give

\[
 N(t)=\|a(t)\|^2+\|b(t)\|^2\le C_0e^{-r_ht},\qquad
 C_0={W_h(0)\over1-\gamma},\qquad
 r_h={\gamma^2\over4(1+\gamma)}.
\]

Starting with the actual post-word `W_0=W(0)`, integration of the forced
scalar inequality yields, for all t>=0,

\[
 W(t)\le W_0e^{-rt}
  +{10\gamma^2 C_0^2\over2r_h-r}
       (e^{-rt}-e^{-2r_ht}),\qquad 2r_h-r>0.
\]

The homogeneous term cannot be discarded: the word did not leave a
matched projected state. For the unchanged section 9 witness,

\[
 m_x=0,\quad m_\theta={\eta\over43\kappa+2},\quad
 p(0)=0,\quad
 q(0)=\eta\left(e_0-{\mathbf1\over43\kappa+2}\right),\qquad
 W_0={\eta^2(43\kappa+1)\over\kappa(43\kappa+2)}>0,
 \quad C_0={b^2\over1-\gamma}.
\]

Accordingly the forcing coefficient is
`600*(1+gamma)*b^4/((30-kappa)*(1-gamma)^2)` in this case. The weighted
relative joint squared error is at most `W(t)/(1-gamma)`; because the
maximum capacity is one, it also bounds the ordinary joint squared norm.

Weighted orthogonality gives exact, separate accounts of prediction and
reconstruction:

\[
 \begin{split}
 \|\Pi z_f-z_c\|_M^2
 &=\|p\|_M^2+\|q\|_M^2+R(m_x^2+m_\theta^2),\\
 \|z_f-z_c\|_M^2
 &=2N+\|p\|_M^2+\|q\|_M^2+R(m_x^2+m_\theta^2).
 \end{split}
\]

Joint norms here sum the form and phase-offset squared norms. The second
line uses the unit capacities on the paired children; their two fine
copies supply the factor two. These decompositions use M. An ordinary
Euclidean decomposition with `45*m_theta^2` would generally have a missing
cross term, since weighted centering need not give zero ordinary mean.
For the mean-corrected core the last term is absent, leaving the separate
relative allowance and hidden reconstruction contribution. For the raw
core section 9 still gives `||e_theta(t)||_infinity>2^-61` at all times.

### 10.5 Prospectively fixed finite discriminator

Use the same exact incoming source and actual ideal successors, without
new parameters or a trajectory. Let shared outward arithmetic give
`0<g_lo<=gamma<=g_hi<1/2`. Define exact rational comparison quantities

\[
 \begin{gathered}
 r_c={\kappa g_{\rm lo}^2\over60(1+g_{\rm hi})},\quad
 h_c={15r_c\over\kappa},\quad a_c=2h_c,\quad T={40\over r_c},\\
 C={b^2\over1-g_{\rm hi}},\quad F=10g_{\rm hi}^2C^2,\quad
 U_0=\eta_{\max}^2{43\kappa+1\over\kappa(43\kappa+2)}.
 \end{gathered}
\]

Here eta_max is an independently rebuilt outward upper bound on `|eta|`
from both actual resultants. It is not a fitted error tolerance. The
declared checks are

\[
 \begin{split}
 B_{\rm rel}(T)&=
 {U_0 e^{-r_cT}_{\rm upper}
   +F\,\mathcal C_{\rm upper}(r_c,a_c,T)\over1-g_{\rm hi}}<2^{-124},\\
 B_{\rm hidden}(T)&=2C e^{-h_cT}_{\rm upper}<2^{-124},\qquad
 B_{\rm corrected\ fine}(T)=B_{\rm rel}(T)+B_{\rm hidden}(T)<2^{-123}.
 \end{split}
\]

`C_upper(r_c,a_c,T)` bounds the integral of
`exp(-r_c*(T-s))*exp(-a_c*s)` over `[0,T]` using the
[shared decay convolution](../../src/tnfr/mathematics/_comparison_flow.py).
The shared rational negative-exponential owner supplies the associated
exponential enclosures. The actual factor has `kappa>1/2`, so the exponents
are `40`, `600/kappa` and `1200/kappa`, all below its work cap 4096.
That computational cap does not restrict the analytical theorem's time domain.

These targets were fixed before the new arithmetic control. A loose
independent budget explains them: `g_hi/g_lo<2`, `eta_max<b^2/4` and
`kappa>1/2` imply
`[U_0+F/(a_c-r_c)]/(1-g_hi)<1024*b^4=2^-70`. The positive Taylor
terms through degree eight give `exp(5)>128`, hence
`exp(-40)<2^-56`. Even this loose relative bound is below `2^-126`.
The hidden exponent exceeds 600 and easily gives its separate threshold.
No acceptance threshold is chosen from an evaluated discrepancy.

Scalar products, powers, exponentials and convolution budgets remain exact
Fractions after the needed outward transcendental endpoints are established.
Rounding all later products onto a fixed 128-bit absolute interval grid
would unnecessarily obscure allowances near or below that grid. No float
underflow or zero substituted for unavailable arithmetic supplies a result.

The finite joint relative norm is consequently below `2^-62`, less than
half the already proved raw phase-mean floor `2^-61`. This distinguishes
an accurately mean-corrected prediction from an uncorrected raw one on the
same admitted source. T is a conservative mathematical deadline in the
declared structural clock, not a measured or physically calibrated settling
time. The corrected fine bound additionally includes the hidden coordinates.

### 10.6 Controls, reuse and the remaining information obligation

The [post-word tracking controls](../../tests/research/test_organized_postword_tracking.py)
rebuild the exact ideal images and collars, weighted centering and orbit
masses, full-support coercivity, complete error/metric identities, edge
forcing and the nonzero initial metric. They evaluate both finite budgets
with shared pure arithmetic and retain the raw mean term separately.
The earlier phase-memory controls retain the word's admission and work
premises. The old unit-capacity matched-state tracking reader is unchanged;
it does not certify these nonunit-capacity, nonzero-error successors.

This is a conditional predictive contract with a specified hidden-state
envelope and mean jump. The shared projected incoming state alone still
cannot determine eta, as section 9 proves. A bound on error does not
acquire missing information, authenticate context, choose an event or
guarantee a discontinuous structural decision. Finite error and mean
availability remain separate obligations for any consuming predictor.
The result supplies neither a universal reduced TNFR state nor physical
validation or a represented-runtime roundoff guarantee.

The prospective discriminator, static controls, independent reviews and
preservation records belong to `artifacts/organized-postword-tracking-20261023/`.
No runtime word, first-hit search, flow integration, source search or retained
response producer is evaluated. Existing shared proof/arithmetic owners
suffice; no new production helper or execution path is introduced.
