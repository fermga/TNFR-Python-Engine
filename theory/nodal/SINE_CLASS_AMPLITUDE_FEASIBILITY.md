# Amplitude feasibility for mediator-organization discrimination

<a id="sine-class-amplitude-feasibility"></a>

The [central cubic response](SINE_CLASS_CUBIC_RESPONSE.md#sine-cubic-finite-result)
has a resolved organization-dependent sign, while its original
recording allowance admits scalar cancellation. Combining it with the
assessed spatial statistic does not remove that
[joint recording limitation](SINE_CLASS_SPATIAL_OBSERVATION.md#sine-spatial-joint-cancellation).
This owner changes only a common positive scale of the two existing
donor form impulses. It proves a nonempty scale interval satisfying
response, noise, work and identity requirements simultaneously.

The argument uses the retained coefficient through exact homogeneity
and a derived delayed-work estimate. It evaluates no new coefficient,
source, nonlinear trajectory or response sweep. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns selection of a later intervention or evaluation.

<a id="sine-amplitude-model-and-scaling"></a>
## Fixed model and one changed input

Retain the complete three-C9 law
\[
 x'=-Ax+\gamma KS(\theta),\qquad \theta'=\gamma Ax,\qquad
 A=D^{-1}L,\quad \gamma=\frac1{1023\pi},\quad
 \tau=et,\quad e=\frac{1023}{1024}.
 \tag{1}
\]
Here \(D\) is the actual degree matrix, \(K=D^{-1}\), the contacts
are \((4,13)\), \((13,22)\), and every capacity is one. All
54 coordinates, both classes \((1,1,1)\), \((1,2,1)\), their
acquired full source families and independent residuals remain as in
the [source owner](SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-source).
In particular, each original component has separate zero-sum form and
target-phase deviations with Euclidean norms at most
\(\epsilon=10^{-32}\). The four histories within each class share
one actual source; the two classes need not share their residuals.

The only variable is a common positive amplitude scale:
\[
 a_0=b_0=\frac1{2000},\qquad
 a=b=\lambda a_0,\quad \lambda>0,\qquad
 s=1,\quad T=2.
 \tag{2}
\]
The form impulses retain their original donor node \(4\) and event
times. All phase, mediator and receiver coordinates carry; no state
is reset. Retain the eight final receiver-center readings, their
separate errors \(\delta=10^{-30}\), radius \(r=1/12\), contact
allowance \(10^{-12}\) and each probe-work allowance
\(2\,10^{-6}\).

Let \(D_{\mathrm c}(\lambda)\) be the class-one-minus-class-two
difference of the two four-history mixed central responses. Its
eight scalar errors give
\[
 |\widehat D_{\mathrm c}(\lambda)-D_{\mathrm c}(\lambda)|
 \le8\delta.
 \tag{3}
\]
The separately noisy eight-record zero-contrast alternative has its
own \(8\delta\) allowance. Consequently a true oriented contrast
strictly greater than \(16\delta\) separates these statistic images;
the weaker \(8\delta\) condition guarantees only the recorded sign.
Neither threshold is changed by scaling the input.

<a id="sine-amplitude-homogeneous-coefficient"></a>
## Homogeneous coefficient and complete finite error

The complete amplitude hierarchy is triangular: its first level is
linear in the supplied impulses, its second forcing is quadratic in
level one, and its third forcing consists of a level-one/level-two
product and a level-one cube. The second event changes only level one
and carries every previous coefficient. Induction first on amplitude
degree and then across that event proves
\[
 (x_n,y_n)(\lambda a_0,\lambda b_0;t)
   =\lambda^n(x_n,y_n)(a_0,b_0;t),\qquad n=1,2,3.
 \tag{4}
\]
This holds separately for all four histories and both classes.
It retains full phase feedback and the quadratic internal recoupling.
It is not a homogeneity assumption on the nonlinear trajectory or
on its arbitrary actual residuals.

The previously enclosed complete cubic contrast has the conservative
retained outer interval
\[
 J_0=[-7.013764940694,-7.013764940692]\,10^{-30}.
 \tag{5}
\]
It includes the complete coefficient, its time truncation and outward
arithmetic. Therefore \(\lambda^3J_0\) encloses the scaled complete
cubic contrast without a new coefficient calculation. Scaling this
whole enclosure also scales its retained coefficient error; it does
not assert that a fresh numerical execution would have identical
rounding widths.

Use the fixed rational upper bound \(g=1/3000>\gamma\). The
[complex-amplitude theorem](SINE_CLASS_CUBIC_RESPONSE.md#sine-cubic-amplitude-tail)
gives
\[
 R_5(A_*,T)
 =\frac{256g^6A_*^5T}
 {(1-2g^2T^2)(1-4g^2A_*^2)}
 \tag{6}
\]
when \(16g^2T^2<1\) and \(2gA_*<1\). The zero history has zero
nominal remainder. Define
\[
 E(\lambda)=
 2\{R_5(2\lambda a_0,T)+2R_5(\lambda a_0,T)\}
 +\frac{8\epsilon}{1-2gT}.
 \tag{7}
\]
The first factor of two accounts for the two classes; the inner
factor accounts for the two separate single-impulse histories.
The source allowance is unchanged by \(\lambda\): real sine is
globally Lipschitz in phase, and heat contraction compares each
actual history with its matching nominal history under identical
scaled jumps. Thus
\[
 D_{\mathrm c}(\lambda)
 \in \lambda^3J_0+[-E(\lambda),E(\lambda)].
 \tag{8}
\]
The coefficient-generating execution remains a premise of the
retained evidence. Admitting an arbitrary supplied numerical pair
as a calculator input cannot prove that it encloses this coefficient.

For an interval \(0<\lambda_-\le\lambda\le\lambda_+\) in the
domain of (6), \(E\) is increasing. If
\(J_0=[j_-,j_+]\) with \(j_+<0\), a uniform enclosure is
\[
 D_{\mathrm c}(\lambda)\in
 [\lambda_+^3j_- -E(\lambda_+),\
  \lambda_-^3j_+ +E(\lambda_+)].
 \tag{9}
\]
In particular,
\(-\lambda_-^3j_+-E(\lambda_+)>16\delta\)
is a sufficient whole-interval separation condition. It is stronger
than testing a few scales or scaling the earlier full-error interval
as though every error were cubic.

<a id="sine-amplitude-carried-work"></a>
## A sharper bound on the actual delayed work

The complete storage is
\(\mathcal H=\tfrac12x^{\mathsf T}Lx+
\sum_{ij}(1-\cos(\theta_j-\theta_i))\). Its continuous derivative
is nonpositive. At a donor form jump \(q\), however, the exact work is
\[
 W(q;x^-)=q(Lx^-)_4+\frac32q^2.
 \tag{10}
\]
The first term is the full preevent donor Laplacian, not \(3qx_4^-\),
and the delayed event must use the actual carried history.

A spectral estimate bounds its heat contribution without evaluating
a heat trajectory. Set \(P(t)=\exp(-At)\). The similarity
\(D^{1/2}AD^{-1/2}=D^{-1/2}LD^{-1/2}\) is symmetric positive
semidefinite with spectrum in \([0,2]\). For each normalized
coordinate vector, the diagonal of \(A\exp(-A)\) is therefore a
probability-weighted mean of \(\mu e^{-\mu}\), which lies in
\([0,1/\exp(1)]\). Since the donor degree is three,
\[
 0\le (LP(1))_{44}
    =3(A\exp(-A))_{44}
    \le\frac3{\exp(1)}<\frac98.
 \tag{11}
\]
The last strict inequality follows from
\(\exp(1)>1+1+1/2+1/6=8/3\). The exponential constant here is
unrelated to the model's weight \(e=1023/1024\).

Suppose the first impulse is \(u\ge0\), including \(u=0\)
for a control. Over \(0\le t\le1\), define
\[
 Q_u=\frac{u+\epsilon+2g\epsilon}{1-2g^2},\qquad
 Z_u=\epsilon+2g\epsilon+2g^2Q_u.
 \tag{12}
\]
Diffusion contraction and the complete phase row give
\(\|x(t)\|_\infty\le Q_u\) and
\(\|y(t)\|_\infty\le\epsilon+2gtQ_u\).
Relative to the ideal positive heat impulse \(uP(t)e_4\), the
full actual form at time one satisfies
\[
 \|x(1^-)-uP(1)e_4\|_\infty
 \le\epsilon+2g\int_0^1(\epsilon+2gtQ_u)\,dt=Z_u.
 \tag{13}
\]
The donor row of \(L\) has absolute row sum six. Combining (11)
and (13) proves the complete carried-pressure enclosure
\[
 (Lx(1^-))_4\in[-6Z_u,\ (9/8)u+6Z_u].
 \tag{14}
\]
This explicitly retains source errors and sine feedback. It is not
a pure-diffusion replacement for the original law.

For a positive delayed impulse \(q\), (10) now gives
\[
 W_1(u)\le\tfrac32u^2+6u\epsilon,\qquad
 W_2(u,q)\le\tfrac32q^2+\tfrac98uq+6qZ_u.
 \tag{15}
\]
The second-only history uses \(u=0\), not the probed preevent
state. These upper bounds are monotone in \(u,q\ge0\), so the
both-probe history at \(\lambda_+\) dominates all history/event
upper bounds throughout the claimed interval.

<a id="sine-amplitude-identity"></a>
## Storage, identity and conserved means

Contact work remains in \([0,8\epsilon^2]\), and the initial
joined excess storage is at most \(22\epsilon^2\). Without
crediting any amount of continuous dissipation, every history has
the excess ceiling
\[
 \mathcal E\le22\epsilon^2+W_1(u)+W_2(u,q).
 \tag{16}
\]
This bounds the first event as well as the second because the used
work upper bounds are nonnegative. Storage can only decrease between
jumps; no amount of continuous loss is credited as an event reserve.

For \(A_*=u+q\), use the original full-history representative bounds
\[
 X=\frac{A_*+\epsilon+2gT\epsilon}{1-2g^2T^2},
 \qquad Y=\epsilon+2gTX,\qquad
 Z_{\mathrm{radius}}^2\le27(X^2+Y^2).
 \tag{17}
\]
The quotient radius is no larger than this global Euclidean
representative bound. The
[matched capture theorem](SINE_CLASS_NONLINEAR_SUPERPOSITION.md#sine-superposition-events-and-identity)
has barrier strictly greater than \(r^2/2700\).
Its sufficient guards remain
\[
 \mathcal E<r^2/2700,\qquad 27(X^2+Y^2)<r^2.
 \tag{18}
\]
Both must hold; a work allowance or small total amplitude alone
does not certify identity.

The graph's degree mass is 58. Initially the separate component
zero sums give joined form and phase means bounded in magnitude by
\(4\epsilon/58\). A donor jump \(u\) changes the weighted form
mean by \(3u/58\), and a later jump \(q\) changes it by \(3q/58\).
Phase means are unchanged. The continuous law preserves both
weighted means between events. Each history consequently retains
its own mean leaf; none is reset to a common postevent mean for
comparison. Under (18), the cycle identities and unforced recovery
after the last event retain their original conditional meaning.

<a id="sine-amplitude-feasible-interval"></a>
## A certified nonempty scale interval

Take the complete interval
\[
 \boxed{\quad \frac43\le\lambda\le\frac75\quad}.
 \tag{19}
\]
Use \(\lambda_+\) in every increasing error, work and radius
majorant, and \(\lambda_-\) in the oriented cubic lower bound.
This is a uniform proof, not interpolation between sampled responses.
The largest per-event amplitude is \(7/10000\), and the largest
history amplitude is \(7/5000\). Hence the complex-amplitude and
real comparison denominators are strictly positive throughout.

Direct exact rational substitution in (6)-(9) gives
\[
 \begin{aligned}
 2\{R_5(2\lambda_+a_0,2)+2R_5(\lambda_+a_0,2)\}
       &<8.027\,10^{-33},\\
 \frac{8\epsilon}{1-4g}&<8.011\,10^{-32},\\
 E(\lambda_+)&<8.814\,10^{-32},\\
 -19.334\,10^{-30}<D_{\mathrm c}(\lambda)
       &<-16.537\,10^{-30}.
 \end{aligned}
 \tag{20}
\]
Thus the whole recorded interval lies below
\(-8.537\,10^{-30}\), whereas the separately noisy
zero-contrast alternative has statistic in
\([-8,8]\,10^{-30}\). Their separation margin is greater than
\(5.37\,10^{-31}\), uniformly on (19). In particular the recorded
sign is guaranteed under every allowed eight-error vector. This
excludes that named alternative's record set through the statistic;
it does not select a unique law or identify a physical interaction.

The same exact substitution in (12)-(18) gives, for every history
and both classes,
\[
 \begin{aligned}
 W_1&<7.36\,10^{-7}<2\,10^{-6},\\
 W_2&<1.287\,10^{-6}<2\,10^{-6},\\
 \mathcal E&<2.022\,10^{-6}<1/388800=r^2/2700,\\
 27(X^2+Y^2)&<5.293\,10^{-5}<1/144=r^2.
 \end{aligned}
 \tag{21}
\]
The contact ceiling \(8\epsilon^2=8\,10^{-64}\) is below its
unchanged allowance \(10^{-12}\). The delayed work improvement
uses the full donor Laplacian and the source/full-law correction.
No storage loss, initial precision, observation accuracy or work
ceiling has been changed to obtain these inequalities.

The result admits an interval of interventions mathematically.
It does not choose or execute a new reserved response, claim that
all amplitudes preserve identity outside this interval, or establish
laboratory preparation, clock or noise feasibility. The original
unit-scale cancellation results remain valid for their unchanged
protocols.

<a id="sine-amplitude-implementation-and-evidence"></a>
## Implementation and evidence boundary

The [conditional calculator](../../src/tnfr/physics/relational_sine_class_amplitude_feasibility.py)
admits a positive scale interval and a supplied base cubic interval.
It reconstructs homogeneous response bounds, higher-amplitude and
source allowances, carried work, identity guards and mean intervals.
The upper-scale work ledger is explicitly an endpoint majorant;
its individual mean shifts must not be substituted for the separately
reported mean intervals covering all scales.

The supplied coefficient endpoints are a mathematical premise, not
an authenticated observation or a certificate generated by this API.
For (5), that premise is supplied by the
[retained cubic calculation](SINE_CLASS_CUBIC_RESPONSE.md#sine-cubic-finite-result)
and its [read-only evidence audit](../../tests/physics/test_sine_class_cubic_evidence.py).
The [independent amplitude controls](../../tests/physics/test_sine_class_amplitude_feasibility_algebra.py)
check homogeneity, spectral pressure, complete error propagation and
the whole interval with exact arithmetic. Failed sufficient guards
report that the method has not certified the requested regime;
they do not prove dynamical infeasibility.

No new coefficient or response archive is needed for these deductions.
The existing coefficient evidence is reused without modification or
duplication, and its generation remains an explicit execution premise.
Any later changed-input observation requires its own fixed source,
event, numerical and recording protocol before evaluation.
