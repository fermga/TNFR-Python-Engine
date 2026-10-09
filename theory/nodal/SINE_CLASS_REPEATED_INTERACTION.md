# Repeated interaction of joined acquired classes

<a id="sine-class-repeated-interaction"></a>

The [retained class comparison](SINE_CLASS_COMPARISON_PROTOCOL.md#sine-comparison-reserved-result)
encloses one nonlinear interaction under actual preparation and reading
uncertainty. This theorem gives a finite return condition for repeating the
same supplied word on the same joined system. Every branch keeps all 54
coordinates, its accumulated means and its own residual history. There is
no repeated acquisition, state reset or new response evaluation.

The proof reuses the strict modified-energy method of
[isolated-class maintenance](SINE_FORMED_CLASS_MAINTENANCE.md#a-strict-lyapunov-bound-on-each-retained-acute-chart),
with newly derived joined spectral constants and the
[degree-metric isometry](SINE_TWO_PORT_DIPOLE.md#sine-two-port-dipole).
The new forward return family is not identified with the original
formation image.

<a id="sine-repeated-complete-state"></a>
## Complete state, branch means and repeated observation

Keep the three C9 rings, contacts \((4,13),(13,22)\), unit capacities
and actual degree matrix \(D\). Write \(d=D\mathbf1\), so
\(d^T\mathbf1=58\), and retain
\[
 x'=-Ax+\gamma D^{-1}S(\theta),\qquad
 \theta'=\gamma Ax,\qquad A=D^{-1}L,\qquad
 \gamma=\frac1{1023\pi},\quad \tau=\frac{1023}{1024}t.
 \tag{1}
\]
The class targets \(\Theta_k\) have component windings \((1,k,1)\),
\(k=1,2\), and local lifts \(2\pi k_c(j-4)/9\). Their bridges
have zero phase gaps; their internal sine forces cancel. Common form
and common phase translations therefore give equilibria on every
conserved-mean leaf. All reference closing-edge integers remain fixed.

Let \(h=(A_h,B_h)\) range over \(00,10,01,11\). During each word,
apply form increments \(aA_h e_4\) at time zero and \(bB_h e_4\)
at time one, with
\[
 a=b=\frac7{10000},\quad T=2,\quad
 \epsilon=10^{-32},\quad \delta=10^{-30},\quad r=\frac1{12}.
 \tag{2}
\]
Read the receiver form at time \(T\), then leave that same full state
unforced for a common dwell \(W\) derived below. The period of the
supplied word is \(P=T+W\). The contact event occurs only once.
Events at every later word retain their complete preevent states.

For a full state define
\[
 \mu_x=\frac{d^Tx}{58},\qquad
 \mu_\theta=\frac{d^T(\theta-\Theta_k)}{58},\qquad
 P_D=I-\frac{\mathbf1d^T}{58},\qquad
 \bar x=P_Dx,\quad y=P_D(\theta-\Theta_k).
 \tag{3}
\]
These are proof coordinates, not operations performed on the state.
For Euclidean capture geometry use the different projection
\(P_0=I-\mathbf1\mathbf1^T/27\). Its squared full-state quotient
radius is \(\|P_0x\|_2^2+\|P_0(\theta-\Theta_k)\|_2^2\).
The degree-centered \(P_D\) is orthogonal only in the degree metric;
no Euclidean contraction property is attributed to it.
Both means are conserved between jumps. A donor jump \(q\) changes
\(\mu_x\) by \(3q/58\); it does not change \(\mu_\theta\).
Consequently, before word \(n\), counting the initial word as zero,
\[
 \mu_{x,k,h,n}=\mu_{x,k,0}+n\,m_h,
 \qquad m_h=\frac{3(aA_h+bB_h)}{58},
 \qquad \mu_{\theta,k,h,n}=\mu_{\theta,k,0}.
 \tag{4}
\]
The initial means can differ between the two acquired classes. They
are common to the four branches within each class only at acquisition;
(4) keeps all subsequent differences explicitly.

Use \(\sigma_{00}=\sigma_{11}=1\) and
\(\sigma_{10}=\sigma_{01}=-1\). The repeated statistic is
\[
 M_{k,n}=\sum_h\sigma_h x_{k,h,22}(nP+T),\qquad
 D_n=M_{1,n}-M_{2,n}.
 \tag{5}
\]
Both \(\sum_h\sigma_h\) and \(\sum_h\sigma_h m_h\) vanish.
Thus the accumulated common forms cancel exactly in (5), without
resetting them or treating the branch residuals as equal. Eight fresh
scalar reading errors contribute at most \(8\delta\) in each word.

<a id="sine-repeated-word-admission"></a>
## Recurrent source family, work and trapping

On each branch's own mean leaf define the new return set
\[
 \mathcal K_k=\{\|\bar x\|_D\le\epsilon,
                    \ \|y\|_D\le\epsilon\},
 \qquad \|z\|_D^2=z^TDz.
 \tag{6}
\]
Products of these sets permit different residuals in all branches.
The means retain (4). Since every degree is at least two, (6) implies
coordinate errors at most \(\epsilon/\sqrt2<\epsilon\).

The initial acquired family is handled separately. Its componentwise
zero-sum errors have Euclidean norms at most \(\epsilon\), as in the
[original source proof](SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-source).
Its joined means are bounded by \(4\epsilon/58\), so quotient
centering only gives a coordinate bound \(31\epsilon/29\), not
membership in (6). The already proved first-word work, trapping and
comparison apply to that original family. The first unforced dwell
will place it in (6). No stronger initial preparation is introduced.

For every subsequent word, \(L\preceq2D\) and criticality of
\(\Theta_k\) give the initial excess-storage bound
\[
 \mathcal H-\mathcal H_*\le
 \|\bar x\|_D^2+\|y\|_D^2\le2\epsilon^2,
 \quad
 \mathcal H=\tfrac12x^TLx+
       \sum_{ij}\{1-\cos(\theta_j-\theta_i)\}.
 \tag{7}
\]
For the phase term use Taylor's integral remainder and the global
upper Hessian bound \(\nabla^2U\preceq L\). Componentwise zero sums
are not assumed for the recurrent family. Common translations do
not change this storage.

Put \(g=1/3000>\gamma\). For a first impulse \(u\in\{0,a\}\)
and delayed impulse \(q\in\{0,b\}\), the
[carried-pressure estimate](SINE_CLASS_AMPLITUDE_FEASIBILITY.md#sine-amplitude-carried-work)
uses only the coordinate errors and the joined graph, and yields
\[
 Q_u=\frac{u+\epsilon+2g\epsilon}{1-2g^2},\qquad
 Z_u=\epsilon+2g\epsilon+2g^2Q_u,
\]
\[
 W_1\le\tfrac32u^2+6u\epsilon,\qquad
 W_2\le\tfrac32q^2+\tfrac98uq+6qZ_u.
 \tag{8}
\]
The exact delayed work remains \(q(Lx^-)_{4}+3q^2/2\).
Subtracting a constant initial form mean in a proof leaves that
Laplacian unchanged. It does not replace the delayed state by its
initial value. The spectral bound \((L\exp(-A))_{44}<9/8\) at
unit time includes the actual donor degree three.

Starting from (7), the weaker already admitted ceiling
\(22\epsilon^2+W_1+W_2\) is conservative. Likewise, for
\(A_*=u+q\), a representative of the whole word obeys
\[
 X=\frac{A_*+\epsilon+2gT\epsilon}{1-2g^2T^2},\qquad
 Y=\epsilon+2gTX,
 \qquad Z_{\rm quotient}^2\le27(X^2+Y^2).
 \tag{9}
\]
Arithmetic-mean projection can only decrease the Euclidean norm in
the last inequality. Exact rational substitution, inherited from
the [whole scale interval](SINE_CLASS_AMPLITUDE_FEASIBILITY.md#sine-amplitude-feasible-interval),
gives
\[
 W_1<7.36\,10^{-7},\quad W_2<1.287\,10^{-6},\quad
 \mathcal H-\mathcal H_*<2.022\,10^{-6}<1/388800,
 \quad Z_{\rm quotient}^2<5.293\,10^{-5}<r^2.
 \tag{10}
\]
Each actual probe meets its \(2\,10^{-6}\) work allowance. The
initial contact already met its separate \(10^{-12}\) allowance;
it is not installed or charged again.

The [joined acute barrier](SINE_REDUCED_PORT_COMPOSITION.md#actual-source-handoff-support-work-and-joined-identity)
exceeds \(r^2/2700=1/388800\). Together with (9), (10) traps every
postword state under uninterrupted relaxation in the same radius
\(r\) quotient chart. All relevant edge cosines exceed \(c=1/20\).
Indeed, each reference gap has magnitude at most \(4\pi/9\), and
its perturbation is at most \(\sqrt2r\). Using the certified
\(\pi>31/10\) and \(\sqrt2<17/12\), the remaining angle to
\(\pi/2\) exceeds \(13/240\). Hence the edge cosine is greater
than \(13/240-(13/240)^3/6>1/20\). The prescribed closing-edge
integers specify the same acute circular gaps throughout.
The events and flows throughout each word remain inside the admitted
identity domain. No lower estimate of continuous dissipation is
used as a work reserve.

<a id="sine-repeated-joined-lyapunov"></a>
## Joined spectral constants and strict nonlinear decay

The fine graph has diameter at most ten. For \(d^Tz=0\), weighted
bounded-range variance and a shortest path give
\[
 \|z\|_D^2\le\frac{58}{4}(\max z-\min z)^2
             \le145\,z^TLz.
 \tag{11}
\]
The normalized Laplacian is self-adjoint in the degree metric.
On its mean-free subspace its spectrum therefore lies in
\([\lambda,\Lambda]=[1/145,2]\). The upper bound follows from
\(\sum_{ij}(z_i-z_j)^2\le2\sum_i d_i z_i^2\).

Set \(\widehat A=D^{-1/2}LD^{-1/2}\),
\(\widehat x=D^{1/2}\bar x\),
\(\widehat y=D^{1/2}y\), and remove the common null vector.
The auxiliary coordinates and potential
\[
 \xi=\widehat A^{-1/2}\widehat y,\qquad
 v=\gamma\widehat A^{1/2}\widehat x,\qquad
 W_k(\xi)=\gamma^2\left[
 U(\Theta_k+\mu_\theta\mathbf1+
       D^{-1/2}\widehat A^{1/2}\xi)-U(\Theta_k)\right]
 \tag{12}
\]
obey exactly \(\xi'=v\),
\(v'=-\widehat A v-\nabla W_k(\xi)\).
The full phase and form rows are both used. This is a degree-metric
isometry, not a change in the supplied law or clock.

On the trapped chart, \(cL\preceq\nabla^2U\preceq L\).
The line from the target to any admitted state stays in that chart.
Congruence by \(D^{-1/2}\widehat A^{1/2}\), without any
commutation premise, bounds the transformed Hessian by
\[
 \mu I\preceq\nabla^2W_k\preceq M I,\qquad
 \eta_- =\frac1{11000000}<\gamma^2<\eta_+=\frac1{9000000},
 \quad \mu=\eta_-c\lambda^2,\quad M=\eta_+\Lambda^2.
 \tag{13}
\]
These gamma bounds follow from the same rational pi bounds as the
selected complete law; no fitted rate is substituted.

With \(\beta=\lambda/4\), define
\[
 V=\tfrac12\|v\|^2+W_k+
       \beta\xi^Tv+\tfrac\beta2\xi^T\widehat A\xi.
\]
The shared modified-energy calculation gives
\[
 \begin{split}
 a_-&=\mu/2+\lambda^2/16,\qquad
 a_+=M/2+\beta\Lambda/2+\beta^2,\\
 \tfrac14\|v\|^2+a_-\|\xi\|^2
 &\le V\le\tfrac34\|v\|^2+a_+\|\xi\|^2,\\
 V'&\le-(\lambda-\beta)\|v\|^2-
                   \beta\mu\|\xi\|^2\le-\kappa V,\\
 \kappa&=\min\{4(\lambda-\beta)/3,\beta\mu/a_+\}.
 \end{split}
 \tag{14}
\]
The cross terms cancel by the complete second row in (12).
Exact rational constants are
\[
 \mu=\frac1{4625500000000},\quad
 a_+=\frac{6537091}{3784500000},\quad
 a_->\frac1{336400},\quad
 \kappa=\frac9{41706640580000}>\frac1{5\,10^{12}}.
 \tag{15}
\]
The same constants hold for both classes and every branch.

<a id="sine-repeated-exact-return"></a>
## One common finite dwell without an interval-grid floor

At the end of any admitted word the Euclidean quotient radius is
less than \(r\). Weighted orthogonal projection minimizes the
degree norm over common translations. Since the maximum degree is
four, for either residual vector \(z\),
\[
 \|P_Dz\|_D\le\|P_0z\|_D\le2\|P_0z\|_2,
 \qquad
 \|P_0z\|_2\le\|P_Dz\|_2\le\|P_Dz\|_D/\sqrt2.
\]
Thus both \(\|\bar x\|_D\) and \(\|y\|_D\) are at most
\(2r=1/6\), and a common postword energy upper bound is
\[
 V_0=\frac34\eta_+\Lambda\frac1{36}
          +\frac{a_+}{\lambda}\frac1{36}
     =\frac{130741907}{18792000000}<\frac1{100}.
 \tag{16}
\]
Fix the structural dwell and exact decay bound
\[
 W=1280000000000000=256(5\,10^{12}),\qquad
 \rho=2^{-256}.
 \tag{17}
\]
Since \(\kappa W>256\) and \(\exp(1)>2\),
\(\exp(-\kappa W)<\rho\). Keep \(\rho\) as an exact rational
number. Materializing it first on a dyadic128 grid would destroy
the required squared-norm resolution; no such operation is used.

Returning to the original degree norms gives
\[
 \begin{split}
 \|\bar x(T+W)\|_D^2
 &\le\frac{4V_0\rho}{\eta_-\lambda}
       <63800000\,2^{-256}<\epsilon^2/4,\\
 \|y(T+W)\|_D^2
 &\le\frac{\Lambda V_0\rho}{a_-}
       <6728\,2^{-256}<\epsilon^2/4.
 \end{split}
 \tag{18}
\]
The last two strict comparisons are integer inequalities, not
floating exponentials. Their respective upper bounds are below
\(5.51\,10^{-70}\) and \(5.82\,10^{-74}\).
The dwell in the original clock is \(1024W/1023\); this very large
conservative time is not a laboratory feasibility claim.

The original first word is trapped by its own already proved
certificate, so (18) places its actual endpoints in
\(\tfrac12\mathcal K_k\). Thereafter (7)-(10) and (18) yield
\[
 \Phi_k^W\mathcal W_{k,h}(\mathcal K_k)
       \subset\tfrac12\mathcal K_k\subset\mathcal K_k,
 \tag{19}
\]
on each successively shifted mean leaf. Here \(\mathcal W_{k,h}\)
is the complete two-event word of duration two. Induction retains
every actual branch for arbitrarily many supplied repetitions.
It does not require equality of branch residuals, an exact equilibrium
return or membership in the old formation image after the first word.

<a id="sine-repeated-response-and-comparator"></a>
## Recurrent response and the comparator's own memory allowance

For any word beginning in (6), compare each branch separately with
its exact nominal reference, translated by that branch's initial
common form and phase. Heat contraction and the real sine Lipschitz
bound give original-coordinate errors at most
\(\epsilon/(1-2gT)\) through the word. Identical jumps do not
change these errors. Formula (4) cancels the reference translations
in (5), even though the eight residuals are different. Hence
\[
 D_n\in I_{\rm ref}+[-B_N,B_N],\qquad
 B_N=\frac{8\epsilon}{1-4g},\qquad n\ge1.
 \tag{20}
\]
The exact retained nominal reference interval is
\[
 I_{\rm ref}=
 \frac{[-6549026775,-6548966240]}{2^{128}}.
 \tag{21}
\]
It is inherited from the independently integrated full nonlinear
reference, not from a new coefficient or response evaluation. The
[retained comparison and its numerical premises](SINE_CLASS_COMPARISON_PROTOCOL.md#sine-comparison-reserved-result)
remain explicit evidence dependencies. Its own target-rounding and
numerical errors are already contained in (21).

Name the alternative before applying a null threshold: a fixed full
tangent law about each joined class target, with the same graph,
gamma, clock and repeated donor events. Its phase Hessian is the
fixed cosine-weighted Laplacian at that target. Admit independent
branch residuals in (6), with the compatible mean progression (4),
at the beginning of its repeated protocol. This family contains the
exactly correlated tangent histories from one shared initial state in
(6); it does not assert that every original acquired source lies in (6).

The tangent model retains all 54 coordinates. Its quadratic potential
has the Hessian bounds (13) globally. Explicitly, in its target-phase
coordinates it obeys
\(x'=-Ax-\gamma C_k y,\ y'=\gamma Ax\), where
\(C_k=D^{-1}\nabla^2U(\Theta_k)\); its weighted quadratic form
satisfies \(cA\preceq_D C_k\preceq_D A\).
Its normalized phase operator
has maximum-norm bound two, so the same word envelopes (9) hold.
In particular, the postword degree norms are bounded by
\(\sqrt{58}X\) and \(\sqrt{58}Y\), both less than \(1/6\).
Weighted centering can only reduce these degree norms. Equations (16)-(19) then
prove its own common finite return. No nonlinear trapping theorem,
reset to an actual nonlinear state or assumed observation zero is
used to supply that comparator continuation.

For its nominal histories, linearity makes the mixed response zero.
Independent returned residuals supply instead the necessary allowance
\[
 |D_n^{\rm lin}|\le B_L=\frac{8\epsilon}{1-4g}.
 \tag{22}
\]
If the tangent alternative starts from one common source and carries
the four complete repeated schedules, exact linearity preserves their
additive correlation and its mixed response is exactly zero. That
narrower fact does not justify replacing (22) by zero on the larger
independent-residual comparator family used here.
The correlated case also covers an original acquired common source
outside (6), by linearity alone; it does not need a new source preparation.

The mean condition is necessary. For example, add the constant form
one only to class one's \(00\) branch and leave every phase and all
other branches unchanged. Every quotient residual, work increment
and return bound is unchanged, but the class contrast acquires the
constant one. Thus quotient balls alone cannot bound (5); the actual
mean progression and its mixed cancellation must be supplied and retained.

The nonlinear recorded band adds its \(8\delta\) allowance to
(20); the independent alternative's recorded band is
\([-B_L-8\delta,B_L+8\delta]\). Exact rational comparison gives
\[
 -\sup I_{\rm ref}-B_N-B_L-16\delta
   >3.085468428957\,10^{-30}>0.
 \tag{23}
\]
Thus every repeated word separates the two eight-record sets through
the fixed statistic. Both source-memory allowances appear explicitly.
The original first word has the same \(B_N\) allowance from its
original source proof, so the comparison also covers \(n=0\) without
asserting initial membership of the acquired family in (6).
This conclusion is uniform in the number of repetitions, not an
assertion that the branch states or their individual readings recur.

<a id="sine-repeated-scope"></a>
## Scope and continuing costs

The first word retains its original acquired-source result. Later
words use the new proved return family and its independent branch
errors. Absolute means are retained by (4); they can grow without
bound while the relative storage and geometry stay trapped. No
periodic full state or unique periodic quotient orbit is claimed.

Every positive impulse is an externally supplied event. Over finitely
many words its cumulative work is the sum of its actual preevent work;
the uniform per-event allowances do not supply a finite reservoir for
indefinite repetition. Continuous dissipation is not used to pay a
jump. The chosen support, event schedule, long dwell, tiny source and
reading allowances and the complete constitutive law remain premises.

This is a conditional repeatability theorem for an interaction property
of organized patterns. It is not a new acquisition theorem, an autonomous
intervention selector, an independently calibrated physical measurement
or identification of a fundamental-particle property. No frozen evidence
is changed and no new scientific trajectory is evaluated.

## Implementation and retained-evidence boundary

The [private proof calculator](../../src/tnfr/physics/_sine_class_repeated_interaction.py)
accepts only a supplied nominal contrast interval. It rebuilds the
complete graph, weighted spectral certificates, recurrent work and
radius bounds, the rational Lyapunov return and both residual/noise
allowances. It reuses shared scalar admission, coordinate envelopes
and the exact modified-energy kernels. The supplied interval is a
conditional premise; scalar admission does not establish its scientific
provenance, entry into the return family or mixed-mean compatibility.
There is no new SDK execution path or solver.

The [independent controls](../../tests/physics/test_sine_class_repeated_interaction.py)
check the actual graph and metric, carried event work, return arithmetic,
mean counterexample and exact error margins. Their retained-response
association uses the existing
[read-only comparison audit](../../tests/physics/test_sine_class_comparison_evidence.py)
to reconstruct (21) from frozen complete histories. Stored derivative
and Picard generation remain explicit numerical premises; this
arithmetic audit neither regenerates them nor evaluates a new response.
