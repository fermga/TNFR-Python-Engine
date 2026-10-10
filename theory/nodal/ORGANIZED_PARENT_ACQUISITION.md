# Parent registration acquired from organized children

<a id="organized-parent-acquisition"></a>

## Fixed analytic question and source

This admission tests whether the matching-contact 36-node law in
[repeated composition](REPEATED_ORGANIZED_COMPOSITION.md) can acquire its
declared parent identity from children whose relative registration initially
lacks that identity. The source, clock, deadlines and readout below are fixed
before assessing the analytic bounds. No numerical trajectory or reserved
response is evaluated by this declaration.

| Item | Fixed choice |
| --- | --- |
| Complete law and support | The matching control of the repeated-composition law: four marked C9s, unit cycle and intraparent edges, matching outer edges, weighted degree four, held unit capacities |
| Clock | `tau=(1023/1024)*t`, `gamma=1/(1023*pi)`; slow clock `s=gamma^2*tau/2` transforms both evolution rows |
| Original children | Four independently admitted isolated class-one C9 source families, each with the original form and phase allowances `1e-10`, including all their fine directions and the unchanged fixed recovery dwell |
| Recovered uncertainty | The original recovery certificate must bound each child's actual form and phase residual Euclidean norms by `1e-70`; no endpoint is reset to its nominal pattern |
| Supplied child origins | `phi[g,a]=(2*g-1)/40+(2*a-1)/4`; the nominal intraparent gap is `1/2`, twice the identity threshold, and the nominal outer gap is `1/20` |
| Source means | Each child's form mean is zero; its phase residual about its supplied origin and winding carrier has mean zero |
| Attachment | Both contact layers are supplied at release; their actual storage work is retained separately from the preceding isolated recovery |
| Acquisition deadline | `s=4`, four slow relaxation units after attachment |
| Entry domain | Both form and lifted-phase residuals about each actual parent mean are strictly below `1/100` in coordinate sup norm |
| Identity | Every child retains winding one from release through `s=5`; every intraparent same-slot gap is strictly below `1/4` throughout `4<=s<=5` |
| Predictive handoff | Continue from the actual parent means and residuals at `s=4`; do not reset forms or restore the original outer phase gap |
| Readout and resolution | Predict the parent lifted-phase-gap change from `s=4` to `s=5` with absolute error below `1e-5`; require a decrease greater than `1e-4` for the entire original source family |
| Decision | Source-inclusive entry, retained identity and resolved predictive handoff, or an explicit scoped obstruction; a failed sufficient inequality is unavailable |

The half-radian intraparent source expresses missing registration, independently
of its eventual response. Four relaxation units and the subsequent single-unit
readout are fixed analytic horizons, not values chosen by a trajectory search.
Relative origins, contact occurrence and their work remain supplied premises.
These clocks and observations are structural quantities without a laboratory
measurement bridge.

An exact antipodal child-registration control may test nonentry under the same
complete law. Such an invariant control concerns its explicitly restricted
source; it cannot replace the four independently uncertain acquired families
or establish nonentry for a neighborhood without a separate proof.

The declaration above was committed as `da12bdb6` before evaluating its
certificate. The following derivation concerns that unchanged analytic
question. It uses no source refinement, contact scan or trajectory replay.

## 1. Complete law, state and the two moving modes

Retain all 36 signed forms and primitive phases, with continuous phase lifts
on the marked nodes `18*g+9*a+j`. Let `alpha=2*pi/9` and
`psi[j]=alpha*(j-4)`. The graph is the Cartesian product
\(C_9\mathbin\square K_2\mathbin\square K_2\), with the fixed unit
conductances specified above. Its unnormalized Laplacian is \(L\). The
complete rows and storage are exactly those of the matching control:

\[
 x_\tau=\frac{-Lx+\gamma S(\theta)}4,\qquad
 \theta_\tau=\frac{\gamma Lx}4,\qquad
 S_i=\sum_{k\sim i}\sin(\theta_k-\theta_i),
 \tag{1}
\]
\[
 H=\tfrac12x^TLx+\sum_{\{i,k\}}[1-\cos(\theta_k-\theta_i)],\qquad
 H_\tau=-\tfrac14\|Lx\|_2^2.
 \tag{2}
\]

The first row independently supplies the pressure for the held unit capacity.
Support, the two child/parent markings, capacity and structural clock remain
supplied premises. There is no additional input after attachment. All phases
and both global means remain part of the law; a circular resultant is not
substituted for the lifted phase observations.

Define a moving reference, not a replacement of the recovered source:

\[
 \begin{split}
 x^*_{g,a,j}&=\tfrac12[(2g-1)u+(2a-1)v],\\
 \theta^*_{g,a,j}&=\psi_j+
                 \tfrac12[(2g-1)\Delta+(2a-1)B].
 \end{split}\tag{3}
\]

The two coordinate directions separate under the actual matching contacts.
Internal cycle currents cancel, intraparent currents depend on \(B\), and
outer currents depend on \(\Delta\). Consequently, both pairs
\((w,D)=(v,B)\) and \((u,\Delta)\) obey

\[
 w_\tau=-\tfrac12w-\tfrac\gamma2\sin D,\qquad
 D_\tau=\tfrac\gamma2w.
 \tag{4}
\]

This follows by substituting into both fine rows, so smooth uniqueness
preserves the separated family. In the slow clock, put
\(\varepsilon=\gamma^2\) and \(Y=-w/\gamma\). Both rows transform to

\[
 D_s=-Y,\qquad \varepsilon Y_s=\sin D-Y,
 \qquad (D(0),Y(0))=(d,0),
 \quad d\in\{1/2,1/20\}.
 \tag{5}
\]

The form coordinate is retained as \(Y\); replacing the second row by
\(Y=\sin D\) would be a different, approximate law.

## 2. Source admission and full fine-state error

Rebuild four independent copies of the
[isolated own-law recovery](ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-carried-handoff)
at deformation zero from the original `1e-10` allowances. Each carries its
actual state after the unchanged formation and recovery times, with
\(\|r_c(0)\|_2,\|z_c(0)\|_2\le q=10^{-70}\), winding one and both
residual means zero. Translating the four disconnected phase origins does
not change their internal storage or dynamics. These translations supply
new preparation information and do not assert autonomous registration.

For the joined state set \(r=x-x^*\) and \(z=\theta-\theta^*\). Then
\(\|r(0)\|_2,\|z(0)\|_2\le2q\). The source retains 64 independent
fine residual directions: eight zero-mean coordinates in each channel of
each child. It is not the earlier 68-direction parentwise residual ball.
After attachment only the global means are conserved; no child mean is reset.

The fixed support obeys

\[
 \lambda_2(L)>1/3,\qquad \|L\|_2\le8.
 \tag{6}
\]

Indeed its eigenvalues are sums of those of \(C_9,K_2,K_2\).
The C9 gap is \(2-2\cos(2\pi/9)>104/243>1/3\), using
\(2\pi/9>2/3\) and the cosine Taylor bound. Degree four gives the upper
bound eight. For any two lifted phase states, the mean-value edge Hessian
gives \(\|S(\theta)-S(\theta^*)\|_2\le8\|z\|_2\), even without
acute gaps. Its possibly signed conductances have magnitude at most one.

Integrating the form row first and substituting it into the phase row yields
the exact identity

\[
 \begin{split}
 z(\tau)={}&z(0)+\gamma(I-e^{-L\tau/4})r(0)\\
 &+\frac{\gamma^2}{4}\int_0^\tau
 (I-e^{-L(\tau-t)/4})[S(\theta(t))-S(\theta^*(t))]dt.
 \end{split}\tag{7}
\]

Both matrix norms \(\|e^{-Lt/4}\|_2\) and
\(\|I-e^{-Lt/4}\|_2\) are at most one. Gronwall therefore gives
\(\|z(\tau)\|_2\le2q(1+\gamma)e^{2\gamma^2\tau}\).
The form error is globally mean zero; (6) bounds its heat convolution.
For the complete declared interval \(0\le s\le5\), retain the exact
rational bounds, with \(\bar\gamma\) the outward upper bound of \(\gamma\):

\[
 Z=2q(1+\bar\gamma)3^{20},\qquad
 R=2q+24\bar\gamma Z,\qquad
 \|z\|_2\le Z<10^{-60},\quad \|r\|_2\le R<10^{-60}.
 \tag{8}
\]

Here \(2\gamma^2\tau=4s\le20\) and \(e<3\). The factor 24 is
\((8/4)/(\lambda_2(L)/4)\le24\). These are bounds on all original
fine perturbations about a moving reference, not a symmetry assumption on
the actual source. Their smallness comes from the already required recovery;
no original allowance or dwell is changed. Tiny exact quantities remain
rational rather than being rounded to zero on the interval grid.

## 3. Whole-source entry and child retention

Let \(a=1+2\varepsilon\), with \(0<\varepsilon<10^{-7}\).
For each equation (5), the region

\[
 0<D\le d,\qquad 0\le Y\le aD
 \tag{9}
\]

is invariant. At its upper sloped boundary,
\(\varepsilon(Y-aD)_s\le(1-a+\varepsilon a^2)D<0\); at \(Y=0\),
the derivative is nonnegative. In particular \(D\) decreases and stays
positive, with \(D(s)\ge d e^{-as}\). Since \(D\le1/2\),
\(\sin D\ge(23/24)D\). The exponential convolution for \(Y\) and
monotonicity of \(\sin D\) along the trajectory give

\[
 Y(s)\ge\sin D(s)(1-e^{-s/\varepsilon}),\qquad
 d e^{-as}\le D(s)\le
 d\exp[-(23/24)(s-\varepsilon)].
 \tag{10}
\]

The right inequality integrates \(D_s/D\); dropping the positive
\(\varepsilon e^{-s/\varepsilon}\) term only enlarges the bound.
At the fixed deadline it implies

\[
 B(4)<1/64,\qquad \Delta(4)>1/1200.
 \tag{11}
\]

These strict elementary inequalities are checked with outward exponential
bounds. For example \(e^{4+8\varepsilon}<60\), using
\(e<11/4\) and \(e^{8\varepsilon}\le(1-8\varepsilon)^{-1}\),
gives the second inequality. Both gaps keep decreasing in the continuation.

At release every actual intraparent same-slot gap is at least
\(1/2-2q>1/4\). Thus the entire source initially lacks parent identity.
Throughout passage, every oriented internal edge has its continuous branch
in \([\alpha-2Z,\alpha+2Z]\subset(0,\pi/2)\). The closing edge retains
its original branch integer, and the residual differences telescope to a
total \(2\pi\). Every child keeps winding one on \(0\le s\le5\).

Use the residual convention of
[repeated composition](REPEATED_ORGANIZED_COMPOSITION.md#5-a-full-transverse-domain-and-identity-bound):
subtract the *actual* parent form mean from form, and subtract the winding
carrier \(\psi_j\) and actual parent phase mean \(\Phi_g\) from phase.
Thus \(r_\theta=\max_{g,a,j}|\theta_{g,a,j}-\psi_j-\Phi_g|\), not the
distance of all primitive phases from a common phase. The carrier has zero
mean, so the mean of phase offsets equals the mean of these declared phase
lifts. For every \(4\le s\le5\), the fine residual sup norms satisfy

\[
 r_\theta\le B(4)/2+2Z<1/128+2Z<1/100,
 \quad
 r_x\le\bar\gamma(1+2\bar\varepsilon)B(4)/2+2R<1/100,
 \tag{12}
\]

where \(\bar\varepsilon\) is an outward upper bound of \(\gamma^2\).
Every actual intraparent
same-slot gap is smaller than \(1/64+2Z<1/4\). This proves entry by
the fixed time and subsequent maintenance on the declared interval, for the
entire source, while all four child identities remain intact.

## 4. Predictive use from the reached state

The outer current on the moving separated reference is exactly
\(\sin\Delta\), although its children still have different origins.
Both parents carry the same internal mode \(B\), so the actual port
association matters. Bound (8) retains its perturbation during the whole
slow interval. Membership in the parent residual ball alone does not prove
this long prediction: the older generic theorem has horizon one in \(\tau\),
not one in \(s\).

First initialize the nonlinear four-coordinate parent law from all four
actual observations at \(s=4\): \((X_0,X_1,\Phi_0,\Phi_1)\).
There is no replacement by zero forms or the original `Delta=1/20`.
The two-node version of (7), with Laplacian norm two, implies over the
following slow unit

\[
 |\Delta_{\rm actual}(s)-\Delta_{\rm parent}(s)|
 \le 2Z+e^{s-4}(2Z+2\bar\gamma R)
 <8Z+6\bar\gamma R.
 \tag{13}
\]

The first term compares the actual fine state with the original moving
reference; the second carries the reduced predictor's true initial
discrepancy. Thus the controlled parent dynamics keeps actual forms and
means, with a quantitative source-specific error rather than an exact quotient.

An explicit forecast for the selected observation uses

\[
 F(A)=2\arctan(e^{-1}\tan(A/2)),\qquad A=\Delta_{\rm actual}(4).
 \tag{14}
\]

This is the time-one map of \(D_s=-\sin D\), a declared approximation.
Its error is justified from the full retained history. For the nominal outer
mode, write \(h=Y-\sin D\). Variation of constants in (5) gives

\[
 h(s)=-\sin(d)e^{-s/\varepsilon}
       +\int_0^s e^{-(s-t)/\varepsilon}\cos D(t)Y(t)dt.
 \tag{15}
\]

For \(s\ge4\), \(e^{-s/\varepsilon}\le\varepsilon/4\), while
\(Y\le ad\) and \(d=1/20\). Therefore
\(|h|\le d\varepsilon(1/4+a)<\varepsilon/10\).
The positive small-angle gradient flow is nonexpansive; its comparison with
the nominal pendulum for one unit costs at most \(\varepsilon/10\).
The initial and final actual-gap errors each cost at most \(2Z\). Hence

\[
 |\Delta_{\rm actual}(5)-F(\Delta_{\rm actual}(4))|
 \le\bar\varepsilon/10+4Z<10^{-5}.
 \tag{16}
\]

The observed initial gap is not a nominal reset. Forms and source history
enter the lag bound (15); (14) by itself certifies neither that history nor
a universal phase-only law.

The readout is resolved, not merely enclosed near zero. On \([4,5]\),
(10) implies \(Y>(3/4)D\), so \(D(5)<e^{-3/4}D(4)<D(4)/2\).
Using (11) and retaining both fine errors proves

\[
 \Delta_{\rm actual}(4)-\Delta_{\rm actual}(5)
 >1/2400-4Z>1/2500>10^{-4}.
 \tag{17}
\]

The forecast error bound is independent of evaluating a reduced or full
trajectory. This is a finite analytic prediction about the declared source
family, without a sensor or laboratory-time interpretation.

## 5. Preparation and attachment work

The four isolated phase translations preserve internal storage. At attachment,
both sets of new contacts contribute their actual instantaneous storage jump:

\[
 W=\sum_{\text{new }\{i,k\}}
 [\tfrac12(x_k-x_i)^2+1-\cos(\theta_k-\theta_i)].
 \tag{18}
\]

At the nominal source,
\(W^*=18[1-\cos(1/2)]+18[1-\cos(1/20)]>2\).
All first-order potential corrections cancel by the original childwise zero
sums, despite nonzero nominal contact currents. The contact Laplacian has
norm four, and \(\|r(0)\|_2,\|z(0)\|_2\le2q\). Taylor's second-order
bound gives

\[
 0\le W-W^*\le16q^2.
 \tag{19}
\]

The lower bound uses convexity along the acute initial contact gaps; the
upper bound includes both form and phase terms. Likewise,
\(0\le H(0)-[36(1-\cos\alpha)+W^*]\le32q^2\), using \(\|L\|_2\le8\).
These exact tiny corrections are retained separately from nominal irrational
intervals. The support event requires external work; prior continuous loss
is not reused as an undeclared reservoir. Source storage, acquired registration
and positive dissipation do not establish a physical binding energy.

## 6. Acquired-source antipodal nonentry control

Keep the same matching support and complete law. Within each parent choose
two identical copies of an actual recovered child state, and translate one
copy's phase by exactly \(\pi\). Fix their supplied origins as
\(\phi^\pi_{g,a}=(2g-1)/40+(2a-1)\pi/2\), retaining the outer nominal
gap \(1/20\). Thus

\[
 x_{g,1,j}=x_{g,0,j},\qquad
 \theta_{g,1,j}=\theta_{g,0,j}+\pi.
 \tag{20}
\]

This is a correlated source subset with genuine nonzero fine residuals
allowed. Intraparent form and sine currents vanish; internal cycle currents
and same-child outer currents are identical in the two copies. Both evolution
rows preserve (20) by uniqueness. Consequently every intraparent gap remains
antipodal for all future time, and parent registration never enters its
quarter-radian identity domain. This uses the exact circular identity
\(\sin\pi=0\), not the sine of a floating-point approximation to pi.

The same global perturbation bound around the antipodal moving reference
preserves each child's winding through \(s=5\). No all-future winding claim
or independently perturbed nonentry neighborhood follows. Its nominal work is
\(W^*_{\pi}=36+18[1-\cos(1/20)]\), with correction in \([0,8q^2]\).
This is a different preparation and work budget, not an equal-work comparison.
It proves that the admitted law and acquired children alone do not force
every registration to organize; relative preparation remains consequential.

## 7. Evidence and scope

The [detached assessor](../../src/tnfr/physics/_organized_parent_acquisition.py)
re-admits all primitive allowances before rebuilding the four isolated
recoveries. It reuses the existing matching geometry and shared outward
arithmetic, retaining source admission separately from conditional bounds.
The [independent controls](../../tests/physics/test_organized_parent_acquisition.py)
check fine and separated rows, both clocks, spectral bounds, every strict
entry/readout margin, work corrections, primitive rejection and the
antipodal symmetry. The forecast arithmetic helper supplies only (14), not
an acquisition certificate for an arbitrary supplied observation.

The result supplies acquired parent registration and a resolved subsequent
interaction from the original uncertain children under one supplied law and
support. The inherited long recovery dwell and the slow structural horizon
remain mathematical preparation premises. Autonomous preparation, contact
selection, structural separation resistance, indefinite maintenance, other
constitutive laws and physical identification require their own arguments.
The matching support is symmetric under exchange of its two binary factors;
their nominal modes obey the same relaxation law. The different initial
amplitudes and the marked grouping do not select a unique hierarchy or a
separation of internal and external timescales.

The separate [restoring-response admission](ORGANIZED_PARENT_RESTORING_RESPONSE.md)
carries this reached source through one supplied internal phase intervention,
accounts for its work and tests finite recovery and sufficient predictive state.
It does not extend this theorem's unchanged-source prediction to an intervened
state without that additional argument.
