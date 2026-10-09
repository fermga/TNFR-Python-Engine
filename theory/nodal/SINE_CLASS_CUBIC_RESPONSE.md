# Complete cubic-amplitude response of an acquired mediator

<a id="sine-class-cubic-response"></a>

The [organization contrast](SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-unresolved)
isolates a finite mediator contribution but leaves its sign unresolved by
an independent-class heat remainder. This owner retains the complete
cubic-amplitude response instead. Linear phase feedback and quadratic
internal corrections are part of the coefficient, not discarded into that
remainder. A complex-parameter estimate controls all higher amplitude
orders without changing the real nodal model.

This is a conditional coefficient calculation and finite-error theorem.
It executes no formation assessment or nonlinear reserved trajectory and
does not reuse the retained class-two response as new cross-class data.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns its research status.

<a id="sine-cubic-source-and-observer"></a>
## Complete source, events and observation

Retain both actual class families \((1,k,1)\), \(k=1,2\), on the
same three-C9 graph, with contacts \((4,13)\), \((13,22)\), unit
capacities and all 54 original coordinates. Use the supplied law
\[
 x'=-Ax+\gamma KS(\theta),\qquad \theta'=\gamma Ax,
 \qquad A=KL,\quad\gamma=1/(1023\pi),\quad \tau=et,
 \quad e=1023/1024.
 \tag{1}
\]
Each class's four histories share its complete actual acquired source;
the two classes may have independent residuals. The
[original source and cost obligations](SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-source)
are unchanged, including separate zero component sums and form and
target-phase Euclidean error bounds \(\epsilon\). No phase, form,
mean or hidden coordinate is reset at either event.

Donor central form impulses occur at times zero and \(s\). The fixed
design remains
\[
 a=b=1/2000,\quad s=1,\quad T=2,\quad
 \epsilon=10^{-32},\quad\delta=10^{-30},\quad r=1/12.
 \tag{2}
\]
Let \(M_k=R_{k,11}-R_{k,10}-R_{k,01}+R_{k,00}\), where \(R\)
is receiver central form at \(T\), and let \(D=M_1-M_2\).
The eight scalar recording errors satisfy \(|e_j|\le\delta\).
Consequently \(|\widehat D-D|\le8\delta\); no probability law or
laboratory error calibration is assumed.

The nominal equilibrium is only a proof reference for the already
admitted full source family. For one history scale all its supplied form
impulses by an auxiliary parameter \(\zeta\), leaving times and every
law unchanged. Its nominal trajectory has ordinary Taylor coefficients
\[
 x(\zeta,t)=\sum_{n\ge1}\zeta^n x_n(t),\qquad
 y(\zeta,t)=\theta-\Theta_k=\sum_{n\ge1}\zeta^n y_n(t).
 \tag{3}
\]
Thus the coefficients include the usual Taylor factorial normalization.
Amplitude derivatives are derived from the law, not measured extra
channels or new initial preparations.

<a id="sine-cubic-triangular-variation"></a>
## The complete triangular variation equations

Write \(C_k\) for the degree-normalized phase Laplacian at \(\Theta_k\),
and put \(d_{ij}=\Theta_{k,j}-\Theta_{k,i}\). Define normalized
multilinear maps
\[
 Q_k(v,w)_i=-\frac1{2d_i}\sum_{j\sim i}
       \sin d_{ij}\,(v_j-v_i)(w_j-w_i),
 \qquad
 T_k(v,w,z)_i=-\frac1{6d_i}\sum_{j\sim i}
       \cos d_{ij}\,(v_j-v_i)(w_j-w_i)(z_j-z_i).
 \tag{4}
\]
Their signs and factors follow by expanding the original sine current.
In original coordinates, with
\(J_k=\left(\begin{smallmatrix}-A&-\gamma C_k\\\gamma A&0\end{smallmatrix}\right)\),
the first three levels satisfy
\[
 \begin{aligned}
 (x_1,y_1)'&=J_k(x_1,y_1),\\
 (x_2,y_2)'&=J_k(x_2,y_2)+(\gamma Q_k(y_1,y_1),0),\\
 (x_3,y_3)'&=J_k(x_3,y_3)
       +(2\gamma Q_k(y_1,y_2)+\gamma T_k(y_1,y_1,y_1),0).
 \end{aligned}
 \tag{5}
\]
Every present impulse adds its amplitude to \(x_{1,D}\). Levels two
and three have no jump, and all phase coefficients are continuous. The
second event uses every carried coefficient from the first window.

For conditioned coefficient arithmetic use
\[
 x_1=u_1,\ y_1=\gamma w_1,\qquad
 x_2=\gamma^3u_2,\ y_2=\gamma^4w_2,\qquad
 x_3=\gamma^4u_3,\ y_3=\gamma^5w_3,
 \quad \eta=\gamma^2.
 \tag{6}
\]
All three pairs then have the same linear block
\(\mathcal J_k=\left(\begin{smallmatrix}-A&-\eta C_k\\A&0\end{smallmatrix}\right)\).
The second form forcing is \(Q_k(w_1,w_1)\); the third is
\(2\eta Q_k(w_1,w_2)+T_k(w_1,w_1,w_1)\). The factor
\(\eta\) in the quadratic recoupling must remain. These scaled
coordinates are used for the nominal coefficient calculation only;
actual phase uncertainty is never divided by \(\gamma\).

The port-fixing reflection gives
\(\mathcal R(x_n,y_n)=(-1)^{n+1}(x_n,y_n)\). Hence the nominal
central coefficient vanishes at every even amplitude degree. Quadratic
internal coefficients generally do not vanish: their reflection-odd
phase component feeds the third level through (5). The complete tangent
mixed response cancels within each class, so the nominal cubic mixed
coefficient is
\[
 C_k^{(3)}=\gamma^4
   [u_{3,k,11,R}(T)-u_{3,k,10,R}(T)-u_{3,k,01,R}(T)],
 \qquad D^{(3)}=C_1^{(3)}-C_2^{(3)}.
 \tag{7}
\]
The nominal unprobed coefficient is exactly zero. Equations (5)-(7)
retain full tangent feedback and all intermediate nodes, unlike the
earlier heat-cubic approximation.

<a id="sine-cubic-amplitude-tail"></a>
## A rigorous bound beyond cubic amplitude

Fix \(g=1/3000>\gamma\). For a history with total absolute impulse
budget \(A_*>0\), require \(2gA_*<1\), and choose
\[
 \rho=\frac1{2gA_*}>1,\qquad
 D_T=1-2g^2T^2>0,\qquad 16g^2T^2<1.
 \tag{8}
\]
Complex values of \(\zeta\) below are auxiliary analytic variables;
physical form and phase are still real. The heat reference on
\(|\zeta|\le\rho\) has phase
\[
 y_{\rm heat}(t)=\gamma\zeta\sum_{t_j\le t}
       q_j[I-P(t-t_j)]e_D,\qquad P(t)=e^{-At}.
 \tag{9}
\]
Markov contraction and nonnegativity imply
\(\|[I-P(v)]e_D\|_\infty\le1\), so
\(\|y_{\rm heat}\|_\infty\le\gamma\rho A_*\le1/2\).
This bound retains both events; replacing it by a linear-in-time phase
estimate would needlessly shrink the analytic disk.

For real \(d\) and complex \(z\), derivatives of sine have modulus
at most \(\cosh|\operatorname{Im}z|\). On the bootstrap
\(\|y\|_\infty\le1\), all edge increments have modulus at most
two and \(\cosh2<4\). Thus the normalized sine difference is at
most \(8\|y\|_\infty\). Comparing full and heat form, then
integrating the complete phase row, gives
\[
 Y(t)\le\frac12+16g^2\int_0^t(t-v)Y(v)\,dv,
 \qquad \sup_{v\le t}Y(v)\le\frac12+8g^2t^2\sup_{v\le t}Y(v).
 \tag{10}
\]
At a first point with \(Y=1\), (8) makes the right side strictly
less than one. This closes the bound on the disk. The form remains bounded
by its heat value and a finite forcing integral. The entire sine field,
linear jumps and strict bound give holomorphic parameter dependence on
a neighborhood of the closed disk throughout the complete finite history.
This is a mathematical extension for the error proof, not a complex-state
replacement of the model.

The normalized sine defect after its linear part is at most
\(8\|y\|_\infty^2\). Therefore its form forcing is at most
\(8g\) on this disk. Diffusion contraction and the phase integral
bound the complete nonlinear-minus-tangent form difference by
\[
 \sup_{|\zeta|\le\rho,t\le T}
 |x_R(\zeta,t)-\zeta x_{1,R}(t)|\le\frac{8gT}{D_T}.
 \tag{11}
\]
Indeed the linear phase feedback contributes at most \(2g^2T^2\)
times the form-error supremum. No exponential bound for the full block
generator is needed here.

This central difference is odd in \(\zeta\) and starts at degree
three. Cauchy's coefficient estimate, summed over degrees five, seven
and higher at \(\zeta=1\), proves
\[
 |x_R(1,T)-x_{1,R}(T)-x_{3,R}(T)|
 \le R_5(A_*,T)
 :=\frac{256g^6A_*^5T}
  {(1-2g^2T^2)(1-4g^2A_*^2)}.
 \tag{12}
\]
If \(A_*=0\), the nominal history is stationary and this remainder
is exactly zero; no division by zero in (8) is required. Actual residuals
need not have the reflection symmetry used in this proof.

<a id="sine-cubic-time-enclosure"></a>
## Fixed finite evaluation of the triangular coefficient

The coefficient policy fixes dyadic128 outward arithmetic, time order
\(N=64\), and \(L=201/100\) before evaluating any new coefficient.
The two time pieces are \([0,1]\) and \([1,2]\). For each class,
one first-probe prefix supplies the first-only and both-probe suffixes;
the delayed-only suffix starts from the exact nominal unprobed state.
This uses eight analytic segments in total. Every level carries at the
second event. There is no adaptive order, subdivision or parameter search.
These are coefficients of (5), not new nonlinear full-state observations.

The full block obeys \(\|\mathcal J_k\|_\infty\le2+2\eta<L\).
The normalized forcing norms in scaled coordinates are bounded by
\(2\|w_1\|^2\) and
\(4\eta\|w_1\|\|w_2\|+(4/3)\|w_1\|^3\).
Let \(v_1,v_2,v_3\) bound the complete scaled pairs at a segment's
initial point. The following positive series dominate their absolute
time coefficients coefficient by coefficient:
\[
 \begin{aligned}
 V_1(t)&=v_1e^{Lt},\\
 V_2(t)&=(v_2+2v_1^2t)e^{2Lt},\\
 V_3(t)&=[v_3+(4\eta v_1v_2+4v_1^3/3)t
                        +4\eta v_1^3t^2]e^{3Lt}.
 \end{aligned}
 \tag{13}
\]
One proof first uses the coefficientwise comparison system
\(V_1'=LV_1\), \(V_2'=LV_2+2V_1^2\),
\(V_3'=LV_3+4\eta V_1V_2+(4/3)V_1^3\). Its convolution
coefficients are bounded by the positive exponential-polynomial series
in (13): substitution leaves nonnegative coefficients in each differential
supersolution residual, so induction on the Taylor degree gives the
comparison. Pointwise growth bounds alone would not justify a Taylor tail.
Numerical majorants replace \(\eta\) by its admitted outward upper
bound; coefficient intervals retain the original parameter enclosure.

For \(z\ge0\) and \(z<m+1\), define
\[
 E_m(z)=\frac{z^m}{m!\,[1-z/(m+1)]}.
 \tag{14}
\]
It bounds the exponential series starting at power \(m\), by its
first term and the largest subsequent ratio. For segment length \(h\),
the tails after time degree \(N\) are bounded by
\[
 \begin{aligned}
 T_1&=v_1E_{N+1}(Lh),\\
 T_2&=v_2E_{N+1}(2Lh)+2v_1^2hE_N(2Lh),\\
 T_3&=v_3E_{N+1}(3Lh)
 +(4\eta v_1v_2+4v_1^3/3)hE_N(3Lh)
 +4\eta v_1^3h^2E_{N-1}(3Lh).
 \end{aligned}
 \tag{15}
\]
Thus order 64 uses the starts 65, 64 and 63 for the constant, linear
and quadratic prefactors. Each outward endpoint includes its own tail
before being carried into the next segment. Source, sine/cosine,
coefficient arithmetic, time truncation and amplitude truncation are
separate obligations; the ordinary full-state solver's dimension or
time-order cap is not changed by this triangular coefficient calculation.

<a id="sine-cubic-source-and-records"></a>
## Transfer to the actual families and recording limits

Let \(J^{(3)}\) be an outward interval for (7), including the complete
triangular coefficient and time tails. The true contrast satisfies
\[
 D\in J^{(3)}+[-F,F],\qquad
 F=2\{R_5(|a|+|b|,T)+R_5(|a|,T)+R_5(|b|,T)\}
       +\frac{8\epsilon}{1-2\bar\gamma T}.
 \tag{16}
\]
Here \(\bar\gamma\) is an admitted outward upper bound for
\(\gamma\), with \(\bar\gamma\le g=1/3000\). The source comparison
may use this sharper bound while the Cauchy disk and (12) retain the fixed
\(g\). The last term compares each actual history with its own nominal
class reference in the original coordinates. It grants no cancellation of
independent cross-class residuals. At (2), exact rational arithmetic gives
\[
 2\{R_5(1/1000,2)+2R_5(1/2000,2)\}<1.6\,10^{-33},\qquad
 \frac{8\epsilon}{1-2\bar\gamma T}<8.02\,10^{-32}.
 \tag{17}
\]
This bounds approximation and source allowances independently of the
coefficient. It proves neither sign nor separation without the coefficient
enclosure and is not a new measured error.

Recorded-sign certification requires the true interval (16) to lie
strictly beyond \(\pm8\delta\). Excluding a separately noisy
eight-record null model through this statistic retains the different
\(16\delta\) condition. Neither threshold is inferred from the
coefficient's sign.

There is also a precise scalar observation obstruction. If (16) instead
proves \(|D|\le8\delta\) throughout the actual family, write
\(\sigma_j\in\{-1,1\}\) for the eight coefficients defining \(D\).
For each actual pair of class sources choose
\[
 e_j=-\sigma_jD/8.
 \tag{18}
\]
Every \(|e_j|\le\delta\), and the recorded statistic is exactly
zero. Thus every such actual source pair admits at least one allowed
error vector erasing this scalar contrast. It does not follow that all
records are uninformative, that full eight-record sets overlap, or that
the underlying organization-dependent response vanishes. Favorable
errors or different functions of those records can have different
discrimination properties.

The [source costs and event/identity proof](SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-events)
are unchanged by improving the approximation. They retain different
class preparations, independent residuals, the full delayed state, both
work allowances and each resulting conserved-mean leaf. Their validity
is not inferred from a signed coefficient or a noise obstruction.

<a id="sine-cubic-evaluation-boundary"></a>
## Evaluation boundary

The fixed policy above precedes evaluation of the refined coefficient.
Its finite outcome, whether a resolved sign, a scalar-resolution
obstruction or an unresolved interval, must be stated separately from
the formulas. No refined coefficient or complete-law observation is
asserted by the admission alone. Earlier frozen responses and the valid
limitation of the independent heat-remainder method remain unchanged.

<a id="sine-cubic-finite-result"></a>
## Finite coefficient result and scalar recording obstruction

The first calculation under the fixed policy completed all eight
analytic segments at order 64, without subdivision or a repeated
coefficient call. Its complete cubic contrast, including interval
arithmetic and the carried time tails, satisfies the outward bound
\[
 J^{(3)}\subset
 [-7.013764940694,-7.013764940692]\,10^{-30}.
 \tag{19}
\]
The exact retained interval is much narrower than these displayed
endpoints: its width is less than \(1.182\,10^{-49}\).
The higher-amplitude allowance is less than \(1.493\,10^{-33}\),
and the independent-source allowance is less than
\(8.010\,10^{-32}\). Reconstructing (16) from those exact rational
quantities gives
\[
 -7.096\,10^{-30}<D<-6.932\,10^{-30}<0.
 \tag{20}
\]
Both complete class families therefore have a strictly ordered finite
nonlinear mixed response in the declared class-one-minus-class-two
orientation. This sign includes the quadratic internal recoupling,
every higher amplitude order through its bound, and the independently
admitted actual residuals. The shared source/work ledger also satisfies
the original contact and probe allowances and both identity guards;
the coefficient calculation does not replace those premises.

At the unchanged \(\delta=10^{-30}\), however, the whole true interval
lies strictly inside \((-8\delta,8\delta)\). The recorded interval
obeys only
\[
 \widehat D\in[-15.096,1.068]\,10^{-30},
 \tag{21}
\]
and (18) supplies an allowed error vector making this statistic zero
for every admitted actual pair. The exact cancellation margin exceeds
\(9.046\,10^{-31}\). Thus the result proves both an
organization-dependent true response and a worst-case scalar recording
limitation at the fixed noise budget. It certifies neither a recorded
sign nor exclusion of a noisy null alternative through this statistic.
It does not assert equality or overlap of the complete eight-record
vectors, and it does not identify a physical property.

This calculation evaluates amplitude coefficients and rigorous
remainders, not a new complete nonlinear response or formation run.
The earlier independent-class heat estimate remains a valid, weaker
unresolved enclosure. Its limitation has been removed here by retaining
the omitted dynamics in the coefficient, without changing the law,
source, events, horizon or recording budget.

The maintained [coefficient implementation](../../src/tnfr/physics/relational_sine_class_cubic_response.py),
[implementation controls](../../tests/physics/test_sine_class_cubic_response.py),
[independent algebra controls](../../tests/physics/test_sine_class_cubic_response_algebra.py)
and [shared contrast decisions](../../tests/physics/test_sine_class_contrast.py)
keep coefficient calculation, source admission and observation decisions
separate.

The [byte-preserving evidence bundle](../../docs/assets/sine_formed_classes/class-cubic-response-v1.evidence.zip)
retains the declared policy, first attempt ledger, exact outcome,
evaluated source and driver. A source-packaging operation initially used
an incorrect dependency path and stopped before creating the attempt
ledger or importing/calling the coefficient calculation. Its partial
archive and error record are preserved alongside the separately named
complete archive. Correcting that path changed no policy, inputs,
numerical budget or runtime bytes; there was one coefficient evaluation.
The bundle stores the original 33,439,119-byte outcome losslessly.

The relevant SHA-256 identities are:

| Object | SHA-256 |
| --- | --- |
| Policy | `f958646ea96ca8d850563e12b18e72ce0d51b8605967de208f29ffd9e1281245` |
| Complete evaluated-source archive | `6f559fdc6d1fb69739915723e78ac9690701236328a34c23542ef35d935d5209` |
| Exact outcome | `a56348c90862d0acecbedb610c6599b2de956dba11b7524f005989eb655dbe12` |
| Evidence bundle | `991d771464d84f65a25c0a82c9b08f7e5a579e450885141b32fef1a1ed8239cb` |

The [read-only evidence audit](../../tests/physics/test_sine_class_cubic_evidence.py)
rebuilds consumed downstream interval arithmetic and decisions from
retained primitives. Generating the stored coefficient enclosures remains
an explicit execution premise, supported by the derivation and independent
controls; reading or hashing the bundle is not a replay of that calculation
or a proof of physical identification.
