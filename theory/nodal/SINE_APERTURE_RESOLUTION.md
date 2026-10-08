# Finite-noise and horizon resolution for the four boxcar observations

<a id="sine-aperture-resolution"></a>

## Question and scope

The [finite-aperture observation law](SINE_APERTURE_INFERENCE.md#sine-aperture-source-and-law)
has four substantial windows, not point samples. Its earlier conditional
resolution example uses a very small error allowance. This owner asks what
happens when the horizon, source uncertainty, clock drift and observation
error are varied **before selecting a response**.

There are two different results. A sufficient budget gives conditional
outer-width guarantees and can fail because its bounds are conservative.
A separate complete-history construction proves actual indistinguishability
under sufficiently large bounded sensor error. Neither result calibrates
an instrument, identifies a physical time unit or selects the supplied
nodal law. No saved response, inverse output or producer is used to choose
the regimes below; earlier frozen owners and evidence retain their scope.

Keep the same complete eighteen-node support, unit capacities, phase events,
source family and held sensor law. With the degree metric \(M\), normalized
Laplacian \(A\), dipole \(q=e_4-e_5\), and \(\gamma=1/(1023\pi)\),

\[
x_\tau=-Ax+\gamma f(\theta),\qquad
\theta_\tau=\gamma Ax,\qquad d\tau/ds=\rho(s).
\tag{1}
\]

Retain the fixed priors and inputs

\[
b\in[11/8,3/2],\quad c\in[2/3,1],\quad
G\in[1,2],\quad \rho(s)\in[1/2,2],\quad
(a_1,a_2)=(1/4,3/4).
\tag{2}
\]

The phase jumps are \(a_1q\) at zero and \((a_2-a_1)q\) at \(H\).
All form coordinates stay continuous. The complete source has arbitrary
common form and phase means and centered residual norms at most \(X,Y\).
The original long-arc mean remains \(B=b-(v_1-v_0)/8\), hence

\[
|B-b|\le Y/8.
\tag{3}
\]

The four windows are \([0,H/3]\), \([H/3,2H/3]\),
\([2H/3,H]\), and \([H,2H]\). Every reading is its normalized
observation-time average of \(Gq^Tx+O\), plus one bounded scalar error.
The gain is positive and the offset is arbitrary; both are held throughout.
The target clock remains
the first-window mean \(\bar\rho=H^{-1}\int_0^H\rho(s)\,ds\),
with effective gain \(J=G\bar\rho\).

The clock is positive and \(C^1\) on the entire interval \([0,2H]\).
The six variable budget primitives are

\[
0<H\le1/4,\qquad X,Y,\delta,t,\Lambda\ge0,
\qquad |\rho'(s)|\le\Lambda.
\tag{4}
\]

Here \(\delta\) bounds each physical sensor error; \(t\) bounds each
primitive recorded interval's half-width. The latter can include source
materialization and numerical enclosure width. They are different premises,
even where a sufficient inequality consumes their sum
\(\varepsilon=t+\delta\). A hypothetical zero-width reading is admissible
within any nonnegative \(t\); no actual numerical producer is promised to
achieve it.

For the uniform sufficient chart guard, require \(Y<1/256\). The smallest
nominal acute margin on (2) is \(11-7\pi/2\). The rational upper bound
\(\pi<355/113\) gives \(11-7\pi/2>1/226>1/256\).
Every edge difference has dual degree-metric norm at most one, so this
guard preserves the initial chart for every admitted residual. Failure of
this conservative guard does not prove that a particular source is nonacute.
The complete later flow may leave the initial acute sector; no maintenance
claim is needed by the global finite bounds.

<a id="sine-aperture-resolution-budget"></a>
## A response-free sufficient budget

Set \(g=1/3069>\gamma\) and \(h_*=2H\). The
[global derivative and transfer proof](SINE_APERTURE_INFERENCE.md#sine-aperture-clock-transfer)
gives

\[
\begin{aligned}
Q_0&=X+28gH,& U&=2Q_0+7g,\\
M_2&=4(1+g^2)Q_0+14g,\\
M_{3,g}&=8(1+2g^2)Q_0+28g(1+g^2)+8g^3Q_0^2.
\end{aligned}\tag{5}
\]

For a uniform budget, using the derivative branch of the clock-transfer
bound is sufficient even when the alternative rate-range bound is sharper.
Define

\[
c_*=2U\Lambda H^2,\quad e=4M_{3,g}H^3/27,\quad
e_m=M_{3,g}H^3/144,\quad \ell=4M_2H^2/3.
\tag{6}
\]

The exact moment matrix of the aperture owner gives virtual point-band
half-widths no larger than

\[
\begin{aligned}
r_0=r_2&=(10/3)\varepsilon+(91/324)c_*+e,\\
r_1&=(7/6)\varepsilon+(11/81)c_*+e_m,\\
r_3&=r_2+2\varepsilon+(5/6)c_*+\ell.
\end{aligned}\tag{7}
\]

These coefficients follow by applying the absolute moment matrix to the
four numerical/sensor budgets and averaged exposure bounds; the Peano
remainders are event-separated. The original curvature coefficients are
\((9/4,-9/2,9/4,0)\), with exact sensor support \(9\delta\).
The same coefficient norm gives \(9t\) for numerical half-widths. There
is no new error attached independently to each virtual point.

Keep the coarse inverse's specialized bound separate from (5):

\[
\begin{aligned}
Q_s&=\frac{X+4gH(7+2Y)}{1-64g^2H^2},\\
E_*&=4HQ_s+4gHY+32g^2H^2Q_s,\\
R&=\frac{18000}{H}(2E_*+r_2+r_3).
\end{aligned}\tag{8}
\]

The denominator is positive throughout (4). The
[two-input coefficient bound](SINE_TWO_PULSE_INFERENCE.md#sine-two-pulse-conditioning)
is \(\sqrt{S_1^2+S_2^2}<36000/h_*\). Since
\(r_0=r_2\le r_3\), the inherited three-reading row bounds give the
last line of (8), exactly as in the aperture conditioning proof.
The auxiliary radial prior is \([1/4,2]\). If \(R<1/8\), compatible
observations therefore have conditional exact-arithmetic width bounds

\[
W_J=4R,\qquad W_b=\frac{2R}{1/4-2R},\qquad
W_B=W_b+Y/4.
\tag{9}
\]

For the first-window curvature remainder, use

\[
\begin{aligned}
S_s&=\min\{7,5+2Y+16gHQ_s\},\\
M_{3,s}&=8(1+2g^2)Q_s+4g(1+g^2)S_s+8g^3Q_s^2,\\
D&=8(1+g^2)X+8gY+2HM_{3,s}.
\end{aligned}\tag{10}
\]

The global and specialized third-derivative quantities have different
roles. Replacing \(Q_s\) or \(M_{3,s}\) by (5) is not justified by their
names or by a comparison at one horizon.

The diameter allowance for \(4C/H^2\) simplifies exactly to

\[
\boxed{d_A=\frac{8(r_0+2r_1+r_2)}{H^2}
 =\frac{72\varepsilon}{H^2}
  +\frac{40}{3}U\Lambda+\frac{67}{27}M_{3,g}H.}
\tag{11}
\]

This exposes the noise amplification, drift floor and finite-flow growth.
The source contribution additionally enters \(D\), \(R\) and the original
angle width. Shortening the window is not a uniform improvement.

The positive natural curvature coefficient has the
[uniform lower bound](SINE_FINITE_CURVATURE_INFERENCE.md#sine-curvature-conditioning)
\(k_0=1/18000\), and its interval width is at most
\((11g/8)W_b\). For a response-independent positive-division guard set

\[
L_A=\tfrac12(\tfrac12 k_0-D)-d_A,
\qquad L_A>0,\qquad L_A/4-D>0.
\tag{12}
\]

The guards imply \(D<k_0/2\), so the lower coefficient used below is
positive and multiplication by the lower bound on \(J\) is legitimate.
Indeed, compatibility, \(J\ge1/2\), \(\bar\rho\ge1/2\), and the
finite curvature theorem bound the true normalized numerator below by
\(\tfrac12(\tfrac12k_0-D)\). Subtracting its diameter gives \(L_A\).
Division by \(J\le4\), then allowance \(D\), proves positivity of the
next numerator. These are sufficient guards, not necessary properties of
every possible informative record.

Put

\[
\beta=\frac{11gW_b}{8k_0},\qquad
z=2W_J+\beta+2W_J\beta.
\]

The inherited positive-quotient argument gives

\[
W_{\bar\rho}=\frac{2d_A}{k_0}
 +(2+D/k_0)z+\frac{2D}{k_0},\qquad
W_G=2W_J+4W_{\bar\rho}.
\tag{13}
\]

One sufficient resolution certificate is the chart guard, (12), \(R<1/8\),
and the four strict policy targets

\[
W_B<1/1024,\quad W_J<1/2048,\quad
W_G<1/16,\quad W_{\bar\rho}<1/64.
\tag{14}
\]

These statements are conditional on a compatible full history and the
declared reading-width allowance. They concern the mathematical enclosure
method. Actual inverse arithmetic must still admit its outward coefficients,
quotients and source; a budget calculation does not generate observations
or assert that every Cartesian tuple of marginal bounds is realizable.

<a id="sine-aperture-resolution-horizons"></a>
## A finite positive window interval and explicit budget obstructions

There is an informative regime with a larger error allowance than the
earlier illustrative budget, without selecting any response:

\[
X,Y\le2^{-48},\quad\Lambda\le2^{-22},\quad
\delta,t\le2^{-80},\qquad
3\cdot2^{-26}\le H\le2^{-24}.
\tag{15}
\]

The absolute apertures vary with \(H\); their relative widths remain the
same fixed thirds and full second window. Both sensor and numerical budgets
are finite independent inputs. The error allowance is 1024 times the earlier
\(2^{-90}\) example; this numerical comparison is not an attainability claim.

For an explicit uniform verification let \(H_-=3\cdot2^{-26}\),
\(H_+=2^{-24}\), and take the largest other budgets in (15).
Expand (8) as

\[
R=18000\left[8Q_s+8gY+64g^2HQ_s+
 \frac{26\varepsilon}{3H}+\frac{226}{81}U\Lambda H
 +\frac8{27}M_{3,g}H^2+\frac43M_2H\right].
\tag{16}
\]

Every term except the explicit inverse powers of \(H\) is nondecreasing
on this interval. Evaluate those increasing terms at \(H_+\), and replace
\(\varepsilon/H\), \(\varepsilon/H^2\) by their values at \(H_-\).
Use the resulting upper bounds in (9)-(13), evaluating \(D\) at \(H_+\).
Exact rational substitution proves (12) and

\[
R<1/11700,\quad W_B<1/1450,\quad W_J<1/2900,
\quad W_{\bar\rho}<3/200,\quad W_G<3/50.
\tag{17}
\]

These bounds imply (14) uniformly on (15). This is not a search over
observed responses or a claim that arbitrarily small windows are useful.

Several simple lower bounds on this **sufficient budget** explain its
limits. They are not lower bounds on the information in every observation:

\[
\begin{aligned}
R&\ge144000(X+gY),\\
D&\ge8(1+g^2)X+8gY,\\
d_A&\ge\frac{72\varepsilon}{H^2}
 +\frac{40}{3}(2X+7g)\Lambda.
\end{aligned}\tag{18}
\]

For example, \(X=2^{-30}\) alone gives \(R>1/8200\), so this
budget cannot guarantee the angle target at any horizon. Even with all
other uncertainties zero, \(\Lambda=2^{-16}\) gives
\(W_{\bar\rho}\ge(80/(3k_0))7g\Lambda>1/64\).
These are uniform-source or uniform-clock budget floors; they do not prove
that every source or clock in the corresponding class is indistinguishable.

Noise and finite flow also impose competing horizon restrictions. From
\(Q_s\ge28gH\), \(M_2\ge14g\), and (16),

\[
R\ge4368000gH.
\]

The angle target requires \(R<1/8200\); the mean-rate target and (11)
require \(H^2>9216\varepsilon/k_0\). Thus simultaneous certification
requires

\[
\sqrt{9216\varepsilon/k_0}<H<
H_{\max}:=\frac1{8200\cdot4368000g}
=\frac{1023}{11939200000}.
\tag{19}
\]

If

\[
\varepsilon\ge\frac{k_0H_{\max}^2}{9216}
=\frac{116281}{2627380162068480000000000000},
\tag{20}
\]

there is no horizon satisfying this budget's two targets. In particular,
\(\delta=t=2^{-75}\) satisfies (20), even with \(X=Y=\Lambda=0\).
This result only rules out this sufficient width guarantee. It does not
show that another sharper enclosure or an actual record cannot resolve
the parameters.

<a id="sine-aperture-resolution-noise-ambiguity"></a>
## Genuine bounded-error ambiguity for two complete histories

A different argument proves an information obstruction for an explicit
subfamily. Fix the same admitted nominal \(b,c\), common means, zero
centered residuals \(u=v=0\), fixed events, kernels and held offset.
Compare the two constant-clock models

\[
(G_1,\rho_1)=(3/2,1),\qquad
(G_2,\rho_2)=(1,3/2),\qquad J_1=J_2=3/2.
\tag{21}
\]

They lie within (2) for every horizon in (4), and their derivative is zero,
so any nonnegative drift allowance admits them. Their initial full nodal
states agree. This subfamily is not an equilibrium continuum: the nominal
state may have port forcing and its whole state evolves.

Let \(\phi_j=\theta_0+a_jq\) denote the nominal phase after the
corresponding cumulative input, and set \(v_j=\gamma q^Tf(\phi_j)\).
The common first-order sensor signal is the broken affine function

\[
y_{\rm lead}(s)=O+J\,[v_1\min(s,H)+v_2(s-H)_+].
\tag{22}
\]

It includes the first-window history in the second window. No state is
reset at \(H\), and (22) is used only as a reference for an explicit
full-flow error bound, not as the measured response.

Heat-semigroup contraction and \(\|f\|_M<7\) give
\(\|P_Mx(\tau)\|_M\le7g\tau\). Since the declared phase jump
is exactly the jump of \(\phi_j\), the continuous phase deviation from
the piecewise nominal phase accumulates throughout both windows:

\[
\|\theta(\tau)-\phi_j\|_M\le7g^2\tau^2.
\]

Common phase means are included in \(\theta_0\). By the norm bound
\(\|A\|_M\le2\), sine Lipschitz constant two and
\(\|q\|_{M^{-1}}=1\), on each continuous segment

\[
|q^Tx_\tau-v_j|\le14g\tau+14g^3\tau^2.
\]

Form continuity permits integration across the phase event. For either
complete history this proves

\[
|Gq^Tx(s)+O-y_{\rm lead}(s)|
\le G\left[7g\rho^2s^2+\frac{14}{3}g^3\rho^3s^3\right].
\tag{23}
\]

Every averaging kernel is nonnegative. The largest second and third
time moments among the four windows occur on \([H,2H]\), where
they are \(7H^2/3\) and \(15H^3/4\). In (21),
\(\sum G_i\rho_i^2=15/4\) and \(\sum G_i\rho_i^3=39/8\).
Since the leading signal is common, the two exact noiseless average
vectors \(m^{(1)},m^{(2)}\in\mathbb R^4\) therefore obey

\[
\max_j|m_j^{(1)}-m_j^{(2)}|
\le\frac{245}{4}gH^2+\frac{1365}{16}g^3H^3.
\tag{24}
\]

Consequently, if the **sensor** allowance satisfies

\[
\boxed{\delta\ge B_{\rm amb}(H):=
\frac{245}{8}gH^2+\frac{1365}{32}g^3H^3,}
\tag{25}
\]

the common exact recorded vector
\(z=(m^{(1)}+m^{(2)})/2\) is admissible for both complete models:
take their four sensor errors as \(z-m^{(i)}\). Each error has magnitude
at most \(\delta\). These are the four errors of the declared averaged
sensor, not an invented pointwise noise process. They differ between the
two alternative unknown realizations, as allowed by the bounded-error
observation model. Exact recorded point intervals have \(t=0\), hence
fit every nonnegative declared numerical half-width allowance.

The mathematical midpoint need not be rational. Since \(\gamma<g\)
and \(H>0\), the bounds used in (23)-(24) have strict slack for these
two histories. At the threshold in (25), each pair of admissible sensor
intervals therefore has an overlapping interior. A common rational value
can be selected in each interior instead of the exact midpoint, so
rational singleton reading intervals with \(t=0\) also realize the
ambiguity. This is an existence argument, not a computed acquisition.

Thus there exists an identical four-average record for two fully evolving
histories whose gain and first-window mean rate each differ by \(1/2\).
Any uniformly valid covering inference for that record must retain both;
it cannot guarantee gain or mean-rate marginal width below \(1/2\).
This is a genuine bounded-error obstruction, not merely equality of a
leading coefficient or overlap of two loose response intervals. It is
existential: it does not say that every record at that error level is
ambiguous. For any fixed \(\delta>0\), sufficiently small positive \(H\)
satisfies (25), so arbitrarily short acquisition cannot uniformly recover
these parameters at fixed absolute noise.

## Implementation and remaining boundary

The [detached budget calculation](../../src/tnfr/physics/relational_sine_aperture_budget.py)
consumes only (4), the fixed model/prior constants and policy targets.
The [independent controls](../../tests/physics/test_sine_aperture_budget.py)
check the moment/error algebra, uniform horizon budget and distinct
ambiguity threshold. The calculation consumes no reading, saved response,
inverse report, hidden preparation or producer certificate. A positive
sufficient budget establishes a conditional mathematical regime;
unavailable or failed sufficient bounds must remain distinct from (25).

There is a deliberate gap between the sufficient regime (15), the budget
obstructions (18)-(20), and the genuine ambiguity condition (25). These
arguments do not determine the sharp information threshold. Improving a
bound cannot erase the explicit indistinguishable pair when (25) holds.
Nor does a successful algebraic budget establish empirical attainability
of its source, clock, gain, aperture or error premises.

The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns further admission. Physical application still needs independently
justified preparation, observation/calibration and time-unit maps under
the same complete model. No new response campaign or physical-data claim
is implied by this theoretical resolution audit.
