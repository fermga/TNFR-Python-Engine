# Finite-aperture observations and mean-clock inference

<a id="sine-aperture-inference"></a>

## Question and conditional result

A sensor that averages a response over time has a different observation
law from the point sensor in the [clock-drift owner](SINE_CLOCK_DRIFT_INFERENCE.md#sine-clock-drift-inference).
Replacing such an average by an endpoint value can introduce a first-order
bias, large compared with the second-order curvature signal. This owner
specifies four positive boxcar kernels and derives necessary inference
from their averages without shrinking their widths relative to the probe.

The first three apertures each span one third of the first window; the
fourth spans the complete second window. Exact polynomial moments give
an algebraic reconstruction whose first-window error is cubic in duration.
The second-window endpoint needs only a quadratic error bound. A separate
clock-transfer bound permits a positive varying observation clock while
retaining the first-window mean-rate target. The resulting adapter reuses
one existing curvature inverse and preserves the original sensor errors
explicitly.

These are conditional mathematical results and fresh implementation
controls. They introduce no reserved response, physical calibration or
reinterpretation of frozen point-reading evidence. The
[ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization) retains
the distinction between collective observation and physical identification.

<a id="sine-aperture-source-and-law"></a>
## F1-F2: complete dynamics and a fixed averaging law

Retain the [complete source, support and clock class](SINE_CLOCK_DRIFT_INFERENCE.md#sine-clock-drift-source-and-law).
On the eighteen-node two-port support, with degree matrix \(M\),
\(A=M^{-1}L\), \(q=e_4-e_5\), held unit capacities and
\(\gamma=1/(1023\pi)\), the structural rows are
\[
x_\tau=-Ax+\gamma f(\theta),\qquad \theta_\tau=\gamma Ax.
\tag{1}
\]
Both rows multiply by the same positive \(C^1\) rate \(\rho(s)\)
in observation time. Its supplied premises remain
\(\rho_-\le\rho(s)\le\rho_+\), \(|\rho'(s)|\le\Lambda\)
on \([0,2H]\), with \(H>0\), \(\Lambda\ge0\), and
\(h_*:=\rho_+H\le1/2\). The target is
\(\bar\rho_1=H^{-1}\int_0^H\rho(s)\,ds\).
The full source retains arbitrary common means and centered residuals
bounded by \(X,Y\), original nominal geometry priors and their acute
admission. The actual angle remains \(B_{\rm initial}=b-(v_1-v_0)/8\).

There are phase-only jumps \(a_1q\) at zero and
\((a_2-a_1)q\) at \(H\), for \(0<a_1\le a_2\le1\).
Every form coordinate is continuous at these events. No additional event,
prehistory, state reset, capacity change or support change is supplied.

Define the observed-time apertures and normalized densities
\[
\begin{aligned}
W_0&=[0,H/3],& W_1&=[H/3,2H/3],\\
W_2&=[2H/3,H],& W_3&=[H,2H],
\qquad k_i(s)=\mathbf1_{W_i}(s)/|W_i|.
\end{aligned}\tag{2}
\]
The four measured quantities are
\[
\mathcal A_i=\int_{W_i}k_i(s)\,[Gq^Tx(s)+O]\,ds+\eta_i,
\qquad |\eta_i|\le\delta,
\tag{3}
\]
with one held positive gain \(G\in[G_-,G_+]\) and offset \(O\).
The primitive intervals \([\ell_i,u_i]\) enclose these four quantities.
Their numerical widths differ from the supplied sensor-error allowance.
There is one error per averaged channel; no pointwise noise process or
independent error at each reconstructed point is assumed.

The kernels are nonnegative, normalized, fixed in observation time and
independently supplied. The adapter does not calibrate them or accept
arbitrary windows under the same name. All support lies inside the declared
horizon. No aperture straddles the phase event; its measure-zero endpoint
does not affect an average of continuous form.

### Why a direct endpoint substitution is insufficient

A nonnegative normalized measure supported in \([0,2H]\) reproduces
every affine function at zero only if its first moment is zero. Positivity
then forces all its mass to be at zero. The corresponding statement at
\(2H\) follows by reflection. Thus a genuine endpoint aperture generally
has first-order timing bias; normalization alone does not remove it.

Nor does symmetry automatically remove event bias. Near a phase event,
a continuous readout can have
\(y(H+t)=y(H)+v_-t+(v_+-v_-)t_++O(t^2)\).
A symmetric kernel cancels the affine term but retains
\((v_+-v_-)\int t_+k(t)\,dt\). In (1), a nonzero slope jump can
arise from the changed phase pressure while form itself stays continuous.
The old smooth remainder must not be extended across that jump.

The finite-sample [Peano and timing method](SINE_ENVIRONMENTAL_MEMORY.md#sine-finite-sample-admission)
is a useful mathematical precedent, with different source and event
premises. Its coefficients and verdicts do not transfer to this sensor.

<a id="sine-aperture-moment-reconstruction"></a>
## Exact moment reconstruction at fixed relative aperture

First consider a scalar function \(y(s)\) and its four noiseless
averages under (2). In normalized time \(z=s/H\), the first three
averages of \(1,z,z^2\) form the matrix
\[
V=\begin{pmatrix}1&1/6&1/27\\1&1/2&7/27\\1&5/6&19/27\end{pmatrix}.
\]
This invertible moment map gives the reconstruction
\[
P=\begin{pmatrix}
11/6&-7/6&1/3&0\\
-1/24&13/12&-1/24&0\\
1/3&-7/6&11/6&0\\
-1/3&7/6&-11/6&2
\end{pmatrix}.
\tag{4}
\]
The first three rows reproduce quadratic values at \(0,H/2,H\).
The last row is \(2\mathcal A_3-(P\mathcal A)_2\): it estimates
the second endpoint by a trapezoid relation on its own continuous window.
Every row sums to one, so a held offset remains one common offset.
The negative reconstruction coefficients are algebraic operations on
positive sensor averages, not negative physical kernel weights.

The curvature row and absolute row sums are
\[
(1,-2,1,0)P=(9/4,-9/2,9/4,0),\qquad
\|P_{i,:}\|_1=(10/3,7/6,10/3,16/3)_i.
\tag{5}
\]
Consequently constant and linear terms cancel exactly in the reconstructed
curvature while a quadratic retains its \(H^2\) signal. This moment
identity does not claim exact reconstruction of an arbitrary trajectory.

<a id="sine-aperture-clock-transfer"></a>
## Finite transfer of the averages to one mean-clock reference

Use the same source, phase events and held sensor with constant rate
\(\bar\rho_1\) as a reference. Its full state agrees exactly with
the actual history at zero and \(H\), including the post-event state.
The drift owner's [global contraction bound](SINE_CLOCK_DRIFT_INFERENCE.md#sine-clock-drift-finite-transfer)
supplies, with \(g=1/3069>\gamma\),
\[
Q_0=X+14gh_*,\qquad U=2Q_0+7g,
\qquad \|P_Mx\|_M\le Q_0,\quad \|x_\tau\|_M\le U.
\tag{6}
\]
The bounds cover both complete histories and the comparison arcs.
Phase evolves throughout; its forcing is bounded globally.

For \(0\le s\le H\), write the exposure discrepancy as
\[
\tau(s)-\bar\rho_1s
=\frac1H\int_0^s\int_s^H[\rho(u)-\rho(v)]\,dv\,du.
\]
For the second window, use
\(\tau(H+t)-\tau(H)-\bar\rho_1t
=H^{-1}\int_0^t\int_0^H[\rho(H+v)-\rho(u)]\,du\,dv\).
The derivative and global range \(\Delta\rho=\rho_+-\rho_-\)
therefore give
\[
\begin{aligned}
|\tau(s)-\bar\rho_1s|&\le
 \min\{\Lambda s(H-s)/2,\Delta\rho s(H-s)/H\},\\
|\tau(H+t)-\tau(H)-\bar\rho_1t|&\le
 \min\{\Lambda t(H+t)/2,\Delta\rho t\}.
\end{aligned}\tag{7}
\]
Integrate each bound against its nonnegative kernel. With
\((c_0,c_1,c_2)=(7,13,7)/108\), sufficient averaged exposure bounds are
\[
d_i=\min\{c_i\Lambda H^2,2c_i\Delta\rho H\}\ (i<3),
\qquad d_3=\min\{5\Lambda H^2/12,\Delta\rho H/2\}.
\tag{8}
\]
Taking the minimum after integration remains a valid upper bound.
Since \(\|q\|_{M^{-1}}=1\), the actual/reference noiseless sensor
averages differ by at most
\[
b_i=G_+Ud_i.\tag{9}
\]
This step transfers averages before reconstructing reference points. It
does not substitute the mean rate into each actual window, differentiate
\(\rho'\), or assume that a \(C^1\) clock has a bounded second
derivative.

<a id="sine-aperture-reconstruction-error"></a>
## Complete-law reconstruction errors without crossing an event

The contraction bound, \(\|A\|_M\le2\),
\(\|Df\|_M\le2\), and
\(\|D^2f[h,k]\|_M\le2\|h\|_M\|k\|_M\) give
global structural derivative bounds on each continuous window:
\[
\begin{aligned}
M_2&=4(1+g^2)Q_0+14g,\\
M_3&=8(1+2g^2)Q_0+28g(1+g^2)+8g^3Q_0^2,\\
\|x_{\tau\tau}\|_M&\le M_2,\qquad
\|x_{\tau\tau\tau}\|_M\le M_3.
\end{aligned}\tag{10}
\]
For example, differentiating (1) yields
\(x_{\tau\tau}=-Ax_\tau+\gamma Df\,\theta_\tau\)
and
\(x_{\tau\tau\tau}=-Ax_{\tau\tau}
+\gamma D^2f[\theta_\tau,\theta_\tau]
+\gamma Df\,\theta_{\tau\tau}\).
These bounds include every form and phase coordinate and use no frozen
phase approximation. Their application below is to the constant-clock
reference, whose observed derivatives introduce powers of \(\bar\rho_1\).

For the first three reconstruction rows, define the order-three Peano
kernel on \([0,1]\) by
\[
K_i(t)=\sum_{j=0}^2P_{ij}\,3\int_{j/3}^{(j+1)/3}
                 (z-t)_+^2/2\,dz-(z_i-t)_+^2/2,
\quad (z_0,z_1,z_2)=(0,1/2,1).
\]
Exact polynomial integration gives
\[
\int_0^1|K_i(t)|\,dt=(1/108,1/2304,1/108)_i.
\tag{11}
\]
One explicit verification uses the nonnegative endpoint kernel
\[
K_0(t)=
\begin{cases}
t^2/2-11t^3/12,&0\le t\le1/3,\\
(1-t)^3/6-3(2/3-t)^3/4,&1/3\le t\le2/3,\\
(1-t)^3/6,&2/3\le t\le1,
\end{cases}
\]
and \(K_2(t)=-K_0(1-t)\). For \(t\ge1/2\),
\(K_1(t)=-(1-t)^3/48+(9/16)(2/3-t)_+^3\le0\);
its reflection has opposite sign. Integration establishes (11), rather
than bounding unrelated endpoint Taylor expansions independently.

On the second window the exact trapezoid error satisfies
\[
\left|\frac2H\int_H^{2H}y(s)\,ds-y(H)-y(2H)\right|
\le H^2\sup_{[H,2H]}|y''|/6.
\tag{12}
\]
Form continuity identifies the two values at \(H\); derivatives are
taken on the appropriate side. Combining (10)-(12), sufficient recorded
reconstruction allowances are
\[
e_0=e_2=G_+M_3h_*^3/108,\qquad
e_1=G_+M_3h_*^3/2304,\qquad
e_3=e_2+G_+M_2h_*^2/6.
\tag{13}
\]
Only the first three entries enter curvature. Their combined error is
\((67/3456)G_+M_3h_*^3\), retaining cubic order at the fixed
aperture fractions in (2). The quadratic last-entry error affects the
coarse two-input inference, not the first-window curvature directly.

<a id="sine-aperture-necessary-inference"></a>
## Necessary inference and the original four errors

Re-admit all eleven primitives. Apply the exact rational matrix \(P\)
to the four primitive average intervals before outward interval
materialization. Let this linear interval image be \(J_i\). The
constant-mean reference's **noiseless** point readings at \(0,H/2,H,2H\)
belong to
\[
\widehat I_i=J_i+[-a_i,a_i],\qquad
a_i=e_i+\sum_j|P_{ij}|b_j+\delta\sum_j|P_{ij}|.
\tag{14}
\]
The raw numerical half-width contributes through \(J_i\); the three
terms in \(a_i\) respectively account for reconstruction, clock transfer
and the original sensor errors. Offset cancellation occurs exactly in
normalized rational arithmetic, even for a large common offset.

Invoke the [existing curvature inverse](SINE_FINITE_CURVATURE_INFERENCE.md#sine-curvature-necessary-projections)
once with \(\widehat I_i\), the unchanged source/input/gain/rate priors,
observed duration \(H\), and child `readout_error_bound=0`.
The child's zero means no additional error is attached to these auxiliary
noiseless-point bands; the measured averages retain their original
\(\delta\). Its held clock denotes \(\bar\rho_1\), and
its effective gain denotes \(J_1=G\bar\rho_1\).

Every compatible averaged history induces such a reference history, so
the resulting marginals are necessary bounds on the original geometry,
held gain and first-window mean. A strict exclusion rules out the admitted
averaged model. Nonempty rectangular constraints need not have a jointly
realizable source, clock or error vector. The implementation retains the
shared four-error map \(P\), the curvature row (5), and the coarse
inverse rows composed with \(P\); independent interval projection can
discard their remaining correlations.

For curvature specifically, there is no additional sensor support loss
from the rectangular projection: its propagated allowance is
\((10/3+2\cdot7/6+10/3)\delta=9\delta\), exactly the support
of the composed row in (5). This equality does not extend automatically
to the other rows or certify joint feasibility.

The report distinguishes original numerical widths, projected sensor
allowances, clock-transfer candidates and reconstruction candidates.
Certified upper bounds require the freshly rebuilt child's original-source
admission. `bounded_candidate`, `incompatible` and `unavailable` retain
their necessary-constraint meanings; no unavailable quantity becomes zero.
The inferred clock field explicitly names the first-window mean.

<a id="sine-aperture-conditioning"></a>
## Informative finite-width regime without selecting a response

Use the previous positive-drift budget
\[
H=2^{-24},\quad X=Y=2^{-48},\quad (a_1,a_2)=(1/4,3/4),\quad
G\in[1,2],\quad\rho\in[1/2,2],\quad
\Lambda=2^{-22},\quad\delta=2^{-90},\quad t_i\le\delta.
\tag{15}
\]
Here \(t_i\) are numerical half-widths of the four original averages,
separate from their four bounded errors. No source realization or response
is selected. The apertures remain \(H/3,H/3,H/3,H\).

Put \(h_*=2H\), \(c=2U\Lambda H^2\),
\(e=2M_3h_*^3/108\), \(e_m=2M_3h_*^3/2304\), and
\(l=2M_2h_*^2/6\). Sufficient virtual-point half-widths are
\[
\begin{aligned}
r_0=r_2&=20\delta/3+91c/324+e,\\
r_1&=7\delta/3+11c/81+e_m,\\
r_3&=r_2+4\delta+5c/6+l.
\end{aligned}\tag{16}
\]
These include both \(t_i\) and \(\delta\). For the separate
coarse-inverse remainder, retain its own specialized quantities
\[
Q_s=\frac{X+2gh_*(7+2Y)}{1-16g^2h_*^2},\qquad
E_*=2h_*Q_s+2gh_*Y+8g^2h_*^2Q_s.
\tag{17}
\]
Do not replace \(Q_s\) by the global \(Q_0\) merely because both
are form bounds. For the [two shared inverse rows](SINE_TWO_PULSE_INFERENCE.md#sine-two-pulse-conditioning), let \(S_j\) be
the sum of their absolute coefficients. Their combined bound is
\(\sqrt{S_1^2+S_2^2}<36000/h_*\).
For a row \(w=(w_1,w_2)\), the three reading-radius contribution is
\(|w_1|r_0+|w_1-w_2|r_2+|w_2|r_3\).
Since \(r_0=r_2\le r_3\), it is at most
\((|w_1|+|w_2|)(r_2+r_3)\). Hence
\[
R\le\frac{36000}{h_*}(2E_*+r_2+r_3)<1/11700,
\quad w_{J_1}\le4R<1/2900,
\quad w_b\le\frac{2R}{1/4-2R}<1/1450.
\tag{18}
\]

The child curvature remainder also keeps its own first-window bound:
\(S_s=5+2Y+8gh_*Q_s<7\),
\(M_{3,s}=8(1+2g^2)Q_s+4g(1+g^2)S_s+8g^3Q_s^2\),
\(\varepsilon_2=4(1+g^2)X+4gY\).
Exact rational substitution gives
\(D=2\varepsilon_2+2HM_{3,s}<10^{-9}<1/800000000\).
The normalized curvature numerator diameter obeys
\[
d_A\le\frac{8(r_0+2r_1+r_2)}{H^2}<1/100000000.
\tag{19}
\]
The [positive natural coefficient bound](SINE_FINITE_CURVATURE_INFERENCE.md#sine-curvature-conditioning)
remains \(K_->k_0=1/18000\), with
\(w_K\le11gw_b/8\). Compatibility and these error bounds preserve
positive quotient numerators. Set \(\beta=w_K/k_0\) and
\(t=2w_{J_1}+\beta+2w_{J_1}\beta\). The same endpoint-width
argument yields
\[
w_{\bar\rho_1}\le2d_A/k_0+(2+D/k_0)t+2D/k_0<1/64,
\qquad w_G\le2w_{J_1}+4w_{\bar\rho_1}<1/16.
\tag{20}
\]
The conservative rational caps in (18)-(19) and the stated cap on \(D\)
already suffice. Also \(w_b+Y/4<1/1024\) and
\(w_{J_1}<1/2048\). These are conditional exact-arithmetic resolution
budgets. Executable outward arithmetic and inverse availability remain
separate checks; the stipulated precision is not a laboratory performance
claim. Fixed relative aperture, rather than an arbitrarily tiny additional
window, is what the moment calculation admits.

<a id="sine-aperture-scope"></a>
## Implementation and remaining boundaries

The [aperture adapter](../../src/tnfr/physics/relational_sine_aperture_inference.py)
uses exact primitive admission, fixed moment coefficients and one fresh
curvature calculation. The [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-aperture-inference)
owns its API; the [controls](../../tests/physics/test_sine_aperture_inference.py)
separate polynomial moment identities, complete-law transfer and averaged
synthetic histories from any reserved evidence.

Point-sample exposure equivalence does not automatically give equal
observed-time averages: the latter consume the response inside the
intervals as well as their endpoints. Thus the earlier cosine-companion
certificate cannot be transferred to this observation law. No converse
claim of clock-profile identification follows either.

General kernels, calibration uncertainty in their timing/weights,
event-straddling apertures, variable sensor gain/offset and physical
clock units require their own premises and bounds. Source formation,
support selection and future maintenance remain separate. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns subsequent admission. No frozen response is averaged, replayed or
reclassified by this theoretical extension.
