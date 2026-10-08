# Finite curvature from one additional reading

<a id="sine-finite-curvature-inference"></a>

## Question and conditional claim

The [uncertain-clock inverse](SINE_CLOCK_INFERENCE.md#sine-clock-inference)
retains necessary bounds on geometry and the effective response scale
\(J=G\rho\), where \(G\) is sensor gain and \(\tau=\rho s\)
converts the observation clock to structural time. Its exact curvature
witness disproves a universal gain/clock symmetry, but supplies no
derivative measurement from noisy samples. Here a fourth scalar reading
inside the first window yields a genuine finite-curvature constraint with
full source and flow error bounds. It can narrow clock and gain separately
under the same supplied complete law.

The result is a necessary interval enclosure. Local identifiability is
proved separately on an ideal fixed-nuisance subfamily; it is not promoted
to global recovery of the full uncertain state. No new reserved response
is evaluated, and the earlier protocols and verdicts remain unchanged.
The [ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization) retains
the distinction between collective observation and physical identification.

<a id="sine-curvature-source-and-observation"></a>
## F1-F2: one full source, two inputs and four readings

Retain the [complete clocked model](SINE_CLOCK_INFERENCE.md#sine-clock-source-and-law):
eighteen nodes in two unit C9 rings, contacts `(0,9)` and `(1,10)`, degree
matrix \(M\), \(A=M^{-1}L\), and \(q=e_4-e_5\). Its structural rows are
\[
x_\tau=-Ax+\gamma f(\theta),\qquad
\theta_\tau=\gamma Ax,\qquad
\gamma=1/(1023\pi),\qquad
f_i(\theta)=M_{ii}^{-1}\sum_{j\sim i}\sin(\theta_j-\theta_i).
\tag{1}
\]
In observation time \(s\), **both** rows multiply by one held unknown
\(\rho\in[\rho_-,\rho_+]\), with \(\rho_->0\). The original full
source has arbitrary common means and centered residuals
\[
x^-=m_x\mathbf1+u,\quad
\theta^-=m_\theta\mathbf1+\theta^0(b,c)+v,\quad
\|u\|_M\le X,\quad\|v\|_M\le Y.
\tag{2}
\]
The nominal geometry and source guard retain their original owners:
\([b_-,b_+]\subseteq[11/8,3/2]\),
\([c_-,c_+]\subseteq[2/3,1]\), continuous lifts and the full weighted
centering. The actual original long-arc mean is still
\(B_{\rm initial}=b-(v_1-v_0)/8\). No derivative of a reconstructed
pressure, ideal target state or discarded intermediate state is supplied.

For \(0<a_1\le a_2\le1\), apply \(a_1q\) to phase at \(s=0\)
and \((a_2-a_1)q\) at \(s=H>0\). Form is continuous at each jump;
all thirty-six coordinates evolve between them. Require
\(h_*:=\rho_+H\le1/2\). The additional reading is passive: it changes
neither state nor events. The four observed times and true recorded values
are
\[
(s_0,s_1,s_2,s_3)=(0,H/2,H,2H),\qquad
r_i=Gq^Tx(s_i)+O+\eta_i,\quad |\eta_i|\le\delta,
\tag{3}
\]
with one held \(G\in[G_-,G_+]\), \(G_->0\), and one held arbitrary
offset \(O\). Each primitive reading interval has exact midpoint \(c_i\)
and half-width \(t_i\); its numerical width is separate from \(\delta\).
The old endpoint inverse consumes indices \((0,2,3)\). It must not mistake
the extra half-time reading for the old middle endpoint.

The finite contrast is
\[
C_r=r_2-2r_1+r_0,\qquad
m_C=c_2-2c_1+c_0,\qquad
\varepsilon_C=t_0+2t_1+t_2+4\delta.
\tag{4}
\]
Compute \(m_C\) exactly before outward interval arithmetic: its offset
cancels. Then \(I_C=[m_C-\varepsilon_C,m_C+\varepsilon_C]\) encloses
the **noiseless** sensor contrast. The fourth reading has zero coefficient
in (4), but enters the old two-input bound on \(b,J\).

<a id="sine-curvature-full-state-error"></a>
## F3: initial curvature and a complete third-derivative bound

All norms below use \(M\). On this support,
\(\|A\|\le2\), \(\|q\|_{M^{-1}}=1\), and
\(\|Df(\theta)\|\le2\). The additional bound needed here is
\[
\|D^2f(\theta)[w,z]\|_M\le2\|w\|_M\|z\|_M.
\tag{5}
\]
For an oriented incidence matrix \(D\),
\(\|M^{-1/2}D\|\le\sqrt2\). Every degree is at least two, so
\(|w_i-w_j|\le\sqrt{1/d_i+1/d_j}\,\|w\|_M\le\|w\|_M\).
Use this edgewise maximum in the incidence representation of the Hessian,
whose sine weights have absolute value at most one, and bound the remaining
edge difference norm by \(\sqrt2\|z\|_M\). This proves (5) globally,
without an acute-trajectory assumption.

Set \(a=a_1\) and \(\phi=\theta^0(b,c)+aq\). At the original
post-event state, differentiating **both** rows of (1) gives
\[
x_{\tau\tau}(0^+)=A^2u-\gamma Af(\phi+v)
 +\gamma^2Df(\phi+v)Au.
\tag{6}
\]
The nominal scalar curvature is
\[
\begin{aligned}
K(b,a)&=-\gamma q^TAf(\phi)\\
 &=\frac\gamma2[4\sin(b+a)-3\sin(b-2a)-\sin b].
\end{aligned}\tag{7}
\]
Only rows three through six enter \(q^TAf\); other rows retain their
actual noncritical values. The initial full-state error satisfies
\[
|q^Tx_{\tau\tau}(0^+)-K(b,a)|\le
\varepsilon_2:=4(1+g^2)X+4gY,\qquad g=1/3069>\gamma.
\tag{8}
\]
Indeed, the form terms in (6) contribute at most \(4X+4g^2X\),
and the sine-map difference followed by \(A\) contributes at most
\(4gY\). A nonzero initial form residual cannot be dropped because the
nominal constant-form case has zero initial phase rate.

For stable positive coefficient bounds, rewrite (7) as
\[
\begin{aligned}
P(a)&=2\sin^2(a/2)(1+3\cos a),\\
R(a)&=\sin a(2+3\cos a),\\
K(b,a)&=\gamma[P(a)\sin b+R(a)\cos b]>0.
\end{aligned}\tag{9}
\]
Both coefficients and both trigonometric coordinates are positive on the
admitted domain. This avoids subtracting nearly equal sine values when
\(a\) is small. Numerical positivity still needs an outward lower bound
strictly above zero; an unresolved bound is not a zero physical curvature.

Reuse the freshly rebuilt maximum-history form bound
\[
Q=\frac{X+2gh_*(4+4a_2+2Y)}{1-16g^2h_*^2}.
\tag{10}
\]
It retains the complete first-to-second-window history. On the first
window, the actual phase is \(\phi+v+d(\tau)\), with
\(\|d\|_M\le4gh_*Q\). Therefore a valid first-window forcing bound is
\[
S=\min\{7,\ 4+4a_1+2Y+8gh_*Q\}.
\tag{11}
\]
The second bound follows from the nominal forcing bound four and global
Lipschitz constant two. The first follows independently from
\(|f_i|\le1\) and \(\sum_i d_i=40<49\).

Writing \(U=2Q+gS\) gives
\[
\|x_\tau\|\le U,\quad
\|\theta_\tau\|\le2gQ,\quad
\|x_{\tau\tau}\|\le2U+4g^2Q,\quad
\|\theta_{\tau\tau}\|\le2gU.
\]
Differentiating once more,
\[
x_{\tau\tau\tau}=-Ax_{\tau\tau}
 +\gamma D^2f[\theta_\tau,\theta_\tau]
 +\gamma Df\,\theta_{\tau\tau}.
\]
Consequently
\[
\boxed{\|x_{\tau\tau\tau}\|_M\le
M_3:=8(1+2g^2)Q+4g(1+g^2)S+8g^3Q^2.}\tag{12}
\]
These are complete-law derivatives on the first continuous segment. No
Taylor expansion crosses a phase jump. The endpoint \(x(H^-)\) equals
\(x(H^+)\), so the reading at the second event remains admissible in (4).

<a id="sine-curvature-finite-contrast"></a>
## A finite difference, not assumed access to derivatives

For any twice differentiable scalar \(y\), the exact identity
\[
y(h)-2y(h/2)+y(0)
 =\int_0^{h/2}\int_0^{h/2}y''(u+v)\,du\,dv
\tag{13}
\]
has a positive kernel of area \(h^2/4\). If \(|y'''|\le M_3\),
its difference from \(h^2y''(0)/4\) has absolute value at most
\(M_3h^3/8\), since the double integral of \(u+v\) is \(h^3/8\).
Apply this to \(q^Tx\), use (8), and set \(h=\rho H\). The actual
noiseless sensor contrast obeys
\[
\left|C-\frac{G\rho^2H^2}{4}K(b,a_1)\right|
\le G\left[\frac{\rho^2H^2}{4}\varepsilon_2
 +\frac{\rho^3H^3}{8}M_3\right].
\tag{14}
\]
Dividing by the **actual positive** \(J=G\rho\) yields
\[
\left|\frac{4C}{H^2J}-\rho K(b,a_1)\right|
\le D_C:=\rho_+\varepsilon_2+\frac{\rho_+^2H}{2}M_3.
\tag{15}
\]
The source term does not vanish merely by shortening \(H\). Meanwhile
the observation error after normalization scales as \(H^{-2}\).
A shorter window reduces the smooth-flow error but can worsen inference
from noisy readings; no automatic refinement policy is justified.

<a id="sine-curvature-necessary-projections"></a>
## F4: necessary clock and gain refinement with shared observations

Re-admit the four primitive intervals before constructing the old
[clock envelope](SINE_CLOCK_INFERENCE.md#sine-clock-marginal-inference)
from indices \((0,2,3)\). Let its bounds be \(I_b,I_J,I_G,I_\rho\).
Intersect the effective-scale interval with the exact positive prior
\([G_-\rho_-,G_+\rho_+]\) before any quotient. Likewise preserve the
positive primitive clock/gain endpoints when an outward interval lower
bound happens to round to zero.

Enclose \(K(b,a_1)\) over \(b\in I_b\) using (9), obtaining
\(I_K\) with positive lower bound. Interval projection of (4), (15) gives
\[
I_Z=\frac{4I_C}{H^2I_J},\qquad
I_\rho^{\rm new}=I_\rho\cap
 \frac{I_Z+[-D_C,D_C]}{I_K},\qquad
I_G^{\rm new}=I_G\cap\frac{I_J}{I_\rho^{\rm new}}.
\tag{16}
\]
Use exact endpoint products and positive quotients before interval
materialization, particularly for \(H^2\) and positive clock scales.
The result retains the old nominal and original actual-angle bounds;
it does not silently reinvert \(K\) or iterate these projections as a fit.

Every full trajectory and held sensor satisfying the four observations
belongs to all these necessary constraints. The shared errors have not
become independent realizations: if \(w=(w_1,w_2)\) is an old inverse
row, its four reading coefficients are
\[
(w_1,0,w_2-w_1,-w_2),
\]
whereas the curvature coefficients are \((1,-2,1,0)\). Both act on the
same four bounded errors. The separate interval constraints conservatively
discard correlations between \(I_J\), \(I_C\), geometry and state.
Intersection is sound, but nonempty intersections or their Cartesian
products do not prove a common error realization or a realizable trajectory.

Strict exclusion gives `incompatible`. Missing base inference, unresolved
curvature positivity or quotient arithmetic gives `unavailable` for the
refinement, retaining the coarse report for inspection. A surviving
`bounded_candidate` reports necessary outer bounds, not independently
calibrated time or gain. Source admission, original-angle meaning and
optional whole-window acuteness retain their separate scopes.

<a id="sine-curvature-conditioning"></a>
### An informative finite-error regime without evaluating a response

The following separately declared arithmetic example shows that the
finite bounds can distinguish clock and gain, rather than only their
product. It selects no source realization or recorded values:
\[
\begin{gathered}
H=2^{-24},\quad X=Y=2^{-48},\quad (a_1,a_2)=(1/4,3/4),\\
G\in[1,2],\quad\rho\in[1/2,2],\quad
\delta=2^{-90},\quad t_i\le\delta.
\end{gathered}\tag{17}
\]
Keep the full original angle priors. Then \(h_*=2^{-23}\).
The [shared conditioning argument](SINE_CLOCK_INFERENCE.md#sine-clock-conditioning)
gives coordinate radius
\[
R_*:=72000E(h_*)/h_*+144000\delta/h_*<1/12500.
\]
The auxiliary radial prior is still \([1/4,2]\). For compatible
observations, the ideal arithmetic enclosures therefore have widths
\[
w_J\le4R_*<1/3000,\qquad
w_b\le\frac{2R_*}{1/4-2R_*}<1/1500.
\tag{18}
\]
Direct rational substitution into (8), (10)-(12), (15) also yields
\(D_C=2\varepsilon_2+2HM_3<10^{-9}\).

The natural positive-coefficient interval (9), even over the whole
original angle prior, admits the uniform lower bound
\(K_->1/18000=:k_0\). For example, the elementary sine/cosine bounds
\[
\sin(1/8)>3/25,\quad\sin(1/4)>6/25,\quad
\cos(1/4)>24/25,\quad\sin b>9/10,\quad\cos b>7/100
\]
imply
\[
P(1/4)\sin b+R(1/4)\cos b>14262/78125,
\qquad \gamma>1/3216,
\]
whose product exceeds \(k_0\). Alternating Taylor bounds give these
inequalities; the degree-six cosine lower bound at \(3/2\), for example,
is \(0.0701171875>7/100\). The same coefficient representation has
\(P(1/4)<1/8\), \(R(1/4)<5/4\); sine and cosine are one-Lipschitz.
Thus its ideal interval width satisfies
\[
w_K\le\frac{11g}{8}w_b.
\tag{19}
\]
This statement concerns the natural positive interval, not an unjustified
replacement of a wider computed interval by a pointwise minimum.

Let \(I_A=4I_C/H^2\). Its diameter is at most
\(d_A=64\delta/H^2=2^{-36}\). Since the actual response belongs to
all the constraints, the conservative lower bounds
\[
(I_A)_-\ge\tfrac12(\tfrac12 k_0-D_C)-d_A>0,
\qquad (I_A)_-/4-D_C>0
\]
also certify positivity of the numerator of the clock quotient. After
intersecting the original priors, \((I_J)_-\ge1/2\). Set
\[
\beta=w_K/k_0,\qquad
t=2w_J+\beta+2w_J\beta.
\]
Positive endpoint division in (16) gives
\[
w_\rho\le\frac{2d_A}{k_0}
 +(2+D_C/k_0)t+\frac{2D_C}{k_0}<\frac1{80}.
\tag{20}
\]
To see the width estimate, before prior intersection the clock quotient
width equals
\[
\frac{d_A}{J_-K_-}
 +\frac{A_-}{J_+K_+}
   \left(\frac{J_+K_+}{J_-K_-}-1\right)
 +D_C\left(\frac1{K_-}+\frac1{K_+}\right)
\]
when \(d_A\) is the actual numerator diameter, and is bounded by this
expression when \(d_A\) is its allowance. Compatibility gives
\(A_-/(J_+K_+)\le2+D_C/K_+\), and the product ratio minus one is at
most \(t\). This proves (20). The final strict inequality follows even
after replacing \(w_J,w_b,D_C\) by \(1/3000,1/1500,10^{-9}\).
Similarly, \(\rho_-\ge1/2\) and compatibility with \(G\le2\) give
\[
w_G\le2w_J+4w_\rho<1/16.\tag{21}
\]
The original actual-angle width remains at most \(w_b+Y/4\).

These are conditional exact-arithmetic budgets, not generated observations.
An executable report still has to admit its outward coefficient and
quotient arithmetic. The noise budget in (17) is newly stated for a
second-difference calculation; it neither recalibrates a laboratory sensor
nor changes the earlier frozen experiments. A reserved numerical test, if
admitted later, must preserve its own first outcome.

<a id="sine-curvature-local-information"></a>
## Why one extra reading supplies new information on an ideal subfamily

The following local theorem justifies the observation design more strongly
than counting unknowns. Fix receiver parameter \(c\), common means,
\(u=v=0\), zero sensor errors and an interior point of the positive
\((b,G,\rho)\) priors. Keep \(a_1<a_2\), both events and the full
continuous law. Allow only these three parameters to vary. In particular,
the unknown geometry remains part of the admitted source family.

Define
\[
A_j(b)=-k_j\cos(b-a_j/2),\qquad
k_j=2\gamma\sin(3a_j/2)>0.
\]
The exact full-flow increments \(D_1=r(H)-r(0)\),
\(D_2=r(2H)-r(H)\) depend smoothly and analytically near \(H=0\)
on this subfamily, including the actual first endpoint and second jump.
With \(J=G\rho\), their duration-normalized map extends smoothly as
\[
(D_1/H,D_2/H)=(JA_1(b),JA_2(b))+O(H).
\tag{22}
\]
The extension includes derivatives with respect to the parameters, by
smooth dependence of the full flow and fixed jump maps. Moreover,
\[
W:=A_1'A_2-A_1A_2'
 =-k_1k_2\sin((a_2-a_1)/2)\ne0.
\tag{23}
\]
For every sufficiently small fixed positive \(H\), the exact two-increment
Jacobian in \((b,G)\), at fixed \(\rho\), is therefore nonsingular.
The implicit-function theorem provides a local curve
\(b(\rho),G(\rho)\) of **exactly equal full-flow increments** through
the chosen interior point. Since \(q^Tx^-=0\), the same held offset
also gives the same original reading. Thus the existing three readings
have an actual local collision family, not just equal leading coefficients.
No numerical horizon for this theorem is claimed.

Adding the finite contrast gives the smooth limiting map
\[
\left(D_1/H,D_2/H,4C/H^2\right)
 \longrightarrow (JA_1(b),JA_2(b),J\rho K(b,a_1)).
\tag{24}
\]
Its Jacobian in \((b,J,\rho)\) has determinant
\[
J^2K(b,a_1)W\ne0.\tag{25}
\]
The inverse-function theorem therefore gives local injectivity for each
sufficiently small fixed positive \(H\). Positive \(\rho\) makes the
coordinate change \(J=G\rho\) regular, so the same statement holds for
\((b,G,\rho)\). The three contrasts are invertible linear combinations
of the three offset-free readings at \(H/2,H,2H\); no derivative access
is assumed. One added scalar reading is thus sufficient and minimal to
remove the exhibited local ambiguity while keeping the original protocol
and this ideal three-parameter subfamily.

Neither theorem grants global injectivity, identification of the receiver
or fine residuals, nor local injectivity after those nuisance coordinates
are also free. Positive residual and measurement allowances use the finite
necessary bounds (8)-(16). The exact common-law-rate/clock equivalence
from the [previous owner](SINE_CLOCK_INFERENCE.md#sine-clock-common-rate-equivalence)
also survives any extra reading of the identical state history if a free
multiplier of every evolution row is admitted. The present law holds that
multiplier fixed.

<a id="sine-curvature-implementation-boundary"></a>
## Implementation and evidence boundary

The [curvature inverse](../../src/tnfr/physics/relational_sine_curvature_inference.py)
reuses the existing clock and two-input owners through freshly admitted
primitives. It adds the fourth reading, the full derivative bounds and
one necessary refinement; it accepts no cached report as evidence. The
[contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-curvature-inference)
owns the input ordering and availability behavior, and the
[tests](../../tests/physics/test_sine_curvature_inference.py) exercise full
graph derivatives, finite contrasts, representation boundaries and fresh
complete-flow controls. Numerical implementation controls are not a
reserved response or an independent physical observation.

This result concerns inference from a supplied source, calibrated input
amplitudes, held affine observation law and fixed normalized-sine dynamics.
It derives neither the source nor its support, event funding, future
maintenance or a unique fundamental law. Any finite response campaign must
first freeze its complete preparation, clock/sensor model, all four reading
times, error budgets and stopping rules in the
[sole execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).


<a id="sine-curvature-reserved-protocol"></a>
## Prospective four-reading software protocol

This section admits a new reserved software evaluation after the conditional
proof above. Its preparation, response generator, public inverse packets and
stopping rules are fixed before any selected response is evaluated. It does
not reinterpret an earlier known-clock record as a new observation. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns this single active gate. The frozen machine-readable protocol and source
archive, under the stem `curvature-inference-v1`, retain the exact constants
and the prospective committed source revision.

### F1-F2: hidden source and declared observation

Use the complete support, full thirty-six-coordinate law and weighted source
chart of (1)-(3). The public prior and supplied input are fixed as
\[
\begin{gathered}
b\in[11/8,3/2],\quad c\in[2/3,1],\quad
G\in[1,2],\quad\rho\in[1/2,2],\\
X=Y=2^{-48},\quad H=2^{-24},\quad
(a_1,a_2)=(1/4,3/4),\quad\delta=2^{-90}.
\end{gathered}\tag{26}
\]
Angles are in radians, \(H\) uses the declared observation clock and
\(\tau=\rho s\) applies to both continuous rows. Gain converts the signed
form readout to recorded units; \(\delta\) and numerical reading radii use
those recorded units. There is no independent laboratory calibration.

The generator receives these three exact cases, indexed by \(k=0,1,2\).
The final column selects the full source residual recipe; it is not a new
coordinate of the model.

| Case \(k\) | \(b\) | \(c\) | \(G\) | \(\rho\) | Recipe \(r\) |
| --- | --- | --- | --- | --- | --- |
| 0 | \(45/32\) | \(3/4\) | \(3/2\) | \(1\) | 0 |
| 1 | \(45/32\) | \(3/4\) | \(6/5\) | \(5/4\) | 0 |
| 2 | \(47/32\) | \(11/12\) | \(7/4\) | \(7/8\) | 2 |

For node indices \(i=0,\ldots,17\), set
\[
\widetilde u_i=\frac{(i+2)(r+2)}{2^{62}},\qquad
\widetilde v_i=\frac{(5i+3r)\bmod23+1}{2^{62}},
\quad
u=\widetilde u-\langle\widetilde u\rangle_M\mathbf1,
\quad v=\widetilde v-\langle\widetilde v\rangle_M\mathbf1.
\tag{27}
\]
Set \(m_x=(r+2)/11\) and
\(m_\theta=-(r+2)/13\), with the nominal phases centered by the same full
weighted mean. Retain every coordinate and the realized exact arrays.
Weighted centering preserves the residual norm bound, and
\(40\cdot76^2/2^{124}<2^{-96}\) admits both squared norms. The recipe
has nonzero residuals at every node. Its original actual long-arc mean is
\[
B_{\rm initial}=b-5/2^{65},\tag{28}
\]
because \(v_1-v_0=5/2^{62}\). The retained initial interval source also
supplies an outer interval for that same initial statistic; it is not an
endpoint mean or an exact recovery of every residual.

Cases 0 and 1 have the same complete source and the same
\(J=G\rho=3/2\). Their gains and clocks differ. Equality of this leading
response scale does not assert equal exact endpoint responses, and these
nonzero-residual cases are not the ideal local-collision family in (22)-(25).

The sensor has one held offset \(O_k=(2k+3)/11\) per case. Its four
realized errors are fixed before response generation:
\[
(\eta_0,\eta_{1/2},\eta_1,\eta_2)
 =2^{-91}(k-1,\ 1-k,\ (-1)^k,\ k-1).
\tag{29}
\]
Each has magnitude at most \(\delta\). There is exactly one recorded
reading at each of \(0,H/2,H,2H\), under the same gain, offset and clock.
Reuse those four primitive intervals in all differences and controls.
The numerical half-width of each retained reading must be at most
\(\delta\), separately from the per-reading error allowance.

### F3: complete uninterrupted response and information separation

Use the [full-state readout producer](../../src/tnfr/physics/relational_sine_two_port_readout.py)
and shared [direct source-box Taylor kernel](../../src/tnfr/mathematics/_validated_taylor.py).
For each case, the three consecutive structural durations and phase jumps
are
\[
(\rho H/2,\rho H/2,\rho H),\qquad(1/4,0,1/2)q.
\tag{30}
\]
The zero entry means continuous continuation with no event at the passive
half-time observation. The first event occurs at observed time zero and
the final event at \(H\). Both are phase-only jumps. Carry each complete
thirty-six-coordinate endpoint interval into the next segment before
applying its declared jump; do not reset, recenter, narrow or reprepare the
intermediate state. Source common means remain part of the same trajectory.

Each segment uses order four, the fixed sixteen-attempt strict Picard
inclusion budget, shared outward dyadic-128 interval arithmetic and the
shared elementary-function enclosures. Retain its initial box, complete
tube, all source Taylor coefficients, fifth-order remainder, full endpoint,
strict inclusion margin and achieved horizon. Local segment time may start
at zero; the record separately retains its global observation and structural
time association. No smaller-step search, precision escalation, endpoint
selection or budget retry is admitted. A positive smooth-law domain flag
establishes neither acute geometry nor maintenance after either pulse.

Construct the sensor readings from the original baseline and the three
complete endpoints. The identical middle readings enter every applicable
increment and the contrast \(C=r_H-2r_{H/2}+r_0\). Offset cancellation and
error correlation must be retained algebraically before interval projection.
The original source boxes, hidden parameters, sensor error realizations and
response certificates stay on the generator/evidence side.

The inverse receives only these ten public keywords:
`bulk_angle_bounds`, `receiver_short_angle_bounds`, `form_radius`,
`phase_radius`, `phase_increments`, `probe_duration`,
`recorded_reading_bounds`, `readout_error_bound`, `readout_gain_bounds`
and `clock_rate_bounds`. In particular, its duration is observed \(H\),
not a hidden structural segment duration. No true \(b,c,G,\rho\), common
mean, offset, realized error, state, response certificate or cached verdict
may enter the inverse. Retain the exact allowlisted packets and their hashes
and invoke the inverse through the separated worker. This is auditable
software information exclusion, not cryptographic secrecy or provenance
authentication.

### F4: fixed stopping criteria and controls

The reserved evaluation passes only if every declared case satisfies all
source, clock, sensor, full-horizon and information-separation checks and
these inference criteria:

- The primary four-reading inverse is available as `bounded_candidate`,
  with its coarse child, finite-curvature bound and positive divisions
  admitted. Its necessary marginals cover the exact nominal \(b\),
  original actual \(B_{\rm initial}\), effective gain \(J\), gain \(G\)
  and clock \(\rho\). The actual-angle marginal also contains the
  retained initial-box enclosure of \(B_{\rm initial}\).
- Strict widths are below `1/1024` for the actual original angle,
  `1/2048` for \(J\), `1/16` for \(G\) and `1/80` for \(\rho\).
  These are marginal resolution checks, not point recovery or joint
  realizability of every retained tuple.
- Cases 0 and 1 have strictly disjoint retained recorded-curvature
  intervals. The comparison uses their actual four-reading records,
  not separately supplied derivative values or fitted curvature.
- Reusing the same readings with false clock prior `[25/32,13/16]` or
  false gain prior `[31/16,2]` leaves the coarse child available but makes
  the curvature refinement `incompatible`. This checks the contribution
  of the additional reading; coarse outer compatibility is not a proof
  of an alternative full trajectory.
- The false angle prior `[11/8,353/256]` returns `incompatible`. The
  equal cumulative amplitudes `(1/4,1/4)` return `unavailable` through
  the leading-rank limitation. The latter is an altered-declaration
  method control, not an evaluated equal-pulse physical alternative.
- The complete phase-blind alternative
  \(x_\tau=-Ax,\ \theta_\tau=\gamma Ax\), with the same source,
  phase events and held sensor/clock priors, is excluded by each retained
  full-window recorded increment. Its necessary band is
  \([-B_{\rm heat},B_{\rm heat}]\), where
  \(B_{\rm heat}=2G_+\rho_+HX+2\delta\). Compare complete interval
  differences with this band; do not silently drop numerical widths.

All three segment margins must be strictly positive and all three declared
horizons complete. The optional whole-window acute flag is retained without
requiring it to pass or interpreting its failure as a proved sector exit.
Neither work allowance nor future identity/recovery is a stopping criterion
for this inference protocol.

Freeze the prospective proof, machine-readable protocol, committed runtime
source, evaluator and isolated inverse worker before the first response.
Retain their archive inventory and hashes together with the complete source,
observations, reports and every stopping result. Preserve the first outcome,
including solver unavailability, failed criteria or export failure. Any
necessary correction requires a separately identified record and may not
rewrite the original prediction or silently replace its first outcome.
Do not rerun an earlier frozen producer or reinterpret an old sensor budget.

Success would establish finite software clock/gain resolution under the
supplied normalized-sine law and held observation model. It would not select
that law, establish a laboratory clock or sensor, eliminate the exact common
all-row-rate/clock equivalence, or identify physical constituents. Any next
research gate requires separate admission in the execution plan.
