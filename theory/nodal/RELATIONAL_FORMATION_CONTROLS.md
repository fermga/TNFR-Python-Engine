# Native validated transit and preparation controls

Validated continuous transit, bounded constitutive robustness and zero/reversed/consensus-form controls; frozen responses retain their original numerical verdicts.

Part of [Native relational exchange admission](RELATIONAL_EXCHANGE_ADMISSION.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

### Validated continuous transit on the exact reflected subsystem

<a id="relational-validated-transit"></a>

The read-only owner
[`physics/relational_transit.py`](../../src/tnfr/physics/relational_transit.py)
checks the original continuous IVP through the
[existing invariant reduction](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-reflected-capture-boundary).
It uses the supplied two-ring support, exact copied/reflected initial state,
unit held capacities and the same positive-coefficient, source-free relational
law. This is a proof computation; it does not advance a live graph or replace
the shared production integrator. The SDK delegate and exact report exporter
retain the full initial admission and every accepted interval step.

With `q=3A-B`, `r=2B-A`, let `d=a/2-b`,

\[
C_0=1+\cos(2a)+\cos(a-b),\quad S_0=-\sin(2a)-\sin(a-b),
\quad u=S_0/C_0,\quad R(u)=\operatorname{atan}(u)/u,
\]

where `R(0)=1`. On the admitted positive-real chamber,

\[
g_0=uR(u)/\pi,\quad H_0^{-1}=R(u)/(\pi C_0),\quad
g_4=d/\pi,\quad H_4^{-1}=[2\pi\cos(a/2)\operatorname{sinc}(d)]^{-1}.
\]

The original proof used the zero-safe analytic series on `abs(u)<=1/2`
and the sinc series on `abs(d)<=1`. These are sufficient numerical domains,
not new physical limits. The current arctangent-ratio owner retains that
same series near zero and, outside it, uses `atan(u)'=u'/(1+u^2)` followed
by formal division when the entire expansion interval avoids zero. A wide
interval crossing zero outside the first domain is still unavailable.
This analytic extension was tested before evaluating the zero-form control;
for example, the already proved consensus-basin state `q=r=0,a=b=1` has
`u` about `-0.5741`, outside the old numerical domain. It is not a new law.
Positive `2*cos(a/2)*cos(d)` together with `abs(d)<=1`
selects exactly the displayed branch for `g_4`. The zero-safe analytic ratios
avoid dividing by a pressure source that crosses zero. The reduced rows are

\[
\begin{aligned}
\dot q&=-eq+(e/2)r+w(3g_0-g_4),\\
\dot r&=(e/3)q-er+w(2g_4-g_0),\\
\dot a&=(w/\beta)qH_0^{-1},\qquad
\dot b=(w/\beta)rH_4^{-1}.
\end{aligned}
\]

The actual frozen form values are binary64 `0.4` and `0.2`, so the exact
initial values are **`q=1+2^(-54)`, `r=0`**, not `(1,0)`. The phase value is
the exact represented rational `875483625981347/562949953421312` at both
coordinates. The earlier ideal-pi/rational-fifths short-crossing theorem is
not silently substituted for this different IVP.

#### Whole-time enclosure and error propagation

All arithmetic bounds use rational endpoints rounded outwards to a fixed
128-bit dyadic grid. Mathematical pi and cosine bounds reuse the existing
Machin-series and cosine owners. Their optional higher-precision path leaves
the production cosine defaults unchanged. Sinc and arctangent-ratio series
bound every normalized derivative through the requested order; a scalar
series remainder alone would not validate time derivatives.

For a current box `Y`, an attempted compact convex tube `B` must satisfy

\[
Y+[0,h]F(B)\subset\operatorname{int}B,
\qquad C_0>0,\quad 2\cos(a/2)\cos(d)>0,\quad 2\cos b>0
\quad\text{throughout }B.
\]

The last condition retains the full central-node resultant even though its
row vanishes under reflection. A first-exit argument and smoothness on an
open neighborhood of the tube prove existence and containment throughout
the step. This does not require a Picard contraction or identify an Euler
chord with a continuous trajectory. Inflation constructs a candidate tube;
only the strict displayed inclusion admits it.

Let `c` be the exact midpoint of `Y`. Generate normalized flow coefficients
`a_j` recursively by `j*a_j = [s^(j-1)]F(sum_l a_l*s^l)`. The center solution
at the endpoint is enclosed by

\[
P_h(c)+h^{p+1}a_{p+1}(B),\qquad
P_h(c)=\sum_{j=0}^{p}h^j a_j(c).
\]

The last coefficient is evaluated over the **entire tube**, bounding the
normalized derivative at every possible remainder point. Outward arithmetic
also encloses the represented midpoint and all polynomial operations.

To avoid discarding the linear damping while propagating earlier uncertainty,
form a Metzler comparison matrix from a first-order interval jet of the
same vector field:

\[
M_{ii}\ge\sup_B\partial_iF_i,\qquad
M_{ij}\ge\sup_B|\partial_jF_i|\quad(i\ne j).
\]

The diagonal keeps its sign. Both the center solution and every solution
starting in `Y` remain in the convex tube, so the upper-Dini comparison
inequality gives componentwise separation at most `exp(h*M)*rad(Y)`.
The shared comparison kernel shifts the diagonal to a nonnegative matrix,
uses a positive rational series with a norm-bounded tail, and encloses the
compensating scalar exponential. Adding this propagated radius to the center
Taylor enclosure gives `Y_next`. Its intersection with `B` remains valid
because both sets already enclose the endpoint. No midpoint projection of
the actual state is performed.

#### Joining transit to maintenance

The endpoint test applies to the whole box, not to a synthetic midpoint
graph. By default it requires `a>2*pi/3`, `a<pi`, `b>0`, `b<pi/2` and

\[
\mathcal E=\frac45(q+r/2)^2+r^2
+\beta[10-2\cos(2a)-4\cos(a-b)-4\cos b]<7\beta.
\]

Exact initial symmetry and uniqueness preserve the invariant slice. These
inequalities therefore join directly to the protected-capture theorem:
the ideal continuation is regular for all future time and converges to the
positive aligned twist modulo its constant common offsets. The proof of
initial winding zero additionally requires exact symmetry and each initial
raw cycle gap strictly inside `(-pi,pi)`. At a positive-rectangle endpoint
only the edge `-2a` gains `2*pi` under principal wrapping; the ring period is
one. No separate numerical estimate of the crossing instant is needed.

The current certificate also accepts `requested_sector=0` or `-1`, and
`None` classifies any admitted one of the three disjoint protected rectangles.
The point and interval certificates evaluate one shared affine-margin ledger;
their precision and input scopes remain distinct. The actual sector, selected
rectangle and all candidate bounds are retained. `positive_rectangle_margins`
remains a compatibility field, not the only gate. Consensus is sector zero,
not an unavailable or falsey target. The original positive-only audit and
its exact archived source remain unchanged.

The caller must declare the horizon, step and order. Unsupported analytic
domains, unresolved tube inclusion or inconclusive endpoint margins return
unavailable evidence, retaining the accepted prefix and first failed tube.
They do not prove that the true trajectory fails. A proof audit of an already
evaluated preparation is explicitly post-evaluation verification; neither a
successful proof nor an improved enclosure rewrites the original frozen
finite-executor prediction.

#### Retained original-IVP proof audit

The [producer](../../benchmarks/relational_transit_proof.py) froze its inputs,
numerical policy, runtime and all 604 source files before this post-evaluation
proof computation. It reused the immutable represented upper-corner seed,
`e=w=1/2`, `beta=nu_i=1`, fixed horizon 32, step `1/8`, Taylor order 12,
128-bit outward intervals, at most 16 Picard inflations per step, and no retry.

All **256** whole-time steps pass. The complete endpoint box lies in `R+`,
and its storage is enclosed in the conservative decimal interval
`[6.97054580874334, 6.97054580874345]`, strictly below 7. Every endpoint
coordinate width is below `7e-15`. Across all tubes, the port, interior and
central relative-resultant real lower bounds exceed `0.98293`, `0.84597`
and `0.03114543`, respectively; every strict Picard inclusion margin exceeds
`1.4389e-7`. Exact rational inequalities and every tube are in the report;
the displayed decimals summarize them. No tube failed and no bound was
repaired by changing the initial state, law, step, order or horizon.

Consequently this exact represented **winding-zero initial state** generates
a winding-one pattern and converges to the maintained aligned twist under
the declared continuous law. This joins formation to maintenance for the
original IVP, rather than restarting the law at a numerical endpoint.

| Separate immutable proof evidence | SHA-256 |
| --- | --- |
| [Frozen proof protocol](../../docs/assets/relational_capture_response/continuous-transit.audit.protocol.json) | `d8913d6af0d40fd1c903eaf24be29fb8ebeb6d3be28d6415d4e8dafb9974db99` |
| [Validated transit report](../../docs/assets/relational_capture_response/continuous-transit.audit.json) | `ac3104c42cf9054968eac770fee5c1b1e174547a408875edac0b635fe04ca781` |
| [Executed source archive](../../docs/assets/relational_capture_response/continuous-transit.audit.source.zip) | `381212374968f32fe59751c752b4dc812844a3309e00f771704a72b31a154c93` |

An independent retained-record audit recomputes the strict self-inclusions,
all resultant bounds, time chain, exact initial seed and whole-endpoint
conditions without re-integrating the trajectory. Tests also bind every
archived source byte, check independent analytic references for the interval
kernels, and preserve the original finite-executor `false` verdict.
This is a computer-assisted conditional mathematical result. The proof is
not a formally verified implementation and does not validate the constitutive
law physically. Initial support, capacity and geometric/storage preparation
remain supplied; neither substrate creation nor generic pattern selection
has been established.

#### Synergy: qualitative robustness beyond exact reflection

Whenever the validated transit above succeeds from a strictly winding-zero
preparation, it also implies a qualitative **open-set** formation result for
the full state. Exact reflection is needed by this proof's reduced enclosure;
it is not thereby necessary for every nearby trajectory to form the pattern.

The reflected capture theorem gives convergence to the aligned positive
twist. Its limiting storage is
`E_*=beta*[10-10*cos(2*pi/5)]`, strictly below the full-state acute-sector
barrier `B_*=beta*[10-5*cos(2*pi/5)-4*cos(3*pi/8)]`. Positivity of their
gap is exact: `5*cos(2*pi/5)>4*cos(3*pi/8)` reduces, by squaring positive
sides, to `176*sqrt(2)>239`, whose squared comparison is `61952>57121`.
The limiting edge gaps are strictly acute. The reference solution therefore
enters the open full-state protected sector at some finite time `T_*`.
This argument does not locate `T_*`; an endpoint in the reflected basin
need not already be in the smaller acute-energy basin.

The reference trajectory is compact and regular on `[0,T_*]`. Smooth
dependence on the initial state pulls that open target basin back to an
open neighborhood of the **original** preparation. All sufficiently nearby
full states remain regular through `T_*` and then converge to the same
positive-twist orbit. For the frozen upper-corner preparation, the smallest
initial antipodal margin is `pi-2*a_0>1/32`, so a sufficiently small phase
perturbation also preserves initial ring winding zero. Intersecting the two
open neighborhoods proves winding-zero formation followed by maintenance
without requiring exact copy or reflection of the perturbed state.

The same conclusion permits small unequal **held positive capacities**:
augment the ODE by `nu_dot=0`, use smooth dependence on this parameter, and
apply the full-state sector theorem, which already admits heterogeneous
positive capacities. Keep the graph, unit conductances, positive `e,w,beta`
and unforced constitutive law fixed. Common form/phase offsets remain neutral
and their limiting values may differ between preparations.

This consequence supplies neither a quantified perturbation radius nor a
uniform entry time. It does not cover arbitrary positive capacities, topology
changes, capacity events, forcing or finite binary64 execution of neighboring
states. It is a conditional dynamical robustness result, not an identification
of the pattern with a physical constituent.

### Formation robustness under small admitted changes of phase law

<a id="relational-formation-law-robustness"></a>

The retained transit also supplies a quantitative comparison with the two
admitted passive completions. This uses the same exact represented initial
state, supplied support, unit held capacities, `e=w=1/2`, `beta=1`, clock
and horizon `T=32`. Only the phase law changes. No reference producer is
rerun, and no changed-law trajectory is sampled.

The target used in this comparison is the **reflected** rectangle
`R+` with storage below 7. The retained endpoint storage is about
`6.9705458087434`, above the full-state acute-sector barrier of about
`6.924181298665`. It would therefore be incorrect to call that endpoint an
admitted shared acute-sector basin. The following symmetry-preserving
extension of the reflected theorem closes the actual endpoint obligation.

#### The reflected barrier also permits a strict-loss class

Keep the native form row and chosen storage, and require a smooth autonomous
phase law preserving the exact ring-copy and simultaneous form/phase
reflection symmetries. On an open regular neighborhood of the relevant
reflected sublevel, suppose `E_dot<=-c*L` for a fixed `c>0`, and require
rest at `q=g=0`. These are whole-law premises, not snapshot observations.
The four boundary costs in [the reflected theorem](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-protected-capture)
depend only on geometry and storage. They still prevent exit from `R+`
when `E<7`, and give a compact regular sublevel. Zero loss forces `q=r=0`;
the unchanged two form rows require `g_0=g_4=0` to remain there. Their
unique solution in `R+` is the aligned twist. LaSalle's argument therefore
gives convergence for this reflected class too. For laws also satisfying
the full-neighborhood premises of [the law-class theorem](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-sector-law-class),
its local Hurwitz argument gives eventual exponential recovery at the acute
target. No full-state convexity is being asserted throughout `R+`.

Both named completions meet these conditions. The `rho*N*g` correction is
odd under phase reversal. In the eta correction, `h(q/(d*sqrt(beta)))` is
odd under form reversal and `g^2` is even under phase reversal. Both respect
node relabeling, ring copying and common shifts, and preserve the simultaneous
reflection. The central rows have `q_3=g_3=0` and remain zero for both laws.
Thus their exact full states retain the derived four-coordinate form, with
the actual common form and phase offsets constant on this preparation.
No mean equation is discarded or externally frozen. The rho law admits
`c=1`; the eta law admits `c=1-eta/2>0`.

#### A corridor around the retained reference tubes

Let `y=(q,r,a,b)` and let `y_0(t)` be the exact reference solution enclosed
by the original 256 whole-time tubes. Inflate each retained tube in every
coordinate by

\[
\delta=\frac1{1024}.
\]

Exact rational comparisons with the retained tube bounds give, throughout
these enlarged boxes,

\[
|q|<\frac{101}{100},\qquad |r|<\frac14,\qquad
C_0>\frac{97}{100},\qquad C_4>\frac{21}{25},\qquad
C_3>\frac{29}{1000}.
\]

Here `C_i=Re(z_i)` for the port, interior and central resultants. Their
phase-coordinate Lipschitz constants are respectively `4,3,2`. Consequently
the retained lower bounds minus `4*delta`, `3*delta` and `2*delta` establish
the displayed strictly positive corridor margins. The enlarged boxes also
satisfy `0<a<pi`, `|a/2-b|<pi/2`, so the same branch has
`g_4=(a/2-b)/pi`. Every complete nodal resultant is regular there, including
the central row absent from the reduced equations.

All comparison norms below are the infinity norm of these four exact
coordinates. Their scaling is fixed by the retained reference preparation;
it is not an inferred physical unit convention. The inverse relations
`A=(2*q+r)/5`, `B=(q+3*r)/5`, together with the retained common offsets,
reconstruct every nodal form and phase. An error bound in these four
coordinates therefore also bounds the full nodal infinity error; this
comparison discards no changing mean or hidden state. The reference field `f_0`
has a Lipschitz constant

\[
L_0=\frac{25}{8}
\]

on each convex enlarged tube. To obtain an independent bound, write
`c_0=97/100`, `c_4=21/25`, `Q=101/100` and `R=1/4`. Then

\[
\begin{aligned}
\|\nabla g_0\|_1&\le\frac4{\pi c_0},&
\|\nabla g_4\|_1&=\frac3{2\pi},\\
H_0^{-1}&\le\frac1{\pi c_0},&
\|\nabla H_0^{-1}\|_1&\le\frac4{\pi c_0^2},\\
H_4^{-1}&\le\frac1{\pi c_4},&
\|\nabla H_4^{-1}\|_1&\le\frac3{\pi c_4^2}.
\end{aligned}
\]

For completeness, when `z=C+iS` and `C>0`, the metric has the zero-safe
representation

\[
H^{-1}=\frac1\pi\int_0^1\frac{C}{C^2+t^2S^2}\,dt.
\]

The integrand's Euclidean gradient in `(C,S)` has norm at most
`1/(C^2+t^2*S^2)<=1/C^2`. The summed phase-direction norms of `z_0`
and `z_4` are at most 4 and 3, proving the inverse-metric derivative
bounds without a division by a possibly zero imaginary part. Substitution
into the four unchanged reference rows, using `pi>3`, bounds their absolute
Jacobian row sums by

\[
\frac{297}{97},\qquad \frac{1079}{582},\qquad
\frac{8350}{9409},\qquad \frac{1325}{3528},
\]

respectively. Each is strictly below `25/8`. This controls the derivative
on the whole enlarged tubes, not merely the saved centers or endpoints.

#### Bounded discrepancy and a first-exit argument

The named phase corrections alter only the reduced `a_dot,b_dot` rows:

\[
\begin{aligned}
f_\rho-f_0&=(0,0,\rho g_0,\rho g_4),\\
f_\eta-f_0&=\left(0,0,
\frac{\eta e}{\pi}h(q/3)g_0^2,
\frac{\eta e}{\pi}h(r/2)g_4^2\right),\qquad
h(s)=\frac{s^3}{1+s^2}.
\end{aligned}
\]

These are their full reduced differences at the same state. The subsequent
form evolution changes through the native phase-to-form feedback. On the
corridor, positive real resultants give `|g_i|<1/2`, and `|h(s)|<=|s|`.
For nonnegative parameters this yields

\[
\|f_\rho-f_0\|_\infty\le\frac\rho2,\qquad
\|f_\eta-f_0\|_\infty\le\frac\eta{64}.
\]

For example, the eta port bound per unit eta is at most `101/7200<1/64`;
the interior bound is at most `1/192<1/64`.

As long as the changed solution remains within `delta` of `y_0(t)`, both
points and their connecting segment belong to the relevant enlarged tube.
If `b_lambda` bounds the displayed whole-field discrepancy, the upper-Dini
comparison with the same exact initial state gives

\[
\|y_\lambda(t)-y_0(t)\|_\infty
\le b_\lambda\frac{e^{L_0t}-1}{L_0}
\le b_\lambda t e^{L_0t}.
\]

The estimate carries continuously across the retained tube boundaries; it
does not restart at a numerical center. Since `L_0*T=100` and
`exp(100)<2^150`, the following separate sufficient parameter ranges keep
the error **strictly below** `delta` through the entire horizon:

\[
\boxed{\quad 0\le\rho\le2^{-164},\qquad
0\le\eta\le2^{-159}\quad}
\]

for the rho and eta laws respectively. These are independent analytic
fallback bounds, before the sharper static comparison below. The exponential
inequality follows, for instance, from `log(2)>2/3`: the integral of `1/t`
on `[1,2]` strictly exceeds its midpoint bound. The worst bounds are
`rho*16*exp(100)<2^-10` and `eta*exp(100)/2<2^-10`.
At a putative first exit they contradict equality with `delta`, so the
assumed corridor containment proves itself. Smoothness and the strict
regular margin give existence and continuation to `T=32`. This is a new
comparison enclosure around the exact reference solution, not a claim that
the changed solution lies inside the old, uninflated tubes.

#### A static logarithmic-norm refinement

The existing interval Jacobian owner also retains the negative diagonal
derivatives rather than replacing them by absolute values. Applying it to
the same enlarged boxes, without integrating either law, gives a Metzler
matrix for each retained time interval:

\[
M^{(k)}_{ii}\ge\sup\partial_i(f_0)_i,\qquad
M^{(k)}_{ij}\ge\sup|\partial_j(f_0)_i|\quad(i\ne j).
\]

Define its nonnegative logarithmic-norm bound

\[
\mu_k=\max\left(0,\max_i\left[M^{(k)}_{ii}
+\sum_{j\ne i}M^{(k)}_{ij}\right]\right).
\]

The first-order interval jets evaluate derivatives on every point of each
inflated tube. Exact rational evaluation over all 256 boxes verifies

\[
\sum_k h_k\mu_k<9.
\]

The detached
[`audit_relational_formation_robustness`](../../src/tnfr/research/relational_formation_robustness.py)
reader binds the three original files to their declared hashes, verifies
the archived source membership and checks the reference formulas before
performing this new static calculation. It retains every comparison matrix,
its logarithmic-norm bound, corridor and endpoint margins, and the separate
law bounds. The [focused controls](../../tests/physics/test_relational_formation_robustness.py)
check the exact comparison arithmetic, endpoint distinction, export and
unavailable/inconsistent evidence paths while forbidding trajectory execution.

The mean-value Jacobian along the segment between the two states obeys the
same bound. At a coordinate attaining the infinity norm of their difference,
the signed diagonal term can be retained in the upper-Dini inequality;
thus `D+error<=mu_k*error+b_lambda`. Nonnegative `mu_k` makes the cumulative
bound apply also at every earlier time. The same exact initial state and
first-exit construction therefore give

\[
\|y_\lambda(t)-y_0(t)\|_\infty
\le b_\lambda t\exp\left(\int_0^t\mu(s)\,ds\right)
\le32b_\lambda\exp(9),\qquad 0\le t\le32.
\]

Using `exp(9)<(11/4)^9<2^14` gives the larger, still conservative,
**certified sufficient ranges**

\[
\boxed{\quad 0\le\rho\le2^{-29},\qquad
0\le\eta\le2^{-24}\quad}
\]

for the separate rho and eta comparisons. Each has uniform error less than
`1/2048`, half the admitted corridor inflation. The elementary estimate
`exp(1)<11/4` follows from its series: the terms through order 3 sum to
`8/3`, and the tail from order 4 is at most `(1/24)/(1-1/5)=5/96`.
The inequality `(11/4)^9<2^14` is an integer comparison. No optimized
logarithmic norm, fitted response or parameter search is needed. These
stronger radii depend on the separately checked static Jacobian sum; the
preceding smaller bounds remain an analytic fallback if that check is
unavailable.

#### Entry and the resulting formation conclusion

Every retained endpoint rectangle margin exceeds `17/100`, and its exact
upper storage bound is below `6971/1000`. The first-exit error is less than
`1/1024`, so the changed endpoint remains in `R+`. On the connecting
endpoint segment, the reduced storage

\[
\mathcal E=\frac45(q+r/2)^2+r^2
+10-2\cos(2a)-4\cos(a-b)-4\cos b
\]

has gradient one-norm at most

\[
\frac{12}{5}Q+\frac{16}{5}R+16
=\frac{2403}{125}<20.
\]

Consequently its changed value is strictly below
`6971/1000+20/1024<7`. The reflected strict-loss theorem just proved now
applies. Each named law in its displayed parameter range takes the unchanged
winding-zero preparation to winding one and then converges to the same
aligned twist. Once close to that acute target, the full-state local
stability argument also applies. Formation followed by maintenance is thus
not isolated to the exact reference phase row within these two admitted
families.

The radii are deliberately small, conservative sufficient bounds from a
uniform Gronwall estimate. They are neither optimal thresholds nor evidence
of robustness over a practically useful parameter range, physical scales or
selection of either law by observation. The theorem evaluates no new
trajectory and changes no frozen source, response, endpoint or verdict.
The exact preparation, coefficients and clock matter; no conclusion for
arbitrary preparations, nonunit capacities, new support, combined corrections
or the production finite-step executor is being supplied here.

For a broader family of full-state phase laws, the existing reference
convergence still gives **qualitative** openness: at some finite `T_*` it
enters the strict full-state acute basin, and sufficiently small uniform
field discrepancies on a compact regular corridor preserve that entry by
continuous dependence. A changed law must independently meet the shared
capture premises after entry. This requires no reflection preservation, but
the present retained horizon-32 endpoint does not locate that acute entry
or quantify its corridor. The explicit radii above therefore belong only
to the two symmetry-preserving comparisons just proved.

### Zero initial form contrast: a prospective phase-to-form discriminator

<a id="relational-zero-form-control"></a>

The prospectively frozen zero-form control changes only the initial form of the successful reference:
`x_i=0` for every node. Its represented phases, supplied two-ring support,
unit capacities, `e=w=1/2` and `beta=1` remain identical. The common producer
[`relational_transit_proof.py --zero-form`](../../benchmarks/relational_transit_proof.py)
freezes a distinct protocol and source archive. No subsequent response is
used to choose another preparation, horizon or outcome gate.

#### What is predicted before evolving the control

For arbitrary uniform initial form `x(0)=m*1` under this unit-capacity law,

\[
\dot x(0)=w g(\theta_0),\qquad \dot\theta(0)=0,\qquad
\ddot\theta(0)=\frac{w^2}{\beta}H(\theta_0)^{-1}B g(\theta_0).
\]

The acceleration is a derivative of the existing two coupled first-order
rows, not an added inertial equation. Therefore zero form contrast need not
be an equilibrium: nonuniform phase pressure can create form contrast and
then change phase through the same feedback. In the selected preparation,
rigorous initial derivative intervals give

\[
\begin{aligned}
0.10885&<\dot q(0)<0.10886,&
-0.24255&<\dot r(0)<-0.24254,\\
\dot a(0)&=\dot b(0)=0,&
0.01730&<\ddot a(0)<0.01732,\\
&&-0.03003&<\ddot b(0)<-0.03001.
\end{aligned}
\]

Although `q` initially grows, the original form coordinates have
`A_dot=w*g0<0` and `B_dot=w*g4<0`: this dynamically generated direction is
different from the supplied positive reference seed. Initial local signs
alone cannot select a terminal basin.

Form Dirichlet storage grows at quadratic order,
`E_D''(0)=w^2*g^T*B*g>0`, while phase storage loses the same leading amount.
For the joint storage,

\[
\dot{\mathcal E}(0)=\ddot{\mathcal E}(0)=0,\qquad
\mathcal E^{(3)}(0)=-2e\left[\frac43\dot q(0)^2+2\dot r(0)^2\right]<0.
\]

Its initial value is about `7.93652606007>7`; dissipation starts at cubic
order. These are static mathematical consequences evaluated before any
control trajectory. They do not imply capture from the initial energy test.

The protocol fixes horizon 32, step `1/8`, order 12 and 128-bit outward
arithmetic, with no retry. Terminal positive twist, consensus, negative twist
and unresolved admission are predeclared alternatives. An admitted basin
settles a conditional asymptotic limit; merely failing to enter one by the
fixed horizon does not. A numerical proof-domain failure remains distinct
from a zero resultant of the actual law. Prepared phase geometry supplies
storage and pressure: none of these claims describes creation from nothing
or emergence of the initial graph.

#### Retained result: transient winding followed by proved consensus

The first frozen control completes all 256 validated steps to time 32, with
no unresolved tube or retry. Its full endpoint box lies in the consensus
rectangle `S`, with joint storage conservatively enclosed in
`[2.11675502876186, 2.11675502876328]`, strictly below 7. Maximum endpoint
coordinate width is below `1.5e-13`. Every tube retains positive port,
interior and central resultants; their respective global lower bounds
exceed `0.96412`, `1.01502` and `0.03098482`. Thus this original continuous
initial state converges to consensus under the same conditional law.

| Preparation, with identical phase/support/capacity/law | Proved limiting basin | Whole endpoint storage upper bound at time 32 |
| --- | --- | --- |
| Retained supplied form contrast | Positive aligned twist | `6.97054580874345 < 7` |
| Zero initial form contrast | Consensus | `2.11675502876328 < 7` |

The source manifest and every saved Picard inclusion, resultant bound,
initial input and terminal inequality were independently checked without
re-integrating either trajectory. The three protected rectangles share one
definition in the point and interval owners, so consensus is explicitly
`target_sector=0` with `terminal_basin_admitted=true`. The positive-pattern
flag remains false for this control; it is not an unavailable outcome.

| Immutable control evidence | SHA-256 |
| --- | --- |
| [Frozen prospective protocol](../../docs/assets/relational_zero_form_response/result.protocol.json) | `920413db8d67dc46c13914bdd550bb097b416ade8dadd0417883b5b5eeca49d1` |
| [Validated response](../../docs/assets/relational_zero_form_response/result.json) | `bb30b7b2ac8812871b7f7667c4a898d79293a4e25e7b96bff13ec29fdf5996c4` |
| [604 fingerprinted source files](../../docs/assets/relational_zero_form_response/result.source.zip) | `dcd7b49d4f596c9703ea191b74fadb0d28c6952d529363ce559b245984dbe8ee` |

A **post-evaluation** inspection of retained endpoint enclosures also proves
both ring windings zero through time `15/8`, one at every saved endpoint
from `2` through `23/4`, and zero from `47/8` through `32`. Raw cycle gaps
telescope; only `-2a` acquires an additional `2*pi` wrap in the middle group.
Consequently continuity gives at least one crossing in each of
`(15/8,2)` and `(23/4,47/8)`. These are retrospective crossing brackets,
not a frozen crossing-time prediction, exact event times or proof of an
uninterrupted winding-one lifetime between every sampled point.

This control therefore generates form contrast and a transient winding-one
configuration, but does **not** maintain the reference pattern. The selected
initial form preparation changes the asymptotic basin. That does not establish
a universal need for initial form contrast: other phase preparations remain
unclassified. Nor does it prove that scalar storage alone chooses a basin;
removing form changes its signed direction as well as its magnitude and energy.
The equal-storage control below separates those two effects.

#### General mechanism and its exact stationary boundary

The initial exchange does not depend on the special two-ring graph. On any
connected fixed unit support admitted by the relational law, let held
`N=diag(nu_i)>0`, regular `H>0`, `w,beta>0` and `x_0=m*1`. Then

\[
\dot x_0=wNg,\qquad \dot\theta_0=0,\qquad
\ddot\theta_0=\frac{w^2}{\beta}H^{-1}NBNg.
\]

Reciprocity gives `1^T H g=0` by cancellation of edge sine contributions.
If `Ng=c*1`, this identity gives `c*sum_i(H_i/nu_i)=0`, hence `g=0`.
For any nonzero phase source, `Ng` is therefore nonconstant, `BNg` is nonzero,
and form contrast and relative phase acceleration arise. An acceleration
proportional to `1` would similarly imply `BNg=c*H*N^-1*1`; summing its
entries forces `c=0`, which excludes nonzero acceleration without relative
phase change. Furthermore,

\[
\ddot E_D(0)=w^2g^TNBNg>0,\qquad
\beta\ddot V(0)=-\ddot E_D(0),
\]

and, for `e>0`,

\[
\mathcal E^{(3)}(0)
=-2ew^2(BNg)^TND^{-1}(BNg)<0.
\]

Conversely, uniform form with `g=0` is exactly stationary under the held
unforced law. The model does not depart spontaneously from a completely
balanced equilibrium. This theorem identifies a local conversion of supplied
nonequilibrium phase structure into form, not the eventual identity or lifetime
of the configuration it generates. The maintained acute geometries already
have their [circulation classification](RELATIONAL_RECOVERY_AND_INTERACTION.md#equilibria-reuse-the-existing-circulation-classification);
that result should be reused, not counted as a new discovery from this control.

Common ideal shifts `x -> x+m*1` and `theta -> theta+c*1` leave both rows,
contrast storage and winding invariant (`B*1=L*1=0`). The zero-form control
therefore represents the entire common-uniform-form family and all common
phase rotations. Its zero origin is not a privileged physical value.
Arbitrary rounded additions to stored binary64 phases need fresh admission
because their exact represented differences may no longer coincide.

### Equal-storage form reversal: separating energy from direction

<a id="relational-reversed-form-control"></a>

The prospectively frozen control exactly negates the successful reference's
stored initial EPI values and changes no phase, support, capacity or coefficient.
In the reflected coordinates, `(q,r) -> (-q,-r)` with `(a,b)` unchanged.
The represented `q_0=1+2^(-54)` becomes its exact negative; signed zero in
the raw JSON is retained and represents the same mathematical zero.
This is neither simultaneous form/phase reflection nor time reversal.

#### Exact initial match and local discriminator

On the admitted fixed graph with held positive capacity, write `y=Bx` and
`K=N*D^-1`. A common-offset reflection `x^- = 2m*1-x^+` has `y^-=-y^+`.
Identical phase geometry keeps `g,H,V` fixed. Consequently,

\[
E_D^-=E_D^+,\quad V^-=V^+,\quad
\mathcal E^- =\mathcal E^+,\quad
\dot{\mathcal E}^- =\dot{\mathcal E}^+=-e\,y^TKy,
\qquad \dot\theta^-=-\dot\theta^+.
\]

The signed exchange term

\[
J=w\,y^TNg,\qquad
\dot E_D=-e\,y^TKy+J,\qquad \beta\dot V=-J
\]

reverses sign. It is a read-out of work already present in the joint law,
not a new pressure channel, state primitive or control policy. Equal total
loss therefore allows opposite initial transfer between phase and form.
For the actual reference, certified intervals give
`J_ref` about `-0.0198749634744`: form initially supplies phase storage.
The reversed preparation has opposite transfer while losing total storage
at exactly the same instantaneous rate.

Since `y_dot=-e*B*K*y+w*B*N*g`,

\[
\ddot{\mathcal E}
=2e^2y^TKBKy-2ew\,y^TKBNg,\qquad
\ddot{\mathcal E}^- -\ddot{\mathcal E}^+
=4ew\,y^TKBNg.
\]

For the selected two-ring state `r_0=0`, put
`h=w*(3*g0-g4)>0`. This reduces to

\[
\ddot{\mathcal E}(-q_0)-\ddot{\mathcal E}(q_0)
=\frac{16}{3}e q_0 h>0.
\]

Static exact-input calculations, performed before evolution, retain
rigorous intervals around `E_0=8.736526060070739`, equal ideal
`E_dot(0)=-(2/3)*(1+2^(-54))^2`, and an acceleration difference between
`0.29026` and `0.29028`. Initial `a_dot` changes from about `+0.159026` to
`-0.159026`; `b_dot=0` in both. Native engine fields independently check
represented form-storage equality, loss equality and phase-rate negation;
their floating work residual is not substituted for the ideal identities.

Thus `(E,E_dot)` already fails as an autonomous state description: two
identical summary states have different derivatives of `E_dot`. Before
evolution this did not decide the **limiting basin**. The frozen criterion
was that different admitted terminal sectors would also exclude a basin
selector based only on those initial summaries, even with identical phase
geometry and model parameters. The same limit would not establish storage
sufficiency; numerical unavailability would remain inconclusive.

The shared producer's `--reverse-form` mode freezes one control with horizon
32, step `1/8`, order 12 and 128-bit outward arithmetic. Its initial-match
gates must pass before a protocol can be prepared. The whole endpoint may
admit any existing protected basin; no preferred sector, new phase law,
retry or changed horizon is introduced. The reference and zero-form records
remain immutable and are not re-executed by this control.

#### Retained result: equal initial energy, different limiting identity

The first evaluation validates all 256 whole-time steps to time 32, without
retry or unresolved interval. Its entire endpoint box lies in the consensus
rectangle `S`, with joint storage conservatively enclosed in
`[1.37994484500693, 1.37994484500711] < 7`. The smallest rectangle margin is
greater than `1.05417`; the maximum endpoint coordinate width is below
`1.889e-14`. All resultants remain positive, with a whole-run lower bound
greater than `0.03080034`. The smallest strict Picard inclusion margin
exceeds `3.0648e-5`. Conditional continuation therefore converges to consensus.

| Initial form, with identical phase/support/capacity/law | Initial joint storage | Initial loss | Proved limiting basin |
| --- | --- | --- | --- |
| Reference contrast | About `8.73652606007` | `(2/3)*(1+2^(-54))^2` | Positive aligned twist |
| Zero contrast | About `7.93652606007` | `0` | Consensus |
| Negated reference contrast | Exactly equal to reference | Exactly equal to reference | Consensus |

The reference/reversal pair now disproves a deterministic limiting-basin
selector using only initial `(E,E_dot)`, even with the phase geometry and
other model inputs supplied. It does not disprove every possible scalar
encoding, a classifier using additional state, or a model using the subsequent
energy history. The initial acceleration of energy already differs, and no
claim of identical losses throughout the trajectory was made.

The retained report has `terminal_basin_admitted=true`, `target_sector=0`
and `initial_storage_selector_refuted=true`. Its historical positive-pattern
flag is false because consensus is a different resolved basin. The earlier
finite reference verdict also remains false; the later continuous proof and
these controls do not rewrite that experiment's acceptance criterion.

| Immutable control evidence | SHA-256 |
| --- | --- |
| [Frozen prospective protocol](../../docs/assets/relational_reversed_form_response/result.protocol.json) | `050e265497f9bd7c483c20ab77bbf58baf763bb13e42575d8227bca9801bd9a8` |
| [Validated response](../../docs/assets/relational_reversed_form_response/result.json) | `f3de2b74e9f0ab9ba3b621dfc471266e22f23a391bb1f4f76153538d924e2963` |
| [604 fingerprinted source files](../../docs/assets/relational_reversed_form_response/result.source.zip) | `9816d178479941e1d9ac007452b06c41d75e2fe7eb5d20d83c81bb1c42494e20` |

An independent retained-record audit checks every source hash, the sign-only
intervention, initial identities, all strict Picard inclusions, chained
endpoints, resultant bounds and terminal inequalities, without rerunning the
trajectory. A **retrospective** consequence of the same whole-time tubes is
that every raw oriented ring gap `(-2a,a-b,b,b,a-b)` stays strictly inside
`(-pi,pi)`, with margin greater than `0.02616569`. Their sum telescopes to
zero, so both ring windings remain zero throughout `[0,32]`. This is stronger
than endpoint sampling but was not a frozen winding-time prediction. The
zero-form control's separately reported transient winding remains a different
case.

#### What the three preparations establish together

The uniform-form result shows how nonbalanced phase geometry first produces
form contrast. Form contrast in turn moves relative phase; reversing it
changes both that direction and the signed exchange `J` while preserving
initial scalar storage and loss. Dissipation and the existing circulation
classification then allow distinct final phase geometries with uniform form.
Thus transient form can affect the maintained full-state identity even when
the final scalar EPI profile is uniform. The earlier regional interaction
result supplies transmission through the same declared law and supplied
connections. These mechanisms share one model; they do not require a new
operator, telemetry-based selector or primitive state variable.

This closes the named energy-versus-direction discriminator. It does not
close arbitrary graph formation, autonomous support/capacity evolution,
physical identification or selection of this constitutive law by nature.
The [shared work integration](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-work-integration) now exposes signed
exchange and regional boundary accounting without extending this preparation
into another basin sweep. The [composition owner](RELATIONAL_PATTERN_COMPOSITION.md)
retains the tangent reduction, nonlinear closure obstructions and conditional
passive bridge-relocation result. The
[sole execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
selects any further formation or preparation question; the controls here
remain reusable evidence rather than a separate task queue.

### Phase consensus with a bounded reflected form preparation cannot form the target

<a id="relational-consensus-preparation-obstruction"></a>

The preceding successful formation proof supplies both nonuniform phase and
form. Removing the initial phase structure is a different initial-value
question from either of its retained form controls. The following analytic
result excludes an entire bounded class under the same native law; it does
not depend on a sampled response or on the numerical transit solver's domain.

#### The exact preparation class and conclusion

Keep the same two unit C5 rings with the two corresponding adjacent bridges,
unit held capacities, `e=w=1/2`, `beta=1`, structural clock, and no inputs,
events or changing support. After removing a common phase origin, set every
initial phase to zero. On each ring supply the copied reflected form

\[
x(0)=(A_0,-A_0,-B_0,0,B_0),\qquad A_0,B_0\in\mathbb R.
\]

Common uniform form offsets are immaterial to this relative statement.
Write the existing sufficient reflected state as
`p=(A,B)^T`, `z=(a,b)^T`, and define

\[
K=\begin{pmatrix}3&-1\\-1&2\end{pmatrix},\qquad
D=\operatorname{diag}(3,2),\qquad
\binom q r=Kp.
\]

Its initial joint storage is exactly its form storage,

\[
F_0=2p(0)^TKp(0)=6A_0^2-4A_0B_0+4B_0^2.
\]

**Conditional obstruction.** If `F_0<=9`, the ideal full native solution
exists through structural time one, remains in `|a|,|b|<1/2` throughout
that interval, and satisfies

\[
\boxed{E(1)\le\frac23F_0\le6<7.}
\]

It then belongs to the already proved consensus capture basin, so its
continuation is regular for all forward time and converges to consensus.
In particular it cannot converge to either maintained aligned unit-winding
target. The number `9` is an analytic preparation ceiling, not a selected
amplitude, a changed laboratory unit or a candidate obtained by trial.

If `F_0=0`, positive definiteness forces `A_0=B_0=0`; the complete state is
stationary. For `0<F_0<=7`, the previous strict sublevel argument already
settles the result: the exact loss is initially strictly positive, so an
initial equality `F_0=7` immediately enters `E<7` while the phases remain
inside the consensus rectangle. The proof below additionally covers the
whole upper part `7<F_0<=9`, rather than treating the necessary energy
barrier as a sufficient formation condition.

#### A regular small-phase interval is guaranteed before any basin test

Reuse the exact four-coordinate rows from the
[reflection reduction](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-reflected-capture-boundary):

\[
\dot p=-eD^{-1}Kp+w g(z),\qquad
\dot z=w H(z)^{-1}Kp,
\qquad H=\operatorname{diag}(H_0,H_4).
\]

The phase metric is the actual native metric, not its consensus
linearization. While `|a|,|b|<=1/2`, every incident fine-edge phase
difference lies in `[-1,1]`: the nonzero ring gaps are
`(-2a,a-b,b,b,a-b)` and the bridge gaps vanish by copying.
Since `cos(1)>1/2` and `1<pi/2`, every relative resultant has
strictly positive real part, and its argument `alpha_i` lies in
`(-pi/2,pi/2)`. The native identity

\[
H_i=\pi\operatorname{Re}(z_i)
       \frac{\tan\alpha_i}{\alpha_i},
\qquad \frac{\tan0}{0}:=1,
\]

therefore gives `H_i>pi*d_i/2`, including the central rows of the
full graph. In the two retained phase rows this implies
`H>(pi/2)D` as positive diagonal matrices. No zero-resultant
continuation or special treatment of a formally vanishing central
numerator is used.

The existing work identity gives `E(t)<=F_0` while the law is
regular, and phase storage is nonnegative. The exact form quadratic
in the gradient coordinates is

\[
F_D=\frac{4q^2+4qr+6r^2}{5}
 =\frac23q^2+\frac65(r+q/3)^2
 =r^2+\frac45(q+r/2)^2.
\]

Thus `q^2<=3F_0/2` and `r^2<=F_0`, yielding the complete-row
speed bounds

\[
|\dot a|\le\frac{\sqrt{3F_0/2}}{3\pi}<\frac12,\qquad
|\dot b|\le\frac{\sqrt{F_0}}{2\pi}<\frac12
\qquad(F_0\le9).
\]

Starting from `a=b=0`, neither coordinate can first reach the box
boundary by time one: integration up to such a first exit would
still give absolute coordinates strictly below `1/2`. Form is bounded
by the positive quadratic, and the entire closed phase box is a
compact subset of the full regular domain. Smooth continuation
therefore supplies the claimed whole interval. More precisely,

\[
|a(t)|\le\frac{t\sqrt{3F_0/2}}{3\pi},\qquad
|b(t)|\le\frac{t\sqrt{F_0}}{2\pi}
\qquad(0\le t\le1).
\]

These are analytic bounds for the actual nonlinear trajectory.
They neither hold form fixed nor identify a tangent or Euler
extrapolation with the solution.

#### Form loses storage faster than this interval can build phase storage

Introduce only proof norms, not new dynamical coordinates,

\[
Y(t)^2=2p(t)^TKp(t)=F_D(t),\qquad
Z(t)^2=2z(t)^TKz(t).
\]

The symmetric matrix `K^(1/2) D^(-1) K^(1/2)` has eigenvalues
`1-1/sqrt(6)` and `1+1/sqrt(6)`. The latter is less than `3/2`.
The phase-metric bound and the induced `K` norm therefore give

\[
\sqrt{2\dot z^TK\dot z}
\le w\lambda_{\max}(K^{1/2}H^{-1}K^{1/2})Y
\le\frac{3}{2\pi}Y\le\frac12Y.
\]

Since `Y(t)<=sqrt(F_0)` and `z(0)=0`, the integral triangle
inequality implies `Z(t)<=sqrt(F_0)*t/2`. The global elementary
bound `1-cos(s)<=s^2/2` applied to the actual reflected phase storage
then gives

\[
\begin{aligned}
V(a,b)&=2[1-\cos(2a)]+4[1-\cos(a-b)]+4[1-\cos b]\\
&\le6a^2-4ab+4b^2=Z^2
\le\frac{F_0t^2}{4}.
\end{aligned}
\]

Independently, the exact loss satisfies

\[
\mathcal L=4e\,(Kp)^TD^{-1}(Kp)
\ge2e\left(1-\frac1{\sqrt6}\right)Y^2
\ge\frac12Y^2.
\]

All quantities are evaluated along the same native solution. With
`beta=1`, `E=Y^2+V`, so throughout the guaranteed interval

\[
\dot E=-\mathcal L
\le-\frac12E+\frac{F_0t^2}{8}.
\]

The explicit comparison function

\[
B(t)=F_0\left(1-\frac t2+\frac{t^2}{6}\right)
\]

starts at `B(0)=E(0)=F_0` and obeys

\[
B'(t)-\left[-\frac12B(t)+\frac{F_0t^2}{8}\right]
=\frac{F_0t(2-t)}{24}\ge0\qquad(0\le t\le1).
\]

Equivalently, the positive part of `E-B` cannot grow from zero
under the displayed scalar differential inequality. Therefore
`E(t)<=B(t)` on the entire interval, proving `E(1)<=2F_0/3`.
This bound retains nonlinear phase feedback through its metric and
potential bounds; it is not a calculation with that feedback removed.

At time one the phase box lies strictly inside
`(-2*pi/3,2*pi/3) x (-pi/2,pi/2)`, and joint storage is below
the exact barrier `7`. The
[existing consensus capture theorem](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-protected-capture)
now supplies global regular continuation and convergence to consensus.
The proof did not construct a synthetic endpoint or restart a different
law there: all inequalities concern the same continuous initial-value
problem.

#### Application to the frozen successful reference's storage budget

The [retained original protocol](../../docs/assets/relational_capture_response/continuous-transit.audit.protocol.json)
fixes the successful source's represented state and law independently of
its response (SHA-256
`d8913d6af0d40fd1c903eaf24be29fb8ebeb6d3be28d6415d4e8dafb9974db99`).
Reconstruct its graph, form, phases, capacities and model from that protocol,
then use the shared native capture's rational `storage_bounds`, not the
field's floating storage or a current producer's defaults. This encloses

\[
8.7365260600707<E_{\rm ref}<8.7365260600729<9.
\]

That total includes its initial phase storage as well as form storage;
its former form amplitude is not substituted for the total budget.
The decimal bounds above are outward summaries of the retained rational
enclosure, not an exact decimal definition of equal energy.

Every phase-consensus preparation in the requested ellipse
`6A_0^2-4A_0B_0+4B_0^2<=E_ref` therefore belongs to the proved
`F_0<=9` class. Its storage by time one is at most `2E_ref/3`,
and it converges to consensus. The ceiling `9` is only a convenient
rational outer bound for the analytic proof; the requested reference
budget has not been raised, and no larger-amplitude response was selected
or evaluated.

This is a class-level exclusion of **maintained target formation**
for the supplied phase-consensus restriction. Nonuniform form can still
generate phase motion: at the initial state,
`a_dot=w*q/(3*pi)` and `b_dot=w*r/(2*pi)`, so both vanish only
when `A_0=B_0=0`. What fails is delivery of the maintained nonzero
winding from this bounded initial class. The first-unit phase bounds
keep its raw cycle gaps inside `(-pi,pi)`, hence winding zero there.
Convergence to consensus alone is not used to classify every possible
intermediate winding after that interval.

Simultaneously negating `(A,B,a,b)` is a symmetry of the same law.
At initial phase consensus this also represents form reversal, and
both orientations have the same consensus conclusion. It is not
dissipative time reversal. There is no contradiction with the earlier
reference/reversal result, whose common initial phases were already
nonuniform and were held fixed under that different intervention.

The result does not extend to arbitrary ten-node form states, broken
reflection or ring-copy symmetry, larger budgets, changed capacities,
other supports, or the sine and alternative-mobility laws. It does
not establish that all formation requires an initial phase seed,
or rule out other preparations under the native paradigm. It identifies
a precise obstruction under fixed supplied structure and a declared
budget, rather than a universal no-formation theorem.

#### Shared analytic admission and independent checks

`certify_relational_consensus_capture` and
`RelationalConsensusCaptureCertificate` in the
[native capture owner](../../src/tnfr/physics/relational_capture.py)
reuse the existing support, copied-reflected state, coefficient and
full-law admission. They check exact initial phase consensus and the
exact form-storage ceiling before exposing the analytic time-one
storage and phase bounds. An initial snapshot need not already pass
the strict `E<7` capture test for this new theorem to apply.

The certificate records a future bound, not an evaluated endpoint
state. Refusal of one theorem hypothesis leaves this conclusion
unavailable; it is not evidence of successful formation. The prior
native finite-executor, continuous-transit and control verdicts remain
unchanged. No producer was regenerated and no new trajectory or basin
sweep was used to prove this exclusion.

The [phase-consensus contract](../../docs/contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#phase-consensus-capture)
and [formation-admission tests](../../tests/physics/test_relational_formation_admission.py)
own the executable scope and independent controls. Static admission,
the nonlinear class theorem and a selected numerical trajectory are
different kinds of evidence. None identifies physical matter or
selects the original microscopic law, initial form or support.

<a id="a-full-form-phase-storage-bound-excludes-the-same-maintained-target"></a>
<a id="relational-full-consensus-formation-obstruction"></a>

### Full-form nonlinear phase-consensus obstruction

The copied-reflected restriction can be removed for **target exclusion**.
Keep the same two C5 rings, the two bridges at matching positions zero and
one, unit held capacities, `beta=1`, effective `e=w=1/2`, and the unforced
native regular-domain law. Prepare all phases equal, but allow any real
ten-node form modulo a common offset. Write `L` for the combinatorial
Laplacian, `D=diag(d_i)`, and

\[
F=\tfrac12x^TLx,\qquad P=V_\phi,\qquad E=F+P,
\qquad \mathscr L=\tfrac12q^TD^{-1}q,\quad q=Lx.
\]

Suppose `F_0<=9`. Then on **every regular existence interval** of the
ideal nonlinear solution,

\[
\boxed{P(t)\le e^{-\pi/7}F_0\le\frac7{10}F_0\le\frac{63}{10}.}
\]

The middle inequality is strict when `F_0>0`. The aligned unit-winding
target in either orientation has phase storage greater than `55/8`, so
neither entry into its acute component nor convergence to that target is
possible. This covers the reference budget `F_0<=E_ref<9` without imposing
copy or reflection. Unlike the preceding theorem, this result does **not**
prove global regular continuation or convergence to consensus.

#### Argument pressure has a global storage bound on its regular domain

Let `z_i=C_i+i*S_i`, `alpha_i=Arg(z_i)` and `g_i=alpha_i/pi`. The
incident phase cost is `d_i-C_i`. If `C_i>=0`, then `|alpha_i|<=pi/2`
and `|z_i|<=d_i` give

\[
d_i-C_i\ge d_i(1-\cos\alpha_i)
\ge 2d_i\alpha_i^2/\pi^2\ge d_i g_i^2.
\]

The trigonometric bound follows from concavity of sine on `[0,pi/2]`.
If `C_i<0`, the cost exceeds `d_i`, whereas `g_i^2<1` throughout the
regular domain. Thus in either case

\[
d_i g_i^2\le\sum_{j\sim i}(1-\cos(\theta_j-\theta_i)),
\qquad \sum_i d_i g_i^2\le2P.
\]

This includes zero pressure and zero real resultant with nonzero imaginary
part. It assigns no pressure at a zero resultant or on the excluded branch.
The exact storage balance and weighted Cauchy--Schwarz now give

\[
\dot E=-\mathscr L,\qquad
\dot P=-\tfrac12q^Tg,\qquad
|\dot P|\le\tfrac12\sqrt{q^TD^{-1}q}\sqrt{g^TDg}
\le\sqrt{\mathscr L P}.
\tag{A}
\]

This controls the actual nonlinear exchange, not a frozen tangent or a
phase-metric approximation. The potentially large native phase velocity
near a regular-domain boundary does not invalidate the work identity.

#### The actual full support supplies a uniform loss bound

The graph contains the spanning cycle
`(0,4,3,2,1,6,7,8,9,5,0)` in the supplied ring labels, and its maximum
degree is three. Let `lambda_*` be the nonzero spectral gap of
`D^(-1/2)L D^(-1/2)`. For any nonconstant form, its weighted Rayleigh
denominator is `min_c sum_i d_i(x_i-c)^2`, at most three times its
ordinary centered squared norm. The spanning C10 supplies a lower bound
on the numerator. Consequently

\[
\lambda_*\ge\frac{\lambda_2(L_{C_{10}})}3
=\frac{3-\sqrt5}{6}>\frac19.
\]

The last inequality follows from `sqrt(5)<7/3`. The same symmetric
normalized Laplacian gives `q^TD^-1q>=lambda_* x^TLx`, hence

\[
\boxed{\mathscr L\ge\frac19 F.}
\tag{B}
\]

All bridge degrees and costs are retained. No isolated-ring rate, sampled
eigenvalue or restriction to a few preparation modes enters this bound.

#### A nonlinear comparison includes all signs of the exchange

On the positive stratum `F,P>0`, put

\[
f=\sqrt F,\quad y=\sqrt P,\quad
k=\mathscr L/F\ge1/9,\quad h=\dot P/(2fy).
\]

Equations (A)--(B) imply

\[
\dot f=-\tfrac k2f-hy,\qquad \dot y=hf,
\qquad 2h\le\sqrt k\le3k.
\]

The sign of `h` is unrestricted: exchange may return storage to form.
Let `varphi=atan2(y,f)` in `[0,pi/2]` and define the auxiliary comparison
quantity

\[
\Psi(\varphi)=\frac27(\varphi+\sin\varphi\cos\varphi),\qquad
W=(F+P)e^{\Psi(\varphi)}.
\]

This is a derived bound, not a replacement storage or a new evolution law.
Since `Psi'=4*cos(varphi)^2/7` and
`varphi_dot=h+(k/2)sin(varphi)cos(varphi)`,

\[
\frac{\dot W}{W}
=-k\cos^2\varphi+
\frac47\cos^2\varphi\left(h+\frac k2\sin\varphi\cos\varphi\right)
\le k\cos^2\varphi
\left[-1+\frac27\left(3+\sin\varphi\cos\varphi\right)\right]
\le0.
\]

**The zero strata are included.** On any compact time interval within the
regular existence interval, `x,theta` are continuously differentiable.
The quantities `f` and `y` are norms of, respectively, real edge-form
differences and complex phase-chord differences, so they are locally
Lipschitz in time. The displayed `W(f,y)` extends locally Lipschitz to
the closed quadrant, with value zero at the origin. It is therefore
absolutely continuous along the solution. A nonnegative Lipschitz function
has derivative zero almost everywhere on its zero set. At `f=0`, `q=0`
makes both loss and phase work zero; at `y=0`, `g=0` makes phase work zero
and `W_dot=-mathscr L` almost everywhere on that stratum. Both zero means
the stationary uniform state. Thus `W_dot<=0` almost everywhere, including
boundary visits, and absolute continuity gives `W(t)<=W(0)=F_0`.

Finally, `P/W=sin(varphi)^2 exp[-Psi(varphi)]` increases on `[0,pi/2]`:
its logarithmic derivative in the interior is
`2*cot(varphi)-4*cos(varphi)^2/7>0`. Its largest value is `exp(-pi/7)`.
This proves the boxed bound. The rational relaxation follows from
`exp(pi/7)>1+pi/7>10/7`, using `pi>3`.

The same comparison also gives a necessary preparation condition without
assuming initial phase consensus. For any initial regular state of this
same complete law, put `varphi_0=atan2(sqrt(P_0),sqrt(F_0))`. Then

\[
P(t)\le E_0\exp[\Psi(\varphi_0)-\pi/7].
\]

In particular, approaching the aligned target requires
`E_0*exp(Psi(varphi_0))>=P_*exp(pi/7)`. This is a necessary condition on
the initial form/phase storage split, not a sufficient formation criterion
or a selection of that preparation. It introduces no phase-threshold fit
or additional response campaign.

#### What is excluded and what remains unresolved

The existing acute-sector Jensen bound gives phase storage at least

\[
P_*=10(1-\cos(2\pi/5))=\frac{25-5\sqrt5}{2}
>\frac{55}{8}>\frac{63}{10}
\]

whenever both supplied rings have the same unit winding and all edges
are acute. Here `sqrt(5)<9/4` proves the first rational inequality.
The target itself has exactly `P_*`. The all-time strict separation
therefore excludes entry into that acute target component and approach to
the aligned target, even asymptotically. It is not merely a failed
sufficient capture test or absence of formation in sampled trajectories.

The native maximal solution might instead approach another admissible
state or reach a singular regular-domain boundary. The phase ceiling
alone does not decide between those cases, exclude all transient winding,
or define continuation through a missing native field. No general
consensus claim follows. The earlier copied-reflected theorem retains
its stronger continuation and limiting-consensus conclusions on its subset.
The prepared positive formation, zero-form and reversal controls retain
their nonuniform initial phases and are not contradicted.

The shared `certify_relational_consensus_formation_obstruction` and
`RelationalConsensusFormationObstruction` in the
[capture owner](../../src/tnfr/physics/relational_capture.py) admit one
fresh full native source and the exact support. Their prospective bound is
`7*F_0/10`; the target cost reuses the existing mathematical-cosine owner.
Failed phase, capacity, coefficient or budget premises withhold that
prospective conclusion. The report explicitly leaves continuation
uncertified. No reflection projection, solver, frozen producer or changed
constitutive law is used. Independent algebra and actual-field admission
controls belong to the existing formation and capture tests.

**Supplied events have a separate comparison obligation.** For a piecewise
native path on this same support, capacity, storage scale and coefficient
law, with regular reset endpoints and finitely many resets on the interval,
`W` is nonincreasing on every regular continuous segment. A reset may
change it. If `Delta W_j=W_j^+-W_j^-` denotes its actual signed jumps, entry
into the acute target requires

\[
\sum_j\Delta W_j\ge P_*e^{\pi/7}-F_0,
\qquad
\sum_j(\Delta W_j)_+\ge P_*e^{\pi/7}-F_0.
\]

This follows by telescoping the continuous decreases and jumps and using
`W>=P*exp(pi/7)` at the endpoint. It is an accounting bound for this derived
comparison quantity, not an energetic reservoir or an event-occurrence law.
Structural-storage passivity alone does not enforce it: from the source
`x_2=3`, all other forms zero and all phases zero, `F_0=9`. A supplied
reset to uniform form and the aligned two-ring twist has `E^+=P_*<9` but
`W^+=P_*exp(pi/7)>275/28>9=W^-`. Both endpoints are native regular states.
Thus an explicitly supplied passive reset can escape the continuous
preparation obstruction. This comparison does not establish that any named
operator executes that reset or explain when a node selects it. A support,
capacity or coefficient change requires new constants and admission; it
cannot silently reuse this same-model jump inequality.
