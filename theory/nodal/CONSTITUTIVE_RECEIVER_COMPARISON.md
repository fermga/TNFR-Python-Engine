# The original receiver source across the admitted constitutive interval

<a id="constitutive-receiver-comparison"></a>

This owner admits the unchanged two-ramp source under every law in the
fixed interval \(|\lambda|\le1/10000\), then compares its actual slow
phase with the preserved sine reference enclosures. Source preparation,
support, capacities, clocks and error allowances stay fixed. The comparison
does not integrate a new trajectory or transfer the sine outcome through
matching target derivatives. It rebuilds the changed-law source transport,
velocity defect, residual and observation margins.

The retained numerical horizon is the 76 evaluated sine steps, ending at
elapsed slow time \(19/2\). This was the stopping horizon of the
[preserved response](ORGANIZED_RECEIVER_ENCLOSURE.md#organized-receiver-enclosure-result),
whose original prospective maximum was \(32\). No continuation to that
unevaluated maximum, parameter-interval reduction or preparation change is
part of this comparison. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
remains the sole queue.

<a id="constitutive-receiver-source"></a>
## Complete law and primitive preparation

Use the actual equal-weight 27-node support, law and units from the
[constitutive barrier owner](CONSTITUTIVE_IDENTITY_BARRIERS.md#constitutive-barrier-model).
Set
\[
 \Lambda=1/10000,\quad
 P(d)=\cos d(1-\cos9d)^2,\quad
 V_\lambda(d)=1-\cos d+\lambda P(d),\quad
 j_\lambda=V_\lambda',\quad |\lambda|\le\Lambda.
\]
The parameter is constant along each trajectory. With \(A=L/4\),
\(\gamma=1/(1023\pi)\) and \(\varepsilon=\gamma^2\), define
\[
 F_{\lambda,i}(\theta)=\tfrac14\sum_jw_{ij}
                           j_\lambda(\theta_j-\theta_i),
 \qquad
 x_\tau=-Ax+\gamma F_\lambda(\theta),\quad
 \theta_\tau=\gamma Ax.                                  \tag{1}
\]
Here \(\tau=(1023/1024)t\), capacities are held at one, and support
and conductances are fixed. Both rows acquire the factor \(1023/1024\)
in the original clock. Phase angles are measured in radians; form, time
and storage retain this model's structural units, with no laboratory-unit
bridge supplied. There is no forcing, event, reset or installed
operator word. All form and phase coordinates remain, on the conserved
global mean-zero leaf. The complete own-law storage obeys
\[
 H_\lambda=\tfrac12x^TLx+\sum_{\{i,j\}}w_{ij}
       V_\lambda(\theta_j-\theta_i),\qquad
 (H_\lambda)_\tau=-\tfrac14\|Lx\|^2.                     \tag{2}
\]

Keep the original primitive source
\[
 \begin{split}
 q_j&=j-4,\quad m=\tfrac{2046}{9}(355/113)^2,
       \quad \alpha=2\pi/9,\\
 Q_{r,j}&=q_j\quad(r=0,1),\qquad Q_{2,j}=0,\\
 x(0)&=mQ+\eta_x,\qquad \theta(0)=\eta_\theta,\\
 |\eta_{x,r,j}|&\le r_x=10^{-10},\qquad
 |\eta_{\theta,r,j}|\le r_\theta=10^{-10}.
 \end{split}                                               \tag{3}
\]
Each error vector has its separately declared zero sum in each original
region. Every allowed combination is retained; only the global means
remain conserved after evolution begins. In particular, the nominal
reflection/sign symmetry does not restrict the original error directions.
The rational ramp coefficient is not replaced by an exact multiple of
\(\pi\).

The nominal initial storage is \(150m^2\), with \(72m^2\) internal
and \(78m^2\) on contacts, for every admitted law. Rebuilding the
actual support gives \(\|L(mQ)\|_1=117m\) and
\(\|L(mQ)\|_2=m\sqrt{2277/2}\). At nonzero phase error,
\(|P(d)|\le9^4d^4/4\), and the total edge weight is \(54\).
Consequently the complete initial storage is bounded below by
\(150m^2-117mr_x\), and above by
\[
 150m^2+117mr_x+108(r_x^2+r_\theta^2)
             +216\,9^4\Lambda r_\theta^4.                \tag{4}
\]
The lower bound uses the actual nonnegative potential; the added quartic
term accounts for the changed potential at the original phase errors.

<a id="constitutive-receiver-release"></a>
## Uniform own-law transport and slow-velocity defect

The Fourier coefficients of \(P'\) give
\[
 |P'|\le57/2,\quad |P''|\le328,
 \quad J=1+(57/2)\Lambda,\quad
 M=\sqrt{27}J,\quad K=2(1+328\Lambda)=1291/625.           \tag{5}
\]
Thus \(\|F_\lambda\|_2\le M\) and
\(\|DF_\lambda\|_2\le K\) on all phase lifts. The same actual
support has \(\|A\|_2\le2\) and mean-zero heat gap
\(g=1/10\). These graph properties are independent of \(\lambda\);
the changed force and derivative bounds are freshly supplied by (5).

Put \(a=\gamma x\), \(z=\theta+a\), and
\(P_*=\alpha Q\). Both exact rows become
\[
 a_\tau=-Aa+\varepsilon F_\lambda(\theta),\qquad
 z_\tau=\varepsilon F_\lambda(\theta).                   \tag{6}
\]
Define
\[
 A_0=\gamma(m\sqrt{120}+\sqrt{27}r_x),\qquad
 d_0=\sqrt{120}|\gamma m-\alpha|
                +\sqrt{27}(r_\theta+\gamma r_x),
\]
\[
 B(T)=e^{-gT}A_0+\varepsilon M/g,\qquad
 b(T)=d_0+B(T)+\varepsilon MT.                            \tag{7}
\]
The mean-zero heat estimate applied to the variation-of-constants formula
for \(a\), followed by the exact mixed row, proves
\[
 \|\gamma x(T)\|_2\le B(T),\qquad
 \|\theta(T)-P_*\|_2\le b(T).                            \tag{8}
\]
The form bound \(B(400)/\gamma\) remains valid for every
\(\tau\ge400\). No phase-only model has replaced the form dynamics.

At \(T=200\), (8) gives donor winding one and receiver winding zero.
On \(200\le\tau\le400\), use the uniform phase bound
\(d_0+B(200)+400\varepsilon M\) to retain the donor branch integers.
Receiver retention before release has a separate whole-time proof. The
commuting internal/contact Markov heat kernels on the fixed support give
\(|\Delta(e^{-A\tau}Q)_{2,j}|<4\), as derived in the
[heat-shape argument](ORGANIZED_RECEIVER_MECHANISM.md#organized-receiver-delay).
Applying its variation-of-constants identity to the actual changed force,
with \(|F_{\lambda,i}|\le J\), gives, for \(0\le\tau\le400\),
\[
 |\Delta\theta_{2,j}(\tau)|
 <4\gamma m+2r_\theta+4\gamma r_x+4\varepsilon\,400J<\pi.
                                                                    \tag{9}
\]
This reuses a law-independent heat estimate with its new force bound,
not the sine source's winding verdict.

In slow time \(s=\varepsilon\tau\), the actual phase velocity is
\(\theta_s=Ax/\gamma\). Its defect from the changed descent field is
\[
 r_\lambda=\theta_s-F_\lambda(\theta)=-x_\tau/\gamma.
\]
Differentiating both actual rows gives
\[
 (r_\lambda)_\tau
   =-(A+\varepsilon DF_\lambda)r_\lambda
          -\varepsilon DF_\lambda F_\lambda.             \tag{10}
\]
This vector stays on the global mean-zero leaf. With
\[
 \kappa=g-\varepsilon K>0,\qquad
 R_0=\frac{m\sqrt{2277/2}}{4\gamma}
          +\frac{2\sqrt{27}r_x}{\gamma}+K\sqrt{27}r_\theta,
\]
the symmetric Jacobian and (5) imply
\[
 \|r_\lambda(\tau)\|_2
 \le R_0e^{-\kappa\tau}+\frac{\varepsilon KM}{\kappa}.
                                                                    \tag{11}
\]
The shared outward rational kernels rebuild all inequalities for the
full interval and original errors. They certify
\[
 b(400)<21/100000,\qquad B(400)/\gamma<1/50,
 \qquad \|r_\lambda(\tau)\|_2<\rho=11/10^6
       \quad(\tau\ge400).                                \tag{12}
\]
The [source assessment](../../src/tnfr/physics/_constitutive_receiver_source.py)
retains the primitive law interval, source cost, support, clock, source
errors and all margins. The numerical budget in (12) follows from the
changed-law inequalities; it is not admitted merely because the sine
assessment used the same displayed budget.

<a id="constitutive-receiver-retained-reference"></a>
## Preserved reference evidence and eight derivative cones

Use elapsed slow time \(u=\varepsilon(\tau-400)\). Equation (12)
admits every original-source trajectory as
\[
 (\theta_\lambda)_u=F_\lambda(\theta_\lambda)+r_\lambda,
 \quad \|r_\lambda\|_2<\rho,
 \quad \|\theta_\lambda(0)-P_*\|_2\le b(400).             \tag{13}
\]
The actual defect retains form memory; it is not an independently supplied
forcing in the complete law.

The original response archive contains one outcome and 76 immutable step
records at their recorded paths. This comparison pins those bytes:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `organized-receiver-enclosure-v1.response.zip` | 741711 | `77b38c4785a57a355aa0d86a318b7acd22702e69de317687e93222ab5758d044` |
| Retained outcome JSON inside the archive | 189956 | `2dbb6e1b738f33f1189ba0de1436b5682c8109000d0ea400250d9a7ae763b79c` |

The retained outcome's ordered step receipts check all 76 step payloads;
strict JSON and exact-number admission precede interval construction.
Support, source, clock, node order, step order and reference-center carry
must match the original validated-reference owner. Byte association is
not mathematical proof or external authentication. The original proof
supplies the reference enclosures: the retained Picard and Taylor derivative
enclosures remain explicit numerical premises and are not derived from
their byte receipts. The new calculation rebuilds every consumed changed-law
residual, cone, growth bound and observation margin.
It does not accept the old source verdict or logarithmic growth rates as
changed-law certificates.

Each original step has length \(h=1/8\), starts from a rational local
reference center \(c\), and encloses the exact sine reference
\(y'=F_0(y)\), \(y(0)=c\). The retained data include its whole-step
box \(T\), endpoint polynomial and remainder enclosures, and next rational
center. The reader rebuilds the validated endpoint box \(Y\) from those
enclosures, then recomputes its full Euclidean recentering radius \(d\)
relative to that next center, using the lifted metric \(2I_{13}\).
The old cached radius does not supply this new radius. Thirteen symmetric
coordinates are lifted to all 27 coordinates as in the original reference proof.
The actual uncertainty always occupies the full mean-zero tangent leaf.

Rebuild \(V=F_0(T)\). For each of eight fixed subintervals
\([a_k,b_k]=[(k-1)h/8,kh/8]\), form the coordinatewise intersection
\[
 T_k=T\cap\bigl(c+[a_k,b_k]V\bigr)
          \cap\bigl(Y-[h-b_k,h-a_k]V\bigr).               \tag{14}
\]
The first cone follows by integrating \(y'\in V\) forward from
\(c\); the second follows by integrating backward from its validated
endpoint. Both hold for the same exact reference trajectory. Therefore
\(y(u)\in T_k\) throughout that subinterval. An empty intersection
rejects the reconstruction; it does not give a zero residual. Equation
(14) uses no new trajectory, Taylor coefficients, time integration or
new reference center.

<a id="constitutive-receiver-residual-transport"></a>
## Full-state residual transport on fresh prospective domains

Write
\[
 F_\lambda=F_0+\lambda G,\qquad
 G_i(y)=\tfrac14\sum_jw_{ij}P'(y_j-y_i).
\]
On each cone, evaluate the factored derivative directly,
\[
 P'(d)=-\sin d(1-\cos9d)^2
                  +18\cos d(1-\cos9d)\sin9d,             \tag{15}
\]
accumulate signed edge currents at each node, and let
\(g_k\) be the outward Euclidean norm upper bound. Subtracting two
independent current intervals and dividing by \(\Lambda\) would
discard the cancellation in (15) and is not this residual construction.
The residual uses the reference cone only; it is not evaluated at a
hypothetical changed-law endpoint.

For the actual-reference difference,
\[
 (\theta_\lambda-y)'=
       F_\lambda(\theta_\lambda)-F_\lambda(y)
                    +\lambda G(y)+r_\lambda.             \tag{16}
\]
The derivative perturbation is bounded in operator norm by
\[
 \|DF_\lambda-DF_0\|_2
    \le328\Lambda\|L/4\|_2\le656\Lambda=41/625.          \tag{17}
\]
This follows edgewise from the weighted Hessian and remains valid on the
entire mean-zero leaf, including all transverse source errors.

Define the outward scalar comparison
\[
 \Psi(e,\mu,t,\beta)=e^{\mu t}e+\phi(\mu,t)\beta,
 \quad
 \phi(\mu,t)=
 \begin{cases}(e^{\mu t}-1)/\mu,&\mu\ne0,\\t,&\mu=0.
 \end{cases}                                               \tag{18}
\]
Given the carried full phase-error radius \(e\), first construct
\[
 \beta_{\max}=\rho+\Lambda\max_k g_k,\qquad
 R=\tfrac54\left[\Psi(e,K,h,\beta_{\max})+2^{-80}\right].
                                                                    \tag{19}
\]
The global derivative bound \(K\) proves that the actual changed-law
phase stays within \(R\) of the reference throughout the step. This
prospective domain does not rely on a future local growth estimate.

On the full convex domain \(T+\{v:\mathbf1^Tv=0,\|v\|\le R\}\),
rebuild a sine logarithmic-growth upper bound \(\mu_0\) using the
shared full-leaf cosine/Loewner and exact PSD construction. Its reflected
blocks cover all 26 mean-zero directions. The original step's cached
growth rate applies to its original radius and cannot substitute for this
new calculation. Equation (17) then supplies
\[
 \mu=\mu_0+656\Lambda.                                   \tag{20}
\]
This one fresh domain covers all eight subintervals. With
\(\Delta=h/8=1/64\), carry
\[
 e_k=\Psi(e_{k-1},\mu,\Delta,\rho+\Lambda g_k),
 \quad k=1,\ldots,8,
 \qquad E_{\rm step}=\max_{0\le k\le8}e_k.               \tag{21}
\]
The scalar comparison is monotone or constant on each subinterval, so
the endpoint maximum bounds the whole subinterval. Require the strict
check \(E_{\rm step}<R\). At the original step boundary, change to
the next retained rational reference center and add its full recentering
radius exactly once:
\[
 e_{\rm next}=e_8+d.                                     \tag{22}
\]
There are no reference resets at the eight artificial subinterval boundaries.
The actual complete state is never reset. Initial admission proves that
the release error plus the retained reference's rounding error is strictly
below \(21/100000\), then starts the comparison at exactly
\(e_0=21/100000\); it does not discard that rounding error.

The fixed comparison uses all 76 original steps, the full symmetric law
interval, eight derivative cones per step, outward rational arithmetic and
fresh full-leaf growth bounds. It keeps the same structural clocks and
original source errors. A broad bound may lose sign and temporal correlations;
such loss is a limitation of this sufficient scalar comparison, not a
restriction on the actual parameter's constancy or a new stochastic law.

## Observations and decision scope

Rebuild winding margins from the actual reference boxes and new full-state
errors. Whole-time receiver retention uses its fixed winding-zero branch
integers and requires every internal edge to remain strictly inside
\((-\pi,\pi)\) throughout every validated interval. Combined with (9),
a passing corridor certifies receiver winding zero from preparation through
the retained stopping horizon for the entire source and law family.

Terminal donor loss is a different observation. It requires the complete
new endpoint ball to lie in a winding different from one for each claimed
donor. A center winding or the original sine endpoint margin alone is
insufficient. A failed sufficient donor radius cannot establish donor
retention, donor loss or future acquisition. Receiver retention may be
certified even when the donor endpoint observation is unavailable.

A failed comparison domain, budget or observation remains explicitly
unavailable for that claim. It is neither a negative physical result nor
an exclusion of the compared constitutive laws. Any newly certified
finite nonacquisition concerns only the fixed source, law interval and
horizon; it does not decide later formation or regeneration. No physical
measurement bridge, preferred law or autonomous preparation follows.

<a id="constitutive-receiver-comparison-result"></a>
## Result: uniform finite receiver nonacquisition

The fixed eight-cone comparison closes all 76 prospective domains. Its
terminal full-source phase radius satisfies
\[
 e_{76}<287/1000,
 \qquad e_{76}\ \text{is approximately }0.286189004.       \tag{23}
\]
The receiver's winding-zero corridor passes throughout every subinterval:
the minimum strict edge-branch margin exceeds \(27/25=1.08\) radians
after including the complete new phase error. Its terminal margin also
exceeds \(1.08\). Combining this retained-reference comparison with the
own-law initial-layer result (9) proves
\[
 w_2(\tau)=0\quad\text{for every original admitted source,
 every }|\lambda|\le1/10000,
 \quad 0\le\tau\le400+\frac{19}{2\gamma^2}.              \tag{24}
\]
Thus the receiver has not acquired winding one anywhere in this entire
finite interval. In particular, receiver acquisition with continuously
retained donors and joined winding-one capture does not occur within
this horizon for the admitted source and constitutive family.

The donor endpoint decision is separate. The new ball is wider than
the endpoint radius permitting a robust winding-zero reading for either
donor. Both corresponding sufficient branch margins are below
\(-349/1000\). Their negativity makes this changed-law donor observation
**unavailable**; it proves neither retention nor loss. The original
whole-source donor-loss conclusion at \(\lambda=0\) remains valid from
its narrower, separately preserved enclosure. It is not extended to all
\(\lambda\) by the present comparison.

Equation (24) is an analytic consequence of freshly admitted complete
laws and preserved reference evidence. It is not a new reserved response,
an evaluation of changed-law trajectories or an experimental measurement.
The law interval and source are unchanged, and the result does not come
from a scan that selects favorable parameters. The global Euclidean
comparison discards some parameter and state correlations; its uncertain
donor reading records that limitation explicitly.

The new statement changes the quantified model class: finite receiver
nonacquisition holds uniformly over the admitted interval, even though
its true first-boundary minima and same-state regional powers differ.
Those static constitutive changes therefore do not suffice to produce
receiver formation on this source and horizon. They do not rule out
later acquisition, other preparations or other justified complete laws.
An all-future claim would need a source-valid invariant domain or another
complete-state argument retaining form, contacts and own-law dissipation;
neither this finite comparison nor a donor endpoint label supplies it.

The internal research entry point
[`compare_constitutive_receiver_prefix(root)`](../../src/tnfr/research/constitutive_receiver_comparison.py)
re-admits the fixed source, reads the pinned archive and applies the shared
[`_compare_constitutive_receiver_step`](../../src/tnfr/physics/_constitutive_receiver_comparison.py)
kernel. Its report retains the numerical premises and makes the unavailable
donor endpoint explicit. From the repository root, reproduce the source,
scalar comparison, kernel and complete-prefix checks with:

```sh
python -m pytest tests/physics/test_constitutive_receiver_source.py tests/physics/test_constitutive_receiver_comparison.py tests/research/test_constitutive_receiver_evidence.py tests/test_comparison_flow.py -q
```

These checks rebuild bounds from the retained evidence; they neither regenerate
the reference nor evaluate a changed-law trajectory.
