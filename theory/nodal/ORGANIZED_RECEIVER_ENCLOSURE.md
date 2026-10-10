# One source-preserving enclosure of the organized receiver

<a id="organized-receiver-enclosure"></a>

This owner defines one prospective finite enclosure for the unchanged
two-donor preparation. It asks whether the initially unprepared receiver
acquires winding one while both donors retain that identity and the complete
state enters the joined capture region. It also admits a certified donor-loss
outcome. Failure of a sufficient enclosure or exhaustion of its frozen budget
is unavailable evidence, not proof that receiver acquisition is impossible.

The [slow-admission theorem](ORGANIZED_RECEIVER_SLOW_ADMISSION.md)
supplies the source-inclusive phase inclusion. The numerical reference below
is an instrument for enclosing the actual complete law; it is not a
replacement phase-only model and does not discard form or preparation errors.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
remains the sole task queue.

## Complete model and retained source

Use exactly the [source and support](ORGANIZED_RECEIVER_MECHANISM.md#organized-receiver-source)
of the receiver-mechanism theorem: three labelled C9 regions, unit internal
weights, and reciprocal contacts from the oriented region pairs
\((0,1),(1,2),(2,0)\) with offsets zero and one, each of weight one half.
All 27 nodes have weighted degree four. Support is present from preparation;
capacities remain one. With \(A=L/4\),
\[
 F_i(\theta)=\frac14\sum_jw_{ij}\sin(\theta_j-\theta_i),\qquad
 x_\tau=-Ax+\gamma F(\theta),\qquad
 \theta_\tau=\gamma Ax,
 \quad \gamma=\frac1{1023\pi},\quad \tau=\frac{1023}{1024}t.
\]
The law is detached reciprocal sine dynamics, without forcing, events,
resets or changing conductances. It does not execute a registered operator.

The original form ramps remain on regions zero and one only:
\[
 x_{r,j}(0)=m(j-4)+\eta_{x,r,j}\quad(r=0,1),\qquad
 x_{2,j}(0)=\eta_{x,2,j},\qquad
 \theta_{r,j}(0)=\eta_{\theta,r,j},
 \quad m=\frac{2046}{9}(355/113)^2.
\]
Every form and phase error has magnitude at most \(10^{-10}\), with separate
exact zero sums in each original region. They need not share the nominal
symmetry. Only the global form and phase means are conserved during the
connected evolution. Retain the continuous phase lifts and the original
storage, including the initial contact cost; no recovered-source certificate
is substituted.

The earlier theorems certify donor winding one at \(\tau=200\), its retention
through \(\tau=400\), and receiver winding zero at release. Set
\[
 u=\gamma^2(\tau-400),\qquad
 P=\alpha(q,q,0),\quad \alpha=2\pi/9,\quad q_j=j-4.
\]
For every allowed original source, the freshly reconstructed release bounds
supply
\[
 \|\theta(0)-P\|_2\le b_0,\qquad
 \theta_u=F(\theta)+r(u),\quad \|r(u)\|_2<\rho=11/10^6,
 \qquad \|x(u)\|_2\le X_0.                                  \tag{1}
\]
Here \(b_0\) and \(X_0\) are the outward source bounds at structural time
400, not newly chosen preparations. The global means of \(\theta\), \(F\)
and \(r\) are zero. The actual defect \(r\) retains the form memory specified
by the exact second-order law. The inclusion merely bounds that defect; it
does not supply an arbitrary forcing to the original model.

<a id="organized-receiver-enclosure-protocol"></a>
## Frozen numerical policy and observations

The finite policy is fixed before evaluating any reference response:

| Item | Declared value or rule |
| --- | --- |
| Elapsed slow horizon | \(u=32\), starting at \(\tau=400\) |
| Step and maximum count | \(h=1/8\), at most 256 steps |
| Taylor order | Eight, with order-nine whole-tube remainder |
| Arithmetic | Shared outward 128-bit dyadic rational intervals |
| Reference | Thirteen nominal reflection/sign coordinates |
| Complete-state uncertainty | One Euclidean phase ball on the full 26-dimensional global mean-zero leaf, plus the carried form norm \(X_0\) |
| Velocity discrepancy | Fixed \(\rho=11/10^6\), with original-source admission rebuilt |
| Local growth search | Fixed bracket \([-1,2]\), eight bisections; no bracket expansion |
| Cosine majorant | Downward 20-bit dyadic bounds, minimized on reflected edge orbits |
| Prospective phase radius | \(5/4\) times the global one-step envelope plus \(2^{-80}\) inside that factor |
| Uncertainty budget | Stop if the carried endpoint phase-error radius exceeds \(1/2\) |
| Adaptation | No step refinement, horizon extension, source retuning or alternative route |
| Positive observation | Continuous donor retention and one common endpoint passing full target-cell and storage-capture conditions |
| Negative observation | A robust endpoint winding different from one for at least one donor |
| Other stops | Validation failure, uncertainty budget or horizon: unavailable for the requested acquisition claim |

The original-time endpoint corresponding to the full horizon is
\((1024/1023)(400+32/\gamma^2)\). This is a structural time, not laboratory
seconds. The single frozen run retains the protocol, source/configuration,
ordered support, exact bounds, partial valid prefix, response and verdict.
A protocol identity associates those artifacts; it does not authenticate an
external physical preparation.

## A nominal symmetry that does not restrict source errors

Let \(R\) be the node permutation
\[
 R(0,j)=(1,8-j),\quad R(1,j)=(0,8-j),\quad R(2,j)=(2,8-j).
\]
It preserves the complete weighted support, has thirteen two-node orbits and
one fixed node \((2,4)\), and satisfies \(R^2=I\).
Because sine is odd and the support is invariant,
\(F(-Rz)=-RF(z)\). Thus the antisymmetric space
\(\mathcal V_-:=\{z:Rz=-z\}\) is invariant for the reference equation
\(y_u=F(y)\), and \(P\in\mathcal V_-\).

Choose one node \(a_k\) from each pair and define
\(C_-e_k=e_{a_k}-e_{Ra_k}\). The nominal reference can be integrated in
thirteen coordinates, with lift \(y=C_-v\) and
\[
 C_-^TC_-=2I,\qquad \|C_-v\|_2^2=2\|v\|_2^2.              \tag{2}
\]
The full initial-error family is not projected onto this space. Its transverse
directions stay in the radius carried under (1). Choosing a symmetric
reference changes the cost of the calculation, not the quantified source.

At the initial reference, materialize a rational midpoint of the interval
enclosure of each retained coordinate of \(P\). Its lifted outward rounding
error is added to \(b_0\). At every later step, the rational reference center
comes from the preceding Taylor endpoint enclosure. Its complete lifted
recentering error is added to the carried radius. No recentering resets the
actual uncertainty to zero.

<a id="organized-receiver-enclosure-step"></a>
## Validated reference and full-state error transport

At one step, let \(c\in\mathcal V_-\) be the rational reference center and
let \(e\) bound the Euclidean difference between the actual phase and \(c\).
The shared metric Taylor owner is called on the thirteen coordinates with
metric \(2I\), initial radius zero, growth bound two, duration \(h\) and
order eight. Strict Picard inclusion gives a whole-time reference tube;
the normalized solution jets and whole-tube order-nine remainder give its
endpoint enclosure. The metric is exactly the lifted norm (2).

Radius zero applies only to this local exact reference solution. The actual
source error \(e\) remains external and is propagated by the following
comparison. In particular, the reference validation is not evidence that
the actual full law starts at its center.

The global Jacobian bound \(\|DF\|_2\le2\) holds on all phase lifts.
Writing
\[
 \phi(\mu,s)=
 \begin{cases}(e^{\mu s}-1)/\mu,&\mu\ne0,\\s,&\mu=0,\end{cases}
\]
it gives the prospective global discrepancy bound
\[
 G=e^{2h}e+\rho\phi(2,h),\qquad
 R_*=\tfrac54(G+2^{-80}).                                  \tag{3}
\]
All arithmetic in (3) is outward. The convex sum of the lifted reference
Picard box and the mean-zero ball of radius \(R_*\) contains both local
solutions throughout the step. The strict margin follows from \(R_*>G\),
not from presuming the validity of a future local growth estimate.

For each fine edge, its reference phase difference is bounded from that
box; the complete discrepancy enlarges the edge interval by
\(\sqrt2R_*\) on both sides. This factor is the norm of the edge incidence
vector. It retains arbitrary mean-zero errors, including errors outside
\(\mathcal V_-\).

Let \(\ell_e\) be a lower bound for cosine on the enlarged edge interval.
Round it downward to the declared dyadic grid, with the exact global floor
\(-1\), and take the minimum over the reflected edge orbit. For every point
in the prospective convex domain,
\[
 DF=-\tfrac14 B\operatorname{diag}(w_e\cos\delta_e)B^T
 \preceq J_*:=-\tfrac14B\operatorname{diag}(w_e\ell_e)B^T.   \tag{4}
\]
This Loewner bound follows term by term from
\(w_e(\cos\delta_e-\ell_e)b_eb_e^T\succeq0\), including
edges with negative cosine. Independent interval matrix entries are not
mistaken for an exact Jacobian.

The orbit minimum makes \(J_*\) commute with \(R\). Both \(J_*\) and the
actual Jacobians annihilate the common-phase vector. To certify an upper
logarithmic growth rate on the entire mean-zero leaf, the calculation uses
both reflection sectors. The antisymmetric sector has basis \(C_-\) and
Gram matrix \(2I\). A basis for the symmetric mean-zero sector is
\[
 C_+e_k=e_{a_k}+e_{Ra_k}-2e_{(2,4)},\qquad
 C_+^TC_+=2I+4\mathbf1\mathbf1^T.                         \tag{5}
\]
These orthogonal sectors each have dimension thirteen and together span
all 26 mean-zero directions. Hence the two exact rational PSD tests
\[
 \mu C_\pm^TC_\pm-C_\pm^TJ_*C_\pm\succeq0                 \tag{6}
\]
prove \(DF|_{\mathbf1^\perp}\preceq\mu I\). Testing only the
antisymmetric block would omit source directions and is insufficient.
The upper search endpoint two is valid because \(\ell_e\ge-1\) and
\(J_*\preceq L/4\preceq2I\). The lower search endpoint minus one is
a search policy, not an asserted lower bound on the actual logarithmic norm.
Each retained rate is checked by exact rational semidefinite elimination.

Pairing the actual-reference difference equation with its normalized error
and using (1), (4)--(6) yields
\[
 D^+\|\theta-y\|_2\le\mu\|\theta-y\|_2+\rho.
\]
Therefore a whole-step discrepancy bound is
\[
 E(s)=e^{\mu s}e+\rho\phi(\mu,s),\quad 0\le s\le h,
 \qquad E_{\max}=\max\{e,E(h)\}.                         \tag{7}
\]
The scalar solution is monotone or constant on a step, so its maximum is
at an endpoint. Outward scalar exponential enclosures handle either sign
of \(\mu\). Strict inclusion \(E_{\max}<R_*\) is checked explicitly.
An unavailable inclusion stops the validated prefix; it cannot be replaced
by a favorable endpoint alone.

Let \(d\) be the full \(2I\)-metric Taylor endpoint recentering error.
The next exact rational reference center and carried full phase radius obey
\[
 e_{\rm next}=E(h)+d.                                      \tag{8}
\]
No observation box is converted back into a smaller radius, and neither
the Taylor error nor the complete-law defect is discarded. The reference
may change at the step boundary; (8) accounts for its entire jump. The
original complete trajectory itself has no reset or discontinuity.

<a id="organized-receiver-enclosure-observations"></a>
## Identity, capture and the meaning of a failed margin

For continuous donor retention, use the release winding-one integers on
each fixed internal donor edge. The whole reference tube, enlarged by
\(\sqrt2E_{\max}\), must keep every registered edge strictly within
\((-\pi,\pi)\). The previously proved time-200 to time-400 bridge covers
the interval before this calculation starts. Receiver branch crossing is
allowed: imposing the receiver's original winding-zero cell at every step
would exclude the requested acquisition by construction.

A failed sufficient donor margin is recorded permanently for the positive
whole-history claim, but does not end the enclosure. The actual state may
still be enclosed at later steps, and its donor winding may become decidable
there. For an endpoint, freshly determine principal-difference integers with
strict branch margins for every internal edge. Their signed sum gives the
integer winding; do not infer it from a rounded phase plot.

If a donor's entire endpoint ball has winding different from one, the
continuous full law must have lost that donor identity after its certified
acquisition. Since the ball encloses every original admitted source, such
a verdict is a whole-family donor-loss result. It refutes the requested
donor-retaining acquisition claim for this preparation and law. It does not
prove that the receiver can never acquire winding, that other preparations
cannot do so, or that all compatible TNFR laws share the outcome.

For positive joined capture, the endpoint center \(c_*\) and radius \(e_*\)
must pass the [same complete-state contract](ORGANIZED_RECEIVER_SLOW_ADMISSION.md#organized-receiver-capture-contract):
\[
 \max_{\text{fine }e}|\delta_e(c_*)|+\sqrt2e_*<2,           \tag{9}
\]
\[
 U(c_*)-U(\Theta)+\|\nabla U(c_*)\|_2e_*
       +4e_*^2+4X_0^2<196/625,\qquad
 \Theta=\alpha(q,q,q).                                   \tag{10}
\]
The fine-edge integers in (9) are those registered to \(\Theta\), not
freely chosen local seams. The potential, gradient and form contribution
refer to the same complete weighted support and same endpoint. The linear
gradient term in (10) remains present. Joined capture plus certified
continuous donor history gives the requested positive result and invokes
the existing future trapping and convergence theorem.

An endpoint capture inequality without that history may establish an
endpoint property, but does not repair unavailable donor retention. Likewise,
a reference donor losing winding without the complete error margins proves
nothing about the actual source family. Numerical validation failure,
phase-error budget exhaustion, unresolved winding or a horizon without
capture remain explicit unavailable outcomes; none is silently promoted to
nonformation.

<a id="organized-receiver-enclosure-result"></a>
## Evaluated result: loss of both donor identities

The first response to the [frozen protocol](../../docs/assets/sine_formed_classes/organized-receiver-enclosure-v1.protocol.json)
certifies **whole-family donor loss**. The
[response archive](../../docs/assets/sine_formed_classes/organized-receiver-enclosure-v1.response.zip)
preserves the first outcome JSON and every original step JSON byte-for-byte
at their recorded repository-relative paths. Its outcome contains the complete
source admission, ordered step manifest and terminal observation. This lossless
transport does not replace or regenerate any evaluated value. The
[freeze record](../../docs/assets/sine_formed_classes/organized-receiver-enclosure-v1.freeze.json)
and [source archive](../../docs/assets/sine_formed_classes/organized-receiver-enclosure-v1.source.zip)
preserve the evaluated implementation and policy. There were 76 accepted
steps, with no refinement or retry. The declared donor-loss stopping rule
ended the calculation at
\[
 u_*=19/2,\qquad
 \tau_*=400+\frac{19}{2\gamma^2},\qquad
 t_*=\frac{1024}{1023}\left(400+\frac{19}{2\gamma^2}\right).
\]
The final carried Euclidean phase radius is approximately
\(0.0031002376916330595<0.00311\). Every source in the original error family
has endpoint windings \((0,0,0)\). Each donor's principal branch margin at
that endpoint exceeds \(1/20\), including the complete source error and
velocity defect. This is a validated property of the actual complete law,
not a winding read from the nominal reference alone.

The retained history distinguishes the loss proof from the earlier failure
of a sufficient corridor:

| Retained endpoint | Actual winding enclosure | Continuous donor-history evidence |
| --- | --- | --- |
| Step 74, \(u=37/4\) | \((1,1,0)\) | Both donors retained through this entire prefix |
| Step 75, \(u=75/8\) | \((1,1,0)\) | The whole-step donor corridor is unavailable; its endpoint remains certified |
| Step 76, \(u=19/2\) | \((0,0,0)\) | Both donors have definitely lost winding one |

Thus, for each original source and each donor, a winding-one to winding-zero
transition occurs between \(75/8\) and \(19/2\). The first donor loss is
enclosed by \([37/4,19/2]\): retention is proved through the former time,
and loss is established by the latter. The narrower endpoint interval does
not establish the first or unique crossing, because the step-75 interior
margin is unavailable. That unavailable margin was not treated as a negative
verdict; validation continued to the strict endpoint loss certificate.

A further consequence follows from the retained full-time tubes without a
new trajectory evaluation. Reconstructing their original receiver edge
differences and enlarging by the certified \(\sqrt2E_{\max}\) gives a
strict winding-zero branch margin greater than \(13/10\) on every one of
the 76 intervals. The receiver therefore remains winding zero throughout
\(0\le u\le19/2\), including the donor losses. This is an additional
retained-tube consequence, distinct from the prospectively declared terminal
donor-loss verdict. Together with the earlier initial-layer exclusion, the
receiver has not acquired winding one anywhere from preparation to this stop.

The requested acquisition with donor retention and joined capture is
therefore false for this full source family under this complete law. Initial
motion toward the donor phases, reciprocal interaction, monotone complete
storage and the existence of the joined attracting target are insufficient
to establish that claim. The loss concerns the declared regional winding
identity; winding zero does not imply that the phase state is uniform or
that every possible organization has disappeared. Neither future receiver
acquisition nor global synchronization is decided by this finite result.

The [three-ramp acquisition theorem](ORGANIZED_JOINT_ACQUISITION.md) remains
valid under its different preparation. This result supplies a counterexample
to transferring its acquisition verdict to a locally unprepared receiver,
even though support, constitutive law and attracting target are shared.
Supplied support, donor preparation and structural clocks remain premises;
no physical identification or unique TNFR law follows. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the subsequent foundational question.
