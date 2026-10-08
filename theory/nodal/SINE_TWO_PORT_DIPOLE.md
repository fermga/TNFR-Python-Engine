# An interior phase probe of the acquired two-port deformation

<a id="sine-two-port-dipole"></a>

## Question and independent preparation

The [uniform form probe](SINE_TWO_PORT_PROBE.md#sine-two-port-probe-result)
certifies supplied-contact transmission and retained identity, but pure
form diffusion also transmits that input. This result instead supplies the
same interior phase dipole to the acquired two-port pair and an independently
evolved unjoined pair. It asks whether the different donor bulk angles
produce distinguishable finite local form increments. A separate complete
phase-blind alternative tests whether ordinary form diffusion alone can
supply that new contrast under the same preparation and input.

This is a new experiment from the unchanged original capture preparation.
The earlier uniform form intervention is not inserted into its trajectory.
There is no equilibrium reset, reduced source uncertainty or response-selected
warmup. The [capture handoff](SINE_TWO_PORT_CAPTURE.md#sine-two-port-capture-result)
and its explicit retained-execution premise supply the joined state at
\(\sigma_*=1025\). The source still contains all independent nodewise
form and radian phase errors of magnitude at most \(1/65536\).

Supply the same original nominal nodal phases and error allowances to the
unjoined control, whose support consists of the two isolated C9 cycles from
preparation onward. Independent errors are allowed in the two experiments.
Common phase origins are removed only in proof coordinates; the components
are not physically rotated or reset. The control is not obtained by deleting
contacts from an evolved joined state.

## Complete laws, means and existing trapping domains

The joined graph has donor nodes `0,...,8`, receiver nodes `9,...,17`,
unit cycle edges, and unit contacts `(0,9)` and `(1,10)`. Its ports have
degree three and other nodes degree two. The unjoined graph omits the
contacts and has degree two at every node. In either graph let \(L\)
be its own Laplacian, \(M\) its own degree matrix, \(A=M^{-1}L\),
and \(f(\theta)=M^{-1}S(\theta)\), with
\(S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i)\). Keep unit held
capacities, beta one, \(e=1023/1024\), \(w=1/1024\), and
\[
\gamma=\frac1{1023\pi},\qquad \eta=\gamma^2,\qquad
\tau=et,\qquad \sigma=\eta\tau.
\]
The complete sine rows between events remain
\[
x_\tau=-Ax+\gamma f(\theta),\qquad
\theta_\tau=\gamma Ax. \tag{1}
\]
All thirty-six form and phase coordinates of each experiment remain.

Let \(P_M\) subtract each connected component's conserved weighted
mean. The joined graph has one form and one lifted phase mean; the control
has separate form and phase means in each C9. Write \(y=\theta-\theta_*^m\)
for the difference from the corresponding exact target with those means.
The joined target is the unique compatible \((2,1,0)\) geometry. The
control target consists of the two uniform twists with periods two and one.
On these respective relative spaces use the degree norm and the constants
\[
\lambda=\frac1{90},\quad \Lambda=2,\quad c_* =\frac1{25},\quad
R=\frac1{12},\quad B=\frac1{648000}. \tag{2}
\]
The joined normalized spectral bounds are
\(\lambda I\preceq A\preceq\Lambda I\). Each isolated ring has
the stronger lower gap \(1/18\), by its degree mass eighteen and
diameter four; the weaker common \(\lambda\) remains valid on the
control's componentwise relative space.

The joined target's minimum acute margin exceeds \(1/8\) radian.
The uniform control donor's margin is \(\pi/18>1/6\), and the
receiver's is larger. Thus, in either combined relative ball
\[
\mathcal R^2=\|P_Mx\|_M^2+\|y\|_M^2\le R^2,
\]
every phase segment from the target has edge cosine at least
\(\sin(1/24)>c_*\). The phase Hessian in the degree metric lies
between \(c_*A\) and \(A\). With the unchanged storage
\[
H=\tfrac12x^TLx+U(\theta),\qquad
U=\sum_{\{i,j\}\in E}[1-\cos(\theta_j-\theta_i)],
\qquad H_\tau=-\|Ax\|_M^2, \tag{3}
\]
criticality gives
\[
\frac{\mathcal R^2}{4500}\le H-H_*\le\mathcal R^2
\]
within that ball. Its boundary storage is at least \(B\).

The previous joined capture theorem places the entire joined family inside
this strict trapping domain at \(\sigma_*\). For the full unjoined
control, the initial combined norm squared and storage are bounded by
\[
36r_x^2+36r_\theta^2
 =\frac{72}{65536^2}=\frac9{536870912}<B<R^2. \tag{4}
\]
This uses both components together after their own mean projections.
The same first-exit storage argument traps the full control from its initial
time onward. Consequently both experiments are already forward trapped at
the common warmup start
\[
\tau_*=1025\cdot1023^2\pi^2,\qquad
t_*=1025\cdot1023\cdot1024\pi^2. \tag{5}
\]

The common coarse bounds \(X_0=Y_0=R\) below apply to these trapped
states. They do not assert that every point of the independent product
\(\|P_Mx\|_M\le R,\ \|y\|_M\le R\) is in the trapping ball.
A generic warmup certificate retains forward trapping as an independent
hypothesis. Source acquisition and this hypothesis are not supplied by a
cached passing flag or by the two radii alone.

## A finite warmup in the original coordinates

The [strict modified-energy mechanism](SINE_FORMED_CLASS_MAINTENANCE.md#a-strict-lyapunov-bound-on-each-retained-acute-chart)
extends to this degree metric without any commutation assumption. The
following isometry makes the reuse explicit. On the appropriate relative
space put
\[
\widehat A=M^{-1/2}LM^{-1/2},\quad
\widehat x=M^{1/2}P_Mx,\quad \widehat y=M^{1/2}y,
\quad \xi=\widehat A^{-1/2}\widehat y,
\quad v=\gamma\widehat A^{1/2}\widehat x.
\]
The nullspace of \(\widehat A\) is removed componentwise. Thus these
inverse square roots are well defined, and Euclidean norms of hatted
coordinates equal the original degree norms. Define
\[
W(\xi)=\eta\left[
 U(\theta_*^m+M^{-1/2}\widehat A^{1/2}\xi)-U(\theta_*^m)
\right].
\]
Differentiating both complete rows gives exactly
\[
\xi'=v,\qquad v'=-\widehat A v-\nabla W(\xi). \tag{6}
\]
No companion row or original form coordinate is discarded.

Use the rational bounds
\[
\underline\eta=\frac1{11000000}<\eta<
\overline\eta=\frac1{9000000},\qquad
\mu=\underline\eta c_*\lambda^2,\quad
M_W=\overline\eta\Lambda^2.
\]
The trapped-chart Hessian satisfies
\(\mu I\preceq\nabla^2W\preceq M_WI\).
Set \(\epsilon=\lambda/4\) and
\[
V=\tfrac12\|v\|^2+W+
 \epsilon\xi^Tv+\tfrac\epsilon2\xi^T\widehat A\xi.
\]
Young's inequality and cancellation of the mixed damping terms yield
\[
\begin{aligned}
a_-&=\mu/2+\lambda^2/16,\\
a_+&=M_W/2+\epsilon\Lambda/2+\epsilon^2,\\
\tfrac14\|v\|^2+a_-\|\xi\|^2
 &\le V\le\tfrac34\|v\|^2+a_+\|\xi\|^2,\\
V'&\le-(\lambda-\epsilon)\|v\|^2
       -\epsilon\mu\|\xi\|^2\le-\kappa V,\\
\kappa&=\min\{4(\lambda-\epsilon)/3,\epsilon\mu/a_+\}
       =\frac1{2233865700000}. \tag{7}
\end{aligned}
All derivatives here are in fast time \(\tau\). This is a bound for
the actual nonlinear potential throughout the retained trapping domain,
not merely for the tangent matrix at equilibrium. The shared
\(\_sine\_lyapunov\) arithmetic kernels consume precisely these
isometric Euclidean coefficients and norms.

For initial norm bounds \(X_0,Y_0\), the modified-energy upper bound
is
\[
V_0=\frac34\overline\eta\Lambda X_0^2
       +\frac{a_+}{\lambda}Y_0^2.
\]
With \(X_0=Y_0=R\), this is \(4512863/2592000000\).
Fix the finite common warmup
\[
N=128,\qquad T=N/\kappa=285934809600000
\quad\hbox{in fast time}. \tag{8}
\]
Since \(\exp(1)>2\), the exact bound
\(\exp(-\kappa T)=\exp(-128)<2^{-128}\) needs no numerical exponential
evaluation. Retain that rational upper bound directly; do not round it
through an interval grid and then silently restore discarded precision.
The original returned norms satisfy
\[
\begin{aligned}
\|P_Mx(T)\|_M^2
 &\le\frac{4V_0 2^{-128}}{\underline\eta\lambda},\\
\|y(T)\|_M^2
 &\le\frac{\Lambda V_0 2^{-128}}{a_-}. \tag{9}
\end{aligned}
Exact rational comparison makes both quantities strictly less than
\(2^{-80}\). Their approximate upper bounds are respectively
\(2.03\times10^{-32}\) and \(1.33\times10^{-36}\); these displays
are not replacement thresholds. Hence use the common pre-probe allowances
\[
X=Y=2^{-40}. \tag{10}
\]
The warmup retains the entire original source family. These are proved
finite-time output bounds, not narrowed preparation errors.

The probe time is \(\tau_p=\tau_*+T\), or
\(t_p=t_*+1024T/1023\). Equivalently its slow time is
\(1025+T/(1023^2\pi^2)\). The large conservative structural dwell
makes no claim of practical speed or laboratory duration.

## Identical interior event and local observation

Fix
\[
q=e_4-e_5,\qquad a=2^{-12},\qquad h=2^{-10}.
\]
At the common \(\tau_p\), supply one event in each sine experiment:
\[
\theta^+=\theta^-+a q,\qquad x^+=x^-. \tag{11}
\]
This is an externally supplied instantaneous phase action. It is not an
autonomous event selector, finite-duration pressure pulse or source reset.
Nodes four and five have degree two, the same neighbors, and the same
three affected donor edges `(3,4)`, `(4,5)`, `(5,6)` in both supports.
Consequently
\[
\|q\|_M=2,\qquad \|q\|_{M^{-1}}=1,\qquad q^TLq=6. \tag{12}
\]
The event preserves all relevant weighted form and phase means, including
the control's separate component means.

At elapsed fast time \(h\), observe the same linear functional in
both experiments:
\[
Y_G=q^T[x_G(h)-x_G(0^-)],\qquad
C=Y_J-Y_0, \tag{13}
\]
where \(J\) denotes joined and \(0\) unjoined. Thus this probe uses
the identical local coefficient vector, not the different receiver-mean
weights of the previous experiment. The arbitrary common form origins
cancel exactly. Four scalar readings of \(q^Tx\), two per experiment,
have independent absolute errors at most
\[
\delta=2^{-50}.
\]
Here the error contract is per scalar form-difference reading, not per
individual node sensor. For the recorded contrast,
\[
|\widehat C-C|\le4\delta. \tag{14}
\]
No favorable correlation of those errors is assumed. The added original
readout duration is \(1024h/1023=1/1023\).

## Exact local coefficient and a finite full-law error

Let \(A_*\) be the joined donor short-arc coordinate in turns. The
compatible donor's uniform bulk angle is
\[
b_J=\frac{2\pi(2-A_*)}{8},\qquad b_0=\frac{4\pi}{9}.
\]
Both targets are critical. At either target, the phase jump changes the
three affected oriented bulk gaps to \(b+a,b-2a,b+a\). Directly
summing the two actual degree-two sine rows gives
\[
q^Tf(\theta_*+a q)=\sin(b-2a)-\sin(b+a)
 =-2\sin(3a/2)\cos(b-a/2). \tag{15}
\]
Thus the exact-equilibrium leading contrast is
\[
L=2\gamma h\sin(3a/2)
   [\cos(b_0-a/2)-\cos(b_J-a/2)]. \tag{16}
\]
This local coefficient depends on the acquired bulk deformation. Equal
local adjacency fixes this immediate calculation; it does not make the
two full-graph propagators equal at finite time.

The previously certified strict root bracket has upper endpoint
\(6573559493/38654705664<7/40\) turns. Consequently
\[
b_J-b_0>\frac{17\pi}{1440}>\frac{17}{480}.
\]
For the fixed \(a\), every angle between \(b_0-a/2\) and
\(b_J-a/2\) lies below \(\pi/2\) and within \(1/5\) of it:
\(\pi/18+a/2<1/5\). Therefore its sine exceeds \(49/50\), and
\[
\Delta_c:=\cos(b_0-a/2)-\cos(b_J-a/2)
 >\frac{833}{24000}>\frac1{32}. \tag{17}
\]
This is an analytical design bound from the existing root evidence. The
new fixed certificate also requires fresh target admission and an outward
positive bound \(\Delta_c>1/32\); it does not treat an old target
box or rounded midpoint as a new equilibrium.

To retain the actual residual states, put \(g=1/3069>\gamma\),
\(c_h=2gh\) and
\[
Q_{\max}=\frac{X+c_h(Y+2a)}{1-c_h^2},\qquad
P_{\max}=Y+2a+c_hQ_{\max}. \tag{18}
\]
The denominator is positive. On each graph the semigroup
\(T_G(s)=e^{-sA_G}\) contracts its own degree norm,
\(\|A_G\|\le2\), and \(f_G\) is globally two-Lipschitz in
that norm. Variation of constants for (1) gives whole-window relative
form and target-phase bounds \(Q_{\max},P_{\max}\), as in the
preceding probe. For \(\psi_0=\theta_*^m+a q\), it also gives
\[
\begin{aligned}
Y_G={}&q^T[T_G(h)-I]x^-+\gamma h q^Tf_G(\psi_0)\\
 &+\gamma\int_0^h q^T[T_G(h-s)-I]f_G(\psi_0)\,ds\\
 &+\gamma\int_0^h q^TT_G(h-s)
       [f_G(\theta(s))-f_G(\psi_0)]\,ds.
\end{aligned} \tag{19}
\]
Since \(\|T_G(s)-I\|\le2s\),
\(\|f_G(\psi_0)\|_M\le4a\), and
\(\|\theta(s)-\psi_0\|_M\le Y+2ghQ_{\max}\), the three
nonleading terms have combined absolute bound
\[
E=2hX+4gah^2+2ghY+4g^2h^2Q_{\max}. \tag{20}
\]
This includes the pre-existing form background, phase error, nonlinear
phase motion and both actual supports' finite propagation. No linearized
law, identical heat kernel, hidden equilibrium reset or favorable source
symmetry is substituted for the complete flow.

Combining (14)--(20) gives
\[
L-2E-4\delta\le\widehat C\le L+2E+4\delta. \tag{21}
\]
For a prospective sufficient rational check, use
\(\gamma>1/3216\) and
\(\sin(3a/2)>(5/4)a\), the latter following from
\(\sin z\ge z-z^3/6\) with \(0<z=3a/2<1\). Equations
(16)--(17) imply
\[
L>\frac{5ah}{64\cdot3216}.
\]
Exact rational substitution of the fixed constants proves
\[
\frac{5ah}{64\cdot3216}-2E-4\delta> C_{\min}=2^{-38}. \tag{22}
\]
For scale, this deliberately loose observed lower bound is greater than
\(5.17\times10^{-12}\), while the threshold is approximately
\(3.64\times10^{-12}\). These are derived design margins, not an
evaluated response or a claim of sensor resolution in physical units.

## Event work, winding identity and later recovery

The event changes only phase storage. At the target its exact work is
\[
W_*(b)=3\cos b-2\cos(b+a)-\cos(b-2a). \tag{23}
\]
Criticality cancels the first variation. Along the target jump segment,
\(q^TLq=6\) and the common cosine bound give
\(3c_*a^2\le W_*\le3a^2\).
For any actual pre-probe phase error of norm at most \(Y\), the
global Hessian norm bound two gives
\[
|W-W_*|\le4aY.
\]
Therefore each sine experiment has the signed event-work enclosure
\[
3a^2/25-4aY\le W\le3a^2+4aY. \tag{24}
\]
At the fixed constants its lower endpoint is positive, and its upper
endpoint is strictly below the common allowance
\(W_{\rm allowed}=2^{-21}\).

The post-event combined radius and storage satisfy the strict admissions
\[
X^2+(Y+2a)^2<R^2,\qquad
X^2+Y^2+3a^2+4aY<B. \tag{25}
\]
These inequalities apply to both the joined graph and the full unjoined
control on their respective mean leaves. Continuous loss then traps each
post-event family for all later uninterrupted times. The periods remain
\((2,1,0)\) in the joined graph and \((2,1)\) in the control.
LaSalle's argument on the compact relative sublevel, followed by uniqueness
of the acute target in each component or sector, gives recovery to the
same phase geometry and conserved form means. This event shifts none of
those means.

The work in (24) is supplied separately from the loss in (3). A positive
work allowance is not an inferred reservoir or passive event. One admitted
phase probe does not establish autonomous maintenance against arbitrary
inputs, contact selection or physical binding.

## A complete phase-blind alternative and its bounded background

On each of the same supports consider the explicitly different complete law
\[
x_\tau=-Ax,\qquad \theta_\tau=\gamma Ax. \tag{26}
\]
It has the same capacities, clocks and original preparation family, and
receives the same phase event (11) at \(\tau_p\). Its form row is
independent of phase. Thus kicked and un-kicked trajectories with the same
pre-event state have exactly identical form for all later times. This is
an exact causal null; an arbitrary before/after raw increment need not be
zero because an initial form background can still relax.

That background is independently bounded for the unchanged source.
Initially \(\|P_Mx\|_M\le7/65536\) on either graph, with
componentwise projection for the control. Since
\(\kappa\le\lambda\), (8) implies \(\lambda T\ge128\).
Without using a sine-law warmup to certify (26), its own heat contraction
therefore gives
\[
\|P_Mx(\tau_p)\|_M
 \le\frac7{65536}e^{-\lambda(\tau_*+T)}
 <\frac7{65536}2^{-128}<X. \tag{27}
\]
Each raw increment (13) has magnitude at most \(2hX\). Including
the four same readout-error allowances, its recorded joined-minus-unjoined
contrast obeys
\[
|\widehat C_{\rm blind}|\le4hX+4\delta
 =2^{-47}<2^{-38}. \tag{28}
\]
Thus the finite contrast separates the stated sine-law experiment from
this phase-blind alternative under the common preparation and observation
contracts. No winding or storage-trapping conclusion for (26) is needed
for its form response bound, and the sine potential is not assigned as
that alternative's conserved or dissipated storage.

This separation is not unique identification of the sine law. Other
phase-sensitive laws or preparations can share such a response. The result
relates one coefficient to the acquired internal deformation and bounds
the complete finite-time confounders; it does not define a topology-free
material constant, identify a physical constituent or supply an independent
laboratory measurement bridge.

## Prospective protocol and stopping boundary

<a id="sine-two-port-dipole-protocol"></a>

Freeze the following before the first new response-certificate assessment:

- Retain the original thirty-six-coordinate preparation uncertainty,
  support-specific complete sine rows, held capacities and all clocks.
  Re-admit the original joined capture handoff without rerunning its
  producer; retain the declared Picard, remainder and endpoint-center
  execution premise. Separately admit the full unjoined source and its
  initial trapping bound (4).
- Both source families continue unprobed to (5). Use the already-trapped
  norm envelopes \(X_0=Y_0=1/12\), common constants (2), rational
  \(\eta\) bounds above, \(N=128\), \(T=N/\kappa\), and the exact
  upper decay \(2^{-128}\). Require the original-coordinate squared
  returns in (9) to be strictly below \(2^{-80}\). No root, source,
  dwell, precision or error-budget search is admitted.
- At the fixed \(\tau_p\) supply exactly (11), with \(q=e_4-e_5\)
  and \(a=2^{-12}\), in both sine models and the declared phase-blind
  alternatives. Form and phase means retain their own support-dependent
  conservation laws; no relative state is reset.
- Use \(h=2^{-10}\), the identical local form-increment map (13),
  per-scalar-readout error \(2^{-50}\), contrast threshold \(2^{-38}\)
  and per-event work allowance \(2^{-21}\). Keep all four errors.
- Rebuild the compatible `(2,1)` target with thirty-two outer and
  sixty-four inner strict-sign refinements. Require the fresh target's
  analytic criticality and acute margin above \(1/8\), and the outward
  local cosine contrast \(\Delta_c>1/32\). The implicit root remains
  correlated; its midpoint or interval product is not an actual target.
- Rebuild the full-law error (18)--(21), strict response margin (22),
  positive work and allowance (24), both post-event trapping admissions
  (25), and the alternative's independent warmup and upper response (27)--(28).
  Equality at a strict threshold is unavailable. A candidate bound without
  its target, source or trapping premises cannot supply a passing result.
- Use shared exact rational and outward dyadic128 trigonometric arithmetic
  for the fresh target and coefficient enclosure. The warmup uses the
  proved elementary exponential bound, not numerical integration. There
  is no new trajectory, adaptive response search or repeated capture run.
- Archive this prospective proof, protocol and complete producing source
  before assessment; retain the response, verdict and any failure separately.
  An unavailable sufficient bound is not observed dynamical failure and
  cannot be replaced by a longer dwell, smaller source or different probe
  without a separate prospective experiment.

The endpoint-domain and warmup mathematics is conditional on its stated
trapping premises. The fixed research record additionally binds those
premises to the original full source families. At this declaration no new
dipole response report has been assessed; any actual retained outcome must
be appended separately. The analytical design margins above do not replace
that execution record or alter the preceding gates' frozen evidence.

## Retained first dipole result

<a id="sine-two-port-dipole-result"></a>

The first assessment of the unchanged frozen protocol was saved successfully
as `certified_dipole`, with no unavailable reasons. All thirteen stopping
conditions passed. They include the actual joined source handoff, the full
unjoined source's trapping bound, finite warmup, fresh target and local cosine
contrast, both nonlinear error budgets, four readout errors, separation from
the phase-blind alternative, positive supplied work within both allowances,
and preservation and recovery of both winding identities.

The retained report contains exact rational and outward interval bounds.
The following approximate displays describe those bounds; they are not
measured responses, sampled trajectories or replacement thresholds:

| Retained quantity | Approximate bound |
| --- | ---: |
| Joined-minus-unjoined recorded contrast | `[8.39509270e-12, 9.62315088e-12]` |
| Lower response margin above `2^-38` | `4.75711390e-12` |
| Absolute recorded phase-blind contrast upper bound | `7.10542736e-15` |
| Lower separation from the phase-blind bound | `8.38798728e-12` |
| Local cosine difference | `[0.0404805704426, 0.0404805704629]` |
| Complete-law remainder per sine model | `3.05238185e-13` |
| Joined supplied phase-jump work | `[2.37655527e-8, 2.38874643e-8]` |
| Unjoined supplied phase-jump work | `[3.10650433e-8, 3.10650451e-8]` |
| Joined post-event capture-storage margin | `1.51932241e-6` |
| Unjoined post-event capture-storage margin | `1.51214483e-6` |

The full warmup returned squared-norm bounds of approximately
`2.02615606e-32` for original relative form and `1.32621120e-36` for
phase. Both are strictly below `2^-80`, so the declared full-family
pre-probe allowances `X=Y=2^-40` are admitted. These bounds follow from
the unchanged finite duration `285934809600000` in fast time and the
strict modified-energy argument. They do not shrink the original source
errors, substitute an asymptotic target for the actual state, or imply a
short laboratory recovery time.

The positive contrast concerns the identical interior phase action and
the identical local form-difference observation on both sine supports.
Its leading coefficient depends on the acquired donor bulk-angle
deformation; the retained remainders account for both complete nonlinear
flows and their different full-support propagators. The recorded lower
bound remains above the threshold after every permitted initial residual
and all four observation errors. The phase-blind alternative's own
source-to-probe contraction bounds its remaining form background below
the separate comparison ceiling.

Consequently every member of the original prepared sine family, with
its own conserved means, satisfies this finite discrimination after the
proved acquisition and warmup. The joined periods remain `(2,1,0)` and
the unjoined periods remain `(2,1)` through the phase input and subsequent
uninterrupted flow. Both recover to their respective phase geometries and
unchanged form means. Supplied event work is positive and bounded in each
model, and remains distinct from later continuous dissipation.

The source handoff retains the earlier capture execution as a premise.
Its read-only audit re-admits primary preparation and law data, strict
target brackets, the saved metric chain and the complete-state tail;
validity of the archived Picard inclusions, local remainders and endpoint
centers remains explicit. This gate ran no new capture trajectory or
probe trajectory, and did not rerun the original capture producer.

The retained
[protocol](../../docs/assets/sine_formed_classes/two-port-dipole-v1.protocol.json),
[producing source archive](../../docs/assets/sine_formed_classes/two-port-dipole-v1.source.zip),
[exact report](../../docs/assets/sine_formed_classes/two-port-dipole-v1.json)
and [manifest](../../docs/assets/sine_formed_classes/two-port-dipole-v1.manifest.json)
preserve this first assessment and its thirteen stopping conditions.
The source archive SHA-256 is
`68ea24ad6301a53fa492fa6b8f53409cb910e700f7d0b9ec7e12e4b59285fb50`,
based on revision `4183b788ee63a61e1c44afc08b7b3950a0bdd311` with the
two declared runtime overlays. The prospective proof remains a
byte-identical prefix of this owner. Source hashes establish recoverable
content associations, not independent chronology or execution authentication.

The result establishes conditional finite sensitivity to the acquired
internal geometry and separates the declared phase-blind alternative.
Other phase-sensitive laws can share this response. It does not select a
unique fundamental law, remove the dependence on supplied topology,
derive an autonomous probe or contact rule, or identify physical binding
or a laboratory measurement map.
