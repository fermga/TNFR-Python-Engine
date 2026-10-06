# Native phase domains and capture

Regular-domain continuation, winding passage, reflected basins and the full-state acute-sector barrier, including capacity/storage-scale covariance and regularity margins.

Part of [Native relational exchange admission](RELATIONAL_EXCHANGE_ADMISSION.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 17. Regular-domain admission before a formation experiment

<a id="relational-formation-domain"></a>
<a id="relational-regular-domain-admission"></a>

### Four boundaries with different meanings

For the full continuous law of sections 1-3, write the local relative
resultant and displacement as

\[
z_i=\sum_{j\sim i}e^{i(\theta_j-\theta_i)},\qquad
\alpha_i=\operatorname{Arg}z_i\in(-\pi,\pi),\qquad
g_i=\alpha_i/\pi,\quad
H_i=\pi|z_i|\operatorname{sinc}\alpha_i.
\]

The open regular domain is `z_i` outside the nonpositive real axis for every
node. Here `H_i>0` and the field is smooth. Its definition uses only the
existing state and support. The separate [execution admission](#relational-full-regular-execution)
certifies sufficient point and proposal-chord margins; a domain definition
alone is not a numerical guarantee.

| Boundary | Consequence for the stated continuous law |
| --- | --- |
| An ordinary edge reaches `abs(gap)=pi/2` | The acute executor stops; the resultant and complete field can remain regular |
| An ordinary edge reaches antipodal separation | Its ordinary cycle winding is undefined at that instant; the nodal field need not be singular |
| A local resultant vanishes | Its direction and native phase source are undefined; the phase metric tends to zero |
| A nonzero resultant lies on the negative real axis | The principal displacement has its `-pi/pi` branch; its two source limits differ and `sinc(alpha)` tends to zero |

At zero resultant, different admitted approaches can give different limits
for the native source. The phase velocity also generically diverges when
`(Bx)_i` stays nonzero while `H_i` vanishes. Even if that numerator vanishes,
the law does not select a value for the resulting ratio or a source direction
on an open full-state neighborhood. Likewise the negative-real branch does
not become regular merely because the resultant magnitude is positive.
A special limiting trajectory would require its own existence and uniqueness
argument; assigning a direction or clipping `H^-1` would add a new law.

### A derived cycle invariant gives a genuine pure-cycle obstruction

<a id="relational-cycle-resultant-obstruction"></a>

On a pure `C_n`, `n>=5`, every node has exactly its two cycle neighbors.
Consequently

\[
|z_i|=|e^{i\theta_{i-1}}+e^{i\theta_{i+1}}|=0
\quad\Longleftrightarrow\quad
\theta_{i+1}-\theta_{i-1}=\pi\pmod{2\pi}.
\]

Consider the derived oriented edges `i -> i+2` modulo n. They form one
cycle when n is odd, and two parity cycles when n is even. Their non-antipodal
winding integers are observations of the same phases, not added support or
new primitive variables. Every such edge has a non-antipodal gap whenever
the original local resultants are nonzero. The ordinary cycle-period argument
from [the existing classification](FORCED_PHASE_LOCKING.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods)
therefore preserves these derived windings along any continuous path with
nonzero resultants, independently of the dynamics or the Arg-branch admission.

For an acute original phase geometry, let its oriented gaps be `delta_i` and
ordinary winding be ell. Each `delta_i+delta_(i+1)` is strictly inside
`(-pi,pi)`, so it is exactly the shortest two-hop gap. Summing gives

\[
\boxed{\quad
W_{\mathrm{skip}}=2\ell\quad(n\text{ odd}),\qquad
W_{\mathrm{even}}=W_{\mathrm{odd}}=\ell\quad(n\text{ even}).
\quad}
\]

For odd n the sum includes every original edge twice; each parity sum for
even n includes every original edge once. In particular, C5 consensus has
derived winding zero, while its acute winding-one twist has derived winding
two on `(0,2,4,1,3)`. **Every continuous fixed-C5 path between them must
encounter a zero local resultant.** Allowing obtuse ordinary edges or removing
the acute executor cutoff cannot remove this constitutive obstruction. The
same reasoning separates different acute winding sectors on any pure cycle.
It does not apply unchanged when extra neighbors contribute to a resultant.

Ordinary winding alone does not identify these regular components. For
example, on C5 take

\[
\theta=(0,\pi,3\pi/4,\pi/2,\pi/4).
\]

The edge `(0,1)` is antipodal, but the relative resultants are
`(-1+exp(i*pi/4), -1+exp(-i*pi/4), sqrt(2), sqrt(2), sqrt(2))`.
Every one is regular. Replacing `theta_1` by `pi+tau` for sufficiently small
positive/negative tau keeps all resultants regular while ordinary winding
changes from zero to minus one. The derived winding remains zero on both
sides. This is a regular geometric crossing, not yet an assertion that the
prescribed path is a trajectory, nor a connection to the acute minus-one
twist (whose derived winding would be minus two).

### A sharp C5 zero-resultant storage barrier

Suppose a C5 resultant vanishes. Its two neighbors are antipodal, so the
phase-storage contributions of the two edges incident at that node sum to
exactly two, independently of the node's own phase. The remaining three-edge
path connects antipodal endpoints. Fix them at `0,pi`, with internal phases
`A,B`. The sum of its three cosines is

\[
\cos A+\cos(B-A)-\cos B
\le \cos A+2|\sin(A/2)|
=\tfrac32-2(|\sin(A/2)|-\tfrac12)^2\le\tfrac32.
\]

Thus that path costs at least `3/2`, attained at `A=pi/3,B=2*pi/3`, and

\[
\boxed{\qquad V_\phi\ge\tfrac72
\quad\text{at every zero-resultant C5 configuration}.\qquad}
\]

The bound is sharp: choose those three equal path gaps; the remaining node
may have any phase. Nonnegative Dirichlet storage and the complete loss
identity imply the necessary budget `E(0)>=7*beta/2` for a trajectory to
reach such a configuration. With positive capacities and `e>0`, any
nonstationary phase evolution consumes positive integrated form loss, so a
trajectory reaching this barrier after that evolution needs strictly more
initial storage. Equality cannot pay both the barrier and that loss.

For initial phase consensus, the entire budget is initial form storage.
The final acute winding-one phase storage is only
`5*(1-cos(2*pi/5))`, approximately `3.454915`, below the crossing barrier
`3.5`. Comparing only initial and target energies therefore misses a necessary
intermediate cost. Even sufficient stored energy would not define continuation
through the singularity or establish a connecting solution of the current law.
The regular ordinary-antipodal example above has
`V_phi=6-2*sqrt(2)<7/2`; that lower cost does not contradict the distinct
zero-resultant obstruction.

### Added support gives a regular route to an admitted target

<a id="relational-regular-winding-crossing"></a>

The pure-cycle obstruction is not a universal restriction on the full law.
Prepare two unit cycles `(0,1,2,3,4)` and `(5,6,7,8,9)`, with bridges
`(0,5)` and `(1,6)`. The second bridge is an explicit support change relative
to section 16's one-bridge preparation; it is not generated by an operator
or hidden in a pressure coefficient. On both rings choose the phase path

\[
\theta_{0:4}(\delta)=\theta_{5:9}(\delta)
 =(0,-4\delta,-3\delta,-2\delta,-\delta),
\qquad 0\le\delta\le2\pi/5.
\]

Each bridge joins equal phases. Interior degree-two nodes have
`z_i=2*cos(delta)>0`; ports 0 and 5 have

\[
z_0=z_5=1+e^{-i\delta}+e^{-4i\delta},\qquad
\operatorname{Re}z_0=1+\cos\delta+\cos4\delta
\ge\cos\delta\ge\cos(2\pi/5)>0.
\]

Ports 1 and 6 have the conjugate resultant. This is a compact regular path
with a positive lower bound on every metric, from consensus to acute
winding-one geometry in both rings. The additional four-cycle has winding
zero wherever its ordinary winding is defined; it also encounters antipodal
edges at `delta=pi/4`. At `delta=pi/3`, the two ring-neighbor phasors cancel at each port,
but the aligned bridge neighbor contributes one. That is precisely a
resultant-zero encounter on the isolated C5 which the supplied extra support
prevents; no direction is assigned to an undefined resultant.

Each ring's oriented gaps are `wrap(-4*delta),delta,delta,delta,delta`.
Its winding changes from zero to one when delta crosses `pi/4`. At the
endpoint `delta=2*pi/5`, all ring gaps are acute and uniform, bridge gaps
are zero, and native phase pressure vanishes. Uniform form there supplies
an equilibrium covered by section 15's local-recovery theorem. This proves
regular geometric connectivity between the preparations. It does not yet
show that their connecting path obeys the joint equations.

The phase storage along this specified path is
`V_phi(delta)=2*[1-cos(4*delta)+4*(1-cos(delta))]`; bridge contributions
vanish. At `delta=pi/3`, all resultants equal one and `g=0`, but the
phase geometry is nonacute. Here `V_phi=7` and the second derivative along
the path is `-12`, while changing only an interior node has positive phase
curvature. It is a saddle of phase storage. The final acute target has
`V_phi=10*(1-cos(2*pi/5))<7`. This locates a barrier on this particular
regular path, not a proven minimum barrier over all paths or a connecting
orbit of the joint ODE.

### The same coupled law realizes a transversal crossing

A geometric path alone would be insufficient. Here the same coupled law can
realize a transversal crossing. At any regular phase state with positive
capacities, let `M=N*H^-1` and prescribe an instantaneous relative phase
velocity represented by v. Define

\[
\gamma=-\frac{\mathbf1^TM^{-1}v}{\mathbf1^TM^{-1}\mathbf1},\qquad
Bx=\frac\beta w M^{-1}(v+\gamma\mathbf1).
\]

The right side has zero sum. Connectedness of the graph supplies a finite
real x, unique modulo a common offset. The actual phase row then satisfies
`theta_dot=v+gamma*1`; the form row remains the unchanged native nodal law.
This realizes any instantaneous relative phase direction through supplied
form. It is not arbitrary control of a whole phase path, because subsequent
form and phase must both follow the joint equations.

The two-ring state gives a concrete witness without a fitted tangent. At
`delta=pi/4`, unit capacities, `beta=1`, `e=w=1/2`, take identical form
on both rings:

\[
x_{0:4}=x_{5:9}
=\frac{2\sqrt2}{5}(\pi+8,-\pi-8,-3\pi-4,0,3\pi+4).
\]

The full-graph `Bx` on each ring is
`(8*sqrt(2),-8*sqrt(2),-2*pi*sqrt(2),0,2*pi*sqrt(2))`.
Bridge differences vanish. Port metrics are `2*sqrt(2)` and interior
metrics are `pi*sqrt(2)`, so the **actual** phase row is

\[
\dot\theta_{0:4}=\dot\theta_{5:9}=(2,-2,-1,0,1).
\]

This is the path tangent `(0,-4,-3,-2,-1)` plus the permitted representative
offset `gamma=2`, obtained from the phase row, not externally imposed.
Smooth local ODE existence and uniqueness give a solution on both time
sides with `theta_1(t)-theta_0(t)=-pi-4*t+O(t^2)` and the same crossing
on the second ring. The remaining ring edges stay non-antipodal locally.
Thus an **actual local solution** changes ordinary winding from zero to
one in both rings under the same law. No phase forcing, support event,
telemetry selector or new evolution rule is inserted.

This supplied form carries finite Dirichlet storage
`128*pi/5+48*pi^2/5+512/5`; it is not free phase actuation or evidence
that a small perturbation of consensus forms the target. Neither arrival
from consensus at the crossing state nor later entry into the acute
winding-one equilibrium's recovery basin has been proved. The regular path,
local crossing solution and admitted stable target are distinct facts;
concatenating them verbally would not establish one connecting trajectory.

### Admission verdict and execution boundary

The same mathematics yields a negative and a positive result: pure-cycle
acute sectors require an undefined resultant between them, whereas the
specified higher-degree support admits a regular geometric route to a
recoverable target and a local topological crossing under the complete law.
Neither a universal formation obstruction nor a complete autonomous
formation trajectory follows. The full regular domain is meaningful and
can exceed the engine's default acute domain, but global continuation
within it remains an independent obligation.

The default acute execution contract and every earlier frozen response remain
unchanged. Section 18 supplies opt-in certified admission for positive
resultants and the full regular domain. A prescribed geometric route does not choose the
numerical trajectory or its outcome. No broader formation sweep, new state
primitive or unsupported singular continuation is justified by this local
existence result. The sole execution queue owns the next gate.

The [exact cycle observer](../../src/tnfr/physics/phase_resultant_sectors.py)
reuses the existing support geometry and rational scalar owner. It records
ordinary/derived winding and both resultant boundary types from declared exact
turns, leaving live radians and runtime admission separate. The
[admission controls](../../tests/physics/test_relational_formation_admission.py)
check those observations, the sharp C5 budget and the two-port phasor, tangent
and native-pressure identities. They contain no time integration or formation
parameter search. [API scope](../../docs/contracts/relational/OBSERVATION_AND_INFORMATION.md#exact-cycle-resultant-sectors)
owns inputs and availability; the symbolic proof above owns the conditional
continuous conclusions.

## 18. Certified phase-domain execution and a retained crossing control

<a id="relational-positive-resultant-execution"></a>

### A sufficient regular chamber for the unchanged joint law

The production model accepts explicit
`phase_domain="positive_resultant"`; `"acute"` remains the default. This
option requires `Re(z_i)>0` at every node. This right-half-plane chamber is
strictly narrower than the full regular domain in section 17, which also
admits left-half-plane resultants away from the negative-real branch. It is
nevertheless broad enough for the two-port witness and its prescribed
geometric route. Support, capacities, storage scale and phase/form equations
remain the same. This extends execution admission, not constitutive selection.

The [shared rational helper](../../src/tnfr/mathematics/_phase_resultant_chamber.py)
reuses the existing Machin enclosure of mathematical pi. For each exact
rational difference of materialized phase coordinates it encloses cosine:
integer true-turn reduction, a rational midpoint Taylor enclosure, and a
Lipschitz allowance for the reduction interval. Outward dyadic rounding
retains a rational lower bound. Near zero an elementary quadratic/quartic
enclosure suffices. Bounded numerical work can return an inconclusive bound;
neither that outcome nor a nonpositive lower bound proves an actual singularity.
The ordinary edge's antipodal branch requires no special cosine continuation.

For the nodewise lower bounds and actual represented endpoint increments,
write

\[
L_i\le\sum_{j\sim i}\cos(\theta_j-\theta_i),\qquad
d_i=\theta_i^{\rm after}-\theta_i^{\rm before}.
\]

Cosine is globally one-Lipschitz. Every point of the straight Euler proposal
chord therefore satisfies

\[
\operatorname{Re}z_i(s)\ge L_i-\sum_{j\sim i}|d_j-d_i|,
\qquad 0\le s\le1.
\]

Strict positivity of this exact rational right side certifies that the
**whole represented phase chord** remains in the chamber. On that chord,
`H_i>=pi*Re(z_i)>0`: for `|delta_i|<pi/2`,
`sin(delta_i)/delta_i>=cos(delta_i)`. The engine additionally evaluates
materialized positive cosine sums and metrics and re-admits the endpoint.
The exact margins are retained as `resultant_real_lower_bounds` and
`segment_resultant_real_lower_bounds`. Rejection precedes live graph commit.

Using only endpoint admission would be insufficient: two regular P2
endpoints can differ by a full relative turn whose interior crosses the
excluded chart. Conversely, a common phase increment contributes zero to
the exact drift bound. The increments used here are the actual rounded
endpoint differences, not an uncommitted `dt*phase_rate` approximation.
The [execution controls](../../tests/test_relational_regular_execution.py)
exercise these distinctions, atomic rejection, zero capacity, the analytic
crossing tangent, and admission independent of libm cosine signs.

This certificate does not enclose the exact continuous ODE trajectory, bound
its global numerical error, or prove Euler storage decrease. Native pressure
and joint rates still use their existing materialized arithmetic; pressure
split, work and step defects remain measured. Raw relative angles are used
for trigonometric evaluation in this chamber; the default acute path
retains its previous represented wrapping. Earlier frozen producers and
records are untouched. Their historical source fingerprints are not rewritten
to claim compatibility with this source revision.

### Certified admission on the full regular phase domain

<a id="relational-full-regular-execution"></a>

The separate `phase_domain="regular"` option admits the same joint
equations when each relative resultant avoids the entire nonpositive real
axis. A negative real component is allowed if the imaginary component is
nonzero. No extra phase coordinate, pressure channel, operator or material
property is introduced. The domain follows from the existing source and
metric; selecting the complete joint law still retains the independent
storage, capacity and support premises of sections 1-2.

Write `z_i=C_i+i*S_i` and `alpha_i=Arg(z_i)` in `(-pi,pi)`. The equivalent
metric expression

\[
H_i=\pi S_i/\alpha_i\quad(\alpha_i\ne0),\qquad
H_i=\pi C_i\quad(\alpha_i=0,\ C_i>0)
\]

is positive throughout this open domain: `S_i` and `alpha_i` have the same
sign. In particular, `C_i<0` does not make the metric negative. This is the
same `H_i=pi*|z_i|*sinc(alpha_i)` identity, with its regular positive-real
extension. Therefore `H*g=-grad(V_phi)`, unchanged native pressure, nodewise
phase/form exchange cancellation and total storage loss all retain their
proofs. Nothing here continues the law through the excluded zero or
negative-real resultant.

Numerical admission reuses the rational trigonometric enclosure owner. For
exact represented raw phase differences it produces a rectangle

\[
C_i\in[C_i^-,C_i^+],\qquad S_i\in[S_i^-,S_i^+].
\]

Its sufficient separation margin is

\[
m_i=\max(0,C_i^-,S_i^-,-S_i^+).
\]

Let `K={r+i*0:r<=0}` be the excluded closed ray. Each positive term in
this maximum is a lower bound for `dist(z_i,K)`, hence so is their maximum.
Strict `m_i>0` certifies a regular point even in the left half-plane. A
zero margin is inconclusive or outside the domain; finite enclosure failure
does not prove that the mathematical resultant actually lies on the ray.
The helper's bounded work is part of numerical admission, not an additional
constitutive boundary.

The field retains the component rectangle as `resultant_bounds` and its
margin as `resultant_regular_margin_lower_bounds`. Computed component signs
must agree with the separation certificate. A materialized principal argument
at either represented branch endpoint is rejected, even if the exact point
has a smaller unresolved separation. The regular numerical metric uses the
displayed `pi*S/Arg` identity and positive-axis limit. Native pressure consumes
the same captured phase source, avoiding a second branch selection; pressure
split, rate and work residuals are still reported from their actual arithmetic.

For actual represented endpoint increments `d_i`, the straight proposal is
`theta_i(s)=theta_i(0)+s*d_i`, `0<=s<=1`. The elementary chord inequality
`|exp(i*a)-exp(i*b)|<=|a-b|` yields

\[
|z_i(s)-z_i(0)|\le\sum_{j\sim i}|d_j-d_i|.
\]

Distance to a closed set is one-Lipschitz, so

\[
\operatorname{dist}(z_i(s),K)
\ge m_i-\sum_{j\sim i}|d_j-d_i|.
\]

Strict positivity at every node certifies the whole represented chord.
The step retains `segment_resultant_regular_margin_lower_bounds`.
It may reject a longer regular chord whose actual geometry is safer than
this conservative bound. Common phase increments cost zero margin. Checking
only the two endpoints would miss a branch crossing in the interior.
Materialized endpoint rates and metrics are additionally admitted before
atomic commit; their arithmetic residuals remain observations rather than
being replaced by their ideal zero values. This certificate neither encloses
the continuous solution nor guarantees future regularity or a storage
decrease during a finite Euler step.

### Regular-domain continuation and boundary access

<a id="regular-domain-continuation-and-boundary-access"></a>

All statements here retain the fixed connected simple unit support,
nonnegative held capacities, `e>=0,w,beta>0`, no input and no events. They
concern the ideal continuous law. Neither finite Euler steps nor the narrower
acute execution chamber inherit a new invariance guarantee.

**Derived resultant dynamics.** Put `v_i=theta_dot_i`,
`delta_ij=theta_j-theta_i`, and `z_i=C_i+i*S_i`. Direct differentiation gives

\[
\dot z_i=i\sum_{j\sim i}e^{i\delta_{ij}}(v_j-v_i),\qquad
\dot C_i=-\sum_j\sin\delta_{ij}(v_j-v_i),\quad
\dot S_i=\sum_j\cos\delta_{ij}(v_j-v_i).
\]

The actual law supplies `v_i=(w/beta)*nu_i*q_i/H_i`. These identities do not
close a law in resultants alone: form, capacities and neighboring rates remain
necessary. A useful cancellation is

\[
\frac{d}{dt}|z_i|^2
=2\sum_j(S_i\cos\delta_{ij}-C_i\sin\delta_{ij})v_j.
\]

The node's own phase rate rotates its relative resultant without directly
changing its magnitude. Its effect on magnitude returns through neighboring
rates. For nonzero `z_i`,
`Arg(z_i)'=(C_i*S_dot_i-S_i*C_dot_i)/|z_i|^2`. The kinematic derivative
of `z_i` can exist at zero even though this quotient and the full nodal law
are undefined.

For the excluded ray `K`, the same chord estimate used by the step admission
gives the continuous conditional bound

\[
\operatorname{dist}(z_i(t),K)\ge
\operatorname{dist}(z_i(0),K)
-\int_0^t\sum_j|v_j(s)-v_i(s)|\,ds.
\]

A whole-time independent rate bound can make this a continuation certificate.
An instantaneous rate is insufficient for that conclusion.

**A sufficient storage theorem.** Let `d_min` be minimum degree and
`v=E(0)/beta`. If

\[
\boxed{\mathcal E(0)<\beta\max(2,d_{\min}),}
\]

the ideal regular law continues for all finite times. To prove this, count
each undirected edge once in the nonnegative storage. At a node of degree
`d_i>=2`,

\[
C_i=d_i-\sum_{j\sim i}(1-\cos\delta_{ij})\ge d_i-v>0,\qquad
H_i\ge\pi C_i\ge\pi(d_i-v).
\]

If leaves exist, the hypothesis gives `v<2`. A leaf has `|z_i|=1` and its
principal edge gap obeys `cos(delta)>=1-v>-1`. Moreover
`sinc(delta)>=(1+cos(delta))/2`: for `0<|delta|<pi` this is
`tan(|delta|/2)>=|delta|/2`, with the continuous value at zero. Hence

\[
H_i\ge\frac{\pi}{2}(2-v)>0\quad\text{at a leaf}.
\]

All resultants are uniformly separated from the excluded ray. With leaves
a common conservative lower distance is `min(1,2-v)`: when a leaf has
negative real part its distance is `|S_i|`, and
`S_i^2=(1+C_i)(1-C_i)>=2-v`; otherwise its distance is one. Without leaves,
`d_min-v` is a common lower bound. Storage bounds centered form on connected
support. The source is bounded and the phase metrics stay positive, so form
and phase rates are bounded on every finite interval, including the common
form velocity. Smooth local existence therefore extends globally.

This proves continuation, not convergence, a physical identification or
stability of a numerical step. Zero held capacities are permitted; no inverse
capacity enters the bound. The universal `2*beta` boundary-storage scale
cannot be increased using only such graph-independent energy information:
with uniform form, P2 at antipodal phase has a nonzero branch resultant with
storage `2*beta`, and P3 at phases `(-pi/2,0,pi/2)` has a zero central
resultant at that storage.
Graph-specific sharper bounds remain possible.

**A nonzero branch can be reached in finite time.** On P2 take
`e=w=1/2`, `beta=nu_0=nu_1=1`, `x=(p,-p)`, `theta=(a,-a)` and initially
`p=3,a=pi/4`. For `pi/4<=a<pi/2`,

\[
H=\pi\frac{\sin(2a)}{2a},\qquad
\dot p=-p-a/\pi,\qquad \dot a=p/H.
\]

While `p>=1`, `0<H<=2` and
`dp/da=-H-sin(2a)/(2p)>=-5/2`. Thus
`p(a)>=3-5*pi/8>1`, closing that bootstrap up to `a=pi/2`.
The smooth scalar equation for `p(a)` extends to this endpoint and
`t(a)=integral(H/p) da` is strictly increasing in the interior with finite
terminal value `T<=pi/2`. Its inverse is an actual solution of the original
law until `T`. Both resultants approach `-1`, while `a_dot` diverges.
Positive dissipation and nonzero resultant magnitude do not prevent this
branch collision. Unwrapping the argument would change the prescribed
pressure and is not a harmless chart continuation.

The [two-ring zero-resultant construction](RELATIONAL_PATTERN_COMPOSITION.md#reflected-boundary-exit)
separately proves finite-time access to zero from acute states with `E<7*beta`
in the central reflected component. Its full-state vector field has no
continuous extension at the endpoint. Neither counterexample determines the
future of the frozen source/receiver preparation.

**Shared implementation.**
[`observe_relational_regularity`](../../src/tnfr/physics/relational_regularity.py)
captures one admitted field, encloses its initial storage with mathematical
trigonometry and applies the strict sufficient bound. It reports explicit
unavailability when that bound cannot certify continuation. Its resultant
derivative enclosure uses captured numerical rates as a supplied tangent;
it does not certify that tangent as the ideal ODE rate. The shared
[`relative_resultant_rate_bounds`](../../src/tnfr/mathematics/_phase_resultant_chamber.py)
owns that kinematic calculation, including common-rate cancellation.
No observer installs a boundary law or executes a trajectory.

### Which earlier conclusions retain their scope

The wider execution option removes a numerical restriction; it does not
erase theorem hypotheses or historical evidence:

| Existing result | Scope after full-regular admission |
| --- | --- |
| Metric, native-pressure alignment and ideal storage/work balances | Already valid throughout the regular domain, including left-half-plane resultants |
| Acute equilibrium classification, local recovery and explicit capture basins | Their acute or basin hypotheses remain required; they do not classify every nonacute regular equilibrium |
| Ordinary cycle winding preservation | Holds while relevant edge gaps stay nonantipodal; full resultant regularity alone does not guarantee this |
| Pure-cycle derived-winding obstruction | Still holds on every continuous path with nonzero resultants, even outside the positive-resultant chamber |
| [Uniform-receiver storage deficit](RELATIONAL_PATTERN_COMPOSITION.md#induced-formation-storage-obstruction) | Its initial positive-resultant restriction remains essential; the full-regular two-port counterexample already prevents promoting it to a universal obstruction |
| Retained crossing, formation and memory evidence | Keeps its declared preparation, phase domain, coefficients, clock, numerical budget and source identity; broader admission is not a new response evaluation |

The full-regular source/receiver preparation is consequently a justified
candidate for further analysis, not a demonstrated formation trajectory.
Enough initial storage is necessary for its proposed acute target, but is
not sufficient for reachability, regular continuation or capture.

### Fixed preparation and independently stated questions

<a id="relational-reserved-crossing-response"></a>

The [bounded producer](../../benchmarks/relational_formation_response.py)
uses the supplied two-ring, two-bridge support of section 17, unit capacities,
`beta=1`, and `e=w=1/2`. Each ring starts at
`theta=(0,-4*delta_0,-3*delta_0,-2*delta_0,-delta_0)`, with
`delta_0=pi/4-1/256`, and the explicit transverse form state derived there.
Thus both ordinary windings initially equal zero, but this is already a
structured, finite-storage preparation close to a crossing; it is not phase
consensus, a random perturbation or substrate birth.

Before evolution, the protocol fixes one horizon `T=16`, three Euler grids
`dt=(1/64,1/128,1/256)`, and common checkpoints. The short positive hypothesis
is winding `(1,1)` by `t=1/64`, retained at `t=1/32`. Later acute-domain
entrance, distance to the prepared acute winding-one equilibrium and a local
basin indicator are separate questions, without a predeclared positive
capture claim. The complete quotient state includes both rings together, so
relative regional offsets are not discarded.

For the basin indicator, the target edge margin is `m=pi/10`. Choose
`r=pi/(20*sqrt(2))`, giving `c_r=sin(pi/20)>1/10`. Every connected ten-node
unit graph has `lambda_2>=2/81>1/45`: summing the path bound
`(y_i-y_j)^2<=9*y^T*B*y` over all pairs gives this conservative estimate.
Section 15 then yields `k_r>1/900`, `r^2>9/800`, and
`k_r*r^2>1/80000`. In the declared beta-one quotient normalization,

\[
\|y\|^2<9/800,\qquad 0\le E_{\rm rel}<1/100000
\]

are sufficient **ideal-law** basin conditions. The retained finite-state
norm and excess storage use materialized phases and trigonometry relative to
the supplied target. Their threshold verdict is therefore a numerical
indicator, not a rigorous transcendental energy enclosure or an exact-ODE
capture certificate. Negative estimated excess is retained, not clipped.

Every grid stops at its first shared-owner rejection, retaining its last
admitted full state, attempted step and unchanged-graph check. It neither
invents a rejected proposal report nor retries with a smaller step or broader
chamber. Refinement compares only checkpoints reached by all grids. No
later form tuning, prescribed phase path or new pressure term is allowed.

### Retained response: crossing succeeded; final capture is unavailable

The [prediction](../../docs/assets/relational_formation_response/result.prediction.json)
was frozen before the single
[evaluation](../../docs/assets/relational_formation_response/result.json).
Its SHA-256 is
`4ed42b2a94138e5a5ed7318c39aa9fdb485234f38cbdcab97ab8a2da75959956`;
the response SHA-256 is
`651cf5bcdeec81d64c4289c56110d052ea23d48a0ecdd08af3c76e65ab258d25`.
The records retain the complete package/producer fingerprints, declared
binary64 runtime, exact arithmetic defects and all saved full states. No
random initialization, retry, altered pressure or subsequent recalibration
was used.

| Euler step | Completed steps | Last admitted time | First observed winding `(1,1)` | First observed acute edges | Last minimum real-resultant bound |
| --- | --- | --- | --- | --- | --- |
| `1/64` | 37 | `0.578125` | `0.015625` | `0.46875` | `0.00289508` |
| `1/128` | 73 | `0.5703125` | `0.0078125` | `0.46875` | `0.0278649` |
| `1/256` | 147 | `0.57421875` | `0.0078125` | `0.46875` | `0.0121522` |

Both rings start with winding zero, and **all three grids satisfy the frozen
winding-one prediction at both `1/64` and `1/32`**. The common checkpoint
state discrepancies decrease under refinement: at `t=1/4`, the combined
form/phase infinity differences are approximately `1.26349e-3` and
`6.27253e-4` for successive grids. This is finite refinement evidence, not a
validated bound against the exact ODE. The maximum measured work residual is
below `2.64e-14`; the declared `1e-10` numerical policy passes. Euler energy
step defects remain nonzero and recorded separately.

Each run subsequently enters the acute phase region but stops when the next
whole-chord positive-resultant bound cannot be certified. Every last state
is still regular, winding-one and acute. Rejection leaves its live graph
unchanged. The limiting rows are nodes 3 and 8, whose phase metrics approach
zero as their two neighbors approach antipodal separation. The endpoint
evidence and the symmetry calculation below locate the relevant boundary;
the stopped Euler runs do not prove that the exact ODE reaches it in finite
time or lacks every other continuation.

None reaches the declared horizon 16. Initial joint storage is about
`283.894`; last-state values are about `203.45`--`204.46`, while the target
phase storage is about `6.90983`. The final saved quotient squared norms are
about `276.25`--`277.64`, well outside the conservative `9/800` basin indicator.
Thus acute entrance alone is plainly insufficient for capture. Later common
checkpoints and final-horizon basin verdicts are **unavailable**, not failed
completed predictions or inferred equilibria.

The [retained-response controls](../../tests/physics/test_relational_formation_response.py)
reconstruct winding/availability/acute geometry from every saved phase state,
check the original preparation and time accounting, and replay detached
last-state observations. They do not rerun the trajectories. The result
establishes an executed, same-law change of geometric sector under supplied
initial structure. It does not establish autonomous formation and maintenance
of the target, spontaneous support creation, or physical particles.

### Reflection reduction explains the next mathematical obligation

<a id="relational-reflected-capture-boundary"></a>

The fixed preparation has more structure than two arbitrary five-node rings.
It is invariant under copying the rings and under simultaneous form-sign and
phase-reflection about a constant phase c. Each ring can be written

\[
x=(A,-A,-B,0,B),\qquad
\theta=c\mathbf1+(a,-a,-b,0,b).
\]

The graph automorphism exchanges ports 0/1, nodes 2/4 and the corresponding
second-ring nodes, while fixing nodes 3 and 8. Unit capacities and the
unforced joint law respect these symmetries on its regular domain. Uniqueness
therefore preserves this four-coordinate subspace of the **ideal** ODE; it
is not imposed on the numerical states to alter their evolution. In the
component containing the preparation, the central form and phase rows vanish
because `q_3=g_3=0`, so c remains constant.

Writing `q=3*A-B` and `r=2*B-A`, the nonredundant relative resultants are

\[
z_0=1+e^{-2ia}+e^{i(b-a)},\qquad
z_4=2\cos(a/2)e^{i(a/2-b)},\qquad z_3=2\cos b.
\]

The other port/interior rows are conjugates. For `|a|<pi`, `|b|<pi/2`
and the retained regular branch, define
`g_0=Arg(z_0)/pi`, `H_0=pi*|z_0|*sinc(Arg(z_0))`,
`g_4=(a/2-b)/pi`, `H_4=2*pi*cos(a/2)*sinc(a/2-b)`.
The unchanged complete joint equations reduce exactly to

\[
\begin{aligned}
\dot A&=-e\,q/3+w\,g_0,&
\dot B&=-e\,r/2+w\,g_4,\\
\dot a&=(w/\beta)\,q/H_0,&
\dot b&=(w/\beta)\,r/H_4.
\end{aligned}
\]

These are derived coordinates of the existing law, not a new pressure,
controller or independent reduced solver. In particular, positive r still
drives b upward after the edges become acute. The retained last states have
`r` near `7.5`, so their remaining form contrast has not disappeared merely
because their winding and acute status now match the target's labels.

The full two-ring storage in this subspace is

\[
\begin{aligned}
E_D&=6A^2-4AB+4B^2,\\
V(a,b)&=10-2\cos(2a)-4\cos(a-b)-4\cos b,\\
\dot E&=-e\left[\tfrac43(3A-B)^2+2(2B-A)^2\right]\le0,
\qquad E=E_D+\beta V.
\end{aligned}
\]

The invertible proof coordinates `A=(2*q+r)/5`, `B=(q+3*r)/5` give
`E_D=(4*q^2+4*q*r+6*r^2)/5` and diagonal loss
`-e*(4*q^2/3+2*r^2)`. They connect the existing first-exit proof to a
potential interval storage bound without projecting the runtime state.
The full central-resultant admission `2*cos(b)>0` remains necessary even
though its exact symmetric dynamical row vanishes.

At the observed central cancellation face,

\[
V(a,\pi/2)=7+4(\sin a-1/2)^2\ge7.
\]

This is the same `7/2` cycle barrier from section 17 applied to both reflected
rings, with zero bridge storage. An ideal trajectory with `E<7*beta` cannot
reach this face while the nonincrease law remains admitted. The acute target
`(a,b)=(4*pi/5,2*pi/5)` has `V=10*(1-cos(2*pi/5))<7`; the nonacute saddle
`(2*pi/3,pi/3)` has `V=7`. These facts connect storage loss to a potentially
protected region around the target. They do not prove that a winding-zero
preparation can enter that region, nor that `E<7*beta` alone identifies its
basin rather than another allowed region.

As `b` approaches `pi/2`, the full central resultant vanishes. The exactly
symmetric central row may look removable because its numerator is identically
zero, but arbitrary nearby full states do not share that cancellation. A
continuation of four formal coordinate equations through the face would not
by itself extend the full phase pressure and inverse-metric law. Beyond the
face, the central resultant is negative real: assigning it zero phase source
would change the native law, while either Arg-side limit breaks the imposed
central zero-form row. Conversely,
the finite guard stops before that face and is not a singular-hitting theorem.

Section 19 uses these reduced equations and the existing local recovery
theorem to prove protected capture and distinguish its possible limits. Entry
from winding zero still requires a crossing and loss budget. The supplied
high-storage response remains the first retained test; it is not retuned
after observing its stop.

Static [formation-admission controls](../../tests/physics/test_relational_formation_admission.py)
independently reconstruct the reduced Laplacian, resultants and storage from
the full support, check a nonacute regular field against the four equations,
and verify both signs of the sharp central-cancellation barrier. They add
neither a reduced execution path nor another temporal campaign.

## 19. Protected capture and conditional basin selection

<a id="relational-protected-capture"></a>

The reflection reduction supplies a global capture result within an explicit
region, without another integration. Retain the same two-ring, two-bridge
support, unit held capacities and fixed `e,w,beta>0`. Work in the exact ideal
reflection subspace of section 18. Arbitrary common form and phase offsets
may be added; both are constant here because the central rows vanish.

### A protected region around the winding-one target

Define the positive-target phase rectangle

\[
\mathcal R=(2\pi/3,\pi)\times(0,\pi/2).
\]

Every relative resultant has positive real part there. In particular,
`Re(z_0)=1+cos(2a)+cos(a-b)>cos(a)*(2*cos(a)+1)>0`;
`z_4=2*cos(a/2)*exp(i*(a/2-b))` has positive modulus and argument in
`(-pi/6,pi/2)`; and `z_3=2*cos(b)>0`. Thus the full joint law is smooth,
not merely its formal four-coordinate expression. The four boundary costs are

\[
\begin{aligned}
V(2\pi/3,b)&=11-4\cos(b-\pi/3)\ge7,\qquad V(\pi,b)=8,\\
V(a,0)&=9-4(\cos a+1/2)^2\ge8,\\
V(a,\pi/2)&=7+4(\sin a-1/2)^2\ge7.
\end{aligned}
\]

**Conditional capture theorem.** Any exact reflected initial state with
`(a,b) in R` and `E=E_D+beta*V<7*beta` has a global forward solution and
converges to `(a,b,A,B)=(4*pi/5,2*pi/5,0,0)`.

Indeed, `E_D=6*A^2-4*A*B+4*B^2` is positive definite, so a fixed sublevel
bounds form. Storage nonincrease and the boundary costs prevent a first exit
from the rectangle. The phase sublevel stays a positive distance from its
boundary; together these give a compact subset of the smooth full-law
domain. Continuation is therefore global. Zero loss forces `q=r=0`, hence
`A=B=0`. To remain in that set requires `g_0=g_4=0`. The latter gives
`b=a/2`, and the former reduces to

\[
\sin(2a)+\sin(a/2)=2\sin(5a/4)\cos(3a/4)=0.
\]

Its unique solution in this rectangle is `a=4*pi/5`. LaSalle's invariance
principle then gives convergence to that target; the local-recovery theorem
supplies an eventual exponential rate, without a graph-independent bound.
The limiting storage is `beta*10*(1-cos(2*pi/5))`, not zero.

Every phase point in `R` already has ordinary winding one on both rings:
the positively wrapped ring gaps are `2*pi-2*a`, `a-b`, `b`, `b`, `a-b`,
whose sum is `2*pi`. This theorem establishes capture of a supplied sector;
it does not establish its formation from winding zero. The strict storage
test concerns the ideal law and exact trigonometry, not a rounded energy
estimate or successful admission of an arbitrary Euler step.

Simultaneous sign reversal `(a,b,A,B) -> (-a,-b,-A,-B)` preserves the same
law and storage. Hence the mirror rectangle
`R_minus=(-pi,-2*pi/3) x (-pi/2,0)` has the identical `E<7*beta` capture
theorem, with target `(-4*pi/5,-2*pi/5,0,0)` and both windings equal to
minus one. These are distinct protected regions of the same completion,
not different operator policies or extra dynamical rules.

### The saddle has a target-directed unstable branch

<a id="relational-saddle-capture-route"></a>

At the nonacute regular equilibrium
`s=(2*pi/3,pi/3,0,0)`, both active metrics equal `pi`, storage equals
`7*beta`, and the phase-storage Hessian is `[[-2,-2],[-2,4]]`. Put

\[
K=\begin{pmatrix}3&-1\\-1&2\end{pmatrix},\quad
G=\begin{pmatrix}1/2&1/2\\1/2&-1\end{pmatrix},\quad
D=\operatorname{diag}(3,2),\quad \omega^2=\frac{w^2}{\beta\pi^2}.
\]

In phase/form coordinates `(u,p)=(delta(a,b),delta(A,B))`, the Jacobian is

\[
J_s=\begin{pmatrix}0&(w/(\beta\pi))K\\
(w/\pi)G&-eD^{-1}K\end{pmatrix}.
\]

Eliminating form gives the symmetric quadratic eigenvalue pencil
`P(lambda)=lambda^2*K^(-1)+e*lambda*D^(-1)-omega^2*G`.
At zero it has one negative and one positive eigenvalue; for real
`lambda>=0`, its derivative is positive definite and its large-lambda
limit is positive definite. Hence it has exactly one simple positive root.
Zero is excluded because `G` is invertible. For a nonreal root and nonzero
phase vector `u`, the imaginary part of `u^*P(lambda)u=0` gives

\[
\operatorname{Re}\lambda
=-\frac{e\,u^*D^{-1}u}{2u^*K^{-1}u}<0.
\]

Thus the equilibrium is hyperbolic with a one-dimensional unstable manifold
in this reflected subspace. Its unstable phase vector has nonzero first
component, since `P_22=3*lambda^2/5+e*lambda/2+omega^2>0`. Orient it by
`u_a>0`; its slope is

\[
\frac{u_b}{u_a}
=\frac{\omega^2/2-\lambda^2/5}
       {3\lambda^2/5+e\lambda/2+\omega^2}\in(0,1/2).
\]

For the lower bound, `P` is already positive definite when
`lambda^2=5*omega^2/2`, where its off-diagonal entry vanishes, so the
unstable root lies below that value. The displayed ratio then gives both
bounds. These are properties of the local linearization, not a simulated
connection or a finite-time error estimate.

Every nontrivial unstable-manifold orbit approaches the saddle as
`t -> -infinity`. Its strict energy deficit follows from the actual balance:

\[
E(t)=7\beta-e\int_{-\infty}^{t}
\left[\tfrac43q(s)^2+2r(s)^2\right]ds<7\beta.
\]

The integral converges by the local exponential unstable asymptotics and is
positive because the unstable form component is nonzero. The branch oriented
by increasing `a` enters `R`. The capture theorem therefore continues that
branch globally forward to the acute winding-one target: a conditional
same-law heteroclinic orbit from the supplied nonacute saddle.

### The opposite branch approaches consensus

The other local branch enters
`S=(-2*pi/3,2*pi/3) x (-pi/2,pi/2)` with the same strict storage deficit.
Here the boundary values are
`V(±2*pi/3,b)=11-4*cos(b∓pi/3)>=7` and
`V(a,±pi/2)=7+4*(sin(a)∓1/2)^2>=7`. Within a sublevel `V<7`,

\[
V\ge8-4\operatorname{Re}z_0,\qquad
V\ge8-4\operatorname{Re}z_4,
\]

so both real parts exceed `1/4`; `z_3=2*cos(b)>0` as well. Consequently this
sublevel is regular in the same positive-resultant chamber, despite `S`
itself not supplying that assertion at arbitrary energy. Compactness and
LaSalle apply again. Now `g_4=0` gives `b=a/2`, and the displayed critical
equation has only `a=0` inside `S`. Thus every exact reflected state in `S`
with `E<7*beta` converges to consensus, including the opposite unstable branch.
It starts with winding one near the saddle and eventually has winding zero
near consensus. This is a forward loss of winding, not a reversal of the
target-directed orbit. In particular it reaches `a<pi/2` forward in time.
The mirrored saddle has the corresponding two branches toward the negative
target and consensus. This gives three sufficient capture regions; it is not
a classification of every initial state or a proof that no other behavior
exists outside them.

The exact saddle itself never departs. These are two distinct trajectories
with the same backward limit, not pieces that can be concatenated into a
winding-zero-to-target solution of the autonomous law. Capture and consensus
selection here remain conditional on supplied support, constitutive premises
and the prepared state; no spontaneous substrate or physical identity follows.

The exact reflection restriction is sufficient, not an assertion that every
slightly asymmetric state fails. From each state in any of these three
protected regions, a finite compact regular segment reaches the corresponding
full-state local-recovery neighborhood. Continuous dependence therefore gives
an open neighborhood of that state in the full form/phase state, with support,
capacity and model held, whose trajectories also converge modulo common
offsets. This is an existence statement: no radius has been computed, and no
rounded preparation, retained earlier trajectory or numerical capture
indicator is thereby certified.

### Shared admission of the sufficient ideal-law basins

[`certify_relational_capture`](../../src/tnfr/physics/relational_capture.py)
and `Network.relational_capture(model, cycles=...)` reuse one fresh detached
field and the shared rational cosine/pi enclosures. They check the exact
support, copied/reflected represented state, unit capacities, positive
coefficients, rectangle and strict energy bound, without projecting the
state or integrating a second solver. The output retains all symmetry
defects, candidate rectangle margins, exact energy bounds and unavailable
reasons. A target sector is supplied only on complete admission, independently
of current winding. The certificate concerns the ideal continuous law from
that represented state; Euler trajectories retain their separate obligations.
The [API contract](../../docs/contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#conditional-relational-capture)
owns implementation details, and [SDK usage](../../docs/guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#check-a-protected-relational-basin)
supplies a nonacute zero-form preparation.

Static [production controls](../../tests/test_relational_capture.py) exercise
both twist sectors, consensus from a winding-one state, strict energy equality,
exact symmetry, common offsets, unsupported domains and exact report export.
The [symbolic controls](../../tests/physics/test_relational_formation_admission.py)
derive boundary costs, resultant lower bounds, critical geometry and the
saddle pencil from the full support. No new temporal campaign is used to
establish these conclusions, and no old frozen outcome is rewritten.

The same owner also exposes `certify_relational_local_capture` and
`Network.relational_local_capture`. These apply section 15's full-state
local theorem on this support, with `beta=nu=1`, to a declared target sector.
They use all centered form/phase coordinates, exact affine-pi reference
errors and outward cosine energy bounds. The conservative conditions are
`norm_squared<9/800` and `excess_storage<1/100000` from section 17; exact
reflection is unnecessary. Their proof is independent of any observed
trajectory. If applied to a numerical endpoint, the certificate concerns
the ideal continuation restarted at that precise represented state. It does
not supply an error tube from the original initial-value problem.

### A distinct winding-zero preparation with a bounded crossing theorem

<a id="relational-upper-corner-preparation"></a>

The previous high-storage preparation is not the only way to drive the same
law. Retain unit capacities, `e=w=1/2`, `beta=1`, both supplied rings and both
bridges, and the exact reflection coordinates. A separate preparation is

\[
a_0=b_0=\pi/2-\varepsilon,\qquad
A_0=2/5,\quad B_0=1/5,\qquad \varepsilon=1/64.
\]

It has `q_0=1`, `r_0=0` and winding zero on each ring. All initial relative
resultants have positive real part:

\[
z_0=2-\cos(2\varepsilon)-i\sin(2\varepsilon),\quad
z_4=1+\sin\varepsilon-i\cos\varepsilon,\quad z_3=2\sin\varepsilon>0.
\]

The form storage is exactly `4/5`; the full storage is

\[
E_0=44/5-4\sin\varepsilon-4\sin^2\varepsilon,
\qquad 44/5-65/1024<E_0<44/5.
\]

This state lies above the protected `7*beta` threshold. It is not admitted
by the capture theorem simply because its storage is smaller than that of
the earlier frozen response.

Put `chi=atan(sin(2*epsilon)/(2-cos(2*epsilon)))`. Then
`0<chi<2*epsilon`, `g_0=-chi/pi`, `g_4=-1/4+epsilon/(2*pi)`,
`H_0=pi*sin(2*epsilon)/chi` and `H_4=2*pi*cos(epsilon)/a_0`.
The initial derivatives obey

\[
\begin{aligned}
\dot q_0&=-3/8-(6\chi+\varepsilon)/(4\pi)<0,\\
\dot r_0&=-1/12+(\varepsilon+\chi)/(2\pi)<-29/384<0,\\
\dot a_0&=1/(2H_0)>0,\qquad
\dot b_0=0,\qquad \ddot b_0=\dot r_0/(2H_4)<0.
\end{aligned}
\]

Thus the form imbalance initially moves a toward the winding boundary while
moving b away from the central-resultant cancellation face. As
`epsilon -> 0+`, the derivatives tend to
`qdot=-3/8`, `rdot=-1/12`, `adot=1/(2*pi)` and `bddot=-1/96`.
The limiting corner itself is singular and is not an admissible initial
state; the selected positive epsilon remains part of the preparation.

**Bounded crossing, not capture.** For this fixed ideal preparation, the
solution remains regular through `t=1/4` and both windings change from zero
to one by `t=7/48`. This follows without numerical integration. Use the box

\[
\begin{gathered}
\pi/2-1/64\le a\le\pi/2+1/16,\qquad
\pi/2-1/32\le b\le\pi/2-1/64,\\
3/4\le q\le1,\qquad -1/8\le r\le0.
\end{gathered}
\]

Here `0<=a-b<=3/32`, `abs(a+b-pi)<=3/64`, and
`Re(z_0),Re(z_4)>2039/2048` (the non-strict lower bound also suffices).
The elementary bounds `sin(t)<=t`, `cos(t)>=1-t^2/2` and `3<pi<22/7`
give

\[
|g_0|<1/60,\quad -1/4<g_4<-11/48,\quad
5/2<H_0<7/2,\quad H_4>5/2.
\]

The reduced equations therefore imply `-1/2<qdot<0`, `rdot>-1/6`, and,
on the face `r=0`, `rdot<-13/240`. Consequently, before any first exit,

\[
\begin{aligned}
q(t)&\ge1-t/2,& r(t)&\ge-t/6,\\
3/28&<\dot a(t)<1/5,&
b_0-t^2/60&\le b(t)\le b_0.
\end{aligned}
\]

For `0<=t<=1/4` these inequalities prevent exit through every face;
`r<0` for positive time prevents exit through the upper b face. The central
resultant stays at least `2*sin(1/64)>0`. Smooth continuation on this compact
box establishes the interval, and the lower bound on `adot` gives
`a(7/48)>pi/2`. Since a increases, while b remains positive and
`0<=a-b<pi`, the ordinary ring winding crosses once at the antipodal
edge `0--1` (and its copied edge). The full resultants remain regular at
that crossing. The winding itself is undefined at the crossing instant.

The signs and corridor motivate a separate finite prediction through the
shared executor. They are an ideal-law theorem, not an Euler-error bound or
a result for the binary representation of the supplied phases. In particular,
this argument does not show entry into `R`, enough subsequent storage loss
for `E<7`, or convergence to the winding-one target. The
[static controls](../../tests/physics/test_relational_formation_admission.py)
check the initial identities and the rational corridor estimates without
executing a trajectory.

### Frozen upper-corner response: regular sector acquisition and a remaining certificate gap

<a id="relational-upper-corner-response"></a>

[`relational_capture_response.py`](../../benchmarks/relational_capture_response.py)
froze one materialized preparation, `e=w=1/2`, `beta=nu=1`, no Gamma,
the supplied ten-node/twelve-edge support, horizon 64 and three Euler steps
`1/64,1/128,1/256` before evaluation. Form and phase offsets were zero.
The binary values corresponding to `(2/5,1/5)` have exact `r=0` but
`q=1+2^-54`; the preceding ideal-input proof is not silently identified
with those represented values. Python 3.13 and dependency/platform versions,
all 599 package/producer source fingerprints, exact inputs and numerical
settings are retained in the prediction.

The protocol stopped each grid at its first admitted target certificate,
shared-owner rejection, or horizon. No projection, retry, phase forcing,
parameter sweep or horizon extension was performed. All three grids reached
64 without rejection; the target certificate was unavailable on every grid.
The compound `finite_prediction_passed` is therefore **false**, while the
numerical work-residual policy passed.

| Euler step | Executed steps | Final storage upper bound, rounded | Final squared quotient norm, rounded | Final target excess upper bound, rounded |
| --- | --- | --- | --- | --- |
| `1/64` | 4096 | 6.919042007 | 0.03467713 | 0.009211950 |
| `1/128` | 8192 | 6.919040986 | 0.03467310 | 0.009210930 |
| `1/256` | 16384 | 6.919040485 | 0.03467112 | 0.009210429 |

Every grid starts with windings `(0,0)` and has `(1,1)` at all retained
checkpoints from `t=1/4` through 64. This is sampled numerical evidence,
separate from the ideal short-crossing theorem. At the retained times 32
and 64, all first-ring `(a,b)` coordinates lie in the positive-twist
rectangle and the full-state certified storage is below 7. For the finest
grid at 32, `(a,b)` is approximately `(2.267268628,1.109711608)` and
the storage upper bound is approximately `6.970551621`.

However, the represented states have small **nonzero** reflection defects.
At 64 the largest form/phase defects across the grids are below `2e-18`,
while ring-copy defects are zero. The reflected theorem rejects those states
solely on its exact-symmetry premise. The full-state local certificate also
rejects them: both distance and excess storage remain above its conservative
thresholds. Neither certificate can be promoted by rounding those defects
away or by treating energy below 7 as a support-independent basin test.

The maximum native work residual across the three traces is below
`2.176e-16`; clock defects are zero. Maximum absolute energy-step defects
decrease from about `2.503e-5` to `6.257e-6` to `1.564e-6`.
All retained states and represented Euler chords satisfy the positive-
resultant guard. At 64 the coarse/middle and middle/fine state infinity
differences are about `4.391e-6` and `2.154e-6`; differences decrease under
refinement at every common checkpoint. These are numerical comparisons,
not rigorous global trajectory-error bounds or asymptotic convergence.

| Immutable record | SHA-256 |
| --- | --- |
| [Prediction](../../docs/assets/relational_capture_response/result.prediction.json) | `4045ab698013f2bb071c5083a99e8aea760c7ce42e568e1be5a230af9629db7c` |
| [Response](../../docs/assets/relational_capture_response/result.json) | `7085c782f0b85c465c530d2e84ff2990e71dfcc89d950f1f993025a952310ed9` |

The [retained-response controls](../../tests/physics/test_relational_capture_response.py)
recompute certificate decisions from saved states without re-executing the
campaign. Earlier frozen responses remain unchanged. This result supports
a regular same-law route toward the maintained geometry on supplied support;
it does not yet certify the original continuous trajectory's entry into the
protected basin, autonomous support creation, or a physical NFR identity.

## 20. A full-state acute-sector barrier without reflection

<a id="relational-acute-sector-capture"></a>

The retained response exposes a gap between the dynamics and the available
sufficient certificates: its tiny symmetry defects invalidate the reflected
theorem, while its distance from the target exceeds the deliberately small
local bound. Neither restriction is intrinsic to every capture proof. The
same support admits a wider full-state theorem based on its cycle periods
and cosine storage, with no symmetry projection or changed evolution law.

Retain the two supplied C5 rings and the two bridges between matching
positions 0 and 1. Let `e,w,beta>0` and all held capacities be strictly
positive. Suppose every true wrapped support-edge phase difference is
strictly acute and both supplied oriented rings have winding `s`, where
`s=+1` or `s=-1`. Define

\[
V_* =5\bigl(1-\cos(2\pi/5)\bigr),\qquad
V_{\rm face}=5-4\cos(3\pi/8),\qquad
\mathcal B=V_*+V_{\rm face}.
\]

**Conditional full-state capture.** If the actual full-state storage satisfies

\[
E_D+\beta V<\beta\mathcal B,
\]

the ideal trajectory remains in its acute component for all forward time
and converges to uniform form and the aligned winding-s twist, modulo common
offsets. Exact ring copying, form/phase reflection and a prescribed small
Euclidean radius are not premises. The theorem concerns circular phase
geometry: the final raw phase lift can additionally contain fixed nodewise
integer multiples of `2*pi`, determined by its initial acute component.

### The cycle geometry supplies the barrier

For `s=+1`, the five oriented acute differences of either ring sum to
`2*pi`. Convexity of `1-cos(delta)` on `[-pi/2,pi/2]` and Jensen's
inequality give ring storage at least `V_*`. At a first acute-boundary
encounter on that ring, a difference of `-pi/2` is impossible: the other
four would need sum `5*pi/2`, exceeding their maximum `2*pi`. Thus a
boundary edge has difference `pi/2`, and the other four have sum `3*pi/2`.
Another application of Jensen gives

\[
V_{\rm ring}\ge1+4\bigl(1-\cos(3\pi/8)\bigr)=V_{\rm face}.
\]

The other ring still costs at least `V_*`; bridge costs are nonnegative.
Consequently every first boundary through a ring edge has total phase
storage at least `mathcal B`. At a bridge boundary, its own cost is 1,
so the total is at least `1+2*V_*`, which is larger: since
`cos(3*pi/8)>cos(2*pi/5)`,

\[
1+V_*-V_{\rm face}
=1-5\cos(2\pi/5)+4\cos(3\pi/8)
>1-\cos(2\pi/5)>0.
\]

Sign reversal proves the same statements for `s=-1`. The one-ring Jensen
bound is sharp, but `mathcal B` is only a sufficient whole-support boundary
bound; simultaneous equality need not respect both bridges. No claim of an
optimal basin or a complete classification of the regular domain is made.

The barrier has an exact radical expression and a simple rational lower bound:

\[
\mathcal B=\frac{45-5\sqrt5}{4}-2\sqrt{2-\sqrt2}
>\frac{6924179}{1000000},\qquad
\mathcal B\simeq6.924181298665.
\]

For example, the certified inequalities
`cos(2*pi/5)<309017/1000000` and
`cos(3*pi/8)<382684/1000000` imply the displayed lower bound.
They follow by squaring positive rational bounds in the exact radical
formulas, independently of binary trigonometric evaluation. An executable
certificate can instead reuse the shared rational pi/cosine enclosures.

### Compactness, the target and the zero-loss set

The two rings and the four-cycle through both bridges form a cycle basis.
Four strictly acute differences cannot sum to a nonzero integer multiple
of `2*pi`, so the four-cycle has period zero. The three periods therefore
agree with the aligned twist: both rings have phase step `s*2*pi/5` and
both bridges have zero difference.

Choose compatible node lifts of that reference and the initial phases.
Equality of all cycle periods makes their edge-difference discrepancy an
exact graph gradient. The common acute component is therefore a convex
polytope in lifted phase coordinates modulo their common offset, with each
edge difference restricted to its selected interval of length `pi`.
The aligned twist belongs to this component. The phase-storage Hessian is
the graph Laplacian with edge weights `cos(delta)>0`, hence is positive
definite on the common-phase quotient. Its sole critical geometry in the
component is consequently the aligned twist.

Storage nonincrease and the strict boundary gap prevent a first exit.
Connected support bounds phase differences modulo the common offset;
Dirichlet storage bounds form modulo its common offset. Thus a fixed joint
sublevel below `beta*mathcal B` is compact in this quotient and stays a
positive distance from the acute boundary. All H entries stay strictly
positive. Quotient continuation is global, and bounded common-offset rates
also exclude finite-time failure of the full state.

For the invariant zero-loss set, positive capacity and `e>0` force
`Bx=0`. Uniform form alone does not yet imply zero common-form drift.
To preserve that condition requires `N*g=c*1` for a scalar c. But the
undirected phase potential satisfies

\[
\sum_i H_i g_i=0,\qquad
c\sum_i H_i/\nu_i=0.
\]

Every `H_i/nu_i` is positive, so `c=0` and `g=0`. LaSalle's invariance
principle on the compact quotient now gives convergence to the unique twist
geometry and uniform form. The existing local theorem supplies eventual
exponential recovery and finite limiting common offsets. Those offsets need
not equal their initial means in an asymmetric state.

### Capture shared by a declared strictly dissipative law class

<a id="relational-sector-law-class"></a>

The sector barrier also separates the maintained geometry from a uniquely
selected transient law. Keep the supplied support, fixed positive capacities,
positive `e,w,beta`, native form row and chosen storage:

\[
\dot x=N(-eD^{-1}Bx+w g),\qquad
E=\tfrac12x^TBx+\beta V,\qquad
L=e\,q^TND^{-1}q,\quad q=Bx.
\]

Consider a phase law `theta_dot=F(x,theta)` with the following **additional
law-level premises**:

1. The law is autonomous and `C1`, well-defined on circular phase states, and
   invariant under both common form shifts and common phase rotations. Its
   domain contains an open regular neighborhood of the entire protected
   storage sublevel, including all sufficiently small perturbations of the
   target. The fixed support, capacities and constitutive coefficients do not
   change during the flow; there is no input or event.
2. It has rest at every state with `q=g=0`: `F=0` there, including its common
   phase component.
3. For some fixed `c>0`, its actual full vector field satisfies the pointwise
   inequality

   \[
   \dot E\le-cL
   \]

   throughout that neighborhood. A sampled decrease, a condition only along
   one trajectory, or merely `E_dot<=0` does not supply this premise.

Under these premises, every preparation admitted by the strict acute-sector
barrier above remains in that sector and converges to the same aligned twist
and uniform form. Convergence is eventually exponential on the joint quotient,
and the full trajectory has finite limiting common form and phase offsets.
This is a conditional theorem for the declared class; the storage and
dissipation assumptions are not derived from the nodal identity.

**Retention and continuation use the unchanged geometry.** Nonincrease of the
same storage gives exactly the previous first-exit contradiction and compact
quotient sublevel. The geometric acute-edge, resultant and metric lower bounds
in [the quantitative consolidation](#relational-sector-consolidation) therefore
apply with the same initial storage deficit. They do not depend on a shared
transient trajectory. The `C1` field is bounded on the compact quotient;
shift invariance bounds the common-offset velocities there as well. These
facts give global continuation without requiring bounded absolute offsets
before convergence has been proved.

**The native form row fixes the invariant zero-loss set.** Since `c,e,nu_i`
are positive, zero loss requires `q=0`. A trajectory remaining in that set
must satisfy `BNg=0`, hence `Ng=a*1`. The identity
`sum(H_i*g_i)=0` and positivity of `H_i/nu_i` give `a=0` and `g=0`, exactly
as above. The phase rest premise makes this set invariant. The phase potential
has only the aligned-twist critical geometry in the acute component, so
LaSalle's argument gives quotient convergence to that geometry and uniform
form for every law in the class.

**Local exponential recovery also follows from the declared conditions.**
An extra assumption about a named law's Jacobian is unnecessary here. At the
target, write `K=Hess(V)` and let the columns of `R` be an orthonormal basis
of the complement of the constant vector. Define

\[
B_q=R^TBR,\qquad K_q=R^TKR,\qquad
M_D=R^TND^{-1}R,\qquad M_H=R^TNH^{-1}R.
\]

All four matrices are symmetric positive definite: the support is connected,
the target is acute, and `N,D,H` are positive diagonal matrices. In quotient
coordinates `z=(u,v)`, the Hessian of storage is

\[
P=\operatorname{diag}(B_q,\beta K_q)>0.
\]

The identity `grad(V)=-H*g` gives `Dg=-H^{-1}K` at `g=0`. Consequently the
first block row of the quotient Jacobian `A` is fixed, independently of the
choice of `F`:

\[
(Az)_x=-eM_DB_qu-wM_HK_qv.
\]

Here `BR=RB_q` and `KR=RK_q` justify the projected products. The product
`M_H*K_q` is invertible; it need not be symmetric, and the proof does not
treat it as a symmetric positive definite matrix.

Expanding the pointwise loss inequality to quadratic order at rest gives

\[
\frac{PA+A^TP}{2}
\preceq-ce
\begin{pmatrix}
B_qM_DB_q&0\\
0&0
\end{pmatrix}.
\]

This expansion requires the inequality on a full neighborhood, not only on
the selected orbit. For a nonzero complex eigenvector `z=(u,v)` of `A`,

\[
\operatorname{Re}(\lambda)\,z^*Pz
\le-ce\,u^*B_qM_DB_qu.
\]

Thus no eigenvalue has positive real part. Equality of its real part to zero
would force `u=0`; the first block row would then force
`w*M_H*K_q*v=0`, hence `v=0`, a contradiction. The quotient Jacobian is
therefore Hurwitz for every admitted law. The `C1` nonlinear stability
theorem gives local exponential recovery, and the previously proved quotient
convergence supplies eventual entry into that neighborhood. Shift invariance
and rest make the common-offset velocities vanish at the target and bound
them by a constant times the quotient deviation nearby. Their exponential
decay is integrable, giving finite limiting common offsets. These limits
need not be the preparation's means.

**The named comparison laws satisfy the conditions separately.** The reference
law has `E_dot=-L`, so it admits `c=1`. The
[phase-damping completion](RELATIONAL_EXCHANGE_ADMISSION.md#relational-passive-loss-completion)
`F_rho=F_rel+rho*N*g`, with fixed `rho>=0`, satisfies

\[
\dot E=-L-\beta\rho\sum_i\nu_iH_i g_i^2\le-L,
\]

and also admits `c=1`. The
[nonlinear signed-form completion](RELATIONAL_EXCHANGE_ADMISSION.md#relational-nonlinear-passive-completion),
with fixed `0<eta<2`, satisfies

\[
\dot E=-L+\mathcal R\le-(1-\eta/2)L,
\]

so it admits `c=1-eta/2`. Each law is smooth on the regular neighborhood,
shift invariant and at rest when `q=g=0`. Their previous independent
calculations establish these premises; sector geometry alone does not.
The nonlinear completion retains the reference Jacobian, whereas positive
phase damping changes it. Both nevertheless share this capture conclusion.

**Geometry evidence is not law admission.** A strict snapshot certificate
can establish the support, winding, acute-margin and storage premises. It
cannot establish an arbitrary callback's pointwise dissipation, regularity,
shift invariance or rest. Those require a separately identified law and
proof. The reference certificate keeps its declared reference-model meaning;
reuse of its geometry does not silently install an alternative law.

The theorem supplies neither a rate uniform over all admitted laws nor the
reference law's previously calculated rates, validated transient tubes,
formation-entry times or frozen responses for a changed law. In particular,
it does not establish that an alternative law carries the original
winding-zero preparation into this sector. That requires a separate transit
argument. Zero capacities, `w=0`, a nonacute target, nonsmooth laws, inputs
and support events also fall outside this proof. The conclusion is robust
maintenance on supplied support under stated constitutive premises, not
autonomous support birth or physical identification of the pattern.

### Evidence and execution boundary

The hypotheses can be checked from a saved full-state snapshot using strict
pi-enclosed edge gaps, exact integer cycle periods and certified cosine
storage bounds. Numerical winding labels, rounded symmetry or an estimated
distance to the target cannot replace those checks. The shared sector
certificate admits the theorem's arbitrary held positive capacities and
positive beta; the separate reflected and local owners retain their own
more restrictive premises.

This theorem was derived **after** the reserved response in section 19.
Applying it to a retained endpoint is a separately identified mathematical
reanalysis. It cannot turn the failed original frozen prediction into a
successful preregistered capture test, change its earlier certificate
decisions, or justify rerunning it with another gate. An admitted endpoint
certifies the ideal law restarted at that precise represented state. It
still does not bound the discrepancy between the numerical trace and the
original continuous trajectory from its winding-zero preparation. No
longer trajectory, support event, additional pressure, controller or
physical-identity claim is introduced by this argument.

The [static controls](../../tests/physics/test_relational_formation_admission.py)
check the support's cycle basis, target periods, one-ring face costs and the
strict rational barrier bound without executing any evolution.

### Shared certificate and retained endpoint reanalysis

`certify_relational_sector_capture` and `Network.relational_sector_capture`
in the [shared capture owner](../../src/tnfr/physics/relational_capture.py)
implement this full-state theorem. The first retained endpoint audit used
unit capacity and unit beta; extending the shared admission to the theorem's
positive held capacities and beta does not change that record. The existing
exact topology, phase-storage and pi/cosine owners are reused. Each candidate wrapped edge
gap is represented as a rational plus an integer multiple of mathematical
pi; two strict enclosed inequalities must certify its acute interval.
Summing the admitted integer turns yields both ring periods and the bridge
square's period. Floating winding telemetry remains separate. The full
storage upper bound must lie below a certified lower bound on the barrier.
The [API contract](../../docs/contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#conditional-relational-capture)
owns report fields, unsupported domains and restart-only scope.

The separate [read-only auditor](../../benchmarks/relational_capture_audit.py)
binds the immutable section-19 records and reconstructs their three final
states without modifying any coordinate. All three states pass the new
theorem. Their lower bounds on the energy margin `mathcal B-E` are about
`0.00513929210`, `0.00514031242`, `0.00514081319` in coarse-to-fine order;
every strict acute margin exceeds `0.16215` radians. Both exact ring periods
are one and the bridge-square period is zero. Consequently the ideal law
restarted at each actual represented endpoint remains regular and converges
to the aligned twist without a reflection premise or further forcing.

| Separate immutable evidence | SHA-256 |
| --- | --- |
| [Post-evaluation endpoint audit](../../docs/assets/relational_capture_response/endpoint-capture.audit.json) | `8d5dee62f7514118d69a19173599a6a536f1c516d2ee486c98abf3ea4e22352a` |
| [All 599 original fingerprinted source files](../../docs/assets/relational_capture_response/source-at-evaluation.zip) | `eb0600da8e479e8b7f1bfede31dbf802ef420ed8419fad982ca49e3e7ba8d049` |
| [Four-file source overlay for the endpoint audit](../../docs/assets/relational_capture_response/source-at-endpoint-audit.delta.zip) | `a0c9075f1514ab0b8fdac5d54a1c7e8c79e7c535d48348b9f5c1ce72730d23b2` |

The source archive was verified against every original prediction digest
before post-evaluation integration changed the engine/SDK source. It does
not bundle dependencies; their versions remain in the original prediction.
The new audit retains its own source/runtime manifest and explicitly records
both `all_endpoints_admitted=true` and the unchanged original
`finite_prediction_passed=false`. Tests check archive byte identity, detached
report data and the old/new decisions without re-executing any trajectory.
The small overlay plus the base source archive reproduces every source hash
in the endpoint audit; it preserves the original unit-restricted owner before
the generalization below without duplicating the complete package archive.

The endpoint audit alone leaves an original-IVP obligation: a validated
finite-time transit enclosure, now supplied [below](RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit).
The exact represented preparation is initially
reflected, so its ideal trajectory stays in the existing four-coordinate
slice even though finite Euler arithmetic breaks that symmetry. Enclosing
that same flow to an admitted reflected or acute-sector basin would join
formation and maintenance for the original continuous state. It requires
no new constitutive law, tuned preparation or longer finite observation.

### Consolidation: capacity, storage scale and a quantitative regularity margin

<a id="relational-sector-consolidation"></a>

The sector theorem already uses the same independent-capacity law as the
heterogeneous-capacity balance and local recovery theorems. No equality of
capacities appears in its barrier or zero-loss argument. All capacities must
remain strictly positive and held; `e,w,beta>0`, the supplied support, the
strict acute component and its exact cycle periods remain premises.
Multiplying every capacity by the same `a>0` multiplies **both** differential
rows by a, so it changes only their common clock. Heterogeneous positive
capacities can change transients and limiting offsets while preserving this
sufficient geometric basin. None of these statements admits capacity events
or extends the separate reflected/local production certificates.

Dividing storage by beta gives `E_D/beta+V_phi`. Thus, on exact real form
charts, replacing beta by `beta_new` and multiplying all form contrasts by
`sqrt(beta_new/beta)` preserves the sector energy test. It need not preserve
the trajectory under a time change: with `y=x/sqrt(beta)`, form damping has
coefficient e while the two exchange terms have coefficient `w/sqrt(beta)`.
Beta therefore controls relative storage and exchange as well as admission;
it is not generally a pure clock scale.

An admitted energy deficit supplies an explicit lower bound on future
regularity. It is a derived quantity, not a new primitive or policy threshold.
Put

\[
\eta=\mathcal B-\mathcal E(0)/\beta>0,\qquad
\kappa=2\pi/5,\qquad
F(t)=1-\cos t+4\left[1-\cos\left((2\pi-t)/4\right)\right],
\]

and `d=F(pi/2)-F(kappa)=mathcal B-2*V_*`. Every oriented signed gap
`y=s*delta` on either acute ring is positive: the four other gaps are each
strictly below `pi/2` and all five sum to `2*pi`. Jensen's inequality on
those other four gaps gives ring storage at least `F(y)`. On
`kappa<=t<=pi/2`,

\[
0\le F'(t)=\sin t-\sin((2\pi-t)/4)
\le M=1-\sin(3\pi/8)<1/13.
\]

The final strict rational inequality has an exact static check:
`sin(3*pi/8)=sqrt(2+sqrt(2))/2>12/13`, because
`238^2=56644<57122=2*169^2`.
The factor 13 is a conservative rational choice below `1/M`, not a
distinguished TNFR constant or an evolution parameter.
Also
`0<eta<=d<M*pi/10<pi/130`, using the two-ring Jensen minimum and
integrating the displayed derivative. Storage nonincrease makes these same
initial-budget estimates valid at every future time. For a ring gap
`y>=kappa`, they imply

\[
\eta\le F(\pi/2)-F(y)\le M(\pi/2-y),\qquad
\pi/2-y>13\eta.
\]

For `y<=kappa`, its margin is at least `pi/10>13*eta`. A bridge instead
has cost `1-cos(delta)<=d-eta<pi/130<1/2`, hence `abs(delta)<pi/3` and
margin greater than `pi/6>13*eta`. Every support edge therefore obeys the
uniform, all-future strict acute margin

\[
\boxed{\quad \pi/2-|\delta_{ij}(t)|>13\eta>0.\quad}
\]

Let `z_i=sum_{j~i} exp(i*(theta_j-theta_i))` be the relative neighbor
resultant and `d_i` the support degree. Sine concavity on `[0,pi/2]`
gives `cos(delta_ij)>=sin(13*eta)>=26*eta/pi`. In the resulting
positive-real chamber, `alpha_i=Arg(z_i)` lies in `(-pi/2,pi/2)` and
`H_i=pi*Re(z_i)*tan(alpha_i)/alpha_i`, with ratio one at zero. Thus

\[
\boxed{\quad
\operatorname{Re}z_i(t)\ge\frac{26d_i\eta}{\pi},\qquad
|z_i(t)|\ge\frac{26d_i\eta}{\pi},\qquad
H_i(t)\ge26d_i\eta>0.
\quad}
\]

The phase Hessian likewise satisfies
`Hess(V_phi)>= (26*eta/pi)*B` as quadratic forms. These bounds quantify
the compact regular domain used by the proof; they do not assert a global
exponential rate or introduce additional phase damping.

The shared certificate can retain a rational
`eta_lower=B_lower-E_upper/beta`, where `B_lower` encloses the geometric
barrier and `E_upper` encloses the actual full-state storage. Only complete
admission of all sector hypotheses makes the future bounds available.
Replacing eta by this positive lower bound and pi by its certified upper
bound on the right-hand side preserves every non-strict lower bound above.
Unit-capacity and unit-beta flags remain descriptive evidence, not admission
requirements. These are exact ideal-law implications from the captured
represented snapshot. They neither bound the preceding numerical error nor
by themselves close the finite-time transit obligation for the original
initial value problem; the validated proof below supplies that separate link.
The frozen response and its later endpoint audit remain separate.
