# Native source preparations, formation barriers and boundary access

Sections 17–23 retain native-law source/receiver preparations, storage
and symmetry restrictions, equilibrium classification, singular-boundary
results and frozen finite responses. Admission, a hypothetical path and
a bounded response are distinct from proved formation. A boundary
obstruction does not authorize continuation under another law.

Part of [Native regional composition](RELATIONAL_PATTERN_COMPOSITION.md).
Section numbers remain stable across this collection. Each result keeps
its full hypotheses, implementation and checks; the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone assigns research work.

## 17. Storage admission for a formed source and an initially uniform receiver

<a id="induced-formation-storage-obstruction"></a>

The retained-receiver result in
[pattern memory](RELATIONAL_PHASE_MEMORY.md#relational-retained-receiver-record)
concerns interaction between two already formed winding-one patterns.
Here the receiver initially lacks that phase geometry. The question is
whether a state-preserving supplied connection can form the receiver while
the source also ends in its maintained winding-one sector.

Fix two simple unit C5 rings, unit held capacities, `beta=1`, `e=w=1/2`,
the native unforced relational law and one common structural clock. Initially
all ten nodes have the same uniform form. The source phases are the exact
twist `theta_L,i=i*kappa`, with `kappa=2*pi/5`; the receiver phases are all
`alpha`, where the relative common phase `alpha` is retained as a parameter.
Use either the single bridge `(L0,R0)` or the two bridges
`(L0,R0),(L1,R1)`. Both fine cycles and the candidate bridge set are supplied;
this is an internal formation question, not a construction of nodes or
primitive support.

The attachment changes no nodal coordinate. Its phase-edge storage is part
of the initial joined storage and must be accounted as event work. There
is no form impulse, phase reset, reservoir input or later event in the
continuation considered below.

### The positive-resultant chamber bounds the available initial storage

Write

\[
c=\cos\kappa=\frac{\sqrt5-1}{4},\qquad
V_*=5(1-c),\qquad \delta_j=\alpha-j\kappa.
\]

At each occupied source port the relative resultant is

\[
z_{L,j}=2c+e^{i\delta_j}.
\]

At its uniform receiver partner it is `z_R,j=2+exp(-i*delta_j)`.
Unoccupied source and receiver nodes have positive real resultants `2c`
and `2`. Consequently, membership in the ideal
[`positive_resultant` chamber](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-positive-resultant-execution)
selected by that engine option is equivalent for this family to

\[
2c+\cos\delta_j>0
\]

at every occupied source port. The receiver conditions are automatic.
Each admitted bridge therefore costs strictly less than `1+2c`.
With `k` equal to one or two bridges, the entire initial form storage is
zero and

\[
E_{\rm initial}
=V_*+\sum_{j=0}^{k-1}(1-\cos\delta_j)
<V_*+k(1+2c).
\]

This bound already includes the attachment work. Supplying an additional
unrecorded work budget without changing the declared state or equations
would not change this initial-value problem. The two-port phase gaps are
not independent, but applying this per-port bound remains valid; no claim
that its supremum is attained simultaneously is needed.

### Both maintained acute twists require more storage

The target requires both original rings to have strict acute oriented
edge gaps with winding `+1`, as in their maintained-twist basin. For each
such ring the five gaps sum to `2*pi`. Convexity of `1-cos` on the acute
interval gives phase storage at least `V_*`. Form and bridge storage are
nonnegative, so every target state obeys

\[
E_{\rm target}\ge2V_*.
\]

The fixed-support native law obeys

\[
\dot E=-\frac12\sum_i\frac{q_i^2}{d_i}\le0.
\]

Yet its initial deficit from this necessary target level is bounded below by

\[
\begin{aligned}
2V_*-E_{\rm initial}&>4-7c &&(k=1),\\
2V_*-E_{\rm initial}&>3-9c &&(k=2).
\end{aligned}
\]

Both bounds are strictly positive: `c<1/3`, and in particular
`3-9c=(21-9*sqrt(5))/4>0`. Thus **neither of the two interfaces can reach
the stated target from an initially positive-resultant member of this
equal-form family under the unforced storage-nonincreasing continuation**.
No trajectory or phase-offset search is needed to establish the obstruction.

The proof does not assume the trajectory stays acute. It also cannot be
evaded merely by allowing a later regular excursion outside the positive-real
chamber while retaining the same storage balance: the initial deficit has
already been established. Even replacing the initial strict inequalities
by their nonnegative-real limiting values leaves a positive deficit. Such
a boundary state is outside strict `positive_resultant` admission; it can
still belong to the separate `regular` domain when its imaginary resultant
components keep it away from the excluded ray.

The target qualification matters. An arbitrary nonacute state with ordinary
winding one need not have storage at least `V_*`; winding alone is not this
maintenance target. For example, the positive wrapped gaps
`(7*pi/8,9*pi/32,9*pi/32,9*pi/32,9*pi/32)` sum to `2*pi` but their
cosine storage is below `V_*`, as the shared exact interval controls verify.
The result does not forbid transient receiver structure,
a changed source identity, or formation from another preparation with a
different admitted resource budget.

### The regular option admits a wider initial family

The initial positive-real condition is a computational admission restriction,
not a universal requirement of the already defined
[full regular relational law](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-regular-domain-admission).
This distinction is essential for the two-port case. Consider the same
equal-form preparation and two bridges, but choose

\[
\alpha=\pi+\frac\kappa2.
\]

The two source resultants are then

\[
2c-\cos(\kappa/2)\mp i\sin(\kappa/2).
\]

Their real parts are negative and their imaginary parts nonzero. They are
outside `positive_resultant`, but avoid both the zero resultant and the
negative-real branch itself. Their native metrics
`H_i=pi*Im(z_i)/Arg(z_i)` are finite and strictly positive. The receiver
ports have positive real part `2-cos(kappa/2)`; all other resultants retain
their earlier positive values. Hence this is a locally regular state of
the same mathematical law, with a smooth local flow. Its represented
preparation is admitted by the explicit
[`phase_domain="regular"` option](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-full-regular-execution),
while `positive_resultant` correctly rejects it. The wider option introduces
no new constitutive law; it certifies the domain already used in the derivation.

Its two-bridge cost exceeds the receiver twist's necessary phase storage:

\[
\begin{aligned}
E_{\rm initial}-2V_*
&=2+2\cos(\kappa/2)-V_*\\
&=\frac{7\sqrt5-15}{4}>0.
\end{aligned}
\]

This is a counterexample to extending the preceding energy deficit to all
regular initial states. It is not a formation trajectory: local regularity
and sufficient total storage do not prove global continuation, preservation
of the source, crossing into the target sector or capture. With only one
bridge, its cost is at most two, so `V_*+2<2V_*` still obstructs the
equal-form target even without the initial positive-real restriction.

The scoped conclusion therefore separates a proved obstruction for both
initially positive-resultant interfaces from an unresolved two-port formation
candidate admitted by `regular`. Treating the computational chamber
as a physical law, or treating the wider state's energy surplus as successful
formation, would conflate different obligations. The shared regular engine
retains certified point/chord admission and represented pressure/work defects;
these are not an exact trajectory enclosure or target-entry certificate.
The acute default and historical evaluated protocols remain unchanged.
The [regular reachability audit](#regular-seeded-reachability-audit) below
resolves the component and necessary-crossing questions for the fixed wider
preparation; its continuous trajectory and maintenance remain separate.

The shared
[`certify_relational_seeded_formation_obstruction`](../../src/tnfr/physics/relational_capture.py)
returns a `RelationalSeededFormationObstruction` with separate interface
cases, exact outward storage bounds and an explicit `obstructed` verdict.
It has no successful-formation or ordinary capture-admission interpretation.
The [capture API controls](../../tests/test_relational_capture.py) check the
shared report contract. The
[independent formation controls](../../tests/physics/test_relational_seeded_formation.py)
verify the radical gaps, native initial budgets and full-regular metric
witness. The [regular execution controls](../../tests/test_relational_regular_execution.py)
check its admission by `regular` and rejection by `positive_resultant`.
Neither certificate nor control advances a trajectory or changes live support.

## 18. Regular-domain geometry of the fixed source/receiver preparation

<a id="regular-seeded-reachability-audit"></a>

Freeze section 17's two-port preparation with `alpha=6*pi/5`, all forms zero,
unit held capacities, `beta=1`, `e=w=1/2`, no forcing and no subsequent event.
Write `A_*=4*pi/5`, `c=cos(2*pi/5)` and `V_*=5*(1-c)`. Its total joined
storage, including the supplied attachment, is

\[
E_0=V_*+2+2\cos(\pi/5)=\frac{35-3\sqrt5}{4},\qquad
E_0-7=\frac{7-3\sqrt5}{4}>0.
\]

The last inequality follows from `49>45`. This audit distinguishes regular
geometric connectivity, a necessary phase crossing, and the actual unforced
initial-value problem. No phase path below is imposed on that problem.

### The unchanged law preserves a reflection subspace

On each ring, reflection `j -> 1-j` modulo five exchanges ports zero and one,
exchanges nodes two and four, and fixes node three. The joined graph is
invariant under the simultaneous reflection of both rings. Compose that
permutation with `x -> -x` and `theta -> 2*alpha-theta` modulo `2*pi`.
The frozen preparation is fixed by this transformation. The full regular
law commutes with it: relative resultants are conjugated, `g` changes sign,
`H` is unchanged, and both evolution rows transform with the corresponding
signs. Smooth uniqueness therefore preserves this subspace for as long as
the solution stays regular.

Use continuous phase lifts based at the fixed node's phase `alpha`. The
source and receiver then have the respective coordinates

\[
\begin{aligned}
x_L&=(p,-p,-r,0,r),&\quad \theta_L-\alpha\mathbf1&=(a,-a,-b,0,b),\\
x_R&=(P,-P,-R,0,R),&\quad \theta_R-\alpha\mathbf1&=(A,-A,-B,0,B).
\end{aligned}
\]

Initially `(a,b,A,B)=(A_*,A_*/2,0,0)` and all four form coordinates vanish.
The degree-two middle node has resultant `2*cos(b)` on the source and
`2*cos(B)` on the receiver. Regularity excludes both zero and negative real
resultants, so the lifts stay in `|b|,|B|<pi/2`. The other degree-two
source resultants are

\[
z_{L,2}=2\cos(a/2)e^{i(b-a/2)},\qquad z_{L,4}=\overline{z_{L,2}}.
\]

A continuous lift cannot cross `a=+/-pi` without making them zero. Thus
`|a|<pi`; likewise `|A|<pi`. Within these bounds, `|b-a/2|<pi` and its
receiver counterpart hold automatically. The remaining domain conditions
are exactly the two port-resultant conditions and their conjugates:

\[
\begin{aligned}
z_{L,0}&=e^{-2ia}+e^{i(b-a)}+e^{i(A-a)},\\
z_{R,0}&=e^{-2iA}+e^{i(B-A)}+e^{i(a-A)}.
\end{aligned}
\]

These reductions follow from the supplied symmetry; they are not new
primitive coordinates or a closure for arbitrary asymmetric states.
They concern the exact ideal preparation with mathematical `pi`.
Independently materializing its phases as binary floating values can break
exact reflection. Point/chord admission by the regular executor does not
itself certify this symmetry or a continuous reduced trajectory for those
represented coordinates.

### Receiver formation must cross an internal cancellation

In these lifts a strict acute winding-one ring necessarily satisfies
`a>3*pi/4` and `b>a-pi/2>pi/4`; replace lower-case coordinates by capitals
for the receiver. Indeed, the oriented gaps are the wrapped versions of
`(-2*a,a-b,b,b,a-b)`. Acute admission forces `|a-b|<pi/2`, and winding one
then requires `wrap(-2*a)=2*pi-2*a` in `(0,pi/2)`. Consequently every such
target has `a+b>pi` and `A+B>pi`. The initially uniform receiver must cross
`A+B=pi` before reaching this target.

At that crossing its two internal neighbor phasors cancel at both ports:

\[
e^{-2iA}+e^{i(B-A)}=0,\qquad
z_{R,0}=e^{i(a-A)},\qquad z_{R,1}=e^{-i(a-A)}.
\]

An isolated C5 would have zero resultants there. The supplied bridge
neighbors instead leave unit nonzero resultants. They are regular whenever
the bridge gap is not antipodal. In particular, if the source already has
`a+b>=pi`, the lift bounds imply `a,A>pi/2`, hence `|a-A|<pi/2`; the receiver
port resultants even have positive real part at this necessary crossing.
This conclusion admits those ports, not automatically every other node.

The pure-cycle skip-winding invariant therefore does not forbid this joined
crossing. Only the two occupied ports can bypass the isolated-ring
cancellation: the other three nodes still have exactly their two ring
neighbors. This identifies the role of the supplied support without deriving
an autonomous contact or a crossing time.

The necessary boundary itself contains a regular state below the frozen
initial budget while keeping the source twist unchanged. Take

\[
(a,b)=(4\pi/5,2\pi/5),\qquad (A,B)=(3\pi/4,\pi/4),\qquad x=0.
\]

Then `A+B=pi` and the bridge gaps are `+/-pi/20`. The source ports have
positive real part `2*c+cos(pi/20)`; the receiver ports are exactly
`exp(+/-i*pi/20)`. The receiver's other resultants are
`2*cos(3*pi/8)*exp(+/-i*pi/8)` and `sqrt(2)`, all with positive real part.
Its ring storage is `5-sqrt(2)`, so the joined boundary storage `E_b` obeys

\[
E_0-E_b
=2\cos(\pi/5)+2\cos(\pi/20)-5+\sqrt2
>\frac{3}{400}>0.
\]

For an exact elementary bound, use `cos(u)>=1-u^2/2`,
`sqrt(5)>559/250`, `sqrt(2)>7071/5000` and `pi<22/7`. They give the strict
lower bound `1839/245000=3/400+3/490000`; the square-root bounds follow by
squaring the positive rationals. Thus the required cancellation boundary
is not everywhere above the available initial storage. This point is not
a reached state or a demonstrated connection within a nonincreasing-energy
path; preceding dissipation and all intermediate states still matter.

### The preparation and target share a regular storage sublevel

There is an explicit continuous geometric path with zero form throughout
from the preparation to the aligned two-twist target, entirely within
`E<=E_0`. It has two legs. This construction does not keep storage monotone
and does not preserve the source winding throughout.

First hold the receiver uniform, `A=B=0`, and decrease the source coordinates
`(a,b)=(u,u/2)` from `u=A_*` to zero. Its degree-two resultants are positive
real. The receiver ports have `z=2+exp(+/-i*u)`, with real part at least one.
The source port zero has

\[
z=e^{-2iu}+e^{-iu/2}+e^{-iu},\qquad
-\operatorname{Im}z=\sin(2u)+\sin(u/2)+\sin u
=\sin(u/2)(8h^3-2h+1),\quad h=\cos(u/2).
\]

Here `h>=c>3/10`. The polynomial `8*h^3-2*h+1` is increasing for
`h>=3/10` and at `3/10` equals `77/125>0`. Thus this imaginary part is
strictly negative for `0<u<=A_*`; the conjugate port has the opposite sign.
At zero both resultants equal three. The whole first leg is regular even
where a source port's real part is negative.

Define the phase storage of one ring on this curve as

\[
V(u)=5-\cos(2u)-4\cos(u/2).
\]

The joined storage on the first leg is

\[
F(u)=V(u)+2(1-\cos u),\qquad
F'(u)=2[\sin(2u)+\sin(u/2)+\sin u]>0
\]

for `0<u<=A_*`. Decreasing `u` therefore lowers storage from `F(A_*)=E_0`
to zero. The common endpoint is joint phase consensus.

For the second leg increase both rings together:
`a=A=u`, `b=B=u/2`, from zero to `A_*`. Both bridge gaps vanish. Port
resultants satisfy

\[
\operatorname{Re}(1+e^{-2iu}+e^{-iu/2})
=1+\cos(2u)+\cos(u/2)\ge\cos(u/2)\ge c>0.
\]

The interior resultants are again positive. This is the previously derived
[added-support regular path](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-regular-winding-crossing),
in the reflection coordinates. Its total storage is `2*V(u)`. Since

\[
V'(u)=2\sin(u/2)(8h^3-4h+1)
=2\sin(u/2)(2h-1)(4h^2+2h-1),
\]

and `h` decreases from one to the positive root `c` of the last quadratic,
`V` increases up to `u=2*pi/3` and then decreases to its twist value.
The maximum joined storage is exactly `2*V(2*pi/3)=7<E_0`.
Every resultant stays regular on the compact two-leg path. Thus neither
disconnected regular components nor an intermediate minimum storage barrier
above `E_0` can rule out this initial/target pair without additional
constraints on the path.

This is not a trajectory of the unforced law. On the second leg storage
increases from zero, whereas an actual trajectory must obey `E_dot<=0`.
Furthermore the first leg unwinds the source before the second leg rebuilds
both ring geometries. The construction cannot certify formation while
preserving the source identity at all times. It establishes geometric
connectivity in a common sublevel, and nothing stronger about the selected
initial-value problem or the dissipation it accumulates.

### Eight coordinates retain the actual symmetric dynamics

The invariant restriction closes exactly without equating the source and
receiver. Define the full-graph form gradients at ports zero and four by

\[
q_L=4p-r-P,\quad v_L=2r-p,\qquad
q_R=4P-R-p,\quad v_R=2R-P.
\]

For the source let `g_L=Arg(z_L,0)/pi`,
`H_L=pi*Im(z_L,0)/Arg(z_L,0)` with its positive-real extension. The interior
row has

\[
g_{L,4}=\frac{a/2-b}{\pi},\qquad
H_{L,4}=2\pi\cos(a/2)\operatorname{sinc}(a/2-b).
\]

Use the same expressions with `(a,b)` replaced by `(A,B)` for the receiver.
The existing full law restricts to

\[
\begin{aligned}
\dot p&=-\frac e3q_L+w g_L,&
\dot r&=-\frac e2v_L+w g_{L,4},\\
\dot a&=\frac w\beta\frac{q_L}{H_L},&
\dot b&=\frac w\beta\frac{v_L}{H_{L,4}},
\end{aligned}
\]

and the four receiver rows obtained by exchanging upper/lower-case
coordinates. All other nodal rows follow by reflection; both fixed-node
rows are zero. This is an exact invariant restriction of the ten-node law,
not an assumed coarse law or an elimination of hidden state. Its form storage
is

\[
E_D=2p^2+(p-r)^2+r^2+2P^2+(P-R)^2+R^2+(p-P)^2.
\]

With `V(a,b)=5-cos(2*a)-2*cos(a-b)-2*cos(b)`, total storage is
`E_D+beta*[V(a,b)+V(A,B)+2*(1-cos(a-A))]`. The inherited loss is

\[
\dot E=-e\left[\frac23(q_L^2+q_R^2)+v_L^2+v_R^2\right].
\]

The shared native field already evaluates these equations after full-state
reconstruction. A future interval proof must admit both port resultants and
the degree-two rows throughout its tubes. The existing copied-ring,
positive-resultant transit proof has different hypotheses: it identifies the
two rings and retains only four coordinates. Applying that solver here would
discard the source/receiver difference that drives the question.

### The initial native response is not the constructed path

Put `s=sin(pi/5)`, `d=cos(pi/5)` and

\[
\rho=\frac1\pi\arctan\frac{s}{2-d}>0,\qquad
h_L=\frac{5s}{3},\qquad h_R=\frac{s}{\rho}.
\]

In original node order, the initial phase sources and metrics are

\[
g_L=(-3/5,3/5,0,0,0),\quad g_R=(\rho,-\rho,0,0,0),
\]
\[
H_L=(h_L,h_L,2\pi c,2\pi c,2\pi c),\quad
H_R=(h_R,h_R,2\pi,2\pi,2\pi).
\]

Equal zero form gives `x_dot=g/2` and `theta_dot=0`, but the next phase
derivative is nonzero. Write

\[
Q_L=\frac65+\frac\rho2,\quad Q_R=\frac3{10}+2\rho,
\quad \ell=\frac{Q_L}{2h_L},\quad k=\frac3{40\pi c},
\quad m=\frac{Q_R}{2h_R},\quad n=\frac\rho{8\pi}.
\]

Since `q_dot=B*x_dot` and the initial `q` vanishes, differentiation of the
native phase row gives

\[
\ddot\theta_L=(-\ell,\ell,-k,0,k),\qquad
\ddot\theta_R=(m,-m,n,0,-n).
\]

Thus `a_ddot=-ell`, `b_ddot=k`, `A_ddot=m`, `B_ddot=-n`. In particular,
the receiver starts deforming even though its initial phase velocity is zero.
Neither of the prescribed geometric paths supplies these accelerations.
These rates can also be evaluated by multiplying the existing uniform-form
joint tangent by the initial full field; no finite-step difference or
additional evolution rule is needed.

For the source port zero, let `C_0=2*c-d`, `S_0=-s` and
`psi=Arg(z_L,0)`. Initially `psi=-3*pi/5` and all first phase derivatives
vanish. Direct differentiation gives

\[
\begin{aligned}
\ddot C_0&=s(\ell+m)-\sin(2\pi/5)(\ell-k),\\
\ddot S_0&=c(3\ell+k)-d(\ell+m),\\
\ddot\psi(0)&=
\frac{C_0\ddot S_0-S_0\ddot C_0}{C_0^2+S_0^2}<0.
\end{aligned}
\]

The shared outward interval controls enclose the last derivative near
`-0.204238` and `S_0_ddot` near `0.041121`, with strict signs. The source
argument initially moves toward `-pi`; since `C_0<0`, its distance to the
excluded ray also initially decreases at second order. This is a local
direction, not a proved finite-time collision or loss of regularity.

There is initially no dissipation because `q=0`, but storage is not conserved.
Define

\[
D_2=\frac{Q_L^2+Q_R^2}{3}+\frac9{200}+\frac{\rho^2}{8}>0.
\]

Smoothness and the exact loss identity imply

\[
E(t)=E_0-\frac{D_2}{3}t^3+O(t^4).
\]

An initial zero storage derivative therefore does not certify equilibrium or
free evolution. Without an explicit remainder bound, this local expansion
cannot give a target-budget exhaustion time, branch-contact time or future
formation verdict.

The [static reachability controls](../../tests/physics/test_relational_seeded_reachability.py)
reuse the shared field, uniform tangent and rational interval kernels. They
check the exact symmetry, crossing budget and initial derivative signs against
independent expressions. They do not integrate this new initial-value problem,
modify a frozen response, install an event or certify a finite-time outcome.
The [single queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the subsequent whole-time validation obligation.

## 19. A bounded continuous response of the fixed regular preparation

<a id="regular-seeded-continuous-response"></a>

The [reflected field](../../src/tnfr/physics/_relational_reflected_flow.py)
implements section 18's exact eight-coordinate restriction for interval states
and Taylor jets. The [continuous proof owner](../../src/tnfr/physics/relational_reflected_transit.py)
consumes that field with the shared Picard/Taylor/comparison kernel. This is a
proof calculation on the ideal mathematical-pi initial-value problem, not an
Euler trajectory or a projection of separately rounded phases onto symmetry.
The same field is checked against the native ten-node owner at independently
prepared represented states. Native execution and continuous enclosure keep
their separate arithmetic provenance.

### Whole-time admission and storage accounting

For each initial box `X`, a convex tube `Y` must satisfy strict inclusion
`X+[0,h]*F(Y)` inside `Y`. Every full resultant and the declared continuous
reflection lifts are admitted on that tube. This establishes existence and
uniqueness throughout the step for every initial point in `X`. In particular,
the true exact-pi preparation is enclosed without identifying pi with a float.

The center solution is bounded by its order-six Taylor polynomial plus an
order-seven derivative remainder on `Y`. Normalized interval jets obtain those
derivatives from the same law. A Metzler matrix of Jacobian bounds on `Y`
propagates initial radii; its shared nonnegative-series comparison bound adds
outward rounding and a rigorous tail. Intersecting the endpoint enclosure with
the independently proved whole tube removes only overestimation. The older
four-coordinate transit owner now delegates this generic step arithmetic to
the same kernel, preserving its separate model and capture contract.

The six independent nodal resultants use interval principal arguments. If
`Re(z)>0` throughout a box, the inverse metric is evaluated as
`atan_ratio(S/C)/(pi*C)`, including the removable `S=0` limit. Otherwise a
separated imaginary sign permits `Arg(z)/(pi*S)`. The shared argument jet
differentiates `(C*S'-S*C')/(C^2+S^2)`; scalar function error is not substituted
for derivative uncertainty. Unresolved rectangles reject the proof step.

For each accepted tube, integrating the interval loss gives
`h*D(Y)`; summing these intervals encloses accumulated loss. The direct endpoint
storage is intersected with `E(0)-integrated_loss`, and the corresponding loss
is tightened by the same identity. A disjoint intersection reports failure.
The target budget is `E(t)-2*V_*`: a positive lower bound only retains this
necessary resource; a negative upper bound would exclude subsequent entry to
the two acute twist sectors under the same unforced loss law. Ordinary winding
flags are checked on every whole tube, separately from target capture.

### Frozen prediction and separately identified numerical correction

The [producer](../../benchmarks/relational_seeded_response.py) freezes the
existing preparation, support, unit capacities, `e=w=1/2`, `beta=1`, no inputs
or events, structural horizon `T=1/8`, step `h=1/64`, Taylor order six and
128-bit outward interval arithmetic. Its endpoint-width budget is `2^-20`.
Before evaluation it predicts an admitted full window, positive receiver port
phase `A(T)`, a source-port argument below its initial `-3*pi/5`, positive
accumulated loss, positive remaining target budget and winding `(1,0)` throughout.
None is a prediction of receiver formation within that short window.

The first record, `response-v1.json`, is retained as **inconclusive**. It
certifies two steps to `t=1/32` but its next comparison bound exceeds the
kernel's admitted norm budget. The responsible interval expression used
`Arg(z)/S` even for tiny sign-separated `S` with positive `C`. Its removable
singularity was mathematically harmless but inflated derivative intervals.
A separate static control reproduces this issue and verifies the equivalent
positive-real `atan_ratio` expression against an independent local expansion.
No constitutive law, initial state, horizon, step, Taylor order or prediction
was changed to obtain a different physical result.

The separately frozen `response-v2.json` records that correction and the v1
record's digest. It passes all declared gates, with eight accepted steps to
`T=1/8`. This is a corrected computational verification, not an independent
blind replication or a replacement of the first verdict. Outward rounded
summaries of its exact retained intervals are:

| Observation | Certified enclosure or verdict |
| --- | --- |
| Receiver port phase `A(T)` in the fixed frame | `(0.000548810055118, 0.000548810055123)` |
| Source-port argument change | `(-0.001570247840, -0.001570247837)` |
| Accumulated storage loss | `(0.000423056798, 0.000423056800)` |
| Endpoint total storage | `(7.072525960075, 7.072525960078)` |
| Remaining two-twist target budget | `(0.162695903824, 0.162695903828)` |
| Smallest retained whole-tube resultant/ray margin | greater than `0.58425` |
| Maximum endpoint coordinate width | less than `1.424e-13` |
| Whole-window ordinary winding | source `+1`, receiver `0` |

The receiver therefore begins responding while remaining in its zero-winding
sector over the entire certified window. The source approaches its branch
locally but does not encounter it during this interval. The retained positive
target budget does not select a future route or admit a maintenance basin.

The local evidence bundle is under
`artifacts/research/relational_seeded_response/`, with separate protocol and
source-archive siblings for each response. SHA-256 record identifiers are:

```text
response-v1.json  d65eb3c9fc88979d71d6abed3178a1f02307b5b37b4cb6bd0d077cdfd4c0e867
response-v2.json  183c762fa4853a1f42c566a3683eb948263e0327f60c50819684cf4bb51ece75
```

The hashes bind retained bytes, not chronology, physical authenticity or
dependency binaries. [Static flow controls](../../tests/physics/test_relational_reflected_flow.py),
[proof admission](../../tests/test_relational_reflected_transit.py) and
[producer plumbing](../../tests/test_relational_seeded_protocol.py) remain
separate from evaluating this response. The optional
[retained-response audit](../../tests/physics/test_relational_seeded_response.py)
checks both archives, strict Picard inclusion, endpoint chains, storage/loss
and saved gates without replaying the IVP; it skips absent local bundles.
The single execution plan owns any
new continuation; these artifacts must not be regenerated to follow unrelated
source changes.

## 20. Fixed-IVP continuation and the remaining target budget

<a id="regular-seeded-budget-continuation"></a>

The separately frozen horizon-one computation retained exactly section 19's
initial state, law, fixed support, clock, interval precision, step `1/64`,
Taylor order six and endpoint-width budget `2^-20`, with total horizon `T=1`.
It started from the original mathematical-pi enclosure, not a rounded retained
midpoint.
The shared producer's `target-budget` study froze this declaration and its
source archive before evaluation. The earlier records remain separate.

The discriminant is `B(T)=E(T)-2*V_*`, with
`2*V_*=10*(1-cos(2*pi/5))`. Section 17's acute-ring bound and the same unforced
loss identity imply the following conditional outcomes:

- An upper bound below zero excludes later entry into states in which both
  original rings are strictly acute and have ordinary winding `+1`, wherever
  the same fixed-support regular continuation exists.
- A lower bound above zero retains only the necessary storage resource. It
  proves neither a route, ordinary winding change nor target capture.
- An interval containing zero leaves this test unresolved, including exact
  equality. Nonacute winding configurations need not obey the acute threshold.

Numerical horizon completion and width admission are separate from that sign.
Either strict sign is an informative numerical verdict. A failed tube cannot
prove a singularity; an earlier certified negative budget remains a scoped
exclusion even if a later endpoint is unavailable. The report retains the first
such certified time, full storage/loss evidence and independent winding flags.
False winding flags indicate missing certification, not proved winding change.
The frozen stop condition allowed one retained continuation verdict,
without automatic horizon extension or an adjusted law.

### Retained horizon-one result

The separately frozen `budget-t1-v1.json` completes all 64 steps through `T=1`
with no failed tube or evaluation error. Its three numerical gates pass:
full-horizon admission, endpoint width and a strict budget sign. The sign is
positive, so the scoped verdict is **target not excluded by storage**. This
does not certify target formation. Outward decimal summaries are:

| Observation at `T=1` | Certified enclosure or verdict |
| --- | --- |
| Remaining target budget | `(0.032840217948, 0.032840218115)` |
| Accumulated loss | `(0.130278742510, 0.130278742677)` |
| Total storage | `(6.942670274198, 6.942670274366)` |
| Receiver port phase `A` | `(0.024811891363, 0.024811891367)` |
| Source port phase `a` | `(2.259008131438, 2.259008131465)` |
| Source-port argument change from preparation | `(-0.062890231290, -0.062890231102)` |
| Maximum endpoint coordinate width | less than `2.396e-11` |
| Smallest retained whole-tube domain/lift margin | greater than `0.29240` |
| Whole-window ordinary winding | source `+1`, receiver `0` |

The budget has fallen from approximately `0.16312` initially to `0.03284`.
The initial response therefore persists beyond the earlier window without
receiver winding formation. In particular, preservation of the source's
ordinary winding does not imply preservation of its acute ring geometry.
At the endpoint its wrapped port-to-port gap is `2*pi-2*a`, enclosed in
`(1.765169044251, 1.765169044302)`, strictly above `pi/2`. The saved endpoint
at `49/64` already has a strictly nonacute source gap; this statement does not
locate the first exact crossing time. The source can deform outside its
acute ring sector while all full nodal resultants remain regular.

A read-only evaluation of the retained endpoint gives
`D(1)` in `(0.321460727563, 0.321460727582)`. This positive instantaneous loss
and the small remaining resource motivated the separately frozen nine-eighths
test below. Their ratio is not a certified exhaustion time: the loss rate
evolves. These horizon-one observations alone do not establish a later
trajectory, singularity or formation result.

The evidence bundle remains in `artifacts/research/relational_seeded_response/`:

```text
budget-t1-v1.json           f142e75070b141644541e91ac549e7fa9d6c5bfbce8700735130b870a6a0c483
budget-t1-v1.protocol.json  5c748341025197b1c6219c2b553b37f29bee8199ca1088edf157e33154189bf4
budget-t1-v1.source.zip     37e2150bed71757de7399e03bf13158919c090330af1b5909784f7cfa66f059d
```

The [budget disposition controls](../../tests/test_relational_seeded_budget.py)
check strict signs, partial horizons and protocol admission without evolving
this IVP. The [optional retained audit](../../tests/physics/test_relational_seeded_response.py)
checks the archive, interval chain, loss balance and independent geometry
without replay. Record digests establish byte consistency, not independent
physical evidence or authenticated chronology. The single execution plan
owns any new continuation.

### Prospective extension to nine eighths

The separately frozen nine-eighths record retained the same state, law, clock,
step, order and width budget, declaring `T=9/8` through the shared
`target-budget` protocol. It started from the original exact initial enclosure.
The strict budget-sign test is unchanged; either resolved sign is informative.
The horizon and source archive were frozen separately before evaluation, and
no adaptive continuation or replay of the earlier evidence was used to choose
the result.

The retained `budget-t9over8-v1.json` completes all 72 steps with all numerical
gates admitted. Its final target budget lies in
`(-0.010524533169, -0.010524532905)`: the specified two-acute-twist target is
excluded under subsequent unforced fixed-support regular continuation.
The first saved endpoint certifying a negative budget is `71/64`; the previous
one, `35/32`, is still positive. These endpoints bracket a budget crossing,
not an exact exhaustion time or a phase singularity.

At `9/8`, total storage lies in `(6.899305523081, 6.899305523346)` and accumulated
loss in `(0.173643493529, 0.173643493795)`. Maximum coordinate width is less than
`3.796e-11`; all retained domain/lift margins exceed `0.28885`. Ordinary
winding remains `(1,0)` throughout. The first 64 steps and observations equal
the preceding horizon-one record exactly. This calculation closes the declared
budget test for this preparation; it does not exclude receiver formation
accompanied by source loss, nonacute patterns, other preparations or laws.

```text
budget-t9over8-v1.json           d3174948f69f600114a3dce2e4915d79cef75230c7a7c7100b6da7de0c6ef332
budget-t9over8-v1.protocol.json  a278574a78e65cd3d55376256f4e230eebb631ac645d34884907eb51b1c8cf1d
budget-t9over8-v1.source.zip     b76ce10679b49c180b21bd5491e2fd10a24922727e0ba342302df0097990f92a
```

## 21. A sharp collective energy envelope and transition barrier

<a id="reflected-collective-energy-barrier"></a>

Reusing the exact reflection, edge storage and nonincrease law gives a stronger
obstruction than the target minimum. This is a retrospective theorem applied
to retained evidence, not a changed prediction or replacement of section 20's
original verdicts. The argument requires the same fixed two-C5 support,
matching bridges, reflected state and regular unforced law with `e>=0`,
`w,beta>0`. It supplies no new pressure term or evolution rule.

### The copied geometry is a sharp envelope, not a closed mean dynamics

Introduce exact coordinates

\[
m=\frac{a+A}{2},\quad d=\frac{a-A}{2},\quad
u=\frac a2-b,\quad v=\frac A2-B.
\]

The full phase potential from section 18 satisfies the identity

\[
\begin{aligned}
V={}&W(m)+4\cos^2(m)(1-\cos(2d))\\
 &+8\cos(m/2)(1-\cos(d/2))\\
 &+4\cos(a/2)(1-\cos u)+4\cos(A/2)(1-\cos v),\\
W(m)={}&10-2\cos(2m)-8\cos(m/2).
\end{aligned}
\]

To derive it, use
`cos(a-b)+cos(b)=2*cos(a/2)*cos(a/2-b)` on each ring, then collect the two
port terms and both bridge costs using `a=m+d`, `A=m-d`. In the inherited
lift `|a|,|A|<pi`, all four remainders are nonnegative. Nonnegative form
storage therefore gives the sharp bound

\[
\boxed{\quad E\ge\beta W(m).\quad}
\]

Equality is attained by zero forms, `a=A=m` and `b=B=m/2`. Thus the copied-ring
phase profile already used in the restricted transit study also bounds the
full unequal eight-coordinate state. No equality of the evolving rings or
closed scalar equation for `m` is inferred. Form and mismatch coordinates
remain necessary to determine its evolution.

### Regularity protects the lift; storage protects a separating surface

The source's two nonport resultants include

\[
z_4=2\cos(a/2)\exp(i(a/2-b)),\qquad z_3=2\cos b.
\]

They vanish at `a=+/-pi` and `b=+/-pi/2`, respectively. The receiver obeys
the same identities. A continuous regular solution starting inside the stated
lift cannot leave it without hitting a zero resultant. Changing phase
representatives does not evade this boundary of the continuous lift.

Since `W(2*pi/3)=W(-2*pi/3)=7`, every point on `a+A=+/-4*pi/3` costs at least
`7*beta`. This bound is attained at zero forms, `a=A=+/-2*pi/3`,
`b=B=+/-pi/3`, whose full nodal resultants are regular. Hence this is a sharp
transition barrier, rather than a numerical-domain threshold.

Within this lift, strict acute winding `+1` on a ring requires `a>3*pi/4`
(and analogously `A>3*pi/4`). Indeed, strict acuteness forces the internal
gap `a-b` into `(-pi/2,pi/2)`; the positive winding then comes from wrapping
the port gap `-2*a`. Both target rings consequently have `a+A>3*pi/2`.

It follows that any admitted reflected state with

\[
E<7\beta,\qquad a+A<4\pi/3
\]

cannot subsequently reach the two acute winding-`+1` rings under the same
regular unforced continuation. More symmetrically, `E<7*beta` and
`|m|<2*pi/3` trap the collective mean between both separating surfaces,
excluding both same-sign acute pairs. The sharp target minimum `2*V_*`
remains unchanged: a target can cost less than the barrier required to reach
its component. A geometric route inside the *initial* sublevel does not imply
a route after dissipative evolution has lowered that sublevel.

### Consequence for retained evidence and limits

The saved endpoint at `51/64` already has
`E` in `(6.999362696581, 6.999362696649)` and `a+A` in
`(2.359699409647, 2.359699409659)`, strictly below `4*pi/3`.
Thus the geometric barrier excludes this preparation's two-acute-`+1` target
earlier than the `2*V_*` budget does. This is the first saved endpoint passing
this sufficient test, not the exact first loss of reachability. The original
positive-budget horizon-one verdict remains correct for its weaker criterion.

The shared
[`certify_relational_reflected_barrier`](../../src/tnfr/physics/relational_reflected_transit.py)
computes storage, exact separator margins and regular-lift admission from an
interval state. It never evolves a trajectory or accepts a fitted storage
value. The [independent static controls](../../tests/test_relational_reflected_barrier.py)
check the algebra, sharp boundary, already formed target and invalid-domain
counterexamples. The [retained audit](../../tests/physics/test_relational_seeded_response.py)
applies the theorem separately to saved endpoints without changing records.

This barrier is not global convergence or regularity. For example, zero forms
with `a=A=pi/6`, `b=pi/2`, `B=pi/12` give
`V=8-4*cos(pi/12)<7` but `z_3=0`; regular interior states approach this boundary.
Thus a central low-storage component need not be compactly contained in the
regular domain. Equilibrium classification and future boundary exclusion
remain separate obligations. Neither this theorem nor the finite response
identifies physical matter or excludes all possible pattern formation.

## 22. Complete reflected regular equilibria and full-network stability

<a id="reflected-regular-equilibria"></a>

Keep section 18's two-C5 support, matching bridges, unit capacities and
inherited regular lift. Equilibrium classification requires `e>=0,w,beta>0`;
the dissipative stability statements below additionally require `e>0`.
The classification is complete within this reflection subspace, not among
all possible states of an arbitrary graph.

### Exact stationary reduction

Vanishing phase rates and positive inverse metrics give
`q_L=v_L=q_R=v_R=0`. Thus `r=p/2`, `R=P/2`, `P=7*p/2` and `p=7*P/2`, so all
four form coordinates vanish. In the inherited lift the nonport argument is
exactly `a/2-b`, since `z_4=2*cos(a/2)*exp(i*(a/2-b))`. Vanishing form rates
therefore imply `b=a/2` and `B=A/2`. The remaining conditions are

\[
f(a)=\sin(A-a),\qquad f(A)=-f(a),\qquad
f(t)=\sin(2t)+\sin(t/2),
\]

with strictly positive port real resultants

\[
C_L=\cos(2a)+\cos(a/2)+\cos(A-a),\quad
C_R=\cos(2A)+\cos(A/2)+\cos(A-a).
\]

At a stationary point the imaginary parts vanish, so a negative real
resultant is outside the regular law, even if the sine equations alone hold.
This two-angle stationary reduction is not a transient two-coordinate law.

### Exhaustion of the stationary branches

Put `m=(a+A)/2`, `d=(a-A)/2`, `M=cos(m)`, `D=cos(d)`,
`u=cos(m/2)>0`, `v=cos(d/2)>0`. The inherited lift gives `|m|+|d|<pi`.

If `d=0`, the equation factors, with `h=cos(a/2)>0`, as

\[
\sin(a/2)(2h-1)(4h^2+2h-1)=0.
\]

It gives the five diagonal states in the table below. If `m=0` and `d!=0`,
stationarity instead gives `16*h^3-8*h+1=0`, with `h=cos(d/2)>0`.
The port resultant is `2-8*h^2`, so regularity requires `h<1/2`.
There is exactly one root there: the cubic decreases to its sole minimum in
that interval and stays negative thereafter up to `1/2`. Its signs at
`1/8` and `1/7` isolate the root `k` between those rationals.

There are no additional asymmetric regular states. For `m*d!=0`, the two
stationary relations reduce to

\[
4uM(2D^2-1)+v=0,\qquad u=-8M^2vD.
\]

They imply `D<0`, `|d|>pi/2`, `|m|<pi/2`, and hence `M>0`. Eliminating `u,v`
gives

\[
32M^3D(2D^2-1)=1,\qquad
M=\frac{2D^2-1}{1+2D},\qquad -1/\sqrt2<D<-1/2.
\]

On this interval both `M(D)` and `D*(2*D^2-1)` are positive and strictly
increasing. At `D=-5/8`, `M=7/8` and their product equation's left side is
`12005/4096>1`. Thus a stationary root has `D<-5/8` and `M<7/8`.
The exact port product under those same relations is

\[
C_L C_R=\frac{2M^2+D-1}{4M^2(M+1)}<0,
\]

because its numerator is below `49/32-5/8-1=-3/32`. One port is therefore
negative real and the candidate is inadmissible. Squared equations have not
been used to admit a state without its original signs and resultants.

### Seven equilibria, five recoverable shapes

Let `d_*=2*acos(k)` for the unique cubic root just defined. Forms are zero,
`b=a/2`, `B=A/2` in every row. Common form/phase offsets in the full graph
are understood separately from the fixed reflected representative.

| Family | `(a,A)` | Ordinary winding | Storage divided by beta | Full quotient modes: stable / unstable |
| --- | --- | --- | --- | --- |
| Consensus | `(0,0)` | `(0,0)` | `0` | `18 / 0` |
| Aligned twists | `(+/-4*pi/5, +/-4*pi/5)` | `(+1,+1)` or `(-1,-1)` | `10*(1-cos(2*pi/5))` | `18 / 0` |
| Aligned saddles | `(+/-2*pi/3, +/-2*pi/3)` | `(+1,+1)` or `(-1,-1)` | `7` | `17 / 1` |
| Opposite twists | `(d_*,-d_*)` and its negative | `(+1,-1)` or `(-1,+1)` | `8+16*k^2-6*k` | `18 / 0` |

The opposite twists are acute in their **wrapped edge differences**. Their
internal path cosine is `k>0`; port and bridge cosines are
`8*k^4-8*k^2+1>0` using `1/8<k<1/7`. A large continuous lift `a` does not mean
a nonacute edge. Their storage is nevertheless

\[
E/\beta=7+16(k-3/16)^2+7/16\ge119/16>7.
\]

For consensus and both twist families every cosine edge weight is positive
on connected support. The phase Hessian consequently has inertia `(9,0,1)`.
The [full-network stiffness theorem](RELATIONAL_RECOVERY_AND_INTERACTION.md#regular-equilibrium-stiffness)
gives local exponential recovery in all 18 quotient coordinates; this includes
perturbations outside reflection. Two common-offset modes are neutral in the
full twenty-coordinate state.

At either diagonal saddle, each ring has weight `-1/2` on its port edge and
`+1/2` on its other four edges; both bridges have weight one. Ring exchange
splits the Hessian into `K_r` and `K_r+2*diag(1,1,0,0,0)`.
The first block has exactly one negative direction: it is a positive path
Laplacian minus a rank-one edge, and `(1,-1,-1/2,0,1/2)` has quadratic value
`-3/2`. The second block is positive definite, since its quadratic form is
the positive half-weight path sum plus `(v_0+v_1)^2+(v_0-v_1)^2/2`.
Thus the full phase inertia is `(8,1,1)` and the joint quotient has exactly
one growing real mode. Numerical eigenvalues do not supply this proof.

### What the classification does and does not settle

In `E<7*beta` the only regular equilibria are consensus and the aligned
twists. In the central component `|m|<2*pi/3`, only consensus remains. Thus
the retained failed formation trajectory has no stationary reflected
mixed-winding `(1,0)` destination. This is not yet its convergence verdict.

If `e>0` and its continuation stays uniformly separated from all excluded
resultant rays, the existing storage law and reflection bounds give a
precompact invariant set. On its invariant zero-loss subset all forms must
vanish and remain zero, requiring `g=0`. The classification then leaves only
consensus, and the existing LaSalle argument gives convergence to it.
The uniform regularity premise is not supplied by the energy sublevel: zero
forms with `a=A=B=0` and `b` approaching `pi/2` from below are regular with
`E=4*beta*(1-cos(b))<4*beta`, yet approach a zero central resultant. No new
global convergence or boundary-crossing rule is inferred.

The shared
[`certify_relational_reflected_equilibrium`](../../src/tnfr/physics/relational_reflected_equilibria.py)
encloses these ideal families and reports exact-definition provenance,
stiffness intervals and full-network stability for `e>0`. The opposite branch
uses a bounded rational cubic isolation and the shared certified cosine to
enclose its angle. It accepts named analytic states, not graphs recognized
as equilibria by a tolerance. [Independent controls](../../tests/physics/test_relational_reflected_equilibria.py)
check the full Hessian and the existing native tangent without evolving a
trajectory. Formation, autonomous support and physical identification remain
separate from the existence and local recovery of these supplied geometries.

## 23. Finite-time zero-resultant access in the reflected law

<a id="reflected-boundary-exit"></a>

Section 22's missing uniform-regularity premise cannot be deduced from the
central component and `E<7*beta`. This section proves an exact counterexample
by local existence; it does not compute or re-evaluate a trajectory.
Keep the same two-C5 support, unit held capacities, `e>=0,w,beta>0` and
unforced regular law. Choose `rho>0` and define the boundary state

\[
y_*=(p,r,P,R,a,b,A,B)=(0,\rho,0,0,0,\pi/2,0,0).
\]

In the established row order `(0,4,3,5,9,8)`, its resultants are
`(2+i,-2i,0,3,2,2)`. Node 3 alone is singular. Put
`h=atan(1/2)/pi>0` and `c=w*rho/beta>0`.
The four noncentral coefficients actually consumed by the eight reduced
equations remain smooth near this point. Their algebra gives the auxiliary
limiting vector

\[
\widetilde F(y_*)=
(e\rho/3+wh,\ -e\rho-w/2,\ 0,\ 0,\ -ch,\ c/2,\ 0,\ 0).
\]

This auxiliary field is a proof construction, not an extension of the
admitted full engine field. In particular no value is assigned to the
undefined central metric. The central form and phase rows are identically
zero along every regular reflected state, which is why they are absent from
the eight consumed rows.

### Transverse arrival from an admitted open time interval

Local existence for the auxiliary smooth field gives a two-sided solution
`gamma(t)` with `gamma(0)=y_*`. Its first-order phase terms are

\[
a(t)=-ch\,t+O(t^2),\qquad
b(t)=\pi/2+ct/2+O(t^2),\qquad A(t),B(t)=O(t^2).
\]

For all sufficiently small negative `t`, both `b<pi/2` and
`b-a<pi/2`. The remaining gaps are near zero. Every edge is therefore
strictly acute, and all six full resultants are regular. On that time
interval the auxiliary equations equal the native complete law exactly.
Selecting any time in this interval as an initial time gives a valid
trajectory that reaches the zero resultant in finite future time.
No fixed initial sample or quantitative exit time is asserted.

The derived resultant kinematics give, at the endpoint,

\[
\dot z=\bigl(-c(h+1/2)+3ich,\ -c(h+1),\ -c,\ -ich,\ 0,\ 0\bigr).
\]

In particular `z_3'= -c<0`, a transverse arrival at zero along the real
axis. The limiting storage and loss are

\[
E_*=2\rho^2+4\beta,\qquad \mathcal L_*=14e\rho^2/3.
\]

For `rho^2<3*beta/2`, a sufficiently short preceding segment retains
`E<7*beta` and `|(a+A)/2|<2*pi/3`. At the reference coefficients
`e=w=1/2, beta=rho=1`, the limiting storage is exactly six and
`z_3'=-1/2`. This disproves forward regularity based on that component and
energy alone, even with acute initial edges and positive dissipation.
It leaves the frozen uniform-receiver initial-value problem undecided.

### The full-state singularity is not removable

Approach the same endpoint with `a=A=B=0`, `b<pi/2` tending to `pi/2`,
the same other forms, and central form `x_3=k*cos(b)`. Then

\[
q_3=2k\cos b,\quad z_3=2\cos b>0,\quad H_3=2\pi\cos b,\qquad
\dot\theta_3=\frac{wk}{\beta\pi}.
\]

Every fixed real `k` gives the same limiting full state, but a different
limiting phase rate. The other rows have identical limits. These approaches
already lie in the positive-resultant chamber; no branch wrapping is needed
to produce the ambiguity. Taking `x_3=sqrt(cos b)` even makes that rate
unbounded at the same limit. Hence the full vector field has no continuous
extension at this point. Cancelling the symmetric zero row would select an
extra boundary convention, not derive a globally defined law.

This is a restriction on this constitutive completion, not on the nodal
identity or every possible TNFR law. A regular chart change cannot remove
the full-state obstruction. Any revised pressure, exchange law, state or
generalized solution convention needs its own F1-F4 admission and must retain
valid interior results with their original hypotheses.

### Static engine evidence and preserved execution boundary

[`certify_relational_reflected_boundary_exit`](../../src/tnfr/physics/relational_reflected_boundary.py)
encloses the named ideal boundary, limiting consumed rates and derivative
signs. It reuses the reflected rate algebra without relaxing the full
evaluator's six-resultant admission. Its statement is local-flow existence,
not a certified response from supplied initial data or permission to step
across zero. [Controls](../../tests/physics/test_relational_reflected_boundary.py)
check those formulas and independent full-state limiting paths.
The [general domain theorem](RELATIONAL_DOMAIN_AND_CAPTURE.md#regular-domain-continuation-and-boundary-access)
separately protects sufficiently small total storage and distinguishes
zero-resultant collisions from nonzero negative-real branch collisions.
