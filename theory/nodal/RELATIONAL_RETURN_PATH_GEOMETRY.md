# Return-path geometry and collective response

Native return-path equilibrium and causal oscillatory response, with separately declared sine/cubic storage-family geometry and its inverse response.

Part of [Native pattern reduction and memory](RELATIONAL_PATTERN_MEMORY.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="mediated-geometric-boundary"></a>
## 11. Return-path geometry and the formation boundary

The present 11-node, 12-edge graph has cycle rank two: its independent cycles
are the two supplied C5 rings. Both edges along the mediator path are graph
bridges. At an equilibrium, sum `grad(V_phi)=0` over one side of either
bridge. Internal sine terms cancel and the only remaining term is the sine
of its phase gap. Strict acute admission then forces that gap to be zero.
Thus the current equilibrium has inherited ring periods and aligned connecting
phases, but no additional independent inter-region circulation period.

This does not exclude a composite NFR or stable binding on this topology.
An additional cycle is only one possible geometric discriminator, not a
necessary condition for every collective relation. Conversely, repeating
the local recovery test cannot manufacture that extra cycle invariant.

A supplied return path changes the cycle space. For example, adding
`1--6` to the current graph supplies a five-edge mixed cycle, whereas the
older two-direct-bridge geometry has a four-edge mixed cycle. Strictly acute
gaps force the latter's integer period to zero; a longer cycle merely removes
that elementary length obstruction. It does not prove a nonzero period is
compatible with the two existing ring periods or with the critical sine
balance. The following admission solves those simultaneous constraints without
a trajectory search. Supplying a return path still does not explain the origin
or timing of a primitive edge.

<a id="return-path-equilibrium"></a>
### 11.1 Complete state, support and equilibrium equations

Keep both oriented C5 rings `0->1->2->3->4->0` and `5->6->7->8->9->5`,
the path `0->10->5`, and add the supplied unit edge `1--6`. This connected
11-node, 13-edge support has cycle rank three. Use the same relational law,
held strictly positive capacities, `e,w,beta>0`, no forcing or events, and
strictly acute gaps. These are constitutive and support premises, not a
derived rule for creating the return edge.

The phase row at equilibrium implies `Bx=0`, hence uniform form. The form
row then requires `g=0`. In the acute domain, every neighbor resultant has
positive real part relative to its node; therefore `g=0` is equivalent to
zero neighbor sine sum. This reuses the algebraic circulation and integral
period criterion of [support balance, Section 30](FORCED_PHASE_LOCKING.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods).
Its separately supplied sine phase evolution is **not** imported into this
relational model. Positive capacities affect recovery rates, not these
equilibrium equations.

Choose the two ring cycles and the mixed cycle `0->10->5->6->1->0`.
They form an integer cycle basis: the edges `1->2`, `6->7` and `0->10`
give an identity submatrix, and removing them leaves a spanning tree.
Write their sine-circulation coordinates as `a,b,q`. The special ring edges
`0->1` and `5->6` carry `a-q` and `b+q`; the other four edges of each ring
carry `a` and `b`. The three connecting edges, oriented `0->10`, `10->5`
and `6->1`, carry `q`. Thus the necessary and sufficient equations are

\[
\begin{aligned}
\arcsin(a-q)+4\arcsin a&=2\pi,\\
\arcsin(b+q)+4\arcsin b&=2\pi\sigma,\\
3\arcsin q+\arcsin(b+q)-\arcsin(a-q)&=2\pi m,
\end{aligned}
\]

where `sigma=+1` or `-1`, `m` is an integer, all currents have magnitude
strictly below one, and arcsines use their acute principal branch.
The possible mixed period must be solved jointly, not assigned by cycle length.

### 11.2 Existence, uniqueness and the forced zero mixed period

For a positive ring period, write its special angle as `t`. The period equation
forces `0<t<pi/2`, bulk angle `A=pi/2-t/4`, and bulk sine `cos(t/4)`.
Define

\[
Q(t)=\cos(t/4)-\sin t,\qquad
Q'(t)=-\tfrac14\sin(t/4)-\cos t<0.
\]

Its range is `(-d,1)`, where `d=1-cos(pi/8)<1/2`.

**Equal ring periods `(1,1)`.** If the other special angle is `s`, then
`q=Q(t)=-Q(s)`, so `abs(q)<d`. The mixed angle sum
`3*arcsin(q)+s-t` has magnitude below `pi/2+3*arcsin(d)<pi`.
Consequently `m=0`. For nonzero `q`, monotonicity makes `s-t` have the
same sign as `q`, so their sum cannot vanish. The unique solution is

\[
q=0,\qquad t=s=2\pi/5,\qquad a=b=\sin(2\pi/5).
\]

Both original twists survive unchanged; all three connecting gaps vanish.

**Opposite ring periods `(1,-1)`.** Write the right special angle as `-s`.
Then `q=Q(t)=Q(s)`, hence `s=t` and `b=-a`. The mixed sum
`3*arcsin(Q(t))-2*t` lies strictly between `-3*pi/2` and `3*pi/2`.
Again only `m=0` is possible. Therefore the connecting angle is `u=2*t/3`
and the remaining scalar equation is

\[
\boxed{F(t)=\cos(t/4)-\sin t-\sin(2t/3)=0.}
\]

Here `F'(t)=-sin(t/4)/4-cos(t)-(2/3)*cos(2t/3)<0` throughout
`(0,pi/2)`. Its signs bracket a unique root `t_*`:

\[
\pi/6<t_*<\pi/4.
\]

For an elementary endpoint proof, `pi<4`, `cos(z)>=1-z*z/2` and
`sin(z)<z` give `F(pi/6)>71/72-1/2-4/9=1/24>0`.
At the other endpoint, `F(pi/4)<1-1/sqrt(2)-1/2<0`.
Continuity and strict monotonicity establish existence and uniqueness
independently of a numerical solver.

Set `u=2*t_*/3` and `A=pi/2-t_*/4`. A nodal reconstruction, modulo `2*pi`, is

\[
\begin{aligned}
\theta_L&=(0,t_*,t_*+A,t_*+2A,t_*+3A),\\
\theta_R&=(2u,2u-t_*,2u-t_*-A,2u-t_*-2A,2u-t_*-3A),\\
\theta_{10}&=u.
\end{aligned}
\]

It has periods `(1,-1,0)` and nonzero connecting sine circulation
`q=sin(u)>0`. The largest absolute gap is `A`, so its minimum acute margin
is exactly `t_*/4`, between `pi/24` and `pi/16`. A zero mixed period
therefore does not imply zero connecting circulation.

### 11.3 Recovery, observable distinction and storage cost

For either exact equilibrium, the existing
[local recovery theorem](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
applies: a sufficiently small joint form/phase perturbation recovers
exponentially modulo one common form offset and one common phase rotation.
There are twenty stable quotient directions. The statement admits unequal
strictly positive held capacities. It supplies neither a quantified basin
here nor a capture certificate for attachment from the previous support.

The independent reflection of one ring used in Section 9 is no longer a
symmetry: it sends the supplied return edge to an absent edge. Only the
identity and whole-ring exchange preserve this graph. In particular, the
mediator is its unique degree-two node adjacent to two degree-three nodes;
fixing it fixes or swaps both ring ports, leaving no independent reflection.
The connecting gap distinguishes the two equilibria: zero versus `2*t_*/3`.
This is an actual geometric distinction, not merely the loss of a symmetry
argument. Neither equilibrium has nonzero pressure or nodal rates. Its sine
circulation is an algebraic balance quantity, not dissipative transport or
an identified magnetic current.

Connection interpreted as a causally shared pulse remains a dynamical
hypothesis. These equilibria provide reference geometries, not persistent
oscillations. The existing [pulse boundary](RELATIONAL_RESPONSE_IDENTIFICATION.md#relational-pulse-scope)
distinguishes damped collective motion from a nontrivial recurrent full state
under strict unforced dissipation. Moreover, freezing the mediator on this
new support leaves the direct return edge active: the null of Section 10
cannot be transferred to this graph as a no-influence control.

Their phase storage is

\[
\begin{aligned}
V_+&=10(1-\cos(2\pi/5)),\\
V_-&=8(1-\sin(t_*/4))+2(1-\cos t_*)+3(1-\cos(2t_*/3)),\\
V_-&>V_+.
\end{aligned}
\]

The strict inequality needs no numerical fit: `1-cos(z)` is strictly convex
on the acute interval. Each ring's five signed gaps have fixed sum `+2*pi`
or `-2*pi`, so Jensen's inequality bounds its storage below by that of the
uniform twist. The opposite solution has nonuniform ring gaps and positive
connecting storage. This comparison does not select a winding or imply that
the dynamics can change sectors while remaining acute.

It also gives a **formation obstruction**. The exact opposite-winding
equilibrium on the original mediator-only support has uniform form and storage
`beta*V_+`. A zero-supply passive event, including a simultaneous nodal reset,
followed by unforced fixed-support relational evolution cannot converge to
the new opposite equilibrium with greater storage `beta*V_-`. Even immediate
formation of that target requires excess initial storage or declared supply
of at least `beta*(V_--V_+)`; dissipation adds its own nonnegative cost.
An available budget is only necessary, not a selector or a sufficient
capture condition. Passivity is an additional event premise throughout.
For equal windings, inserting the zero-gap return edge is storage-neutral
and preserves the old equilibrium, but no occurrence time follows.
Simply adding the edge to the old opposite equilibrium fails the
acute and positive-resultant options: its gap is `-4*pi/5`, and each new port resultant
has relative real part `2*cos(2*pi/5)+cos(4*pi/5)=(sqrt(5)-3)/4<0`.
The nonzero imaginary parts instead permit the full regular law and its
explicit regular-domain executor. This removes that particular chart
restriction, not the preceding passive-storage obstruction to the final target.

### 11.4 Static computational contract

The [return-geometry instrument](../../benchmarks/relational_return_geometry.py)
reuses the engine's exact cycle reconstruction, one-chord topology extension,
rational interval trigonometry and detached relational field. It encloses
the scalar root in turns, with certified opposite endpoint signs, and
constructs a rational midpoint witness for represented native checks.
Exact periods and strict acute admission hold for that witness; its sine
balance remains explicitly unresolved by the rational reconstruction owner.
The exact equilibrium belongs to the analytic root above. Small binary64
rates at the midpoint are reported residuals, not a zero-rate proof.
With forty bisections, the exact turn bracket is
`[2619154484885/26388279066624, 436525747481/4398046511104]`.
Its midpoint gives approximately `t=0.6236341875541443` radians,
connecting gap `0.4157561250360962` and acute margin `0.1559085468885361`.
The ideal storage-excess interval is contained in `[0.47999183896,0.47999183898]`;
these are structural units with `beta=1`, not laboratory energy measurements.
The instrument also reports native midpoint residuals without subtracting
represented drift. Run it with `python -m benchmarks.relational_return_geometry`;
the default preparation fixes unit capacities, `e=w=1/2`, `beta=1`, ascending
node order and no randomness. The analytic equilibrium and storage comparison
retain the wider parameter hypotheses above.
The [static controls](../../tests/physics/test_relational_return_geometry.py)
also reject simply copying the old opposite twists onto the added edge.
No trajectory, automatic connection, new pressure law or physical bridge
is introduced by this instrument.

<a id="return-path-storage-dependence"></a>

### 11.5 Coupled-cycle geometry changes with admitted phase storage

This section changes the complete law explicitly. Sections 11.1-11.4 above
retain their native pressure, positive loss and original evidence. Here reuse
only their supplied eleven-node support, cycle basis and geometric
reconstruction. Choose the
[sine/cubic storage family](SINE_CONSTITUTIVE_INFORMATION.md#phase-storage-selection-boundary),
unit capacities, `e=0,w=beta=1`, no inputs or events, and `tau=t/pi`:

\[
x'=K S_\epsilon(\theta),\qquad \theta'=KLx,\qquad
K=\operatorname{diag}(1/d_i),\qquad
S_{\epsilon,i}=\sum_{j\sim i}j_\epsilon(\theta_j-\theta_i),
\]

\[
j_\epsilon(\delta)=\sin\delta+\epsilon\sin^3\delta,
\qquad U_\epsilon'=j_\epsilon,\quad U_\epsilon(0)=0,\qquad
E_\epsilon=\tfrac12x^{\mathsf T}Lx+\sum_e U_\epsilon(\delta_e),
\qquad E_\epsilon'=0.
\]

Each finite `epsilon>=0` specifies one law with the same clock, held support
and capacity. The prescribed sector is `(1,-1,0)` in the two oriented ring
cycles and the mixed cycle `0->10->5->6->1->0`. Changing storage does not
select these periods or create the return edge.

#### The scalar equation describes every acute critical state in this sector

At any equilibrium, the positive phase mobility implies uniform form and
the form row then requires zero current divergence. On the acute interval,

\[
j_\epsilon'(\delta)=\cos\delta(1+3\epsilon\sin^2\delta)>0.
\]

Consecutive bulk edges along each degree-two path therefore have equal
gaps. Write the left special gap as `v` and the right one as `-s`. Their
ring periods and strict acuity force `v,s in (0,pi/2)` and bulk magnitudes
`A(v)=pi/2-v/4`, `A(s)=pi/2-s/4`. The connecting current is both
`Q_epsilon(v)` and `Q_epsilon(s)`, where

\[
Q_\epsilon(v)=j_\epsilon(A(v))-j_\epsilon(v),\qquad
Q_\epsilon'(v)=-\tfrac14j_\epsilon'(A(v))-j_\epsilon'(v)<0.
\]

Hence `v=s`. All three oriented connecting edges have the same current,
so strict monotonicity gives a common gap `u`. The zero mixed period forces
`3u-2v=0`. Thus every acute critical geometry in this sector must satisfy

\[
\boxed{F_\epsilon(v):=
j_\epsilon(\pi/2-v/4)-j_\epsilon(v)-j_\epsilon(2v/3)=0.}
\]

This is not a restriction to a guessed reflection-symmetric ansatz: the
equal gaps follow from full nodal balance, the supplied periods and acute
current injectivity. Conversely the reconstruction below gives a complete
critical state from any root.

#### Existence, uniqueness and constitutive dependence for every finite coefficient

Put `a=cos(v/4)`, `b=sin(v)` and `c=sin(2v/3)`, so
`F_epsilon=F_0+epsilon*F_3`, where `F_0=a-b-c` and
`F_3=a^3-b^3-c^3`. The scalar derivative is strictly negative on
`0<v<pi/2`:

\[
\partial_vF_\epsilon=
-\tfrac14j_\epsilon'(\pi/2-v/4)
-j_\epsilon'(v)-\tfrac23j_\epsilon'(2v/3)<0.
\]

At `v=pi/6`, the elementary bounds used above give
`a>71/72` and `b+c<17/18`, hence `F_0>1/24` and `F_3>0`.
At `v=2*pi/5`, the bulk and special angles coincide, so
`F_epsilon=-j_epsilon(4*pi/15)<0`. Continuity and strict monotonicity
therefore prove exactly one root `v_epsilon`, with the uniform bracket

\[
\boxed{\pi/6<v_\epsilon<2\pi/5\qquad(\epsilon\ge0).}
\]

At zero coefficient this is the retained root `t_*` from Section 11.2,
whose sharper bracket `pi/6<t_*<pi/4` remains valid. That sharper upper
endpoint is not a bound asserted for the whole storage family.

The root varies smoothly, and strictly increases, for all finite
`epsilon>=0`. At a root, `F_3` must be positive: otherwise
`a< b+c` follows from `a^3<=b^3+c^3` and `b,c>0`, making
`F_0+epsilon*F_3<0`. Implicit differentiation now gives

\[
\boxed{\frac{dv_\epsilon}{d\epsilon}
=\frac{F_3(v_\epsilon)}{
\tfrac14j_\epsilon'(A(v_\epsilon))+
j_\epsilon'(v_\epsilon)+\tfrac23j_\epsilon'(2v_\epsilon/3)}>0.}
\]

This distinguishes actual critical geometries, rather than only changing a
response around an unchanged pattern. Their bulk gaps decrease while the
special and connecting gaps increase. In particular, for `epsilon>0`,
`F_0(v_epsilon)=-epsilon*F_3(v_epsilon)<0`: the new geometry is not an
equilibrium of the unchanged sine or native law.

For interval arithmetic, the same root equation can be evaluated as
`(1-lambda)*F_0+lambda*F_3=0`, where
`lambda=epsilon/(1+epsilon)`. This is multiplication of the stationary
equation by a positive number. It does not rescale one dynamical row,
change the structural clock, or install an additional parameter or limiting
constitutive law.

#### Reconstruction, identity and local protection

Set `v=v_epsilon`, `u=2v/3` and `A=pi/2-v/4`. The same nodal reconstruction
as before gives

\[
\theta_L=(0,v,v+A,v+2A,v+3A),\qquad
\theta_R=(2u,2u-v,2u-v-A,2u-v-2A,2u-v-3A),\qquad
\theta_{10}=u,
\]

with phases taken modulo `2*pi` and uniform form. The oriented cycle periods
are exactly `(1,-1,0)`. Its bulk current is `a_epsilon=j_epsilon(A)`,
right bulk current is `-a_epsilon`, and all three connecting currents are
`q_epsilon=j_epsilon(u)>0`. The equation
`a_epsilon=j_epsilon(v)+q_epsilon` verifies every port balance; internal
path nodes cancel pairwise. These are balanced stationary edge currents,
not a claim of a persistent oscillation.

Because `v<2*pi/5`, the largest gap is `A>v>u`. Its acute margin is
`pi/2-A=v/4>pi/24`, uniformly over the family. The full phase Hessian
`Z diag(j_epsilon'(delta_e)) Z^T` is therefore positive definite modulo
common rotation. Together with the form Hessian `L`, it gives a strict local
minimum of the conserved storage on the relative-state quotient. Its target
value is

\[
E_\epsilon^*=8U_\epsilon(A)+2U_\epsilon(v)+3U_\epsilon(u).
\]

The [local conserved-storage argument](RESONANCE_FOUNDATIONS.md#storage-family-pattern-robustness)
then supplies nonlinear Lyapunov protection: for each fixed coefficient,
sufficiently small joint perturbations remain near this geometry, after
retaining the degree-weighted form mean and a consistent local lifted phase
mean. The neighborhood and excess storage must use this law and this target.
No numerical basin or acceptance of a captured preparation follows merely
from a positive Hessian. This is local protection under zero loss, not the
native positive-loss attraction result in Section 11.3 and not formation
from an initial state outside the sector.

The [cycle-geometry owner](../../src/tnfr/physics/phase_cycle_geometry.py)
provides `assess_return_path_storage_geometry` and its
`ReturnPathStorageGeometryAssessment` with a `PhaseRootBracket`. It retains
the supplied support and coefficient while enclosing the scalar root and
reconstructing target geometry. A rational midpoint is an approximate target
with a residual, not an exactly critical state; source association does not
mean the captured network occupies this target or follows the alternative
law. Exact periods, root enclosures and nonlinear evolution remain different
kinds of evidence.

The transferable result is a persistent winding identity with a uniquely
deforming acute critical geometry under this explicit storage freedom.
Neither topology alone nor the common consensus tangent fixes that geometry.
The selected support, winding preparation and constitutive family remain
premises; no event, universal phase potential or physical identity is derived.

<a id="return-path-geometry-response"></a>

### 11.6 A geometric interval constrains a law and a separate nodal response

Retain exactly the support, periods, unit capacities, conservative
`j_epsilon`, `w=beta=1` and `tau=t/pi` of Section 11.5. The new input is an
independently supplied closed interval for the special edge angle `v`,
including its origin and uncertainty. It is a constraint on an equilibrium
of this supplied family, not an assertion that an arbitrary captured state
is at equilibrium. No response, pressure reconstructed from a response, or
root midpoint is used to choose the coefficient.

#### The inverse exists on a proper, half-open geometric branch

Write `v_0` for the unique zero of `F_0` and `v_infinity` for the unique
zero of `F_3` in `(pi/6,2*pi/5)`. The latter root exists because `F_3` is
positive at `pi/6`, negative at `2*pi/5`, and strictly decreasing:

\[
F_3'(v)=-\tfrac34\cos^2(v/4)\sin(v/4)
        -3\sin^2(v)\cos(v)
        -2\sin^2(2v/3)\cos(2v/3)<0.
\]

At `v_0`, the identity `a=b+c` gives `F_3(v_0)=3bc(b+c)>0`, so
`v_0<v_infinity`. At `v_infinity`, `a^3=b^3+c^3` with positive `b,c`
implies `a<b+c` and hence `F_0(v_infinity)<0`. It follows that

\[
\boxed{\epsilon(v)=-\frac{F_0(v)}{F_3(v)},
       \qquad v\in[v_0,v_\infty)}
\]

is nonnegative, continuous, zero only at `v_0`, and unbounded as
`v` approaches `v_infinity` from below. On that interval,

\[
\frac{d\epsilon}{dv}
 =-\frac{F_0'(v)+\epsilon(v)F_3'(v)}{F_3(v)}>0.
\]

Thus it is a bijection from `[v_0,v_infinity)` to all finite
`epsilon>=0`. In particular, the limiting angle is not an equilibrium
of any finite member of the family. The equation `F_3=0` describes a
geometric limit; it does not silently install an infinite coefficient or
a rescaled pure-cubic time-evolution law.

For an exact observation interval `I=[v_lo,v_hi]`, intersect it with this
half-open branch before inferring any coefficient. The compatible set is
empty if `v_hi<v_0` or `v_lo>=v_infinity`. Otherwise its lower endpoint is
zero when `v_lo<=v_0`, and `epsilon(v_lo)` when `v_lo>v_0`. Its upper
endpoint is `epsilon(v_hi)` if `v_hi<v_infinity`; if
`v_hi>=v_infinity`, the set is unbounded above. Equality at `v_0` is
admissible, whereas equality at `v_infinity` is excluded for every finite
law. These distinctions cannot be replaced by clipping the coefficient to
a convenient numerical search range.

In turn coordinates `s=v/(2*pi)`, the same statements use the prior
enclosure `(1/12,1/5)` and the two roots in that interval. A numerical
reader can certify endpoint signs with outward intervals. A sign interval
containing zero supplies neither a strict comparison nor an exact root
equality; a finite arithmetic budget may therefore leave classification
unresolved. This is an arithmetic limitation, distinct from an incompatible
geometric observation.

#### Finite geometric accuracy does not imply uniform coefficient accuracy

Let

\[
C_\infty=\frac{F_0(v_\infty)}{F_3'(v_\infty)}>0.
\]

Taylor expansion at the simple zero of `F_3` gives

\[
\epsilon(v)=\frac{C_\infty}{v_\infty-v}+O(1),\qquad
v_\infty-v_\epsilon=\frac{C_\infty}{\epsilon}
                     +O(\epsilon^{-2}),
\]
\[
\frac{d\epsilon}{dv}(v_\epsilon)
       \sim\frac{\epsilon^2}{C_\infty}.
\]

The geometry saturates while the unscaled law becomes stronger. A fixed
angle error therefore becomes increasingly uninformative about the
coefficient near the limiting root. If the geometric interval reaches that
limit, the absence of a finite upper coefficient bound is intrinsic to the
observation, even with exact arithmetic. This does not invalidate the
forward equilibrium or its local stability for any fixed finite coefficient.

#### The full nodal response probes curvature rather than only current balance

Orient the thirteen edges consistently with the shared geometry owner, and
let `Z` be the incidence matrix with `-1` at the tail and `+1` at the
head. At the exact equilibrium reconstructed from `v`, define

\[
H_\epsilon(v)=Z\operatorname{diag}
 \left[\cos\delta_e(v)
       \left(1+3\epsilon\sin^2\delta_e(v)\right)\right]Z^{\mathsf T}.
\]

Here `S_epsilon=-Z j_epsilon(Z^T theta)` and its phase derivative is
`-H_epsilon`; fixed integer lift offsets do not change the derivative.
Every edge curvature is positive on the admitted acute branch. With
`y=delta x` and `eta=delta theta`, the complete twenty-two-coordinate
linearization in the declared clock is

\[
\begin{pmatrix}y'\\\eta'\end{pmatrix}
 =\begin{pmatrix}0&-K H_\epsilon\\KL&0\end{pmatrix}
  \begin{pmatrix}y\\\eta\end{pmatrix}.
\]

For a prescribed form preparation `y(0)=h`, `eta(0)=0`, it follows that

\[
y'(0)=0,\qquad \eta'(0)=KLh,\qquad
\boxed{y''(0)=-K H_\epsilon(v) K Lh.}
\]

The first zero form rate does not imply equilibrium: phase initially moves
and subsequently changes the form-driving current. The new observation is
this full nodal acceleration, in units of form per `tau` squared. The
geometric constraint uses `j_epsilon`; the reserved response tests
`j_epsilon'` together with the complete support and mobility.

For the specified mediator preparation `h=e_10`, set

\[
c_A=j_\epsilon'(A),\quad c_v=j_\epsilon'(v),\quad
c_u=j_\epsilon'(u),\qquad B=\frac{c_A+c_v+4c_u}{9}.
\]

The actual degrees give `(KLh)_10=1`, `(KLh)_0=(KLh)_5=-1/3`, with
all other coordinates zero. Therefore, in node order `0,...,10`,

\[
\boxed{y''(0)=
 (B,-c_v/9,0,0,-c_A/6,
  B,-c_v/9,0,0,-c_A/6,-4c_u/3).}
\]

This expression contains all three geometric curvature classes. Its
degree-weighted sum is exactly zero, as required by conservation of
`sum_i d_i*x_i`. No projection of the mediator preparation onto a chosen
mode is required. For the full nonlinear preparation
`x(0)=x_*+alpha*h`, `theta(0)=theta_*`, its initial second form derivative
is exactly `alpha` times the displayed expression. That local derivative
identity still gives no certified finite-time or finite-amplitude trajectory.

#### Outward bounds retain the same implicit geometry and law

Suppose the geometric observation implies a finite coefficient enclosure
`[epsilon_lo,epsilon_hi]`. For every compatible angle retain the exact
relations `A=pi/2-v/4`, `u=2v/3` and the nodal affine reconstruction from
Section 11.5. Enclose the three edge curvatures by outward trigonometric
and arithmetic evaluation over this set, and propagate them through the
fixed matrix product `-K H K Lh`. This yields a componentwise enclosure
of the complete acceleration for every compatible equilibrium and
coefficient. Separate angle and coefficient intervals can widen the answer
by discarding correlations for interval evaluation, but they cannot create
additional admitted equilibria. In particular, choosing unrelated angles
from different edge boxes is not a new geometry.

Endpoint evaluation of the curvatures alone is insufficient without a
monotonicity proof for those complete expressions. The inverse coefficient
map is monotone; this does not imply monotonicity of every product of
`cos(delta)`, `sin(delta)^2` and the changing coefficient. Strict endpoint
sign bounds also remain necessary before dividing by an enclosure of `F_3`.

An unbounded compatible coefficient set does not generally permit a finite
response bound. For this mediator preparation,
`c_u=cos(2v/3)*(1+3*epsilon*sin(2v/3)^2)` tends to positive infinity
as `v` approaches `v_infinity`. The mediator acceleration consequently
tends to negative infinity. A finite enclosure cannot be manufactured from
the geometric limit. Special preparations such as constant form may still
have an identically zero response and must be distinguished from this case.

The shared [phase-cycle owner](../../src/tnfr/physics/phase_cycle_geometry.py)
exposes `assess_return_path_geometry_response` and
`ReturnPathGeometryResponseAssessment`. It re-admits the source premises,
two ordered cycles, mediator, supplied `special_turn_bounds`,
`form_direction` and `observation_origin`. Its coefficient status separates
`bounded`, `unbounded_above`, `incompatible` and
`unresolved_interval_arithmetic`; a missing finite bound is not a point fit.
Only a bounded compatible input supplies the reader's finite full response
enclosure. The supplied observation-origin label is provenance metadata,
not independent authentication or physical validation. The captured source
association does not certify that its stored phases occupy the inferred
equilibrium.

#### Geometry does not determine the clock or the missing dynamical scales

Multiplying every phase current by a positive constant leaves the zero
divergence condition and equilibrium geometry unchanged. More generally,
the supplied laws `x'=a*K*S_epsilon`, `theta'=b*K*Lx`, with independent
positive `a,b`, have the same equilibria and replace the predicted
acceleration by `-a*b*K*H_epsilon*K*Lh`. Positive held capacities also
affect `K` without changing this zero-current equilibrium condition.
Changing from `tau` to `r*tau` multiplies accelerations by `r^-2`.
The angle observation cannot recover these freedoms.

Consequently the result is conditional on the independently fixed clock,
capacities, current normalization and phase mobility. A frozen mathematical
control can infer the coefficient interval from geometry, then compare a
separately evaluated full nodal response without changing that interval.
Success checks joint consistency and the implementation of this declared
family. It does not select the family among arbitrary potentials, infer
microscopic formation, or validate a physical correspondence.

#### Frozen known-source control

The [declaration](../../docs/assets/return_path_geometry_response/declaration.json)
fixes coefficient `1` for a mathematical source, a `10^-6` turn observation
grid, the mediator form direction, all eleven response coordinates and
95-decimal independent arithmetic. The source coefficient is excluded from
the predictor's inputs. The independently quantized special angle is
`s in [0.137625,0.137626]`; strict outward signs of the source root equation
certify that this interval contains its exact equilibrium angle.

Separate producer invocations retained the
[source protocol](../../docs/assets/return_path_geometry_response/response-v1.protocol.json),
[prediction](../../docs/assets/return_path_geometry_response/response-v1.prediction.json)
and [response](../../docs/assets/return_path_geometry_response/response-v1.json).
The exact predicted coefficient bounds are contained in the conservative
decimal display `[0.99999715,1.00005518]`. Thus geometry excludes the sine
baseline within this family. All eleven independently differentiated nodal
responses lie in their saved intervals. For example the mediator acceleration
is approximately `-2.1142183971`, inside the predicted interval, whose outward
decimal display is `[-2.11428951,-2.11420582]` in form per `tau` squared.
The coefficient was not refitted after evaluating that response.

The [producer](../../benchmarks/return_path_geometry_response.py) reuses the
existing write-once evidence transport and retains a
[source archive](../../docs/assets/return_path_geometry_response/response-v1.source.zip).
It re-admits the frozen geometric premises before consuming the prediction;
hashes alone do not authenticate its mathematical content or chronology.
The independent high-precision response is a finite cross-check, not an
interval-certified oracle or physical measurement. The enclosure proof
above and the finite implementation check have different evidential roles.
The bounded geometry-to-response gate is complete; coefficient sweeps or
finite-amplitude forecasts are not required by this result.

<a id="shared-collective-pulse"></a>
## 12. A causally shared oscillatory response, with a dissipation limit

**Question and scope.** Can the admitted form/phase dynamics itself generate
an oscillatory response shared by the two regions? Here a local collective
oscillation means a nonreal mode of the native equilibrium derivative that
contributes to a donor-to-receiver response. Matching frequencies or an
eigenvector drawn across both rings is insufficient. A maintained pulse would
add a separate persistence obligation. Neither meaning is a definition of
every possible NFR or a physical identification.

Fix the native opposite-winding equilibrium of Sections 11.1-11.3, unit
capacities, `e=w=1/2`, `beta=1`, its existing structural clock and no inputs or
events.
The graph, constitutive law and preparation remain explicit premises.
An initial donor perturbation probes the response; it is not a periodic
forcing or an autonomous explanation of how that perturbation arose.

### 12.1 The oscillatory mechanism uses the existing two rows

Let `B` be the unweighted Laplacian, `D` the degree diagonal, and `K` the
Laplacian weighted by the equilibrium edge cosines. At this equilibrium
`H_i=pi*sum_j cos(delta_ij)>0`. With `u=delta x`, `v=delta theta`, the
[existing recovery derivative](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
is

\[
\frac{d}{dt}\begin{pmatrix}u\\v\end{pmatrix}
=J\begin{pmatrix}u\\v\end{pmatrix},\qquad
J=\begin{pmatrix}
-eD^{-1}B&-wH^{-1}K\\
(w/\beta)H^{-1}B&0
\end{pmatrix}.
\]

Phase deformation changes pressure and hence form; form contrast drives
phase motion back. This is a derived feedback within the supplied nodal law,
not an additional oscillator equation or a prescribed frequency. Its restoring
and damping rates depend on geometry through `B,K,H,D` and on the declared
model coefficients and clock. The same coefficients can be overdamped on
P2; oscillation is not guaranteed merely by naming a phase coordinate.

The shared engine now provides `evaluate_relational_uniform_tangent` and
`Network.relational_uniform_tangent`. It admits exactly uniform represented
form even away from phase equilibrium, retaining the native field and the
general Arg derivative. At an ideal equilibrium that derivative reduces to
`-H^-1 K`. The observer does not declare equilibrium from a small residual,
repair the numerical common-offset defect, or change the time-evolution law.
Its [contract](../../docs/contracts/relational/RELATIONAL_EXECUTION.md#uniform-form-tangent-observation)
separates materialized coefficients from exact derivatives and spectral proof.

### 12.2 A nonzero transfer is stronger than a shared eigenvector

Let `a` be a declared initial perturbation and `O_R` an observation of the
receiving ring. The Laplace transform of its tangent response is

\[
T_R(s)=O_R(sI-J)^{-1}a.
\]

For a simple eigenvalue, a right vector `v_k` and a left row `l_k` normalized
by `l_k v_k=1` give residue `O_R v_k (l_k a)`. Both preparation and observation
must participate. This expression is invariant to eigenvector rescaling;
using a Hermitian Laplacian projector for this generally nonnormal `J` would
give a different calculation. Degenerate independent oscillators permit
extended eigenvectors while their cross-region transfer is exactly zero.

There is also an exact causal result for the ideal return graph. Order the
two coordinates by node. Every off-diagonal edge block from node `j` to `i` is

\[
J_{ij}=\begin{pmatrix}
e/d_i&w\cos\delta_{ij}/H_i\\
-w/(\beta H_i)&0
\end{pmatrix},\qquad
\det J_{ij}=\frac{w^2\cos\delta_{ij}}{\beta H_i^2}>0.
\]

Suppose a right eigenvector vanishes on the entire receiving ring. Its
equations at nodes 5 and 6 force respectively node 10 and node 1 to vanish;
the equation at 10 then forces node 0. The equations at 0, 1 and 2 force
nodes 4, 2 and 3 in turn. No nonzero eigenvector can be invisible on that
ring. The transpose argument starting with a left eigenvector zero on the
donor forces nodes 10, 6, 5, 9, 7 and 8 to vanish. Thus complete donor
form/phase preparation and complete receiver observation meet the eigenvector
controllability/observability criteria: no mode is absent from the full
donor-to-receiver transfer. This does not assert that every single-node
preparation or scalar observation detects every mode, nor that the mediator
moves in every mode.

For the numerical comparison, fix `a=e_x0` before evaluation. The primary
receiver outputs are `x_5-mean_R(x)` and `v_5-mean_R(v)`. Mediator outputs
are its two **rates**, avoiding a spurious apparent motion from subtracting
a changing common reference. No phase angle derived from a mode is identified
with primitive nodal phase.

The causal null sets cross-region blocks of the full tangent to zero,
retaining its three diagonal blocks for donor, receiver and mediator. This
is an explicit linear ablation with held local coefficients, not a newly
admitted disconnected nonlinear geometry. Its donor-only subspace is
invariant, so receiver response is identically zero for every time and every
resolvent point outside the spectrum, even if eigenvalues are degenerate.
Its offsets need not have the native quotient symmetry; use the complete
22-coordinate null. Merely freezing node 10 leaves edge `1--6` active and
cannot serve as that null.

### 12.3 Certified existence of an ideal damped complex mode

A scalar spectral moment proves presence without certifying individual
numerical eigenvalues. Write `L=D^-1 B` and `M=H^-1`. Block multiplication
at the exact equilibrium gives

\[
\operatorname{tr}J^3=-e^3\operatorname{tr}L^3
 +\frac{3ew^2}{\beta}\operatorname{tr}(LMKMB).
\]

The graph is triangle-free. Its degrees give
`sum_edges 1/(d_i*d_j)=7/3`, so `tr(L^3)=11+6*(7/3)=25`.
Set `r=cos(t_*)`, `z=sin(t_*/4)`, `b=cos(2*t_*/3)` and `s=r+z+b`.
The cosine strengths `s_i=H_i/pi` are `s` on four ports, `2*z` on the
six internal ring nodes and `2*b` on the mediator. Expanding the remaining
trace into diagonal and two-edge walks gives

\[
\begin{aligned}
T:=\pi^2\operatorname{tr}(LMKMB)
 &=\sum_i\frac{d_i}{s_i}
  +\sum_{\{i,j\}\in E}\left(\frac1{d_i s_j}+\frac1{d_j s_i}
                          +\frac{4\cos\delta_{ij}}{s_i s_j}\right)\\
 &=\frac{29}{s}+\frac{38}{3z}+\frac4{3b}
                         +\frac{4(2r+b)}{s^2},\\
\operatorname{tr}J^3&=-\frac{25}{8}+\frac{3T}{8\pi^2}.
\end{aligned}
\]

The exact root bracket of Section 11 and shared outward rational trigonometry
enclose this ideal trace strictly inside `[0.72428869449,0.72428869451]`.
No numerical eigensolver or trajectory enters that sign certificate.
The two common-offset eigenvalues are zero; all twenty quotient eigenvalues
have negative real part by local recovery. If they were all real, the sum
of their cubes would be negative. The certified positive trace contradicts
that possibility. Therefore at least one damped complex-conjugate pair exists
for this exact geometry and model. Section 12.2 further proves that some donor
preparation and receiver observation retain such a pair causally.

This theorem does not count all complex pairs, certify individual frequencies,
or prove that this same pair moves the mediator. Those more specific statements
retain their separate numerical evidence below. The certificate is conditional
on the stated geometry and coefficients, not a universal pulse law for TNFR.

### 12.4 Persistence is a separate claim

All nonneutral modes of the ideal acute equilibrium have negative real part
by the existing recovery theorem. More strongly, the
[storage recurrence argument](RELATIONAL_RESPONSE_IDENTIFICATION.md#relational-pulse-scope)
excludes nonstationary full-state periodic or relative-periodic motion in the
specified regular chamber under strictly positive dissipation and capacity.
These statements do not exclude every recurrent lossy diagnostic or all
other TNFR completions. They do exclude using this model as evidence of a
permanent unforced full-state pulse.

A mode with `lambda=-alpha+i*omega` contributes a damped sinusoid to the
infinitesimal response. Its period `2*pi/abs(omega)` and amplitude decay time
`1/alpha` have units of the declared structural clock. Neither is a calibrated
physical time. A sum of such modes need not visibly complete a cycle; a
nonzero complex residue alone is not a nonlinear limit cycle or phase-locking
theorem. For a fixed finite horizon, smooth dependence supplies the usual
first-order response `epsilon*exp(J*t)*a` about the exact equilibrium, with
a local higher-order remainder; no numerical remainder bound is certified here.

### 12.5 Retained static evidence and numerical boundaries

The [collective-pulse instrument](../../benchmarks/relational_collective_pulse.py)
uses the shared uniform-form tangent and the same forty-bisection rational
midpoint as Section 11. Its
[static record](../../docs/assets/relational_collective_pulse/result.json)
retains the complete state, matrices, declared observations, initial direction,
all poles and residues, controls, root enclosure and ideal trace certificate.
There is no finite-amplitude evolution or fitted frequency in this calculation.

Common form and phase offsets are removed by the exact difference map
`D_0 z=(x_i-x_10, v_i-v_10)`, `i=0,...,9`, with lift `L_0` setting the
mediator coordinates to zero. Here `D_0` is an observation map, distinct from
the degree diagonal `D` above. `D_0 L_0=I`; the ideal law induces
`J_q=D_0 J L_0`. The materialized generator's nonzero row sums and the defect
`D_0 J-J_q D_0` are retained, not silently corrected. Their maximum magnitude
was `4.17e-17`. This numerical quotient is not asserted to close the represented
generator exactly.

The numerical quotient has six conjugate pairs and eight real poles. The
positive-imaginary representatives are listed below; the report retains their
conjugates and the real modes as well. "Unresolved" means the mediator-rate
residue magnitude is below the stated numerical resolution, not a missing
coordinate or a proof of zero motion.

| Real part | Positive imaginary part | Period | Decay time | Mediator-rate response |
| --- | --- | --- | --- | --- |
| -0.438279 | 0.530782 | 11.8376 | 2.28165 | Unresolved |
| -0.438120 | 0.534008 | 11.7661 | 2.28248 | Resolved |
| -0.265023 | 0.269281 | 23.3332 | 3.77326 | Resolved |
| -0.242613 | 0.272205 | 23.0825 | 4.12179 | Unresolved |
| -0.134737 | 0.050636 | 124.0860 | 7.42187 | Resolved |
| -0.045952 | 0.058180 | 107.9960 | 21.76184 | Unresolved |

All six have resolved primary receiver residues for the declared donor form
direction. For example, the `-0.265023+0.269281i` component has receiver
form/phase residue magnitudes approximately `(0.0438520,0.0273921)` and
mediator-rate magnitudes `(0.0195353,0.00876979)`. Its amplitude falls to
`exp(-1)` in only `0.161712` cycles; after one period its modal envelope is
approximately `0.002063` of its initial value. The existence of an oscillatory
component therefore should not be presented as a long train of shared pulses.

Numerical pole and residue zero policies were `5.71e-14` and `6.93e-13`,
respectively, based on binary64 precision, matrix/observation scale and
eigenbasis conditioning. They are resolution policies, not physical thresholds.
The eigenbasis condition number was `24.35`, the smallest distinct-pole
separation `0.00323028`, and the maximum eigenvector residual below `3.44e-15`.
Reassembling the full transfer at the declared `s=1+2i` differed from direct
solution by `1.06e-16`; full and quotient resolvents differed by `1.56e-17`.
These are consistency checks, not rigorous error bounds on individual ideal
poles or residues. The certified trace sign remains separate evidence.

The all-route block-diagonal control has identically zero receiver transfer.
Freezing only the mediator does not: the receiver's second derivative per
unit donor direction is approximately `(-0.00424688,0.00281912)` through the
remaining return route. This is why apparent shared timing and intervention
on just one route cannot settle connection.

Execution used Python 3.13.6, NumPy 2.5.3, NetworkX 3.6.1, Windows AMD64,
binary64 native coefficients and eigensystems, exact rational observation
maps/defects, the shared interval kernel, and no randomness. Listed owner
fingerprints are a **partial** source manifest, not a complete dependency
archive or provenance authentication. The stored matrices permit independent
checks without regenerating the producer. Its
[tests](../../tests/physics/test_relational_collective_pulse.py) also use an
analytic oscillator transfer and disconnected degenerate oscillators as
independent controls; those test matrices are not added to TNFR dynamics.

This closes local pulse admission: oscillatory causal participation follows
under the specified law and geometry, while the individual residues and
strong damping have the stated numerical scope. A visibly oscillating finite
nonlinear response and a sustained pulse remain distinct further claims.
