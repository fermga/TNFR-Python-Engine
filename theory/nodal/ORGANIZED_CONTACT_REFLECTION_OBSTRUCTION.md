# A reflection obstruction for full-fine contact identity

<a id="organized-contact-reflection-obstruction"></a>

## Declared numerical illustration

The proof below is analytic. A separate detached illustration is declared
here before evaluating its algebraic example; it does not discover a
stationary state by fitting a numerical residual.

| Item | Fixed illustration scope |
| --- | --- |
| Algebraic object | The unique zero z of `16*z^3+36*z^2+24*z+1` in the exact rational bracket `(-1/20,0)`, whose existence and uniqueness are proved in Section 4 |
| Enclosure budget | Exactly 128 rational bisections, without adaptive extension, followed by the shared dyadic128 square-root, circular-angle and edge-field arithmetic |
| Complete state | Form zero, internal profile `psi_j=(2*pi/9)*(j-4)`, rim origins `(0,b,a,-a,-b)` and hub zero, with a and b defined by the unit-circle identities in Section 4; preserve the complete law (1) and all 108 coordinates |
| Required observations | Enclose all fifteen principal winding sums and certify winding one away from cuts; audit enclosing stationary residuals while retaining the exact cubic identities as the stationarity proof |
| Instability witness | Center the indicator of the fixed-rim constituent by subtracting its mean `1/6`; certify its full phase quadratic form is strictly negative. Internal edges make zero contribution to this vector |
| Execution boundary | No source assessment, trajectory, eigensolver, scan, parameter tuning or root search beyond enclosure of this single analytically proved cubic root |

No numerical result or passing verdict is asserted by this declaration.
The [research program](../RESEARCH_PROGRAM.md#research-current-question)
owns activation and the next task; this illustration creates no second queue.

## 1. Question, complete model and candidate sector

Can a rearranged constituent profile support a stable stationary outer
identity on the existing joined support? The
[continued-contact owner](ORGANIZED_CONTACT_MAINTENANCE.md) fixes the
54-node unit graph `C9 square W6`, held unit capacities, complete two-row
law, clock and conserved means. This result retains that model:

\[
 x'=(-Lx+\gamma S(\theta))/4,\qquad
 \theta'=\gamma Lx/4,\qquad
 \gamma=1/(1023\pi),
 \tag{1}
\]

where prime denotes `tau=(1023/1024)*t` and
`S_i=sum_{b~i} sin(theta_b-theta_i)`. No coefficient, support edge,
input, event, capacity or preparation is changed.

The marked identity consists of winding one on all six internal C9
cycles and all nine old outer C5 cycles. Each winding is the sum of its
oriented principal gaps divided by `2*pi`. All marked cycle edges stay
away from their cuts at absolute gap pi. Internal and outer gaps may be
nonacute. Spoke edges carry no separately prescribed winding identity.

Choose a reflection R of the five rim vertices of W6. It fixes one rim
vertex f and the hub h, and exchanges two pairs of rim vertices. Write
`psi_j=theta_{h,j}`. The candidate sector requires the circular identities

\[
 \theta_{R(c),j}+\theta_{c,j}=2\psi_j
 \pmod {2\pi}
 \quad\text{for every rim vertex c and slot j}.
 \tag{2}
\]

This is a restriction on proposed stationary states. It retains every
fine coordinate before applying stationarity and the marked windings;
it does not assume a carrier or a prescribed internal profile. Equation
(2) is not asserted to define an invariant sector of the complete flow.
In particular, the independently perturbed acquired source is not assumed
to obey it, and no source trajectory or future source symmetry is inferred.

**Conditional obstruction.** No stationary state in this sector, retaining
all fifteen marked windings, has a positive-semidefinite phase Hessian.
Consequently any such stationary state is unstable for the complete
two-row law on its fixed-mean leaf. This excludes stable stationary
identity in the specified reflection sector, including nonacute charts;
it does not exclude stationary targets that break that reflection.

The argument below is analytic and independent of the declared numerical
illustration. It performs no trajectory, source assessment, parameter scan
or reserved-response evaluation. The illustration supplies implementation
evidence for its fixed example, not the proof of the obstruction.

## 2. Stationarity and identity force slot-independent relative origins

Connectedness and the phase row in (1) imply that stationary form is
common. The form row then requires `S(theta)=0`. Introduce the hub's
internal principal gaps

\[
 \rho_j=\operatorname{wrap}(\psi_{j+1}-\psi_j),\qquad
 \sum_{j=0}^{8}\rho_j=2\pi.
 \tag{3}
\]

Indices are cyclic. Reflection pairs have opposite spoke sines, while
the fixed rim vertex has relative phase either zero or pi and contributes
zero spoke sine. Hub stationarity therefore gives

\[
 \sin\rho_j=\sin\rho_{j-1}=s.
 \tag{4}
\]

Because hub edges avoid the winding cut, `s=0` would make every rho zero;
`s<0` would make every rho negative. Both contradict (3). Thus

\[
 s>0,\qquad 0<\rho_j<\pi\quad\text{for every j}.
 \tag{5}
\]

These conclusions do not require an acute hub. Its gaps need not be
equal: for example, one gap `6*pi/7` and eight gaps `pi/7` have the same
sine and sum to `2*pi`.

### The fixed rim vertex cannot alternate between its two origins

Put `epsilon_j=theta_{f,j}-psi_j` modulo `2*pi`. Equation (2) gives
`epsilon_j` in `{0,pi}`. On an internal edge where epsilon does not change,
the fixed rim gap is rho_j. On an edge where it changes, its principal
gap is `rho_j-pi`, since (5) places rho_j strictly between zero and pi.
If N is the number of changes, the fixed rim gap sum is `2*pi-N*pi`.
Its marked winding one forces `N=0`. Hence its relative origin is one
constant epsilon, either zero or pi.

### A reflected pair cannot redistribute internal winding

For either exchanged pair, write its phases as `psi_j+u_j` and
`psi_j-u_j` modulo `2*pi`. Let eta_j be a principal representative of
`u_{j+1}-u_j`. Adding the two stationary nodal equations cancels their
wheel currents: reflection reverses every corresponding relative phase,
including edges to the fixed rim vertex at epsilon zero or pi. Their
internal currents give

\[
 2\sin\rho_j\cos\eta_j
 -2\sin\rho_{j-1}\cos\eta_{j-1}=0.
 \tag{6}
\]

Equations (4)--(5) imply constant `cos(eta_j)`. There is a single
`d` in `[0,pi]` such that each eta_j is congruent to `+d` or `-d`.
The two child internal principal gaps at this slot are
`wrap(rho_j+d)` and `wrap(rho_j-d)`, in an order determined by that sign.
Their sum is

\[
 \begin{cases}
 2\rho_j,&d<\pi-\rho_j,\\
 2\rho_j-2\pi,&d>\pi-\rho_j.
 \end{cases}
 \tag{7}
\]

Equality would put a marked child edge on its winding cut and is excluded.
The formula also covers `d=pi` through its second case. There is no lower
cut crossing: `rho_j-d>-pi` follows from (5).

Both children have winding one, so the sum of their two gap sums must
be `4*pi`. The hub gap sum is `2*pi`; therefore (7) permits no slot of
its second kind. In particular `d<pi` and no child gap requires wrapping.
One child's gap sum is consequently

\[
 \sum_j(\rho_j+\sigma_j d)=2\pi,
 \qquad\sigma_j\in\{-1,1\}.
 \tag{8}
\]

It follows that `d*sum_j sigma_j=0`. Nine signs have an odd, nonzero sum,
so `d=0`. The pair's relative origins are constant across all slots.
The argument applies separately to both reflected pairs.

Every candidate stationary state in this sector thus has the derived form

\[
 \theta_{c,j}=\psi_j+\Phi_c\pmod {2\pi},\qquad
 \Phi_h=0,\qquad \Phi_f\in\{0,\pi\}.
 \tag{9}
\]

This reduction uses the odd length nine, both members' retained internal
windings and their complete stationary equations. Equality of cosine
alone would not justify equal offsets. The shared internal profile may
remain nonuniform and nonacute. Its currents cancel because of (4), so
the remaining stationary conditions are exactly those of the unit wheel.

## 3. Unit-wheel classification in the reflected sector

At any slot, list the five rim origins in cyclic order as
`(epsilon,b,a,-a,-b)`, with hub origin zero. Take principal representatives
for a and b. Rim branch cuts remain excluded; coincident neighboring
origins, which give zero gaps, are allowed.

For any phase vector, the phase Hessian is the signed weighted Laplacian

\[
 M=B\operatorname{diag}(\cos\delta_e)B^T.
 \tag{10}
\]

A vector constant along each internal C9 has no internal edge difference.
Its full quadratic form is nine times its wheel quadratic form. Thus
every wheel cut used below is an exact full-state Hessian test, regardless
of the shared internal profile in (9).

### Fixed rim origin zero

The stationary equations at b and a are

\[
 \sin(a-b)=2\sin b
          =-\sin a\,(1+2\cos a).
 \tag{11}
\]

If b were pi modulo `2*pi`, its edge to the fixed zero vertex would lie
on a marked cut. If b were zero, (11) would force a zero or pi. The pi
case again has a marked cut; the zero case has outer winding zero.
Consequently a retained nonzero outer winding requires `b!=0` and
`b!=pi` modulo `2*pi`.

The wheel equations and Hessian are unchanged under reversal of every
origin. This reverses the outer winding sign but preserves its being
nonzero. For the following exclusion take `0<b<pi` without loss of
generality.

Suppose that M were positive semidefinite. Test the vector equal to one
on the fixed rim constituent and zero on the others, constant in j.
Its quadratic form is `9*(1+2*cos(b))`, so necessarily

\[
 1+2\cos b\ge0.
 \tag{12}
\]

Equation (11) gives `2*sin(b)<=1`. On `(0,pi)`, this requires either
`b<=pi/6` or `b>=5*pi/6`; (12) excludes the second alternative. Hence
`0<b<=pi/6`.

Now `sin(a-b)>0`. With a principal representative for a, there are only
two possible branches: `a>b`, or `a<b-pi<=-5*pi/6`. In the latter,
`-sin(a)*(1+2*cos(a))<0`, contradicting (11). Therefore `a>b>0`.
The endpoint a=pi is also impossible in (11), and positivity of its last
expression forces

\[
 2\pi/3<a<\pi,\qquad a-b>\pi/2.
 \tag{13}
\]

Test the vector equal to one on the two constituents with origins a and
-a, and zero on the other four constituents, again constant in j. Only
the two edges to the b pair and the two spokes cross its cut. The full
quadratic form is

\[
 9\,[2\cos(a-b)+2\cos a]<0,
 \tag{14}
\]

by (13), contradicting positive semidefiniteness. Thus every stationary
candidate of this kind with retained nonzero outer winding has a negative
full-fine phase direction.

### Fixed rim origin pi

At the vertex with origin b, its hub and fixed-rim sine contributions
cancel. Its stationary equation is `sin(a-b)=0`. A difference pi would
be a marked rim cut, so a and b coincide modulo `2*pi`. The remaining
stationary equation is

\[
 \sin b\,(1+2\cos b)=0.
 \tag{15}
\]

The possibilities are b zero, pi or `+/-2*pi/3`, modulo `2*pi`. The zero
case puts an edge to the fixed pi vertex on a marked cut. The pi case
makes all rim origins coincide, with outer winding zero. For
`b=2*pi/3`, the rim principal gaps are
`(-pi/3,0,2*pi/3,0,-pi/3)`; for `b=-2*pi/3` they have the opposite
signs. Both have sum zero. No stationary candidate in this case retains
the prescribed outer winding one.

## 4. An exact retained-identity stationary saddle

The obstruction is to stability, not to existence of any stationary state.
An exact nonacute example makes that distinction explicit. Let z be the
unique root in `(-1/20,0)` of

\[
 P(z)=16z^3+36z^2+24z+1=0.
 \tag{16}
\]

Existence and uniqueness follow from `P(-1/20)=-14/125`, `P(0)=1` and
`P'(z)=24*(2*z+1)*(z+1)>0` throughout this interval. This definition
requires no numerical root evaluation. Put `t=sqrt(1-z^2)` and `k=1+2*z`;
choose the circular origins a and b by

\[
 (\cos a,\sin a)=(z,-t),\qquad
 (\cos b,\sin b)=(-(2+z)k/2,tk/2).
 \tag{17}
\]

The second pair has unit norm because
`(5+4*z)*(1+2*z)^2=4`, which is exactly (16). These choices give
`-2*pi/3<a<-pi/2` and `5*pi/6<b<pi`. Direct substitution yields

\[
 \sin(a-b)=tk=2\sin b=-\sin a(1+2\cos a),
 \qquad \cos(a-b)=-k^2/2.
 \tag{18}
\]

Thus the unit wheel with rim origins `(0,b,a,-a,-b)` and hub zero is
stationary. Its five principal ring gaps are

\[
 (b,\ 2\pi+a-b,\ -2a-2\pi,\ 2\pi+a-b,\ b).
 \tag{19}
\]

They avoid every marked cut and sum to `2*pi`. All five are nonacute:
the second and fourth have negative cosine by (18), and the middle gap
lies in `(-pi,-2*pi/3)`. Set `psi_j=(2*pi/9)*(j-4)` in (9) and form
equal to zero. The complete 54-node state is stationary, retains all six
internal windings one and all nine outer windings one, and has zero
global form and lifted phase means.

Its fixed-rim cut, constant along the nine slots, has quadratic form

\[
 9(1+2\cos b)=9(-1-5z-2z^2)<-27/4.
 \tag{20}
\]

It is therefore a strict full-fine saddle. This explicit organization
has internal defect zero but lies outside the earlier acute outer chart;
it is not a counterexample to that chart's no-equilibrium theorem. A
stationary source substitution, capture or maintained actual preparation
does not follow from its construction.

## 5. Complete-law consequence and scope

Section 2 reduces every stationary candidate in the declared reflection
sector to (9), without imposing an acute internal or outer chart.
Section 3 excludes a positive-semidefinite phase Hessian when the outer
winding is retained. Subtract the mean from a negative Hessian test
vector: its quadratic form does not change because `M*1=0`.

The
[complete tangent argument](ORGANIZED_CONTACT_MAINTENANCE.md#5-a-full-fine-obstruction-to-stable-retention-near-the-carrier)
then applies on the zero-mean subspace. Its symmetric pencil is

\[
 P(s)=s^2L^{-1}+sI/4+\gamma^2M/16.
 \tag{21}
\]

A negative eigenvalue at `s=0` and positive definiteness for sufficiently
large positive s supply a positive growth root. The stationary state is
therefore unstable for both coupled rows, not only for a wheel model or
a phase-only surrogate. The two conserved common means remain neutral.

This is not another small-defect estimate. Nonacute shared profiles are
admitted: the example with one internal gap `6*pi/7` and eight `pi/7`
can have internal phase defect far outside the earlier near-carrier
domain. The new obstruction instead uses reflection, stationary current
balance, constituent winding and odd cycle length. It does not transfer
to another cycle length or another coupling law without a new argument.

The result leaves open stationary targets that break (2), even if they
retain all marked windings. It also leaves open actual source transport,
capture, winding loss, a loss deadline and nonstationary maintenance.
No instability of an actual moving source follows from this stationary
classification. A source's independent fine perturbations cannot be
discarded to place it in the excluded sector.

The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next admission and the property-first physical comparison.
This conditional obstruction changes neither the earlier finite response
nor the separately captured aligned outer-zero control, and identifies
no fundamental physical property.

## 6. Fixed algebraic enclosure and implementation boundary

The declaration and analytic proof were committed as `9c7c3ef4` before
evaluating the numerical illustration. The
[detached implementation](../../src/tnfr/physics/_organized_contact_reflection.py)
encloses only the root in (16), using exactly the declared 128 rational
bisections. It then uses the shared outward square-root and argument
owners to reconstruct (17), and the existing complete contact geometry
and sine edge-field kernel for all 54 nodes and 144 edges. It admits no
incoming report, alternate root, source or numerical-budget override.

The resulting root bracket retains exact rational endpoints and opposite
polynomial signs; the derivative stays positive throughout the original
bracket. Materialization on the dyadic128 grid remains separate from that
root isolation. The resulting principal-gap enclosures establish all
fifteen marked windings one, including the five nonacute outer gaps.
Their shared exact branch identities fix the winding sums; interval
overlap alone would not determine an integer winding.

The full-edge quadratic enclosure satisfies the conservative displayed
bound

\[
 -7.030 < v^TMv < -7.029 < -27/4,
 \tag{22}
\]

where v is the centered fixed-rim cut. The independently evaluated
stationary-current boxes all contain zero and have absolute upper bounds
below `5e-37`. These residuals check the implemented arithmetic; the
cubic and trigonometric identities (16)--(18) prove stationarity. A
passing residual flag cannot replace those identities or certify an
arbitrary numerical candidate. A failed arithmetic audit makes the
combined implementation certificate unavailable.

The [dedicated tests](../../tests/physics/test_organized_contact_reflection.py)
independently build edge fields, winding sums and the full negative cut,
check exact root signs and the fixed budget, and retain the source-free
and immutable-report boundaries. No trajectory or frozen source producer
is invoked. This one illustrated saddle is not the proof of the general
reflection-sector obstruction in Sections 2--3, and no actual acquired
source is substituted by it.
