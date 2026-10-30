

## Acquisition and retention with evolving nonuniform capacity

<a id="native-dynamic-acquisition"></a>

**Conditional result.** For every fixed `kappa>0`, the admitted native
three-row law on the retained **two-bridge** support has an open set of
initially winding-zero full states that acquires and retains the aligned
winding-one organization. The set includes an open, nonempty subset with
nonuniform, genuinely evolving capacities. The construction supplies an
existential neighborhood and entry time, not numerical bounds for either.

This resolves the acquisition obligation selected in the preceding review.
It extends the demonstrated capability without changing the prepared
single-bridge theorem or treating its domain as the acquisition domain.
Earlier pending-acquisition statements record the scope before this result;
the program owns the current question. The four prerequisite receipts retain
their explicitly prepared scope, with their source associations renewed
after this append. The present theorem has its own independent review;
physical readiness remains absent.

This is a new analytical consequence of existing conditional proofs and the
retained computer-assisted transit certificate. No new trajectory, symbolic
evaluation, test or historical producer is executed. Source associations
and the independent review belong to
`artifacts/native-dynamic-acquisition-20261028/`. The historical numerical
certificate remains evidence from its actual archived source, not a fresh
evaluation of the current implementation.

### DA1. Complete law and the exact reference being extended

Take cycles `(0,1,2,3,4)` and `(5,6,7,8,9)`, unit conductances and both
supplied bridges `(0,5)` and `(1,6)`. Let `B=D-A` be this graph's ordinary
Laplacian. Degrees are three at **four** ports `0,1,5,6` and two elsewhere.
Use signed form, circular phase, positive capacity, the ordinary mean-one
capacity leaf and the same structural clock. Fix `kappa>0`, `e=w=1/2` and
`beta=1`. With the existing native `g(theta),H(theta)`, the complete law is

\[
\dot x=N(-D^{-1}Bx/2+g/2),\qquad
\dot\theta=H^{-1}NBx/2,\qquad
\dot\nu=-\kappa B\nu/2,\quad N=\operatorname{diag}\nu.
\tag{DA1}
\]

There are no inputs, events or consumed histories. The full regular phase
domain excludes the nonpositive real axis for every relative neighbor
resultant. It admits ordinary antipodal edge crossings when the resultants
remain regular. The field is smooth there, with pressure and metric refreshed
from the full state; the acute chart is needed only for the eventual local
target neighborhood. The CT capacity hull and mean conservation apply to
this actual connected support.

Use the exact represented reference from the
[retained transit](RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit).
On each ring its initial data are

\[
x^0=(A,-A,-B_0,0,B_0),\qquad
\theta^0=(a_0,-a_0,-a_0,0,a_0),\qquad \nu^0=\mathbf1,
\tag{DA2}
\]

where `A` and `B_0` are the exact binary64 values displayed as `0.4` and
`0.2`, and `a_0=875483625981347/562949953421312`. In the archived reduced
coordinates, `q=3*A-B_0=1+2^(-54)` and `r=2*B_0-A=0`; a floating diagnostic
that displays `q=1` is not the admitted initial interval. All node order,
bridge associations, coefficients and source records are retained. No
idealized rational-fifths or ideal-pi preparation replaces (DA2).

For `nu=1`, `B*nu=0` on this support. Uniqueness therefore makes that slice
invariant, and the first two rows are exactly the archived held-unit-capacity
native equations, for every fixed positive `kappa`. The original four-row
reduction is used only on this exactly copied/reflected reference. Perturbed
trajectories below use all thirty nodal coordinates.

The retained certificate establishes regular transit on `[0,32]` and entry
into the reflected capture domain. The existing reflected theorem then
gives all-forward continuation and convergence to uniform form and the
copied phase shape

\[
\theta_*=(4\pi/5,-4\pi/5,-2\pi/5,0,2\pi/5)
\quad\hbox{on each ring},\qquad \nu_*=\mathbf1,
\tag{DA3}
\]

modulo the reference's common offsets. This is the aligned winding-one
shape: all principal ring gaps are `2*pi/5`, both bridge gaps are zero,
and the bridge-square period is zero. It is equivalent to the usual twist
through a common origin and nodewise whole turns. The `T=32` endpoint is
not asserted to lie in the joint neighborhood constructed next.

### DA2. A joint recovery neighborhood on the actual two-bridge graph

Write `B_cyc=B(C5 disjoint union C5)` and
`L_ij=(e_i-e_j)(e_i-e_j)^T`, where `e_i` denotes a node basis vector.
With `gamma=cos(2*pi/5)>0`, the actual target geometry is

\[
B=B_{\rm cyc}+L_{05}+L_{16},\quad
K_*=\gamma B_{\rm cyc}+L_{05}+L_{16},\quad
(H_*)_i=\begin{cases}\pi(1+2\gamma)&i\in\{0,1,5,6\},\\
2\pi\gamma&\text{otherwise}.\end{cases}
\tag{DA4}
\]

Opposite ring-neighbor sine contributions cancel at every node; bridge
contributions vanish. Thus `g(theta_*)=0`. The positive edge cosines and
connectedness give `B,K_*>0` on `1^perp`, and every target resultant is
strictly positive real. Both bridge modes remain in the state; independent
ring rotations are not symmetries of the joined system.

Choose orthonormal `R` spanning `1^perp` and write
`x=c*1+R*u`, `theta=theta_*+alpha*1+R*v`, `nu=1+R*z` locally.
The mean-one leaf has 29 dimensions; quotienting only the two common
origins gives `y=(u,v,z)` with 27 coordinates. Set
`B_q=R^T B R`, `K_q=R^T K_* R`,
`A_q=R^T D^-1 R` and `M_q=R^T H_*^-1 R`. All are positive definite.
Since `p=Bx=g=0` at the target, capacity variations contribute zero to the
two native tangent rows there. The actual relative Jacobian is

\[
J_\kappa=\operatorname{diag}\left(
\begin{pmatrix}-A_qB_q/2&-M_qK_q/2\\M_qB_q/2&0\end{pmatrix},
-\kappa B_q/2\right).
\tag{DA5}
\]

Storage similarity by `diag(B_q^(1/2),K_q^(1/2))` transforms the native
block to `[-Dcal,-Ccal;Ccal^T,0]`, where
`Dcal=B_q^(1/2)*A_q*B_q^(1/2)/2>0` and
`Ccal=B_q^(1/2)*M_q*K_q^(1/2)/2` is invertible. The eigenvector work identity
is `Re(lambda)*(||s||^2+||t||^2)=-s^*Dcal*s`. A nonnegative real part
forces `s=0`, then `Ccal*t=0`, contradicting a nonzero eigenvector. The
native block is Hurwitz; `-kappa*B_q/2` is strictly negative. This proves
joint linear stability using (DA4), without transferring the one-bridge
JO matrices or a held-capacity basin.

The relative field is autonomous by the two exact common-origin symmetries.
In a sufficiently small positive-capacity, acute neighborhood it is smooth,
so `y_dot=J_kappa*y+O(||y||^2)`. Its positive Lyapunov matrix

\[
P_\kappa=\int_0^\infty e^{J_\kappa^Tt}e^{J_\kappa t}\,dt
\quad\hbox{satisfies}\quad
J_\kappa^TP_\kappa+P_\kappa J_\kappa=-I.
\tag{DA6}
\]

For a small enough ball, `d(y^T P_kappa y)/dt <= -||y||^2/2`.
Choose an open sublevel `W_kappa` with closure inside that ball and the
regular acute domain. It is forward invariant and relative deviations
decay exponentially. The common-origin rates are smooth, vanish at `y=0`
and are bounded by a constant times `||y||`. Their integrals converge.
These bounds rule out finite-time escape and give a unique full forward
solution tending to `(c_infinity*1,theta_*+alpha_infinity*1,1)`.
The arithmetic form and phase means need not be conserved.

This is a joint neighborhood of the complete dynamic-capacity law on the
actual support. It includes capacity contrasts and phase/form perturbations
breaking copy and reflection. Its size and decay estimates may depend on
the fixed `kappa`; no uniform zero-transport limit is asserted.

### DA3. Pulling the joint basin back to the original preparation

Let `s_ref(t)` be the exact reference (DA2), embedded in (DA1). Its proved
convergence to (DA3) implies that, for the fixed `kappa`, there is a finite
`T_kappa>32` at which its relative state lies strictly inside `W_kappa`.
Choose such a time; no computed value is required by this existence claim.

The reference image on `[0,T_kappa]` is compact and lies in the full regular
domain with positive capacities. Smooth local existence and dependence on
initial state, covered along that compact path, supply an open neighborhood
`U_0` of (DA2) in the mean-one leaf. Every solution from a sufficiently small
`U_0` exists, remains regular through `T_kappa`, and has its endpoint in
`W_kappa`. This is finite-transit dependence for the complete field (DA1),
not for the four-coordinate reduction or the held-capacity equation.
The target neighborhood then supplies continuation and retention for all
later times.

Initially the two ring windings of (DA2) are zero. Their smallest antipodal
margin satisfies `pi-2*a_0>1/32`, as retained in the original preparation
proof. Winding is locally constant away from antipodal gaps, so there is an
open initial phase neighborhood with both windings zero. Intersect `U_0`
with that neighborhood and positive capacities to obtain `U_kappa`.
It is nonempty, open in the 29-dimensional mean-one leaf and contains the
reference. Every initial state in it lacks the target winding-one identity;
by `T_kappa` it has entered the retained joint organization and thereafter
converges to its orbit. Ordinary winding may be undefined at intermediate
antipodal crossings; the full field remains regular along these solutions.

Nonuniform capacity is genuinely active in this set. The exact capacity row
gives

\[
\nu(t)-\mathbf1=e^{-\kappa Bt/2}(\nu(0)-\mathbf1).
\tag{DA7}
\]

The exponential is invertible at every finite time, and `B` is invertible
on `1^perp`. Thus `nu(0)!=1` implies both a nonzero contrast and
`nu_dot(t)!=0` as vectors at every finite time. The subset
`U_kappa intersect {nu!=1}` is open and nonempty; removing the uniform
slice does not leave merely a symmetric reference trajectory. Capacities
remain positive and converge to one. Individual node derivatives can vanish;
the claim concerns the full redistribution vector.

### What this adds, and what the controls prevent inferring

The same complete law now supports target absence, acquisition and
all-forward retention on an open set with evolving capacity, in addition
to the earlier prepared recovery and response. Initial geometric and form
structure is supplied: absence means absence of the declared winding-one
identity, not absence of every organization or of the substrate.

The result does not establish that capacity transport is necessary for
formation, improves a basin or determines the target. The matched held
model already has an open acquisition result. More sharply, uniform form
and phase consensus make `Bx=g=0`; both native rows then remain stationary
even when nonuniform capacities redistribute. Transport alone does not
create a phase organization from that state. On an isolated C5, the existing
zero-resultant obstruction still separates consensus from the acute twist;
capacity evolution does not alter that geometric obstruction. The supplied
two-bridge support is therefore substantive, not a decorative connection.

No numerical basin radius, entry time, maximal basin, rate advantage,
probability of formation or physical interpretation follows. The proof does
not install a dynamic-capacity executor or transfer the one-bridge receiver
derivative to this different graph. It retains the full fine state; an
autonomous regional NFR still needs a sufficient observation/closure or
memory account for its intended prediction.
