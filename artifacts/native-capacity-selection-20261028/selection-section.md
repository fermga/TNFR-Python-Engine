

## Capacity transport can change the selected organization

<a id="native-capacity-selection"></a>

**Conditional selection theorem.** For every fixed `kappa>0`, the admitted
two-bridge native model has shared positive mean-one initial states for which
evolving capacity and holding that same capacity profile fixed converge to
different phase organizations: one to the positive aligned twist and the
other to consensus. Such matched states exist arbitrarily close to the
uniform-capacity reflected saddle. Each admits an open neighborhood of
shared full states with those distinct destinations. No particular numerical
state, perturbation radius or orientation of the two outcomes is selected
by this existence proof.

The construction uses symmetry to prove a witness exists, then proves
robustness outside that symmetry. Near the saddle both initial ring windings
are already one. This is a selection/retention-versus-loss result, distinct
from DA's winding-zero acquisition theorem. It shows a structural role for
capacity transport beyond changing a transient or a clock. It does not
establish constitutive uniqueness, a preferred organization, or a physical
identification.

This is an analytical result, independently reviewed against the actual
model. No trajectory, symbolic evaluation, numerical witness or historical
producer is evaluated. Source associations, the full-state scope amendment
and reviews are in `artifacts/native-capacity-selection-20261028/`.
The preceding pending-selection statements retain their earlier scope; the
program owns the present question. The four prerequisite approvals retain
their prepared single-bridge scope with renewed source associations, and
physical readiness remains absent.

### CS1. An invariant capacity profile and the matched complete laws

Keep the two C5 cycles, bridges `(0,5),(1,6)`, unit conductances,
`e=w=1/2`, `beta=1` and one structural clock. Temporarily retain `e,w,beta`
in formulas to display the existing work identities. On each copied ring
use the same reflected native coordinates as the
[capture owner](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-reflected-capture-boundary):

\[
x=(A,-A,-B_0,0,B_0),\quad
\theta=(a,-a,-b,0,b),\quad
\nu=(s,s,t,u,t),\quad 2s+2t+u=5.
\tag{CS1}
\]

Capacities are strictly positive. Here `B_0` is a form coordinate, whereas
`B_G` below denotes the full graph Laplacian. Copying the rings and the
reflection/sign symmetry preserve this six-dimensional state space. The
capacity profile is even under reflection; form and phase are odd about
their supplied constant common origins. The central native rows vanish;
their full positive-resultant condition is still required.

Set `q=3A-B_0`, `r=2B_0-A`. With the actual regular native `g_0,g_4,H_0,H_4`,
the four native rows are

\[
\begin{aligned}
\dot A&=s(-eq/3+wg_0),&\dot B_0&=t(-er/2+wg_4),\\
\dot a&=(w/\beta)sq/H_0,&\dot b&=(w/\beta)tr/H_4.
\end{aligned}
\tag{CS2}
\]

The dynamic model is exactly `nu_dot=-kappa*B_G*nu/2`, hence

\[
\dot s=\tfrac\kappa2(t-s),\qquad
\dot t=\tfrac\kappa2(s+u-2t),\qquad
\dot u=\kappa(t-u).
\tag{CS3}
\]

The held model has `s_dot=t_dot=u_dot=0` and the identical native rows.
Both start from the **same** six coordinates, not a dynamic profile paired
with a unit-capacity substitute. Capacity transport preserves the weighted
mean in (CS1), positivity and the initial capacity hull. For
`zeta=(s-1,t-1)`, `u=1-2*zeta_1-2*zeta_2`, its matrix is

\[
\dot\zeta=C_\kappa\zeta,\qquad
C_\kappa=\frac\kappa2\begin{pmatrix}-1&1\\-1&-4\end{pmatrix}.
\tag{CS4}
\]

It is strictly stable. An explicit decreasing positive quadratic form is
the capacity variance below; its small ellipsoids provide invariant chart
domains without assuming Euclidean contraction.

### CS2. Capture remains valid under each actual capacity law

The same full two-ring storage and geometric barriers apply:

\[
E=6A^2-4AB_0+4B_0^2+\beta V(a,b),\qquad
\dot E=-e\left(\tfrac43s q^2+2t r^2\right),
\tag{CS5}
\]

where `V=10-2*cos(2a)-4*cos(a-b)-4*cos(b)`. It has no capacity coordinate,
so the dynamic row introduces no omitted chain-rule term. In
`R+=(2*pi/3,pi) x (0,pi/2)` or
`S=(-2*pi/3,2*pi/3) x (-pi/2,pi/2)`, the
[retained barrier and regularity proofs](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-protected-capture)
use only the phase geometry and `E<7*beta`. Equation (CS5) preserves those
inequalities for either model. The fixed energy sublevel times the compact
positive capacity hull is forward invariant and stays inside the full
regular domain. Thus continuation is global.

For held capacities, zero storage loss forces `q=r=0`, hence `A=B_0=0`.
Invariance then forces `g_0=g_4=0`. The retained critical-point argument
gives only `(a,b)=(4*pi/5,2*pi/5)` in `R+`, and only `(0,0)` in `S`.
LaSalle therefore gives respectively the twist and consensus. This step
uses the positive heterogeneous capacities in (CS2), not the old unit-only
verdict without a new argument.

For dynamic capacity define

\[
V_\nu=\tfrac12[2(s-1)^2+2(t-1)^2+(u-1)^2],\qquad
\dot V_\nu=-\kappa[(s-t)^2+(t-u)^2].
\tag{CS6}
\]

On the already confined compact product, `E+V_nu` has zero derivative only
when the native zero-loss conditions hold and `s=t=u=1`. Its largest
invariant zero-loss set is the corresponding target. LaSalle establishes
the same geometric capture and capacity convergence to one. The boundary
condition remains **`E<7*beta`**; imposing `E+V_nu<7*beta` would introduce
an unnecessary restriction. The held limits retain their initial capacity
profile, while the dynamic limits generally do not.

### CS3. The saddle and two comparable local boundaries

At `a_*=2*pi/3,b_*=pi/3,A=B_0=0`, the native field vanishes for every
positive capacity profile. Let `y=(a-a_*,b-b_*,A,B_0)`. At uniform capacity,
the [retained regular saddle](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-saddle-capture-route)
has one simple positive eigenvalue and three strictly stable native
directions. In this order its invertible Jacobian is

\[
J_0=\begin{pmatrix}
0&0&3w/(\beta\pi)&-w/(\beta\pi)\\
0&0&-w/(\beta\pi)&2w/(\beta\pi)\\
w/(2\pi)&w/(2\pi)&-e&e/3\\
w/(2\pi)&-w/\pi&e/2&-e
\end{pmatrix},\qquad
J(s,t)=\operatorname{diag}(s,t,s,t)J_0.
\tag{CS7}
\]

The nearby held family remains hyperbolic in its four native coordinates
with the same index. Its three-dimensional stable manifolds depend smoothly
on the two capacity parameters. Their union is the five-dimensional local
center-stable boundary `W_H` of the held equilibrium family. Here it means
that specific union of stable fibers, not an arbitrary center manifold.
The dynamic linearization at `(y,zeta)=(0,0)` is
`diag(J_0,C_kappa)`: all capacity-to-native columns vanish there because
the native field is zero. It has five stable directions and one unstable
direction, and a five-dimensional stable boundary `W_D`.

These uses of the smooth stable-manifold theorem and its parameter version
require smooth fields and a nonzero spectral gap, both verified above.
For the standard statements see Feldman's
[stable-manifold notes, slides 7 and 40](https://www.math.utah.edu/~feldman/ODE-slides4.pdf).
The boundary comparison and capture arguments below are applications to
this supplied TNFR law, not consequences asserted by that reference.

Choose fixed linear native coordinates `(xi,v)` from the unstable line and
stable hyperplane of `J_0`, with `v` three-dimensional. Both boundaries are
graphs `xi=h_D(v,zeta)` and `xi=h_H(v,zeta)` near the origin. They agree
on `zeta=0`, since the uniform slice has identical native dynamics. They
also contain `y=0` for all small capacities: the dynamic flow only relaxes
capacity on that exact native saddle, while the held point is an equilibrium.
Their tangents at the uniform saddle coincide. None of these facts implies
that the two nonlinear boundaries coincide.

### CS4. Their equality would force an impossible common stable hyperplane

Suppose the boundaries agreed in some neighborhood, with common graph
`xi=h(v,zeta)`. Both vector fields would be tangent to that graph. Their
difference is the pure capacity field `(0,C_kappa*zeta)`, so

\[
D_\zeta h(v,\zeta)C_\kappa\zeta=0.
\tag{CS8}
\]

Take a small capacity ellipsoid from (CS6) inside the graph chart. Following
its forward capacity flow holds the native coordinates fixed and contracts
`zeta` to zero while remaining in the chart. Integrating (CS8) gives
`h(v,zeta)=h(v,0)`. Consequently every nearby held stable fiber would have
the same native tangent hyperplane `E_s=ker(ell)` at its saddle, where
`ell` is a fixed nonzero row covector.

Let `lambda_0>0` be the unstable eigenvalue of `J_0`. Invariance of this
common stable hyperplane under every `J(s,t)` implies
`ell*J(s,t)=lambda(s,t)*ell`, with the nearby positive quotient eigenvalue.
Since `ell*J_0=lambda_0*ell` and `J_0` is invertible, (CS7) then gives

\[
\ell\operatorname{diag}(s,t,s,t)
  =\frac{\lambda(s,t)}{\lambda_0}\ell.
\tag{CS9}
\]

The parameters `s,t` vary independently near one while `u=5-2s-2t`
remains positive. Thus (CS9) forces `ell` to be supported only on the
`(a,A)` group or only on `(b,B_0)`. Neither is possible for `J_0`:

- If `ell=(alpha,0,eta,0)`, its `b` column equation forces
  `eta*w/(2*pi)=0`, then its `B_0` column forces
  `-alpha*w/(beta*pi)=0`.
- If `ell=(0,delta,0,zeta_0)`, its `a` column forces
  `zeta_0*w/(2*pi)=0`, then its `A` column forces
  `-delta*w/(beta*pi)=0`.

Both contradict `ell!=0`. Therefore `W_D` and `W_H` differ in every
neighborhood of the uniform saddle, for each fixed `kappa>0`. This proves
nonidentity of their local shapes without computing a curvature coefficient
or assigning a sign to the displacement. A nonzero gap requires nonuniform
capacity, because their uniform slices agree.

### CS5. Opposite local sides actually reach opposite organizations

The local comparison alone does not establish global outcomes. The uniform
reference saddle has two unstable branches: the increasing-`a` branch enters
`R+`, and the decreasing-`a` branch enters `S`. On both, integrating the
strict native storage loss from the backward saddle limit gives `E<7*beta`
at every nontrivial branch point. The dynamic unstable manifold remains
exactly in the invariant uniform-capacity slice, so these are also its
branches. The nearby held branches depend smoothly on capacity.

Select sufficiently small, fixed exit sections around one point on each
uniform branch, inside the strict sets `R+,E<7*beta` and `S,E<7*beta`.
Small capacity variations leave these section neighborhoods in the same
capture sets by CS2. The following local exit argument connects boundary
sign to these sections; no global-basin identification is assumed.

For the dynamic saddle, straighten its stable and unstable manifolds
simultaneously in a small neighborhood. In these local coordinates
`xi_dot=a(xi,eta)*xi` with `a>=lambda_0/2>0`; the stable row has the form
`eta_dot=S(xi,eta)*eta`. Its matrix stays close to the stable linear block,
so an adapted quadratic norm gives uniform exponential decay of `eta`
while the trajectory remains local. A nonzero `xi` keeps its sign and
exits through its corresponding fixed section. As initial states approach
the saddle, the exit time tends to infinity and the stable exit coordinate
tends to zero: exits approach the appropriate unstable branch point.

For held capacity, apply the same construction to each nearby native saddle,
using the smooth parameter-dependent charts and a uniform local spectral
gap. Capacity is a fixed parameter in this argument. If the complete initial
state tends to the uniform saddle, the native exits approach the same two
reference branch points. Orient both charts by the increasing-`a` branch;
the sign of `xi-h_D` or `xi-h_H` in the common graph chart agrees with the
respective exit orientation after shrinking the neighborhood.

Since the two boundary graphs differ arbitrarily close to the origin,
choose `(v,zeta)` with `h_D!=h_H`, and take the shared native unstable
coordinate strictly between them. Both models have exactly the same initial
state but opposite local exit signs. Such points can be chosen arbitrarily
close to the saddle, so their exits lie in the selected capture neighborhoods.
By CS2 one converges to the twist and the other to consensus. This proves
different limiting geometries, not merely different exit times. The argument
does not specify which law reaches which destination for a chosen numerical
profile, nor assert that both orientations occur.

### CS6. The selection contrast extends beyond exact reflection

Each reduced matched witness has two regular trajectories, with distinct
limiting geometries under the two complete laws. To extend it, one needs
full-state target neighborhoods for **both** geometries and **both** laws.
At the twist, the actual two-bridge matrices are (DA4). At consensus,
`K_*=B_G` and `H_*=pi*D`. Both are acute critical geometries. The
[native full-state recovery proof](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
therefore gives held-capacity recovery for every nearby positive capacity
profile, with smooth local dependence on that profile.

For dynamic capacity, use the full 27-coordinate mean-one quotient as in
DA2. The native block at either target has the same positive-storage
similarity and is Hurwitz; capacity adds `-kappa*R^T B_G R/2<0`, with zero
native crosscolumns at the equilibrium. The local Lyapunov and origin-drift
argument supplies a full-state invariant attracting neighborhood at each
target. This explicitly verifies consensus as well as the earlier twist.

For each of the two trajectories from a reduced witness, choose a finite
time at which it lies strictly inside its own full-state target neighborhood.
The times and neighborhoods need not coincide. Each preceding reference
segment is compact and regular. Continuous dependence of the corresponding
complete law pulls its target neighborhood back to an open set of full
initial states near the witness, with all node and port associations retained.
Intersect the two pullbacks on `sum_i nu_i=10`. This is a nonempty open set
in the 29-dimensional mean-one leaf, and every state in it is used unchanged
as the initial condition for both models. Each model reaches its own distinct
target. The held capacity profile is treated as a parameter with its admitted
law; it is not discarded or replaced when taking that pullback.

Shrink the intersection to keep capacities positive and nonuniform and to
retain the initial ring windings of the witness. Near the saddle those
windings are `(1,1)`; its phase gaps are away from antipodal separation.
The dynamic limit has uniform capacity, whereas the held limit preserves
its initial profile. The distinguishing observation is the phase geometry:
`(1,1)` at the aligned twist versus `(0,0)` at consensus. Symmetry is thus a
construction tool for the witness, not a required condition on every member
of the resulting open full-state set. No radius or concrete witness has
been computed.

### Interpretation and controls

The selected capacity law can change a basin boundary and the ultimate
organization under matched preparation, support, native equations and clock.
This is a functional distinction between two complete supplied laws. It
neither establishes that transport always helps preserve organization nor
that it is necessary for acquisition: DA and the held reference admit shared
acquisition neighborhoods.

The uniform-capacity slice is an exact null comparison. At the exact native
saddle, even nonuniform capacity changes no native coordinate, so it cannot
serve as the selection witness. Uniform form and phase consensus are another
stationary native control. For `kappa=0` the compared laws coincide; no
uniform separation as `kappa` tends to zero is asserted. These exclusions
are compatible with the arbitrarily nearby nonuniform off-boundary witnesses.

The result is existential: it does not locate the shifted boundary, quantify
basin volume, give a transition time, select an optimal transport rate or
provide a numerically admitted pair. It supplies neither an executable
dynamic-capacity integrator nor a laboratory mapping. A concrete numerical
discriminator still requires an explicit matched preparation, rigorous
admission/error bounds and its own prospective evaluation declaration.
