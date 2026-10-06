# Sine pattern geometry and dissipative capture

These results use the complete smooth reciprocal normalized-sine law.
Supplied support, coefficients, held capacities, initial state, clock and
phase-domain assumptions remain part of each theorem. The native
neighbor-argument law and its event analysis have a separate
[composition owner](RELATIONAL_PATTERN_COMPOSITION.md). Shared variables and
methods do not transfer conclusions between those complete laws.

<a id="reading-map"></a>

## Chapters and scope

| Chapter | Responsibility |
| --- | --- |
| [Exact and controlled form-phase reductions](SINE_FORM_PHASE_REDUCTION.md) | Equivalent state/memory representations, slow-time comparison and full-state capture handoff. |
| [Conservative regional formation and its barriers](SINE_REGIONAL_FORMATION.md) | Finite winding entry, channel balance, source geometry, barriers and robust passage. |
| [Collective phase families and live contact feedback](SINE_COLLECTIVE_PHASE_DYNAMICS.md) | Exact partitions, moving windows, contact/storage exchange and receiver rigidity. |
| [Conservative source preparation and finite retention](SINE_CONSERVATIVE_PREPARATION.md) | Retention bounds, saddle/corridor construction and its operational and constitutive limitations. |

<a id="combine-results-only-across-a-compatible-handoff"></a>

A formation-to-retention claim must connect an identity-absent preparation
to the full admitted retention set under the same law, including its live
environment, conserved storage, clock and phase lifts. Local growth, available
storage or a compatible invariant family alone does not supply that connection.
Positive-loss capture and conservative finite retention keep distinct premises;
exact hidden-state representations retain their independent initialization.
None of these results derives support birth or physical identification.

Section numbers remain stable. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone assigns research work.

## 24. Equilibrium geometry of components joined by bridges

<a id="sine-bridge-tree-composition"></a>

The nine attracting geometries of the
[eleven-node sine model](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-eleven-node-equilibrium-stability)
are an instance of a more general composition rule. This section concerns
the **smooth reciprocal sine law**, not the native Arg law in the
[reflected-domain analyses](RELATIONAL_NATIVE_FORMATION.md#regular-seeded-reachability-audit).
It combines component geometry under
the existing complete dynamics without assuming independent component
motion or installing a support law.

### Complete law and decomposition premises

Let `G` be a finite connected simple undirected unit graph with `n>=2`
nodes. Partition its nodes into `m>=1` nonempty sets `G_a`, each inducing
a connected graph; a singleton is allowed. Require exactly `m-1`
intercomponent edges, with the quotient graph a tree. These edges are then
bridges of the full graph. In particular, a second connection between the
same components or a cycle in the quotient is outside this decomposition.
Internal component edges need not themselves be bridges.

Retain strictly positive held capacities, fixed support, no forcing or
events, and `e,w,beta>0`. Define full-graph degrees `d_i`,
`K=diag(nu_i/d_i)`, the unit Laplacian `L`,
`S_i(theta)=sum_{j~i} sin(theta_j-theta_i)`, `a=w/pi` and
`b=w/(beta*pi)`. The consumed rows are

\[
\dot x=-eKLx+aKS(\theta),\qquad
\dot\theta=bKLx.
\]

Bridge degrees and all nodes remain in these rows. No isolated-component
degree normalization replaces `K`; a singleton's full degree is positive
because the full graph is connected. The total one-node graph is excluded
from this row-normalized statement.

At equilibrium, positivity of `K` gives `Lx=0`, hence uniform form, and
then `S(theta)=0`. Choose an exact critical phase geometry, not a state
declared critical merely because its residual is small. Let its phase
Hessian be

\[
v^\mathsf THv=
\sum_{\{i,j\}\in E(G)}
\cos(\theta_j^*-\theta_i^*)(v_j-v_i)^2.
\]

### A bridge cannot carry a stationary sine current

Delete a bridge `{u,v}` and let `A` be the resulting component containing
`u`. Summing `S_i(theta*)=0` over `i in A` cancels every internal sine
current against its reversed edge. The single remaining term is
`sin(theta_v*-theta_u*)`. Consequently every bridge has gap zero or pi
modulo `2*pi`, and its cosine is respectively `+1` or `-1`.

Removing these zero bridge currents from a node's balance leaves exactly
the internal component balance. The restricted phase geometry is therefore
critical on every `G_a`. Conversely, choose a critical phase geometry on
each component and independently prescribe zero or pi on each bridge.
Starting at one root component, shift each child's common phase to satisfy
its joining bridge. The quotient tree makes this consistent and unique
modulo the common phase of the whole graph. Thus these choices produce
all full critical geometries on this decomposition.

This argument uses the conservative *pairwise sine current* `S`. The
native neighbor-resultant direction is not generally such a current, so
the same cut sum cannot silently be applied to it.

### Relative Hessian inertia adds across the bridge tree

For each component let `(p_a,q_a,z_a)` be the positive, negative and zero
inertia of its internal Hessian after removing that component's common
phase direction. Their sum is `|G_a|-1`; a singleton contributes `(0,0,0)`.
Let `b_+` and `b_-` count zero-gap and pi-gap bridges, respectively.
Then the full relative inertia is

\[
\boxed{\quad
\operatorname{inertia}(H_{\rm relative})=
\left(\sum_a p_a+b_+,\ \sum_a q_a+b_-,\ \sum_a z_a\right).
\quad}
\]

To prove the direct sum, coordinatize a nodal perturbation modulo its one
common origin by every component's internal relative perturbation and the
`m-1` actual bridge endpoint differences. Given these coordinates, solve
the component common offsets successively along the quotient tree. This
gives a linear isomorphism: the dimensions are
`sum_a(|G_a|-1)+(m-1)=n-1`, and only the global common offset is discarded.
The Hessian quadratic form in these coordinates is the sum of the internal
component forms and the independent scalar bridge forms `+v_e^2` or
`-v_e^2`. Sylvester's law of inertia proves the formula. The unrestricted
Hessian adds one common-phase zero direction.

The coordinate transformation is a congruence, not a dynamical
decoupling. It does not diagonalize the full generator, remove mediated
interaction or identify component degrees with full degrees.

### Full nodal stability follows under positive loss

Write `r=n-1`, `j=sum_a q_a+b_-` and `z=sum_a z_a`. Reuse the
[full reciprocal Jacobian argument](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-eleven-node-equilibrium-stability),
which depends on connected support and the Hessian inertia rather than
the number eleven. On the conserved weighted-mean leaf, put
`B=K^(1/2)LK^(1/2)>0` and `C=K^(1/2)HK^(1/2)`, restricted to
`(K^(-1/2)1)^perp`. The latter has the relative inertia just derived.
The exact two-row variation reduces to the quadratic pencil

\[
P(s)=s^2I+esB+abB^{1/2}CB^{1/2}.
\]

For a nonreal characteristic root, its imaginary-part identity forces
strictly negative real part. On `s>=0`, `P'(s)=2sI+eB` is positive
definite. Each initially negative stiffness direction therefore crosses
zero once at positive `s`, counted with multiplicity. There are exactly
`j` positive real characteristic roots.

If `z=0`, no relative zero root remains. The full relative equilibrium
is hyperbolic, with `j` unstable directions and `2r-j` stable directions.
In particular it is locally exponentially attracting precisely when
every component has positive-definite relative Hessian and every bridge
is aligned. A negative component direction or an antipodal bridge makes
the full equilibrium nonlinearly unstable.

If `z>0`, a negative direction still proves instability, but the relative
zero modes require their own nonlinear analysis. A positive-semidefinite
Hessian with additional kernel is not certified as an attractor by this
test. The common form and phase origins are separate symmetry freedoms;
they are removed before counting these relative zero modes.

Strictly positive changes of the held capacities or coefficients preserve
this classification while changing rates and potentially basins. The
claim does not extend to zero capacity, zero loss, another phase law,
time-dependent support or forcing.

### Trees and assemblies of C5 components

For a tree, take every component to be one node. All critical phase gaps
are zero or pi, and the negative index equals the number of antipodal
edges. Hence **consensus is the only locally attracting relative
equilibrium**; every other critical geometry is unstable. This says
nothing about convergence of each preparation or finite-time transients.

For `m` disjoint unit C5 components, with any additional singleton
intermediaries, suppose the joining edges form a tree on these components.
The [exact C5 classification](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-eleven-node-equilibrium-stability)
supplies three positive-definite relative critical geometries on each
cycle: consensus and the two uniform unit windings. Every other C5
critical geometry is nondegenerate and has a negative direction.
All bridges must therefore align for attraction, and independent cycle
choices yield exactly **`3^m` locally attracting relative geometries**.
The count keeps the supplied node labels and the two handedness choices;
it removes one global phase origin, not graph automorphisms or reflection.
The two-C5/intermediary graph gives nine. This is neither a bound on
TNFR geometries nor a claim that all these states are accessible from a
particular preparation.

Adding a connection that creates a cycle between components removes the
independent-bridge premise. Nonzero stationary currents and coupled
cycle constraints can then appear. The bridge formula is unavailable;
the actual full Hessian or an appropriate cycle-space argument must be
used instead of extrapolating `3^m`.

### Shared geometry reader and evidence boundary

[`compose_bridge_tree_hessian_inertia`](../../src/tnfr/physics/phase_cycle_geometry.py)
centralizes the direct-sum calculation, retaining the declared component
dimensions, relative inertias and bridge signs. Its result is a conditional
geometric certificate: component inertia inputs require independent
derivation, and arithmetic composition alone does not verify a supplied
phase state, select a complete law or execute its dynamics. The specialized
C5 report can supply those component premises from its exact catalog.
The [independent bridge controls](../../tests/physics/test_bridge_tree_hessian_inertia.py)
compare the shared calculation with full Hessians and invalid decomposition
cases rather than treating a sum of supplied labels as a formation test.

These results explain how restoring geometry can survive composition
while the full system remains coupled. They do not create the support,
choose its partition or occurrence time, prove entry into a basin,
generate a persistent pulse or identify a physical constituent. Formation,
maintenance and autonomous hierarchical organization remain distinct
questions under their stated complete laws.

## 25. Cycle periods constrain joint composition

<a id="sine-cycle-sector-compatibility"></a>

Retain Section 24's finite connected simple unit support, positive held
capacities, `e,w,beta>0`, complete two-row normalized-sine law and absence
of forcing or events. Remove only the bridge-tree decomposition premise.
All internal nodes, full degrees and edge currents remain in the model.
The result concerns strictly acute phase gaps; nonacute equilibria and
formation from a specified preparation require separate arguments.

### Existing circulation criterion on the complete nodal law

Use `D` for the oriented incidence matrix and let `C` contain the integer
fundamental-cycle columns supplied by `PhaseCycleGeometry` (the transpose
of its `cycle_rows`). Write `b_1=|E|-|V|+1` for the cycle rank. With
`delta_e=wrap(theta_head-theta_tail)` and `f_e=sin(delta_e)`, the nodal
sine sum is `S=-Df`. At a full equilibrium, the phase row and positive
`K` force `Lx=0`, hence uniform form; the form row then forces `Df=0`.
Conversely these two conditions make both rows vanish.

The [circulation and reconstruction theorem](FORCED_PHASE_LOCKING.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods)
therefore supplies the exact necessary and sufficient equations

\[
f=Cz,\qquad |(Cz)_e|<1,\qquad
C^{\mathsf T}\arcsin(Cz)=2\pi\mathbf{k},\qquad
\mathbf{k}\in\mathbb Z^{b_1}.
\]

Its integer-lattice qualification, principal acute arcsine and at-most-one
geometry per sector all apply unchanged. This transfers its phase
geometry, not its separately supplied common-capacity phase dynamics:
here the full equilibrium is stationary and capacities may be unequal.
Existence still requires a solution of these coupled equations. A zero
mixed period need not give zero connecting circulation, as the
[return-path equilibrium](RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-equilibrium)
already demonstrates. Component winding labels omit these currents and
the internal phase deformation needed to satisfy the full nodal balances.

At any such equilibrium, every edge cosine is positive. Consequently
`H=D diag(cos(delta)) D^T` is positive definite modulo the one common
phase direction. Section 24's full-Jacobian quadratic-pencil argument
then gives local exponential recovery on the conserved weighted-mean
leaf, with `2*(|V|-1)` stable relative directions. There are no additional
relative zero modes in this strict acute domain. Uniform form and common
phase remain the two independent origin freedoms. This conditional
recovery conclusion does not establish existence, an acute invariant
domain for arbitrary preparations or entry into the local basin.

### A general exact obstruction from combined cycle periods

Set `t=delta/(2*pi)` in turns. Circular reconstruction requires
`C^T t=k`, while strict acuity requires `|t_e|<1/4`. For any nonzero
integer cycle-coordinate vector `r`, form the actual signed edge chain
`c=Cr`. Cancellation of shared edges occurs before taking its norm.
Every strictly acute realization must satisfy

\[
\boxed{\quad
|r^{\mathsf T}\mathbf{k}| < \frac{\|Cr\|_1}{4}.
\quad}
\]

Indeed, `r^T k=c^T t` and
`|c^T t|<=sum_e |c_e||t_e|<||c||_1/4`; full column rank makes
`c` nonzero. Equality also excludes a strictly acute realization. The
bound uses the combined chain, not the sum of separate cycle lengths.
It therefore detects conflicts that each basis-cycle bound can miss.
It is an obstruction to phase geometry itself, hence also to an acute
equilibrium under the complete law, independently of capacities or
positive coefficient values.

There is also an exact geometry-only existence characterization. For
`b_1>0`, define the open zonotope

\[
\mathcal Z=C^{\mathsf T}(-1/4,1/4)^{|E|}.
\]

An integer sector `k` has a strictly acute circular phase realization
if and only if `k` belongs to `Z`. Equivalently, the boxed inequality
holds for every nonzero real `r`. To see the converse, `Z` is the interior
of the full-dimensional compact zonotope
`C^T[-1/4,1/4]^|E|`; its support function in direction `r` is
`||Cr||_1/4`. Separation excludes every point outside that interior by
one failed strict support inequality. Since this polytope is rational,
its facets have rational normals, so testing all nonzero integer `r`
is equivalent as well. The original integral-period reconstruction
then supplies circular phases. For a tree the cycle-period vector is empty and this
period condition is vacuous; its equilibrium condition still forces
zero edge currents.

This equivalence supplies phase compatibility only. It does not impose
`D sin(2*pi*t)=0`; even a phase realization is not a full equilibrium
certificate. Without a completeness proof, a finite collection of
successful inequalities does not certify membership in `Z`. One failed
inequality suffices for exclusion.

The difference is substantive. Take three internally disjoint paths
from `A` to `B` of lengths `(4,4,1)`, all oriented from `A` to `B`.
Prescribe period one on each long path followed by the reversed single
edge. Giving every long-path edge turn `5/24` and the single edge turn
`-1/6` supplies a strictly acute circular realization of both periods.
Nevertheless this sector has no acute sine equilibrium. At such an
equilibrium, every internal degree-two node would force consecutive
sines equal, hence consecutive acute gaps equal. If the single-edge
gap is `phi`, each long-path gap must then be `(2*pi+phi)/4`. Acuity
requires `-pi/2<phi<0`, so the outgoing sine sum at `A` would obey

\[
2\sin((2\pi+\phi)/4)+\sin\phi
>2\sin(3\pi/8)-1>0.
\]

This contradicts nodal balance. Both five-edge cycles separately admit
a stable unit twist, and the joint sector passes every geometric support
inequality, but circulating-current compatibility still fails.

For a simple triangle or square, the integer period and strict length
bound force its period to zero. This retains the earlier short-cycle
consensus obstruction but also constrains individual sectors when those
short cycles span only part of the cycle space. For example, let three
internally disjoint paths between two nodes have lengths `(3,2,2)`.
The cycles consisting of the first path followed by either reversed
short path each have length five. Proposed periods `(1,-1)` pass both
individual bounds `1<5/4`. Their difference is the four-edge cycle
between the short paths, however, and has proposed period two, violating
`2<4/4`. Independent admissible cycle labels cannot be composed here.

### Independent components: a compatible and an excluded composition

Take two disjoint oriented C5 components with nodes `L_i,R_i`, indices
modulo five, and add all five matching edges `L_i--R_i`. Each supplied
component admits either uniform unit winding and has positive relative
phase Hessian. Any one matching edge can separately be aligned by a
common rotation of one component. These facts do not decide whether all
five connections can coexist in a strictly acute phase state.

Orient each square as
`Q_i=(L_i,L_(i+1),R_(i+1),R_i)`. Strict acuity gives every square period
zero, while the exact integer-chain identity is

\[
\sum_{i=0}^{4}Q_i=c_L-c_R.
\]

Thus any strictly acute state on the complete graph must have equal
inherited ring periods `k_L=k_R`. Opposite unit windings cannot compose
in this domain, even with arbitrary internal phase deformations. This
excludes any strictly acute full equilibrium preserving those two
periods; it does not exclude nonacute states or a history that changes
the supplied sectors.

On the same support, equal windings have the exact positive control
`theta_(L_i)=theta_(R_i)=2*pi*i/5`. Ring gaps are `2*pi/5`, all matching
gaps vanish, and opposite ring currents cancel at every node. Uniform
form completes a stationary state. Its full phase Hessian is positive
on relative coordinates, so the preceding full-law recovery result
applies for every choice of strictly positive held capacities. The
negative-winding counterpart and consensus give further controls. The
short-cycle span here has rank five while the full cycle rank is six:
the old consensus-only topology flag cannot express the exclusion of
opposite sectors while retaining these nonzero compatible geometries.

### Shared certificate and retained formation boundary

[`assess_acute_cycle_periods`](../../src/tnfr/physics/phase_cycle_geometry.py)
returns `AcuteCyclePeriodAssessment` for a supplied integer fundamental
sector and one nonzero rational cycle combination. The bound is homogeneous,
so rational rescaling preserves its verdict; supported represented-real
coefficients use the shared exact-materialization boundary. It reconstructs the
support owner, computes the exact combined chain, period, strict bound
and margin, and marks an obstruction only when that inequality fails.
A passing witness remains unresolved: the function neither searches
all separating directions nor solves the sine-current equations.
The certificate is unchanged by consistent changes of integer cycle
basis, though its displayed coordinates change. It reads no live nodal
state, installs no edge and does not replace execution admission.

The composition mechanism is therefore constrained circulation with
integral phase periods, with a full nodal restoring test once a compatible
equilibrium exists. Neither compatible geometry nor local recovery
selects the support, its creation event, its winding preparation or its
attainment from an initial state. No lower-dimensional autonomous
component law or physical identification follows from the certificate.

## 26. Storage below every sector face forces capture

<a id="sine-target-free-sector-capture"></a>

Retain Section 25's complete positive-loss sine model: finite connected
simple unit support, held `nu_i>0`, `e,w,beta>0`, the declared clock, and no
forcing or events. Put `K=diag(nu_i/d_i)`, `a=w/pi`, `b=w/(beta*pi)`.
The actual two rows and their storage balance are

\[
\dot x=-eKLx+aKS,\qquad \dot\theta=bKLx,\qquad
E=\tfrac12x^{\mathsf T}Lx+\beta V(t),\qquad
\dot E=-e(Lx)^{\mathsf T}K(Lx),
\]

where `S=-D sin(2*pi*t)` and `V(t)=sum_e[1-cos(2*pi*t_e)]`.
These are the [existing complete-law identities](SINE_PATTERN_RECOVERY.md#sine-relative-pattern-state),
not a new phase-only dynamics. In particular, all form coordinates remain
in the initial storage and subsequent evolution.

### The entire acute cell, without an equilibrium target

For the full integer fundamental-cycle matrix `C` and a supplied integer
sector `k`, define

\[
\mathcal A_k=\{t:C^{\mathsf T}t=k,\ |t_e|<1/4\},\qquad
\overline{\mathcal A}_k=
\{t:C^{\mathsf T}t=k,\ |t_e|\le1/4\}.
\]

Require an initial state in `A_k`. This establishes nonemptiness, so the
displayed closed polytope really is the closure: mix any closed-cell point
with the strict initial point. Integral periods reconstruct circular phases;
on this cell, edge turns determine the complete phase geometry modulo one
common rotation. Its tangent space is `ker(C^T)=im(D^T)`. The relative
boundary is the union of **all** signed edge faces

\[
F_{e,\sigma}=\{t\in\overline{\mathcal A}_k:
t_e=\sigma/4\},\qquad \sigma\in\{-1,1\}.
\]

Some faces can be empty. Neither a subset of basis cycles nor selected
boundary samples can replace this full union. For a tree there are no
cycle-period constraints: the acute cell is the nonempty full open edge
cube, its closure is the closed edge cube, and the minimum boundary phase
storage is one.

**Target-free capture theorem.** Let `B` be any proved lower bound on `V`
on every nonempty face. If the full initial storage satisfies

\[
\boxed{\qquad E(x_0,t_0)<\beta B,\qquad t_0\in\mathcal A_k,\qquad}
\]

then the sector has exactly one sine-critical phase geometry. The complete
solution remains strictly inside this sector for all forward time and
converges to that geometry and uniform form, modulo the common origins.
No critical coordinates are an input to this criterion.

To prove existence, minimize `V` on the compact closed cell. Since
`beta*V(t_0)<=E(x_0,t_0)<beta*B`, no minimizer lies on its boundary.
At an interior minimizer, stationarity in `im(D^T)` gives
`D sin(2*pi*t)=0`, hence `S=0`. Uniform form completes an equilibrium
of both rows. The Hessian of `V` is diagonal with strictly positive
entries `(2*pi)^2*cos(2*pi*t_e)` in the open cube. Thus `V` is strictly
convex on the affine cell and has only this one critical geometry.

The smooth complete vector field is globally Lipschitz in real form and
phase lifts, so it has unique global solutions. Before any putative first
exit, storage is nonincreasing. A boundary state would instead have
`E>=beta*B>E(x_0,t_0)`, a contradiction. The strict energy gap also keeps
the entire trajectory away from the closed-cell boundary.

For compactness and convergence, use the two conserved lifted means

\[
\overline x_K=
\frac{\sum_i(d_i/\nu_i)x_i}{\sum_i d_i/\nu_i},\qquad
\overline\theta_K=
\frac{\sum_i(d_i/\nu_i)\theta_i}{\sum_i d_i/\nu_i}.
\]

The first fixes the eventual uniform form. The second is defined after
choosing the cell's continuous lift and fixes its common phase origin;
it is not an additional circular observable. Their conservation follows
from `1^T L=0` and `1^T S=0`. Connectedness and the form energy bound
make the fixed-form-mean leaf bounded, while the phase cell modulo its
common origin is compact. Hence the trapped storage sublevel is compact.

In the set `dot(E)=0`, positivity of `e` and `K` forces `Lx=0`.
Then `dot(theta)=0`. To remain in that set requires
`L dot(x)=a LKS=0`, so `KS=c*1`; summing `S` and using positivity of
`1^T K^-1 1` gives `c=0`. Its largest invariant subset therefore has
`Lx=0,S=0`. The invariance principle and sector uniqueness imply
convergence to the single equilibrium on the fixed-mean leaf. Section
25's local exponential result applies once the trajectory is near it;
the certificate does not supply a uniform global convergence rate.

### A computable lower bound covering every face

The shared recovery owner uses convex supporting planes and exact rational
linear algebra, without solving for a critical target or sampling a
trajectory. Write `R=C^T`. On face `(e,sigma)`, remove column `e` to obtain
`B_e`, and put `d=k-sigma*R[:,e]/4`. For any auxiliary cube vector `u`
on the remaining edges, let
`f(u_j)=1-cos(2*pi*u_j)` and `g_j=2*pi*sin(2*pi*u_j)`. For **any**
rational multiplier vector `lambda`, convexity and `B_e*t=d` imply

\[
\begin{aligned}
V(t)&\ge L_{e,\sigma}(u,\lambda),\\
L_{e,\sigma}
&=1+\lambda^{\mathsf T}d+
\sum_{j\ne e}\bigl[f(u_j)-g_ju_j\bigr]
-\frac14\sum_{j\ne e}
\left|g_j-(B_e^{\mathsf T}\lambda)_j\right|.
\end{aligned}
\]

Indeed, each supporting line bounds its convex edge potential from below.
Subtracting the equality multiplier leaves a linear residual, whose
minimum over the remaining closed cube is minus one quarter of its
`l1` norm. The fixed edge contributes exactly one. The auxiliary `u`
need not satisfy the sector or face constraints and is not an equilibrium
candidate. An empty face also satisfies this bound vacuously.

[`_sine_sector_boundary.py`](../../src/tnfr/physics/_sine_sector_boundary.py)
chooses `u` by clipping the exact least-norm solution
`B_e^T (B_e B_e^T)^-1 d` to the closed cube. It chooses rational
`lambda=(B_e B_e^T)^-1 B_e g_mid`, where `g_mid` contains interval
derivative midpoints. Deleting a single edge leaves the cycle rows
independent: a nonzero cycle-space vector supported on one nonloop edge
would have nonzero incidence divergence. Thus the reduced Gram inverse
exists; a rank-one update reuses the full Gram inverse. The tree case
needs no inverse.

Outward rational sine, cosine and arithmetic bound the displayed dual
expression from below. Taking the maximum with the elementary bound one
on each face, then the minimum over all `2*|E|` faces, gives `B`. The
fixed arithmetic budget and deterministic supporting planes can make
this conservative; failure to obtain a strict storage margin is
**unavailable**, not nonexistence, instability or a failed physical law.
No exact minimization or optimal barrier is claimed.

For an independent nonzero-sector control, a unit-winding C5 has phase
storage `V_5=5*(1-cos(2*pi/5))`. On a feasible boundary face the remaining
four oriented turns sum to `3/4`. Convexity gives
`B_5=1+4*(1-cos(3*pi/8))`; the opposite signed face is infeasible.
The kernel's supporting planes recover this lower bound, with the strict
outward-certified gap `B_5-V_5>1/100`. This comparison remains strict
for small nonuniform form and phase perturbations, so it is not limited
to a stationary initial state. For example, perturb one node's phase by
`1/1000` turn and its form by `1/1000` with `beta=1`: the form cost is
`10^-6` and the phase increase is below `4*10^-5`. Two such cycles joined
through a supplied intermediary retain the full-support boundary bound
`B_5+V_5`, including the intermediary-edge faces; this follows from the
disjoint cycle potentials and is also obtained by the same kernel.

The feasible `(4,4,1)` no-equilibrium sector of Section 25 cannot pass
this theorem. Its closed-cell minimum must occur on the boundary;
otherwise the interior-minimizer argument would produce the excluded
equilibrium. Therefore every true boundary lower bound is at most the
initial phase storage for any admitted state in that sector, and the
strict full-storage margin is impossible. A conservative unavailable
report preserves that obstruction without pretending to decide all
other sectors.

This is capture **inside a supplied sector** under the complete sine law.
It selects no support event or preparation, proves no entry from another
sector, and does not transfer the native-law continuation certificate,
an operator-runtime claim, a closed coarse dynamics or physical identity.

## 27. Prepared form can acquire a nonzero phase sector

<a id="sine-prepared-sector-entry"></a>

The preceding theorem retains a sector already present at its initial
instant. A separate global estimate now supplies a sufficient route into
such a sector from **equal initial phases**, under the same complete sine
law throughout the transit. The preparation includes a supplied signed
form profile on the full fixed graph. It is structured initial information,
not an unstructured substrate or a derived source of support.
The full-state uncertainty extension below quantifies an open neighborhood
of this mechanism, including nonflat initial phase, under the same held
capacities and complete law.

### A global form-to-phase estimate with a finite declared horizon

Keep Section 26's finite connected simple unit support, positive held
capacities, `e,w,beta>0`, and absence of inputs or events. Set

\[
M=K^{-1},\quad A=KL,\quad r=w/e,\quad
\tau=et,\quad z=(b/e)(x-\overline x_K\mathbf1),\quad
\eta=ab/e^2=\frac{r^2}{\beta\pi^2}.
\]

Here `tau` is a rescaled variable used in the proof; the declared
original-clock endpoint is reported in the model clock `t`. Choose the common
initial phase origin so that `theta(0)=0`, and write
`v=z(0)`. The conserved form mean gives `1^T M z=0`. Rewriting both
complete rows, without deleting their nonlinear sine term, gives

\[
z'=-Az+\eta KS(\theta),\qquad
\theta'=Az,\qquad
(\theta+z)'=\eta KS(\theta),
\]

where primes denote `d/dtau`. These real phase lifts remain legitimate
through nonacute and antipodal configurations because the sine field is
globally smooth; no native-Arg continuation theorem is being imported.
More precisely, its linear form rows and globally bounded sine derivatives
make the full vector field globally Lipschitz on all real form and phase
lifts. Every finite declared preparation therefore has a unique global
solution, so the endpoint being enclosed exists and the enclosure is
nonempty.

Use `||q||_M^2=q^T Mq`. The operator `A` is self-adjoint in this inner
product, annihilates the common mode and has a positive gap on
`1^T Mq=0`. Let `lambda>0` be any certified lower bound on that gap.
One reusable choice is

\[
\lambda=k_{\min}\lambda_{L},\qquad
k_{\min}=\min_i\nu_i/d_i,
\]

where `lambda_L` is the shared exact rational lower bound on the
combinatorial Laplacian's positive gap. Indeed, on the weighted-mean-zero
leaf,

\[
\|q\|_M^2=\min_c\sum_i\frac{(q_i-c)^2}{k_i}
\le\frac1{k_{\min}}\min_c\sum_i(q_i-c)^2,
\qquad q^{\mathsf T}Lq\ge\lambda\|q\|_M^2.
\]

The forcing remains in this leaf because `1^T S=0`. The graph alone
supplies the **global** bound

\[
\|KS(\theta)\|_M^2=\sum_i k_iS_i^2
\le\sum_i\nu_i d_i=:F^2,
\]

using `|S_i|<=d_i`; this does not assume that phase gaps stay small.
Fix a finite supplied horizon `tau>0` and any proved
`rho>=exp(-lambda*tau)`. In particular the fixed rational expression
`rho=(1+lambda*tau/16)^(-16)` is valid, since
`(1+s/16)^16<=exp(s)` for `s>=0`. Put

\[
\boxed{\begin{aligned}
R_z&=\rho\|v\|_M+\eta F\min(\tau,1/\lambda),\\
R_\theta&=R_z+\eta F\tau.
\end{aligned}}
\]

Then every solution of the declared preparation satisfies

\[
\|z(\tau)\|_M\le R_z,\qquad
\|\theta(\tau)-v\|_M\le R_\theta.
\]

For the first inequality, variation of constants and the self-adjoint
semigroup estimate give
`||z(tau)||_M<=exp(-lambda*tau)||v||_M+
eta*F*(1-exp(-lambda*tau))/lambda`.
The integral factor is at most `min(tau,1/lambda)`. Integrating the
exact third row gives
`theta(tau)-v=-z(tau)+eta*integral_0^tau KS ds`, proving the second
inequality. Thus the complete nonlinear response is bounded without
evaluating or fitting that response.

### Convert the estimate into a full-state capture certificate

At the original-clock time `T=tau/e`, the entire endpoint belongs to
the explicit coordinate box

\[
\begin{aligned}
|x_i(T)-\overline x_K|&\le(e/b)\sqrt{k_i}\,R_z,\\
|\theta_i(T)-v_i|&\le\sqrt{k_i}\,R_\theta.
\end{aligned}
\]

This is an **analytic endpoint enclosure**, not a measured snapshot or
a validated Taylor trajectory. Independent coordinate intervals may
discard correlations, but still contain the complete true endpoint.
If retained, the dual norm gives sharper edge errors
`|(z_j-z_i)(T)|<=sqrt(k_i+k_j)*R_z` and
`|(theta_j-theta_i)(T)-(v_j-v_i)|<=sqrt(k_i+k_j)*R_theta`.
When these sharper bounds are consumed, the admitted endpoint set is the
intersection of the node box, the correlated edge constraints and the
conserved weighted-mean leaf. Its actual endpoint is in that intersection;
arbitrary corners of the full rectangular box need not satisfy the sharper
constraints. A claim about the entire product box instead requires bounds
valid on that larger set, as in the independent control below.
Also `Lambda=2*max_i nu_i` bounds the largest eigenvalue of `A`, since
`q^T Lq<=2*sum_i d_i*q_i^2<=Lambda*||q||_M^2`; hence

\[
E_{\rm form}(T)\le\frac{\Lambda}{2}(e/b)^2R_z^2.
\]

[`certify_sine_prepared_entry`](../../src/tnfr/physics/relational_sine_entry.py)
retains the captured complete-law source, the supplied scaled horizon,
both full endpoint coordinate bounds and their sharper edge bounds.
Its `SinePreparedEntry` report uses the shared exact weighted diffusion
gap and outward exponential/trigonometric arithmetic, then consumes the
same private capture core as `certify_sine_sector_capture`. A tighter
proved gap or exponential bound can improve this estimate without
changing the theorem. Missing law premises or unresolved strict margins
remain explicit; an unavailable enclosure is not failed formation.
The [prepared-entry tests](../../tests/physics/test_relational_sine_entry.py)
exercise the preparation, full-law hypotheses, endpoint provenance and
capture handoff.

Supply integer edge offsets for this admitted endpoint set. If every resulting
edge-turn interval is strictly acute, all its states lie in the same
sector `k=C^T h`. If the **whole** endpoint storage upper bound is below
the full-face barrier of Section 26, apply that theorem at time `T`.
This proves subsequent retention and convergence to the sector's unique
equilibrium, without requiring its phase coordinates in advance. When
`k!=0`, the initially phase-flat trajectory has acquired a nonzero
sector. Continuity then also implies at least one intervening antipodal
edge crossing; no event or discontinuity is added to the sine law.

The inequalities are finite conditions on the supplied preparation,
coefficients and horizon. A small ratio alone at a fixed horizon is not
a certificate: the remaining homogeneous form term is multiplied by
`e/b=beta*pi/r` when converted back to the original form coordinate.
For an asymptotic family, fix the support, `K`, `beta` and a weighted-mean-zero
profile `v` while `r` tends to zero. Then
one sufficient choice is `tau=1/r`. With the displayed fixed rational
bound, `rho=O(r^16)`, `R_z=O(r^2)` and `R_theta=O(r)`, so the endpoint
form vanishes and its phase approaches `v`. Consequently any `v` with
strictly acute edges and phase storage strictly below a proved sector
boundary has such prepared entrants for sufficiently small positive
`r`. This corollary follows from the quantitative bounds; it does not
replace their evaluation at a selected finite preparation.

That family carries the explicit initial storage

\[
E(0)=\frac12(e/b)^2v^{\mathsf T}Lv
=\frac{\beta^2\pi^2}{2r^2}v^{\mathsf T}Lv.
\]

It diverges as `r^(-2)` for fixed nonconstant `v`. The phase information
was encoded in the supplied form profile. Varying `r=w/e` changes the
constitutive ratio, whereas a common clock rescaling preserves it;
neither a bounded source budget nor a universal coefficient ratio is
established by this construction.

### One exact prepared control and its zero-form comparison

Take the complete unit C5 with node order `(0,1,2,3,4)`, unit held
capacities, `beta=1`, and the exactly representable effective weights

\[
e=1023/1024,\qquad w=1/1024,\qquad
x(0)=4092(-2,-1,0,1,2),\qquad\theta(0)=0.
\]

Its form mean is zero and
`v=(-8,-4,0,4,8)/pi`. This is the phase profile predicted by the
preparation, not an equilibrium supplied to the capture theorem. Four
oriented cycle gaps are `4/pi`; the closing gap is `2*pi-16/pi`.
They are all strictly acute and sum to `2*pi`, so the prospective sector
has unit winding. The unequal closing gap also shows that this profile
need not itself be sine-critical.

Choose `tau=100`, so `T=102400/1023` in the declared original clock.
An independent elementary gap bound suffices: shortest paths between
the ten unordered C5 node pairs have equal edge load five in the
path-counting inequality of the
[whole-support recovery owner](SINE_PATTERN_RECOVERY.md#sine-interacting-recovery).
Thus `lambda_2(L)>=5/5=1`, and `lambda>=1/2` because `K=I/2`.
Here `F=sqrt(10)`, `||v||_M=sqrt(320)/pi` and outward rational
evaluation of the displayed finite formulas gives

\[
R_z<6.132\,10^{-7},\qquad
R_\theta<3.123\,10^{-5}.
\]

The independent node intervals have form radius below `0.001394` and
phase radius below `0.000022083` radians. On their complete product box,
all five wrapped edges remain strictly acute, the winding is one, and
full storage is less than `3.456051`. The independently derived C5
full-face bound in Section 26 exceeds `3.469266`. The strict margin
therefore proves finite-time entry followed by permanent sector
retention and convergence. No trajectory samples or search over
preparations are needed for this control. The initial form storage is
exactly `160*1023^2=167444640`; the large transient energy budget is
part of the preparation, not omitted phase-only information.

With the same graph, capacities, coefficients and equal phases but
uniform initial form, both complete rows vanish identically. That
control remains in consensus and cannot acquire nonzero winding.
The positive result is thus a conditional conversion of supplied
nonuniform form into a maintained phase sector. It neither reopens
the separately bounded eleven-node receiver source class nor derives
support birth, an autonomous preparation, a unique fundamental law or
physical matter.

### Quantitative robustness for independent form and phase uncertainty

The exact witness's strict margins and continuous dependence already imply
some open neighborhood of successful preparations. The additional result
here is a computable bound for a declared full-state family. It uses the
existing [`SineRelativePattern`](../../src/tnfr/physics/relational_sine_pattern.py)
observation set, with every node retained and capacities exact:

\[
\begin{aligned}
x_i(0)&=x_i^{\rm nom}+c_x+\xi_i,
&|\xi_i|&\le\epsilon_i^x,\\
\theta_i(0)&=\theta_i^{\rm nom}+c_\theta+\zeta_i,
&|\zeta_i|&\le\epsilon_i^\theta.
\end{aligned}
\]

The two common offsets are arbitrary and independent. The residual errors
in form and phase are independently bounded; phase centers and errors
refer to the owner's supplied consistent real lifts. No radius is a fitted
confidence level, and nominal capacities do not replace an uncertain
capacity law. The theorem retains the original common-offset-plus-residual
set, not the larger rectangular relative box that forgets reference-error
correlations.

Write `m_i=M_ii`, `W=sum_i m_i`, and
`P_M q=q-1*(sum_i m_i*q_i)/W`. This is the orthogonal centering projector
for `||.||_M`. For each actual initial state put
`v=alpha*P_M x(0)`, where `alpha=b/e>0`. The previous form row and global
forcing bound are unchanged, while the exact integrated phase identity is
now

\[
\theta(T)=\theta(0)+v-z(\tau)
             +\eta\int_0^\tau KS(\theta(s))\,ds.
\]

The nonflat initial phase is an essential term. Its omission would replace
the declared preparation by a different one. Since `P_M` is a contraction,
the triangle inequality gives the uniform initial norm bound

\[
\|P_Mx(0)\|_M\le N_x:=
\|P_Mx^{\rm nom}\|_M+
\left(\sum_i m_i(\epsilon_i^x)^2\right)^{1/2}.
\]

Use `alpha*N_x` in place of `||v||_M` in the same finite formulas for
`R_z` and `R_theta`. They then bound every preparation in the original
observation set, including its nonlinear phase feedback during nonacute
passage. No nominal trajectory or new approximation to the sine law is
introduced. With zero residual radii, this estimate recovers the exact
initial norm used above.

For a directed edge `i--j`, let `X_ij` and `Theta_ij` be the initial form
and phase difference intervals obtained directly from the original
residuals. Their radii are respectively `epsilon_i^x+epsilon_j^x` and
`epsilon_i^theta+epsilon_j^theta`; the common offsets cancel before any
interval construction. The same actual endpoint therefore satisfies

\[
\begin{aligned}
x_j(T)-x_i(T)&\in
\left[-\alpha^{-1}\sqrt{k_i+k_j}R_z,
       \alpha^{-1}\sqrt{k_i+k_j}R_z\right],\\
\theta_j(T)-\theta_i(T)&\in
\Theta_{ij}+\alpha X_{ij}+
\left[-\sqrt{k_i+k_j}R_\theta,
       \sqrt{k_i+k_j}R_\theta\right].
\end{aligned}
\]

These direct edge bounds retain the common-mean cancellation needed by the
shared sector capture calculation. They are not obtained by differencing
independent outer node intervals. If every initial raw phase-edge interval
is strictly acute, the supplied real lifts telescope around every cycle
without wrapping, proving initial sector zero for the whole family. Apply
the supplied final integer edge offsets to the endpoint intervals and
require the same strict acute and total-storage margins as before. A
nonzero captured sector then proves acquisition and subsequent convergence
for every state in the family; acuteness is not required during transit.

The uncertainty report also distinguishes centered and absolute endpoint
coordinates. Define

\[
d_i^x=(1-m_i/W)\epsilon_i^x+
       \sum_{j\ne i}(m_j/W)\epsilon_j^x
\]

and define `d_i^theta` in the same way. These bound the respective centered
residual coordinates. Centered endpoint form has radius
`alpha^(-1)*sqrt(k_i)*R_z` about zero; centered phase has radius
`d_i^theta+alpha*d_i^x+sqrt(k_i)*R_theta` about
`(P_M theta^nom)_i+alpha*(P_M x^nom)_i`. Arbitrary common offsets leave
absolute endpoint coordinates and weighted means unavailable for a relative
source. Each actual state has its own conserved weighted means, so this
family ranges over a union of conserved leaves, not one common leaf.
For an exact captured source, its known means separately reconstruct the
absolute endpoint bounds. The correlated edge constraints remain distinct
from the larger product of centered node intervals in either case.

#### One fixed preparation neighborhood

Keep the published C5 nominal preparation, support, held capacities,
`e=1023/1024`, `w=1/1024`, `beta=1` and `tau=100`. Declare, before
evaluating the bound, the same independent residual radii at every node:

\[
\epsilon_i^x=1/16,\qquad
\epsilon_i^\theta=1/65536.
\]

There is no coefficient, horizon or radius search. Every initial phase
gap has magnitude at most `1/32768` radian, so all preparations have zero
winding. The shared weighted estimate and direct edge enclosures give
endpoint storage below `3.456318`, against the complete boundary lower
bound above `3.469266`; the computed strict margin exceeds `0.0129487`.
Consequently the complete family acquires unit winding and converges to
the captured sector's unique phase geometry, with each preparation's own
conserved common origins. Its positive radii contain an
open neighborhood in all initial form and phase coordinates. Arbitrary
common translations and rotations do not change the conclusion.

The storage budget remains part of that family. The initial form storage
is bounded above by
`21433437701/128=167448732.0390625`, and the initial phase potential is
at most `5/(2*32768^2)` by `1-cos(u)<=u^2/2`. The shared source retains
its own outward storage enclosure; the exact nominal storage cannot
replace the uncertain preparation's budget.

For the same uncertainty radii but uniform nominal form and phase, the
whole initial set instead has zero-sector storage at most
`5/128+5/(2*32768^2)<1`. The elementary full-face bound one therefore
certifies zero-sector capture and convergence to consensus directly.
This uncertain control family is not stationary: its nonuniform residuals
can evolve. Exact uniform form and phase remain the stationary control
proved above. These two controls distinguish robust conversion of the
supplied structured preparation from generic claims about small noise,
autonomous preparation or the separately deferred receiver source class.

## 31. A fixed original preparation budget forces consensus at small ratio

<a id="sine-budget-consensus"></a>

The successful preparation in Sections 27 and 30 retains nonzero scaled
form while its original form budget increases as the exchange/loss ratio
decreases. A different question fixes that original budget. For exactly
phase-flat preparations, the exact mixed-coordinate identity from
Section 28 gives an all-time obstruction to winding acquisition, stronger
than a finite slow-time comparison.

### The preparation class and sufficient condition

Fix a finite connected simple unit graph with at least two nodes, positive
held capacities, `beta>0` and complete-law coefficients `e,w>0`. There are
no external inputs or support events. Retain
`K=diag(nu_i/d_i)`, `M=K^-1`, `A=KL`, `alpha=b/e` and
`eta=ab/e^2=beta*alpha^2`. Let `lambda>0` be a certified weighted quotient
gap, so that

\[
x^{\mathsf T}Lx\ge\lambda\|P_Mx\|_M^2.
\]

The preparation class consists of **all** signed form vectors and exactly
equal initial circular phases satisfying

\[
\frac12x(0)^{\mathsf T}Lx(0)\le B,
\qquad B\ge0.
\]

Choose equal initial real phase lifts. Common form and phase origins are
arbitrary and may differ between members. Independent nonzero phase
residuals do not in general belong to this exactly flat class.

Write `k_i=K_ii`, `kappa=max_edges(k_i+k_j)` and

\[
\overline W=\frac{2\eta B}{\beta\lambda}.
\]

For `B>0`, a sufficient condition is

\[
\boxed{\qquad
0<\eta\le1,\qquad
2\sqrt{k_i+k_j}\sqrt{\overline W}<\pi
\quad\text{on every edge}.
\qquad}
\]

Equivalently, the strict edge condition is
`8*kappa*eta*B<beta*lambda*pi^2`. Under these hypotheses every member
retains zero principal cycle periods for all future time and converges
to full consensus: form tends to its conserved weighted mean and phase
to its conserved common phase lift. The result has no selected slow
horizon. If `B=0`, connectedness already forces constant initial form;
the entire class is exactly stationary for every positive `e,w,beta`,
without the restriction `eta<=1`.

### An exact auxiliary Lyapunov function

For each member, center the phase on its conserved weighted mean and set

\[
\vartheta=P_M\theta,\qquad z=\alpha P_Mx,
\qquad y=\vartheta+z,\qquad \tau=e t.
\]

The exact rows are `vartheta'=Az`, `z'=-Az+eta*KS(vartheta)` and
`y'=eta*KS(vartheta)`. Both centered coordinates have zero weighted mean.
Define the auxiliary function

\[
W=\frac12\|\vartheta+z\|_M^2+\frac12\|z\|_M^2
=\left\|z+\frac{\vartheta}{2}\right\|_M^2
 +\frac14\|\vartheta\|_M^2.
\]

This is a positive definite function of the retained joint state on the
centered leaf. It is not the original storage `E`, a new evolution law,
or a physical energy identification. Since `vartheta(0)=0`, the
preparation budget implies

\[
W(0)=\|z(0)\|_M^2
\le\frac{2\alpha^2B}{\lambda}=\overline W.
\]

Orient each edge `ij` arbitrarily, and write its raw real-lift differences
as `delta_ij=vartheta_j-vartheta_i` and `zeta_ij=z_j-z_i`.
The exact edge summation identity
`u^T S(vartheta)=-sum_edges (u_j-u_i)*sin(delta_ij)` gives

\[
\begin{aligned}
W'
&=-z^{\mathsf T}Lz
  +\eta(\vartheta+2z)^{\mathsf T}S(\vartheta)\\
&=-\sum_{ij}\left(\zeta_{ij}+\eta\sin\delta_{ij}\right)^2
  -\eta\sum_{ij}
   \left(\delta_{ij}\sin\delta_{ij}
                     -\eta\sin^2\delta_{ij}\right).
\end{aligned}
\]

For `|delta|<pi`, the real numbers `delta` and `sin(delta)` have the
same sign and `|sin(delta)|<=|delta|`. Hence
`delta*sin(delta)>=sin(delta)^2`. If `0<eta<=1`, every term in the
last sum is nonnegative. It is strictly positive whenever `delta!=0`
in this open strip, including when `eta=1`. Thus `W'<=0` throughout
the strip. This calculation uses the weighted inner product and the
edge identity directly; it requires no commutation of `K` and `L`.

### First-exit exclusion and full-state convergence

Before any possible first exit from the raw strip `|delta_ij|<pi`,
monotonicity gives `W<=overline W`, and weighted duality gives

\[
|\delta_{ij}|
\le\sqrt{k_i+k_j}\|\vartheta\|_M
\le2\sqrt{k_i+k_j}\sqrt{\overline W}<\pi.
\]

The final margin is uniform in time, so continuity excludes that first
exit. Global existence follows from the complete field's global
Lipschitz property. Consequently the entire future remains in the same
strip. Every principal phase difference equals its raw lift difference,
and raw differences telescope around every cycle; all principal cycle
periods therefore stay zero. No acuteness assumption is needed for this
conclusion. The stronger optional condition
`32*kappa*eta*B<beta*lambda*pi^2` keeps every edge strictly acute.

The sublevel `W<=overline W` is compact on the finite-dimensional
centered leaf and, by the strict margin, lies entirely inside the raw
strip. It is positively invariant. Within it, `W'=0` requires
`delta_ij=0` on every edge, followed by `zeta_ij=0`. Connectedness and
the two zero-mean constraints give `vartheta=z=0`. LaSalle's invariance
principle therefore proves `vartheta(t),z(t)->0`. Since the fixed
coefficient `alpha` is strictly positive, `P_Mx(t)=z(t)/alpha->0` as
well. Each member retains its own conserved origins throughout this
argument; no common unknown origin is assigned a value.

The same invariant sublevel also supplies uniform full-state bounds
`||z(t)||_M<=sqrt(2*overline W)` and
`||P_Mx(t)||_M<=sqrt(4*B/lambda)`. The latter follows by cancelling
the exact positive `alpha` algebraically, so its evaluation need not
divide by a rounded interval for a very small form-to-phase scale.

This proves consensus directly from a different invariant sublevel. It
does not assert that the initial original storage lies below the
ordinary acute-sector boundary barrier. In particular, the allowed
budget can exceed that barrier while the sufficient ratio condition
still holds.

### Budget dependence, shared certificate and fixed controls

At fixed support, capacities, `beta` and finite `B`, the strict condition
holds for all sufficiently small `r=w/e`, since
`eta=r^2/(beta*pi^2)`. This is an all-time statement for each admitted
positive ratio, not an exchange of a finite-time approximation with an
infinite-time limit. Its contrapositive supplies a conservative necessary
cost: if an exactly phase-flat preparation reaches nonzero principal
cycle period while `0<eta<=1`, its original initial form storage must
satisfy

\[
\frac12x(0)^{\mathsf T}Lx(0)
\ge\frac{\beta\lambda\pi^2}{8\kappa\eta}
=\frac{\beta^2\lambda\pi^4}{8\kappa r^2}.
\]

This lower bound is not sufficient for formation and is not claimed
sharp. It derives the inverse-square budget requirement for the whole
phase-flat class, rather than only evaluating the cost of one successful
preparation.

[`certify_sine_budget_consensus`](../../src/tnfr/physics/relational_sine_budget.py)
implements this sufficient budget condition using admitted support,
held capacities, the shared weighted gap and outward arithmetic. It
does not integrate a trajectory or install a phase reference. A failed
sufficient margin remains unavailable, not an instability or formation
verdict.

The [budget tests](../../tests/physics/test_relational_sine_budget.py)
retain the published C5 law `e=1023/1024`, `w=1/1024`,
`beta=nu_i=1` and fix the whole preparation class `B=160`.
The independent gap `lambda>=1/2` from Section 27 and `kappa=1` give

\[
\overline W\le\frac{640}{1023^2\pi^2},\qquad
|\delta_{ij}(t)|
\le\frac{2\sqrt{640}}{1023\pi}<\frac1{50}
\quad\text{for all }t\ge0.
\]

For example, `x_i(0)=4*(i-2)` has exactly this form budget, but the
certificate covers every phase-flat form vector within the ceiling and
every common origin. The earlier nonzero-sector source
`x_i(0)=4092*(i-2)` has budget `160*1023^2` and does not meet this
sufficient strip condition. Its retained acquisition certificate and
the present bounded-budget exclusion concern distinct preparation
classes under the same law. No ratio, support or budget is searched to
alter either control.

The theorem does not extend to arbitrary initial phase records, select
the origin of a supplied preparation or support, or establish physical
identity. It distinguishes what this complete law can do with a fixed
original form budget from its previously certified large-preparation
mechanism.

## 32. Equal original budgets do not determine the acquired geometry

<a id="sine-equal-budget-preparation"></a>

Section 31 supplies a necessary resource bound for winding acquisition.
That scalar budget does not specify the organization reached by the full
dynamics. The following fixed comparison separates the budget from the
geometric information in the preparation. Its symmetry mechanism reuses
the [sine orientation argument](RELATIONAL_MEDIATOR_DYNAMICS.md#mediator-orientation-scope)
and the [reflection obstruction to receiver formation](SINE_PATTERN_RECOVERY.md#sine-interacting-recovery);
it is not a new symmetry law.

### A capacity-compatible automorphism restricts cycle periods

Consider the complete unforced sine rows on admitted fixed unit support,
with held capacities `nu_i>=0`, `beta>0`, `w>0` and `e>=0`. Let `p`
be a graph automorphism preserving the capacities, and let `P` be its
permutation matrix. Degrees are preserved as well, so for
`K=diag(nu_i/d_i)` one has `PK=KP`, `PL=LP` and
`S(Ptheta)=P*S(theta)`. Thus the full field is equivariant:

\[
F(Px,P\theta)=P F(x,\theta),
\]

where the same permutation acts on each row. This argument requires
neither positive loss nor strictly positive capacity; it does not use
`K^-1`. There are no inputs or events that break the supplied symmetry.

Suppose the actual initial form and phase lifts are exactly fixed:
`Px(0)=x(0)` and `Ptheta(0)=theta(0)`. Global uniqueness for the smooth
complete field then gives `Px(t)=x(t)` and `Ptheta(t)=theta(t)` for all
`t>=0`. These are raw equalities on consistent real lifts, without a
floating tolerance or an assumed nodewise phase rewrapping. Symmetric
error intervals alone would not establish them for every independently
uncertain member. Independent common form and phase shifts preserve the
equalities.

For an oriented cycle chain `c`, let `p_*c` be the chain obtained by
mapping each ordered edge through the automorphism. At a time when the
relevant edges are not antipodal, their principal angle increments
`Delta_e` are uniquely defined in `(-pi,pi)` and reverse sign when an
edge orientation reverses. Define

\[
k(c)=\frac1{2\pi}\sum_e c_e\Delta_e.
\]

Raw state invariance implies `k(p_*c)=k(c)`. Consequently,

\[
\boxed{\qquad p_*c=-c\quad\Longrightarrow\quad k(c)=0.\qquad}
\]

More generally the cycle-period functional is invariant under the
automorphism's induced action on cycle chains. The reversal test can
therefore use a supplied simple cycle on a larger support; it need not
be a particular fundamental cycle or the entire graph.

At an antipodal edge, a half-open wrapping convention need not satisfy
`wrap(-delta)=-wrap(delta)`. No zero-period conclusion is assigned at
that boundary by this argument. The raw symmetry itself remains exact,
and the smooth sine evolution continues through the boundary. In
particular, an invariant source can never enter a strict acute state
with nonzero winding on a reversed cycle. This is an exclusion, not a
proof of consensus or of avoidance of antipodal edges.

Any circular limiting configuration with no antipodal edge on this cycle has
a neighborhood where its winding is constant. A nonzero-winding limit would
therefore contradict the zero-winding restriction at all sufficiently late
times. This also excludes nonacute limits with nonzero winding when their
cycle edges stay strictly away from antipodes; no convergence is assumed.

### The fixed C5 preparations have exactly equal storage

Use nodes `0,...,4`, the oriented cycle `0,1,2,3,4,0`, unit capacities,
`beta=1`, `e=1023/1024`, `w=1/1024` and initially zero phase. Retain
Section 27's acquisition source and fix the comparison source as

\[
\begin{aligned}
x_+(0)&=4092(-2,-1,0,1,2)
       =1023(-8,-4,0,4,8),\\
x_R(0)&=1023(8,4,-8,-8,4).
\end{aligned}
\]

Both sums vanish, so both conserved weighted form means are zero.
Initial phase potential is zero. Directly from the five support edges,

\[
\begin{aligned}
\frac{E_+(0)}{1023^2}
 &=\frac12(4^2+4^2+4^2+4^2+16^2)=160,\\
\frac{E_R(0)}{1023^2}
 &=\frac12(4^2+12^2+0+12^2+4^2)=160.
\end{aligned}
\]

Thus the full initial storage is exactly `160*1023^2=167444640` in
both cases, under the identical support, capacities, law and clock.
The equality does not assume equality of their other state coordinates
or observations.

The reflection `p(i)=-i mod 5` fixes `x_R(0)` and its flat phase,
preserves the capacities, and reverses the declared cycle. Its full
trajectory therefore has the raw phase form `(a,b,c,c,b)` at every
time. Away from antipodal edges the cycle increments occur in opposite
pairs, with the edge `2--3` having zero increment, so the winding is
zero. The control cannot enter either strict acute nonzero winding
sector or converge to either corresponding C5 twist: those phase
geometries are outside the closed reflection-fixed set, even after a
common phase rotation.

This control is not stationary. Initially
`KLx_R=1023*(4,4,-6,-6,4)`, so its full phase row is nonzero. The
reflection obstruction neither identifies its eventual equilibrium
nor proves convergence or high-budget consensus.

In contrast, Section 27 already certifies that `x_+(0)` reaches the
strict winding-one capture sector at `tau=100`, or original time
`t=102400/1023`, and subsequently converges within that sector. That
certificate is reused without another trajectory or horizon search.
Reflecting this successful full preparation while holding the declared
cycle orientation fixed reverses the certified winding sign. Relabeling
both the state and the oriented observation cycle instead leaves the
observation unchanged. There is no orientation-independent absolute
sign selected by this comparison.

The successful ramp still has a combined symmetry. The reflection
`p(i)=4-i mod 5` sends `x_+(0)` to `-x_+(0)`, and the complete law
also commutes with the global sign reversal `(x,theta)->(-x,-theta)`.
Their composition fixes the flat-phase preparation and is preserved by
uniqueness. Each operation separately reverses the oriented winding;
their composition preserves it, so this stabilizer is compatible with
the certified nonzero winding. The preparation breaks the pure
reflection that would force zero winding; it does not lack all
symmetries. The reader below checks pure node permutations only.

### What the comparison establishes

[`assess_sine_cycle_symmetry`](../../src/tnfr/physics/relational_sine_symmetry.py)
checks an admitted exact comparison, a supplied node permutation and a
supplied simple cycle. It validates the support automorphism, capacity
compatibility, raw full-state equalities and reversed cycle chain. The
[symmetry tests](../../tests/physics/test_relational_sine_symmetry.py)
combine this reusable restriction with the retained entry certificate
and the independently checked equal-budget preparations. An unsupported
symmetry does not imply acquisition, and the reader makes no conclusion
about a winding value on an antipodal boundary.

The budget supplies a necessary resource; the initial form supplies
information that breaks or preserves geometric symmetries. Both sources
start without phase winding. The complete reciprocal rows convert the
successful form preparation into actual phase winding, while the capture
theorem establishes its later retention and convergence. This does not
identify form with an initially present primitive phase winding.

Scalar budget alone therefore cannot predict the organization, and the
successful preparation is not evidence that the model autonomously
selects that preparation or a handedness from a reflection-fixed state.
The comparison supplies no preparation-occurrence law, support origin,
new constitutive selector or physical identification.


## 33. Retained relative origins permit a full-state composition handoff

<a id="sine-relative-frame-composition"></a>

This result connects the existing relative-state observation to Section 26's
capture theorem under the **same complete positive-loss sine law**. It does
not select that law. The [common-origin result](SINE_CONSTITUTIVE_INFORMATION.md#form-balance-common-origin)
explains why eliminating one global origin differs from independently
eliminating two origins before an interaction. Here the missing relative
information is declared explicitly and kept in a reusable state handoff.

### The admitted family and the exact offset map

Let A and B have disjoint ordered node sets, connected simple unit supports,
positive held capacities, identical admitted `e,w,beta>0` and one declared
structural clock. At a declared common observation/event time, each relative
source supplies nominal form and real phase lifts with per-node residuals:

\[
x_i=\bar x_i+C^x_A+\epsilon_i,\qquad
\theta_i=\bar\theta_i+C^\theta_A+\eta_i\quad(i\in A),
\]

and similarly on B, with `|epsilon_i|<=r_i`, `|eta_i|<=s_i`.
The common offsets can be unknown. These are admitted preparation/observation
premises, not an assertion that separate measurements were synchronized.
Declare exact relative **common-origin** differences

\[
\Delta_x=C^x_B-C^x_A,\qquad
\Delta_\theta=C^\theta_B-C^\theta_A.
\]

They are not actual reference-node differences: the latter also contain the
nominal reference difference and both reference residuals. A phase difference
uses a supplied consistent real lift; no unwrap or favorable branch is inferred.
Define the joint nominal rows by retaining A and adding `Delta_x,Delta_theta`
to every B node, then concatenate the original residual radii. Every member
of the two-source family is represented exactly by these rows plus **one**
common form and phase origin. Conversely each member of that joint residual
family splits into the two declared source families with those offsets.
No Cartesian relative-coordinate box or independently chosen nominal member
is substituted. This exact map relies on scalar known origin differences;
uncertain origin differences require a separately justified correlated or
outer-enclosure contract.

Rebasing a source's nominal form by `s_A` or `s_B` requires
`Delta_x -> Delta_x+s_A-s_B`, and likewise for phase. The joined family then
changes by just the common rebase `s_A`. Holding the offset fixed while
rebasing one component describes a different proposed contact.

### A supplied bridge changes weights and adds storage

Admit one unit bridge `a in A, b in B`, with no nodal reset and unchanged
capacities. Put `d_i^+=d_i^-+1_{i=a,b}`, `K_i^+=nu_i/d_i^+` and
`rho_i^+=d_i^+/nu_i`. The pressure and phase rows after the event must use
these new degrees. Reusing either isolated mobility gives the wrong field at
the bridge endpoints. For the weights and charges in a common frame,

\[
W^+=W_A+W_B+\frac1{\nu_a}+\frac1{\nu_b},\qquad
Q^+=Q_A+Q_B+\frac{x_a}{\nu_a}+\frac{x_b}{\nu_b}.
\]

This is event reweighting, not a continuous flux or violation of a held-support
invariant. Only the postevent weighted mean determines the eventual uniform
form if full-state capture holds. An unknown common form origin leaves the
absolute mean unknown; an interval for the representative with A's common
origin zero must be labelled as such. Lifted phase means have the analogous
frame dependence and are not additional circular observables.

The exact storage jump for each member is

\[
\boxed{\Delta E=\frac12(x_b-x_a)^2+
 \beta[1-\cos(\theta_b-\theta_a)]\ge0.}
\]

Internal edge costs are unchanged. The bridge form-gap interval is centered at
`bar(x_b)+Delta_x-bar(x_a)` with radius `r_a+r_b`; the phase-gap interval has
the corresponding phase center and radius `s_a+s_b`. Outward interval square
and cosine give a conservative jump bound. Rebuilding all internal and bridge
bounds gives a bound for the **whole** postevent storage. Bounds are
per-member statements about one family; subtracting two independently rounded
storage intervals is not needed to identify the event work.

A nonnegative supplied event-work allowance can be compared against this jump.
A certified within-allowance result does not establish capture or event
occurrence. Conversely, a postevent capture theorem does not supply the work
required to perform the event. Continuous storage dissipation is not an
undeclared reserve from which this jump can be paid.

### Missing relative information is a mathematical obstruction

If `Delta_x` is unspecified, a common shift of B leaves all of its internal
observations and isolated certificates unchanged while making the bridge form
cost arbitrarily large. If `Delta_theta` is unspecified, a relative rotation
can make the bridge antipodal. There is then no whole-family strict acute
admission. These obstructions concern what the sources determine; they do not
say that every particular contact fails. Missing offsets therefore remain
explicitly unavailable and never default to alignment.

Even known offsets and two isolated capture certificates are insufficient.
Take two P2 components with unit capacities, `beta=1`, zero phase and forms
`(-1,0)` and `(0,1)`. Each isolated energy is `1/2`, below its tree boundary
barrier 1. Join the two zero-form endpoints. The bridge jump is zero but total
postevent storage is 1, so the strict whole-tree energy criterion is
unavailable. Its failure is not a proof that the joined trajectory cannot
converge. It proves why isolated verdicts or bridge cost alone cannot replace
Section 26's full-state premises.

### Shared implementation and fixed controls

[`assess_sine_pattern_composition`](../../src/tnfr/physics/relational_sine_composition.py)
re-admits both `SineRelativePattern` sources and their authoritative model
coefficients, then rebuilds their fields from exact primitive rows. It shifts
only the right nominal coordinates, preserves every residual radius, joins
one supplied edge and rebuilds all joint fields through the shared relative
pattern owner. No graph-to-binary64 roundtrip changes an admitted rational
state. The existing sector owner receives this entire joined source and
explicit integer edge offsets; it checks every signed boundary face. Its
absolute weighted means remain unavailable, even when the frame-relative
representative means have finite bounds.

The [composition controls](../../tests/physics/test_relational_sine_composition.py)
fix two P2 sources with nominal zero form/phase, capacities `(1,2)` and `(3,4)`,
`e=w=1/2,beta=1`, bridge the inner endpoints and declare
`Delta_x=Delta_theta=1/4`. In the resulting P4 order,

\[
d^-=(1,1,1,1),\quad d^+=(1,2,2,1),\quad
W^-=25/12,\quad W^+=35/12,
\]
\[
Q^-_A=7/48,\quad Q^+_A=11/48,\quad
M^-_A=7/100,\quad M^+_A=11/140.
\]

Here the subscript A denotes the representative with A's common form origin
zero, not an absolute charge measurement. The jump is
`1/32+1-cos(1/4)<1`, so the full tree is captured without supplying an
equilibrium target. Missing-frame, large-phase-gap, full-energy and per-node
uncertainty controls test separate obligations. The implementation neither
installs a live bridge nor evolves a trajectory; it supplies the previously
missing preparation-to-capture handoff under a declared event and law.

A second fixed control joins two C5 sources with zero form, unit capacities,
phases `5*j/4` for `j=0,...,4`, zero relative common origins and a bridge
between their first nodes. A `-1` edge turn on each canonically oriented
closing edge retains unit winding in both cycles. The same whole-set reader
certifies capture against all 22 signed boundary faces of the joined support.
This checks composition of nonzero winding patterns; their preparation and
the occurrence of the bridge are still supplied premises.

## 34. Analytic acquisition can hand off to maintained composition

<a id="sine-prepared-composition"></a>

Sections 27 and 33 can be connected without replacing an analytic endpoint
by an independently supplied observation. Each component evolves on its own
declared support up to one common event time; a supplied unit bridge then
changes the support without resetting any nodal coordinate. The isolated
entry theorem proves the finite prefix. Only a new full-support certificate
proves maintenance after the event. This construction still assumes the
preparations, complete sine law and bridge event.

### Preserve the actual endpoint family and its clock

For component `c` retain its original relative preparation, positive held
capacities and the complete law of Section 27. Let `rho_ci=d_ci/nu_ci`,
`W_c=sum_i rho_ci`, and let its nominal weighted means be `mbar_c^x` and
`mbar_c^theta`. The memberwise conserved means, excluding the unknown common
origins, are

\[
m_c^x=\bar m_c^x+\frac{\sum_i\rho_{ci}\epsilon_{ci}}{W_c},\qquad
m_c^\theta=\bar m_c^\theta+\frac{\sum_i\rho_{ci}\eta_{ci}}{W_c}.
\]

Consequently the original residual radii imply intervals for these means;
using only the nominal means would lose admitted states. Section 27 provides
centered endpoint intervals `U_ci^x,U_ci^theta` and tighter internal edge
intervals for the **actual endpoint set**. Every endpoint has the form

\[
x_{ci}(T)=C_c^x+m_c^x+u_{ci}^x(T),\qquad
\theta_{ci}(T)=C_c^\theta+m_c^\theta+u_{ci}^\theta(T).
\]

The centered intervals use the original component degrees and weighted norm.
The common mean residual and centered errors need not be independent;
interval sums give safe outer bounds without asserting such independence.
The internal edge intervals retain their separate, stronger proof. They do
not describe every corner of the product of endpoint node intervals.

With declared initial structural times `t_A,t_B` and scaled durations
`tau_A,tau_B`, an instantaneous join requires

\[
T=t_A+\tau_A/e=t_B+\tau_B/e.
\]

Equality is checked from admitted original preparations, coefficients and
scaled durations, not cached horizon fields. Different starting times may
give the same endpoint time. Unequal endpoint times cannot be repaired by
relabelling an observation or silently holding the earlier state fixed.
Clock agreement is a preparation premise, not authenticated measurement data.

### Bound the bridge without discarding the isolated constraints

Declare exact right-minus-left common origins `Delta_x,Delta_theta` as in
Section 33. For either coordinate `q`, the bridge from `a in A` to `b in B`
has the interval bound

\[
q_b-q_a\in J_q:=\Delta_q+M_B^q-M_A^q+U_{Bb}^q-U_{Aa}^q,
\]

where `M_c^q` encloses the memberwise mean above. Retain every original
internal endpoint edge bound and add this one bridge bound. The actual
joint set is the product of the isolated flow images with the declared
frame relation, not a fabricated `SineRelativePattern` endpoint. The smooth
held-law flow supplies a nonempty image for every admitted preparation member.
These bounds therefore satisfy the shared sector core's producer contract.

The new degrees, mobility and exact event-work identity are those of Section
33. In particular, endpoint error bounds use `d^-`, whereas subsequent joint
dynamics uses `d^+`. The bridge work is bounded by
`J_x^2/2 + beta*(1-cos(J_theta))`. A supplied allowance is an independent
comparison; positive width of a work enclosure does not establish positive
actual work.

The post-event weighted mean also retains the old conserved inventory. In
the representative with A's common origin zero, put
`M_B^q <- M_B^q+Delta_q`. A valid bound is

\[
\boxed{
M_+^q\subseteq
\frac{(W_A+1/\nu_a)M_A^q+(W_B+1/\nu_b)M_B^q
      +U_{Aa}^q/\nu_a+U_{Bb}^q/\nu_b}
     {W_A+W_B+1/\nu_a+1/\nu_b}.}
\]

Here the inclusion means that every actual representative mean belongs to
the interval on the right. Summing independent node-box corners would forget
the conserved inventory. The absolute global origin remains unavailable;
the lifted phase mean is not an additional circular observable.

If the entire joined endpoint set is acute and its total storage upper bound
lies strictly below every full-support boundary face, Section 26 applies at
`T`. The complete post-event trajectory then remains in that sector and
converges on its new conserved-mean leaf. When both isolated preparations
also pass Section 27's zero-to-nonzero winding test, this proves a conditional
**acquisition -> supplied join -> maintenance** chain. Neither isolated
capture flags nor bridge work alone imply this conclusion.

### Shared reader and frozen analytic control

[`assess_sine_prepared_composition`](../../src/tnfr/physics/relational_sine_composition.py)
rebuilds both entry reports from their original `SineRelativePattern` sources,
scaled times and declared integer sectors. Zero residual radii are supported.
The report retains the initial times and derived common endpoint time, the
analytic source provenance, original edge bounds, mean intervals and the new
joint capture. It reuses the static composition's support/work accounting and
the shared sector core; it neither runs a solver nor installs a live edge.
Missing relative origins remain unavailable. Invalid declarations and unequal
derived endpoint times reject. A failed sufficient bound remains inconclusive.

The [handoff controls](../../tests/physics/test_relational_sine_prepared_composition.py)
freeze two relabelled C5 copies of Section 27's preparation:

\[
x_j(0)=4092(j-2),\quad\theta_j(0)=0,\quad\nu_j=1,
\quad e=1023/1024,\quad w=1/1024,\quad\beta=1.
\]

Both begin at zero with `tau=100`, so `T=102400/1023`. Nodes are ordered
`0,...,4` then `5,...,9`; the bridge joins ports `0,5`, and both relative
common origins are zero. The canonical edge turns are
`(0,-1,0,0,0,0,0,-1,0,0,0)`. No coefficients, preparation or horizon are fitted
to the returned result.

Outward rational bounds certify joint cycle periods `(1,1)` against all 22
signed boundary faces, with total storage below `6.912`, boundary lower bound
above `6.924` and a strict margin greater than `9/1000`. The bridge work
enclosure contains zero and has upper bound below `1/20000`. An allowance is
not inferred from those bounds. In fact the identical exact copies have equal
port trajectories by symmetry and uniqueness; the general enclosure does not
use that extra correlation, so its positive upper endpoint is conservative.

The total post-event weight is 22. Its representative form mean has a small
interval around zero, while its phase mean lies near `-8/(11*pi)` because
the new weights include the two port phases. No nodal reset causes that
reweighting. The controls also retain source uncertainty and reject cached
endpoint substitutions or mismatched times. This joins two existing theorems
under one explicit hybrid protocol; it does not derive when contacts occur
or identify the resulting patterns with physical matter.

## Section link directory

These aliases route existing citations to their substantive owner.

- <a id="smooth-sine-pattern-dynamics"></a>[Smooth-sine pattern dynamics](#smooth-sine-pattern-dynamics)

- <a id="28-exact-form-phase-and-memory-representations"></a>[28. Exact form, phase and memory representations](SINE_FORM_PHASE_REDUCTION.md#28-exact-form-phase-and-memory-representations)

- <a id="sine-form-phase-memory-equivalence"></a>[sine-form-phase-memory-equivalence](SINE_FORM_PHASE_REDUCTION.md#sine-form-phase-memory-equivalence)

- <a id="the-sum-coordinate-retains-the-full-joint-state"></a>[The sum coordinate retains the full joint state](SINE_FORM_PHASE_REDUCTION.md#the-sum-coordinate-retains-the-full-joint-state)

- <a id="exact-elimination-gives-second-order-phase-and-retained-memory"></a>[Exact elimination gives second-order phase and retained memory](SINE_FORM_PHASE_REDUCTION.md#exact-elimination-gives-second-order-phase-and-retained-memory)

- <a id="form-reconstruction-scope"></a>[form-reconstruction-scope](SINE_FORM_PHASE_REDUCTION.md#form-reconstruction-scope)

- <a id="what-an-emergent-form-hypothesis-must-distinguish"></a>[What an emergent-form hypothesis must distinguish](SINE_FORM_PHASE_REDUCTION.md#what-an-emergent-form-hypothesis-must-distinguish)

- <a id="the-nonlinear-storage-representation-contains-the-existing-resonance-pencil"></a>[The nonlinear storage representation contains the existing resonance pencil](SINE_FORM_PHASE_REDUCTION.md#the-nonlinear-storage-representation-contains-the-existing-resonance-pencil)

- <a id="a-phase-flat-preparation-distinguishes-the-full-law-from-phase-descent"></a>[A phase-flat preparation distinguishes the full law from phase descent](SINE_FORM_PHASE_REDUCTION.md#a-phase-flat-preparation-distinguishes-the-full-law-from-phase-descent)

- <a id="29-a-controlled-fast-form-and-slow-phase-comparison"></a>[29. A controlled fast-form and slow-phase comparison](SINE_FORM_PHASE_REDUCTION.md#29-a-controlled-fast-form-and-slow-phase-comparison)

- <a id="sine-controlled-slow-phase"></a>[sine-controlled-slow-phase](SINE_FORM_PHASE_REDUCTION.md#sine-controlled-slow-phase)

- <a id="complete-state-reference-initialization-and-clocks"></a>[Complete state, reference initialization and clocks](SINE_FORM_PHASE_REDUCTION.md#complete-state-reference-initialization-and-clocks)

- <a id="explicit-composite-phase-and-form-bounds"></a>[Explicit composite, phase and form bounds](SINE_FORM_PHASE_REDUCTION.md#explicit-composite-phase-and-form-bounds)

- <a id="uniform-finite-time-meaning-and-the-retained-storage-budget"></a>[Uniform finite-time meaning and the retained storage budget](SINE_FORM_PHASE_REDUCTION.md#uniform-finite-time-meaning-and-the-retained-storage-budget)

- <a id="means-uncertain-sources-and-phase-potential"></a>[Means, uncertain sources and phase potential](SINE_FORM_PHASE_REDUCTION.md#means-uncertain-sources-and-phase-potential)

- <a id="shared-certificate-and-bounded-numerical-controls"></a>[Shared certificate and bounded numerical controls](SINE_FORM_PHASE_REDUCTION.md#shared-certificate-and-bounded-numerical-controls)

- <a id="30-full-state-capture-from-controlled-phase-geometry"></a>[30. Full-state capture from controlled phase geometry](SINE_FORM_PHASE_REDUCTION.md#30-full-state-capture-from-controlled-phase-geometry)

- <a id="sine-slow-phase-capture-handoff"></a>[sine-slow-phase-capture-handoff](SINE_FORM_PHASE_REDUCTION.md#sine-slow-phase-capture-handoff)

- <a id="a-proved-reference-neighborhood-from-the-preparation"></a>[A proved reference neighborhood from the preparation](SINE_FORM_PHASE_REDUCTION.md#a-proved-reference-neighborhood-from-the-preparation)

- <a id="the-actual-endpoint-and-its-complete-storage"></a>[The actual endpoint and its complete storage](SINE_FORM_PHASE_REDUCTION.md#the-actual-endpoint-and-its-complete-storage)

- <a id="handoff-to-the-whole-sector-theorem"></a>[Handoff to the whole-sector theorem](SINE_FORM_PHASE_REDUCTION.md#handoff-to-the-whole-sector-theorem)

- <a id="fixed-analytic-controls-and-scope"></a>[Fixed analytic controls and scope](SINE_FORM_PHASE_REDUCTION.md#fixed-analytic-controls-and-scope)

- <a id="35-conservative-exchange-can-produce-finite-regional-winding"></a>[35. Conservative exchange can produce finite regional winding](SINE_REGIONAL_FORMATION.md#35-conservative-exchange-can-produce-finite-regional-winding)

- <a id="conservative-regional-winding-entry"></a>[conservative-regional-winding-entry](SINE_REGIONAL_FORMATION.md#conservative-regional-winding-entry)

- <a id="a-global-full-law-phase-enclosure"></a>[A global full-law phase enclosure](SINE_REGIONAL_FORMATION.md#a-global-full-law-phase-enclosure)

- <a id="one-frozen-two-port-preparation"></a>[One frozen two-port preparation](SINE_REGIONAL_FORMATION.md#one-frozen-two-port-preparation)

- <a id="sine-regional-storage-balance"></a>[sine-regional-storage-balance](SINE_REGIONAL_FORMATION.md#sine-regional-storage-balance)

- <a id="regional-work-retains-the-cross-boundary-storage"></a>[Regional work retains the cross-boundary storage](SINE_REGIONAL_FORMATION.md#regional-work-retains-the-cross-boundary-storage)

- <a id="conservative-regional-phase-transport"></a>[conservative-regional-phase-transport](SINE_REGIONAL_FORMATION.md#conservative-regional-phase-transport)

- <a id="acute-winding-requires-internal-phase-redistribution"></a>[Acute winding requires internal phase redistribution](SINE_REGIONAL_FORMATION.md#acute-winding-requires-internal-phase-redistribution)

- <a id="the-complete-rows-already-transmit-phase-motion-to-the-inner-path"></a>[The complete rows already transmit phase motion to the inner path](SINE_REGIONAL_FORMATION.md#the-complete-rows-already-transmit-phase-motion-to-the-inner-path)

- <a id="conservative-regional-work-retention"></a>[conservative-regional-work-retention](SINE_REGIONAL_FORMATION.md#conservative-regional-work-retention)

- <a id="a-finite-regional-work-budget-can-retain-an-admitted-acute-state"></a>[A finite regional work budget can retain an admitted acute state](SINE_REGIONAL_FORMATION.md#a-finite-regional-work-budget-can-retain-an-admitted-acute-state)

- <a id="conservative-regional-organization-control"></a>[conservative-regional-organization-control](SINE_REGIONAL_FORMATION.md#conservative-regional-organization-control)

- <a id="a-bounded-full-state-organization-control"></a>[A bounded full-state organization control](SINE_REGIONAL_FORMATION.md#a-bounded-full-state-organization-control)

- <a id="controls-implementation-and-limits"></a>[Controls, implementation and limits](SINE_REGIONAL_FORMATION.md#controls-implementation-and-limits)

- <a id="sine-regional-channel-accessibility"></a>[sine-regional-channel-accessibility](SINE_REGIONAL_FORMATION.md#sine-regional-channel-accessibility)

- <a id="36-regional-channel-balance-and-phase-accessibility"></a>[36. Regional channel balance and phase accessibility](SINE_REGIONAL_FORMATION.md#36-regional-channel-balance-and-phase-accessibility)

- <a id="separate-internal-conversion-from-boundary-input"></a>[Separate internal conversion from boundary input](SINE_REGIONAL_FORMATION.md#separate-internal-conversion-from-boundary-input)

- <a id="equal-storage-does-not-determine-the-initial-phase-response"></a>[Equal storage does not determine the initial phase response](SINE_REGIONAL_FORMATION.md#equal-storage-does-not-determine-the-initial-phase-response)

- <a id="a-running-phase-barrier-for-any-acute-unit-winding-entry"></a>[A running phase barrier for any acute unit-winding entry](SINE_REGIONAL_FORMATION.md#a-running-phase-barrier-for-any-acute-unit-winding-entry)

- <a id="shared-implementation-and-evidence-boundary"></a>[Shared implementation and evidence boundary](SINE_REGIONAL_FORMATION.md#shared-implementation-and-evidence-boundary)

- <a id="retrospective-channel-diagnosis-of-the-frozen-control"></a>[Retrospective channel diagnosis of the frozen control](SINE_REGIONAL_FORMATION.md#retrospective-channel-diagnosis-of-the-frozen-control)

- <a id="37-distributed-source-geometry-can-produce-finite-acute-winding"></a>[37. Distributed source geometry can produce finite acute winding](SINE_REGIONAL_FORMATION.md#37-distributed-source-geometry-can-produce-finite-acute-winding)

- <a id="sine-conservative-source-geometry"></a>[sine-conservative-source-geometry](SINE_REGIONAL_FORMATION.md#sine-conservative-source-geometry)

- <a id="the-environmental-source-image-uses-full-degrees"></a>[The environmental source image uses full degrees](SINE_REGIONAL_FORMATION.md#the-environmental-source-image-uses-full-degrees)

- <a id="a-source-image-plus-a-global-remainder-gives-actual-finite-passage"></a>[A source image plus a global remainder gives actual finite passage](SINE_REGIONAL_FORMATION.md#a-source-image-plus-a-global-remainder-gives-actual-finite-passage)

- <a id="one-supplied-distributed-preparation"></a>[One supplied distributed preparation](SINE_REGIONAL_FORMATION.md#one-supplied-distributed-preparation)

- <a id="the-same-storage-and-mean-do-not-select-the-acquired-winding"></a>[The same storage and mean do not select the acquired winding](SINE_REGIONAL_FORMATION.md#the-same-storage-and-mean-do-not-select-the-acquired-winding)

- <a id="shared-admission-and-scope"></a>[Shared admission and scope](SINE_REGIONAL_FORMATION.md#shared-admission-and-scope)

- <a id="38-low-regional-storage-does-not-close-the-retention-handoff"></a>[38. Low regional storage does not close the retention handoff](SINE_REGIONAL_FORMATION.md#38-low-regional-storage-does-not-close-the-retention-handoff)

- <a id="sine-conservative-handoff-obstruction"></a>[sine-conservative-handoff-obstruction](SINE_REGIONAL_FORMATION.md#sine-conservative-handoff-obstruction)

- <a id="admit-the-actual-full-support-velocity-ramp"></a>[Admit the actual full-support velocity ramp](SINE_REGIONAL_FORMATION.md#admit-the-actual-full-support-velocity-ramp)

- <a id="acquired-entry-near-the-storage-minimum"></a>[Acquired entry near the storage minimum](SINE_REGIONAL_FORMATION.md#acquired-entry-near-the-storage-minimum)

- <a id="strict-finite-exit-and-a-necessary-positive-work-transfer"></a>[Strict finite exit and a necessary positive work transfer](SINE_REGIONAL_FORMATION.md#strict-finite-exit-and-a-necessary-positive-work-transfer)

- <a id="zero-instantaneous-work-does-not-remove-the-missing-information"></a>[Zero instantaneous work does not remove the missing information](SINE_REGIONAL_FORMATION.md#zero-instantaneous-work-does-not-remove-the-missing-information)

- <a id="relative-rhythm-means-compatible-phase-rates-not-identical-phases"></a>[Relative rhythm means compatible phase rates, not identical phases](SINE_REGIONAL_FORMATION.md#relative-rhythm-means-compatible-phase-rates-not-identical-phases)

- <a id="shared-admission-and-boundary-of-the-result"></a>[Shared admission and boundary of the result](SINE_REGIONAL_FORMATION.md#shared-admission-and-boundary-of-the-result)

- <a id="39-compatible-internal-geometry-and-collective-phase-dynamics"></a>[39. Compatible internal geometry and collective phase dynamics](SINE_COLLECTIVE_PHASE_DYNAMICS.md#39-compatible-internal-geometry-and-collective-phase-dynamics)

- <a id="sine-phase-offset-partition"></a>[sine-phase-offset-partition](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-phase-offset-partition)

- <a id="a-declared-partition-with-fixed-internal-phase-offsets"></a>[A declared partition with fixed internal phase offsets](SINE_COLLECTIVE_PHASE_DYNAMICS.md#a-declared-partition-with-fixed-internal-phase-offsets)

- <a id="necessity-and-sufficiency-use-both-consumed-rows"></a>[Necessity and sufficiency use both consumed rows](SINE_COLLECTIVE_PHASE_DYNAMICS.md#necessity-and-sufficiency-use-both-consumed-rows)

- <a id="reciprocity-inherited-storage-and-conserved-means"></a>[Reciprocity, inherited storage and conserved means](SINE_COLLECTIVE_PHASE_DYNAMICS.md#reciprocity-inherited-storage-and-conserved-means)

- <a id="one-moving-realization-on-the-retained-private-leaf-support"></a>[One moving realization on the retained private-leaf support](SINE_COLLECTIVE_PHASE_DYNAMICS.md#one-moving-realization-on-the-retained-private-leaf-support)

- <a id="exact-compatibility-is-not-finite-time-formation-into-the-family"></a>[Exact compatibility is not finite-time formation into the family](SINE_COLLECTIVE_PHASE_DYNAMICS.md#exact-compatibility-is-not-finite-time-formation-into-the-family)

- <a id="shared-admission-distinguishes-proof-from-undecided-equality"></a>[Shared admission distinguishes proof from undecided equality](SINE_COLLECTIVE_PHASE_DYNAMICS.md#shared-admission-distinguishes-proof-from-undecided-equality)

- <a id="40-a-finite-width-window-around-compatible-moving-geometry"></a>[40. A finite-width window around compatible moving geometry](SINE_COLLECTIVE_PHASE_DYNAMICS.md#40-a-finite-width-window-around-compatible-moving-geometry)

- <a id="sine-moving-pattern-window"></a>[sine-moving-pattern-window](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-moving-pattern-window)

- <a id="reference-motion-error-chart-and-declared-duration"></a>[Reference motion, error chart and declared duration](SINE_COLLECTIVE_PHASE_DYNAMICS.md#reference-motion-error-chart-and-declared-duration)

- <a id="the-moving-reference-relative-storage-identity"></a>[The moving-reference relative-storage identity](SINE_COLLECTIVE_PHASE_DYNAMICS.md#the-moving-reference-relative-storage-identity)

- <a id="internal-identity-and-actual-work-remain-compatible"></a>[Internal identity and actual work remain compatible](SINE_COLLECTIVE_PHASE_DYNAMICS.md#internal-identity-and-actual-work-remain-compatible)

- <a id="conserved-storage-and-backward-time-restrict-acquisition-claims"></a>[Conserved storage and backward time restrict acquisition claims](SINE_COLLECTIVE_PHASE_DYNAMICS.md#conserved-storage-and-backward-time-restrict-acquisition-claims)

- <a id="41-actual-contact-motion-can-exchange-storage-with-internal-structure"></a>[41. Actual contact motion can exchange storage with internal structure](SINE_COLLECTIVE_PHASE_DYNAMICS.md#41-actual-contact-motion-can-exchange-storage-with-internal-structure)

- <a id="sine-collective-pulse-transfer"></a>[sine-collective-pulse-transfer](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-collective-pulse-transfer)

- <a id="mean-contact-rows-from-the-complete-state"></a>[Mean-contact rows from the complete state](SINE_COLLECTIVE_PHASE_DYNAMICS.md#mean-contact-rows-from-the-complete-state)

- <a id="the-storage-split-retains-its-phase-lift-dependence"></a>[The storage split retains its phase-lift dependence](SINE_COLLECTIVE_PHASE_DYNAMICS.md#the-storage-split-retains-its-phase-lift-dependence)

- <a id="a-phase-flat-preparation-has-a-discriminating-fourth-derivative"></a>[A phase-flat preparation has a discriminating fourth derivative](SINE_COLLECTIVE_PHASE_DYNAMICS.md#a-phase-flat-preparation-has-a-discriminating-fourth-derivative)

- <a id="one-phase-flat-preparation-above-the-necessary-phase-path-budget"></a>[One phase-flat preparation above the necessary phase-path budget](SINE_COLLECTIVE_PHASE_DYNAMICS.md#one-phase-flat-preparation-above-the-necessary-phase-path-budget)

- <a id="retained-full-law-control-excludes-acute-winding-on-its-declared-horizon"></a>[Retained full-law control excludes acute winding on its declared horizon](SINE_COLLECTIVE_PHASE_DYNAMICS.md#retained-full-law-control-excludes-acute-winding-on-its-declared-horizon)

- <a id="a-structural-alternative-separates-storage-release-from-acquisition"></a>[A structural alternative separates storage release from acquisition](SINE_COLLECTIVE_PHASE_DYNAMICS.md#a-structural-alternative-separates-storage-release-from-acquisition)

- <a id="42-a-larger-cycle-sector-gives-a-global-conservative-barrier"></a>[42. A larger cycle sector gives a global conservative barrier](SINE_REGIONAL_FORMATION.md#42-a-larger-cycle-sector-gives-a-global-conservative-barrier)

- <a id="sine-cycle-sector-barrier"></a>[sine-cycle-sector-barrier](SINE_REGIONAL_FORMATION.md#sine-cycle-sector-barrier)

- <a id="the-boundary-minimum-is-exactly-72"></a>[The boundary minimum is exactly `7/2`](SINE_REGIONAL_FORMATION.md#the-boundary-minimum-is-exactly-72)

- <a id="a-conserved-full-budget-separates-the-sector-in-both-directions"></a>[A conserved full budget separates the sector in both directions](SINE_REGIONAL_FORMATION.md#a-conserved-full-budget-separates-the-sector-in-both-directions)

- <a id="negative-collective-feedback-can-still-lie-below-this-barrier"></a>[Negative collective feedback can still lie below this barrier](SINE_REGIONAL_FORMATION.md#negative-collective-feedback-can-still-lie-below-this-barrier)

- <a id="compatible-conserved-quantities-still-do-not-determine-reachability"></a>[Compatible conserved quantities still do not determine reachability](SINE_REGIONAL_FORMATION.md#compatible-conserved-quantities-still-do-not-determine-reachability)

- <a id="43-fast-contact-motion-can-exclude-finite-receiver-organization"></a>[43. Fast contact motion can exclude finite receiver organization](SINE_REGIONAL_FORMATION.md#43-fast-contact-motion-can-exclude-finite-receiver-organization)

- <a id="sine-contact-averaging"></a>[sine-contact-averaging](SINE_REGIONAL_FORMATION.md#sine-contact-averaging)

- <a id="the-full-rows-bound-the-accumulated-contact-currents"></a>[The full rows bound the accumulated contact currents](SINE_REGIONAL_FORMATION.md#the-full-rows-bound-the-accumulated-contact-currents)

- <a id="an-exact-change-of-variables-retains-the-nonlinear-receiver"></a>[An exact change of variables retains the nonlinear receiver](SINE_REGIONAL_FORMATION.md#an-exact-change-of-variables-retains-the-nonlinear-receiver)

- <a id="a-prospective-finite-exclusion-despite-a-large-total-budget"></a>[A prospective finite exclusion despite a large total budget](SINE_REGIONAL_FORMATION.md#a-prospective-finite-exclusion-despite-a-large-total-budget)

- <a id="44-a-distributed-preparation-has-a-robust-full-state-passage"></a>[44. A distributed preparation has a robust full-state passage](SINE_REGIONAL_FORMATION.md#44-a-distributed-preparation-has-a-robust-full-state-passage)

- <a id="robust-conservative-passage"></a>[robust-conservative-passage](SINE_REGIONAL_FORMATION.md#robust-conservative-passage)

- <a id="direct-full-row-bounds-compose-into-an-independent-target-box"></a>[Direct full-row bounds compose into an independent target box](SINE_REGIONAL_FORMATION.md#direct-full-row-bounds-compose-into-an-independent-target-box)

- <a id="an-exact-distributed-source-and-a-fixed-short-window"></a>[An exact distributed source and a fixed short window](SINE_REGIONAL_FORMATION.md#an-exact-distributed-source-and-a-fixed-short-window)

- <a id="what-this-passage-establishes-and-what-it-does-not"></a>[What this passage establishes and what it does not](SINE_REGIONAL_FORMATION.md#what-this-passage-establishes-and-what-it-does-not)

- <a id="45-complete-contact-feedback-distinguishes-cancellation-from-rigidity"></a>[45. Complete contact feedback distinguishes cancellation from rigidity](SINE_COLLECTIVE_PHASE_DYNAMICS.md#45-complete-contact-feedback-distinguishes-cancellation-from-rigidity)

- <a id="sine-receiver-rigidity"></a>[sine-receiver-rigidity](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-receiver-rigidity)

- <a id="sine-relative-phase-feedback"></a>[sine-relative-phase-feedback](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-relative-phase-feedback)

- <a id="the-complete-centered-rows-and-their-derivatives"></a>[The complete centered rows and their derivatives](SINE_COLLECTIVE_PHASE_DYNAMICS.md#the-complete-centered-rows-and-their-derivatives)

- <a id="vanishing-first-and-second-rates-do-not-imply-a-fixed-geometry"></a>[Vanishing first and second rates do not imply a fixed geometry](SINE_COLLECTIVE_PHASE_DYNAMICS.md#vanishing-first-and-second-rates-do-not-imply-a-fixed-geometry)

- <a id="exact-rigidity-has-no-additional-moving-compensation-family"></a>[Exact rigidity has no additional moving compensation family](SINE_COLLECTIVE_PHASE_DYNAMICS.md#exact-rigidity-has-no-additional-moving-compensation-family)

- <a id="a-retained-second-order-description-exposes-the-same-feedback"></a>[A retained second-order description exposes the same feedback](SINE_COLLECTIVE_PHASE_DYNAMICS.md#a-retained-second-order-description-exposes-the-same-feedback)

- <a id="consequence-for-the-longer-retention-question"></a>[Consequence for the longer retention question](SINE_COLLECTIVE_PHASE_DYNAMICS.md#consequence-for-the-longer-retention-question)

- <a id="46-a-conserved-budget-bounds-finite-relative-phase-travel"></a>[46. A conserved budget bounds finite relative-phase travel](SINE_CONSERVATIVE_PREPARATION.md#46-a-conserved-budget-bounds-finite-relative-phase-travel)

- <a id="sine-energy-speed-retention"></a>[sine-energy-speed-retention](SINE_CONSERVATIVE_PREPARATION.md#sine-energy-speed-retention)

- <a id="an-edge-speed-bound-from-the-actual-full-form-storage"></a>[An edge-speed bound from the actual full form storage](SINE_CONSERVATIVE_PREPARATION.md#an-edge-speed-bound-from-the-actual-full-form-storage)

- <a id="first-exit-certificate-including-full-state-uncertainty"></a>[First-exit certificate, including full-state uncertainty](SINE_CONSERVATIVE_PREPARATION.md#first-exit-certificate-including-full-state-uncertainty)

- <a id="an-explicit-target-with-uncertainty-in-all-twenty-coordinates"></a>[An explicit target with uncertainty in all twenty coordinates](SINE_CONSERVATIVE_PREPARATION.md#an-explicit-target-with-uncertainty-in-all-twenty-coordinates)

- <a id="what-this-closes-and-what-it-leaves-open"></a>[What this closes, and what it leaves open](SINE_CONSERVATIVE_PREPARATION.md#what-this-closes-and-what-it-leaves-open)

- <a id="a-directed-preparation-for-a-possible-reverse-time-check"></a>[A directed preparation for a possible reverse-time check](SINE_CONSERVATIVE_PREPARATION.md#a-directed-preparation-for-a-possible-reverse-time-check)

- <a id="47-a-reversible-enclosure-can-certify-an-independent-formation-source"></a>[47. A reversible enclosure can certify an independent formation source](SINE_CONSERVATIVE_PREPARATION.md#47-a-reversible-enclosure-can-certify-an-independent-formation-source)

- <a id="sine-reversible-preparation"></a>[sine-reversible-preparation](SINE_CONSERVATIVE_PREPARATION.md#sine-reversible-preparation)

- <a id="complete-state-clock-and-global-error-bound"></a>[Complete state, clock and global error bound](SINE_CONSERVATIVE_PREPARATION.md#complete-state-clock-and-global-error-bound)

- <a id="from-a-reverse-endpoint-to-a-forward-source-ball"></a>[From a reverse endpoint to a forward source ball](SINE_CONSERVATIVE_PREPARATION.md#from-a-reverse-endpoint-to-a-forward-source-ball)

- <a id="numerical-admission-is-part-of-the-composition"></a>[Numerical admission is part of the composition](SINE_CONSERVATIVE_PREPARATION.md#numerical-admission-is-part-of-the-composition)

- <a id="frozen-bounded-evaluation-and-its-scope"></a>[Frozen bounded evaluation and its scope](SINE_CONSERVATIVE_PREPARATION.md#frozen-bounded-evaluation-and-its-scope)

- <a id="retained-response-the-directed-checkpoint-does-not-supply-a-zero-winding-source"></a>[Retained response: the directed checkpoint does not supply a zero-winding source](SINE_CONSERVATIVE_PREPARATION.md#retained-response-the-directed-checkpoint-does-not-supply-a-zero-winding-source)

- <a id="48-an-exact-signreflection-reduction-retains-the-sector-saddle"></a>[48. An exact sign/reflection reduction retains the sector saddle](SINE_CONSERVATIVE_PREPARATION.md#48-an-exact-signreflection-reduction-retains-the-sector-saddle)

- <a id="sine-involution-saddle-reduction"></a>[sine-involution-saddle-reduction](SINE_CONSERVATIVE_PREPARATION.md#sine-involution-saddle-reduction)

- <a id="exact-invariant-family-and-reconstruction"></a>[Exact invariant family and reconstruction](SINE_CONSERVATIVE_PREPARATION.md#exact-invariant-family-and-reconstruction)

- <a id="inherited-storage-and-oriented-winding"></a>[Inherited storage and oriented winding](SINE_CONSERVATIVE_PREPARATION.md#inherited-storage-and-oriented-winding)

- <a id="the-full-saddles-unique-hyperbolic-pair-lies-in-this-family"></a>[The full saddle's unique hyperbolic pair lies in this family](SINE_CONSERVATIVE_PREPARATION.md#the-full-saddles-unique-hyperbolic-pair-lies-in-this-family)

- <a id="a-finite-nonlinear-envelope-for-the-tangent-comparison"></a>[A finite nonlinear envelope for the tangent comparison](SINE_CONSERVATIVE_PREPARATION.md#a-finite-nonlinear-envelope-for-the-tangent-comparison)

- <a id="a-nonlinear-local-sector-passage-with-an-explicit-error-budget"></a>[A nonlinear local sector passage with an explicit error budget](SINE_CONSERVATIVE_PREPARATION.md#a-nonlinear-local-sector-passage-with-an-explicit-error-budget)

- <a id="what-the-reduction-does-not-remove"></a>[What the reduction does not remove](SINE_CONSERVATIVE_PREPARATION.md#what-the-reduction-does-not-remove)

- <a id="49-a-directed-nonlinear-corridor-reaches-the-winding-seam"></a>[49. A directed nonlinear corridor reaches the winding seam](SINE_CONSERVATIVE_PREPARATION.md#49-a-directed-nonlinear-corridor-reaches-the-winding-seam)

- <a id="sine-directed-saddle-corridor"></a>[sine-directed-saddle-corridor](SINE_CONSERVATIVE_PREPARATION.md#sine-directed-saddle-corridor)

- <a id="the-principal-seam-has-a-lower-minimum-than-the-sector-saddle"></a>[The principal seam has a lower minimum than the sector saddle](SINE_CONSERVATIVE_PREPARATION.md#the-principal-seam-has-a-lower-minimum-than-the-sector-saddle)

- <a id="exact-transverse-storage-and-a-directed-momentum"></a>[Exact transverse storage and a directed momentum](SINE_CONSERVATIVE_PREPARATION.md#exact-transverse-storage-and-a-directed-momentum)

- <a id="a-uniform-force-bound-throughout-the-corridor"></a>[A uniform force bound throughout the corridor](SINE_CONSERVATIVE_PREPARATION.md#a-uniform-force-bound-throughout-the-corridor)

- <a id="a-directional-exit-condition-keeps-the-lower-face-closed"></a>[A directional exit condition keeps the lower face closed](SINE_CONSERVATIVE_PREPARATION.md#a-directional-exit-condition-keeps-the-lower-face-closed)

- <a id="why-this-exit-changes-winding-and-what-it-leaves-open"></a>[Why this exit changes winding, and what it leaves open](SINE_CONSERVATIVE_PREPARATION.md#why-this-exit-changes-winding-and-what-it-leaves-open)

- <a id="50-conservative-formation-and-retention-on-the-same-full-state-orbit"></a>[50. Conservative formation and retention on the same full-state orbit](SINE_CONSERVATIVE_PREPARATION.md#50-conservative-formation-and-retention-on-the-same-full-state-orbit)

- <a id="sine-conservative-formation-retention"></a>[sine-conservative-formation-retention](SINE_CONSERVATIVE_PREPARATION.md#sine-conservative-formation-retention)

- <a id="one-exact-preparation-supplies-both-corridor-admissions"></a>[One exact preparation supplies both corridor admissions](SINE_CONSERVATIVE_PREPARATION.md#one-exact-preparation-supplies-both-corridor-admissions)

- <a id="outer-and-inner-nonlinear-connections"></a>[Outer and inner nonlinear connections](SINE_CONSERVATIVE_PREPARATION.md#outer-and-inner-nonlinear-connections)

- <a id="an-explicit-acute-band-supplies-a-unit-of-retained-evolution"></a>[An explicit acute band supplies a unit of retained evolution](SINE_CONSERVATIVE_PREPARATION.md#an-explicit-acute-band-supplies-a-unit-of-retained-evolution)

- <a id="positive-widths-include-all-twenty-form-and-phase-coordinates"></a>[Positive widths include all twenty form and phase coordinates](SINE_CONSERVATIVE_PREPARATION.md#positive-widths-include-all-twenty-form-and-phase-coordinates)

- <a id="51-rational-preparation-and-retained-metric-propagation"></a>[51. Rational preparation and retained-metric propagation](SINE_CONSERVATIVE_PREPARATION.md#51-rational-preparation-and-retained-metric-propagation)

- <a id="sine-operational-saddle-preparation"></a>[sine-operational-saddle-preparation](SINE_CONSERVATIVE_PREPARATION.md#sine-operational-saddle-preparation)

- <a id="a-rational-preparation-must-retain-the-same-finite-gates"></a>[A rational preparation must retain the same finite gates](SINE_CONSERVATIVE_PREPARATION.md#a-rational-preparation-must-retain-the-same-finite-gates)

- <a id="why-independent-coordinate-boxes-lose-essential-cancellations"></a>[Why independent coordinate boxes lose essential cancellations](SINE_CONSERVATIVE_PREPARATION.md#why-independent-coordinate-boxes-lose-essential-cancellations)

- <a id="an-exact-rational-metric-retains-all-relative-coordinates"></a>[An exact rational metric retains all relative coordinates](SINE_CONSERVATIVE_PREPARATION.md#an-exact-rational-metric-retains-all-relative-coordinates)

- <a id="a-finite-neighborhood-nonlinear-bound"></a>[A finite-neighborhood nonlinear bound](SINE_CONSERVATIVE_PREPARATION.md#a-finite-neighborhood-nonlinear-bound)

- <a id="conversion-to-primitive-coordinates-and-execution-boundary"></a>[Conversion to primitive coordinates and execution boundary](SINE_CONSERVATIVE_PREPARATION.md#conversion-to-primitive-coordinates-and-execution-boundary)

- <a id="a-validated-step-retains-the-metric-radius"></a>[A validated step retains the metric radius](SINE_CONSERVATIVE_PREPARATION.md#a-validated-step-retains-the-metric-radius)

- <a id="sine-execution-with-retained-full-state-uncertainty"></a>[Sine execution with retained full-state uncertainty](SINE_CONSERVATIVE_PREPARATION.md#sine-execution-with-retained-full-state-uncertainty)

- <a id="certifying-growth-on-a-full-picard-tube-beyond-the-saddle-neighborhood"></a>[Certifying growth on a full Picard tube beyond the saddle neighborhood](SINE_CONSERVATIVE_PREPARATION.md#certifying-growth-on-a-full-picard-tube-beyond-the-saddle-neighborhood)

- <a id="retained-finite-connection-and-its-preparation-boundary"></a>[Retained finite connection and its preparation boundary](SINE_CONSERVATIVE_PREPARATION.md#retained-finite-connection-and-its-preparation-boundary)

- <a id="an-inverse-inclusion-test-separates-an-image-from-a-preparation"></a>[An inverse-inclusion test separates an image from a preparation](SINE_CONSERVATIVE_PREPARATION.md#an-inverse-inclusion-test-separates-an-image-from-a-preparation)

- <a id="sine-constitutive-robustness"></a>[sine-constitutive-robustness](SINE_CONSERVATIVE_PREPARATION.md#sine-constitutive-robustness)

- <a id="52-a-constitutive-change-preserves-equilibria-but-blocks-the-same-sources-formation"></a>[52. A constitutive change preserves equilibria but blocks the same source's formation](SINE_CONSERVATIVE_PREPARATION.md#52-a-constitutive-change-preserves-equilibria-but-blocks-the-same-sources-formation)

- <a id="the-complete-changed-law-and-fixed-source"></a>[The complete changed law and fixed source](SINE_CONSERVATIVE_PREPARATION.md#the-complete-changed-law-and-fixed-source)

- <a id="the-exact-changed-acquisition-barrier"></a>[The exact changed acquisition barrier](SINE_CONSERVATIVE_PREPARATION.md#the-exact-changed-acquisition-barrier)

- <a id="applying-the-barrier-to-the-retained-source"></a>[Applying the barrier to the retained source](SINE_CONSERVATIVE_PREPARATION.md#applying-the-barrier-to-the-retained-source)

- <a id="the-complete-critical-geometry-and-its-inertia-remain-unchanged"></a>[The complete critical geometry and its inertia remain unchanged](SINE_CONSERVATIVE_PREPARATION.md#the-complete-critical-geometry-and-its-inertia-remain-unchanged)

- <a id="the-sufficient-error-transport-method-has-a-separate-role"></a>[The sufficient error-transport method has a separate role](SINE_CONSERVATIVE_PREPARATION.md#the-sufficient-error-transport-method-has-a-separate-role)
