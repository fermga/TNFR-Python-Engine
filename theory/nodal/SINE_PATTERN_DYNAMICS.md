# Smooth-sine pattern dynamics

**Scope:** conditional geometry, capture, preparation, memory, reduction and
symmetry results for the complete smooth reciprocal normalized-sine law.
The supplied support, coefficients, held capacities and retained initial state
remain part of each theorem. Capacity, loss, uncertainty and phase-domain
hypotheses are stated with their proofs; their conclusions do not transfer
automatically to another complete law or a discrete execution path.

The native neighbor-argument model, its connection and event analysis, and
its reflected-domain formation obstructions remain in
[Composition, interaction and formation of coherent regions](RELATIONAL_PATTERN_COMPOSITION.md).
[Pattern memory](RELATIONAL_PATTERN_MEMORY.md) retains the complementary
model-specific interaction, observability, recovery and receiver results.
The exact representations below preserve hidden initialization rather than
turning an observation into a complete state.

Sections 24–32 retain their published numbers so existing section citations
remain valid. Sections 1–23 and compatibility links remain in the composition
owner. Definitions, complete-law hypotheses and proofs live here; the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
continues to own the research queue. No result below selects support birth,
a preparation-occurrence law, a unique constitutive law or physical identity.

## Reading map

| Question | Sections | Retained boundary |
| --- | --- | --- |
| Which component geometries compose? | [24: bridge trees](#sine-bridge-tree-composition), [25: cycle periods and currents](#sine-cycle-sector-compatibility) | Equilibrium compatibility and stability do not imply entry from a preparation. |
| When is a full state captured? | [26: whole-sector capture](#sine-target-free-sector-capture), [30: controlled-reference handoff](#sine-slow-phase-capture-handoff) | Form storage, every boundary face and the actual endpoint set remain explicit. |
| Can supplied form acquire phase winding? | [27: prepared entry and uncertainty](#sine-prepared-sector-entry) | The initial form budget and memberwise conserved origins are retained. |
| What is preserved by eliminating or approximating form? | [28: exact state and memory representations](#sine-form-phase-memory-equivalence), [29: finite slow-time comparison](#sine-controlled-slow-phase) | Exact equivalence, transient corrections and controlled approximation are distinct claims. |
| What do budget and preparation geometry permit? | [31: fixed-budget consensus](#sine-budget-consensus), [32: equal-budget symmetry discriminator](#sine-equal-budget-preparation) | Necessary resources do not select organization; symmetry exclusions retain their branch and preparation premises. |

## 24. Equilibrium geometry of components joined by bridges

<a id="sine-bridge-tree-composition"></a>

The nine attracting geometries of the
[eleven-node sine model](RELATIONAL_PATTERN_MEMORY.md#sine-eleven-node-equilibrium-stability)
are an instance of a more general composition rule. This section concerns
the **smooth reciprocal sine law**, not the native Arg law in the
[reflected-domain analyses](RELATIONAL_PATTERN_COMPOSITION.md#regular-seeded-reachability-audit).
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
[full reciprocal Jacobian argument](RELATIONAL_PATTERN_MEMORY.md#sine-eleven-node-equilibrium-stability),
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
The [exact C5 classification](RELATIONAL_PATTERN_MEMORY.md#sine-eleven-node-equilibrium-stability)
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

The [circulation and reconstruction theorem](../FORCED_SUPPORT_BALANCE.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods)
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
[return-path equilibrium](RELATIONAL_PATTERN_MEMORY.md#return-path-equilibrium)
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
These are the [existing complete-law identities](RELATIONAL_PATTERN_MEMORY.md#sine-relative-pattern-state),
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
[whole-support recovery owner](RELATIONAL_PATTERN_MEMORY.md#sine-interacting-recovery).
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

## 28. Exact form, phase and memory representations

<a id="sine-form-phase-memory-equivalence"></a>

The completed composition, capture and prepared-entry results share more
than their graph. They use the same reciprocal exchange, conserved means
and phase potential. The following identities connect those results to
the existing memory and resonance owners without introducing a new
primitive coordinate, deleting the initial form information or assuming
a slow phase approximation.

Retain finite connected simple unit support with at least two nodes, positive
held capacities, `e,w,beta>0`, the declared clock and no forcing or events. Use
`A=KL`, `M=K^-1`, `W=1^T M1`, and the weighted centering projector
`P_M=I-1*(1^T M)/W`. The means `mu_x` and `mu_theta` are those of the
actual state; the latter belongs to a chosen continuous phase lift.
Uncertain relative sources retain one conserved pair of means per member,
with their absolute common origins unobserved. Neither mean is replaced
by a nominal value in the identities below.

### The sum coordinate retains the full joint state

In Section 27's variables `tau=e*t`, `z=(b/e)P_Mx` and
`eta=ab/e^2`, define the mixed coordinate `y=theta+z`. The exact rows become

\[
\boxed{\qquad
\theta'=A(y-\theta),\qquad y'=\eta KS(\theta).
\qquad}
\]

This change is invertible on consistent real lifts:
`z=y-theta`, `x=mu_x*1+(e/b)z`, with `1^T Mz=0`. Thus it retains both
consumed coordinates and the conserved form mean. A change of circular
representative sends `(theta,y)` to `(theta+2*pi*m,y+2*pi*m)` for the
**same** integer vector `m`; their real difference is unchanged.
Equivalently, retain real `z` and circular `exp(i*y)`, from which
`exp(i*theta)=exp(i*y)*exp(-i*z)` is reconstructed.

Treating `theta` and `y` as two independently wrapped phase vectors loses
information. On the unit P2 with unit capacities, `K=I`. The preparations
`theta=0,z=0` and `theta=0,z=(2*pi,-2*pi)` give the same two separately
wrapped phase vectors, but their scaled phase rates are respectively
`0` and `A z=(4*pi,-4*pi)`. The real form contrast has been discarded by
that wrapping, although it is consumed by the phase row.

Nor does `y` alone close the dynamics. On the same P2, `y=0` can arise
from `theta=z=0`, giving `y'=0`, or from
`theta=(delta,-delta),z=(-delta,delta)`, giving
`y'=eta*(-sin(2*delta),sin(2*delta))`. For `0<delta<pi/4` these rates
differ even on the same conserved-mean leaf. A first-order law depending
only on `y` therefore needs an additional approximation or restricted
invariant family.

The cancellation behind this coordinate is already used operationally in
the [hidden-state inverse](RELATIONAL_PATTERN_MEMORY.md#sine-hidden-state-observability):
`dot(x)+(e/b)*dot(theta)=aKS` removes the form-gradient contribution
from paired rate observations. Its time derivative supplies the paired
acceleration channel there. The present transformation and the entry
estimate reorganize that same complete-law identity; they do not select
a new pressure law from those observations.

### Exact elimination gives second-order phase and retained memory

In the original clock, differentiate the complete phase row and use the
form row. With the actual ordered matrix products this gives

\[
\boxed{\ddot\theta+eA\dot\theta=abAKS(\theta),\qquad
\dot\theta(0)=bAP_Mx(0).}
\]

The constraint `1^T M dot(theta)=0` must remain. An unconstrained
second-order equation on all phase lifts would also permit a constant
common phase velocity, which the original positive-capacity law does not
supply. On the weighted-mean-zero space, `A` is invertible and the phase
velocity reconstructs the full centered form as
`P_Mx=b^(-1)A^(-1)dot(theta)`. The constant form mean must be retained
separately if the original absolute form is needed. This is a
same-information representation, not a removal of a state variable.

Alternatively, variation of constants and integration give the exact
nonlinear Volterra equation

\[
\boxed{\begin{aligned}
\theta(t)={}&\theta(0)
+\frac be\left(I-e^{-eAt}\right)P_Mx(0)\\
&+\frac{ab}{e}\int_0^t
\left(I-e^{-eA(t-s)}\right)KS(\theta(s))\,ds.
\end{aligned}}
\]

The source term is the retained initial form, not an external input.
The phase current in the integral uses the actual earlier phase state;
this is not a fixed linear convolution in `theta`. Differentiating the
identity with its supplied initial state recovers the second-order row,
and reconstructing form recovers the original joint system. Global
Lipschitz continuity of the full sine field supplies uniqueness. The
projector could be omitted only from the first source term because
`I-exp(-eAt)` annihilates the common form mode; displaying it makes the
mean accounting explicit.

No commutation of `K` with `L` is used. In particular `AK=KLK`, whereas
`KA=K^2L` is generally different. The corresponding phase-velocity
memory operator is

\[
\mathcal R_e(s)=A e^{-eAs}K
 =K^{1/2}B e^{-eBs}K^{1/2},\qquad
B=K^{1/2}LK^{1/2}.
\]

It is symmetric positive semidefinite for each nonnegative lag, but its
individual entries need not be nonnegative. Spectral integration gives

\[
\int_0^\infty\mathcal R_e(s)\,ds
=\frac1e\left(K-\frac{\mathbf1\mathbf1^{\mathsf T}}W\right).
\]

Acting on a sine-current vector, whose ordinary sum is zero, this
integrated operator gives `KS/e`. This explains the frozen-history gain
behind a candidate slow phase law; it does **not** bound the error made
by replacing the moving history by its current value. It also does not
remove the initial source or justify exchanging an infinite-time limit
with a constitutive limit.

The [single-intermediary sine owner](RELATIONAL_PATTERN_MEMORY.md#causal-sine-environmental-pressure)
already derives a nonlinear second-order representation and derivative-free
Volterra memory, including the hidden initial state and moving visible
boundary. The [general elimination owner](../DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state)
owns the variation-of-constants principle and its source obligation.
The identities here specialize that principle to eliminating the entire
form coordinate of this complete nonlinear sine model. They neither turn
the existing signed intermediary kernel into a positive-entry kernel nor
transfer a pure-diffusion or conservative-memory conclusion between models.

### The nonlinear storage representation contains the existing resonance pencil

Remove the conserved origins, and put

\[
h=K^{-1/2}\mathbf1,\quad\mathcal Q=h^\perp,\quad
\xi=K^{-1/2}P_Mx,\quad
\vartheta=K^{-1/2}P_M\theta,\quad
\Phi(\vartheta)=V(K^{1/2}\vartheta).
\]

The symmetric matrix `B` above is positive definite on `Q`; all inverse
matrices in this paragraph act only there. Since
`grad(Phi)=-K^(1/2)S`, the exact nonlinear transformed rows and their
equivalent second-order equation are

\[
\dot\xi=-eB\xi-a\nabla\Phi(\vartheta),\qquad
\dot\vartheta=bB\xi,\qquad
B^{-1}\ddot\vartheta+e\dot\vartheta
+ab\nabla\Phi(\vartheta)=0.
\]

Their storage and loss are still the original quantities:

\[
b^2E=\frac12\dot\vartheta^{\mathsf T}B^{-1}\dot\vartheta
+ab\Phi(\vartheta),\qquad
\frac{d}{dt}(b^2E)=-e\|\dot\vartheta\|^2.
\]

The velocity contribution is the original form storage in different
coordinates. The derived quotient matrix `B^-1` depends on the graph and
held capacities; this supplies no identification of capacity with inverse
physical mass. Initial velocity, nonlinear phase history and the conserved
origins remain part of the model.

At a critical phase geometry with Hessian `H`, the Hessian of `Phi` is
`C=K^(1/2)HK^(1/2)`. Linearizing the displayed equation gives precisely
the [existing full-support resonance pencil](RESONANCE_FOUNDATIONS.md#resonance-tangent),
`B^-1*ddot(vartheta)+e*dot(vartheta)+ab*C*vartheta=0` for perturbations.
The matrices `B` and `C` need not commute, so scalar cycle discriminants
still cannot classify a general attachment. Strict acute-sector convexity,
equilibrium recovery and the full storage capture barrier use the same
phase curvature; the nonnegative velocity/form term is why a phase-only
storage check cannot replace the capture certificate.

### A phase-flat preparation distinguishes the full law from phase descent

Take equal initial phases and any nonconstant initial form on the retained
connected graph. Then `S(0)=0`, `H(0)=L` and

\[
V(0)=\dot V(0)=0,\qquad
\ddot V(0)=b^2(Ax(0))^{\mathsf T}L(Ax(0))>0,
\qquad
\dot E(0)=-e(Lx(0))^{\mathsf T}K(Lx(0))<0.
\]

For strict positivity, `Ax(0)` has zero weighted mean. If it were a
constant vector, it would vanish; positivity of `K` and connectedness
would then force `x(0)` constant, contrary to preparation. Thus phase
potential initially increases while total storage decreases. The form
reservoir and reciprocal phase row account for both signs under one law.

In contrast, the candidate phase-gradient flow `dpsi/dsigma=KS(psi)`
initialized at that same uniform `theta(0)` remains there by uniqueness.
It would discard the prepared form information. Initializing a proposed
comparison from `theta(0)+z(0)` retains that information, but the exact
identities alone do not prove closeness to this comparison flow. Section 29
derives the finite-horizon error bound while retaining the initial transient.
A later capture handoff, support creation and physical identification remain
separate obligations.

The [full-law equivalence controls](../../tests/physics/test_relational_sine_exchange_equivalence.py)
differentiate the shared field with interval jets, check the two opposite
storage signs and the failure of a sum-coordinate-only closure, and retain
noncommuting capacity/Laplacian products. Independent numerical quadrature
checks the nonlinear memory identity along a complete trajectory; it is an
equivalence check, not a reserved-response prediction or the analytic proof.

## 29. A controlled fast-form and slow-phase comparison

<a id="sine-controlled-slow-phase"></a>

The exact representations in Section 28 permit a quantitative comparison
with a first-order phase flow. This is an approximation theorem for the
same complete sine law, with an explicit transient and error. It does not
install that reference flow as a replacement pressure or remove the
prepared form information.

### Complete state, reference initialization and clocks

Fix the finite connected unit support, positive held `K`, positive `beta`
and positive complete-law coefficients `e,w`. Retain Section 27's
`M=K^-1`, `A=KL`, `alpha=b/e`, `z=alpha*P_Mx`, `tau=e*t` and
`eta=ab/e^2=beta*alpha^2`. For the actual initial state write `z_0=z(0)`
and take a proved bound `Z>=||z_0||_M`. Define the slow time and reference
on consistent phase lifts by

\[
\sigma=\eta\tau=\frac{ab}{e}t,\qquad
\frac{d\psi}{d\sigma}=f(\psi),\qquad
f(\theta)=KS(\theta),\qquad
\psi(0)=\theta(0)+z_0.
\]

Both the complete and reference fields are globally Lipschitz, so their
solutions exist uniquely for all finite times. The reference is initialized
with the actual retained form contribution, not only the initial phase.
As in Section 28, the mixed coordinate uses real `z_0` and consistent
lifts; wrapping its two summands independently would discard information.

Let `lambda>0` be a certified weighted quotient gap for `A`, and set

\[
F=\left(\sum_i\nu_i d_i\right)^{1/2},\qquad
\ell=2\max_i\nu_i.
\]

The earlier forcing bound gives `||f(theta)||_M<=F` globally. Its derivative
is `-KH(theta)`, where `H` is the cosine-weighted phase Hessian. The edge
formula gives `-B<=K^(1/2)H(theta)K^(1/2)<=B` for
`B=K^(1/2)LK^(1/2)`. The similar matrix `KL` has Gershgorin intervals
`[0,2*nu_i]`, so the symmetric matrix `B` has norm at most `ell`.
Integrating the derivative
along a segment therefore proves the global Lipschitz estimate
`||f(u)-f(v)||_M<=ell*||u-v||_M`. This bound needs no acute phase domain
for either path or the segment between them.

### Explicit composite, phase and form bounds

For a supplied finite `sigma>=0`, put `tau=sigma/eta` and
`D(tau)=exp(-A*tau)`. Define

\[
\boxed{\begin{aligned}
R(\tau)&=\frac{\eta F}{\lambda}
                  \left(1-e^{-\lambda\tau}\right),\\
C(\sigma)&=
\frac{\eta(\ell Z+F)}{\lambda+\eta\ell}
           \left(e^{\ell\sigma}-e^{-\lambda\tau}\right).
\end{aligned}}
\]

Then the exact complete solution and its reference satisfy

\[
\boxed{\begin{aligned}
\|z(\tau)-D(\tau)z_0\|_M&\le R(\tau),\\
\|\theta(\tau)+D(\tau)z_0-\psi(\sigma)\|_M&\le C(\sigma),\\
\|\theta(\tau)-\psi(\sigma)\|_M
&\le Ze^{-\lambda\tau}+C(\sigma).
\end{aligned}}
\]

The middle quantity is the **composite correction**. It retains the fast
initial form contribution instead of calling the corrected angle the
actual phase. For `y=theta+z`, the same estimates also give
`||y(tau)-psi(sigma)||_M<=C(sigma)+R(tau)`.

The form estimate follows from the exact variation-of-constants identity

\[
z(\tau)=D(\tau)z_0+
\eta\int_0^\tau D(\tau-s)f(\theta(s))\,ds.
\]

All integrand vectors have zero weighted mean. Thus the quotient semigroup
bound `||D(s)||_M<=exp(-lambda*s)` applies to them and gives `R`.
Section 28's exact phase memory, now in scaled time, gives

\[
\theta(\tau)+D(\tau)z_0
=\theta(0)+z_0+
\eta\int_0^\tau\left[I-D(\tau-s)\right]f(\theta(s))\,ds.
\]

Subtract the reference integral equation. If
`q(tau)=theta(tau)+D(tau)z_0-psi(eta*tau)`, the forcing and Lipschitz
bounds imply

\[
\|q(\tau)\|_M\le
\eta\ell\int_0^\tau\|q(s)\|_M\,ds+
\frac{\eta(\ell Z+F)}{\lambda}
                \left(1-e^{-\lambda\tau}\right).
\]

Here the `ell*Z` term retains the initial transient inside
`theta-psi=q-D(s)z_0`; the `F` term bounds the remaining memory integral.
The scalar comparison function with zero initial value solves
`c'=eta*ell*c+eta*(ell*Z+F)*exp(-lambda*tau)`. Its explicit solution is
the displayed `C`, proving the composite bound by Gronwall's inequality.
The uncorrected phase bound then follows by the triangle inequality.
No phase linearization, commuting `K,L` assumption or sampled response
enters this argument.

Conversion back to the original form coordinate is also explicit:

\[
\left\|P_Mx(t)-e^{-eAt}P_Mx(0)\right\|_M
\le\frac ae\frac F\lambda
              \left(1-e^{-\lambda\tau}\right)
=\sqrt{\beta\eta}\frac F\lambda
              \left(1-e^{-\lambda\tau}\right).
\]

Thus the original form remainder after its homogeneous transient is
`O(sqrt(eta))` at fixed support, `K` and `beta`. A small scaled `z`
remainder alone would not establish this original-coordinate statement.
The homogeneous form term remains until its own transient has decayed.

### Uniform finite-time meaning and the retained storage budget

Fix a finite slow-time horizon `Sigma`, the support, `K`, `beta` and a
uniform bound `Z` on the scaled initial form. For `0<=sigma<=Sigma`,

\[
C(\sigma)\le
\eta\frac{\ell Z+F}{\lambda}e^{\ell\Sigma}.
\]

This is a uniform `O(eta)` composite comparison including the initial
instant. It is not uniform `O(eta)` proximity of the actual phase from
that instant: exactly `theta(0)-psi(0)=-z_0`. For any fixed
`sigma_0>0`, the uncorrected estimate on `[sigma_0,Sigma]` instead has
the additional transient `Z*exp(-lambda*sigma_0/eta)`. More generally,
the finite displayed inequalities decide whether a chosen transient
is sufficiently small, without retuning the requested horizon.

The original-clock comparison time is

\[
t=\frac{\sigma}{e\eta}
  =\beta\pi^2\frac e{w^2}\sigma.
\]

As `eta` changes, the constitutive ratio changes; this is not just a common
clock rescaling. Keeping a nonzero scaled preparation fixed also retains
the original initial storage

\[
E(0)=\frac{\beta}{2\eta}z_0^{\mathsf T}Lz_0
                       +\beta V(\theta(0)).
\]

Its form term grows as `1/eta`. For uncertain preparations the expression
and its enclosing budget apply to every actual member, rather than to its
nominal source alone.

### Means, uncertain sources and phase potential

Since `1^T M f=1^T S=0`, the reference conserves its weighted phase mean.
Its initialization has the same weighted phase mean as the complete
trajectory because `z_0` is centered. Common form shifts cancel from
`z_0`; common phase shifts move the actual and reference solutions
together. All comparison differences and the composite correction have
zero weighted mean on the selected lift.

For a `SineRelativePattern`, reuse Section 27's original residual set and
its weighted initial norm bound. Each member is compared with **its own**
reference initialized at that member's `theta(0)+z_0`. A uniform `Z`
makes the displayed error uniform over the family but does not replace
these references by a single nominal reference trajectory. Doing that
would require a separate bound on the initialization differences.
Unknown absolute origins and memberwise conserved means remain explicit.

The potential satisfies the global bound
`|V(u)-V(v)|<=F*||u-v||_M`, since
`||grad(V)||_(M^-1)=||KS||_M<=F`. Hence

\[
|V(\theta(\tau))-V(\psi(\sigma))|
\le F\left[Ze^{-\lambda\tau}+C(\sigma)\right].
\]

The reference itself obeys
`dV(psi)/dsigma=-S(psi)^T K S(psi)<=0`. This does not make the actual
phase potential monotone: Section 28's phase-flat, nonconstant-form
preparation has strictly positive initial second phase-potential derivative
while total storage decreases. Initializing the reference only at that
uniform phase would leave it at consensus and discard the preparation
mechanism; the shifted initialization and explicit transient resolve that
mismatch in the comparison theorem.

### Shared certificate and bounded numerical controls

[`bound_sine_slow_phase`](../../src/tnfr/physics/relational_sine_reduction.py)
accepts an admitted exact or relative source and a supplied rational slow
time. It reuses the complete-law preparation admission, weighted norm and
exact reversible gap owners, and returns outward analytic comparison
bounds. The scaled and original times are enclosed directly from their
mathematical-pi formulas; these are enclosures of the declared comparison
instant, not a new class of independently uncertain times. The reader runs
neither the full trajectory nor the reference phase solver. It does not
issue an endpoint capture or final-basin verdict.

The [reduction tests](../../tests/physics/test_relational_sine_reduction.py)
freeze a small heterogeneous support with edges
`(0,1),(1,2),(2,3),(0,2)`, capacities `(1,3/2,2,5/4)` and `beta=2`.
The primary constructor weight ratio is `16:1`; the stored normalized
coefficients remain authoritative, with exact ratio `w/e=1/16`.
The declared preparation is
`x=(beta*e/w)*(1/2,-3/4,5/4,-1)` and
`theta=(1/8,-1/4,1/2,-3/8)`, with slow horizon `1/16`.
A separately fixed `8:1` comparison retains the same scaled initial form
and checks the feedback-parameter dependence. Independent SciPy integrations
of the complete and reference rows cross-check the inequalities under the
test's explicit numerical settings; they are not validated trajectory
enclosures or a search over sources, ratios or horizons. The analytic
certificate does not depend on their sampled responses.

The error grows with the declared slow horizon and is not an infinite-time
theorem. It proves neither a final basin match nor an exchange of the
limits `eta->0` and `t->infinity`. Section 30 supplies a separate capture
criterion that admits the actual full-state endpoint, including the form
remainder and all sector margins. The result controls a mechanism within the supplied law
and preparation; it does not select that law, support, source budget or
physical interpretation.

## 30. Full-state capture from controlled phase geometry

<a id="sine-slow-phase-capture-handoff"></a>

Section 29 bounds the difference from a reference flow but does not by
itself locate that reference at the requested endpoint. A sufficient
handoff needs both a justified reference neighborhood and the remaining
original form storage. The following construction supplies these from the
admitted preparation and an exactly verified stationary phase geometry.
It requires neither a supplied response endpoint nor a reference solver.

### A proved reference neighborhood from the preparation

Retain all hypotheses and notation of Section 29. Supply exact rational
node turns `q_i`, in the source node order, and put `phi_*=2*pi*q` on the
selected real lift. Require every principal target edge angle
`delta_*e` to lie strictly between `-pi/2` and `pi/2`, and require
`S(phi_*)=0`. The shared target owner verifies these hypotheses using the
[exact acute geometry and sine cancellation](../../src/tnfr/physics/relational_sine_recovery.py);
a floating residual close to zero does not prove criticality. This
implemented algebraic admission is sufficient, not a classification of
all possible sine equilibria.

For each actual preparation member, shift `phi_*` by a common constant so
that its weighted mean equals that of `theta(0)`. Write the resulting lift
as `phi_*^m`. Its phase geometry and potential `V_*` are unchanged.
The reference `psi(0)=theta(0)+alpha*P_Mx(0)` has this same weighted mean.
Choose a proved bound

\[
D_0\ge
\left\|P_M\left(\theta(0)+\alpha P_Mx(0)-\phi_*\right)\right\|_M.
\]

Since `f(phi_*^m)=0`, the constant curve `phi_*^m` solves the reference
equation. The global Lipschitz bound from Section 29 and Gronwall give

\[
\|\psi(\sigma)-\phi_*^m\|_M
\le D_{\rm ref}(\sigma):=e^{\ell\sigma}D_0.
\]

This estimate assumes no contraction or acute evolution of the reference.
The supplied stationary geometry is a proof reference; it is not inserted
into the complete law or assigned to the evolving state.

For a relative source, retain exactly Section 27's original residual
family. If its componentwise form and phase error radii are
`epsilon_xi,epsilon_thetai`, a uniform choice is

\[
\begin{aligned}
D_0={}&
\left\|P_M\left(\theta_{\rm nom}
                +\alpha P_Mx_{\rm nom}-\phi_*\right)\right\|_M\\
&+\left(\sum_i m_i\epsilon_{\theta i}^2\right)^{1/2}
+\alpha\left(\sum_i m_i\epsilon_{x i}^2\right)^{1/2},
\qquad m_i=M_{ii}.
\end{aligned}
\]

This follows from the triangle inequality and the contractivity of the
weighted orthogonal projection `P_M`. Every member keeps its own
reference, mean-matched target and conserved means. Unknown common
origins cancel before bounding the mismatch; no absolute phase or form
origin is inferred. The estimate uses the selected consistent phase
lifts, with the chart and quotient distinctions from Section 28.

### The actual endpoint and its complete storage

At the supplied slow time `sigma`, set `tau=sigma/eta` and define

\[
\begin{aligned}
Q(\sigma)&=Ze^{-\lambda\tau}+C(\sigma),\\
\rho(\sigma)&=D_{\rm ref}(\sigma)+Q(\sigma),\\
X(\sigma)&=e^{-\lambda\tau}N_x+
\frac ae\frac F\lambda\left(1-e^{-\lambda\tau}\right),
\qquad N_x\ge\|P_Mx(0)\|_M.
\end{aligned}
\]

Section 29 proves that the actual endpoint satisfies
`||theta(t)-phi_*^m||_M<=rho` and `||P_Mx(t)||_M<=X`, where
`t=sigma/(e*eta)`. The phase bound includes the initial transient;
substituting only the composite error `C` would not prove this statement.
The form bound is in the original coordinate, with its conversion factor
retained.

For an edge `ij`, write `k_i=K_ii` and
`g_ij=sqrt(k_i+k_j)`. Weighted duality gives

\[
\left|(\theta_j-\theta_i)
       -(\phi_{*j}^m-\phi_{*i}^m)\right|\le g_{ij}\rho,
\qquad |x_j-x_i|\le g_{ij}X.
\]

Thus a sufficient strict acute margin on every edge is

\[
\boxed{\quad |\delta_{*ij}|+g_{ij}\rho<\frac\pi2
\quad\text{for every retained edge}.\quad}
\]

Use the integer edge offsets that reduce the supplied target lift to its
principal acute angles. They place every enclosed actual endpoint in the
target's cycle-period sector, without inferring a future unwrapping from
samples.

A useful storage bound retains the correlations lost by independent edge
intervals. Let `h=theta(t)-phi_*^m`. The phase Hessian satisfies `H<=L`
globally, since every edge cosine is at most one. Taylor's integral
formula and the exact criticality of the target therefore give

\[
V(\phi_*^m+h)
=V_*+\int_0^1(1-s)h^{\mathsf T}H(\phi_*^m+sh)h\,ds
\le V_*+\frac12h^{\mathsf T}Lh
\le V_*+\frac\ell2\rho^2.
\]

The linear term vanishes because `grad V(phi_*^m)=0`. The segment need
not remain acute for this upper bound. Likewise
`x^T Lx<=ell*||P_Mx||_M^2`, so the complete storage obeys

\[
\boxed{\qquad
E(t)\le E_{\rm handoff}:=
\frac\ell2X^2+\beta\left(V_*+\frac\ell2\rho^2\right).
\qquad}
\]

This uses the same cancellation principle as local recovery, now with a
preparation-derived phase neighborhood. Bounding every cosine separately
can lose that cancellation and fail to establish a margin that the
correlated bound proves. The admitted endpoint set retains the norm and
storage constraints together with the edge intervals and memberwise mean
leaf; arbitrary corners of the independent outer intervals need not obey
the correlated storage bound. Global existence of the complete law makes
the set nonempty for every admitted preparation member.

### Handoff to the whole-sector theorem

Let `B_k` be the shared certified lower bound for the phase potential on
**every** boundary face of the target sector, as in Section 26. If all
strict edge margins above hold and

\[
\boxed{\qquad E_{\rm handoff}<\beta B_k,\qquad}
\]

then Section 26 applies to the actual full-state endpoint of every source
member. Subsequent complete evolution stays in that sector and converges
on its conserved-mean leaf to its unique acute equilibrium. The admitted
target already supplies the stationary geometry in that sector, so
uniqueness identifies the limiting phase geometry with it. The theorem
does not assume that geometry was the initial winding or that the
reference and actual paths shared their earlier sectors.

[`certify_sine_slow_capture`](../../src/tnfr/physics/relational_sine_reduction.py)
recomputes the preparation and slow comparison, admits the exact target,
and passes producer-proved edge and correlated storage bounds to the
shared sector-capture owner. The outer report retains the enclosed
original endpoint time; its nested capture is neither a timestamped
observation nor a sampled forecast. Outward rational arithmetic encloses
the pi, trigonometric, norm and exponential quantities. Malformed inputs
are rejected; insufficient strict margins return unavailable. An
unavailable sufficient condition proves neither instability nor
impossibility.

### Fixed analytic controls and scope

The [handoff tests](../../tests/physics/test_relational_sine_slow_capture.py)
reuse Section 27's published C5 preparation unchanged:
`nu_i=beta=1`, `e=1023/1024`, `w=1/1024`,
`x_i(0)=4092*(i-2)` and `theta_i(0)=0` for `i=0,...,4`.
The supplied target is `q_i=(i-2)/5` and the fixed slow time is `1/16`.
Its initial form budget remains `160*1023^2`; neither that budget nor the
ratio is retuned to obtain the handoff. The analytic bounds give

\[
\begin{aligned}
D_0&<0.074249,&\rho&<0.084137,\\
\frac\ell2X^2&<0.000002028,&
E_{\rm handoff}&<3.461996083,\\
B_k&>3.469266270,&
\beta B_k-E_{\rm handoff}&>0.00727018.
\end{aligned}
\]

All edge margins are positive. The same previously fixed independent
radii `epsilon_x=1/16` and `epsilon_theta=1/65536` also pass, with total
storage below `3.462017026` and margin above `0.00724924`. These are
analytic endpoint certificates, with no integrated reference or complete
trajectory used as their producer. Initial zero winding is separately
known for these preparations, so their captured nonzero sector also
establishes acquisition. The general API does not assume zero initial
winding and therefore does not label every capture an acquisition.

Here the declared original time is approximately `646182.7393` structural
units, obtained from `t=beta*pi^2*e*sigma/w^2`. This fixed handoff is not
an optimized entry time or a replacement for the earlier `tau=100` entry
certificate. Controls with resolved phase but excessive remaining form
storage, and with unresolved phase margins, remain unavailable under
their respective sufficient conditions.

The result connects one justified reference geometry to complete-state
capture at a finite declared instant. It supplies neither autonomous
support creation, selection of the preparation or target, a unique
constitutive law, nor a physical identification. It also does not turn
the finite comparison into a uniform infinite-time approximation.

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
the [sine orientation argument](RELATIONAL_PATTERN_MEMORY.md#mediator-orientation-scope)
and the [reflection obstruction to receiver formation](RELATIONAL_PATTERN_MEMORY.md#sine-interacting-recovery);
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
