# Mediated connections and pattern detachment

Sections 14–16 distinguish causal influence, restoring interaction and
support birth. Exact hidden-state elimination retains initial state and
memory; local recovery and detachment use their stated native-law premises.
The results assume fine support and do not create it autonomously.

Part of [Native regional composition](RELATIONAL_PATTERN_COMPOSITION.md).
Section numbers remain stable across this collection. Each result keeps
its full hypotheses, implementation and checks; the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone assigns research work.

<a id="connection-mechanisms-and-mediators"></a>
## 14. Connection mechanisms: mediation, reinforcement and primitive birth

The [physical binding comparisons](../PHYSICAL_REGIME_CORRESPONDENCES.md#physical-binding-and-interaction)
motivate a precise distinction. A channel of influence, a dynamically bound
configuration and a primitive support event are different model claims.
Existing interactions can organize a composite without creating a new
fundamental interaction. This does not explain the origin of the fine support.

### A mediated effective connection follows from existing nodal transport

Reuse [exact hidden-state elimination](../DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state)
on unit P3, `A--M--B`, with common held capacity `nu>0` and pure-EPI pressure.
Observe `y=(x_A,x_B)` and hide `h=x_M`. The actual normalized nodal law gives

\[
\dot y=-\nu y+\nu\mathbf1h,\qquad
\dot h=\frac\nu2\mathbf1^Ty-\nu h.
\]

Eliminating the mediator gives the exact endpoint law

\[
\dot y(t)=-\nu y(t)+\nu e^{-\nu t}\mathbf1h(0)
+\int_0^t\frac{\nu^2}{2}e^{-\nu(t-s)}
\mathbf1\mathbf1^Ty(s)\,ds.
\]

Its off-diagonal memory term transmits influence without a direct A--B edge.
For an initial donor-only difference `(delta x_A,delta h,delta x_B)=(a,0,0)`,

\[
\delta x_B(t)=\frac a4(1-e^{-\nu t})^2
=\frac{a\nu^2}{4}t^2+O(t^3).
\]

The recipient's initial rate difference is zero; its acceleration difference
is `a*nu^2/2`.
Removing the mediator paths removes that influence. An instantaneous direct
edge with positive conductance instead gives a nonzero initial recipient rate
difference for `a!=0`. This is a bounded analytic discriminator, not a new
simulation campaign, autonomous edge birth or a bound-state theorem.

Replacing the hidden row by `h=(x_A+x_B)/2` gives a direct reduced coupling,
but here it is exact only on that invariant preparation manifold; the donor-
only control is outside it. The
[minimal realization](../DERIVED_EPI_MEMORY.md#11-minimal-linear-state-retaining-a-declared-observation)
requires three linear coordinates for arbitrary preparations: endpoint
observation rows have rank two, and their first generator products raise the
rank to three. The partition-based `observe_epi_memory` API cannot be called
with the mediator omitted from its partition. Reuse the elimination algebra
or the general linear-realization owner instead. The older unnormalized Kron
example has a different capacity law, as that document already specifies.

The [joint mediator extension](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction)
now uses the existing nonlinear form/phase law on two C5 regions linked through
one nodal intermediary. It derives a two-coordinate tangent memory, a
capacity-dependent transient and a direct-versus-mediated onset discriminator;
the existing local theorem admits recovery of the supplied joint geometry.
This extends the mechanism beyond pure diffusion without explaining primitive
support origin or replacing a finite capture study.

<a id="mediated-restoring-geometry"></a>
### The same phase geometry induces a restoring relation through a mediator

The [local recovery theorem](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
uses the Hessian of the existing phase storage, with edge weights
`k_ij=cos(delta_ij)>0` at an acute equilibrium. These are derived curvatures,
not new conductances or physical spring constants. Restrict its quadratic
form to a two-edge path `A--M--B`, holding regional shapes fixed and allowing
phase offsets `a,b` at its endpoints and `h` at its mediator. Its contribution is

\[
Q(a,h,b)=\frac\beta2\big[k_1(h-a)^2+k_2(b-h)^2\big].
\]

Completing the square gives the exact quadratic identity

\[
Q=\frac{\beta(k_1+k_2)}2(h-h_*)^2
+\frac{\beta k_{\rm eff}}2(b-a)^2,\qquad
h_* = \frac{k_1a+k_2b}{k_1+k_2},\quad
k_{\rm eff}=\frac{k_1k_2}{k_1+k_2}>0.
\]

Thus eliminating a hidden coordinate in a **static constrained minimum**
leaves positive curvature in the endpoints' relative phase despite no direct
edge. A common offset costs nothing. Other routes and internal deformations
retain their own contributions; this two-edge expression is not the complete
stiffness of the return-path graph.

For an aligned unit path, `k_1=k_2=1`, so `k_eff=1/2`. More than the local
quadratic term is available here. With lifted endpoint difference
`d=b-a` satisfying `abs(d)<pi` and both path gaps acute, its exact constrained
phase-storage minimum is

\[
\min_h\beta[2-\cos(h-a)-\cos(b-h)]
=2\beta[1-\cos(d/2)]
=\frac\beta4d^2+O(d^4).
\]

Indeed the expression before minimization is
`2*beta*[1-cos(d/2)*cos(h-(a+b)/2)]`; its unique minimum in this acute lift
is at the midpoint. This calculation restricts the existing storage rather
than selecting another pressure law or introducing a synchronization threshold.

Static minimization is **not dynamic elimination**. Endpoints and ring shapes
are fixed only for this calculation; that constrained family need not be
invariant under the actual flow. Neither `h=h_*` nor a direct endpoint law
may replace the mediator's form and phase rows without another argument.
The [derived memory](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction)
retains those rows, their initial state and their capacity. In particular a
zero-capacity mediator can stay away from this constrained minimum: geometric
curvature does not by itself make an inactive node move.
The [native mediation controls](../../tests/physics/test_relational_mediation.py)
check the exact path cost and its gradient against engine observations,
including this zero-capacity boundary, without a trajectory campaign.

Under the full local theorem's positive-capacity premises, phase deformation
drives form through pressure, and form contrast feeds back into phase; form
loss then damps the joint deviation. A restored configuration can remain at
rest with all rates zero. What follows is a conditional restoring collective
relation, not a requirement for perpetual oscillation. Holding fine support
fixed still explains why its edges persist in the model; this curvature
calculation does not explain their primitive birth or physical identification.

### Simultaneous nodal loss can admit continuous reinforcement

The shared [transport derivative](../../src/tnfr/physics/support_transport.py)
and [joint response](JOINT_PARAMETER_RESPONSE.md#10-joint-pressure-response-and-the-capacity-product-rule)
already separate changing conductance from nodal change. Fix a simple bare
support `U`, positive symmetric conductances `a_ij`, strengths `d_i>0`, held
positive `N=diag(nu_i)` and regular `H_U>0`, with `H_U*g_U=-grad(V_U)`.
Let `e>=0`, `w>0`, `beta>0` and declare the conductance/storage scales.
Consider the **explicit weighted extension**, not the unit-only executor,

\[
q=B_a x,\quad
\dot x=N(-eD_a^{-1}q+w g_U),\quad
\dot\theta=(w/\beta)H_U^{-1}Nq,\quad
S=\tfrac12x^TB_a x+\beta V_U.
\]

For held `e,w,beta,nu_i`, differentiating and cancelling the exchange terms gives

\[
\dot S=-e\sum_i\frac{\nu_iq_i^2}{d_i}
+\frac12\sum_{\{i,j\}\in U}\dot a_{ij}(x_i-x_j)^2.
\]

Thus contemporaneous form loss can accommodate positive conductance work
under an additional `S_dot<=0` premise. This is not expenditure of past loss
from an invented reservoir. It is an admissibility inequality, leaving the
allocation, speed and occurrence of reinforcement undetermined. It supplies
no evolution law for `a`, and no weighted executor or global recovery theorem
is installed by this calculation.
For common-capacity P2 with nonzero form contrast it reduces to
`a_dot<=4*e*nu*a`: positive reinforcement can be admitted without selecting
its rate. Conductance scaling leaves normalized form transport unchanged but
changes this extension's phase row at fixed `beta`; the storage budget is not
an independently calibrated physical growth law.

This calculation holds bare phase support fixed. A zero-weight edge already
belongs to that support and participates in its phase pressure; it is not an
absent relation. The existing fixed-active-edge derivative rejects birth from
zero, and a zero-strength row needs separate admission. Adding a genuinely
absent edge also changes `V_U`, `g_U`, `H_U` and support degrees, so its event
needs the [complete reset budget](RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact).
At unchanged phase, its added phase cost `beta*(1-cos(theta_j-theta_i))`
does not vanish as the newborn conductance tends to zero. Small transport
weight therefore does not regularize native bare-support birth.
Likewise a regular multiplicative rule `a_dot=a*f` preserves an initially zero
weight when `f` is locally bounded: `a(t)=a(0)*exp(integral(f))`. Naming such a
rule adaptive does not make it a birth mechanism.

### What the repository supplies, and what is still missing

The [relation foundation](RELATION_FOUNDATIONS.md) defines support, conductance,
metric and causal dependence as distinct objects. Its
[zero-relation admission](RELATION_FOUNDATIONS.md#zero-relation-boundary)
shows why a small transport weight need not describe a weak or newly forming
interaction, including the separate first-neighbor normalization boundary.
That foundational contract precedes a proposed law for creating a relation;
the effective-link results below retain their supplied fine support.

UM can add a chord and create a new cycle sector; the existing
[sector-birth control](FORCED_PHASE_LOCKING.md#31-one-added-chord-extends-the-cycle-lattice-not-a-phase-generation-law)
separates that event from simultaneous phase writes and its configured Si
gate. Topological REMESH constructs MST/kNN support from supplied EPI distances
and settings. The runtime's ordinary REMESH gate instead invokes delayed EPI
mixing. Topology and coupling-support observations do not install either law.
These are reusable mechanisms and controls, not a hidden autonomous selector.

With a component-local fixed-support vector field, genuinely disconnected
components have no causal cross-response. Shared initial rhythms do not alter
that fact. A prospective binding model must therefore declare whether its
precursor interaction is an existing fine path, an explicitly supplied field
or candidate relation, or a newly postulated support law. Relabeling a
zero-weight edge or a globally read candidate as no prior interaction conceals
that premise. A pre-material interpretation remains open until this state and
law are justified and a prospective response is derived. The
[sole queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) admits that
mechanism before extending the composite-identity experiment.

<a id="effective-link-admission"></a>
## 15. Sufficient conditions for a restoring effective link

### Operational meaning and fixed premises

Here a **local restoring effective link** means two properties under the same
declared law: interventions in a region's retained form/phase state affect the
other region, and sufficiently small joint perturbations recover the reference
geometry modulo common offsets. This is an operational criterion for the
present model, not a universal definition of an NFR, a microscopic edge-birth
law or a physical spatial binding claim. A reference equilibrium need not
oscillate or carry a nonzero signal to have this response to perturbations.

Fix the [complete relational law](RELATIONAL_EXCHANGE_ADMISSION.md#1-state-inherited-geometry-and-the-independent-premise)
on a finite connected simple unit graph, with held positive capacities,
`e,w,beta>0`, no forcing or events, and an acute equilibrium
`x_*=c*1`, `theta_*` with zero neighbor sine sums. Use local lifted phase
deviations and interleave the coordinates as `z_i=(u_i,v_i)` for each node.
This is a permutation of the native tangent's all-form, then all-phase order.
The following identities concern the ideal equilibrium and real-valued field;
a materialized tangent retains its own numerical residuals.

### A unique shortest path gives a full-pair causal response

Let `J=DF(z_*)`. Its off-diagonal block on an edge `j--i` is

\[
J_{ij}=\begin{pmatrix}
e\nu_i/d_i & w\nu_i\cos(\theta_{*,j}-\theta_{*,i})/H_i\\
-w\nu_i/(\beta H_i)&0
\end{pmatrix},\qquad
\det J_{ij}=\frac{w^2\nu_i^2\cos(\theta_{*,j}-\theta_{*,i})}
 {\beta H_i^2}>0.
\]

Nonadjacent off-diagonal blocks vanish. This follows directly from the
[equilibrium Jacobian](RELATIONAL_RECOVERY_AND_INTERACTION.md#quotient-linearization-and-the-restoring-mechanism),
not from a graph-wave equation. Suppose distinct nodes `a,b` have a unique
shortest support path `a=v_0,...,v_ell=b`, with length `ell>=1`. Locality gives
`(J^k)_{ba}=0` for `k<ell`. At order `ell`, a contributing walk cannot contain
a diagonal stay or a detour; uniqueness therefore gives

\[
P_{ba}=(J^\ell)_{ba}
=J_{v_\ell v_{\ell-1}}\cdots J_{v_1v_0},\qquad \det P_{ba}>0.
\]

For the flow `Phi_t`, the tangent transfer of the donor's two coordinates to
the receiver's two coordinates is consequently

\[
T_{ba}(t)=D_{z_a}(\Phi_t)_b(z_*)=(e^{tJ})_{ba}
=\frac{t^\ell}{\ell!}P_{ba}+O(t^{\ell+1}),
\]

\[
\det T_{ba}(t)
=\frac{t^{2\ell}}{(\ell!)^2}\det P_{ba}+O(t^{2\ell+1})>0
\]

for every sufficiently small positive `t`. The reversed unique path gives
the reciprocal statement. This is a rank-two response, not a guarantee that
each matrix entry, a chosen scalar sensor or a regional average is nonzero.
For each such fixed time, smoothness and the inverse function theorem also
make the actual nonlinear map from the donor pair to the receiver pair locally
invertible, with other initial coordinates held fixed. The neighborhood may
depend on time; no arbitrary-amplitude or global response follows.

With several shortest paths the leading coefficient is the **sum** of their
ordered block products. Invertibility of each product alone does not establish
invertibility of the sum. The unique-path result is sufficient, not necessary;
the following consensus specialization removes that restriction.

### At consensus, all shortest active paths reinforce the same block

At phase consensus, `H_i=pi*d_i` and every edge cosine is one. Here even held
nonnegative capacities are allowed for the transfer calculation. Put

\[
A=ND^{-1}B,\qquad
T=\begin{pmatrix}e&w/\pi\\-w/(\beta\pi)&0\end{pmatrix},
\qquad J=-A\otimes T,\qquad \det T=\frac{w^2}{\beta\pi^2}>0.
\]

Call the directed step `j -> i` active when `j--i` is a support edge and
`nu_i>0`: the receiving row determines transmission. If the shortest active
path from `a` to `b` has length `ell`, the same walk expansion gives

\[
(J^\ell)_{ba}=s_{ba}T^\ell,\qquad
s_{ba}=\sum_{\substack{a=v_0\to\cdots\to v_\ell=b\\
                         \text{shortest active paths}}}
               \prod_{r=1}^\ell\frac{\nu_{v_r}}{d_{v_r}}>0.
\]

Thus all such paths have the same matrix factor and there is no leading-block
cancellation, regardless of their number. The full-pair small-time conclusion
holds whenever an active path exists. The degrees remain those of the actual
support, including neighbors of zero capacity. In particular, `nu_a=0` does
not prevent a perturbed donor value from acting as a fixed boundary source;
capacity zero freezes its response, not its neighbors' dependence on its state.
This extension concerns transfer only: the whole-network recovery theorem
below still requires strictly positive capacities. Positive `e` is required
there for attraction, not for the displayed off-diagonal determinants.

### A frozen separator is an exact nonlinear causal null

Let a set of zero-capacity vertices separate the donor and receiver in the
support, and hold capacities and support fixed. Compare two admitted solutions
with identical initial separator and receiver-side states, changing only the
donor side. Both rows of every separator node are identically zero under the
selected joint law. Its state therefore stays the same in both solutions.
The receiver-side equations consume only their own evolving state and this
same fixed boundary. Local uniqueness makes their solutions identical for
their common existence interval. The argument is nonlinear and does not use
a tangent approximation. A remaining active route invalidates this null.

The [grounded recovery control](RELATIONAL_MEDIATOR_DYNAMICS.md#finite-mediated-response)
shows why this distinction matters: independent regions can each restore a
geometry imposed by one frozen boundary without transmitting changes to one
another. Neither similar shapes nor equal rhythms establishes causal linkage.

### Static effective curvature and dynamic recovery are complementary

Let `K` be the acute equilibrium's cosine-weighted phase Hessian. Retain two
distinct endpoints `R={a,b}` and minimize its quadratic storage over the other
vertices `I`. When `I` is nonempty, connected positive edge weights give
`K_II>0`; the unique constrained minimum has effective Hessian

\[
K_{\rm eff}=K_{RR}-K_{RI}K_{II}^{-1}K_{IR}
=k_{ab}\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\qquad k_{ab}>0.
\]

Indeed the minimized quadratic is nonnegative, vanishes for common endpoint
offsets, and cannot vanish for unequal offsets: a zero full-graph quadratic
requires every phase deviation to be equal. These facts give the displayed
rank-one form and strict coefficient. For two vertices with no interior, use
`K` itself. The minimum phase contribution is
`beta*k_ab*(v_b-v_a)^2/2`. This generalizes the
[two-edge calculation](#mediated-restoring-geometry); it is static curvature,
not a new pressure, conductance or instantaneous evolution law.

Under the positive-capacity premises, the existing
[local recovery theorem and sufficient basin](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
complete the restoring claim. In its notation, `||z(0)||<r` and
`E_rel(0)<k_r*r^2` keep the joint state in an admitted acute neighborhood and
give convergence to the reference geometry modulo common form/phase offsets.
The effective-curvature coefficient `k_ab` is distinct from that basin bound
`k_r`. The causal theorem plus this recovery result supplies a sufficient
restoring effective link for the stated pairs; no extra synchronization
threshold, pulse variable or operator schedule is required.

If hidden nodes are eliminated dynamically, the resulting effective law must
retain the [derived memory and hidden initial state](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction).
The [pressure-state obstruction](JOINT_PARAMETER_RESPONSE.md#pressure-state-closure)
also rules out replacing this environment in general by its instantaneous
total pressure: equal pressure can conceal different future responses under
the same law. Zero pressure is not an empty substrate.
Static Schur minimization cannot replace that law. A trajectory entering the
sufficient basin can be said to acquire the maintained joint geometry; proving
entry from another preparation is a separate capture obligation. When a fine
active path already exists, causal influence has no positive waiting interval
in this local ODE argument. A detection threshold does not create its onset,
and neither this result nor capture explains the birth of primitive support.

### An explicit local formation-to-maintenance preparation

The same two unit C5 rings with path `0--10--5` provide a capture statement
without another trajectory calculation. Fix unit capacities,
`e=w=1/2`, `beta=1`, and the aligned winding-one reference of the
[mediator owner](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction).
Prepare uniform zero form, leave the first ring's phases at the reference,
rotate the entire second ring by `delta`, and set the mediator phase to
`delta/2`. Both internal windings are unchanged. The common phase offset is
`delta/2`, so the initial quotient norm and excess storage are exactly

\[
\|z(0)\|^2=\frac52\delta^2,\qquad
\mathcal E_{\rm rel}(0)=2[1-\cos(\delta/2)]\le\frac{\delta^2}4.
\]

This graph has eleven nodes and diameter six. For any centered vertex vector,
its squared norm is at most `n*(max-min)^2/4`, while Cauchy--Schwarz along a
shortest path between its extrema bounds the graph energy below by
`(max-min)^2/diameter`. Thus `lambda_2(B)>=4/(n*diameter)=2/33`.
The reference acute margin is `m=pi/10`. Taking

\[
r=\frac{\pi}{20\sqrt2},\qquad
c_r=\sin(\pi/20),\qquad
\underline k=\frac{\sin(\pi/20)}{33}\le k_r
\]

in the existing basin theorem proves capture whenever

\[
|\delta|<\min\{r\sqrt{2/5},\ 2r\sqrt{\underline k}\}.
\]

For example, `delta=1/128` satisfies the strict bounds without a floating
evaluation: `pi>3` and `sin(pi/20)>1/10` give `r^2>9/800` and
`underline(k)*r^2>9/264000>1/65536`, whereas the preparation has
`||z(0)||^2=5/32768` and `E_rel(0)<=1/65536`.

The full continuous law therefore keeps this preparation acute and converges
to the aligned joint geometry modulo one common phase and one common form
offset. It restores an initially nonzero relative regional offset and retains
both windings. This establishes local acquisition and maintenance of the
joint geometry on supplied support; it is not creation of the already active
causal path. No monotone regional phase difference, finite-time exact locking,
global capture, autonomous preparation or Euler-trajectory certificate is
asserted. The static midpoint preparation does not keep the mediator or the
ring shapes constrained during this subsequent evolution.

The [focused controls](../../tests/physics/test_relational_effective_link.py)
check the path, consensus and frozen-separator distinctions through the shared
native tangent and field owners. Finite represented checks support integration;
they do not replace the ideal proofs or certify a numerical recovery trajectory.

<a id="environmental-capture-domain"></a>
### A capture domain retaining the intermediary's initial state

Keep the preceding two-C5 plus mediator support, aligned winding-one reference,
unit capacities, `e=w=1/2` and `beta=1`. A wider preparation gives the mediator
form `a` while all ring forms are zero. Leave the first ring's phases at their
reference, rotate the second ring by `delta`, and set the mediator phase to
`delta/2+eta`. These are three supplied initial coordinates, not an invariant
restriction on the later evolution or a new environment law.

The common offsets of the deviations from the reference are `a/11` in form
and `delta/2+eta/11` in phase. Subtracting them gives exactly

\[
\|z(0)\|^2=\frac{10}{11}(a^2+\eta^2)+\frac52\delta^2.
\]

Only the two mediator edges change storage relative to the reference. Their
form contribution is `a^2`, and their phase gaps are `delta/2+eta` and
`delta/2-eta`. Therefore

\[
\begin{aligned}
\mathcal E_{\rm rel}(0)
 &=a^2+2[1-\cos(\delta/2)\cos\eta]\\
 &\le a^2+\eta^2+\frac{\delta^2}4
 =:\mathcal B(a,\eta,\delta).
\end{aligned}
\]

The inequality follows by applying `1-cos(u)<=u^2/2` to each gap. Reuse the
same `r` and `underline(k)` as above and define
`kappa=underline(k)*r^2`. The single sufficient condition

\[
\boxed{\quad \mathcal B(a,\eta,\delta)<\kappa\quad}
\]

implies both basin hypotheses: `E_rel(0)<kappa<=k_r*r^2`, and
`||z(0)||^2<=10*B<10*underline(k)*r^2<r^2`, since
`10*underline(k)<10/33<1`. Thus the initial state is in the acute neighborhood
and its full continuous evolution remains there and approaches the same joint
geometry modulo common offsets. The mediator's nonzero initial form and phase
are included in this theorem, rather than replaced by their equilibrium values.

A wholly nonzero rational example is
`delta=1/256`, `a=eta=1/512`. It has
`B=3/262144<9/264000<kappa`, using the preceding exact lower bound. No
trajectory or transcendental rounding assumption enters this sufficient
admission. Violating this conservative bound establishes neither escape nor
failure to recover; it leaves the stated certificate unavailable.

### Identical mediator pressure does not specify the environment

The family also connects capture to the
[pressure-state obstruction](JOINT_PARAMETER_RESPONSE.md#pressure-state-closure).
In the admitted lift the two port phases are zero and `delta`, so their
resultant points at `delta/2`. The mediator phase source and total pressure are

\[
g_m=-\eta/\pi,\qquad p_m=-ea-\frac w\pi\eta.
\]

Fix the same `delta` and compare the reference intermediary `a=eta=0` with
a compensated intermediary

\[
\eta\ne0,\qquad a=-\frac{w\eta}{\pi e}.
\]

Both preparations have identical ring form/phase coordinates, identical
capacities and the same instantaneous mediator pressure `p_m=0`. Nevertheless
the receiving port has form gradient `q_5=-a`, hence

\[
\dot\theta_5=-\frac{wa}{\beta H_5}\ne0
\]

in the compensated preparation, whereas its phase rate is zero in the
reference preparation. For example, with `c=cos(2*pi/5)` and
`alpha=Arg(2*c+exp(i*(eta-delta/2)))`, the metric is the native
`H_5=pi*abs(2*c+exp(i*(eta-delta/2)))*sinc(alpha)>0`. The distinction uses the
existing phase row and the complete hidden state, not a pressure reconstructed
from the receiver's observed derivative.

At the fixed coefficients, choose `delta=1/256`, `eta=1/512` and
`a=-eta/pi`. Then `B=(2+1/pi^2)/262144<3/262144<kappa`; the reference
intermediary also satisfies the bound. Thus both environments belong to the
same proved capture domain but produce different initial receiver responses.
The claim concerns the **mediator's** pressure, not equality of every nodal
pressure or every subsequent trajectory.

The [existing two-coordinate memory result](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction)
already proves why both hidden mediator coordinates must be retained for the
specified all-ring tangent observation. This example shows concretely why
replacing them by the single mediator pressure loses information. The memory
owner retains hidden initial state and nonlinear forcing; it does not replace
the mediator by instantaneous static minimization. The initial preparation
family above need not remain rigid, pressure-compensated or otherwise closed
under the full flow.

The [effective-link controls](../../tests/physics/test_relational_effective_link.py)
check these static identities and native response distinctions without treating
finite arithmetic as the continuous capture proof. The
[sole execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns subsequent work. Supplied fine support, model premises and preparation
remain explicit; this result does not establish physical vacuum, autonomous
substrate creation or a universal zero-pressure criterion.

## 16. Formation-enabling support and later pattern detachment

<a id="relational-pattern-detachment"></a>

The two supplied bridges can enable formation without being necessary for
the eventual maintenance of either ring. This distinction follows from the
existing formation, storage and cycle-geometry results; it does not require
another trajectory or an autonomous edge-deletion rule.

### Event and post-event state card

Start with the two unit C5 rings and the two matching bridges at positions
0 and 1 used by the [formation proof](RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit).
At a supplied time, delete both bridges and retain every nodal form, phase
and capacity coordinate. The resulting support consists of two C5 components.
Evaluate each component separately with its own newly computed degree,
gradient, resultant and phase metric. The connected relational executor is
not thereby extended to a disconnected graph.

On each component, retain positive held capacities and `e,w,beta>0`, the
native form row and the chosen Dirichlet/cosine storage. The phase law must
be admitted on that **post-event** support: autonomous, `C1`, invariant
under common form/phase shifts, at rest when `q=g=0`, and satisfying
`E_dot<=-c*L` for a fixed `c>0` on the relevant regular neighborhood. No
forcing, further event or capacity change occurs during the claimed
continuation. The same component-local rule can be used before and after
the cut; its actual support-dependent quantities must be refreshed. The
reference, rho and eta laws have this component-local property and the
previously proved loss bounds. A law that reads another component's state
does not acquire independent-component dynamics merely because the graph
was cut.

### A sufficient maintenance basin for one isolated C5

For an oriented C5, let its true wrapped edge gaps be strictly acute and
have winding `s`, with `s=+1` or `-1`. Define

\[
V_*=5[1-\cos(2\pi/5)],\qquad
V_{\rm face}=5-4\cos(3\pi/8),\qquad
E_R=\frac12x_R^TB_Rx_R+\beta V_R.
\]

**Conditional isolated-ring capture.** If

\[
\boxed{\qquad E_R<\beta V_{\rm face},\qquad}
\]

then every law admitted by the preceding state card preserves the acute
winding sector and converges to uniform component form and the winding-s
uniform twist. Convergence is eventually exponential modulo the component's
common form and phase offsets, which have finite limits. Neither reflection
nor uniform capacity is a premise.

For `s=+1`, the five acute gaps sum to `2*pi`. A first boundary gap cannot
be `-pi/2`: the other four cannot supply the required remaining `5*pi/2`.
A boundary gap of `pi/2` leaves sum `3*pi/2` for the other four. Convexity
of `1-cos(delta)` on the closed acute interval gives

\[
V_R\ge1+4[1-\cos(3\pi/8)]=V_{\rm face}
\]

there. Reversing all gaps proves the negative-winding case. Storage
nonincrease therefore prevents a first exit. The strict sublevel is compact
modulo common offsets and stays a positive distance from the acute boundary.
Every resultant and phase-metric entry remains regular and positive.

In the phase quotient the fixed-period acute component is convex, and the
phase Hessian is the cycle Laplacian with positive weights `cos(delta)`.
Its unique critical geometry is the uniform twist: zero phase gradient
makes all oriented edge sines equal, and sine is injective on the acute
interval. The common gap is consequently `s*2*pi/5`.

The strict loss bound and positive capacities force `q=B_R*x_R=0` on the
invariant zero-loss set. To preserve this condition, the unchanged form row
requires `B_R*N_R*g_R=0`, so `N_R*g_R=a*1`. The identity
`sum(H_i*g_i)=0` then gives `a*sum(H_i/nu_i)=0`, hence `a=0` and `g_R=0`.
Rest makes that target invariant. Compactness and LaSalle's principle give
convergence. The [local law-class proof](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-sector-law-class)
uses only connected support, positive capacities, a positive acute phase
Hessian and the full-neighborhood strict loss bound for its Hurwitz step;
those hypotheses hold on this C5. Its exponential-recovery and integrable
common-offset arguments therefore apply independently to each component.
The barrier is a sufficient capture test, not a classification of every
initial state that could recover.

### The joined sector certificate already admits both detached rings

Let `E_total` be the full two-ring storage immediately before deletion.
Suppose the existing [joined acute-sector conditions](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-acute-sector-capture)
hold, including common winding `s=+1` or `-1` and

\[
E_{\rm total}<\beta(V_*+V_{\rm face}).
\]

Write `E_1,E_2` for the two internal ring storages and `C_bridge>=0` for
the two bridge costs. Jensen's inequality gives `E_j>=beta*V_*` for each
acute winding-s ring. Thus

\[
E_{\rm total}=E_1+E_2+C_{\rm bridge},\qquad
E_1\le E_{\rm total}-\beta V_*<\beta V_{\rm face},
\]

and the same calculation applies to `E_2`. Deleting the bridges changes
neither internal phase gaps nor either internal storage. **Every snapshot
admitted by the joined geometric sector test therefore already satisfies
the two isolated-ring capture tests**, provided the post-cut laws and
capacities are admitted separately. No smaller Euclidean neighborhood or
extra numerical formation run is needed. Directly checking the two component
basins may also admit states outside this sufficient joined criterion.

For arbitrary full states the exact event budget is

\[
\Delta E=-\sum_{\{i,j\}\in\mathrm{bridges}}
\left\{\tfrac12(x_i-x_j)^2+\beta[1-\cos(\theta_i-\theta_j)]\right\}
\le0.
\]

This is the [existing deletion identity](RELATIONAL_SUPPORT_EVENTS.md#support-event-premise-admission).
The geometric basin conditions supply the additional future-identity
obligation; nonpositive event cost alone does not do so. After the cut,
component means can evolve differently, and common phase offsets need not
remain mutually aligned. The theorem maintains each ring's winding geometry,
not a restoring interaction between disconnected components or a joint
future determined by an aggregate observation.

### Zero bridge cost throughout formation does not make an early cut safe

In the exact copied preparation, corresponding endpoints of each bridge
have equal form and phase throughout the ideal pre-cut flow. Both bridge
costs are therefore identically zero, not merely zero at equilibrium.
Nevertheless, removing them changes each port's degree and resultant even
when its form gradient is unchanged. A zero storage jump does not imply
unchanged pressure, phase metric or future evolution.

The original represented seed supplies an exact early-cut control. Each
ring has phases

\[
(a_0,-a_0,-a_0,0,a_0),\qquad
a_0=\frac{875483625981347}{562949953421312},\qquad
0<a_0<\pi/2.
\]

The ordinary ring winding is zero. On the pure C5, the five resultants are
`1+exp(-2*i*a_0)`, `1+exp(2*i*a_0)`, `1+exp(i*a_0)`,
`2*cos(a_0)` and `1+exp(-i*a_0)`. Their real parts are strictly positive,
so the early cut itself has an admitted regular post-event state. It is
not rejected merely because the domain was undefined.

The [pure-cycle resultant invariant](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-cycle-resultant-obstruction)
now distinguishes it from either maintained acute twist. Around the derived
skip cycle `(0,2,4,1,3)`, its true wrapped gaps are

\[
(-2a_0,\ 2a_0,\ -2a_0,\ a_0,\ a_0),
\]

all strictly inside `(-pi,pi)` and summing to zero. Its skip winding is
therefore zero. Every acute winding-s C5 twist has skip winding `2*s`.
A continuous pure-C5 path with nonzero resultants preserves that skip
winding, independently of the pressure or phase kinetics. Thus the early
detached seed cannot reach either acute winding-one identity along a
regular continuation of any of the named laws. The conclusion does not
assert consensus, finite-time singularity, or the absence of other regular
behavior. It is a precise obstruction to the target already obtained with
the supplied bridges.

### A sufficiently late detachment window follows from formed-state recovery

The [reference formation result](RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit)
and its [bounded phase-law comparisons](RELATIONAL_FORMATION_CONTROLS.md#relational-formation-law-robustness)
converge to the same joined aligned twist. Its edge gaps are strictly acute,
each ring storage tends to `beta*V_*`, and
`V_*<V_face`. Consequently there is a finite time after which every
pre-cut state lies in the strict joined acute-sector basin and hence in
both inherited isolated-ring basins. A supplied deletion of both bridges
at any such later time retains the formed winding identities under the
admitted component laws. This establishes the existence of a detachment
window without locating its first time or producing a new trajectory.

The previously retained horizon-32 endpoint is certified in the larger
reflected `E<7*beta` basin, not in this smaller acute-sector sublevel. It
must not be silently labeled an already certified cut time. A particular
detachment snapshot needs its own post-cut geometry and law admission.
The conclusion also does not select the cut, its candidates or its clock,
and makes no claim of autonomous substrate birth, physical particles or
the origin of the supplied preparation. It distinguishes support that
enables a formation route from support required for subsequent maintenance.

### Shared observation and implementation scope

[`certify_relational_cycle_capture` and `observe_relational_detachment`](../../src/tnfr/physics/relational_capture.py)
reuse the native field, exact acute-gap/period and cosine-storage kernels.
The detachment report evaluates one connected pre-cut field and two fresh
post-cut component fields, with unchanged primitive state. It retains
component capture certificates, per-node pressure/metric/rate changes and
the shared reset budget without deleting live edges. The reset's represented
wrapped phase cost and the native fields' raw-gap phase costs retain their
separate reconciliation residual. A direct component certificate need not
require that the stronger joined-sector condition passed first.

These APIs certify the selected reference relational model. The broader
strict-loss theorem requires the chosen alternative law's independent
admission; the report does not install or validate an arbitrary phase
callback. The [engine controls](../../tests/test_relational_detachment.py)
cover independent component admission, state-preserving accounting and
early-cut limitations. The [SDK controls](../../tests/sdk/test_relational_detachment.py)
check delegation and export while retaining that same scope. No control
replays the frozen formation trajectory or chooses a cut time.
