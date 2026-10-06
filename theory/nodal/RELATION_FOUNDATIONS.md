# Relations, support and the zero-interaction boundary

This owner asks what a connection means in a declared TNFR model before
asking how one forms. It distinguishes implemented graph data, causal
properties of a complete law and additional support-evolution premises.
It does not install an edge law or identify a graph relation with a physical
bond. The [parameter ledger](../NODAL_PARAMETER_FOUNDATIONS.md#2-parameter-and-dependency-ledger)
owns the general variable definitions; the
[research plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next task.

## 1. A relation state card

Write the nodal coordinates as `X=(x,theta,nu)`. The graph can carry distinct
relational data; only the selected model's consumed coordinates belong to
its dynamical state.

`U` is discrete and unit-free; `W` is a nonnegative matrix with a declared
conductance scale, while its normalized row ratios are dimensionless.
Directionality, reciprocity and allowed loops follow the selected model.
`ell` has an independently declared metric unit. A proposed continuous
amplitude `a>=0` also needs a scale: in `W=a*W_0`, dimensionless `W_0` gives
`[a_dot]=[W]/[t]`; with the conductance scale retained in `W_0`, `a` is instead
dimensionless and `[a_dot]=1/[t]`. The nodal, channel and storage units remain
with the [joint-law unit owner](RELATIONAL_EXCHANGE_ADMISSION.md#4-origin-units-and-exact-replication).
Numerically normalized coefficients do not erase those chart/unit premises.
Present support, weights and lengths are supplied or held inputs where consumed;
derived field geometry and effective interactions are outputs. A candidate
equation for `a` would add a dynamical law, not merely rename an existing input.

| Object | Operational meaning | What it does not establish |
| --- | --- | --- |
| Bare support `U` | Current directed or undirected neighborhood incidences consumed by the selected law | A nonzero current, an event time or an independently derived origin |
| Conductance `W` | Nonnegative coefficients of weighted form transport on that support | Absolute interaction strength in a row-normalized law; `W_ij=0` does not remove a bare neighbor |
| Structural length `ell` | The metric/path input consumed by specified field diagnostics | Physical distance or a law deriving support from geometry |
| Candidate relation | Which pairs a support-changing action is permitted to inspect | An existing transport edge or a justified occurrence rule |
| Direct dynamical dependence | Changing a donor coordinate can change a receiver's row in the complete vector field | A nonzero derivative at every state or a persistent material bond |
| Effective relation | A specified donor intervention changes a retained receiver through the full dynamics, possibly via hidden nodes and memory | A new primitive edge between the observed regions |

The current mixed pressure uses weighted EPI means, but unique bare neighbors
for phase, capacity and degree contrast. Directed rows read outgoing neighbors;
the relevant input/output convention must therefore accompany any causal claim.
Parallel conductances and diagnostic lengths also have different reductions.
These contracts remain centralized in the
[pressure owner](../NODAL_PARAMETER_FOUNDATIONS.md#21-the-implemented-pressure-is-a-specified-relational-map)
and [edge semantics](../../src/tnfr/physics/_edge_semantics.py). The fallback
from missing `length` to `weight` is a compatibility convention, not a physical
identification or an equation for emergent distance.

Three tests of absence must not be conflated. A row can depend on another
node although its instantaneous rate vanishes at equilibrium. An instantaneous
Jacobian block can vanish although a higher derivative or a mediated response
does not. A causal-null claim instead specifies the donor intervention,
receiver observation, hidden initial state and complete evolution. In the
held-capacity relational law, zero receiver capacity freezes both receiver
rows, but that node can still supply a fixed boundary to active neighbors.
The [effective-link theorem](RELATIONAL_EFFECTIVE_CONNECTIONS.md#effective-link-admission)
and [hidden-memory owner](RELATIONAL_MEDIATOR_DYNAMICS.md#9-a-nodal-intermediary-mediates-joint-formphase-interaction)
retain these distinctions.

## 2. What currently changes support

The continuous relational executor selects a simple connected unit support,
held capacities and an admitted phase domain. Its equations do not contain
a support derivative. This is a declared model boundary, not a proof that
support must be fundamental or immutable in every TNFR realization.

UM can create actual edges through the
[shared coupling proposal](../../src/tnfr/operators/_coupling_stage_kernel.py).
It first needs an existing compatible neighbor at its target, then searches
all nodes or a supplied sample for candidates. Its phase/form/Si score, threshold,
sampling and invocation are configured. Topological
[REMESH](../../src/tnfr/operators/remesh.py) similarly constructs support from
declared candidate access and settings; ordinary delayed REMESH mixes form on
existing support. Neither is an unrecorded autonomous connection theorem.
The [all-operator mechanism map](../STRUCTURAL_OPERATORS.md#operator-mechanism-and-activation-audit)
owns their complete scope.

Candidate access is logically distinct from realized coupling. Supplying all
unordered pairs is a possible kinematic domain; it does not assert that every
pair already transmits. But a rule that reads across otherwise disconnected
components adds a cross-component dependency and must declare it. A strictly
component-local, fixed-support law cannot select and execute a cross-component
change using information it never consumes.

Joint nodal reorganization can pay a positive attachment cost in the same
declared action. The [actual UM reset witness](RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact)
already proves this conditional possibility. Conversely,
[symmetry, clock covariance, passivity and recovery](RELATIONAL_SUPPORT_EVENTS.md#support-law-choice-and-clock)
do not select a unique event or its time. The question below is different:
whether a continuously varying relation coordinate has a consistent zero limit.

<a id="zero-relation-boundary"></a>
## 3. Native support and zero conductance have different limits

First retain the implemented distinction between bare phase support and
weighted form transport. Start with edges `(0,1)` and `(2,3)`, unit weights,
uniform form and phases `(0,0,delta,delta)`, `0<delta<pi/2`. Without the
bridge `(1,2)`, both components have zero phase pressure. With that bridge
present at any positive conductance `a`, the native phase sources at its ports
are

\[
g_1=\delta/(2\pi),\qquad g_2=-\delta/(2\pi),
\]

independently of `a`. They are the wrapped displacements to the unweighted
two-neighbor means. A supported zero-weight bridge has these same sources;
deleting it gives zero. Its unweighted phase-storage contribution
`beta*(1-cos(delta))` likewise does not shrink with `a`.

Even coincidence of endpoint phases and forms does not generally make a
support event dynamically invisible. With phases zero and forms `(0,1,1,1)`,
adding `(1,2)` costs zero storage. At port 1, its form gradient remains one,
but the phase metric changes from `pi` to `2*pi`. The unit-support relational
phase rate therefore changes from `w*nu_1/(beta*pi)` to half that value.
This compares the separate component field and the joined unit-support field,
not a weighted relational execution.
The [attachment owner](RELATIONAL_SUPPORT_EVENTS.md#8-one-supplied-bridge-sufficient-interface-and-endpoint-only-obstruction)
already retains the necessary degree and resultant information.

Thus making only transport conductance small does not regularize native
bare-support birth. A continuous relation model must either revise that
phase/support identification or keep the change as a hybrid event with its
actual reset and budget.

## 4. Weighting every channel does not resolve an empty row

To separate the second obstruction, consider an explicitly different,
**conditional mathematical extension**, not an installed engine model.
Take symmetric nonnegative weights on a finite potential-pair domain and set

\[
\begin{gathered}
d_i=\sum_j W_{ij}>0,\quad D=\operatorname{diag}(d),\quad
B=D-W,\quad q=Bx,\\
Z_i=\sum_jW_{ij}e^{i(\theta_j-\theta_i)},\quad
g_i=\arg(Z_i)/\pi,\quad
H_i=\pi|Z_i|\operatorname{sinc}(\arg Z_i),\\
V_W=\sum_{i<j}W_{ij}[1-\cos(\theta_j-\theta_i)].
\end{gathered}
\]

Use an admitted regular phase chart with `H_i>0`; strict acute gaps suffice.
Then `H*g=-grad_theta(V_W)`. With held positive capacities `N`, fixed
`e,w,beta>0`, the rows

\[
\dot x=N(-eD^{-1}q+wg),\qquad
\dot\theta=(w/\beta)H^{-1}Nq
\]

recover the existing relational model at unit weights. This extension weights
phase storage and resultants too; it differs from the fixed-bare-phase
[reinforcement calculation](RELATIONAL_EFFECTIVE_CONNECTIONS.md#simultaneous-nodal-loss-can-admit-continuous-reinforcement).
Their common unit specialization does not select between their other premises.

Let `W=a*W_0`, `a>0`, for any fixed admitted positive-strength geometry. Then

\[
D=aD_0,\quad B=aB_0,\quad H=aH_0,\quad
g=g_0,\qquad F_{aW_0}(X)=F_{W_0}(X).
\]

Every factor `a` cancels from the complete nodal field at fixed capacity.
For generic state it therefore has no continuous extension to a zero-weight
configuration whose nodes have no cross-node interaction. Different directions
`W_0` can also give different limiting responses. Weighting phase consistently
removes the earlier bare-neighbor jump on a regular positive background, but
does not remove this empty-strength normalization obstruction.

The implemented EPI channel already gives a minimal exact witness. On two
nodes with one edge of weight `a>0`,

\[
\dot x_i=e\nu_i(a)(x_j-x_i),
\]

whereas its zero-strength convention makes that channel inactive. At fixed
nonzero contrast and `e>0`, continuity of this rate at zero requires
`nu_i(a)->0`. A locally Lipschitz extension, or a finite derivative there,
further requires `nu_i(a)=O(a)`. Mere continuity allows other vanishing rates;
none is selected by this argument. Pressure itself retains its nonzero
directional limit even when its product with capacity vanishes.

For the complete weighted field, vanishing capacities are sufficient along
a fixed regular geometry, or uniformly on a phase cone where `H_i/d_i` stays
bounded below and forms stay bounded. They are not sufficient without such
phase regularity: a vanishing resultant can make `q_i/H_i` unbounded. A relation
between capacity and strength, an absolute mobility factor or a revised
unnormalized pressure would each be an additional model premise. An evolving
capacity must also retain its actual law; it is not the held-capacity model.

## 5. Passive onset has different branches

For the fully weighted extension define

\[
S=\tfrac12x^TBx+\beta V_W,\qquad
\dot S=-e\sum_i\nu_iq_i^2/d_i
+\sum_{i<j}\dot W_{ij}
 \left[\tfrac12(x_i-x_j)^2+\beta(1-\cos(\theta_j-\theta_i))\right].
\]

This follows by differentiation and cancellation of the form/phase exchange
terms. Support work uses the same weights in both storage terms. Requiring
`S_dot<=0` is an additional passivity premise, not a consequence of the nodal
identity and not an occurrence law.

For uniform onset `W=a*W_0`, write `S=a*S_0(X)` and
`Diss_0=e*sum_i nu_i*q_0i^2/d_0i`. Wherever `S_0>0`, passivity requires

\[
\dot a\le a\,\mathrm{Diss}_0/S_0\le4e\nu_{\max}a.
\]

The last inequality is Cauchy's inequality applied to each weighted gradient:
`sum_i q_0i^2/d_0i <= 4*E_0`, where `E_0=x^T B_0 x/2 <= S_0`.
With locally bounded capacities, a nonnegative absolutely continuous `a`
starting at zero cannot become positive while `S_0` remains positive and
this passive regular description holds. Gronwall's inequality gives this
boundary invariance; a multiplicative rule `a_dot=a*f` with locally bounded
`f` is its familiar special case. At `S_0=0`, division is invalid and this
argument does not restrict zero-cost activation. Nor does it select activation
there. Additional stored variables, inputs or hybrid nodal resets change
the premises and require their own balance.

This is **not** a general obstruction to a new bridge between active regions.
With positive existing strengths and regular resultants at both ports, the
fully weighted candidate has a smooth `a=0` limit. Its storage has the form
`S_base+a*c`, where `c` is the new edge cost in brackets above. Existing nodal
loss can be positive at zero. The instantaneous passive budget permits
`a_dot(0)*c <= Diss(0)`, including positive onset when strict slack is available.
This proves feasibility of a budget, not an independently selected derivative,
candidate or time. It also does not transfer to native unweighted phase support.

<a id="autonomous-contact-completions"></a>
## 6. Autonomous contact is possible, but its constitutive completion is open

Retain the fully weighted extension in section 4. Consider two components
with positive existing strengths and regular phase resultants, one supplied
potential bridge `{u,v}` of variable dimensionless weight `a>=0`, and fixed
other weights, capacities, coefficients and clock. The potential pair is
part of the declared model before its transport weight becomes positive.
Use the same state-dependent edge cost and total loss as above, and define
a port-local loss allocation

\[
c=\tfrac12(x_u-x_v)^2+\beta[1-\cos(\theta_v-\theta_u)],\qquad
D_i=e\nu_iq_i^2/d_i,\qquad
\mathcal D=\sum_iD_i,\qquad \mathcal B=D_u+D_v\le\mathcal D.
\]

The port gradients consume their current weighted neighborhoods. Unlike a
global-loss allocation, `mathcal B` does not read an unrelated disconnected
component. It still supplies a chosen allocation and reads both endpoints
of the candidate pair; locality relative to this potential pair does not
derive candidate access or the allocation itself.

### Two complete local autonomous countermodels

Combine the same nodal rows with either of the two **postulated** support rows

\[
\dot a=f_1=\frac{\mathcal B}{\beta+c},\qquad\text{or}\qquad
\dot a=f_2=\frac{\beta\mathcal B}{(\beta+c)^2}.
\]

These are autonomous equations for `(X,a)`, not an external invocation or
thresholded diagnostic. On a regular positive-background chart they are
smooth, nonnegative at the boundary and nonsingular because `beta>0`.
Local existence is the ordinary smooth-ODE result, stopped at any first
exit from that chart. No global continuation or later recovery is asserted.
If `mathcal B(0)>0`, each produces `a(t)>0` for sufficiently small `t>0`.
An initially zero allocation alone does not prove permanent inactivity:
subsequent nodal evolution can change it.

More generally, a locally Lipschitz joint law
`X_dot=F(X,a), a_dot=mathcal H(X,a)` preserves `a=0` if
`mathcal H(X,0)=0` on the entire admitted boundary. The restricted solution
`(X(t),0)` then solves the full equations, so local uniqueness enforces
invariance. A zero derivative only at the initial state does not meet this
hypothesis. This statement concerns the proposed support row, not the phase
metric `H`.

Both retain the stated storage and obey its full balance:

\[
\begin{aligned}
\dot S_1&=-(\mathcal D-\mathcal B)
 -\frac{\beta\mathcal B}{\beta+c}\le0,\\
\dot S_2&=-(\mathcal D-\mathcal B)
 -\mathcal B\left[1-\frac{\beta c}{(\beta+c)^2}\right]
 \le-(\mathcal D-\mathcal B)-\tfrac34\mathcal B\le0.
\end{aligned}
\]

Here `beta*c/(beta+c)^2<=1/4`. More generally
`a_dot=(mathcal B/beta)*h(c/beta)` is passive for any smooth nonnegative `h`
with `r*h(r)<=1` for `r>=0`. The two displayed laws choose
`h(r)=1/(1+r)` and `h(r)=1/(1+r)^2`. Having no additional numerical parameter
does not remove this functional constitutive freedom.

Both laws respect relabeling of the entire declared candidate, exchange of
its endpoints, common form shifts and common phase rotations. Under the
complete form/clock conversion `x_new=lambda*x`, `t_new=b*t`,
`beta_new=lambda^2*beta`, the unit owner gives
`mathcal B_new=lambda^2*mathcal B/b` and `c_new=lambda^2*c`.
Consequently `f_new=f/b`, as required for dimensionless `a`.
These restrictions do not distinguish the two laws.

### An exact first-contact discriminator

Take base edges `(0,1)` and `(2,3)` with unit conductance, candidate `(1,2)`,
`a(0)=0`, forms `(0,1,0,0)`, phases zero, unit capacities and
`e=w=1/2`, `beta=1`. Both components have positive strengths and capacities;
the left component initially supplies loss and the right is at equilibrium.
The exact initial quantities and predictions are

\[
c=\tfrac12,\quad \mathcal D=1,\quad\mathcal B=\tfrac12,\qquad
\begin{array}{c|cc}
 &f_1&f_2\\\hline
\dot a(0)&1/3&2/9\\
\dot S(0)&-5/6&-8/9\\
\ddot x_2(0)&1/6&1/9\\
\ddot\theta_2(0)&-1/(6\pi)&-1/(9\pi)
\end{array}
\]

Both receiver rates initially vanish. On this preparation their derivatives
with respect to the new weight are `partial_a F_x2=e` and
`partial_a F_theta2=-w/(beta*pi)`. The old nodal-flow contribution to their
acceleration is zero, so multiplication by the respective `a_dot(0)` gives
the displayed values. Thus the two autonomous models predict

\[
x_2^{(1)}(t)-x_2^{(2)}(t)=t^2/36+O(t^3),\qquad
\theta_2^{(1)}(t)-\theta_2^{(2)}(t)=-t^2/(36\pi)+O(t^3).
\]

This is a local analytic prediction of the explicitly weighted completions,
not a trajectory of the unit-only executor or proof of their physical status.
It establishes constructive feasibility of contact and different nodal
consequences under the same symmetry, units and passive budget. The full
system already contains a latent dependency through its support row before
the realized transport is positive; it does not create interaction from an
unspecified empty substrate. In contrast, a complete locally unique system
whose full state remains factorized as `X_L_dot=F_L(X_L)` and
`X_R_dot=F_R(X_R)` cannot acquire cross-component causal dependence under
that unchanged law. Introducing the candidate row revises that premise.

### Fixed-support dynamics leaves support storage undetermined

There is a deeper freedom than choosing the allocation function. With the
selected storage and other coordinates held fixed, `partial_a S=c>=0`.
A pure scalar gradient descent `a_dot=-M*c`, `M>=0`, weakens an existing
edge; its projection onto `a>=0` cannot leave zero. This statement concerns
that additional gradient premise, not all passive contact laws: the preceding
countermodels already permit growth funded by simultaneous nodal loss.

The fixed-support nodal equations cannot distinguish

\[
\widetilde S(X,a)=S(X,a)+\Psi(a)
\]

from `S` by differentiating in `X`. For every held `a` the nodal gradients,
field and storage decay rate are identical. Once `a` varies, however,

\[
\dot{\widetilde S}=-\mathcal D+[c+\Psi'(a)]\dot a.
\]

For example, the independently chosen, bounded-below potential
`Psi(a)=beta*(a^2/2-a)` gives support gradient `c+beta*(a-1)`.
Writing `k=c+Psi'(a)`, a supplied smooth nonnegative scalar mobility `M` defines
the projected gradient rule

\[
\dot a=\begin{cases}
-M k,&a>0,\\
\max(0,-M k),&a=0.
\end{cases}
\]

In this dimensionless-conductance chart, `[M]=1/([x]^2[t])`.
Where the gradient row is active, the full balance is
`S_tilde_dot=-mathcal D-M*k^2`; on the projected boundary hold it is
`-mathcal D`. With `Psi=0`, the nonnegative cost forbids positive boundary
drive. The displayed quadratic instead grows from zero when `c<beta` and
`M>0`. The number one and the shape
of this potential encode a constitutive choice even though `beta` already
appears in the nodal model. The resulting threshold is not derived by the
fixed-support equations. This example revises the chosen storage and
kinetics; it does not invalidate the conditional bounds for `Psi=0`.
The positive-onset demonstration has an ordinary smooth field locally where
`c<beta`; no globally smooth projected boundary field is asserted.
Equivalently, adding the irrelevant constant `beta/2` gives
`Psi_plus=beta*(a-1)^2/2`: `a=0` has excess `beta/2` above this potential's
minimum. That available difference is an added relation-storage premise,
not energy obtained by changing the reference zero. This example remains
within the regular positive-background extension and does not repair the
first-neighbor normalization boundary.

**Equilibrium elimination does not supply the missing negative gradient.**
On an `a`-independent hidden domain, suppose the finite reduced storage is
`S_eff(y,a)=inf_h [S_base(y,h)+a*c(y,h)]`, with `c>=0`.
For `a_2>=a_1`, every hidden-state value is nondecreasing, so taking the
infimum preserves `S_eff(y,a_2)>=S_eff(y,a_1)`. If a differentiable interior
minimizing branch `h*(y,a)` exists, stationarity and the chain rule give
`partial_a S_eff=c(y,h*)>=0`. The same derivative holds for the value along
any differentiable interior stationary branch, without asserting that it
realizes the infimum. Thus eliminating equilibrium hidden coordinates alone
cannot produce a negative support-gradient drive from this storage.
Moving constraints, new `a`-dependent terms, nonequilibrium memory and
nodal-loss-funded contact lie outside this statement. The
[controlled hidden reduction](RELATIONAL_MEDIATOR_DYNAMICS.md#fast-mediator-reduction)
already distinguishes static elimination from exact finite-capacity dynamics;
no additional closure or evolution law follows from this monotonicity proof.

The freedom in `Psi` and its mobility is the support-dependent version of
the undetermined potential and mobility in the
[variational owner](../TNFR_VARIATIONAL_PRINCIPLE.md#131-mixed-derivatives-require-reciprocal-coupling).
It exposes a concrete missing premise: whether a relation carries intrinsic
storage or work beyond nodal mismatch, and how that quantity couples to nodal
reorganization. Pure nodal evolution at fixed support cannot answer it.
Further progress requires an independently justified restriction on this
support sector, not tuning an allocation until links appear. Neither a unique
activation law nor a new weighted runtime is installed by these countermodels.
The [constitutive support controls](../../tests/physics/test_constitutive_support_scope.py)
separate native weighted-form observations, exact proposed support budgets
and the conditional receiver jets; they do not execute a weighted trajectory.

<a id="contact-and-internal-reorganization"></a>
## 7. Contact and internal reorganization belong to one joint state

The connection's effect on its endpoints is already part of the admitted
problem. In the continuous completion write, with held capacity,

\[
\dot X=F(X,a),\qquad \dot a=\mathcal H(X,a),\qquad X=(x,\theta).
\]

Changing `a` changes the weighted gradients, normalization and phase geometry
consumed by `F`; changing `X` changes the cost and local loss consumed by the
candidate support laws in section 6. These are two directions of dynamical
dependence. They do not establish a symmetric kinetic coefficient or select
`mathcal H`. Holding the *nodal law* fixed in that comparison did not freeze
the *nodal state* or its rates.

On a smooth admitted chart the chain rule gives

\[
\ddot X=D_XF\,F+(\partial_aF)\,\mathcal H.
\]

The first term retains internal nodal evolution; the second records the
instantaneous effect of contact growth on acceleration. The first-contact
witness above already has nonzero receiver acceleration from this second
term even though its initial receiver rates vanish. This is an alteration of
the subsequent nodal trajectory, not an instantaneous jump in `X`. The
result does not show that contact must alter every coordinate at every state.
For example, common form and phase give `q=g=0` on either compatible support;
the nodal field remains zero. Capacity is held by this model and cannot be
assigned a new evolution merely because a connection appears.

The storage also retains this joint dependence. With
`S_tilde=S_base(X)+a*c(X)+Psi(a)`, its exact balance is

\[
\dot{\widetilde S}
=\bigl[\nabla_X S_{\rm base}+a\nabla_Xc\bigr]\cdot\dot X
 +\bigl[c+\Psi'(a)\bigr]\dot a.
\]

In particular, `partial_a grad_X S_tilde=grad_X c`: the contact contribution
changes the nodal storage gradients too. The existing form/phase exchange
cancellation already uses these gradients at the current contact weight.
Reorganization and contact work must therefore be accounted for in one
balance. This identity supplies no extra reservoir and does not remove the
undetermined `Psi` or kinetics. If a further reciprocal transfer is proposed,
its change to the nodal rows must be specified and checked in this same
balance; counting the same work twice would not constitute a mechanism.

### Conditional kinetic-reciprocity audit

On the regular chart of the fully weighted extension, write
`S_tilde=S_base(X)+a*c(X)+Psi(a)` and
`k=partial_a S_tilde=c+Psi'(a)`. The mixed derivative identity is

\[
\partial_a\nabla_X\widetilde S=\nabla_X k=\nabla_Xc.
\]

This is reciprocity of the chosen state potential. It does **not** determine
the kinetic matrix. In particular, the existing nodal rows admit a conditional
block-diagonal completion: retain their form/phase exchange operator and add
only a nonnegative scalar support mobility `m`. The resulting support row is
`a_dot=-m*k` in the interior (with the declared projection at `a=0`). Its
contribution to the storage rate is $-m k^2$; the nodal rows remain unchanged
and contribute their existing $-\mathcal D$. Thus the joint rate is
$-\mathcal D-m k^2$ when the support row is active. This completion
demonstrates compatibility, not a selected law. With the current `Psi=0`,
`k=c>=0`, so this gradient row cannot give positive onset at the zero boundary.
A different boundary drive requires an additional support-storage or kinetic
premise, as in the conditional countermodel above.

By contrast, if one **additionally** requires an Onsager-symmetric cross
mobility between `X` and `a`, its support component contributes a term
proportional to `-nabla_X S_tilde` in `a_dot`, while the same cross block
contributes a term proportional to `-k` in `X_dot`. A nonzero reciprocal
kinetic block therefore changes the admitted nodal rows on a general open
state domain; preserving those rows requires its net nodal contribution to
vanish there, or requires explicitly revised nodal equations. An
antisymmetric reversible cross block has its own paired contributions and
the same row-preservation obligation. Neither type of cross block follows
from `partial_a grad_X S_tilde=grad_X k`, from passivity alone, or from the
two-way dependence of `F` and `H`. The storage choice, diagonal mobilities,
cross kinetics and any reversible exchange remain separate premises. This
conditional audit does not select an activation law or imply a regime
transition.

For hybrid contact the corresponding object is a complete reset
`(U,X)->(U_plus,X_plus)`. The
[full-reset owner](RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact)
already separates nodal reorganization from support work in the same action.
A support-only event can have `X_plus=X` while changing subsequent rates,
as the [one-bridge interface](RELATIONAL_SUPPORT_EVENTS.md#one-bridge-interface-admission)
shows. Actual UM and RA-then-UM controls instead retain their changes to nodal
state. Neither event contract selects its own invocation.

Finally, a primitive scalar node and a composite NFR have different state
cards. The former contains only the declared coordinates of its model. The
latter can reorganize through its internal fine nodes, their form/phase
configuration and any admitted support evolution. A regional mean is not
generally sufficient: the [composition and memory results](RELATIONAL_PATTERN_MEMORY.md)
retain internal degrees of freedom or their justified memory. This is the
existing route for investigating internal deformation, not evidence of new
unmodeled coordinates inside every scalar node.

These results make joint node/contact evolution an explicit requirement for
the next admission. The audit below classifies what could supply the missing
nonzero boundary drive; it derives none. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns what remains.

### Zero-boundary audit: what can drive a first contact

Keep the regular weighted extension, one supplied candidate `{u,v}` and held
capacities. The storage rate is `S_tilde_dot=-mathcal D+k*a_dot` with
`k=c+Psi'(a)`, `c>=0` and `mathcal D>=0`: the nodal part supplies a cost and a
budget, neither of which is a drive. At an **uncoupled equilibrium** `(X*,0)`,
each component has `q=Bx=0` and `g=0`; hence `F=0` and `mathcal D=0`.

**Passive onset needs a nonpositive support force.** Passivity at that state
gives `k* a_dot(0)<=0`. Boundary admission also requires `a_dot(0)>=0`.
Thus `k*>0` forces `a_dot(0)=0`, and positive instantaneous onset requires
`k*<=0`, i.e. `Psi'(0)<=-c(X*)`. If `c(X*)>0`, this requires a negative
relation-storage slope within this unchanged balance. At `c=0`, however,
`Psi=0` already gives `k=0`: no negative storage is necessary for a zero-cost
rate, and the inequality selects none. A zero initial rate alone does not
exclude higher-order onset away from a full equilibrium.

On the regular `a=0` boundary near `X*`, with `k>=k_0>0`,
bounded capacities and positive background strengths, loss is quadratic in q.
A passive row therefore obeys `a_dot<=mathcal D/k=O(|X-X*|^2)`. The rows `f_1,f_2` of
section 6 are of this kind: `mathcal B<=mathcal D` vanishes at equilibrium, so
they preserve it and open contact only while the port nodes still carry a form
gradient. At `k*=0`, passivity neither forbids nor selects growth.

**A derived relation coordinate has no independent drive.** Let `a=A(X)>=0`
with closed evolution `X_dot=F(X,A(X))`. The chain rule gives exactly
`a_dot=grad A . F`, so no support row remains to choose; `A` reads both
endpoints across components, a declared cross-component dependence. If `A` is
differentiable at an interior zero, that zero minimizes `A`, so `grad A=0` and
`a_dot=0`. If A is twice differentiable there, then
`a_ddot=F^T(grad^2 A)F>=0`; this may also vanish and does not guarantee onset.
A positive first-order rate requires leaving this smooth-interior-minimum
hypothesis, for example a kink with `a_dot=A'(X;F)` a one-sided derivative.
If along an outward family `A'(X;F)>=b*epsilon`, `b>0`, while
`mathcal D=O(epsilon^2)` and `k>=k_0>0`, passivity fails for sufficiently
small epsilon. This is a directional scaling condition, not a universal
exclusion of every nonsmooth gate.

*Exact witness.* Take forms `(1+s,1,0,0)` with `s<0`, zero phases, unit
capacities, `e=w=1/2`, `beta=1` and `Psi=0`. The native `a=0` fields give
`c=1/2`, `mathcal D=s^2` and `c_dot=s/2`, so the cost falls. With threshold
`kappa=c(0)`, the linear gate
`A=(kappa-c)_+/kappa` has `a_dot(0+)=|s|` and support work `|s|/2`; the
quadratic gate `A=((kappa-c)_+/kappa)^2` has `a_dot(0)=0` and
right acceleration `a_ddot(0+)=2*s^2`; this gate is C1, not C2 at its threshold.
The linear budget `|s|/2<=s^2` holds exactly for
`|s|>=1/2`: the first-contact fixture (`s=-1`) passes and a smaller internal
gradient does not. These are first jets of the closed derived evolution; they
select neither gate.

**Boundary storage slope is not a bistability theorem.** Retain the projected
gradient row above and a C2 potential, with `kappa=-Psi'(0)`. At zero weight
and component consensus, `S_base=0`. If `c(X*)>kappa`, continuity gives a
local constrained storage minimum, neutral along the component offset family.
If `c(X*)<kappa`, increasing a at fixed X lowers storage. With `m(X*,0)>0`
the support rate is then positive: this point is not an equilibrium of the
joint system, rather than an equilibrium proved unstable. With `m=0` even
that rate conclusion fails. Equality requires higher-order information.

If Psi has an interior global minimizer `a_*>0` with the additional strict
inequality `Psi(a_*)<Psi(0)`, the completely uniform connected state attains
that lower storage. Merely having a positive minimizer does not imply the
strict inequality. Nonincrease of a semidefinite storage and two static minima
alone establish neither two attracting regimes nor a bifurcation: positivity
of the kinetics, invariant neighborhoods, neutral directions and asymptotic
behavior must be checked. The existing controls evaluate slopes, storage
values and a Hessian, not those dynamical obligations.

| Candidate source | It must still supply | At an uncoupled equilibrium |
| --- | --- | --- |
| Nodal rows: cost `c`, loss `mathcal D` | Nothing | Cost obstructs and loss only funds: no drive |
| Passive allocation of loss | The allocation rule | Rate bounded by zero when `k>0`; zero-cost case separate |
| Gradient row with relation storage | `Psi'(0)` and a mobility | Positive rate for `c(X*)<kappa` only with positive boundary mobility |
| Smooth derived coordinate | The function `A` | No first-order onset at an interior zero; higher orders need checking |
| Kinked derived coordinate (gate) | `A` and its threshold | Fails under the stated linear-drive/quadratic-loss scaling |
| Hybrid reset | Candidate, time, reset and budget | [Existing event contracts](RELATIONAL_SUPPORT_EVENTS.md#support-law-choice-and-clock) |

This audit derives no first contact. It reduces the missing information to a
relation-storage slope, an allocation, a gate threshold or an event rule; the
nodal rows supply none of them and none is installed. The
[constitutive support controls](../../tests/physics/test_constitutive_support_scope.py)
check symbolic jets and represented storage/Hessian controls, including a
numerical spectrum. They execute no weighted trajectory.

### Composite internal storage reduces to the elimination obstruction

A composite NFR carries internal fine nodes, and the
[sufficient interface card](RELATIONAL_SUPPORT_EVENTS.md#8-one-supplied-bridge-sufficient-interface-and-endpoint-only-obstruction)
retains their state behind each port. One might expect that large internal
storage, such as a protected winding, could supply the negative slope the
scalar node lacks. It does not, for a first contact at an uncoupled
equilibrium: that internal storage is exactly the hidden-coordinate domain of
the [elimination argument](#autonomous-contact-completions)
in section 6.

Write the composite internal coordinates as `h` and the external partner state
as `y`. A single supplied candidate adds `a*c(h,y)` with the nonnegative port
cost `c=\tfrac12(x_u-x_v)^2+\beta(1-\cos(\theta_v-\theta_u))>=0`; the internal
graph, and hence the domain of `h`, does not depend on `a`. The reduced storage
is `S_eff(y,a)=\inf_h[S_int(h)+a*c(h,y)]`. By the envelope identity at a
differentiable interior minimizer `h*(a)`,

\[
\left.\frac{\partial S_{\rm eff}}{\partial a}\right|_{a=0}=c\bigl(h^*(0),y\bigr)\ge 0,
\]

because the implicit variation `dh*/da` is annihilated by internal
stationarity. The slope reads only the ports, at the composite's own internal
equilibrium `h*(0)`. Internal reorganization can fund contact only while the
chosen allocation has loss available. For the port-local allocation this needs
nonzero endpoint form gradients. Those nodal gradients vanish at equilibrium;
they are not the support derivative c, which may remain positive.

For a protected winding, use that fixed sector as the hidden domain, or retain
only the stationary-branch identity. A local minimum is not automatically the
global infimum over an unrestricted phase domain.

At a stationary internal preparation its first-order internal storage variation
vanishes; this does not make its whole stored cost constant during deformation.
A winding is retained along continuous paths avoiding its branch boundary.
Changing it can involve a phase reset on unchanged support or a path leaving
that protected chamber, not necessarily deletion of an internal cycle.
The [complete-reset accounting](RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact)
retains both state and support effects. Nonequilibrium dynamics leaves the
stationary-elimination premise, but need not change the hidden-state domain.

*Exact witness.* Take two disjoint acute winding-one rings as composites and
one supplied unit port bridge, read through the shared reset observer. On an
`n`-ring the equal-gap winding state has gap `2\pi/n<\pi/2`, local phase cost
`1-\cos(2\pi/n)` per edge with `\cos(2\pi/n)>0`, so it is a strict local
minimum of phase storage modulo common rotation in the fixed winding class.
Its value `n(1-\cos(2\pi/n))` differs with `n`
(`5(1-\cos 72^\circ)` against `7(1-\cos(360^\circ/7))`), yet with aligned ports
the first-contact storage change is exactly zero for both. Offsetting the port
phase by `\delta` gives change `1-\cos\delta`, and an EPI contrast `r` at the
ports gives `r^2/2`; both are nonnegative and read only the ports, never the
internal winding. These are detached storage snapshots of the port-cost jump,
not a weighted trajectory or an occurrence law.

This is the composite branch of the stationary, fixed-hidden-domain
obstruction, not an exclusion of every composite mechanism. Moving constraints,
nonequilibrium internal dynamics and new support-dependent storage terms must
be assessed with their own complete laws.

<a id="nonequilibrium-internal-memory-is-a-prepared-dissipating-excess"></a>

### Nonequilibrium memory retains preparation and evolving boundaries

Static elimination assumed the internal coordinate sits at its equilibrium.
A genuinely distinct branch keeps a slow internal coordinate `h` out of
equilibrium. Its exact evolution retains the initial state,

\[
h(t)=e^{A_ht}h_0+\int_0^t e^{A_h(t-s)}R_h(y(s),h(s))\,ds,
\]

through the [hidden-memory owner](RELATIONAL_PATTERN_MEMORY.md#3-exact-hidden-memory-retains-its-initial-condition).
The initial state must be retained, but a zero h0 need not stay zero: the
nonlinear source and moving boundary can generate hidden state. The existing
[zero-hidden-state counterexample](RELATIONAL_PATTERN_MEMORY.md#5-zero-initial-hidden-state-is-not-an-invariant-preparation)
already makes this distinction.

For the [fast-mediator chart](RELATIONAL_MEDIATOR_DYNAMICS.md#fast-mediator-reduction),
`E_exc=S_full-S_eff=u^2+2*beta*cos(delta/2)*(1-cos v)>=0` on its stated domain.
This is excess over the instantaneous frozen-boundary reconstruction, not
necessarily over a full joint equilibrium. Along the full law,
`E_exc_dot=-mathcal D-grad(S_eff) dot F(y,z)`, so total storage loss alone
does not fix the sign of the excess derivative.

A static P3 control makes this concrete without a new solver. Put the mediator
at node 1, forms `(1,1/2,0)`, phases zero, capacities `(1,3,2)` and
`e=w=1/2`, `beta=1`. The same two-edge midpoint decomposition has zero initial
excess. Native rates give `u_dot=-1/8`, `v_dot=1/(8*pi)` relative to the moving
endpoint means, while total storage rate is `-3/8`. Consequently
`E_exc_dot(0)=0` and `E_exc_ddot(0)=(1+1/pi^2)/32>0`.
Hidden excess can grow while total storage falls; its presence is not always
an independently supplied nonzero hidden preparation. The
[shared-field control](../../tests/physics/test_relational_coefficient_scope.py)
keeps represented rates distinct from this ideal derivative identity.

The prepared P2 storage/loss check remains a valid special case, not a theorem
of monotone hidden memory. None of these rewrites alone selects a support row,
but neither do they exclude a separately justified internal mechanism.

<a id="declared-forcing-or-gamma-is-an-external-input-not-a-derivation"></a>

### Declared forcing changes work, not the support gradient

The live Gamma registry supplies `dx/dt=nu_f*p+Gamma`. If only this form source
is added to the weighted comparison, the chain rule becomes
`S_tilde_dot=-mathcal D+q^T Gamma+k*a_dot`. Source work can have either sign;
moving form at zero capacity does not by itself fund contact. In particular
`q=0` gives zero instantaneous source work despite a possible nonzero form rate.
Gamma changes the work balance, not the storage derivative `k=c+Psi'(a)`.

The source law and provenance must be declared, whether it is prescribed
externally or depends on retained state. The unforced relational executor
rejects nonzero Gamma; this forced comparison is not installed. Neither a
source nor its budget alone selects support evolution or event occurrence.

### The origin of relation storage is a nonselection classification

The reviewed branches expose remaining premises; they are not an exhaustive
classification of all possible support laws. A negative support slope, work
available to pay positive support cost, and a rule selecting occurrence are
different quantities. At an uncoupled equilibrium the unchanged unforced
balance has vanishing nodal loss; its zero-cost case remains separate.

| Mechanism reviewed | Remaining obligation |
| --- | --- |
| Relation-storage potential | Storage slope and kinetics; negative slope is not required at zero cost |
| Loss allocation | An allocation of contemporaneous loss, not a negative support gradient |
| Derived relation coordinate | Chain-rule closure, domain and any gate premise |
| Stationary composite reduction | Nonnegative slope on the fixed hidden domain; moving constraints are outside this proof |
| Nonequilibrium memory | Initial state and boundary-driven dynamics; hidden excess need not decay monotonically |
| Declared forcing/Gamma | Signed source work and a support law; forcing alone does not determine a storage slope |
| Hybrid reset | Candidate, time, reset and budget via the [event contracts](RELATIONAL_SUPPORT_EVENTS.md#support-law-choice-and-clock) |

The distinct autonomous countermodels in section 6 prove nonselection by the
compared premises, not impossibility of autonomous continuous formation. The
event-law comparison retains different support and occurrence hypotheses;
placing both results in a table does not exclude every future completion.
Primitive formation and independent selection of its law remain open. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next bounded admission, not a theorem that these premises are exhaustive.

### A possible regime transition is an open interpretation

The passage from separately evolving patterns to a maintained coupled
organization could be investigated as a transition of the joint dynamics.
This meaning of transition is distinct from changing the angular phase
`theta` of a node. The current smooth positive-onset examples establish
neither a critical threshold nor a change of stability or a maintained
coupled regime. A weight becoming positive is not sufficient evidence for
those claims.

Once a joint law is justified, identify its uncoupled and coupled invariant
states or persistent regimes and the declared preparation or model parameter
being varied. A stability change or coexistence of distinct regimes would
provide a concrete mechanism to examine; a threshold assigned to a diagnostic
would not derive it. The gradient row of the
[zero-boundary audit](#zero-boundary-audit-what-can-drive-a-first-contact)
locates a boundary-force sign change for a supplied potential, not by itself
a bifurcation or two attracting regimes. Its kappa is declared and the
projected joint dynamics needs the separate stability analysis above.
A finite-network dynamical bifurcation, if
established, would retain its own scope rather than establish a physical
thermodynamic transition. This is an interpretation criterion for the same
contact study, not a new campaign or an additional phase-transition law.

## 8. Reuse and the remaining foundation

The [normalized-neighbor owner](../../src/tnfr/mathematics/_neighbor_differences.py),
[support derivative](../../src/tnfr/physics/support_transport.py) and
[phase geometry](../../src/tnfr/physics/phase_response.py) supply the existing
calculations. Their fixed-active-edge derivative deliberately excludes birth.
The [shared pressure controls](../../tests/core_physics/test_stable_neighbor_pressure.py)
check the zero-strength and positive-background limits on five execution paths.
The [relational executor](../../src/tnfr/dynamics/relational.py) remains
unit-support execution; none of the weighted alternatives above is installed.
Existing [attachment controls](../../tests/test_relational_attachment.py) and
[actual operator reset controls](../../tests/physics/test_coupling_attachment_budget.py)
exercise distinct hybrid cases without selecting their occurrence.

The foundational obligation is to specify which relation is being formed,
its zero state, candidate access, normalization and work accounting before
deriving its evolution. Current nodal dynamics already derives effective
relations on supplied fine support. Continuous creation of primitive relations
requires a justified completion at the boundary, while hybrid creation requires
a justified event mechanism. These alternatives remain revisable; neither
negative result proves that TNFR can never supply such a completion.
