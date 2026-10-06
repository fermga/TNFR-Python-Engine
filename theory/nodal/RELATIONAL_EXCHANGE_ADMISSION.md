# Native relational exchange admission

This owner derives a conditional native joint law from explicit storage,
locality and capacity premises on supplied support. Signed form, circular
phase, held capacities and the declared structural clock remain part of the
model. The [opt-in implementation](#relational-engine-integration) preserves
its execution contract; these premises do not select a unique physical law.

<a id="consolidated-claim-boundary"></a>

## Chapters and scope

| Chapter | Responsibility |
| --- | --- |
| [Relational coefficient and capacity identification](RELATIONAL_RESPONSE_IDENTIFICATION.md) | Spectral and finite-response discrimination, temporal acquisition and capacity-intervention evidence. |
| [Sine law and constitutive information boundaries](SINE_CONSTITUTIVE_INFORMATION.md) | Global sine comparison, balance/locality premises, storage alternatives and phase-information restrictions. |
| [Native local recovery and regional interaction](RELATIONAL_RECOVERY_AND_INTERACTION.md) | Full-law recovery, equilibrium stiffness, regional transmission and work accounting. |
| [Native phase domains and capture](RELATIONAL_DOMAIN_AND_CAPTURE.md) | Regular-domain continuation, winding passage, reflected basins and full-state sector barriers. |
| [Native validated transit and preparation controls](RELATIONAL_FORMATION_CONTROLS.md) | Continuous transit, bounded law changes and preparation controls with their original evidence scope. |

<a id="reuse-without-changing-the-complete-law"></a>

Shared form/phase variables do not make native and
[smooth-sine results](SINE_PATTERN_DYNAMICS.md#reading-map) interchangeable.
A closure theorem, storage identity or contact response consumes its complete
law and retained state; it neither selects pressure nor removes independent
hidden initialization. Formation-to-maintenance handoffs must preserve those
premises and the full environmental state. Continuous theorems, admitted
finite steps and validated trajectory enclosures provide different evidence.

Section numbers remain stable. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone assigns research work.

## 1. State, inherited geometry and the independent premise

Fix a finite connected simple reciprocal graph with at least two nodes, unit
conductances, no loops, and held capacities `N=diag(nu_i)>=0`. Write
`D=diag(d_i)`, `B=D-A` and `L=D^-1 B`. All degrees are positive. Use signed
real form `x`, primitive circular phase `theta`, and a regular phase chart:
every neighbor resultant is nonzero and its displacement from the node phase
lies in `(-pi,pi)`. There are no inputs, events, clipping or capacity changes.

The existing [phase metric](../TNFR_VARIATIONAL_PRINCIPLE.md#136-exact-state-dependent-metric-for-canonical-phase-pressure)
and [Dirichlet balance](../TNFR_VARIATIONAL_PRINCIPLE.md#35-exact-restricted-dirichlet-balance)
supply, without a new primitive,

\[
g_i=\frac{\operatorname{wrap}(\arg\sum_{j\sim i}e^{i\theta_j}-\theta_i)}\pi,
\quad H_i=\pi\left|\sum_{j\sim i}e^{i\theta_j}\right|
             \operatorname{sinc}(\pi g_i)>0,
\quad Hg=-\nabla V_\phi,
\]

\[
V_\phi=\sum_{\{i,j\}\in E}(1-\cos(\theta_j-\theta_i)),\qquad
E_D=\tfrac12x^TBx,\qquad q=\nabla E_D=Bx.
\]

Within the explicitly edge-additive storage class,
[native-pressure alignment](../TNFR_VARIATIONAL_PRINCIPLE.md#native-phase-storage-classification)
selects the phase cost above up to a positive factor. Continuity selects the
metric's values at zero phase pressure. We use unit cost normalization, so
the free positive factor is absorbed into `beta`. This restricted result
does not itself derive Dirichlet form storage, joint separability or physical
energy. Section [11.3](#joint-storage-locality-classification) separately derives
the quadratic/cosine split from an admitted joint edge cost, prescribed exact
loss and uniform primitive-local completion with e,w>0. It does not replace
the prescribed loss by mere passivity or identify physical energy.

Retain the native two-channel form row, with fixed `e>=0`, `w>0`:

\[
\dot x=N(-eLx+wg).
\]

For an engine comparison, `e,w` must be the effective coefficients after its
weight normalization; with only these channels active their sum is one.

The new **independent premise** is that `E_D+beta V_phi`, `beta>0`, measures
joint storage and that phase/form transfer cancels at each nodal work port:

\[
\boxed{\quad w\nu_i q_i g_i-\beta H_i g_i\dot\theta_i=0
       \quad\text{for every }i.\quad}
\]

These are the coordinate contributions to the total storage chain rule;
they do not assert an independent locally conserved energy at each node.
The premise is stronger than zero **total** transfer. It chooses which
nodal contributions cancel, a storage ratio and a phase reference for the
exchange row. Global phase rotation makes total phase work unchanged, but
adding an arbitrary common phase velocity generally breaks nodewise
cancellation. Neither the nodal product nor the phase metric forces this
extra premise. Its merit here is a precise, falsifiable dependency record.
Section 11 gives an alternative admission: independent local capacity
dependence and zero-capacity freezing derive the nodewise cancellation from
global balance. Those capacity conditions are additional premises too.
Section [11.1](#primitive-locality-exchange-selection) gives another route
using a uniform primitive-neighborhood information contract across supports,
while retaining the zero-capacity phase condition and chosen storage.
Section [11.2](#primitive-locality-phase-clock) removes individual phase
freezing and classifies the remaining freedom as a common clock only.
Adding `gamma*1` to the phase velocity changes local allocations but not
their sum, since `1^T Hg=0`. It also leaves `J_g*dot(theta)` unchanged.
The displayed representative fixes this common-rotation convention; it
does not derive an absolute phase clock.

## 2. Forced phase row and its domain

At `g_i!=0`, cancellation fixes the phase velocity. Requiring that velocity
to be continuous on the full regular phase chart gives

\[
\boxed{\quad \dot\theta=\frac w\beta H^{-1}NBx.\quad}
\]

This extension is forced, not a division by zero at `g_i=0`. Since the
support has no loops, holding neighboring phases fixed and varying
`theta_i` gives `partial_i g_i=-1/pi` in a sufficiently small regular chart.
The points `g_i!=0` are therefore dense locally. Equality there fixes any
continuous extension at its zeros. For `nu_i=0` the chosen joint law freezes
both local rows; the nodal identity alone would freeze only form.

Both rows use the node and its neighbors. The pressure is unchanged:
there is no cotangent `Kx` correction, hidden mean projection or reconstructed
pressure. The phase row is a new conditional law, not the default runtime's
phase coordination or the separate independent-pressure extension.

## 3. Complete source, storage and mean balances

With `J_g=Dg=(R-I)/pi` from the shared
[phase response owner](../../src/tnfr/physics/phase_response.py), the source
and complete pressure satisfy

\[
\dot g=\frac w\beta J_gH^{-1}NBx,\qquad
\dot p=-eL\dot x+w\dot g.
\]

This derives source evolution from the proposed phase row, before observing
an EPI response. It does not define pressure from an observed derivative.
The storage identity is exact on this regular chart:

\[
\begin{aligned}
\dot E_D&=-e q^TND^{-1}q+wq^TNg,\\
\beta\dot V_\phi&=-w g^TNq,\\
\boxed{\dot{\mathcal E}=-e q^TND^{-1}q\le0},
\qquad\mathcal E=E_D+\beta V_\phi.
\end{aligned}
\]

This admits heterogeneous and zero held capacities without inverse-capacity
storage. It uses the actual nonnegative diagonal `ND^-1`; no Euclidean
dissipation assumption about `NL` is needed. The storage is semidefinite in
form, retaining the common-offset freedom, rather than controlling absolute
EPI. The complete arithmetic mean remains

\[
\frac d{dt}\overline x=
-\frac e n\mathbf1^TND^{-1}Bx+
 \frac w n\mathbf1^TNg.
\]

If all capacities are positive, the fixed diffusion weights
`rho_i=d_i/nu_i` instead give

\[
\frac d{dt}\frac{\rho^Tx}{\rho^T\mathbf1}
=\frac{w\,d^Tg}{\rho^T\mathbf1}.
\]

The native Arg source need not make this last numerator zero. Relational
storage and origin covariance therefore do not imply conserved form mean.
At zero capacities use the complete raw mean formula, not `rho`.
The existing [passive-transfer budget](../TNFR_VARIATIONAL_PRINCIPLE.md#1318-source-sensitivity-joint-work-and-the-passive-transfer-limit)
explains why a nonincreasing finite storage cannot establish indefinite
nonzero dissipative form contrast. This candidate supplies exchange from
prepared phase structure, not an inexhaustible maintenance mechanism.

A conditional forward domain is available without simulation. If
`E(0)<beta`, let `a=acos(1-E(0)/beta)<pi/2`. Nonincrease keeps every edge gap
at most `a`, all resultants at least `d_i*cos(a)`, and
`H_i>=pi*d_i*cos(a)*sinc(a)>0`. Dirichlet storage bounds centered form on
this connected graph. The source and complete mean velocity are bounded,
so absolute form remains bounded on every finite interval. These estimates
keep the smooth quotient field away from its singularities and give global
ODE continuation under the stated held law. They do not prove persistence.
The [regular-domain theorem](RELATIONAL_DOMAIN_AND_CAPTURE.md#regular-domain-continuation-and-boundary-access)
below enlarges this sufficient domain to
`E(0)<beta*max(2,d_min)`; that larger domain need not keep all edges acute.

## 4. Origin, units and exact replication

For a constant form shift `x_new=x+c*1`, both rows and storage are unchanged,
because `B*1=L*1=0`. This is an actual symmetry of the stated joint model;
it is not obtained by suppressing the mean equation.

For chart and clock conversion `x_new=a*x`, `t_new=b*t`, `a,b>0`, retain
the same phase and dimensionless graph, and transform

\[
N_{new}=N/b,\qquad e_{new}=e,\qquad
w_{new}=a w,\qquad \beta_{new}=a^2\beta.
\]

Then both velocities transform correctly and `E_new=a^2 E`. In this
convention capacity has inverse-clock units, `w` has form units and `beta`
has squared-form units. The usual numerical coefficients can be
dimensionless in a chosen normalized form chart. Changing form units while
retaining every raw configured coefficient would change the model.
This covariance does not determine the invariant ratio `w^2/beta`.
Equivalently one can hold numerical capacity fixed and absorb the clock
conversion into `e_new=e/b`, `w_new=a*w/b`; this is a representation choice,
not a change to only one term of the nodal product.

If the two-channel engine normalization `e+w=1` must be retained, define
`k=e+a*w`. The equivalent representation has `e_new=e/k`,
`w_new=a*w/k`, `N_new=k*N/b` and `beta_new=a^2*beta`. The capacity
rescaling is essential: simply renormalizing the transformed pressure
weights changes both rates. This reparametrization belongs to the declared
two-channel held-capacity family; it does not preserve arbitrary capacity
source terms, adaptation policies or other engine channels.

The existing [counted-support construction](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#joint-constitutive-reduction-with-inherited-support-counts)
also supplies an exact comparison. Replace each base node by `m` replicas,
with no within-block edges and all `m^2` unit edges for each original edge.
Prepare equal form, phase and capacity within each block. Permutation
equivariance preserves that prepared submanifold. On it,

\[
g_{fine}=g,\quad L_{fine}x=LX,\quad
(B_{fine}x)_{i\alpha}=m(BX)_i,\quad H_{fine,i\alpha}=mH_i.
\]

Consequently both joint rows descend with unchanged `beta`, capacity and
clock. Total `E_D` and `V_phi` each scale by `m^2`. This differs from the
cotangent comparison's inherited exchange scale `eta_eff=m`; the difference
comes from the storage/exchange premise, not from new fine-node capacity.
The partition and synchronized preparation are supplied. No block formation
or transverse stability is inferred.

## 6. Why changing cotangent storage alone cannot retain native pressure

There is an independent obstruction on unit P2 with unit capacity. Let
`delta=theta_1-theta_0` lie in `(-pi,pi)`,
`h(delta)=pi*sinc(delta)` and `m=(x_0+x_1)/2`. Native pressure gives
`m_dot=0`. Suppose instead that `pi_i=h*x_i` is canonical momentum and
the Hamiltonian is invariant under common phase rotation. Noether momentum
is `P=2h*m`; the native diffusion contributes zero to its rate. Therefore

\[
0=\dot P=2m h'(\delta)\dot\delta.
\]

For `delta!=0`, `h'(delta)!=0`: the odd function
`sin(delta)-delta*cos(delta)` has positive derivative `delta*sin(delta)`
on `(0,pi)`. On an open full-state domain, `m!=0, delta!=0` is dense, so
continuity forces `delta_dot=0` everywhere, including zero-mean and
consensus points. This does not assume a quadratic form storage.

More precisely, set `q_theta=(theta_0+theta_1)/2` and
`J=h*(x_1-x_0)/2`. The canonical one-form is
`P*dq_theta+J*ddelta`. Rotation invariance and `delta_dot=0` imply locally
that the Hamiltonian is independent of `q_theta` and `J`. The native phase
source then forces, on connected fibers,

\[
\mathcal H=w(1-\cos\delta)+F(P).
\]

Thus this premise cannot retain native pressure and supply nontrivial
relative-phase exchange by choosing a different relative-form storage.
The isolated zero-mean sector is not an open full-state remedy. The
[cotangent P2 results](../TNFR_VARIATIONAL_PRINCIPLE.md#cotangent-p2-nonzero-momentum)
remain valid because their corrected form row differs from native pressure.
Other momentum maps, non-Hamiltonian interconnections and constrained
models are not excluded.

## 7. Admission verdict and remaining information

The dependency is explicit: **relational storage plus nodewise exchange**
forces the phase row; the existing nodal pressure then fixes source evolution,
mean drift, loss and the spectral discriminator. The new premise avoids the
cotangent pressure correction by making a different constitutive choice.
Its skew power-canceling representation need not satisfy Jacobi; it is not
promoted to a Hamiltonian law.

Remaining choices are the storage itself, `beta`, the capacity-composition
or nodewise-cancellation premise, phase-reference convention and admitted
support/capacity scope. Sections 10-12 distinguish what global balance forces,
what capacity separability additionally selects and which alternatives remain.
The nodal equation does not select those premises. Their independent physical
justification remains open. The capacity-separable family is admitted for
conditional implementation and testing; no autonomous NFR, physical energy
or universal law is claimed.

Controls: [relational exchange admission](../../tests/physics/test_relational_exchange_admission.py)
uses exact algebra for the premise, P2 obstruction, units, spectral ratios and
replica reduction. A separate finite local response check reuses native
pressure, the shared transport snapshot and `derive_joint_nodal_response`,
with represented-coefficient errors distinguished from symbolic identities.
It evaluates no formation trajectory and installs no phase-law default.

## 8. Comparing the premises before selecting a model

<a id="relational-cotangent-premise-comparison"></a>

Neither completion follows from the nodal product alone. Their exact
consequences should be compared against explicit requirements, not ranked by
whether a pressure formula is already implemented:

| Requirement | Relational exchange | Cotangent exchange |
| --- | --- | --- |
| Retain the native two-channel pressure | Yes | Requires the derived `Kx` correction |
| Treat a common EPI offset as redundant | Exact symmetry of both rows | Absolute form enters momentum and storage; a coordinate shift must transform them too |
| Depend on primitive state within one hop | Both rows do | The correction generally reaches two hops |
| Admit held heterogeneous and zero capacities with nonincreasing storage | Section 3 supplies the balance | The original pullback/storage has the section 13.24 obstructions |
| Supply genuine Poisson geometry | Power cancellation does not establish Jacobi | The coordinate pullback establishes Jacobi and a conditional Noether charge |
| Reduce equal coherent complete replicas | Same rows with unchanged `beta` | Inherits `eta_eff=m*eta` |

For preservation of the present relational pressure sector, the relational
candidate requires a smaller change to that sector and admits more of its
held-capacity domain. This is a reason to prioritize its conditional study,
not a proof that its premises are physically preferable. If a canonical
momentum and genuine symplectic geometry are independently required, the
cotangent construction supplies an advantage the relational balance alone
does not. Offset redundancy, strict one-hop locality and these particular
storage choices are not universal TNFR axioms; the
[pressure-premise owner](PRESSURE_CONSTITUTIVE_SCOPE.md#42-when-scale-covariance-and-regularity-force-linear-form-response)
already distinguishes offset symmetry from changing a physically relevant
form origin. Neither candidate has independent physical validation here.

**A possible justification for relational storage, with its assumptions
visible.** Suppose form storage is additive over undirected support edges,
uses the same scalar function `U` of the signed endpoint difference on every
edge, has no other edge attributes, and does not depend on edge orientation.
Require `U(0)=0`, nonnegative nontrivial storage and exact quadratic amplitude
homogeneity `U(a*r)=a^2*U(r)` for every `a>0`. Orientation independence gives
`U(-r)=U(r)`; taking `a=abs(r)` gives

\[
U(r)=U(1)r^2=\frac{\kappa}{2}r^2,\qquad
S(x)=\sum_{\{i,j\}\in E}U(x_j-x_i)=\kappa E_D(x),\quad\kappa>0.
\]

Dividing the joint storage by `kappa` leaves the relative phase-storage
scale `beta`. This selects the displayed form storage within that class,
but the nodal equation does not impose edge additivity, identical edge
contributions or quadratic homogeneity. Nor does the argument derive
nodewise work cancellation or fix `beta`.

**A nonlinear experiment that cannot distinguish the candidates.** On P2
with unit capacity, let `x=(a,-a)` and `delta=theta_1-theta_0`, with
`h(delta)=pi*sinc(delta)>0`. The invariant zero-mean cotangent sector has
`Kx=0`. With its general exchange scale `eta>0`, both models obey

\[
\dot a=-2ea+w\delta/\pi,
\qquad
\dot\delta_{\rm cot}=-\frac{2a}{\eta h(\delta)},
\qquad
\dot\delta_{\rm rel}=-\frac{4wa}{\beta h(\delta)}.
\]

Their common phase velocities are zero. Thus `beta=2*w*eta` makes the
complete nonlinear sector identical, including its diffusion, rather than
merely matching a linear frequency. Relational storage is then twice the
cotangent storage on this sector. A symmetric two-node exchange or one
fitted response cannot select these premises. The two distinct graph modes
in section 5 instead supply an invariant ratio after the common scale is
eliminated; section 9 supplies the finite prepared-state domain and error bounds.

**Counting edges is distinct from counting nodes.** Disjoint copies add
both relational storages linearly. The complete replica construction of
section 4 multiplies every base edge into `m^2` edges, so its `m^2` storage
scaling is consistent with edge additivity. Node-count extensivity under
that different topology is an additional requirement, not a contradiction
of the stated law. The inherited cotangent scale likewise cannot be reset
merely by naming a coherent block one node.

## 10. What global exchange already forces

<a id="global-exchange-consensus-law"></a>

The consensus discriminator has weaker premises than the full nodewise
exchange law. Retain section 1's finite connected simple reciprocal unit
graph, held `N>=0`, native form row and storage `E_D+beta*V_phi`. Let
`F(x,theta)` be a phase velocity continuous on the full regular phase chart.
Instead of nodewise cancellation, require only the global exchange identity

\[
w(Bx)^TNg(\theta)+\beta\nabla V_\phi(\theta)^TF(x,\theta)=0
\]

for all admitted form and phase states. This is precisely the exchange
condition remaining after the known diffusion loss is subtracted from the
joint storage derivative. It does not impose a portwise decomposition.

Fix any admitted `x`, not necessarily small, and any common phase `c`.
For an arbitrary real vector `v`, the regular perturbation
`theta=c*1+epsilon*v` gives

\[
g(\theta)=-\frac{\epsilon}{\pi}D^{-1}Bv+o(\epsilon),\qquad
\nabla V_\phi(\theta)=\epsilon Bv+o(\epsilon).
\]

Divide the global identity by `epsilon` and use continuity of `F`. Since
`v` is arbitrary and `B` is symmetric, the result is

\[
B\left[\beta F(x,c\mathbf1)-\frac w\pi D^{-1}NBx\right]=0.
\]

Connectedness makes `ker(B)=span{1}`. Thus

\[
\boxed{F(x,c\mathbf1)=\frac{w}{\beta\pi}D^{-1}NBx
       +\gamma(x,c)\mathbf1.}
\]

This conclusion includes heterogeneous and zero held capacities. The
remaining common rotation need not pause at a zero-capacity node; pausing
every phase row there was part of the stronger nodewise convention. At
consensus `J_g=-L/pi` and `J_g*1=0`, so the source derivative is fixed:

\[
\dot g=-\frac{w}{\beta\pi^2}L D^{-1}NBx.
\]

With unit capacity, the pure-mode phase projection and form acceleration
of section 9 are therefore forced by this weaker identity too. Common
rotation changes neither projection nor pressure response. The ratios
`3` and `9` do not depend on selecting the nodewise completion away from
consensus. These are consequences of the chosen storage and global balance,
which remain constitutive postulates, not of the nodal product alone.

Away from consensus, let `F_rel=(w/beta)*H^-1*NBx`. Global balance allows
`F=F_rel+Z` whenever `grad(V_phi)^T Z=0`. At a noncritical phase state this
is one linear constraint on an `n`-component velocity. Common rotation
already lies in that kernel; quotienting it leaves `n-2` instantaneous
relative directions before imposing locality, symmetry or compatibility
across states. Counting these directions does not construct a continuous
local law that realizes them. On P2 the regular noncritical domain has no
remaining relative direction, and continuity fixes its consensus limit as
well. Larger networks retain a separate admission question: whether the
additional structural requirements remove or retain relative motion along
phase-storage level sets. No longer trajectory can replace that premise
test merely by confirming a selected completion's response.

## 11. Capacity separability can select the complete exchange row

<a id="capacity-separable-exchange"></a>

Retain section 10's graph, native two-channel pressure, joint storage and
global exchange identity. Regard held capacity as an independently variable
vector `nu` in the full nonnegative orthant. Require the same constitutive
phase law `F(x,theta;nu)` to obey that identity for every such vector, with
the following additional admission conditions:

1. At fixed form, phase and support, row `F_i` depends on capacity only
   through its own `nu_i`.
2. `F_i=0` whenever `nu_i=0`.
3. The phase law is continuous across the full regular phase chart.

No linear dependence on form or capacity is assumed. These conditions force

\[
\boxed{F_i(x,\theta;\nu)=\frac{w}{\beta}\,
             \frac{\nu_i(Bx)_i}{H_i(\theta)}}
\]

throughout that chart, not only at consensus. In particular, nodewise work
cancellation is then a consequence of the admitted capacity structure and
global balance, rather than a separate cancellation premise.

**Proof.** Write `q=Bx`. Hold `x,theta` fixed and set every capacity except
`nu_i` to zero. Their phase rows vanish by condition 2. Row `i` has the
same value as before the change, by condition 1. Global balance becomes

\[
g_i\left(w\nu_i q_i-\beta H_iF_i\right)=0.
\]

For `g_i!=0` this determines the displayed row. At `g_i=0`, hold neighboring
phases fixed and vary `theta_i` inside a sufficiently small regular chart.
The loopless support gives `partial_i g_i=-1/pi`, so nonzero `g_i` states
are locally dense. Both the proposed row and `F_i` are continuous, while
`H_i>0`; equality extends to the zero set. This also covers zero capacity.

**A compositional formulation.** Instead of assuming condition 1, require
capacity additivity at fixed remaining state:

\[
F(x,\theta;\nu+\mu)=F(x,\theta;\nu)+F(x,\theta;\mu),
\qquad \nu,\mu\ge0.
\]

Together with the zero-row condition, decomposition into coordinate-axis
capacity vectors gives
`F_i(nu)=sum_j F_i(nu_j*e_j)=F_i(nu_i*e_i)`. Thus additivity supplies the
needed separability, and the same proof fixes the row. Additivity expresses
an independent composition rule for capacity contributions; it is not
implied by the nodal product. Common clock homogeneity
`F(a*nu)=a*F(nu)` alone does not give this rule or exclude nonlinear
dependence on capacity ratios.

Likewise an affine capacity law `F_i=b_i+sum_j T_ij*nu_j` plus the zero-row
condition forces `b_i=0` and `T_ij=0` for `j!=i`, providing another sufficient
admission. Affinity is not a weaker assumption than allowing arbitrary
own-capacity dependence; the theorem derives that dependence's linearity.

**Scope.** The zero-row condition also fixes a common-rotation convention:
an extra capacity-independent phase clock is not admitted. Neither phase
freezing at zero capacity nor independence from neighboring capacities is
already required by `xdot=nu*p`. Holding one capacity vector, or varying
only a common capacity multiplier, is insufficient for the isolation proof.
The existing form pressure's capacity-contrast channel is inactive here;
restoring it changes the global work identity and falls outside this result.
Capacity evolution likewise introduces separate closure and storage duties.
The selected row uses only neighboring form and phase. Spatial dependence
on one fixed support alone does not supply the capacity-composition rule;
the graph-family locality premise in section 11.1 is a different sufficient
restriction when combined with the same balance and zero-row conditions.

This is a conditional model-admission theorem, not a declaration that the
new law is canonical or physically selected. The remaining foundational
question is whether capacity really composes through independent local
activity, or whether neighboring capacities mediate additional phase
exchange. The distinction must be justified from a capacity mechanism or
independent evidence, not by giving a preferred completion a new label.

### 11.1 Uniform primitive locality supplies a different selection premise

<a id="primitive-locality-exchange-selection"></a>

Section 11 assumes independence from neighboring capacities. The following
alternative derives that independence from a **uniform local information
contract across supports**, the same global storage balance, and zero-capacity
phase freezing. It does not derive locality or freezing from the nodal product.

**Information and domain.** Fix the same positive `w,beta` and nonnegative `e`
on every finite connected simple unit graph with at least two nodes. Admit
arbitrary real forms, held nonnegative capacities and the full regular phase
domain of section 1. A row can consume the induced rooted neighborhood

\[
\mathcal B_i=G[\{i\}\cup N(i)]
\]

and the primitive `(x,theta,nu)` values on its vertices. This includes the
root's degree and edges between its neighbors. It excludes total neighbor
degrees, their external incidences, graph size, remote states and derived
neighbor pressure/gradients that depend on those excluded data. Model
coefficients and the structural clock are fixed and shared, not extracted
from global diagnostics. Require the same row on isomorphic rooted data,
regardless of how the graph continues outside that neighborhood. This is
stronger than sparse dependence on state on each separately supplied graph:
it also excludes hidden support-specific coefficients.

Suppose the phase law `F` is continuous in the admitted primitive phases,
vanishes at row `i` whenever `nu_i=0`, and satisfies

\[
\beta\sum_i a_i F_i+w\sum_i\nu_i q_i g_i=0,
\qquad a=\nabla V_\phi=-Hg,
\]

at every admitted state on every such graph. Neighboring capacities may enter
`F_i` arbitrarily at this stage; no capacity additivity, linearity, form
covariance or capacity differentiability is assumed. Then necessarily

\[
\boxed{F_i=\frac w\beta\frac{\nu_i q_i}{H_i}}
\]

on the full regular domain. This also holds if the law is posed only on
strictly acute edges with all other premises unchanged.

**Proof by a local completion.** Retain any root neighborhood `B_i`, with
all its forms, capacities, phases and internal edges, initially assuming
that no internal edge has antipodal phases. For each neighbor `j` of the
root and each `k` adjacent to `j` within `B_i`, add three private leaves
attached only to `j`: one with phase `2*theta_j-theta_k` and two with phase
`theta_j`, all interpreted on the circle. Give every new leaf zero capacity
and any finite form. No new vertex touches the root, so its entire admitted
information is unchanged. At `j`, each old incidence and its three new
leaves contribute the relative phasor sum

\[
e^{\mathrm i(\theta_k-\theta_j)}
+e^{-\mathrm i(\theta_k-\theta_j)}+2
=2[1+\cos(\theta_k-\theta_j)]>0.
\]

Thus every original neighbor has `a_j=g_j=0` and positive metric. Each new
leaf has a single nonantipodal neighbor, hence a nonzero resultant and a
regular displacement; its phase and form velocities vanish by zero capacity.
The root retains its original regular resultant. The completed graph is
finite, simple, connected and wholly admitted, including for nonacute
internal edges. The two aligned leaves prevent a negative resultant at a
neighbor when a gap is obtuse. In the acute restriction all newly added
edges remain acute as well.

Global exchange on this completed graph therefore reduces to

\[
g_i(w\nu_i q_i-\beta H_i F_i)=0.
\]

For `g_i!=0`, this forces the displayed row. Uniform locality transfers it
back to the original graph, since the root's data did not change. To cover
the exceptional states, perturb the finitely many neighborhood phases
arbitrarily slightly while remaining in the original open regular domain.
Avoiding antipodal internal edges is dense. At a regular root with `g_i=0`,
the neighbor resultant points along its own phase, so
`partial a_i/partial theta_i=|sum_j exp(i*theta_j)|>0`; hence nonzero `g_i`
is locally dense too. Continuity of the original root row and `H_i>0`
extend the equality to all exceptional states. No limiting regularity of
the auxiliary leaves' metrics is required: the limit is taken only in the
original root's admitted row. This proves the claim.

**Meaning and limits.** The extra leaves are mathematical comparison states,
not events added to a live network or a physical mechanism that creates
support. The proof uses zero capacities exactly, so a model defined only
at strictly positive capacities needs an independently justified continuous
zero-capacity extension. A single K3, a fixed degree class, a fixed graph
size or one common capacity value does not provide the graph/state family
used in the proof. Allowing neighbor degree or external-incidence metadata
falls outside this theorem; no impossibility result for that larger class
is asserted. Dropping zero-capacity phase freezing at least admits an
additional common phase velocity, invisible to total phase storage.

The reference law already satisfies the information contract: `q_i` and
`H_i` use only incident primitive form/phase and `F_i` uses own capacity.
The theorem derives nodewise cancellation and capacity separability within
this class; it does not fix `chi`, pressure weights, storage choice, capacity
evolution, support birth or a physical interpretation. The section 12
countermodel retains its valid balance but consumes excluded neighbor
gradients on general supports. Its one-hop K3 example therefore does not
contradict this uniform result.

No exact closure for an unprepared coarse partition has been assumed. Hidden
state elimination can still produce effective nonlocality and memory under
the [composition and memory contracts](RELATIONAL_PATTERN_MEMORY.md).
Those reduced descriptions are not primitive rows in this theorem's class.
The earlier [primitive phase-lift obstruction](PRIMITIVE_PHASE_CLOSURE.md)
concerns a different prescribed acceleration and is not overridden.

The [exact locality controls](../../tests/physics/test_relational_locality_selection.py)
check the completion, exceptional-domain boundaries and the old countermodel;
the [engine controls](../../tests/test_relational_exchange_execution.py)
check actual remote-extension behavior and whole-graph rejection. These finite
controls supplement the proof rather than replace its graph quantifiers.
Mathematical locality concerns the row: graph-wide validation and storage
aggregation remain necessary in the public executor, and backend-dependent
floating arithmetic is not a promise of bitwise equality across all graphs.

### 11.2 Without phase freezing, only a common clock remains

<a id="primitive-locality-phase-clock"></a>

Remove **only** the zero-capacity phase-row condition from section 11.1.
Keep the same autonomous model, uniform induced-neighborhood information,
all finite connected simple unit supports, arbitrary independent form/capacity
states, full regular phase domain, phase continuity and global exchange
identity. Model constants remain fixed across graphs and states; no global
state-dependent rate is an admitted local input. Then

\[
\boxed{F=F_{\rm rel}+\gamma\mathbf1,
\qquad F_{\rm rel}=\frac w\beta H^{-1}NBx,}
\]

where `gamma` is a single state- and support-independent angular rate for
that fixed model. It may depend on its declared constitutive constants.
It is not the independent form forcing `Gamma`. In particular, dropping
individual phase freezing cannot create a different **relative** phase law
within this class. Common-phase covariance is a consequence here, not a
further premise inserted into the proof.

**1. Reduce to a degree-one row.** Let `Z=F-F_rel`. The reference already
satisfies the global identity, so `a^T Z=0`. Write `h(u,v)` for the correction
at a degree-one node with primitive state `u` and its neighbor's state `v`.
Their induced neighborhood contains just the edge; the neighbor's external
degree is unavailable. On P2, with a nonzero nonantipodal phase gap `delta`,
the balance gives `sin(delta)[h(u,v)-h(v,u)]=0`. Thus `h` is symmetric;
phase continuity includes the zero-gap case. Antipodal P2 is not admitted.

**2. Use phase-balanced stars.** Fix a center phase `theta` and any finite
list of nonantipodal gaps `delta_1,...,delta_m` satisfying
`sum_r sin(delta_r)=0`. Add one leaf at each gap and `m+1` further leaves
aligned with the center. These are distinct vertices, even if some phases
coincide. All primitive forms and capacities are arbitrary. The center's
relative resultant is real and obeys

\[
\sum_{r=1}^m\cos\delta_r+m+1\ge1.
\]

Every leaf is regular, and the center has `a=g=0`. Its unknown correction
therefore performs no phase work. Aligned leaves also have zero gradient,
without requiring their velocities to vanish. The global balance and pair
symmetry imply

\[
\sum_{r=1}^m\sin\delta_r\,h(u,v_r)=0.
\tag{L}
\]

Taking the two gaps `delta,-delta` shows that `h(u,v_+)=h(u,v_-)` for
arbitrarily and independently chosen leaf forms and capacities. Holding
one leaf fixed while changing the other removes dependence on the latter's
form and capacity. Pair symmetry removes those attributes of the center
as well. Continuity extends this statement to zero gap. Denote the remaining
phase function by `h_theta(delta)=h(theta,theta+delta)`; it is even in the gap.
No differentiability or continuity in form/capacity was used.

**3. The balanced-star equation fixes the phase function.** Define

\[
f_\theta(s)=s\,h_\theta(\arcsin s),\qquad -1\le s\le1.
\]

The balanced pair `delta,-arcsin(sin(delta))` in (L) gives
`sin(delta)*h_theta(delta)=f_theta(sin(delta))`, including obtuse gaps;
aligned leaves keep these comparison stars regular. The balanced triple
with gaps `arcsin(s),arcsin(t),-arcsin(s+t)` then gives

\[
f_\theta(s+t)=f_\theta(s)+f_\theta(t)
\quad\text{whenever }s,t,s+t\in[-1,1].
\]

Phase continuity makes `f_theta` continuous. Repeated addition gives
`f_theta(p/q)=(p/q)*f_theta(1)` for rational points in the interval;
continuity extends this to every real point. Hence
`f_theta(s)=c_theta*s` and `h_theta(delta)=c_theta` for nonzero sine,
with zero gap covered by continuity. Pair symmetry gives
`c_theta=c_(theta+delta)` for every nonantipodal pair. An intermediate phase
connects antipodal pairs, so all `c_theta` equal one constant `gamma`.
Every degree-one correction therefore has this value, at any capacity.

**4. Transfer to every root.** Put `Y=Z-gamma*1`. Since `sum_i a_i=0`,
`a^T Y=0`, and every degree-one row of `Y` vanishes. Reuse section 11.1's
induced-neighborhood completion: all original neighbors have zero phase
gradient, and the new private leaves have zero `Y` regardless of their
capacities. The leaves need not have zero capacity in this proof. Thus
`a_i*Y_i=0` isolates the root. The same locality and continuity argument
covers zero root gradients and antipodal internal edges of an originally
regular graph. Consequently `Y=0`, proving the displayed classification.

**Reference choice and the role of capacity.** The theorem leaves an absolute
common clock, not a second relative interaction mechanism. Any one of the
following additional choices fixes `gamma=0`:

- The individual zero-capacity phase-row condition of section 11.1.
- The weaker all-at-rest condition `F(x,theta;0)=0` for all admitted `x,theta`.
- Same-model capacity homogeneity `F(x,theta;lambda*nu)=lambda*F(x,theta;nu)`
  for all positive `lambda`, with coefficients held fixed: the reference
  scales by `lambda`, so the remaining equality requires `gamma=lambda*gamma`.

Each is an explicit condition. Changing clock units alone does not impose
the third one: an independently supplied rate `gamma` could transform with
those units. A model admitting explicit shared time dependence would instead
leave a common function `gamma(t)` by the same pointwise argument. It is
outside the autonomous class stated here; no primordial clock is derived.

**Exact relative equivalence.** In the unforced held-support model, for any
reference solution and the same initial state at `t_0`,

\[
x_{\rm alt}(t)=x_{\rm rel}(t),\qquad
\theta_{\rm alt}(t)=\theta_{\rm rel}(t)+\gamma(t-t_0)\mathbf1
\]

is a solution of the alternative law throughout the same regular interval.
All edge gaps, native phase pressure, metric, storage and form trajectories
coincide. In particular `J_g*1=0` makes the phase-source derivative identical.
This is an exact transformation, not a small-step approximation or a claim
that arbitrary absolute-phase observations are redundant. An external phase
reference, forcing or event law must be transformed and admitted separately.

The [exact controls](../../tests/physics/test_relational_locality_selection.py)
exercise balanced stars, the common-clock null direction, root isolation and
the homogeneity distinction. The existing executor keeps `gamma=0` and its
current phase chambers; this theorem introduces no runtime mode or expanded
admission. Native pressure, joint storage and uniform primitive locality
remain constitutive premises. The coefficients, their physical justification
and primitive support origin are not selected by this result.

### 11.3 Joint edge storage is selected by the prescribed loss and locality

<a id="joint-storage-locality-classification"></a>

The preceding phase-cost classification assumed separate form and phase
storage. The following result tests that split rather than imposing it.
Retain the native form row, with fixed `e>0,w>0`, held independently variable
nonnegative capacities, the same structural clock, and every finite connected
simple unit graph on its full regular phase domain. There are no inputs or
events. Let one twice continuously differentiable cost satisfy

\[
\mathcal S(x,\theta)=\sum_{\{i,j\}\in E}
 W(x_i-x_j,\theta_i-\theta_j),\qquad
W(r,\delta+2\pi)=W(r,\delta),
\]

\[
W\ge0,\qquad W(0,0)=0,\qquad
W(-r,-\delta)=W(r,\delta).
\]

The last condition expresses undirected edge reversal. Neither separate
evenness nor either axis cost is assumed. Suppose there exists a continuous
phase row `F` with section 11.1's uniform induced-neighborhood information
contract, same-model capacity homogeneity

\[
F(x,\theta;\lambda\nu)=\lambda F(x,\theta;\nu),\qquad\lambda>0,
\]

and the specified exact balance

\[
\frac{d\mathcal S}{dt}=-e\sum_i\frac{\nu_i q_i^2}{d_i},
\qquad q=Bx,\qquad
\dot x_i=\nu_i\left(-e\frac{q_i}{d_i}+wg_i\right).
\]

Homogeneity holds with the cost and all constitutive coefficients fixed.
No individual zero-capacity phase condition, gradient-flow representation,
native-aligned storage metric or previously selected phase row is imposed.
Then one constant `beta>0` necessarily gives

\[
\boxed{W(r,\delta)=\frac12r^2+\beta(1-\cos\delta),\qquad
F_i=\frac w\beta\frac{\nu_i q_i}{H_i}.}
\]

Thus the stated class selects the separated storage and existing phase
completion together. The prescribed loss, native pressure, locality, capacity
homogeneity and edge-additive cost remain premises of this conditional result.

**1. A necessary condition at zero phase-storage gradient.** Write
`f=partial_r W` and `h=partial_delta W`. Simultaneous reversal makes both
functions odd under `(r,delta)->(-r,-delta)`; in particular `f(0,0)=h(0,0)=0`.
At a root with incident differences `(r_j,delta_j)`, define

\[
Q=\sum_jr_j,\quad P=\sum_j f(r_j,\delta_j),\quad
J=\sum_j h(r_j,\delta_j).
\]

Initially take each gap in `(-pi,pi)` and make the root regular by adding
aligned zero-form leaves if needed. Give only the root positive capacity.
For each neighbor `j`, add three private leaves: one with
`x_leaf=2*x_j-x_root` and `theta_leaf=2*theta_j-theta_root`, and two with
the neighbor's own form and phase. All these leaves have zero capacity.
The old incidence and its reflected counterpart cancel `h` at `j`; the
aligned incidences contribute zero. Every original neighbor therefore has
zero phase-storage gradient. Its relative phase resultant is
`2*(1+cos(delta_j))>0`, and all private leaves have nonantipodal degree-one
phase data. The completed finite tree is regular everywhere.

Each private leaf sees only zero capacities in its admitted neighborhood.
Scaling the sole active root capacity leaves those local data unchanged,
while capacity homogeneity would scale its phase velocity. Consequently its
phase row is zero. This uses locality and homogeneity, not the unassumed rule
that any node with zero own capacity must freeze. All nonroot form velocities
also vanish. Global balance therefore reduces, whenever `J=0`, to

\[
\boxed{P\left(e\frac Qd-wg\right)=e\frac{Q^2}{d}.}
\tag{JS1}
\]

The phase row at the root cancels from this identity. Thus it is necessary
for every candidate completion, regardless of its dependence on neighboring
capacities. The auxiliary vertices are comparison states, not live support
events or a proposed mechanism of formation.

**2. Aligned padding resolves the native resultant.** For a finite collection
with `J=0`, put `X=sum_j cos(delta_j)`, `Y=sum_j sin(delta_j)` and let `d` be
its size. Append `m` additional incidences with `(r,delta)=(0,0)`. They change
neither `P,Q,J,Y` nor any cost derivative, but the degree becomes `d+m` and
the real resultant becomes `X+m`. For all sufficiently large integers `m`,
the root is regular with

\[
g=-\frac1\pi\arctan\frac{Y}{X+m}.
\]

Constructing the preceding completion for each padded star gives, with
`k=w/pi>0`,

\[
P\left[eQ+k(d+m)\arctan\frac{Y}{X+m}\right]=eQ^2.
\tag{JS2}
\]

If `Q!=0`, positivity of `e` forces `P!=0`, so
`(d+m)*arctan(Y/(X+m))` must be independent of `m`. Yet

\[
(d+m)\arctan\frac{Y}{X+m}
 =Y+\frac{(d-X)Y}{m}+O(m^{-2}).
\]

Here `d-X=sum_j(1-cos(delta_j))>=0`, and it is strictly positive if `Y!=0`.
A constant sequence has its limiting value `Y`, so its displayed first
correction must vanish. It follows that `Y=0`, and (JS2) then gives `P=Q`.
Only integer padding counts are used; no weighted-graph limit or uniform
bound on the auxiliary phase velocities is assumed.

**3. Derive an aligned form anchor.** Fix any `r!=0`. The periodic smooth
function `delta->W(r,delta)` has a stationary point away from the single
excluded antipodal phase. Indeed, a nonconstant periodic function attains
distinct minimum and maximum values, so at least one extremum is
nonantipodal; a constant function has every point stationary. Apply step 2
to that singleton incidence. Since `Q=r!=0`, its sine must vanish, so its
nonantipodal phase is zero. Moreover `P=Q` gives

\[
h(r,0)=0,\qquad f(r,0)=r.
\]

Continuity includes `r=0`; integrating with `W(0,0)=0` proves
`W(r,0)=r^2/2`. These axis properties have been derived, not assumed.

Now consider any finite collection with `J=0`, including one with `Q=0`.
Append a single aligned incidence `(a,0)`, choosing `a` so that `Q+a!=0`.
Its contributions are `h=0`, `f=a` and zero sine. Step 2 applies and gives
`Y=0` and `P+a=Q+a`. Consequently every such collection satisfies

\[
\sum_jh(r_j,\delta_j)=0
\quad\Longrightarrow\quad
\sum_j\sin\delta_j=0,\qquad
\sum_jf(r_j,\delta_j)=\sum_jr_j.
\tag{JS3}
\]

**4. Classify the phase derivative without assuming separability.** The image
`I` of the continuous function `h` on the connected domain
`R times (-pi,pi)` is an interval, symmetric about zero by simultaneous
reversal. It is not the singleton `{0}`: that would contradict (JS3) for
a single gap with nonzero sine. If two states have the same value of `h`,
apply (JS3) to the first and the reversal of the second. Their sines agree.
There is therefore a well-defined bounded function

\[
T:I\longrightarrow[-1,1],\qquad T(h(r,\delta))=\sin\delta.
\]

Balanced triples in (JS3) give `T(a+b)=T(a)+T(b)` whenever
`a,b,a+b` belong to `I`. This bounded additive function is linear. To see
the needed regularity directly, choose `[-eta,eta]` inside `I`. For any
positive integer `n` and sufficiently small `t`, repeated addition gives
`n*T(t)=T(n*t)`, hence `|T(t)|<=1/n`. Thus `T` is continuous at zero;
rational addition and subdivision extend linearity across `I`.
Its slope cannot be zero, since nonzero sine values occur. Accordingly,
for one nonzero constant `beta`,

\[
h(r,\delta)=\beta\sin\delta.
\]

Integrating in phase and using the derived form axis gives
`W(r,delta)=r^2/2+beta*(1-cos(delta))`. Continuity and periodicity include
antipodal endpoints. Nonnegativity at `r=0` forces `beta>=0`; the nonzero
slope excludes `beta=0`, so `beta>0`.

**5. Select the phase completion.** For this now-derived storage, section
[11.2](#primitive-locality-phase-clock) applies to the candidate phase law:
global balance and the uniform information contract leave only
`F=F_rel+gamma*1`. The reference is homogeneous in capacity. Keeping all
coefficients fixed, homogeneity of the candidate then requires
`gamma=lambda*gamma` for every positive `lambda`, so `gamma=0`.
Conversely, the existing reference storage and phase row satisfy all the
stated requirements. No new default law or runtime parameter is introduced.

**Scope and remaining premises.** The prescribed exact loss is essential:
its positive quadratic form term supplies the nonzero-form restriction in
(JS2), and positive phase coupling supplies the padding constraint. Neither
`e=0` nor `w=0` is covered. The argument uses arbitrary support sizes and
degrees, independent capacities, and the full regular phase domain; a fixed
P2, a fixed degree family, or acute-only observations do not supply these
quantifiers. In particular the periodic extremum step cannot silently be
restricted to acute gaps. Nonadditive storage, different losses, additional
state variables, inputs and events require separate admission.

The arbitrary comparison leaves do not derive support, capacities or a
physical clock. The positive coefficient `beta` remains free and is the
existing storage-scale parameter. This theorem does not establish physical
energy or the uniquely correct pressure of nature. Its normalization also
retains a boundary: a constant per edge is excluded by `W(0,0)=0`; without
that convention it is invisible to fixed-support evolution but affects
support-event storage jumps. The older auxiliary-potential and cotangent
models retain their different laws and balance premises.

**Why smaller checks do not select the cost.** At consensus let the edge
Hessian have entries `(a,b;b,c)`, with `c>0`, and put `k=w/pi`.
For `t<e/k`, set `sigma=e/(e-k*t)`, `b=c*t` and `a=sigma+c*t^2`.
The positive quadratic tangent storage then has the mixed form
`sigma*E_D(x)+c*E_D(theta+t*x)`. With `p=B*theta`, the local tangent rows

\[
\dot x=-ND^{-1}(eq+kp),\qquad
\dot\theta=ND^{-1}\big[(k\sigma/c+et)q+ktp\big]
\]

satisfy the prescribed quadratic loss exactly. This uses a local real phase
lift, not a globally periodic storage. The periodic candidate
`W=sigma*r^2/2+c*(1-cos(delta+t*r))` has that Hessian but need not satisfy
the nonlinear balance: at `e=w=1/2`, `c=1`, `t=-pi`, `r=-2/3`, `delta=pi/3`
and total two-node capacity `3`, its phase derivative vanishes while
`Sdot=-1/6` and required `-L=-2/3`. No finite phase completion can supply
the missing work at that state.

Another candidate `W=r^2/2+(beta+alpha*r^2)*(1-cos(delta))`, with
`beta>0, alpha>0`, passes the regular two-node phase-critical check.
On a six-cycle with phase gaps `pi/3` and alternating form `+a,-a`, however,
both native phase pressure and the entire phase-storage gradient vanish.
Its form-storage gradient is `(1+alpha)*q`, so `Sdot=-(1+alpha)*L`,
independently of the phase row. This violates the specified equality even
though storage decreases at that state. Exact loss and passivity are distinct
conditions; this example alone does not give a complete passive alternative.

The [joint-storage controls](../../tests/physics/test_relational_joint_storage.py)
check the reflected completion, aligned-padding restriction and named mixed
counterexamples with independent exact arithmetic. A compatible mixed
quadratic tangent law shows why a consensus expansion alone cannot establish
this theorem; its natural periodic continuation can fail even on an acute
two-node state. A form-dependent phase-cost amplitude can pass a two-node
critical-point check and fail on a phase-balanced cycle. These are controls
of the proof's obligations, not a substitute for its all-state argument.
The existing [phase-law locality controls](../../tests/physics/test_relational_locality_selection.py)
and [engine execution tests](../../tests/test_relational_exchange_execution.py)
retain the continuous-law versus represented-evaluation boundary.

### 11.4 Passive phase loss is a distinct admissible completion

<a id="relational-passive-loss-completion"></a>

The exact loss in section 11.3 is an independent premise. To examine it,
**retain the already specified comparison storage**
`E=E_D+beta*V_phi`, rather than asserting that its classification still holds
after weakening that premise. On the same unforced held-support regular
domain, with `e,w,beta>0` and held `N>=0`, declare the complete family

\[
\dot x=N(-eD^{-1}Bx+wg),\qquad
F_\rho=\dot\theta=\frac w\beta H^{-1}NBx+\rho Ng,
\qquad \rho\ge0.
\]

Here `rho` is a supplied constant coefficient, not a diagnostic, an inferred
reservoir or a consequence of the nodal product. The case `rho=0` is the
existing reference; other values are comparison laws, not production modes.
The added row uses only primitive neighbors and own capacity. It is linear
in capacity and preserves common form shifts, common phase rotations,
relabeling and simultaneous `(x,theta)->(-x,-theta)` reversal. It also
vanishes at a node with zero capacity.

The older [passive-transfer comparison](../TNFR_VARIATIONAL_PRINCIPLE.md#1318-source-sensitivity-joint-work-and-the-passive-transfer-limit)
already used an additional multiple of `g`. That result employs centered
Euclidean form storage and unit capacity on a supplied prism. Its cancellation
method is reused here; its storage, rates and numerical bounds are not
transferred to this Dirichlet-storage family.

**Exact work and unchanged form equation.** Since `grad(V_phi)=-H*g`,

\[
\boxed{\dot E=-L-R_\rho,\qquad
L=e\sum_i\frac{\nu_iq_i^2}{d_i},\qquad
R_\rho=\beta\rho\sum_i\nu_iH_i g_i^2\ge0.}
\]

Thus passivity, locality, capacity homogeneity and the listed symmetries do
not select the exact reference loss. The extra phase motion dissipates
storage directly whenever `rho>0` and `Ng!=0`; the reference transfers
phase/form work before dissipation through form contrast. This accounting
does not identify either loss with a physical bath or make it available to
fund an event.

The complete form-mean formula is still

\[
\dot{\bar x}=\frac1n\mathbf1^TN(-eD^{-1}q+wg).
\]

It agrees at the same prepared state, but its later values generally differ
because the phase trajectory changes. The mean angular velocity acquires
`rho*mean(Ng)`; it is not generally a common rotation. Likewise the pressure
derivative remains `pdot=-eD^-1B*xdot+w*J_g*F_rho`, with the chosen phase row
included. No additional term has been silently inserted into `xdot`.

**Units and replicas.** The new coefficient obeys
`[rho*nu_f]=1/[time]` for dimensionless angular phase and native `g`.
In section 4's convention, capacity has inverse-clock units and `rho` is
dimensionless. Under `x_new=a*x`, `t_new=b*t`, `N_new=N/b`, it stays unchanged.
If capacity is held numerically fixed instead, `rho_new=rho/b`. When the
pressure weights are renormalized with `k=e+a*w` and `N_new=k*N/b`, the
equivalent coefficient is `rho_new=rho/k`. Failing to transform it changes
the comparison law. The ratio `rho*sqrt(beta)/w` is an additional invariant
under these form/clock conventions; their covariance does not choose it.
Equal complete replicas retain `g` and each prepared capacity, so the added
row descends under the same supplied synchronized replication as the reference.
This is a compatibility check, not a derivation of `rho`.

**Equilibria and inactive nodes.** For every held nonnegative capacity vector,
both laws have precisely the equilibrium conditions

\[
Nq=0,\qquad Ng=0.
\]

At an equilibrium the nonnegative losses must vanish. Since `e>0`, this gives
`Nq=0`; the form equation then gives `Ng=0`. Conversely these two conditions
annul both complete rows. With all capacities positive, connectedness gives
uniform form and a regular critical point of phase storage, which can include
winding patterns. Zero capacities retain inactive frozen coordinates and do
not permit replacing these conditions by global consensus. Equal equilibria
do not imply equal trajectories, attraction basins or rates.

**One prepared analytic discriminator.** On P2 set `e=w=1/2`, `beta=1`,
capacities `(1,2)`, uniform form, and the phase gap
`theta_0-theta_1=pi/3`. Both nodes lie strictly inside the acute domain. Then

\[
g=(-1/3,1/3),\qquad H_0=H_1=3\sqrt3/2,\qquad
\dot x=(-1/6,1/3).
\]

The reference initially has `F_rel=0` and `Edot=0`. Relative to it, the declared
alternative predicts

\[
\Delta F=(-\rho/3,2\rho/3),\qquad
\Delta(\dot\theta_0-\dot\theta_1)=-\rho,\qquad
R_\rho=\rho\sqrt3/2,
\]

\[
\Delta\ddot x=wN J_g\Delta F
 =\left(\frac{\rho}{2\pi},-\frac{\rho}{\pi}\right),
\qquad
J_g=\frac1\pi\begin{pmatrix}-1&1\\1&-1\end{pmatrix}.
\]

The initial form response is identical; the relative phase velocity, storage
loss and form acceleration distinguish the laws. These are exact conditional
initial-rate predictions, not a sampled response, a fitted coefficient or a
physical validation. No previously frozen acquisition is rerun or reinterpreted.

**Local recovery survives with different damping.** Take a strictly acute
phase equilibrium, uniform form and positive held capacities. Let
`K=Hess(V_phi)` and evaluate `H` at that equilibrium. For form/phase
perturbations `(u,v)`, the linearization is

\[
\begin{aligned}
\dot u&=-eND^{-1}Bu-wNH^{-1}Kv,\\
\dot v&=(w/\beta)NH^{-1}Bu-\rho NH^{-1}Kv.
\end{aligned}
\]

On the quotient by the two common offsets,
`Q=(u^TBu+beta*v^TKv)/2` is positive definite and

\[
\dot Q=-e(Bu)^TND^{-1}(Bu)
       -\beta\rho(Kv)^TNH^{-1}(Kv).
\]

For `rho>0` this is strictly negative on every nonzero quotient state, so the
linearization is Hurwitz and the smooth nonlinear law has local exponential
recovery. The `rho=0` case uses the reference's separately proved exchange
observability argument. This establishes existence of a local recovery
neighborhood; it does not preserve a previous numerical rate or capture tube.

At phase consensus with common capacity `nu`, a nonzero random-walk
Laplacian mode `lambda` has characteristic polynomial

\[
s^2+\nu\lambda(e+\rho/\pi)s
+\nu^2\lambda^2\left(\frac{e\rho}{\pi}
  +\frac{w^2}{\beta\pi^2}\right)=0.
\]

The reference polynomial is recovered at zero `rho`; extra dissipation changes
both coefficients. For the explicit comparison `e=w=1/2`, `beta=1`, its
discriminant divided by `(nu*lambda)^2` is `1/4-1/pi^2>0` at `rho=0`, but
`1/4-1/pi<0` at `rho=1`. Thus the same equilibrium and local-recovery verdict
coexist with real decaying modes in one law and damped oscillatory modes in
the other. This is a tangent response distinction, not a perpetual nonlinear
pulse or a change in the supplied graph. On an acute uniform cycle twist,
replace the last `w^2/(beta*pi^2)` term by `w^2/(beta*pi^2*cos(kappa))`.
No monotonic optimal decay claim, global attraction, zero-capacity recovery
or unchanged validated formation/capture evidence follows from these tangent
formulas.

**A sufficient selection premise under total passivity.** One can recover the
reference without assuming its exact loss directly, but another condition is
needed. Keep the fixed comparison storage and native form row, and suppose
the phase row is exactly homogeneous of degree one in signed form:

\[
F(ax,\theta;\nu)=aF(x,\theta;\nu),\qquad a\in\mathbb R,
\]

with coefficients, phases and capacities held fixed. This is a condition on
different preparations of the same model, not a conversion of form units.
For an arbitrary proposed phase law write the signed exchange residual

\[
\mathcal R(x,\theta;\nu)=wq^TNg-\beta g^THF,
\qquad \dot E=-L+\mathcal R.
\]

Under this form condition, `R(a*x)=a*R(x)` while `L(a*x)=a^2*L(x)`.
Requiring only total passivity for every signed amplitude therefore gives
`a*R-a^2*L<=0`. Both signs and arbitrarily small amplitudes force `R=0`.
The uniform-locality theorem then leaves only a common angular rate, and
same-model capacity homogeneity removes it. The resulting law is `F_rel`.

Exact degree-one form response is an additional constitutive premise, not a
consequence of the nodal product or of mere oddness under form reflection.
The supplied `rho*Ng` term violates it away from zero phase pressure. The
storage remained fixed throughout this comparison; passivity has not replaced
the loss hypothesis in section 11.3's storage-classification theorem. Higher
order form dependence and other passive completions require their own test,
rather than being excluded by calling the reference canonical.

The [shared admission controls](../../tests/physics/test_relational_exchange_admission.py)
reuse the reference phase geometry and verify the added work, inactive rows,
unit/replica conversion, exact P2 jet, consensus spectrum and signed-amplitude
argument. Detached production field/tangent readings retain their represented
defects separately from these ideal identities. No comparison coefficient is
added to the engine or SDK, and no frozen response is regenerated.

### 11.5 Nonlinear signed form response survives passivity and local recovery

<a id="relational-nonlinear-passive-completion"></a>

The degree-one form response in section 11.4 is sufficient for selection,
but the other retained properties do not force it. Keep the **fixed comparison
storage** `E=E_D+beta*V_phi`, native form pressure, `e,w,beta>0`, held `N>=0`,
and the full regular phase domain on supplied connected simple unit support.
Declare one constant `0<eta<2` and the complete comparison law

\[
\dot x=N(-eD^{-1}q+wg),\qquad F_\eta=F_{\rm rel}+Z,
\]

\[
\boxed{Z_i=\frac{\eta e}{\pi}\nu_i h(s_i)g_i^2,\qquad
s_i=\frac{q_i}{d_i\sqrt\beta},\qquad
h(s)=\frac{s^3}{1+s^2}.}
\]

This is an explicit countermodel, not a derived coefficient or an engine
default. The interval for `eta` is a sufficient declared bound; no optimal
range or search over alternatives is claimed. The denominator is positive
for every finite signed form, so the row is smooth wherever the native
phase geometry is regular.

**Information, parity and units.** The correction uses own capacity and
primitive incident form/phase only. It is capacity-linear, vanishes at zero
capacity, and preserves relabeling, common form shifts and common phase
rotations. Since `h` is odd and `g^2` is even under phase reflection, both
the complete phase row and its correction obey signed-form oddness and
simultaneous `(x,theta)->(-x,-theta)` reversal. At uniform form the correction
and reference phase row vanish. However `h(a*s)` is not generally `a*h(s)`:
oddness and degree-one form response remain distinct requirements.

The coordinate `s_i` is dimensionless. Under section 4's transformations
`x_new=a*x`, `beta_new=a^2*beta`, `N_new=N/b`, it stays unchanged and
`Z_new=Z/b`, with `eta` unchanged. If the pressure weights are renormalized,
`e_new=e/k` and `N_new=k*N/b` compensate in the prefactor as well. Thus no
form or clock conversion is mistaken for a change of the nonlinear response.
On the supplied synchronized complete-replica construction,
`q_fine=m*q`, `d_fine=m*d` and `g_fine=g`; consequently `s` and the phase row
descend unchanged. This prepared compatibility is not closure of an arbitrary
coarse observation or a mechanism of block formation.

**A global pointwise work bound on the regular domain.** Write the signed
extra exchange work as

\[
\mathcal R=-\beta\sum_i g_iH_iZ_i
 =-\frac{\eta e\beta}{\pi}\sum_i\nu_iH_i h(s_i)g_i^3,
\qquad \dot E=-L+\mathcal R.
\]

The native geometry and elementary scalar inequality give

\[
0<H_i\le\pi d_i,\qquad |g_i|<1,\qquad
|h(s)|\le\frac{s^2}{2}.
\]

The last inequality follows from `2*|s|<=1+s^2`. Since
`L=e*beta*sum_i nu_i*d_i*s_i^2`, these independent bounds imply

\[
\boxed{|\mathcal R|\le\frac\eta2L,\qquad
\dot E\le-\left(1-\frac\eta2\right)L\le0.}
\]

The residual can have either sign. It must not be called a nonnegative added
loss: the alternative can dissipate less storage than the reference at one
preparation and more at another, while retaining total passivity. The bound
holds whenever the ideal state is regular; by itself it is not a claim of
global domain invariance or a discrete-step storage guarantee. No loss is
made available to fund a support event without its own reservoir law.

**Equilibria and the retained local mechanism.** For every held nonnegative
capacity vector, the equilibrium conditions remain exactly

\[
Nq=0,\qquad Ng=0.
\]

Indeed, zero velocity implies zero storage derivative, and the strict factor
`1-eta/2>0` forces `L=0`, hence `Nq=0`. The form row then forces `Ng=0`.
Conversely these conditions annul the reference and added rows. Inactive
coordinates retain their frozen-value obstructions; this statement does not
infer global consensus or positive-capacity recovery when some capacities
are zero.

At uniform form and a regular phase equilibrium, `q=0` and `g=0`. For small
joint form/phase perturbations `(u,v)`,

\[
Z=O(\|u\|^3\|v\|^2)=O(\|(u,v)\|^5).
\]

Consequently the reference Jacobian is unchanged. This is a statement about
the vector field's expansion in **state amplitude**; it does not assert that
the first four temporal derivatives agree at a finite perturbed preparation.
At a strictly acute phase equilibrium and positive held capacity, the existing
Hurwitz result therefore gives local exponential recovery for the comparison
as well. Neither the equilibria, signed symmetries nor that linear response
select the complete nonlinear phase law.

**An exact finite-amplitude discriminator.** Fix P2 with
`eta=1`, `e=w=1/2`, `beta=1`, capacities `(1,2)`, forms `(1/2,-1/2)` and
phases `(pi/3,0)`. The state is strictly acute. At this same preparation,

\[
q=(1,-1),\quad g=(-1/3,1/3),\quad H_0=H_1=3\sqrt3/2,
\quad \dot x=(-2/3,4/3),
\]

\[
Z=\left(\frac1{36\pi},-\frac1{18\pi}\right),\qquad
\mathcal R=\frac{\sqrt3}{24\pi}>0,\qquad L=\frac32.
\]

Relative to the reference, the predictions are

\[
\Delta(\dot\theta_0-\dot\theta_1)=\frac1{12\pi},\qquad
\Delta\ddot x=wN J_gZ
 =\left(-\frac1{24\pi^2},\frac1{12\pi^2}\right).
\]

Reflecting the form while holding phase fixed reverses `Z` and `R`, while
leaving the storage and `L` unchanged. Both preparations remain passive.
These are analytic initial-rate distinctions, not sampled trajectories or
physical evidence. The full form-mean equation remains
`mean(xdot)=mean(N*(-eD^-1q+w*g))`: the two laws agree on its initial value,
but their later phase responses change it. In this preparation the difference
of mean form acceleration is `1/(48*pi^2)`. The mean of `Z` is
`-1/(72*pi)` and its relative component is nonzero, so the correction is not
a removable common angular clock.

**The same local geometric basin, with a new proof.** A stronger local
statement can be recovered without transferring a stored trajectory. Use
the [local recovery owner's](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery) strictly acute
critical phase `theta_*`, its quotient coordinates `z=(u,v)`, and positive
held capacities. Choose

\[
m=\min_{\{i,j\}\in E}(\pi/2-|\delta_{*,ij}|),\qquad
0<r<m/\sqrt2,
\]

\[
c_r=\min_{\{i,j\}\in E}\cos(|\delta_{*,ij}|+\sqrt2r)>0,
\qquad k_r=\frac{\lambda_2(B)}2\min(1,\beta c_r).
\]

The unchanged storage and phase Hessian give, inside `||z||<=r`,

\[
E_{\rm rel}=E_D+\beta[V_\phi(\theta)-V_\phi(\theta_*)]
\ge k_r\|z\|^2.
\]

For `||z(0)||<r` and `E_rel(0)<k_r*r^2`, the newly proved inequality
`Edot_rel<=-(1-eta/2)*L<=0` prevents a first exit from that same ball.
The quotient stays in a compact interior sublevel; the phase metric is bounded
away from zero and the smooth law continues. Thus the **same analytically
specified continuous energy sublevel** is a sufficient domain for this
comparison, by its own argument.

Within it, the largest invariant subset of `L=0` has `q=0`. There `Z=0` and
`F_rel=0`. Keeping `q=0` requires `Ng` to be common across the connected graph;
`sum_i H_i*g_i=0` and positive `H,N` force that common value to vanish. Strict
convexity in the acute quotient neighborhood then selects the reference phase
shape. LaSalle's argument gives convergence, and the unchanged Hurwitz
Jacobian gives exponential convergence near the limit. Common-offset
velocities are smooth functions of the quotient state and vanish there, so
their time integrals converge and the offsets approach finite limits.

This result does not retain a previous quantitative decay constant, validated
trajectory tube, arrival time or proof that a remote formation preparation
enters the basin. Those concern the changed vector field and require their
own evidence. Zero capacities, nonacute targets, events and changing support
are outside this local capture statement.

**Minimal premise ledger and the next scope.** The
[joint-cost theorem](#joint-storage-locality-classification) selects storage
and phase response using its prescribed exact loss and edge-cost class.
With storage fixed, [signed degree-one form response and total passivity](#relational-passive-loss-completion)
are another sufficient route to exact exchange and the reference row. The
present nonlinear alternative meets the listed parity, locality, capacity,
equilibrium and local-recovery requirements while violating that degree-one
condition. Those properties therefore do not derive it. The fixed storage
used for the comparison is not circularly promoted to a theorem about all
passive models, and the supplied `eta` has not become a selected mechanism.

The [shared exact exchange controls](../../tests/physics/test_relational_exchange_admission.py)
check the bound, scaling, equilibrium conditions and finite-amplitude witness.
This closes the declared necessity check. The next formation/interaction
question is whether the full-state two-C5/two-bridge protected-sector capture
argument extends to this admitted comparison. That is a separate gate: the
local-ball proof above neither establishes its larger sector barrier and
capture obligations nor repeats the original formation computation.

## 12. A capacity-mediated alternative and a discriminating intervention

<a id="capacity-mediated-exchange-counterexample"></a>

The missing capacity premise is consequential. Reuse the earlier
[power-null construction's method](../TNFR_VARIATIONAL_PRINCIPLE.md#1319-structural-closure-tests-exchange-jacobi-and-the-remaining-potential),
but retain this owner's Dirichlet storage and exact native form row. Write
`a=grad(V_phi)`, `ell=D^-1 Bx` and define the symmetric pair mobility

\[
\mu_{ij}=\begin{cases}
\nu_i\nu_j/(\nu_i+\nu_j),&\nu_i+\nu_j>0,\\
0,&\nu_i=\nu_j=0.
\end{cases}
\]

For adjacent nodes set

\[
S_{ij}=\frac w\beta\frac{\mu_{ij}}{d_i d_j}
       (\ell_i+\ell_j)\sin(\theta_j-\theta_i),\qquad
S_{ii}=0,\qquad Z=Sa.
\]

Nonadjacent entries vanish. The factors other than the sine are symmetric,
so `S^T=-S` and `a^T Z=0` identically. Both `F_rel` and `F_rel+Z` therefore
have the same complete storage loss and unmodified form pressure. In
general they redistribute nonzero nodal phase work while its sum cancels.
This is an explicit **countermodel**, not a selected correction or a new
production phase rule.

This alternative uses only existing form, phase, capacity and support:

- It is equivariant under node relabeling, common form shifts, common phase
  rotations and simultaneous `(x,theta)->(-x,-theta)` reversal.
- `nu_i=0` makes row `S_i` zero. The pair mobility is homogeneous of degree
  one in all capacities, but not additive in independent capacity vectors.
  It is continuous at joint zero and smooth on positive strata; no capacity
  derivative at the joint-zero corner is claimed. Held capacities give a
  smooth field in evolving form and phase throughout the regular chart.
- The section 4 unit/clock conversions preserve this field, including the
  compensating capacity change when channel weights are renormalized.
- Equal complete replicas preserve the correction: `ell` is unchanged,
  `a_fine=m*a`, each pair coefficient becomes `S_ij/m^2`, and each base
  neighbor supplies `m` identical contributions. Disjoint admitted components are
  also independent. Thus these composition checks do not imply capacity
  additivity at a fixed state and support.

**Locality boundary.** On K3 every primitive consumed is within one hop.
On general graphs, reading a neighbor's `ell` and `a` reaches two primitive
neighborhoods. The replica identity does not turn that full fine law into
a globally one-hop implementation. This example disproves uniqueness from
the stated fixed-K3 balance, symmetry and zero-capacity conditions, and
from their general finite-locality counterpart. It does **not** disprove a
stronger theorem demanding one uniform primitive-neighborhood law on all
admitted graphs. [Section 11.1](#primitive-locality-exchange-selection) now
proves that conditional result without assuming closure of unprepared
coarse refinements.

**Exact nonconsensus witness.** On unit K3 take
`x=(1/3,0,-1/3)`, `theta=(0,pi/6,-pi/6)`, `nu=(1,1,1)`. All resultants and
phase displacements lie in the admitted chart. With `k=w/beta`,

\[
Z=\frac{k}{64}
\begin{pmatrix}1+\sqrt3\\-3-\sqrt3\\-3-\sqrt3\end{pmatrix},
\qquad
J_gZ=\frac{k(2+\sqrt3)}{64\pi}
\begin{pmatrix}-2\\1\\1\end{pmatrix}\ne0.
\]

Both laws have identical `x`, pressure and instantaneous `xdot`. Their
phase-source rates differ by `J_gZ`, and their form accelerations by
`w*N*J_gZ`. This is relative evolution, not a removable common rotation.
At phase consensus the correction vanishes, so the frozen section 9
discriminator cannot separate these two members of the relational-storage
family. The new witness addresses the missing premise directly.

Hold that form and phase fixed, keep `nu_0=nu_2=1`, and change only
`nu_1` from `1` to `1/2`. The selected capacity-separable law leaves `F_0`
unchanged. The alternative changes its extra row by

\[
\Delta F_0=\Delta Z_0=-\frac{w(1+\sqrt3)}{192\beta}\ne0.
\]

The same intervention admits a relative-phase observation, independent of a
common phase reference. Put `R=F_0-F_2`. Both unchanged-capacity baseline rows
are invariant under this intervention, whereas the countermodel also has
`Delta Z_2=k*(3+sqrt(3))/192`. Therefore

\[
\Delta R_{\rm separable}=0,\qquad
\Delta R_{\rm mediated}=-\frac{k(2+\sqrt3)}{96},\qquad k=w/\beta>0.
\]

These are ideal differences at the same declared nonconsensus preparation,
not a finite-sample acquisition. Relative phase removes a common rotation;
it does not remove rate units or observation uncertainty. A uniform positive
separation from zero requires a positive lower bound on `k` in that clock,
or an admitted normalized observation cancelling the unknown scale.
Identifying `chi` alone does not supply such a bound. For instance,
`(e,w,beta)=(1/2,1/2,1)` and `(1/3,2/3,4)` both have `chi=1`, but `k=1/2`
and `1/6`, respectively. Other retained coefficient/clock information must
justify any absolute prediction. Both candidates must keep the same admitted
preparation and parameter information through the reserved comparison.

No pressure is reconstructed from a response and no scale is refitted.
This exact constitutive intervention distinguishes local independent
activity from one admissible pair-mediated activity. Independent physical
preparation/observation of such an intervention remains unestablished.

Controls: the
[capacity-identification test](../../tests/physics/test_relational_exchange_admission.py)
solves for unknown nonlinear own-capacity coefficients using the complete
storage chain rule on an irregular P3. The
[selection counterexample tests](../../tests/physics/test_relational_exchange_selection.py)
reuse the shared exact phase geometry and check the full work identity,
zero capacities, nonadditivity, source-response difference, capacity
intervention, transformations and complete replicas. These controls add
no temporal campaign or default engine dynamics.

## 13. Shared engine execution of the conditional reference

<a id="relational-engine-integration"></a>

The admitted capacity-separable family is implemented in
[dynamics/relational.py](../../src/tnfr/dynamics/relational.py). Evaluation and
one simultaneous Euler step share that owner; the SDK delegates to it through
`Network.relational_exchange` and `Network.step_relational`. The required
`RelationalExchangeModel.storage_scale` declares beta rather than deriving or
fitting it. Native pressure resolves the normalized EPI/phase weights. The
implementation admits fixed connected simple unit-conductance support and held
nonnegative capacities. Its default domain requires strict acute represented
edge phases; [section 18](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-positive-resultant-execution) defines the
opt-in positive-resultant chamber. Both reject unsupported forcing and states
before any live commit.

The field computes `H` from the local phasor resultant and its displacement,
using the continuous sinc limit at zero. It never obtains `H` by dividing a
possibly zero pressure by a phase gradient. Native pressure and the separately
materialized form/phase split retain their difference. The phase row and shared
nodal product are evaluated from the same supplied state; no telemetry selects
an operator, partitions the network or supplies a missing capacity law.

Exact-rational work on materialized values retains the measured residual
`r=storage_rate+continuous_loss`. For one admitted step, with `D` its recorded
Euler storage defect, the accounting identity is

\[
E_{next}-E_{old}=-h\,\mathrm{loss}_{old}+h\,r_{old}+D,
\qquad D=E_{next}-E_{old}-h\,\mathrm{storage\_rate}_{old}.
\]

The positive or negative value of `D` is retained; it is not an a priori
error bound or a new physical term. Mathematical continuous dissipation,
represented instantaneous work and numerical endpoint nonincrease remain
separate. In particular a lossless Euler step can increase storage. Keeping
both endpoints and the numerical proposal segment acute does not prove
that an exact future ODE trajectory stays there for arbitrary duration.

The [execution contract](../../docs/contracts/relational/RELATIONAL_EXECUTION.md#conditional-relational-execution)
owns scalar/clock admission, atomicity, caches, aliases and immutable report
semantics; the [SDK guide](../../docs/guides/relational/RELATIONAL_EXECUTION.md#execute-the-conditional-relational-model)
owns invocation. [Example 180](../../examples/08_emergent_geometry/180_relational_exchange.py)
executes four small declared steps, including a zero-capacity node. The
[routine regression owner](../../tests/test_relational_exchange_execution.py)
checks analytic rates, the neighbor-capacity intervention, shared pressure,
domain failures, zero-capacity behavior, atomic commit and numerical defects.
The symbolic conditional theorems remain in their research controls. This
integration does not install the countermodel, replace default runtime
policies or turn successful finite execution into physical validation.

The shared [pattern observer](../../src/tnfr/physics/relational_observations.py)
and `Network.relational_pattern` now retain the regional information used by
the recovery and transmission studies below: complete centered form and
reference-lift phase coordinates, separate squared norms and common offsets.
They reuse winding and full-support regional transport accounting with the
independent phase source, preserving explicit unavailable domains and pressure
defects. A supplied frame is not a demonstrated equilibrium or temporal identity.
The [observation/export contract](../../docs/contracts/relational/RELATIONAL_EXECUTION.md#relational-pattern-observation)
owns admission and exact JSON projection. Routine controls compare selected
retained checkpoints with this owner without altering or rerunning the frozen
campaigns; their historical benchmark wrappers remain immutable evidence.

## Section link directory

These aliases route existing citations to their substantive owner.

- <a id="relational-storage-and-local-phaseform-exchange-admission"></a>[Relational storage and local phase/form exchange admission](#relational-storage-and-local-phaseform-exchange-admission)

- <a id="5-a-prospective-spectral-discriminator"></a>[5. A prospective spectral discriminator](RELATIONAL_RESPONSE_IDENTIFICATION.md#5-a-prospective-spectral-discriminator)

- <a id="coefficient-synergy-audit"></a>[coefficient-synergy-audit](RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-synergy-audit)

- <a id="coefficient-audit-selection-covariance-and-identifiable-response"></a>[Coefficient audit: selection, covariance and identifiable response](RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-audit-selection-covariance-and-identifiable-response)

- <a id="coefficient-memory-identification"></a>[coefficient-memory-identification](RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-memory-identification)

- <a id="the-same-invariant-in-hidden-phase-memory-and-temporal-poles"></a>[The same invariant in hidden-phase memory and temporal poles](RELATIONAL_RESPONSE_IDENTIFICATION.md#the-same-invariant-in-hidden-phase-memory-and-temporal-poles)

- <a id="native-consensus-full-state-tangent"></a>[native-consensus-full-state-tangent](RELATIONAL_RESPONSE_IDENTIFICATION.md#native-consensus-full-state-tangent)

- <a id="full-state-consensus-derivative-and-a-bounded-linear-response"></a>[Full-state consensus derivative and a bounded linear response](RELATIONAL_RESPONSE_IDENTIFICATION.md#full-state-consensus-derivative-and-a-bounded-linear-response)

- <a id="prepared-coefficient-identification"></a>[prepared-coefficient-identification](RELATIONAL_RESPONSE_IDENTIFICATION.md#prepared-coefficient-identification)

- <a id="a-prepared-mode-identifies-chi-from-its-initial-response-jet"></a>[A prepared mode identifies chi from its initial response jet](RELATIONAL_RESPONSE_IDENTIFICATION.md#a-prepared-mode-identifies-chi-from-its-initial-response-jet)

- <a id="coefficient-jet-uncertainty"></a>[coefficient-jet-uncertainty](RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-jet-uncertainty)

- <a id="finite-samples-require-a-declared-error-budget-and-abstention"></a>[Finite samples require a declared error budget and abstention](RELATIONAL_RESPONSE_IDENTIFICATION.md#finite-samples-require-a-declared-error-budget-and-abstention)

- <a id="coefficient-temporal-acquisition"></a>[coefficient-temporal-acquisition](RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-temporal-acquisition)

- <a id="a-bounded-computational-temporal-acquisition-on-p2"></a>[A bounded computational temporal acquisition on P2](RELATIONAL_RESPONSE_IDENTIFICATION.md#a-bounded-computational-temporal-acquisition-on-p2)

- <a id="relational-pulse-scope"></a>[relational-pulse-scope](RELATIONAL_RESPONSE_IDENTIFICATION.md#relational-pulse-scope)

- <a id="pulse-recurrence-and-the-nondissipative-boundary"></a>[Pulse, recurrence and the nondissipative boundary](RELATIONAL_RESPONSE_IDENTIFICATION.md#pulse-recurrence-and-the-nondissipative-boundary)

- <a id="9-exact-finite-amplitude-discriminator-and-prepared-state-control"></a>[9. Exact finite-amplitude discriminator and prepared-state control](RELATIONAL_RESPONSE_IDENTIFICATION.md#9-exact-finite-amplitude-discriminator-and-prepared-state-control)

- <a id="finite-modal-exchange-discriminator"></a>[finite-modal-exchange-discriminator](RELATIONAL_RESPONSE_IDENTIFICATION.md#finite-modal-exchange-discriminator)

- <a id="116-global-closure-obstruction-and-a-normalized-sine-comparison"></a>[11.6 Global closure obstruction and a normalized-sine comparison](SINE_CONSTITUTIVE_INFORMATION.md#116-global-closure-obstruction-and-a-normalized-sine-comparison)

- <a id="global-closure-pressure-comparison"></a>[global-closure-pressure-comparison](SINE_CONSTITUTIVE_INFORMATION.md#global-closure-pressure-comparison)

- <a id="the-incompatibility-is-stronger-than-the-local-completion-theorem"></a>[The incompatibility is stronger than the local-completion theorem](SINE_CONSTITUTIVE_INFORMATION.md#the-incompatibility-is-stronger-than-the-local-completion-theorem)

- <a id="reuse-an-existing-phase-reading-with-an-explicit-changed-premise"></a>[Reuse an existing phase reading with an explicit changed premise](SINE_CONSTITUTIVE_INFORMATION.md#reuse-an-existing-phase-reading-with-an-explicit-changed-premise)

- <a id="complete-balance-continuation-and-inherited-geometry"></a>[Complete balance, continuation and inherited geometry](SINE_CONSTITUTIVE_INFORMATION.md#complete-balance-continuation-and-inherited-geometry)

- <a id="conditional-selection-and-remaining-freedom"></a>[Conditional selection and remaining freedom](SINE_CONSTITUTIVE_INFORMATION.md#conditional-selection-and-remaining-freedom)

- <a id="an-exact-discriminator-before-any-trajectory"></a>[An exact discriminator before any trajectory](SINE_CONSTITUTIVE_INFORMATION.md#an-exact-discriminator-before-any-trajectory)

- <a id="117-common-exchange-geometry-composition-and-a-controlled-common-regime"></a>[11.7 Common exchange geometry, composition and a controlled common regime](SINE_CONSTITUTIVE_INFORMATION.md#117-common-exchange-geometry-composition-and-a-controlled-common-regime)

- <a id="common-exchange-geometry-and-composition"></a>[common-exchange-geometry-and-composition](SINE_CONSTITUTIVE_INFORMATION.md#common-exchange-geometry-and-composition)

- <a id="geometric-attenuation-is-not-generally-a-change-of-clock"></a>[Geometric attenuation is not generally a change of clock](SINE_CONSTITUTIVE_INFORMATION.md#geometric-attenuation-is-not-generally-a-change-of-clock)

- <a id="a-controlled-near-alignment-regime"></a>[A controlled near-alignment regime](SINE_CONSTITUTIVE_INFORMATION.md#a-controlled-near-alignment-regime)

- <a id="composition-exposes-an-independent-premise"></a>[Composition exposes an independent premise](SINE_CONSTITUTIVE_INFORMATION.md#composition-exposes-an-independent-premise)

- <a id="118-closed-form-balance-can-select-primitive-pressure-composition"></a>[11.8 Closed form balance can select primitive pressure composition](SINE_CONSTITUTIVE_INFORMATION.md#118-closed-form-balance-can-select-primitive-pressure-composition)

- <a id="closed-form-balance-and-source-selection"></a>[closed-form-balance-and-source-selection](SINE_CONSTITUTIVE_INFORMATION.md#closed-form-balance-and-source-selection)

- <a id="diffusion-fixes-the-weight-of-a-possible-linear-form-invariant"></a>[Diffusion fixes the weight of a possible linear form invariant](SINE_CONSTITUTIVE_INFORMATION.md#diffusion-fixes-the-weight-of-a-possible-linear-form-invariant)

- <a id="whole-graph-balance-is-equivalent-to-an-odd-pairwise-source"></a>[Whole-graph balance is equivalent to an odd pairwise source](SINE_CONSTITUTIVE_INFORMATION.md#whole-graph-balance-is-equivalent-to-an-odd-pairwise-source)

- <a id="cosine-exchange-selects-sine-with-a-separate-phase-row-premise"></a>[Cosine exchange selects sine, with a separate phase-row premise](SINE_CONSTITUTIVE_INFORMATION.md#cosine-exchange-selects-sine-with-a-separate-phase-row-premise)

- <a id="the-combined-premises-also-cover-induced-neighborhood-pressure"></a>[The combined premises also cover induced-neighborhood pressure](SINE_CONSTITUTIVE_INFORMATION.md#the-combined-premises-also-cover-induced-neighborhood-pressure)

- <a id="branching-tests-information-that-paths-and-cycles-cannot-determine"></a>[Branching tests information that paths and cycles cannot determine](SINE_CONSTITUTIVE_INFORMATION.md#branching-tests-information-that-paths-and-cycles-cannot-determine)

- <a id="phase-storage-selection-boundary"></a>[phase-storage-selection-boundary](SINE_CONSTITUTIVE_INFORMATION.md#phase-storage-selection-boundary)

- <a id="local-phase-storage-nonselection"></a>[local-phase-storage-nonselection](SINE_CONSTITUTIVE_INFORMATION.md#local-phase-storage-nonselection)

- <a id="exact-local-agreement-does-not-select-global-winding-passage"></a>[Exact local agreement does not select global winding passage](SINE_CONSTITUTIVE_INFORMATION.md#exact-local-agreement-does-not-select-global-winding-passage)

- <a id="first-phase-moment-sufficiency"></a>[first-phase-moment-sufficiency](SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency)

- <a id="what-restricting-primitive-phase-information-selects"></a>[What restricting primitive phase information selects](SINE_CONSTITUTIVE_INFORMATION.md#what-restricting-primitive-phase-information-selects)

- <a id="balanced-neighborhood-information-admission"></a>[balanced-neighborhood-information-admission](SINE_CONSTITUTIVE_INFORMATION.md#balanced-neighborhood-information-admission)

- <a id="balanced-geometry-does-not-independently-select-the-phase-current"></a>[Balanced geometry does not independently select the phase current](SINE_CONSTITUTIVE_INFORMATION.md#balanced-geometry-does-not-independently-select-the-phase-current)

- <a id="equal-first-moments-conceal-a-higher-harmonic-response"></a>[Equal first moments conceal a higher-harmonic response](SINE_CONSTITUTIVE_INFORMATION.md#equal-first-moments-conceal-a-higher-harmonic-response)

- <a id="finite-phase-information-response"></a>[finite-phase-information-response](SINE_CONSTITUTIVE_INFORMATION.md#finite-phase-information-response)

- <a id="a-finite-observation-of-the-information-restriction"></a>[A finite observation of the information restriction](SINE_CONSTITUTIVE_INFORMATION.md#a-finite-observation-of-the-information-restriction)

- <a id="phase-motion-information"></a>[phase-motion-information](SINE_CONSTITUTIVE_INFORMATION.md#phase-motion-information)

- <a id="phase-information-must-retain-its-pairing-with-motion"></a>[Phase information must retain its pairing with motion](SINE_CONSTITUTIVE_INFORMATION.md#phase-information-must-retain-its-pairing-with-motion)

- <a id="existing-acute-star-observations-separate-the-premises"></a>[Existing acute-star observations separate the premises](SINE_CONSTITUTIVE_INFORMATION.md#existing-acute-star-observations-separate-the-premises)

- <a id="119-form-balance-common-origins-and-the-primitive-information-contract"></a>[11.9 Form balance, common origins and the primitive information contract](SINE_CONSTITUTIVE_INFORMATION.md#119-form-balance-common-origins-and-the-primitive-information-contract)

- <a id="form-balance-common-origin"></a>[form-balance-common-origin](SINE_CONSTITUTIVE_INFORMATION.md#form-balance-common-origin)

- <a id="exact-relative-equivalence-and-its-retained-reconstruction-coordinate"></a>[Exact relative equivalence and its retained reconstruction coordinate](SINE_CONSTITUTIVE_INFORMATION.md#exact-relative-equivalence-and-its-retained-reconstruction-coordinate)

- <a id="the-balancing-pressure-consumes-global-state-and-capacity-ratios"></a>[The balancing pressure consumes global state and capacity ratios](SINE_CONSTITUTIVE_INFORMATION.md#the-balancing-pressure-consumes-global-state-and-capacity-ratios)

- <a id="zero-capacity-rates-and-pressures-have-different-limits"></a>[Zero-capacity rates and pressures have different limits](SINE_CONSTITUTIVE_INFORMATION.md#zero-capacity-rates-and-pressures-have-different-limits)

- <a id="frozen-static-discriminators-and-existing-consumers"></a>[Frozen static discriminators and existing consumers](SINE_CONSTITUTIVE_INFORMATION.md#frozen-static-discriminators-and-existing-consumers)

- <a id="a-scale-independent-reserved-capacity-test"></a>[A scale-independent reserved capacity test](RELATIONAL_RESPONSE_IDENTIFICATION.md#a-scale-independent-reserved-capacity-test)

- <a id="normalized-capacity-discriminator"></a>[normalized-capacity-discriminator](RELATIONAL_RESPONSE_IDENTIFICATION.md#normalized-capacity-discriminator)

- <a id="prior-uniform-temporal-admission-for-the-normalized-test"></a>[Prior-uniform temporal admission for the normalized test](RELATIONAL_RESPONSE_IDENTIFICATION.md#prior-uniform-temporal-admission-for-the-normalized-test)

- <a id="capacity-discriminator-temporal-admission"></a>[capacity-discriminator-temporal-admission](RELATIONAL_RESPONSE_IDENTIFICATION.md#capacity-discriminator-temporal-admission)

- <a id="bounded-software-verification-of-the-phase-sampling-protocol"></a>[Bounded software verification of the phase-sampling protocol](RELATIONAL_RESPONSE_IDENTIFICATION.md#bounded-software-verification-of-the-phase-sampling-protocol)

- <a id="capacity-discriminator-sampled-response"></a>[capacity-discriminator-sampled-response](RELATIONAL_RESPONSE_IDENTIFICATION.md#capacity-discriminator-sampled-response)

- <a id="14-finite-capacity-intervention-response"></a>[14. Finite capacity-intervention response](RELATIONAL_RESPONSE_IDENTIFICATION.md#14-finite-capacity-intervention-response)

- <a id="finite-capacity-intervention-response"></a>[finite-capacity-intervention-response](RELATIONAL_RESPONSE_IDENTIFICATION.md#finite-capacity-intervention-response)

- <a id="a-relative-observation-tied-directly-to-pressure"></a>[A relative observation tied directly to pressure](RELATIONAL_RESPONSE_IDENTIFICATION.md#a-relative-observation-tied-directly-to-pressure)

- <a id="the-complete-receiver-identity-and-what-the-leading-orders-mean"></a>[The complete receiver identity and what the leading orders mean](RELATIONAL_RESPONSE_IDENTIFICATION.md#the-complete-receiver-identity-and-what-the-leading-orders-mean)

- <a id="prospective-numerical-protocol"></a>[Prospective numerical protocol](RELATIONAL_RESPONSE_IDENTIFICATION.md#prospective-numerical-protocol)

- <a id="first-retained-evaluation-2026-09-27"></a>[First retained evaluation: 2026-09-27](RELATIONAL_RESPONSE_IDENTIFICATION.md#first-retained-evaluation-2026-09-27)

- <a id="retained-evidence-consolidation"></a>[Retained-evidence consolidation](RELATIONAL_RESPONSE_IDENTIFICATION.md#retained-evidence-consolidation)

- <a id="relational-response-consolidation"></a>[relational-response-consolidation](RELATIONAL_RESPONSE_IDENTIFICATION.md#relational-response-consolidation)

- <a id="15-local-recovery-under-the-complete-relational-law"></a>[15. Local recovery under the complete relational law](RELATIONAL_RECOVERY_AND_INTERACTION.md#15-local-recovery-under-the-complete-relational-law)

- <a id="relational-local-recovery"></a>[relational-local-recovery](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)

- <a id="equilibria-reuse-the-existing-circulation-classification"></a>[Equilibria reuse the existing circulation classification](RELATIONAL_RECOVERY_AND_INTERACTION.md#equilibria-reuse-the-existing-circulation-classification)

- <a id="quotient-linearization-and-the-restoring-mechanism"></a>[Quotient linearization and the restoring mechanism](RELATIONAL_RECOVERY_AND_INTERACTION.md#quotient-linearization-and-the-restoring-mechanism)

- <a id="regular-equilibrium-stiffness"></a>[regular-equilibrium-stiffness](RELATIONAL_RECOVERY_AND_INTERACTION.md#regular-equilibrium-stiffness)

- <a id="full-network-stiffness-at-a-regular-equilibrium"></a>[Full-network stiffness at a regular equilibrium](RELATIONAL_RECOVERY_AND_INTERACTION.md#full-network-stiffness-at-a-regular-equilibrium)

- <a id="an-explicit-local-domain-and-the-offset-limits"></a>[An explicit local domain and the offset limits](RELATIONAL_RECOVERY_AND_INTERACTION.md#an-explicit-local-domain-and-the-offset-limits)

- <a id="cycle-rates-competing-laws-and-excluded-boundaries"></a>[Cycle rates, competing laws and excluded boundaries](RELATIONAL_RECOVERY_AND_INTERACTION.md#cycle-rates-competing-laws-and-excluded-boundaries)

- <a id="bounded-c5-recovery-control-and-first-retained-evaluation"></a>[Bounded C5 recovery control and first retained evaluation](RELATIONAL_RECOVERY_AND_INTERACTION.md#bounded-c5-recovery-control-and-first-retained-evaluation)

- <a id="16-interaction-of-two-recoverable-regions-under-the-same-law"></a>[16. Interaction of two recoverable regions under the same law](RELATIONAL_RECOVERY_AND_INTERACTION.md#16-interaction-of-two-recoverable-regions-under-the-same-law)

- <a id="relational-region-interaction"></a>[relational-region-interaction](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-region-interaction)

- <a id="joined-support-changes-the-shared-geometry-not-the-evolution-rule"></a>[Joined support changes the shared geometry, not the evolution rule](RELATIONAL_RECOVERY_AND_INTERACTION.md#joined-support-changes-the-shared-geometry-not-the-evolution-rule)

- <a id="a-transmitted-response-that-changes-internal-geometry"></a>[A transmitted response that changes internal geometry](RELATIONAL_RECOVERY_AND_INTERACTION.md#a-transmitted-response-that-changes-internal-geometry)

- <a id="regional-accounting-and-the-information-that-cannot-be-discarded"></a>[Regional accounting and the information that cannot be discarded](RELATIONAL_RECOVERY_AND_INTERACTION.md#regional-accounting-and-the-information-that-cannot-be-discarded)

- <a id="shared-signed-work-and-regional-rate-integration"></a>[Shared signed-work and regional-rate integration](RELATIONAL_RECOVERY_AND_INTERACTION.md#shared-signed-work-and-regional-rate-integration)

- <a id="relational-work-integration"></a>[relational-work-integration](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-work-integration)

- <a id="relational-local-composition"></a>[relational-local-composition](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-composition)

- <a id="prospective-finite-control"></a>[Prospective finite control](RELATIONAL_RECOVERY_AND_INTERACTION.md#prospective-finite-control)

- <a id="first-retained-transmission-evaluation-2026-09-27"></a>[First retained transmission evaluation: 2026-09-27](RELATIONAL_RECOVERY_AND_INTERACTION.md#first-retained-transmission-evaluation-2026-09-27)

- <a id="17-regular-domain-admission-before-a-formation-experiment"></a>[17. Regular-domain admission before a formation experiment](RELATIONAL_DOMAIN_AND_CAPTURE.md#17-regular-domain-admission-before-a-formation-experiment)

- <a id="relational-formation-domain"></a>[relational-formation-domain](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-formation-domain)

- <a id="relational-regular-domain-admission"></a>[relational-regular-domain-admission](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-regular-domain-admission)

- <a id="four-boundaries-with-different-meanings"></a>[Four boundaries with different meanings](RELATIONAL_DOMAIN_AND_CAPTURE.md#four-boundaries-with-different-meanings)

- <a id="a-derived-cycle-invariant-gives-a-genuine-pure-cycle-obstruction"></a>[A derived cycle invariant gives a genuine pure-cycle obstruction](RELATIONAL_DOMAIN_AND_CAPTURE.md#a-derived-cycle-invariant-gives-a-genuine-pure-cycle-obstruction)

- <a id="relational-cycle-resultant-obstruction"></a>[relational-cycle-resultant-obstruction](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-cycle-resultant-obstruction)

- <a id="a-sharp-c5-zero-resultant-storage-barrier"></a>[A sharp C5 zero-resultant storage barrier](RELATIONAL_DOMAIN_AND_CAPTURE.md#a-sharp-c5-zero-resultant-storage-barrier)

- <a id="added-support-gives-a-regular-route-to-an-admitted-target"></a>[Added support gives a regular route to an admitted target](RELATIONAL_DOMAIN_AND_CAPTURE.md#added-support-gives-a-regular-route-to-an-admitted-target)

- <a id="relational-regular-winding-crossing"></a>[relational-regular-winding-crossing](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-regular-winding-crossing)

- <a id="the-same-coupled-law-realizes-a-transversal-crossing"></a>[The same coupled law realizes a transversal crossing](RELATIONAL_DOMAIN_AND_CAPTURE.md#the-same-coupled-law-realizes-a-transversal-crossing)

- <a id="admission-verdict-and-execution-boundary"></a>[Admission verdict and execution boundary](RELATIONAL_DOMAIN_AND_CAPTURE.md#admission-verdict-and-execution-boundary)

- <a id="18-certified-phase-domain-execution-and-a-retained-crossing-control"></a>[18. Certified phase-domain execution and a retained crossing control](RELATIONAL_DOMAIN_AND_CAPTURE.md#18-certified-phase-domain-execution-and-a-retained-crossing-control)

- <a id="relational-positive-resultant-execution"></a>[relational-positive-resultant-execution](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-positive-resultant-execution)

- <a id="a-sufficient-regular-chamber-for-the-unchanged-joint-law"></a>[A sufficient regular chamber for the unchanged joint law](RELATIONAL_DOMAIN_AND_CAPTURE.md#a-sufficient-regular-chamber-for-the-unchanged-joint-law)

- <a id="certified-admission-on-the-full-regular-phase-domain"></a>[Certified admission on the full regular phase domain](RELATIONAL_DOMAIN_AND_CAPTURE.md#certified-admission-on-the-full-regular-phase-domain)

- <a id="relational-full-regular-execution"></a>[relational-full-regular-execution](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-full-regular-execution)

- <a id="regular-domain-continuation-and-boundary-access"></a>[Regular-domain continuation and boundary access](RELATIONAL_DOMAIN_AND_CAPTURE.md#regular-domain-continuation-and-boundary-access)

- <a id="which-earlier-conclusions-retain-their-scope"></a>[Which earlier conclusions retain their scope](RELATIONAL_DOMAIN_AND_CAPTURE.md#which-earlier-conclusions-retain-their-scope)

- <a id="fixed-preparation-and-independently-stated-questions"></a>[Fixed preparation and independently stated questions](RELATIONAL_DOMAIN_AND_CAPTURE.md#fixed-preparation-and-independently-stated-questions)

- <a id="relational-reserved-crossing-response"></a>[relational-reserved-crossing-response](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-reserved-crossing-response)

- <a id="retained-response-crossing-succeeded-final-capture-is-unavailable"></a>[Retained response: crossing succeeded; final capture is unavailable](RELATIONAL_DOMAIN_AND_CAPTURE.md#retained-response-crossing-succeeded-final-capture-is-unavailable)

- <a id="reflection-reduction-explains-the-next-mathematical-obligation"></a>[Reflection reduction explains the next mathematical obligation](RELATIONAL_DOMAIN_AND_CAPTURE.md#reflection-reduction-explains-the-next-mathematical-obligation)

- <a id="relational-reflected-capture-boundary"></a>[relational-reflected-capture-boundary](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-reflected-capture-boundary)

- <a id="19-protected-capture-and-conditional-basin-selection"></a>[19. Protected capture and conditional basin selection](RELATIONAL_DOMAIN_AND_CAPTURE.md#19-protected-capture-and-conditional-basin-selection)

- <a id="relational-protected-capture"></a>[relational-protected-capture](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-protected-capture)

- <a id="a-protected-region-around-the-winding-one-target"></a>[A protected region around the winding-one target](RELATIONAL_DOMAIN_AND_CAPTURE.md#a-protected-region-around-the-winding-one-target)

- <a id="the-saddle-has-a-target-directed-unstable-branch"></a>[The saddle has a target-directed unstable branch](RELATIONAL_DOMAIN_AND_CAPTURE.md#the-saddle-has-a-target-directed-unstable-branch)

- <a id="relational-saddle-capture-route"></a>[relational-saddle-capture-route](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-saddle-capture-route)

- <a id="the-opposite-branch-approaches-consensus"></a>[The opposite branch approaches consensus](RELATIONAL_DOMAIN_AND_CAPTURE.md#the-opposite-branch-approaches-consensus)

- <a id="shared-admission-of-the-sufficient-ideal-law-basins"></a>[Shared admission of the sufficient ideal-law basins](RELATIONAL_DOMAIN_AND_CAPTURE.md#shared-admission-of-the-sufficient-ideal-law-basins)

- <a id="a-distinct-winding-zero-preparation-with-a-bounded-crossing-theorem"></a>[A distinct winding-zero preparation with a bounded crossing theorem](RELATIONAL_DOMAIN_AND_CAPTURE.md#a-distinct-winding-zero-preparation-with-a-bounded-crossing-theorem)

- <a id="relational-upper-corner-preparation"></a>[relational-upper-corner-preparation](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-upper-corner-preparation)

- <a id="frozen-upper-corner-response-regular-sector-acquisition-and-a-remaining-certificate-gap"></a>[Frozen upper-corner response: regular sector acquisition and a remaining certificate gap](RELATIONAL_DOMAIN_AND_CAPTURE.md#frozen-upper-corner-response-regular-sector-acquisition-and-a-remaining-certificate-gap)

- <a id="relational-upper-corner-response"></a>[relational-upper-corner-response](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-upper-corner-response)

- <a id="20-a-full-state-acute-sector-barrier-without-reflection"></a>[20. A full-state acute-sector barrier without reflection](RELATIONAL_DOMAIN_AND_CAPTURE.md#20-a-full-state-acute-sector-barrier-without-reflection)

- <a id="relational-acute-sector-capture"></a>[relational-acute-sector-capture](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-acute-sector-capture)

- <a id="the-cycle-geometry-supplies-the-barrier"></a>[The cycle geometry supplies the barrier](RELATIONAL_DOMAIN_AND_CAPTURE.md#the-cycle-geometry-supplies-the-barrier)

- <a id="compactness-the-target-and-the-zero-loss-set"></a>[Compactness, the target and the zero-loss set](RELATIONAL_DOMAIN_AND_CAPTURE.md#compactness-the-target-and-the-zero-loss-set)

- <a id="capture-shared-by-a-declared-strictly-dissipative-law-class"></a>[Capture shared by a declared strictly dissipative law class](RELATIONAL_DOMAIN_AND_CAPTURE.md#capture-shared-by-a-declared-strictly-dissipative-law-class)

- <a id="relational-sector-law-class"></a>[relational-sector-law-class](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-sector-law-class)

- <a id="evidence-and-execution-boundary"></a>[Evidence and execution boundary](RELATIONAL_DOMAIN_AND_CAPTURE.md#evidence-and-execution-boundary)

- <a id="shared-certificate-and-retained-endpoint-reanalysis"></a>[Shared certificate and retained endpoint reanalysis](RELATIONAL_DOMAIN_AND_CAPTURE.md#shared-certificate-and-retained-endpoint-reanalysis)

- <a id="consolidation-capacity-storage-scale-and-a-quantitative-regularity-margin"></a>[Consolidation: capacity, storage scale and a quantitative regularity margin](RELATIONAL_DOMAIN_AND_CAPTURE.md#consolidation-capacity-storage-scale-and-a-quantitative-regularity-margin)

- <a id="relational-sector-consolidation"></a>[relational-sector-consolidation](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-sector-consolidation)

- <a id="validated-continuous-transit-on-the-exact-reflected-subsystem"></a>[Validated continuous transit on the exact reflected subsystem](RELATIONAL_FORMATION_CONTROLS.md#validated-continuous-transit-on-the-exact-reflected-subsystem)

- <a id="relational-validated-transit"></a>[relational-validated-transit](RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit)

- <a id="whole-time-enclosure-and-error-propagation"></a>[Whole-time enclosure and error propagation](RELATIONAL_FORMATION_CONTROLS.md#whole-time-enclosure-and-error-propagation)

- <a id="joining-transit-to-maintenance"></a>[Joining transit to maintenance](RELATIONAL_FORMATION_CONTROLS.md#joining-transit-to-maintenance)

- <a id="retained-original-ivp-proof-audit"></a>[Retained original-IVP proof audit](RELATIONAL_FORMATION_CONTROLS.md#retained-original-ivp-proof-audit)

- <a id="synergy-qualitative-robustness-beyond-exact-reflection"></a>[Synergy: qualitative robustness beyond exact reflection](RELATIONAL_FORMATION_CONTROLS.md#synergy-qualitative-robustness-beyond-exact-reflection)

- <a id="formation-robustness-under-small-admitted-changes-of-phase-law"></a>[Formation robustness under small admitted changes of phase law](RELATIONAL_FORMATION_CONTROLS.md#formation-robustness-under-small-admitted-changes-of-phase-law)

- <a id="relational-formation-law-robustness"></a>[relational-formation-law-robustness](RELATIONAL_FORMATION_CONTROLS.md#relational-formation-law-robustness)

- <a id="the-reflected-barrier-also-permits-a-strict-loss-class"></a>[The reflected barrier also permits a strict-loss class](RELATIONAL_FORMATION_CONTROLS.md#the-reflected-barrier-also-permits-a-strict-loss-class)

- <a id="a-corridor-around-the-retained-reference-tubes"></a>[A corridor around the retained reference tubes](RELATIONAL_FORMATION_CONTROLS.md#a-corridor-around-the-retained-reference-tubes)

- <a id="bounded-discrepancy-and-a-first-exit-argument"></a>[Bounded discrepancy and a first-exit argument](RELATIONAL_FORMATION_CONTROLS.md#bounded-discrepancy-and-a-first-exit-argument)

- <a id="a-static-logarithmic-norm-refinement"></a>[A static logarithmic-norm refinement](RELATIONAL_FORMATION_CONTROLS.md#a-static-logarithmic-norm-refinement)

- <a id="entry-and-the-resulting-formation-conclusion"></a>[Entry and the resulting formation conclusion](RELATIONAL_FORMATION_CONTROLS.md#entry-and-the-resulting-formation-conclusion)

- <a id="zero-initial-form-contrast-a-prospective-phase-to-form-discriminator"></a>[Zero initial form contrast: a prospective phase-to-form discriminator](RELATIONAL_FORMATION_CONTROLS.md#zero-initial-form-contrast-a-prospective-phase-to-form-discriminator)

- <a id="relational-zero-form-control"></a>[relational-zero-form-control](RELATIONAL_FORMATION_CONTROLS.md#relational-zero-form-control)

- <a id="what-is-predicted-before-evolving-the-control"></a>[What is predicted before evolving the control](RELATIONAL_FORMATION_CONTROLS.md#what-is-predicted-before-evolving-the-control)

- <a id="retained-result-transient-winding-followed-by-proved-consensus"></a>[Retained result: transient winding followed by proved consensus](RELATIONAL_FORMATION_CONTROLS.md#retained-result-transient-winding-followed-by-proved-consensus)

- <a id="general-mechanism-and-its-exact-stationary-boundary"></a>[General mechanism and its exact stationary boundary](RELATIONAL_FORMATION_CONTROLS.md#general-mechanism-and-its-exact-stationary-boundary)

- <a id="equal-storage-form-reversal-separating-energy-from-direction"></a>[Equal-storage form reversal: separating energy from direction](RELATIONAL_FORMATION_CONTROLS.md#equal-storage-form-reversal-separating-energy-from-direction)

- <a id="relational-reversed-form-control"></a>[relational-reversed-form-control](RELATIONAL_FORMATION_CONTROLS.md#relational-reversed-form-control)

- <a id="exact-initial-match-and-local-discriminator"></a>[Exact initial match and local discriminator](RELATIONAL_FORMATION_CONTROLS.md#exact-initial-match-and-local-discriminator)

- <a id="retained-result-equal-initial-energy-different-limiting-identity"></a>[Retained result: equal initial energy, different limiting identity](RELATIONAL_FORMATION_CONTROLS.md#retained-result-equal-initial-energy-different-limiting-identity)

- <a id="what-the-three-preparations-establish-together"></a>[What the three preparations establish together](RELATIONAL_FORMATION_CONTROLS.md#what-the-three-preparations-establish-together)

- <a id="phase-consensus-with-a-bounded-reflected-form-preparation-cannot-form-the-target"></a>[Phase consensus with a bounded reflected form preparation cannot form the target](RELATIONAL_FORMATION_CONTROLS.md#phase-consensus-with-a-bounded-reflected-form-preparation-cannot-form-the-target)

- <a id="relational-consensus-preparation-obstruction"></a>[relational-consensus-preparation-obstruction](RELATIONAL_FORMATION_CONTROLS.md#relational-consensus-preparation-obstruction)

- <a id="the-exact-preparation-class-and-conclusion"></a>[The exact preparation class and conclusion](RELATIONAL_FORMATION_CONTROLS.md#the-exact-preparation-class-and-conclusion)

- <a id="a-regular-small-phase-interval-is-guaranteed-before-any-basin-test"></a>[A regular small-phase interval is guaranteed before any basin test](RELATIONAL_FORMATION_CONTROLS.md#a-regular-small-phase-interval-is-guaranteed-before-any-basin-test)

- <a id="form-loses-storage-faster-than-this-interval-can-build-phase-storage"></a>[Form loses storage faster than this interval can build phase storage](RELATIONAL_FORMATION_CONTROLS.md#form-loses-storage-faster-than-this-interval-can-build-phase-storage)

- <a id="application-to-the-frozen-successful-references-storage-budget"></a>[Application to the frozen successful reference's storage budget](RELATIONAL_FORMATION_CONTROLS.md#application-to-the-frozen-successful-references-storage-budget)

- <a id="shared-analytic-admission-and-independent-checks"></a>[Shared analytic admission and independent checks](RELATIONAL_FORMATION_CONTROLS.md#shared-analytic-admission-and-independent-checks)

- <a id="a-full-form-phase-storage-bound-excludes-the-same-maintained-target"></a>[A full-form phase-storage bound excludes the same maintained target](RELATIONAL_FORMATION_CONTROLS.md#a-full-form-phase-storage-bound-excludes-the-same-maintained-target)

- <a id="relational-full-consensus-formation-obstruction"></a>[relational-full-consensus-formation-obstruction](RELATIONAL_FORMATION_CONTROLS.md#relational-full-consensus-formation-obstruction)

- <a id="full-form-nonlinear-phase-consensus-obstruction"></a>[Full-form nonlinear phase-consensus obstruction](RELATIONAL_FORMATION_CONTROLS.md#full-form-nonlinear-phase-consensus-obstruction)

- <a id="argument-pressure-has-a-global-storage-bound-on-its-regular-domain"></a>[Argument pressure has a global storage bound on its regular domain](RELATIONAL_FORMATION_CONTROLS.md#argument-pressure-has-a-global-storage-bound-on-its-regular-domain)

- <a id="the-actual-full-support-supplies-a-uniform-loss-bound"></a>[The actual full support supplies a uniform loss bound](RELATIONAL_FORMATION_CONTROLS.md#the-actual-full-support-supplies-a-uniform-loss-bound)

- <a id="a-nonlinear-comparison-includes-all-signs-of-the-exchange"></a>[A nonlinear comparison includes all signs of the exchange](RELATIONAL_FORMATION_CONTROLS.md#a-nonlinear-comparison-includes-all-signs-of-the-exchange)

- <a id="what-is-excluded-and-what-remains-unresolved"></a>[What is excluded and what remains unresolved](RELATIONAL_FORMATION_CONTROLS.md#what-is-excluded-and-what-remains-unresolved)
