# Relational storage and local phase/form exchange admission

**Status:** Conditional constitutive admission, not a selected TNFR law.
This owner tests a new premise while retaining the native two-channel pressure.
Section 13 links its opt-in engine implementation; section 14 retains the
first evaluated nonlinear capacity response. Section 15 proves conditional
local recovery under that full continuous law and retains its bounded engine
control. Section 16 derives and tests interaction between two prepared regions.
Section 17 separates genuine regular-domain obstructions from the executor's
acute cutoff. None changes production defaults or makes a physical-identification claim.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
remains the sole task queue.

### Consolidated claim boundary

| Evidence level | What this study establishes | What remains supplied or open |
| --- | --- | --- |
| Model premises | An explicit connected graph, initial triad, held capacities, pressure coefficients, storage scale and structural clock | Their physical origin and autonomous preparation |
| Conditional derivation | The phase row follows from the chosen storage balance, independent own-capacity dependence and zero-capacity freezing | Those premises are not selected by the nodal product alone |
| Exact continuous results | Joint storage loss, initial response orders, complete receiver identity and local recovery of acute equilibria under positive capacity/dissipation | No unrestricted attraction, autonomous formation or one-node closure follows |
| Engine contract | One explicit joint Euler step with admission and represented defects | A step need not inherit continuous storage nonincrease |
| Retained numerical evidence | Four law/capacity cases at three step sizes meet the prior finite-response criteria | These are twelve simulations, not twelve independent physical experiments or an exact-flow error certificate |
| Prepared geometry | The local theorem and two-grid C5 control support recovery; the paired-region control tests transmitted form/phase deformation | Autonomous formation, physical identification and unique capacity selection remain open |
| Formation-domain admission | Pure cycles have a resultant-protected obstruction between acute winding sectors; a higher-degree example permits a local ordinary-winding crossing under the same regular law | No remote preparation-to-attractor formation, global continuation or wider engine execution is established |

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

## 5. A prospective spectral discriminator

Take common positive capacity `nu`, zero form and phase consensus. Here
`H=pi D`, `Dg=-L/pi` and `Bx=0`. A nonzero Laplacian mode with eigenvalue
`lambda` has the exact linearization

\[
\binom{\dot x_\lambda}{\dot\theta_\lambda}
=\nu\lambda
 \begin{pmatrix}-e&-w/\pi\\ w/(\beta\pi)&0\end{pmatrix}
 \binom{x_\lambda}{\theta_\lambda},
\]

\[
\boxed{s^2+\nu e\lambda s+
 \frac{\nu^2w^2\lambda^2}{\beta\pi^2}=0.}
\]

All nonzero spatial modes share the same damped/critical/oscillatory class:
the discriminant has the sign of `e^2-4w^2/(beta*pi^2)`. Their rates scale
linearly with `lambda`. For the existing regular-support cotangent model,
the determinant instead scales with `lambda` at held degree, not `lambda^2`.
Two distinct nonzero modes can therefore distinguish these complete laws
without tuning a coefficient to a trajectory or repeating the C8 experiment.
For example, the C6 eigenvalues `1/2` and `3/2` give determinant ratio `9`
for this exchange premise and `3` for the cotangent premise. These ratios
cancel the respective unknown storage/exchange scale and common clock.
This is a conditional linear prediction. Section 9 establishes a separate
finite-amplitude projected identity and a prepared-state implementation
control; neither identifies physical matter.

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

## 9. Exact finite-amplitude discriminator and prepared-state control

<a id="finite-modal-exchange-discriminator"></a>

Use one unit regular cycle of degree `d=2`, unit held capacity, no forcing or
events, and the same fixed `e,w>0`. Prepare each mode separately:
`x(0)=a*v`, `a!=0`, `L*v=lambda*v`, `1^T*v=0`, and primitive phase consensus.
For the cotangent comparison use its complete general-scale law

\[
\dot x=-eLx+wg+\eta^{-1}Kx,\qquad
\dot\theta=\eta^{-1}H^{-1}x,\qquad
K_{ij}=\frac{x_j\partial_iH_j-x_i\partial_jH_i}{H_iH_j}=-K_{ji}.
\]

At consensus, `H=pi*d*I`, `DH=0`, `K=0`, `Dg=-L/pi`, and both form
velocities are `-e*lambda*a*v`. Let `A=v^T*x/(v^T*v)`. Although differentiating
the cotangent correction can give a nonzero cubic vector, its projection is
exactly zero: `v^T*dot(K)*x=a*v^T*dot(K)*v=0`. Consequently

\[
\begin{array}{c|cc}
 & \text{relational} & \text{cotangent}\\\hline
\Omega_\lambda=\dfrac{v^T\dot\theta}{a\,v^Tv}
 & \dfrac{w\lambda}{\beta\pi} & \dfrac{1}{\eta\pi d}\\[5pt]
Q_\lambda=e^2\lambda^2-\dfrac{\ddot A(0)}a
 & \dfrac{w^2\lambda^2}{\beta\pi^2} & \dfrac{w\lambda}{\eta\pi^2d}
\end{array}
\]

These are exact **instantaneous finite-amplitude** identities, with no
small-amplitude remainder. On C6 choose `(lambda_low,lambda_high)=(1/2,3/2)`.
The phase-gain ratio is `3` versus `1`; the restoring ratio is **`9` versus
`3`**. The respective unknown storage scale and a common clock cancel.
Under a clock conversion, the diffusion rate subtracted in `Q` must transform
too. Different per-mode clocks or refitted coefficients are not admitted.

**Boundary controls.** This does not make a nonlinear Fourier mode invariant.
For `v=(1,1/2,-1/2,-1,-1/2,1/2)` and `eta=1`, the cotangent contribution
`dot(K)*x` is `a^3*(-1,1,-1,1,-1,1)/(16*pi^2)`: a nonzero orthogonal harmonic.
Adding a common form offset `mu` instead changes its projected acceleration
divided by `a` by `-5*mu^2/(24*pi^2)`. A mixed preparation can transfer
connection work between the observed modes. The exact controls differentiate
the complete connection before projection and retain these exclusions.

**Finite constitutive probe, distinct from temporal evolution.** Initialize
two nearby states on the tangent line of each declared law,

\[
x_\pm=(1\mp he\lambda)a v,\qquad
\theta_\pm=\pm h\dot\theta(0),\qquad h>0.
\]

Keep their lifted phase range below `pi/2`. On this degree-two acute chart,
the neighbor resultant has the neighbor midpoint angle, hence `g=-L*theta/pi`.
At both endpoints `x` remains collinear with `v`, so the additional cotangent
`v^T*K*x` vanishes. Therefore, writing `f` for the complete model form field,

\[
\frac{v^Tf(x_+,\theta_+)-v^Tf(x_-,\theta_-)}{2h\,v^Tv}
=\ddot A(0).
\]

This finite prepared-state identity has no time-discretization remainder.
The endpoints are **not evolved trajectory samples**. The two candidates
supply their own phase directions; the same native pressure implementation
evaluates the projected responses. This verifies constitutive implementation
consistency, not an independent experiment selecting an autonomous law.
The full cotangent vector is not computed by dropping `Kx`; only its proven
zero projection is omitted.

The maintained [producer](../../benchmarks/phase_form_exchange_comparison.py)
fixes `a=1/16`, `h=1/8`, `e=w=1/2`, `eta=1`, `beta=1/2`, and integer modes
`(2,1,-1,-2,-1,1)` and `(2,-1,-1,2,-1,-1)`. The scales match the low mode
**a priori** through `beta=eta*d*w*lambda_low`; neither is refitted on the high
mode. Both preparations also lie inside their respective conditional
regular-domain storage bounds. The probe phases stay in their separately
checked acute chart.

Before pressure execution, preparation freezes every materialized state,
exact-rational midpoint references using represented binary64 `pi`, source
fingerprints, runtime versions and a restoring allowance `epsilon=1e-10`.
Given reference restoring values `q_l>epsilon,q_h`, its decision interval is
`[(q_h-epsilon)/(q_l+epsilon),(q_h+epsilon)/(q_l-epsilon)]`. Both prospective
intervals exclude the fixed separator `6`. Evaluation records actual native
pressure and the propagated bound

\[
\varepsilon_Q\le
\frac{|P_v(f_+-f_+^{ref})|+|P_v(f_--f_-^{ref})|}{2h|a|},
\qquad P_v(y)=\frac{v^Ty}{v^Tv},
\]

and rejects if this exceeds the frozen allowance. Fraction arithmetic makes
this an exact defect calculation on materialized rates. It is not an interval
certificate for transcendental real arithmetic or a temporal solver estimate.
The independently derived symbolic identities and represented-input checks
have distinct roles. No random input is used.

**Retained first evaluation (2026-09-26).** The
[prediction](../../docs/assets/phase_form_exchange_comparison/result.prediction.json)
was written before the separate
[response](../../docs/assets/phase_form_exchange_comparison/result.json).
Observed ratios were `9.000000000000036` (relational) and
`3.000000000000035` (cotangent). Their largest propagated restoring defects
were respectively `5.63e-17` and `2.76e-16`, below the predeclared `1e-10`.
All eight probes used the checked `fused_canonical` NumPy path, binary64,
and the retained node order; there were no random inputs or temporal steps.
The JSON records contain exact represented rates, fractions and fingerprints.
Their SHA-256 hashes are, respectively,
`34fa669cb5a1c494f5df3e38d71d1d6670fd3e02f31c3511143e1a6186591486`
and `1246771b7d0f6cd4ec78251f4ecdc31b027fe7aa805a1d94b9fb439c2029dd48`.
The producer fingerprint is
`ea19b78874c2dc8cac9a9b0869417fcc5bec0a3aa9a6b942ed17b99f649c50e7`.
The listed file hashes are not a complete transitive dependency archive.

Controls: [focused comparison tests](../../tests/physics/test_phase_form_exchange_comparison.py)
cover full projected acceleration, harmonic leakage, excluded preparations,
finite tangent-line algebra, actual pressure and pre-execution protocol
rejection. The [benchmark guide](../../benchmarks/README.md#running-and-reporting)
owns invocation and immutable-record handling.

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
The selected row happens to use only neighboring form and phase, but local
spatial dependence by itself has not supplied the capacity-composition rule.

This is a conditional model-admission theorem, not a declaration that the
new law is canonical or physically selected. The remaining foundational
question is whether capacity really composes through independent local
activity, or whether neighboring capacities mediate additional phase
exchange. The distinction must be justified from a capacity mechanism or
independent evidence, not by giving a preferred completion a new label.

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
stronger theorem demanding one uniform one-hop law on all graphs and their
unprepared refinements. No such stronger theorem is asserted here.

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
edge phases; [section 18](#relational-positive-resultant-execution) defines the
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

The [execution contract](../../docs/API_CONTRACTS.md#conditional-relational-execution)
owns scalar/clock admission, atomicity, caches, aliases and immutable report
semantics; the [SDK guide](../../docs/CLI_AND_SDK.md#execute-the-conditional-relational-model)
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
The [observation/export contract](../../docs/API_CONTRACTS.md#relational-pattern-observation)
owns admission and exact JSON projection. Routine controls compare selected
retained checkpoints with this owner without altering or rerunning the frozen
campaigns; their historical benchmark wrappers remain immutable evidence.

## 14. Finite capacity-intervention response

<a id="finite-capacity-intervention-response"></a>

This F4 control evolves the admitted reference and the section 12
countermodel after the same capacity intervention. The latter remains a
benchmark-only comparison, not a new production policy. Both use the native
pressure and the same supplied unit K3, form and phase. Set `e=w=1/2`,
`beta=1`; compare held capacities `(1,1,1)` and `(1,1/2,1)`, with
`x=(1/3,0,-1/3)` and `theta=(0,pi/6,-pi/6)` in node order `(0,1,2)`.
The support, initial state, clock and storage scale are independent inputs.

### A relative observation tied directly to pressure

Use the common-rotation-invariant contrast

\[
\chi=\theta_0-\frac{\theta_1+\theta_2}{2}.
\]

Use the common real lift fixed by the supplied preparation and retained by
the admitted Euler segments. On this chart, the exact neighbor-midpoint identity gives
`g_0=-chi/pi`. Thus the proposed phase observation directly determines the
receiver's phase-pressure contribution; no new diagnostic or fitted mapping
is required. For each law `M` define the intervention response
`R_M[y](t)=y_{M,intervened}(t)-y_{M,baseline}(t)` and the difference between
those responses `D_y=R_alt[y]-R_rel[y]`.
The contrast is invariant under common phase rotation. It is not a global
real-valued function of arbitrary independently shifted `2*pi` phase
representatives; those representatives require a consistent lift first.

The initial form gradient is `Bx=(1,0,-1)`. Changing only capacity 1 therefore
leaves the entire initial reference phase row unchanged. It does **not**
leave subsequent trajectories unchanged: node 1 already has nonzero native
form pressure, and its changed form rate feeds back into other nodes later.
In particular `R_rel[chi]'(0)=0` is an instantaneous statement, not an
identically zero finite response.

For the alternative, the active pair mobility changes from `1/2` to `1/3`.
Its complete correction vector at this preparation is therefore multiplied
by `2/3`. Section 12 then gives the exact initial relative response

\[
D_\chi(0)=0,\qquad
D_\chi'(0)=-\frac{2+\sqrt3}{192}<0.
\]

The two laws have identical initial form rates for each capacity preparation.
Differentiating the native receiver row, with `nu_0=1` in both cases, cancels
their equal initial diffusion contribution and gives

\[
D_{x_0}(0)=D_{x_0}'(0)=0,\qquad
D_{x_0}''(0)=-\frac{w}{\pi}D_\chi'(0)
=\frac{2+\sqrt3}{384\pi}>0.
\]

Smoothness on the admitted chart implies a first-order phase difference and
a second-order form difference for sufficiently small positive time. This
derivation identifies how a constitutive capacity assumption reaches the
form row through native pressure. It does not specify an error enclosure at
an arbitrary finite time or choose which capacity premise is physically true.

### The complete receiver identity and what the leading orders mean

The connection is valid along each admitted continuous trajectory, not only
at its initial state. On the same common acute lift, every receiver row has

\[
\dot x_0=-e\left[x_0-\frac{x_1+x_2}{2}\right]-\frac w\pi\chi.
\]

The unchanged receiver capacity `nu_0=1` and common pressure coefficients are
essential. Taking the same four-way linear difference gives the exact identity

\[
\boxed{\quad
\dot D_{x_0}=-e\left[D_{x_0}-\frac{D_{x_1}+D_{x_2}}2\right]
             -\frac w\pi D_\chi.
\quad}
\]

For the equal initial states, integrating the receiver row yields

\[
D_{x_0}(T)=\int_0^T e^{-e(T-s)}
\left[\frac e2(D_{x_1}+D_{x_2})(s)-\frac w\pi D_\chi(s)\right]\,ds.
\]

Thus the phase difference contributes through the existing pressure law,
while neighboring form differences remain part of the response. Omitting
those terms would manufacture an autonomous one-node description that this
comparison has not derived. The sign of `D_chi` alone does not determine
the form response at every later time. Changing the receiver's own capacity
would add product terms and would require a different intervention formula.

The leading `D_chi=O(t)` and `D_x0=O(t^2)` orders describe smooth response
near the initial state. They do not assert a finite propagation delay: both
can be nonzero at every sufficiently small positive time. In an explicit
Euler execution the two laws first separate their phase endpoints, then
their form endpoints at a following step. That discrete ordering is not a
new physical clock or signal-speed law.

At the three retained checkpoints, the same identity can be checked on actual
stored state and native rates, with a represented-arithmetic residual. The
continuous integral and all intermediate numerical states cannot be recovered
from those checkpoints alone. Exact conditional identities, snapshot checks
and three-grid trajectory evidence are therefore separate levels of support.

### Prospective numerical protocol

The [producer](../../benchmarks/relational_capacity_response.py) separates
preparation from evaluation. Before executing any trajectory, freeze:

- Structural horizon `T=1/32`, with `128`, `256` and `512` Euler steps; all
  four law/capacity combinations start from the same supplied form and phase.
- Signed finite hypotheses
  `D_chi(T)/(T*D_chi'(0)) in [3/4,5/4]` and
  `D_x0(T)/(T^2*D_x0''(0)/2) in [3/4,5/4]`. These intervals are prospective
  numerical hypotheses motivated by the initial Taylor coefficients, **not**
  proved Taylor-remainder bounds.
- Each trajectory stays inside the selected chart with minimum acute-edge
  margin greater than `pi/8`; actual represented work residual is at most
  `1e-12` in the declared normalization. The cutoff is an arithmetic
  acceptance policy, not a new structural constant.
- Successive full-state endpoint differences and both discriminating signals
  decrease under step refinement: the fine-to-middle difference is at most
  `3/4` of the middle-to-coarse difference plus `1e-12`. Each finest signal
  exceeds `32` times its fine-to-middle difference plus the same numerical
  floor inside that factor. These checks supply empirical numerical
  separation, not a rigorous ODE error bound.
- Source/runtime fingerprints, scalar precision, effective coefficients,
  native pressure path and preparation order accompany the prediction.
  Initial, intermediate and final state evidence, phase margins and actual
  work/Euler defects accompany each execution. The reference uses the shared
  production step; the comparison reuses its field and Euler arithmetic.

There is no fitting, trajectory-based pressure reconstruction or independent
physical measurement in this control. Failure of a sign, interval, chart or
refinement condition must remain a failed or unresolved frozen result; it
does not authorize adjusting the response afterward. Earlier constitutive
and source-relative records remain immutable.

### First retained evaluation: 2026-09-27

The [frozen prediction](../../docs/assets/relational_capacity_response/result.prediction.json)
preceded all twelve trajectories. The
[response record](../../docs/assets/relational_capacity_response/result.json)
retains their complete initial, midpoint and final states and per-trace
evidence. All prospective gates passed without changing the inputs or
acceptance rules. Source hashes and Python/NumPy/NetworkX versions are in the
prediction; the observed native pressure path was `fused_canonical` throughout.

| Observation at `T=1/32`, finest grid | Difference of intervention responses | Ratio to the leading Taylor prediction | Change from the middle grid |
| --- | --- | --- | --- |
| Relative phase `chi` | `-6.004469361879845e-4` | `0.9885036850` | `1.3625854652e-8` |
| Receiver form `x_0` | `1.4825332026457971e-6` | `0.9814494374` | `2.8041313160e-9` |

The refinement changes approximately halved. Full-state midpoint and endpoint
refinements passed separately for every law/capacity pair. The smallest chart
margin was `0.5207904177` radians, above the predeclared `pi/8`; the smallest
phase metric was `5.4369609494`. Maximum actual represented work residual was
`1.377e-16`. The report separately retains nonzero Euler storage defects;
that small instantaneous residual is not a bound on time-integration error.

The reference's own finite phase response to the capacity change was
`-6.909006713886213e-6`, despite its zero initial response slope. This confirms
the need to distinguish local instantaneous dependence from later network
feedback. The alternative's response was `-6.073559429018707e-4`. Their
difference supplies the first row of the table rather than assuming the
reference response remains zero.

**Disposition.** This closes the named finite-response F4 obligation in its
numerical scope. The two supplied capacity laws produce the predicted distinct
phase-to-pressure-to-form responses while retaining the same ideal total
storage balance. This demonstrates a discriminating internal model prediction,
not evidence preferring one law in nature, a certified ODE enclosure, formation
of a new NFR or indefinite persistence. The
[controls](../../tests/physics/test_relational_capacity_response.py) independently
check initial identities, a comparison step, fail-closed protocol handling and
the retained endpoint/refinement arithmetic without regenerating the campaign.

Here "same balance" means the same functional loss identity evaluated on each
law's own state. Once the trajectories differ, their numerical storage and
loss values need not be identical. Agreement with that identity therefore
cannot replace an intervention-sensitive response test.

### Retained-evidence consolidation

<a id="relational-response-consolidation"></a>

The [read-only audit](../../benchmarks/relational_capacity_audit.py) is the
single reconstruction owner for the retained F4 record. Tests reuse it rather
than maintaining another copy of the numerical decision logic. It does not
advance a state, refit a parameter, replace the prediction or rerun the twelve
simulations. Its checks separate the following obligations:

| Check | Evidence available | Boundary |
| --- | --- | --- |
| Preparation | All twelve initial checkpoints agree with the frozen support, form, phase frame and capacity choice | Internal record binding, not proof of physical acquisition |
| Checkpoint state and arithmetic | Thirty-six initial/midpoint/final states admit their stored field/rate/balance data; the K3 midpoint pressure and receiver identity are checked independently | A matching `nu*p` product alone does not validate pressure |
| Frozen numerical verdict | Intervention responses, Taylor ratios and refinement decisions are reconstructed from retained coordinates | Rechecking the evaluated prediction is not a new reserved prediction |
| Current source/runtime | Listed byte fingerprints and runtime versions determine availability of current-engine snapshot replay; numerical replay agreement is reported separately | The first manifest does not identify the operating system, libm or every dependency, so listed agreement alone does not guarantee bitwise reproduction |
| Unretained steps | Reported extrema must be compatible with the saved checkpoints | Their exact values over omitted steps and the complete nodal time integral cannot be independently recovered |

The receiver check retains its actual represented defect, rather than assigning
zero. With `rho=x_0-(x_1+x_2)/2` and represented `pi64`, it decomposes as

\[
\begin{aligned}
\varepsilon&=\dot x_0+e\rho+w\chi/\pi_{64}\\
&=(\dot x_0-p_0)
 +(p_0+e\rho-wg_0)
 +w(g_0+\chi/\pi_{64}).
\end{aligned}
\]

These terms separate nodal arithmetic, the native pressure split and the
midpoint representation. Their four-way difference checks the complete receiver
identity at the retained times. The audit's `1e-12` comparison allowance is
a read-only arithmetic policy; it neither changes the original frozen gates
nor becomes a physical constant or an exact-flow error bound.

Current-source replay is reported as unavailable if the installed source or
runtime differs. That does not change the historical verdict reconstructed
from the record. If the listed source/runtime matches but exact numerical
replay disagrees, the audit retains that disagreement separately from its
independent record-consistency checks. Conversely, source agreement cannot
make contradictory state, pressure, storage or decision fields acceptable.
The original producer and
prediction/response bytes remain unchanged by this consolidation.

## 15. Local recovery under the complete relational law

<a id="relational-local-recovery"></a>

### Equilibria reuse the existing circulation classification

Fix the same connected simple unit graph, now with **every held capacity
strictly positive** and `e,w,beta>0`. Let `theta_*` have strictly acute edge
gaps and `grad(V_phi)(theta_*)=0`. The equilibrium family is

\[
\mathcal M_*=
\{(x,\theta)=(c\mathbf1,\theta_*+\alpha\mathbf1):c\in\mathbb R,
\ \alpha\in S^1\}.
\]

These are all acute equilibria: the phase row and invertibility of `N,H`
force `Bx=0`, then the form row forces `g=0`. Conversely those two conditions
make both rows vanish. In the acute chart, zero `g` is equivalent to zero
neighbor sine sums. Therefore the integral-cycle circulation classification
in [the existing owner](../FORCED_SUPPORT_BALANCE.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods)
applies to the *geometry*, without importing that owner's supplied sine
evolution law or its common-capacity premise. Each admitted winding sector
contains at most one such phase shape up to common rotation. Trees and
graphs whose cycle space is spanned by triangles/quadrilaterals admit only
consensus; a unit cycle admits twists `2*pi*ell/n` with `abs(ell)<n/4`.

The resulting form is uniform, but the phase geometry need not be. In
particular, winding-one C5 is an admissible nonconsensus geometry. Its
existence is inherited from the supplied support and initial sector; it is
not spontaneous formation or a differentiated stationary scalar form.

This also separates geometric identity from scalar summaries: at the exact
winding-one C5 lock, the mean phasor is zero and every local mean absolute
edge gap is `2*pi/5`, while `g=0` and both nodal rows vanish. A zero global
phase-order parameter or a configured gradient warning cannot rule out this
locally recoverable geometry. These observations do not select its dynamics.

### Quotient linearization and the restoring mechanism

Choose an orthonormal matrix `R` spanning `1^perp` and local lifted deviations
`x=c*1+R*u`, `theta=theta_*+alpha*1+R*v`. These are local differences from
the prepared circle-valued phase, not a common semicircle assumption on
`theta_*`. Let `K_*` be the Hessian of `V_phi` there: its edge weights are
`cos(delta_*)>0`. Write

\[
\begin{gathered}
B_q=R^TBR>0,\quad K_q=R^TK_*R>0,\\
A_q=R^TND^{-1}R>0,\qquad M_q=R^TNH_*^{-1}R>0.
\end{gathered}
\]

At equilibrium, differentiating `Hg=-grad(V_phi)` gives
`Dg=-H_*^-1 K_*`. The derivative of `H^-1` contributes nothing to the phase
row there because `Bx=0`. The exact quotient Jacobian is therefore

\[
J_q=\begin{pmatrix}
-eA_qB_q&-wM_qK_q\\
(w/\beta)M_qB_q&0
\end{pmatrix}.
\]

This includes arbitrary positive heterogeneous capacities and irregular
degrees. Passing to storage coordinates
`(B_q^(1/2)*u, sqrt(beta)*K_q^(1/2)*v)` gives the similar matrix

\[
\widetilde J_q=\begin{pmatrix}-\mathsf D&-\mathsf C\\
\mathsf C^T&0\end{pmatrix},\quad
\mathsf D=eB_q^{1/2}A_qB_q^{1/2}>0,\quad
\mathsf C=\frac w{\sqrt\beta}B_q^{1/2}M_qK_q^{1/2}.
\]

The coupling matrix `C` is invertible. For a complex eigenvector `(a,b)`,
the real part of its work identity is
`Re(lambda)*(||a||^2+||b||^2)=-a^* D a`. Thus no eigenvalue has positive
real part. Equality would force `a=0`; the first block would then give
`C*b=0`, hence `b=0`, contradicting an eigenvector. All quotient eigenvalues
have strictly negative real part.

**Conditional local theorem.** Smoothness of the admitted field and this
Hurwitz Jacobian imply local exponential recovery of `M_*`: sufficiently
small joint form/phase perturbations remain acute and their deviations
modulo the two common offsets decay exponentially. A pure phase deformation
can initially have zero storage loss, but it induces form contrast through
native pressure; exchange makes that deformation accessible to form
dissipation. No added phase damping, telemetry selector or restorative
operator is needed under this specified law.

### An explicit local domain and the offset limits

An energy barrier supplies a sufficient continuous admission neighborhood.
Let `m=min_edges(pi/2-abs(delta_*))`, choose `0<r<m/sqrt(2)`, and define

\[
c_r=\min_{\{i,j\}\in E}\cos(|\delta_{*,ij}|+\sqrt2r)>0,
\quad k_r=\frac{\lambda_2(B)}2\min(1,\beta c_r).
\]

Within `||z||<=r`, `z=(u,v)`, every interpolated edge gap stays acute because
`abs((Rv)_j-(Rv)_i)<=sqrt(2)*||v||`. Taylor's formula about the critical
phase and the graph spectral gap give

\[
\mathcal E_{rel}=E_D(x)+\beta[V_\phi(\theta)-V_\phi(\theta_*)]
\ \ge k_r\|z\|^2,\qquad
\dot{\mathcal E}_{rel}=-e(Bx)^TND^{-1}Bx\le0.
\]

If `||z(0)||<r` and `E_rel(0)<k_r*r^2`, the trajectory cannot first exit
this ball. The quotient stays in a compact interior sublevel, its metric
remains nonsingular, and the smooth law continues for all future time.
The largest invariant set with zero loss has `Bx=0`. To retain that equality,
`N*g` must be a common vector; `1^T H*g=0` and positive `H,N` force it to be
zero. Sector uniqueness then gives precisely the reference phase shape.
LaSalle's argument consequently gives convergence in this sufficient basin;
the local Hurwitz result supplies exponential convergence near its limit.
These statements do not give a graph-independent decay rate or an error
bound for a numerical integrator.

The common form and phase velocities are smooth functions of the quotient
state and vanish at its equilibrium. Exponential quotient convergence makes
both velocities integrable, so the actual offsets tend to finite limits
`c_infinity, alpha_infinity`. They need not equal their initial values: the
native Arg pressure does not generally conserve a form mean, and the phase
row does not generally conserve a phase mean. The theorem recovers the
prepared geometry modulo those symmetries, not a prescribed absolute offset.
The limiting raw storage is `beta*V_phi(theta_*)`, which is positive for a
nonzero twist; only the local excess storage tends to zero.
Unlike section 3's low-total-storage condition, this local excess-storage
barrier can surround a winding equilibrium whose total storage exceeds beta.

### Cycle rates, competing laws and excluded boundaries

For a unit `C_n` with common capacity `nu>0` and uniform twist
`kappa=2*pi*ell/n`, `K_*=cos(kappa)*B` and `H_*=pi*D*cos(kappa)`. Each
nonzero random-walk Laplacian mode `lambda` satisfies

\[
\boxed{\quad s^2+e\nu\lambda s+
\frac{\nu^2w^2\lambda^2}{\beta\pi^2\cos\kappa}=0.\quad}
\]

At phase consensus this reproduces section 5. With `e=w=1/2`,
`beta=nu=1`, winding-one C5 has damped oscillatory linear modes, whereas
winding-one C6 has two negative real roots per mode. Those are predictions
of this joint law, not the relaxation rates of the earlier supplied sine
law and not a claim about particles or a physical clock.

For positive heterogeneous capacities on the same cycle, replace `nu*lambda`
in this polynomial by an eigenvalue `mu` of
`T=(R^T N R)*B_q/2`. This matrix is similar to a symmetric positive definite
matrix, and `mu_min>=nu_min*lambda_2(B)/2`. All modes still have the same
damped/critical/oscillatory class. For winding-one C5 with `e=w=1/2`,
`beta=1` and `nu_min=3/4`, their real parts satisfy
`Re(s)<=-3*lambda_2(B)/32`. This linear spectral bound does not remove the
nonlinear neighborhood restriction or supply an Euler error estimate.

The section 12 countermodel has the same equilibria and linearization here.
Indeed its skew correction is `Z=S*grad(V_phi)`, with `S=0` at uniform form
and `grad(V_phi)=0` at the phase equilibrium; locally
`Z=O(||u||*||v||)`. Its ideal storage balance is unchanged. Thus it too has
the stated local recovery: stability alone cannot select the independent
capacity premise. The finite capacity-intervention discriminator remains
necessary to separate their nonlinear responses.

The assumptions have concrete limits. At `e=0` local excess storage is
conserved, so a nonzero quotient perturbation cannot converge to the reference
equilibrium family; the linearized nonzero modes are purely imaginary. If
`w=0`, phase is frozen and arbitrary phase damage cannot recover (that
boundary is outside the admitted production model). Zero capacities require
new observability conditions: for example, two inactive nodes with distinct
frozen forms prevent convergence to common form, however small their
initial difference. No conclusion here covers changing capacity/support,
forcing, operator events, nonacute equilibria, global attraction across
winding sectors or indefinite stability of finite Euler execution.

### Bounded C5 recovery control and first retained evaluation

The [shared-engine instrument](../../benchmarks/relational_local_recovery.py)
uses one supplied winding-one C5 with `e=w=1/2`, `beta=1` and held capacities
`(1,5/4,3/4,3/2,1)`. At `epsilon=2^-8`, prepare

\[
x(0)=\epsilon(1,-1,0,0,0),\qquad
\theta_i(0)=2\pi i/5+\epsilon(0,1,-1,0,0)_i.
\]

The ideal preparation lies in the sufficient continuous basin *before any
trajectory is computed*. Take `r=pi/(20*sqrt(2))` in the preceding barrier.
Then `c_r=sin(pi/20)>1/10`, `lambda_2(B)=(5-sqrt(5))/2>1`, and
`k_r*r^2>9/16000`. The initial quotient norm is `2*epsilon=1/128<r`.
Using the unweighted graph Hessian as an upper bound on the cosine Hessian
along the initial perturbation gives
`E_rel(0)<=6*epsilon^2=3/32768<9/16000`. This is an ideal continuous
admission argument, not a floating-point trajectory enclosure.

The [prediction](../../docs/assets/relational_local_recovery/result.prediction.json)
was frozen before the two positive-capacity executions. Its finite hypotheses
are: at structural time `T=32`, joint quotient distance is at most half its
initial value on both grids; endpoint grid separation is at most `1/64` of
the initial distance; the acute margin stays at least `pi/20`; actual work
residual stays at most `1e-12`; and support, capacities, pressure path and
winding one are retained. Grids use `dt=1/64` and `1/128`. These tolerances
are numerical acceptance policies, not new dynamics or rigorous ODE error
bounds. The common form and phase offsets are retained separately.

The [response](../../docs/assets/relational_local_recovery/result.json), first
evaluated on 2026-09-27, passed every frozen criterion without retuning:

| Euler steps | Initial joint distance | Final joint distance | Final / initial |
| --- | --- | --- | --- |
| 2048 | `0.0078125` | `3.8753314601e-5` | `0.0049604243` |
| 4096 | `0.0078125` | `3.9089447971e-5` | `0.0050034493` |

The two endpoints differ by `3.9732190974e-7` in the quotient, approximately
`5.086e-5` of the initial perturbation. Both traces retained winding one and
minimum acute margin `0.3102530154` radians. Maximum observed actual work
residual was about `1.008e-18`. No positive storage increment was observed;
nonzero Euler storage defects remain in the record and are distinct from
instantaneous work residuals.

On the fine grid the final form mean is `0.00016167095`, and the common
phase-offset change is approximately `-0.00042482317`. The relative geometry
recovers while these common offsets move, as the theorem permits. Neither
the final distance nor these finite offsets are asserted to be exact limits.
The same perturbed state with all capacities zero stays exactly frozen in
one shared engine step despite nonzero pressure; the declared clock advances.
That control distinguishes inactive retention from restorative dynamics.

The [controls](../../tests/physics/test_relational_local_recovery.py) compare
the production tangent with independently derived cycle blocks, check common
offsets and the zero-capacity boundary, and reuse the saved checkpoints and
decisions. They do not regenerate the full two-grid campaign. The artifact
contains four checkpoints per trace and summaries over every executed step;
saved snapshots alone do not independently authenticate omitted steps.

**Disposition.** The named F3 local-recovery obligation is complete for this
conditional family: a proof covers positive held capacities on the stated
acute equilibria, and the finite control exercises the existing integrated
law on a nonuniform phase geometry. It adds no solver, stabilizing controller
or default policy. Supplied support and winding, formation of that sector,
interaction with other regions and physical identification remain distinct.

## 16. Interaction of two recoverable regions under the same law

<a id="relational-region-interaction"></a>

### Joined support changes the shared geometry, not the evolution rule

Prepare two oriented unit cycles `A=(0,1,2,3,4)` and `R=(5,6,7,8,9)` with
the fixed bridge `(0,5)`, all capacities one and the same `e,w,beta` as
above. For `k=0,...,4`, let `theta_*[k]=theta_*[k+5]=k*kappa`, where
`kappa=2*pi/5`. The bridge endpoints agree. Existing acute circulation
classification requires zero phase gap on a bridge at equilibrium; the
two ring circulations balance independently. Uniform form and these phases
are therefore an equilibrium of the complete joined law.

Only the geometry and balance accounting of the historical
[contact study](../COHERENT_PATTERN_CONTACT.md) are reused. Its supplied
sine-law trajectories and maximum principle do not apply to this joint law.
There is no support event during the experiment, imposed interaction force,
phase driver, selector or retuned pressure. Joining is a supplied preparation.

The joined graph has degrees three at ports 0 and 5, and two elsewhere.
Writing `c=cos(kappa)` and `h=1+2*c=(1+sqrt(5))/2`, its equilibrium phase
metric is `H_ports=pi*h`, `H_other=2*pi*c`. Both native diffusion normalization
and the phase metric must be evaluated on the joined support. An isolated
ring's row cannot be copied to a port after joining.

The disconnected reference admits two independent form offsets and two
independent phase rotations. The connected graph admits only one of each:
relative regional offsets are now part of the dynamical state. Section 15
supplies local recovery of this joined equilibrium modulo the remaining
global symmetries. Prepared regional winding is retained in that local
domain; neither the bridge nor the winding sectors are generated by the law.

### A transmitted response that changes internal geometry

Perturb only an internal left node: `x(0)=epsilon*e_1`, with both phases
unchanged. The receiver's complete form and phase velocities initially vanish.
At the donor port,

\[
\dot x_0(0)=e\epsilon/3,\qquad
\dot\theta_0(0)=-w\epsilon/(\beta\pi h).
\]

At the receiving port `q_5=(Bx)_5=0`, so differentiating `H_5^-1` supplies
no initial term in its phase acceleration. The native pressure derivative
uses `partial_0 g_5=1/(pi*h)` at the prepared lock. Consequently

\[
\boxed{\quad
\ddot x_5(0)=\epsilon\left(\frac{e^2}{9}
                      -\frac{w^2}{\beta\pi^2h^2}\right),\qquad
\ddot\theta_5(0)=-\frac{ew\epsilon}{3\beta\pi h}.
\quad}
\]

All other receiver accelerations vanish initially. The form response contains
competing diffusion and phase-mediated contributions, both from the existing
law. Its sign is not universal over storage scales: for `e=w=1/2`, `beta=1`
the displayed form coefficient is positive and the phase coefficient negative.

On consistent local lifts define the receiver's port-versus-rest contrasts

\[
\chi_x=x_5-\tfrac14\sum_{i=6}^9x_i,\qquad
\chi_\theta=(\theta_5-\theta_{*,5})
             -\tfrac14\sum_{i=6}^9(\theta_i-\theta_{*,i}).
\]

These remove a common regional shift and have the same initial accelerations.
They therefore detect internal deformation, not merely a translated regional
mean. In the ideal disconnected comparison the recipient remains at its
equilibrium for all time. Let `D_chi` be joined recipient minus separately
evolved disconnected recipient; its leading coefficients are
`D_chi(T)=T^2*chi''(0)/2+O(T^3)`. The order is a smooth local response order,
not a finite signal delay or physical propagation speed. Floating-point
drift of the disconnected preparation is measured separately rather than
assigned the ideal zero value.

The ideal preparation with `epsilon=1/256` lies within the sufficient
continuous basin before any simulation. The joined graph has ten nodes and
diameter five. For zero-mean `v`, summing pathwise Cauchy inequalities gives
`10*||v||^2=sum_(i<j)(v_i-v_j)^2<=45*5*v^T Bv`, hence
`lambda_2(B)>=2/45`. With `r=pi/(20*sqrt(2))` and `c_r>1/10`, section 15's
barrier exceeds `pi^2/360000>1/40000`. The preparation has
`E_rel(0)=epsilon^2=1/65536` and quotient norm
`epsilon*sqrt(9/10)<r`. This proves continuous admission and eventual local
recovery, not the finite Taylor band or an Euler error enclosure.

### Regional accounting and the information that cannot be discarded

For the fixed receiver, the form total weighted by full-graph degrees has
the exact balance

\[
M_R=3x_5+2\sum_{i=6}^9x_i,\qquad
\dot M_R=e(x_0-x_5)+w\sum_{i\in R}d_i g_i.
\]

The phase row likewise implies

\[
\sum_{i\in R}H_i(\theta)\dot\theta_i
       =\frac w\beta(x_5-x_0).
\]

The latter is an instantaneous weighted rate, **not** the derivative of a
weighted phase total: `H` changes with phase. The former is an EPI accounting
identity, not conservation of physical mass. It does not discard the internal
phase source or imply that the ordinary arithmetic mean obeys a pure cut-flux
law. The existing
[regional support observer](../../src/tnfr/physics/support_transport.py)
owns the represented form-total and variance decomposition. Its forcing
argument is supplied as `F=w*g` directly from the phase state, independently
of the form rate; native pressure split defects remain separate.

#### Shared signed-work and regional-rate integration

<a id="relational-work-integration"></a>

The engine now retains the exact represented graph gradient `q=Bx` in
`field.work`, alongside the older floating gradient. Using represented
capacity, phase source, phase gradient and materialized model rates, define

\[
D_i=e\nu_iq_i^2/d_i,\qquad J_i=w\nu_iq_i g_i,\qquad
F_i=q_i\dot x_i,\qquad P_i=\beta(\nabla V)_i\dot\theta_i.
\]

In exact real arithmetic the admitted law has `F_i=-D_i+J_i` and `P_i=-J_i`.
In the represented engine these are independently retained defects:

\[
\delta F_i=F_i+D_i-J_i
=q_i(\nu_i\,\epsilon_i+\eta_i),\qquad
\delta P_i=P_i+J_i,
\]

where `epsilon=pressure_split_residual` and
`eta=nodal_rate_rounding_defect`. Thus
`sum_i(delta F_i+delta P_i)=field.balance_residual` exactly. Positive `J_i`
transfers phase storage toward form storage; `D_i>=0` is dissipation.
`P_i` uses the retained trigonometric gradient and represented rate, so its
defect includes their failure to satisfy the ideal metric/source identity.
Exact rational arithmetic here does not certify the trigonometric evaluation
error. The existing global loss and work now sum these same nodal arrays;
neither rates, pressure nor constitutive premises change.

These are nodal contributions to the derivative of **total** storage, not
derivatives of independently assigned node energies. Regional sums retain
that distinction. They sum globally for a partition; overlapping regions
count shared nodes repeatedly. Uniform form has `q=0`, hence zero immediate
work and exchange, even if `g!=0` generates nonzero form rates. Zero work
therefore does not imply equilibrium or absence of a subsequent response.

For arbitrary positive capacities on a supplied region, the two ideal
weighted balances generalize the unit-capacity equations above:

\[
Q_R=\sum_{\substack{i\in R\\j\notin R}}A_{ij}(x_i-x_j)
=\sum_{i\in R}q_i,
\qquad
\sum_{i\in R}\frac{d_i}{\nu_i}\dot x_i
=-eQ_R+w\sum_{i\in R}d_i g_i,
\]
\[
\sum_{i\in R}\frac{H_i}{\nu_i}\dot\theta_i=\frac w\beta Q_R.
\]

The shared support owner computes the outward cut once for both balances.
The actual represented form rate includes `sum_R d_i*epsilon_i` and
`sum_R (d_i/nu_i)*eta_i`; subtracting them gives an exact zero accounting
residual. The actual weighted phase rate retains its difference from
`(w/beta)*Q_R`. Its weights use the captured `H`, so it remains an
instantaneous rate, not the derivative of a weighted phase total.
For the full support the cut is empty. Zero diffusion weight removes the
form boundary term without removing phase exchange. A zero capacity inside
the region makes divided rates unavailable; all nodal work and undivided
cut/model terms remain defined. A zero capacity outside the selected region
does not invalidate its divided rates. These local requirements are separate
from the older regional variance observer's positive-full-capacity contract.

[`observe_relational_pattern`](../../src/tnfr/physics/relational_observations.py)
captures one fresh field, reuses this cut owner and exposes regional `work`
and `boundary` through `Network.relational_pattern` and exact JSON projection.
No diagnostic chooses an operator, evolves a graph, or closes regional
dynamics. The [API contract](../../docs/API_CONTRACTS.md#relational-pattern-observation)
owns field names and availability; static analytic, sign-reversal, exact
gradient and retained interaction checks exercise the shared implementation
without rerunning a trajectory. This integration adds explanatory accounting,
not new evidence for physical identity or another constitutive law.

A concrete obstruction prevents replacing each ring by only its mean and
storage. Move the same impulse from node 1 to node 2. The donor mean,
form norm, Dirichlet storage, phase geometry, capacities and port form are
unchanged. Both receiver accelerations are now zero, because the donor port
has zero initial velocity in both rows. Thus identical aggregate observations
can have different transmitted responses. A reduced interaction description
must retain sufficient internal/port information or justify an appropriate
memory term; this example does not prove that every reduction is impossible.
The two donor mean derivatives already differ (`-e*epsilon/30` versus zero),
so a description that retains those rates has additional information. The
obstruction concerns the stated instantaneous means/storage, not every
augmented observation or a sufficiently resolved boundary history.

<a id="relational-local-composition"></a>

The [local composition theorem](RELATIONAL_PATTERN_COMPOSITION.md) now fixes
six regional mean/port observations on this same single-bridge equilibrium.
They require ten coordinates for exact first-variation closure; an explicit
mixed form/phase counterexample excludes nonlinear closure of that minimal
linear observation. The theorem owns its symmetry, rank and coefficient
scope; it does not replace the full nonlinear engine state.
Its [nonlinear regional budget](RELATIONAL_PATTERN_COMPOSITION.md#regional-phase-mobility-balance)
identifies the missing mobility/form covariance: even the same zero outward
cut can coexist with different mean phase response. The shared observer
retains that contribution, its squared bound and actual rate-rounding defect.

### Prospective finite control

The [instrument](../../benchmarks/relational_region_interaction.py) keeps
`epsilon=1/256`, `e=w=1/2`, `beta=1`, and structural horizon `T=1/16`.
It compares the joined graph with both disconnected cycles executed separately
through the same connected-graph engine owner, at 64, 128 and 256 steps.
The prior criteria require each signed contrast divided by its quadratic
prediction to lie in `[3/4,5/4]`; successive signal differences must contract
by at least the declared factor `3/4` plus `1e-12`, and each finest signal
must exceed `32*(last_grid_difference+1e-12)`. Winding one, an acute margin
of at least `pi/20`, held support/capacities and work residual at most `1e-12`
are separate checks. Disconnected receiver drift is bounded by the separately
declared `1e-12` numerical gate.

These are prospective numerical hypotheses, not a proved Taylor remainder,
an autonomous regional closure or empirical physical evidence. The complete
prediction must be frozen before finite-horizon campaign execution; a failed criterion
remains a failed result. No prior recovery or capacity-response artifact is
changed by this control.

### First retained transmission evaluation: 2026-09-27

The [frozen prediction](../../docs/assets/relational_region_interaction/result.prediction.json)
preceded every finite-horizon campaign trace. Detached field/Jacobian checks
and a one-step Euler implementation control preceded that freeze; none
evaluated the reserved finite-horizon response. The
[retained response](../../docs/assets/relational_region_interaction/result.json)
contains nine traces (three scenarios on three grids), with initial, midpoint
and final checkpoints, runtime/source provenance and numerical decisions.
All five gates passed on the first evaluation, without retuning:

| Finest-grid contrast, joined recipient minus disconnected recipient | Observed response | Ratio to quadratic prediction | Last grid difference |
| --- | --- | --- | --- |
| Form | `1.3432193095e-7` | `0.9725660824` | `5.0174490386e-10` |
| Relative phase error | `-1.2133275087e-7` | `0.9700781649` | `4.5079212794e-10` |

The successive differences approximately halved. Every trace retained its
declared support, unit capacities and winding; minimum acute margin was
`0.3139754258` radians and maximum actual work residual was `6.661e-19`.
The largest disconnected-recipient full-state drift was `2.769e-18`; it
was subtracted as observed rather than assigned zero. These finite checks
are neither a full continuous error enclosure nor independent validation
of every omitted step from the retained checkpoints alone.

At the fine-grid endpoint, the receiver's degree-weighted form-total rate
decomposes into positive cut contribution `1.9770232650e-5`, negative phase
source contribution `-6.8497774475e-6` and native pressure-split defect about
`4.203e-17`. The existing regional observer checks the exact bookkeeping on
represented values. This is a concrete balance of the two transmission
channels, not a physical-mass measurement or a new regional evolution law.
The ordinary form mean (`2.7112836143e-8`) and full-graph transport-weighted
mean (`3.6881703848e-8`) differ and retain their distinct meanings.

The [controls](../../tests/physics/test_relational_region_interaction.py)
independently check initial accelerations, the equal-summary obstruction,
regional accounting, first-step Euler ordering and frozen-protocol admission.
They bind saved preparations and capacities, replay 27 detached checkpoint
fields and reconstruct the finite contrasts without repeating the trajectories.

**Disposition and next boundary.** The named prepared-interaction obligation
is complete in its stated mathematical/numerical scope. Recovery and
transmission now use one unchanged conditional law. The next foundational
question is formation/selection: acute continuous fixed-support execution
preserves cycle winding, so this experiment cannot create its supplied
winding sectors. A regular-domain and energy-admission study must distinguish
that executor restriction from genuine constitutive singularities before
another formation experiment is justified. The sole queue owns that next task.

## 17. Regular-domain admission before a formation experiment

<a id="relational-formation-domain"></a>
<a id="relational-regular-domain-admission"></a>

### Four boundaries with different meanings

For the full continuous law of sections 1-3, write the local relative
resultant and displacement as

\[
z_i=\sum_{j\sim i}e^{i(\theta_j-\theta_i)},\qquad
\alpha_i=\operatorname{Arg}z_i\in(-\pi,\pi),\qquad
g_i=\alpha_i/\pi,\quad
H_i=\pi|z_i|\operatorname{sinc}\alpha_i.
\]

The open regular domain is `z_i` outside the nonpositive real axis for every
node. Here `H_i>0` and the field is smooth. Its definition uses only the
existing state and support; it supplies no wider numerical execution contract.

| Boundary | Consequence for the stated continuous law |
| --- | --- |
| An ordinary edge reaches `abs(gap)=pi/2` | The acute executor stops; the resultant and complete field can remain regular |
| An ordinary edge reaches antipodal separation | Its ordinary cycle winding is undefined at that instant; the nodal field need not be singular |
| A local resultant vanishes | Its direction and native phase source are undefined; the phase metric tends to zero |
| A nonzero resultant lies on the negative real axis | The principal displacement has its `-pi/pi` branch; its two source limits differ and `sinc(alpha)` tends to zero |

At zero resultant, different admitted approaches can give different limits
for the native source. The phase velocity also generically diverges when
`(Bx)_i` stays nonzero while `H_i` vanishes. Even if that numerator vanishes,
the law does not select a value for the resulting ratio or a source direction
on an open full-state neighborhood. Likewise the negative-real branch does
not become regular merely because the resultant magnitude is positive.
A special limiting trajectory would require its own existence and uniqueness
argument; assigning a direction or clipping `H^-1` would add a new law.

### A derived cycle invariant gives a genuine pure-cycle obstruction

<a id="relational-cycle-resultant-obstruction"></a>

On a pure `C_n`, `n>=5`, every node has exactly its two cycle neighbors.
Consequently

\[
|z_i|=|e^{i\theta_{i-1}}+e^{i\theta_{i+1}}|=0
\quad\Longleftrightarrow\quad
\theta_{i+1}-\theta_{i-1}=\pi\pmod{2\pi}.
\]

Consider the derived oriented edges `i -> i+2` modulo n. They form one
cycle when n is odd, and two parity cycles when n is even. Their non-antipodal
winding integers are observations of the same phases, not added support or
new primitive variables. Every such edge has a non-antipodal gap whenever
the original local resultants are nonzero. The ordinary cycle-period argument
from [the existing classification](../FORCED_SUPPORT_BALANCE.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods)
therefore preserves these derived windings along any continuous path with
nonzero resultants, independently of the dynamics or the Arg-branch admission.

For an acute original phase geometry, let its oriented gaps be `delta_i` and
ordinary winding be ell. Each `delta_i+delta_(i+1)` is strictly inside
`(-pi,pi)`, so it is exactly the shortest two-hop gap. Summing gives

\[
\boxed{\quad
W_{\mathrm{skip}}=2\ell\quad(n\text{ odd}),\qquad
W_{\mathrm{even}}=W_{\mathrm{odd}}=\ell\quad(n\text{ even}).
\quad}
\]

For odd n the sum includes every original edge twice; each parity sum for
even n includes every original edge once. In particular, C5 consensus has
derived winding zero, while its acute winding-one twist has derived winding
two on `(0,2,4,1,3)`. **Every continuous fixed-C5 path between them must
encounter a zero local resultant.** Allowing obtuse ordinary edges or removing
the acute executor cutoff cannot remove this constitutive obstruction. The
same reasoning separates different acute winding sectors on any pure cycle.
It does not apply unchanged when extra neighbors contribute to a resultant.

Ordinary winding alone does not identify these regular components. For
example, on C5 take

\[
\theta=(0,\pi,3\pi/4,\pi/2,\pi/4).
\]

The edge `(0,1)` is antipodal, but the relative resultants are
`(-1+exp(i*pi/4), -1+exp(-i*pi/4), sqrt(2), sqrt(2), sqrt(2))`.
Every one is regular. Replacing `theta_1` by `pi+tau` for sufficiently small
positive/negative tau keeps all resultants regular while ordinary winding
changes from zero to minus one. The derived winding remains zero on both
sides. This is a regular geometric crossing, not yet an assertion that the
prescribed path is a trajectory, nor a connection to the acute minus-one
twist (whose derived winding would be minus two).

### A sharp C5 zero-resultant storage barrier

Suppose a C5 resultant vanishes. Its two neighbors are antipodal, so the
phase-storage contributions of the two edges incident at that node sum to
exactly two, independently of the node's own phase. The remaining three-edge
path connects antipodal endpoints. Fix them at `0,pi`, with internal phases
`A,B`. The sum of its three cosines is

\[
\cos A+\cos(B-A)-\cos B
\le \cos A+2|\sin(A/2)|
=\tfrac32-2(|\sin(A/2)|-\tfrac12)^2\le\tfrac32.
\]

Thus that path costs at least `3/2`, attained at `A=pi/3,B=2*pi/3`, and

\[
\boxed{\qquad V_\phi\ge\tfrac72
\quad\text{at every zero-resultant C5 configuration}.\qquad}
\]

The bound is sharp: choose those three equal path gaps; the remaining node
may have any phase. Nonnegative Dirichlet storage and the complete loss
identity imply the necessary budget `E(0)>=7*beta/2` for a trajectory to
reach such a configuration. With positive capacities and `e>0`, any
nonstationary phase evolution consumes positive integrated form loss, so a
trajectory reaching this barrier after that evolution needs strictly more
initial storage. Equality cannot pay both the barrier and that loss.

For initial phase consensus, the entire budget is initial form storage.
The final acute winding-one phase storage is only
`5*(1-cos(2*pi/5))`, approximately `3.454915`, below the crossing barrier
`3.5`. Comparing only initial and target energies therefore misses a necessary
intermediate cost. Even sufficient stored energy would not define continuation
through the singularity or establish a connecting solution of the current law.
The regular ordinary-antipodal example above has
`V_phi=6-2*sqrt(2)<7/2`; that lower cost does not contradict the distinct
zero-resultant obstruction.

### Added support gives a regular route to an admitted target

<a id="relational-regular-winding-crossing"></a>

The pure-cycle obstruction is not a universal restriction on the full law.
Prepare two unit cycles `(0,1,2,3,4)` and `(5,6,7,8,9)`, with bridges
`(0,5)` and `(1,6)`. The second bridge is an explicit support change relative
to section 16's one-bridge preparation; it is not generated by an operator
or hidden in a pressure coefficient. On both rings choose the phase path

\[
\theta_{0:4}(\delta)=\theta_{5:9}(\delta)
 =(0,-4\delta,-3\delta,-2\delta,-\delta),
\qquad 0\le\delta\le2\pi/5.
\]

Each bridge joins equal phases. Interior degree-two nodes have
`z_i=2*cos(delta)>0`; ports 0 and 5 have

\[
z_0=z_5=1+e^{-i\delta}+e^{-4i\delta},\qquad
\operatorname{Re}z_0=1+\cos\delta+\cos4\delta
\ge\cos\delta\ge\cos(2\pi/5)>0.
\]

Ports 1 and 6 have the conjugate resultant. This is a compact regular path
with a positive lower bound on every metric, from consensus to acute
winding-one geometry in both rings. The additional four-cycle has winding
zero wherever its ordinary winding is defined; it also encounters antipodal
edges at `delta=pi/4`. At `delta=pi/3`, the two ring-neighbor phasors cancel at each port,
but the aligned bridge neighbor contributes one. That is precisely a
resultant-zero encounter on the isolated C5 which the supplied extra support
prevents; no direction is assigned to an undefined resultant.

Each ring's oriented gaps are `wrap(-4*delta),delta,delta,delta,delta`.
Its winding changes from zero to one when delta crosses `pi/4`. At the
endpoint `delta=2*pi/5`, all ring gaps are acute and uniform, bridge gaps
are zero, and native phase pressure vanishes. Uniform form there supplies
an equilibrium covered by section 15's local-recovery theorem. This proves
regular geometric connectivity between the preparations. It does not yet
show that their connecting path obeys the joint equations.

The phase storage along this specified path is
`V_phi(delta)=2*[1-cos(4*delta)+4*(1-cos(delta))]`; bridge contributions
vanish. At `delta=pi/3`, all resultants equal one and `g=0`, but the
phase geometry is nonacute. Here `V_phi=7` and the second derivative along
the path is `-12`, while changing only an interior node has positive phase
curvature. It is a saddle of phase storage. The final acute target has
`V_phi=10*(1-cos(2*pi/5))<7`. This locates a barrier on this particular
regular path, not a proven minimum barrier over all paths or a connecting
orbit of the joint ODE.

### The same coupled law realizes a transversal crossing

A geometric path alone would be insufficient. Here the same coupled law can
realize a transversal crossing. At any regular phase state with positive
capacities, let `M=N*H^-1` and prescribe an instantaneous relative phase
velocity represented by v. Define

\[
\gamma=-\frac{\mathbf1^TM^{-1}v}{\mathbf1^TM^{-1}\mathbf1},\qquad
Bx=\frac\beta w M^{-1}(v+\gamma\mathbf1).
\]

The right side has zero sum. Connectedness of the graph supplies a finite
real x, unique modulo a common offset. The actual phase row then satisfies
`theta_dot=v+gamma*1`; the form row remains the unchanged native nodal law.
This realizes any instantaneous relative phase direction through supplied
form. It is not arbitrary control of a whole phase path, because subsequent
form and phase must both follow the joint equations.

The two-ring state gives a concrete witness without a fitted tangent. At
`delta=pi/4`, unit capacities, `beta=1`, `e=w=1/2`, take identical form
on both rings:

\[
x_{0:4}=x_{5:9}
=\frac{2\sqrt2}{5}(\pi+8,-\pi-8,-3\pi-4,0,3\pi+4).
\]

The full-graph `Bx` on each ring is
`(8*sqrt(2),-8*sqrt(2),-2*pi*sqrt(2),0,2*pi*sqrt(2))`.
Bridge differences vanish. Port metrics are `2*sqrt(2)` and interior
metrics are `pi*sqrt(2)`, so the **actual** phase row is

\[
\dot\theta_{0:4}=\dot\theta_{5:9}=(2,-2,-1,0,1).
\]

This is the path tangent `(0,-4,-3,-2,-1)` plus the permitted representative
offset `gamma=2`, obtained from the phase row, not externally imposed.
Smooth local ODE existence and uniqueness give a solution on both time
sides with `theta_1(t)-theta_0(t)=-pi-4*t+O(t^2)` and the same crossing
on the second ring. The remaining ring edges stay non-antipodal locally.
Thus an **actual local solution** changes ordinary winding from zero to
one in both rings under the same law. No phase forcing, support event,
telemetry selector or new evolution rule is inserted.

This supplied form carries finite Dirichlet storage
`128*pi/5+48*pi^2/5+512/5`; it is not free phase actuation or evidence
that a small perturbation of consensus forms the target. Neither arrival
from consensus at the crossing state nor later entry into the acute
winding-one equilibrium's recovery basin has been proved. The regular path,
local crossing solution and admitted stable target are distinct facts;
concatenating them verbally would not establish one connecting trajectory.

### Admission verdict and execution boundary

The same mathematics yields a negative and a positive result: pure-cycle
acute sectors require an undefined resultant between them, whereas the
specified higher-degree support admits a regular geometric route to a
recoverable target and a local topological crossing under the complete law.
Neither a universal formation obstruction nor a complete autonomous
formation trajectory follows. The full regular domain is meaningful and
can exceed the engine's default acute domain, but global continuation
within it remains an independent obligation.

The default acute execution contract and every earlier frozen response remain
unchanged. Section 18 supplies an opt-in sufficient regular chamber for a
wider numerical control. A prescribed geometric route does not choose the
numerical trajectory or its outcome. No broader formation sweep, new state
primitive or unsupported singular continuation is justified by this local
existence result. The sole execution queue owns the next gate.

The [exact cycle observer](../../src/tnfr/physics/phase_resultant_sectors.py)
reuses the existing support geometry and rational scalar owner. It records
ordinary/derived winding and both resultant boundary types from declared exact
turns, leaving live radians and runtime admission separate. The
[admission controls](../../tests/physics/test_relational_formation_admission.py)
check those observations, the sharp C5 budget and the two-port phasor, tangent
and native-pressure identities. They contain no time integration or formation
parameter search. [API scope](../../docs/API_CONTRACTS.md#exact-cycle-resultant-sectors)
owns inputs and availability; the symbolic proof above owns the conditional
continuous conclusions.

## 18. Certified positive-resultant execution and a reserved crossing control

<a id="relational-positive-resultant-execution"></a>

### A sufficient regular chamber for the unchanged joint law

The production model now accepts explicit
`phase_domain="positive_resultant"`; `"acute"` remains the default. The new
option requires `Re(z_i)>0` at every node. This right-half-plane chamber is
strictly narrower than the full regular domain in section 17, which also
admits left-half-plane resultants away from the negative-real branch. It is
nevertheless broad enough for the two-port witness and its prescribed
geometric route. Support, capacities, storage scale and phase/form equations
remain the same. This extends execution admission, not constitutive selection.

The [shared rational helper](../../src/tnfr/mathematics/_phase_resultant_chamber.py)
reuses the existing Machin enclosure of mathematical pi. For each exact
rational difference of materialized phase coordinates it encloses cosine:
integer true-turn reduction, a rational midpoint Taylor enclosure, and a
Lipschitz allowance for the reduction interval. Outward dyadic rounding
retains a rational lower bound. Near zero an elementary quadratic/quartic
enclosure suffices. Bounded numerical work can return an inconclusive bound;
neither that outcome nor a nonpositive lower bound proves an actual singularity.
The ordinary edge's antipodal branch requires no special cosine continuation.

For the nodewise lower bounds and actual represented endpoint increments,
write

\[
L_i\le\sum_{j\sim i}\cos(\theta_j-\theta_i),\qquad
d_i=\theta_i^{\rm after}-\theta_i^{\rm before}.
\]

Cosine is globally one-Lipschitz. Every point of the straight Euler proposal
chord therefore satisfies

\[
\operatorname{Re}z_i(s)\ge L_i-\sum_{j\sim i}|d_j-d_i|,
\qquad 0\le s\le1.
\]

Strict positivity of this exact rational right side certifies that the
**whole represented phase chord** remains in the chamber. On that chord,
`H_i>=pi*Re(z_i)>0`: for `|delta_i|<pi/2`,
`sin(delta_i)/delta_i>=cos(delta_i)`. The engine additionally evaluates
materialized positive cosine sums and metrics and re-admits the endpoint.
The exact margins are retained as `resultant_real_lower_bounds` and
`segment_resultant_real_lower_bounds`. Rejection precedes live graph commit.

Using only endpoint admission would be insufficient: two regular P2
endpoints can differ by a full relative turn whose interior crosses the
excluded chart. Conversely, a common phase increment contributes zero to
the exact drift bound. The increments used here are the actual rounded
endpoint differences, not an uncommitted `dt*phase_rate` approximation.
The [execution controls](../../tests/test_relational_regular_execution.py)
exercise these distinctions, atomic rejection, zero capacity, the analytic
crossing tangent, and admission independent of libm cosine signs.

This certificate does not enclose the exact continuous ODE trajectory, bound
its global numerical error, or prove Euler storage decrease. Native pressure
and joint rates still use their existing materialized arithmetic; pressure
split, work and step defects remain measured. Raw relative angles are used
for trigonometric evaluation in the new chamber; the default acute path
retains its previous represented wrapping. Earlier frozen producers and
records are untouched. Their historical source fingerprints are not rewritten
to claim compatibility with this source revision.

### Fixed preparation and independently stated questions

<a id="relational-reserved-crossing-response"></a>

The [bounded producer](../../benchmarks/relational_formation_response.py)
uses the supplied two-ring, two-bridge support of section 17, unit capacities,
`beta=1`, and `e=w=1/2`. Each ring starts at
`theta=(0,-4*delta_0,-3*delta_0,-2*delta_0,-delta_0)`, with
`delta_0=pi/4-1/256`, and the explicit transverse form state derived there.
Thus both ordinary windings initially equal zero, but this is already a
structured, finite-storage preparation close to a crossing; it is not phase
consensus, a random perturbation or substrate birth.

Before evolution, the protocol fixes one horizon `T=16`, three Euler grids
`dt=(1/64,1/128,1/256)`, and common checkpoints. The short positive hypothesis
is winding `(1,1)` by `t=1/64`, retained at `t=1/32`. Later acute-domain
entrance, distance to the prepared acute winding-one equilibrium and a local
basin indicator are separate questions, without a predeclared positive
capture claim. The complete quotient state includes both rings together, so
relative regional offsets are not discarded.

For the basin indicator, the target edge margin is `m=pi/10`. Choose
`r=pi/(20*sqrt(2))`, giving `c_r=sin(pi/20)>1/10`. Every connected ten-node
unit graph has `lambda_2>=2/81>1/45`: summing the path bound
`(y_i-y_j)^2<=9*y^T*B*y` over all pairs gives this conservative estimate.
Section 15 then yields `k_r>1/900`, `r^2>9/800`, and
`k_r*r^2>1/80000`. In the declared beta-one quotient normalization,

\[
\|y\|^2<9/800,\qquad 0\le E_{\rm rel}<1/100000
\]

are sufficient **ideal-law** basin conditions. The retained finite-state
norm and excess storage use materialized phases and trigonometry relative to
the supplied target. Their threshold verdict is therefore a numerical
indicator, not a rigorous transcendental energy enclosure or an exact-ODE
capture certificate. Negative estimated excess is retained, not clipped.

Every grid stops at its first shared-owner rejection, retaining its last
admitted full state, attempted step and unchanged-graph check. It neither
invents a rejected proposal report nor retries with a smaller step or broader
chamber. Refinement compares only checkpoints reached by all grids. No
later form tuning, prescribed phase path or new pressure term is allowed.

### Retained response: crossing succeeded; final capture is unavailable

The [prediction](../../docs/assets/relational_formation_response/result.prediction.json)
was frozen before the single
[evaluation](../../docs/assets/relational_formation_response/result.json).
Its SHA-256 is
`4ed42b2a94138e5a5ed7318c39aa9fdb485234f38cbdcab97ab8a2da75959956`;
the response SHA-256 is
`651cf5bcdeec81d64c4289c56110d052ea23d48a0ecdd08af3c76e65ab258d25`.
The records retain the complete package/producer fingerprints, declared
binary64 runtime, exact arithmetic defects and all saved full states. No
random initialization, retry, altered pressure or subsequent recalibration
was used.

| Euler step | Completed steps | Last admitted time | First observed winding `(1,1)` | First observed acute edges | Last minimum real-resultant bound |
| --- | --- | --- | --- | --- | --- |
| `1/64` | 37 | `0.578125` | `0.015625` | `0.46875` | `0.00289508` |
| `1/128` | 73 | `0.5703125` | `0.0078125` | `0.46875` | `0.0278649` |
| `1/256` | 147 | `0.57421875` | `0.0078125` | `0.46875` | `0.0121522` |

Both rings start with winding zero, and **all three grids satisfy the frozen
winding-one prediction at both `1/64` and `1/32`**. The common checkpoint
state discrepancies decrease under refinement: at `t=1/4`, the combined
form/phase infinity differences are approximately `1.26349e-3` and
`6.27253e-4` for successive grids. This is finite refinement evidence, not a
validated bound against the exact ODE. The maximum measured work residual is
below `2.64e-14`; the declared `1e-10` numerical policy passes. Euler energy
step defects remain nonzero and recorded separately.

Each run subsequently enters the acute phase region but stops when the next
whole-chord positive-resultant bound cannot be certified. Every last state
is still regular, winding-one and acute. Rejection leaves its live graph
unchanged. The limiting rows are nodes 3 and 8, whose phase metrics approach
zero as their two neighbors approach antipodal separation. The endpoint
evidence and the symmetry calculation below locate the relevant boundary;
the stopped Euler runs do not prove that the exact ODE reaches it in finite
time or lacks every other continuation.

None reaches the declared horizon 16. Initial joint storage is about
`283.894`; last-state values are about `203.45`--`204.46`, while the target
phase storage is about `6.90983`. The final saved quotient squared norms are
about `276.25`--`277.64`, well outside the conservative `9/800` basin indicator.
Thus acute entrance alone is plainly insufficient for capture. Later common
checkpoints and final-horizon basin verdicts are **unavailable**, not failed
completed predictions or inferred equilibria.

The [retained-response controls](../../tests/physics/test_relational_formation_response.py)
reconstruct winding/availability/acute geometry from every saved phase state,
check the original preparation and time accounting, and replay detached
last-state observations. They do not rerun the trajectories. The result
establishes an executed, same-law change of geometric sector under supplied
initial structure. It does not establish autonomous formation and maintenance
of the target, spontaneous support creation, or physical particles.

### Reflection reduction explains the next mathematical obligation

<a id="relational-reflected-capture-boundary"></a>

The fixed preparation has more structure than two arbitrary five-node rings.
It is invariant under copying the rings and under simultaneous form-sign and
phase-reflection about a constant phase c. Each ring can be written

\[
x=(A,-A,-B,0,B),\qquad
\theta=c\mathbf1+(a,-a,-b,0,b).
\]

The graph automorphism exchanges ports 0/1, nodes 2/4 and the corresponding
second-ring nodes, while fixing nodes 3 and 8. Unit capacities and the
unforced joint law respect these symmetries on its regular domain. Uniqueness
therefore preserves this four-coordinate subspace of the **ideal** ODE; it
is not imposed on the numerical states to alter their evolution. In the
component containing the preparation, the central form and phase rows vanish
because `q_3=g_3=0`, so c remains constant.

Writing `q=3*A-B` and `r=2*B-A`, the nonredundant relative resultants are

\[
z_0=1+e^{-2ia}+e^{i(b-a)},\qquad
z_4=2\cos(a/2)e^{i(a/2-b)},\qquad z_3=2\cos b.
\]

The other port/interior rows are conjugates. For `|a|<pi`, `|b|<pi/2`
and the retained regular branch, define
`g_0=Arg(z_0)/pi`, `H_0=pi*|z_0|*sinc(Arg(z_0))`,
`g_4=(a/2-b)/pi`, `H_4=2*pi*cos(a/2)*sinc(a/2-b)`.
The unchanged complete joint equations reduce exactly to

\[
\begin{aligned}
\dot A&=-e\,q/3+w\,g_0,&
\dot B&=-e\,r/2+w\,g_4,\\
\dot a&=(w/\beta)\,q/H_0,&
\dot b&=(w/\beta)\,r/H_4.
\end{aligned}
\]

These are derived coordinates of the existing law, not a new pressure,
controller or independent reduced solver. In particular, positive r still
drives b upward after the edges become acute. The retained last states have
`r` near `7.5`, so their remaining form contrast has not disappeared merely
because their winding and acute status now match the target's labels.

The full two-ring storage in this subspace is

\[
\begin{aligned}
E_D&=6A^2-4AB+4B^2,\\
V(a,b)&=10-2\cos(2a)-4\cos(a-b)-4\cos b,\\
\dot E&=-e\left[\tfrac43(3A-B)^2+2(2B-A)^2\right]\le0,
\qquad E=E_D+\beta V.
\end{aligned}
\]

The invertible proof coordinates `A=(2*q+r)/5`, `B=(q+3*r)/5` give
`E_D=(4*q^2+4*q*r+6*r^2)/5` and diagonal loss
`-e*(4*q^2/3+2*r^2)`. They connect the existing first-exit proof to a
potential interval storage bound without projecting the runtime state.
The full central-resultant admission `2*cos(b)>0` remains necessary even
though its exact symmetric dynamical row vanishes.

At the observed central cancellation face,

\[
V(a,\pi/2)=7+4(\sin a-1/2)^2\ge7.
\]

This is the same `7/2` cycle barrier from section 17 applied to both reflected
rings, with zero bridge storage. An ideal trajectory with `E<7*beta` cannot
reach this face while the nonincrease law remains admitted. The acute target
`(a,b)=(4*pi/5,2*pi/5)` has `V=10*(1-cos(2*pi/5))<7`; the nonacute saddle
`(2*pi/3,pi/3)` has `V=7`. These facts connect storage loss to a potentially
protected region around the target. They do not prove that a winding-zero
preparation can enter that region, nor that `E<7*beta` alone identifies its
basin rather than another allowed region.

As `b` approaches `pi/2`, the full central resultant vanishes. The exactly
symmetric central row may look removable because its numerator is identically
zero, but arbitrary nearby full states do not share that cancellation. A
continuation of four formal coordinate equations through the face would not
by itself extend the full phase pressure and inverse-metric law. Beyond the
face, the central resultant is negative real: assigning it zero phase source
would change the native law, while either Arg-side limit breaks the imposed
central zero-form row. Conversely,
the finite guard stops before that face and is not a singular-hitting theorem.

Section 19 uses these reduced equations and the existing local recovery
theorem to prove protected capture and distinguish its possible limits. Entry
from winding zero still requires a crossing and loss budget. The supplied
high-storage response remains the first retained test; it is not retuned
after observing its stop.

Static [formation-admission controls](../../tests/physics/test_relational_formation_admission.py)
independently reconstruct the reduced Laplacian, resultants and storage from
the full support, check a nonacute regular field against the four equations,
and verify both signs of the sharp central-cancellation barrier. They add
neither a reduced execution path nor another temporal campaign.

## 19. Protected capture and conditional basin selection

<a id="relational-protected-capture"></a>

The reflection reduction supplies a global capture result within an explicit
region, without another integration. Retain the same two-ring, two-bridge
support, unit held capacities and fixed `e,w,beta>0`. Work in the exact ideal
reflection subspace of section 18. Arbitrary common form and phase offsets
may be added; both are constant here because the central rows vanish.

### A protected region around the winding-one target

Define the positive-target phase rectangle

\[
\mathcal R=(2\pi/3,\pi)\times(0,\pi/2).
\]

Every relative resultant has positive real part there. In particular,
`Re(z_0)=1+cos(2a)+cos(a-b)>cos(a)*(2*cos(a)+1)>0`;
`z_4=2*cos(a/2)*exp(i*(a/2-b))` has positive modulus and argument in
`(-pi/6,pi/2)`; and `z_3=2*cos(b)>0`. Thus the full joint law is smooth,
not merely its formal four-coordinate expression. The four boundary costs are

\[
\begin{aligned}
V(2\pi/3,b)&=11-4\cos(b-\pi/3)\ge7,\qquad V(\pi,b)=8,\\
V(a,0)&=9-4(\cos a+1/2)^2\ge8,\\
V(a,\pi/2)&=7+4(\sin a-1/2)^2\ge7.
\end{aligned}
\]

**Conditional capture theorem.** Any exact reflected initial state with
`(a,b) in R` and `E=E_D+beta*V<7*beta` has a global forward solution and
converges to `(a,b,A,B)=(4*pi/5,2*pi/5,0,0)`.

Indeed, `E_D=6*A^2-4*A*B+4*B^2` is positive definite, so a fixed sublevel
bounds form. Storage nonincrease and the boundary costs prevent a first exit
from the rectangle. The phase sublevel stays a positive distance from its
boundary; together these give a compact subset of the smooth full-law
domain. Continuation is therefore global. Zero loss forces `q=r=0`, hence
`A=B=0`. To remain in that set requires `g_0=g_4=0`. The latter gives
`b=a/2`, and the former reduces to

\[
\sin(2a)+\sin(a/2)=2\sin(5a/4)\cos(3a/4)=0.
\]

Its unique solution in this rectangle is `a=4*pi/5`. LaSalle's invariance
principle then gives convergence to that target; the local-recovery theorem
supplies an eventual exponential rate, without a graph-independent bound.
The limiting storage is `beta*10*(1-cos(2*pi/5))`, not zero.

Every phase point in `R` already has ordinary winding one on both rings:
the positively wrapped ring gaps are `2*pi-2*a`, `a-b`, `b`, `b`, `a-b`,
whose sum is `2*pi`. This theorem establishes capture of a supplied sector;
it does not establish its formation from winding zero. The strict storage
test concerns the ideal law and exact trigonometry, not a rounded energy
estimate or successful admission of an arbitrary Euler step.

Simultaneous sign reversal `(a,b,A,B) -> (-a,-b,-A,-B)` preserves the same
law and storage. Hence the mirror rectangle
`R_minus=(-pi,-2*pi/3) x (-pi/2,0)` has the identical `E<7*beta` capture
theorem, with target `(-4*pi/5,-2*pi/5,0,0)` and both windings equal to
minus one. These are distinct protected regions of the same completion,
not different operator policies or extra dynamical rules.

### The saddle has a target-directed unstable branch

<a id="relational-saddle-capture-route"></a>

At the nonacute regular equilibrium
`s=(2*pi/3,pi/3,0,0)`, both active metrics equal `pi`, storage equals
`7*beta`, and the phase-storage Hessian is `[[-2,-2],[-2,4]]`. Put

\[
K=\begin{pmatrix}3&-1\\-1&2\end{pmatrix},\quad
G=\begin{pmatrix}1/2&1/2\\1/2&-1\end{pmatrix},\quad
D=\operatorname{diag}(3,2),\quad \omega^2=\frac{w^2}{\beta\pi^2}.
\]

In phase/form coordinates `(u,p)=(delta(a,b),delta(A,B))`, the Jacobian is

\[
J_s=\begin{pmatrix}0&(w/(\beta\pi))K\\
(w/\pi)G&-eD^{-1}K\end{pmatrix}.
\]

Eliminating form gives the symmetric quadratic eigenvalue pencil
`P(lambda)=lambda^2*K^(-1)+e*lambda*D^(-1)-omega^2*G`.
At zero it has one negative and one positive eigenvalue; for real
`lambda>=0`, its derivative is positive definite and its large-lambda
limit is positive definite. Hence it has exactly one simple positive root.
Zero is excluded because `G` is invertible. For a nonreal root and nonzero
phase vector `u`, the imaginary part of `u^*P(lambda)u=0` gives

\[
\operatorname{Re}\lambda
=-\frac{e\,u^*D^{-1}u}{2u^*K^{-1}u}<0.
\]

Thus the equilibrium is hyperbolic with a one-dimensional unstable manifold
in this reflected subspace. Its unstable phase vector has nonzero first
component, since `P_22=3*lambda^2/5+e*lambda/2+omega^2>0`. Orient it by
`u_a>0`; its slope is

\[
\frac{u_b}{u_a}
=\frac{\omega^2/2-\lambda^2/5}
       {3\lambda^2/5+e\lambda/2+\omega^2}\in(0,1/2).
\]

For the lower bound, `P` is already positive definite when
`lambda^2=5*omega^2/2`, where its off-diagonal entry vanishes, so the
unstable root lies below that value. The displayed ratio then gives both
bounds. These are properties of the local linearization, not a simulated
connection or a finite-time error estimate.

Every nontrivial unstable-manifold orbit approaches the saddle as
`t -> -infinity`. Its strict energy deficit follows from the actual balance:

\[
E(t)=7\beta-e\int_{-\infty}^{t}
\left[\tfrac43q(s)^2+2r(s)^2\right]ds<7\beta.
\]

The integral converges by the local exponential unstable asymptotics and is
positive because the unstable form component is nonzero. The branch oriented
by increasing `a` enters `R`. The capture theorem therefore continues that
branch globally forward to the acute winding-one target: a conditional
same-law heteroclinic orbit from the supplied nonacute saddle.

### The opposite branch approaches consensus

The other local branch enters
`S=(-2*pi/3,2*pi/3) x (-pi/2,pi/2)` with the same strict storage deficit.
Here the boundary values are
`V(±2*pi/3,b)=11-4*cos(b∓pi/3)>=7` and
`V(a,±pi/2)=7+4*(sin(a)∓1/2)^2>=7`. Within a sublevel `V<7`,

\[
V\ge8-4\operatorname{Re}z_0,\qquad
V\ge8-4\operatorname{Re}z_4,
\]

so both real parts exceed `1/4`; `z_3=2*cos(b)>0` as well. Consequently this
sublevel is regular in the same positive-resultant chamber, despite `S`
itself not supplying that assertion at arbitrary energy. Compactness and
LaSalle apply again. Now `g_4=0` gives `b=a/2`, and the displayed critical
equation has only `a=0` inside `S`. Thus every exact reflected state in `S`
with `E<7*beta` converges to consensus, including the opposite unstable branch.
It starts with winding one near the saddle and eventually has winding zero
near consensus. This is a forward loss of winding, not a reversal of the
target-directed orbit. In particular it reaches `a<pi/2` forward in time.
The mirrored saddle has the corresponding two branches toward the negative
target and consensus. This gives three sufficient capture regions; it is not
a classification of every initial state or a proof that no other behavior
exists outside them.

The exact saddle itself never departs. These are two distinct trajectories
with the same backward limit, not pieces that can be concatenated into a
winding-zero-to-target solution of the autonomous law. Capture and consensus
selection here remain conditional on supplied support, constitutive premises
and the prepared state; no spontaneous substrate or physical identity follows.

The exact reflection restriction is sufficient, not an assertion that every
slightly asymmetric state fails. From each state in any of these three
protected regions, a finite compact regular segment reaches the corresponding
full-state local-recovery neighborhood. Continuous dependence therefore gives
an open neighborhood of that state in the full form/phase state, with support,
capacity and model held, whose trajectories also converge modulo common
offsets. This is an existence statement: no radius has been computed, and no
rounded preparation, retained earlier trajectory or numerical capture
indicator is thereby certified.

### Shared admission of the sufficient ideal-law basins

[`certify_relational_capture`](../../src/tnfr/physics/relational_capture.py)
and `Network.relational_capture(model, cycles=...)` reuse one fresh detached
field and the shared rational cosine/pi enclosures. They check the exact
support, copied/reflected represented state, unit capacities, positive
coefficients, rectangle and strict energy bound, without projecting the
state or integrating a second solver. The output retains all symmetry
defects, candidate rectangle margins, exact energy bounds and unavailable
reasons. A target sector is supplied only on complete admission, independently
of current winding. The certificate concerns the ideal continuous law from
that represented state; Euler trajectories retain their separate obligations.
The [API contract](../../docs/API_CONTRACTS.md#conditional-relational-capture)
owns implementation details, and [SDK usage](../../docs/CLI_AND_SDK.md#check-a-protected-relational-basin)
supplies a nonacute zero-form preparation.

Static [production controls](../../tests/test_relational_capture.py) exercise
both twist sectors, consensus from a winding-one state, strict energy equality,
exact symmetry, common offsets, unsupported domains and exact report export.
The [symbolic controls](../../tests/physics/test_relational_formation_admission.py)
derive boundary costs, resultant lower bounds, critical geometry and the
saddle pencil from the full support. No new temporal campaign is used to
establish these conclusions, and no old frozen outcome is rewritten.

The same owner also exposes `certify_relational_local_capture` and
`Network.relational_local_capture`. These apply section 15's full-state
local theorem on this support, with `beta=nu=1`, to a declared target sector.
They use all centered form/phase coordinates, exact affine-pi reference
errors and outward cosine energy bounds. The conservative conditions are
`norm_squared<9/800` and `excess_storage<1/100000` from section 17; exact
reflection is unnecessary. Their proof is independent of any observed
trajectory. If applied to a numerical endpoint, the certificate concerns
the ideal continuation restarted at that precise represented state. It does
not supply an error tube from the original initial-value problem.

### A distinct winding-zero preparation with a bounded crossing theorem

<a id="relational-upper-corner-preparation"></a>

The previous high-storage preparation is not the only way to drive the same
law. Retain unit capacities, `e=w=1/2`, `beta=1`, both supplied rings and both
bridges, and the exact reflection coordinates. A separate preparation is

\[
a_0=b_0=\pi/2-\varepsilon,\qquad
A_0=2/5,\quad B_0=1/5,\qquad \varepsilon=1/64.
\]

It has `q_0=1`, `r_0=0` and winding zero on each ring. All initial relative
resultants have positive real part:

\[
z_0=2-\cos(2\varepsilon)-i\sin(2\varepsilon),\quad
z_4=1+\sin\varepsilon-i\cos\varepsilon,\quad z_3=2\sin\varepsilon>0.
\]

The form storage is exactly `4/5`; the full storage is

\[
E_0=44/5-4\sin\varepsilon-4\sin^2\varepsilon,
\qquad 44/5-65/1024<E_0<44/5.
\]

This state lies above the protected `7*beta` threshold. It is not admitted
by the capture theorem simply because its storage is smaller than that of
the earlier frozen response.

Put `chi=atan(sin(2*epsilon)/(2-cos(2*epsilon)))`. Then
`0<chi<2*epsilon`, `g_0=-chi/pi`, `g_4=-1/4+epsilon/(2*pi)`,
`H_0=pi*sin(2*epsilon)/chi` and `H_4=2*pi*cos(epsilon)/a_0`.
The initial derivatives obey

\[
\begin{aligned}
\dot q_0&=-3/8-(6\chi+\varepsilon)/(4\pi)<0,\\
\dot r_0&=-1/12+(\varepsilon+\chi)/(2\pi)<-29/384<0,\\
\dot a_0&=1/(2H_0)>0,\qquad
\dot b_0=0,\qquad \ddot b_0=\dot r_0/(2H_4)<0.
\end{aligned}
\]

Thus the form imbalance initially moves a toward the winding boundary while
moving b away from the central-resultant cancellation face. As
`epsilon -> 0+`, the derivatives tend to
`qdot=-3/8`, `rdot=-1/12`, `adot=1/(2*pi)` and `bddot=-1/96`.
The limiting corner itself is singular and is not an admissible initial
state; the selected positive epsilon remains part of the preparation.

**Bounded crossing, not capture.** For this fixed ideal preparation, the
solution remains regular through `t=1/4` and both windings change from zero
to one by `t=7/48`. This follows without numerical integration. Use the box

\[
\begin{gathered}
\pi/2-1/64\le a\le\pi/2+1/16,\qquad
\pi/2-1/32\le b\le\pi/2-1/64,\\
3/4\le q\le1,\qquad -1/8\le r\le0.
\end{gathered}
\]

Here `0<=a-b<=3/32`, `abs(a+b-pi)<=3/64`, and
`Re(z_0),Re(z_4)>2039/2048` (the non-strict lower bound also suffices).
The elementary bounds `sin(t)<=t`, `cos(t)>=1-t^2/2` and `3<pi<22/7`
give

\[
|g_0|<1/60,\quad -1/4<g_4<-11/48,\quad
5/2<H_0<7/2,\quad H_4>5/2.
\]

The reduced equations therefore imply `-1/2<qdot<0`, `rdot>-1/6`, and,
on the face `r=0`, `rdot<-13/240`. Consequently, before any first exit,

\[
\begin{aligned}
q(t)&\ge1-t/2,& r(t)&\ge-t/6,\\
3/28&<\dot a(t)<1/5,&
b_0-t^2/60&\le b(t)\le b_0.
\end{aligned}
\]

For `0<=t<=1/4` these inequalities prevent exit through every face;
`r<0` for positive time prevents exit through the upper b face. The central
resultant stays at least `2*sin(1/64)>0`. Smooth continuation on this compact
box establishes the interval, and the lower bound on `adot` gives
`a(7/48)>pi/2`. Since a increases, while b remains positive and
`0<=a-b<pi`, the ordinary ring winding crosses once at the antipodal
edge `0--1` (and its copied edge). The full resultants remain regular at
that crossing. The winding itself is undefined at the crossing instant.

The signs and corridor motivate a separate finite prediction through the
shared executor. They are an ideal-law theorem, not an Euler-error bound or
a result for the binary representation of the supplied phases. In particular,
this argument does not show entry into `R`, enough subsequent storage loss
for `E<7`, or convergence to the winding-one target. The
[static controls](../../tests/physics/test_relational_formation_admission.py)
check the initial identities and the rational corridor estimates without
executing a trajectory.

### Frozen upper-corner response: regular sector acquisition and a remaining certificate gap

<a id="relational-upper-corner-response"></a>

[`relational_capture_response.py`](../../benchmarks/relational_capture_response.py)
froze one materialized preparation, `e=w=1/2`, `beta=nu=1`, no Gamma,
the supplied ten-node/twelve-edge support, horizon 64 and three Euler steps
`1/64,1/128,1/256` before evaluation. Form and phase offsets were zero.
The binary values corresponding to `(2/5,1/5)` have exact `r=0` but
`q=1+2^-54`; the preceding ideal-input proof is not silently identified
with those represented values. Python 3.13 and dependency/platform versions,
all 599 package/producer source fingerprints, exact inputs and numerical
settings are retained in the prediction.

The protocol stopped each grid at its first admitted target certificate,
shared-owner rejection, or horizon. No projection, retry, phase forcing,
parameter sweep or horizon extension was performed. All three grids reached
64 without rejection; the target certificate was unavailable on every grid.
The compound `finite_prediction_passed` is therefore **false**, while the
numerical work-residual policy passed.

| Euler step | Executed steps | Final storage upper bound, rounded | Final squared quotient norm, rounded | Final target excess upper bound, rounded |
| --- | --- | --- | --- | --- |
| `1/64` | 4096 | 6.919042007 | 0.03467713 | 0.009211950 |
| `1/128` | 8192 | 6.919040986 | 0.03467310 | 0.009210930 |
| `1/256` | 16384 | 6.919040485 | 0.03467112 | 0.009210429 |

Every grid starts with windings `(0,0)` and has `(1,1)` at all retained
checkpoints from `t=1/4` through 64. This is sampled numerical evidence,
separate from the ideal short-crossing theorem. At the retained times 32
and 64, all first-ring `(a,b)` coordinates lie in the positive-twist
rectangle and the full-state certified storage is below 7. For the finest
grid at 32, `(a,b)` is approximately `(2.267268628,1.109711608)` and
the storage upper bound is approximately `6.970551621`.

However, the represented states have small **nonzero** reflection defects.
At 64 the largest form/phase defects across the grids are below `2e-18`,
while ring-copy defects are zero. The reflected theorem rejects those states
solely on its exact-symmetry premise. The full-state local certificate also
rejects them: both distance and excess storage remain above its conservative
thresholds. Neither certificate can be promoted by rounding those defects
away or by treating energy below 7 as a support-independent basin test.

The maximum native work residual across the three traces is below
`2.176e-16`; clock defects are zero. Maximum absolute energy-step defects
decrease from about `2.503e-5` to `6.257e-6` to `1.564e-6`.
All retained states and represented Euler chords satisfy the positive-
resultant guard. At 64 the coarse/middle and middle/fine state infinity
differences are about `4.391e-6` and `2.154e-6`; differences decrease under
refinement at every common checkpoint. These are numerical comparisons,
not rigorous global trajectory-error bounds or asymptotic convergence.

| Immutable record | SHA-256 |
| --- | --- |
| [Prediction](../../docs/assets/relational_capture_response/result.prediction.json) | `4045ab698013f2bb071c5083a99e8aea760c7ce42e568e1be5a230af9629db7c` |
| [Response](../../docs/assets/relational_capture_response/result.json) | `7085c782f0b85c465c530d2e84ff2990e71dfcc89d950f1f993025a952310ed9` |

The [retained-response controls](../../tests/physics/test_relational_capture_response.py)
recompute certificate decisions from saved states without re-executing the
campaign. Earlier frozen responses remain unchanged. This result supports
a regular same-law route toward the maintained geometry on supplied support;
it does not yet certify the original continuous trajectory's entry into the
protected basin, autonomous support creation, or a physical NFR identity.

## 20. A full-state acute-sector barrier without reflection

<a id="relational-acute-sector-capture"></a>

The retained response exposes a gap between the dynamics and the available
sufficient certificates: its tiny symmetry defects invalidate the reflected
theorem, while its distance from the target exceeds the deliberately small
local bound. Neither restriction is intrinsic to every capture proof. The
same support admits a wider full-state theorem based on its cycle periods
and cosine storage, with no symmetry projection or changed evolution law.

Retain the two supplied C5 rings and the two bridges between matching
positions 0 and 1. Let `e,w,beta>0` and all held capacities be strictly
positive. Suppose every true wrapped support-edge phase difference is
strictly acute and both supplied oriented rings have winding `s`, where
`s=+1` or `s=-1`. Define

\[
V_* =5\bigl(1-\cos(2\pi/5)\bigr),\qquad
V_{\rm face}=5-4\cos(3\pi/8),\qquad
\mathcal B=V_*+V_{\rm face}.
\]

**Conditional full-state capture.** If the actual full-state storage satisfies

\[
E_D+\beta V<\beta\mathcal B,
\]

the ideal trajectory remains in its acute component for all forward time
and converges to uniform form and the aligned winding-s twist, modulo common
offsets. Exact ring copying, form/phase reflection and a prescribed small
Euclidean radius are not premises. The theorem concerns circular phase
geometry: the final raw phase lift can additionally contain fixed nodewise
integer multiples of `2*pi`, determined by its initial acute component.

### The cycle geometry supplies the barrier

For `s=+1`, the five oriented acute differences of either ring sum to
`2*pi`. Convexity of `1-cos(delta)` on `[-pi/2,pi/2]` and Jensen's
inequality give ring storage at least `V_*`. At a first acute-boundary
encounter on that ring, a difference of `-pi/2` is impossible: the other
four would need sum `5*pi/2`, exceeding their maximum `2*pi`. Thus a
boundary edge has difference `pi/2`, and the other four have sum `3*pi/2`.
Another application of Jensen gives

\[
V_{\rm ring}\ge1+4\bigl(1-\cos(3\pi/8)\bigr)=V_{\rm face}.
\]

The other ring still costs at least `V_*`; bridge costs are nonnegative.
Consequently every first boundary through a ring edge has total phase
storage at least `mathcal B`. At a bridge boundary, its own cost is 1,
so the total is at least `1+2*V_*`, which is larger: since
`cos(3*pi/8)>cos(2*pi/5)`,

\[
1+V_*-V_{\rm face}
=1-5\cos(2\pi/5)+4\cos(3\pi/8)
>1-\cos(2\pi/5)>0.
\]

Sign reversal proves the same statements for `s=-1`. The one-ring Jensen
bound is sharp, but `mathcal B` is only a sufficient whole-support boundary
bound; simultaneous equality need not respect both bridges. No claim of an
optimal basin or a complete classification of the regular domain is made.

The barrier has an exact radical expression and a simple rational lower bound:

\[
\mathcal B=\frac{45-5\sqrt5}{4}-2\sqrt{2-\sqrt2}
>\frac{6924179}{1000000},\qquad
\mathcal B\simeq6.924181298665.
\]

For example, the certified inequalities
`cos(2*pi/5)<309017/1000000` and
`cos(3*pi/8)<382684/1000000` imply the displayed lower bound.
They follow by squaring positive rational bounds in the exact radical
formulas, independently of binary trigonometric evaluation. An executable
certificate can instead reuse the shared rational pi/cosine enclosures.

### Compactness, the target and the zero-loss set

The two rings and the four-cycle through both bridges form a cycle basis.
Four strictly acute differences cannot sum to a nonzero integer multiple
of `2*pi`, so the four-cycle has period zero. The three periods therefore
agree with the aligned twist: both rings have phase step `s*2*pi/5` and
both bridges have zero difference.

Choose compatible node lifts of that reference and the initial phases.
Equality of all cycle periods makes their edge-difference discrepancy an
exact graph gradient. The common acute component is therefore a convex
polytope in lifted phase coordinates modulo their common offset, with each
edge difference restricted to its selected interval of length `pi`.
The aligned twist belongs to this component. The phase-storage Hessian is
the graph Laplacian with edge weights `cos(delta)>0`, hence is positive
definite on the common-phase quotient. Its sole critical geometry in the
component is consequently the aligned twist.

Storage nonincrease and the strict boundary gap prevent a first exit.
Connected support bounds phase differences modulo the common offset;
Dirichlet storage bounds form modulo its common offset. Thus a fixed joint
sublevel below `beta*mathcal B` is compact in this quotient and stays a
positive distance from the acute boundary. All H entries stay strictly
positive. Quotient continuation is global, and bounded common-offset rates
also exclude finite-time failure of the full state.

For the invariant zero-loss set, positive capacity and `e>0` force
`Bx=0`. Uniform form alone does not yet imply zero common-form drift.
To preserve that condition requires `N*g=c*1` for a scalar c. But the
undirected phase potential satisfies

\[
\sum_i H_i g_i=0,\qquad
c\sum_i H_i/\nu_i=0.
\]

Every `H_i/nu_i` is positive, so `c=0` and `g=0`. LaSalle's invariance
principle on the compact quotient now gives convergence to the unique twist
geometry and uniform form. The existing local theorem supplies eventual
exponential recovery and finite limiting common offsets. Those offsets need
not equal their initial means in an asymmetric state.

### Evidence and execution boundary

The hypotheses can be checked from a saved full-state snapshot using strict
pi-enclosed edge gaps, exact integer cycle periods and certified cosine
storage bounds. Numerical winding labels, rounded symmetry or an estimated
distance to the target cannot replace those checks. The shared sector
certificate admits the theorem's arbitrary held positive capacities and
positive beta; the separate reflected and local owners retain their own
more restrictive premises.

This theorem was derived **after** the reserved response in section 19.
Applying it to a retained endpoint is a separately identified mathematical
reanalysis. It cannot turn the failed original frozen prediction into a
successful preregistered capture test, change its earlier certificate
decisions, or justify rerunning it with another gate. An admitted endpoint
certifies the ideal law restarted at that precise represented state. It
still does not bound the discrepancy between the numerical trace and the
original continuous trajectory from its winding-zero preparation. No
longer trajectory, support event, additional pressure, controller or
physical-identity claim is introduced by this argument.

The [static controls](../../tests/physics/test_relational_formation_admission.py)
check the support's cycle basis, target periods, one-ring face costs and the
strict rational barrier bound without executing any evolution.

### Shared certificate and retained endpoint reanalysis

`certify_relational_sector_capture` and `Network.relational_sector_capture`
in the [shared capture owner](../../src/tnfr/physics/relational_capture.py)
implement this full-state theorem. The first retained endpoint audit used
unit capacity and unit beta; extending the shared admission to the theorem's
positive held capacities and beta does not change that record. The existing
exact topology, phase-storage and pi/cosine owners are reused. Each candidate wrapped edge
gap is represented as a rational plus an integer multiple of mathematical
pi; two strict enclosed inequalities must certify its acute interval.
Summing the admitted integer turns yields both ring periods and the bridge
square's period. Floating winding telemetry remains separate. The full
storage upper bound must lie below a certified lower bound on the barrier.
The [API contract](../../docs/API_CONTRACTS.md#conditional-relational-capture)
owns report fields, unsupported domains and restart-only scope.

The separate [read-only auditor](../../benchmarks/relational_capture_audit.py)
binds the immutable section-19 records and reconstructs their three final
states without modifying any coordinate. All three states pass the new
theorem. Their lower bounds on the energy margin `mathcal B-E` are about
`0.00513929210`, `0.00514031242`, `0.00514081319` in coarse-to-fine order;
every strict acute margin exceeds `0.16215` radians. Both exact ring periods
are one and the bridge-square period is zero. Consequently the ideal law
restarted at each actual represented endpoint remains regular and converges
to the aligned twist without a reflection premise or further forcing.

| Separate immutable evidence | SHA-256 |
| --- | --- |
| [Post-evaluation endpoint audit](../../docs/assets/relational_capture_response/endpoint-capture.audit.json) | `8d5dee62f7514118d69a19173599a6a536f1c516d2ee486c98abf3ea4e22352a` |
| [All 599 original fingerprinted source files](../../docs/assets/relational_capture_response/source-at-evaluation.zip) | `eb0600da8e479e8b7f1bfede31dbf802ef420ed8419fad982ca49e3e7ba8d049` |
| [Four-file source overlay for the endpoint audit](../../docs/assets/relational_capture_response/source-at-endpoint-audit.delta.zip) | `a0c9075f1514ab0b8fdac5d54a1c7e8c79e7c535d48348b9f5c1ce72730d23b2` |

The source archive was verified against every original prediction digest
before post-evaluation integration changed the engine/SDK source. It does
not bundle dependencies; their versions remain in the original prediction.
The new audit retains its own source/runtime manifest and explicitly records
both `all_endpoints_admitted=true` and the unchanged original
`finite_prediction_passed=false`. Tests check archive byte identity, detached
report data and the old/new decisions without re-executing any trajectory.
The small overlay plus the base source archive reproduces every source hash
in the endpoint audit; it preserves the original unit-restricted owner before
the generalization below without duplicating the complete package archive.

The endpoint audit alone leaves an original-IVP obligation: a validated
finite-time transit enclosure, now supplied [below](#relational-validated-transit).
The exact represented preparation is initially
reflected, so its ideal trajectory stays in the existing four-coordinate
slice even though finite Euler arithmetic breaks that symmetry. Enclosing
that same flow to an admitted reflected or acute-sector basin would join
formation and maintenance for the original continuous state. It requires
no new constitutive law, tuned preparation or longer finite observation.

### Consolidation: capacity, storage scale and a quantitative regularity margin

<a id="relational-sector-consolidation"></a>

The sector theorem already uses the same independent-capacity law as the
heterogeneous-capacity balance and local recovery theorems. No equality of
capacities appears in its barrier or zero-loss argument. All capacities must
remain strictly positive and held; `e,w,beta>0`, the supplied support, the
strict acute component and its exact cycle periods remain premises.
Multiplying every capacity by the same `a>0` multiplies **both** differential
rows by a, so it changes only their common clock. Heterogeneous positive
capacities can change transients and limiting offsets while preserving this
sufficient geometric basin. None of these statements admits capacity events
or extends the separate reflected/local production certificates.

Dividing storage by beta gives `E_D/beta+V_phi`. Thus, on exact real form
charts, replacing beta by `beta_new` and multiplying all form contrasts by
`sqrt(beta_new/beta)` preserves the sector energy test. It need not preserve
the trajectory under a time change: with `y=x/sqrt(beta)`, form damping has
coefficient e while the two exchange terms have coefficient `w/sqrt(beta)`.
Beta therefore controls relative storage and exchange as well as admission;
it is not generally a pure clock scale.

An admitted energy deficit supplies an explicit lower bound on future
regularity. It is a derived quantity, not a new primitive or policy threshold.
Put

\[
\eta=\mathcal B-\mathcal E(0)/\beta>0,\qquad
\kappa=2\pi/5,\qquad
F(t)=1-\cos t+4\left[1-\cos\left((2\pi-t)/4\right)\right],
\]

and `d=F(pi/2)-F(kappa)=mathcal B-2*V_*`. Every oriented signed gap
`y=s*delta` on either acute ring is positive: the four other gaps are each
strictly below `pi/2` and all five sum to `2*pi`. Jensen's inequality on
those other four gaps gives ring storage at least `F(y)`. On
`kappa<=t<=pi/2`,

\[
0\le F'(t)=\sin t-\sin((2\pi-t)/4)
\le M=1-\sin(3\pi/8)<1/13.
\]

The final strict rational inequality has an exact static check:
`sin(3*pi/8)=sqrt(2+sqrt(2))/2>12/13`, because
`238^2=56644<57122=2*169^2`.
The factor 13 is a conservative rational choice below `1/M`, not a
distinguished TNFR constant or an evolution parameter.
Also
`0<eta<=d<M*pi/10<pi/130`, using the two-ring Jensen minimum and
integrating the displayed derivative. Storage nonincrease makes these same
initial-budget estimates valid at every future time. For a ring gap
`y>=kappa`, they imply

\[
\eta\le F(\pi/2)-F(y)\le M(\pi/2-y),\qquad
\pi/2-y>13\eta.
\]

For `y<=kappa`, its margin is at least `pi/10>13*eta`. A bridge instead
has cost `1-cos(delta)<=d-eta<pi/130<1/2`, hence `abs(delta)<pi/3` and
margin greater than `pi/6>13*eta`. Every support edge therefore obeys the
uniform, all-future strict acute margin

\[
\boxed{\quad \pi/2-|\delta_{ij}(t)|>13\eta>0.\quad}
\]

Let `z_i=sum_{j~i} exp(i*(theta_j-theta_i))` be the relative neighbor
resultant and `d_i` the support degree. Sine concavity on `[0,pi/2]`
gives `cos(delta_ij)>=sin(13*eta)>=26*eta/pi`. In the resulting
positive-real chamber, `alpha_i=Arg(z_i)` lies in `(-pi/2,pi/2)` and
`H_i=pi*Re(z_i)*tan(alpha_i)/alpha_i`, with ratio one at zero. Thus

\[
\boxed{\quad
\operatorname{Re}z_i(t)\ge\frac{26d_i\eta}{\pi},\qquad
|z_i(t)|\ge\frac{26d_i\eta}{\pi},\qquad
H_i(t)\ge26d_i\eta>0.
\quad}
\]

The phase Hessian likewise satisfies
`Hess(V_phi)>= (26*eta/pi)*B` as quadratic forms. These bounds quantify
the compact regular domain used by the proof; they do not assert a global
exponential rate or introduce additional phase damping.

The shared certificate can retain a rational
`eta_lower=B_lower-E_upper/beta`, where `B_lower` encloses the geometric
barrier and `E_upper` encloses the actual full-state storage. Only complete
admission of all sector hypotheses makes the future bounds available.
Replacing eta by this positive lower bound and pi by its certified upper
bound on the right-hand side preserves every non-strict lower bound above.
Unit-capacity and unit-beta flags remain descriptive evidence, not admission
requirements. These are exact ideal-law implications from the captured
represented snapshot. They neither bound the preceding numerical error nor
by themselves close the finite-time transit obligation for the original
initial value problem; the validated proof below supplies that separate link.
The frozen response and its later endpoint audit remain separate.

### Validated continuous transit on the exact reflected subsystem

<a id="relational-validated-transit"></a>

The read-only owner
[`physics/relational_transit.py`](../../src/tnfr/physics/relational_transit.py)
checks the original continuous IVP through the existing invariant reduction.
It uses the supplied two-ring support, exact copied/reflected initial state,
unit held capacities and the same positive-coefficient, source-free relational
law. This is a proof computation; it does not advance a live graph or replace
the shared production integrator. The SDK delegate and exact report exporter
retain the full initial admission and every accepted interval step.

With `q=3A-B`, `r=2B-A`, let `d=a/2-b`,

\[
C_0=1+\cos(2a)+\cos(a-b),\quad S_0=-\sin(2a)-\sin(a-b),
\quad u=S_0/C_0,\quad R(u)=\operatorname{atan}(u)/u,
\]

where `R(0)=1`. On the admitted positive-real chamber,

\[
g_0=uR(u)/\pi,\quad H_0^{-1}=R(u)/(\pi C_0),\quad
g_4=d/\pi,\quad H_4^{-1}=[2\pi\cos(a/2)\operatorname{sinc}(d)]^{-1}.
\]

The original proof used the zero-safe analytic series on `abs(u)<=1/2`
and the sinc series on `abs(d)<=1`. These are sufficient numerical domains,
not new physical limits. The current arctangent-ratio owner retains that
same series near zero and, outside it, uses `atan(u)'=u'/(1+u^2)` followed
by formal division when the entire expansion interval avoids zero. A wide
interval crossing zero outside the first domain is still unavailable.
This analytic extension was tested before evaluating the zero-form control;
for example, the already proved consensus-basin state `q=r=0,a=b=1` has
`u` about `-0.5741`, outside the old numerical domain. It is not a new law.
Positive `2*cos(a/2)*cos(d)` together with `abs(d)<=1`
selects exactly the displayed branch for `g_4`. The zero-safe analytic ratios
avoid dividing by a pressure source that crosses zero. The reduced rows are

\[
\begin{aligned}
\dot q&=-eq+(e/2)r+w(3g_0-g_4),\\
\dot r&=(e/3)q-er+w(2g_4-g_0),\\
\dot a&=(w/\beta)qH_0^{-1},\qquad
\dot b=(w/\beta)rH_4^{-1}.
\end{aligned}
\]

The actual frozen form values are binary64 `0.4` and `0.2`, so the exact
initial values are **`q=1+2^(-54)`, `r=0`**, not `(1,0)`. The phase value is
the exact represented rational `875483625981347/562949953421312` at both
coordinates. The earlier ideal-pi/rational-fifths short-crossing theorem is
not silently substituted for this different IVP.

#### Whole-time enclosure and error propagation

All arithmetic bounds use rational endpoints rounded outwards to a fixed
128-bit dyadic grid. Mathematical pi and cosine bounds reuse the existing
Machin-series and cosine owners. Their optional higher-precision path leaves
the production cosine defaults unchanged. Sinc and arctangent-ratio series
bound every normalized derivative through the requested order; a scalar
series remainder alone would not validate time derivatives.

For a current box `Y`, an attempted compact convex tube `B` must satisfy

\[
Y+[0,h]F(B)\subset\operatorname{int}B,
\qquad C_0>0,\quad 2\cos(a/2)\cos(d)>0,\quad 2\cos b>0
\quad\text{throughout }B.
\]

The last condition retains the full central-node resultant even though its
row vanishes under reflection. A first-exit argument and smoothness on an
open neighborhood of the tube prove existence and containment throughout
the step. This does not require a Picard contraction or identify an Euler
chord with a continuous trajectory. Inflation constructs a candidate tube;
only the strict displayed inclusion admits it.

Let `c` be the exact midpoint of `Y`. Generate normalized flow coefficients
`a_j` recursively by `j*a_j = [s^(j-1)]F(sum_l a_l*s^l)`. The center solution
at the endpoint is enclosed by

\[
P_h(c)+h^{p+1}a_{p+1}(B),\qquad
P_h(c)=\sum_{j=0}^{p}h^j a_j(c).
\]

The last coefficient is evaluated over the **entire tube**, bounding the
normalized derivative at every possible remainder point. Outward arithmetic
also encloses the represented midpoint and all polynomial operations.

To avoid discarding the linear damping while propagating earlier uncertainty,
form a Metzler comparison matrix from a first-order interval jet of the
same vector field:

\[
M_{ii}\ge\sup_B\partial_iF_i,\qquad
M_{ij}\ge\sup_B|\partial_jF_i|\quad(i\ne j).
\]

The diagonal keeps its sign. Both the center solution and every solution
starting in `Y` remain in the convex tube, so the upper-Dini comparison
inequality gives componentwise separation at most `exp(h*M)*rad(Y)`.
The shared comparison kernel shifts the diagonal to a nonnegative matrix,
uses a positive rational series with a norm-bounded tail, and encloses the
compensating scalar exponential. Adding this propagated radius to the center
Taylor enclosure gives `Y_next`. Its intersection with `B` remains valid
because both sets already enclose the endpoint. No midpoint projection of
the actual state is performed.

#### Joining transit to maintenance

The endpoint test applies to the whole box, not to a synthetic midpoint
graph. By default it requires `a>2*pi/3`, `a<pi`, `b>0`, `b<pi/2` and

\[
\mathcal E=\frac45(q+r/2)^2+r^2
+\beta[10-2\cos(2a)-4\cos(a-b)-4\cos b]<7\beta.
\]

Exact initial symmetry and uniqueness preserve the invariant slice. These
inequalities therefore join directly to the protected-capture theorem:
the ideal continuation is regular for all future time and converges to the
positive aligned twist modulo its constant common offsets. The proof of
initial winding zero additionally requires exact symmetry and each initial
raw cycle gap strictly inside `(-pi,pi)`. At a positive-rectangle endpoint
only the edge `-2a` gains `2*pi` under principal wrapping; the ring period is
one. No separate numerical estimate of the crossing instant is needed.

The current certificate also accepts `requested_sector=0` or `-1`, and
`None` classifies any admitted one of the three disjoint protected rectangles.
The point and interval certificates evaluate one shared affine-margin ledger;
their precision and input scopes remain distinct. The actual sector, selected
rectangle and all candidate bounds are retained. `positive_rectangle_margins`
remains a compatibility field, not the only gate. Consensus is sector zero,
not an unavailable or falsey target. The original positive-only audit and
its exact archived source remain unchanged.

The caller must declare the horizon, step and order. Unsupported analytic
domains, unresolved tube inclusion or inconclusive endpoint margins return
unavailable evidence, retaining the accepted prefix and first failed tube.
They do not prove that the true trajectory fails. A proof audit of an already
evaluated preparation is explicitly post-evaluation verification; neither a
successful proof nor an improved enclosure rewrites the original frozen
finite-executor prediction.

#### Retained original-IVP proof audit

The [producer](../../benchmarks/relational_transit_proof.py) froze its inputs,
numerical policy, runtime and all 604 source files before this post-evaluation
proof computation. It reused the immutable represented upper-corner seed,
`e=w=1/2`, `beta=nu_i=1`, fixed horizon 32, step `1/8`, Taylor order 12,
128-bit outward intervals, at most 16 Picard inflations per step, and no retry.

All **256** whole-time steps pass. The complete endpoint box lies in `R+`,
and its storage is enclosed in the conservative decimal interval
`[6.97054580874334, 6.97054580874345]`, strictly below 7. Every endpoint
coordinate width is below `7e-15`. Across all tubes, the port, interior and
central relative-resultant real lower bounds exceed `0.98293`, `0.84597`
and `0.03114543`, respectively; every strict Picard inclusion margin exceeds
`1.4389e-7`. Exact rational inequalities and every tube are in the report;
the displayed decimals summarize them. No tube failed and no bound was
repaired by changing the initial state, law, step, order or horizon.

Consequently this exact represented **winding-zero initial state** generates
a winding-one pattern and converges to the maintained aligned twist under
the declared continuous law. This joins formation to maintenance for the
original IVP, rather than restarting the law at a numerical endpoint.

| Separate immutable proof evidence | SHA-256 |
| --- | --- |
| [Frozen proof protocol](../../docs/assets/relational_capture_response/continuous-transit.audit.protocol.json) | `d8913d6af0d40fd1c903eaf24be29fb8ebeb6d3be28d6415d4e8dafb9974db99` |
| [Validated transit report](../../docs/assets/relational_capture_response/continuous-transit.audit.json) | `ac3104c42cf9054968eac770fee5c1b1e174547a408875edac0b635fe04ca781` |
| [Executed source archive](../../docs/assets/relational_capture_response/continuous-transit.audit.source.zip) | `381212374968f32fe59751c752b4dc812844a3309e00f771704a72b31a154c93` |

An independent retained-record audit recomputes the strict self-inclusions,
all resultant bounds, time chain, exact initial seed and whole-endpoint
conditions without re-integrating the trajectory. Tests also bind every
archived source byte, check independent analytic references for the interval
kernels, and preserve the original finite-executor `false` verdict.
This is a computer-assisted conditional mathematical result. The proof is
not a formally verified implementation and does not validate the constitutive
law physically. Initial support, capacity and geometric/storage preparation
remain supplied; neither substrate creation nor generic pattern selection
has been established.

#### Synergy: qualitative robustness beyond exact reflection

Whenever the validated transit above succeeds from a strictly winding-zero
preparation, it also implies a qualitative **open-set** formation result for
the full state. Exact reflection is needed by this proof's reduced enclosure;
it is not thereby necessary for every nearby trajectory to form the pattern.

The reflected capture theorem gives convergence to the aligned positive
twist. Its limiting storage is
`E_*=beta*[10-10*cos(2*pi/5)]`, strictly below the full-state acute-sector
barrier `B_*=beta*[10-5*cos(2*pi/5)-4*cos(3*pi/8)]`. Positivity of their
gap is exact: `5*cos(2*pi/5)>4*cos(3*pi/8)` reduces, by squaring positive
sides, to `176*sqrt(2)>239`, whose squared comparison is `61952>57121`.
The limiting edge gaps are strictly acute. The reference solution therefore
enters the open full-state protected sector at some finite time `T_*`.
This argument does not locate `T_*`; an endpoint in the reflected basin
need not already be in the smaller acute-energy basin.

The reference trajectory is compact and regular on `[0,T_*]`. Smooth
dependence on the initial state pulls that open target basin back to an
open neighborhood of the **original** preparation. All sufficiently nearby
full states remain regular through `T_*` and then converge to the same
positive-twist orbit. For the frozen upper-corner preparation, the smallest
initial antipodal margin is `pi-2*a_0>1/32`, so a sufficiently small phase
perturbation also preserves initial ring winding zero. Intersecting the two
open neighborhoods proves winding-zero formation followed by maintenance
without requiring exact copy or reflection of the perturbed state.

The same conclusion permits small unequal **held positive capacities**:
augment the ODE by `nu_dot=0`, use smooth dependence on this parameter, and
apply the full-state sector theorem, which already admits heterogeneous
positive capacities. Keep the graph, unit conductances, positive `e,w,beta`
and unforced constitutive law fixed. Common form/phase offsets remain neutral
and their limiting values may differ between preparations.

This consequence supplies neither a quantified perturbation radius nor a
uniform entry time. It does not cover arbitrary positive capacities, topology
changes, capacity events, forcing or finite binary64 execution of neighboring
states. It is a conditional dynamical robustness result, not an identification
of the pattern with a physical constituent.

### Zero initial form contrast: a prospective phase-to-form discriminator

<a id="relational-zero-form-control"></a>

The prospectively frozen zero-form control changes only the initial form of the successful reference:
`x_i=0` for every node. Its represented phases, supplied two-ring support,
unit capacities, `e=w=1/2` and `beta=1` remain identical. The common producer
[`relational_transit_proof.py --zero-form`](../../benchmarks/relational_transit_proof.py)
freezes a distinct protocol and source archive. No subsequent response is
used to choose another preparation, horizon or outcome gate.

#### What is predicted before evolving the control

For arbitrary uniform initial form `x(0)=m*1` under this unit-capacity law,

\[
\dot x(0)=w g(\theta_0),\qquad \dot\theta(0)=0,\qquad
\ddot\theta(0)=\frac{w^2}{\beta}H(\theta_0)^{-1}B g(\theta_0).
\]

The acceleration is a derivative of the existing two coupled first-order
rows, not an added inertial equation. Therefore zero form contrast need not
be an equilibrium: nonuniform phase pressure can create form contrast and
then change phase through the same feedback. In the selected preparation,
rigorous initial derivative intervals give

\[
\begin{aligned}
0.10885&<\dot q(0)<0.10886,&
-0.24255&<\dot r(0)<-0.24254,\\
\dot a(0)&=\dot b(0)=0,&
0.01730&<\ddot a(0)<0.01732,\\
&&-0.03003&<\ddot b(0)<-0.03001.
\end{aligned}
\]

Although `q` initially grows, the original form coordinates have
`A_dot=w*g0<0` and `B_dot=w*g4<0`: this dynamically generated direction is
different from the supplied positive reference seed. Initial local signs
alone cannot select a terminal basin.

Form Dirichlet storage grows at quadratic order,
`E_D''(0)=w^2*g^T*B*g>0`, while phase storage loses the same leading amount.
For the joint storage,

\[
\dot{\mathcal E}(0)=\ddot{\mathcal E}(0)=0,\qquad
\mathcal E^{(3)}(0)=-2e\left[\frac43\dot q(0)^2+2\dot r(0)^2\right]<0.
\]

Its initial value is about `7.93652606007>7`; dissipation starts at cubic
order. These are static mathematical consequences evaluated before any
control trajectory. They do not imply capture from the initial energy test.

The protocol fixes horizon 32, step `1/8`, order 12 and 128-bit outward
arithmetic, with no retry. Terminal positive twist, consensus, negative twist
and unresolved admission are predeclared alternatives. An admitted basin
settles a conditional asymptotic limit; merely failing to enter one by the
fixed horizon does not. A numerical proof-domain failure remains distinct
from a zero resultant of the actual law. Prepared phase geometry supplies
storage and pressure: none of these claims describes creation from nothing
or emergence of the initial graph.

#### Retained result: transient winding followed by proved consensus

The first frozen control completes all 256 validated steps to time 32, with
no unresolved tube or retry. Its full endpoint box lies in the consensus
rectangle `S`, with joint storage conservatively enclosed in
`[2.11675502876186, 2.11675502876328]`, strictly below 7. Maximum endpoint
coordinate width is below `1.5e-13`. Every tube retains positive port,
interior and central resultants; their respective global lower bounds
exceed `0.96412`, `1.01502` and `0.03098482`. Thus this original continuous
initial state converges to consensus under the same conditional law.

| Preparation, with identical phase/support/capacity/law | Proved limiting basin | Whole endpoint storage upper bound at time 32 |
| --- | --- | --- |
| Retained supplied form contrast | Positive aligned twist | `6.97054580874345 < 7` |
| Zero initial form contrast | Consensus | `2.11675502876328 < 7` |

The source manifest and every saved Picard inclusion, resultant bound,
initial input and terminal inequality were independently checked without
re-integrating either trajectory. The three protected rectangles share one
definition in the point and interval owners, so consensus is explicitly
`target_sector=0` with `terminal_basin_admitted=true`. The positive-pattern
flag remains false for this control; it is not an unavailable outcome.

| Immutable control evidence | SHA-256 |
| --- | --- |
| [Frozen prospective protocol](../../docs/assets/relational_zero_form_response/result.protocol.json) | `920413db8d67dc46c13914bdd550bb097b416ade8dadd0417883b5b5eeca49d1` |
| [Validated response](../../docs/assets/relational_zero_form_response/result.json) | `bb30b7b2ac8812871b7f7667c4a898d79293a4e25e7b96bff13ec29fdf5996c4` |
| [604 fingerprinted source files](../../docs/assets/relational_zero_form_response/result.source.zip) | `dcd7b49d4f596c9703ea191b74fadb0d28c6952d529363ce559b245984dbe8ee` |

A **post-evaluation** inspection of retained endpoint enclosures also proves
both ring windings zero through time `15/8`, one at every saved endpoint
from `2` through `23/4`, and zero from `47/8` through `32`. Raw cycle gaps
telescope; only `-2a` acquires an additional `2*pi` wrap in the middle group.
Consequently continuity gives at least one crossing in each of
`(15/8,2)` and `(23/4,47/8)`. These are retrospective crossing brackets,
not a frozen crossing-time prediction, exact event times or proof of an
uninterrupted winding-one lifetime between every sampled point.

This control therefore generates form contrast and a transient winding-one
configuration, but does **not** maintain the reference pattern. The selected
initial form preparation changes the asymptotic basin. That does not establish
a universal need for initial form contrast: other phase preparations remain
unclassified. Nor does it prove that scalar storage alone chooses a basin;
removing form changes its signed direction as well as its magnitude and energy.
The equal-storage control below separates those two effects.

#### General mechanism and its exact stationary boundary

The initial exchange does not depend on the special two-ring graph. On any
connected fixed unit support admitted by the relational law, let held
`N=diag(nu_i)>0`, regular `H>0`, `w,beta>0` and `x_0=m*1`. Then

\[
\dot x_0=wNg,\qquad \dot\theta_0=0,\qquad
\ddot\theta_0=\frac{w^2}{\beta}H^{-1}NBNg.
\]

Reciprocity gives `1^T H g=0` by cancellation of edge sine contributions.
If `Ng=c*1`, this identity gives `c*sum_i(H_i/nu_i)=0`, hence `g=0`.
For any nonzero phase source, `Ng` is therefore nonconstant, `BNg` is nonzero,
and form contrast and relative phase acceleration arise. An acceleration
proportional to `1` would similarly imply `BNg=c*H*N^-1*1`; summing its
entries forces `c=0`, which excludes nonzero acceleration without relative
phase change. Furthermore,

\[
\ddot E_D(0)=w^2g^TNBNg>0,\qquad
\beta\ddot V(0)=-\ddot E_D(0),
\]

and, for `e>0`,

\[
\mathcal E^{(3)}(0)
=-2ew^2(BNg)^TND^{-1}(BNg)<0.
\]

Conversely, uniform form with `g=0` is exactly stationary under the held
unforced law. The model does not depart spontaneously from a completely
balanced equilibrium. This theorem identifies a local conversion of supplied
nonequilibrium phase structure into form, not the eventual identity or lifetime
of the configuration it generates. The maintained acute geometries already
have their [circulation classification](#equilibria-reuse-the-existing-circulation-classification);
that result should be reused, not counted as a new discovery from this control.

Common ideal shifts `x -> x+m*1` and `theta -> theta+c*1` leave both rows,
contrast storage and winding invariant (`B*1=L*1=0`). The zero-form control
therefore represents the entire common-uniform-form family and all common
phase rotations. Its zero origin is not a privileged physical value.
Arbitrary rounded additions to stored binary64 phases need fresh admission
because their exact represented differences may no longer coincide.

### Equal-storage form reversal: separating energy from direction

<a id="relational-reversed-form-control"></a>

The prospectively frozen control exactly negates the successful reference's
stored initial EPI values and changes no phase, support, capacity or coefficient.
In the reflected coordinates, `(q,r) -> (-q,-r)` with `(a,b)` unchanged.
The represented `q_0=1+2^(-54)` becomes its exact negative; signed zero in
the raw JSON is retained and represents the same mathematical zero.
This is neither simultaneous form/phase reflection nor time reversal.

#### Exact initial match and local discriminator

On the admitted fixed graph with held positive capacity, write `y=Bx` and
`K=N*D^-1`. A common-offset reflection `x^- = 2m*1-x^+` has `y^-=-y^+`.
Identical phase geometry keeps `g,H,V` fixed. Consequently,

\[
E_D^-=E_D^+,\quad V^-=V^+,\quad
\mathcal E^- =\mathcal E^+,\quad
\dot{\mathcal E}^- =\dot{\mathcal E}^+=-e\,y^TKy,
\qquad \dot\theta^-=-\dot\theta^+.
\]

The signed exchange term

\[
J=w\,y^TNg,\qquad
\dot E_D=-e\,y^TKy+J,\qquad \beta\dot V=-J
\]

reverses sign. It is a read-out of work already present in the joint law,
not a new pressure channel, state primitive or control policy. Equal total
loss therefore allows opposite initial transfer between phase and form.
For the actual reference, certified intervals give
`J_ref` about `-0.0198749634744`: form initially supplies phase storage.
The reversed preparation has opposite transfer while losing total storage
at exactly the same instantaneous rate.

Since `y_dot=-e*B*K*y+w*B*N*g`,

\[
\ddot{\mathcal E}
=2e^2y^TKBKy-2ew\,y^TKBNg,\qquad
\ddot{\mathcal E}^- -\ddot{\mathcal E}^+
=4ew\,y^TKBNg.
\]

For the selected two-ring state `r_0=0`, put
`h=w*(3*g0-g4)>0`. This reduces to

\[
\ddot{\mathcal E}(-q_0)-\ddot{\mathcal E}(q_0)
=\frac{16}{3}e q_0 h>0.
\]

Static exact-input calculations, performed before evolution, retain
rigorous intervals around `E_0=8.736526060070739`, equal ideal
`E_dot(0)=-(2/3)*(1+2^(-54))^2`, and an acceleration difference between
`0.29026` and `0.29028`. Initial `a_dot` changes from about `+0.159026` to
`-0.159026`; `b_dot=0` in both. Native engine fields independently check
represented form-storage equality, loss equality and phase-rate negation;
their floating work residual is not substituted for the ideal identities.

Thus `(E,E_dot)` already fails as an autonomous state description: two
identical summary states have different derivatives of `E_dot`. Before
evolution this did not decide the **limiting basin**. The frozen criterion
was that different admitted terminal sectors would also exclude a basin
selector based only on those initial summaries, even with identical phase
geometry and model parameters. The same limit would not establish storage
sufficiency; numerical unavailability would remain inconclusive.

The shared producer's `--reverse-form` mode freezes one control with horizon
32, step `1/8`, order 12 and 128-bit outward arithmetic. Its initial-match
gates must pass before a protocol can be prepared. The whole endpoint may
admit any existing protected basin; no preferred sector, new phase law,
retry or changed horizon is introduced. The reference and zero-form records
remain immutable and are not re-executed by this control.

#### Retained result: equal initial energy, different limiting identity

The first evaluation validates all 256 whole-time steps to time 32, without
retry or unresolved interval. Its entire endpoint box lies in the consensus
rectangle `S`, with joint storage conservatively enclosed in
`[1.37994484500693, 1.37994484500711] < 7`. The smallest rectangle margin is
greater than `1.05417`; the maximum endpoint coordinate width is below
`1.889e-14`. All resultants remain positive, with a whole-run lower bound
greater than `0.03080034`. The smallest strict Picard inclusion margin
exceeds `3.0648e-5`. Conditional continuation therefore converges to consensus.

| Initial form, with identical phase/support/capacity/law | Initial joint storage | Initial loss | Proved limiting basin |
| --- | --- | --- | --- |
| Reference contrast | About `8.73652606007` | `(2/3)*(1+2^(-54))^2` | Positive aligned twist |
| Zero contrast | About `7.93652606007` | `0` | Consensus |
| Negated reference contrast | Exactly equal to reference | Exactly equal to reference | Consensus |

The reference/reversal pair now disproves a deterministic limiting-basin
selector using only initial `(E,E_dot)`, even with the phase geometry and
other model inputs supplied. It does not disprove every possible scalar
encoding, a classifier using additional state, or a model using the subsequent
energy history. The initial acceleration of energy already differs, and no
claim of identical losses throughout the trajectory was made.

The retained report has `terminal_basin_admitted=true`, `target_sector=0`
and `initial_storage_selector_refuted=true`. Its historical positive-pattern
flag is false because consensus is a different resolved basin. The earlier
finite reference verdict also remains false; the later continuous proof and
these controls do not rewrite that experiment's acceptance criterion.

| Immutable control evidence | SHA-256 |
| --- | --- |
| [Frozen prospective protocol](../../docs/assets/relational_reversed_form_response/result.protocol.json) | `050e265497f9bd7c483c20ab77bbf58baf763bb13e42575d8227bca9801bd9a8` |
| [Validated response](../../docs/assets/relational_reversed_form_response/result.json) | `f3de2b74e9f0ab9ba3b621dfc471266e22f23a391bb1f4f76153538d924e2963` |
| [604 fingerprinted source files](../../docs/assets/relational_reversed_form_response/result.source.zip) | `9816d178479941e1d9ac007452b06c41d75e2fe7eb5d20d83c81bb1c42494e20` |

An independent retained-record audit checks every source hash, the sign-only
intervention, initial identities, all strict Picard inclusions, chained
endpoints, resultant bounds and terminal inequalities, without rerunning the
trajectory. A **retrospective** consequence of the same whole-time tubes is
that every raw oriented ring gap `(-2a,a-b,b,b,a-b)` stays strictly inside
`(-pi,pi)`, with margin greater than `0.02616569`. Their sum telescopes to
zero, so both ring windings remain zero throughout `[0,32]`. This is stronger
than endpoint sampling but was not a frozen winding-time prediction. The
zero-form control's separately reported transient winding remains a different
case.

#### What the three preparations establish together

The uniform-form result shows how nonbalanced phase geometry first produces
form contrast. Form contrast in turn moves relative phase; reversing it
changes both that direction and the signed exchange `J` while preserving
initial scalar storage and loss. Dissipation and the existing circulation
classification then allow distinct final phase geometries with uniform form.
Thus transient form can affect the maintained full-state identity even when
the final scalar EPI profile is uniform. The earlier regional interaction
result supplies transmission through the same declared law and supplied
connections. These mechanisms share one model; they do not require a new
operator, telemetry-based selector or primitive state variable.

This closes the named energy-versus-direction discriminator. It does not
close arbitrary graph formation, autonomous support/capacity evolution,
physical identification or selection of this constitutive law by nature.
The [shared work integration](#relational-work-integration) now exposes signed
exchange and regional boundary accounting without extending this preparation
into another basin sweep. The [composition result](RELATIONAL_PATTERN_COMPOSITION.md)
now answers the first-variation question and isolates the nonlinear geometric
response; the sole execution plan owns the remaining predictivity gate.
