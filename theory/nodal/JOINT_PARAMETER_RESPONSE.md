# Joint form, phase and capacity response

Signed representation, pressure derivatives and finite phase/capacity compatibility.

Section numbers are stable locators across this document family.
The [parameter reference](../NODAL_PARAMETER_FOUNDATIONS.md) owns the
reading map; the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns active tasks. Each result retains its stated model and scope.

## 9. Signed EPI and phase: an explicit representation test

Consider the proposed observation `z_i=x_i*exp(i*theta_i)`, retaining capacity,
support and coefficients. It might seem to package form and phase in one
complex coordinate. A necessary condition for a closed differentiable law on
this observation is that two underlying states with the same `z` give the
same derivative of every smooth function of `z`. In particular,

\[
|z_i|^2=x_i^2,\qquad \frac{d|z_i|^2}{dt}=2x_i\nu_i p_i.
\]

The angular velocity cancels from this identity. We can therefore test
projectability without assuming any missing phase law.

### 9.1 A sign ambiguity with different radial responses

At fixed support write `p=-w_e*L*x+F(theta,nu,G)`. Under the common change
`T:(x,theta)->(-x,theta+pi*1)`, exact circle algebra gives the same `z`.
The non-EPI source `F` is unchanged under a common phase rotation on its
regular branch, while the EPI difference changes sign. Thus

\[
p(Ts)=-p(s)+2F(s),\qquad
\left.\frac{d|z_i|^2}{dt}\right|_{Ts}
-\left.\frac{d|z_i|^2}{dt}\right|_s=-4x_i\nu_iF_i.
\]

This supplies an obstruction whenever the displayed product is nonzero.
It excludes a closed law on this observation for the stated domain, regardless
of any real angular velocities one might subsequently propose.

A bounded two-node witness uses all four configured channel weights `1/4`,
capacities `(1,2)`, unit reciprocal conductance and phase consensus:

| State | EPI | Phase | Non-EPI source | Pressure | Squared-modulus rate |
| --- | --- | --- | --- | --- | --- |
| A | `(1/4,1/2)` | `(0,0)` | `(1/4,-1/4)` | `(5/16,-5/16)` | `(5/32,-5/8)` |
| B | `(-1/4,-1/2)` | `(pi,pi)` | `(1/4,-1/4)` | `(3/16,-3/16)` | `(-3/32,3/8)` |

Both exact observations are `(1/4,1/2)`, both phase gaps satisfy strict U3,
and all form values lie inside the default scalar interval. No clipping,
selected operator word or trajectory is needed for the obstruction. The
pure-EPI global-flip control has `F=0` and no such radial defect; that control
does not prove full complex-state closure or specify angular motion.

The proof uses exact circle components `(1,0)` and `(-1,0)`. Executable
pressure controls separately materialize uniform binary64 phase values and
verify zero relative-phase pressure; they do not assert that evaluating
`exp(i*float(pi))` gives exactly `(-1,0)`.

### 9.2 Zero form does not erase a node's phase

For `x=(0,1)`, `nu=(1,1)` and the same weights, change the zero-form node's
phase from zero to `alpha`, where `0<alpha<pi/2`, holding the other phase zero.
The complex observation remains `(0,1)`. On the singleton-neighbor phase
branch, the positive node's pressure changes by `w_phi*alpha/pi`; its
squared-modulus rate changes by `2*w_phi*alpha/pi`. Phase at a zero EPI
coordinate thus affects an observable neighbor response.

The production represented quarter-pi control has pressures `(1/4,-1/4)`
versus `(3/16,-3/16)` and radial rates `(0,-1/2)` versus `(0,-3/8)`.
Those exact rational outputs describe the retained binary64 calculation;
the symbolic proof above does not depend on its transcendental rounding.

This also locates a conceptual choice. If a new theory declares phase absent
at zero form, its pressure law must respect that equivalence. The current
phase channel does not. Calling the zero coordinate a vacuum cannot resolve
the conflict; the node, its neighbors and its other attributes still exist.

### 9.3 A faithful encoding, without a new law

Away from zero and on a fixed sign sector, the polar map is locally invertible:
its Jacobian determinant is `x`. Globally it identifies opposite signed forms
with a half-turn of phase; at zero it loses an entire phase circle.
Restricting to positive form needs a separate invariant-domain argument.
Indeed `x=(0,1/2)`, `nu=(2,1)` at phase consensus gives
`xdot=(-1/4,1/8)` in the same pressure model: the nonnegative boundary points
outward. A clipping rule would be an additional map, not a proof of invariance
of the unclipped law.

A faithful node representation is `(x,u,nu)` with `x` real and `|u|=1`.
It retains phase at zero form and is the cylinder embedding
`(x,cos(theta),sin(theta))` with the existing capacity coordinate. Its two
form/phase tangent columns have identity Gram matrix, including at `x=0`.
Equivalently retain `(z,u,nu)` with `z*conj(u)` real and recover
`x=Re(z*conj(u))`. This is an encoding of the existing state, not an assertion
that extra physical degrees of freedom have emerged.

For a differentiable unit-circle path, `u_dot=i*omega*u` for some real
`omega`; that kinematic identity does not determine `omega`. The EPI law
still supplies only `xdot=nu*P`. Capacity, phase, changing support and history
require their own justified closure or reduction. The linear `epi_memory`
and `operator_quotient` owners supply the established closure principles;
their linear certificates cannot certify this nonlinear observation without
the displayed fiber test.

Five detached controls in
[test_epi_phase_representation_scope.py](../../tests/physics/test_epi_phase_representation_scope.py)
cover these statements. They reject this particular lossy packaging, not every
complex-valued formulation of TNFR, and do not derive autonomous NFR formation.

## 10. Joint pressure response and the capacity product rule

### 10.1 One identity, with three distinct neighborhood responses

Fix unique support, active conductance edges and effective channel coefficients
on a differentiable segment; positive symmetric conductances may vary smoothly.
Write `U` for the unweighted unique-neighbor averaging matrix, `L_U=I-U`,
and `L_W` for the weighted EPI random-walk Laplacian. On a regular phase
branch with nonzero neighbor resultants, let `R` be the circular-mean
derivative from [phase_response](../../src/tnfr/physics/phase_response.py) and
`J=R-I`. The current pressure realization is

```text
p = -e L_W x + w_phi g(theta) - v L_U nu - w_topo L_U k,
Dg = J/pi,               x_dot = nu*p.
```

Here `k` is the fixed unique-support degree. Products between nodal vectors
are componentwise. Set `q=theta_dot/pi` and `a=nu_dot`; these are supplied
velocities, not new state primitives or selected laws. Differentiation gives

```text
p_dot = -e L_W (nu*p) - e L_W_dot x + w_phi J q - v L_U a,
x_ddot = a*p + nu*p_dot.
```

The first term transports the current EPI rate. The second differentiates
the normalized transport geometry. For `d_i=sum_j W_ij` and
`g_E_i=sum_j W_ij*(x_j-x_i)/d_i`, it is

```text
(-L_W_dot x)_i = [sum_j W_dot_ij*(x_j-x_i) - d_dot_i*g_E_i]/d_i.
```

Its zero-strength rows are zero within this fixed active-edge domain.
The row-strength correction is necessary; differentiating only the weighted
numerator changes the law. The phase derivative uses
`R`, not the final IL/UM stage Jacobian, whose merge and gain have a different
meaning. Capacity uses `L_U`, not `L_W`: zero-weight edges still participate
in the capacity and phase channels, parallel edges count once in unique
support, and self-neighbors count once. Fixed topology has zero derivative;
this does not say its baseline pressure contribution vanishes. The product
term `a*p` is essential even when capacity differences remain unchanged.

With `[x]=[p]=X`, `[nu]=T^-1`, `[q]=T^-1`, `[a]=T^-2`, the coefficient units
are `[e]=1`, `[w_phi]=X` and `[v]=X*T`. Thus all pressure-rate terms have units
`X/T` and both acceleration terms have units `X/T^2`. Expressing angular
velocity in pi units preserves exact rational controls without pretending
that represented `float(pi)` is exact pi or identifying phase speed with nu.

`derive_joint_nodal_response` centralizes this conditional identity. Optional
`conductance_rates` align with the materialized effective-edge entries,
including both directions; omission retains the fixed-conductance behavior.
The returned `transport` is the shared derivative record. The joint result's
computed
`epi_flow_pressure_rate` and `epi_geometry_pressure_rate` sum to the complete
`epi_pressure_rate`, without duplicating the transport formula. Input edges
and their rates are associated before canonical ordering; both may be
reordered together without changing the result. The shared
[transport derivative](../../src/tnfr/physics/support_transport.py) materializes
them once and rebuilds the snapshot's untrusted cached fields. The joint
observer verifies that the declared phase-reference neighborhoods match
the unique support.
Its first domain requires nonempty neighborhoods; isolates are rejected,
not assigned a fictitious phasor mean. Coefficients are never renormalized.
The supplied stored pressure is declared input. The observer does not verify
that it is a refreshed canonical pressure, that the ideal cosine Gram belongs
to the captured phases, or that the live wrap branch is regular. Tests obtain
compatible control states through
[forcing_realization](../../src/tnfr/physics/forcing_realization.py) and separately
check its fresh/stored/represented residuals. A returned identity is neither
a derivative of binary64 arithmetic nor a runtime execution certificate.

The effective channel coefficients are held fixed after configuration;
this observer never normalizes them. Changing coefficients adds their
derivatives times the corresponding channels;
support changes, operator jumps and REMESH history require their existing
reset/event descriptions. Finite sampled capacity secants are not silently
substituted for a smooth instantaneous `a`. A zero-conductance support edge
cannot gain conductance through this fixed-active-edge derivative. Changing
transport weights may also change tetrad metric distances when explicit
lengths are absent; no fixed-metric tetrad claim follows from this identity.

### 10.2 Pressure-invisible motion can change form acceleration

At the same initial state and identical supplied conductance rates,
differences between two supplied phase/capacity velocity pairs
obey

```text
delta(p_dot) = w_phi J delta(q) - v L_U delta(a),
delta(x_ddot) = nu*delta(p_dot) + p*delta(a).
```

Common rotation lies in `ker J`. A common capacity rate `delta(a)=c*1` also
leaves `p_dot` unchanged, but changes acceleration by `c*p`. Thus observing
only pressure response does not identify the full response of form. If all
`p_i` are nonzero, equality of both pressure rate and EPI acceleration forces
`delta(a)=0`; for positive phase weight the remaining freedom lies in `ker J`.
At zero-pressure coordinates that inference degenerates. In particular, at
`p=0`, pressure tangency implies zero instantaneous EPI acceleration but does
not prove that the trajectory remains on the zero-pressure set.

The existing two-completion P2 control in
[test_constitutive_capacity_scope.py](../../tests/physics/test_constitutive_capacity_scope.py)
now uses the shared owner instead of a separate acceleration calculation.
Its identical initial state, tetrad and EPI rate still permit different
accelerations. These are countermodels to a claimed implication, not two
outcomes of one fully specified autonomous law.

### 10.3 Which phase-source motions can capacity compensate?

On connected undirected unique support with positive degrees `d_U`,
`range(L_U)={b: d_U^T b=0}`. This follows from
`diag(d_U)L_U` being the connected symmetric combinatorial Laplacian.
For positive `v` and `w_phi`, a capacity velocity can cancel a supplied
phase-source velocity precisely when

```text
d_U^T J q = 0.
```

If solvable, `v L_U a=w_phi J q` determines `a` only up to a uniform vector.
This is an algebraic instantaneous criterion; positivity boundaries and a
physical evolution law add independent requirements. The joint source map
`[w_phi J, -v L_U]` has rank `n-1` if `d_U^T J=0`, and rank `n` otherwise:
the capacity image already fills the codimension-one hyperplane, and a
phase column outside it completes the range. Its tangent kernel therefore
has dimension `n+1` or `n`, respectively. Weighted conductance strengths
cannot replace the unique-support degrees in this statement.

The undirected-support premise must be checked independently. Symmetric
effective conductance alone is insufficient: reciprocal unit edges
`0<->1<->2` plus a zero-weight arc `0->2` retain symmetric EPI transport,
but their directed support has `d_U=(2,2,1)` and
`d_U^T(U-I)=(-1,0,1)`. The shared joint derivative correctly covers that
snapshot; this undirected capacity-range theorem does not. Its portable
regression prevents transferring the theorem merely from conductance admission.
Likewise the existing `derive_forced_support_balance` solves a weighted EPI
Poisson problem. A capacity reconstruction must reuse the exact algebra with
the support operator and its own gauge, not pass the live weighted EPI
problem unchanged.

At consensus `R=U`; cancellation is equivalent to
`w_phi*q+v*a` being spatially constant. Away from consensus it can fail even
strictly inside U3. On a three-leaf star with phases `(0,0,0,alpha)`,
`cos(alpha)=3/5`, `sin(alpha)=4/5`, the center row is
`R_0=(0,13/37,13/37,11/37)` and each leaf row points to the center. Therefore

```text
d_U=(3,1,1,1),        d_U^T J=(0,2/37,2/37,-4/37).
```

All edge gaps are below pi/2 and all resultants are nonzero. For `q=(0,0,0,1)`
the weighted sum is `-4/37`, so no capacity velocity cancels this phase-source
motion. This extends the same strict-star geometry used by the
[reciprocal-closure audit](../TNFR_VARIATIONAL_PRINCIPLE.md#13-forced-potential-family-and-reciprocal-closure-constraints)
without assuming a variational law or adding a controller.

There are also exact finite compatible families. In one fixed normalized
structural unit on unit P2, choose `e=1/2`, `w_phi=v=1/4`, `w_topo=0`, and

```text
x=(1/4,0),  theta(t)=pi*t*(1,-1),  nu(t)=(1-t,3/2+t),  |t|<1/4.
```

Singleton means give `g=(-2t,2t)` and all pressure channels sum to zero for
the whole interval. Capacity remains positive and U3 strict, so the constant
nonuniform form satisfies the nodal equation along this supplied family.
Holding capacity fixed was essential when applying the earlier phase-only
rigidity results to a held total source; joint compensation supplies another
possibility. The family does not select
its own angular speed or capacity law, establish stability, or explain its
preparation. It is a finite compatibility control, not autonomous emergence.

### 10.4 Full tetrad and the remaining closure obligation

The same primitive dependencies must be carried into the tetrad. In the table,
`delta_ij=wrap(theta_j-theta_i)` retains the oriented separation:

| Field | Conditional response and additional domain |
| --- | --- |
| `Phi_s` | At fixed metric, `Phi_s_dot=B_G*p_dot`. Actual telemetry reads stored pressure; refresh and metric provenance remain necessary. Moving metric adds `B_G_dot*p`. |
| Phase gradient | At nonzero wrapped gaps away from the cut, its derivative is the neighbor mean of `sign(delta_ij)*(theta_dot_j-theta_dot_i)`. At zero gaps, the absolute value generally has only directional derivatives. |
| `K_phi` | On its regular branch, `K_phi_dot=(I-R)*theta_dot=-pi*J*q`. Resultant and wrap boundaries remain explicit. |
| `xi_C` | The implemented fit uses static pressure-coherence products and metric distances, not a phase autocorrelation. Differentiating it needs absolute-pressure, sample/bin and estimator-branch conditions; the spectral fallback is a different observation. |

Consequently a smooth pressure derivative is not automatically a smooth
whole-tetrad evolution, and pressure-invisible phase motion can still change
local phase observations. The
[source tangency identity](FORCED_SOURCE_AND_CLOCK.md#22-source-tangency-without-a-telemetry-controller)
and reciprocal variational requirements constrain a proposed completion;
neither selects the missing velocities. The shared exact controls are in
[test_joint_nodal_response.py](../../tests/physics/test_joint_nodal_response.py).
The single execution plan records the open closure obligation.

### 10.5 Geometry response is an input to closure, not its selection law

The integrated smooth-conductance response reuses the transport derivative
already present in the repository. It closes an accounting omission in the
joint observer, not the missing evolution law. Positive weights, fixed
incidence, a regular phase chart and declared differentiable rates remain
premises. No connection intensity is chosen to create a desired response.

A common instantaneous conductance scaling `W_dot=c*W` gives
`L_W_dot=0`: numerator and row-strength effects cancel. Yet its explicit
Dirichlet work is `c*E_D`, which can be positive. Thus a rise in this
geometry-dependent energy need not alter the EPI pressure or mean
acceleration. Nonuniform edge changes can instead alter both, even when
phase and capacity velocities vanish. Full pressure and acceleration, not
an energy sign alone, identify which channel supplies the response.

One exact control uses unit P3, `x=(3/2,1,0)`, `nu=(1,2,4)`, consensus
phase, `e=1/2`, `w_phi=v=1/4` and zero topology weight. Initially `p=0`
because `x=2*1-nu/2`. A declared unit rate on edge `0--1`, zero on
`1--2`, with zero phase/capacity rates gives `p_dot=(0,3/16,0)` and
`x_ddot=(0,3/8,0)`, so the ordinary mean acceleration is `1/8`.
This separates instantaneous zero pressure from invariance under changing
conductance. It supplies no law causing that edge rate.

The [joint geometry controls](../../tests/physics/test_joint_geometry_response.py)
differentiate the pressure expression independently, retain the full mean
and product rule, and check compatibility with the zero-rate default.
The fixed-conductance binary64 Dirichlet observer separately rejects any
represented asymmetry. Its former approximate symmetry admission could
report a zero identity residual although its displayed energy derivative
was different; the [boundary regression](../../tests/physics/test_heterogeneous_vf.py)
now preserves that distinction. Numerical energy diagnostics remain
different from the exact rational transport budgets.

The conservation, variational and scale-reduction owners provide no missing
rate selection here. Their residuals, supplied Hamiltonians and projected
fine-model rates depend on an already specified evolution. Using their
outputs to invent that evolution would reverse the dependency. The single
G3 plan records their ownership and keeps autonomous maintenance unresolved.

Historical source-bound validation record:
`artifacts/research/joint_geometry_admission_validation_2026_09_19.json`.

<a id="pressure-state-closure"></a>
### 10.6 Pressure as an observation or a sufficient state coordinate

The previous controls distinguish possible completions and supplied changes.
There is also a pressure-only closure obstruction within **one fixed complete
law**. Use the [native relational exchange model](RELATIONAL_EXCHANGE_ADMISSION.md#1-state-inherited-geometry-and-the-independent-premise)
on unit P2 with equal held capacity `nu>0`, effective pressure coefficients
`e,w>0`, storage scale `beta>0`, and a lifted acute phase difference
`abs(delta)<pi/2`. There are no inputs, events, clipping or support changes.
With the two nodes numbered 1 and 2, write

\[
d=x_2-x_1,\qquad \delta=\theta_2-\theta_1,\qquad
a=\frac w\pi,\qquad H(\delta)=\pi\operatorname{sinc}\delta>0,
\qquad p=p_1=-p_2=ed+a\delta.
\]

Here `sinc(delta)=sin(delta)/delta`, extended by one at zero. The full law
gives the exact relative-state equations

\[
\dot d=-2\nu p,\qquad
\dot\delta=\frac{2\nu w d}{\beta H(\delta)},\qquad
\dot p=-2e\nu p+\frac{\kappa d}{H(\delta)},\qquad
\kappa=\frac{2\nu w^2}{\beta\pi}.
\]

Choose any nonzero acute `delta` and `d=-a*delta/e`. Both nodal pressures and
both form rates then vanish, but

\[
\dot p=-\frac{2\nu w^3\delta}{\beta\pi^2eH(\delta)}\ne0.
\]

At consensus `d=delta=0`, the same pressure vector `(0,0)` has zero pressure
rate and remains stationary. Therefore no autonomous first-order map of the
instantaneous pressure vector alone can reproduce both states, even with
support, capacity, coefficients and clock fixed. The difference comes from
retained phase/form geometry, not an unspecified external phase velocity.
The compensated state has positive joint storage
`E=d^2/2+beta*(1-cos(delta))` and instantaneous loss
`E_dot=-2*e*nu*d^2<0`: phase changes while form is instantaneously stationary.
Zero net pressure is a cancellation of channels, not absence of structure.

This does **not** exclude pressure coordinates. Keeping one additional
relative coordinate gives an exact alternative chart:

\[
(d,\delta)\longleftrightarrow(p,d),\qquad
\delta=\frac{p-ed}{a},\qquad
\dot d=-2\nu p,\qquad
\dot p=-2e\nu p+
\frac{\kappa d}{H((p-ed)/a)}.
\]

The transformation is invertible on the admitted domain
`abs((p-ed)/a)<pi/2`, since `a>0`. These are the existing equations in different
coordinates, not a new pressure law or primitive variable. On this equal-
capacity P2 the common form and lifted phase means are constant; retain their
initial values too when reconstructing absolute nodal state. The two displayed
coordinates describe only the relative state.

Alternatively, eliminating the retained form difference gives exact pressure
memory. Set

\[
d_p(t)=d(0)-2\nu\int_0^t p(s)\,ds.
\]

Substituting `d=d_p(t)` into the displayed pressure row produces a closed
history-dependent equation on its admitted domain. The initial hidden value
`d(0)` must remain supplied: `p(0)` cannot determine it. This is the same
[hidden-state obligation](RELATIONAL_PATTERN_MEMORY.md#3-exact-hidden-memory-retains-its-initial-condition)
as in the joint memory construction, here with an explicit one-coordinate
elimination. It is not a memory-free pressure law, a claim that one hidden
scalar suffices on arbitrary networks, or a reconstruction of pressure from
an evaluated form derivative.

The [substrate and scale distinction](../FUNDAMENTAL_THEORY.md#29-assumed-substrate-and-emergence-between-scales)
still applies. A uniform form origin, pressure cancellation and an empty
support are different statements. This calculation neither identifies a
physical vacuum nor derives nodes, support or an external source from
pressure alone. The [focused native controls](../../tests/physics/test_relational_pressure_state.py)
check the cancellation and sufficient-coordinate identities against the
shared evaluator, with represented arithmetic kept separate from this
ideal continuous derivation.

## 11. Finite joint phase-capacity source compatibility

The instantaneous condition in section 10.3 has a finite algebraic counterpart
without choosing velocities. Fix a connected reciprocal unique
support, with nonempty neighbor rows, and fixed channel coefficients in one
declared unit system. Let `C_ij=1` when `j` is in row `i`, `d_i=|N(i)|`,
`D=diag(d)`, `B=D-C` and `L_U=D^-1 B`. Self-neighbors count once and parallel
neighbors once; zero-conductance support edges remain present. Symmetry of
positive EPI conductance does not establish reciprocity of this support.

For an oriented phase-pressure vector `g`, held non-EPI source `f`,
`w_phi,w_topo>=0` and `v=w_vf>0`, the finite level equation is

\[
f=w_\phi g-vL_U\nu-w_{\rm topo}L_Ud.
\]

This is a source equation, not the nodal time-evolution law. A zero total
pressure additionally requires `f=w_epi*L_W*x`. The existing weighted EPI
balance owner solves a different problem with a different invariant measure.

### 11.1 Necessary and sufficient condition and all capacities

Define `r=w_phi*g-w_topo*L_U*d-f`. Then

\[
\exists\nu:\ vL_U\nu=r
\quad\Longleftrightarrow\quad
d^\top r=0
\quad\Longleftrightarrow\quad
d^\top(w_\phi g-f)=0.
\]

**Proof.** Reciprocal connected support makes `B` a symmetric positive
semidefinite Laplacian with kernel `span{1}`. Thus `B*nu=D*r/v` is solvable
exactly when its right-hand side sums to zero. Also `d^T L_U=0`, so the fixed
topology channel changes the solution profile but cannot remove an
incompatible weighted phase-source mean. This includes a self-supported
singleton; an isolated node is outside the stated domain.

There is exactly one support-degree-centered solution `z`, and all solutions
are

\[
z=(vB+dd^\top)^{-1}Dr,\qquad d^\top z=0,
\qquad \nu=z+c\mathbf1.
\]

The displayed rank-one term fixes a computational coordinate in the declared
unit system; it is not a new physical interaction or source coefficient.
Its matrix is positive definite because the only null direction of `B` is
not orthogonal to `d`. Multiplication by `1^T` proves `d^T z=0`, after which
the solve reduces to the original equation. An incompatible residual is
reported without a profile. Subtracting its mean would change the held
source and would answer a different question.

The uniform coordinate `c` leaves this source unchanged, but it is not a
physical equivalence: away from zero pressure it changes `xdot=nu*p` by
`c*p`. Neither the centered representative nor a minimum-norm solution is
a derived capacity-selection law.

### 11.2 Positive capacity and declared bands

Strict positivity requires `c>-min(z)`. For optional finite closed nodewise
bands `0<=ell_i<=nu_i<=u_i`, put

\[
L=\max_i(\ell_i-z_i),\qquad
U=\min_i(u_i-z_i),\qquad a=-\min_i z_i.
\]

The allowed shifts are `[L,U]` intersected with `(a,infinity)`. Since
`ell_i>=0`, `L>=a`: the lower endpoint is included precisely when `L>a`.
The intersection is nonempty exactly when `L<=U` and `U>a`. Missing upper
bounds mean no upper limit; every compatible finite profile then admits
positive capacities. A closed singleton above `a` is allowed; the singleton
at `a` is not. These are supplied admissibility bands, not inferred runtime
clipping rules or forward-invariant bounds.

### 11.3 Exact compatible P2 family and strict-U3 obstruction

The P2 family in section 10.3 supplies a compatible control for the entire
interval `|t|<1/4`. Here `t` labels configurations; assigning it a physical
clock would be an additional choice. Its data and complete reconstruction are

```text
w_epi=1/2,  w_phi=v=1/4,  w_topo=0,
x=(1/4,0),  theta/pi=(t,-t),  g=(-2t,2t),  f=(1/8,-1/8),
z=(-1/4-t,1/4+t),  nu=z+c*1.
```

The choice `c=5/4` recovers the previously supplied `nu=(1-t,3/2+t)`;
it is not selected by the theorem. Bands `[3/4,7/4]` on both nodes give
`c in [1+t,3/2-t]`, which contains `5/4` throughout the strict domain.
Singleton neighbor means give the exact displayed `g`, all edge gaps are
below `pi/2`, and `f=w_epi*L_W*x`, so total pressure is zero. The theorem
describes a finite set of possible states, not their preparation or motion.

A three-leaf star gives a contrasting exact obstruction with rational
phase-pressure coordinates. Set

\[
\theta/\pi=(0,1/3,1/3,-1/3),\qquad d=(3,1,1,1).
\]

The center's neighbor sum is `3/2+i*sqrt(3)/2`, whose argument is `pi/6`;
each leaf sees the center's zero phase. Therefore

\[
g=(1/6,-1/3,-1/3,1/3),\qquad d^\top g=1/6.
\]

Every edge gap is `pi/3<pi/2`, all resultants are nonzero, and the oriented
wrap charts are regular. Yet for `f=0` and `w_phi>0`, the residual is
`w_phi/6`, so no capacity vector exists. Changing the positive capacity
weight, uniform capacity, positivity bands or topology weight cannot fix
it. U3 admission is therefore insufficient for finite source compensation.
This does not rule out other source levels or other constitutive models.

The phase realizations above are analytic constructions, not rounded
trigonometric tests. Reflection preserves a cosine Gram and its response
matrix while reversing `g`; a Gram alone cannot authenticate the oriented
source used in this theorem.

### 11.4 Shared implementation, tetrad and remaining freedom

`derive_phase_capacity_balance` in
[phase_response.py](../../src/tnfr/physics/phase_response.py) reuses snapshot
rebuilding, the unweighted support-gradient owner, exact scalar/vector
admission and the existing rational inverse. It explicitly checks reciprocal
connected support, reconstructs derived snapshot fields, returns exact
compatibility/equation/centering residuals, and reports the full shift
interval. Declared rational inputs remain rational; represented real inputs
are treated as their exact binary64 values. The theorem also holds for exact
real data, but this executable owner does not manipulate arbitrary
transcendental phase sources. It neither certifies a phase chart nor writes
a graph or chooses a trajectory.

The existing `capture_non_epi_forcing` / `decompose_non_epi_forcing` path
can supply its represented source and coefficients. The resulting centered
profile recovers captured capacity minus its unique-support weighted mean.
That integration check is exact algebra on captured data; phase realization,
stored-pressure freshness and binary64 kernel defects keep their existing
separate scope. The portable controls are in
[test_phase_capacity_balance.py](../../tests/physics/test_phase_capacity_balance.py).

All tetrad fields retain the dependencies in section 10.4. In particular,
on the exact zero-pressure P2 family with fixed graph-distance metric, `Phi_s=0` while
the phase gradient is `2*pi*abs(t)` and curvature is `(2*pi*t,-2*pi*t)`.
The static pressure-coherence inputs and distances used by `xi_C` do not
change; availability and fit/fallback semantics still belong to that
observer. Thus constant pressure does not imply constant full tetrad.

Finite level membership is stronger than tangent cancellation at one point.
The existing higher-order escape example in
[source tangency](FORCED_SOURCE_AND_CLOCK.md#22-source-tangency-without-a-telemetry-controller)
and the phase-only double-star obstruction remain necessary boundaries.
Even finite compatible families leave angular motion, uniform capacity,
source evolution and support evolution unselected. They do not prove
autonomous generation, attraction, stability or a physical particle. The
single execution plan carries the next constitutive-closure gate.

The [restoring-contribution test](../TNFR_VARIATIONAL_PRINCIPLE.md#135-necessary-restoring-contribution-on-the-compatible-p2-preparation)
now separates this finite compatibility from a conditional joint energy
minimum. It derives the missing capacity slope and curvature requirements
without choosing their physical origin. A source-compatible P2 perturbation
can leave all four tetrad read-outs unchanged while changing that potential;
neither identical diagnostics nor zero pressure closes the autonomous laws.
