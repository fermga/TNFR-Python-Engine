# Exact scale, coherence-geometry and bridge results

This note establishes restricted scale reductions, symmetry quotients, field
geometry and retained internal dynamics under explicitly stated laws. Each
claim keeps its own domain and evidence; none establishes closure of all
13 operators or a coupled global TNFR geometry. The
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns
research priorities; the reductions below are reusable results.

The nonlinear results distinguish the normalized-sine law from supplied
alternative reciprocal mobilities. A shared storage balance does not make
their complete dynamics identical. Their support and held capacities remain
premises; quotient coordinates remove only information proved redundant
under the actual law.

### Read by mathematical question

| Question | Result owners in this note | Main boundary |
| --- | --- | --- |
| When does a coarse state predict its own evolution? | [Pure-EPI reduction](#1-pure-epi-coarse-graining), [sine inheritance](#sine-replica-inheritance), [unordered pair state](#sine-replica-unordered-state) | Block means need not close; retained internal state or memory can be necessary |
| Which geometry or execution observation is defined? | [Field geometry](#2-geometry-forced-by-canonical-coherence), [direct product](#3-dissipative-symplectic-direct-product), [inverse identifiability](#4-inverse-identifiability), [S16 endpoint](#5-executable-s16-endpoint-certificate) and [temporal boundary](#6-time-resolved-s16-boundary) | Each construction keeps its own model and observation contract |
| Can internal motion coexist with a persistent collective identity? | [Prepared pulse](#sine-replica-internal-pulse), [complete variation](#sine-replica-pulse-variation), [transverse splitting](#sine-replica-pulse-splitting), [joint persistence](#sine-replica-joint-persistence) | Exact periodicity, perturbation stability, circulation and almost-everywhere recurrence are distinct claims |
| What survives unequal capacities or asymmetric attachments? | [Capacity correlations](#sine-replica-capacity-asymmetry), [support-symmetry criterion](#sine-pair-support-symmetry), [mixed collective state](#sine-mixed-pair-state) | Discard member labels only where the complete law permits their interchange |
| Which groups and equilibria can be recognized? | [Phase pairing](#sine-phase-pairing), [acute equilibrium classification](#sine-replica-acute-critical) | Recognition does not create support or select an equilibrium |
| Can a new grouping become observable under fixed dynamics? | [Local onset and loss](#sine-pairing-transition), [whole-box finite window](#sine-pairing-window) | A changing phase-distance observation does not establish permanent capture |
| Which conclusions survive a constitutive change? | [Positive-mobility grouping and discriminator](#sine-pairing-constitutive-scope), [protected relative geometry](#sine-mobility-relative-geometry) | The common barrier survives; old means, periods and invariant-volume claims do not transfer automatically |
| Which collective actions occur or can be represented? | [Local and whole-pair AL descent](#sine-pair-emission-descent), [autonomous regional transfer](#sine-autonomous-regional-transfer) | A projected reset, a continuous internal transfer and a selected event are distinct |
| What identifies the exchanging regions when their phases coincide? | [Joint form-phase identity](#sine-joint-identity-window) | Exact reference identification persists; full-state perturbations have a separate finite-window bound |
| Does the established phase-grouping transition change joint identity? | [Same-source joint comparison](#sine-joint-grouping-comparison) | The two observations retain different information and can select different partitions |
| Can the complete law acquire a support-compatible joint pairing? | [Critical-boundary acquisition](#sine-joint-boundary-acquisition) | A supplied critical preparation crosses the observation boundary locally; the support and quotient already exist |
| Can an acquired joint identity disappear and reappear autonomously? | [Recurrent finite episodes](#sine-joint-recurrent-episodes) | Almost every state in an open acquisition neighborhood has repeated finite episodes; no chosen-state verdict, period or return deadline follows |
| Can internal structure cancel and restore a collective form current? | [Zero-resultant restoration](#sine-zero-resultant-restoration) | An isolated cancellation can end under the complete law; the phase channel and support remain, and exact interval silence cannot restart later |

These results separate **formation, observation and maintenance**. The
[conservative formation boundary](nodal/RESONANCE_FOUNDATIONS.md#conservative-formation-boundary)
shows why protection of a two-sided invariant family cannot itself explain
entry into that family from outside. The local grouping and finite-window
results instead concern a noninvariant observation of an evolving supplied
state. A formation claim must specify its initial set, target and lifetime
without treating a prepared pattern as its own explanation. The separate
[validated native-law transit](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-validated-transit)
does join formation and maintenance for a declared dissipative preparation;
its law and capture proof do not transfer to the conservative families here.

Repository graphs keep two edge channels distinct. `weight` is transport
conductance: it defines `W`, diffusion, Dirichlet energy, quotient generators
and the conductance coordinate of the fixed-topology state metric; it defaults
to one.
`length` is the independent shortest-path channel used by structural potential
geometry and is the second edge coordinate of that metric. An explicit `length`
wins there; if it is absent, path geometry falls back to `weight` for
compatibility, and then to one if neither is present.
Consequently `length` does not change EPI diffusion, while `weight` also changes
legacy path geometry unless `length` is declared. Models in which coupling and
distance differ should declare both attributes explicitly. The executable
policy is centralized in
[`_edge_semantics.py`](../src/tnfr/physics/_edge_semantics.py).

## 1. Pure-EPI coarse-graining

On fixed symmetric nonnegative conductance with positive fine row strengths
and positive capacities, take a partition with positive macro row strengths.
These hypotheses make the inverse metric and the displayed positive macro
capacities well defined. Isolates require a separate zero-row treatment and
are outside this construction. Write

```text
x' = -A x,                 A = H^-1 B,
H = diag(h_i),             h_i = d_i/nu_i.
```

For a node partition, let `P` copy one macro value to every node in a block.
The reversible block average is

```text
R = (P^T H P)^-1 P^T H,   R P = I.
```

The quotient conductance between two blocks is the sum of all micro
conductances crossing between them. If `h_bar=P^T h`, its generator is

```text
A_bar = diag(h_bar)^-1 B_bar.
```

This is again a TNFR pure-EPI nodal equation. Its macro capacity is
`nu_bar_a=d_bar_a/h_bar_a` and macro EPI is `R x`.

### Exact closure condition

The macro observable is autonomous for every micro state exactly when

```text
R A = A_bar R.
```

Equivalently in this reversible construction, the block-constant subspace is
invariant:

```text
A P = P A_bar.
```

Thus exact coarse-graining is an intertwining structural morphism, not a
fourteenth operator. The information removed by the quotient has dimension
`N-m`, where `m` is the number of blocks. A nonzero intertwining defect measures
the influence of those unresolved within-block modes; the effective dynamics
then needs memory or additional macro variables and the nodal equation is not
closed in `R x` alone.

[`certify_epi_coarse_graining`](../src/tnfr/physics/structural_morphism.py)
constructs the quotient, reports both closure defects and a clearly named
within-tolerance decision, then delegates morphism classification to the
existing certificate. The exact theorem is the zero-residual identity; a caller
tolerance cannot make a nonzero defect algebraically exact. An equitable
reflection partition of a path has zero residual to machine precision. A path partition
whose middle block mixes boundary and interior nodes is an executable
counterexample.

### Inherited topology source and same-law scope

Closure of EPI diffusion does not imply that every canonical pressure channel
can be recomputed from the bare quotient graph. This can fail even at uniform
phase and capacity, with no unresolved EPI memory. The following exact
fixed-support result uses the existing unique-neighbor topology channel,
not an additional physical interaction.

Take unit undirected complete bipartite support `K_(a,b)`, with positive
integers `a,b`, `a+b>2`, and common capacity `nu>0`. Partition into its two
sides. Hold support, capacity and common phase fixed and keep the declared
effective channel coefficients fixed. The fine unique degrees are `b` on
the first side and `a` on the second. The existing topology response is the
neighbor-mean degree minus the node's degree, hence

```text
g_topo = (a-b) on side A, (b-a) on side B.
```

The reversible projection is the arithmetic mean on each side: fine weights
are `b/nu` and `a/nu`, and each block has total metric weight `a*b/nu`.
The quotient conductance is `a*b`, its capacity is again `nu`, and

```text
A_bar = nu * [[1,-1],[-1,1]],       R A = A_bar R.
```

Indeed every fine neighbor mean is exactly the opposite block's average;
averaging each fine derivative proves the intertwining identity for all
fine EPI fields, not only block-constant ones. Uniform capacity and phase
make their gradient channels zero. With `y=R*x`, the complete enabled
pressure therefore gives the exact affine coarse law

```text
y_dot = -w_epi*A_bar*y + nu*w_topo*(a-b, b-a).
```

But the simple two-node quotient has one unique neighbor at each node,
regardless of the conductance `a*b`. Its freshly recomputed topology gradient
is `(0,0)`. For `a!=b` and `w_topo!=0`, applying the unchanged four-channel
recipe to that bare quotient loses a nonzero source. Changing only its
scalar coefficient cannot repair a zero gradient. In `K_(2,3)` at uniform
EPI, `nu=1` and all four effective weights `1/4`, the actual projected nodal
rate is `(-1/4,+1/4)` while the naive quotient rate is zero.

**Positive inheritance result.** Retaining the fine degree profile `(b,a)`
or its calculated source restores the displayed exact affine law. This is
structure inherited from the declared graph, not pressure solved from a
desired trajectory. The existing
[forced closure observer](../src/tnfr/physics/epi_memory.py) verifies its
all-state affine closure and zero hidden-to-macro coupling. Thus the result
does not refute coarse autonomy; it refutes losing inherited structure while
claiming the identical bare-graph pressure recipe. For this held domain the
source is constant; a changing support would need its own inherited evolution
and event law.

[Nine executable controls](../tests/physics/test_joint_scale_topology_scope.py)
compare the production pressure writer, exact represented-coefficient
support/forcing records and both macro constructions. They include hidden
nonuniform fine EPI, the inherited affine response, balanced bipartite graphs
and a zero-topology-weight control. The default topology weight is zero, so
this is not a defect in the default configuration. The all-state real identity
above and the finite binary64 controls have distinct scope. Neither supplies
autonomous phase/capacity/support laws, spontaneous partition selection or
physical emergence. This complements the existing phase-fiber and inherited
potential obstructions below rather than replacing them.

### Joint constitutive reduction with inherited support counts

The next constructive bridge retains structural information rather than
reapplying the fine formulas to a bare simple graph. Consider connected
symmetric positive transport and a fixed partition with exactly
block-constant primitive phases `Theta_a` and positive fine capacities
`kappa_a`. Hold phase, capacity, support and effective channel coefficients
fixed. Fine scalar EPI evolves by the declared multichannel pressure and
need not be block-constant. These held rows specify a conditional fine
model; the nodal product does not derive their stationarity.

Require equitable **unique-support** counts

```text
N_ab = number of distinct neighbors in block b of each node in block a,
s_a = sum_b N_ab.
```

Internal neighbors contribute to `N_aa`; zero-conductance support edges
still contribute to these counts. They are different from transport
conductance. On the defined circular-mean branch, the fine non-EPI channels
are block-constant and are reconstructed by

```text
g_phase,a = wrap(Arg(sum_b N_ab exp(i Theta_b)) - Theta_a) / pi,
g_vf,a    = (sum_b N_ab kappa_b)/s_a - kappa_a,
g_topo,a  = (sum_b N_ab s_b)/s_a - s_a.
```

These follow by grouping the existing neighbor sums. They do not introduce
chosen phase weights, a new pressure channel or an autonomous phase clock.
The phase formula retains the production wrap orientation; a zero phasor
resultant requires an explicit implementation policy and is outside an
exact regular-branch interpretation.

Keep `d_i` for fine **transport strength**, so `h_i=d_i/kappa_a` and
`hbar_a=sum_(i in a) h_i`. The existing cross-only quotient removes internal
transport from its displayed adjacency, giving strength `dbar_a` and
effective capacity `nu_eff,a=dbar_a/hbar_a`. Consequently

```text
sigma_a = kappa_a / nu_eff,a = (sum_(i in a) d_i)/dbar_a,
f_N,a   = w_phase*g_phase,a + w_vf*g_vf,a + w_topo*g_topo,a,
p_eff   = -w_epi*L_rw,bar*y + diag(sigma)*f_N,
y_dot   = diag(nu_eff)*p_eff.
```

This is the projected nodal law for every fine EPI state when the existing
exact EPI intertwining test passes. Indeed `R diag(nu)` applied to a
block-constant source equals `diag(kappa)` applied to that source, while
`diag(nu_eff) diag(sigma)=diag(kappa)`. Thus the rescaling is fixed by the
inherited metric and the selected transport normalization, independently
of measured rates or desired responses. Fine capacity `kappa` remains
the input to the capacity-pressure formula; replacing it by `nu_eff`
generally changes that formula.

**A second normalization obstruction.** On unit `K3`, partition
`((0),(1,2))`, uniform fine capacity one descends to `nu_eff=(1,1/2)`.
Here `N=((0,2),(1,1))`, so counted capacity pressure is still zero.
Using `nu_eff` as the capacity field on a new bare P2 instead produces
gradients `(-1/2,+1/2)`. Retaining a self-neighbor changes the second
incorrect value but does not repair the first. This issue can occur with
the default positive capacity-channel weight, unlike the earlier
zero-default-topology counterexample. It concerns a naive reduction, not
the original fine engine. The counted construction preserves the actual
zero channel without a fitted cancellation.

The shared [support observer](../src/tnfr/physics/quotient_structure.py)
owns exact multiplicities, both conductance normalizations, the inherited
metric, both capacities and counted capacity/topology gradients. It rejects
inequitable support or nonconstant block capacity; weighted EPI closure is
a separate obligation. The [joint observer](../src/tnfr/physics/joint_quotient.py)
then reuses the existing pressure capture and exact affine closure owners.
It evaluates counted phase with the same non-JIT NumPy kernel using repeated
target indices for distinct fine neighbors, rather than adding another
phase implementation.

**Numerical and dynamical scope.** Grouped phase sums can round differently
from the fine neighbor order. The observer keeps that unit-phase defect,
its projected rate contribution, fresh pressure assembly error and stale
stored-pressure residual separately. Its exact rational identities connect
those represented-coefficient models. The inherited NumPy kernel uses an
explicit zero phase-pressure extension at a represented zero resultant,
where the geometric curvature reader reports an unavailable direction. The explicit
`phase_geometry_scope` retains that distinction; successful rate identities
do not certify geometric phase availability. They are not a certificate of exact
transcendental evaluation, continuous phase evolution or repeated runtime
execution. Held phases/capacities have zero rates in this specified model;
no history or event is executed. The selected partition has not emerged.
This is a derived effective description with inherited structure, not
proof that the unchanged simple-graph engine represents every scale.

Controls are centralized in
[support tests](../tests/physics/test_quotient_structure.py) and
[joint-channel tests](../tests/physics/test_joint_quotient_contract.py).
They distinguish the counted channels from simple-graph recomputation,
including unequal phase-neighbor multiplicities and internal neighbors.
Field inheritance remains a separate requirement, addressed next.

### Joint evolving phase/form reduction on fixed K3

The counted construction also permits a joint evolving calculation on a
restricted invariant subspace. Take fixed unit undirected `K3`, the selected
partition `((0),(1,2))`, and identical positive fine capacity `kappa`. Supply
the existing U3-gated sine phase law with coupling `K>0`, identifying its
free angular rate with that fine capacity. The gate is `pi/2`. Use fresh
canonical multichannel pressure with fixed effective coefficients
`e=w_epi>0`, `w=w_phase>=0`; the capacity and topology coefficients may be
nonzero, but their gradients vanish because fine capacities and degrees
are uniform. Hold capacities, support and the pressure recipe fixed; exclude
Gamma, operator events and clipping. These are constitutive premises, not
consequences of the nodal equation alone.

Assume block-constant fine EPI and primitive phase. Write their two distinct
values as `x0,x1` and continuous phase lifts `theta0,theta1`, with

```text
q = x0-x1,                 delta = theta1-theta0,
abs(delta) < pi/2.
```

All fine neighbors are then U3-compatible. Node 0 has two neighbors with
phase `theta1`; each node in block 1 has one neighbor of each phase. Their
phasor directions are respectively `theta1` and the midpoint
`(theta0+theta1)/2` in this chart. Therefore the actual counted pressure and
the supplied averaged-sine phase law give

```text
p0 = -e*q + w*delta/pi,
p1 =  e*q/2 - w*delta/(2*pi),        p2 = p1,

x0_dot = kappa*p0,
x1_dot = kappa*p1,

theta0_dot = kappa + K*sin(delta),
theta1_dot = kappa - (K/2)*sin(delta).
```

The two fine nodes in block 1 receive identical exact-real rows, so the
block-constant subspace is invariant. This statement does not cover
within-block perturbations, arbitrary partitions or unequal capacities.
Eliminating the common coordinates yields a triangular relative system:

```text
c = 3*K/2,                A = 3*kappa*e/2,
B = 3*kappa*w/(2*pi),

delta_dot = -c*sin(delta),
q_dot     = -A*q + B*delta.
```

The geometry supplies the factors `3/2` and the unequal block rates. It
does not create a nonzero intrinsic-frequency difference. In particular,

```text
m = (x0+2*x1)/3,          m_dot = 0,
b = (theta0+2*theta1)/3,   b_dot = kappa,

x0 = m + 2*q/3,           x1 = m - q/3,
theta0 = b - 2*delta/3,    theta1 = b + delta/3.
```

Here `b` is a lifted weighted phase coordinate, not a global circular mean
or a clock derived from the tetrad. The common advancing phase still comes
from the declared microscopic law. The EPI mean is precisely the inherited
metric mean, since the fine transport metric is `(2/kappa)*I` and its block
weights are `(2/kappa,4/kappa)`.

**Why effective mobility must not be reused as an intrinsic phase rate.**
The cross-only quotient has conductance two and effective capacities
`nu_eff=(kappa,kappa/2)`. Its counted source multipliers are `(1,2)`, hence
its effective nodal-pressure vector is

```text
p_eff = (-e*q+w*delta/pi, e*q-w*delta/pi),
(x0_dot,x1_dot) = diag(kappa,kappa/2)*p_eff.
```

These unequal effective capacities derive from transport normalization;
the fine phase law still reads the common microscopic `kappa` and the
neighbor multiplicities `N=((0,2),(1,1))`. Substituting `nu_eff` for the free
angular rate on a new bare P2 would instead produce
`delta_dot=-kappa/2-2*K*sin(delta)`. It already disagrees at `delta=0` and
can manufacture a nonzero lock from a uniform microscopic clock. That
would be a changed model, not emergent detuning. Recomputing its capacity
pressure from `nu_eff` would introduce the additional false source
identified in the preceding normalization obstruction.

**Exact continuous restoration ends at consensus.** Choose
`0<rho<pi/2` with `abs(delta(0))<=rho`. The phase interval `[-rho,rho]` is
forward invariant because its vector field points inward. More explicitly,

```text
delta(t) = 2*atan(tan(delta(0)/2)*exp(-c*t)).
```

Set `mu=c*cos(rho)>0`. Since `sin(delta)/delta>=cos(rho)` on this interval,

```text
abs(delta(t)) <= abs(delta(0))*exp(-mu*t),

abs(q(t)) <= abs(q(0))*exp(-A*t)
             + B*abs(delta(0))*I(A,mu,t),

I(A,mu,t) = (exp(-mu*t)-exp(-A*t))/(A-mu)   if A != mu,
I(A,A,t)  = t*exp(-A*t).
```

The second inequality follows directly from variation of constants. Both
relative coordinates therefore approach zero, and fine EPI tends to the
constant `m`. The unique relative equilibrium in this chart is
`(delta,q)=(0,0)`, with linearized eigenvalues `-c,-A`. Common EPI level
and common phase remain separate neutral coordinates. Nonzero form
contrast is not maintained by the unequal effective mobilities. Phase may
temporarily increase `abs(q)` before relaxing, so this is not a claim of
monotone form contrast or monotone diagnostic coherence.

No additional pressure law is needed to express the joint response:

```text
p0_dot = -A*p0 - (c*w/pi)*sin(delta),      p1 = -p0/2.
```

This identity differentiates the already declared pressure along the
coupled rows. It does not add an independently adjustable force. For
`w=0`, form is isolated diffusion with `q(t)=q(0)*exp(-A*t)`; the phase
trajectory is unchanged. A runtime control must retain the same effective
`e` when disabling `w`, because setting a raw channel weight to zero can
renormalize the other channels. Moving that raw weight to the identically
zero topology channel on K3 supplies such a control.

**Ideal Euler scope.** A simultaneous exact-real explicit step is

```text
delta_next = delta - h*c*sin(delta),
q_next     = (1-h*A)*q + h*B*delta.
```

For `0<h<=min(1/c,1/A)`, the phase keeps its sign, decreases in magnitude
and remains in `[-rho,rho]`. Let `a=1-h*mu`, `b_e=1-h*A`; then

```text
abs(delta_n) <= a^n*abs(delta_0),
abs(q_n) <= b_e^n*abs(q_0)
            + h*B*abs(delta_0)*sum_(j=0)^(n-1) b_e^(n-1-j)*a^j.
```

Both terms tend to zero; when `a=b_e`, the sum is `n*a^(n-1)` for `n>=1`.
The EPI mean is preserved and the lifted weighted phase advances by
`h*kappa` per step. Since `q_next` is a convex combination of `q` and
`w*delta/(e*pi)`, every ideal contrast stays in

```text
[min(q_0,-w*rho/(e*pi)), max(q_0,w*rho/(e*pi))].
```

Together with `x0=m+2*q/3`, `x1=m-q/3`, this gives an explicit sufficient
ideal no-clipping condition by checking the two interval endpoints against
the allowed EPI range. It does not establish a future binary64 bound.

**Implementation and evidence boundary.**
[propose_joint_nodal_phase_step](../src/tnfr/physics/joint_quotient.py)
returns a detached `JointNodalPhaseStep`: its existing joint observation
retains the same-snapshot EPI pressure, effective mobility and fine capture,
while its new phase proposal compares actual fine endpoints with lifted
counted endpoints. Both phase paths reuse the arithmetic owner in
[phase_evolution.py](../src/tnfr/dynamics/phase_evolution.py); counted
neighbors retain the multiplicities supplied by the existing quotient
observer. The exact signed `phase_step_defect` records the difference
between represented endpoints, not a circular distance or an authenticated
runtime certificate. This observer does not integrate EPI or write the
graph. The [joint phase tests](../tests/physics/test_joint_phase_evolution.py)
separately pass the simultaneous pressure/capacity inputs to the actual
shared `DefaultIntegrator`, compare fine and reduced one-step endpoints,
and refresh pressure after the EPI and phase proposals. These finite
one-step and lift controls do not establish repeated runtime closure.
Binary64 trigonometry, neighbor summation, phase wrapping and Euler updates
have separate represented defects; the exact equations above are not
identities for every rounded row. No asymptotic binary64 convergence or
future gate/clip admission follows. The theorem closes the joint
calculation on the declared symmetric K3 subspace. Autonomous
capacity/support evolution, partition selection, transverse stability outside
the following fixed-law domain and a maintained nonuniform NFR remain open.

### Transverse restoration and omission error on the same K3

The preceding invariant-subspace calculation can be tested against differences
inside the selected block. Retain its fixed unit K3, common positive fine
capacity `kappa`, supplied averaged-sine phase law `K>0`, and fresh pressure
coefficients `e>0`, `w>=0`. Capacity and topology gradients remain zero; exclude
Gamma, events, changing support and clipping. Choose continuous fine phase
lifts with initial width at most `rho`, where `0<rho<pi/2`. This is a condition
on all three fine phases, not merely on their two block averages. Require the
fixed effective U3 gate to admit this width, so every fine neighbor contributes
to the supplied phase law throughout the invariant chart.

Define the visible and omitted coordinates by

```text
delta = (theta1+theta2)/2-theta0,     eta = theta1-theta2,
q     = x0-(x1+x2)/2,               u   = x1-x2,

c = 3*K/2,     A = 3*kappa*e/2,     B = 3*kappa*w/(2*pi).
```

Each node has two neighbors whose relative phases lie in its open half-pi
chart, so the neighbor-phasor direction is their midpoint. Subtracting the
actual fine rows gives the exact continuous system

```text
delta_dot = -c*sin(delta)*cos(eta/2),
eta_dot   = -K*(cos(delta)*sin(eta/2)+sin(eta)),
q_dot     = -A*q+B*delta,
u_dot     = -A*u-B*eta.
```

The minus sign in the hidden form forcing follows from the declared
orientation `u=x1-x2`, `eta=theta1-theta2`. The mean form `m=(x0+x1+x2)/3`
is conserved and the lifted mean phase `b=(theta0+theta1+theta2)/3` advances
at the supplied rate `kappa`. Reconstruction is exact:

```text
x0 = m+2*q/3,      x1 = m-q/3+u/2,          x2 = m-q/3-u/2,
theta0 = b-2*delta/3,
theta1 = b+delta/3+eta/2,    theta2 = b+delta/3-eta/2.
```

Thus an omitted form difference alone does not affect these macro rows.
An omitted phase difference does: it changes the visible phase relaxation
through `cos(eta/2)`, which subsequently changes the visible form response.
At `delta=0` that macro defect vanishes even when `eta` is nonzero. Generic
off-fiber phase states therefore lack exact two-coordinate macro autonomy;
the special zero-visible-phase case must not be used to infer it.

**An invariant chart and attraction of the omitted coordinates.** Subtract
the common rotation `kappa*t`. At a maximal lifted phase every sine term is
nonpositive; at a minimal phase every term is nonnegative. The phase width
therefore never exceeds its initial bound. In particular,

```text
abs(eta) <= rho,       abs(delta)+abs(eta)/2 <= rho,
mu = c*cos(rho) > 0.
```

To obtain a quantitative rate, use `sin(z)/z>=cos(z)` inside this chart,
continuously extended at zero, and

```text
cos(delta)*cos(eta/2)
  = (cos(delta+eta/2)+cos(delta-eta/2))/2 >= cos(rho).
```

The scalar damping coefficient in each phase row is consequently at least
`mu`. Variation of constants in the form row then yields

```text
abs(delta(t)) <= abs(delta0)*exp(-mu*t),
abs(eta(t))   <= abs(eta0)*exp(-mu*t),
abs(u(t)) <= abs(u0)*exp(-A*t)+B*abs(eta0)*I(A,mu,t).
```

Here `I(a,b,t)=integral_0^t exp(-a*(t-s))*exp(-b*s) ds` is the nonnegative
convolution already evaluated above, including `I(a,a,t)=t*exp(-a*t)`.
The selected block becomes coherent in both phase and form. The whole
triangle also reaches consensus in its relative coordinates: this is
attraction of a supplied grouping, not selection or persistent identity
of a distinct two-node object. Every relabeled pair has the same symmetry.

**A prospective absolute error for omitting the phase difference.** Let
`delta_bar,q_bar` follow the preceding `eta=0` macro model with the same
initial visible coordinates, common means and supplied coefficients.
Its phase interval also remains inside `[-rho,rho]`. The omitted term in
the visible phase equation is exactly

```text
r_delta = c*sin(delta)*(1-cos(eta/2)),
abs(r_delta) <= (3*K/16)*abs(sin(delta))*eta^2.
```

This uses `1-cos(z)<=z^2/2`, not a truncated Taylor equality. Define
`R0=c*abs(delta0)*eta0^2/8`. The phase bounds imply
`abs(r_delta(t))<=R0*exp(-3*mu*t)`. For
`D=abs(delta-delta_bar)` and `Q=abs(q-q_bar)`, the mean-value theorem gives
the scalar comparison `D_dot<=-mu*D+abs(r_delta)` in the upper-derivative
sense, including at `D=0`. Hence

```text
D(t) <= R0*I(mu,3*mu,t),
Q(t) <= B*R0*(I(A,mu,t)-I(A,3*mu,t))/(2*mu).
```

The difference of convolutions is nonnegative; equivalently the second
bound is `B*R0*integral_0^t exp(-A*(t-s))*I(mu,3*mu,s) ds`.
These expressions remain valid when `A=mu` or `A=3*mu` by the stated
continuous definition of `I`. Both bounds tend to zero. Their all-time
upper bounds are

```text
sup D <= R0/(3*sqrt(3)*mu),
sup Q <= (B/A)*R0/(3*sqrt(3)*mu).
```

The first maximum follows by maximizing
`I(mu,3*mu,t)=(exp(-mu*t)-exp(-3*mu*t))/(2*mu)`; the second uses the
positive form convolution with integral at most `1/A`. Thus the visible
omission error is quadratic in initial `eta0`, with no fitted coefficient
or reserved-trajectory adjustment. This does not make the fine state
recoverable from its block means. Against the reference lifted to the
same fine graph, reconstruction gives, for example,

```text
max_i abs(x_i-x_bar_i) <= max(2*Q/3, Q/3+abs(u)/2).
```

The omitted fine form and phase remain first-order information even when
their effect on the macro response is second order.

**Pressure and potential reuse the same error coordinates.** Set
`P=-e*(q-q_bar)+w*(delta-delta_bar)/pi` and `H=e*u+w*eta/pi`.
The fine pressure error against that lifted reference is

```text
p-p_bar = (P, -P/2-3*H/4, -P/2+3*H/4).
```

Consequently, with `M=e*Q+w*D/pi` and `J=e*abs(u)+w*abs(eta)/pi`,
its maximum absolute value is at most `max(M,M/2+3*J/4)`. Every fresh
exact pressure on this unit triangle sums to zero; its fine inverse-square
potential therefore satisfies `Phi_s(i)=sum_(j!=i) p_j=-p_i`. The same
bound applies to this potential error. Both read-outs use the original
fine graph and unit path lengths. Recomputing potential on a bare quotient
is a different observation. This identity neither bounds every tetrad
reconstruction nor removes the correlation-fit availability boundaries.

**Ideal Euler scope.** A simultaneous exact-real Euler step of the four
displayed rows preserves the lifted phase-width bound for
`0<h<=min(1/c,1/A)`. Indeed each fine phase update, after removing `h*kappa`,
is a convex combination with weights
`(h*K/2)*sin(theta_j-theta_i)/(theta_j-theta_i)` and a nonnegative
remaining self-weight. Let `a=1-h*mu`, `b_e=1-h*A`. The visible and hidden
phase magnitudes contract by at most `a`, while

```text
abs(u_n) <= b_e^n*abs(u0)
  + h*B*abs(eta0)*sum_(j=0)^(n-1) b_e^(n-1-j)*a^j.
```

For the ideal `eta=0` comparison initialized with the same macro state,

```text
D_n <= h*R0*sum_(j=0)^(n-1) a^(n-1-j)*a^(3*j),
Q_n <= h*B*sum_(j=0)^(n-1) b_e^(n-1-j)*D_j.
```

These are exact-model discrete convolution bounds. They do not combine
an exact continuous reference with an unmeasured Euler error. A condition
excluding future EPI clipping must additionally use the reconstruction
and form bounds; phase admission alone cannot supply it.

**Implementation and arithmetic boundary.**
[observe_k3_transverse_state](../src/tnfr/physics/joint_quotient.py)
reuses the fine capture to read the represented lifted coordinates,
common capacity and effective pressure recipe.
`bound_k3_transverse_euler` evaluates an exact rational upper envelope of
the ideal-Euler bounds for a declared finite step count. Its computational
domain uses the narrower sufficient raw phase-width condition `rho<=1`
radian, without wrapping the supplied chart. It replaces the transcendental
constants by the justified inequalities

```text
mu >= c*(1-rho^2/2) > 0,       B <= 3*kappa*w/(2*3),
1/pi < 1/3.
```

The first follows from `cos(rho)>=1-rho^2/2`; the others use `pi>3`.
The evaluator propagates nonnegative rational upper bounds for the two
phase magnitudes, hidden form, macro omission errors and reconstructed
pressure/potential error. It does not evaluate a new trajectory or replace
the existing phase/pressure/integrator owners. The one-radian restriction
and finite evaluator limit are implementation scope, not an inferred
structural constant or the full theorem's boundary.

The [transverse controls](../tests/physics/test_k3_transverse.py) exercise
the existing averaged-sine phase proposal, refreshed canonical pressure
and shared nodal integrator, keeping the reduced comparison fixed from its
initial projected state. The rational envelope concerns ideal arithmetic;
comparison with represented finite endpoints must retain its numerical
defect scope. Binary64 trigonometry, phase normalization, midpoint rounding
and integration defects are distinct from the exact-real and ideal-Euler
claims. An observed finite restoration or omission error is not an
asymptotic binary64 certificate. This result does not derive capacities,
support, a privileged partition, the supplied phase law or a maintained
nonuniform NFR.

The deterministic finite control uses node order `(0,1,2)`, unit conductances
and path lengths, `kappa=1`, `K=1/2`, effective `e=w=1/2`, no Gamma, hard EPI
limits `[-1,1]` and the full half-pi U3 gate. At initialization,
`x=(1/4,3/16,1/16)` and `theta=(1/4,7/8,5/8)`, giving
`(delta,eta,q,u)=(1/2,1/4,1/8,1/8)`. The comparison retains the same initial
`delta,q,m,b` and sets `eta=u=0`. Both invoke the shared simultaneous phase
proposal and `DefaultIntegrator` with fresh pressure for 32 steps at `h=1/8`;
no stochastic seed or named operator word is involved. No clipping or phase
wrap occurs in these finite traces. Final macro omission errors are:

| Coordinate | Observed absolute error | Precomputed ideal-Euler upper bound |
| --- | --- | --- |
| Phase contrast delta | 0.00010148122492026346 | 0.00022025172008861333 |
| Form contrast q | 0.00008750642343340054 | 0.00014777434227552532 |

The final hidden phase is approximately `0.0110270752` and hidden form
`-0.00620438720`; form changes sign under the phase forcing and is not claimed
to decrease monotonically in magnitude. Tests compare every finite endpoint
with explicit `1e-12` headroom and separately record signed phase-staging,
pressure-sum, field-summation and Euler defects. These observations validate
the prepared finite control, not an unmeasured runtime error enclosure.

### A prospective regional window on the unit barbell

This finite comparison uses two unit triangles joined by one unit edge,
with node order `(0,1,2,3,4,5)` and bridge `(2,3)`. Conductances and path lengths
are one. This supplied geometry distinguishes two regions up to their exchange;
it does not derive support or choose an autonomously maintained NFR. The model
retains common fixed capacity `kappa>0`, fresh canonical pressure with effective
weights `e>0`, `w>=0`, and the supplied averaged-sine phase law with `K>0`.
Capacity and topology pressure weights are explicitly zero. There are no Gamma
terms, operator events, capacity updates, clipping or controllers in the ideal
comparison. In particular, zero topology weight is a configuration choice:
the raw topology-gradient vector on this graph is

```text
(1/2, 1/2, -2/3, -2/3, 1/2, 1/2).
```

It cannot be discarded by claiming the support is regular. If a topology weight
`t>0` were retained, its source would be `-t*L*d`, and after phase synchronization
the corresponding fixed-support equilibrium would instead be
`x=constant-(t/e)*d`. That is a different declared pressure recipe.

**What can be grouped.** Let `D=diag(2,2,3,3,2,2)` and `L=I-D^(-1)W`.
The four fibers `((0,1),(2),(3),(4,5))` retain bridge endpoints and have the
counted neighbor-mean matrix

```text
P4 = ((1/2, 1/2, 0,   0),
      (2/3, 0,   1/3, 0),
      (0,   1/3, 0,   2/3),
      (0,   0,   1/2, 1/2)).
```

The equal-interior subspace is preserved by the ideal form and phase laws.
Writing its phase coordinates as `(a,b,c,d_phase)`, the first two phase rows are

```text
a_dot = kappa + (K/2)*sin(b-a),
b_dot = kappa + (K/3)*(2*sin(a-b)+sin(c-b));
```

the other two follow by reflection. The first pressure-phase row is
`(b-a)/(2*pi)` in the admitted chart; the second is
`Arg(2*exp(i*(a-b))+exp(i*(c-b)))/pi`. The internal same-fiber neighbor remains
in both the denominator and phasor sum. Replacing those counted rows by a
simple four-node path would change the model. Two triangle averages alone are
not closed: states with the same averages can have different bridge values,
and hence different boundary flux and future averages.

**Local joint response and its two clocks.** In the common rotating chart
`vartheta=theta-kappa*t*1`, the linearization at synchronization is

```text
vartheta_dot = -K*L*vartheta,
x_dot = -kappa*e*L*x - (kappa*w/pi)*L*vartheta.
```

The sorted spectrum of `L` is

```text
0, (11-sqrt(73))/12, 7/6, 3/2, 3/2, (11+sqrt(73))/12.
```

Thus `lambda_s=(11-sqrt(73))/12` is about `0.2046663546`, whereas every faster
nonconstant mode has eigenvalue at least `7/6`. Right eigenvectors are
orthonormal in `<u,v>_D=u^T*D*v`. The two initialized unit modes are

```text
r = (sqrt(73)-5)/6,
v_s = (1,1,r,-r,-1,-1)/sqrt(8+6*r^2),
v_f = (3,3,-4,-4,3,3)/sqrt(168).
```

Their first nonzero entries are positive. The slow profile itself has bridge
to interior ratio `r`, about `0.5906672909`; a dominant slow mode therefore
does not make each triangle uniform. A regional mean and its remaining
bridge profile must be retained separately.

For a modal phase coefficient `c_lambda` and form coefficient `d_lambda`,
put `A=kappa*e` and `B=kappa*w/pi`. The continuous linear response is

```text
vartheta_lambda(t) = exp(-K*lambda*t)*c_lambda,
x_lambda(t) = exp(-A*lambda*t)*d_lambda
  - B*lambda*c_lambda*integral_0^t
      exp(-A*lambda*(t-s))*exp(-K*lambda*s) ds.
```

The clocks `K` and `A` are distinct in general. At `K=A`, the integral is
`t*exp(-A*lambda*t)`; there is no singular response. The finite comparison uses
ideal simultaneous Euler rather than silently treating this continuous formula
as an exact numerical endpoint. With `a_lambda=1-h*A*lambda` and
`b_lambda=1-h*K*lambda`, its closed prediction is

```text
vartheta_lambda,n = b_lambda^n*c_lambda,
x_lambda,n = a_lambda^n*d_lambda
  - h*B*lambda*c_lambda*sum_(j=0)^(n-1)
      a_lambda^(n-1-j)*b_lambda^j.
```

For distinct factors the sum is
`(a_lambda^n-b_lambda^n)/(a_lambda-b_lambda)`; for equal factors it is
`n*a_lambda^(n-1)`. At `n=0` the forcing sum is empty. The constant mode has no
forcing or decay in this predictor. The reciprocal sine law conserves the
ideal degree-weighted rotating phase mean by edgewise cancellation, but the
nonlinear neighbor-phasor pressure need not conserve the degree-weighted form
mean. Its drift is part of the source discrepancy, not removed by recentering
the predicted trajectory after execution.

**Prospective nonlinear envelope.** Suppose the initial raw lifted phase
width is at most `rho<=1`, the full support stays U3-admitted, and
`h*K<=1`, `h*kappa*e<=1`. After subtracting the common drift, each ideal sine
Euler row is a convex combination: its edge weights are
`h*K*sin(theta_j-theta_i)/(degree_i*(theta_j-theta_i))`, interpreted continuously
at zero. They are nonnegative and their sum is at most `h*K`. The phase width
therefore remains at most `rho`. For the actual canonical range convention,
the declared finite horizon must also stay inside the selected unwrapped chart.

The sine remainder is bounded by `rho^3/6`. For a neighbor arithmetic mean `m`,
the displacements `z_j=theta_j-m` satisfy `mean(z)=0` and `abs(z_j)<=rho`.
Consequently `abs(mean(sin(z)))<=rho^3/6` and
`mean(cos(z))>=1-rho^2/2>0`. Using `abs(atan(y))<=abs(y)` gives a phasor-direction
defect at most `rho^3/[6*(1-rho^2/2)]`. The following rational bounds thus apply
to the phase-rate defect and the pressure-phase defect, respectively:

```text
R_theta = K*rho^3/6,
R_p = rho^3/[18*(1-rho^2/2)],
```

where the second uses `pi>3`. The linear Euler phase and form matrices are
row stochastic, hence contractions in the sup norm, and `||L||_infinity=2`.
The executable predictor restricts both step products to at most `1/2`, a
sufficient subdomain making every ideal modal factor nonnegative. It uses the
observed initial width, which can give tighter bounds than the chosen ceiling.
Starting from the same initial full state, telescoping gives

```text
epsilon_theta(n) <= n*h*R_theta,
epsilon_x(n) <= kappa*w*(n*h*R_p
                       + h^2*n*(n-1)*R_theta/3).
```

These bounds compare the nonlinear ideal Euler law with its fixed linear
prediction. They include all modes and possible form-mean drift. They do not
bound the difference between ideal arithmetic and an arbitrary binary64
trajectory. Pressure realization, phase staging, trigonometry, spectral
projection and nodal integration retain their separate numerical evidence.

**Frozen finite control.** The deterministic protocol fixes
`kappa=1`, `e=w=K=1/2`, `h=1/8`, `N=48`, no seed or named operator word,
the full half-pi U3 gate and hard EPI limits `[-1,1]`. Set

```text
theta0 = (1/8)*1 + (1/32)*(v_s+v_f),
x0 = -(theta0-(1/8)*1)/pi.
```

The predictor reads the materialized initial modal projections, including
their small represented residuals, before any trajectory is evaluated; it
does not replace them with ideal fixture coefficients. The chosen width
ceiling `rho=1/16` holds at initialization and the full time-six phase range
stays below the wrap boundary in the ideal model. This width ceiling, the
amplitude and the observation times are experimental design choices, not new
TNFR constants. The reference is fixed from that initial state and never
refitted to an observed endpoint.

Let `Pi_s` be the D-orthogonal slow-mode projector, `Pi_0` the constant-mode
projector and `Q_fast=I-Pi_0-Pi_s`. At the reserved steps `n=(32,40,48)`,
corresponding to times `(4,5,6)`, require all of the following:

- `||Q_fast*vartheta_n||_D <= 0.2*||Pi_s*vartheta_n||_D`;
- `||Q_fast*x_n||_D <= 0.4*||Pi_s*x_n||_D`;
- `||Pi_s*vartheta_n||_D >= 0.5*||Pi_s*vartheta_0||_D`;
- full-state sup-norm prediction errors remain within the precomputed
  nonlinear envelopes plus the explicit finite numeric allowance `1e-12`.

These tests mean that faster disturbances have subsided while a predictable
interregional contrast remains. Projector errors are at most
`sqrt(14)*epsilon` when a full-state sup bound is `epsilon`, since `sum(d)=14`.
Before execution, the displayed ideal prediction and rational envelopes give
phase-ratio upper bounds approximately `(0.151,0.104,0.079)`, form-ratio bounds
`(0.377,0.284,0.229)`, and slow-phase retention lower bounds
`(0.652,0.585,0.524)` at the three selected times. These are prospective bounds,
not measured responses. The `1e-12` allowance is an explicit finite numerical
comparison tolerance, not a proof of future runtime accuracy.

The [phase-response owner](../src/tnfr/physics/phase_response.py) supplies the
fixed linear predictor and envelopes; the
[reserved barbell controls](../tests/physics/test_barbell_joint_response.py)
exercise the existing shared phase proposal, canonical pressure refresh and
nodal integrator. No second nonlinear solver is introduced. The finite control
must retain source and execution defects, chart/clipping checks, weighted mean
and bridge-profile observations. Its acceptance or failure concerns this
configured transient regional window only, not autonomous phase-law selection,
permanent differentiated maintenance, emergent topology or physical NFR identity.

**Finite outcome under the frozen protocol.** The shared engine passed all three
reserved acceptance checks. Rounded observations were:

| Supplied time | Fast/slow phase norm | Fast/slow form norm | Slow phase retained |
| --- | --- | --- | --- |
| 4 | 0.133885 | 0.332843 | 0.662348 |
| 5 | 0.0809865 | 0.221152 | 0.597526 |
| 6 | 0.0489883 | 0.144229 | 0.539048 |

The observed initial width is approximately `0.03252325`. At time six, full-field
phase and form sup errors were `5.92e-8` and `3.05e-8`, versus respective ideal
nonlinear bounds `1.72e-5` and `2.26e-5`, with the separately declared `1e-12`
finite numerical allowance. The weighted form mean changed by `1.59e-8`:
the nonlinear source was retained, not replaced by the linear Laplacian.
The maximum observed nonlinear pressure-phase discrepancy was `1.35e-7`.
Independent 70-digit source evaluations separate this discrepancy from
binary64 source materialization (below `2.83e-16`); phase staging was below
`6.94e-16`, and signed nodal Euler defects below `4.21e-19`. These are measured
finite residuals, not interval certificates. Exact regional endpoint budgets
retain internal dissipation, bridge exchange, forcing and numerical defects.
The result admits this temporary regional window; it does not establish
permanent maintenance or an autonomous origin of the supplied phase law.

### Geometric observation can require an additional state coordinate

For the same held fine law, write `p=-e L x+f`, `x_dot=-A x+b`,
and let `K` be the actual inverse-square fine distance kernel. Observing
the averaged potential gives

```text
O = -e R K L,             o0 = R K f,
R Phi_s = O P y + O Q x + o0,       Q=I-PR.
```

EPI autonomy requires `R A Q=0`. Potential determination by the current
macro EPI additionally requires `O Q=0`. This is weaker than the full
arbitrary-pressure factorization `R K=Kbar R`; the latter must not replace
the actual observation test when only this constitutive family is claimed.

A minimal exact countercontrol uses unit-conductance P3, common capacity
one, pure EPI pressure and blocks `((0,2),(1))`. Set the existing explicit
edge lengths to `1,2`, so the endpoint distance is `3`. The transport still
closes exactly, but for the hidden vector `delta_x=(1,0,-1)`:

```text
R delta_x=0,       R A delta_x=0,       -R K L delta_x=(0,-3/4).
```

The missing geometric observation is retained by one derived coordinate,
`h=(x0-x2)/2`. With `u=(x0+x2)/2`, `v=x1`,

```text
u_dot=v-u,       v_dot=u-v,       h_dot=-h,
R Phi_s=(-37*(v-u)/72, 5*(v-u)/4 - 3*h/4).
```

The stacked linear observation `(R;O)` has rank three. Thus three scalar
coordinates are necessary and sufficient for this exact instantaneous
linear state/field description; the explicit equations also close its
evolution. No new force was introduced: the extra coordinate was already
part of fine EPI. Equal edge lengths supply the two-coordinate positive
control, with inherited kernel `((1/4,1),(2,0))`. The asymmetric lengths
are declared input geometry, not a claimed spontaneously selected metric.

[Independent rational and production-field controls](../tests/physics/test_joint_quotient_geometry.py)
verify the two domains and the decoder. This is one potential observation,
not full-tetrad sufficiency. In particular, primitive phase fields and the
nonlinear coherence-length fit have their own dependencies.

### Joint EPI/potential realization and field provenance

The [joint geometry observer](../src/tnfr/physics/geometry_realization.py)
now derives the joint observation directly from a captured held fine law.
With `A=e diag(nu) L`, it constructs

```text
O_joint = (R; -R K diag(1/nu) A),
o_joint = (0; R K f).
```

The inverse-capacity conversion is essential: the potential aggregates
pressure, whereas `A` generates the EPI rate. Using `-R K A` would silently
change the field at heterogeneous capacity. The fixed source `f` comes from
the existing non-EPI channel capture, independently of a desired response or
stored pressure. Its potential offset stays in the decoder, without adding
a constant state coordinate.

The observer delegates these rows to the single
[affine realization owner](DERIVED_EPI_MEMORY.md#11-minimal-linear-state-retaining-a-declared-observation).
That owner derives a minimal closed **linear** state `s=Cx` with
`s_dot=-G s+C b` and outputs `D s+o_joint`. Both EPI means and averaged
model potential, including their rates, are reproduced for every initial
scalar EPI in the held model. EPI partitions that fail instantaneous closure
are allowed: the same invariant-row construction retains the missing
directions. Redundant field rows do not inflate the state dimension. The
two P3 controls give dimensions two and three as derived above; dimension
three is a change of coordinates of the complete P3 form, not compression.

**Distance provenance.** The exact kernel uses the shared edge reader's
materialized binary64 lengths: explicit `length`, legacy `weight`, then
unit default, with the minimum over parallel edges. This differs from the
summed parallel conductance used in transport. Detached rational shortest
paths and inverse squares give `K`; self, zero-distance and unreachable
pairs contribute zero, following the canonical field policy. The original
raw edge attribute need not equal its binary64 materialization. Runtime
path sums, inverse powers and field accumulation are separate numerical
operations: for example, exact represented edge lengths `1` and `2^-54`
have rational path sum `1+2^-54`, while binary64 addition gives `1`.

For fresh pressure-kernel defect `delta_p_kernel`, stored-pressure residual
`delta_p_stored` and the actually materialized fine field `Phi_runtime`,
the observer verifies the exact diagnostic telescope

```text
R Phi_runtime = (O x + R K f)
              + R K delta_p_kernel + R K delta_p_stored + delta_field,
delta_field  = R Phi_runtime - R K p_stored,
O            = -R K diag(1/nu) A.
```

Thus numerical field evaluation is not silently identified with the exact
model observation. These potential corrections have no capacity multiplier.
They are detached measurements, not certified future error bounds.
[Geometry controls](../tests/physics/test_forced_support_geometry.py)
independently check capacities, source offsets, nonclosed partitions,
length/transport conventions, the separate defects and graph immutability.
Phase, capacity, support, materialized lengths and channel coefficients
remain held data. This result introduces no graph evolution, partition
selection, complete-tetrad reconstruction or physical-emergence claim.

### Remaining tetrad dependencies and the limit of linear compression

[`observe_forced_support_tetrad_dependencies`](../src/tnfr/physics/geometry_realization.py)
composes the same geometry realization with the existing phase and coherence
readers. It does not create another field implementation or reduction method.
The retained observation is `s=Cx`, with `CT=I`. Set

```text
M = -diag(1/nu) A,       Dp = M T,       Ep = M - Dp C,
p = Dp s + f + Ep x.
```

Thus all fine **model pressure** is determined by `s` exactly when `Ep=0`.
If column `j` of `Ep` is nonzero, `delta=(I-TC)e_j` satisfies `C delta=0`
and `M delta=Ep e_j != 0`: it is a same-state/different-pressure witness.
The observer tests the whole residual matrix, not just its action on the
current state, which can vanish accidentally.

There is a useful limit. On the admitted connected positive-capacity model,
`ker(M)=span(1)`. Because `C` retains `R` and `R1=1`,
`ker(C) intersect ker(M)={0}`. Consequently `Ep=0` implies `ker(C)={0}`;
the converse follows from `TC=I` at full rank. **Retaining all fine pressure
and the partition means therefore requires the full fine linear dimension.**
Pressure completeness is sufficient input for a pressure-based model
diagnostic, but it is not necessary for every nonlinear scalar diagnostic.
Failure of this test must not be labeled failure of every possible xi decoder.

The dependency contract is:

| Read-out | Retained input and result | Boundary |
| --- | --- | --- |
| Averaged model potential | The shared affine realization supplies the exact linear output and its rate | Actual stored pressure and numerical field residuals remain separate |
| Averaged phase gradient and curvature | Fine primitive phases and unique-neighbor support are held; the existing phase observer supplies these constant read-outs | They are not derived form angles; represented cancellation leaves curvature unavailable in any average containing it |
| Coherence-length fit | Nonlinear function of fine pressure products, runtime floating paths and the estimator's sample/bin policy | A current value from full stored pressure is not a forecast from `s`; model pressure completeness only supplies a sufficient input criterion |
| Coherence-length fallback | A selected normalized-Laplacian mode scale with explicit method metadata | It is dimensionless, not fitted path-length correlation; the cutoff can skip the true smallest positive mode |

The observer evaluates xi on a detached graph copy so that the canonical
spectral cache does not mutate the caller's graph. Its `observed_coherence_length`
uses **actual stored pressure**. It preserves the `autocorrelation_fit`,
historical `spectral_gap`, or `unavailable` method and associated provenance.
The fallback selects the smallest numerical eigenvalue greater than `1e-9`;
it equals the usual gap scale only when that gap passes the selection. An
independent P4 with conductances `(1,2^-40,1)` demonstrates a skipped small
mode. The fit bins equality of floating shortest-path distances, not the
rational edge-path model used by the potential realization. These are
estimation conventions, not new TNFR equations.

**An exact nonlinear obstruction on the existing P5 reduction.** Use the
unit path, common capacity one, pure EPI pressure, equal primitive phases
and reflection blocks `((0,4),(1,3),(2))`. Both its transport and inverse-square
kernel respect reflection. The existing three-coordinate orbit projection
therefore retains EPI means and averaged potential and is already invariant.
For the hidden form

```text
x(a)=a*(-2,-1,0,1,2),      Cx(a)=0,
p(a)=(a,0,0,0,-a),        R Phi_s(a)=0,
```

the retained state and all its modeled outputs are the same for every `a`.
Nevertheless the canonical coherence products differ. At `a=1`, the three
accepted distance bins `1,2,3` have exact means `(3/4,2/3,1/2)`; at `a=2`
they have `(2/3,5/9,1/3)`. Their pair counts are `(4,3,2)`; the tenth pair
at distance four is a singleton and is excluded from regression. The exact
real log-linear fits give

```text
xi(a=1)=2/log(3/2),       xi(a=2)=2/log(2).
```

The production reader agrees numerically on prepared inputs whose stored
and model pressures coincide. These are successful fits, not two values of
the spectral fallback. Hence no single-valued xi output of this three-state
realization exists on the full admitted fine-form domain. No trajectory or
new force is needed for the counterexample. Completing its pressure inputs
with the shared affine algorithm gives dimension five, as the theorem predicts.

The converse boundary matters: every finite-pressure P3 has fewer than the
required ten pairs, so its fitted branch is unavailable independently of
pressure. Its fixed selected spectral value can be constant despite missing
pressure coordinates. Also `a` and `-a` on P5 have opposite pressure but
identical xi by reflection. These facts motivate a symmetry-based nonlinear
observation test; they do not establish a minimal nonlinear state or an
autonomously selected geometry.

Controls reuse the existing P5 reduction owner in
[tetrad dependency tests](../tests/physics/test_tetrad_geometry_scope.py),
with separate [spectral selection controls](../tests/physics/test_coherence_spectral_selection_scope.py).
Held phase, capacity, source and support are still premises. This completes
the explicit observation/dependency contract, not full-tetrad minimality or
a complete evolving law for the fine substrate.

### A complete reflection-invariant form state on the retained P5

The P5 observation obstruction can be resolved without declaring the missing
shape irrelevant. Write the existing fine form as

```text
x=(a+r, p+s, c, p-s, a-r).
```

Reflection fixes `(a,p,c)` and sends `(r,s)` to `(-r,-s)`. Reuse the same
unit-support, pure-EPI P5 generator with common positive capacity `nu`.
Its equations give

```text
a_dot=nu*(p-a),       p_dot=nu*((a+c)/2-p),       c_dot=nu*(p-c),
r_dot=nu*(s-r),       s_dot=nu*(r/2-s).
```

The three quadratic observations

```text
J00=r^2,       J01=r*s,       J11=s^2
```

are unchanged by reflection. Their valid image requires **both**
`J00>=0,J11>=0` and `J00*J11=J01^2`; the determinant equality alone is
insufficient. They determine the hidden pair up to its simultaneous sign:
when `J00>0`, choose `r=sqrt(J00)` and `s=J01/r`; when `J00=0`, necessarily
`J01=0`, so choose `r=0,s=sqrt(J11)`. The zero triple gives zero hidden form.
Thus `(a,p,c,J00,J01,J11)` distinguishes exactly the two-element reflection
orbits, with the symmetric form as the one-element orbit.

The product rule derives a closed law, rather than selecting one:

```text
J00_dot=2*nu*(J01-J00),
J01_dot=nu*(J00/2+J11-2*J01),
J11_dot=nu*(J01-2*J11).
```

For `Delta=J00*J11-J01^2`, `Delta_dot=-4*nu*Delta`. More fully, the matrix
`J=h h^T` evolves by `J_dot=B J+J B^T`, where
`B=nu*((-1,1),(1/2,-1))`. Congruence with the exact linear hidden flow
preserves positive semidefiniteness and rank; the valid invariant image is
forward invariant in the exact continuous model. Keeping only the diagonal
squares fails: `(r,s)=(1,1)` and `(1,-1)` have the same squares, but
`J00_dot` is respectively `0` and `-4*nu`. The cross term carries necessary
shape information.

**What is preserved.** Reflection commutes with this transport generator
and with the unit-path inverse-square kernel. Pressure and fine potential
are therefore recovered **up to reflection**; their labels are not recovered
uniquely. Orbit-averaged potential and EPI, uniform held-phase read-outs and
the scalar coherence diagnostic are invariant. Reflection bijects the full
unordered node pairs while preserving their distances and coherence products,
so it preserves the fit input and its branch, as well as the selected spectral
fallback. This exact statement concerns the declared model and estimator;
arbitrary stored-pressure defects and numerical order effects need their own
provenance. The invariant state retains the previously missing magnitude
information, so it distinguishes the P5 `a=1` and `a=2` hidden-line examples.

**Dimension and boundary.** Six displayed coordinates obey one independent
constraint away from zero hidden form: the quotient still has intrinsic
dimension five. It removes the duplication between reflected descriptions,
not two continuous physical degrees of freedom. The observation Jacobian
has rank five off `r=s=0` and rank three on that symmetric stratum; the latter
does not make its surrounding state space three-dimensional. A chosen
representative can change sign branch at `r=0` even while the invariants
remain continuous. That change of representative is not a physical jump.
The exact symmetric stratum stays symmetric under this unforced law; no
spontaneous symmetry breaking or geometry selection has been derived.

The implementation lives beside the existing reduction in
[`p5_reduction.py`](../src/tnfr/physics/p5_reduction.py).
`observe_p5_reflection_invariants` derives the observation and its rates from
the shared fine generator. `decode_p5_reflection_invariants` supplies an exact
rational representative on the rationally liftable image, including both
axes and the zero state. A valid real image with irrational lift is outside
that exact rational decoder and is explicitly rejected; no floating square
root silently replaces it. Generated rational/represented fine snapshots
lie in the supported image. The mathematical real quotient is broader.

No integrator is added. In particular, advancing quadratic invariants with
ordinary Euler is not the pushforward of a fine Euler step: squaring
`h+dt*h_dot` also gives `dt^2*h_dot*h_dot^T`. An independently supplied Euler
step on the displayed J law need not stay on its rank-one cone.
[Independent controls](../tests/physics/test_p5_reflection_invariants.py)
verify the quotient, mixed-term obstruction, derivative and singular domains,
decoder, and field invariance using the existing readers.

### Continuity boundary for a later approximate reduction

Removing a discrete reflection redundancy differs from neglecting a small
hidden amplitude. For the preceding hidden line, write its amplitude as
`alpha>0` to distinguish it from the even coordinate `a`. Endpoint coherence
is `z=1/(1+alpha)`. The retained bin means are
`((1+z)/2,(1+2*z)/3,z)`, and their exact-real fit is

```text
xi(alpha)=2/log(1+alpha/2).
```

This expression applies while all three bins pass the current floor,
`0<alpha<10^9-1`. Hence it diverges as `alpha -> 0+`, whereas the exactly
uniform pressure case has no admissible fit and returns the distinct finite
P5 spectral scale `sqrt(2+sqrt(2))`. This is a discontinuity between tagged
estimator branches, not a divergence of the nodal state or a phase transition
proved by the dynamics. Binary64 coherence rounding can switch branches
earlier; the expression is not a binary64 asymptotic theorem. It follows
that small pressure error alone cannot justify a uniform error bound for
the unqualified displayed xi value near this boundary. Any later dynamical
reduction must retain that distinction instead of imposing continuity or
altering a nodal force to improve the diagnostic.

### Absolute error from omitting the decaying odd form on P5

Keep the preceding fixed unit-conductance P5, unit structural edge lengths,
common constant capacity `nu>0`, and pure-EPI law `x_dot=-nu*L_rw*x`.
Primitive phases are equal and held; no additional source, capacity/support
motion or hybrid event is admitted. Replacing the full form by its lifted
reflection means gives

```text
x_hat=(a,p,c,p,a),       h=x-x_hat=(r,s,0,-s,-r),
z=(r,s),               z_dot=-nu*D*z,
D=((1,-1),(-1/2,1)).
```

Both full and lifted forms follow the same declared fine law. Their difference
therefore evolves in the invariant odd subspace. This **five-to-three
approximation** discards actual form information; it is not the preceding
reflection quotient, which retains five intrinsic dimensions. It also retains
the even contrast `u=c-p`: the separate three-to-two memory approximation
below discards a different coordinate.

**Decay in the inherited metric.** Use the degree metric
`H=diag(1,2,2,2,1)`, equal to `nu` times the existing reversible metric.
The squared norm, without a factor of one half, satisfies

```text
N=||h||_H^2=2*r^2+4*s^2,
N_dot=-4*nu*(r^2-2*r*s+2*s^2).
```

In coordinates `(sqrt(2)*r,2*s)`, the hidden generator is symmetric with
diagonal entries one and off-diagonal entries `-1/sqrt(2)`. Its dimensionless
rates are `lambda_minus=1-1/sqrt(2)` and `lambda_plus=1+1/sqrt(2)`. Hence,
for every `t>=0`,

```text
N(t) <= N(0)*exp(-2*nu*lambda_minus*t).
```

The rate is sharp over the real state space, attained by a nonzero slow mode
with `s=r/sqrt(2)`. The rational evaluator does not replace this irrational
eigenvector by an approximate vector and call it exact.

**Pressure and potential errors.** Write signed errors as full model minus
lifted model. Fresh model pressure is `p_model=-L_rw*x`, independently of
`nu`. Thus

```text
delta_p=(s-r, r/2-s, 0, s-r/2, r-s).
```

The same unit-path kernel `B_ij=1/|i-j|^2` off the diagonal, `B_ii=0`, gives
`delta_Phi=B*delta_p`. On odd pressure `(q0,q1,0,-q1,-q0)`, its first two
components are `K*(q0,q1)`, where

```text
K=((-1/16,8/9),(8/9,-1/4)),
delta_Phi=(73*r/144-137*s/144, -73*r/72+41*s/36, 0,
           73*r/72-41*s/36, -73*r/144+137*s/144).
```

Cauchy-Schwarz in the hidden metric `diag(2,4)` bounds a row `(u,v)` by
`(u*r+v*s)^2 <= (u^2/2+v^2/4)*N`. Applying it to these rows proves

```text
(delta_p_i(t))^2   <= C_p[i]*N(0)*exp(-2*nu*lambda_minus*t),
(delta_Phi_i(t))^2 <= C_Phi[i]*N(0)*exp(-2*nu*lambda_minus*t),
C_p   =(3/4, 3/8, 0, 3/8, 3/4),
C_Phi =(9809/27648, 2897/3456, 0, 2897/3456, 9809/27648).
```

The row constants are sharp instantaneous bounds in this norm; joint
attainment with the slow-rate envelope is not asserted. All three orbit
means of `h`, `delta_p` and `delta_Phi` are exactly zero. Thus exact orbit
means and orbit-averaged potential can coexist with nonzero fine-field error.
No commutation of the potential kernel with the diffusion generator is
required: reflection invariance of each suffices.

**Admitting an absolute approximation.** For positive requested tolerances
`eps_p,eps_Phi`, both infinity-norm errors are within budget for all `t>=T`,
where the sufficient analytical time is

```text
M=max(1, (3/4)*N(0)/eps_p^2, (2897/3456)*N(0)/eps_Phi^2),
T=log(M)/(2*nu*lambda_minus).
```

Zero initial hidden form gives exact agreement immediately. Nonzero hidden
form cannot give identically zero pressure or potential error at finite time;
the hidden flow and both restricted output maps are invertible. Tolerances
are approximation budgets, not new parameters in the nodal dynamics.

[`bound_p5_hidden_form`](../src/tnfr/physics/p5_hidden_form.py) derives its
hidden generator and metric from the shared P5 reduction, and the potential
from the existing exact geometry kernel. It returns rational squared bounds,
using a positive rational lower rate `g<=nu*lambda_minus` certified by the
shared semidefinite test and an upper rational enclosure of `exp(-2*g*t)`.
`sample.within_tolerances(pressure=..., potential=...)` compares those bounds
directly with squared budgets. A false result means the bounds are
insufficient, not that the actual error exceeds the budget. Conservative
rate/enclosure slack is distinct from model omission error. The evaluator
supports `nu*t<=2048`; this resource limit does not restrict the all-time
analytical theorem. Binary64 execution defects are outside both claims.
[Independent controls](../tests/physics/test_p5_hidden_form.py) cover the
shared matrices, factors, rational bounds and admission boundaries.

**Identity and tetrad boundary.** Absolute decay alone does not justify
forgetting the remaining shape. For nonzero pure odd form the lifted
approximation is zero, and the relative H-norm error is one at every finite
time. More generally, the retained nonuniform even modes decay at `nu` and
`2*nu`, both faster than the slow omitted odd mode. With that slow component
present, its fraction of the remaining nonuniform form can approach one.
The result therefore admits an absolute pressure/potential approximation,
not preservation of relative pattern identity. Held equal phases give the
same phase read-outs, but the preceding xi fit/fallback discontinuity still
precludes an unqualified uniform xi error guarantee. This is a conditional
reduction of a supplied passive law, not autonomous NFR formation,
maintenance, or a law for the full evolving substrate.

### Derived memory when closure fails

For fixed reversible pure-EPI diffusion, the unresolved coordinate
`z=(I-PR)x` can now be eliminated exactly. With `Q=I-PR`, the projected
equation has instantaneous generator `RAP`, kernel
`K(t)=RA exp(-QAQ t) QAP`, and initial hidden source
`f(t)=-RA exp(-QAQ t) Qx_0`. The convolution enters with a positive sign.
The weighted identity `H_bar K(0)=(QAP)^T H(QAP)` proves that its kernel
vanishes exactly when the quotient closes for every state.

The proof and independent P4/P5 controls are centralized in
[Derived EPI memory](DERIVED_EPI_MEMORY.md).
[`observe_epi_memory`](../src/tnfr/physics/epi_memory.py) reuses the same
partition geometry and records finite numerical samples of the full equation
and the two omission controls. Numerical generator/propagator checks reject
loss of the necessary diffusion invariants; they are not accuracy enclosures.
This is an offline observation, without a REMESH or complete-runtime claim.

For the explicit P5 partition, the same note's section 8 derives uniform and
causal error bounds for a shortened history window. Its separate rational
reference preserves the initial hidden source, encloses the actual truncated
solution and keeps evaluation width separate from model error. This restricted
approximation has no fitted memory rate or asserted REMESH correspondence.

The P5 reflection partition `((0,4),(1,3),(2,))` supplies a complementary exact
closure on a genuine three-node quotient with conductances `2,2`. Observing
only its endpoint mean and combined inner/center mean is precisely the second
reduction that generates memory. Uniform unclipped REMESH commutes with this
reflection projection and its lift; the fine stationary-history energy splits
into quotient and discarded energies. These identities reuse the existing
companion theorem in the quotient's metric. They do not identify a REMESH
echo with a forward diffusion step, as an exact causal-history counterexample
shows. See section 9 of [Derived EPI memory](DERIVED_EPI_MEMORY.md).

### Consequence for S9

The pure-EPI diffusion channel is a fixed family under every exact quotient:
its generator remains `diag(nu_bar)L_rw,bar`. This does not show that Emission,
Coherence, REMESH or the other nonlinear operators close under the same map.
Operator RG flow therefore remains open beyond this one channel.

Field inheritance is another independent obligation. On the unit triangular
prism, the triangle-mean quotient closes exactly and has zero memory kernel,
yet a self-excluded scalar macro potential reverses the sign of the averaged
fine potential. The inherited kernel retains within-block sources. Keeping
four internal modes and the mean contrast instead reconstructs full exact
model pressure and potential; primitive phase and field provenance remain
separate. See the single
[macro-state and tetrad derivation](nodal/INHERITED_FORM_DYNAMICS.md#13-faithful-macro-state-and-tetrad-inheritance-on-the-retained-prism).

### Generic fixed-vector operator test

For any declared micro map `F`, macro map `F_bar`, projection `R` and right
inverse lift `P`, two logically independent identities are relevant:

```text
R F = F_bar R,        F P = P F_bar.
```

The first makes the projected dynamics autonomous for every micro state; the
second keeps lifted macro states inside the lifted subspace. Exact strong
closure requires both. A deterministic counterexample shows that projected
closure need not imply lift invariance.

[`certify_operator_quotient`](../src/tnfr/physics/operator_quotient.py) checks
both identities globally for matrices and names every Boolean decision as
within the declared numerical tolerance. For nonlinear callables it reports only
sampled residuals and repeated-evaluation consistency on declared macro and
micro probes, including the defect
between two micro states with the same projection. A cubic componentwise map
closes on block-constant probes yet depends on unresolved within-block fibers,
providing an explicit nonlinear obstruction. Graph mutation, operator history
and nested-EPI changes are rejected as outside a fixed-dimensional vector-map
test. Thus the certificate advances S9 without claiming closure or completeness
of the 13-operator catalog.

### Circular phase extension and obstruction

Choose an open semicircle chart `q` so every relevant wrapped pairwise
difference stays on one branch. The pairwise phase-pressure realization is then

```text
r_pair(q) = -(1/pi) diag(nu_f) L_rw q.
```

It is linear and inherits the reversible pure-EPI projection and lift
identities. This exact fixed-branch statement concerns pairwise edge
differences. It is distinct from the engine's canonical phase channel, which
uses the argument of the unweighted sum of neighboring phasors:

```text
g_i(q) = -(1/pi) wrap(q_i - Arg sum_{j in N(i)} exp(i q_j)),
r_i(q) = nu_f_i g_i(q).
```

For a lifted block-constant phase field, this nonlinear channel closes on the
macro support under a sufficient fixed-support domain: neighbor-count profiles
are equitable inside each fiber, fibers have no internal edges, every active
macro neighbor has the same multiplicity within its source block, capacity is
exactly block-constant, the chart stays on one wrap branch, and all phasor resultants
are nonzero. The canonical support is the unweighted NetworkX adjacency,
including edges whose transport `weight` is zero; the reversible pairwise
projection continues to use the conductance-degree-over-capacity metric.

That lifted-subspace result does not make the projected canonical dynamics
autonomous for arbitrary micro phases. On `K3,3`, a nonconstant micro state and
its lifted representative have the same macro chart coordinate but different
projected nodal rates. This is a constructive counterexample to global
canonical phase closure: unresolved within-fiber phase affects the macro
derivative. Branch crossing, zero resultants, evolving capacity, changing
support and finite-time phase evolution remain outside the certificate.

[`certify_phase_nodal_coarse_graining`](../src/tnfr/physics/phase_quotient.py)
reports the pairwise matrix quotient, the restricted canonical lift test and
the same-macro-state counterexample without reading or changing EPI.
The capacity hypothesis uses exact represented equality within each fiber;
the separately rounded macro-capacity match has its own numerical residual.
In a `K3,3` fiber, capacities `1` and `1+2^-35` can pass a `1e-10` tolerance
while unequal lifted nodal rates still exclude exact closure. Numerical
closeness must not satisfy that algebraic premise. A small nonzero phasor
resultant can likewise fail the implementation's numerical margin without
being an exact circular-mean singularity. The regression owner is
[the phase quotient tests](../tests/physics/test_phase_quotient.py).

## 2. Geometry forced by canonical coherence

For the signed local chart `p=DeltaNFR`, `v=dEPI`, the constitutive kernel is

```text
C(p,v) = 1 / (1+|p|+|v|).
```

For `0<c<1`, its exact level set is

```text
|p|+|v| = 1/c-1.
```

It is an L1 diamond. It is smooth on each open edge and nondifferentiable at
the four axis vertices. Its Euclidean distance from equilibrium is not fixed:
it ranges from `r/sqrt(2)` to `r`, whereas its L1 distance is exactly
`r=1/c-1`. At `c=1` the level collapses to `(0,0)`.

[`coherence_level_set_geometry`](../src/tnfr/physics/coherence_geometry.py)
exposes this exact geometry. The result rejects the assumption that canonical
coherence alone supplies a smooth Riemannian manifold. It naturally supplies
an L1 gauge in the local constitutive chart. A smooth information metric on the
full graph state would require an additional modeling choice.

For a fixed nonempty network with `N` nodes, canonical mean aggregation gives

```text
C_N = 1 / (1 + (sum_i |p_i| + sum_i |v_i|) / N).
```

Thus `C_N=c` is the boundary of a `2N`-dimensional cross-polytope with
total L1 radius `R=N(1/c-1)`, intrinsic dimension `2N-1`, `4N` vertices
and

```text
f_k = 2^(k+1) binom(2N,k+1)
```

`k`-faces. Its `2^(2N)` open facets form the regular locus; every lower
face is a nonsmooth absolute-value stratum. The Euclidean radius ranges from
`R/sqrt(2N)` to `R`, the regular gradient norm is
`c^2 sqrt(2/N)`, and the superlevel set `C_N>=c` is closed and convex.

The fixed-capacity nodal equation `v_i=nu_f_i p_i` cuts this ambient level
down to

```text
sum_i (1+nu_f_i)|p_i| = N(1/c-1).
```

This is a weighted `N`-dimensional cross-polytope. The executable certificate
reports its pressure-axis vertices, Euclidean radii in the induced
`(p_i,nu_f_i p_i)` metric, face stratification and regular gradient norm.
[`network_coherence_level_set_geometry`](../src/tnfr/physics/coherence_geometry.py)
and
[`fixed_capacity_coherence_level_set_geometry`](../src/tnfr/physics/coherence_geometry.py)
make both scopes explicit. They describe one instantaneous chart; capacity
evolution, changing node count, basin boundaries and temporal attraction
remain outside the result.

### Fixed-topology structural-state metric

One such explicit choice is now available within a declared topology and label
class. For each node, form the five-nodal-channel state

```text
q_i = (EPI_i, nu_f_i, phase_i, DeltaNFR_i, dEPI_i),
```

and assign each edge both its `weight` conductance coordinate (unit when
`weight` is absent) and its effective structural-length coordinate (`length`,
then the compatibility fallback `weight`, then one). Declare one finite positive
reference scale per channel, use circular geodesic distance for phase, and
minimize the scaled Euclidean product over node and edge coordinates across all
topology- and declared label-preserving graph isomorphisms. The compatibility
API lets an omitted `edge_length` scale reuse `edge_conductance`; dimensional
studies should declare both. Because a finite graph has finitely many
isomorphisms and they act by isometries, the minimum is an exact metric on the
resulting state-isomorphism classes. The certificate also reports the two
`L_infinity` residuals of
`dEPI=nu_f*DeltaNFR`; it does not project inconsistent inputs onto the nodal
equation.

[`fixed_topology_structural_state_distance`](../src/tnfr/physics/structural_state_distance.py)
implements this quotient metric for finite simple graphs. It preserves phase
wrapping and node relabeling, accepts explicit node and edge labels, validates
finite nonnegative conductance and structural length, and rejects nonisomorphic
supports, mixed
direction and multigraphs. It reports every distance minimizer, uses an explicit
lexicographic component refinement, and withholds a node mapping when that
refinement remains ambiguous. The exhaustive
isomorphism search can be factorial. Cross-topology edit costs, nested EPI
identity and operator-history geometry remain open, so this is a structural
state metric rather than a completed global information geometry.

The structural-length coordinate is a numeric cost. A caller that requires
identical path geometry may additionally include `"length"` in
`edge_label_attributes`, which makes exact attribute equality part of the
admissible-isomorphism test rather than merely charging a nonzero distance.

## 3. Dissipative-symplectic direct product

Let `z` be the `4N`-dimensional harmonic substrate coordinate and `x` the EPI
field. Define

```text
X = (z,x),
H = (1/2)||z||^2,
V = (1/2)x^T Bx,
J = diag(J_sub,0),
G = diag(0,diag(nu_i/d_i)).
```

Then

```text
X' = J grad(H) - G grad(V)
```

simultaneously reproduces the specified auxiliary harmonic-substrate flow and
fixed symmetric EPI diffusion. The degeneracy identities

```text
J grad(V) = 0,             G grad(H) = 0
```

give `H'=0` and `V'<=0`. This is an exact metriplectic-style **direct product**.
[`verify_metriplectic_product`](../src/tnfr/physics/metriplectic.py) checks the
antisymmetry, positive semidefiniteness, degeneracies and both vector-field
residuals.

The graph's stored `DeltaNFR` initializes part of the auxiliary coordinate `z`
through `Phi_s` and `J_DeltaNFR`. The dissipative EPI block nevertheless
constructs its pressure independently as `p_epi=-D^-1 Bx` from the `weight`
conductance and supplied EPI. It does not replace the stored value. The
certificate returns
`stored_pressure_consistency_residual=||DeltaNFR_stored-p_epi||_2` and the
separate relative-tolerance decision `stored_pressure_matches_epi_channel`.
Neither is included in `is_decoupled_metriplectic_bridge`: the block-product
identity can pass while stored pressure contains other channels or is otherwise
inconsistent with pure-EPI diffusion. A caller asserting a pure-EPI graph state
must therefore require both the bridge Boolean and the pressure-consistency
Boolean. This separation is exercised by
[`test_metriplectic_product.py`](../tests/physics/test_metriplectic_product.py).

The cross blocks are zero, so the result does not derive how a pulse changes
EPI relaxation or how dissipation feeds back into the substrate. A coupled
bridge requires a TNFR derivation and preservation of the realizable graph-field
image, as well as its stated balance laws. Nonzero cross tensors alone are
insufficient. Fixed mutual-singleton P2 already excludes nonzero continuous
harmonic evolution of the same geometric read-outs; its fixed-phase pure-EPI
potential/flux sector instead has a closed dissipative law with rate
`nu_0+nu_1`. The exact proof and production regressions are centralized in
[the variational note, section 3.7](TNFR_VARIATIONAL_PRINCIPLE.md#37-p2-read-out-realizability-obstruction-and-derived-flow).
Broader TNFR-derived realizability and coupling remain open under S12.

## 4. Inverse identifiability

### Contract snapshot

[`contract_identifiability_certificate`](../src/tnfr/operators/operator_contracts.py)
partitions the catalog using only declared executable contract features. With
the richest non-tautological tuple `(channel, direction, scale, context)`, eleven
operators are singletons and one class contains both Silence and Contraction.
Both are node-scale decreases of `nu_f` observed at network context.

This is an exact negative result for instantaneous contract identification. It
does not say their trajectories are always identical: Silence targets latency,
whereas Contraction also densifies pressure. Resolving them requires temporal
state changes or a quantitative postcondition observer. Catalog coverage is
therefore distinct from catalog identifiability and from universal catalog
completeness.

### Quantitative one-step signatures

[`probe_canonical_operator_identifiability`](../src/tnfr/physics/temporal_identifiability.py)
enumerates the 13 operators from the current contract catalog and applies each
one to a fresh deterministic heterogeneous path graph at the known target node
`n_nodes//2`. Its features
contain the target-node change, the across-node mean and RMS change in the four
raw channels EPI, `nu_f`, `DeltaNFR` and wrapped phase; the corresponding target,
mean and RMS change in nodal velocity `nu_f*DeltaNFR`; and node/edge-count
changes. Operator names, glyphs and contract categories are evaluation labels
and never enter the feature matrix. Executed history is checked separately to
reject a fallback mislabeled as the requested operator.

With the declared default three probes, all 13 rows are distinct after the
declared decimal quantization. Matrix rank and affine rank are 12. Row
distinctness exactly partitions this finite generated hypothesis matrix; rank is
only a numerical diagnostic. Silence and Contraction separate because both
reduce `nu_f` while only Contraction changes EPI on these probes. The result is
measured on this finite family.

This is closed-set classification against the fixed current catalog and a known
target, not target localization or operator discovery. Although mean and RMS
summaries of the raw changes are included, the protocol does not invert from
aggregate TNFR telemetry alone: `C(t)`, Si, phase synchronization and the tetrad
are absent, and target-indexed raw coordinates are present. Identification from
those aggregate read-outs, localization of an unknown target, unseen states,
compositions and arbitrary grammar words remain open.

For any declared positive feature scales, let `delta` be the minimum pairwise
distance between these finite prototypes in the scaled L-infinity or L2 norm.
The triangle inequality gives an exact robustness statement: an additive error
strictly smaller than `delta/2` cannot cross a nearest-prototype boundary. The
noise-margin certificate compares the supplied binary64 values through exact
rational arithmetic, rounds the minimum distance downward and halves that
lower bound. It therefore reports a conservative radius and marks midpoint
ties as uncertified. Stable L2 accumulation avoids false underflow, large
declared feature scales avoid false subtraction overflow, and unrepresentable
scaled separations are rejected. The default operator probe attaches a
unit-scaled L-infinity margin after applying
the same decimal quantization as its row partition. Those unit scales define a
coordinate convention, not a calibrated sensor or process-noise law. This
handles bounded perturbations around the fixed prototypes only. It does not
provide a stochastic noise law or extend the prototypes beyond the declared
states.

## 5. Executable S16 endpoint certificate

[`certify_core_research_integration`](../src/tnfr/physics/core_research_integration.py)
turns the restricted S16 intersection into one inspectable endpoint
certificate. It accepts two independently frozen states, one partition and
explicit `StructuralChannelScales`. The inputs must be finite undirected simple
graphs with exactly the same node identifiers and bare edge support. At each
endpoint the effective positive-conductance graph must also be connected, every
capacity must be positive and frozen, and all raw state channels required by the
constituent certificates must exist. Identical bare support alone is therefore
insufficient when zero `weight` disconnects effective conductance. Conductance,
structural length and state values may differ between endpoints.

At each endpoint it runs the fixed heterogeneous pure-EPI stability certificate,
the graph-specific full-potential EPI reconstruction certificate and the
reversible pure-EPI partition certificate. It also computes the declared
fixed-topology structural-state distance between the endpoints. Finally, it
checks two state-consistency conditions independently at each endpoint: stored
`DeltaNFR` must match `-L_rw EPI`, and stored `dEPI` must match
`nu_f*DeltaNFR`, after scaling by the declared pressure and EPI-rate scales.

The result exposes all constituent certificates, raw and scaled consistency
residuals, `numerical_conditions` with fifteen named entries, and
`failed_conditions`. `joint_numerical_conditions_pass` is true only when every
entry passes. Numerical rank, closure, balance and consistency decisions use the
single declared relative tolerance; the exact metric-on-isomorphism-classes
entry remains a structural Boolean. A nonclosing partition, a pure-EPI pressure
mismatch or an independent nodal-equation mismatch blocks joint promotion while
leaving the other evidence inspectable; changed support is rejected before
composition. These positive and negative paths are executable in
[`test_core_research_integration.py`](../tests/physics/test_core_research_integration.py).

A passing Boolean demonstrates simultaneous endpoint membership in this shared
restricted numerical hypothesis class under the declared tolerance. It does not
certify a trajectory or persistence between the endpoints. Phase dynamics,
nonlinear or multichannel pressure,
operator histories and words, S15 inverse identification, REMESH/nesting,
changing support/topology and nonzero dissipative-symplectic coupling remain
open. The exact L1 coherence level-set result also remains a separate theorem;
the current S16 Boolean does not consume it. Explicit `length` values contribute
to the numeric state distance; they may still differ unless the caller includes
`"length"` among the exact edge-label constraints.

## 6. Time-resolved S16 boundary

[`certify_core_research_trajectory`](../src/tnfr/physics/core_research_trajectory.py)
extends the endpoint intersection to a finite ordered sample path without
claiming unobserved interpolation. It retains persistent node identifiers and
fixed bare edge support, applies the endpoint certificate to every adjacent
pair, and checks the left-explicit update

`EPI^(k+1) - EPI^k = dt_k diag(nu_f^k) DeltaNFR^k`

in scaled L-infinity norm. Stored `dEPI` remains an independent endpoint channel;
it is not substituted for the vector field in this step test.

Two extra stability conditions prevent a merely self-consistent data sequence
from being called stable. First, each timestep must satisfy the modal
explicit-Euler condition for its left-hand transport regime. Its stationary-mode
resolution uses a separate dimensionless tolerance relative to the fastest
decay rate; the EPI residual tolerance is not reused as an absolute frequency
cutoff. Second, all sampled regimes must share the exact projective `d_i/nu_i`
metric required by the common switching theorem. The implementation evaluates
the resulting common quadratic at every supplied state. At each snapshot it
recomputes that metric's instantaneous weighted projection onto the consensus
subspace before evaluating disagreement energy; it does not freeze the first
snapshot's consensus coordinate. This is necessary because the represented
binary64 generator can contract disagreement in the displayed metric without
preserving that metric's weighted mean as an exact rational identity. The
certificate limits both each
positive increment and the cumulative positive variation over the whole path
by one declared EPI-squared scale budget. This prevents individually small
increases from accumulating without bound as samples are added.

The Euler residual and both local and cumulative Lyapunov-budget decisions
compare exact rational values of the represented binary64 inputs with the exact
rationalization of the caller tolerance. Their exposed float magnitudes are
diagnostics only and do not decide a boundary case.

The analytic switching theorem covers arbitrary
piecewise-constant continuous solutions among this finite regime family. The
snapshot certificate only checks its supplied Euler steps: it does not prove
that the samples lie on such a continuous solution, identify a switching law
between them, or cover an unseen regime.

The companion refinement comparison certifies both paths before comparing
them. It separates the strict path tolerance from the coarser agreement
tolerance, requires every coarse time to match exactly one fine time, and
requires a smaller fine-grid maximum step. Direct EPI differences on persistent
ids determine agreement. Quotient structural distance is exposed only as a
diagnostic because a different minimizing isomorphism at each time is not a
node trajectory. The joint result also requires the caller to set
`same_dynamics_declared=True`. That recorded assertion closes the previous
semantic gap in which identical equilibrium samples from different generators
could be promoted as a refinement pair; snapshots cannot independently verify
the declaration.

The deterministic
[`161_core_research_trajectory.py`](../examples/02_physics_regimes/161_core_research_trajectory.py)
uses two nested stable Euler meshes for one non-equilibrium fixed generator and
also evaluates the analytic fixed-generator semigroup numerically through the symmetric similarity
`H^(-1/2) B H^(-1/2)`. The fine solution is closer at every noninitial common
time in this experiment. This finite result does not prove numerical
convergence or its order. Phase evolution, nonlinear/multichannel pressure,
operator histories, REMESH/nesting, adaptive/event timesteps, changing support
and nonzero metriplectic cross-coupling remain open.

<a id="sine-replica-inheritance"></a>

## 7. Nonlinear sine inheritance with retained internal coordinates

This result uses the complete
[normalized-sine relational law](nodal/RESONANCE_FOUNDATIONS.md#reciprocal-exchange),
with `e=0`, `w,beta>0`, fixed unit support, strictly positive held
capacities, one common structural clock, and no input, clipping or operator
event. It extends the existing
[synchronized replica construction](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#4-origin-units-and-exact-replication)
by retaining the coordinates transverse to that construction. It does not
transfer the native resultant-pressure law or the independently supplied
phase law used in earlier sections to this model.

### 7.1. Supplied support and an invertible description

Let a finite connected simple unit base graph have degrees `d_i>0`.
Replace each node `i` by the ordered pair `(i,+),(i,-)`. For every base
edge `{i,j}` retain all four fine edges `{(i,s),(j,t)}`, with
`s,t in {+,-}`, and no other edge. In particular, there is no edge within
a pair. The fine degree is `2*d_i`. Both members of pair `i` have the
same held capacity `nu_i>0`; different pairs may have different capacities.
The support and partition are supplied, not produced by the dynamics.

Use the exact invertible form coordinates and declared continuous phase
lifts

\[
x_{i,\pm}=X_i\pm u_i,\qquad
\theta_{i,\pm}=\Theta_i\pm\delta_i .
\]

Here `X,Theta` describe the paired means and `u,delta` their internal
differences. No fine node or coordinate has disappeared. Near synchronized
phases, choose the pair chart `|delta_i|<pi/2`, so the two phases are not
antipodal and `Theta_i` is their unambiguous circular midpoint. Beyond a
chosen chart, real-lift averages are not a globally defined circular
observation; another lift may change the midpoint by `pi`. The equations
below hold on supplied continuous lifts, with the circular state obtained
by projection, rather than by repeatedly guessing a phase unwrapping.

Put `a=w/pi`, `b=w/(beta*pi)` and
`Delta_ji=Theta_j-Theta_i`. The fine nodal rows are

\[
\dot x_{i,s}
 =\frac{a\nu_i}{2d_i}
   \sum_{j\sim i}\sum_{t=\pm1}
   \sin(\Theta_j+t\delta_j-\Theta_i-s\delta_i),
\]

\[
\dot\theta_{i,s}
 =\frac{b\nu_i}{2d_i}
   \sum_{j\sim i}\sum_{t=\pm1}
   (X_i+s u_i-X_j-t u_j).
\]

Taking their half-sum and half-difference gives the **complete nonlinear
retained-coordinate law**

\[
\boxed{\begin{aligned}
\dot X_i
 &=\frac{a\nu_i}{d_i}\cos\delta_i
   \sum_{j\sim i}\cos\delta_j\sin\Delta_{ji},\\
\dot\Theta_i
 &=\frac{b\nu_i}{d_i}\sum_{j\sim i}(X_i-X_j),\\
\dot u_i
 &=-\frac{a\nu_i}{d_i}\sin\delta_i
   \sum_{j\sim i}\cos\delta_j\cos\Delta_{ji},\\
\dot\delta_i&=b\nu_i u_i .
\end{aligned}}
\]

The identities
`sin(A+B)+sin(A-B)=2*sin(A)*cos(B)` and the cancellation of the two
neighbor form differences prove these rows directly. They are not a
tangent approximation or a fitted effective pressure. In particular,
the coarse phase row is autonomous in `X`, but the coarse form row still
consumes the internal phases.

### 7.2. Internal phase geometry modulates inherited coupling

The mean fine phasor in a pair is exactly

\[
Z_i=\frac{e^{i\theta_{i,+}}+e^{i\theta_{i,-}}}{2}
   =e^{i\Theta_i}R_i,\qquad R_i=\cos\delta_i .
\]

In the nonantipodal midpoint chart, `R_i>0` is the magnitude of this
resultant. The inherited form row and its amplitude derivative are

\[
\dot X_i=\frac{a\nu_i}{d_i}
  \sum_{j\sim i}R_iR_j\sin(\Theta_j-\Theta_i),
\qquad
\dot R_i=-b\nu_i u_i\sin\delta_i.
\]

Thus the factor `R_i*R_j` is calculated from retained internal phase
geometry. It is a dynamical contribution to the paired response, not a
threshold that activates a control action. Its evolution still requires
internal information. Using this resultant does not justify dropping
`u_i` or the phase difference sign, and no general closure in
`(X,Theta,R)` is claimed.

The original fine support remains fixed. The varying factor neither
creates a new edge nor derives an occurrence time for one. Moreover,
`Theta_dot` still contains the unmodified base Laplacian and the original
degree `d_i`. Replacing every adjacency or degree by `R_i*R_j` would
change that row and define another model. This is an inherited
phase-to-form interaction with internal state, not the unchanged bare
coarse law on an arbitrarily weighted graph.

### 7.3. Exact synchronized inheritance and storage scaling

If `u=delta=0` initially, their two rows vanish. Smooth uniqueness makes
this synchronized submanifold invariant. The remaining rows are exactly
the original conservative sine law on the base graph, with the same
`nu_i,beta,w` and clock. This is inheritance on a supplied invariant
submanifold, not synchronization from a general preparation.

Write

\[
E_c(X,\Theta)=\frac12\sum_{\{i,j\}}(X_i-X_j)^2
              +\beta\sum_{\{i,j\}}[1-\cos(\Theta_j-\Theta_i)] .
\]

Summing over the four fine edges associated with each base edge yields
the exact identity

\[
\boxed{
E_f=4E_c+
       2\sum_i d_i u_i^2+
       4\beta\sum_{\{i,j\}}\cos(\Theta_j-\Theta_i)
                   [1-\cos\delta_i\cos\delta_j] .}
\]

For the form term, the four squared differences sum to
`4*((X_i-X_j)^2+u_i^2+u_j^2)`. For phase, their four cosines sum to
`4*cos(Theta_j-Theta_i)*cos(delta_i)*cos(delta_j)`. These two
identities give the displayed formula and, on synchronization, `E_f=4*E_c`.
Neither capacity nor the clock has been rescaled to obtain the factor four.

Both full fine storage and synchronized coarse storage are conserved in
their respective conservative laws. Off the synchronized submanifold,
`E_c` is not independently conserved in general: the other retained terms
exchange storage with it. Their displayed phase contribution is
nonnegative in a local acute coarse chart, but not for arbitrary coarse
phase differences. Thus the algebraic decomposition alone is not a
global positive split into separate reservoirs.

### 7.4. A nearby exact obstruction to autonomous block means

Take the base `C5`, ordered `0,1,2,3,4`, with
`alpha=2*pi/5`, constant `X` and `Theta_j=j*alpha`. First prepare
`u=delta=0`. This is an exact equilibrium of the fine and coarse laws.
Now keep the same `X,Theta,u=0` and change only
`delta_0=epsilon`, with every other `delta_i=0` and

\[
0<|\epsilon|<\pi/2-\alpha.
\]

The fine edges remain strictly acute, and the entire preparation can be
arbitrarily close to the synchronized target. Nevertheless its exact
coarse form rates are

\[
\dot X_1=\frac{a\nu_1}{2}\sin\alpha(1-\cos\epsilon)>0,\qquad
\dot X_4=-\frac{a\nu_4}{2}\sin\alpha(1-\cos\epsilon)<0,
\]

with `X_dot=0` at the other three nodes. All coarse phase rates remain
zero at this instant. The first preparation had all these rates zero.
The two states therefore have identical observed `(X,Theta)` and
different derivatives. No autonomous first-order vector field on those
block means can reproduce every fine state in any neighborhood of this
target. This is a failure of that proposed coarse closure, not a failure
of the complete retained-coordinate nodal law.

The all-state closure obstruction also holds for any connected base graph
in the stated domain. Choose a node `j`, set its coarse phase to a small
`gamma!=0` and all other coarse phases to zero, and compare `delta=0`
against a preparation with only `delta_j=epsilon!=0`. At every neighbor
`i` of `j` the coarse form defect is
`(a*nu_i/d_i)*(cos(epsilon)-1)*sin(gamma)!=0`. The two preparations have
the same coarse state; choosing `|gamma|+|epsilon|<pi/2` keeps the
comparison within the pair chart with all fine phase gaps acute.
This general obstruction does not require that
either preparation be a critical target.

Internal form can also be hidden from an instantaneous rate comparison.
At the same target, prepare `delta=0` and only `u_k=U!=0`. Define the
coarse form defect relative to the base sine row by

\[
D_i=\frac{a\nu_i}{d_i}\sum_{j\sim i}
       (\cos\delta_i\cos\delta_j-1)\sin\Delta_{ji}.
\]

Initially `D=0` and `D_dot=0`. For a neighboring node `i`, however,

\[
\ddot D_i(0)
 =-\frac{a\nu_i}{2}
     \sin(\Theta_k-\Theta_i)(b\nu_kU)^2 .
\]

This follows by differentiating the exact cosine factors twice and using
`delta_k_dot=b*nu_k*U`; their value and first derivative vanish in the
defect at the preparation. It is an exact field-jet statement, not a
trajectory or a tangent replacement of the nonlinear law. Internal form
changes internal phase and hence the later coarse response even when the
initial coarse rate defect vanishes.

### 7.5. Trapping permits persistent internal organization

For this full ten-node fine graph, the lifted twist is itself an exact
acute critical target. On pair-constant vectors its combinatorial
Laplacian is twice the `C5` Laplacian; on pair-antisymmetric vectors it is
`4*I`. Thus its smallest positive eigenvalue is

\[
\lambda_{2,f}=\min(2\lambda_{2,C5},4)=5-\sqrt5>0 .
\]

Every target edge has cosine `cos(alpha)>0`. The existing
[local first-exit proof](nodal/RELATIONAL_PATTERN_MEMORY.md#sine-relative-pattern-state)
therefore applies to the **full** fine graph. With centered fine form
and target phase deviations of combined squared norm `Z_f^2`, choose
`r>0` such that `alpha+sqrt(2)*r<pi/2` and set

\[
\kappa_f=\frac{5-\sqrt5}{2}
  \min\{1,\beta\cos(\alpha+\sqrt2r)\}.
\]

If initially

\[
Z_f^2<r^2,\qquad E_f-4E_{*,C5}<\kappa_f r^2,
\]

conservation and the strict boundary coercivity exclude a first exit.
The argument is valid in both time directions. It keeps both internal
and collective coordinates in the local tube: in the declared chart,

\[
Z_f^2=
2\|PX\|^2+2\|P(\Theta-\Theta_*)\|^2
+2\|u\|^2+2\|\delta\|^2,\qquad
P=I-\frac15\mathbf1\mathbf1^T .
\]

This is a sufficient trapping condition on the full state, not a sum of
five independent pair certificates. All target cycles retain their
winding inside the acute tube. The previous small-`epsilon` obstruction
can be chosen within such a tube, so persistent fine organization and
failure of block-mean closure are compatible.

The transverse linearization at the synchronized twist is

\[
\dot u_i=-a\nu_i\cos\alpha\,\delta_i,\qquad
\dot\delta_i=b\nu_i u_i .
\]

Its eigenvalues are
`+/-i*nu_i*sqrt(a*b*cos(alpha))`. These conditional tangent modes describe
internal exchange, not decay or a nonlinear period certificate. The full
conservative flow preserves volume. The existing
[nonlinear recurrence theorem](nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
implies that almost every point outside the synchronized submanifold in
an admitted finite-volume trapped family cannot converge to it: near
returns to its initial state retain a positive distance from that closed
submanifold. Conservative trapping therefore does not establish attracting
synchronization. This does not assert recurrence of a selected moving
state or supply an observed return time.

The same support also distinguishes an unstable geometry without changing
the law. The uniform `C5` twist with winding two is another exact critical
target, with edge cosine `cos(4*pi/5)<0`. Its transverse equation is

\[
\ddot\delta_i
 =-ab\nu_i^2\cos(4\pi/5)\,\delta_i ,
\]

so the tangent has a positive real eigenvalue
`nu_i*sqrt(-a*b*cos(4*pi/5))`. The smooth full sine equilibrium is
therefore locally unstable. In contrast, winding one's positive edge
cosine supports the preceding full-state trapping domain. This is a
conditional geometric distinction under identical support and coefficients:
losing the winding-two organization does **not** mean that its storage is
dissipated. With `e=0` the total storage remains constant. It also does
not prove that every perturbation of the unstable target follows the same
transition or ends in the winding-one family.

For the different law `e>0`, decay and target recovery must use the
existing [positive-loss theorem](nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
and [local recovery hypotheses](nodal/RELATIONAL_PATTERN_MEMORY.md#sine-relative-pattern-state).
The conservative recurrence argument no longer applies, and the
conservative transverse modes cannot be relabeled as its attractors.

### 7.6. Meaning and limits of the scale result

The paired description preserves a larger organization together with its
internal fine patterns. It does not require that the fine patterns vanish
when a collective description is introduced. Conversely, grouping two
nodes does not by itself establish either constituent or the group as an
autonomously formed NFR; their preparation, identity and support remain
explicit premises here.

Exact same-law inheritance is proved on synchronization. Near that
submanifold, the exact larger description retains internal coordinates,
and their emergent resultant amplitudes modify collective interactions.
This supplies a specific relation between organization and internal
exchange. It does not establish generic autonomous coarse closure,
spontaneous hierarchical formation, a scale-independent fractal dimension,
a universal pulse, or a microscopic physical identification.

The detached
[replica observer](../src/tnfr/physics/relational_sine_scale.py)
retains the complete captured graph and independently compares fine
half-sum/half-difference rates with the factored expressions. It uses exact
represented form values and supplied phase-lift integers, and retains
interval bounds for trigonometric quantities. Zero-containing arithmetic
residuals are checks of that capture, distinct from the all-state
identities proved above. The
[replica tests](../tests/physics/test_relational_sine_replica.py)
separate synchronized inheritance, hidden internal motion, nonlinear
coarse nonclosure and the two target geometries. The observer does not
certify an arbitrary capture as lying in the local trapping domain or
execute a trajectory.

<a id="sine-replica-unordered-state"></a>

## 8. An exact collective state for unordered pairs

Keep the complete conservative sine law, complete two-replica support,
strictly positive held paired capacities and nonantipodal pair chart of
[section 7](#sine-replica-inheritance). There are no external inputs or
node-selected events. Swapping the two members of any pair is then an
exact symmetry: it preserves the support, capacities and complete nodal
rows. This section removes only those independent swap labels. Internal
form and phase information remain in the collective description.

### 8.1. Realizable invariants and their fibers

For each pair define

\[
R_i=\cos\delta_i,\qquad U_i=u_i^2,\qquad
Q_i=u_i\sin\delta_i .
\]

The symbol `Q_i` here denotes an internal form-phase correlation, not
nodal pressure `p=DeltaNFR`. Their units are respectively `1`, `[x]^2`
and `[x]`. Each is unchanged by the pair swap
`(u_i,delta_i)->(-u_i,-delta_i)`. The retained coarse coordinates
`X_i,Theta_i` are unchanged as well.

Their exact realized domain is

\[
\boxed{\quad
0<R_i\le1,\qquad U_i\ge0,\qquad
Q_i^2=U_i(1-R_i^2).
\quad}
\]

Necessity follows from `|delta_i|<pi/2` and the sine-cosine identity.
For sufficiency and uniqueness of the unordered pair:

- If `0<R_i<1`, choose
  `delta_i=arccos(R_i)>0` and
  `u_i=Q_i/sqrt(1-R_i^2)`. The constraint gives `u_i^2=U_i`.
  The other possible lift is its simultaneous negative, exactly the pair
  swap. This also covers `U_i=Q_i=0` with nonzero phase separation.
- If `R_i=1`, the chart forces `delta_i=0` and the constraint forces
  `Q_i=0`. The choices `u_i=+sqrt(U_i)` and `u_i=-sqrt(U_i)`
  are the same unordered pair.
- At `R_i=1,U_i=Q_i=0` there is just one lift,
  `u_i=delta_i=0`.

Thus `(X,Theta,R,U,Q)` separates all fine states in this chart up to
the independent pair swaps. Together with the supplied support and held
capacities, it specifies the unordered fine configuration. In particular,
the fully coherent phase value `R_i=1` does not imply synchronized
form: `U_i>0` remains distinguishable.

This is a quotient by a finite symmetry group, not a reduction in the
generic number of continuous degrees of freedom. Each internal pair has
two degrees of freedom, represented by three invariants with one
constraint. At the synchronized tip `(R,U,Q)=(1,0,0)`, the gradient of
`Q^2-U*(1-R^2)` vanishes and the swap fixes the original state. These
invariants are not an ordinary smooth coordinate chart at that tip.
Their algebraic singularity must not be interpreted as an extra physical
degree of freedom, a dynamical instability or failure of the original
smooth nodal law.

### 8.2. Closed dynamics with internal state retained

Let

\[
A_i=\frac{a\nu_i}{d_i}
       \sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),
\qquad c_i=b\nu_i .
\]

These are calculated from the retained state and declared coefficients,
with units `[A]=[x]/[t]` and `[c]=1/([x][t])`. They introduce no new
constitutive parameter. The original internal rows are
`u_i_dot=-A_i*sin(delta_i)` and `delta_i_dot=c_i*u_i`.
Differentiating the invariants gives

\[
\boxed{\begin{aligned}
\dot X_i&=\frac{a\nu_i}{d_i}
             \sum_{j\sim i}R_iR_j\sin(\Theta_j-\Theta_i),\\
\dot\Theta_i&=\frac{b\nu_i}{d_i}
             \sum_{j\sim i}(X_i-X_j),\\
\dot R_i&=-c_iQ_i,\\
\dot U_i&=-2A_iQ_i,\\
\dot Q_i&=-A_i(1-R_i^2)+c_iU_iR_i .
\end{aligned}}
\]

Every consumed quantity is determined by the unordered collective state.
These are exact nonlinear quotient rows. Unlike block means alone, the
augmented state closes because it retains the internal information that
caused the earlier obstruction. No hidden initial condition was set to
zero, fitted to a response or erased.

The constraint is an exact first integral of these rows:

\[
\begin{aligned}
\frac{d}{dt}[Q_i^2-U_i(1-R_i^2)]
={}&2Q_i[-A_i(1-R_i^2)+c_iU_iR_i]\\
 &+2A_iQ_i(1-R_i^2)-2c_iU_iR_iQ_i=0 .
\end{aligned}
\]

This algebra alone does not prove realizability: the same equality also
has nonphysical solutions such as `R_i>1,U_i<0`. The inequalities,
phase chart and lifting argument are essential.

### 8.3. Evolution, boundaries and lift independence

Start with a realizable collective state and choose any fine lift supplied
by section 8.1. Evolve the original smooth fine nodal law. On every time
interval for which all pairs remain in `|delta_i|<pi/2`, its invariants
solve the displayed collective rows and remain realizable.

Conversely, those rows define a smooth vector field in the ambient
`(X,Theta,R,U,Q)` variables; there is no division by `U`, `Q` or
`1-R^2` in them. Local uniqueness therefore makes every collective
solution from that initial state equal to the invariant image of the
fine solution. Choosing the other fine lift merely performs fixed pair
swaps, which are symmetries of the full law. It yields the same collective
solution. This proves dynamical sufficiency even at a tip where the
inverse invariant coordinates are not differentiable.

Two boundary controls make the role of the retained variables explicit:

\[
\begin{array}{ll}
R_i=1,\ U_i>0:
 &\dot Q_i=c_iU_i,\quad
   \ddot R_i=-c_i^2U_i<0,\\[2mm]
U_i=0,\ 0<R_i<1:
 &\dot Q_i=-A_i(1-R_i^2),\quad
   \ddot U_i=2A_i^2(1-R_i^2)\ge0 .
\end{array}
\]

The first state has aligned primitive phases but a form difference that
immediately changes its internal correlation and subsequently its
resultant amplitude. The second can acquire a form difference from
phase separation when `A_i!=0`. At the tip `R_i=1,U_i=Q_i=0` all
three internal derivatives vanish. That internal tip remains invariant
even when the means or neighboring internal states evolve.

No further chart guarantee follows just from the collective equality
constraint. At `R_i=0` the primitive phases are antipodal, their mean
phasor vanishes, and its circular midpoint is not uniquely defined.
The original fine law remains smooth there. The chosen collective chart
must end or be replaced by a separately justified chart; continuing a
signed `R` or clipping it to zero is not an automatic continuation of
this unordered observation. The present result is exact up to that chart
boundary unless an independent trapping condition excludes it.

### 8.4. Storage and identity pass to the exact quotient

The full fine storage becomes the invariant function

\[
\boxed{
\mathcal H(X,\Theta,R,U)=
2\sum_{\{i,j\}}(X_i-X_j)^2+
2\sum_i d_iU_i+
4\beta\sum_{\{i,j\}}
 [1-R_iR_j\cos(\Theta_j-\Theta_i)] .
}
\]

This is exactly `E_f` from section 7, not an additional Hamiltonian or
an independently calibrated energy. Its conservation follows by lifting
the collective rows to the same conservative law, or by direct
differentiation using those rows. It retains internal form storage even
at `R_i=1`. On the realized domain every displayed contribution is
nonnegative. Setting `R=1,U=Q=0` recovers `4*E_c` and the supplied
synchronized same-law sector.

The previous full ten-node acute-twist barrier can also be expressed
without the swap labels. In its declared target chart,

\[
Z_f^2=
2\|PX\|^2+2\|P(\Theta-\Theta_*)\|^2
+2\sum_i[U_i+(\arccos R_i)^2],
\qquad P=I-\frac15\mathbf1\mathbf1^T .
\]

This equals the fine centered norm because
`delta_i^2=(arccos(R_i))^2` in the admitted pair chart. Thus the
existing sufficient conditions
`Z_f^2<r^2` and
`H-4*E_*,C5<kappa_f*r^2` admit exactly the same trapping conclusion
when read through these invariants. In particular,

\[
U_i<r^2/2,\qquad
R_i>\cos(r/\sqrt2)>0
\]

throughout such a trapped trajectory. The acute radius condition in
section 7 implies `r/sqrt(2)<pi/2`. Consequently that independent
full-state barrier keeps the unordered pair chart valid for all time;
the constraint by itself did not do so. This rewrites a proved barrier
rather than creating another approximation, recovery criterion or
trajectory calculation.

### 8.5. What has and has not been reduced

The result removes a redundant choice of member labels while retaining
the complete unordered pair, including its internal form variance,
phase separation and their signed correlation. The collective object
therefore still contains its smaller constituents and their dynamics.
It is not a claim that a pair has become a single fundamental scalar
node with the original bare nodal state space.

The complete bipartite support, partition, equal paired capacities and
phase chart remain assumptions. Unequal member capacities, selective
forcing, member-specific events or another fine support can break the
swap symmetry and require another observation model. No autonomous
partition formation, generic fractality, unlimited hierarchical closure,
material particle or microscopic law selection follows from this finite
symmetry quotient.

The [existing replica observer](../src/tnfr/physics/relational_sine_scale.py)
exposes the invariant state of an admitted fine capture and compares
its directly calculated derivatives with these closed rows. Its exact
represented `U` and interval `R,Q` retain the fine capture and
arithmetic provenance; an interval constraint residual containing zero
does not admit every tuple in that surrounding rectangular box.
The [same test owner](../tests/physics/test_relational_sine_replica.py)
checks pair-swap invariance, boundary controls, direct fine-to-collective
rates and storage. This is not an API for accepting arbitrary collective
tuples, reconstructing a chosen numerical lift, or running a trajectory.

<a id="sine-replica-inherited-poisson"></a>

### 8.6. The collective state inherits the same Poisson structure

The exact quotient inherits more than a closed vector field. Use the
[same-law Poisson tensor](nodal/RESONANCE_FOUNDATIONS.md#reciprocal-exchange),
with bracket convention `dot f={f,E_f}`. Each fine member has degree
`2*d_i`, hence

\[
\{x_{i,\pm},\theta_{i,\pm}\}
   =-\frac{b\nu_i}{2d_i}.
\]

All other fine coordinate brackets vanish. The mean/half-difference
change therefore gives

\[
\{X_i,\Theta_i\}=\{u_i,\delta_i\}=-\kappa_i,\qquad
\kappa_i=\frac{b\nu_i}{4d_i}>0,
\]

with zero brackets between distinct pairs and between a pair's mean
and internal coordinates. The factor four follows from both the fine
degree and the two factors of one half in the coordinate change.
Holding support and paired capacities fixed makes each `kappa_i`
constant.

The chain rule for `R=cos(delta), U=u^2, Q=u*sin(delta)` gives

\[
\boxed{
\{R_i,U_i\}=-2\kappa_iQ_i,\qquad
\{R_i,Q_i\}=-\kappa_i(1-R_i^2),\qquad
\{U_i,Q_i\}=-2\kappa_iU_iR_i .
}
\]

Together with `{X_i,Theta_i}=-kappa_i`, these are all the
nonzero independent brackets. They descend under every pair swap:
simultaneously negating `u_i,delta_i` preserves their constant
bracket and leaves the invariant functions unchanged.

Jacobi follows by pushforward of the fine bracket on the invariant
algebra. It can also be checked without choosing a lift. Set

\[
\mathcal C_i=Q_i^2-U_i(1-R_i^2).
\]

In the ordered coordinates `(R_i,U_i,Q_i)`, the internal tensor is
`J_ab=-kappa_i*epsilon_abc*partial_c C_i`. A three-dimensional
bracket of this form satisfies Jacobi because its defining vector
`-kappa_i*grad C_i` has zero curl. Moreover
`J*grad C_i=0`, so `C_i` is a **Casimir**: its bracket with
every smooth retained function vanishes, not only with this storage.
The direct sum with the constant mean blocks is again Poisson.

For a realized nontip state the internal tensor has rank two.
If `R_i<1`, its `(R_i,Q_i)` entry is nonzero; if `R_i=1`
and `U_i>0`, its `(U_i,Q_i)` entry is nonzero. At the tip
`(R_i,U_i,Q_i)=(1,0,0)` it has rank zero. The independent mean
block retains rank two there. This is the finite-symmetry quotient's
singular internal stratum, not an additional continuous degree of
freedom or a singularity of the fine nodal law.

The Hamiltonian is exactly the already derived `H=E_f` in section
8.4. Its nonzero coordinate derivatives are

\[
\begin{aligned}
\partial_{X_i}\mathcal H&=4\sum_{j\sim i}(X_i-X_j),\\
\partial_{\Theta_i}\mathcal H
 &=-4\beta R_i\sum_{j\sim i}R_j\sin(\Theta_j-\Theta_i),\\
\partial_{R_i}\mathcal H
 &=-4\beta\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),\\
\partial_{U_i}\mathcal H&=2d_i,\qquad
\partial_{Q_i}\mathcal H=0 .
\end{aligned}
\]

Using `a=beta*b`, the mean brackets reproduce the exact `X,Theta`
rows, while the internal brackets give

\[
\begin{aligned}
\{R_i,\mathcal H\}&=-4\kappa_i d_iQ_i=-c_iQ_i,\\
\{U_i,\mathcal H\}
 &=-8\kappa_i\beta Q_i\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i)
   =-2A_iQ_i,\\
\{Q_i,\mathcal H\}
 &=-4\kappa_i\beta(1-R_i^2)
       \sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i)
      +4\kappa_i d_iU_iR_i\\
 &=-A_i(1-R_i^2)+c_iU_iR_i .
\end{aligned}
\]

Thus all five collective rows, the internal constraint and storage
conservation share one inherited structure. No Hamiltonian, pressure
channel or feedback controller is added.

The polynomial Poisson identity also exists outside the realized set;
it does not admit those states. The inequalities, lifting and chart
continuation conditions of sections 8.1–8.3 remain essential. In
particular, a Casimir value or a numerical residual cannot establish
realizability, and `R=0` still requires a different collective chart.
Likewise, a recurrence argument must retain the fine-state measure or
a justified quotient measure: the constrained invariants are not
independent variables carrying full-dimensional Lebesgue volume.
The `e=0` law, supplied support and equal held member capacities remain
premises; this identity selects neither microscopic loss nor a physical
interpretation of storage.

<a id="sine-replica-internal-observability"></a>

### 8.7. Temporal resultant information recovers the unordered internal state

The same closed rows yield an exact observation consequence, without
introducing another variable or law. Retain the section 8 premises and
the structural clock. Suppose the means `X,Theta`, all neighboring
resultant amplitudes `R` and the held coefficients are independently
known. Thus `c_i=b*nu_i>0` and
`A_i=(a*nu_i/d_i)*sum_j R_j*cos(Theta_j-Theta_i)` are known;
only relative neighboring mean phases are consumed by `A_i`.
For observations of an admitted smooth fine trajectory,

\[
\dot R_i=-c_iQ_i,\qquad
\ddot R_i=c_iA_i(1-R_i^2)-c_i^2U_iR_i.
\]

Consequently, everywhere in the local chart `R_i>0`,

\[
\boxed{
Q_i=-\frac{\dot R_i}{c_i},\qquad
U_i=\frac{c_iA_i(1-R_i^2)-\ddot R_i}{c_i^2R_i}.
}
\]

These observations recover the exact unordered internal state, not
the constituent labels or a preferred signed lift. Held capacity is
essential: if `c_i` varied, the second derivative would also contain
`-dot(c_i)*Q_i`. The formula does not estimate an unknown capacity,
select a clock, or identify a laboratory sensor with this resultant.

For `0<R_i<1`, the Casimir additionally gives the first-derivative
formula

\[
U_i=\frac{\dot R_i^2}{c_i^2(1-R_i^2)}.
\]

That expression becomes singular at aligned phases. The second-derivative
formula remains regular there: at `R_i=1`, any admitted trajectory
has `dot R_i=Q_i=0` but

\[
\boxed{\qquad U_i=-\ddot R_i/c_i^2.\qquad}
\]

Thus a perfectly aligned instantaneous pair can conceal a form imbalance
that its resultant curvature distinguishes. The neighbor term vanishes
at this exact boundary, although it is needed for the general
second-derivative reconstruction. This reuses the boundary dynamics
of section 8.3; zero first derivative alone did not imply a stationary
internal state.

The inverse retains admission and observation limits. Its output must
satisfy `0<R_i<=1`, `U_i>=0` and the exact Casimir constraint.
Small `c_i` or proximity to `R_i=0` amplifies uncertainty in the
second-derivative inverse; the first-derivative inverse instead loses
conditioning near `R_i=1`. Independent intervals for state and
derivatives need not come from one joint trajectory. Numerical
differentiation requires its own sampling, timing and remainder bounds;
neither interval overlap nor this exact identity supplies them.
There is no arbitrary noisy-tuple admission, raw-data reconstruction
API or replacement of retained internal state by the single value R.

<a id="sine-replica-internal-pulse"></a>

## 9. A finite-amplitude internal pulse with persistent collective identity

Keep the complete double-replica `C5`, the same conservative sine law
`e=0,w,beta>0`, no inputs or events, and now a **common** strictly
positive held capacity `nu` at all ten fine nodes. This additional
capacity premise is needed for the invariant family below; paired
equality alone was enough for sections 7 and 8.

### 9.1. The exact invariant preparation

Let `alpha=2*pi/5` and `c_alpha=cos(alpha)>0`. Supply the exact
collective state

\[
X_i=\bar X,\qquad
\Theta_i=\bar\Theta+i\alpha\pmod{2\pi},
\qquad i=0,\ldots,4,
\]

and identical signed internal coordinates

\[
u_i=u,\qquad \delta_i=\delta
\]

for all five pairs. The constants `bar X` and `bar Theta` specify the
common form and phase origins. The target angles are exact turns `i/5`;
replacing them by rounded radian values and accepting a small balance
residual would not establish this invariant family.

At each coarse node the two phase differences are `+alpha` and
`-alpha` modulo `2*pi`. Their sine currents cancel, while their
cosines add. The exact retained-coordinate rows of section 7 therefore
give

\[
\dot X_i=0,\qquad \dot\Theta_i=0,\qquad
\dot u_i=-a\nu c_\alpha\cos\delta\sin\delta,\qquad
\dot\delta_i=b\nu u.
\]

The internal derivatives are identical at all five pairs, so smooth
uniqueness proves that the full preparation remains in this family.
Its complete nonlinear evolution reduces exactly to

\[
\boxed{\quad
\dot u=-a\nu c_\alpha\cos\delta\sin\delta,\qquad
\dot\delta=b\nu u .
\quad}
\]

The ten constituent coordinates remain
`x_i,plus=bar X+u`, `x_i,minus=bar X-u` and
`theta_i,plus/minus=bar Theta+i*alpha+/-delta`. Static collective
means do not mean that the constituents are stationary. Neither an
external periodic input nor an additional oscillator equation has been
installed: the two displayed rows are an invariant restriction of the
already supplied complete law.

### 9.2. Conserved amplitude and exact periods

Because `a=beta*b`, the internal quantity

\[
\boxed{\quad H=u^2+\beta c_\alpha\sin^2\delta\quad}
\]

is conserved:

\[
\dot H
=-2a\nu c_\alpha u\cos\delta\sin\delta
 +2\beta b\nu c_\alpha u\sin\delta\cos\delta=0.
\]

It is part of the existing full storage, not a second independently
postulated energy. Indeed, section 7 gives exactly

\[
E_f=4E_{*,C5}+20H,\qquad
E_{*,C5}=5\beta(1-c_\alpha).
\]

Define

\[
\Omega=\nu\sqrt{ab c_\alpha}
       =\frac{w\nu\sqrt{c_\alpha}}{\pi\sqrt\beta},
\qquad
m=\frac{H}{\beta c_\alpha}.
\]

For `0<m<1` and a preparation in `|delta|<pi/2`, the level set is a
nonstationary closed curve around `(u,delta)=(0,0)`. Its amplitudes are

\[
|\delta|_{\max}=\arcsin\sqrt m<\pi/2,\qquad
|u|_{\max}=\sqrt H .
\]

The pair chart consequently remains valid for the entire motion. Set
`eta=2*delta`; differentiating the actual phase row gives

\[
\ddot\eta+\Omega^2\sin\eta=0 .
\]

This pendulum equation is a derived form of the retained internal
system. The conserved level already proves that the orbit is a
libration, and direct quadrature gives its period. With the elliptic
**parameter** `m`, define

\[
K(m)=\int_0^{\pi/2}\frac{d\psi}{\sqrt{1-m\sin^2\psi}}.
\]

Since `sin(delta)=sqrt(m)*sin(psi)` on a quarter orbit,

\[
\boxed{\quad
T_{\rm labeled}=\frac{4}{\Omega}K(m)
 =\frac{4\pi\sqrt\beta}{w\nu\sqrt{c_\alpha}}K(m).
\quad}
\]

This is the period of the full labeled circular fine state, not merely
of a tangent mode. On a nonzero libration the half-period map is
`(u,delta)->(-u,-delta)`. It swaps the two members of every pair while
leaving `X,Theta` fixed. The unordered state of section 8 therefore has
the smaller fundamental period

\[
\boxed{\quad T_{\rm unordered}=T_{\rm labeled}/2.\quad}
\]

To see why it is the fundamental unordered period, the nonstationary
closed energy curve is traversed once in `T_labeled` and each unordered
state has exactly two fine lifts on it. Those lifts are related by the
half-period swap; there is no fixed lift on a positive-energy orbit.
The invariants `R,U,Q` change along this orbit, so the unordered state
is not stationary despite the fixed collective means.

The same elementary integral bounds used for the
[two-node pulse](nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
give

\[
T_0=\frac{2\pi}{\Omega},\qquad
T_0<T_{\rm labeled}\le\frac{T_0}{\sqrt{1-m}},
\]

and half of each bound for the unordered period. The strictly positive
amplitude is essential: `T_0` is the small-amplitude limiting period,
not a period assigned to a stationary state. In particular, the pulse
frequency depends on the supplied capacity, coefficients, geometry and
amplitude. As `m` approaches one, `K(m)` diverges. This is a persistent
periodic internal motion under a declared law, not a universal constant
frequency.

### 9.3. Collective identity and the stronger fine-edge condition

Throughout every admitted libration, the collective means remain exactly
the winding-one target. The inherited phase-current factor
`R_i*R_j=cos(delta)^2` varies periodically on every coarse edge, while
its two incident sine currents still cancel at each coarse node.
Consequently a static collective mean can contain changing internal
interaction strength. The fine phase differences along an oriented
base edge are

\[
\alpha,\qquad \alpha+2\delta,\qquad \alpha-2\delta
\pmod{2\pi}.
\]

Preserving the nonantipodal pair chart is therefore weaker than keeping
every fine edge acute. The latter holds for the complete period under
the sharper condition

\[
|\delta|_{\max}<\frac{\pi/2-\alpha}{2}
\quad\Longleftrightarrow\quad
\boxed{\quad m<\sin^2(\pi/20).\quad}
\]

This nonempty finite-amplitude subfamily has fixed collective winding,
moving constituents and strictly acute fine edges at all times. Its fine
cycle windings also retain those of the duplicated target: continuously
varying strictly acute edges cannot cross an antipodal wrapping boundary.
This conclusion follows from the exact orbit and its amplitude bound;
it does not require a numerical trajectory.

For larger librations with `0<m<1`, the static collective target and
valid pair chart still follow, but the all-fine-edge acute certificate
does not. A loss of that sufficient condition is not by itself a
formation, instability or winding-change verdict. General transverse
perturbations away from the common-internal-state family remain governed
by the full retained law and its independent
[trapping conditions](#sine-replica-inheritance). Exact period, common
internal phase and the half-period swap cannot be inferred for those
perturbations from an energy bound alone.

### 9.4. Boundaries and scientific scope

Within the admitted pair chart:

- `H=0` forces `u=delta=0` and is stationary. No pulse starts
  spontaneously from that exact equilibrium.
- `H=beta*c_alpha` is the separatrix level. Nonstationary preparations
  inside the chart approach its boundary `|delta|=pi/2` asymptotically
  and do not have a finite libration period. The boundary states
  `u=0,delta=+/-pi/2` are stationary states of the full fine law,
  outside this pair-midpoint chart.
- `H>beta*c_alpha` gives
  `u^2>=H-beta*c_alpha>0`, so `delta` moves monotonically and leaves
  the admitted pair chart in finite time. This excludes this
  **libration certificate**, not every periodic or rotating full
  circular-state solution. The full fine law continues smoothly and
  would need another observation chart.

The result establishes a family of finite-amplitude internal pulses
inside a persistent collective organization. The common capacity, exact
twist, zero loss and identical signed internal preparation are explicit
premises. This invariant family has lower dimension than the ambient
fine state space; its exact periodic behavior is not an almost-everywhere
or generic-attraction theorem.

Nothing here selects microscopic zero loss, supplies the initial
amplitude, forms the paired support, forces all internal modes to align,
or identifies a material particle. The pulse is sustained by conservative
exchange already present in the chosen nodal law, and the distinction
between labeled and unordered periods follows from its exact symmetry.

The [existing scale owner](../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_pulse` for this declared exact family, with
the common signed internal preparation and capacity supplied explicitly.
It retains symbolic target turns and marks captured-graph membership as
uncertified. Interval bounds classify the internal energy, enclose the
two periods and independently test the stronger all-fine-edge acute
condition; an unresolved boundary does not become a passing certificate.
The period enclosure reuses the
[shared elliptic-integrand bound](../src/tnfr/physics/relational_sine_resonance.py).
The [replica tests](../tests/physics/test_relational_sine_replica.py)
compare full nodal rows, exact storage identities and independent
elliptic-function values without an ODE trajectory or a rounded-target
membership assertion.

<a id="sine-replica-pulse-variation"></a>

## 10. Complete variation around the finite internal pulse

Retain the exact doubled-`C5` pulse and all hypotheses of
[section 9](#sine-replica-internal-pulse). The instantaneous variational
identities hold at any declared point of its invariant family while the
pair chart is valid. The finite-period return statements additionally
require the nonstationary libration condition `0<m<1`. Write the
reference internal phase as `d(t)` to distinguish it from a perturbation,
and put

\[
\alpha=\frac{2\pi}{5},\quad c=\cos\alpha,\quad s=\sin\alpha,\quad
C(t)=\cos d(t),\quad S(t)=\sin d(t).
\]

The perturbations below keep the supplied graph, common held capacity
and law fixed. They include all twenty real nodal-state directions, not
just perturbations that preserve the common signed internal preparation.
They do not include perturbations of the support or constitutive laws.

### 10.1. Direct differentiation of the full retained law

Perturb `(X_i,Theta_i,u_i,delta_i)` by `(xi_i,eta_i,v_i,zeta_i)`.
On the ordered cycle define

\[
(Lf)_i=2f_i-f_{i+1}-f_{i-1},\qquad
(Df)_i=f_{i+1}-f_{i-1},
\]

with indices modulo five. Differentiating the complete rows of section 7,
before restricting a spatial mode, gives

\[
\boxed{\begin{aligned}
\dot\xi&=-\frac{a\nu}{2}
       [cC^2L\eta+sCS\,D\zeta],\\
\dot\eta&=\frac{b\nu}{2}L\xi,\\
\dot v&=-a\nu c
       [\cos(2d)I+\tfrac12S^2L]\zeta
       +\frac{a\nu sCS}{2}D\eta,\\
\dot\zeta&=b\nu v .
\end{aligned}}
\]

For example, the derivative of the mean form row contains the neighbor
internal-phase term
`-(a*nu/2)*C*S*sum_j sin(Theta_j-Theta_i)*zeta_j`.
The two target sine values are `+s,-s`, producing `D*zeta`.
The term from differentiating the local `cos(delta_i)` vanishes
because the unperturbed incident sine sum is zero. Differentiating the
internal form row gives the opposite signed `D*eta` term and the
coefficient
`C^2*zeta_i-(S^2/2)*(zeta_(i+1)+zeta_(i-1))`, equal to the displayed
`cos(2d)I+(S^2/2)L` expression.

Thus transverse perturbations couple internal and collective coordinates.
Treating the five pairs as independent pendula would omit the two
`D` terms and generally change the variation being tested.

### 10.2. Exact spatial decomposition without discarding directions

Use the Fourier convention `f_i=hat f_k*exp(i*q_k*i)`, with
`q_k=2*pi*k/5`. Then

\[
\lambda_k=2-2\cos q_k,\qquad
L\mapsto\lambda_k,\qquad D\mapsto2i\sin q_k .
\]

In the order `(hat xi,hat eta,hat v,hat zeta)`, the complex block is

\[
\mathcal A_k(t)=
\begin{pmatrix}
0&-a\nu cC^2\lambda_k/2&0&-ia\nu sCS\sin q_k\\
b\nu\lambda_k/2&0&0&0\\
0&ia\nu sCS\sin q_k&0&
   -a\nu c[\cos(2d)+S^2\lambda_k/2]\\
0&0&b\nu&0
\end{pmatrix}.
\]

Real nodal perturbations obey `hat f_(5-k)=conjugate(hat f_k)`.
For each representative `k=1,2`, change only the analysis basis to

\[
(\hat\xi_k,\hat\eta_k,i\hat v_k,i\hat\zeta_k).
\]

The resulting real-coefficient matrix is

\[
\boxed{
A_k(t)=
\begin{pmatrix}
0&-a\nu cC^2\lambda_k/2&0&-a\nu sCS\sin q_k\\
b\nu\lambda_k/2&0&0&0\\
0&-a\nu sCS\sin q_k&0&
   -a\nu c[\cos(2d)+S^2\lambda_k/2]\\
0&0&b\nu&0
\end{pmatrix}.}
\]

Its real and imaginary parts follow two identical four-real-dimensional
systems. This is a basis transformation, not complexification of the
scalar physical form coordinate. Mode `k=0` uses the original four real
coordinates and the same displayed matrix with `lambda_0=sin(q_0)=0`.
The dimensions are exactly

\[
4+2\cdot4+2\cdot4=20 .
\]

No relative mean, hidden capacity, fine coordinate or spatial perturbation
has been dropped. Capacities are held parameters of the stated model,
so they are not additional tangent state directions.

The common mode contains two constant origin perturbations
`xi_0,eta_0` and the pendulum variation

\[
\dot v_0=-a\nu c\cos(2d)\zeta_0,\qquad
\dot\zeta_0=b\nu v_0 .
\]

The nonzero spatial modes are the sixteen real directions transverse to
the common-preparation family. They are the relevant blocks when asking
whether arbitrary fine perturbations preserve internal coordination.

### 10.3. Zero-amplitude control and dimensionless parameters

At `u=d=0`, `C=1,S=0` and the internal-collective cross terms vanish.
The exact spectral control is

\[
\text{mean sector: }\quad
\pm i\Omega\,\lambda_k/2,\qquad
\text{internal sector: }\quad \pm i\Omega,
\qquad \Omega=\nu\sqrt{ab c}.
\]

Mode zero contributes two zero origin modes instead of a nonzero mean
frequency. The two nonzero mean-frequency ratios are

\[
\frac{\lambda_1}{2}=\frac{5-\sqrt5}{4},\qquad
\frac{\lambda_2}{2}=\frac{5+\sqrt5}{4}.
\]

Including Fourier multiplicities, there are two zero directions, ten
internal oscillator directions and eight nonzero mean oscillator
directions. This agrees with the full fine-graph decomposition in
section 7. All are a **stationary-target** control or a small-amplitude
limit. A nonstationary finite-amplitude period is not assigned to the
zero-amplitude equilibrium, and this limiting spectrum does not decide
finite-amplitude stability.

There is also an exact reduction of the apparent parameter freedom.
Divide both form perturbations by `sqrt(beta*c)`, leave phase
perturbations unchanged, and use `tau=Omega*t`. The block becomes

\[
\boxed{
\widetilde A_k(\tau)=
\begin{pmatrix}
0&-C^2\lambda_k/2&0&-\tan\alpha\,CS\sin q_k\\
\lambda_k/2&0&0&0\\
0&-\tan\alpha\,CS\sin q_k&0&
   -[\cos(2d)+S^2\lambda_k/2]\\
0&0&1&0
\end{pmatrix}.}
\]

The base pulse itself satisfies

\[
\frac{dd}{d\tau}=\widehat u,\qquad
\frac{d\widehat u}{d\tau}=-\cos d\sin d,\qquad
\widehat u=\frac{u}{\sqrt{\beta c}},\qquad
m=\widehat u^2+\sin^2d .
\]

Consequently the return multipliers depend only on `m` and the fixed
geometry. Changing the reference position on
the same orbit conjugates the return matrix and does not change its
spectrum. At fixed `m`, `nu,w,beta` change the time or coordinate
scales, not independent stability parameters. Changing `beta` while
holding raw `u,d` fixed usually changes `m`, so it is not that
fixed-amplitude comparison.

### 10.4. Symplectic blocks and the correct half-period return

Let

\[
J_0=\operatorname{diag}
\left(
\begin{pmatrix}0&-1\\1&0\end{pmatrix},
\begin{pmatrix}0&-1\\1&0\end{pmatrix}
\right).
\]

For every displayed real block, `-J_0*A_k(t)` is symmetric. Equivalently

\[
A_k(t)^TJ_0+J_0A_k(t)=0 .
\]

Thus a fundamental matrix `Phi_k(t)` initialized by `Phi_k(0)=I`
satisfies `Phi_k(t)^T*J_0*Phi_k(t)=J_0`. This is a tangent consequence
of the same conservative law, not an auxiliary Hamiltonian added to the
fine evolution.

Write `T=T_labeled` and `h=T/2`. At the half period the base pulse has
`(u,d)->(-u,-d)`. The induced member-swap derivative is

\[
S_*=\operatorname{diag}(1,1,-1,-1),\qquad S_*^2=I .
\]

Since `C` is unchanged while `S` changes sign,

\[
A_k(t+h)=S_*A_k(t)S_* .
\]

The raw matrix is therefore generally **not** half-periodic: its two
cross terms reverse sign. They happen to vanish in mode zero, which
does not remove the swap identification of the base state.

The full labeled return and the symmetry-correct unordered return are

\[
\boxed{\qquad
M_k=\Phi_k(T),\qquad
B_k=S_*\Phi_k(h),\qquad M_k=B_k^2 .
\qquad}
\]

Indeed the second-half propagator is `S_*Phi_k(h)S_*`, and multiplying
it by the first-half propagator proves the square identity. Applying
`S_*` after the first half is essential: it identifies the tangent
space at the swapped endpoint with the original one. A spectrum of the
raw `Phi_k(h)` alone is not the unordered return spectrum.

Both `S_*` and `Phi_k(h)` are symplectic, hence so are `B_k` and
`M_k`. Their determinant is one and their multipliers occur in
reciprocal and complex-conjugate pairs. These restrictions are exact;
they supply no numerical value for a nontrivial finite-amplitude
multiplier.

### 10.5. Amplitude-dependent phase shear is a neutral-family effect

The period is strictly increasing with the conserved internal amplitude:

\[
\frac{dT}{dH}
 =\frac{4}{\Omega\beta c}K'(m)>0,\qquad
K'(m)=\frac12\int_0^{\pi/2}
 \frac{\sin^2\psi}{(1-m\sin^2\psi)^{3/2}}\,d\psi .
\]

This follows by differentiation under the integral for `0<m<1`. For
example,

\[
\frac{\pi}{2\Omega\beta c}
 <\frac{dT}{dH}
 \le\frac{\pi}{2\Omega\beta c(1-m)^{3/2}} .
\]

To identify its tangent effect, choose the smooth initial section
`d(0)=0,u(0)=sqrt(H)` with the common origins fixed. Let `v_t` be
the flow tangent there and `v_H` the derivative of that initial state
with respect to `H`. They are independent for `H>0`. Differentiate
the exact identity `z(H,T(H))=z(H,0)` to obtain

\[
M_0v_t=v_t,\qquad
M_0v_H=v_H-T'(H)v_t.
\]

Differentiating the corresponding half-period swap gives

\[
B_0v_t=v_t,\qquad
B_0v_H=v_H-\tfrac12T'(H)v_t.
\]

The two common origins have identity returns as well. Thus the common
mode has a nontrivial neutral time-amplitude shear. Repeated comparison
at the same clock phase can accumulate a linear-in-cycle phase shift
between neighboring amplitudes, even though each preparation remains
on its own conserved libration. This is not an exponentially growing
transverse mode or proof of nonlinear orbital instability. Fixing
`H` removes that amplitude direction; comparing only equally timed
waveforms does not.

### 10.6. A discriminating stability question, not a stability verdict

After the common-mode directions have been identified, the prospective
linear test is confined to the `k=1,2` representative blocks:

- A certified return multiplier `|lambda|>1` yields exponential
  transverse growth. For the smooth periodic orbit here, an unstable
  multiplier of its transverse return map also certifies nonlinear
  orbital instability. The criterion can be checked through `B_k`
  or `M_k=B_k^2`, with the correct return convention.
- If all transverse multipliers lie on the unit circle and their return
  matrices are semisimple, the transverse linear solutions are bounded:
  powers of the return matrices are bounded, and the continuous
  propagator over one compact period is bounded.
- Unit-circle eigenvalues with nontrivial Jordan blocks can yield
  polynomial growth and do not pass that bounded-linear criterion.
  Neutral amplitude shear in the separately identified common family
  must not be counted as a transverse instability.

Bounded linear variation alone does not prove nonlinear orbital
stability, attracting synchronization or retention of one exact
frequency after a general perturbation. Nonlinear resonances and
higher-order terms remain separate obligations. In particular, even
when every fine edge stays acute along the pulse, positivity of an
instantaneous storage Hessian does not by itself solve this problem.
The quadratic variation has a time-dependent coefficient:

\[
\frac{d}{dt}\frac12 z^T H_2(t)z
 =\frac12 z^T\dot H_2(t)z
\]

when the Hamiltonian tangent terms cancel. It is not generally a
conserved positive quadratic norm. The earlier all-state energy barrier
keeps admitted solutions near the static winding target, but need not
keep them near this particular periodic orbit or phase-locked to it.

This reduction supplies all perturbation directions, exact symmetry
constraints and a one-amplitude stability question. It does not evaluate
a monodromy matrix, prove a finite-amplitude instability, establish a
stability interval, or run a trajectory or Floquet sweep.
The [following small-amplitude calculation](#sine-replica-pulse-splitting)
resolves the leading transverse signs with a separately proved
existential amplitude range.

The [existing scale owner](../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_pulse_variation` with the exact family
premises reused from the pulse assessment. It reports the three real
instantaneous block types, their multiplicities, the dimensionless
versions, fixed symplectic form, swap matrix and stationary-limit
frequencies. Instantaneous coefficients are not transition matrices.
Periodic-reference and half-return claims remain conditional on the
separate libration certificate, and orbital stability is explicitly
unassessed. The [replica tests](../tests/physics/test_relational_sine_replica.py)
compare these blocks with the full fine nodal Jacobian and check the
coordinate transformations, symmetry and zero-amplitude controls.

<a id="sine-replica-pulse-splitting"></a>

## 11. Small-amplitude transverse splitting with collective response retained

This result continues the exact conservative doubled-`C5` pulse and
the complete real mode blocks of
[section 10](#sine-replica-pulse-variation). It resolves a local
small-amplitude question; it does not search over amplitudes, integrate
a variational trajectory or introduce a different law.

Write `epsilon=sqrt(m)` and choose the reference pulse section
`d(0)=0,u(0)/sqrt(beta*c)=epsilon`, with `c=cos(alpha)` and
`alpha=2*pi/5`. The two representatives have

\[
\ell_k=\lambda_k/2=1-\cos q_k,\qquad
g_k=\tan\alpha\sin q_k,\qquad q_k=2\pi k/5,\quad k=1,2 .
\]

In particular `0<ell_k<2`, and neither value is an integer. Their
collective limiting multipliers are
`exp(+/-2*pi*i*ell_k)` for the full return and
`exp(+/-pi*i*ell_k)` for the swap-correct half return. They are
distinct from the internal limiting multiplier `+1`. This separation
permits a local two-dimensional internal spectral reduction without
discarding its coupling to the collective coordinates.

### 11.1. Fix the period before expanding the coefficients

The dimensionless pulse period is `4*K(m)` in `tau=Omega*t`.
Use the period-normalized coordinate

\[
\sigma=\omega(m)\tau,\qquad
\omega(m)=\frac{\pi}{2K(m)}
         =1-\frac m4+O(m^2).
\]

The full period is now exactly `2*pi`. The analytic pulse equations
and analytic period give, uniformly on this fixed interval,

\[
d(\sigma,\epsilon)=\epsilon\sin\sigma+O(\epsilon^3),
\quad
\cos^2d=1-\epsilon^2\sin^2\sigma+O(\epsilon^4),
\quad
\cos d\sin d=\epsilon\sin\sigma+O(\epsilon^3).
\]

Oddness of the initial section and vector field makes `d` odd in
`epsilon`. These are analytic Taylor expansions on a fixed compact
interval, with bounded remainders for sufficiently small amplitude;
they do not use a simulated periodic response.

For one representative mode suppress the index `k`, put
`gamma=sqrt(ell)*g` and use
`y=eta/sqrt(ell), z=zeta` in its real block. Eliminating the two
form derivatives yields the symmetric second-order equations. After
the time change they read

\[
\begin{aligned}
y_{\sigma\sigma}
 &+\ell^2[1+\epsilon^2(1/2-\sin^2\sigma)]y
   +\gamma\epsilon\sin\sigma\,z
   +O(\epsilon^4)y+O(\epsilon^3)z=0,\\
z_{\sigma\sigma}
 &+[1+\epsilon^2 f(\sigma)]z
   +\gamma\epsilon\sin\sigma\,y
   +O(\epsilon^4)z+O(\epsilon^3)y=0,\\
f(\sigma)
 &=\frac12-(2-\ell)\sin^2\sigma .
\end{aligned}
\]

The `1/2` terms come from `omega(m)^(-2)=1+m/2+O(m^2)`.
Leaving the reference period uncorrected would change the internal
resonant coefficients.

### 11.2. The induced collective motion changes the splitting

At zero amplitude the internal solution is
`z_0=A*cos(sigma)+B*sin(sigma)`. The internal spectral subspace
induces a collective correction `y=epsilon*y_1+O(epsilon^2)`.
Its leading `2*pi`-periodic solution satisfies

\[
y_1''+\ell^2y_1=-\gamma\sin\sigma\,z_0 .
\]

There is a unique periodic solution because `ell` is not an integer.
It is

\[
\boxed{
y_1=
-\frac{\gamma A}{2(\ell^2-4)}\sin2\sigma
-\frac{\gamma B}{2\ell^2}
+\frac{\gamma B}{2(\ell^2-4)}\cos2\sigma .}
\]

The denominators `ell^2` and `ell^2-4` retain respectively the
constant and second-harmonic collective responses. Setting `y_1=0`
would not be the full fine variational problem.

Substitution in the internal equation, followed by projection onto
`cos(sigma),sin(sigma)`, gives the resonant coefficients

\[
\boxed{\begin{aligned}
P&=\frac{\ell}{4}-\frac{\gamma^2}{4(\ell^2-4)},\\
Q&=\frac{3\ell-4}{4}
   -\frac{\gamma^2}{2\ell^2}
   -\frac{\gamma^2}{4(\ell^2-4)} .
\end{aligned}}
\]

Here `P,Q` are coefficient names local to this perturbation calculation;
`Q` is not the retained internal correlation of section 8. The
diagonal part alone would give `ell/4` and `(3*ell-4)/4`; the
remaining terms are the calculated feedback from the collective
response.

To see their return-map meaning, use rotating internal amplitudes
`z=A(sigma)*cos(sigma)+B(sigma)*sin(sigma)` with the usual
variation-of-constants condition. The resonant part of their equation
is

\[
\binom{A}{B}'=
mG\binom{A}{B}+\text{higher-order and removable periodic terms},
\qquad
G=\begin{pmatrix}0&Q/2\\-P/2&0\end{pmatrix}.
\]

For example, forcing `-m*(P*A*cos(sigma)+Q*B*sin(sigma))` gives
the averaged rows `A'=m*Q*B/2` and `B'=-m*P*A/2`.
The nonresonant harmonics can be removed by a periodic near-identity
change of variables; integrating their zero-mean part over one fixed
period gives no additional leading return term. Equivalently this is
the leading analytic reduction of the isolated internal spectral
subspace. Its full return matrix in such a basis is

\[
M_{\rm int}=I+2\pi mG+O(m^{3/2}),\qquad
\sigma_*^2=-PQ/4 .
\]

The symbol `sigma_*` denotes a splitting coefficient, not the
period-normalized time `sigma`.

### 11.3. Exact algebraic signs for the two spatial modes

Using `tan(alpha)^2=5+2*sqrt(5)` and the exact cycle angles gives:

| Representative | `P` | `Q` | `sigma_*^2=-P*Q/4` |
| --- | --- | --- | --- |
| `k=1` | `(95+sqrt(5))/164` | `-(479+245*sqrt(5))/164` | `(23365+11877*sqrt(5))/53792 > 0` |
| `k=2` | `(55+21*sqrt(5))/41` | `(14+21*sqrt(5))/41` | `-(2975+1449*sqrt(5))/6724 < 0` |

All signs follow directly from positive integers and `sqrt(5)>0`.
They are exact identities, not signs extracted from a numerical
monodromy. The first internal pair therefore has a real leading
splitting and the second an imaginary one.

### 11.4. Analytic remainder and the actual sufficiently-small conclusion

The return matrices are analytic in `epsilon`: their coefficients,
the fixed-period pulse and the finite-interval linear initial-value
problem are analytic in that parameter. At `epsilon=0` the collective
pair is separated from `+1`. A sufficiently small fixed contour around
`+1` therefore defines an analytic real two-dimensional spectral
subspace of the full return. The displayed leading matrix is the
restriction to that subspace in an analytic basis.

That subspace is symplectic for sufficiently small amplitude. At zero
it is the nondegenerate internal oscillator plane, and nondegeneracy
persists by continuity. Since the full return is symplectic, the
restricted return has determinant one.

There is also an exact parity restriction. Replacing `epsilon` by
`-epsilon` changes the reference pulse by the member swap, so the
full return matrices are conjugate by `S_*`. Let `P_epsilon` be
the analytic internal spectral projector and let `V_0` be a fixed
basis for the limiting internal plane, ordered as the rotating
amplitudes. Then

\[
P_{-\epsilon}=S_*P_\epsilon S_*,
\qquad S_*V_0=-V_0,\qquad
V_\epsilon=P_\epsilon V_0,\qquad
V_{-\epsilon}=-S_*V_\epsilon .
\]

The columns of `V_epsilon` remain independent for sufficiently small
amplitude. In this basis the restricted return matrix is exactly even
in `epsilon`, because
`M(-epsilon)=S_*M(epsilon)S_*` intertwines the displayed bases.
It is therefore analytic in `m=epsilon^2` and satisfies

\[
M_{\rm int}=I+2\pi mG+O(m^2),\qquad
\frac{\log M_{\rm int}}{2\pi}=mG+O(m^2).
\]

The logarithm is the real branch near the identity. It is an effective
return generator, not the raw instantaneous rotating-amplitude row.
Trace and determinant are basis-independent, so their resulting
expansions hold even when another analytic basis is used.

Let `t_int` be the trace. Determinant one implies

\[
t_{\rm int}
 =2+4\pi^2\sigma_*^2m^2+O(m^3).
\]

Indeed `det(G)=-sigma_*^2`, so the order-two trace coefficient is
fixed by the determinant identity. In particular,

\[
\boxed{\quad
t_{\rm int}=2+4\pi^2\sigma_*^2m^2+O(m^3),\qquad
t_{\rm int}^2-4=16\pi^2\sigma_*^2m^2+O(m^3).
\quad}
\]

This establishes a remainder-controlled **existence** statement. For
each mode there are positive constants `C,m_0` bounding the remainder
by `C*m^3` on `0<m<m_0`. Because the exact leading coefficient is
nonzero, shrinking that interval makes it dominate the remainder.
No numerical values of those constants or an explicit amplitude radius
are obtained here.

For `k=1`, put

\[
\sigma_1=
\sqrt{\frac{23365+11877\sqrt5}{53792}}>0 .
\]

The internal labeled multipliers then satisfy

\[
\Lambda_\pm=1\pm2\pi\sigma_1m+O(m^2),
\]

and for every sufficiently small positive `m` they are a real
reciprocal pair with one member strictly greater than one. The
swap-correct half-return multipliers satisfy
`1+/-pi*sigma_1*m+O(m^2)`, consistently with `M=B^2`.
This is an exponentially unstable transverse mode of the prepared
periodic orbit. Both real spatial copies of this representative have
the same conclusion.

For `k=2`, put

\[
\omega_2=
\sqrt{\frac{2975+1449\sqrt5}{6724}}>0 .
\]

Its internal labeled multipliers satisfy

\[
\Lambda_\pm=1\pm2\pi i\omega_2m+O(m^2).
\]

For every sufficiently small positive `m` they are nonreal and,
because their product is one, lie exactly on the unit circle. They
are distinct and semisimple. The corresponding collective pair starts
as a simple isolated nonreal unit-circle pair and remains so for a
sufficiently small amplitude interval: leaving the circle would require
a reciprocal-conjugate collision, excluded locally by its initial
spectral separation. Thus this representative passes the bounded
**linear** transverse criterion locally. It is not a nonlinear
orbital-stability theorem for that mode, and it cannot stabilize the
whole pulse against the unstable `k=1` directions.

### 11.5. Interpretation and what remains unmeasured

The same full nonlinear law admits the exact periodic family and also
makes its sufficiently small nonzero members transversely unstable.
There is no contradiction: invariant preparation proves that the orbit
exists, whereas the return splitting asks what nearby states do. An
unstable transverse return multiplier gives nonlinear orbital
instability of this smooth periodic orbit; it does not imply that every
perturbation grows or that the exact orbit ceases to exist.

The amplification is produced by the periodic internal coefficients
and their coupled collective response. It does not require an external
periodic source or a newly installed pressure law. In the complete
conservative system, departure from that particular internal waveform
can transfer storage among its retained modes without changing total
storage. This is distinct from the neutral time-amplitude shear in
mode zero.

The harmonic mechanism is explicit. The `sin(sigma)^2` stiffness
term contains a constant part and a second harmonic. The order-
`epsilon*sin(sigma)` cross term sends the internal first harmonic
into the collective constant and second-harmonic responses; those feed
back into the internal first harmonic through the same cross term.
This is parametric feedback in the derived tangent law, not an externally
prescribed oscillatory drive, a claim about all states, or invocation of
the named Resonance operator.

Nor does orbital instability mean unbounded fine form, loss of every
coherent geometry or unavoidable winding change. For sufficiently
small pulses, the earlier acute-target trapping domain contains open
neighborhoods of their initial states: both their centered norm and
excess storage tend to zero with amplitude. Solutions from an admitted
neighborhood remain near that static phase identity, even while some
depart from the common-internal-state periodic orbit. A robust
geometric organization and a fragile synchronized waveform can
therefore coexist under the same storage balance.

The result supplies no certified numerical interval of unstable
amplitudes, no verdict for a chosen nonzero represented amplitude
solely because it is called small, no growth time at that amplitude,
and no finite-amplitude continuation of either mode class. It does
not justify changing the constitutive law to preserve a preferred
pulse, or identify this prepared organization with physical matter.

The [existing scale owner](../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_pulse_splitting`. It retains the exact
coefficients as `a+b*sqrt(5)`, encloses their signs and the leading
return-logarithm slopes, and reuses the same declared model, capacity
and symbolic target through a stationary reference template. That
reference is not itself labeled unstable. The API accepts no finite
amplitude and supplies no numerical amplitude radius, remainder
constant, computed multiplier or selected-preparation verdict.
The [replica tests](../tests/physics/test_relational_sine_replica.py)
independently check the forced collective harmonics, resonant
projections, algebraic coefficients and return normalization.

<a id="sine-replica-joint-persistence"></a>

## 12. Collective identity with independently active constituents

This theorem combines the full-state barrier of
[Section 7.5](#75-trapping-permits-persistent-internal-organization),
the retained internal law of
[Section 8](#sine-replica-unordered-state), and
[nonlinear recurrence](nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence).
It uses the complete doubled `C5` support, `e=0`, common held
capacity `nu>0`, `w,beta>0` and one structural clock. There are
no inputs, events, clipping, discarded constituents or changing edges.
The support, pair partition and zero-loss law remain supplied premises.
No attracting pulse or new constitutive mechanism is introduced.

### 12.1. An admitted open full-state family

Let `a=w/pi`, `b=w/(beta*pi)` and `alpha=2*pi/5`.
For each pair `i=0,...,4` the exact target phase is
`theta_(i,+),*=theta_(i,-),*=i*alpha`, with uniform target form.
The fine graph has ten nodes, degree four and twenty edges. Its
target storage and spectral gap are

\[
E_*=20\beta(1-\cos\alpha),\qquad
\lambda_{2,f}=5-\sqrt5 .
\]

Reversing the base cycle orientation gives the same conclusions for
winding minus one, with `|alpha|` in the radius condition.

Write `P_f=I-11^T/10`, `v=P_f x` and use the phase chart

\[
\theta=\theta_*+c\mathbf1+h\pmod{2\pi},
\qquad h\perp\mathbf1,\qquad c\in\mathbb R/2\pi\mathbb Z .
\]

The common circular phase origin is retained. In the same fixed model
coordinates as the preceding barrier, define

\[
Z_f^2=\|v\|^2+\|h\|^2,\qquad
0<r,\quad \alpha+\sqrt2r<\frac\pi2,
\]

\[
c_r=\cos(\alpha+\sqrt2r)>0,\qquad
\kappa_f=\frac{5-\sqrt5}{2}\min(1,\beta c_r)>0 .
\]

This is a genuine chart throughout the closed radius ball and a
slightly larger neighborhood. Indeed two representations of the same
circular state would have
`(h_i-h_j)-(h'_i-h'_j)=2*pi*(n_i-n_j)`. Its magnitude is
at most `2*sqrt(2)*r<2*pi`, forcing all integers equal;
centering then gives `h=h'` and the same `c` on its circle.
The chart differential is invertible. In particular there is no hidden
wrapping boundary inside the ball.

For `Z_f<=r` every fine edge has its target-compatible increment
`+/-alpha+h_j-h_i`, strictly inside `(-pi/2,pi/2)`.
The target is critical. Its vanishing first variation, the phase Hessian
bound `H_phase>=c_r*L_f` along the segment to the target, and the
fine spectral gap give

\[
\mathcal E:=E_f-E_*
\ge\frac{\lambda_{2,f}}2\|v\|^2
+\frac{\beta c_r\lambda_{2,f}}2\|h\|^2
\ge\kappa_f Z_f^2 .
\]

Choose an excess ceiling and finite positive-width mean interval,

\[
0<\varepsilon_*<\kappa_f r^2,\qquad
m_{\rm lo}<m_{\rm hi}.
\]

Here `epsilon_*` is a family ceiling, not the small-amplitude
parameter of Section 11. Because fine degree and capacity are common,
the conserved weighted form mean is the ordinary mean
`m(x)=sum(x)/10`. Define

\[
\boxed{\mathcal U=
\{Z_f^2<r^2,\quad \mathcal E<\varepsilon_*,
                 \quad m_{\rm lo}<m(x)<m_{\rm hi}\}.}
\]

Energy and mean conservation prevent a first radius exit in either
time direction: an exit would require
`mathcal E>=kappa_f*r^2>epsilon_*`. Consequently

\[
Z_f(t)^2\le\frac{\mathcal E(0)}{\kappa_f}
 <\frac{\varepsilon_*}{\kappa_f}<r^2
\qquad(t\in\mathbb R).
\]

The full smooth flow is complete on the bounding energy/mean slab
by the recurrence owner's compactness argument. The chart, both strict
inequalities and the mean interval are preserved, so
`Phi_t(U)=U` for every real `t`. Fine edges remain acute and
every cycle retains its target winding. In particular every five-edge
loop following the positive base orientation and choosing one constituent
from each consecutive pair retains winding one.

The family is open in the full twenty-dimensional state manifold,
not just in a synchronized subspace. Form coordinates are
`x=m*1+v` with `v perpendicular 1`. Together with the phase
chart these are locally invertible coordinates, and all inequalities
are strict. A uniform-form target with its mean inside the interval
has an open neighborhood in `U`, proving positive volume. Its
closure is compact: `|x_i-m|<=||v||<=r` bounds form, and phases
belong to a compact torus. Thus `U` has finite positive volume
for fine-form Lebesgue measure times phase-torus Haar measure.

### 12.2. Every nontip pair remains internally active

In the retained pair coordinates the fine norm splits exactly as

\[
Z_f^2=2\|P_5X\|^2+
      2\|P_5(\Theta-\Theta_*)\|^2+
      2\sum_i(u_i^2+\delta_i^2).
\]

Hence `|delta_i|<D:=r/sqrt(2)<pi/2`, so the nonantipodal
pair chart remains valid for all time. Let

\[
\mathcal T_i=\{u_i=\delta_i=0\},\qquad
\mathcal U_{\rm active}
=\mathcal U\setminus\bigcup_{i=0}^4\mathcal T_i .
\]

Each synchronized tip `T_i` is invariant under the **full** smooth
flow: `u_i_dot=-A_i*sin(delta_i)` and
`delta_i_dot=b*nu*u_i` vanish there, irrespective of the motion
in other pairs. Two-sided uniqueness then makes its complement invariant.
A nontip pair cannot arrive at that tip at a finite time. This conclusion
uses realizable fine coordinates, rather than only the polynomial
constraint among `R,U,Q`.

Each `T_i` has codimension two in the fine chart and zero ambient
volume. Therefore `U_active` is still open, of positive finite
volume, and invariant for every real time. Removing the five tips does
not remove any positive-volume portion of `U`.

The acute full-state geometry supplies a uniform restoring sign. Define

\[
F_i=\frac12\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),
\qquad A_i=a\nu F_i,\qquad c_i=b\nu .
\]

For a base edge `{i,j}`, averaging the cosines of its four actual
fine edge increments gives

\[
R_iR_j\cos(\Theta_j-\Theta_i)
 =\frac14\sum_{s,t=\pm1}
       \cos(\Theta_j+t\delta_j-\Theta_i-s\delta_i)
 \ge c_r .
\]

Since `0<R_i<=1`, each summand
`R_j*cos(Theta_j-Theta_i)>=c_r`. Its upper bound is one.
Consequently

\[
\boxed{c_r\le F_i\le1,\qquad A_i\ge a\nu c_r>0.}
\]

At any nontip state the unordered internal velocity is nonzero.
If `Q_i!=0`, then `R_i_dot=-c_i*Q_i!=0`. If `Q_i=0`,
the remaining nontip possibilities in this chart are:

- `R_i=1,U_i>0`, where `Q_i_dot=c_i*U_i>0`;
- `U_i=0,R_i<1`, where
  `Q_i_dot=-A_i*(1-R_i^2)<0`.

Thus every pair of every state in `U_active` has a moving internal
state `(R_i,U_i,Q_i)` at every finite time. This does not require
each separate scalar coordinate or each fine node rate to be nonzero.
It also does not give a uniform positive lower bound on the velocity
norm: the open family contains states arbitrarily close to a tip.

### 12.3. Bounded internal circulation without a common period

There is a stronger all-state consequence of the same retained rows.
Normalize internal form only for this calculation:

\[
y_i=\frac{u_i}{\sqrt\beta},\qquad
\Omega=\frac{w\nu}{\pi\sqrt\beta}>0.
\]

Then along the full interacting trajectory

\[
\dot\delta_i=\Omega y_i,\qquad
\dot y_i=-\Omega F_i(t)\sin\delta_i,\qquad c_r\le F_i(t)\le1 .
\]

The coefficient `F_i(t)` is generated by the other retained
coordinates; it is not an imposed periodic drive. Since
`(delta_i,y_i)` never equals zero, its argument has a continuous
real lift `psi_i=arg(delta_i+i*y_i)` for all real time. Direct
differentiation yields

\[
-\dot\psi_i
=\Omega\frac{F_i(t)\delta_i\sin\delta_i+y_i^2}
                 {\delta_i^2+y_i^2}.
\]

For `|delta_i|<D<pi/2`,
`sinc(D)<=sin(delta_i)/delta_i<=1`, taking the continuous
value one at zero. The quadratic quotient therefore satisfies

\[
\boxed{\Omega c_r\,\operatorname{sinc}(D)
        \le-\dot\psi_i\le\Omega,\qquad
        \operatorname{sinc}(D)=\frac{\sin D}{D}>0.}
\]

Each constituent pair undergoes endlessly repeated turns in this
internal phase plane in forward time, and oppositely in backward time.
From any initial angular value, the unique next full-turn crossing
has elapsed time `Delta t_i` bounded by

\[
\frac{2\pi}{\Omega}\le\Delta t_i
\le\frac{2\pi}{\Omega c_r\operatorname{sinc}(D)}.
\]

Successive axis crossings have analogous quarter-turn bounds. These are bounds
on angular traversal under the declared structural clock, **not**
on recurrence of the internal amplitude or the full state. Different
pairs may change amplitude, modulate their traversal times and exchange
storage; no exact shared frequency or phase locking has been assumed
or proved. Each pair is active, but their dynamics are still coupled;
no statistical independence is claimed. A pair swap shifts `psi_i`
by `pi` and leaves its derivative unchanged. A projective advance of
`pi` likewise need
not return the unordered state unless its amplitude also returns.
No positive minimum amplitude follows from these angular bounds.

### 12.4. Almost-everywhere recurrence in the same family

The full conservative sine flow has zero divergence in fine form and
circular phase, as proved by the
[recurrence owner](nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence).
Restrict that preserved measure to the finite-volume invariant
`U_active`. The same finite-measure recurrence argument gives,
for every fixed sampling increment `s>0`,

\[
\Phi_{n_js}(z)\longrightarrow z,\qquad n_j\longrightarrow\infty,
\quad\text{for almost every }z\in\mathcal U_{\rm active}.
\]

All these recurrent states are nonstationary by Section 12.2.
Here recurrence is measured on the full fine state with circular
phases. It is not inferred using ambient Lebesgue measure in the
twenty-five constrained invariant coordinates `(X,Theta,R,U,Q)`.
An absolutely continuous preparation distribution on the fine family
inherits the almost-sure conclusion. A selected state, finite grid,
fixed-energy surface or other singular preparation does not receive a
recurrence certificate from this ambient-measure theorem.

The quantifiers are distinct: **every** admitted state preserves its
geometry and its five active internal constituents, with the angular
traversal bounds above; **almost every** such state also returns
arbitrarily near its full initial state. There is no full-state return
deadline, generic exact period or numerical chosen-state guarantee.

### 12.5. Overlap with the prepared pulse and its instability

Take a pulse preparation from Section 9 at a phase crossing:
`X_i=m` inside the mean interval, `Theta_i=i*alpha+c`,
`delta_i=0` and the same `u_i=sqrt(H)>0` in all five pairs.
At that instant

\[
Z_f^2=10H,\qquad \mathcal E=20H.
\]

Thus the explicit sufficient conditions

\[
0<H<
\min\left(\frac{r^2}{10},
          \frac{\varepsilon_*}{20},\beta\cos\alpha\right)
\]

put the preparation in `U_active` and in the nonlinear libration
family. Every bound on the right is strictly positive. The entire pulse
and an open full-state neighborhood of its preparation consequently
remain in the admitted collective family, with all constituents active.

Section 11 supplies an existential `m_*>0` such that the prepared
pulse is transversely orbitally unstable for
`0<m=H/(beta*cos(alpha))<m_*`. Intersecting this interval with
the preceding positive interval proves a nonempty overlap without
computing `m_*`. There is no contradiction: the larger phase
organization and internal circulation remain protected while an exact
common waveform can be fragile. Nearby states are not thereby proved
periodic, synchronized, or convergent to another selected pulse.

Finally, two-sided invariance itself prevents capture into
`U_active` from its complement. The
[formation boundary](nodal/RESONANCE_FOUNDATIONS.md#conservative-formation-boundary)
still applies. This is a conditional joint **maintenance** result for
a supplied organization, not autonomous construction of its support or
partition, selection of microscopic zero loss, universal fractality,
or identification of a material constituent.

### 12.6. Captured-source admission and evidence

The existing [scale owner](../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_persistence` and
`SineReplicaPersistenceAssessment`. It checks the complete ordered
doubled-cycle support, common positive capacity and conservative model
on one retained capture. Exact target turns are separate from represented
source phases. The
[shared geometry owner](../src/tnfr/physics/relational_sine_recovery.py)
supplies whole-source norm, canceled excess-storage and first-exit bounds;
its general rational spectral lower bound can be weaker than the exact
`5-sqrt(5)` used above without invalidating an admitted certificate.

The reader distinguishes the declared finite-volume family's admission,
source trapping, each pair's exact tip status, and source membership
in the stricter excess/absolute-mean family. Its captured form values
supply the actual mean; a relative observation alone would not provide
that common origin. Trapping and nontip activity may be certified even
when the source is outside the smaller declared mean/excess family.
Family almost-everywhere recurrence never becomes selected-source
recurrence through these checks.

For numerical angular bounds the reader uses the safe lower factor
`c_r*cos(D)<=c_r*sinc(D)`. The inequality follows from
`sin(D)-D*cos(D)>=0` for `0<=D<pi/2`. An outward speed
enclosure may touch zero for an extremely small exact positive capacity;
that numerical limitation does not negate strict angular circulation
under the admitted theorem. A finite upper traversal-time bound remains
an angular bound rather than a state-return deadline.

The [replica tests](../tests/physics/test_relational_sine_replica.py)
check the complete source rows and geometric prerequisites independently.
The reader evaluates no trajectory, waits for no recurrence, changes
no law, and assigns no finite-amplitude instability radius.

<a id="sine-replica-capacity-asymmetry"></a>

## 13. Unequal constituent capacities: persistence and its symmetry boundary

Keep the same complete conservative sine law and doubled-cycle support.
Change only the supplied held capacities: each constituent now has its
own `nu_(i,+)>0` and `nu_(i,-)>0`. Define

\[
v_i=\frac{\nu_{i,+}+\nu_{i,-}}2,\qquad
\eta_i=\frac{\nu_{i,+}-\nu_{i,-}}2,\qquad
v_i>|\eta_i|.
\]

The letter `v_i` in this section is mean capacity, not the centered
fine-form vector used in Section 12. Both `v_i` and `eta_i` are
held parameters. The same form/phase coordinates
`x_(i,+/-)=X_i+/-u_i`, `theta_(i,+/-)=Theta_i+/-delta_i`
remain invertible on supplied lifts. The pair chart is still
`|delta_i|<pi/2`. No new force, state law or capacity evolution is
postulated.

### 13.1. Exact asymmetric mean and internal rows

For a base degree `d_i` define the normalized neighboring sums

\[
S_i=\frac1{d_i}\sum_{j\sim i}R_j\sin(\Theta_j-\Theta_i),
\quad
C_i=\frac1{d_i}\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),
\quad
G_i=\frac1{d_i}\sum_{j\sim i}(X_i-X_j),
\quad R_i=\cos\delta_i .
\]

For the doubled `C5`, `d_i=2`. Before multiplying by the
individual capacity, the two fine form rows are
`a*(R_i*S_i-/+sin(delta_i)*C_i)`, and the two fine phase rows
are `b*(G_i+/-u_i)`. Their exact half-sums and half-differences
therefore give

\[
\boxed{\begin{aligned}
\dot X_i&=a(v_iR_iS_i-\eta_i\sin\delta_i\,C_i),\\
\dot\Theta_i&=b(v_iG_i+\eta_i u_i),\\
\dot u_i&=a(\eta_iR_iS_i-v_i\sin\delta_i\,C_i),\\
\dot\delta_i&=b(\eta_iG_i+v_i u_i).
\end{aligned}}
\]

The additional terms are consequences of multiplying the existing fine
rows by unequal capacities. Setting `eta_i=0` recovers Section 7,
even if `v_i` differs between pairs. Variation **between** equal
member pairs and inequality **within** a pair are thus distinct
changes. No rounded near-equality can replace the exact equality used
in the earlier invariant-tip theorem.

The inherited signed Poisson bracket makes the same distinction.
With `k_i=b/(4*d_i)` its nonzero independent entries are

\[
\{X_i,\Theta_i\}=\{u_i,\delta_i\}=-k_i v_i,\qquad
\{X_i,\delta_i\}=\{u_i,\Theta_i\}=-k_i\eta_i .
\]

Different pairs have zero brackets. These entries follow by taking
half-sums/differences of
`{x_(i,+/-),theta_(i,+/-)}=-b*nu_(i,+/-)/(2*d_i)`.
The bracket is constant for held capacities and hence satisfies Jacobi.
Its determinant in each four-coordinate block is
`k_i^4*(v_i^2-eta_i^2)^2>0`. The same full fine storage from
Section 7 generates all four displayed rows, including the cross terms.
Equal capacities eliminate those cross terms; they do not supply the
underlying storage balance.

### 13.2. Collective trapping and ambient recurrence survive

The exact storage still satisfies `E_f_dot=0`. The full field
still has zero divergence: each fine form row consumes phase, and
each fine phase row consumes form, with all capacities held. Its
conserved form mean is now

\[
m_\rho(x)=
\frac{\sum_{i,s}\rho_{i,s}x_{i,s}}{\sum_{i,s}\rho_{i,s}},
\qquad \rho_{i,s}=\frac{2d_i}{v_i+s\eta_i}>0 .
\]

In retained coordinates this is

\[
\boxed{
m_\rho=
\frac{\displaystyle\sum_i
      \frac{d_i(v_iX_i-\eta_i u_i)}{v_i^2-\eta_i^2}}
     {\displaystyle\sum_i
      \frac{d_i v_i}{v_i^2-\eta_i^2}}.}
\]

The ordinary mean is not a substitute. For the same winding-one
target the fine storage, spectral gap and acute geometry do not depend
on capacity. Retain `r,kappa_f,E_*` from Section 12, choose
`0<epsilon_*<kappa_f*r^2` and a finite open interval
`m_lo<m_hi`, and define

\[
\mathcal U_\rho=
\{Z_f^2<r^2,\ E_f-E_*<\varepsilon_*,
                 \ m_{\rm lo}<m_\rho<m_{\rm hi}\}.
\]

For each supplied positive held capacity list this is an open invariant
family with finite positive fine volume. The same energy argument
prevents radius exit in both time directions. To verify the volume
claim explicitly, use centered form `z=P_f x` and write

\[
x=\left(m_\rho-\frac{\rho^\mathsf Tz}
                          {\rho^\mathsf T\mathbf1}\right)\mathbf1+z .
\]

This is an invertible form coordinate change. Positive `rho` makes
`m_rho` a convex average, so

\[
|x_j-m_\rho|
\le\max_k|x_j-x_k|
\le\sqrt2\,\|z\|<\sqrt2r .
\]

Thus the closure of `U_rho` lies in a bounded form box times the
phase torus; target neighborhoods prove positive volume. The smooth
two-sided flow preserves the family and its fine ambient measure.
The existing recurrence proof gives almost-everywhere nonstationary
recurrence there, with its original selected-state and singular-measure
limitations. This family includes synchronized tips: deleting them
would generally destroy invariance, as the next subsection shows.

These conclusions apply separately to each fixed positive capacity
list. They do not install a time-varying capacity law, give a common
return deadline, or transfer the equal-capacity internal-circulation
claim. The singular boundary at a zero constituent capacity is outside
this theorem.

### 13.3. Exact counterexamples to tip invariance and pointwise activity

At a synchronized tip `u_i=delta_i=0` the asymmetric internal
rows reduce to

\[
\dot u_i=a\eta_i S_i,\qquad
\dot\delta_i=b\eta_i G_i .
\]

They need not vanish. The tip and its complement are therefore not
generally invariant. This is a structural obstruction, not a numerical
threshold effect.

An explicit preparation also shows that a **nontip** pair can have
zero instantaneous internal velocity. Take `v_i=1` for all pairs,
`eta_0=eta` with `0<eta<1`, and all other `eta_i=0`.
Use exact target phases `Theta_i=i*alpha` and `delta_i=0`.
Prepare

\[
X_0=\epsilon,\quad X_{i\ne0}=0,\qquad
u_0=-\eta\epsilon,\quad u_{i\ne0}=0,\qquad \epsilon>0.
\]

All `S_i=0` and `G_0=epsilon`. Hence

\[
\dot u_0=\dot\delta_0=0,\qquad
u_0\ne0,\qquad
\dot\Theta_0=b\epsilon(1-\eta^2)>0 .
\]

The pair's old invariant rates `R_dot,U_dot,Q_dot` all vanish
at this instant, although its mean phase moves. This also makes the
normalized signed internal angular velocity zero, excluding the earlier
strict positive angular lower bound. The storage excess and full norm
are exactly

\[
E_f-E_*=4(1+\eta^2)\epsilon^2,\qquad
Z_f^2=\left(\frac85+2\eta^2\right)\epsilon^2 .
\]

For example `eta=1/2,epsilon=1/1024` gives capacities
`(nu_(0,+),nu_(0,-))=(3/2,1/2)` and forms
`(x_(0,+),x_(0,-))=(epsilon/2,3*epsilon/2)`, with
excess `5*epsilon^2` and norm squared `21*epsilon^2/10`.
Its conserved weighted mean is `5*epsilon/16`, rather than
the ordinary mean `epsilon/5`. The target phases in this proof
are symbolic angles; substituting rounded radians would not prove the
exact zero derivative.

The zero internal rate at pair zero also has an exactly represented
phase control. Choose rational `h` near `alpha` and raw pair
means `(0,h,2h,-2h,-h)`, with respective full-turn lifts
`(0,0,0,1,1)`. The two neighboring phases of pair zero are
`+h,-h`, so their sine currents cancel exactly. The same form
preparation still has `u_0_dot=delta_0_dot=0` without claiming
that the entire phase state is critical. Taking rational `h`
arbitrarily close to `alpha` preserves any strict neighborhood
admission of the symbolic preparation by continuity.

Both the asymmetry `eta` and preparation size `epsilon` can
be arbitrarily small. For any fixed admitted radius and excess ceiling,
sufficiently small `epsilon` puts the state in the collective
trapping family; a common form translation places its weighted mean
inside the chosen slab without changing any displayed rate or bound.
Thus collective geometric persistence can coexist with an internal
stall arbitrarily close to the symmetric regime.

For a separate tip-crossing control keep the same preparation but set
`u_0=0`. Now `delta_0_dot=b*eta*epsilon!=0` at the tip.
Smooth local uniqueness gives nontip states just before and after this
instant on the same full solution. Choosing the preparation inside
`U_rho` retains those states in its two-sided invariant family.
A nontip preparation can consequently reach the tip at a finite time.

Neither control proves permanent internal cessation, loss of the
collective winding, or absence of later turns. They disprove the
specific all-state no-tip-entry and nonzero-internal-velocity statements
outside their equal-member-capacity premises. They do not contradict
the surviving full-state recurrence theorem.

### 13.4. The unordered state must retain capacity correlations

Swapping only `(u_i,delta_i)` while fixing unequal labeled
capacities is no longer a symmetry. For example at `delta_i=0`,
the two forms `u_i` and `-u_i` have the same old `R,U,Q`
but different mean phase rates through `eta_i*u_i`. Even retaining
the signed parameter `eta_i` does not restore closure of those
old observed coordinates.

For an unordered pair, interchange the **whole** constituent,
including its held capacity. This exact relabeling sends
`(eta_i,u_i,delta_i)` to `(-eta_i,-u_i,-delta_i)` while
preserving `v_i,X_i,Theta_i`. The missing correlations can be
retained as

\[
P_i=\eta_i u_i,\qquad T_i=\eta_i\sin\delta_i,\qquad
E_i=\eta_i^2 .
\]

Here `E_i` is squared capacity contrast, not the total fine
storage `E_f`. Together with `R_i,U_i,Q_i` these quantities
identify the exact joint swap orbits on the pair chart. Their realizable
domain is

\[
0<R_i\le1,\quad U_i,E_i\ge0,\quad v_i>0,\quad E_i<v_i^2,
\]

\[
Q_i^2=U_i(1-R_i^2),\quad
P_i^2=E_iU_i,\quad T_i^2=E_i(1-R_i^2),\quad
P_iT_i=E_iQ_i .
\]

Equivalently the symmetric matrix

\[
\begin{pmatrix}
E_i&P_i&T_i\\
P_i&U_i&Q_i\\
T_i&Q_i&1-R_i^2
\end{pmatrix}
\]

is positive semidefinite of rank at most one: it is the outer product
of `(eta_i,u_i,sin(delta_i))` with itself. If `E_i>0`,
choose `eta_i=sqrt(E_i)`, `u_i=P_i/eta_i` and
`delta_i=asin(T_i/eta_i)`; its only other lift is the simultaneous
sign reversal. If `E_i=0`, then `P_i=T_i=0` and the
earlier `R,U,Q` lifting applies. This is a finite symmetry
quotient with retained internal information, not fewer continuous
dynamical degrees of freedom.

The exact forward equations require no division by `eta_i`:

\[
\boxed{\begin{aligned}
\dot X_i&=a(v_iR_iS_i-T_iC_i),\\
\dot\Theta_i&=b(v_iG_i+P_i),\\
\dot R_i&=-b(v_iQ_i+T_iG_i),\\
\dot U_i&=2a(P_iR_iS_i-v_iQ_iC_i),\\
\dot Q_i&=a(R_iT_iS_i-v_i(1-R_i^2)C_i)
              +bR_i(P_iG_i+v_iU_i),\\
\dot P_i&=a(E_iR_iS_i-v_iT_iC_i),\\
\dot T_i&=bR_i(E_iG_i+v_iP_i),\\
\dot E_i&=\dot v_i=0 .
\end{aligned}}
\]

These are exact derivatives of the realized invariants, so their
constraints, the same fine storage and their lift are preserved for
as long as the pair chart persists. The displayed algebra is not
permission to evolve arbitrary incompatible interval tuples; domain
preservation comes from the fine lift and smooth uniqueness. On
`U_rho` the full-state barrier keeps the chart for all time.
At `E_i=0` the correlations vanish and the equal-capacity
closure is recovered without a singular forward limit.

Their joint consistency can also be checked without expanding separate
squared constraints. The vector `z_i=(eta_i,u_i,sin(delta_i))`
obeys `z_i_dot=M_i*z_i` along the full solution, where

\[
M_i=\begin{pmatrix}
0&0&0\\
aR_iS_i&0&-av_iC_i\\
bR_iG_i&bR_iv_i&0
\end{pmatrix}.
\]

Consequently its Gram matrix obeys
`(z_i*z_i^T)_dot=M_i*(z_i*z_i^T)+(z_i*z_i^T)*M_i^T`.
This is an exact consistency identity with coefficients supplied by the
same retained state, not a separately prescribed linear model.

### 13.5. Weighted centers remove direct cross terms, not internal information

One useful coordinate check uses `lambda_i=eta_i/v_i` and
the inverse-capacity weighted pair centers

\[
\widehat X_i=X_i-\lambda_i u_i,\qquad
\widehat\Theta_i=\Theta_i-\lambda_i\delta_i,\qquad
\nu_{{\rm eff},i}=\frac{v_i^2-\eta_i^2}{v_i}
=\frac{2\nu_{i,+}\nu_{i,-}}{\nu_{i,+}+\nu_{i,-}} .
\]

Because capacities are held, the exact rows give

\[
\dot{\widehat X}_i=a\nu_{{\rm eff},i}R_iS_i,\qquad
\dot{\widehat\Theta}_i=b\nu_{{\rm eff},i}G_i .
\]

This explains the harmonic-mean capacity and cancellation of direct
cross terms in weighted coordinates. It does **not** give an autonomous
law for those means: `G_i` still consumes
`X_i=widehat(X)_i+lambda_i*u_i` and `S_i` consumes
`Theta_i=widehat(Theta)_i+lambda_i*delta_i` and neighboring
`R_j`. The weighted phase center is only a local lifted coordinate;
unequal noninteger phase weights do not define a global circular average.
Nor does a change of mean coordinates undo the actual signed
internal stall in Section 13.3.

The symmetry boundary therefore refines the persistent-pattern claim:
positive held capacities preserve the conservative storage, local
collective identity and ambient recurrence result, while the stronger
all-pair circulation theorem requires its stated symmetry. Capacity
correlations restore an exact unordered description of the asymmetric
law; they do not restore that stronger theorem, select capacities,
or prove autonomous formation of the graph or material constituents.

### 13.6. Shared engine scope

The [scale owner](../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_capacity` and
`SineReplicaCapacityAssessment`. It reuses one full capture and
the shared support/lift coordinates, then compares the four signed rows
and eight capacity-aware invariant rows against the fine pushforward.
Its storage gradients and two signed Poisson coefficients retain the
cross terms. Constraint residuals check that capture's realizable image;
the reader does not admit arbitrary independent invariant tuples.
Capacity contrasts refer to the captured represented values, rather
than precision discarded before that capture.

Optional `SineReplicaCapacityFamily` evidence uses the shared acute
barrier, exact degree/capacity mean weights and a declared radius,
excess ceiling and open mean slab on the complete ordered doubled
`C5`. It includes tips and distinguishes family admission, source
trapping, source membership and ambient almost-everywhere recurrence.
It supplies neither an all-time internal-activity certificate nor an
angular-circulation certificate for unequal capacities.

The earlier equal-capacity readers retain their exact capacity
requirements. The [replica tests](../tests/physics/test_relational_sine_replica.py)
separately check fine-row factorization, capacity-state swaps,
constraints, weighted conservation and the symmetry-breaking controls.
Neither reader changes the model, adjusts a capacity, evolves a
trajectory or promotes a finite residual check to recurrence of the
captured state.

<a id="sine-phase-pairing"></a>

## 14. State-derived phase partners and independent law admission

The pair partition in Sections 7–13 specifies which collective description
is being tested. This section asks a separate question: can the **phase
state itself** identify those pairs without consuming the proposed
partition, support edges, capacities, forms or a label convention? On the
protected doubled-`C5` family the answer is yes. On a general snapshot
the same observation must be allowed to abstain, and a detected partition
still needs independent support and complete-law admission.

### 14.1. A branch-independent comparison of circular distances

For any two fine nodes define their circular distance and squared chord,

\[
d_{\mathbb S^1}(\theta_p,\theta_q)
 =\min_{n\in\mathbb Z}|\theta_q-\theta_p+2\pi n|
 \in[0,\pi],
\]

\[
D_{pq}=|e^{i\theta_q}-e^{i\theta_p}|^2
      =2[1-\cos(\theta_q-\theta_p)]
      =4\sin^2\!\left(\frac{d_{\mathbb S^1}(\theta_p,\theta_q)}2\right).
\]

The function `d -> 2*(1-cos(d))` is strictly increasing on
`[0,pi]`. Thus comparing squared chords gives exactly the same
nearest-partner ordering as comparing circular distances. It requires
no inverse trigonometric function, common phase origin, branch cut or
choice of real phase lifts.

This observation is invariant under common circular rotation, integer
full-turn changes of phase representatives, simultaneous reversal of
phase orientation, and permutation of node labels. The returned
partition is an unordered set of unordered pairs; serialization order
is not an extra physical choice. These are mathematical invariances of
the admitted phase state. Adding a large common floating-point offset
can discard previously represented phase differences; such a changed
capture is not an exact symmetry control.

### 14.2. The protected tube separates true partners

Use the exact winding-one doubled-`C5` target and phase chart of
Section 12,

\[
\theta=\theta_*+c\mathbf1+h\pmod{2\pi},\qquad
\|h\|\le Z_f<r,\qquad
\alpha=\frac{2\pi}{5},\qquad
\alpha+\sqrt2r<\frac\pi2 .
\]

For the two members `p,q` of one target pair their target phases
coincide. Cauchy–Schwarz therefore gives

\[
d_{\mathbb S^1}(\theta_p,\theta_q)
\le |h_q-h_p|
\le\sqrt2\,\|h\|
<\sqrt2r<\frac\pi{10}.
\]

For nodes in different target pairs the target circular distance is
either `alpha` or `2*alpha`. The triangle inequality on the
circle, applied after canceling the common origin, gives

\[
d_{\mathbb S^1}(\theta_p,\theta_q)
\ge d_{\mathbb S^1}(\theta_{*,p},\theta_{*,q})
                -d_{\mathbb S^1}(h_p,h_q)
>\alpha-\sqrt2r>\frac{3\pi}{10}.
\]

Every node consequently has exactly one nearest phase neighbor: its
other constituent. Each choice is mutual. The inequalities also give
the strict chord separation

\[
\max_{\text{within pair}}D_{pq}
<2[1-\cos(\sqrt2r)]
<2[1-\cos(\alpha-\sqrt2r)]
<\min_{\text{between pairs}}D_{pq}.
\]

The target and partition appear here to **prove recovery**, not as
inputs to the nearest-partner observation. The proof applies equally
to the reversed winding. It consumes no capacity equality. In the
unequal-positive-held-capacity family `U_rho` of Section 13, the
same full-state barrier preserves the phase tube for every real time.
Hence the phase-only observation identifies the same constituent pairs
throughout the full trajectory, including possible internal stalls or
tip crossings. This is an all-state identification consequence of
geometric trapping, independent of almost-everywhere recurrence.

No common frequency, internal angular monotonicity or threshold for
declaring a link was needed. The numerical observer need not consume
`r` or either angular threshold: it only compares the actual
phase distances.

### 14.3. A prospective nearest-partner certificate and its abstentions

For a finite captured phase list, let `[L_pq,U_pq]` enclose each
squared chord. A sufficient certificate that `q` is the unique
nearest neighbor of `p` is

\[
U_{pq}<L_{pk}\qquad
\text{for every }k\ne p,q .
\]

When each node has one such certified neighbor, retain the full
partition only if those choices are mutual. A complete mutual map has
no fixed points and partitions the nodes into disjoint pairs. This
does not require selecting a distance cutoff, minimizing a global
matching cost or fitting a grouping scale.

Strict comparison is essential. Equal distances do not specify one
partner, and overlapping enclosures do not prove either equality or
unique ordering. Odd cardinality, unmatched nodes and nonmutual
nearest choices prevent a complete pair partition. In those cases
the observation abstains; it does not break ties using node names,
greedily rematch unused nodes or borrow a partition from graph topology.

For example, the four exact phases `(0,1/10,3/10,1)` lie within
one short circular arc and each has a unique nearest neighbor. The
first two choose one another, the third chooses the second and the
fourth chooses the third. The choices do not define a complete mutual
pairing. Three coincident phases instead give actual nearest-neighbor
ties. Both are admissible circular observations, but neither justifies
an arbitrary choice of pairs.

The geometric theorem establishes exact separation throughout its
admitted family. A fixed-precision implementation still must enclose
the captured distances and prove its strict inequalities; numerical
range limits or wide enclosures can cause abstention. Conversely,
an unambiguous match outside that family is only an instantaneous
observation. It supplies no future partner guarantee by itself.

### 14.4. Phase pairing does not certify the paired support

After the phase observation, the proposed pairs must independently
satisfy the existing complete-replica support contract: every fine node
occurs once, there are no within-pair edges, each active base edge has
all four cross-edges, and the remaining law, capacity and phase-chart
hypotheses hold. This check may inspect edges; the preceding observation
may not. An asymmetric-capacity consumer must retain the correlations
of Section 13 rather than silently invoking an equal-capacity theorem.

A state-only permutation supplies an explicit discriminating control.
Start with the actual doubled-`C5` support and give both members
of pair `i` the same phase `phi_i`, using five distinct circular
phases. Now exchange **only** the phase attributes of nodes
`(0,-)` and `(1,-)`. Leave graph edges, node labels, capacities
and forms untouched. The unique observed zero-distance pairs become

\[
\{(0,+),(1,-)\},\quad \{(1,+),(0,-)\},
\]

with the other three pairs unchanged. Every node still has one unique
mutual phase partner. Yet both new pairs contain live edges, because
the original base edge `0--1` carried all four fine cross-edges.
They fail the no-within-pair-edge contract immediately.

This control works with exact distinct represented phases such as
`phi_i=1287*i/1024`. It does not claim they equal the symbolic
critical target. A phase-only permutation is not a graph relabeling;
relabeling the entire graph and all attributes would preserve admission.
Replacing the observed pairs by the old graph twin classes would hide
the intended incompatibility and is not an allowed repair.

After successful support admission, a target-dependent family report
also needs an explicit order and orientation of the five observed
pairs, and declared lifts of the captured phases. Nearest pairing
alone fixes neither a particular cyclic starting pair nor a chosen
winding sign. Those declarations must be recorded before applying the
target's norm/storage certificate; they are not inferred by quietly
reordering phases until a certificate succeeds.

The result therefore distinguishes identification, law admission and
persistence. It recognizes an organization already present in the
state, verifies whether the supplied support permits its retained
collective description, and uses a separate invariant-tube certificate
for future identification. Conservation protects the admitted phase
geometry; it is not a new force creating a pair, an edge or a material
constituent.

### 14.5. Observation and admission interfaces

The [scale owner](../src/tnfr/physics/relational_sine_scale.py)
provides `observe_phase_pairs(nodes=..., phases=...)` and
`PhasePairObservation`. This standalone observation reads only
the labeled phase list. Its squared-chord bounds, certified nearest
indices, strict margins and abstention reasons retain the numerical
evidence. The 64-node limit bounds quadratic arithmetic work; it is
neither a physical length scale nor a grouping threshold. Exact ties
recognized from equal absolute raw phase gaps are sufficient tie
controls, not an exhaustive classification of every circular tie.

`assess_sine_state_pairing` and `SineStatePairingAssessment`
then reuse one graph capture. The phase projection supplies the observed
candidate; the existing capacity-aware owner independently checks its
support, law and supplied phase lifts. A caller's `pair_order` must
agree with that observed unordered partition, and is mandatory for
target-dependent family evidence. Without family bounds, capture order
merely arranges the result for display. No automatic unwrapping or
topological substitution is performed.

The wrapper can preserve a valid phase proposal while reporting its
independent collective-law admission as rejected. When its optional
full-state geometry certifies trapping, the separation theorem justifies
`all_time_pairing_persistence_certified`. This flag does not certify
formation, internal circulation or recurrence of the selected state.
Membership in a smaller declared excess/mean family remains separately
reported by the reused owner. The
[state-pairing contract](../docs/contracts/RELATIONAL_DYNAMICS.md#sine-state-pairing)
owns the public admission and reporting details.

The [replica tests](../tests/physics/test_relational_sine_replica.py)
separate phase-only comparisons, exact and unresolved ambiguities,
nonmutual choices, complete relabeling, independently refused support
and protected-family admission. These are detached observations and
conditional reports; neither interface evolves the graph.

<a id="sine-replica-acute-critical"></a>

## 15. Exhaustive fully acute equilibria on the doubled cycle

### 15.1. Complete equations and the sector being classified

Take the fixed complete doubled-`C5` graph with its twenty unit
edges and no other edges. Its structural pairs have identical neighbor
sets; pair `i` is adjacent to both constituents of pairs `i-1`
and `i+1` modulo five. This is a fact about the supplied support,
not a partition inferred by the phase observer.

Keep `e=0`, `a=w/pi>0`, `b=w/(beta*pi)>0`, arbitrary
strictly positive held fine capacities and no input or event. With
the fine combinatorial Laplacian `L_f`, define

\[
K=\operatorname{diag}(\nu_p/4)>0,\qquad
S_p(\theta)=\sum_{q\sim p}\sin(\theta_q-\theta_p).
\]

The complete law is

\[
\dot x=aKS(\theta),\qquad \dot\theta=bKL_f x .
\]

Full equilibrium requires **both** rows to vanish. A vanishing form
rate or instantaneous sine balance alone is not this condition.
Restrict the classification to configurations in which every actual
fine-edge principal phase gap lies in `(-pi/2,pi/2)`. Equivalently,
every edge cosine is strictly positive. This is an explicit sector
restriction; it does not follow merely from the model being conservative.

The exact classification is

\[
\boxed{
x_{i,+}=x_{i,-}=m,\qquad
\theta_{i,+}=\theta_{i,-}
 =c+\frac{2\pi k i}{5}\pmod{2\pi},
\quad k\in\{-1,0,1\},}
\]

where `m` is any real form origin and `c` any common circular
phase origin. Neither constant form, member-phase equality nor the
integer winding is assumed in deriving this list.

### 15.2. Zero phase rates force uniform form

At full equilibrium, `b*K*L_f*x=0`. Both `b` and every
diagonal entry of `K` are positive, so `L_f*x=0`. The
fine graph is connected; equivalently,

\[
x^\mathsf TL_f x=\sum_{\{p,q\}\ {\rm fine\ edge}}(x_p-x_q)^2=0
\]

forces equality along every edge and hence `x=m*1`.
Capacity heterogeneity affects the motion away from equilibrium but
cannot change this kernel argument. Strict positivity is essential;
the proof is not transferred to frozen zero-capacity rows.

### 15.3. Member-phase equality follows without a pair chart

For a structural pair `i` form the common neighbor phasor

\[
Z_i=\sum_{q\in N_i}e^{i\theta_q}.
\]

Both constituents have exactly this neighbor set. For either member
`p`,

\[
e^{-i\theta_p}Z_i
=\sum_{q\in N_i}\cos(\theta_q-\theta_p)
 +i\sum_{q\in N_i}\sin(\theta_q-\theta_p).
\]

The zero form rate and positive capacity imply that its imaginary
part vanishes. All four incident edge cosines are strictly positive,
so its real part is positive. Thus `Z_i!=0` and

\[
e^{i\theta_p}=\frac{Z_i}{|Z_i|}.
\]

Applying this equation to the other member proves equality of their
phases on the circle. No midpoint, phase unwrapping, equal-capacity
assumption or preselected synchronized pair state was used. This
complex-number identity concerns the sine critical equations; it
does not invoke the separate native resultant-pressure runtime.

The common member phase may now be denoted `Theta_i`. Its
coincident-member chart is a consequence of equilibrium and acuteness.

### 15.4. Acute sine balance and circular closure give exactly three windings

Let `gamma_i` be the principal phase increment from pair `i`
to `i+1`. Every `gamma_i` belongs to `(-pi/2,pi/2)`.
Each fine node has two neighbors in each adjacent pair, so its remaining
sine-balance equation is

\[
2\{\sin\gamma_i-\sin\gamma_{i-1}\}=0 .
\]

Sine is strictly increasing on the acute interval. Hence every
`gamma_i` equals the same `gamma`. Circular closure gives

\[
\prod_{i=0}^4e^{i\gamma_i}=1,\qquad
5\gamma=2\pi k\quad(k\in\mathbb Z).
\]

The strict acute bound becomes `4*abs(k)<5`. Its only integer
solutions are `k=-1,0,1`, and reconstructing the consecutive
phases gives the claimed family.

Conversely, choose any of those integers, any `m,c` and any
positive held fine capacities. Uniform form gives zero phase rates.
At every fine node the two forward sine terms and two backward sine
terms cancel exactly, giving zero form rates. All fine edge gaps are
`0` or `+/-2*pi/5` and are strictly acute. Thus every listed
state is a full equilibrium and every fully acute equilibrium is listed.
The converse uses exact angles, not small computed residuals.

These are three signed winding families relative to a declared base
orientation. Reversing that orientation interchanges `+1` and
`-1`. The two twists also have identical edge cosines and storage,

\[
E_{*,k}=20\beta[1-\cos(2\pi k/5)] .
\]

Consensus has zero storage and the two twists have equal positive
storage. These facts do not select an equilibrium's occurrence under
the conservative law. They are not three physical species or spatial
pentagons. The support specifies relational adjacency, while the
classified geometry is a pattern of circular phases on that support.

### 15.5. Exact families, captured phases and observer abstention are distinct

Consensus is a valid full equilibrium although every node has nine
equally close phase neighbors. The phase-only observer therefore
abstains rather than selecting the support's structural pair classes.
There is no contradiction: the classification uses support to prove
which complete equilibria exist; the observer asks what phase alone
identifies. At either exact nonzero twist, the only zero-distance
neighbor is the other constituent, so the phase observation identifies
the pairs. Nearby protected captures can also have unique partners
without being critical states.

There is a useful exact arithmetic boundary for the current capture
format. All captured raw radians are rational numbers: integers,
Fractions and finite binary values have that property. If a captured
adjacent phase difference represented an exact nonzero twist, it would
have to satisfy

\[
\theta_q-\theta_p=2\pi(n+k/5),\qquad
n\in\mathbb Z,\quad k=\pm1 .
\]

The left side is rational. The right side is irrational because
`n+k/5` is a nonzero rational and `pi` is irrational. Thus no
such raw-radian capture is **exactly** a nonzero member of this critical
family. Exact symbolic turns and rounded radians are different inputs.
Similarly, captured rational phases equal on the circle must have
identical raw values: a nonzero rational difference cannot equal
`2*pi*n`.

It follows that, within a certified fully acute captured sector,
exact full equilibrium is equivalent to exact uniform captured form
and exact equality of all captured raw phases. This yields separate
valid source verdicts:

- Nonuniform captured form excludes full equilibrium on this connected
  positive-capacity support, independently of any acute-sector decision.
- Uniform form and identical captured raw phases certify exact consensus.
- Uniform form, certified fully acute edges and nonidentical captured raw
  phases exclude exact critical membership by the preceding classification
  and arithmetic argument.
- Uniform form outside, or not certified inside, the acute sector does
  not receive an equilibrium classification from this theorem.

The third verdict does not measure the size of a response, exclude
proximity to a protected target, or imply numerical or dynamical
instability. A rounded twist can be extremely close to equilibrium
while failing exact membership. No tolerance promotes that approximate
state to the symbolic critical family.

### 15.6. The strict sector boundary and what has not been selected

Acuteness cannot be removed from the classification. As an exact
symbolic control, set uniform form, give one constituent phase `pi`,
and give every other fine node phase zero. Every edge sine is zero,
so both complete rows vanish for any positive capacities. The two
members of the selected structural pair have different phases, and
its incident `pi` gaps have negative cosine. This is an equilibrium
outside the classified sector. Replacing `pi` by a rounded binary
value would not be the same exact counterexample.

The theorem classifies neither nonacute equilibria nor arbitrary
nonstationary organizations. The pulse and protected families studied
earlier are not required to be equilibria at every time. Existing
acute-barrier results can be reused at the listed targets with their
own hypotheses; this enumeration supplies no attraction theorem,
global stability classification or choice among consensus and twists.
It also supplies no route from an arbitrary preparation into the
two-sided invariant protected family.

What has been derived is the complete list of fully acute critical
phase geometries **conditional on this law and supplied support**.
The mechanism selecting a preparation, changing support or producing
a particular organization remains a separate question.

### 15.7. Shared classification and captured-state evidence

The [scale owner](../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_equilibria` and
`SineReplicaEquilibriaAssessment`. It retains one full source
capture and validates the declared structural pair order against the
complete support, with arbitrary positive fine capacities. Classification
does not require a supplied source pair-phase chart or a successful
phase-only grouping.

The integer condition `4*abs(k)<5` generates the candidate
windings. Each `SineReplicaEquilibriumTarget` reuses the
[phase geometry owner](../src/tnfr/physics/phase_cycle_geometry.py)
for exact-turn reconstruction and symbolic odd-sine cancellation.
Only those algebraic properties are consumed; its separate phase-law
interpretation is not transferred to the complete model.

The report keeps this exhaustive symbolic classification separate from
the captured edge-cosine enclosures, exact form/phase equalities and
source equilibrium verdict. A nonacute source does not invalidate the
conditional family theorem, and a small source residual does not prove
target membership. Nonuniform form provides a global exclusion through
the phase row; otherwise an uncertified acute sector remains outside
the captured-state classification unless exact consensus is present.

The [replica tests](../tests/physics/test_relational_sine_replica.py)
check independent fine-row cancellations, the complete integer list,
capacity independence, exact consensus, rounded nonzero twists and
the nonacute boundary. The reader does not choose a winding, repair
a capture, evolve a trajectory or replace the independent phase-only
observer with topological pair recovery.

<a id="sine-pairing-transition"></a>

## 16. Local onset and loss of a support-compatible observed grouping

### 16.1. Differentiate the observation using the complete law

The squared-chord observation of Section 14 is

\[
D_{pq}=2[1-\cos(\theta_q-\theta_p)] .
\]

Along a smooth solution of the complete sine law its exact derivative is

\[
\boxed{
\dot D_{pq}
=2\sin(\theta_q-\theta_p)(\dot\theta_q-\dot\theta_p).}
\]

This formula is circular and does not differentiate an arbitrary wrapping
branch. The phase observer itself still consumes only phase. Its rate,
however, needs the existing complete phase law. For fine degree `d_p`,
write the exact rate numerator

\[
n_p=\frac{w}{\beta}\frac{\nu_p}{d_p}(L_f x)_p,
\qquad \dot\theta_p=\frac{n_p}{\pi}.
\]

Then

\[
\dot D_{pq}
=\frac{2}{\pi}\sin(\theta_q-\theta_p)(n_q-n_p).
\]

The form, capacities, actual neighbors and structural clock enter through
`n`. They cannot be reconstructed from the phase-distance snapshot
alone. These are rates of the existing observation, not a new force,
controller or event law.

For a node `p` comparing partners `q` and `r`, define
the preference margin

\[
M_{p;q,r}=D_{pr}-D_{pq}.
\]

A positive margin favors `q`. If it is zero at the preparation,
a strictly positive derivative proves that this particular tie is
resolved toward `q` for all sufficiently small positive times.
To infer a complete pairing, every tied choice and every other
competitor must also be accounted for.

### 16.2. One frozen represented preparation

Use the fixed complete unit doubled-`C5` support with structural pairs

\[
(0,1),\ (2,3),\ (4,5),\ (6,7),\ (8,9).
\]

Each adjacent base-pair connection contains all four fine edges and
there are no other edges. Keep `e=0`, `beta=1`, all held
capacities one, the structural clock, and no input, clipping or event.
The normalized zero-loss repository reference has `w=1`; retaining
`b=w/pi>0` in the formulas also displays the clock factor.

Freeze `d=u=1/8` and the exact initial values

\[
\theta(0)=(0,d,2d,3d,1,1,2,2,3,3),
\]

\[
x(0)=(u,-u,u,-u,0,0,0,0,0,0).
\]

These dyadic values are represented exactly. Their `d,u` symbols
name the fixed preparation, not scanned parameters. This state is not
an equilibrium or an assumption of fully acute fine edges: for example
the live edge `0--8` has gap `3>pi/2`. The full smooth sine
law applies; the native resultant-pressure law and the acute
equilibrium classification are not substituted for it.

Each structural pair has zero form sum. Every fine node's four
neighbors consist of two complete adjacent pairs, whose total form
is therefore zero. Since the fine degree is four,

\[
L_f x(0)=4x(0),\qquad
\dot\theta(0)=b\,x(0)
=(\omega,-\omega,\omega,-\omega,0,0,0,0,0,0),
\quad \omega=bu>0 .
\]

These are the actual initial phase rows of the full nonlinear system.
Form continues to evolve through `xdot=a*K*S(theta)`; no frozen
form or straight-line phase extrapolation is being treated as a
trajectory.

### 16.3. All initial competitors and the two transverse ties

The full phase span is `3<pi`, so every initial circular distance
is the ordinary absolute difference of these displayed radians. Put
`F(s)=2*(1-cos(s))` for `0<=s<=pi`. The complete initial
nearest sets and closest outsider distances are:

| Node or nodes | Initial nearest set | Nearest circular distance | Smallest distance to an outsider |
| --- | --- | --- | --- |
| `0` | `{1}` | `d` | `2d` |
| `1` | `{0,2}` | `d` | `2d` |
| `2` | `{1,3}` | `d` | `2d` |
| `3` | `{2}` | `d` | `2d` |
| `4,5` | The other member of `(4,5)` | `0` | `1-3d=5d` |
| `6,7` | The other member of `(6,7)` | `0` | `1=8d` |
| `8,9` | The other member of `(8,9)` | `0` | `1=8d` |

Every comparison against an outsider has strictly positive chord
margin at least

\[
g_0=F(2d)-F(d)=2(\cos d-\cos2d)>0 .
\]

For the last three pairs their outsider margins are larger:
`F(5d)>F(2d)>g_0` or `F(8d)>F(5d)`. Thus the table
checks all competitors, not just the adjacent ties. At time zero the
complete phase-pair observer abstains because nodes `1` and `2`
each have two equally near choices.

The exact chord rates for the three tied-length gaps are

\[
\dot D_{01}(0)=\dot D_{23}(0)=-4\omega\sin d,\qquad
\dot D_{12}(0)=4\omega\sin d .
\]

Define the two margins favoring the structural partners,

\[
M_1=D_{12}-D_{10},\qquad M_2=D_{21}-D_{23}.
\]

They obey

\[
M_1(0)=M_2(0)=0,\qquad
\dot M_1(0)=\dot M_2(0)=8bu\sin d>0 .
\]

For the normalized `w=1` reference and frozen `u=d=1/8`,
the common margin derivative is exactly
`sin(1/8)/pi`. Positivity follows from `0<1/8<pi`,
not from a small sampled trajectory or a fitted tolerance.

### 16.4. A complete local transition follows from smoothness

All finitely many distances and preference margins are smooth along
the actual solution. The strict outsider margins therefore remain
positive on some two-sided neighborhood of time zero. For the two
ties,

\[
M_j(t)=8bu\sin(d)\,t+o(t)\qquad(j=1,2).
\]

There exists a common `epsilon_t>0` such that the two margins
are positive for `0<t<epsilon_t` and negative for
`-epsilon_t<t<0`, while all outsider comparisons retain their
initial signs.

For every sufficiently small **positive** time, the unique nearest map
is consequently

\[
0\leftrightarrow1,\qquad
2\leftrightarrow3,\qquad
4\leftrightarrow5,\qquad
6\leftrightarrow7,\qquad
8\leftrightarrow9 .
\]

This is a complete mutual matching. It equals the actual structural
pair partition, so its independent support check succeeds: no matched
pair contains an edge, and adjacent matched blocks have all four
cross-edges. Initial within-pair phase gaps are `d,d,0,0,0<pi`;
continuity also preserves their local pair charts after shrinking the
same unspecified neighborhood if necessary. The retained collective
law thus applies to this observed grouping.

For every sufficiently small **negative** time, the first four choices
instead are

\[
0\longmapsto1,\qquad
1\longmapsto2,\qquad
2\longmapsto1,\qquad
3\longmapsto2 .
\]

The remaining three pairs retain their unique mates. Every node has
a unique nearest neighbor, but the choices of `0` and `3`
are not mutual. No complete mutual-nearest partition exists. A greedy
rematching would change the observation contract rather than complete
this proof.

The conclusion concerns exact mathematical matching at every
sufficiently small time on either side. Fixed arithmetic enclosures
may abstain at times extremely close to the tie, where strict margins
are too small to resolve. The theorem supplies neither a numerical
value of `epsilon_t` nor a certified finite endpoint, forecast or
sampled trajectory. A first derivative alone cannot supply any of those.

There is also a qualitative robustness consequence for **full initial
states**, with the law, support and capacities held fixed. Choose any two
nonempty closed time windows strictly inside the proved negative and
positive intervals. On each window, every node's chosen nearest partner
is separated from every competitor by a strictly positive margin.
The minimum of these finitely many continuous margins on the compact
windows is positive. Joint continuity of the smooth flow in time and
initial state therefore gives an open neighborhood of the supplied full
preparation for which all those inequalities persist on both windows.
Every preparation in this neighborhood has the same nonmutual backward
choices and complete support-compatible forward matching.

This does not preserve the exact simultaneous tie at time zero. Locally
the two initial tie equations reduce to
`theta_0-2*theta_1+theta_2=0` and
`theta_1-2*theta_2+theta_3=0`, two independent conditions.
Perturbations may separate their crossing times; there is no single
common transverse section asserted for the whole neighborhood.
Neither the windows nor the preparation radius have numerical bounds
here. The argument establishes robustness of the strict observations
away from the crossing, not an implemented finite-window certificate.

### 16.5. Form reversal controls the direction with the same phase geometry

Now reverse every form coordinate at the preparation while preserving
the exact phases, capacities and support:

\[
\widetilde x(0)=-x(0),\qquad
\widetilde\theta(0)=\theta(0).
\]

The initial phase observation, its exact ties and all outsider margins
are unchanged. Linearity of the phase row in form gives
`dot(theta_tilde)(0)=-dot(theta)(0)`. Both tied margin derivatives
are therefore `-8bu*sin(d)<0`. Complete mutual pairing holds
locally on the negative side and nonmutual choices locally on the
positive side for this control.

This reversal is also an exact identity of the complete conservative
flow. If `(x(t),theta(t))` solves
`xdot=a*K*S(theta)`, `thetadot=b*K*L_f*x`, then

\[
\widetilde x(t)=-x(-t),\qquad
\widetilde\theta(t)=\theta(-t)
\]

solves the same equations with the reversed initial form. Direct
differentiation proves both rows. Smooth uniqueness identifies it with
the control solution, so its entire local phase-distance history is
the original history reversed in time. This relies on `e=0` and
the absence of time-directed forcing or events; it is not a transfer
to a positive-loss runtime.

The two supplied states thus have identical instantaneous phase
geometry and opposite local directions of grouping change. Form is
necessary predictive information here. Conservation does not choose
one direction, and the emergence of a distance ordering is not
creation of a law that causes it.

### 16.6. What has appeared and what has not

This result proves local onset and local loss of **observability** of
a support-compatible collective grouping under the existing unforced
fine dynamics. The nodes and all twenty edges were already present.
No edge is added, no operator event is selected, and no constituent
capacity is changed.

Nor is this first entrance into the earlier protected family.
That family is invariant in both time directions, so a point outside
it cannot enter it at a finite time. In particular the tie preparation
cannot lie in the protected winding-one tube, whose strict separation
already identifies partners at every time. The present local interval
therefore supplies no acquisition of permanent trapping, long-term
pair identity, attracting waveform or material constituent.

It does establish a concrete same-law mechanism: retained form fixes
phase velocity, phase velocity changes nearest-distance margins, and
those margins can resolve or destroy a complete compatible observation.
A finite observation window or a stronger formation claim would require
its own independently declared prediction and additional proof.
Section 17 supplies the former at an explicitly frozen budget.

### 16.7. Implementation and evidence

`assess_sine_pairing_transition` in the
[shared scale owner](../src/tnfr/physics/relational_sine_scale.py)
retains one complete sine comparison and the existing phase-only
observation. It consumes
`SineExchangeComparison.phase_rate_numerators()` from the
[comparison owner](../src/tnfr/physics/relational_sine_comparison.py);
the same exact rational phase row also serves resultant kinematics.
The report records chord-rate bounds, exact nearest groups, strict
outsider margins and separate backward/forward conclusions.

For an exact tied group, it preserves the common factor
`2*sin(abs(gap))/pi` and compares the exact rational coefficients
`sign(gap)*(n_q-n_p)`. A resolved factor sign and strict coefficient
order can certify a split without subtracting independently rounded
derivative enclosures. Overlapping distances do not establish a tie;
zero or unresolved factors and tied first derivatives remain
unavailable. Complete nearest choices are required before a matching
or nonmutuality conclusion, and a complete matching still receives
its own fixed-support admission.

The [public contract](../docs/contracts/RELATIONAL_DYNAMICS.md#sine-pairing-transition)
keeps `certified_time_horizon` unavailable. The report does not
assign a numerical robustness radius, install the phase observation
as a controller or evolve the graph. The
[independent replica tests](../tests/physics/test_relational_sine_replica.py)
check the frozen complete-field rates, all competing partners, the
form-reversal control and the distinction between a local mathematical
ordering and finite-precision observation.

<a id="sine-pairing-window"></a>

## 17. Whole-box prediction throughout a fixed observation window

### 17.1. The frozen question and complete state

Retain the full conservative normalized-sine law, fixed unit doubled-`C5`
support, held capacities one and `beta=w=1` from Section 16.
No input, clipping, event or native resultant-pressure row is added.
Let `theta*` and `x*` be the exact ten-node preparations there:
`d=u=1/8`,

\[
\theta^*=(0,d,2d,3d,1,1,2,2,3,3),\qquad
x^*=(u,-u,u,-u,0,0,0,0,0,0).
\]

For each `sigma in {+1,-1}`, admit the entire closed preparation box

\[
\mathcal B_\sigma=
\left\{(x_0,\theta_0):
 |x_{0i}-\sigma x_i^*|\leq\rho,\quad
 |\theta_{0i}-\theta_i^*|\leq\rho\quad\hbox{for every }i
\right\},
\qquad \rho=2^{-20}.
\]

The phase errors are specified in the declared real lifts. The flow
and final observation remain circular. All twenty errors are independent;
form sums, pair means, phase equality within the last three pairs and
the two initial nearest ties are not constraints on either box.
The support and capacities are exact. Reversing form exchanges the two
boxes, including their uncertainty sets.

Freeze the same future structural-time window for both predictions,

\[
t_-=1/128,\qquad T=1/64,\qquad t_-\leq t\leq T.
\]

The claim is the original complete mutual matching for `B_+` and the
nonmutual nearest map of Section 16 for `B_-`, for **every** source in
the relevant box and **every** time in this closed window. The budgets
are fixed before their inequalities are evaluated.

### 17.2. A global full-field remainder, without a reduced trajectory

The complete fine rows are

\[
\dot x_i=\frac1{\pi d_i}
 \sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
\dot\theta_i=\frac1{\pi d_i}
 \sum_{j\sim i}(x_i-x_j),
\qquad d_i=4 .
\]

They imply the global bounds

\[
|\dot x_i|\leq\frac1\pi,\qquad
|\ddot\theta_i|
\leq\frac1{\pi d_i}\sum_{j\sim i}
 (|\dot x_i|+|\dot x_j|)
\leq\frac2{\pi^2}.
\]

These bounds use the actual evolving form and all neighbors, and hold
at arbitrary phase configurations. In particular they do not freeze
the phase-to-form feedback or assume fine-edge acuity. The smooth field
has complete solutions: form grows at most linearly on each finite
time interval and the phase row then has a finite integral.

At the center preparation `L_f x*=4*x*`. An arbitrary initial form
error in the box changes the phase rate by at most `2*rho/pi`,
including both the node's own error and its neighbors' errors. Hence
the exact integral remainder gives, for every source and `0<=t<=T`,

\[
\boxed{\left|
\theta_i(t)-\theta_i^*-\frac{\sigma x_i^*}{\pi}t
\right|
\leq E(t):=\rho+\frac{2\rho}{\pi}t+\frac{t^2}{\pi^2}.}
\]

The three terms retain initial phase error, initial form error and
full nonlinear acceleration, respectively. The central affine
expression is a reference used in an enclosure; it is not asserted to
be a nonlinear trajectory. No local Taylor solver, state truncation
or omitted internal coordinate enters this bound.

The same calculation has a reusable conservative form on any admitted
fixed unit support with held nonnegative capacities. Put
`a=w/pi`, `b=w/(beta*pi)`, and let the independent initial form
and phase radii be `r_xi,r_thetai`. For node `i` define

\[
M_i=a\nu_i,\qquad
V_i=b\nu_i\left(r_{xi}+\frac1{d_i}\sum_{j\sim i}r_{xj}\right),
\qquad
B_i=b\nu_i\left(M_i+\frac1{d_i}\sum_{j\sim i}M_j\right).
\]

If `v_i` is the complete phase rate at the center capture, then

\[
|\theta_i(t)-\theta_{0i}^{\rm center}-v_i t|
\leq r_{\theta i}+V_i t+\tfrac12B_i t^2
\quad(t\geq0).
\]

The center rate uses the shared phase-law numerator, not a fitted
response. This extension requires `e=0` and the declared absence of
inputs/events; the form bound is not transferred unchanged to a
positive-loss law.

### 17.3. Exact rational margins for the whole window

Use the rational bounds `25/8<pi<22/7`, with the coarser lower bound
`3<pi` to simplify the remainder. Since `E(t)` increases for nonnegative
time, every phase error on `[0,T]` is at most

\[
\overline E
=\rho+\frac{2\rho T}{3}+\frac{T^2}{9}
=\frac{8483}{301989888}.
\]

Write `omega=u/pi`. The central first-four phase gaps, in node order,
are

\[
d-2\sigma\omega t,\qquad
d+2\sigma\omega t,\qquad
d-2\sigma\omega t.
\]

Each actual gap differs from its central value by at most
`2*Ebar`. Their common strictly positive lower bound is

\[
d-\frac{2uT}{3}-2\overline E
=\frac{18669277}{150994944}>0 .
\]

Thus the order of the first four phases is unchanged for either box
throughout the window. More generally the total real phase span is at
most

\[
3+\frac{2uT}{3}+2\overline E
=\frac{453189923}{150994944}
<\frac{25}{8}<\pi .
\]

All phase separations in this proof are consequently also their
circular distances; no hidden wrapping transition can alter the
nearest ordering.

For the two initially tied choices, the difference between the
undesired and desired distances, with `desired` chosen according to
the sign `sigma`, is bounded below by

\[
\begin{aligned}
m_{\rm tie}
&=4u\,t_-\,\frac7{22}-4\overline E\\
&=\frac{938879}{830472192}
>\frac1{1024}>0.
\end{aligned}
\]

The `4*Ebar` term is the worst error of the three-coordinate
combination at each tie, retaining the repeated central node's
coefficient two. The bound applies even when a source in the box has
no exact initial tie.

For the first four nodes, every competitor outside the singleton or
tied group in Section 16 starts at distance at least `2*d`,
whereas the chosen partner starts at distance `d`. Bounding the
two central motions and the two gap errors gives

\[
m_{\rm outside}
=d-\frac{4uT}{3}-4\overline E
=\frac{9232093}{75497472}
>\frac1{1024}.
\]

For each of the last three pairs, its internal distance can be as
large as `2*Ebar`; those members have not been kept synchronized.
Every outsider starts at distance at least `5*d`. Their
preference margin is at least
`5*d-2*u*T/3-4*Ebar`, which is larger than
`m_outside`. These inequalities account for every competitor of
every node, not just the two central ties.

It follows that every selected nearest distance has uniform preference
margin greater than `g=1/1024`. This also gives a lower bound for the
actual squared-chord observer. If circular distances satisfy
`0<=s<t<=pi` and `t-s>=g`, then

\[
\begin{aligned}
F(t)-F(s)
&=4\sin\!\left(\frac{t+s}{2}\right)
       \sin\!\left(\frac{t-s}{2}\right)\\
&\geq4\sin^2(g/2)=2(1-\cos g)>0,
\qquad F(z)=2(1-\cos z).
\end{aligned}
\]

Indeed the midpoint angle lies between `(t-s)/2` and
`pi-(t-s)/2`, so its sine is at least the sine of the
half-separation. Thus the whole-box prediction has a strict
quantitative chord margin, not only an ordering of unwrapped angles.

### 17.4. The two reserved predictions and their limits

For every initial state in `B_+` and every `t in [1/128,1/64]`,
the exact nearest map is

\[
0\leftrightarrow1,\quad
2\leftrightarrow3,\quad
4\leftrightarrow5,\quad
6\leftrightarrow7,\quad
8\leftrightarrow9.
\]

All choices are strict and mutual. The complete fixed-support
admission independently succeeds for this partition. Its within-pair
phase gaps also remain below `pi`, so the retained collective chart
is available. This does not make its internal differences zero.

For every initial state in `B_-` on the same window, the nearest
indices are

\[
(1,2,1,2,5,4,7,6,9,8).
\]

All choices are strict, but `0->1` and `3->2` are not mutual.
There is no complete mutual-nearest matching under the declared
observation contract, even though the support itself is unchanged.
Both conclusions cover independent perturbations of every fine
coordinate; neither follows from a center trajectory or a single
evaluated endpoint.
Some sources in these boxes may already have their window's nearest
map at time zero. The whole-box conclusion is therefore a uniform
future observation prediction, not a claim that every source undergoes
one simultaneous onset at a specified time.

The original local reader still supplies only an existential
neighborhood. The present theorem instead proves its separately
frozen finite-window prediction using explicit global remainders.
It gives no conclusion outside `[1/128,1/64]`, no observation-noise
budget beyond the stated preparation uncertainty, and no eventual
capture by the earlier two-sided invariant family. A numerical
observer still has to resolve a margin with its own arithmetic.
The result is finite organizational robustness under a supplied
complete law, not new support, a controller, permanent formation or
physical identification.

### 17.5. Shared consumer and numerical evidence

`assess_sine_pairing_window` in the existing
[scale owner](../src/tnfr/physics/relational_sine_scale.py) consumes a
single comparison capture, mandatory full vectors of form and phase
error radii, and an explicit `0<=window_start<window_end`.
It requires zero form loss and retains the captured support, capacities
and structural clock. The source box is supplied analysis evidence;
the reader does not authenticate a laboratory preparation.

`SinePairingWindowAssessment` records the initial intervals, the
per-node form-speed, initial phase-rate-error and acceleration upper
bounds, the resulting remainders and the complete phase window.
Pair differences reuse the exact shared rate-numerator difference
before applying the common time interval. This preserves information
that subtraction of independent whole-node phase windows would lose.
Where the absolute gap enclosure lies below mathematical `pi`,
monotonicity gives tight chord bounds from certified scalar cosine
endpoints. Elsewhere the shared interval cosine is used without
inferring a new phase lift.

Only strict comparison with every competitor certifies a node's
nearest choice throughout the box and window. The report then
distinguishes complete mutual matching, certified nonmutuality and
unavailable bounds, followed by separate paired-support admission
where a complete matching exists. Wide intervals do not refute a
physical trajectory or the proposed grouping. The existing local
reader keeps its absent numerical horizon.

The [window contract](../docs/contracts/RELATIONAL_DYNAMICS.md#sine-pairing-window)
and [independent replica tests](../tests/physics/test_relational_sine_replica.py)
retain these distinctions, the frozen two-box control, all competing
nodes, invalid domains and unresolved-bound controls. This is a
full-field analytic enclosure on the existing owner, with no
trajectory producer, finite-difference fit or enlarged solver budget.

<a id="sine-pair-support-symmetry"></a>

## 18. Which attachments permit an unordered pair-state law?

### 18.1. Equivalence concerns the entire member state on a fixed graph

Keep the complete conservative normalized-sine law on a fixed finite
simple connected unit graph, with a common held capacity `nu>0`,
`w,beta>0` and no input or event. Let an explicit partition put
every fine node in a two-member pair. Define `G` as the finite
group generated by independently exchanging the two members of each
pair. Each permutation acts on **both** form and circular phase.
The support, capacities and clock stay fixed.

On the pair chart of Section 8, the retained coordinates
`(X,Theta,R,U,Q)` identify exactly these swap orbits. They retain
the internal member information up to interchange; they are not merely
pair means or a reduction in continuous dimension. The all-state
condition for removing member labels from the evolution is

\[
\boxed{F(Pz)=P F(z)\quad\hbox{for every }z
\hbox{ and every independent pair swap }P.}
\]

Equivariance suffices: smooth uniqueness gives
`Phi_t(Pz)=P Phi_t(z)`, so the orbit of a state has a
well-defined future orbit. It is also necessary for the exact
all-state orbit quotient. On the dense set where no pair consists
of identical member states, the finite action is free. Locally the
quotient is invertible up to its finitely many separate lifts.
If two such lifts have the same quotient flow, continuity forces
their initial relative permutation to remain the same for sufficiently
small time. Differentiating at zero gives the displayed identity.
Continuity of the field then extends it to the fixed strata.
The singular invariant chart at a tip does not remove this obligation.

This is different from simultaneous graph/state relabeling.
Relabeling `A` to `P A P^T` while also relabeling `z`
always describes the same unlabeled fine system. Here `A` is
held unchanged; the question is whether exchanging member states
alone can lose predictive information. Nor does failure of the
all-state criterion exclude a separately proved special invariant
subfamily or a smaller joint-swap symmetry.

### 18.2. Necessary and sufficient fixed-support condition

Write `L_rw=I-D^{-1}A` for the normalized Laplacian of this
unit graph. The complete field is

\[
\dot x=a\nu D^{-1}S(\theta),\qquad
\dot\theta=b\nu L_{\rm rw}x,\qquad
a=w/\pi,\quad b=w/(\beta\pi).
\]

Since `b*nu>0`, equivariance of the phase row for every form
already requires

\[
L_{\rm rw}P=P L_{\rm rw}.
\]

For distinct nodes `p,q`, its matrix entry is `-1/d_p`
exactly when an edge is present, and zero otherwise. Permutation
commutation therefore preserves this nonzero pattern: `P` is
an automorphism of the fixed adjacency. Conversely, a graph
automorphism preserves degrees and changes neither the linear
neighbor sum nor the nonlinear sum
`sum_j sin(theta_j-theta_i)` after the same state permutation.
It consequently commutes with **both** complete rows.

Thus, under the stated capacity and support hypotheses,

\[
\boxed{\begin{split}
\text{independent pair-state quotient for all states}
&\ \Longleftrightarrow\
\text{every independent pair swap is a graph automorphism}\\
&\ \Longleftrightarrow\
\text{each interpair block is complete or empty.}
\end{split}}
\]

The last line allows an edge within any pair. More explicitly,
members `p,q` are interchangeable precisely when
`N(p) minus {q} = N(q) minus {p}`; adjacency between `p`
and `q` is itself unchanged by their transposition. For two
different pairs, independent exchange of their row and column
members forces all four entries of their `2-by-2` adjacency
block to agree. Each block is therefore either all four cross-edges
or no edge. The presence or absence of each within-pair edge is
unrestricted by the swap symmetry.

Positivity and the fixed unit graph are substantive premises here.
A frozen zero-capacity row need not reveal its attachments through
the phase field. Unequal capacities require the joint
capacity-state treatment in Section 13, and a weighted graph
requires the corresponding weighted equivariance test. Neither
case is covered by simply dropping a hypothesis.

Equal neighbor counts alone are insufficient. For example, pairs
`(0,1)` and `(2,3)` with internal edges `0--1,2--3`
and cross-edges `0--2,1--3` form an equitable partition:
each member has one neighbor in each relevant block. Exchanging
only `0,1` changes the cross-edge pattern. The full unordered
member-state quotient fails even though a linear mean reduction
can have matching neighbor counts. A purely diffusive or linear
quotient certificate cannot replace the complete-law condition.

### 18.3. Internal edges preserve symmetry but change the inherited rows

The preceding criterion is broader than the existing replica
runtime's no-internal-edge admission. To see the distinction
directly, let `epsilon_i in {0,1}` mark an internal pair edge,
let `d_i` count adjacent base pairs, and put
`D_i=2*d_i+epsilon_i`. Connected support gives `D_i>0`.
Use the same signed coordinates as Section 7 and define

\[
R_i=\cos\delta_i,\qquad
J_i=\sum_{j\sim i}R_j\sin(\Theta_j-\Theta_i),\qquad
C_i=\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),\qquad
q_i=\sum_{j\sim i}(X_i-X_j).
\]

Exact averaging and differencing of the fine rows give

\[
\boxed{\begin{aligned}
\dot X_i&=\frac{2a\nu}{D_i}R_iJ_i,&
\dot\Theta_i&=\frac{2b\nu}{D_i}q_i,\\
\dot u_i&=-\frac{2a\nu}{D_i}
 \sin\delta_i\,(C_i+\epsilon_iR_i),&
\dot\delta_i&=\frac{2b\nu}{D_i}(d_i+\epsilon_i)u_i.
\end{aligned}}
\]

Indeed the optional internal edge contributes
`-epsilon_i*sin(2*delta_i)` to the plus-member sine sum
and `2*epsilon_i*u_i` to its form-gradient sum, with the
opposite contributions for the minus member. The degree used to
normalize every fine row changes at the same time.

For `epsilon_i=0` these expressions recover Section 7.
For `epsilon_i=1` the same finite swap quotient still exists,
but its internal restoring term and normalization differ. An
internal edge also adds
`2*u_i^2+2*beta*sin(delta_i)^2` to the fine storage.
It is therefore incorrect either to reject its mathematical
swap symmetry or to apply the old no-internal-edge formulas
unchanged. The current strict replica consumer keeps its original
admission; the larger symmetry test does not install these broader
rows as a runtime law.

### 18.4. Frozen one-edge control: identical retained state, different future

Return to the ten-node preparation of Section 16, with
`d=u=1/8`, `beta=w=nu=1` and structural pairs
`(0,1),(2,3),(4,5),(6,7),(8,9)`. Remove exactly edge
`(0,8)` in the **detached preparation**. Compare two states
on this same nineteen-edge graph: the original state and the
one obtained by exchanging both `x` and `theta` of
nodes `0` and `1`. No support-removal event or change in
its energy balance is being executed.

The full degree of node `8` is now three, with neighbors
`{1,6,7}`, whereas node `9` has degree four and neighbors
`{0,1,6,7}`. The original forms imply

\[
(L_fx)_8=u,\qquad (L_fx)_9=0,\qquad
\dot\theta_8=\frac{u}{3\pi},\qquad
\dot\theta_9=0.
\]

The retained phase mean of pair `(8,9)` consequently has rate

\[
\dot\Theta_{(8,9)}=\frac{u}{6\pi}
=\frac1{48\pi}.
\]

After the state-only swap, `x_1=u` instead of `-u`;
the two neighbor sums at node `9` still cancel. Therefore

\[
(L_f\widetilde x)_8=-u,\qquad
(L_f\widetilde x)_9=0,\qquad
\dot{\widetilde\Theta}_{(8,9)}
=-\frac{u}{6\pi}=-\frac1{48\pi}.
\]

The derivative difference is exactly `-1/(24*pi)`.
These are the complete-law phase rows with the actual changed
degrees, not a perturbative estimate or an inherited base-row formula.

Nevertheless, every retained unordered pair coordinate is initially
identical in the two states. Only pair `(0,1)` has changed
its signed lift: `u_0` and `delta_0` both negate.
Its means, `R=cos(delta)`, `U=u^2` and
`Q=u*sin(delta)` are unchanged; all other pairs are untouched.
The phase-mean-rate difference in an untouched pair is therefore
an exact obstruction to any autonomous law on these retained
unordered coordinates alone for this support.

The lost information is concrete: which member state of pair
`(0,1)` is attached to node `8`. Keeping the graph fixed
does not recover that state-to-attachment correlation after the
member labels have been discarded. Equal capacities do not
remove it.

### 18.5. What a larger collective description must retain

An observed pair can remain a useful organization even when its
members are not independently interchangeable under the law.
The phase-only observer, the support symmetry and the future
closure are separate claims. A failed symmetry test neither
erases an observed grouping nor proves that it immediately disappears.

The same nineteen-edge source also passes the existing analytic pairing-window
reader with the unchanged independent error radii `2^-20` and window
`[1/128,1/64]`. Its center phase-rate numerators are
`(1/8,-1/8,1/8,-1/8,0,0,0,0,1/24,0)`; the general full-field remainder from
Section 17 retains this changed rate and does not assume pair `(8,9)` stays
synchronized. Every original pair remains the strict mutual nearest choice
throughout the admitted box/window, while strict replica support is rejected.
This supplies a finite-time instance of observed organization without the
proposed fully unordered dynamical closure. It is a reuse of the same bound,
not a support event, retuned budget or trajectory experiment.

On asymmetric support, retain the constituent states together
with their incidence information, or quotient only by actual
symmetries of that joint data. Correlations between member state
and external attachments cannot in general be reconstructed
from pair means or `R,U,Q`. A more compressed sufficient
description needs its own proof; the counterexample does not
select a universal new macro variable or force law.

This result establishes an information boundary under a supplied
network. It neither selects that network, generates a microscopic
connection nor establishes autonomous birth of a collective
constituent. Exact finite-symmetry reduction retains internal
dynamics and removes only labels whose interchange truly
preserves the complete field.

### 18.6. Shared support reader and state-only evidence

`assess_sine_pair_support_symmetry` in the existing
[scale owner](../src/tnfr/physics/relational_sine_scale.py) consumes
one complete sine comparison, an explicit pair partition and,
optionally, a caller-selected `witness_pair`. Its domain keeps
zero form loss and common positive held capacity. It checks
external neighbor equality directly, retains internal pair edges
and every interpair edge count, and records a nonzero exact
phase-row matrix witness for each failed swap. No sampled state
or tolerance decides the all-state criterion.

The optional `SinePairSwapWitness` exchanges form and phase
on the selected pair while keeping the captured graph and
capacities unchanged. It reuses the shared form-gradient,
phase-rate-numerator and sine-rate owners, retains both full
fields, and compares pair phase-mean rates. Exact equality of
the raw member-state multisets establishes the same swap orbit
without testing equality of trigonometric enclosures. Phase
means here use the supplied continuous real lifts locally;
the report does not infer a global circular mean chart.

`SinePairSupportSymmetryAssessment` keeps this concrete
countercontrol separate from its support-wide verdict and
from the existing strict replica admission. A symmetric graph
with an internal pair edge can pass the first test while
remaining outside the no-internal-edge consumer. Conversely,
a chosen state with vanishing witness rates does not restore
all-state closure on asymmetric support.

The [support-symmetry contract](../docs/contracts/RELATIONAL_DYNAMICS.md#sine-pair-support-symmetry)
and [independent replica tests](../tests/physics/test_relational_sine_replica.py)
cover the complete/empty criterion, internal edges, equitable
but noninterchangeable members, exact degree-sensitive
countercontrol and graph/state relabeling distinction. The
reader neither mutates support nor installs a broader
autonomous collective runtime.

<a id="sine-mixed-pair-state"></a>

## 19. Sufficient collective state on asymmetric support

### 19.1. Remove only the independently redundant member labels

Keep the fixed simple connected unit support, common held capacity
`nu>0`, conservative normalized-sine law and explicit pair partition
of Section 18. Write `S` for the pair indices whose individual
member transposition is a graph automorphism, and `O` for the
remaining pair indices. The graph determines this distinction through
the exact external-neighbor criterion. It does not depend on an
instantaneous synchronization score or an observed small rate.

For every pair retain the means `X_i,Theta_i` and the same declared
nonantipodal chart

\[
x_{i,\pm}=X_i\pm u_i,\qquad
\theta_{i,\pm}=\Theta_i\pm\delta_i,
\qquad |\delta_i|<\pi/2.
\]

`Theta_i` denotes the circular midpoint with a supplied local continuous
lift; adding the same full turn to its members changes that lift, not
the circular state. The retained internal coordinates are

\[
\boxed{
\begin{cases}
(u_i,\delta_i), & i\in O,\\
(R_i,U_i,Q_i)=(\cos\delta_i,u_i^2,u_i\sin\delta_i), & i\in S.
\end{cases}}
\]

For an ordered pair, the plus/minus entries remain associated with
their named incident edges. The sign is this supplied member ordering,
not a physical orientation selected by the model. For an unordered
pair, `0<R_i<=1`, `U_i>=0` and
`Q_i^2=U_i*(1-R_i^2)` are essential realizability conditions.
The full support, pair membership, capacity, coefficients and clock
remain part of the specification.

Let `H` be the finite group generated by the individual swaps in `S`.
Two fine circular states in this chart have the same mixed description
if and only if they lie in the same `H` orbit. Indeed each ordered
pair reconstructs its assigned member states directly. For an
unordered pair with `0<R<1`, choose

\[
\delta=\arccos R>0,\qquad
u=Q/\sqrt{1-R^2}.
\]

The other lift negates both coordinates and is exactly its permitted
swap. At `R=1`, `delta=0`, `Q=0` and `u=+sqrt(U)` or
`-sqrt(U)` give the same permitted orbit. At the synchronized tip
`(R,U,Q)=(1,0,0)` there is one fine lift. These pairwise choices
prove both reconstruction and exact separation of the product orbits.
The sign of `Q` is necessary: erasing it would in general identify
states outside a permitted swap orbit.

Every generator of `H` preserves the full field by Section 18, so
these orbits have a well-defined future while the chart is valid.
This is sufficient state for the declared law, not a reduction in the
generic number of continuous degrees of freedom. Each pair still
has its two internal degrees of freedom. Nor is `H` asserted to be
the entire graph automorphism group: a simultaneous exchange of
several otherwise asymmetric pairs may be another symmetry, but
finding or quotienting such joint permutations is outside this result.

### 19.2. Inherited rates use the actual fine attachments

For any representative reconstructed above, evaluate the original rows

\[
f_v=\frac{a\nu}{D_v}\sum_{k\sim v}\sin(\theta_k-\theta_v),
\qquad
h_v=\frac{b\nu}{D_v}\sum_{k\sim v}(x_v-x_k),
\qquad a=w/\pi,\quad b=w/(\beta\pi),
\]

where `D_v` is its actual fine degree. For every pair the means obey
`X_dot=(f_plus+f_minus)/2` and
`Theta_dot=(h_plus+h_minus)/2`. For each ordered pair retain also

\[
\dot u_i=(f_{i,+}-f_{i,-})/2,\qquad
\dot\delta_i=(h_{i,+}-h_{i,-})/2.
\]

For an unordered pair differentiation gives

\[
\dot R_i=-\sin\delta_i\,\dot\delta_i,\quad
\dot U_i=2u_i\dot u_i,\quad
\dot Q_i=\sin\delta_i\,\dot u_i+
          u_i\cos\delta_i\,\dot\delta_i.
\]

These expressions are independent of the choice of permitted fine
lift. Equivariance exchanges the two rates when the member states
are exchanged; their means and the three differentiated invariants
are unchanged. Rates of untouched ordered members also stay the
same. The complete-bipartite replica formulas are not being applied
to an incomplete cross-block.

The closure can also be made explicit without an inverse square root.
For `i in S`, let `T_i` be the common external neighbor set of
its members, `m_i=|T_i|`, and let `epsilon_i` indicate the optional
internal edge. Their common fine degree is `D_i=m_i+epsilon_i>0`.
Define from retained state

\[
J_i=\sum_{k\in T_i}\sin(\theta_k-\Theta_i),\qquad
C_i=\sum_{k\in T_i}\cos(\theta_k-\Theta_i),\qquad
q_i=\sum_{k\in T_i}(X_i-x_k),
\]

\[
A_i=\frac{a\nu}{D_i}(C_i+2\epsilon_iR_i),\qquad
c_i=\frac{b\nu}{D_i}(m_i+2\epsilon_i).
\]

There is no hidden sign choice in these sums. If a member of a
different unordered pair belongs to `T_i`, its other member belongs
as well, since that pair's own swap is a graph automorphism. Its
combined contributions to `J_i,C_i,q_i` are therefore

\[
2R_j\sin(\Theta_j-\Theta_i),\qquad
2R_j\cos(\Theta_j-\Theta_i),\qquad 2(X_i-X_j).
\]

Any ordered neighbor contributes its retained assigned form and
phase individually. The same both-or-neither property lets an
ordered receiver evaluate its neighboring unordered pairs using
`2R_j*sin(Theta_j-theta_v)` and `2*(x_v-X_j)`.
Thus all required sums are functions of the mixed state and support.

Exact averaging of the external sine terms gives `R_i*J_i`;
their half-difference is `-sin(delta_i)*C_i`. An internal edge
adds `-sin(2*delta_i)` to the plus-member sine row and
`2*u_i` to its form-gradient row. Consequently the unordered
rows are exactly

\[
\boxed{\begin{aligned}
\dot X_i&=\frac{a\nu}{D_i}R_iJ_i,&
\dot\Theta_i&=\frac{b\nu}{D_i}q_i,\\
\dot R_i&=-c_iQ_i,&
\dot U_i&=-2A_iQ_i,\\
\dot Q_i&=-A_i(1-R_i^2)+c_iU_iR_i.
\end{aligned}}
\]

This includes symmetric pairs attached to only one member of an
ordered neighboring pair, as well as internal edges. In the special
complete/empty interpair support of Section 18.3, substituting
`m_i=2*d_i` recovers its formulas. The more general rows follow
from the same microscopic field, rather than an added interaction.

### 19.3. Realizability, synchronized tips and the chart boundary

The explicit mixed rows are smooth functions of the ambient
retained variables in a local midpoint chart. They contain no
division by `U`, `Q` or `1-R^2`. Their derivative of the
unordered constraint `Q^2-U*(1-R^2)` vanishes identically,
with the same cancellation as Section 8.2. More strongly, lift
any realizable initial state, evolve the original smooth fine law,
and project it. This produces a realizable solution of the mixed
rows. Local uniqueness of their smooth ambient field implies that
every solution from that mixed initial state is this projection,
including at a singular invariant tip. Thus the inequalities and
the full realizable image are preserved for as long as the phase
chart remains valid; the algebraic constraint alone would not have
proved this.

For an unordered pair at `R=1,U=Q=0`, all internal derivatives
vanish and the tip remains invariant. This follows from its actual
swap symmetry, even when neighboring ordered members differ.
At `R=1,U>0`, the nonzero `Q_dot=c_i*U` retains the form-driven
departure from phase alignment. By contrast, an ordered pair at
`u=delta=0` need not remain synchronized: unequal attachments
can give unequal fine rows immediately. Equality of a pair's
instantaneous member states is not a substitute for graph symmetry.

At `|delta_i|=pi/2` the circular midpoint becomes ambiguous;
for an unordered pair its resultant magnitude is zero. The
chosen mixed chart then ends even though the original sine field
remains smooth. No new global chart or trapping theorem on arbitrary
asymmetric support is supplied here. In numerical admission, a
chart margin not certified strictly positive requires refusal,
not clipping, sign selection or a tolerance-based assertion of
nonantipodality.

### 19.4. The retained receiver control now discriminates the two states

On the fixed nineteen-edge preparation in Section 18.4, the
unordered indices are `S={1,2,3}`, while `O={0,4}` retains
the attachment-bound member states of `(0,1)` and `(8,9)`.
For the original source, pair `(0,1)` has

\[
(X_0,\Theta_0,u_0,\delta_0)
 =(0,1/16,1/8,-1/16).
\]

Its state-only swap on the same graph instead has `u_0=-1/8`
and `delta_0=1/16`. The mixed initial states are now distinct.
The original receiver `(8,9)` has `u_4=delta_4=0` but

\[
\dot\Theta_4=\dot\delta_4=\frac{1}{48\pi};
\]

both signs reverse for the swapped source. Its equal initial
member states therefore do not make its ordered internal
coordinate redundant. These derivatives reproduce the already
known fine-row control; they are not new reserved observations.
The new result is the sufficient state and rate construction
that retains the cause of their distinction without changing
the underlying law.

Independently swapping any subset of pairs `1,2,3` changes
neither the mixed circular state nor its rates. A simultaneous
relabeling of the graph, fine state and ordered pair identities
only relabels the description. Reversing an asymmetric pair's
declared ordering while tracking its incidence reverses `u,delta`
and their rates; exchanging its states on a fixed incidence
assignment is instead the different fine model state
examined by the control. No absolute orientation or extra force
is inferred from this bookkeeping.

### 19.5. Shared observer and scope

`assess_sine_mixed_pair_state` in the
[scale owner](../src/tnfr/physics/relational_sine_scale.py)
combines the existing exact support-symmetry assessment with
the shared pair-lift chart and complete fine sine rates. It
records each pair's ordered or unordered mode, retained coordinates,
reconstruction stratum and pushforward rates. The captured fine
state remains available as provenance; an unordered pair's signed
representative is not an additional retained mixed coordinate.

The theorem concerns exact realized states. Trigonometric interval
enclosures in a numerical report are neither exact invariant values
nor a joint uncertainty inverse for arbitrary proposed collective
data. The reader does not reconstruct a new fine state from
independently supplied `R,U,Q` boxes, silently promote overlap
to equality, or integrate an autonomous macro-node runtime.
The existing strict replica admission remains separate, and
none of its persistence, pulse or recurrence certificates are
broadened merely by accepting this mixed description.

The [mixed-state contract](../docs/contracts/RELATIONAL_DYNAMICS.md#sine-mixed-pair-state)
and [independent replica tests](../tests/physics/test_relational_sine_replica.py)
cover the full-node rates, the fixed receiver distinction, allowed
swaps, simultaneous relabeling, chart refusal and synchronized tips.
The unchanged finite observation window from Section 18.5 does
not become an all-time grouping theorem. The construction removes
a specific information loss under supplied support; it neither
creates connections nor proves spontaneous permanent constituents
or identifies physical matter.

<a id="sine-pairing-constitutive-scope"></a>

## 20. Constitutive scope of the local grouping mechanism

### 20.1. The common work identity leaves a mobility law to specify

Reuse the [reciprocal mobility class](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#common-exchange-geometry-and-composition)
on fixed simple connected unit support. Let `L_f` be its combinatorial
Laplacian, `q=L_f*x`, and
`S_i=sum_{j~i} sin(theta_j-theta_i)`. In the conservative sector the
complete unforced rows are

\[
\boxed{\dot x_i=w\nu_i m_i(\theta)S_i,\qquad
\dot\theta_i=(w/\beta)\nu_i m_i(\theta)q_i.}
\]

Here `w,beta>0`, capacities are held and strictly positive, and the
structural clock is fixed. For the present local result the supplied
mobility functions are finite, strictly positive, phase-only and locally
Lipschitz on a circular neighborhood of the preparation. These are
sufficient hypotheses for a unique differentiable local solution, not
conditions inferred from an observed response. A model must specify
its mobility functions and their domain; positivity is not a complete
law. The frozen comparison below uses smooth functions of incident
phase differences, preserving global phase-origin and graph-relabeling
symmetries.

The nodal pressure is the independently supplied map
`p_i=w*m_i*S_i`. The common storage is

\[
E=\frac12\sum_{\{i,j\}}(x_i-x_j)^2+
  \beta\sum_{\{i,j\}}[1-\cos(\theta_j-\theta_i)].
\]

Its gradients are `grad_x E=q` and `grad_theta E=-beta*S`.
Consequently each node's two exchange work terms cancel:

\[
q_i\dot x_i-\beta S_i\dot\theta_i
 =w\nu_i m_i q_iS_i-w\nu_i m_i S_iq_i=0.
\]

No derivative of the mobility is needed for this identity. It does
not select `m_i=1/(pi*d_i)`, establish an invariant volume, or
give all candidates the same trajectory or weighted means. The
[existing conditional classification](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#global-closure-pressure-comparison)
selects normalized sine pressure only after adding its stated
pairwise-superposition or phase-independent-response premise and
normalization. That selection theorem is not being assumed to hold
for every positive reciprocal mobility.

### 20.2. Positive reciprocal response preserves the local direction of grouping

Retain exactly Section 16's twenty-edge doubled-C5 preparation:
`d=u=1/8`, phases `(0,d,2d,3d,1,1,2,2,3,3)`, forms
`(u,-u,u,-u,0,0,0,0,0,0)`, common capacity `nu=1`, and
`w=beta=1`. Keeping `k=w*nu/beta>0` visible below makes the
common rate factor explicit. Every fine degree is four and the
prepared neighbor form sums vanish, so `q=4*x`. Write `m_i`
for the chosen mobility evaluated at this initial phase state.
The complete phase row gives

\[
(\dot\theta_0,\dot\theta_1,\dot\theta_2,\dot\theta_3)
 =4ku\,(m_0,-m_1,m_2,-m_3),\qquad
\dot\theta_i=0\quad(i=4,\ldots,9).
\]

Keep the same circular squared-chord observation
`D_ij=2-2*cos(theta_j-theta_i)` and the two initially tied
preference margins

\[
M_1=D_{12}-D_{10},\qquad M_2=D_{21}-D_{23}.
\]

Differentiating the chords using the actual phase row, rather than
assuming a phase clock, gives

\[
\boxed{\begin{aligned}
\dot M_1(0)&=8ku\sin d\,(m_0+2m_1+m_2)>0,\\
\dot M_2(0)&=8ku\sin d\,(m_1+2m_2+m_3)>0.
\end{aligned}}
\]

For example, `M1_dot=2*sin(d)*(theta_dot2-2*theta_dot1+theta_dot0)`;
the second expression is
`2*sin(d)*(2*theta_dot2-theta_dot1-theta_dot3)`.
The signs follow solely from this preparation, positive capacity
and mobility, and `0<d<pi`. Equal mobilities are unnecessary.

Section 16.3 already checks every other potential partner at these
same phases. Its strictly positive outsider margins remain positive
on some sufficiently small neighborhood of time zero for each
admitted complete field. The two nonzero margin derivatives resolve
the ties. Thus the original preparation has the same complete mutual
matching `(0,1),(2,3),(4,5),(6,7),(8,9)` for all sufficiently
small positive times, and the same nonmutual choices of Section 16.4
for all sufficiently small negative times. The observed organization
does not require the particular constant sine mobility.

Complete form reversal retains the mobility values, negates `q`
and every initial phase rate, and reverses both margin derivatives.
It therefore reverses the local direction of the observation change.
More strongly, because the complete conservative form row is
phase-only and its phase row is linear in form,
`(-x(-t),theta(-t))` is the solution with reversed initial form.
This exact local time-reversal identity uses the same phase-only
mobility law in both rows. Merely assuming positive state-dependent
mobility without its form-reversal symmetry would not justify that
identity, although the displayed initial sign argument still has
its own pointwise version.

These are local conclusions for each specified law. An arbitrarily
large positive common multiplier already changes its clock speed,
and the class has no shared numerical Lipschitz or acceleration
budget. No uniform positive time interval follows from the strict
initial signs alone. In particular, Section 17's uncertainty box
and window `[1/128,1/64]` are not transferred to another mobility.
Support-compatible observed pairs likewise do not license use of
the sine-specific collective rate, pulse or recurrence formulas.

### 20.3. A frozen relative-rate discriminator

Use only the two previously admitted comparison members

\[
m_i^{(\epsilon)}(\theta)
 =\frac{1+\epsilon(S_i/d_i)^2}{\pi d_i},
\qquad \epsilon\in\{0,1\}.
\]

Both complete rows change together when the mobility changes.
The same source, support, held capacities, coefficients and structural
clock are retained. No parameter is fitted. Before evaluating the
alternative, fix the dimensionless initial-rate observation

\[
\boxed{\mathscr D=
 \frac{\dot M_1(0)-\dot M_2(0)}
      {\dot M_1(0)+\dot M_2(0)}.}
\]

The positive source has a strictly positive denominator; the
form-reversed control has a strictly negative one. Both are valid
denominators. A generic numerical consumer must still report the
ratio unavailable whenever its denominator enclosure includes zero.
The raw margin derivatives remain separate evidence.

A common positive clock rescaling multiplies both derivatives by
the same factor and leaves `mathscr D` unchanged. Adding a common
phase-origin velocity also changes neither derivative, because their
phase-rate coefficient sums are zero. Therefore distinct values
exclude an identification by common clock rescaling and common
phase drift at this source. They do not exclude arbitrary state
redefinitions or establish an independent physical measurement bridge.

Put `s_i=S_i/4` at the frozen preparation. The preceding exact
expressions give

\[
\mathscr D_\epsilon=
\frac{\epsilon(s_0^2+s_1^2-s_2^2-s_3^2)}
 {8+\epsilon(s_0^2+3s_1^2+3s_2^2+s_3^2)}.
\]

Hence `mathscr D_0=0` exactly, without comparing rounded intervals.
To evaluate the other member independently, the required fine currents
are explicitly

\[
\begin{aligned}
S_0&=\sin(2d)+\sin(3d)+2\sin3,\\
S_1&=\sin d+\sin(2d)+2\sin(3-d),\\
S_2&=-\sin(2d)-\sin d+2\sin(1-2d),\\
S_3&=-\sin(3d)-\sin(2d)+2\sin(1-3d).
\end{aligned}
\]

The shared outward rational trigonometric arithmetic at `d=1/8`
certifies the strict bound

\[
\boxed{\frac1{500}<\mathscr D_1<\frac1{400},
\qquad \mathscr D_0=0.}
\]

Rounded displays of the enclosed values are:

| Fixed mobility | `M1_dot(0)` | `M2_dot(0)` | `mathscr D` |
| --- | --- | --- | --- |
| `epsilon=0` | `0.0396852001938463` | `0.0396852001938463` | Exactly `0` |
| `epsilon=1` | `0.0417943681701375` | `0.0415967934157799` | Approximately `0.00236925293520505` |

The rational strict interval, rather than those rounded decimal
displays, establishes separation. Form reversal negates both raw
margin derivatives and leaves their ratio unchanged. Its exact
control follows algebraically; no new trajectory or observation
window was evaluated.

The result separates two questions. Both laws create the same local
nearest-partner ordering from the chosen preparation, so observing
only that qualitative ordering cannot select between them. Their
relative rates nevertheless differ on the same complete state.
Reciprocal work cancellation and the qualitative mechanism are
broader than the selected quantitative constitutive response.

### 20.4. Shared comparison and evidence boundary

`SineExchangeComparison.with_current_squared_mobility(epsilon=...)` in
the [comparison owner](../src/tnfr/physics/relational_sine_comparison.py)
retains the original capture and declares the alternative mobility,
both complete rows and their storage-work evidence. The
`assess_sine_pairing_mobility` reader in the
[scale owner](../src/tnfr/physics/relational_sine_scale.py)
evaluates two explicitly supplied chord-margin triples and their
rate contrast, sum and available ratio. Its triple order is
`(observer, alternative, preferred)`, so `(1,2,0)` and
`(2,1,3)` represent exactly `M1` and `M2` above.

The reader uses the shared fine capture, phase-current arithmetic and
actual phase rows. Exact cancellation is retained where the law and
gap identities provide it; overlapping intervals are not promoted to
equality. The generic report evaluates specified instantaneous margins,
not a theorem about every possible preparation. The positive-class
conclusion is the conditional argument of Section 20.2.

The [constitutive comparison contract](../docs/contracts/RELATIONAL_DYNAMICS.md#sine-pairing-mobility)
and [independent replica tests](../tests/physics/test_relational_sine_replica.py)
retain both frozen controls, invalid-domain and availability cases,
full-node rates and explicit law provenance. Native Arg dispatch and
the selected sine runtime remain separate. No finite-horizon forecast,
unchanged invariant volume, periodic pulse, microscopic law selection
or physical identification follows from this static comparison.

<a id="sine-replica-constitutive-nonselection"></a>

### 20.5. Recursive inheritance and the equilibrium tangent do not select one law

The [existing composition comparison](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#common-exchange-geometry-and-composition)
already distinguishes equal replication from independent neighbor
superposition. Its counterfamily and the frozen discriminator above give
a stronger combined scope statement: exact recursive inheritance and the
same small-signal exchange can coexist with different complete dynamics.
The following is a corollary of those admitted laws, not a new constitutive
model or evidence that a hierarchy forms itself.

Take any finite connected simple undirected unit base graph with at least
two nodes, positive held capacities `nu_i`, and fixed `w,beta>0`. Keep `e=0`
and either existing member `epsilon=0` or `epsilon=1`. For an integer
`k>=1`, replace each base node `i` by `k` constituents `(i,r)`, put no edges
within a fiber, and replace each base edge by all `k^2` cross edges.
Supply synchronized fiber states
`x_(i,r)=X_i`, `theta_(i,r)=Theta_i` and capacity `nu_(i,r)=nu_i`.
With `q=L X` and `S_i=sum_j sin(Theta_j-Theta_i)` on the base, direct
neighbor counting gives, for every base preparation,

\[
d_{i,r}^{\rm fine}=k d_i,\qquad
q_{i,r}^{\rm fine}=k q_i,\qquad
S_{i,r}^{\rm fine}=k S_i,\qquad
m_{i,r}^{(\epsilon),\rm fine}=\frac1k m_i^{(\epsilon)}.
\]

Consequently both actual fine rows are the copied base rows:

\[
\dot x_{i,r}=w\nu_i m_i^{(\epsilon)}S_i,\qquad
\dot\theta_{i,r}=(w\nu_i/\beta)m_i^{(\epsilon)}q_i.
\]

Uniqueness preserves synchronization; no capacity or clock rescaling is
needed. Every base edge contributes the same storage on each of its `k^2`
copies, so the full storage satisfies `E_fine=k^2 E_base` on this invariant
submanifold. Applying an integer `ell>=1`-fold construction to the result is,
up to relabeling, the `(k*ell)`-fold construction, with storage factor
`(k*ell)^2`.
This proves finite iterative same-law inheritance for both candidates.
The support, fiber partition and synchronized preparation remain supplied;
it proves neither attraction to these fibers nor generic off-fiber closure,
autonomous hierarchical formation or a fractal dimension. Section 7 retains
the internal variables required away from synchronization for the sine law.

The common equilibrium tangent is equally insufficient for selection. At
any full equilibrium with `q=0,S=0`, the difference between these complete
fields is

\[
\dot x_i^{(\epsilon)}-\dot x_i^{(0)}
 =\frac{\epsilon w\nu_i}{\pi d_i^3}S_i^3,\qquad
\dot\theta_i^{(\epsilon)}-\dot\theta_i^{(0)}
 =\frac{\epsilon w\nu_i}{\beta\pi d_i^3}S_i^2q_i.
\]

Both corrections are cubic in a joint form/phase perturbation, including
at nonconsensus critical targets. Let `H_*` be the cosine phase Hessian
there and `K=diag(nu_i/d_i)`. Their full Jacobian is therefore identical:

\[
J_*=
\begin{pmatrix}
0&-(w/\pi)K H_*\\
(w/(\beta\pi))K L&0
\end{pmatrix}.
\]

When `H_*` is positive on the common-phase quotient, the existing
[conservative tangent argument](nodal/RESONANCE_FOUNDATIONS.md#finite-conservative-memory)
gives the same free oscillatory modes, with the common origins treated
separately. The same prescribed infinitesimal input and observation also
have the same linear response. This does not transfer a finite-amplitude
periodic family, its period or its transverse stability. In particular,
the bounded positive-frequency work-port peak theorem requires positive
loss; it is not a theorem about a bounded forced response at `e=0`.

Nevertheless, the already frozen non-equilibrium comparison in Section
20.3 gives `mathscr D_0=0` and `1/500<mathscr D_1<1/400`. Thus the candidates
have identical recursive inheritance and equilibrium linearizations but
different relative rates that a common clock rescaling cannot remove.
No frozen response is rerun for this conclusion. To select a microscopic
law one still needs a justified additional restriction or an independent
discriminator; calling either prepared inheritance or an oscillatory
tangent "fractal resonance" cannot supply that restriction.

The [shared comparison tests](../tests/physics/test_relational_sine_comparison.py)
check both admitted mobilities on a nonregular-degree base with unequal
per-fiber capacities and successive two- and three-fold replication. They
retain every fine row and check complete rates and storage against independent
values; this is implementation evidence for the corollary, not its proof.

<a id="sine-mobility-relative-geometry"></a>

## 21. Protected relative geometry without an assumed invariant volume

### 21.1. Quotient only the common origins of the actual complete law

Retain the exact twenty-edge doubled-C5 support and winding-one target
of Section 12, common positive held capacity and the conservative
current-squared mobility of Section 20. The comparison uses only its
already declared members `epsilon=0` and `epsilon=1`, with
`w=beta=nu=1`. The identities below display the existing positive
coefficients and apply to each fixed finite `epsilon>=0`; no parameter
search is needed. All constituent forms, phases and fixed connections
remain in the model. No input, support event or clock change is added.
Reflecting the target gives the same result for winding minus one:
the cosine storage, spectral gap and radius condition use the same
absolute target increment.

Let `n=10`, `d=4`, `P=I-11^T/n`, `L=L_f` and

\[
S_i(\theta)=\sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
m_i(\theta)=\frac{1+\epsilon(S_i/d)^2}{\pi d},
\qquad M=\operatorname{diag}(m_i).
\]

Constant common form translation and common circular phase rotation
are exact symmetries: neither changes `q=Lx`, `S`, `M` or either
complete rate row. In Section 12's target chart write

\[
x=\bar x\mathbf1+v,\qquad
\theta=\theta_*+c\mathbf1+h\pmod{2\pi},\qquad
v,h\perp\mathbf1.
\]

The retained relative state is `(v,h)`, with eighteen continuous
coordinates. Only the two common origins are discarded. Pairwise
form differences and circular phase differences, including internal
member state, remain determined. The common phase origin `c` has
a local continuous lift, rather than a globally defined real mean
on the phase torus.

The induced field in the same structural clock is exactly

\[
\boxed{\dot v=w\nu P M(h)S(h),\qquad
\dot h=(w\nu/\beta)P M(h)Lv.}
\]

Here `S(h)` and `M(h)` mean their evaluation at `theta_*+h`;
the omitted common phase cancels from every gap. No projected
mobility or scalar effective capacity replaces these matrix products.
The removed origins obey the independently determined rows

\[
\dot{\bar x}=\frac{w\nu}{n}\mathbf1^TMS,\qquad
\dot c=\frac{w\nu}{\beta n}\mathbf1^TMLv.
\]

Their initial values and these integrals reconstruct the full state
as long as the chart is valid. Discarding them therefore limits
the claim to relative structure; it does not prove that the original
means are constant. Projecting out their computed motion also does
not reparametrize time or supply an independently chosen phase clock.

### 21.2. The existing storage barrier protects the relative pattern

The exact same fine storage descends to the quotient:

\[
E(v,h)=\tfrac12 v^TLv+
\beta\sum_{\{i,j\}}[1-\cos(\theta_{j,*}-\theta_{i,*}+h_j-h_i)].
\]

Since `q=Lv` and `S` both have zero sum, differentiating using the
projected field cancels the exchange terms exactly:

\[
\dot E=w\nu q^TPMS-w\nu S^TPMq=0.
\]

The projections act trivially on the left vectors `q,S`; the
remaining cancellation uses the symmetry of the same diagonal `M`
in both rows. Conservation holds along the changed law itself, not
along a substituted sine trajectory.

Keep every radius, phase-lift and storage hypothesis from Section 12:

\[
\alpha=2\pi/5,\quad E_*=20\beta(1-\cos\alpha),\quad
0<r,\quad \alpha+\sqrt2r<\pi/2,
\]

\[
c_r=\cos(\alpha+\sqrt2r)>0,\qquad
\kappa_f=\frac{5-\sqrt5}{2}\min(1,\beta c_r),\qquad
0<\eta_*<\kappa_f r^2.
\]

The symbol `eta_*` is the existing excess-storage ceiling, renamed
here to distinguish it from the mobility parameter. Put
`Z^2=||v||^2+||h||^2` and `mathcal E=E-E_*`. The unchanged
target Hessian and fine spectral gap give the same geometric bound

\[
\mathcal E\ge\kappa_f Z^2\qquad(Z\le r).
\]

This inequality concerns the storage and target chart; its derivation
does not contain the mobility. Define the relative family

\[
\boxed{\mathcal V=
\{(v,h):Z^2<r^2,\quad \mathcal E<\eta_*\}.}
\]

Every state in `V` is protected for all positive and negative times
under the declared mobility. If there were a first exit at `Z=r`,
the same conserved excess would have to satisfy both
`mathcal E<eta_*` and `mathcal E>=kappa_f*r^2`, a contradiction.
More explicitly,

\[
\boxed{Z(t)^2\le\frac{\mathcal E(0)}{\kappa_f}
<\frac{\eta_*}{\kappa_f}<r^2\qquad(t\in\mathbb R).}
\]

There is no finite-time escape hidden in this statement. On the phase
torus `|S_i|<=d` and
`1/(pi*d)<=m_i<=(1+epsilon)/(pi*d)`. In particular every full
form rate is bounded by `w*nu*(1+epsilon)/pi`. The full form
can grow at most linearly on a finite time interval, and the phase
rates are at most linear in that finite form. Smooth continuation
therefore exists in both time directions. Alternatively, the relative
field is smooth on a neighborhood of the compact trapped closure;
its two origin rows stay bounded there and can be integrated separately.

The family `V` is open, has finite positive eighteen-dimensional
volume, and has compact closure inside the valid relative chart.
Its invariance does not require a bounded interval for the absolute
form mean. Every fine edge remains acute in its target-compatible
lift, and every cycle retains the target winding. Section 14's
phase-separation argument therefore continues to identify the same
five constituent pairs. These consequences use the proved geometry,
not the sine-specific rates or an invariant-volume assumption.

The individual pair swaps remain exact symmetries of this particular
mobility: they permute degrees, currents and both rows together.
Consequently each synchronized tip set is invariant. Smooth uniqueness
also prevents a state outside such a set from reaching it in finite
time. Removing all five tip sets gives the corresponding open invariant
family `V_*`, still of finite positive relative volume. This fact alone
does not establish a nonzero internal velocity, an angular circulation
bound or a repeating waveform; those are separate dynamical questions.

The conclusion is conditional maintenance of an already admitted
organization. The earlier local-transition preparation is not thereby
placed inside this protected family. Since the family is invariant in
both time directions, it is not a finite-time capture target for states
outside it under this same complete autonomous law. This uses the
[two-sided invariance argument](nodal/RESONANCE_FOUNDATIONS.md#conservative-formation-boundary),
not the unproved invariant measure for positive `epsilon`. A measure-based
obstruction to asymptotic capture requires separate recurrence hypotheses.

### 21.3. Mean motion and divergence change although storage does not

The original weighted-mean calculation exposes the difference without
choosing a new response preparation. More generally, on the admitted
unit support with held positive capacities put
`rho_i=d_i/nu_i` and `W=sum_i rho_i`. For the same mobility family,
the old weighted form mean and a locally lifted phase mean satisfy

\[
\boxed{\begin{aligned}
\frac{d}{dt}\frac{\sum_i\rho_i x_i}{W}
 &=\frac{w\epsilon}{\pi W}\sum_i\frac{S_i^3}{d_i^2},\\
\frac{d}{dt}\frac{\sum_i\rho_i\theta_i}{W}
 &=\frac{w\epsilon}{\beta\pi W}
       \sum_i\frac{S_i^2q_i}{d_i^2}.
\end{aligned}}
\]

The constant-mobility terms vanish because `sum S_i=sum q_i=0`.
The remaining terms are actual drift, not a fitted correction to a
conserved quantity. The weighted phase expression still depends on
declared continuous lifts; it is not a global scalar observable on
the phase torus. For the doubled-C5 coefficients frozen above these
two right-hand sides become respectively
`epsilon*sum(S_i^3)/(640*pi)` and
`epsilon*sum(S_i^2*q_i)/(640*pi)`.

Write `C_i=sum_{j~i} cos(theta_j-theta_i)`, so
`partial_theta_i S_i=-C_i`. The divergence of the full field is

\[
\boxed{\operatorname{div}F
=-\frac{2w\epsilon}{\beta\pi}
  \sum_i\frac{\nu_i q_i S_i C_i}{d_i^3}.}
\]

The form row has no form derivative; the displayed term is the
diagonal phase derivative of the changed phase row. This is also the
Euclidean divergence of the induced relative field. To see this
without treating constrained coordinates as independent, choose a
constant linear basis for the relative form and locally lifted phase
coordinates and append their two common origins. The complete field
is independent of the two origin values. Their two diagonal derivative
entries are zero, leaving precisely the relative-coordinate trace.
The same argument applies to node-reference differences instead of
the centered basis.

For `epsilon=0` both mean drifts and the divergence vanish identically.
For `epsilon=1` these identities fail even arbitrarily near the
existing protected target, not merely outside its acute domain. An
analytic local check makes this explicit without a new numerical
preparation. Put `c_alpha=cos(alpha)>0` and let `H` be any nonzero
centered direction. At `h=s*H`,

\[
S=-s c_\alpha LH+O(s^2),\qquad
C=d c_\alpha\mathbf1+O(s).
\]

Taking `v=-s*H` in the same open family gives

\[
\operatorname{div}F
=-\frac{2w\nu\epsilon c_\alpha^2}{\beta\pi d^2}
    s^2\|LH\|^2+O(s^3),
\]

which is nonzero for sufficiently small nonzero `s` when `epsilon>0`.
For the form mean, `L` maps the centered subspace invertibly onto
itself. That subspace contains vectors `Q` with `sum Q_i^3!=0`.
Choose the direction with `LH=Q`; then

\[
\begin{aligned}
\dot{\bar x}
&=-\frac{w\nu\epsilon c_\alpha^3}{n\pi d^3}
   s^3\sum_i Q_i^3+O(s^4),\\
\dot c
&=-\frac{w\nu\epsilon c_\alpha^2}{\beta n\pi d^3}
   s^3\sum_i Q_i^3+O(s^4).
\end{aligned}
\]

Both are likewise nonzero locally. These are directional proofs about
the existing open family, not a new selected-amplitude experiment or
radius search. Thus the old absolute-mean slab and ordinary-volume
proof cannot simply be inherited. A zero drift or divergence at a
particular captured state would not restore either all-state identity.

### 21.4. What is still required for an almost-everywhere return claim

The relative trapping theorem concerns **every** admitted initial
state. Recurrence requires a separate measure argument. For
`epsilon=0`, the relative field is divergence-free; ordinary relative
volume restricted to `V` or `V_*` is finite and invariant. The
[existing finite-measure proof](nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
then gives relative-state recurrence for almost every point with
respect to that measure. Full-state sine recurrence retains the
additional absolute-mean and phase-torus premises of its owner.

For `epsilon=1`, nonzero Euclidean divergence prevents the same
volume proof, but does not disprove recurrence or exclude a different
invariant density. To recover the comparable almost-everywhere
statement on the relative family, one sufficient missing object is
a finite invariant measure equivalent to its natural relative volume.
For a positive differentiable density `varrho(v,h)`, its exact
obligation is

\[
\operatorname{div}_{v,h}(\varrho F_{\rm rel})=0,
\qquad 0<\int_{\mathcal V}\varrho\,dv\,dh<\infty,
\]

with equivalence to the preparation measure and compatibility with
the complete two-sided flow. Positivity and smoothness on a
neighborhood of the compact closure would supply useful sufficient
finiteness and equivalence conditions. A singular invariant measure
concentrated at the stationary target exists, but its recurrence says
nothing about almost every initial state in the open family.

Even a density depending on phases alone requires a genuine
integrability check. Let `varrho(theta)` be global-phase-shift invariant,
write

\[
B_i=\partial_{\theta_i}\log m_i,\qquad
a_i=\partial_{\theta_i}\log\varrho.
\]

Because `q=Lx` spans the centered subspace, invariance for every
relative form would require a common value

\[
\nu_i m_i(a_i+B_i)=\gamma(\theta)\quad\hbox{for every }i.
\]

The shift condition `sum_i a_i=0` fixes the only candidate gradient:

\[
\boxed{a_i=-B_i+
\frac{\sum_j B_j}{\sum_j1/(\nu_jm_j)}\frac1{\nu_i m_i}.}
\]

This one-form must be an exact gradient on the relative chart, with
the necessary positivity, normalization and any extension conditions.
No such solution is established here. A product of reciprocal nodal
mobilities cannot be assumed to solve it: differentiating that product
also introduces neighboring mobilities' derivatives. Moreover failure
of this phase-only ansatz would not exclude a density depending on
both relative form and phase. The general equation above, rather than
an unproved invariant-volume label, is the remaining obligation.

This theorem therefore certifies relative geometry, while leaving the
alternative's comparable almost-everywhere recurrence claim unavailable.
It neither labels that dynamics nonrecurrent nor imports the old
internal-pulse period or circulation theorem. Relative return, if
subsequently proved, would itself not ensure return of discarded
common origins or provide a deadline for a selected state.

### 21.5. Shared balance and admission evidence

The existing mobility comparison retains its own complete-law
provenance. Its `relative_balance(reference_node=...)` reader uses captured fine rows,
phase currents and cosine sums to report reference-node coordinate
rates, weighted-mean drift and full/relative divergence. Exact
constant-mobility cancellations remain distinct from finite
state-specific interval enclosures.

`assess_sine_mobility_geometry` in the
[scale owner](../src/tnfr/physics/relational_sine_scale.py)
reuses the actual doubled-cycle support, symbolic target, supplied
phase lifts and shared source storage/barrier admission. It separates
an admitted relative family, a trapped captured source and membership
in the stricter family with synchronized tips removed. The same
geometry bounds do not relabel the alternative as the original law.
The report retains recurrence as unavailable when its invariant-measure
obligation is unresolved, rather than turning that absence into a
negative trajectory verdict.

The source capture is not treated as the exact symbolic twist merely
because rounded phases are close to it. No new radius, uncertainty
budget, preparation scan or frozen response is used. The
[relational contract](../docs/contracts/RELATIONAL_DYNAMICS.md#sine-mobility-geometry)
and [independent replica tests](../tests/physics/test_relational_sine_replica.py)
own the executable admission and balance checks. Protected relative
organization remains distinct from autonomous formation, physical
identification and selection of the microscopic mobility.

<a id="sine-pair-emission-descent"></a>
## 22. Local and whole-pair form actions on an exact unordered NFR

### 22.1. Keep the continuous quotient and the supplied event separate

Use the normalized-sine complete law and the nonantipodal pair chart
of Sections 8 and 19. The selected pair has an independently valid
fixed-support swap and equal positive held capacities. Other pairs may
remain ordered where their swaps are not symmetries. Write

\[
x_\pm=X\pm u,\quad \theta_\pm=\Theta\pm\delta,
\quad |\delta|<\pi/2,\qquad
(R,U,Q)=(\cos\delta,u^2,u\sin\delta).
\]

The retained state identifies precisely the two lifts related by
`(u,delta) -> (-u,-delta)`, with a single lift at the synchronized
tip `u=delta=0`. Equal phase alone and equal form alone are not tips.
Unequal held capacities or asymmetric attachments are not silently
removed: their sufficient states remain governed by Sections 13 and 19.

Consider the **primary form action** of registered AL on existing
members. Its shared implementation is
[`emission_epi_proposal`](../src/tnfr/operators/al_sha_stage_proposals.py):
the supplied boost is added, the configured EPI boundary projection
is applied, and a proposal that decreases form is rejected. Let `T(x)`
denote this actual deterministic admitted point map, including projection,
and put `s_+=T(X+u)-(X+u)`, `s_-=T(X-u)-(X-u)`. Both increments are
nonnegative on the comparison domain. All proposals must be finite and
admitted before using the following identities. A soft-boundary rejection
is not an alternative valid endpoint; a saturated no-op is not an
unclipped boost.

This is a supplied hybrid reset attached to the stated sine observation,
not a term already present in its continuous law. The graph, capacity,
phase and clock stay fixed. The general public AL runtime additionally
consumes admission, lifecycle and history state which this quotient does
not retain. No closure of that larger runtime is asserted.

### 22.2. Exact obstruction and the two exceptional cases

Applying the same point map to the first member or to the second gives

\[
\begin{array}{c|ccc}
 &X'&U'&Q'\\\hline
\text{first}&X+s_+/2&U+s_+u+s_+^2/4&Q+(s_+/2)\sin\delta\\
\text{second}&X+s_-/2&U-s_-u+s_-^2/4&Q-(s_-/2)\sin\delta.
\end{array}
\]

Both leave `Theta,R` unchanged and satisfy
`Q'^2=U'(1-R^2)`. Exchanging the input lift while retaining the
same member label exchanges these two candidate outputs. Consequently
the fixed-member action has a well-defined unordered output at this
state **if and only if**

\[
\boxed{(s_+=s_-=0)\quad\text{or}\quad(u=\delta=0).}
\]

Indeed equal means first require `s_+=s_-=s`. If `s=0`, both
projected actions are identities. If `s>0`, equality of `U'` forces
`u=0`, and equality of `Q'` forces `sin(delta)=0`, hence `delta=0`
on the chart. Conversely either displayed condition suffices. This is
an exact statement on the domain where both candidates are admitted;
event admission itself must also agree on equivalent lifts. It does
not depend on a small boost approximation or on an unproved stability
effect of AL.

For an unprojected common boost `b>0`, the two differences are
`U'_first-U'_second=2*b*u` and
`Q'_first-Q'_second=b*sin(delta)`. Thus the equal-phase stratum
`R=1,U>0` still distinguishes the targets through `U'`. The
equal-form stratum `U=0,R<1` distinguishes them through `Q'` even
though their means and new `U'` agree. Losing either stratum would
incorrectly certify a collective action.

At the tip an unprojected singleton action has the unique projected
output `X'=X+b/2`, `R'=1`, `U'=b^2/4`, `Q'=0`. Nevertheless a
deterministic fine-member selector cannot be swap-equivariant there:
the source is fixed by the swap, whereas neither singleton target is.
This is the existing [selector-symmetry obstruction](../src/tnfr/physics/selector_symmetry.py).
A unique projected outcome therefore does **not** require a unique
labeled fine realization. Nor does its uniqueness choose when the
action occurs. At a geometric tip, separate member histories can still
distinguish the full public AL outcomes.

### 22.3. A sufficient marked port and a symmetric control

For an action on a supplied constituent, retain its offsets

\[
u_p=x_p-X,\quad v_p=\sin(\theta_p-\Theta),\qquad
u_p^2=U,\quad v_p^2=1-R^2,\quad u_pv_p=Q.
\]

These constrained data reconstruct that member and its partner:
`delta_p=atan2(v_p,R)`, then
`(x_p,theta_p)=(X+u_p,Theta+delta_p)` and
`(x_other,theta_other)=(X-u_p,Theta-delta_p)`. The chart has `R>0`,
so this also covers equal phase and equal form without division by
`u` or `sin(delta)`. For `R<1`, the sign of `v_p` and the unmarked
state suffice; for `R=1,U>0`, the sign of `u_p` suffices. The missing
choice is generically one binary orientation, not an extra independent
continuous coordinate. At the tip both offsets vanish; persistence of
a separate label, lineage or external port remains additional data.

With `s_p=T(X+u_p)-(X+u_p)`, the marked update is

\[
X'=X+s_p/2,\quad u_p'=u_p+s_p/2,\quad v_p'=v_p,
\quad U'=U+s_pu_p+s_p^2/4,\quad Q'=Q+s_pv_p/2.
\]

The port transforms with its member when the fine graph is relabeled.
A fixed untransformed label is not such a mark. A rule such as selecting
the larger form would itself be a supplied policy, require tie handling,
and would not derive occurrence from the continuous law.

In contrast, apply the same `T` to **both** members from one source
snapshot. If `y_+=T(X+u)` and `y_-=T(X-u)`, then

\[
X'=(y_++y_-)/2,\qquad
U'=(y_+-y_-)^2/4,\qquad
Q'=(y_+-y_-)\sin\delta/2.
\]

The swap exchanges `y_+,y_-` and changes the sign of `sin(delta)`,
so these outputs are invariant. This whole-pair form map descends on
its invariant admission domain without retaining a member port.
For an unprojected boost, `X'=X+b` while `R,U,Q` are unchanged.
With clipping, `U,Q` can change: their preservation is not part of
the symmetry result. The whole-pair action is a different supplied
event, not a way to reinterpret a singleton action after discarding
its target. Neither the quotient nor AL fixes a per-pair rather than
per-member boost convention.

### 22.4. Storage and occurrence remain separate obligations

For the same fixed fine support, let `q=Lx` and let `h` contain
the actual form increments of the admitted event. The sine storage
`H=x^T Lx/2+beta*V(theta)` obeys the exact reset identity

\[
\boxed{H^+-H^-=q^Th+\tfrac12h^TLh.}
\]

The phase part is unchanged. At one member `a` this is
`s*q_a+d_a*s^2/2`, with its actual fine degree. It may be positive
or negative; increasing a signed local form is not an energy or
pattern-maintenance theorem. Equal-form interchangeable members have
equal `q_a` and degree. Their singleton actions therefore have the
same storage jump even when their distinct phases make `Q'` differ.
A common event balance cannot recover the discarded target.

These resets can change the internal state or leave a protected family;
the applicable barrier needs fresh admission. On the unprojected
whole-pair action, retaining `R,U,Q` preserves those internal
coordinates, but changing `X` can still alter relations to neighboring
pairs. A pressure refresh recomputes the selected law's pressure from
the new state; a retained old pressure is not the same continuation.
Continuous loss does not pay for a reset without a declared reservoir,
and no target, amplitude, time, grammar rule or autonomous event trigger
has been derived by the descent test.

The read-only `assess_sine_pair_emission` in the
[scale owner](../src/tnfr/physics/relational_sine_scale.py) reuses the
actual AL proposal and the current mixed-state source admission with
common positive held capacity. It reports both
singleton alternatives and the symmetric whole-pair control without
mutating the source or executing public AL lifecycle effects. Exact
represented form arithmetic and member matching determine the descent
verdict; trigonometric enclosures do not substitute for that equality.
The [independent algebra controls](../tests/physics/test_relational_pair_emission.py)
check the quotient identities and storage distinction. A supplied
collective action is now specified where the criterion holds; a
collective action selected by the dynamics remains a different question.

<a id="sine-autonomous-regional-transfer"></a>
## 23. Autonomous regional transfer and supplied form injection

### 23.1. The regional balance of the complete sine law

Keep finite connected simple unit support, strictly positive held capacities,
and the original normalized-sine complete law, with no input, event,
clipping or added phase velocity. For this balance the form-loss coefficient
may be any `e>=0`; the periodic control below separately requires `e=0`.
Use `a=w/pi`, `b=w/(beta*pi)`, `q=Lx`, `d_i>0` and
`S_i=sum_{j~i}sin(theta_j-theta_i)`. Both rows are retained:

\[
\dot x_i=\frac{\nu_i}{d_i}[-e q_i+aS_i],\qquad
\dot\theta_i=b\frac{\nu_i}{d_i}q_i.
\]

For a supplied region `A`, define its weighted form sum, its fixed weight,
and its weighted form mean by

\[
M_A=\sum_{i\in A}\rho_i x_i,\qquad
\rho_i=d_i/\nu_i>0,\qquad
H_A=\sum_{i\in A}\rho_i,\qquad \bar x_A=M_A/H_A.
\]

This is a structural coordinate, not an identification with physical mass
or stored energy. Sum the fine form row over `A`. Every internal edge
cancels with its reverse orientation in both the form difference and the
odd sine term. Only the boundary remains:

\[
\boxed{\dot M_A=\sum_{\substack{i\in A,\ j\notin A\\j\sim i}}
\left[e(x_j-x_i)+a\sin(\theta_j-\theta_i)\right],
\qquad \dot{\bar x}_A=\dot M_A/H_A.}
\]

The orientation is into `A` from its complement. The same cut contributes
the negative rate to the complement, and the full-support sum `M` is
constant. This recovers the existing weighted-mean invariant and identifies
its exact regional transfer, including the dissipative form-difference
channel. Dissipation of storage does not destroy this particular invariant.
On supplied continuous phase lifts there is also the exact companion row

\[
\frac{d}{dt}\sum_{i\in A}\rho_i\theta_i
=b\sum_{\substack{i\in A,\ j\notin A\\j\sim i}}(x_i-x_j).
\]

The lifted sum is not a globally defined average of circular phases.
Neither boundary identity by itself closes the regional dynamics: the
actual boundary member states, capacities and support remain consumed.
They do not transfer to native Arg dynamics or a state-dependent mobility
merely because those laws share another storage balance.

### 23.2. A positive control inherited from the existing pair pulse

Choose the supplied `K2,2` support, with regions `A={0,1}` and `B={2,3}`,
all four cross-edges and no internal edges. Hold common capacity `nu>0`
and `e=0`, with `w,beta>0` and the same structural clock. Set

\[
x_{0,1}=X_A,\quad x_{2,3}=X_B,\qquad
\theta_{0,1}=\Theta_A,\quad\theta_{2,3}=\Theta_B.
\]

The members of each pair have equal complete rows. Smooth uniqueness
therefore makes this synchronized submanifold invariant, exactly as in
[Section 7](#sine-replica-inheritance). It is a supplied preparation,
not a proof that generic pairs synchronize. Its internal coordinates
remain `R=1,U=Q=0`; means close here because these internal states are
fixed, not because arbitrary internal states are irrelevant.

Let `D=X_A-X_B` and `Delta=Theta_B-Theta_A`. Each fine degree is two,
so the two incoming sine terms cancel that normalization. The induced
complete regional rows are

\[
\begin{aligned}
\dot X_A&=a\nu\sin\Delta,&\dot X_B&=-a\nu\sin\Delta,\\
\dot\Theta_A&=b\nu D,&\dot\Theta_B&=-b\nu D,\\
\dot D&=2a\nu\sin\Delta,&\dot\Delta&=-2b\nu D.
\end{aligned}
\]

In particular `H_A=H_B=4/nu`, and the boundary current is
`M_A_dot=4*a*sin(Delta)=-M_B_dot`, consistent with the full-node
cut calculation. Both common origins `(X_A+X_B)/2` and
`(Theta_A+Theta_B)/2` are fixed on these lifts. Differentiating the
relative phase row yields

\[
\ddot\Delta+4ab\nu^2\sin\Delta=0,\qquad
E_f=4E_c,\qquad E_c=\tfrac12D^2+\beta(1-\cos\Delta).
\]

This is precisely the existing
[nonlinear P2 exchange](nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission),
inherited under a complete two-replica blow-up. No primitive oscillator,
new coupling law or activation threshold has been appended.

For the stated initial control choose equal forms `X_A=X_B=C` and
a supplied lift `0<|Delta_0|<pi`. Then

\[
\begin{aligned}
\dot X_A(0)&=a\nu\sin\Delta_0=-\dot X_B(0)\ne0,\\
\dot\Theta_A(0)&=\dot\Theta_B(0)=0,\\
\ddot\Theta_A(0)&=2ab\nu^2\sin\Delta_0
                 =-\ddot\Theta_B(0),\\
\ddot\Delta(0)&=-4ab\nu^2\sin\Delta_0.
\end{aligned}
\]

Thus one region gains form while the other loses it for a nonzero initial
time interval, determined by the sign of the supplied relative phase.
The initially zero phase velocity does not permit holding phase fixed:
the acquired form difference immediately induces its second-order
response. In particular
`Theta_A(t)=Theta_A(0)+a*b*nu^2*sin(Delta_0)*t^2+o(t^2)`.
Reversing `Delta_0` reverses these signed responses. With equal forms
and `Delta_0=0` the source is stationary; an exactly antipodal relative
phase is also stationary and is outside the strict pulse preparation.

The existing P2 theorem gives a periodic complete exchange for this
strictly sub-separatrix energy. Its energy relation also fixes the
maximum regional form displacement without fitting a response:

\[
\max_t|X_A(t)-C|=\max_t|X_B(t)-C|
=\sqrt\beta\,|\sin(\Delta_0/2)|.
\]

Indeed the orbit reaches `Delta=0`, where
`|D|=sqrt(2*beta*(1-cos(Delta_0)))`, and the two form changes are
`D/2` and `-D/2`. The coefficients `w,nu` set the clock scale;
they do not change this displacement at fixed initial phase and `beta`.
This is a prepared reversible transfer and return, not creation of a
new support, attraction to the synchronized submanifold or permanent
unidirectional supply.

### 23.3. Why a positive AL-only endpoint is not that closed flow

Now retain the original full sine state, support, positive held capacities
and law, and evaluate any admitted structural AL-only reset. Let
`h_i=x_i^+-x_i^-` be its **actual** increments after boundary projection
and rounding, including zero at unaffected nodes. The shared AL
postcondition is `h_i>=0`. Consequently

\[
\boxed{M^+-M^-=\sum_i\rho_i h_i\ge0,
\qquad M^+-M^->0\ \Longleftrightarrow\ \exists i:h_i>0.}
\]

Since the unchanged unforced sine flow conserves `M`, any strictly
positive increment excludes this full reset endpoint at **every**
elapsed time of that flow. This obstruction holds for the whole-pair
map that descends in Section 22, as well as for a marked singleton;
it is independent of whether the reset raises or lowers storage. It
also holds with `e>0` for this original normalized-sine law.

This is a full-form endpoint claim in the same chart. If an observation
discards a common form origin, equality of the observed endpoints is a
different question and cannot be excluded solely by this invariant.

Clipped or rounded no-ops have zero increment and evade this particular
obstruction. They are structural identities, not certificates of a
positive-time return or of public AL lifecycle equivalence. A rejected
projection is not an admitted no-op. Nor is zero weighted increment
sufficient for flow realizability: for example a nonzero balanced form
reset at a uniform-form, common-phase equilibrium preserves `M` but
cannot be produced by its stationary unforced flow.

The positive regional control does not contradict the obstruction.
Observing only `A` discards the compensating response in `B`; the
full state never undergoes an AL-only increment. Internal interaction
already explains its gain through an existing boundary phase contrast,
and the phase response evolves with it. Reproducing a pure supplied
AL endpoint instead requires changes beyond that unchanged closed
flow, such as a declared input or compensating external state; their
law and preparation cannot be inferred by naming them. No autonomous
target, hybrid jump, occurrence time or loss reservoir follows from
the continuous transfer.

### 23.4. Shared current and endpoint observations

`SineExchangeComparison.regional_transfer(region=...)` evaluates this
boundary ledger from one captured source. It retains the form-difference
and sine contributions and an independently accumulated full-node rate
residual. `assess_form_increment(increments=...)` computes the exact
change of the conserved weighted form for a supplied full increment
vector. These readers live with the
[complete sine comparison](../src/tnfr/physics/relational_sine_comparison.py),
so they do not reconstruct the law from a reported response or dispatch
the native runtime. Their finite enclosures remain distinct from the
all-state cancellation proof.

The preceding AL report's `form_increment(outcome=...)` delegates its
actual first-member, second-member or whole-pair increments to this
same ledger. A strictly nonzero weighted increment proves the specified
closed-flow endpoint obstruction; a zero value leaves other realizability
obligations open. No captured boundary rate is a time-integrated transfer,
and no observation executes an event or advances the graph. The
[independent algebra controls](../tests/physics/test_relational_regional_transfer.py)
derive the cut cancellation, the synchronized fine rows and their phase
acceleration, inherited storage, and a storage-decreasing AL endpoint
which still violates weighted-form conservation.

<a id="sine-joint-identity-window"></a>
## 24. Joint form-phase identification during autonomous exchange

### 24.1. A storage-defined observation, not an extra dynamical rule

For the fixed positive storage scale `beta`, define

\[
J_{ij}=(x_i-x_j)^2+2\beta[1-\cos(\theta_i-\theta_j)],
\qquad d_\beta(i,j)=\sqrt{J_{ij}}.
\]

The map `z_i=(x_i,sqrt(beta)*cos(theta_i),sqrt(beta)*sin(theta_i))`
embeds the form and circular phase into Euclidean three-space, with
`d_beta(i,j)=||z_i-z_j||`. Thus `d_beta` is a metric on this state
space: it vanishes precisely for equal form and equal circular phase,
and it inherits the triangle inequality. `J` is its square, not itself
a metric. Both terms use the existing configured storage scale in the
same declared, currently nondimensional coordinate chart. If form units
are assigned, `beta` must carry their square for `d_beta` to have form
units; this bookkeeping does not establish a laboratory bridge for the
law's coefficients or clock. The scale is inherited from the declared
storage, not fitted to identify a desired partition or proved to be a
unique physical geometry.

The observation is invariant under node relabeling, common form shifts,
common phase rotations and independent full-turn changes of phase
representatives. It consumes form and phase; held support and capacity
remain separate model premises. As in Section 14, an inferred pair
requires unique mutual nearest partners. A tie must remain a tie.
Such instantaneous identification does not establish sufficient state
for a future law or select a new support or hierarchy event.

### 24.2. Exact reference identity survives a phase-only collision

Keep the complete conservative normalized-sine law and the synchronized
`K2,2` preparation of Section 23: common positive capacity, fixed support,
equal initial forms and a supplied lift `0<|Delta_0|<pi`. The full
trajectory retains identical members within each pair. For every time,

\[
\boxed{J_{\rm within}=0,\qquad
J_{\rm cross}=D(t)^2+2\beta[1-\cos\Delta(t)]
=2E_c=4\beta\sin^2(\Delta_0/2)=:J_*>0.}
\]

This is an identification consequence of the existing inherited storage
identity, not another conservation postulate. Every node has its other
constituent as its unique nearest partner throughout the entire reference
motion. The observation can recover that partition without reading its
labels or the supplied support; those premises are used to prove the
claim, not to resolve an observed distance tie.

The P2 libration crosses `Delta=0`. At that instant all four phases
coincide, so every phase-only distance is zero and phase-only pairing
must abstain. But then `D^2=2E_c>0`: the joint observation retains
exactly the same strict pair distinction. At the form turning points
`D=0`, the nonzero phase separation instead supplies it. Form and phase
exchange their contributions without erasing the joint distinction.
At `Delta_0=0` with equal forms the reference has `J_*=0`, four
identical observed states and no uniquely identified pairing. This
degenerate reference is not certified by continuity from positive
amplitude.

### 24.3. A full-field perturbation bound over a declared window

Now allow independent preparation errors in the form and phase of
**all four** nodes. The perturbed state need not preserve either
pair's internal synchronization. Both trajectories use the same
`K2,2` graph, common capacity `nu>0`, `beta,w>0`, `e=0`, clock,
and no input, event or clipping. Specify real phase lifts for the
initial comparison and continue them by the smooth phase row.

The error norm is taken in the eight real coordinates

\[
Z=(x_0,\ldots,x_3,\sqrt\beta\theta_0,\ldots,
\sqrt\beta\theta_3),\qquad
r_0^2=\|Z(0)-Z_*(0)\|_2^2.
\]

For componentwise preparation radii `f_i,p_i>=0`, the sufficient
initial bound is `r_0^2<=sum_i(f_i^2+beta*p_i^2)`. Each component
can vary independently within its declared interval. A common-origin
or phase-lift choice must be specified before this bound is evaluated;
small circular distance alone is not a declaration of small error in
an arbitrary lift.

Put `c=w*nu/(pi*sqrt(beta))`. On this degree-two regular graph the
Jacobian of the **full fine field** in the scaled real lifts is

\[
DF(Z)=c\begin{pmatrix}0&-L_{\cos}(\theta)/2\\L/2&0\end{pmatrix},
\]

where `L_cos` is the symmetric edge Laplacian with signed coefficients
`cos(theta_j-theta_i)`. Both `L/2` and `L_cos/2` have Euclidean
operator norm at most two. For example the absolute row sum of the
signed matrix is at most two after division by the degree, and its
symmetry bounds the spectral norm. Equivalently
`-L <= L_cos <= L` as quadratic forms and the largest eigenvalue
of the `K2,2` Laplacian is four. The block Jacobian norm is therefore
at most `L_*=2c`, uniformly in form and phase.

This proves a global Lipschitz bound in the scaled **lift** coordinates,
and hence global continuation and the full-field difference estimate

\[
\|Z(t)-Z_*(t)\|_2\le r_0e^{L_*t}\le r_0e^{L_*T}=:r_T
\qquad(0\le t\le T).
\]

It does not assert that the embedded circular-state vector field has
an equally bounded Jacobian independent of form. The embedding is
used only for its observation error:

\[
\|z_i-z_{*,i}\|^2
=(x_i-x_{*,i})^2+
4\beta\sin^2[(\theta_i-\theta_{*,i})/2]
\le(x_i-x_{*,i})^2+\beta(\theta_i-\theta_{*,i})^2.
\]

For any two nodes the reverse triangle inequality and
`||e_i-e_j||<=sqrt(2)*sqrt(||e_i||^2+||e_j||^2)` give

\[
|d_\beta(i,j)-d_{\beta,*}(i,j)|\le\sqrt2\,r_T.
\]

Writing `s=sqrt(J_*)`, throughout the whole window we consequently have

\[
\max_{\rm within}d_\beta\le\sqrt2r_T,\qquad
\min_{\rm cross}d_\beta\ge\max(0,s-\sqrt2r_T).
\]

In particular the following strict, prospective inequality guarantees
the same mutual nearest pairs for **every** preparation in the error
ball or declared component box and every time in the window:

\[
\boxed{J_*>8r_0^2\exp\!\left(
\frac{4w\nu T}{\pi\sqrt\beta}\right).}
\]

The inequality is sufficient, not necessary. A failed or unresolved
bound cannot by itself prove loss of identity. Zero uncertainty gives
the exact reference result. For nonzero uncertainty the estimate
establishes a finite-window guarantee, not attraction, nonlinear
orbital stability or permanent identification. The held graph and
capacities also do not emerge from this error estimate.

### 24.4. A fixed full-exchange control and genuine degeneracies

Before evaluating the finite report, fix the source to `K2,2` with
`C=1/2`, phases `(0,0,1/4,1/4)`, `beta=nu=1`, `e=0` and effective
`w=1`. Set `T=10` in its structural clock and give each form and
phase component radius `10^-6`. Then `r_0^2<=8*10^-12` and
`J_*=2*(1-cos(1/4))`. These are preparation specifications, not
fitted errors or observed response curves.

The existing P2 period bound gives
`T_pulse<=pi^2/cos(1/8)<10`; for example
`pi<22/7` and `cos(1/8)>=127/128` already give the strict upper
bound `61952/6223<10`. Thus the chosen window includes one full
reference exchange, including its phase-only collision. The strict
identity inequality can also be checked without trajectory evaluation:
`J_*>1/40`, while its right-hand side is less than
`64*10^-12*3^14<1/40`. These conservative inequalities use
`sin(y)>=2y/pi` for `0<=y<=pi/2`, `3<pi<22/7`, and
`exp(40/pi)<exp(14)<3^14`. A finite interval report can retain
sharper margins without changing the declared source or budget.

The zero-amplitude control has no reference separation. The separately
declared large-error control, with every radius `1/8` and the same
nonzero-amplitude reference, includes a preparation with all forms
`1/2` and all phases `1/8`. All four observed node states then
coincide at the initial time. The entire box therefore cannot have
the strict pair-identification property, independently of how sharp
an error bound is. This explicit counterexample must not be generalized
to every preparation budget for which the sufficient inequality fails.

### 24.5. Shared observation and window admission

`observe_joint_pairs(nodes=..., forms=..., phases=..., storage_scale=...)`
in the [scale owner](../src/tnfr/physics/relational_sine_scale.py) observes
the actual signed forms and circular phases. It reuses strict mutual
nearest-partner admission and may abstain; it does not read a proposed
partition or use support to resolve a tie.

`assess_sine_joint_pairing_window` separately admits the complete
`K2,2` reference and its supplied pairs, phase lifts, `form_error_bounds`,
`phase_error_bounds` and `window_end`. The returned
`initial_scaled_error_squared`, `lipschitz_upper` and
`propagated_scaled_error_squared_upper` retain the full-field error
provenance. `identification_budget_margin` applies the strict squared
inequality; the report also requires strict interval nearest-partner
comparisons. `reference_identity_certified` is separate from
`whole_window_identity_certified`. A numerical abstention cannot erase
the exact reference theorem or become evidence of actual perturbed
failure.

The report's `reference_period_bounds` reuse the existing P2 period
owner; `window_covers_reference_period` concerns that reference only,
not a period or a recurrence deadline for perturbed trajectories. No
graph is advanced, and no snapshot is represented as the evolved
endpoint. The [independent algebra controls](../tests/physics/test_relational_joint_identity.py)
check the circular metric identity, conserved reference separation,
complete scaled-lift Jacobian and both observation-error factors.

<a id="sine-joint-grouping-comparison"></a>
## 25. The frozen phase transition does not change the joint pairing

### 25.1. Compare two observations of the same complete preparation

Keep the exact doubled-`C5` graph, law, clock and the two form-reversed
sources from Sections 16–17, with `e=0`, `w=beta=nu=1` and
`d=u=1/8`:

\[
\theta^*=(0,d,2d,3d,1,1,2,2,3,3),\qquad
x^*=(u,-u,u,-u,0,0,0,0,0,0).
\]

For `sigma=+1,-1`, retain **all twenty** independent initial radii
`rho=2^-20` around `(sigma*x*,theta*)` and the original observation
window `[1/128,1/64]`. The supplied structural pairs are
`(0,1),(2,3),(4,5),(6,7),(8,9)`. No parameter, horizon, support,
pressure law or event schedule is adjusted for the joint observation.
The `K2,2` constant-separation result is not applied to this graph.

The joint observation of Section 24 is
`J_pq=(x_q-x_p)^2+beta*D_pq`, where
`D_pq=2*(1-cos(theta_q-theta_p))` is the existing phase-only chord.
Differentiation along the **complete** fine field gives

\[
\boxed{\dot J_{pq}=
2(x_q-x_p)(\dot x_q-\dot x_p)
+2\beta\sin(\theta_q-\theta_p)(\dot\theta_q-\dot\theta_p).}
\]

Both rows are necessary. In particular a measured or computed phase
rate alone does not determine this derivative. The sine form row
depends on the full neighbor phase currents; the phase row depends
on the full form gradients. These are observation rates, not new
terms in either evolution law.

### 25.2. A different joint partition is already strictly present

Write `F(s)=2*(1-cos(s))`. At either source center,

\[
J_{02}=J_{13}=F(2d),\qquad
J_{01}=J_{12}=J_{23}=4u^2+F(d).
\]

The closest opposite-form competitors are farther by

\[
h_0=4u^2+F(d)-F(2d)\ge d^2=1/64>0.
\]

Indeed `u=d` and
`F(2d)-F(d)=integral_d^(2d) 2*sin(s) ds <=3d^2`.
The remaining first-four comparison `0--3` has the same form cost
and a larger phase gap `3d`. Every node outside those four is at
phase distance at least `5d` from them. Here
`F(2d)<=4d^2` while `F(5d)>10d^2`, using the chord bounds below.
The last three pairs have exactly zero internal joint distance and
strictly positive distance to every outsider. All phase gaps lie
below `pi`. Thus every competitor has been checked, and the unique
mutual joint pairs at both centers are

\[
\boxed{\mathcal C=
\{(0,2),(1,3),(4,5),(6,7),(8,9)\}.}
\]

The form-reversed source has the same squared form differences, so
this observation is unchanged. All its joint-distance rates reverse:
the initial form rates stay unchanged, whereas form differences and
phase rates change sign. In particular `Jdot_02=Jdot_13=0` at both
centers, because their form differences and phase-rate differences
are zero. These zero derivatives do not create a tie: their nearest
margins are strictly positive. More generally the exact conservative
reversal of Section 16 gives `J_tilde(t)=J(-t)`; it does not imply
that an initially strict nearest relation must change at time zero.

### 25.3. The same joint pairs persist on the complete frozen boxes

Reuse the full-field phase bound of Section 17. For every source in
either box and `0<=t<=T=1/64`,

\[
|\theta_i(t)-\theta_i^*-(\sigma x_i^*/\pi)t|\le E,
\quad E=\rho+2\rho T/3+T^2/9
=\frac{8483}{301989888}.
\]

The same complete field has `|xdot_i|<=1/pi`. Integration, including
each independent initial form error, therefore gives the additional
bound

\[
\boxed{|x_i(t)-\sigma x_i^*|\le B:=\rho+T/3
=\frac{16387}{3145728}.}
\]

This bound does not freeze form or assume a common form error within
any pair. The phase enclosure uses the original central phase rates
and nonlinear remainder; it is not an extrapolated trajectory. The
span estimate from Section 17 remains below mathematical `pi` on
this entire interval, so absolute phase gaps equal circular gaps.

For `0<=s<=pi`, the elementary chord inequalities are

\[
\frac25s^2\le\frac4{\pi^2}s^2\le F(s)\le s^2,
\]

where the first inequality uses `pi^2<10`. They follow from
`2y/pi<=sin(y)<=y` on `0<=y<=pi/2`. For the desired first-four
pairs `(0,2),(1,3)`, the reference forms and central phase rates
coincide within each pair. Consequently

\[
J_{\rm desired}\le U_1:=4B^2+(2d+2E)^2.
\]

Every remaining first-four competitor has initial form difference
`2u` and phase gap at least `d`. Independently bounding both
coordinates gives

\[
J_{\rm competitor}\ge L_1:=(2u-2B)^2+
\frac25\left(d-\frac{2uT}{3}-2E\right)^2.
\]

Both lower gap bounds inside the squares are strictly positive.
They must not be squared without that check. Their exact margin is

\[
L_1-U_1=
\frac{33345408988471}{37999121855938560}
>\frac1{2048}>0.
\]

For a first-four node and an outsider among the last six, the phase
gap is at least `g_2=5d-u*T/3-2E>0`, giving
`J_competitor>=(2/5)*g_2^2`. Its margin over `U_1` is greater
than `L_1-U_1`. For each of the last three desired pairs,
`J_desired<=U_2:=4B^2+4E^2`; every outsider again has phase gap
at least `g_2`, giving a still larger margin. This includes other
members of the last six, whose initial interpair gap is at least one.

Thus all eighty directed nearest-partner comparisons have the required
strict sign. For **every** source in either frozen box and every time
`0<=t<=1/64`, the joint nearest map is

\[
\boxed{(2,3,0,1,5,4,7,6,9,8).}
\]

In particular this proves the new observation on the original frozen
window `[1/128,1/64]`. The same estimates also cover its preparation
and intervening interval: the identified pairs cannot disappear and
reappear before that window. This is an analytic corollary of the
unchanged budgets, not a replacement or retiming of the earlier
phase-only forecast.

### 25.4. Observation change, collective state and formation are distinct

Under phase-only observation the positive box has the structural
matching on its window, while the negative box has the nonmutual
nearest map proved in Section 17. Under the joint observation both
boxes already have the same strict partition `C` initially and
retain it throughout the interval just proved. Therefore that
specific phase-only onset or loss is **not** onset or loss of joint
pair identity under the specified joint criterion.

This comparison neither invalidates the old phase-only theorem nor
declares a uniquely correct physical NFR metric. It identifies the
different information retained by the two observations. Neither
distance ordering creates support, selects an operator or proves
permanent formation.

The joint pairs also do not automatically inherit the structural
pair law. The proposed pairs `(0,2)` and `(1,3)` contain existing
fine edges, so they fail the strict no-internal-edge replica admission.
More decisively, even the broader independent-swap criterion of
Section 18 fails. For `(0,2)`,

\[
N(0)\setminus\{2\}=\{3,8,9\},\qquad
N(2)\setminus\{0\}=\{1,4,5\}.
\]

Their interchange does not preserve the fixed support. Proximity in
the joint observation consequently cannot discard their distinct
attachment identities; an unordered pair state alone is not the
generic exact dynamical quotient. The mixed-state and boundary-current
owners retain the relevant internal and environmental information.
Recognition of a stable finite-window pattern and sufficiency of its
proposed collective evolution remain separate obligations.

### 25.5. A projection of the existing window, not another forecast

`SinePairingWindowAssessment.joint_observation()` in the
[shared scale owner](../src/tnfr/physics/relational_sine_scale.py) returns
`SineJointPairingProjection` from the existing capture, error box and
declared window. It combines the actual form-speed tube with the
already computed phase chord bounds; it neither rereads an evolved
graph nor changes the original phase-only result. The source
`source_joint_distance_rate_bounds` consume both captured fine rows.

`initial_observation`, `initial_box_candidate_pairs` and `candidate_pairs`
refer respectively to the exact center, the whole initial box and the
whole reported window. `same_pairing_as_initial_box` compares the last
two only when both are certified complete pairings. Equality there
does not establish persistence across an unobserved intervening gap;
the stronger result in Section 25.3 has its own analytic proof.
`support_admission_status` retains the existing strict replica
admission and must not be read as an exhaustive symmetry test. The
[independent algebra and rational controls](../tests/physics/test_relational_joint_grouping.py)
check all competing nodes, the full-row reversal, the source support
obstruction and the fixed-box bound without evaluating a trajectory.

<a id="sine-joint-boundary-acquisition"></a>
## 26. Acquiring a joint pairing at its exact observation boundary

### 26.1. A separately declared critical preparation

Keep the unit doubled-C5 support of Sections 16-17, its ten labeled nodes,
and the complete conservative normalized-sine law with held
`e=0`, `w=beta=nu_i=1`. There are no inputs, events or clipping. Every fine
node has degree four, and the full rows are

\[
\dot x_i=\frac{S_i}{4\pi},\qquad
\dot\theta_i=\frac{(Lx)_i}{4\pi},\qquad
S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i).
\]

Use the same phases and the same joint observation `J` as Section 25, but
declare a **new mathematical preparation**, chosen by an exact observation
boundary rather than a search over responses:

\[
d=\frac18,\quad F(s)=2(1-\cos s),\quad
u_c=\frac12\sqrt{F(2d)-F(d)}>0,
\]
\[
\theta^*=(0,d,2d,3d,1,1,2,2,3,3),\qquad
x^\sigma=\sigma(u_c,-u_c,u_c,-u_c,0,0,0,0,0,0),
\quad\sigma\in\{+1,-1\}.
\]

The positive square root exists because `F` is strictly increasing on
`(0,pi)`. This source is supplied, not autonomously selected. It does not
replace the earlier `u=1/8` source, independent error boxes, fixed horizon
or evaluated evidence. In particular its exact trigonometric form amplitude
is not a rounded graph attribute.

At either preparation, the five distances

\[
J_{01}=J_{12}=J_{23}=J_{02}=J_{13}=J_*:=F(2d)
\]

are exactly equal, because `4*u_c^2=F(2d)-F(d)`. The complete initial
nearest sets are

\[
\{1,2\},\quad\{0,2,3\},\quad\{0,1,3\},\quad\{1,2\},
\quad\{5\},\{4\},\{7\},\{6\},\{9\},\{8\}.
\]

All other comparisons have a strict gap. For `(0,3)` the excess over
`J_*` is `F(3d)-F(d)`. Between a first-four node and a last-six node the
phase gap is at least `5d`, so the excess is at least `F(5d)-F(2d)`.
The last three synchronized pairs have distance zero; their outsiders
have phase gap at least `5d`. The chord inequalities from Section 25 give

\[
F(3d)-F(d)\ge\frac{13}{5}d^2,\qquad
F(5d)-F(2d)\ge6d^2,\qquad F(5d)\ge10d^2.
\]

These are uniform positive margins for every initially untied competitor.
Consequently the only local ordering question is how the five tied edges
separate under the complete law.

### 26.2. Both rows determine the crossing direction

At the positive preparation `Lx=4x`, so `theta_dot=x/pi`. Write the first
four sine sums as

\[
\begin{aligned}
S_0&=\sin2d+\sin3d+2\sin3,\\
S_1&=\sin d+\sin2d+2\sin(3-d),\\
S_2&=-\sin2d-\sin d+2\sin(1-2d),\\
S_3&=-\sin3d-\sin2d+2\sin(1-3d).
\end{aligned}
\]

Apply the full joint-distance derivative from Section 25:

\[
\dot J_{ij}=2(x_j-x_i)(\dot x_j-\dot x_i)
 +2\sin(\theta_j-\theta_i)(\dot\theta_j-\dot\theta_i).
\]

Define the three dimensionless coefficients

\[
\begin{aligned}
A&=5\sin d-\sin3d+2[\sin(3-d)-\sin3],\\
B&=2\sin d-2\sin2d+2[\sin(1-2d)-\sin(3-d)],\\
C&=5\sin d-\sin3d+2[\sin(1-3d)-\sin(1-2d)].
\end{aligned}
\]

The tied-edge rates at the positive source are exactly

\[
\boxed{\frac\pi{u_c}
 (\dot J_{01},\dot J_{12},\dot J_{23},\dot J_{02},\dot J_{13})
 =(-A,B,-C,0,0).}
\]

All three coefficients are strictly positive, by elementary inequalities:

* The triple-angle identity gives
  `5*sin(d)-sin(3d)=2*sin(d)+4*sin(d)^3>0`.
  Since `pi/2<3-d<3<pi`, sine decreases between `3-d` and `3`;
  the additional difference in `A` is positive.
* The same identity and the sine-difference formula give

  \[
  C=4\sin(d/2)[\cos(d/2)-\cos(1-5d/2)]+4\sin^3d>0.
  \]

  Here `d/2=1/16<1-5d/2=11/16<pi`, so both displayed terms are
  positive. This retains the form/phase cancellation in the smallest
  tied-edge response; the phase row alone cannot justify its sign.
* `sin(3/4)>=3/4-(3/4)^3/6=87/128>2/3`.
  Also `0<pi-23/8<1/3`, using `3<pi<22/7`, and hence
  `sin(23/8)=sin(pi-23/8)<1/3`.
  Finally `sin(2d)-sin(d)<=d` by the derivative bound for sine. Therefore
  `B>2*(2/3-1/3-d)=5/12>0`.

No ordering between `A`, `B` and `C` is needed. Reversing all forms leaves
the form row unchanged and negates the phase row. Both terms of every
`J_dot` therefore reverse, while every initial `J` stays unchanged.

### 26.3. One-sided acquisition and qualitative full-state robustness

The field and the joint distances are smooth. A tied margin with a strictly
positive derivative becomes positive on a sufficiently short positive-time
interval. There are finitely many comparisons, and the initially untied
ones have the positive margins just proved. Thus some `tau>0` exists such
that, for **every** `0<t<tau`, the positive source has nearest map

\[
\boxed{(1,0,3,2,5,4,7,6,9,8),}
\]

whereas the negative source has nearest map

\[
\boxed{(2,2,1,1,5,4,7,6,9,8).}
\]

For the positive source, edges `01` and `23` decrease while `12` increases
and `02,13` have zero first derivative. This resolves every tied choice in
favor of the original structural pairs. For the reversed source, `12`
decreases while `01,23` increase; nodes 0 and 3 select the zero-rate
alternatives 2 and 1. This map is not a complete mutual pairing. The last
three pairs remain the unique nearest choices by their initial strict gaps.

The exact conservative reversal `(x(t),theta(t)) ->
(-x(-t),theta(-t))` also shows that the positive orbit has the nonmutual
map on `-tau<t<0`. It crosses from that region, through the declared ties,
into the structural mutual-pairing region. This is local acquisition of the
specified joint observation under the complete autonomous law, not merely
a phase-distance transition or a choice among initially unique joint pairs.

The assertion has qualitative robustness in **all twenty fine coordinates**.
For any compact interval `[a,b]` with `0<a<b<tau`, all desired margins
along either reference are uniformly positive. Continuous dependence of the
full flow yields an open initial-state neighborhood preserving its respective
nearest map throughout `[a,b]`. There is no restriction to synchronized
perturbations, pair sums or the original one-dimensional form family.
The exact simultaneous ties at time zero need not survive a perturbation.

More strongly, choose any small positive `a<tau` on the positive reference.
Its state at `-a` has a strict nonmutual map and its state at `+a` has the
strict structural map. Both strict endpoint observations persist for an
open neighborhood of that earlier state, by the same continuous-dependence
argument. Thus nearby full states also undergo a change between these
observation regions, although their individual tie-crossing times and order
need not coincide. These are existential intervals and neighborhoods;
none is a certified numerical horizon, error radius or permanent lifetime.

### 26.4. What is acquired, and what remains supplied

The positive partition passes the strict replica support contract: its
pairs have no internal edges, neighboring blocks have complete unit
bipartite support, capacities are common, and every within-pair interchange
is a graph automorphism. Their initial absolute within-pair phase half-gaps
are `d/2,d/2,0,0,0`, all strictly within the local midpoint chart; `tau` may
be shortened to keep that admission throughout the crossing. The exact
collective-state result of Sections 7 and 18-19 therefore applies with its
retained internal form/phase variables.
The first two pairs are not synchronized at preparation, so their newly
strict observation does not authorize discarding those internal variables.
The negative map has no complete mutual pairing to admit as such a partition.

The support symmetry and its sufficient quotient existed before the crossing.
The dynamics makes that partition strictly recognizable by the supplied
joint observation; it does not create the support, the quotient degrees of
freedom, an operator-selection law or a physical constituent. No invariant
prepared family has been entered from outside, and no attraction or permanent
maintenance is asserted. The result is consequently compatible with the
conservative formation boundary stated at the beginning of this note.

No new execution law or observer is necessary. The existing joint observation
and full-row derivative owners already retain the required information.
Their represented-real admission must not relabel a rounded `u_c` as the
exact source, infer an equality from overlapping intervals, or promote this
local theorem to a finite-window certificate. Any such numerical claim would
require its own prospectively declared enclosure and budget. The
[independent symbolic controls](../tests/physics/test_relational_joint_boundary.py)
check the exact critical relation, complete fine-row coefficients and every
tied ordering; the proof above supplies their inequality and local-flow scope.
The [outward full-field controls](../tests/physics/test_relational_joint_boundary_bounds.py)
separately bound the symbolic amplitude, all initially untied comparisons
and the complete joint rates through shared rational kernels. Their rounding
control demonstrates why a represented approximation is not the exact tie.

<a id="sine-joint-recurrent-episodes"></a>
## 27. Repeated finite episodes of joint identification

This is a consequence of the local crossing in Section 26 and the existing
[full-state invariant-volume recurrence theorem](nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence).
It uses the **same** complete conservative normalized-sine law: fixed unit
doubled-C5 support, `e=0`, held `nu_i=w=beta=1`, structural time, and no
forcing, events, clipping or changed mobility. All twenty fine coordinates
are retained, with phases on the torus. No trajectory is evaluated and no
recurrence assertion is inferred from a finite numerical return.

### 27.1. Identity, loss and a common acquisition interval

Let `Phi_t` denote the complete flow on
`M=R^10 x (R/(2*pi*Z))^10`. Define two open subsets of that state space:

* `P` consists of states whose strict joint nearest map is the structural
  matching `(1,0,3,2,5,4,7,6,9,8)`.
* `N` consists of states whose strict joint nearest map is
  `(2,2,1,1,5,4,7,6,9,8)`, the nonmutual map in Section 26.

Both predicates use exactly the joint distances `J_ij` already defined.
Each is a finite conjunction of strict continuous inequalities
`J_i,p(i)<J_ij` for all other candidates `j`. Hence `P` and `N` are open
and disjoint. **Identity** here means membership in `P`; its loss means
nonmembership. Visiting `N` establishes a stronger, strict alternative
observation, rather than an unavailable or inconclusive numerical report.

Write `z_c` for the positive exact critical preparation. Section 26 gives
`tau>0` such that `Phi_t(z_c)` lies in `N` for `-tau<t<0` and in `P` for
`0<t<tau`, with the local pair midpoint chart admitted on these intervals.
Choose `a>0` sufficiently small, with `a<tau/4`, and set

\[
z_-:=\Phi_{-a}(z_c).
\]

Then `z_-` lies strictly in `N`, while its entire image over the positive
interval `[2a,3a]` lies strictly in `P`. Compactness of that time interval
and continuous dependence on all fine coordinates give an open neighborhood
`U` of `z_-` with compact closure and a bounded open matching neighborhood
`V` such that

\[
\boxed{\overline U\subset N,\qquad
\Phi_s(\overline U)\subset V\subset P
\quad\text{for every }s\in[2a,3a].}
\]

The neighborhoods can be chosen inside the admitted pair chart during
these initial and matching windows. The compact image tube has a common
strictly positive nearest-partner margin, although no numerical value is
claimed. The same `a,U,V` work for every preparation in `U`, including
independent perturbations of all twenty coordinates. Every such state is
nonstationary already, since a stationary state cannot move between the
disjoint sets `N` and `P`.

The midpoint chart is an additional collective-coordinate admission on
these windows, **not** part of the identity predicate `P`. A longer matching
episode may leave that chart without losing its joint-distance matching.

### 27.2. One finite invariant ambient family

The recurrence theorem needs a finite invariant measure, not just a local
crossing. Its hypotheses can be admitted here with an explicit loose slab.
There are twenty unit edges, and every fine degree is four. The conserved
storage and weighted mean therefore take the forms

\[
E=\frac12\sum_{\{i,j\}\in\mathcal E}(x_i-x_j)^2
  +\sum_{\{i,j\}\in\mathcal E}[1-\cos(\theta_j-\theta_i)],
\qquad m=\frac1{10}\sum_i x_i.
\]

For the closed form box `|x_i|<=1/4`, with **arbitrary circular phases**,

\[
E\le\frac12\,20\left(\frac12\right)^2+2\,20
 =\frac{85}{2}<43,\qquad |m|\le\frac14<\frac12.
\]

Thus the whole box times the phase torus lies strictly inside

\[
\mathcal S=\{z:E(z)\le43,\ -\tfrac12\le m(z)\le\tfrac12\}.
\]

At the critical source, `u_c<d=1/8`, since
`4*u_c^2=F(2d)-F(d)<4d^2`. By choosing `a` smaller if necessary, then
shrinking `U`, its compact closure lies in `|x_i|<1/4`. Therefore
`overline(U)` lies in the interior of `S`. The form box itself need not be
invariant: `S` is the invariant family. These numerical ceilings are
analysis bounds on supplied preparations, not new parameters of the law.

The source theorem applies without a change of model. This is a finite
connected simple unit graph with positive held capacities; both complete
rows use the same constant degree/capacity mobility. Storage and the
weighted form mean are conserved. The full divergence vanishes because
the form row depends only on phase and the phase row only on form.
Consequently `S` is compact, the flow exists in both time directions, and
it preserves finite positive product measure

\[
d\mu=dx_0\cdots dx_9\,d\mathrm{Haar}_{\mathbb T^{10}}.
\]

No bounded interval of unwrapped phase lifts or unproved measure on a
fixed-energy surface is being introduced. Since `U` is full-dimensional
and open with compact closure, `0<mu(U)<infinity`.

### 27.3. Infinitely many distinct finite matching episodes

Fix the single sampling increment `h=4a` and write `T=Phi_h`. First take
**any** `z` in `U` that returns to `U` infinitely often under `T`. This is
an explicit deterministic premise: there are integers

\[
0=n_0<n_1<n_2<\cdots,\qquad n_k\longrightarrow\infty,
\qquad \Phi_{n_kh}(z)\in U.
\]

Put `r_k=n_kh`. Each return is a state of the **same** autonomous system;
it is not a reset or a fresh draw of initial data. The common transit gives

\[
\Phi_{r_k}(z)\in N,\qquad
\Phi_t(z)\in V\subset P\quad
\text{for every }t\in[r_k+2a,r_k+3a].
\]

Also `r_(k+1)-r_k>=h=4a>3a`. Hence each guaranteed matching window lies
strictly between two nonmutual returns. Define the matching-time set

\[
\mathcal I_z=\{t>0:\Phi_t(z)\in P\}.
\]

It is open. Let `I_k` be the connected component containing
`[r_k+2a,r_k+3a]`. Since neither bounding return lies in `P`,

\[
\boxed{
[r_k+2a,r_k+3a]\subset I_k\subset(r_k,r_{k+1}).
}
\]

These components are pairwise distinct, bounded intervals, each of
duration at least `a>0`. There are infinitely many of them. Every return
to the open set `N` also has a nonempty time neighborhood in `N`, so the
selected episodes are separated by actual intervals of strict alternative
identification, not merely an unresolved tie at one instant.

More generally **every** positive-time component of `I_z` is bounded:
an unbounded component would contain all sufficiently late times and
contradict the unbounded sequence of returns to `N`. Only the selected
infinite subfamily has the common lower duration bound `a`; no such bound
is asserted for other episodes. This argument supplies no upper lifetime
bound common to preparations, bound on the next return or exact period.
The constants `a,U,V` and the recurrence times have existence proofs here,
not numerical certificates.

Everything so far follows for each state satisfying the stated return
premise. The existing finite-measure recurrence theorem, applied to the
fixed map `T` and measurable subset `U` of `S`, supplies that premise for
`mu`-almost every state in `U`.

Thus, for almost every preparation in this nonempty open full-state
acquisition neighborhood, the structural joint observation repeatedly
appears, persists for a positive interval, is lost, and reappears. It does
not eventually become permanent for those preparations. This is a scoped
lifetime result for a noninvariant observation; it neither contradicts nor
enters the separately protected two-sided invariant identity families.

### 27.4. Measure, interpretation and integration boundaries

The exceptional set is null in the stated ambient twenty-dimensional
product measure. An initial probability law supported in `U` and absolutely
continuous with respect to that measure inherits the result with probability
one. The dynamics has not selected such a preparation law. No pointwise
recurrence conclusion follows for the exact critical source, its earlier
point `z_-`, an individual captured graph, a fixed-energy or fixed-mean
preparation, or a synchronized or finitely sampled family solely from
their inclusion in `S` or `U`.

The recurring identity is the **same specified matching**, not a proof that
nodes, edges, quotient coordinates or physical constituents are repeatedly
created and destroyed. Its support and sufficient state remain supplied;
only the joint observation changes. The conclusion does not establish a
common pulse, periodic waveform, attraction, dissipation or a clock selected
by synchronization. It does not transfer to the native argument-pressure
law, positive-loss sine dynamics or another reciprocal mobility without
their own measure and lifetime premises.

No additional runtime or report is needed. The existing recurrence reader
admits a family and explicitly leaves individual nonstationary recurrence
unavailable; the joint observer reports a captured state's distances rather
than unknown future return times. The
[recurrence controls](../tests/physics/test_relational_sine_resonance.py)
check complete-law invariants, finite-family admission and that chosen-state
boundary. The [ambient-family integration control](../tests/physics/test_relational_joint_recurrence.py)
independently checks the declared box/slab and existing family API without
asserting a numerical recurrence sequence. The
[joint crossing controls](../tests/physics/test_relational_joint_boundary.py)
and [outward bounds](../tests/physics/test_relational_joint_boundary_bounds.py)
check the local mechanism reused here. Finite tests validate these premises
and their implementation, not the infinite recurrence sequence itself.

<a id="sine-zero-resultant-restoration"></a>
## 28. Zero pair resultant and autonomous restoration of form current

### 28.1. A global current observation without a midpoint angle

Keep the complete conservative normalized-sine law and unit doubled-C5
support of Sections 26-27, with `e=0`, held `nu_i=w=beta=1`, structural
time and no input, clipping or event. The five structural pairs are
`a={2a,2a+1}`, for `a=0,...,4`, with base indices taken modulo five.
Each fine degree is four. Retain all fine real forms and circular phases,
and define the observations

\[
X_a=\frac{x_{2a}+x_{2a+1}}2,\qquad
Z_a=\frac{e^{i\theta_{2a}}+e^{i\theta_{2a+1}}}2.
\]

`Z_a` is a globally defined mean phase phasor, including at zero. It is
neither complex EPI nor a new constitutive state variable. Its argument
is unavailable when `Z_a=0`, which for a pair means exactly antipodal
primitive phases. This does not remove or make either primitive phase
undefined, and the complete sine field has no singularity there.

For adjacent base pairs `a,b`, use the existing boundary balance to define
the contribution to the **mean form rate** in `a` from `b`:

\[
I_{b\to a}:=\frac1{8\pi}
 \sum_{i\in a}\sum_{j\in b}\sin(\theta_j-\theta_i)
 =\boxed{\frac1{2\pi}\operatorname{Im}(\overline Z_a Z_b)}.
\]

The equality follows by factoring the four fine phasors. It is the
globally regular expression of the inherited form row from Section 7,
not another coupling law. In particular

\[
\dot X_a=I_{a-1\to a}+I_{a+1\to a},\qquad
I_{b\to a}=-I_{a\to b},\qquad \sum_a\dot X_a=0.
\]

The weights in Section 23 are `rho_i=d_i/nu_i=4`. Thus the weighted pair
form is `M_a=8X_a`, and its contribution from `b` is `8I_(b->a)`.
This normalization distinguishes a mean-form contribution from the
full weighted cut current; it preserves the existing regional ledger.

If `Z_a=0`, every incident block-form contribution is zero, independently
of neighboring phases. The individual fine edge currents need not vanish:
opposite members can cancel in the block sum. Conversely a zero block
current need not mean a zero resultant; aligned nonzero phasors can also
give a zero imaginary product. Current cancellation is not a nearest-pair
criterion, missing support or full dynamical decoupling.

In particular the nearest-partner inequalities from
[Section 26](#sine-joint-boundary-acquisition) never enter these analytic
current expressions. Crossing one of their equality boundaries cannot
install a causal switch or a discontinuity in the complete field.

### 28.2. The retained phase channel and internal form information

The complete primitive phase row is still

\[
\dot\theta_i=\frac1{4\pi}\sum_{j\sim i}(x_i-x_j).
\]

Its average rate within a pair is the globally meaningful scalar

\[
\boxed{\Omega_a:=\frac{\dot\theta_{2a}+\dot\theta_{2a+1}}2
 =\frac1\pi\left(X_a-\frac{X_{a-1}+X_{a+1}}2\right).}
\]

Rates of primitive angles agree between local lifts that differ by
constant full turns, so this average rate does not require a global
midpoint angle. In particular it must not be called `d(arg Z_a)/dt`
at `Z_a=0`. The form-to-phase susceptibility to either neighboring
block remains `partial Omega_a/partial X_b=-1/(2*pi)` there.
Replacing that row or the graph degree by a factor proportional to
`|Z_a Z_b|` would change the declared complete law.

One useful derivative identity keeps the missing internal information
visible. Define the derived form-phase moment

\[
Y_a=\frac{x_{2a}e^{i\theta_{2a}}
             +x_{2a+1}e^{i\theta_{2a+1}}}2.
\]

Differentiating the phasors with the full primitive phase row gives

\[
\boxed{\dot Z_a=\frac i\pi\left[
Y_a-\frac{X_{a-1}+X_{a+1}}2 Z_a\right].}
\]

This identity is regular at zero resultant. It is an observation of the
retained fine state, not an assertion that `(X,Z,Y)` closes autonomously
or a replacement for the existing sufficient-state theorem. In particular
`Z_a=0` does not force `Z_dot_a=0`: internal form can make `Y_a` nonzero.
This is the zero-resultant instance of the internal-state obligation
already identified in Section 7.

### 28.3. One frozen antipodal preparation and its complete response

Let `A` be structural pair `(0,1)` and declare

\[
(\theta_0,\theta_1)=(0,\pi),\quad
(x_0,x_1)=(u,-u),\quad u\in\{\tfrac18,-\tfrac18,0\},
\]

with all other forms and phases zero. The half-turn is exact mathematical
phase data. A rounded radian approximation to `pi` cannot substitute for
it when asserting exact antipodality, cancellation or equilibrium.
This supplied preparation is separate from the critical-nearest-boundary
source in Section 26 and from all earlier frozen response protocols.

Initially every pair mean form is zero, `Z_A=0`, and all other pair
phasors equal one. All block currents vanish. In this particular witness
every fine sine current also vanishes, because each fine phase difference
is zero or an exact half-turn; the general cancellation identity does
not require that additional property. The fine form and phase rows give

\[
\dot x(0)=0,\qquad
\dot\theta(0)=\frac u\pi(1,-1,0,0,0,0,0,0,0,0),
\qquad \ddot\theta(0)=0.
\]

Thus zero instantaneous form pressure and zero mean phase rate do not
make the nonzero-`u` state an equilibrium. The phase row moves the two
members in opposite directions. Since `Y_A(0)=u`,

\[
\boxed{\dot Z_A(0)=\frac{iu}\pi,\qquad
\dot Z_b(0)=0\quad(b\ne A).}
\]

For completeness, differentiating every fine form row gives

\[
\ddot x_i(0)=\frac1{4\pi}
\sum_{j\sim i}\cos(\theta_j-\theta_i)
                  [\dot\theta_j-\dot\theta_i]\big|_{t=0},
\]
\[
\boxed{\ddot x(0)=\frac u{\pi^2}
(-1,-1,\tfrac12,\tfrac12,0,0,0,0,\tfrac12,\tfrac12).}
\]

The two neighbors of `A` are the blocks `(2,3)` and `(8,9)`. Their
initial collective accelerations are consequently

\[
\boxed{\ddot X_A(0)=-\frac u{\pi^2},\qquad
\ddot X_1(0)=\ddot X_4(0)=\frac u{2\pi^2},\qquad
\ddot X_2(0)=\ddot X_3(0)=0.}
\]

Equivalently, each incident current satisfies

\[
\dot I_{b\to A}(0)
 =\frac1{2\pi}\operatorname{Im}
       (\overline{\dot Z_A(0)}Z_b(0))
 =-\frac u{2\pi^2},\qquad b\in\{1,4\}.
\]

The opposite current is its negative. The leading collective gains in
the two neighbors exactly compensate the change in `A`; this is also
the all-time cancellation in Section 28.1, not just a Taylor residual.
The full storage is `E=8+4u^2` initially and is conserved. The two nonzero
orientations therefore have equal storage and opposite leading transfers.
There is no supplied AL increment, input, event reserve or new edge.

For `u=0`, both complete fine rows vanish exactly at this preparation.
Uniqueness makes it stationary and its currents remain zero. This control
does not assert that every zero-resultant state with initially zero
internal form is stationary; its full surrounding phase and form state
matters. All three preparations share their initial `(X,Z)` observations
and block currents, while their full internal form states differ.

### 28.4. Immediate restoration, a magnitude cusp and exact silence

For either nonzero frozen orientation, the nonzero derivative proves
that on some sufficiently small punctured interval around zero,

\[
Z_A(t)=\frac{iu}\pi t+O(t^2)\ne0,\qquad
I_{b\to A}(t)=-\frac u{2\pi^2}t+O(t^2)\ne0.
\]

For positive time, both neighboring mean-form currents point away from
`A` when `u>0` and toward it when `u<0`. In particular

\[
X_A(t)=-\frac u{2\pi^2}t^2+O(t^3),\qquad
X_1(t)=X_4(t)=\frac u{4\pi^2}t^2+O(t^3).
\]

The equality between the last two full functions also follows from the
reflection symmetry of the support and this supplied preparation.
Only the local signed response is asserted; there is no numerical
duration, permanent current or selected oscillation period here.

The magnitude behaves differently from the analytic complex phasor:

\[
|Z_A(t)|=\frac{|u|}\pi|t|+O(t^2).
\]

It is not two-sided differentiable at zero. The limiting phasor directions
on the two sides differ by a half-turn, and `arg Z_A(0)` is undefined.
Neither fact is a singularity or jump of the fine state or its field.
One may continue `cos(delta)` as a signed quantity on chosen real lifts,
but may not thereby extend its interpretation as the nonnegative
midpoint-chart magnitude through the antipodal boundary. No new global
quotient or replacement midpoint angle has been supplied here.

There is a stronger limit on the word "restoration." On this unchanged
law the finite-dimensional field is real analytic in real form and local
phase lifts. Its solutions are real analytic in time. The global phasor
products and currents are periodic analytic functions of those coordinates,
so each `I_(b->a)(t)` is real analytic on the connected trajectory interval.
The identity theorem then gives

\[
\boxed{I_{b\to a}(t)=0\text{ on a nonempty open time interval}
\ \Longrightarrow\ I_{b\to a}(t)\equiv0
\text{ on the same connected trajectory interval}.}
\]

The present full sine flow is
[globally continuable](nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence),
so the interval can be the entire real time axis. A nonidentically-zero
current instead has isolated zeros, with no accumulation at a finite time.
Thus the witness crosses an isolated instant of exact cancellation; it
does not wait with an exactly silent current for a finite interval and
later switch it on.
Thresholded observations, changed inputs or a declared hybrid event are
different questions. The analytic statement requires this complete law
and its unchanged analytic evolution, not merely smoothness.

The mechanism is therefore an actual, compensated change in collective
form transport caused by retained internal form and evolving primitive
phase. It is more than a relabeling by a nearest-pair rule, but it is
still neither creation of the full coupling channel nor birth of support
or a maintained physical constituent. The phase row and all primitive
edges remain present throughout.

### 28.5. Existing source owners and exact-phase scope

The [complete sine comparison](../src/tnfr/physics/relational_sine_comparison.py)
already provides the needed detached observations.
`regional_transfer(region=...)` reports the weighted cut current, which is
`8*X_dot_a` for these pairs, rather than the block mean rate itself.
`resultant_kinematics()` differentiates the **node-relative neighbor**
resultant `z_i=sum_(j~i) exp(i*(theta_j-theta_i))`. That is distinct from
the pair phasor `Z_a`. Since `Im(z_i)=S_i`, its imaginary derivative gives
`x_ddot_i=Im(z_dot_i)/(4*pi)` in this fixed conservative unit-capacity
law. Neither observation divides by a resultant or needs a derived angle.

The [regional-transfer controls](../tests/physics/test_relational_regional_transfer.py)
check the phasor/cut identity, all complete fine jets, retained phase
susceptibility and compensated response. Exact symbolic half-turn controls
establish the stated cancellation and stationary preparation. Separately,
represented phases on either side of mathematical `pi` are evaluated at
their actual captured values: their small nonzero currents remain visible,
and the zero-form case is not mislabeled as an exact equilibrium.
The shared reader's interval evidence and the exact-phase theorem therefore
keep distinct provenance. No new report, runtime law, source search or
global quotient is required for this result.
