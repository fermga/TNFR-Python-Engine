# Coarse-graining, coherence geometry and bridge results

Sections 1–6 establish restricted coarse reductions, field geometry and
inverse/endpoint contracts. Each result retains its declared pressure,
support, capacities and clock; none supplies closure of all 13 operators
or a coupled global TNFR geometry. The nonlinear chapters distinguish the
normalized-sine law from alternative reciprocal mobilities. Their shared
storage balance does not make their complete dynamics interchangeable.

<a id="read-by-mathematical-question"></a>

## Chapters and scope

| Chapter | Responsibility |
| --- | --- |
| [Sufficient unordered pair state](nodal/SINE_PAIR_STATE.md) | Realizability, temporal reconstruction, capacity correlations and support symmetry, including phase cancellation. |
| [Internal replica pulses and their perturbations](nodal/SINE_REPLICA_PULSE.md) | Prepared pulses, full variation, transverse splitting and persistent constituent activity. |
| [Phase and joint form-phase grouping](nodal/SINE_PAIR_GROUPING.md) | Equilibrium geometry, pair observations and finite grouping windows; observation does not create support. |
| [Pair geometry under alternative reciprocal mobility](nodal/SINE_PAIR_MOBILITY.md) | Grouping and protected geometry under a separately declared complete law. |
| [Pair actions, regional exchange and cancellation](nodal/SINE_PAIR_INTERACTION.md) | Form events, autonomous currents, cancellation and the retained interaction interface. |

Section numbers remain stable. The
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone assigns research work.

Formation, observation and maintenance are distinct. Protection of a
[two-sided invariant family](nodal/RESONANCE_FOUNDATIONS.md#conservative-formation-boundary)
does not prove entry into it. The separate
[validated native-law transit](nodal/RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit)
retains its own dissipative law, preparation and capture proof.

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

## Section link directory

These aliases route existing citations to their substantive owner.

- <a id="exact-scale-coherence-geometry-and-bridge-results"></a>[Exact scale, coherence-geometry and bridge results](#exact-scale-coherence-geometry-and-bridge-results)

- <a id="sine-replica-inheritance"></a>[sine-replica-inheritance](nodal/SINE_PAIR_STATE.md#sine-replica-inheritance)

- <a id="7-nonlinear-sine-inheritance-with-retained-internal-coordinates"></a>[7. Nonlinear sine inheritance with retained internal coordinates](nodal/SINE_PAIR_STATE.md#7-nonlinear-sine-inheritance-with-retained-internal-coordinates)

- <a id="71-supplied-support-and-an-invertible-description"></a>[7.1. Supplied support and an invertible description](nodal/SINE_PAIR_STATE.md#71-supplied-support-and-an-invertible-description)

- <a id="72-internal-phase-geometry-modulates-inherited-coupling"></a>[7.2. Internal phase geometry modulates inherited coupling](nodal/SINE_PAIR_STATE.md#72-internal-phase-geometry-modulates-inherited-coupling)

- <a id="73-exact-synchronized-inheritance-and-storage-scaling"></a>[7.3. Exact synchronized inheritance and storage scaling](nodal/SINE_PAIR_STATE.md#73-exact-synchronized-inheritance-and-storage-scaling)

- <a id="74-a-nearby-exact-obstruction-to-autonomous-block-means"></a>[7.4. A nearby exact obstruction to autonomous block means](nodal/SINE_PAIR_STATE.md#74-a-nearby-exact-obstruction-to-autonomous-block-means)

- <a id="75-trapping-permits-persistent-internal-organization"></a>[7.5. Trapping permits persistent internal organization](nodal/SINE_PAIR_STATE.md#75-trapping-permits-persistent-internal-organization)

- <a id="76-meaning-and-limits-of-the-scale-result"></a>[7.6. Meaning and limits of the scale result](nodal/SINE_PAIR_STATE.md#76-meaning-and-limits-of-the-scale-result)

- <a id="sine-replica-unordered-state"></a>[sine-replica-unordered-state](nodal/SINE_PAIR_STATE.md#sine-replica-unordered-state)

- <a id="8-an-exact-collective-state-for-unordered-pairs"></a>[8. An exact collective state for unordered pairs](nodal/SINE_PAIR_STATE.md#8-an-exact-collective-state-for-unordered-pairs)

- <a id="81-realizable-invariants-and-their-fibers"></a>[8.1. Realizable invariants and their fibers](nodal/SINE_PAIR_STATE.md#81-realizable-invariants-and-their-fibers)

- <a id="82-closed-dynamics-with-internal-state-retained"></a>[8.2. Closed dynamics with internal state retained](nodal/SINE_PAIR_STATE.md#82-closed-dynamics-with-internal-state-retained)

- <a id="83-evolution-boundaries-and-lift-independence"></a>[8.3. Evolution, boundaries and lift independence](nodal/SINE_PAIR_STATE.md#83-evolution-boundaries-and-lift-independence)

- <a id="84-storage-and-identity-pass-to-the-exact-quotient"></a>[8.4. Storage and identity pass to the exact quotient](nodal/SINE_PAIR_STATE.md#84-storage-and-identity-pass-to-the-exact-quotient)

- <a id="85-what-has-and-has-not-been-reduced"></a>[8.5. What has and has not been reduced](nodal/SINE_PAIR_STATE.md#85-what-has-and-has-not-been-reduced)

- <a id="sine-replica-inherited-poisson"></a>[sine-replica-inherited-poisson](nodal/SINE_PAIR_STATE.md#sine-replica-inherited-poisson)

- <a id="86-the-collective-state-inherits-the-same-poisson-structure"></a>[8.6. The collective state inherits the same Poisson structure](nodal/SINE_PAIR_STATE.md#86-the-collective-state-inherits-the-same-poisson-structure)

- <a id="sine-replica-internal-observability"></a>[sine-replica-internal-observability](nodal/SINE_PAIR_STATE.md#sine-replica-internal-observability)

- <a id="87-temporal-resultant-information-recovers-the-unordered-internal-state"></a>[8.7. Temporal resultant information recovers the unordered internal state](nodal/SINE_PAIR_STATE.md#87-temporal-resultant-information-recovers-the-unordered-internal-state)

- <a id="sine-replica-internal-pulse"></a>[sine-replica-internal-pulse](nodal/SINE_REPLICA_PULSE.md#sine-replica-internal-pulse)

- <a id="9-a-finite-amplitude-internal-pulse-with-persistent-collective-identity"></a>[9. A finite-amplitude internal pulse with persistent collective identity](nodal/SINE_REPLICA_PULSE.md#9-a-finite-amplitude-internal-pulse-with-persistent-collective-identity)

- <a id="91-the-exact-invariant-preparation"></a>[9.1. The exact invariant preparation](nodal/SINE_REPLICA_PULSE.md#91-the-exact-invariant-preparation)

- <a id="92-conserved-amplitude-and-exact-periods"></a>[9.2. Conserved amplitude and exact periods](nodal/SINE_REPLICA_PULSE.md#92-conserved-amplitude-and-exact-periods)

- <a id="93-collective-identity-and-the-stronger-fine-edge-condition"></a>[9.3. Collective identity and the stronger fine-edge condition](nodal/SINE_REPLICA_PULSE.md#93-collective-identity-and-the-stronger-fine-edge-condition)

- <a id="94-boundaries-and-scientific-scope"></a>[9.4. Boundaries and scientific scope](nodal/SINE_REPLICA_PULSE.md#94-boundaries-and-scientific-scope)

- <a id="sine-replica-pulse-variation"></a>[sine-replica-pulse-variation](nodal/SINE_REPLICA_PULSE.md#sine-replica-pulse-variation)

- <a id="10-complete-variation-around-the-finite-internal-pulse"></a>[10. Complete variation around the finite internal pulse](nodal/SINE_REPLICA_PULSE.md#10-complete-variation-around-the-finite-internal-pulse)

- <a id="101-direct-differentiation-of-the-full-retained-law"></a>[10.1. Direct differentiation of the full retained law](nodal/SINE_REPLICA_PULSE.md#101-direct-differentiation-of-the-full-retained-law)

- <a id="102-exact-spatial-decomposition-without-discarding-directions"></a>[10.2. Exact spatial decomposition without discarding directions](nodal/SINE_REPLICA_PULSE.md#102-exact-spatial-decomposition-without-discarding-directions)

- <a id="103-zero-amplitude-control-and-dimensionless-parameters"></a>[10.3. Zero-amplitude control and dimensionless parameters](nodal/SINE_REPLICA_PULSE.md#103-zero-amplitude-control-and-dimensionless-parameters)

- <a id="104-symplectic-blocks-and-the-correct-half-period-return"></a>[10.4. Symplectic blocks and the correct half-period return](nodal/SINE_REPLICA_PULSE.md#104-symplectic-blocks-and-the-correct-half-period-return)

- <a id="105-amplitude-dependent-phase-shear-is-a-neutral-family-effect"></a>[10.5. Amplitude-dependent phase shear is a neutral-family effect](nodal/SINE_REPLICA_PULSE.md#105-amplitude-dependent-phase-shear-is-a-neutral-family-effect)

- <a id="106-a-discriminating-stability-question-not-a-stability-verdict"></a>[10.6. A discriminating stability question, not a stability verdict](nodal/SINE_REPLICA_PULSE.md#106-a-discriminating-stability-question-not-a-stability-verdict)

- <a id="sine-moving-pulse-work-response"></a>[sine-moving-pulse-work-response](nodal/SINE_REPLICA_PULSE.md#sine-moving-pulse-work-response)

- <a id="107-internal-motion-produces-a-finite-same-type-work-response-asymmetry"></a>[10.7. Internal motion produces a finite same-type work-response asymmetry](nodal/SINE_REPLICA_PULSE.md#107-internal-motion-produces-a-finite-same-type-work-response-asymmetry)

- <a id="matched-form-inputs-and-storage-work-outputs"></a>[Matched form inputs and storage-work outputs](nodal/SINE_REPLICA_PULSE.md#matched-form-inputs-and-storage-work-outputs)

- <a id="time-ordering-supplies-the-first-nonzero-antisymmetric-term"></a>[Time ordering supplies the first nonzero antisymmetric term](nodal/SINE_REPLICA_PULSE.md#time-ordering-supplies-the-first-nonzero-antisymmetric-term)

- <a id="a-finite-positive-bound-at-the-declared-horizon"></a>[A finite positive bound at the declared horizon](nodal/SINE_REPLICA_PULSE.md#a-finite-positive-bound-at-the-declared-horizon)

- <a id="reversal-controls-and-scope"></a>[Reversal controls and scope](nodal/SINE_REPLICA_PULSE.md#reversal-controls-and-scope)

- <a id="sine-moving-pulse-finite-work-response"></a>[sine-moving-pulse-finite-work-response](nodal/SINE_REPLICA_PULSE.md#sine-moving-pulse-finite-work-response)

- <a id="108-a-controlled-finite-probe-of-the-complete-fine-law"></a>[10.8. A controlled finite probe of the complete fine law](nodal/SINE_REPLICA_PULSE.md#108-a-controlled-finite-probe-of-the-complete-fine-law)

- <a id="uniform-nonlinear-error-before-evaluating-a-finite-response"></a>[Uniform nonlinear error, before evaluating a finite response](nodal/SINE_REPLICA_PULSE.md#uniform-nonlinear-error-before-evaluating-a-finite-response)

- <a id="domain-preparation-and-interpretation"></a>[Domain, preparation and interpretation](nodal/SINE_REPLICA_PULSE.md#domain-preparation-and-interpretation)

- <a id="sine-moving-pulse-parametric-representation"></a>[sine-moving-pulse-parametric-representation](nodal/SINE_REPLICA_PULSE.md#sine-moving-pulse-parametric-representation)

- <a id="109-parametric-response-and-complete-history-reversal"></a>[10.9. Parametric response and complete-history reversal](nodal/SINE_REPLICA_PULSE.md#109-parametric-response-and-complete-history-reversal)

- <a id="sine-pulse-stiffness-discriminator"></a>[sine-pulse-stiffness-discriminator](nodal/SINE_REPLICA_PULSE.md#sine-pulse-stiffness-discriminator)

- <a id="1010-a-clock-independent-stiffness-family-discriminator"></a>[10.10. A clock-independent stiffness-family discriminator](nodal/SINE_REPLICA_PULSE.md#1010-a-clock-independent-stiffness-family-discriminator)

- <a id="exact-obstruction-for-one-affine-stiffness-control"></a>[Exact obstruction for one affine stiffness control](nodal/SINE_REPLICA_PULSE.md#exact-obstruction-for-one-affine-stiffness-control)

- <a id="sine-replica-pulse-splitting"></a>[sine-replica-pulse-splitting](nodal/SINE_REPLICA_PULSE.md#sine-replica-pulse-splitting)

- <a id="11-small-amplitude-transverse-splitting-with-collective-response-retained"></a>[11. Small-amplitude transverse splitting with collective response retained](nodal/SINE_REPLICA_PULSE.md#11-small-amplitude-transverse-splitting-with-collective-response-retained)

- <a id="111-fix-the-period-before-expanding-the-coefficients"></a>[11.1. Fix the period before expanding the coefficients](nodal/SINE_REPLICA_PULSE.md#111-fix-the-period-before-expanding-the-coefficients)

- <a id="112-the-induced-collective-motion-changes-the-splitting"></a>[11.2. The induced collective motion changes the splitting](nodal/SINE_REPLICA_PULSE.md#112-the-induced-collective-motion-changes-the-splitting)

- <a id="113-exact-algebraic-signs-for-the-two-spatial-modes"></a>[11.3. Exact algebraic signs for the two spatial modes](nodal/SINE_REPLICA_PULSE.md#113-exact-algebraic-signs-for-the-two-spatial-modes)

- <a id="114-analytic-remainder-and-the-actual-sufficiently-small-conclusion"></a>[11.4. Analytic remainder and the actual sufficiently-small conclusion](nodal/SINE_REPLICA_PULSE.md#114-analytic-remainder-and-the-actual-sufficiently-small-conclusion)

- <a id="115-interpretation-and-what-remains-unmeasured"></a>[11.5. Interpretation and what remains unmeasured](nodal/SINE_REPLICA_PULSE.md#115-interpretation-and-what-remains-unmeasured)

- <a id="sine-replica-joint-persistence"></a>[sine-replica-joint-persistence](nodal/SINE_REPLICA_PULSE.md#sine-replica-joint-persistence)

- <a id="12-collective-identity-with-independently-active-constituents"></a>[12. Collective identity with independently active constituents](nodal/SINE_REPLICA_PULSE.md#12-collective-identity-with-independently-active-constituents)

- <a id="121-an-admitted-open-full-state-family"></a>[12.1. An admitted open full-state family](nodal/SINE_REPLICA_PULSE.md#121-an-admitted-open-full-state-family)

- <a id="122-every-nontip-pair-remains-internally-active"></a>[12.2. Every nontip pair remains internally active](nodal/SINE_REPLICA_PULSE.md#122-every-nontip-pair-remains-internally-active)

- <a id="123-bounded-internal-circulation-without-a-common-period"></a>[12.3. Bounded internal circulation without a common period](nodal/SINE_REPLICA_PULSE.md#123-bounded-internal-circulation-without-a-common-period)

- <a id="124-almost-everywhere-recurrence-in-the-same-family"></a>[12.4. Almost-everywhere recurrence in the same family](nodal/SINE_REPLICA_PULSE.md#124-almost-everywhere-recurrence-in-the-same-family)

- <a id="125-overlap-with-the-prepared-pulse-and-its-instability"></a>[12.5. Overlap with the prepared pulse and its instability](nodal/SINE_REPLICA_PULSE.md#125-overlap-with-the-prepared-pulse-and-its-instability)

- <a id="126-captured-source-admission-and-evidence"></a>[12.6. Captured-source admission and evidence](nodal/SINE_REPLICA_PULSE.md#126-captured-source-admission-and-evidence)

- <a id="sine-replica-capacity-asymmetry"></a>[sine-replica-capacity-asymmetry](nodal/SINE_PAIR_STATE.md#sine-replica-capacity-asymmetry)

- <a id="13-unequal-constituent-capacities-persistence-and-its-symmetry-boundary"></a>[13. Unequal constituent capacities: persistence and its symmetry boundary](nodal/SINE_PAIR_STATE.md#13-unequal-constituent-capacities-persistence-and-its-symmetry-boundary)

- <a id="131-exact-asymmetric-mean-and-internal-rows"></a>[13.1. Exact asymmetric mean and internal rows](nodal/SINE_PAIR_STATE.md#131-exact-asymmetric-mean-and-internal-rows)

- <a id="132-collective-trapping-and-ambient-recurrence-survive"></a>[13.2. Collective trapping and ambient recurrence survive](nodal/SINE_PAIR_STATE.md#132-collective-trapping-and-ambient-recurrence-survive)

- <a id="133-exact-counterexamples-to-tip-invariance-and-pointwise-activity"></a>[13.3. Exact counterexamples to tip invariance and pointwise activity](nodal/SINE_PAIR_STATE.md#133-exact-counterexamples-to-tip-invariance-and-pointwise-activity)

- <a id="134-the-unordered-state-must-retain-capacity-correlations"></a>[13.4. The unordered state must retain capacity correlations](nodal/SINE_PAIR_STATE.md#134-the-unordered-state-must-retain-capacity-correlations)

- <a id="135-weighted-centers-remove-direct-cross-terms-not-internal-information"></a>[13.5. Weighted centers remove direct cross terms, not internal information](nodal/SINE_PAIR_STATE.md#135-weighted-centers-remove-direct-cross-terms-not-internal-information)

- <a id="136-shared-engine-scope"></a>[13.6. Shared engine scope](nodal/SINE_PAIR_STATE.md#136-shared-engine-scope)

- <a id="sine-phase-pairing"></a>[sine-phase-pairing](nodal/SINE_PAIR_GROUPING.md#sine-phase-pairing)

- <a id="14-state-derived-phase-partners-and-independent-law-admission"></a>[14. State-derived phase partners and independent law admission](nodal/SINE_PAIR_GROUPING.md#14-state-derived-phase-partners-and-independent-law-admission)

- <a id="141-a-branch-independent-comparison-of-circular-distances"></a>[14.1. A branch-independent comparison of circular distances](nodal/SINE_PAIR_GROUPING.md#141-a-branch-independent-comparison-of-circular-distances)

- <a id="142-the-protected-tube-separates-true-partners"></a>[14.2. The protected tube separates true partners](nodal/SINE_PAIR_GROUPING.md#142-the-protected-tube-separates-true-partners)

- <a id="143-a-prospective-nearest-partner-certificate-and-its-abstentions"></a>[14.3. A prospective nearest-partner certificate and its abstentions](nodal/SINE_PAIR_GROUPING.md#143-a-prospective-nearest-partner-certificate-and-its-abstentions)

- <a id="144-phase-pairing-does-not-certify-the-paired-support"></a>[14.4. Phase pairing does not certify the paired support](nodal/SINE_PAIR_GROUPING.md#144-phase-pairing-does-not-certify-the-paired-support)

- <a id="145-observation-and-admission-interfaces"></a>[14.5. Observation and admission interfaces](nodal/SINE_PAIR_GROUPING.md#145-observation-and-admission-interfaces)

- <a id="sine-replica-acute-critical"></a>[sine-replica-acute-critical](nodal/SINE_PAIR_GROUPING.md#sine-replica-acute-critical)

- <a id="15-exhaustive-fully-acute-equilibria-on-the-doubled-cycle"></a>[15. Exhaustive fully acute equilibria on the doubled cycle](nodal/SINE_PAIR_GROUPING.md#15-exhaustive-fully-acute-equilibria-on-the-doubled-cycle)

- <a id="151-complete-equations-and-the-sector-being-classified"></a>[15.1. Complete equations and the sector being classified](nodal/SINE_PAIR_GROUPING.md#151-complete-equations-and-the-sector-being-classified)

- <a id="152-zero-phase-rates-force-uniform-form"></a>[15.2. Zero phase rates force uniform form](nodal/SINE_PAIR_GROUPING.md#152-zero-phase-rates-force-uniform-form)

- <a id="153-member-phase-equality-follows-without-a-pair-chart"></a>[15.3. Member-phase equality follows without a pair chart](nodal/SINE_PAIR_GROUPING.md#153-member-phase-equality-follows-without-a-pair-chart)

- <a id="154-acute-sine-balance-and-circular-closure-give-exactly-three-windings"></a>[15.4. Acute sine balance and circular closure give exactly three windings](nodal/SINE_PAIR_GROUPING.md#154-acute-sine-balance-and-circular-closure-give-exactly-three-windings)

- <a id="155-exact-families-captured-phases-and-observer-abstention-are-distinct"></a>[15.5. Exact families, captured phases and observer abstention are distinct](nodal/SINE_PAIR_GROUPING.md#155-exact-families-captured-phases-and-observer-abstention-are-distinct)

- <a id="156-the-strict-sector-boundary-and-what-has-not-been-selected"></a>[15.6. The strict sector boundary and what has not been selected](nodal/SINE_PAIR_GROUPING.md#156-the-strict-sector-boundary-and-what-has-not-been-selected)

- <a id="157-shared-classification-and-captured-state-evidence"></a>[15.7. Shared classification and captured-state evidence](nodal/SINE_PAIR_GROUPING.md#157-shared-classification-and-captured-state-evidence)

- <a id="sine-pairing-transition"></a>[sine-pairing-transition](nodal/SINE_PAIR_GROUPING.md#sine-pairing-transition)

- <a id="16-local-onset-and-loss-of-a-support-compatible-observed-grouping"></a>[16. Local onset and loss of a support-compatible observed grouping](nodal/SINE_PAIR_GROUPING.md#16-local-onset-and-loss-of-a-support-compatible-observed-grouping)

- <a id="161-differentiate-the-observation-using-the-complete-law"></a>[16.1. Differentiate the observation using the complete law](nodal/SINE_PAIR_GROUPING.md#161-differentiate-the-observation-using-the-complete-law)

- <a id="162-one-frozen-represented-preparation"></a>[16.2. One frozen represented preparation](nodal/SINE_PAIR_GROUPING.md#162-one-frozen-represented-preparation)

- <a id="163-all-initial-competitors-and-the-two-transverse-ties"></a>[16.3. All initial competitors and the two transverse ties](nodal/SINE_PAIR_GROUPING.md#163-all-initial-competitors-and-the-two-transverse-ties)

- <a id="164-a-complete-local-transition-follows-from-smoothness"></a>[16.4. A complete local transition follows from smoothness](nodal/SINE_PAIR_GROUPING.md#164-a-complete-local-transition-follows-from-smoothness)

- <a id="165-form-reversal-controls-the-direction-with-the-same-phase-geometry"></a>[16.5. Form reversal controls the direction with the same phase geometry](nodal/SINE_PAIR_GROUPING.md#165-form-reversal-controls-the-direction-with-the-same-phase-geometry)

- <a id="166-what-has-appeared-and-what-has-not"></a>[16.6. What has appeared and what has not](nodal/SINE_PAIR_GROUPING.md#166-what-has-appeared-and-what-has-not)

- <a id="167-implementation-and-evidence"></a>[16.7. Implementation and evidence](nodal/SINE_PAIR_GROUPING.md#167-implementation-and-evidence)

- <a id="sine-pairing-window"></a>[sine-pairing-window](nodal/SINE_PAIR_GROUPING.md#sine-pairing-window)

- <a id="17-whole-box-prediction-throughout-a-fixed-observation-window"></a>[17. Whole-box prediction throughout a fixed observation window](nodal/SINE_PAIR_GROUPING.md#17-whole-box-prediction-throughout-a-fixed-observation-window)

- <a id="171-the-frozen-question-and-complete-state"></a>[17.1. The frozen question and complete state](nodal/SINE_PAIR_GROUPING.md#171-the-frozen-question-and-complete-state)

- <a id="172-a-global-full-field-remainder-without-a-reduced-trajectory"></a>[17.2. A global full-field remainder, without a reduced trajectory](nodal/SINE_PAIR_GROUPING.md#172-a-global-full-field-remainder-without-a-reduced-trajectory)

- <a id="173-exact-rational-margins-for-the-whole-window"></a>[17.3. Exact rational margins for the whole window](nodal/SINE_PAIR_GROUPING.md#173-exact-rational-margins-for-the-whole-window)

- <a id="174-the-two-reserved-predictions-and-their-limits"></a>[17.4. The two reserved predictions and their limits](nodal/SINE_PAIR_GROUPING.md#174-the-two-reserved-predictions-and-their-limits)

- <a id="175-shared-consumer-and-numerical-evidence"></a>[17.5. Shared consumer and numerical evidence](nodal/SINE_PAIR_GROUPING.md#175-shared-consumer-and-numerical-evidence)

- <a id="sine-pair-support-symmetry"></a>[sine-pair-support-symmetry](nodal/SINE_PAIR_STATE.md#sine-pair-support-symmetry)

- <a id="18-which-attachments-permit-an-unordered-pair-state-law"></a>[18. Which attachments permit an unordered pair-state law?](nodal/SINE_PAIR_STATE.md#18-which-attachments-permit-an-unordered-pair-state-law)

- <a id="181-equivalence-concerns-the-entire-member-state-on-a-fixed-graph"></a>[18.1. Equivalence concerns the entire member state on a fixed graph](nodal/SINE_PAIR_STATE.md#181-equivalence-concerns-the-entire-member-state-on-a-fixed-graph)

- <a id="182-necessary-and-sufficient-fixed-support-condition"></a>[18.2. Necessary and sufficient fixed-support condition](nodal/SINE_PAIR_STATE.md#182-necessary-and-sufficient-fixed-support-condition)

- <a id="183-internal-edges-preserve-symmetry-but-change-the-inherited-rows"></a>[18.3. Internal edges preserve symmetry but change the inherited rows](nodal/SINE_PAIR_STATE.md#183-internal-edges-preserve-symmetry-but-change-the-inherited-rows)

- <a id="184-frozen-one-edge-control-identical-retained-state-different-future"></a>[18.4. Frozen one-edge control: identical retained state, different future](nodal/SINE_PAIR_STATE.md#184-frozen-one-edge-control-identical-retained-state-different-future)

- <a id="185-what-a-larger-collective-description-must-retain"></a>[18.5. What a larger collective description must retain](nodal/SINE_PAIR_STATE.md#185-what-a-larger-collective-description-must-retain)

- <a id="186-shared-support-reader-and-state-only-evidence"></a>[18.6. Shared support reader and state-only evidence](nodal/SINE_PAIR_STATE.md#186-shared-support-reader-and-state-only-evidence)

- <a id="sine-mixed-pair-state"></a>[sine-mixed-pair-state](nodal/SINE_PAIR_STATE.md#sine-mixed-pair-state)

- <a id="19-sufficient-collective-state-on-asymmetric-support"></a>[19. Sufficient collective state on asymmetric support](nodal/SINE_PAIR_STATE.md#19-sufficient-collective-state-on-asymmetric-support)

- <a id="191-remove-only-the-independently-redundant-member-labels"></a>[19.1. Remove only the independently redundant member labels](nodal/SINE_PAIR_STATE.md#191-remove-only-the-independently-redundant-member-labels)

- <a id="192-inherited-rates-use-the-actual-fine-attachments"></a>[19.2. Inherited rates use the actual fine attachments](nodal/SINE_PAIR_STATE.md#192-inherited-rates-use-the-actual-fine-attachments)

- <a id="193-realizability-synchronized-tips-and-the-chart-boundary"></a>[19.3. Realizability, synchronized tips and the chart boundary](nodal/SINE_PAIR_STATE.md#193-realizability-synchronized-tips-and-the-chart-boundary)

- <a id="194-the-retained-receiver-control-now-discriminates-the-two-states"></a>[19.4. The retained receiver control now discriminates the two states](nodal/SINE_PAIR_STATE.md#194-the-retained-receiver-control-now-discriminates-the-two-states)

- <a id="195-shared-observer-and-scope"></a>[19.5. Shared observer and scope](nodal/SINE_PAIR_STATE.md#195-shared-observer-and-scope)

- <a id="sine-star-moment-chart"></a>[sine-star-moment-chart](nodal/SINE_PAIR_STATE.md#sine-star-moment-chart)

- <a id="196-a-regular-phaserate-chart-of-the-existing-unordered-state"></a>[19.6. A regular phase/rate chart of the existing unordered state](nodal/SINE_PAIR_STATE.md#196-a-regular-phaserate-chart-of-the-existing-unordered-state)

- <a id="sine-pairing-constitutive-scope"></a>[sine-pairing-constitutive-scope](nodal/SINE_PAIR_MOBILITY.md#sine-pairing-constitutive-scope)

- <a id="20-constitutive-scope-of-the-local-grouping-mechanism"></a>[20. Constitutive scope of the local grouping mechanism](nodal/SINE_PAIR_MOBILITY.md#20-constitutive-scope-of-the-local-grouping-mechanism)

- <a id="201-the-common-work-identity-leaves-a-mobility-law-to-specify"></a>[20.1. The common work identity leaves a mobility law to specify](nodal/SINE_PAIR_MOBILITY.md#201-the-common-work-identity-leaves-a-mobility-law-to-specify)

- <a id="202-positive-reciprocal-response-preserves-the-local-direction-of-grouping"></a>[20.2. Positive reciprocal response preserves the local direction of grouping](nodal/SINE_PAIR_MOBILITY.md#202-positive-reciprocal-response-preserves-the-local-direction-of-grouping)

- <a id="203-a-frozen-relative-rate-discriminator"></a>[20.3. A frozen relative-rate discriminator](nodal/SINE_PAIR_MOBILITY.md#203-a-frozen-relative-rate-discriminator)

- <a id="204-shared-comparison-and-evidence-boundary"></a>[20.4. Shared comparison and evidence boundary](nodal/SINE_PAIR_MOBILITY.md#204-shared-comparison-and-evidence-boundary)

- <a id="sine-replica-constitutive-nonselection"></a>[sine-replica-constitutive-nonselection](nodal/SINE_PAIR_MOBILITY.md#sine-replica-constitutive-nonselection)

- <a id="205-recursive-inheritance-and-the-equilibrium-tangent-do-not-select-one-law"></a>[20.5. Recursive inheritance and the equilibrium tangent do not select one law](nodal/SINE_PAIR_MOBILITY.md#205-recursive-inheritance-and-the-equilibrium-tangent-do-not-select-one-law)

- <a id="collective-mean-closure-obstruction"></a>[collective-mean-closure-obstruction](nodal/SINE_PAIR_MOBILITY.md#collective-mean-closure-obstruction)

- <a id="206-universal-collective-mean-closure-forces-a-constant-circular-current"></a>[20.6. Universal collective-mean closure forces a constant circular current](nodal/SINE_PAIR_MOBILITY.md#206-universal-collective-mean-closure-forces-a-constant-circular-current)

- <a id="sine-mobility-relative-geometry"></a>[sine-mobility-relative-geometry](nodal/SINE_PAIR_MOBILITY.md#sine-mobility-relative-geometry)

- <a id="21-protected-relative-geometry-without-an-assumed-invariant-volume"></a>[21. Protected relative geometry without an assumed invariant volume](nodal/SINE_PAIR_MOBILITY.md#21-protected-relative-geometry-without-an-assumed-invariant-volume)

- <a id="211-quotient-only-the-common-origins-of-the-actual-complete-law"></a>[21.1. Quotient only the common origins of the actual complete law](nodal/SINE_PAIR_MOBILITY.md#211-quotient-only-the-common-origins-of-the-actual-complete-law)

- <a id="212-the-existing-storage-barrier-protects-the-relative-pattern"></a>[21.2. The existing storage barrier protects the relative pattern](nodal/SINE_PAIR_MOBILITY.md#212-the-existing-storage-barrier-protects-the-relative-pattern)

- <a id="213-mean-motion-and-divergence-change-although-storage-does-not"></a>[21.3. Mean motion and divergence change although storage does not](nodal/SINE_PAIR_MOBILITY.md#213-mean-motion-and-divergence-change-although-storage-does-not)

- <a id="214-what-is-still-required-for-an-almost-everywhere-return-claim"></a>[21.4. What is still required for an almost-everywhere return claim](nodal/SINE_PAIR_MOBILITY.md#214-what-is-still-required-for-an-almost-everywhere-return-claim)

- <a id="215-shared-balance-and-admission-evidence"></a>[21.5. Shared balance and admission evidence](nodal/SINE_PAIR_MOBILITY.md#215-shared-balance-and-admission-evidence)

- <a id="sine-pair-emission-descent"></a>[sine-pair-emission-descent](nodal/SINE_PAIR_INTERACTION.md#sine-pair-emission-descent)

- <a id="22-local-and-whole-pair-form-actions-on-an-exact-unordered-nfr"></a>[22. Local and whole-pair form actions on an exact unordered NFR](nodal/SINE_PAIR_INTERACTION.md#22-local-and-whole-pair-form-actions-on-an-exact-unordered-nfr)

- <a id="221-keep-the-continuous-quotient-and-the-supplied-event-separate"></a>[22.1. Keep the continuous quotient and the supplied event separate](nodal/SINE_PAIR_INTERACTION.md#221-keep-the-continuous-quotient-and-the-supplied-event-separate)

- <a id="222-exact-obstruction-and-the-two-exceptional-cases"></a>[22.2. Exact obstruction and the two exceptional cases](nodal/SINE_PAIR_INTERACTION.md#222-exact-obstruction-and-the-two-exceptional-cases)

- <a id="223-a-sufficient-marked-port-and-a-symmetric-control"></a>[22.3. A sufficient marked port and a symmetric control](nodal/SINE_PAIR_INTERACTION.md#223-a-sufficient-marked-port-and-a-symmetric-control)

- <a id="224-storage-and-occurrence-remain-separate-obligations"></a>[22.4. Storage and occurrence remain separate obligations](nodal/SINE_PAIR_INTERACTION.md#224-storage-and-occurrence-remain-separate-obligations)

- <a id="sine-autonomous-regional-transfer"></a>[sine-autonomous-regional-transfer](nodal/SINE_PAIR_INTERACTION.md#sine-autonomous-regional-transfer)

- <a id="23-autonomous-regional-transfer-and-supplied-form-injection"></a>[23. Autonomous regional transfer and supplied form injection](nodal/SINE_PAIR_INTERACTION.md#23-autonomous-regional-transfer-and-supplied-form-injection)

- <a id="231-the-regional-balance-of-the-complete-sine-law"></a>[23.1. The regional balance of the complete sine law](nodal/SINE_PAIR_INTERACTION.md#231-the-regional-balance-of-the-complete-sine-law)

- <a id="relative-inventory-and-a-physical-property-boundary"></a>[Relative inventory and a physical-property boundary](nodal/SINE_PAIR_INTERACTION.md#relative-inventory-and-a-physical-property-boundary)

- <a id="sine-two-contact-orientation-response"></a>[sine-two-contact-orientation-response](nodal/SINE_PAIR_INTERACTION.md#sine-two-contact-orientation-response)

- <a id="a-finite-orientation-sensitive-response-with-retained-backreaction"></a>[A finite orientation-sensitive response with retained backreaction](nodal/SINE_PAIR_INTERACTION.md#a-finite-orientation-sensitive-response-with-retained-backreaction)

- <a id="232-a-positive-control-inherited-from-the-existing-pair-pulse"></a>[23.2. A positive control inherited from the existing pair pulse](nodal/SINE_PAIR_INTERACTION.md#232-a-positive-control-inherited-from-the-existing-pair-pulse)

- <a id="233-why-a-positive-al-only-endpoint-is-not-that-closed-flow"></a>[23.3. Why a positive AL-only endpoint is not that closed flow](nodal/SINE_PAIR_INTERACTION.md#233-why-a-positive-al-only-endpoint-is-not-that-closed-flow)

- <a id="234-shared-current-and-endpoint-observations"></a>[23.4. Shared current and endpoint observations](nodal/SINE_PAIR_INTERACTION.md#234-shared-current-and-endpoint-observations)

- <a id="sine-joint-identity-window"></a>[sine-joint-identity-window](nodal/SINE_PAIR_GROUPING.md#sine-joint-identity-window)

- <a id="24-joint-form-phase-identification-during-autonomous-exchange"></a>[24. Joint form-phase identification during autonomous exchange](nodal/SINE_PAIR_GROUPING.md#24-joint-form-phase-identification-during-autonomous-exchange)

- <a id="241-a-storage-defined-observation-not-an-extra-dynamical-rule"></a>[24.1. A storage-defined observation, not an extra dynamical rule](nodal/SINE_PAIR_GROUPING.md#241-a-storage-defined-observation-not-an-extra-dynamical-rule)

- <a id="242-exact-reference-identity-survives-a-phase-only-collision"></a>[24.2. Exact reference identity survives a phase-only collision](nodal/SINE_PAIR_GROUPING.md#242-exact-reference-identity-survives-a-phase-only-collision)

- <a id="243-a-full-field-perturbation-bound-over-a-declared-window"></a>[24.3. A full-field perturbation bound over a declared window](nodal/SINE_PAIR_GROUPING.md#243-a-full-field-perturbation-bound-over-a-declared-window)

- <a id="244-a-fixed-full-exchange-control-and-genuine-degeneracies"></a>[24.4. A fixed full-exchange control and genuine degeneracies](nodal/SINE_PAIR_GROUPING.md#244-a-fixed-full-exchange-control-and-genuine-degeneracies)

- <a id="245-shared-observation-and-window-admission"></a>[24.5. Shared observation and window admission](nodal/SINE_PAIR_GROUPING.md#245-shared-observation-and-window-admission)

- <a id="sine-joint-grouping-comparison"></a>[sine-joint-grouping-comparison](nodal/SINE_PAIR_GROUPING.md#sine-joint-grouping-comparison)

- <a id="25-the-frozen-phase-transition-does-not-change-the-joint-pairing"></a>[25. The frozen phase transition does not change the joint pairing](nodal/SINE_PAIR_GROUPING.md#25-the-frozen-phase-transition-does-not-change-the-joint-pairing)

- <a id="251-compare-two-observations-of-the-same-complete-preparation"></a>[25.1. Compare two observations of the same complete preparation](nodal/SINE_PAIR_GROUPING.md#251-compare-two-observations-of-the-same-complete-preparation)

- <a id="252-a-different-joint-partition-is-already-strictly-present"></a>[25.2. A different joint partition is already strictly present](nodal/SINE_PAIR_GROUPING.md#252-a-different-joint-partition-is-already-strictly-present)

- <a id="253-the-same-joint-pairs-persist-on-the-complete-frozen-boxes"></a>[25.3. The same joint pairs persist on the complete frozen boxes](nodal/SINE_PAIR_GROUPING.md#253-the-same-joint-pairs-persist-on-the-complete-frozen-boxes)

- <a id="254-observation-change-collective-state-and-formation-are-distinct"></a>[25.4. Observation change, collective state and formation are distinct](nodal/SINE_PAIR_GROUPING.md#254-observation-change-collective-state-and-formation-are-distinct)

- <a id="255-a-projection-of-the-existing-window-not-another-forecast"></a>[25.5. A projection of the existing window, not another forecast](nodal/SINE_PAIR_GROUPING.md#255-a-projection-of-the-existing-window-not-another-forecast)

- <a id="sine-joint-boundary-acquisition"></a>[sine-joint-boundary-acquisition](nodal/SINE_PAIR_GROUPING.md#sine-joint-boundary-acquisition)

- <a id="26-acquiring-a-joint-pairing-at-its-exact-observation-boundary"></a>[26. Acquiring a joint pairing at its exact observation boundary](nodal/SINE_PAIR_GROUPING.md#26-acquiring-a-joint-pairing-at-its-exact-observation-boundary)

- <a id="261-a-separately-declared-critical-preparation"></a>[26.1. A separately declared critical preparation](nodal/SINE_PAIR_GROUPING.md#261-a-separately-declared-critical-preparation)

- <a id="262-both-rows-determine-the-crossing-direction"></a>[26.2. Both rows determine the crossing direction](nodal/SINE_PAIR_GROUPING.md#262-both-rows-determine-the-crossing-direction)

- <a id="263-one-sided-acquisition-and-qualitative-full-state-robustness"></a>[26.3. One-sided acquisition and qualitative full-state robustness](nodal/SINE_PAIR_GROUPING.md#263-one-sided-acquisition-and-qualitative-full-state-robustness)

- <a id="264-what-is-acquired-and-what-remains-supplied"></a>[26.4. What is acquired, and what remains supplied](nodal/SINE_PAIR_GROUPING.md#264-what-is-acquired-and-what-remains-supplied)

- <a id="sine-joint-recurrent-episodes"></a>[sine-joint-recurrent-episodes](nodal/SINE_PAIR_GROUPING.md#sine-joint-recurrent-episodes)

- <a id="27-repeated-finite-episodes-of-joint-identification"></a>[27. Repeated finite episodes of joint identification](nodal/SINE_PAIR_GROUPING.md#27-repeated-finite-episodes-of-joint-identification)

- <a id="271-identity-loss-and-a-common-acquisition-interval"></a>[27.1. Identity, loss and a common acquisition interval](nodal/SINE_PAIR_GROUPING.md#271-identity-loss-and-a-common-acquisition-interval)

- <a id="272-one-finite-invariant-ambient-family"></a>[27.2. One finite invariant ambient family](nodal/SINE_PAIR_GROUPING.md#272-one-finite-invariant-ambient-family)

- <a id="273-infinitely-many-distinct-finite-matching-episodes"></a>[27.3. Infinitely many distinct finite matching episodes](nodal/SINE_PAIR_GROUPING.md#273-infinitely-many-distinct-finite-matching-episodes)

- <a id="274-measure-interpretation-and-integration-boundaries"></a>[27.4. Measure, interpretation and integration boundaries](nodal/SINE_PAIR_GROUPING.md#274-measure-interpretation-and-integration-boundaries)

- <a id="sine-zero-resultant-restoration"></a>[sine-zero-resultant-restoration](nodal/SINE_PAIR_INTERACTION.md#sine-zero-resultant-restoration)

- <a id="28-zero-pair-resultant-and-autonomous-restoration-of-form-current"></a>[28. Zero pair resultant and autonomous restoration of form current](nodal/SINE_PAIR_INTERACTION.md#28-zero-pair-resultant-and-autonomous-restoration-of-form-current)

- <a id="281-a-global-current-observation-without-a-midpoint-angle"></a>[28.1. A global current observation without a midpoint angle](nodal/SINE_PAIR_INTERACTION.md#281-a-global-current-observation-without-a-midpoint-angle)

- <a id="282-the-retained-phase-channel-and-internal-form-information"></a>[28.2. The retained phase channel and internal form information](nodal/SINE_PAIR_INTERACTION.md#282-the-retained-phase-channel-and-internal-form-information)

- <a id="283-one-frozen-antipodal-preparation-and-its-complete-response"></a>[28.3. One frozen antipodal preparation and its complete response](nodal/SINE_PAIR_INTERACTION.md#283-one-frozen-antipodal-preparation-and-its-complete-response)

- <a id="284-immediate-restoration-a-magnitude-cusp-and-exact-silence"></a>[28.4. Immediate restoration, a magnitude cusp and exact silence](nodal/SINE_PAIR_INTERACTION.md#284-immediate-restoration-a-magnitude-cusp-and-exact-silence)

- <a id="285-existing-source-owners-and-exact-phase-scope"></a>[28.5. Existing source owners and exact-phase scope](nodal/SINE_PAIR_INTERACTION.md#285-existing-source-owners-and-exact-phase-scope)

- <a id="sine-moving-pattern-interface"></a>[sine-moving-pattern-interface](nodal/SINE_PAIR_INTERACTION.md#sine-moving-pattern-interface)

- <a id="286-retained-interaction-contract-for-a-moving-constituent"></a>[28.6. Retained interaction contract for a moving constituent](nodal/SINE_PAIR_INTERACTION.md#286-retained-interaction-contract-for-a-moving-constituent)

- <a id="sine-global-pair-state"></a>[sine-global-pair-state](nodal/SINE_PAIR_STATE.md#sine-global-pair-state)

- <a id="29-global-unordered-pair-state-through-phase-cancellation"></a>[29. Global unordered pair state through phase cancellation](nodal/SINE_PAIR_STATE.md#29-global-unordered-pair-state-through-phase-cancellation)

- <a id="291-derived-coordinates-and-exact-realizability"></a>[29.1. Derived coordinates and exact realizability](nodal/SINE_PAIR_STATE.md#291-derived-coordinates-and-exact-realizability)

- <a id="292-the-inherited-polynomial-field"></a>[29.2. The inherited polynomial field](nodal/SINE_PAIR_STATE.md#292-the-inherited-polynomial-field)

- <a id="293-storage-current-and-global-continuation"></a>[29.3. Storage, current and global continuation](nodal/SINE_PAIR_STATE.md#293-storage-current-and-global-continuation)

- <a id="294-why-cancellation-needs-the-retained-phase-product"></a>[29.4. Why cancellation needs the retained phase product](nodal/SINE_PAIR_STATE.md#294-why-cancellation-needs-the-retained-phase-product)

- <a id="295-implementation-and-numerical-scope"></a>[29.5. Implementation and numerical scope](nodal/SINE_PAIR_STATE.md#295-implementation-and-numerical-scope)

- <a id="sine-pair-cancellation-observability"></a>[sine-pair-cancellation-observability](nodal/SINE_PAIR_STATE.md#sine-pair-cancellation-observability)

- <a id="296-observable-internal-state-at-cancellation"></a>[29.6. Observable internal state at cancellation](nodal/SINE_PAIR_STATE.md#296-observable-internal-state-at-cancellation)
