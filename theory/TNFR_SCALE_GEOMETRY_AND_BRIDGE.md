# Exact scale, coherence-geometry and bridge results

This note records restricted results for core research lines S8, S9, S11, S12,
S14, S15 and S16. Each exact claim states its domain and executable certificate.
None establishes closure of all 13 operators or a coupled global TNFR geometry.

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
those represented-coefficient models. The inherited NumPy `atan2(0,0)`
policy can also return a pressure coefficient where the geometric curvature
reader reports an undefined represented resultant. The explicit
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
[macro-state and tetrad derivation](NODAL_PARAMETER_FOUNDATIONS.md#13-faithful-macro-state-and-tetrad-inheritance-on-the-retained-prism).

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
