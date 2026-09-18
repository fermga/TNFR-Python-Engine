# Nodal parameter foundations and constitutive scope

Reviewed 2026-09-18. This is the cross-channel foundation audit supporting G3
in the [single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md).
It is not another research queue. The definition of structural form, tangent
pressure and chart changes remains in
[FUNDAMENTAL_THEORY sections 2.4-2.6](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation).
The [constitutive audit](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#14-constitutive-closure-audit-from-the-nodal-law)
owns the detailed inventory of implemented auxiliary laws. This document
connects every foundational parameter family to those owners, states additional
derivations and records what the current implementation cannot establish.

## 1. Start from a typed law, not from its defaults

Write the unforced nodal equation as

\[
\dot x_i=\nu_i p_i,\qquad p_i=P_i(z),\qquad
z=(x,\nu,\theta,G,W,\ell,h).
\]

Here `h` denotes whatever history the declared model actually needs. It is not
an assertion that all these coordinates are independent physical primitives.
Some could be observations of a richer form; that requires an explicit map
and a closure proof. Conversely, naming them parts of one structure does not
already provide their equations of motion.

For a chosen real EPI chart, let `[x]=X`, `[t]=T`, `[nu]=T^-1`. Then `[p]=X`.
The phase coordinate is an angle on the circle, expressed in radians; an
explicit structural distance can have a separate unit `L`. All code values
may be nondimensional, but the reference scales and time convention must then
be declared. A finite real number does not certify dimensional consistency.

Three levels must stay distinct:

1. **Definition or identity:** for example, wrapped angles have magnitude at
   most pi in the radian chart.
2. **Conditional consequence:** diffusion dissipates its Dirichlet energy
   under the stated nonnegative reciprocal conductance and mobility premises.
3. **Model or numerical choice:** the pressure mixture, phase/capacity update,
   operator word, clipping interval, threshold, seed or sampling protocol.

An identity involving a chosen coefficient proves properties of that choice;
it does not derive why nature or the nodal law must choose that coefficient.
The naming of `constants.canonical` is not an epistemic classification.

## 2. Parameter and dependency ledger

| Family | Meaning and owner | Established boundary |
| --- | --- | --- |
| EPI `x` | Coherent form in a declared chart; [foundation types](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation), [scalarization](../src/tnfr/mathematics/epi.py) | Signed real storage and finite BEPI storage exist. Neither storage complexity nor temporal entropy selects the necessary physical state space. |
| Capacity `nu` | Nonnegative local mobility/rate in `xdot=nu*p`; [adaptation](../src/tnfr/dynamics/adaptation.py) | Zero freezes this unforced EPI channel; it need not destroy stored form or freeze all other channels. `nu<=2*pi` is a configured rail, not a phase theorem. No unique autonomous capacity law follows from the product. |
| Pressure `p` | Directed tangent response, evaluated from a declared constitutive map; [dnfr](../src/tnfr/dynamics/dnfr.py), [support transport](../src/tnfr/physics/support_transport.py) | The EPI channel is a graph difference. The full pressure is not generally a potential gradient. Its sign is chart-dependent and does not name an operator. Stored pressure need not be freshly evaluated pressure. |
| Phase `theta` | Circle coordinate and wrapped neighbor separation; [phase response](../src/tnfr/physics/phase_response.py) | A common rotation is a symmetry of the regular difference/phasor formulas. Relative phase is independent of scalar EPI in the present representation. The nodal product does not imply `theta_dot=nu` or any synchronization law. |
| Clock `t`, `dt` | Declared time coordinate and numerical increment; [integrator](../src/tnfr/dynamics/integrators.py), [directed structural time](../src/tnfr/physics/directed_diffusion.py) | Physical time, operator position, invocation count and history index are different. A step size alone supplies no stability guarantee. |
| Channel coefficients | `p=w_phi*g_phi+w_epi*g_epi+w_vf*g_vf+w_topo*g_topo`; [defaults](../src/tnfr/config/defaults_core.py) | Numeric sum one is a selected normalization, not physical dimensional analysis. Priority phase/EPI/capacity is configured. See section 3. |
| Support `G` | Which nodes can be neighbors, including zero-conductance support edges | Phase and capacity support need not equal positive EPI transport support. Isolates, self-loops, multiple edges and directionality require explicit conventions. |
| Conductance `W` | Nonnegative transport strength; [support transport](../src/tnfr/physics/support_transport.py) | Common positive rescaling leaves row-normalized EPI transport unchanged. Reciprocal conductance is an additional assumption, not a consequence of undirected support alone. |
| Distance `ell`, kernel exponent | Structural path-distance read-out; [edge semantics](../src/tnfr/physics/_edge_semantics.py) | Explicit `length` wins; absent length, `weight` is a compatibility fallback, then unit length. Undirected distances can be a pseudometric; distinct-node zero distances are omitted from the potential and coherence fit. Parallel lengths combine by minimum; directed distances follow outgoing arcs and can be asymmetric. No equation here derives distance from conductance or selects inverse-square exponent 2 uniquely. |
| Tetrad | Potential, phase gradient, phase curvature, coherence length; [fields](../src/tnfr/physics/fields.py) | Required complementary diagnostics, not a proved closed state or a four-dimensional complete basis. Section 5 states their different units and domains. |
| Coherence `C`, Sense Index `Si` | Shared diagnostic conventions; [metrics common](../src/tnfr/metrics/common.py), [sense index](../src/tnfr/metrics/sense_index.py) | Normalized read-outs are not new dynamical laws. Their use as feedback is a separately configured controller, including where an old runtime already does so. |
| `dEPI`, acceleration | Stored rate or declared event secant, then rate difference per time | A stored pre-projection rate may differ from the realized clipped increment. An event secant is not automatically the continuous derivative; acceleration inherits both timestamps and rate provenance. |
| Currents, energies, charges | Graph contractions and auxiliary models; [conservation](../src/tnfr/physics/conservation.py), [variational principle](TNFR_VARIATIONAL_PRINCIPLE.md) | Positivity of a sum of squares does not prove decay. A charge label does not prove conservation or quantization. Units and a symmetry/action must be supplied for a physical Noether claim. |
| Memory and scale | Hidden-coordinate elimination; [EPI memory](../src/tnfr/physics/epi_memory.py); declared jumps in [REMESH](../src/tnfr/operators/_delayed_remesh_kernel.py) | Exact projected memory is model-dependent. REMESH mixing factors and integer history delays are selected maps, not the automatically derived kernel. Nesting alone is not self-similarity or an autonomous macro-NFR. |
| Operators and grammar | Named transformation contracts, phase gates, word admission and runtime evidence; [contracts](../src/tnfr/operators/operator_contracts.py), [grammar bases](../src/tnfr/operators/grammar_canon.py) | Channel/sign contracts do not uniquely derive magnitudes, order or occurrence. U1-U6 combine definitions, conditions and policies. Completeness of the 13 transformations remains open. |
| Numerical and statistical controls | Clipping, tolerances, discretization, regression bins, sampling and random seeds | They define an experiment or implementation. Reproducibility is not physical necessity; small residuals need an error scale and cannot replace a proof. |

## 3. Joint changes of form and time units

The type audit imposes a useful positive constraint. Hold support fixed and
write the configured channel readings as

\[
p=w_\phi g_\phi+w_e g_e+w_\nu g_\nu+w_Tg_T.
\]

Here `g_phi` is a wrapped angular displacement divided by pi, `g_e` is the
weighted neighbor EPI difference, `g_nu` is the unweighted neighbor capacity
difference, and `g_T` is the declared dimensionless topology reading. Thus

\[
[w_\phi]=[w_T]=X,\quad [w_e]=1,\quad [w_\nu]=XT.
\]

For `a,c>0`, choose a common affine form chart and a constant time-unit change:

\[
y=ax+b\mathbf1,\quad \tau=ct,\quad \nu_\tau=\nu/c.
\]

Then `g_e` scales by `a`, `g_nu` by `1/c`, and the phase/topology readings
are unchanged. Coefficientwise covariance of this constitutive family uses

\[
(w_\phi',w_e',w_\nu',w_T')=
(a w_\phi,w_e,ac w_\nu,a w_T).
\]

This gives `p'=a*p` and `dy/dtau=(a/c)*dx/dt`, as required. Keeping all
numeric coefficients fixed fails in general, even for a change of time unit
alone. Renormalizing the transformed coefficients to sum one also changes
the rate by `1/S`, where `S` is their transformed sum, unless a compensating
factor is explicitly introduced. The production normalized-weight interface
therefore is not automatically covariant under physical unit changes.

The tests in
[test_nodal_parameter_covariance.py](../tests/physics/test_nodal_parameter_covariance.py)
exercise the existing support/forcing owners with exact rational arithmetic
on their retained finite coefficients. They include wrong-fixed-coefficient
and renormalization counterexamples; they do not claim exact transcendental
phase evaluation or derive new dynamics. Bounds, thresholds, external inputs
and every auxiliary law would also need transformation to establish covariance
of a complete runtime.

The same distinction applies to phase: `theta_dot=nu` treats the numeric
capacity as an angular rate. If `nu` instead counts cycles per time, the
conversion is `theta_dot=2*pi*nu`. Neither choice follows from
`xdot=nu*p`; their relationship requires a stated constitutive premise.
Writing angles in radians gives the exact pi bound, not a universal ceiling
on a rate or a unique coefficient for other channels.

## 4. What locality and symmetry can derive

Consider only an affine local EPI pressure on a finite loopless support:

\[
P_i(x)=b_i+\sum_j A_{ij}x_j,
\qquad A_{ij}=0\ \text{off support for }i\ne j.
\]

Uniform-shift invariance gives `A*1=0`. Requiring every uniform form to be an
equilibrium gives `b=0`. Consequently

\[
P_i(x)=\sum_{j\ne i}a_{ij}(x_j-x_i).
\]

The local maximum principle for every `x` is equivalent to `a_ij>=0`: the
sufficiency follows term by term; for necessity set `x_i=0`, one selected
neighbor `x_j=-1`, and all other coordinates zero. This derives a possibly directed
diffusive generator **under those premises**. It does not select its gains.

Reciprocity needs an additional positive measure `h` satisfying
`h_i*a_ij=h_j*a_ji`. Then `W_ij=h_i*a_ij` is symmetric. With unit off-diagonal
row sums, `d_i=h_i` and `P=-L_rw*x`. Undirected support is insufficient: a
triangle with forward rates `2/3` and reverse rates `1/3` has symmetric
support but violates detailed balance, because the two cycle products are
`8/27` and `1/27`.

There is a restricted way to remove conductance ratios without fitting them.
If reciprocal conductances depend only on support and are invariant under
every support automorphism, they are constant on each undirected edge orbit.
On a connected edge-transitive graph with nonzero conductance all conductances
are one common positive value;
row normalization removes it and yields the unweighted random walk. Multiple
edge orbits leave ratios undetermined. This is a genuine consequence of the
added symmetry premises, not a derivation of why a network must have that
symmetry or why it must emerge.

Fixed phase, capacity and topology sources instead give `b=F`; the full
multichannel pressure need not annihilate a uniform EPI field or satisfy a
maximum principle. Removing those sources would change the model. Self-loops
also change normalization without contributing an EPI difference and require
their own convention. The portable covariance controls test these restricted
matrix identities alongside the existing support owner.

## 5. Geometry and the whole tetrad

### 5.1 Potential and local phase fields

At fixed metric kernel, `Phi_s=B_G*p` is linear in pressure. For inverse-square
distance, `[Phi_s]=X/L^2` if distances carry length units. Rescaling all distances
by `k>0` scales potential by `k^-2`; independent conductance rescaling does not
change it when every edge has an explicit fixed length. With the old weight
fallback those two changes are coupled by representation, not by a theorem.

`|grad phi|` is a mean absolute wrapped neighbor separation, not a derivative
per unit metric length. `K_phi=wrap(theta_i-Arg sum_j exp(i theta_j))` is a
circular discrepancy; both are bounded by pi in the radian chart. Interpreting
them as spatial differential operators requires a length convention and a
limit. A nonzero resultant and a regular branch are needed for the displayed
phase-response derivative. At very small resultants the current curvature
read-out must distinguish ill-conditioning from an undefined direction.
The shared `observe_phase_curvature` interface now sums its materialized
binary64 phasor components exactly and records that numerical scope. Every
nonzero represented resultant has a reported numerical direction; the former
`1e-9` arithmetic-angle fallback is removed. At exact represented joint zero,
the evidence marks curvature unavailable and numeric curvature/full-field
adapters raise `UndefinedPhaseCurvatureError`. The gradient remains available;
an isolated node has the explicit empty-neighborhood zero convention.

Exact summation of materialized components is not exact transcendental
evaluation or a conditioning guarantee. For example, represented phases
`(0,0,+float(pi),-float(pi))` can have exactly cancelling represented phasors
although their exact-real trigonometric resultant is nonzero. Unavailability
describes the chosen numerical realization; it does not prove a physical
singularity. Pressure, IL and coordination dynamics retain their separately
declared kernels. This correction selects no new phase evolution law.

### 5.2 One coherence-fit definition across implementations

The scalar and vector implementations now share
[_coherence_fit.py](../src/tnfr/physics/_coherence_fit.py). The fitted field is
the static pressure-only read-out `c_i=1/(1+|p_i|)`, with zero rate argument;
it is not generally the full runtime coherence. The fitted statistic is

\[
q(r)=\operatorname{mean}_{d(i,j)=r}(c_i c_j),\qquad
\log q(r)\approx \log A-r/\xi_C.
\]

This is an **uncentered product fit**, not a connected covariance or a
universal correlation-decay theorem. The shared distance channel uses explicit
`length`, then compatibility `weight`, then one, taking the minimum of parallel
lengths and following outgoing arcs. Undirected graphs use unordered pairs;
directed graphs use ordered reachable pairs. Off-diagonal zero-distance pairs
are excluded: undirected zero-length paths give a pseudometric rather than a
separating metric, and directed distances need not be symmetric.
Negative or nonfinite edge lengths and unrepresentable reachable distances
raise rather than silently selecting the spectral fallback. A caller-supplied
distance matrix must satisfy the explicit numeric domain, zero diagonal and
undirected symmetry; it remains declared data, not authenticated shortest paths.

The fit requires at least ten positive-distance pairs, at least two pairs per
exact represented distance bin, and at least three bins whose mean product
exceeds `1e-9`. It accepts only a finite negative log-linear slope and finite
positive length; no goodness-of-fit acceptance test is asserted. These are
estimator policies, not physical thresholds. Below 1000 nodes all pairs are
used. Larger graphs sample evenly spaced source nodes in insertion order;
undirected pairs incident to selected sources are counted once, while directed
pairs originate at those sources. Both backends use the same selection. The
canonical fit cache includes node/source order, the numerical path, pressure,
topology with both edge channels, and precision mode. The outer structural
telemetry cache additionally binds node and neighbor iteration order and the
coherence numerical path, so it cannot bypass these inputs with a stale result.

The previous implementations used hop distances versus conductance distances,
and the vector path discarded one orientation on directed graphs. The common
owner removes this disagreement. Tests use a weighted star with
`c_center=1`, `c_leaf=exp(-r_leaf/xi)`: every pair product is exactly the
target exponential in real arithmetic. This checks a known fit, metric scaling,
explicit-length independence, directionality and cache refresh without inferring
a physical transition from synthetic data.

The spectral fallback remains a different diagnostic:
`1/sqrt(lambda_positive)` from a normalized graph Laplacian is dimensionless.
It does not scale with explicit metric length and is not interchangeable with
the fitted length. The legacy scalar return alone does not attest which
estimator succeeded; research must retain the estimator path and conditions.
Disconnected/directed graphs need their stated implementation scope rather
than an automatic connected symmetric `lambda_2` interpretation.

## 6. Telemetry, time and energy are not interchangeable

The shared coherence map `C(p,r)=1/(1+|p|+|r|)` is a chosen normalized numeric
diagnostic. Before nondimensionalization, `p` and `r=xdot` have different units;
their raw sum is not a unit-invariant physical observable. Declaring scales
for the two inputs would make the definition meaningful under unit changes,
but selecting those scales is still a model/measurement decision.

Network coherence applies this kernel to mean magnitudes. It generally differs
from mean per-node coherence; a projected parent has another pressure again.
Freshness also matters: stored pressure, stored rate and a newly recomputed
pressure are not automatically simultaneous. The zero-pressure/rate predicate
is not a certificate that phase, capacity and topology are at a fixed point.
The shared P2 joint-response control starts with zero pressure and EPI rate,
yet a supplied relative phase velocity gives nonzero pressure rate and EPI
acceleration. It is an explicit counterexample to promoting the instantaneous
predicate into joint equilibrium, not an autonomous evolution law.
Primary coherence and dispersion now share strict authoritative-alias reads:
invalid provided pressure/rate values raise instead of becoming a later alias
or zero; genuinely missing values retain the public zero default.

The field interfaces follow the same authoritative-source policy: raw
pressure and phase are validated before public cache access. Logical/textual
values, nonfinite values and sources lost to binary64 materialization do not
become apparent equilibrium. Public potential/current/flux and composite
field dictionaries are detached from cache entries; caller edits cannot alter
later readouts. These fixes and partial-field availability belong to the
[field API guide](../docs/STRUCTURAL_FIELDS_TETRAD.md), not a new dynamical law.

The separate dispersion diagnostic `1-std(p)/max|p|` is scale-invariant in
exact arithmetic. Its duplicated implementations previously squared raw
pressures and overflowed or underflowed: `(m,2m)` produced `0.75`, `0`, or
`1` as the numeric scale changed. One shared kernel now normalizes before
computing variance. The same fixture retains `0.75` at extreme and subnormal
represented scales, with sign and backend controls. This repairs a numerical
identity; it does not turn dispersion into the primary coherence or a law.

Si combines relative capacity, phase dispersion and relative pressure with
configured weights and clipping. It changes when the comparison population or
normalizers change. The corrected Python path refreshes the same live capacity
and pressure maxima as NumPy instead of trusting stale maxima. This repairs
backend-dependent values without promoting Si to an autonomous mechanism.
Existing selectors/adaptation that read Si remain declared feedback policies.

The default integrator holds supplied pressure and capacity fixed within a
call. Its `rk4` option integrates the supported time forcing, not a newly
evaluated nonlinear pressure at every RK stage. Clipping is a separate map;
an unconstrained rate is not the clipped step secant. With the optional
additive Gamma source the model is `xdot=nu*p+Gamma`: at positive capacity it
can be rewritten using `p_eff=p+Gamma/nu`, but that rewriting is singular
at zero capacity. Nonzero Gamma can move a zero-capacity node. The four
[forcing-scope controls](../tests/test_nodal_forcing_scope.py) verify the
distinction through both integration backends. Unforced proofs require Gamma
to vanish; source residuals cannot be concealed by reconstructing pressure.

Capacity-rate telemetry now uses the shared
[timestamped observer](../src/tnfr/metrics/capacity_rates.py). For advancing
recorded times, two capacity samples define the interval secant
`r_n=(nu_n-nu_(n-1))/(t_n-t_(n-1))`. Three define
`B_n=2*(r_n-r_(n-1))/(t_n-t_(n-2))`, using the separation of the two interval
midpoints. Exact rational arithmetic on represented samples avoids inventing
a duration from configured `DT` or rounding midpoint timestamps together.
Finite public results remain approximations to those rational values.
With `[nu]=T^-1`, these diagnostics have units `T^-2` and `T^-3`.
Under `tau=c*t`, `nu_tau=nu/c`, their values scale by `c^-2` and `c^-3`,
respectively; the irregular-sample controls check that covariance.

Each node retains at most three samples and a `capacity_rate_diagnostic`
payload with status, availability and source times. Missing time/capacity,
insufficient samples and unrepresentable results are explicit unavailable
values (`None`). A duplicate time creates no interval; changed capacity at
the same time restarts history at the right-hand value. Backward time or
invalid input is rejected before capacity diagnostics are committed. The
metrics callback preflights capacity/time chronology before recording its
history. It records an initial sample on attachment only when runtime time
is already present. Legacy untimestamped derivative fields are not evidence.

`history['B']` is unavailable unless every current node supplies a representable
second difference; `capacity_rate_coverage` reports the coverage. A genuine
measured zero remains distinct from unavailable. `delta_Si` remains an
increment, not a time derivative. The repaired secants can still contain
unobserved events between endpoints, so they do not prove a smooth capacity
law or exact pointwise derivatives. Event resolution and physical clock
calibration remain separate obligations.

Likewise `E_D=1/2*x^T(D-W)x` is a conditional transport energy, while the
tetrad sum of squares and auxiliary substrate Hamiltonian are other functionals.
Adding quantities with different raw units requires scales/metric coefficients.
Packing curvature and current into a complex number supplies neither their
unit conversion nor a physical quantum state. The existing variational and
conservation owners already separate trajectory residuals, auxiliary flows
and restricted dissipation; reuse those distinctions for every new claim.

## 7. Memory, operators and scale

Eliminating hidden coordinates in a declared linear system produces an exact
memory kernel and a hidden-initial-state term. These are derived from that
system and projection; no new sustaining mechanism has been discovered by
renaming the kernel. The existing
[derived-memory analysis](DERIVED_EPI_MEMORY.md) and `epi_memory` owner test
continuous closure and minimal realizations. Event closure additionally
requires the existing event-intertwining check.

REMESH instead references retained pre-jump snapshots at integer positions
`history[-(tau+1)]`. A delay in samples becomes a fixed physical delay only
under a declared uniform sampling convention. Variable cycle durations cannot
be ignored. The mixing coefficient, history support, clipping and metric are
part of its map; existing finite/conditional stability certificates already
record those premises. They do not derive REMESH from every projected kernel.

The 13 operators specify allowed named transformations. They coexist with
declared shared nodal solvers. Reciprocal scalar IL/OZ defaults do not make
the full operators inverse or isometric. Reciprocal NUL capacity/pressure
factors preserve their product only for the ideal two scalar rescalings;
capacity is not a geometric volume. Generator/closure labels, operation-count
debts and phase gates supply admission contracts, not spontaneous event timing.

Noise parameters, finite event counts and seeds likewise require a stated
discrete or stochastic model. A fixed per-call perturbation is not a
time-step-independent continuum noise law. A tolerance is not an exact zero,
and clipping-generated persistence is not evidence for unconstrained stability.

## 8. Reuse and consequence for the generative objective

The review preserves a useful core: typed nodal rates, reversible transport,
phase geometry, exact reduction/memory and finite event evidence. The new
locality and covariance results constrain proposed completions without choosing
one to manufacture a pattern. The repaired telemetry permits consistent tests
of those completions; it does not supply the missing laws.

Any further proposed representation must declare its state/equivalence, explain
which quantities are coordinates and which are observations, and test its
directed response with the existing closure and phase-response owners. A
nonuniform persistent NFR cannot be inferred from fixed positive-capacity pure
diffusion on fixed connected positive-conductance support, which relaxes
spatial disagreement. Sources,
finite accumulated activity, memory, changing geometry or another justified
channel can alter that conclusion, but their origin and work balance must be
part of the same model. This is the link to the original generative objective,
not permission to add a tuned stabilizer or a desired attractor.

The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) alone records the
current gate and next action. Autonomous multichannel closure, a unique
selection of all numerical coefficients, and emergence of laboratory particles
or the observable world remain unproved. The audit covers foundational
parameter families and their principal owners; it is not an exhaustive proof
of every implementation or historical document in this repository.

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
[test_epi_phase_representation_scope.py](../tests/physics/test_epi_phase_representation_scope.py)
cover these statements. They reject this particular lossy packaging, not every
complex-valued formulation of TNFR, and do not derive autonomous NFR formation.

## 10. Joint pressure response and the capacity product rule

### 10.1 One identity, with three distinct neighborhood responses

Fix support, conductance and channel coefficients on a differentiable segment.
Write `U` for the unweighted unique-neighbor averaging matrix, `L_U=I-U`,
and `L_W` for the weighted EPI random-walk Laplacian. On a regular phase
branch with nonzero neighbor resultants, let `R` be the circular-mean
derivative from [phase_response](../src/tnfr/physics/phase_response.py) and
`J=R-I`. The current pressure realization is

```text
p = -e L_W x + w_phi g(theta) - v L_U nu - w_topo L_U k,
Dg = J/pi,               x_dot = nu*p.
```

Here `k` is the fixed unique-support degree. Products between nodal vectors
are componentwise. Set `q=theta_dot/pi` and `a=nu_dot`; these are supplied
velocities, not new state primitives or selected laws. Differentiation gives

```text
p_dot = -e L_W (nu*p) + w_phi J q - v L_U a,
x_ddot = a*p + nu*p_dot.
```

The first term transports the current EPI rate. The phase derivative uses
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

`derive_joint_nodal_response` centralizes this conditional identity. It
rebuilds the detached support snapshot, reuses the
[transport derivative](../src/tnfr/physics/support_transport.py), and verifies
that the declared phase-reference neighborhoods match the unique support.
Its first domain requires nonempty neighborhoods; isolates are rejected,
not assigned a fictitious phasor mean. Coefficients are never renormalized.
The supplied stored pressure is declared input. The observer does not verify
that it is a refreshed canonical pressure, that the ideal cosine Gram belongs
to the captured phases, or that the live wrap branch is regular. Tests obtain
compatible control states through
[forcing_realization](../src/tnfr/physics/forcing_realization.py) and separately
check its fresh/stored/represented residuals. A returned identity is neither
a derivative of binary64 arithmetic nor a runtime execution certificate.

Changing conductance adds the existing transport geometry term. Changing
coefficients adds their derivatives times the corresponding channels;
support changes, operator jumps and REMESH history require their existing
reset/event descriptions. Finite sampled capacity secants are not silently
substituted for a smooth instantaneous `a`.

### 10.2 Pressure-invisible motion can change form acceleration

At the same initial state, differences between two supplied velocity pairs
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
[test_constitutive_capacity_scope.py](../tests/physics/test_constitutive_capacity_scope.py)
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
[reciprocal-closure audit](TNFR_VARIATIONAL_PRINCIPLE.md#13-forced-potential-family-and-reciprocal-closure-constraints)
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
[source tangency identity](FORCED_SUPPORT_BALANCE.md#22-source-tangency-without-a-telemetry-controller)
and reciprocal variational requirements constrain a proposed completion;
neither selects the missing velocities. The shared exact controls are in
[test_joint_nodal_response.py](../tests/physics/test_joint_nodal_response.py).
The single execution plan records the next bounded closure task.
