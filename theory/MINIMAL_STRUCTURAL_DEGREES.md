<a id="structural-tetrad-diagnostic-degrees-and-reconstruction-scope"></a>
# Structural Observation Families and Tetrad Reconstruction Scope

**Status:** Task-dependent observation design, with the historical tetrad
retained as a compatible selected diagnostic view. No fixed number of channel
names determines a minimal or sufficient structural state. This chapter owns
the reconstruction and dependency criteria; the
[field guide](../docs/STRUCTURAL_FIELDS_TETRAD.md) owns field definitions and
public contracts, and the
[mathematical scope note](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) retains the related
conditional dynamical results.

## 1. Statement

The structural tetrad (Φ_s, |∇φ|, K_φ, ξ_C) is the engine's selected readout of
pressure accumulation, local phase stress, local circular curvature, and
static pressure-derived spatial structure. Its four established field
definitions and APIs remain available. The broader structural observation
family may also retain primitive state, resolved profiles, oriented phase
relations and independently measured history when the selected task requires
them. These additions are observations of the model, not new constitutive
parameters merely because they receive separate output names.

Here "four" counts diagnostic channels, not four scalar degrees of freedom.
On an N-node graph, the first three channels each retain up to N nodal
values; ξ_C is one graph-level estimate, with its availability and provenance.
Global summaries discard further information. The fields read stored pressure,
primitive phase, support and the declared geometry. They do not directly read
EPI, capacity or history; dependence on those quantities requires the supplied
pressure law or other model relations, including their freshness conditions.

The selection is a diagnostic design. It does not establish that every scalar
observable is a function of the four reported channels, or that the tetrad
determines a full graph state and its evolution under

    ∂EPI/∂t = ν_f · ΔNFR.

A completeness claim must specify a state space, admissible observables,
gauge identifications, and what information is retained by the readout. A
minimality claim must additionally prove that no selected channel is redundant
for that specified task. These stronger claims are not established by naming
four structural questions or by computing the fields on a collection of graphs.

### 1.1 Specify the task before minimizing its observation

Let `z` denote the admitted state, including every state/history variable
consumed by the selected model. Fix the support, metric, law parameters,
inputs, clock and allowed equivalences as part of the comparison, or include
them among the observed variables when they can change. Let `O(z)` be an
observation and `R(z)` the particular state property or response of interest.
For that task, sufficiency requires

```text
O(z1) = O(z2)  implies  R(z1) = R(z2)
```

on the declared domain. Equivalently, `R` must factor through `O` there.
State reconstruction requires injectivity up to exactly the declared
equivalences. Future prediction concerns the selected response over a
specified horizon under a complete well-posed law. It need not require
reconstructing state directions that never affect that response.

Minimality also requires an admissible class of observations and a cost:
for example, the number of individually measured edge relations, the rank
of a linear observer, or memory/computational cost at stated resolution.
One channel can contain an N-vector; counting its name as one scalar gives
no dimension theorem. Different sufficient families can be incomparable,
and an algebraically minimal one can be poorly conditioned.

Availability and node/edge association belong to the observation contract.
Equal available components of two partial observations do not establish
equality of the missing information. A selected diagnostic view can remain
useful even when its unknown components prevent a sufficiency verdict.

## 2. Four diagnostic questions

| Question | Field | Construction |
|----------|-------|--------------|
| How much pressure accumulates from other nodes? | Φ_s | Distance-weighted aggregation of ΔNFR |
| How misaligned is a node with its neighbors? | \|∇φ\| | Mean magnitude of wrapped phase differences |
| How does its phase differ from the neighborhood mean? | K_φ | Wrapped deviation from the circular mean |
| What distance scale summarizes the pressure-derived product profile? | ξ_C | Static exponential fit; separate spectral reference when the fit is unavailable |

This organizes local phase information and non-local aggregation/correlation.
Φ_s is a per-node value with global dependence; it is not a purely local
quantity. ξ_C compresses a spatial correlation profile into a length. The table
describes this selected view; it does not exhaust graph observations or require
all four channels for every task. The coherence fit and spectral fallback
retain their separate units, meanings and availability.

## 3. The operator-derivative tower

### 3.1 Construction

    ΔNFR → distance-kernel aggregation → Φ_s
    phase → wrapped neighbor differences → |∇φ|
    phase → circular-neighborhood deviation → K_φ
    local coherence → spatial correlation → ξ_C

The first two phase operations are related to first and second derivatives.
For an undirected graph, an oriented incidence matrix gives an edge gradient,
and composing it with its weighted adjoint gives the graph Laplacian.
For small phase spread within a consistent branch and matching neighbor
weights, circular curvature approaches L_rw φ. The equality is a linearized
description; the exact implemented field uses circular means and wrapping.

### 3.2 Algebraic generation does not terminate information at second order

Higher Laplacian powers are compositions of the same operator. This fact does
not mean their actions are linearly dependent or recoverable from lossy
pointwise summaries. If L has at least four distinct eigenvalues, then I, L,
L², and L³ are linearly independent. Otherwise a polynomial of degree at most
three would vanish at four distinct eigenvalues.

The finite-dimensional cutoff is controlled by L's minimal polynomial. It
depends on the graph and is not universally second order. In addition, |∇φ|
discards signs and ξ_C discards most of the correlation profile. A generating
set of operators and a sufficient set of state statistics are different
mathematical objects.

### 3.3 Dynamical non-reconstruction

At fixed graph, phase, EPI, and pure-EPI pressure, a uniform rescaling of ν_f
leaves the tetrad unchanged while scaling ∂EPI/∂t = ν_f ΔNFR. Consequently the
tetrad alone does not determine the full dynamical state. The structural triad,
frequency, pressure law, and graph remain part of the model.

The [same-law P2 witness](TNFR_VARIATIONAL_PRINCIPLE.md#cotangent-tetrad-future-witness)
is stronger: it fixes capacity, clock and a complete conditional cotangent law.
Two states with identical full tetrads, pressure and initial form rates have
different potential derivatives and relative-form accelerations because their
omitted EPI means differ. Thus a snapshot tetrad need not close even its own
evolution under a known law. This does not exclude reconstruction from a time
series or the restricted positive result below.

### 3.4 Conditional reconstruction in the pure EPI channel

The capacity-rescaling result identifies capacity as a necessary predictive model
parameter. A graph-specific positive result is also available when the complete
nodal structural-potential field is retained.

For fixed connected reciprocal nonnegative conductance with positive nodal
strengths and fixed positive capacities, let `K` be the canonical
inverse-square shortest-path kernel on its declared distance domain. With
unit pure-EPI channel coefficient, pressure gives

```text
Phi_s = -K L_rw x,    x = EPI.
```

Write `O=-K L_rw`. Since `L_rw 1=0`, uniform EPI shifts are always invisible to
`Phi_s`. If `rank(O)=N-1`, this is the only invisible EPI direction. Appending
the fixed-capacity conserved mean

```text
m(x) = sum_i (d_i/nu_i)x_i / sum_i(d_i/nu_i)
```

makes `[O; m]` injective and hence reconstructs absolute EPI in exact arithmetic.
Together with the capacity vector and known graph, this determines the future
pure-diffusion trajectory. Thus, for this restricted algebraic task, the full
potential field plus one zero-mode scalar is sufficient for form reconstruction,
while capacity remains necessary to determine the clock/generator. Numerical
promotion additionally requires scale-aware rank and residual checks.

[`epi_diffusion_reconstruction_certificate`](../src/tnfr/physics/observability.py)
computes the potential and augmented ranks with separate scale-aware SVD
thresholds, then requires a small relative reconstruction residual before it
reports numerical success. The exhaustive
NetworkX graph atlas through six nodes gives `rank(O)=N-1` on all 142 connected
simple graphs in that finite domain. This is a measured finite-domain result,
not a proof for every graph, weight or structural-potential kernel.

Algebraic rank is also weaker than robust reconstruction. On a four-node
weighted star with edge weights `[10^-3, 10^3, 10^3]`, the rank remains `N-1`
but the nonzero singular-value condition number exceeds `10^10`. Thus an exact
observer can be practically unstable under extreme geometry. Uniformly scaling
all conductances to `10^-8` or `10^8` also preserves the structural nullity but
can make the unscaled augmented observer numerically singular. The certificate
reports its thresholds, condition number and relative residual so downstream
inverse methods can distinguish algebraic identifiability from numerical
resolution.

Global tetrad summaries remain insufficient because they discard the nodal
potential field. Changing capacity ratios also removes the fixed conserved mean,
as shown by the time-varying result in
[TNFR_DIFFUSION_STABILITY_THEOREM.md](TNFR_DIFFUSION_STABILITY_THEOREM.md).

### 3.5 Pressure reconstruction depends on the actual potential kernel

For a known fixed distance kernel, write `Phi_s = B_G p` in exact arithmetic.
The full potential reconstructs an independently variable pressure vector
if and only if `B_G` is injective on its admitted pressure domain. On a
restricted domain the required condition is

```text
ker(B_G) intersect {p1 - p2 : p1,p2 admitted} = {0}.
```

Invertibility of the ambient matrix suffices, but is not a universal property
of every allowed support. In particular, directed distance kernels can be
singular. Pressure-derived coherence profiles are already determined by
pressure and geometry; adding their output names cannot create independent
information when pressure is reconstructed.

For an unrestricted real pressure space, suppose `dim ker(B_G)=d` and the
columns of `N` form an orthonormal basis of that kernel. Then

```text
p -> (B_G p, N^T p)
```

is injective: a difference in its kernel has the form `N a` and satisfies
`N^T N a=a=0`. At least `d` additional independent linear scalar observations
are necessary because fewer cannot have an injective restriction to the
`d`-dimensional kernel. This is minimality within linear augmentations of
this fixed observer, not within arbitrary nonlinear encodings or a different
pressure domain. Numerical inversion additionally needs condition and
residual bounds. Direct stored pressure is often the clearer observation
when the engine already exposes it.

#### Same full tetrad, different signed pressure and form rate

Take the outward directed star with three leaves at each explicit edge length
`r=1,2,3,4` and unit conductance. Set zero phases, unit capacities and zero EPI.
Prepare leaf coherence `c_i=2^(-r)` and hence leaf pressure `p_i=2^r-1`.
Compare center pressure zero and one, corresponding to center coherence one
and one half. All other inputs agree.

Only the center reaches other nodes, so

$$\Phi_s(0)=3\sum_{r=1}^4\frac{2^r-1}{r^2}
=\frac{499}{48},\qquad \Phi_s(i)=0\quad(i\ne0).$$

Center pressure is in the kernel of this potential map. Gradient and curvature
are zero, with the documented isolate convention at leaves. The twelve
positive-distance pairs give `q_r=c_0*2^(-r)`, three per bin. Both fits have
slope `-log(2)` and `xi_C=1/log(2)`, while their intercepts are zero and
`-log(2)`. Thus the full tetrad is equal in exact arithmetic, whereas the
full coherence profiles distinguish the preparations.

Under the same supplied held-pressure law `x'=p, p'=0`, with unit capacity,
phase and support fixed and no events, the center form rate is respectively
zero and one. This is missing signed driver information under a fixed law,
not just uncertainty about which law applies. The
[fixed observation controls](../tests/physics/test_structural_observation_requirements.py)
compare the public readouts with these analytic targets. Equality of exact
formulas is distinct from tolerance-based agreement of floating regressions.

There is a stronger algebraic corollary with the same kernel and leaves:
center pressures `+1` and `-1` also leave the potential unchanged, and now
their equal magnitudes preserve the entire coherence profile. Their center
form rates under the held-pressure law are `+1` and `-1`. Thus the rich
profile does not universally repair this missing signed driver. This
corollary is an exact consequence of the pressure kernel and absolute-value
definition, not another evaluated fixture.

### 3.6 Constructive circular phase reconstruction on fixed support

Let `z_i=exp(i*theta_i)` be circular phase on a finite known graph, and fix
an orientation for each observed edge. The edge phasor is

$$u_{ij}=z_j\overline{z_i},\qquad u_{ji}=\overline{u_{ij}}.$$

The graph of available observations is considered with its underlying
undirected connectivity. Reversing the orientation uses the inverse unit
phasor; it does not require a reverse dynamical arc. Let the original
observation support have `n` nodes and `k` weak components, including isolates.

**Reconstruction theorem.** Edge phasors on any spanning forest of that
support determine every nodal circular phase up to one common unit factor
per component. Choose one root `r` per component and set its reconstructed
phasor to one. For each node, multiply the edge phasors, or their inverses
when traversed backward, along its unique forest path from the root. The
products telescope:

$$\widehat z_i=
\prod_{(a,b)\text{ along }r\leadsto i}u_{ab}
=z_i\overline{z_r}.$$

Supplying the root phasor recovers `z_i=z_r*hat_z_i`. More generally, the
selected individual edge observations suffice modulo one component factor
if and only if they connect each original component. Necessity follows by
rotating one connected piece of a disconnected observed subgraph while
leaving the rest of its original component fixed. Every observed edge phasor
stays unchanged, but the original component's relative phase changes.

A spanning forest uses `n-k` edges and is therefore minimal among such
individual edge measurements: fewer cannot connect all original components.
Each phasor is one circle-valued coordinate, represented by two real
components with a unit-norm constraint. This count does not provide a global
uncut real-angle chart or minimum cardinality among arbitrary encodings.

If all edge phasors are retained, cycle products must equal one. Conversely,
unit edge data satisfying the fundamental-cycle products reconstruct nodal
phasors by the same forest construction: each non-tree edge agrees with its
reconstructed endpoint ratio. The extra edges provide consistency checks,
not extra independent phase coordinates for already integrable data.
Noisy or rounded edge data require a separate consistency/error contract;
they may not be silently renormalized into an exact certificate.

#### Derived phase fields, winding and absolute references

In exact circular arithmetic, with `S_i=sum_(j in N(i)) u_ij`, the phase
observation formulas satisfy

```text
|grad phi|(i) = mean_j |Arg(u_ij)|,
K_phi(i) = wrap(-Arg(S_i)) when S_i is nonzero.
```

The isolate convention is separate from a zero resultant on nonempty support.
The mean sine current is the mean of `Im(u_ij)`. Thus the full directed edge
phasors generate these local phase readouts with their stated support and
branch conventions; the converse generally loses information.

The implemented phasor/resultant readers separately retain represented-component
and rounding semantics. These ideal identities do not turn rounded relative
phasors into an exact reconstruction certificate for their numerical output.

For a declared oriented cycle away from the wrap cut,

$$W=\frac{1}{2\pi}\sum_{(i,j)\text{ on cycle}}\operatorname{Arg}(u_{ij})$$

is an integer because the edge product is one. It is derived from the
edge phasors, including the non-tree edges determined by reconstruction;
it is not an additional independent continuous coordinate. It can change
when an edge crosses the selected cut. The
[existing winding owner](../src/tnfr/physics/winding_certificates.py) reports
branch margin and availability separately from U3 admission. The
[marked eight-cycle witness](COHERENCE_OBSERVATION_FOUNDATIONS.md#61-an-oriented-phase-observation-beyond-this-full-tetrad)
already proves that winding can distinguish identical full tetrads. That
proof retains orientation and label equivalences rather than treating every
reflection as a different physical state.

For circular-state reconstruction, one root phase per component restores
absolute phase. For prediction those anchors can be omitted only if the
complete law and specified inputs admit the corresponding component
rotations as symmetries, and the target response respects them. An
externally referenced phase law or an interaction joining components can
consume the missing relative reference. Circular reconstruction also does
not determine arbitrary continuous phase lifts or their histories; a law
using those lifts needs its own retained branch information.

Algebraic minimality does not ensure uniform numerical robustness. For unit
phasor data with chord errors bounded by `epsilon_e`, telescoping the product
difference gives a reconstruction chord error no larger than the sum of
`epsilon_e` along the selected forest path, plus any root-anchor error.
Long paths can accumulate error. Redundant edges may support a separately
specified consistency or estimation method; they do not automatically
produce an uncertainty reduction without a noise model.

For the broader estimation problem with noisy relative angles, see
[Amit Singer, Angular Synchronization by Eigenvectors and Semidefinite Programming](https://arxiv.org/abs/0905.3174).
That work studies recovery up to a common phase. The finite exact forest
argument above requires no noise model; statistical recovery claims retain
their separately stated assumptions.

### 3.7 Snapshot reconstruction, temporal observability and closure differ

For a complete smooth law `z'=F(z)` and observation `y=O(z)`, the initial
observed rate is `D O(z) F(z)`. A necessary condition for an autonomous
first-order law in `y` is that this rate agree on every admitted fiber
`O(z)=y`. Existence, uniqueness and domain/event behavior are additional
requirements. The
[same-law coherence tangent](COHERENCE_OBSERVATION_FOUNDATIONS.md#51-same-full-tetrad-and-law-different-length-derivative)
violates that fiber condition even after retaining the expanded static
coherence profile. Its omitted EPI is an existing state coordinate, not an
additional fundamental field derived by naming the obstruction.

For the special linear finite-dimensional system `z'=A z, y=C z`, three
different algebraic questions have exact answers:

| Question | Condition under this fixed known model |
| --- | --- |
| Does one snapshot reconstruct all `z`? | `rank(C)=n` |
| Does an ideal output history reconstruct the initial `z`? | `rank([C; C A; ...; C A^(n-1)])=n` |
| Does `y` obey a closed autonomous linear first-order law? | `ker(C)` is invariant under `A`, equivalently `C A=M C` for some `M` |

The history condition assumes ideal derivatives or equivalently a complete
continuous output record on an interval, known clock and dynamics. It is
not a guarantee for a finite noisy sample. The existing
[`linear_observability_certificate`](../src/tnfr/physics/observability.py)
checks the specified finite-dimensional rank with a numerical tolerance;
it is not a universal tetrad or nonlinear observability theorem.

For a state response `R(z)=Q z`, equality of its whole linear future on
every `C`-fiber is equivalent to

```text
ker(C) subset of intersection_(j=0,...,n-1) ker(Q A^j).
```

The necessity follows by differentiating `Q exp(t A) z` at zero; sufficiency
follows from Cayley-Hamilton and the exponential series. Consequently a
response can be predictable even when the whole state is not observable.
For pure EPI diffusion a uniform form shift is invisible to pressure at all
times, but is needed if the requested response includes absolute EPI.

The oscillator block `x'=p, p'=-x` makes the distinction concrete. Observing
`p` alone does not reconstruct `x` at one instant and does not close a
first-order `p` law. The ideal derivative `p'=-x` resolves the missing form;
the two-row observability matrix has full rank. Describing the same pressure
response by `p''=-p` retains two initial conditions, rather than eliminating
the missing state by renaming the output.

### 3.8 Consumed form, capacity, sources and event history

For the unforced instantaneous form rate, signed pressure and nonnegative
capacity give `x'=nu_f*p`. A known held product can suffice for that rate,
while separate pressure and capacity may remain necessary for their future
laws. With a live source, the complete row is `x'=nu_f*p+Gamma` and the
source must also be retained. Inferring pressure by dividing an evaluated
rate by capacity is circular and fails at zero capacity.

Holding pressure fixed while scaling capacity from one to two leaves the
full tetrad unchanged and doubles a nonzero form rate in the same declared
clock. Adding the admitted capacity resolves this particular ambiguity; it
does not recover omitted form or pressure evolution. Under a supplied
oscillator, two EPI preparations with identical pressure and capacity have
the same initial form rate but different pressure derivatives, as the
linked exact witness shows. Under fixed pure diffusion, the narrower mean
and potential reconstruction in Section 3.4 can instead be sufficient.

Event decisions can require information beyond all instantaneous primitive
values. Mutation consumes a live two-sample EPI secant and independent
grammar context. An operator-name history is categorical context, not the
timestamped or declared-step EPI evidence. The current snapshot and the
nodal-product rate cannot substitute for that observed secant. The
[event contracts](../docs/contracts/OPERATOR_EVENTS.md) and
[mutation admission controls](../tests/operators/test_collective_mutation_admission.py)
own its actual validity requirements. Hidden reservoir, boundary, selector
or memory state remains part of a model whenever its law consumes it.

Thus primitive values, derived views and history are selected against the
actual response and execution path. Neither appending a few diagnostics nor
exposing a categorical history automatically supplies a complete checkpoint
or sufficient state for every operator and continuous law.

## 4. Field scales and selected thresholds

The exact phase-wrap maximum π must be distinguished from policy values
expressed as fractions of π. The numerical values below remain unchanged in
[canonical.py](../src/tnfr/constants/canonical.py).

| Field | Exact or conditional mathematical scale | Canonical telemetry policy |
|-------|----------------------------------------|----------------------------|
| Φ_s | ‖Φ_s‖∞ ≤ ‖B_G‖∞ ‖ΔNFR‖∞ for a fixed distance kernel | Per-node π/4; U6 drift π/2 |
| \|∇φ\| | At most π by wrapped-angle definition | Warning level π/16 |
| K_φ | Absolute value at most π by wrapping | Warning level 0.9π |
| ξ_C | Fit: declared path-length units; spectral reference 1/√λ_gap: dimensionless | Fitted-length comparisons use matching path geometry and units |

### 4.1 Structural potential

The canonical exponent is α = 2 in B_ij = d(i,j)⁻². The resulting field is
linear in ΔNFR and unbounded under pressure rescaling. Even at zero phase,
unit pressure on K₄ gives Φ_s = 3. No phase-wrap argument can enforce π/4 or
π/2 for this linear field without an additional pressure/geometry constraint.

The chain sums Σ d⁻² = ζ(2) and Σ d⁻⁴ = ζ(4) are correct particular values.
They do not uniquely select exponent 2: absolute chain sums converge for all
α > 1 and independent unit-variance pressure sums for all α > 1/2. Graph
families with growing distance shells need their own summability assumptions.
The inverse-square kernel is the canonical modeling choice, not a uniqueness
theorem from those series.

### 4.2 Phase gradient

The mean of absolute wrapped neighbor differences is bounded by π. The
warning value π/16 is selected policy; a measured synchronization onset depends
on the experiment's frequency spread, topology, and coupling protocol.

Canonical aggregate coherence is

    C(t) = 1 / (1 + mean|ΔNFR| + mean|dEPI|).

It does not identify the spatial location of phase stress. It also is not
scale invariant: with dEPI = 0 and uniform |ΔNFR| = 1, C(t) = 1/2; doubling
pressure gives 1/3. Only the separate normalized dispersion statistic
C_disp = 1 − σ_ΔNFR/ΔNFR_max is invariant under proportional positive pressure
scaling when its denominator is nonzero.

### 4.3 Circular curvature

Where the circular mean is available,
K_φ = wrap(φ_i − circular_mean(neighbors)) gives |K_φ| ≤ π by definition.
The 0.9π threshold is an operational margin inside that exact bound. The
shared reader accepts every nonzero represented phasor resultant without an
arbitrary small-magnitude cutoff; exact represented cancellation on nonempty
support makes curvature unavailable. Isolates retain the documented zero
convention. These arithmetic availability conditions do not prove a
trigonometric singularity or a singularity of a continuous manifold.

### 4.4 Correlation length

The current estimator first fits pressure-derived coherence products against
graph distance. If that fit is unavailable under its admission policy, it uses a
spectral reference from the graph: the inverse square root of its smallest
normalized-Laplacian eigenvalue above `1e-9`. On an admitted connected undirected
graph this is 1/√λ₂ only when that gap clears the numerical cutoff. A fitted
correlation length need not equal that reference for every
state; exponential spatial decay is itself a fitting assumption.

The fitted statistic is an uncentered product of static pressure-only
coherences, not a connected covariance. Its acceptance rules require a finite
negative fitted slope, but do not test goodness of fit. The fitted length
scales with an independent uniform path-length rescaling; the dimensionless
normalized-Laplacian fallback does not. Keep these distinct observations and
their provenance when comparing states, scales or thresholds.

On disconnected graphs, a second-smallest eigenvalue and a smallest positive
eigenvalue need not coincide. Reports must identify the estimator/fallback and
graph regime. A failed fit or large estimate alone does not prove a phase
transition.

The [coherence observation foundations](COHERENCE_OBSERVATION_FOUNDATIONS.md)
derive exact finite-profile identities and counterexamples: a perfect prepared
exponential need not measure dynamic correlation, an accepted slope can hide
an oscillatory profile, and exact-distance binning can lose availability after
a small metric change. The optional profile observer retains discarded evidence.
Its moments have explicit algebraic dependencies. A marked-cycle winding
witness distinguishes states with the same full tetrad; a same-law oscillator
witness shows why even the expanded static profile need not close dynamics.

## 5. Diagnostic removal studies

The four established fields address accumulated pressure, local mismatch,
local curvature and a compressed spatial profile. They remain useful selected
views, without requiring all of them in every task. A non-redundancy study should construct
explicit pairs of states with identical retained readouts and a differing
omitted readout, or specify a detection problem and prove the omitted channel
changes the decision.

Descriptions such as “a blind spot appears” require the actual fixture,
readout resolution, and detection threshold. Passing many implementation
tests does not prove universal irreducibility. Nor does usefulness of each
channel prove that no additional observable can carry independent information.

A strong minimality claim is meaningful only after fixing its target state
space, equivalences, admissible measurements and cost. The unrestricted
four-field completeness claim is ruled out by the explicit missing-pressure,
capacity, form and orientation witnesses above. Restricted representations
can still have positive injectivity, closure and minimality results.

A positive example with those choices fixed is the
[P5 reflection quotient](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#a-complete-reflection-invariant-form-state-on-the-retained-p5).
Its three even coordinates and constrained quadratic hidden-mode matrix
reconstruct form up to reflection under the declared pure-EPI law. This is
not the four-field tetrad, a universal state basis, or a reduction to three
dimensions near the reflection-fixed stratum. The linked derivation owns
the injectivity, rate, decoder and field-covariance results.

### 5.1 Dependencies guide which outputs to request

| Task or retained information | Useful observation | What it does not establish |
| --- | --- | --- |
| A compact diagnostic snapshot | Existing tetrad with fieldwise availability and provenance | Complete state or a predictor |
| Pressure structure on known geometry | Signed stored pressure, or a justified potential decoder | Pressure evolution or an autonomous event rule |
| Spatial coherence organization | Resolved profile, endpoint populations, fit evidence and declared distance resolution | Independent new degrees of freedom; the moments have exact dependencies |
| Relative circular phase | Edge phasors on connected observed support; a forest is sufficient modulo component rotations | Absolute phase reference or a continuous lift history |
| Orientation on a declared cycle | Existing winding certificate | General phase reconstruction; it is derived from retained edge phasors |
| Instantaneous form response | Admitted pressure, capacity and consumed source | A finite-time forecast without the remaining laws |
| Future response or event admission | A sufficient state/history for the specified complete model and target | Universal sufficiency across different laws or inputs |

The [coherence observation owner](COHERENCE_OBSERVATION_FOUNDATIONS.md)
derives the exact dependencies among profile moments and their distance
endpoint populations. A circular phase alignment and its circular variogram
are also related by `variogram=1-alignment`; counting both as independent
channels would duplicate information. The phase-resultant magnitude already
retained by the curvature observer can help describe directional conditioning,
but is not a new dynamical law or a substitute for an error certificate.

The [post-Silence parent-child case](nodal/SINE_MAINTAINED_PARENT_CLOSURE.md#post-sha-pair-nonclosure)
makes this distinction concrete on a retained composition. A marked pair's
form and phase differences need the parent's boundary inputs even for their
first derivative. Equal pair state and equal instantaneous inputs can still
have different second derivatives. Adding those two input values therefore
does not itself close the pair dynamics; retain the relevant environmental
state or its derived history. The conditional relational identity and its
matched control belong to that model's owner.

Optimization can therefore mean retaining a lossless generating description
for the chosen task, requesting only useful derived views, and preserving
availability and conditioning. It need not mean increasing or reducing a
universal channel count. A costly or redundant view may still be convenient
for interpretation, while a smaller output is acceptable only when its lost
information does not affect the admitted task.

## 6. Energy, currents, and conservation

The tetrad can participate in energy-like diagnostics with phase and pressure
currents:

    V = ½ Σ_i (Φ_s(i)² + |∇φ|(i)² + K_φ(i)²),
    T = ½ Σ_i (J_φ(i)² + J_ΔNFR(i)²).

These are nonnegative functionals of measured fields. Their existence does not
establish state reconstruction, does not make ΔNFR equal to −∂V/∂EPI, and
does not turn all implemented operators into Hamiltonian transformations.
Conjugate coordinates belong to a separately specified substrate model.

Likewise a continuity residual or measured energy descent is evidence about
the tested trajectory. Universal Noether conservation and asymptotic stability
require the stated symmetry, flow, and Lyapunov hypotheses. The mathematical
scope is developed in [the variational note](TNFR_VARIATIONAL_PRINCIPLE.md) and
[the conservation note](STRUCTURAL_CONSERVATION_THEOREM.md); the tetrad's
canonical diagnostic status does not supply those proofs.

## 7. Symmetry-restricted research

Symmetry can restrict which sectors a graph readout detects. A statement about
the tetrad on a particular prime-ladder graph with uniform parameters must
name that graph, its action, and the parameter restrictions. Invariance under
that action constrains reach; it does not imply that the tetrad spans every
vector in the invariant sector.

The REMESH-∞ and Riemann programs use separate projection and observability
constructions. Their particular graph certificates do not establish a
universal complete “smooth structural sector” on arbitrary networks.
See [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) and
[TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) for the precise
program assumptions and open conjectures.

## 8. Verification scope

| Claim | Supported scope |
|-------|-----------------|
| Four canonical field APIs exist | Definitions and implementation contracts |
| Phase magnitudes are at most π | Exact wrapped-angle bounds |
| Potential is linear in pressure | Exact fixed-kernel identity |
| Potential has a uniform π-fraction maximum | False without additional pressure/geometry assumptions |
| Three operations bound every diffusion relaxation time | False; the 21-node path is a counterexample |
| Laplacian compositions prove tetrad completeness | False inference; operator generation is not reconstruction |
| Tetrad is a universal minimal sufficient state statistic | False on the admitted general state families; task-specific quotients require explicit injectivity and dynamics proofs |
| Full Φ_s plus one zero-mode scalar reconstructs EPI | Conditional on `rank(-K L_rw)=N-1`, fixed graph and pure EPI pressure |
| Rank condition holds on every connected graph | Open; measured on all 142 connected simple graph-atlas cases through six nodes |
| Full-rank potential reconstruction is uniformly well-conditioned | False across unrestricted weights; extreme weighted stars are arbitrarily ill-conditioned candidates |
| Full potential always reconstructs signed pressure | False on directed supports; the explicit outward-star kernel loses center pressure |
| A spanning forest reconstructs circular phase | Exact modulo one component rotation; `n-k` individual edge phasors are sufficient and minimal in that measurement class |
| Cycle winding adds an independent coordinate to full edge phasors | False; it is a branch-aware function of those phasors |
| Temporal observability implies snapshot injectivity or first-order closure | False; the oscillator pressure output separates all three questions |

Operator gains, discretization steps, numerical clamps, and safety thresholds
remain configured policies unless an explicit derivation with its assumptions
is supplied. A formula using π alone is not sufficient evidence of a physical
necessity.

## References

- [AGENTS.md](../AGENTS.md): canonical synthesis.
- [Field definitions and APIs](../docs/STRUCTURAL_FIELDS_TETRAD.md).
- [Mathematical scope and exact witnesses](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).
- [Unified grammar](UNIFIED_GRAMMAR_RULES.md).
- [Fundamental theory](FUNDAMENTAL_THEORY.md): broader theory context.
