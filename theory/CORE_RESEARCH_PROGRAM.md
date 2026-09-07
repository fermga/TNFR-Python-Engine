# TNFR core dynamics research program

This program maps the proposed research lines to current repository evidence.
Every line begins open and changes status only through a scoped derivation,
counterexample or reproducible measurement. The nodal equation and existing
operator contracts remain authoritative.

| Line | Question | Current status |
| --- | --- | --- |
| S1 | Stability for nontrivial dynamics | Fixed and bounded time-varying EPI diffusion derived; exact-common-metric switching plus declared affine EPI resets have a conservative hybrid gain budget; Reception and Resonance provide the first runtime realization boundaries. SDK EN/RA and GPU RA all-target stages now use a shared immutable snapshot and atomic two-phase Jacobi commit; stability of clipped, repeated and mixed-operator words remains open |
| S2 | A TNFR Lyapunov functional | Exact weighted fixed-capacity and common time-varying functionals derived; an affine reset has finite global gain exactly when it preserves the consensus subspace, with a rational Frobenius bound for composition; uniform fixed points and weighted-mean preservation are independent requirements; Reception and Resonance expose weighted-mean drift separately, and RA generally changes the post-flow metric when it amplifies local capacity; full tetrad/nonlinear-operator functional open |
| S3 | Tetrad observability and minimality | Tetrad-only prediction disproved; full `Phi_s` plus one zero-mode scalar conditionally reconstructs EPI; universal rank/minimality open |
| S4 | Spectrum of heterogeneous `nu_f` | Generalized fixed-capacity gap, time-varying bound and minimum common-metric switching-family gap derived. The spectral-coordinate executor now evaluates canonical `-L_rw EPI` in node space and the heterogeneous `nu_f` product pointwise; directed case open |
| S5 | Spectral adaptive grammar | Graph-specific Euler relaxation diagnostic derived; ZHIR now separates timestamped physical secants, instantaneous nodal prediction and three-sample acceleration, but an operator-event duration contract and adaptive policy remain open |
| S6 | Coherence phase transitions | Signed-global classifier corrected and a balanced static-ensemble finite-size protocol added; an actual transition and universality remain open |
| S7 | Dynamic nodal-topology transitions | Descriptive inverse-square-kernel transition certificates align labels with tetrad, C and Si endpoint deltas; precursor prediction open |
| S8 | Coarse-graining and closure | Exact reversible partition condition derived for pure EPI; phase, topology and nonlinear closure open |
| S9 | Renormalization flow of operators | Pure-EPI diffusion stays in-family under certified exact intertwining quotients; generic linear identities and sampled nonlinear defects are measurable; 13-operator flow open |
| S10 | Completeness of the 13 operators | **OPEN**: contract snapshots and catalog-independent quotient machinery are groundwork; the admissible transformation space and generation theorem are undefined |
| S11 | Geometry of coherence level sets | Local constitutive levels are L1 diamonds; an exact scaled product metric now covers one fixed topology/label class; cross-topology strata open |
| S12 | Dissipative-symplectic bridge | Exact decoupled metriplectic product derived, with stored-pressure consistency reported separately; nonzero physical cross-coupling open |
| S13 | Non-normality and dissonance bursts | Logarithmic-norm sign exactly characterizes fixed linear pressure growth; numerical signs within backward error abstain; finite-family prediction measured and canonical directed U2 metric open |
| S14 | Structural information geometry | A relabeling-invariant structural-state metric is exact within a declared finite simple-graph topology/label class; cross-topology, nesting and history geometry open |
| S15 | Inverse identification from telemetry | Known-target, node-level one-step signatures separate all 13 operators on declared probes and have a finite-prototype noise margin; ZHIR adds a replayable local prediction/observation residual with tri-state abstention. The EN/RA stage repair removes their insertion-order and partial-commit confound, while target localization, aggregate inversion, mixed schedules, unseen states and complete words remain open |
| S16 | Observable, stable and scale-persistent NFR structure | Endpoint and sampled-path certificates compose the restricted pure-EPI hypotheses, stable forward-Euler updates and an exact common switching metric. SDK EN/RA and GPU RA now have atomic all-target Jacobi stages; the other operator stages remain operator-major Gauss-Seidel, and inter-sample, phase, nonlinear, history and changing-support dynamics remain open |

## Implementation map

The program is centralized here; implementation and falsification tests remain
next to the subsystem they certify.

| Lines | Executable certificate or diagnostic | Primary tests |
| --- | --- | --- |
| S1, S2, S4, S5 | [`structural_diffusion.py`](../src/tnfr/physics/structural_diffusion.py), [`hybrid_operator_stability.py`](../src/tnfr/physics/hybrid_operator_stability.py), [`reception_realization.py`](../src/tnfr/physics/reception_realization.py), [`resonance_realization.py`](../src/tnfr/physics/resonance_realization.py), [`fft_engine.py`](../src/tnfr/dynamics/fft_engine.py) | [`test_heterogeneous_diffusion_stability.py`](../tests/physics/test_heterogeneous_diffusion_stability.py), [`test_hybrid_operator_stability.py`](../tests/physics/test_hybrid_operator_stability.py), [`test_reception_realization.py`](../tests/physics/test_reception_realization.py), [`test_resonance_realization.py`](../tests/physics/test_resonance_realization.py), [`test_fft_nodal_product.py`](../tests/test_fft_nodal_product.py) |
| S3 | [`observability.py`](../src/tnfr/physics/observability.py) | [`test_tetrad_observability.py`](../tests/physics/test_tetrad_observability.py) |
| S6 | [`phase_transition.py`](../src/tnfr/physics/phase_transition.py), [`phase_scaling.py`](../src/tnfr/physics/phase_scaling.py) | [`test_phase_transition.py`](../tests/physics/test_phase_transition.py), [`test_phase_scaling.py`](../tests/physics/test_phase_scaling.py) |
| S7 | [`topology_transitions.py`](../src/tnfr/physics/topology_transitions.py) | [`test_topology_transitions.py`](../tests/physics/test_topology_transitions.py) |
| S8, S9 | [`structural_morphism.py`](../src/tnfr/physics/structural_morphism.py), [`operator_quotient.py`](../src/tnfr/physics/operator_quotient.py) | [`test_epi_coarse_graining.py`](../tests/physics/test_epi_coarse_graining.py), [`test_operator_quotient.py`](../tests/physics/test_operator_quotient.py) |
| S10 | [`operator_contracts.py`](../src/tnfr/operators/operator_contracts.py), [`operator_quotient.py`](../src/tnfr/physics/operator_quotient.py) | [`test_operator_contracts.py`](../tests/operators/test_operator_contracts.py), [`test_operator_quotient.py`](../tests/physics/test_operator_quotient.py) |
| S11, S14 | [`coherence_geometry.py`](../src/tnfr/physics/coherence_geometry.py), [`structural_state_distance.py`](../src/tnfr/physics/structural_state_distance.py) | [`test_coherence_geometry.py`](../tests/physics/test_coherence_geometry.py), [`test_structural_state_distance.py`](../tests/physics/test_structural_state_distance.py) |
| S12 | [`metriplectic.py`](../src/tnfr/physics/metriplectic.py) | [`test_metriplectic_product.py`](../tests/physics/test_metriplectic_product.py) |
| S13 | [`nonnormal_prediction.py`](../src/tnfr/physics/nonnormal_prediction.py) | [`test_nonnormal_prediction.py`](../tests/physics/test_nonnormal_prediction.py) |
| S15 | [`temporal_identifiability.py`](../src/tnfr/physics/temporal_identifiability.py), [`mutation_trigger.py`](../src/tnfr/physics/mutation_trigger.py) | [`test_temporal_identifiability.py`](../tests/physics/test_temporal_identifiability.py), [`test_mutation_trigger.py`](../tests/physics/test_mutation_trigger.py) |
| S16 | [`core_research_integration.py`](../src/tnfr/physics/core_research_integration.py) composes frozen endpoints; [`core_research_trajectory.py`](../src/tnfr/physics/core_research_trajectory.py) adds the sampled temporal boundary; [`network_stage.py`](../src/tnfr/operators/network_stage.py) supplies atomic two-phase EN/RA stages and labels the remaining schedule | [`test_core_research_integration.py`](../tests/physics/test_core_research_integration.py), [`test_core_research_trajectory.py`](../tests/physics/test_core_research_trajectory.py), [`test_network_stage_schedule_diagnostic.py`](../tests/sdk/test_network_stage_schedule_diagnostic.py) |

## Current restricted results

Lines S1, S2 and S4 share one exact restricted result:
[Heterogeneous EPI diffusion stability theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md).
It proves exponential convergence in the metric `diag(d_i/nu_f_i)` for fixed
positive capacities and a common Dirichlet-energy bound for arbitrary
time-varying capacities inside positive finite bounds. The executable
time-varying result is an exact-real continuous theorem induced by the effective
binary64 conductances and bounds: its rational proof, its ordinary binary64
spectral diagnostics, availability of a positive published float rate, and
verification of a runtime integration path are four separate facts. The
consensus value is schedule-dependent unless capacity ratios remain fixed.
These results explicitly do not promote grammar U2 into a general convergence
theorem.

Line S3 now has both a negative and a conditional positive result. Uniform
capacity rescaling leaves the initial tetrad unchanged while changing EPI
velocity, so the tetrad alone is not predictively sufficient. On a fixed graph,
the full nodal potential observes `-K L_rw EPI`; when this operator has rank
`N-1`, one conserved zero-mode scalar completes EPI reconstruction. The rank
condition holds on all 142 connected simple graph-atlas cases through six nodes,
but its universal validity remains open. Extreme weighted stars retain algebraic
rank while becoming severely ill-conditioned, so robust inverse reconstruction
requires separate scale-aware rank thresholds and a small reconstruction
residual. The certificate distinguishes structural rank from numerical success.
Diffusion uses edge `weight` as conductance; potential geometry uses an explicit
`length` when provided and otherwise preserves `weight` as a compatibility
fallback. Studies with distinct physical channels must declare both.

Line S5 now has a non-normative spectral diagnostic. It computes the exact
explicit-Euler modal factor and step count for frozen symmetric EPI diffusion,
  including heterogeneous capacity and the stability interval
  `0<dt<2/lambda_max`.
The 21-node path requires 231 steps at canonical `dt` while `K4` requires 2;
U4 remains a three-position grammar policy. An adaptive grammar cannot identify
these clocks until an operator-position-to-time contract is defined.

ZHIR now supplies a narrower event-local temporal boundary shared by S5 and
S15. The nodal equation predicts the instantaneous rate
`nu_f * DeltaNFR`; two fresh timestamped EPI samples measure a signed secant
over their actual interval; three strictly ordered samples separately support
a nonuniform finite-difference acceleration. These quantities can disagree and
are never substituted for one another. Only the observed signed secant can
satisfy the strict Mutation threshold, while active capacity and U4b context
remain independent requirements. The physical prediction/observation
`rate_gap` is a finite-interval closure residual, not an acceleration or a
proof that the nodal law failed between samples. Legacy histories retain a
declared unit-operator-step basis and therefore cannot produce that physical
residual. Missing, malformed or stale evidence yields abstention rather than a
fabricated non-crossing. This makes one operator trigger replayable and gives
inverse identification an additional temporal channel; it still does not
define how long an operator position lasts or derive the three-position U4b
window from graph relaxation.

The next restricted extension admits topology switching. A finite family of
connected symmetric graphs has a common quadratic Lyapunov function when all
`d_i/nu_i` metric vectors are exactly proportional. The implementation checks
that projective relation by rational cross-products of the represented binary64
values, separately from normalized caller-tolerance proximity; only the former
promotes its finite family to this theorem. The minimum generalized gap across
the family gives an exponential envelope under arbitrary switching. A family
with changing degree and fixed capacity supplies a counterexample to the
common-metric hypothesis, while leaving stability outside this theorem open.

Lines S1 and S2 now cross one further boundary without turning grammar labels
into dynamics. For a declared affine EPI reset `x+ = Ax+b` in the same positive
metric, a finite global disagreement gain exists exactly when `A1` and `b` are
uniform. Failure produces an explicit consensus input with zero incoming and
positive outgoing disagreement energy. In the passing case the sharp gain is
the induced norm of `QAQ`; the executable certificate uses a rational
weighted-Frobenius upper bound computed exactly on the represented binary64
coefficients so finite hybrid flow/reset words can be composed
without trusting a rounded SVD estimate or a caller-declared multiplier. A
repeated positive-duration word contracts disagreement when its cumulative
log-gain is smaller than the certified diffusion decay. Exact preservation of
the weighted mean additionally fixes the limiting consensus to the initial
weighted mean. The result says nothing about whether a runtime implementation
really equals the declared affine map, and it leaves nonlinear, phase,
pressure, topology, history and REMESH effects open. See
[`hybrid_operator_stability.py`](../src/tnfr/physics/hybrid_operator_stability.py)
and the centralized
[diffusion stability theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md#affine-reset-gain-and-hybrid-word-theorem).
The executable
[`162_hybrid_epi_stability.py`](../examples/02_physics_regimes/162_hybrid_epi_stability.py)
records a mean-preserving amplification absorbed by diffusion, a uniform
translation that separates disagreement decay from consensus drift, and the
exact infinite-gain witness for a local offset.

Reception (EN) and Resonance (RA) now supply the first two runtime realization
bridges for this generic result. Their centralized kernel blends the target
with the **unweighted** arithmetic mean of runtime neighbours, even when the
pure-EPI transport uses weighted conductance. On fixed connected undirected
support, with scalar or uniform-real BEPI input, a convex mix, bounded values,
and inactive hard clipping, the ideal-real EPI action is affine and preserves
constant fields. A nontrivial local blend nevertheless changes every positive
weighted mean as a global functional; disagreement contraction and consensus
drift remain separate questions.

RA narrows that neighbour set before averaging: only nodes satisfying the
individual circular U3 test contribute to its EPI mean, circular phase mean,
or frequency trigger. Its graph phase limit must lie in `[0, pi/2]`, so it may
tighten but cannot relax the canonical gate. Before any state or tracking
mutation, RA also requires `RA_epi_diff` in `[0,1]`, a nonnegative
`RA_vf_amplification`, and `RA_phase_coupling` in `[0,1]`. Its identity contract
does not freeze the scalar EPI. It rejects a strict negative-to-positive or
positive-to-negative proposal, treats exact zero as a neutral sign boundary,
and preserves an established nonempty `epi_kind`; an absent kind may be
initialized. Sign and kind are independent conditions.

The RA certificate keeps four statements separate: the ideal-real convex EPI
blend, the matrix assembled from represented binary64 coefficients, the actual
two-stage binary64 proposal at the supplied snapshot, and the accepted snapshot
after the identity gate. The represented row is promoted to the affine jump
theorem only when its consensus identity passes exactly. Snapshot agreement
with either affine layer is diagnostic; separate rounding, clipping, gates,
phase, and capacity changes prevent any global binary64 runtime-affinity claim.

When an accepted RA amplifies only the target frequency, it generally changes
the diffusion metric `h_i=d_i/nu_i`. The fixed post-RA flow can still receive a
heterogeneous diffusion certificate, and the represented jump is audited in
that post-RA metric. A theorem for arbitrary switching between the pre/post
generators abstains unless their represented metric vectors are exactly
proportional. Optional recovery composes the represented jump with the fixed
post-RA flow; its break-even duration is display-only and its Boolean verdict
comes from the exact hybrid composer.

Both bridges expose the same state-consistency boundary. If the stored pure-EPI
pressure before an accepted update is `p=-L_rw x` and the target changes only
coordinate `i` by `delta`, retaining the old pressure leaves the exact defect
`delta L_rw e_i`. On connected positive conductance this vanishes exactly iff
`delta=0`. Every accepted nontrivial EN or RA EPI update must therefore refresh
pressure before a subsequent segment is interpreted as pure-EPI flow. The
binary64 defect norm is a separate snapshot diagnostic. See
[`reception_realization.py`](../src/tnfr/physics/reception_realization.py) and
[`resonance_realization.py`](../src/tnfr/physics/resonance_realization.py).

Lines S8 and S9 now have an exact pure-EPI scale result. The reversible
partition average `R` closes the macro equation exactly iff it intertwines the
micro and quotient generators. Exact quotients remain in the
`diag(nu_f)L_rw` family; non-equitable partitions expose unresolved within-block
modes. This is structural transport under U5, not an additional operator and
not a proof that all 13 operators close.

Lines S11 and S14 have a local exact result. Since
`C=1/(1+|DeltaNFR|+|dEPI|)`, every finite non-equilibrium constitutive level is
an L1 diamond with four nonsmooth vertices. Coherence therefore supplies an
exact L1 distance to equilibrium but no preferred smooth global information
metric.

They now also have an exact fixed-topology state result. Assign effective
positive scales to EPI, `nu_f`, circular phase, `DeltaNFR`, `dEPI`, edge
conductance and structural edge length; take their scaled Euclidean product over
nodes and edges; and minimize over every topology- and declared
label-preserving isomorphism. The compatibility API lets an omitted length scale
reuse the conductance scale; dimensional studies should declare both. The
resulting quotient distance is a metric on finite simple-graph state isomorphism
classes within that topology/label class.
It is invariant under node relabeling and phase wrapping and reports, rather
than hides, each snapshot's nodal-equation residual. It counts all metric
minimizers, uses an explicit lexicographic component refinement, and withholds
the node mapping if that refinement is still ambiguous. Factorial isomorphism
enumeration, changing topology, multigraphs, nested EPI identity and histories
remain outside this result. See
[`fixed_topology_structural_state_distance`](../src/tnfr/physics/structural_state_distance.py).

The spectral-coordinate executor supplied a separate negative implementation
result. Transforming `nu_f` and `DeltaNFR` independently and multiplying their
graph-Fourier coefficients is not the transform of their nodal product; even a
constant `nu_f` is concentrated in the zero mode. Moreover, the orthonormal
eigenbasis diagonalizes `L_sym`, while canonical EPI pressure is `-L_rw EPI`.
The corrected path reconstructs the scalar EPI field, applies the fixed
canonical random-walk operator, multiplies by heterogeneous capacity pointwise,
then projects the rate back into the orthonormal coordinates. An irregular
weighted graph with an isolated node is the independent regression oracle:
the executor agrees with `structural_diffusion_operator`, and the isolate has
zero pressure and rate. This establishes solver fidelity for one explicit-Euler
step; it does not make the dense full-basis implementation asymptotically fast.

Network-stage scheduling now has a bounded positive result and an explicit
remaining boundary. SDK words and supported GPU blocks remain operator-major
across glyph stages. Within SDK EN/RA and GPU RA stages, every target reads the
same detached stage-start graph; the executor preflights all targets, builds and
validates all proposals, then performs one transactional commit of state,
histories, grammar bookkeeping, telemetry and the pressure refresh. These stages therefore have
two-phase Jacobi semantics, are insertion-order invariant on the executable
fixtures, and roll back completely when validation or refresh fails.

This repair is stage-local. It does not make a mixed word simultaneous or prove
arbitrary relabeling equivariance. The other operator stages retain
operator-major Gauss-Seidel semantics until their cross-target read/write sets
and merge laws are specified. The positive and failing cases live in
[`test_network_stage_schedule_diagnostic.py`](../tests/sdk/test_network_stage_schedule_diagnostic.py)
and the GPU canonical-execution tests.

Line S12 now has an exact baseline: the harmonic substrate and symmetric EPI
diffusion form a metriplectic-style direct product with conserved substrate
energy and decreasing Dirichlet energy. Its cross tensors vanish. The
certificate separately compares stored `DeltaNFR` with the pressure implied by
the EPI channel; agreement is diagnostic and is not required by the decoupled
product identity. Deriving a nonzero coupling compatible with both degeneracy
identities remains the actual bridge problem. The scale, stability,
observability and coarse-graining results form the first conditional S16 chain.
Full treatment is centralized in
[Exact scale, coherence-geometry and bridge results](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md).

Line S16 now has an executable intersection certificate rather than a narrative
concatenation. For two independently frozen states with identical node and bare
edge support, one declared partition and explicit state-channel scales, it runs
the stability, EPI reconstruction, reversible quotient and structural-distance
certificates at both endpoints. It also checks that stored pressure is the pure
EPI channel and that `dEPI=nu_f*DeltaNFR`. The joint Boolean is true only when
all fifteen named numerical conditions pass; failures remain individually
inspectable. A passing finite example demonstrates a nonempty numerical
intersection under the declared tolerance; it does not certify a trajectory
between endpoints. See
[`certify_core_research_integration`](../src/tnfr/physics/core_research_integration.py).
The executable
[`160_core_research_integration.py`](../examples/02_physics_regimes/160_core_research_integration.py)
constructs a nontrivial passing case and emits the fifteen decisions together
with a domain-neutral reproducibility manifest.

The time-resolved extension accepts at least two strictly ordered snapshots.
Every adjacent pair must pass the endpoint certificate. For each persistent
node identifier and interval it separately evaluates the declared forward-Euler
defect

`r_i^k = EPI_i^(k+1) - EPI_i^k - dt_k nu_i^k DeltaNFR_i^k`.

The scaled L-infinity defect must fit the trajectory tolerance, and every step
must lie inside its graph-specific explicit-Euler modal stability interval.
Stationary-mode resolution uses a separate dimensionless spectral tolerance
relative to the fastest rate, avoiding the earlier dimensional error of using
the path residual tolerance as an absolute rate cutoff. An algebraically exact
Euler update can still be an unstable discretization when its timestep is too
large. The certificate then
requires the exact common-metric switching theorem for the finite sampled
transport family and evaluates that common quadratic at every snapshot. Each
snapshot is reprojected onto its own instantaneous weighted consensus
coordinate before the common-metric disagreement energy is evaluated; the
first snapshot's center is not frozen. Any individual increase or the
cumulative positive variation over the entire path
beyond the declared EPI-squared scale tolerance blocks the joint temporal
result. It reports every interval certificate, all failures in the first
failing interval, the sampled energy series and whether the transport operator
was actually fixed; fixed transport is diagnostic because the common metric
theorem also admits a restricted switching family.

[`compare_core_research_trajectory_refinement`](../src/tnfr/physics/core_research_trajectory.py)
certifies both the coarse and fine paths with their own strict trajectory
tolerance, then applies a separately declared agreement tolerance. Every coarse
time must have one unambiguous fine-grid match, and the fine maximum step must be
strictly smaller. The decisive EPI error compares the same persistent node ids.
The relabeling-quotient structural distance remains diagnostic: independently
minimizing a graph isomorphism at each time could otherwise hide an exchange of
nodal identity. Promotion additionally requires an explicit
`same_dynamics_declared=True`; the certificate records this caller assertion
because snapshots alone cannot prove that two meshes used the same generator
and event schedule.

The reproducible
[`161_core_research_trajectory.py`](../examples/02_physics_regimes/161_core_research_trajectory.py)
integrates one non-equilibrium fixed pure-EPI generator on nested Euler meshes.
Both temporal certificates pass; the fine mesh is closer than the coarse mesh
to the independently evaluated fixed-generator semigroup at every noninitial
common time. This is a finite two-mesh measurement. It proves neither a mesh
limit nor an order of convergence, and says nothing about unsampled regimes,
phase evolution, nonlinear pressure, operators, REMESH or support changes.

Lines S10 and S15 share a further negative result. The tuple `(primary channel,
direction, scale, context)` separates eleven operators but leaves Silence and
Contraction in one equivalence class. Names, glyphs and free-text descriptions
are excluded because they would make inverse identification tautological.
Instantaneous contract telemetry is therefore insufficient for all 13
operators; a temporal or quantitative postcondition channel is necessary.
The finite-probe implementation supplies that next layer without placing names,
glyphs or contract categories in the observed features. Across the default
three deterministic heterogeneous path probes, one-step changes in EPI,
`nu_f`, `DeltaNFR`, wrapped phase, nodal velocity and graph size give 13 distinct
rows. The matrix and affine ranks are both 12, while distinctness itself is the
finite-hypothesis identification criterion. Silence and Contraction separate
because only the latter changes EPI on these probes. The tested target node and
operator candidates are supplied to the protocol, and the features contain raw
node-level changes rather than only aggregate C, Si, phase synchronization and
tetrad telemetry. This is a reproducible measurement, not a proof for unseen
states, stochastic observation laws, target localization, compositions or
arbitrary grammar words.

For the resulting finite prototype matrix, choose explicit positive feature
scales and either the scaled L-infinity or L2 norm. If `delta` is the minimum
pairwise prototype distance, the triangle inequality proves that every
observation within the strict radius `delta/2` of its generating prototype has
that prototype as its unique nearest row. The executable certificate reports
`delta`, the closest pairs and the open noise radius, and the classifier refuses
to label a midpoint tie as robust. The default operator experiment attaches the
unit-scaled L-infinity certificate after applying the same declared decimal
quantization used by the row partition. The half-separation theorem is exact.
For supplied binary64 rows and scales, exact rational comparisons determine
the closest pairs and the reported minimum is rounded downward before it is
halved, so the executable radius is conservative. Numerically unrepresentable
scaled separations are rejected instead of being mistaken for duplicate
prototypes. Unit feature scales are a coordinate convention, not a calibrated
observation model. This finite-family guarantee does not estimate a noise
distribution or cover a new graph state.
See
[`probe_canonical_operator_identifiability`](../src/tnfr/physics/temporal_identifiability.py).

Line S6 contained a semantic implementation error: phase classification used
`mean(abs(S))` and `mean(abs(chi))` while documenting `abs(mean(S))` and
`abs(mean(chi))`. Random opposing chirality was therefore promoted to
homochirality. Classification now uses signed global means; mean absolute
magnitudes remain separate local-activity telemetry. Susceptibility maxima,
correlation length and time-series exponents remain measured diagnostics, not
proofs of divergence or a universal transition class.

S6 now also has an explicit finite-size protocol. A rectangular sweep over a
shared control grid reports the sampled susceptibility maximum for each node
count, replicate standard errors, and log-log slopes of peak susceptibility,
order magnitude and coherence length against `N`. These slopes are not named
thermodynamic exponent ratios because no graph-independent map from node count
to linear size has been derived. Every exact sampled maximizer is retained:
plateaus are marked ambiguous, and a plateau touching either grid boundary is
unbracketed even when its first maximizer is interior. The legacy
`symmetry_zscore` is likewise an
operational standardized spatial imbalance: coupled graph nodes are not
independent samples, so it is not a p-value without a separate sampling model.
The bundled example samples static graph-state ensembles and can return
unbracketed boundary maxima; it demonstrates the measurement protocol and does
not itself establish a dynamical phase transition.

S7 now has a read-only finite-sequence detector. It records the canonical
radial/annular/multinodal label together with tetrad, flow, coherence and Si
snapshots, then emits endpoint differences after aligning exact relabelings.
Equal persistent node ids take precedence when both snapshots use the same
support. Node support, edge support, direction, effective conductance and
structural-length changes are reported separately, and circular phase-curvature
deltas stay in the principal wrapped interval. A transition is bracketed by
adjacent samples and is not interpolated.
State-preserving isomorphism uses all five nodal channels as well as the derived
read-outs. When graph symmetry leaves several admissible correspondences with
different pointwise changes, the certificate reports the ambiguity and withholds
mapping-dependent deltas instead of selecting the first isomorphism.
The present classifier reads unit-source graph geometry, so changing EPI,
phase or pressure on a fixed graph cannot by itself change the topology label;
causal operator attribution and universal precursors remain open.
Its radial/annular cuts are calibrated only for the canonical inverse-square
kernel, and the certificate rejects alternate exponents instead of attaching
canonical labels to an uncalibrated geometry.

S9 now separates two closure identities for a declared projection `R`, lift
`P`, micro map `F` and macro map `F_bar`. Projected autonomy requires
`R F = F_bar R`; invariance of lifted macro states requires
`F P = P F_bar`. Neither implies the other in general. Matrices obtain global
finite-dimensional residuals with explicitly tolerance-conditioned decisions,
while nonlinear callables are only tested on declared lifted and arbitrary
probes. Repeated evaluations also detect stateful behavior on those probes.
These checks can expose dependence on unresolved fibers but cannot prove a
global identity or callable purity. Graph,
history and nested-EPI mutations are explicitly outside the fixed-vector
protocol. See
[`certify_operator_quotient`](../src/tnfr/physics/operator_quotient.py).

S13 has an exact restricted linear result. For fixed directed pure-EPI flow,
`p=DeltaNFR=-Lx` obeys `p'=-Lp`. When there is one consensus mode,
`range(L)=ker(pi^T)`, so the restricted semigroup norm is the worst-case
reachable pressure gain. The Euclidean logarithmic norm
`mu_2(-L_sub)` is nonpositive exactly when that semigroup is contractive; a
positive value gives an immediately growing pressure direction. The spectral
abscissa remains an asymptotic diagnostic and cannot exclude this transient.
The implementation evaluates the mathematical sign in binary64 and publishes
positive/negative predictions only outside a matrix-derived backward-error
band. Values inside that band are `numerically_unresolved`; they are excluded
from accuracy scores and prevent a finite scan from being labeled consistent
with the exact theorem.
In the declared deterministic 16-graph family, all spectra are stable, four
graphs burst, the spectral sign rule has balanced accuracy 0.50, and the
logarithmic-norm sign rule has 1.00 on both the even-index calibration split
and odd-index holdout. These scores and magnitude rankings are finite-family
measurements. See
[`benchmark_nonnormal_prediction`](../src/tnfr/physics/nonnormal_prediction.py).

## Falsification ledger

| Line | Next precise test | Result that blocks promotion |
| --- | --- | --- |
| S1-S2 | Certify the all-target EN/RA stage maps across clipping and repetition, then derive realizations for the remaining operators | A staged runtime map leaves its declared domain, violates a gate, loses atomicity, fails pressure refresh, or silently changes the proof metric |
| S3 | Prove or refute `rank(-K L_rw)=N-1` beyond the finite atlas | One connected positive-conductance graph with nullity greater than one |
| S4 | Replace symmetric gaps by a directed contraction quantity | Stable spectrum with an unbounded claimed metric transient |
| S5 | Define operator-event duration, then compare timestamped trigger evidence and modal decay under equivalent timestep refinement | The physical trigger or duration mapping changes solely because an equivalent integration interval is repartitioned |
| S6 | Run the implemented shared-grid protocol across replicated graph families | Size slopes fail held-out family reproducibility intervals |
| S7 | Compare recorded pre-transition trajectories with matched non-transition controls | No precursor beats the matched controls out of sample |
| S8 | Extend the quotient to circular phase and changing support | Macro derivatives depend on unresolved fiber state |
| S9 | Map each fixed-state operator onto the generic quotient protocol | A proposed closed operator depends on an unresolved fiber or leaves the declared macro family |
| S10 | Define the admissible transformation space independently of the catalog | One admissible transformation cannot be generated or classified |
| S11 | Stratify the full network-level `C` map | Claimed smooth chart crosses an absolute-value singular stratum |
| S12 | Introduce nonzero cross tensors satisfying both degeneracies | Either `H` drifts or the dissipative functional increases |
| S13 | Repeat the benchmark across held-out graph families and operator-driven trajectories | The exact linear sign criterion fails in scope, or spectral baselines match finite predictive rankings out of sample |
| S14 | Extend the fixed-class metric across topology, nesting and histories with explicit edit costs | Triangle inequality, relabeling invariance or phase-wrap invariance fails |
| S15 | Estimate observation scales/noise laws, then test unseen graph families, mixed schedules and complete grammar words | Operators or words claimed identifiable have overlapping observation laws, or empirical errors exceed the certified finite-prototype margin |
| S16 | Add a third nested mesh and reference trajectories that mix Jacobi EN/RA with declared operator-major stages | A repaired EN/RA stage regains insertion-order dependence or partial commits, refinement stops reducing persistent-id error, or an intermediate state leaves the hypothesis class |

## Next working order

1. Extend the certified EN and RA realizations from local and single-stage
   boundaries to their all-target two-phase maps, including clipping and
   repetition, while tracking consensus drift and proof-metric changes.
2. Specify cross-target read/write and merge contracts for the remaining
   operators before changing their operator-major Gauss-Seidel stages; test
   insertion-order behavior, relabeling behavior and rollback independently.
3. Define an operator-event duration contract on top of the timestamped ZHIR
   evidence before converting modal solver clocks into adaptive U2/U4 policies.
4. Extend the pure-EPI quotient to circular phase, changing support and
   nonlinear operators while recording unresolved fiber dependence.
5. Test topology precursors and temporal operator signatures on held-out graph
   families, noise models and complete grammar words with declared schedules.
6. Search for nonzero dissipative-symplectic cross tensors satisfying both
   degeneracy identities, and for a canonical directed contraction metric.
7. Define admissible TNFR transformations independently of the existing catalog
   before revisiting catalog completeness.

Reusable certificate functions record their declared inputs, numerical decision
and scope. Publication artifacts must additionally attach a domain-neutral
`CoreExperimentManifest` with graph construction, capacities, solver, telemetry,
canonical `ClaimStatus`, Git revision and an explicit clean/dirty source
declaration. Seed, timestep and operator sequence are recorded when applicable;
dirty source additionally requires a SHA-256 digest of the declared working
snapshot. [`current_git_source_provenance`](../src/tnfr/research/core_manifests.py)
centralizes the scoped Git query and content digest used by executable examples.
The older
`ExperimentManifest` retains its arithmetic-specific factor and bit-size fields.
Passing tests certifies implementation; it does not promote an open statement.
