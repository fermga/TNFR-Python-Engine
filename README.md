# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TNFR is a Python framework for coherent-pattern analysis on graph-coupled
networks. Every node carries form (EPI), structural frequency (`nu_f`, in
`Hz_str`), and phase. Its evolution is organized by the nodal equation

$$
\frac{\partial \mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}(t).
$$

The repository implements 13 canonical structural operators, grammar U1-U6,
network telemetry, the structural-field tetrad, and research programs built on
those primitives. [AGENTS.md](AGENTS.md) is the canonical synthesized reference.
Mathematical scope and counterexamples are stated explicitly in the linked
theory documents.

The graph engine's scalar EPI chart accepts a raw real value or the equivalent
uniform-real `BEPIElement` representation. Its scalar projection retains the
sign; `abs(EPI)` remains the nonnegative Banach-envelope magnitude. Genuinely
nonuniform or complex BEPI payloads keep that magnitude projection for generic
read-outs and are rejected by canonical glyphs that require a real scalar EPI
coordinate, pure-EPI diffusion and scalar certificates.

```bash
pip install tnfr
```

## Quick start

```python
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)
print(net.results().summary())
print(net.tetrad().summary())
print(net.tetrad().is_safe())
```

Current deterministic output for this uniform initial state:

```text
C=1.000, Si=1.000, N=20, E=20, rho=0.105
Phi_s=0.0000, |grad_phi|=0.0000, |K_phi|=0.0000, xi_C=4.5201 (N=20)
{'phi_s_safe': True, 'grad_phi_safe': True, 'k_phi_safe': True, 'xi_c_safe': True, 'overall': True}
```

The same network exposes the principal read-outs:

```python
net.conservation()
net.symplectic_substrate()
net.rhythm()
net.resonance()
net.telemetry()
net.audit_operators()
analysis = TNFR.analyze(net)
```

Grammar-aware evolution validates operator composition before applying it:

```python
net.evolve_grammar_aware(steps=10)
```

## Canonical structure

The public structural-field tetrad is `(Phi_s, |grad phi|, K_phi, xi_C)`.

| Field | Role | Exact or scoped statement |
| --- | --- | --- |
| `Phi_s` | Global pressure aggregation | General magnitude depends on pressure and graph geometry; `pi/4` and `pi/2` are selected warning policies |
| `grad phi` | Local phase stress | Mean absolute wrapped phase difference across neighboring nodes; exact bound `pi`, with `pi/16` as the selected warning threshold |
| `K_phi` | Local wrapped phase curvature | Exact wrapped magnitude bound `pi`; `0.9*pi` is a warning margin |
| `xi_C` | Non-local correlation range | Spectral estimate scales as `1/sqrt(lambda_2)` under its documented hypotheses |

`Phi_s` uses explicit edge `length` for path geometry when available; otherwise
it retains `weight` as a compatibility fallback. EPI diffusion always reads
`weight` as conductance, so models with distinct geometry and transport should
declare both attributes.

For small phase spread on a consistent branch and matching weight conventions,
`K_phi` agrees with the random-walk Laplacian applied to phase. The EPI channel
of `Delta NFR` is exact graph diffusion. These statements do not make every
pressure channel a linear Laplacian or make the tetrad a complete state
reconstruction theorem. See
[Structural Fields](docs/STRUCTURAL_FIELDS_TETRAD.md) and
[Minimal Structural Degrees](theory/MINIMAL_STRUCTURAL_DEGREES.md).

Operators modify four nodal channels:

- capacity `nu_f`: Silence, Expansion, Contraction;
- pressure `Delta NFR`: Coherence, Dissonance, Self-organization, Transition;
- phase: Coupling, Mutation;
- form EPI: Emission, Reception, Resonance, Recursivity.

The authoritative contracts live in
[`operator_contracts.py`](src/tnfr/operators/operator_contracts.py). Grammar
classifications are derived in
[`physics_derivation.py`](src/tnfr/config/physics_derivation.py), materialized in
[`grammar_canon.py`](src/tnfr/operators/grammar_canon.py), and exposed through
[`grammar.py`](src/tnfr/operators/grammar.py).

The operator registry is the canonical semantic interface for named
transformations. Declared numerical solvers may advance EPI only through the
shared nodal-equation integrator from explicit `nu_f` and `DeltaNFR`, with
provenance or a residual; ad hoc state assignment is outside the engine
contract.

SDK words preserve operator order. All thirteen glyph stages use one immutable
stage-start snapshot, validate every target proposal, and commit atomically:
neighbour-reading Reception, Coherence and Resonance; pointwise Emission,
Silence, Expansion, Contraction, Mutation and Transition; Coupling's
overlapping phase/topology proposal; Dissonance's overlapping pressure
proposal; Self-organization's child-support and hierarchy merge; and
Recursivity's deduplicated network advisory.
Coherence contracts each target's pressure magnitude and locks its phase from
that shared snapshot;
its canonical stage/global and radius-local structural `C(t)` fields are
reported separately from the retained legacy pressure-dispersion fields. The
supported GPU Emission and Resonance strategies reuse the corresponding shared
stage. When grammar accepts the requested glyph for every target, these
positions have two-phase Jacobi semantics. Their committed primary structural
channels are target-order invariant **before** the opaque pressure-refresh
callback. Emission and Silence bind every target to one shared stage timestamp.
Transition binds `nu_f`, phase, `DeltaNFR` and per-node RNG progress to the
snapshot; it resolves a missing graph seed inside the transaction, persists it
only after success, and uses one shared instant for every latency calculation.
Stable per-node offsets and draw counts make the committed RNG progress
target-order invariant. Ordered lifecycle, audit/telemetry and monitor streams
retain the requested target order. Coupling merges shortest-arc phase
displacements in snapshot-node order, normalizes final phases, rechecks U3 and
coalesces accepted functional links deterministically. This merge is an explicit
engine policy, not a stability theorem. Dissonance derives every local
pressure and outgoing propagated increment from the snapshot, then reduces
overlapping incoming increments with `math.fsum` in snapshot-node order. Its
local pressure-magnitude contract is checked before propagation is accumulated:
positive incoming increments can partially cancel a negative signed pressure.
Self-organization plans every bifurcation from the snapshot, resolves colliding
child identifiers in snapshot-node rank, validates the complete detached
`sub_nodes`/`sub_epis`/`hierarchy` merge, and only then commits `d2EPI`,
`DeltaNFR` and support. IL precondition warnings are emitted only after every
fallible stage effect succeeds and still participate in rollback
when warning policy raises. Cache state and the opaque refresh remain outside
this invariance claim; identity-bearing caches remain tied to their live graph
and node objects. Relabeling equivariance remains unproved.

The Recursivity word stage keeps the node-level glyph advisory-only, merges one
graph event per telemetry step and leaves EPI, nu_f, phase, DeltaNFR and support
unchanged before pressure refresh. It never invokes the separate
`apply_network_remesh` delayed-EPI operation. That operation now exposes an
immutable `plan_network_remesh` proposal and result. Insufficient history and
empty live support are explicit side-effect-free no-ops. An applicable plan
requires exact live support in both selected history snapshots and separates
the exact represented
three-term recurrence from its nested binary64 and clipped outputs, and commits
graph-owned state atomically. Optional evidence reports weighted-mean drift, a
three-input convex disagreement bound and a conditional fixed-history gain;
it does not prove stability when history evolves across repeated calls.

A grammar replacement or Recursivity execution override falls back to the
transactional Gauss-Seidel path and is reported as such. The explicit legacy
GS runner remains diagnosable as a schedule mismatch. The centralized
stage-contract registry records read/write footprints, merge and rollback
scopes and target-order claims for all 13 operators. These guarantees are
stage-local and do not make a mixed word simultaneous. No GPU Reception
strategy is currently registered.

THOL's `subepi_amplitude_alignment` is a variance-based EPI-amplitude
diagnostic, not canonical `C(t)` and not U5. A concrete U5 target is evaluated
by `assess_u5_parent_child_coherence(..., alpha=...)`; the hierarchy and
nonnegative `alpha` must be supplied explicitly.

### Mutation temporal evidence

Mutation keeps three related quantities separate:

| Read-out | Definition | Scope |
| --- | --- | --- |
| `predicted_depi_dt` | instantaneous `nu_f * DeltaNFR` | Nodal-equation prediction; its crossing is exposed by the legacy SDK alias `near_bifurcation` |
| `observed_depi_dt` | signed two-sample EPI secant | Evidence used by the strict, non-disableable ZHIR threshold gate |
| `d2epi_dt2` | three-sample change between adjacent secant rates | Structural-acceleration diagnostic; timestamped or legacy unit-step, and not the ZHIR gate |

Physical evidence uses timestamped `(time, EPI)` records with finite increasing
time and a fresh final EPI endpoint. If supplied, it is authoritative and does
not fall back when invalid or stale. Legacy `epi_history` and `_epi_history`
instead retain a unit-operator-step interpretation and are explicitly not
resolved in physical time. Direct Mutation requires a valid observed rate
strictly above `ZHIR_THRESHOLD_XI`, together with active capacity and any
configured minimum capacity.

ZHIR supports bifurcation detection only. The legacy
`ZHIR_BIFURCATION_MODE="variant_creation"` setting is rejected before any write:
Mutation changes phase, while topology and sub-EPI creation belong to THOL.

When dynamic selection cannot support a proposed ZHIR from that evidence, it
substitutes Coherence (IL) before ordinary grammar enforcement and records the
requested and applied glyphs with the reason. The SDK whole-word runner
checks all target nodes before executing a word that contains Mutation and
rejects timestamped evidence that an earlier EPI-channel operator in the word
would make stale. `ZHIR_BIFURCATION_VF_THRESHOLD = 0.5` only controls branch
proposal; it is not the Mutation gate. `MutationTriggerCertificate` is an
immutable diagnostic, and `nodal_state()` reads the same evidence without
modifying the graph. Neither evaluates the prior-IL and recent-destabilizer
context required by U4b, and neither certifies execution readiness.
`TNFRNetwork.apply_evidence_gated_mutation()` is the high-level experiment
policy: it runs the requested ZHIR word only after the same preflight;
otherwise it executes a declared Mutation-free exploration word and records
the decision in `NetworkResults.mutation_workflows`. It never synthesizes EPI
history, and malformed evidence remains an error. Direct `apply_sequence()`
calls remain strict. See
[Mutation (ZHIR)](theory/STRUCTURAL_OPERATORS.md#91-mutation-zhir).

### Operator-event time

`build_operator_event_schedule(...)` represents each canonical operator as an
instantaneous jump and requires `m + 1` declared nodal-flow durations for `m`
events. The exact rational value of each materialized binary64 duration is
authoritative; absolute timestamps are display values, and coincident events
remain ordered by `event_index` in `hybrid_event_log`. The companion
`diagnose_continuous_relaxation_duration(...)` uses the fixed symmetric
pure-EPI diffusion certificate and rational logarithm/exponential enclosures to
decide whether one declared interval reaches a requested disagreement-energy
fraction. Both objects are read-only and do not adapt U2/U4.

`execute_operator_event_schedule(...)` binds a valid schedule to the configured
nodal integrator and the shared atomic network-stage dispatcher. It requires
the live graph clock to match every boundary exactly, rejects positive
intervals that collapse or cannot land by direct binary64 addition, freezes the
initial target tuple, and rolls back graph-owned flow and jump state on failure.
Flow boundaries provide timestamped EPI evidence; zero-duration jumps are
recorded separately and restart same-time EPI history after a state change.
Every Mutation event therefore requires a positive immediately preceding flow
whose displayed endpoint subtraction exactly equals its declared duration.

`capture_nodal_flow_state(...)` and
`certify_observed_nodal_flow_interval(...)` provide detached endpoint evidence
for one declared interval. They separate the exact rational nodal identity and
pure-EPI diffusion/quotient result from binary64 held-pressure replays. The
standalone certificate never infers runtime provenance.
Passing `include_flow_certificates=True` to the event executor captures each
positive interval immediately around the actual integrator call and returns an
`ExecutedNodalFlowInterval`. Its broader
`runtime_bound_binary64_held_pressure_interval_identified` result covers one or
more built-in Euler substeps when their represented duration sum equals the
declared interval, Gamma is `none`, clipping and extended dynamics are inactive,
support, capacity and pressure remain fixed, and the sequential binary64 replay
matches. `runtime_bound_binary64_interval_identified` and exact rational
pure-EPI affine promotion remain restricted to one Euler substep; the latter
also requires fixed symmetric nonnegative conductance with positive row
strengths, positive capacity, stored pressure `-L_rw EPI` and the exact nodal
identity. Internal held-pressure substeps do not re-evaluate pressure and do not
establish physical mesh refinement or a modal diffusion decision. No interval
field by itself certifies solver accuracy, a glyph gain or future/repeated
schedule stability.

`observe_event_local_zhir_prejump(...)` pairs one sealed executed flow offline
with a scheduled ZHIR boundary at the same exact and binary64 coordinate. It
records both the exact rational endpoint secant and the subtraction/division
performed by the live Mutation gate; those values can differ, and the latter
alone decides the strict `rate > xi` threshold. The observation is available
whether that threshold passes or fails. The paired objects do not prove common
schedule-execution provenance.

`compare_event_local_zhir_held_pressure_subdivision(...)` compares two such
observations with identical initial state, capacity, pressure, conductance,
support, duration, event coordinate and threshold but different positive
substep counts. It certifies the represented gate decision only when both
decisions agree and every exact binary64 rate difference is strictly smaller
than the baseline distance to `xi`. Equality at the margin abstains. This is an
internal held-pressure subdivision result; pressure-reevaluated refinement,
modal decisions, solver accuracy/order, U4 readiness, adaptive grammar and
future behavior remain open.
Passing `include_stage_certificates=True` implies flow capture and returns one
`ExecutedGlyphStage` for every accepted event. The stage record binds the
executor-owned pointwise certificate for AL/SHA/VAL/NUL/ZHIR/NAV, or the
all-target neighbour certificate for EN/RA, to the EPI endpoints captured around
the actual jump and to any immediately adjacent positive flows. Unsupported or
out-of-domain glyph evidence remains an explicit abstention without changing a
valid stage execution. Every accepted two-phase ZHIR stage separately preserves
one sealed `MutationStageDecisionObservation` per target, including its complete
threshold certificate, phase/regime decision, acceleration/bifurcation read-out
and U4 context. `NetworkStageResult` retains these observations even when no EPI
certificate was requested; opt-in event-stage evidence carries them forward.
Each `ExecutedGlyphStage` is value-sealed across its event, endpoints,
certificate, adjacent-flow records and Mutation observations. The execution
result requires one intact stage per committed event in the same order, so
construction or replacement cannot promote a represented-gain claim.

The result's `represented_epi_schedule_composition` is an
`ObservedRepresentedEPIScheduleComposition`. It records every positive flow and
glyph in chronological order as a `RepresentedEPIScheduleOperation`. It
publishes an exact rational gain product only when every operation exposes an
intact represented affine map, the node order and consecutive observed EPI
endpoints agree exactly, and one normalized positive rational metric spans the
whole finite trace. `represented_map_global_disagreement_contraction_certified`
then concerns those represented maps. `runtime_schedule_global_gain_certified`
remains false: endpoint binding does not identify one global executable
binary64 map or certify solver accuracy, refinement, full multichannel
stability, future schedules or repeated execution.

`execute_event_remesh_cycle(...)` adds one explicit delayed-REMESH boundary:
schedule execution, one canonical full-support pre-REMESH `_epi_hist` sample,
then `apply_network_remesh`, all within an outer graph transaction. Delay `tau`
keeps the runtime index `history[-(tau + 1)]`; no second delayed-history sample
is added after the jump. An applied EPI jump does record its same-time right
endpoint in `epi_time_history`, keeping later Mutation secants physical. One
materialized positive diagonal metric measures the three cycle-level EPI states
and REMESH stability evidence; legacy REMESH metadata keeps unweighted means.
Ordered node support, incoming history, the endpoint clock, the committed event
log, phase, the pressure hook and deterministic REMESH configuration are
protected. Edge support may change during the schedule, while delayed REMESH
observers cannot change it or any stored non-EPI alias. Consensus drift,
capacity changes and pressure
refreshes remain separate. The optional post-REMESH refresh runs once only when
the map applies. Exact observations remain authoritative when a derived float
display is `None`. This is an atomic one-cycle execution contract, not a solver,
delayed-REMESH gain or evolving-history repetition theorem.

The cycle forwards both certificate options to its event execution and can
therefore retain the finite represented flow/glyph composition there. The
separately invoked delayed-REMESH map remains outside that composition; its
one-step evidence is never inserted as another gain factor.

Each cycle result also seals a `RemeshHistoryTransitionObservation`. For bounded
history capacity `M` and exact pre-REMESH vector `x_pre`, it verifies
`H_out = tail_M(tail_M(H_in) || (x_pre,))`, including any rebuild truncation and
oldest-snapshot eviction. Local and global delayed vectors are selected
independently from `H_out`, so one lag may be available while the other is not.
A completed post-REMESH pressure callback is an operational fact only; it does
not prove that the resulting pressure satisfies `DeltaNFR = -L_rw EPI`.

`compose_event_remesh_cycle_observations(...)` binds at least two sealed results
supplied in caller order into an `ObservedEventRemeshCycleSequence`. Each sealed
`EventRemeshCycleBoundaryObservation` compares the exact recorded EPI,
capacity, pressure, phase, schedule clock and complete delayed history across
one adjacent supplied-result boundary. Its zero-based cycle indices are local
ordinals rather than runtime call identifiers. The composer rejects reuse of
the identical result object, but distinct copies do not prove that observations
came from consecutive calls or the same graph. The sequence distinguishes
equality of raw metric weights from equality of their exact normalized rays:
proportional weights share disagreement geometry but rescale its energy. It
also reports tri-state alignment between each cycle ray and its nested
represented schedule metric.
The nested schedule compositions and REMESH results remain separately exposed.
`exact_recorded_boundary_continuity_certified` concerns the sealed recorded
boundaries alone. The stronger
`exact_common_metric_cycle_sequence_certified` additionally requires every
cycle metric to share one exact normalized ray and every nested schedule to
expose that ray; `None` or `False` alignment blocks only this stronger result,
and raw metric equality is not required.

This observation does not multiply schedule and REMESH gains or establish
evolving-history repetition, a runtime-global gain, whole-sequence atomicity,
full graph or grammar-history continuity, solver accuracy, future stability or
shared execution provenance. The boundary is necessary: one explicit sequential
lag-one execution with `alpha=1`, initial EPI `(2, 0)` and prior history `(0, 2)`
alternates to `(0, 2)` and back to `(2, 0)`, although each fixed-history REMESH
certificate has zero current-state coefficient.

The companion example exercises
only the schedule and duration diagnostic, without executing this runtime
binding: [`165_operator_event_relaxation.py`](examples/02_physics_regimes/165_operator_event_relaxation.py).

## Mathematical scope

TNFR provides executable structural models, diagnostics, and reproducible
experiments. Several correspondences are exact within stated finite-graph or
linearized hypotheses; others are measured diagnostics or open conjectures.
The current scope is centralized in:

- [Unified Grammar Rules](theory/UNIFIED_GRAMMAR_RULES.md)
- [Diagnostic and Grammar Scope](theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md)
- [Minimal Structural Degrees](theory/MINIMAL_STRUCTURAL_DEGREES.md)
- [Structural Conservation Theorem](theory/STRUCTURAL_CONSERVATION_THEOREM.md)
- [Core Dynamics Research Program](theory/CORE_RESEARCH_PROGRAM.md), with its
  [diffusion stability theorem](theory/TNFR_DIFFUSION_STABILITY_THEOREM.md) and
  [scale, geometry and bridge results](theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md)

At the scale boundary, pure EPI and the fixed-branch pairwise phase realization
admit exact reversible quotient tests. The actual circular mean-of-phasors
channel closes only on a restricted lifted block-constant subspace; a `K3,3`
same-macro-state construction refutes global projected phase autonomy. At the
instantaneous coherence boundary, fixed-`N` network levels are stratified
`2N`-dimensional cross-polytope boundaries, and fixed positive capacity cuts
them to weighted `N`-dimensional cross-polytopes. Neither result proves temporal
attraction or closure under changing support.

The restricted S16 executable boundary now covers both frozen endpoints and
sampled pure-EPI trajectories. The path certificate checks every nodal update
on persistent node identifiers, the explicit-Euler modal limit and a common
switching Lyapunov metric. It separates spectral and residual tolerances and
limits cumulative positive energy variation. Its mesh comparison requires an
explicit same-dynamics declaration and keeps numerical agreement separate from
a proof of convergence.

Within that common metric, a declared affine EPI reset has a finite global
disagreement gain exactly when it preserves the consensus subspace. The engine
uses a rational quotient gain bound computed exactly on the represented
binary64 coefficients for each passing reset, with the weighted-Frobenius norm
retained as a safe fallback. Exact scalar quotient actions, including identity,
avoid the former dimension-dependent Frobenius penalty. This bound is combined
with a rationally certified lower bound for the represented diffusion decay.
Log-space composition keeps the precise rational gain product and uses separate
upward 32-bit-significand dyadic factors to bound arithmetic growth without
weakening the certificate.
The flow proof separately checks
that its materialized generator preserves the consensus subspace and requires
the stronger exact identity `A 1 = 0` so every uniform EPI field is a fixed
point. Preservation of the displayed weighted mean, `h^T A = 0`, is reported
separately. Spectral rates remain estimates. A rational log/exp enclosure
decides and bounds finite and repeated hybrid words without using caller
tolerance as a theorem gate.

For bounded time-varying capacities, the certificate constructs `W`, `D`,
`B=D-W`, its quotient gap and the Lyapunov rate rationally from the effective
binary64 conductances and declared capacity bounds. It reports the conditional
exact-real theorem, availability of a positive operational float rate, ordinary
spectral diagnostics and numerical-integration verification separately; the
last remains open because no future schedule or solver path is observed.

Reception (EN) and Resonance (RA) are the first two catalog operators connected
to this framework at both local and all-target boundaries. They share one
centralized unweighted-neighbour EPI blend even when transport conductance is
weighted. The all-target certificate assembles the simultaneous stage map and
replays a finite repeated structural trace; regression tests compare every step
with the public runtime. It proves hard-bound forward invariance for convex
ideal-real mixing and separates represented-map gain, consensus drift and RA
diffusion-metric drift. The RA audit keeps four
layers separate: the ideal-real convex blend, the represented binary64 affine
map, the actual two-stage binary64 proposal, and the accepted identity-gated
runtime snapshot. Only neighbours that individually pass U3 participate in
RA's EPI mean, phase mean, and frequency trigger; its configured phase limit
may tighten, but cannot exceed, the canonical `pi/2` gate.

RA permits the scalar EPI to move through convex mixing while preserving its
identity: a strict negative/positive crossing is rejected, exact zero is a
neutral boundary, and an established nonempty `epi_kind` cannot change (an
absent kind may be initialized). These sign and kind conditions are independent.
The runtime also requires `0 <= RA_epi_diff <= 1`, nonnegative
`RA_vf_amplification`, and `0 <= RA_phase_coupling <= 1` before mutation. A
local frequency boost generally changes the post-RA diffusion metric
`h_i=d_i/nu_i`; the fixed post-RA flow can still be certified, while a pre/post
switching claim abstains unless the represented metrics are exactly
proportional. At the proved node-local boundary, a nontrivial EPI write has the
exact pure-EPI pressure defect `delta L_rw e_i`. An all-target stage instead
uses the aggregated defect `L_rw delta`: simultaneous row changes can cancel,
including into a uniform EPI shift, so nontrivial target changes alone do not
imply a nonzero aggregate defect. The shared stage invokes the configured
pressure refresh inside its transaction. RA phase or capacity changes can
independently require a full multichannel refresh even when the aggregate EPI
pressure defect vanishes. Separate rounding, clipping, identity gates and
multichannel effects preclude a global binary64 affinity claim. Canonical labels
alone do not supply a gain.

For AL/SHA/VAL/NUL/ZHIR/NAV, the shared pointwise executor can opt into a
three-level certificate computed from its own detached snapshot and frozen
proposals before commit. A successful `NetworkStageResult` separates exact
represented EPI realization, affine gain in the pre-flow metric, and an
aligned pre/post diffusion metric. Certification rejects unsupported, empty,
grammar-replaced and noncanonical stages before live writes; NUL pressure
effects remain a separate diagnostic. The event runtime can bind these
certificates and the EN/RA stage certificates into one finite represented-map
composition under exact endpoint and common-metric gates. This does not produce
a global binary64 runtime-map or repeated-runtime theorem.

The Riemann, Navier-Stokes, Yang-Mills, P-vs-NP, BSD, and Hodge programs remain
open research programs. They do not claim solutions to the corresponding
classical problems. Their current status is indexed in
[the theory hub](theory/README.md).

## Installation

```bash
pip install tnfr
pip install -e ".[dev-minimal]"   # local development
pip install -e ".[test-all]"      # complete test tooling
pip install -e ".[compute-jax]"   # optional JAX backend
pip install -e ".[compute-torch]" # optional Torch numerical backend
pip install -e ".[docs]"          # documentation build
```

The Torch extra provides a supported numerical backend. TNFR does not currently
ship a dedicated `TNFRGPUEngine` or promise CUDA speedups.

## Repository map

```text
src/tnfr/
├── config/          # runtime configuration and physics-derived classifications
├── constants/       # canonical and operational constants
├── operators/       # operator implementations, contracts, grammar and execution
├── dynamics/        # Delta NFR computation and nodal integration
├── physics/         # tetrad, diffusion, conservation and structural diagnostics
├── metrics/         # coherence, sense index and telemetry kernels
├── core/            # service protocols, defaults and dependency container
├── services/        # orchestration facade
├── sdk/             # simple and fluent public APIs
├── engines/         # optimization and computation services
├── mathematics/     # numerical backends and arithmetic structures
└── research areas   # riemann, navier_stokes, yang_mills and related modules
```

Executable demonstrations are grouped into ten thematic folders under
[`examples/`](examples/README.md). The full architecture and source-of-truth map
are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Development and verification

```bash
python -m pytest
python scripts/verify_internal_references.py --ci
python scripts/check_documentation.py
python scripts/prepare_docs.py
python -m mkdocs build --strict
```

The configured default test run excludes tests marked `slow`. See
[TESTING.md](TESTING.md) for focused suites, optional backends, slow tests, and
reproducibility checks. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution
requirements.

## Documentation

| Resource | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Canonical synthesized TNFR reference and agent doctrine |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Implemented package boundaries and data flow |
| [docs/README.md](docs/README.md) | Technical documentation hub |
| [theory/README.md](theory/README.md) | Theory and research-program index |
| [docs/API_CONTRACTS.md](docs/API_CONTRACTS.md) | Operator contract reference |
| [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md) | Field definitions and safety-policy scope |
| [examples/README.md](examples/README.md) | Executable examples |

The published site is built from these repository sources by the documentation
workflow: [TNFR documentation](https://fermga.github.io/TNFR-Python-Engine/).

## Citation

```bibtex
@software{tnfr_python_engine,
  author = {Martinez Gamo, F. F.},
  title = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  year = {2026},
  version = {0.0.3.5},
  doi = {10.5281/zenodo.17602860},
  url = {https://github.com/fermga/TNFR-Python-Engine}
}
```

MIT licensed. See [LICENSE.md](LICENSE.md).
