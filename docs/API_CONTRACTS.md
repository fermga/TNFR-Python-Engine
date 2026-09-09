# TNFR Operator API Contracts

**Status:** Active normative view
**Repository version:** 0.0.3.5
**Owner:** [`tnfr.operators.operator_contracts`](../src/tnfr/operators/operator_contracts.py)

This document is a readable view of the canonical operator contracts. The code
module above owns channel, scale, direction, postcondition, and source anchors.
Changes begin in that module and must pass its consistency assertions and tests.

## Contract model

Every canonical operator has:

- a lowercase English execution token, a public display/class name, and an
  internal glyph;
- one primary nodal-equation channel;
- node or network scale;
- a direct-effect direction;
- a verifiable postcondition;
- independent grammar and state preconditions.

The channel classification does not replace grammar roles. For example,
Mutation acts primarily on phase while also being a U2 destabilizer and U4
transformer.

`Scale` is the U5 fractality axis: `node` acts at the current fiber/level and
`network` denotes the multi-scale REMESH echo. It is not an execution-footprint
field. Coupling remains node-scale on this axis even though one application may
write neighbouring phases and edge support; those overlaps belong to the
separate stage contract.

## Canonical contracts

| Operator | Glyph | Primary channel | Scale | Direct postcondition |
| --- | --- | --- | --- | --- |
| Emission | AL | EPI | node | EPI does not decrease; frequency, pressure and phase stay unchanged |
| Reception | EN | EPI | node | Coherence does not decrease during coherent integration |
| Resonance | RA | EPI | node | EPI structural identity is preserved |
| Silence | SHA | structural frequency | node | Structural frequency does not increase |
| Expansion | VAL | structural frequency | node | Structural frequency does not decrease |
| Contraction | NUL | structural frequency | node | Structural frequency does not increase |
| Coupling | UM | phase | node | Pressure magnitude does not increase under mutual stabilization |
| Mutation | ZHIR | phase | node | Phase is transformed when mutation preconditions hold |
| Coherence | IL | pressure | node | Pressure magnitude and coherence do not worsen |
| Dissonance | OZ | pressure | node | Pressure magnitude does not decrease |
| Self-organization | THOL | pressure | node | Global form is preserved without catastrophic coherence loss |
| Transition | NAV | pressure | node | At least one controlled state channel changes |
| Recursivity | REMESH | EPI | network | EPI mixes with two delayed per-node snapshots |

The REMESH row describes the separately invoked network operation
apply_network_remesh. Its public planner, plan_network_remesh, returns an
immutable all-node proposal. Insufficient history or empty live support is an
explicit no-op; after the history guard passes, each selected temporal snapshot
must be a node mapping with
exactly the live support. Missing values are never replaced by current EPI.

The executor returns an immutable result that exposes raw affine and bounded
values separately. Optional evidence reports the three-snapshot convex
disagreement bound, weighted-mean drift and only sufficient fixed-history gain
conditions. These fields do not infer stability from the operator name. The
commit is atomic over graph-owned state, topology, metadata, history, caches
and capturable callback state. Effects already emitted to external systems by a
callback cannot be rolled back.

A direct node-level Recursivity glyph and its shared word stage are
advisory-only: the stage derives one immutable advisory from its snapshot,
deduplicates one graph event per telemetry step and leaves structural channels
unchanged. It never triggers delayed EPI mixing implicitly.

For exact wording and measured context, inspect
[`OPERATOR_CONTRACTS`](../src/tnfr/operators/operator_contracts.py).

## Six global invariants

The canonical invariant list is owned by
[AGENTS.md](../AGENTS.md#8-canonical-invariants):

1. nodal-equation integrity;
2. phase-coherent coupling;
3. multi-scale fractality;
4. grammar compliance;
5. structural metrology;
6. reproducible dynamics.

This document does not define a parallel invariant list.

## Valid execution example

```python
from tnfr.operators.definitions import Coherence, Emission, Silence
from tnfr.structural import create_nfr, run_sequence

G, node = create_nfr("seed", epi=0.1, vf=1.0, theta=0.0)
run_sequence(G, node, [Emission(), Coherence(), Silence()])
```

The word starts with a U1 generator, contains a stabilizer, and ends with a U1
closure. State-dependent operator preconditions remain mandatory during
execution. Coupling and Resonance additionally enforce U3 before mutation.

## Operator-event timeline

[`build_operator_event_schedule`](../src/tnfr/operators/event_timing.py)
accepts canonical lowercase execution tokens. It represents operators as
zero-duration jumps and requires exactly one more declared flow interval than
events. Exact rationalized binary64 durations and offsets define physical time;
absolute float timestamps are display-only, and coincident events use their
explicit index order. The schedule validates structure without executing an
operator or writing EPI history.

[`diagnose_continuous_relaxation_duration`](../src/tnfr/physics/event_duration.py)
maps one declared interval to the existing fixed symmetric pure-EPI diffusion
certificate. Its target decision uses the exact rate and rational
transcendental enclosures; eigensolver and libm values remain estimates. This
diagnostic does not alter U2 or U4 policy.

[`execute_operator_event_schedule`](../src/tnfr/operators/event_runtime.py)
executes a valid schedule with the graph's configured nodal integrator and the
shared canonical all-target stage dispatcher. It freezes the initial target
tuple, requires exact agreement with the live binary64 clock at each boundary,
and rejects collapsed or nonadditive positive intervals before writes. One
outer graph transaction covers flows, jumps, histories, runtime caches and the
hybrid event log. Completed pressure-refresh callbacks are counted; effects
already emitted outside the graph cannot be rolled back. Flow boundaries feed
timestamped EPI evidence, while same-time jumps restart that evidence and remain
zero-duration events.

[`capture_nodal_flow_state`](../src/tnfr/physics/runtime_flow_stability.py)
captures a detached ordered endpoint containing only scalar EPI, `nu_f`,
`DeltaNFR` and effective conductance.
[`certify_observed_nodal_flow_interval`](../src/tnfr/physics/runtime_flow_stability.py)
compares two such endpoints over one materialized binary64 duration. Its exact
rational nodal residual, exact pure-EPI pressure identity, binary64
held-pressure Euler replay and exact rational quotient-gain theorem are
independent claims. Caller-supplied endpoint metadata does not prove which
runtime produced the observations.

Setting `include_flow_certificates=True` on
`execute_operator_event_schedule` binds capture to both sides of each actual
positive flow call. Each `ExecutedNodalFlowInterval` records its interval and
actual integrator provenance together with the detached certificate or an
explicit state-capture abstention; failed theorem conditions remain visible in
the nested certificate. Runtime-bound binary64 identification requires the
exact built-in `DefaultIntegrator`, Euler with one substep, live Gamma type
`none`, inactive clipping, disabled extended dynamics, stable node support,
unchanged capacity and pressure, and an exact binary64 endpoint replay. A
custom integrator or subclass, RK4, multiple substeps, any other Gamma type,
active clipping or requested extended dynamics prevents that identification.

Exact rational pure-EPI affine promotion additionally requires at least two
nodes, fixed symmetric nonnegative conductance with positive row strengths,
positive capacity, the exact stored pressure `-L_rw EPI` and the exact rational
nodal identity. Changed support, conductance, capacity or pressure blocks the
corresponding promotion. Stale pressure exposes the three-level boundary: the
held-pressure nodal identity and trusted binary64 runtime replay can pass while
the pure-EPI affine map and quotient theorem abstain.
`flow_certification_requested` distinguishes the
disabled path, while `flow_interval_evidence` stores the positive-interval
records. The tri-state
`all_positive_flow_intervals_binary64_identified`,
`all_positive_flow_intervals_exact_affine` and
`all_positive_flow_intervals_contracting` aggregates return `None` when
capture was not requested and otherwise summarize only those executed positive
intervals. They do not turn endpoint agreement into solver-accuracy,
refinement, glyph-gain or repeated-schedule evidence.

Setting `include_stage_certificates=True` is also opt-in and implies flow
capture. `glyph_stage_evidence` then contains one `ExecutedGlyphStage` for every
accepted scheduled event. Pointwise AL/SHA/VAL/NUL/ZHIR/NAV stages reuse their
executor-owned frozen-proposal certificate; EN/RA reuse their executor-owned
all-target neighbour certificate. IL/OZ/UM/THOL/REMESH and any domain-rejected
certificate expose a specific abstention reason. Each successful stage record
binds the represented certificate to detached EPI snapshots captured around the
actual jump, derives normalized exact pre/post metric rays, and checks its
immediately adjacent positive-flow endpoints and metrics. A pressure callback
that also changes EPI is observed in the right snapshot and prevents exact
endpoint binding.

`represented_epi_schedule_composition` returns an
`ObservedRepresentedEPIScheduleComposition` whenever stage capture was
requested. Its `operations` tuple contains one
`RepresentedEPIScheduleOperation` for every positive flow and every glyph in
chronological order, including explicit ineligibility reasons. Exact factors,
the common metric and their product are published only if the operation count
is complete, every operation has intact represented affine evidence, all node
orders match, consecutive observed EPI endpoints are exactly continuous, and
one normalized positive rational metric applies throughout. On success,
`represented_affine_composition_gain_certified` certifies that product and
`represented_map_global_disagreement_contraction_certified` reports whether it
is strictly below one. `runtime_schedule_global_gain_certified` is always
false. These properties do not identify a global executable binary64 map or
certify solver accuracy, refinement, full multichannel stability, future
schedules or repeated execution. The underlying flow certificate and the
operation/composition records seal their proof fields; wrapper and aggregate
properties revalidate those seals and fail closed if a factor or claim flag is
replaced.

[`execute_event_remesh_cycle`](../src/tnfr/operators/event_remesh_runtime.py)
composes one such schedule with exactly one canonical pre-REMESH `_epi_hist`
sample and one separately invoked delayed map. The outer transaction rejects
ordered node-support changes or schedule-owned delayed-history writes. It also
binds the immutable schedule result to the live event log and freezes the
endpoint clock, phase, pressure-hook identity and deterministic REMESH controls.
Edges may change during the schedule; the EPI-only delayed map and its ON_REMESH
observers must preserve the resulting edge state and all stored non-EPI aliases.

The supplied metric is materialized once, exposed even for the uniform default,
and reused for cycle-level weighted EPI observations and optional REMESH
evidence. Legacy REMESH metadata retains unweighted summaries. Exact means,
drifts and disagreements remain authoritative; an unrepresentable binary64
display is `None`. Capacity vectors, pressure vectors, schedule refreshes and
the optional post-REMESH pressure callback remain distinct. The latter runs
only when REMESH applies and is counted only after returning. Delay `tau` reads
`_epi_hist[-(tau + 1)]`, with no post-jump delayed-history duplicate. A committed
jump separately records its same-time `epi_time_history` endpoint for Mutation.
The cycle forwards `include_flow_certificates` and
`include_stage_certificates`, preserving the resulting interval, glyph and
represented finite-schedule records inside `event_execution`. The delayed
REMESH operation remains separate and its one-step evidence is never a factor
in `represented_epi_schedule_composition`. The cycle does not certify a global
binary64 runtime gain or repeated stability with evolving history. External
callback and integrator effects remain outside rollback.

Each successful cycle also returns a sealed
[`RemeshHistoryTransitionObservation`](../src/tnfr/operators/event_remesh_runtime.py).
Writing `M=history_maxlen`, `H_in` for the exact represented incoming rows and
`x_pre` for the exact pre-REMESH EPI vector, its decisive identity is
`H_out = tail_M(tail_M(H_in) || (x_pre,))`. It records container rebuild and
truncation, append eviction, and the vectors at
`H_out[-(tau_local + 1)]` and `H_out[-(tau_global + 1)]` independently. Either
lag can therefore be unavailable while the other is present. Its proof seal,
and the enclosing `EventRemeshCycleResult` seal, fail closed after replacement
or nested evidence mutation.

[`compose_event_remesh_cycle_observations`](../src/tnfr/operators/event_remesh_sequence.py)
is a pure observer over at least two sealed cycle results supplied in caller
order. It returns an `ObservedEventRemeshCycleSequence` containing one sealed
`EventRemeshCycleBoundaryObservation` per adjacent supplied pair. A passing
boundary requires intact individually atomic cycles, ordered node support, exact
schedule end/start time, post-REMESH/pre-schedule EPI, complete outgoing/incoming
REMESH history, capacity, post-refresh/pre-schedule pressure and phase equality.
When an applied left REMESH changed EPI, the observer additionally requires one
completed explicitly requested post-REMESH pressure callback. Callback
completion is operational evidence; it does not certify the constitutive
identity `DeltaNFR = -L_rw EPI`.

`cycle_indices` are zero-based local ordinals assigned by the composer, not
runtime call identifiers. Reusing the identical result object at two positions
is rejected as a self-pairing guard. Distinct value-equal copies remain
admissible, so neither that guard nor exact boundary equality proves causal
ordering, consecutive calls or shared-graph execution provenance.

The sequence records each exact raw metric vector and its normalized positive
rational ray. `raw_metric_weights_equal` requires the original vectors to be
identical; `exact_common_normalized_metric_ray` also admits exact proportional
vectors, which share disagreement geometry while using different raw energy
scales. `nested_schedule_metric_alignment` is tri-state per cycle and compares
an available nested `ObservedRepresentedEPIScheduleComposition` metric with
that cycle's ray; absence of a nested metric remains `None`. This alignment is
not silently substituted for recorded boundary continuity.
`exact_recorded_boundary_continuity_certified` checks the sealed adjacent-state
conditions without requiring a common metric. The stronger
`exact_common_metric_cycle_sequence_certified` additionally requires one exact
normalized ray and `True` alignment for every nested schedule; exact equality
of raw metric vectors is not a condition. A `None` or `False` alignment blocks
only this stronger result, not exact recorded-boundary continuity.
`schedule_compositions` and `remesh_results` preserve both evidence families
without combining their gain claims.

No sequence field multiplies schedule and REMESH gains or certifies
evolving-history REMESH gain or repetition, a runtime-global gain,
whole-sequence atomicity across separate calls, full graph-state or
grammar-history continuity, solver accuracy, shared graph provenance, or a
future-cycle theorem. A concrete obstruction uses lag one and `alpha=1`:
starting from EPI `(2, 0)` with delayed row `(0, 2)`, two empty-schedule cycles
produce `(0, 2)` and then `(2, 0)`. Both one-step REMESH records have zero
current-state coefficient, so multiplying those fixed-history factors would
contradict the observed alternating history.

## Contract verification

- [`test_operator_contracts.py`](../tests/operators/test_operator_contracts.py)
  verifies catalog coverage and direct effects.
- [`test_u3_hard_invariant.py`](../tests/operators/test_u3_hard_invariant.py)
  verifies rejection before Coupling or Resonance mutation.
- [`test_canonical_operators_modern.py`](../tests/operators/test_canonical_operators_modern.py)
  covers operator behavior and latency.
- [`test_event_remesh_runtime.py`](../tests/operators/test_event_remesh_runtime.py)
  verifies sealed exact history transitions and one-cycle boundaries.
- [`test_event_remesh_cycle_sequence.py`](../tests/operators/test_event_remesh_cycle_sequence.py)
  verifies ordered recorded-state continuity, metric rays, proof seals and
  the alternating-history obstruction to gain multiplication.
- [Grammar Physics Verification Map](grammar/PHYSICS_VERIFICATION.md) maps U1-U6
  to their implementation and scope.

## Extension rule

The catalog is fixed at 13 canonical operators. A domain feature should compose
existing operators or remain a diagnostic/morphism outside the catalog. Any
proposal to alter the catalog requires a nodal-equation channel, scale,
postcondition, grammar classification, tests, and an update to the canonical
synthesis; adding a class or registry entry alone does not establish
canonicity.

## Auxiliary spectral-expectation contract

`SpectralExpectationOperator` evaluates the Hermitian observable
`<psi|A|psi>`. Its value lies in the real spectral interval of `A`; values
above one are valid. This auxiliary value is never the structural coherence
`C(t)`, never inherits a `[0, 1]` bound, and never enters `C_steps`.

`NodeNX`, `create_math_nfr`, the dynamics runtime and the CLI expose the
canonical names `spectral_operator`, `spectral_expectation_threshold` and
`spectral_operator_expectation`. Every result payload contains `value`,
`threshold`, `passed`, `metric_kind`, `range`, `bounded`,
`provenance`, `canonical_coherence_certified=False` and
`records_to_C_steps=False`.

Historical `coherence_operator`, `coherence_threshold`, `coherence_value`,
`coherence_passed` and runtime history keys remain explicit compatibility
aliases. They mirror the same auxiliary spectral value and carry no `C(t)`
meaning. Supplying contradictory canonical and historical inputs is rejected.

The CLI accepts `--math-spectral-expectation-spectrum`,
`--math-spectral-expectation-floor` and
`--math-spectral-expectation-threshold`. The former
`--math-coherence-*` spellings remain accepted as aliases.
