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
and rejects collapsed or nonadditive positive intervals before writes. A ZHIR
pre-flow is also rejected when subtracting its displayed endpoints would differ
from its authoritative declared duration. One outer graph transaction covers
flows, jumps, histories, runtime caches and the hybrid event log. Completed
pressure-refresh callbacks are counted; effects already emitted outside the
graph cannot be rolled back. Flow boundaries feed timestamped EPI evidence,
while same-time jumps restart that evidence and remain zero-duration events.
During schedule execution the executor exclusively owns `epi_time_history`: any
integrator write is rejected and rolls back the graph transaction, and only
declared flow boundaries, including explicit physical-segment boundaries, are
appended.

For each positive call, the integrator write footprint is limited to EPI,
`dEPI_dt`, `d2EPI_dt2` and the runtime clock. A custom integrator or subclass
must additionally realize the held-input nodal equation exactly in the
rationalized represented binary64 values:

```text
EPI_right[i] - EPI_left[i]
    = dt * nu_f_left[i] * DeltaNFR_left[i]
```

Failure to capture either endpoint or any nonzero exact residual rejects the
schedule and restores the graph. Passing this residual establishes the declared
finite nodal identity only; it does not establish solver accuracy or order.

The outer `GraphTransactionSnapshot` retains its originating graph and rejects
use with any other graph, including a value-equal copy. It restores NetworkX
structural mapping identities and their alias topology, graph-reachable mutable
state, and capturable namespace, slot and closure state owned by configured
callbacks. Snapshot preflight rejects mutable node, edge and attribute keys
whose equality or hash is not object-identity based, custom `__deepcopy__`
hooks, and unmodelled opaque interpreter/C state. Common immutable atoms,
including supported date/time, `Fraction`, `Decimal`, regular-expression and
`ZoneInfo` values, are safe atomic metadata. The rollback boundary covers state capturable from the graph;
already emitted I/O and warnings, external resources, and references reachable
only through external aliases are outside it.

[`capture_nodal_flow_state`](../src/tnfr/physics/runtime_flow_stability.py)
captures a detached ordered endpoint containing only scalar EPI, `nu_f`,
`DeltaNFR` and effective conductance.
[`certify_observed_nodal_flow_interval`](../src/tnfr/physics/runtime_flow_stability.py)
compares two such endpoints over one materialized binary64 duration. Its exact
rational nodal residual, exact pure-EPI pressure identity, one-step Euler replay,
sequential held-pressure replay and exact rational quotient-gain theorem are
independent claims. Caller-supplied endpoint metadata does not prove which
runtime produced the observations.

Setting `include_flow_certificates=True` on
`execute_operator_event_schedule` binds capture to both sides of each actual
positive flow call. Each `ExecutedNodalFlowInterval` records its interval and
actual integrator provenance together with the detached certificate or an
explicit state-capture abstention; failed conditions remain visible in the
nested certificate. The broader
`runtime_bound_binary64_held_pressure_interval_identified` property requires the
exact built-in `DefaultIntegrator`, Euler with at least one substep, live Gamma
type `none`, inactive clipping, disabled extended dynamics, stable node support,
unchanged capacity and pressure, equality between the exact sum of represented
substep durations and the declared interval, and a matching sequential replay.
A custom integrator or subclass, RK4, any other Gamma type, active clipping,
extended dynamics or a duration-sum mismatch prevents that identification.

The older one-step binary64 property and exact rational pure-EPI affine
promotion remain restricted to one Euler substep. The latter additionally
requires at least two
nodes, fixed symmetric nonnegative conductance with positive row strengths,
positive capacity, the exact stored pressure `-L_rw EPI` and the exact rational
nodal identity. Changed support, conductance, capacity or pressure blocks the
corresponding promotion. Stale pressure exposes the levels directly: the
held-pressure nodal identity and trusted binary64 sequential replay can pass
while the pure-EPI affine map and quotient theorem abstain. Internal substeps
hold pressure fixed; they are not a pressure-reevaluated physical partition and
do not carry a diffusion modal decision.

[`build_physical_flow_partition`](../src/tnfr/operators/event_timing.py)
materializes at least two strictly positive physical segments over one positive
scheduled interval. Exact represented durations must cover the parent interval,
and every binary64 segment must advance, land by direct addition and subtract
back to its declared duration. These segments are execution boundaries, unlike
an integrator's internal substeps.

Passing the partitions through `physical_flow_partitions=` makes
`execute_operator_event_schedule` refresh stored pressure before every segment
and once more at the terminal boundary. Physical execution therefore performs
`segment_count + 1` pressure callbacks and forces interval capture independently
of the general flow-certificate option. Each sealed
`PressureRefreshBoundaryObservation` binds the callback identity and before/after
state. The callback may write pressure aliases and known derived pressure/cache
outputs only. It must preserve EPI, capacity, phase, derivatives, histories,
node and edge support, every edge attribute, effective conductance, the clock,
persistent graph configuration, capturable state owned by a callable hook and
any existing non-`None` cached `_dnfr_weights`. Only the canonical default hook
may initialize missing or `None` cached weights. The same restriction applies
to stage pressure callbacks. The executor freezes hook presence and identity
for the complete schedule and checks them before invocation. A failed
preservation check, callback or later segment restores the complete graph
transaction; effects emitted outside the graph remain outside rollback.

`ExecutedPressureRefreshedFlowPartition` stores all boundary observations,
segment `ExecutedNodalFlowInterval` records and one segment-start
`PhysicalEulerModalObservation` per segment. For a fixed pure-EPI generator
`A = diag(nu_f)L_rw`, the exact-real held-pressure outer model is `I - T A`,
even when one integrator call uses several internal substeps. Its exact-real
physical counterpart is `product_k(I - h_k A)`. Actual endpoints remain
binary64 observations: a modal diagnostic alone does not identify them with
either exact-real map. The comparison checks trusted held-pressure Euler replays,
node order, capacity and conductance directly; equal decay-rate spectra do not
establish a common eigenbasis. Its ordinary binary64 evaluations of the model
mode factors are respectively `1 - T*mu` and
`product_k(1 - h_k*mu)`, which need not have the same stability decision. Exact
endpoint identification and segment gain composition additionally require
intact exact-affine certificates and one exact normalized metric ray. The modal
records certify no solver order, mesh convergence, adaptive grammar, or future
execution. The result reports physical and stage pressure callback counts
separately, and their sum must equal the total.

`flow_certification_requested` distinguishes the disabled path.
`flow_interval_evidence` stores unpartitioned positive-parent records;
`physical_flow_partition_evidence` stores each physical parent and nests its
segment records in `segment_flow_evidence`. Runtime wrappers are value-sealed,
so direct construction or later replacement cannot assert executor provenance.
The tri-state `all_positive_flow_intervals_binary64_identified`,
`all_positive_flow_intervals_binary64_held_pressure_identified`,
`all_positive_flow_intervals_exact_affine` and
`all_positive_flow_intervals_contracting` aggregates return `None` when capture
is disabled and otherwise summarize all executed positive parents through the
corresponding unpartitioned record or all-segment physical property. They do not
turn endpoint agreement into solver-accuracy, refinement, glyph-gain or
repeated-schedule evidence.

[`observe_event_local_zhir_prejump`](../src/tnfr/physics/event_refinement.py)
requires a sealed, nonempty `ExecutedNodalFlowInterval` with trusted sequential
held-pressure provenance. It pairs that record offline with a canonical
scheduled ZHIR event whose exact and represented coordinate is the flow end and
whose timestamp subtraction equals the declared duration. It reproduces the
runtime gate in its actual binary64 order—EPI subtraction, timestamp
subtraction, then division—and retains the exact rational endpoint secant as a
separate quantity. `xi` must be finite and nonnegative. A rejected threshold is
a valid observation; the object certifies neither a common originating schedule
nor operator admission or U4b readiness.

[`compare_event_local_zhir_held_pressure_subdivision`](../src/tnfr/physics/event_refinement.py)
requires two intact observations with equal event coordinate, interval index,
node order, initial EPI, capacity, pressure, conductance, duration and threshold,
and different positive substep counts. For node `i`, let `r_i` be the exact
representation of the actual binary64 observed rate and `xi` the exact
representation of its threshold. The local sufficient test is
`|r_i(candidate)-r_i(baseline)| < |r_i(baseline)-xi|`, together with observed
decision agreement. It is strict: touching the threshold margin abstains. The
sealed aggregate covers only that offline held-pressure gate comparison and
keeps all physical-refinement, modal, solver, adaptive-policy, U4 and future
claims false.

[`observe_event_local_zhir_physical_prejump`](../src/tnfr/physics/event_refinement.py)
requires one intact pressure-refreshed partition and a coordinate-paired
scheduled ZHIR event. Mutation's authoritative observed rate is the secant over
the terminal physical segment because those are the last two live history
samples. The object also records an offline secant over the complete parent
interval; that longer secant never substitutes for the live gate.

[`observe_executed_event_local_zhir_physical_prejump`](../src/tnfr/physics/event_refinement.py)
accepts an intact `OperatorEventExecutionResult` with stage certification and a
committed ZHIR `event_index`. It identifies the unique scheduled and executed
event, preceding physical parent, glyph stage, ordered target decisions and
trigger certificates, and requires the stage's pre-flow evidence to be the
terminal segment of that same parent. The resulting
`ExecutedEventLocalZHIRPhysicalPrejumpObservation` revalidates those objects by
identity and certifies common provenance for this one executor-owned finite
window. The coordinate-paired observer above remains the reusable offline
component and does not by itself certify that the partition and jump came from
one execution.

[`compare_event_local_zhir_physical_refinement`](../src/tnfr/physics/event_refinement.py)
pairs this observation with an intact held-pressure baseline sharing the initial
binary64 state, capacity, pressure, conductance, duration, event coordinate and
threshold. It reports physical-minus-baseline endpoint and rate differences,
both gate decisions, the held outer modal factors and the composite physical
factors. Decision agreement is an observation rather than a prerequisite: a
physical refresh may legitimately move the ZHIR gate or change modal stability.
Common spectra and modal configuration are required before modal comparison;
otherwise it abstains. The sealed record establishes neither equivalence,
solver accuracy/order, convergence, U4 readiness, adaptive policy nor future
behavior.

Every accepted two-phase ZHIR `NetworkStageResult` also contains one ordered,
sealed `MutationStageDecisionObservation` per target, independent of the EPI
certificate option. It freezes the complete trigger certificate, capacity gate,
phase/regime decision, acceleration and bifurcation result, and U4 context before
live metadata can change. Direct construction and coherent `dataclasses.replace`
records remain unsealed. Event execution carries these observations into the
corresponding opt-in `ExecutedGlyphStage`.

`ExecutedGlyphStage` also seals its complete event, endpoint, certificate,
adjacent-flow and decision payload. `OperatorEventExecutionResult` requires one
intact stage for every committed event, preserves event order, and binds ZHIR
observation order to `target_nodes`. A nested flow abstention remains recordable
but cannot be promoted into a flow or composition certificate.

Setting `include_stage_certificates=True` is also opt-in and implies flow
capture. `glyph_stage_evidence` then contains one `ExecutedGlyphStage` for every
accepted scheduled event. Pointwise AL/SHA/VAL/NUL/ZHIR/NAV stages reuse their
executor-owned frozen-proposal certificate; EN/RA reuse their executor-owned
all-target neighbour certificate. IL/OZ/UM/THOL/REMESH and any domain-rejected
certificate expose a specific abstention reason. Each successful stage record
binds the represented certificate to detached EPI snapshots captured around the
actual jump, derives normalized exact pre/post metric rays, and checks its
immediately adjacent positive-flow endpoints and metrics. A stage pressure
callback that changes EPI or any other non-pressure state rejects the schedule
and rolls back graph-owned state before stage evidence can be published.

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
observers are read-only over all graph-owned state, including the resulting
topology, metadata, histories and stored non-EPI aliases. Capturable state
reachable through a graph-owned observer is therefore restored on failure.

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
binary64 runtime gain or repeated stability with evolving history. Capturable
graph-reachable callback and integrator state is covered by the transaction;
their emitted I/O or warnings and external-only resources or aliases are not.
The optional `physical_flow_partitions` iterable is materialized once inside
the cycle's outer graph transaction and that exact tuple is forwarded unchanged
to `execute_operator_event_schedule`. Iterator failure aborts and restores the
graph, and a stateful iterable cannot define a second partition plan; its own
external-only state is not graph rollback state.

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

[`execute_event_remesh_cycle_sequence`](../src/tnfr/operators/event_remesh_causal_runtime.py)
is the graph-mutating finite causal wrapper. Each
`EventRemeshCycleExecutionSpec` declares one cycle invocation. The executor
runs the specs in order on one graph under one outer graph transaction and
returns an `ExecutedEventRemeshCycleSequence`. Every sealed
`CausalEventRemeshCycleReceipt` binds a zero-based execution ordinal, the exact
spec object and the resulting `EventRemeshCycleResult`; in particular, the
cycle's executed schedule is the schedule carried by that spec. The enclosing
result always retains the ordinary `ObservedEventRemeshCycleSequence`. The
default `require_runtime_telescope=True` additionally requires and retains its
compatible `RuntimeRemeshScheduleSequenceObservation`. With
`require_runtime_telescope=False`, `runtime_telescope` is always exactly `None`;
this lets a word without one common affine schedule metric still produce sealed
causal evidence, and `exact_finite_energy_telescope_certified` is false. The flag and the
optional object are both proof-bound.

The new positive claim is restricted to same-invocation causal order, common
graph identity and finite graph-owned atomicity. Failure anywhere in the
declared block restores the outer graph transaction. This does not compose a
global executable schedule/REMESH gain, certify solver accuracy/order or mesh
convergence, or establish repeated or future stability. Emitted I/O, warnings,
external resources and aliases reachable only outside the graph remain outside
rollback. Caller-ordered results passed directly to either underlying observer
still have no causal provenance.

[`observe_executed_event_remesh_block_margin`](../src/tnfr/physics/runtime_remesh_schedule_block_margin.py)
accepts only an intact `ExecutedEventRemeshCycleSequence` with the compatible
runtime telescope present. Optional
`start_boundary` and `boundary_count` select one nonempty contiguous block of
its identity-bound runtime telescope. The sealed
`RuntimeRemeshScheduleBlockMarginObservation` retains those exact boundary
objects and verifies

```text
D_block = V_before - V_after = K_block + S_block,
S_block >= 0.
```

Here `K_block` is the sum of the gain-based energy-drop lower bounds and
`S_block` is the sum of represented-schedule slacks. For `V_before > 0`, the
stored normalized diagnostics are
`kappa = K_block / V_before`, the observed fraction `D_block / V_before`, and
the endpoint gain upper bound `1 - kappa`. They are `None` at zero initial
energy. `positive_normalized_block_margin_certified` means only that this
recorded block has `K_block > 0`; it is not uniform class coercivity.

No positive absolute lower bound can be uniform across equilibrium and
amplitude-scaled copies because the quadratic energy and drop scale to zero.
The remaining runtime-promotion target is a uniform positive normalized block
margin on a declared forward-invariant executor class plus a finite runtime
intrablock prefix bound. Neither target, repeated/future runtime stability, a
runtime-global gain, solver accuracy/order, mesh convergence nor full TNFR
stability is certified by this finite observer. The separate exact-model
analogue below proves both bounds conditionally from a common schedule gain.

[`observe_event_remesh_three_mesh_refinement`](../src/tnfr/physics/event_remesh_refinement.py)
is a pure observer over three already committed `EventRemeshCycleResult`
objects. Every positive interval must have physical partition evidence, and
the intermediate boundary sets must strictly contain the coarse sets while the
fine sets strictly contain the intermediate sets. The observer rejects an
incompatible schedule, ordered full support, initial EPI/capacity/phase/pressure,
captured effective conductance, normalized cycle metric, executor/integrator or
pressure-callback metadata, incoming REMESH history, or REMESH configuration.
The observer and its six record types are re-exported by the curated
`tnfr.physics` facade and remain available from their defining
`tnfr.physics.event_remesh_refinement` submodule.

The returned `EventRemeshThreeMeshRefinementObservation` retains the three cycle
and schedule-composition objects and their separate delayed REMESH results. Its
checkpoint records cover pre-schedule EPI, every physical boundary,
pre-REMESH EPI and post-REMESH EPI. Pairwise
`EventRemeshPersistentEPIError` records report exact represented component
errors and the exact `L_inf` error on persistent node identifiers at every
coarser checkpoint. ZHIR comparison is available whenever intact
`ExecutedEventLocalZHIRPhysicalPrejumpObservation` records exist for all three
runs. Optional `zhir_xi` validates that their executed thresholds match the
supplied value; omitting it does not request an abstention.
Per-parent modal factors are available only where all runs expose one identical
ordered support, capacity/conductance generator and modal spectrum.

`three_mesh_observation_certified` means that the finite inputs, strict nesting,
compatibility checks and proof seals are intact. The separate
`intermediate_fine_error_decreases_at_coarse_checkpoints` property reports a
finite non-increase with at least one strict decrease; neither property proves
solver accuracy or order, mesh convergence, Lyapunov decrease, runtime-global
gain, a combined schedule/REMESH gain, whole-three-mesh atomicity or future
behavior. `complete_reference_problem_certified` and
`epi_differences_attributable_only_to_mesh_certified` are always false: detached
cycle artifacts do not retain the complete pre-schedule graph namespace,
callback closure or RNG state, node metadata, or sub-EPI state.

[`certify_reversible_single_eigenmode_euler_reference`](../src/tnfr/physics/reversible_eigenmode_reference.py)
is the pure exact-rational reference API for one mode of a general reversible
pure-EPI generator. It accepts ordered `Fraction` sequences only. The
conductance must be square, symmetric, nonnegative and zero-diagonal, with at
least two nodes and connected positive support; every capacity must be
positive. From `d_i=sum_j W_ij` it derives
`A=diag(nu_f)(I-D^-1 W)`, `H=diag(d_i/nu_i)`, the `H`-weighted mean, and the
centered initial field. That field must be nonzero and satisfy the exact
rational identity `A*v=mu*v` for one `mu>0`; mixed modes are rejected rather
than projected.

Every supplied partition must be positive, have the same exact duration `T`
and satisfy `0 < mu*h < 1`. Successive partitions, when present, must form a
strict proper-subdivision chain. The sealed
`ReversibleSingleEigenmodeEulerReferenceCertificate` retains the exact
generator and metric, modal residual, rational enclosure of `exp(-mu*T)`,
Euler products and endpoints, and factor, `L_inf` and `H`-error-energy bounds.
One partition certifies the theorem but has no observed pairwise subdivision;
with at least two partitions the separate strict-improvement property verifies
every declared refinement. For fixed data and an admissible family with
`h_max -> 0`, the certificate exposes the conditional exact-real convergence
theorem. Its binary64-asymptotic, mixed-mode, directed/nonreversible,
changing-generator, glyph/REMESH, solver-order and full-TNFR scope properties
remain false. Direct mutation and inconsistent private resealing fail closed.

The rational exponential implementation requires `mu*T <= 4096` solely to
cap the integer-power exponent used by the enclosure. It places no bound on the
total bit size of arbitrary `Fraction` inputs or derived rational values. The
function and result class are re-exported from `tnfr.physics`, and the complete
typed surface is declared by
[`reversible_eigenmode_reference.pyi`](../src/tnfr/physics/reversible_eigenmode_reference.pyi).
The theorem and proof are centralized in
[`TNFR_DIFFUSION_STABILITY_THEOREM.md`](../theory/TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem).
The public nonregular-`P3` construction is
[`167_reversible_eigenmode_reference.py`](../examples/02_physics_regimes/167_reversible_eigenmode_reference.py).

[`observe_executed_reversible_single_eigenmode_euler_reference`](../src/tnfr/physics/runtime_eigenmode_reference.py)
is the finite executor-binding API for that exact theorem. It accepts a
nonempty ordered iterable of intact
`ExecutedPressureRefreshedFlowPartition` records. Every record must expose the
same ordered support, rationalized conductance, positive capacity, exact initial
EPI and total duration; every physical boundary must carry canonical binary64
pure-EPI pressure, and every segment must identify the trusted held-pressure
Euler replay. Every segment must satisfy `0 < mu*h < 1`, and multiple inputs
must form a strict proper-subdivision chain. The adapter derives the exact
reference certificate from those runtime inputs. It does not accept an
independently supplied reference.

For each segment, the sealed
`ExecutedReversibleSingleEigenmodeEulerPartitionObservation` stores

```text
rho_j     = p64_j - (-L_rw z_j),
eta_j     = z_(j+1) - z_j - h_j diag(nu_f) p64_j,
epsilon_j = h_j diag(nu_f) rho_j + eta_j.
```

All displayed binary64 values are interpreted as exact `Fraction` values in
these identities. The adapter verifies
`z_(j+1)=(I-h_j A)z_j+epsilon_j` and propagates endpoint defects with the full
matrix recurrence
`r_(j+1)=(I-h_j A)r_j+epsilon_j`. This full propagation is required because
`rho_j`, `eta_j` and `epsilon_j` need not remain in the reference eigenmode.
The row also retains rational signed coordinate enclosures for the
runtime-minus-continuous endpoint and exact `L_inf` and `H`-error-energy lower
and upper bounds. Exact-affine-map identification is reported independently;
it is not required when the measured represented residuals are nonzero.

The enclosing
`ExecutedReversibleSingleEigenmodeEulerReferenceObservation` binds the rows to
one internally derived exact reference and revalidates their nested execution
evidence. Its certification means only a finite offline binding of individually
executor-certified partitions. Binary64 asymptotic convergence, runtime mesh
convergence, solver accuracy/order, common causal provenance among the supplied
executions, glyph/REMESH dynamics, repeated behavior and future/full TNFR
stability remain false. Both record classes and the observer are re-exported
from `tnfr.physics`; the typed surface is declared by
[`runtime_eigenmode_reference.pyi`](../src/tnfr/physics/runtime_eigenmode_reference.pyi).
Example
[`168_runtime_reversible_eigenmode_reference.py`](../examples/02_physics_regimes/168_runtime_reversible_eigenmode_reference.py)
exercises the public API on three independently executed nonregular-`P3`
partitions.

[`observe_p2_event_remesh_reference_family`](../src/tnfr/physics/event_remesh_reference.py)
promotes exactly one compatible finite family beyond that generic observation.
It accepts three intact coarse, intermediate and fine cycle results and binds
the generic refinement record to an executor-linked runtime REMESH bridge for
each mesh. It extracts the three physical partitions, invokes the common
runtime eigenmode observer once, and requires each returned row to identify an
exact affine map with zero `rho`, `eta`, `epsilon` and endpoint defect before
applying its stronger P2/REMESH claims. The accepted domain is an event-free
effective two-node path with a
nonuniform initial mode, homogeneous positive capacity, fixed positive
conductance, pure-EPI pressure refreshed at every boundary, and exact
represented Euler segments satisfying `0 < lambda*h < 1`. Both refinements
must be proper positive subdivisions. REMESH must use unit local/global delays,
the exact initial field as its delayed row, and hard clipping on one common
nonempty scalar interval. The exact rational exponential enclosure is limited
to `lambda*T <= 4096` to cap its integer-power exponent. This cutoff does not
bound the bit size of arbitrary `Fraction` inputs.

Each `P2EventRemeshMeshReferenceObservation` retains the exact Euler factor,
the rational continuous-factor interval, both finite-mesh error bounds, the
ideal `beta=(1-alpha)^2` REMESH error scaling, the signed runtime residual and
the resulting `L_inf` runtime error bound. The enclosing
`P2EventRemeshReferenceFamilyObservation` verifies the two strict subdivision
improvements and binds all three rows to one reference problem. Its explicit
false-scope properties withhold solver order, generic mesh convergence,
binary64 asymptotic convergence, arbitrary glyph or mixed-mode dynamics, soft
clipping, changing support or metric, repeated runtime stability and future
behavior. The observer and both result classes are re-exported from
`tnfr.physics`; its complete typed surface is declared by
[`event_remesh_reference.pyi`](../src/tnfr/physics/event_remesh_reference.pyi).
Its continuous/Euler proof is the two-node homogeneous-capacity specialization
of the
[`general reversible eigenmode theorem`](../theory/TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem).
The additional REMESH equations and runtime boundary are centralized in
[`REMESH_INFINITY_DERIVATION.md`](../theory/REMESH_INFINITY_DERIVATION.md#28-effective-p2-three-mesh-reference-family).
The public `2/4/8`-segment construction is
[`166_event_remesh_reference_family.py`](../examples/02_physics_regimes/166_event_remesh_reference_family.py).

[`certify_uniform_remesh_history_stability`](../src/tnfr/physics/remesh_history_stability.py)
constructs a sealed exact-rational certificate for the distinct finite
companion recurrence

```text
x[k+1] = (1-alpha)^2*x[k]
       + alpha*(1-alpha)*x[k-tau_local]
       + alpha*x[k-tau_global].
```

Its public `UniformRemeshHistoryStabilityCertificate` exposes the combined
delay coefficients, row-stochastic companion matrix and invariant stationary
temporal distribution. For any fixed positive diagonal spatial metric,
[`observe_uniform_remesh_history_transition`](../src/tnfr/physics/remesh_history_stability.py)
checks one exact transition and the identity

```text
V_before - V_after
  = pi[0]/2 * sum_(a<b) c[a]*c[b]
      * ||Q_H*x[k-a] - Q_H*x[k-b]||_H^2 >= 0.
```

For `0 < alpha < 1`, the companion is primitive and each spatial coordinate
converges to the preserved stationary history barycenter. `alpha=0` is the
identity. At `alpha=1`, the companion is a pure-delay permutation of order
`tau_global+1`; its augmented energy is conserved and a particular orbit may
have any period dividing that order. These are theorem-level properties of the
uniform exact recurrence. They do not identify `apply_network_remesh` with its
binary64 evaluation and clipping, certify spatial consensus or zero pressure,
admit changing parameters/support/metric, or multiply a represented schedule
gain by a REMESH gain. Both functions and result classes are re-exported from
`tnfr.physics`; the full derivation lives in
[`REMESH_INFINITY_DERIVATION.md`](../theory/REMESH_INFINITY_DERIVATION.md#24-exact-finite-companion-history-stability).

[`certify_uniform_remesh_schedule_policy_stability`](../src/tnfr/physics/remesh_schedule_policy_stability.py)
accepts one intact companion certificate and an exact-rationalized common
schedule disagreement-energy gain bound `q` in `[0,1]`. It describes a
conditional family whose histories are sampled after each schedule: REMESH
first forms the next input from the stored heads, and the next schedule then
acts on that mixture. The componentwise energy envelope is therefore

```text
B_q = diag(q, 1, ..., 1) P.
```

The sealed `UniformRemeshSchedulePolicyStabilityCertificate` stores the exact
one-step envelopes and the three decisive block powers at
`L=active_max_delay+1`, verifies `B_q^L <= q*P^L` entrywise and exposes

```text
prefix gain <= 1,
L-cycle block gain <= q,
normalized block-margin lower bound >= 1-q,
V[k+n] <= q^floor(n/L) V[k].
```

`L` is a sufficient universal companion-path horizon and need not be minimal
for a particular interior coefficient choice. Strict `q<1` proves geometric
decay of spatial disagreement in every history row. It does not control their
spatially uniform temporal means. `q=1` proves nonincrease with zero certified
margin and makes no convergence decision. The caller must separately prove the
common schedule gain and consensus preservation for every member of the
intended exact family. No schedule map or runtime execution is an input, so the
certificate does not establish binary64 invariance, rounding/clipping control,
solver accuracy/order, adaptive grammar or full TNFR stability. The class and
builder are re-exported from `tnfr.physics`; example 171 exercises the strict
mixed-delay, strict pure-delay and `q=1` boundary cases.

[`certify_uniform_remesh_schedule_relative_defect_stability`](../src/tnfr/physics/remesh_schedule_relative_defect_stability.py)
accepts an intact common-`q` policy certificate and a finite nonnegative
relative signed-defect bound `eta`. For the ideal head `y`, runtime-bounded head
`z`, and Jensen input envelope

```text
J = sum_d c_d E_H(history[d]),
delta = E_H(z) - E_H(y) <= eta*J,
q_eff = q*(1+eta),
```

Jensen's inequality and the declared schedule gain imply
`E_H(schedule(z)) <= q_eff*J`. The builder delegates its matrices and block
powers to the common-gain theorem, preserving `q` as the schedule-only bound.
It rejects `q_eff>1`; `q_eff=1` gives nonincrease and zero margin, while
`q_eff<1` gives normalized block margin `1-q_eff` and the repeated exact-model
bound `q_eff^floor(n/L)`. A zero schedule gain absorbs every finite `eta`.
The defect is the signed difference between two centered energies, not the
energy of the state residual. The certificate declares rather than verifies
the per-transition hypothesis and therefore makes no runtime or future claim.

[`observe_executed_event_remesh_relative_defect_block`](../src/tnfr/physics/runtime_remesh_schedule_relative_defect.py)
accepts one intact `ExecutedEventRemeshCycleSequence` with its compatible
runtime telescope present, one intact relative-defect certificate, and an
optional nonempty contiguous boundary range. It
reuses the causal block binding and, for each selected boundary, verifies the
REMESH configuration, represented schedule gain `q_j <= q`, exact signed
defect `delta_j <= eta*J_j`, and the componentwise history-energy envelope
under `diag(q_eff,1,...,1)P`. It records exact defect ratios only when
`J_j>0`; at `J_j=0`, it checks the inequality without division. The endpoint
bound is `q_eff^floor(N/L)` for `N` selected boundaries, so an incomplete first
block retains factor one. The result certifies only that recorded finite
causal block. It does not prove that the bounds are forward invariant, recur
on later executions, establish solver accuracy/order, or stabilize the full
TNFR state.

[`observe_binary64_remesh_pair_relative_defect`](../src/tnfr/physics/binary64_remesh_relative_defect.py)
accepts three strict two-coordinate binary64 pairs plus the represented REMESH
factor and clipping interval. It invokes the same nested scalar evaluator used
by the planner and returns a sealed
`Binary64RemeshPairRelativeDefectObservation` containing

```text
D_c = beta*d_current^2 + gamma*d_local^2 + delta*d_global^2,
rounding defect = raw separation^2 - ideal separation^2,
clipping defect = bounded separation^2 - raw separation^2,
eta_pair = max(0, total defect / D_c) when D_c > 0.
```

At `D_c=0` the observer closes the deterministic equality branch without
division. The value is the exact minimum nonnegative bound for that pair; it
does not maximize over a state class. The retained normal-valued `alpha=1/2`
witness has `eta_pair=2**210-1/4`, giving the strict threshold
`q<4/(2**212+3)`. Hard clipping is pairwise nonexpansive, but that fact cannot
control the preceding rounding amplification.

[`certify_alpha_one_hard_clip_remesh_class`](../src/tnfr/physics/binary64_remesh_relative_defect.py)
accepts one nonempty ordered support, an optional positive diagonal metric,
positive local/global delays and one finite interval. It fixes `alpha=1`, hard
clipping and the runtime history capacity, then reuses the exact REMESH
companion certificate. Every represented chronological history with length
between `max(tau_local,tau_global)+1` and `history_maxlen`, exact row width,
finite binary64 entries and values inside the interval belongs to the class.
REMESH copies the global delayed row numerically, has uniform `eta=0`, and
preserves the class. Equality does not promise preservation of the signed-zero
bit. Schedule, repeated event execution, future-runtime, solver and full-TNFR
properties are explicitly false.

[`certify_half_alpha_antisymmetric_hard_clip_remesh_class`](../src/tnfr/physics/binary64_remesh_relative_defect.py)
accepts exactly two ordered nodes, an optional positive diagonal metric,
positive local/global delays and a positive represented radius
`epi_bound>=4*2**-1074`. It fixes `alpha=0.5` and the symmetric hard interval
`[-epi_bound,epi_bound]`. A represented history belongs only when its length is
between `max(tau_local,tau_global)+1` and `history_maxlen` and every row is a
finite in-interval binary64 pair `(a,-a)`. The production nested recurrence and
symmetric clamp preserve numeric antisymmetry and the interval.

The certificate stores the sharp bound `eta*=135/124`. Its executable proof
splits at `x=sqrt(D)=11*s`, where `s=2**-1074` and
`D=c**2+l**2+2*g**2`. For the large-norm tail, it stores the exact constants
`A=3*u/2+u**2/2` and `C=9/4+9*u/4+u**2/2`, with `u=2**-53`, in
`|r-y|<=A*x+C*s`; the resulting tail ratio is strictly below `eta*`. For
`x<11*s`, it exactly enumerates 6,615 bounded integer triples, retains the
3,890 with `0<D/s**2<121`, and finds the maximum at `(-3,-2,-3)`. The stored
subnormal witness has ideal amplitude `-11*s/4`, runtime amplitude `-4*s`, and
relative defect `135/124`.

`certify_schedule_relative_defect_stability(q)` composes that `eta*` through
the existing robust theorem. Its strict threshold is `q<124/259`; `q=4/9`
produces `q_eff=259/279` and normalized block margin `20/279`, equality at
`q=124/259` produces nonincrease with zero margin, and `q=9/16` is rejected.
This method certifies the conditional exact schedule theorem, not a verified
runtime schedule family. The class is REMESH-forward-invariant only. It does
not include arbitrary positive-metric-centered histories or unrestricted fixed
lattices, and it leaves graph/event binding, repeated complete-runtime
execution, solver and full-TNFR properties false.

[`certify_p2_half_reception_remesh_stability`](../src/tnfr/physics/binary64_p2_reception_stability.py)
requires an intact two-node `alpha=1` class certificate. It fixes mutual
singleton neighbor indices, the exact binary64 configured factor `0.5`, a
common hard clamp and an immutable all-target EPI snapshot. For every finite
represented pair in the interval, the two shared Reception-kernel evaluations
are numerically equal; hence their centered energy is zero in the supplied
metric and the global numeric EPI-kernel gain is `q=0`. The builder reuses the
common-`q` and relative-defect certificates with `eta=q_eff=0`. Its exact cycle
bound is one before the sufficient horizon and zero from
`tau_global+1` onward. `evaluate_binary64_schedule_pair` replays one strict
pair and checks numeric consensus and interval membership. The certificate is
global over its numeric EPI-kernel class and supports arbitrary finite kernel
repetition; it does not bind a real graph, the complete EN stage, grammar,
events, callbacks, transactions or a solver.

[`certify_executed_p2_half_reception_stage`](../src/tnfr/physics/runtime_p2_reception_stage.py)
accepts one intact P2 half-Reception kernel certificate and one intact
`OperatorEventExecutionResult`. Unless `event_index` selects it explicitly,
the execution must contain exactly one Reception event. The adapter requires
the selected event and its executor-owned `ExecutedGlyphStage` to retain
same-invocation identity, zero duration, two-phase scheduling, two exact
captured endpoints and a revalidated one-step neighbor-stage certificate. It
then verifies ordered P2 targets, mutual singleton runtime neighbors, exact
binary64 mix `0.5`, the source hard interval, unchanged capacity and effective
conductance, the source diffusion-metric ray, accepted output identity and a
bit-exact replay through the shared production mean/blend/clipping kernel.

The sealed `ExecutedP2HalfReceptionStageCertificate` therefore binds the
global `q=0` conclusion to one executed EPI stage with finite grammar admission
and whole-schedule graph atomicity. Opposite signed-zero output bits remain
valid numeric consensus. The certificate does not bind the source REMESH
history or configuration to the graph, audit all auxiliary Reception state or
raw topology, retain current live-graph identity, or certify future/repeated
runtime stability, solver accuracy or full TNFR stability.

[`certify_executed_p2_half_reception_remesh_sequence`](../src/tnfr/physics/runtime_p2_reception_remesh_sequence.py)
accepts one intact P2 half-Reception kernel certificate and one intact
`ExecutedEventRemeshCycleSequence`. It selects one Reception event per cycle,
automatically only when the event is unique, and constructs both an
`ExecutedP2HalfReceptionStageCertificate` and a
`RuntimeRemeshHistoryBridgeObservation` for that same cycle. Each selected EN
stage must span the complete schedule EPI endpoints and retain the exact P2
support, half mix, hard interval and metric ray, so the observed schedule EPI
transition has `q=0`. Each applied REMESH must use exact binary64 `alpha=1`, the
same delays, bounds, history capacity and alpha-source provenance, and must be
an exact numeric copy of its selected global-delay row with zero represented
defect.

For `N >= L = tau_global+1`, the adapter checks the active suffix of exactly
`L` chronological rows rather than every retained history row. That suffix lies
inside the source interval, while stale older rows are outside the recurrence
and outside the certificate. The result records exact zero spatial disagreement
for the observed post-horizon REMESH fields and the final committed endpoint,
and inherits same-invocation cycle order and graph-owned atomicity from the
source execution. This is a finite observed certificate. Future or unobserved
repetition, auxiliary Reception state, current live-graph binding, solver
accuracy and full TNFR stability remain explicitly false.

[`execute_p2_half_reception_remesh_policy_invocation`](../src/tnfr/physics/runtime_p2_reception_remesh_policy.py)
is the reusable execution boundary for that restricted finite protocol. A fresh
outer `GraphTransactionSnapshot` is taken before caller-owned specifications or
metric weights are materialized. Static preflight requires an intact P2
kernel certificate; the same ordered undirected two-node mutual-singleton
support and positive diffusion metric ray; positive capacity and symmetric
conductance; exact binary64
`EN_mix=0.5`; matching hard-clipped `alpha=1` REMESH controls, delays, bounds and
history capacity; an in-interval live EPI pair that is not identically zero,
an in-interval active incoming history; and
at least `tau_global+1` distinct one-cycle zero-flow
`Reception -> Coherence -> Recursivity` specifications chained to the live
clock. Physical flow partitions are excluded from this initial policy.
The incoming history may use any supported indexed container; the canonical
append rebuilds it as a bounded deque. Enough rows must exist for the first
append to satisfy `max(tau_local,tau_global)+1`, but only the last `tau_global`
incoming rows are active and therefore required to lie inside the interval.

The function passes a read-only live context to
`execute_event_remesh_cycle_sequence`; schedule materialization therefore
rederives U1a admission from the current EPI pair at every cycle start. If a
REMESH result makes that pair identically zero, the next cycle is rejected and
the whole invocation rolls back. The policy requests no affine telescope and
constructs the existing
`ExecutedP2HalfReceptionRemeshSequenceCertificate` before the outer transaction
can commit. Any preflight, execution, callback, observation or
post-certification failure restores graph-owned state. Each successful call
certifies only its own finite trace. A later call is revalidated independently;
future and unobserved repetition, auxiliary Reception-state stability, solver
properties and full TNFR stability remain false.

[`observe_runtime_remesh_history_bridge`](../src/tnfr/physics/runtime_remesh_history_stability.py)
accepts one intact, applied `EventRemeshCycleResult` and binds it to that exact
companion model. The returned `RuntimeRemeshHistoryBridgeObservation` retains
the exact represented ideal, raw and bounded heads and the signed identity

```text
bounded = ideal_companion + rounding_residual + clipping_residual.
```

It replays the nested affine expression and canonical `structural_clip` call
bit for bit. Its exact lifted augmented-energy balances separate Jensen
dissipation from the signed rounding and clipping defects. The accompanying
absolute perturbation bounds are a posteriori sufficient conditions for this
one step; the signed balances remain available when those bounds are too
conservative. Hard clipping on one common interval is nonexpansive in the
declared positive diagonal disagreement metric. Soft clipping can increase
disagreement and therefore receives no such promotion.

The word `lifted` is part of the contract: the runtime appends the pre-REMESH
head and does not immediately append the bounded result. The bridge does not
certify live history advance, repeated runtime stability, transfer of the
companion convergence theorem, schedule/REMESH composition or future behavior.
Its result class and observer are re-exported from `tnfr.physics`.

[`observe_remesh_schedule_history_transition`](../src/tnfr/physics/remesh_schedule_stability.py)
is the pure exact layer for one REMESH-head/schedule-head transition. Given an
intact companion transition, represented raw and bounded heads, a scheduled
head and a declared nonnegative bound `q` that it verifies against those
supplied heads, it retains separate raw, clipping and schedule
disagreement-energy defects. The returned
`RemeshScheduleHistoryStabilityObservation` checks

```text
exact_drop = Jensen_dissipation - raw_defect - clip_defect - schedule_defect
exact_drop = gain_based_lower_bound + pi[0] * schedule_gain_slack.
```

The lower bound is a sufficient one-step nonincrease condition. The exact
signed balance remains authoritative when that sufficient bound is negative.
The observation also records the stationary-history barycenter drift caused by
replacing the ideal head with the scheduled head. This layer has no executable
provenance and certifies neither repetition nor future stability. Its class and
observer are re-exported from `tnfr.physics`.

[`observe_runtime_remesh_schedule_sequence`](../src/tnfr/physics/runtime_remesh_schedule_stability.py)
accepts an intact `ObservedEventRemeshCycleSequence` only when its exact
recorded boundaries are continuous, every nested schedule exposes one common
normalized positive metric and the REMESH configuration is fixed. For each
adjacent pair, `RuntimeRemeshScheduleBoundaryObservation` binds the left
cycle's applied runtime bridge to the right cycle's represented schedule. The
schedule must start at the bounded REMESH head, finish at the right pre-REMESH
head and satisfy its sealed represented-map gain.

The boundary also reconstructs the next cycle's newest-first history window
and requires it to equal the scheduled head followed by the preceding
companion tail. Its exact energy record therefore applies to the runtime
values retained by those two cycle artifacts. The enclosing
`RuntimeRemeshScheduleSequenceObservation` checks cancellation of every
intermediate augmented energy and publishes the finite identity

```text
total_drop = initial_augmented_energy - final_augmented_energy
total_drop = sum(gain_based_lower_bounds)
             + sum(schedule_augmented_energy_gain_slacks).
```

Each augmented slack is `pi[0] * schedule_energy_gain_slack` in the fixed
companion metric.

This is an additive finite telescope, without multiplication of fixed-history
REMESH gains. The source sequence remains a caller-supplied ordering of
individually atomic cycles, so the result does not certify shared graph
provenance, causal succession, cross-call atomicity, a global executable gain,
repetition or future behavior. Both result classes and the observer are
re-exported from `tnfr.physics`.

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
- [`test_event_remesh_causal_runtime.py`](../tests/operators/test_event_remesh_causal_runtime.py)
  verifies same-invocation receipts, graph identity, outer graph rollback,
  nested sequence bindings, seal integrity and explicit false stability scope;
  [`test_event_remesh_causal_runtime_example.py`](../tests/operators/test_event_remesh_causal_runtime_example.py)
  checks the public facade, stub and finite causal example.
- [`test_runtime_remesh_schedule_block_margin.py`](../tests/physics/test_runtime_remesh_schedule_block_margin.py)
  verifies exact contiguous-block telescoping, normalized diagnostics, zero
  initial energy, source/boundary identity and fail-closed proof seals;
  [`test_runtime_remesh_schedule_block_margin_example.py`](../tests/physics/test_runtime_remesh_schedule_block_margin_example.py)
  checks the public facade, import order and the `139/256` versus `0` examples.
- [`test_event_remesh_refinement.py`](../tests/physics/test_event_remesh_refinement.py)
  verifies strict three-mesh compatibility, checkpoint errors, executed ZHIR
  binding, common-generator modal abstention and fail-closed proof seals.
- [`test_event_remesh_reference.py`](../tests/physics/test_event_remesh_reference.py)
  verifies the effective-P2 hypotheses, rational Euler bounds, strict proper
  subdivisions, exact REMESH scaling, runtime residual bound and fail-closed
  scope; [`test_event_remesh_reference_example.py`](../tests/physics/test_event_remesh_reference_example.py)
  executes and checks the public `2/4/8`-segment example.
- [`test_reversible_eigenmode_reference.py`](../tests/physics/test_reversible_eigenmode_reference.py)
  verifies the general reversible metric and exact-mode hypotheses, exponential
  enclosure, Euler and norm bounds, strict subdivision identities, convergence
  scope and fail-closed proof seal;
  [`test_reversible_eigenmode_reference_example.py`](../tests/physics/test_reversible_eigenmode_reference_example.py)
  verifies the facade, public stub and both nonuniform modes in example 167.
- [`test_runtime_eigenmode_reference.py`](../tests/physics/test_runtime_eigenmode_reference.py)
  verifies runtime-source binding, exact `rho`/`eta`/`epsilon` decomposition,
  full-matrix defect propagation, rational continuous-error enclosures, family
  compatibility, proof sealing and false scope;
  [`test_runtime_eigenmode_reference_example.py`](../tests/physics/test_runtime_eigenmode_reference_example.py)
  covers its facade, stub and executed `P3` report.
- [`test_remesh_history_stability.py`](../tests/physics/test_remesh_history_stability.py)
  verifies the exact companion, stationary distribution, Jensen balance,
  equality case and all three alpha regimes.
- [`test_remesh_schedule_policy_stability.py`](../tests/physics/test_remesh_schedule_policy_stability.py)
  verifies the exact `D_q P` ordering, rational powers, universal path horizon,
  prefix and block inequalities, endpoint regimes, input domain and fail-closed
  seals; [`test_remesh_schedule_policy_stability_example.py`](../tests/physics/test_remesh_schedule_policy_stability_example.py)
  checks the facade, stub, import order and public example 171.
- [`test_remesh_schedule_relative_defect_stability.py`](../tests/physics/test_remesh_schedule_relative_defect_stability.py)
  verifies exact `q_eff`, the reused envelope, zero/equality/rejection
  boundaries, Jensen cancellation and `J=0`, exact rationalization, hostile
  inputs and fail-closed nested seals.
- [`test_runtime_remesh_schedule_relative_defect.py`](../tests/physics/test_runtime_remesh_schedule_relative_defect.py)
  verifies causal identity, normalized signed defects, minimum accepted `eta`,
  represented `q_j<=q`, every history-vector envelope, incomplete and complete
  blocks, zero energy and tamper resistance;
  [`test_runtime_remesh_schedule_relative_defect_example.py`](../tests/physics/test_runtime_remesh_schedule_relative_defect_example.py)
  checks the public facade and example 172.
- [`test_binary64_remesh_relative_defect.py`](../tests/physics/test_binary64_remesh_relative_defect.py)
  verifies shared-kernel replay, the exact normal-valued `alpha=1/2`
  obstruction, zero denominators, hard-clamp nonexpansiveness, signed-zero
  scope, the `alpha=1` class and fail-closed seals;
  [`test_delayed_remesh_contract.py`](../tests/operators/test_delayed_remesh_contract.py)
  verifies that the planner and observer share the production scalar kernel;
  [`test_binary64_remesh_relative_defect_example.py`](../tests/physics/test_binary64_remesh_relative_defect_example.py)
  checks the public facade and example 173.
- [`test_half_alpha_antisymmetric_remesh_class.py`](../tests/physics/test_half_alpha_antisymmetric_remesh_class.py)
  verifies the sharp `eta=135/124` class, exact analytic-tail constants,
  exhaustive 6,615/3,890 finite core, subnormal maximizer, strict/equality/
  rejection gain boundaries, history membership, forward invariance, excluded
  centered and lattice generalizations, public types and fail-closed seals;
  [`test_half_alpha_antisymmetric_remesh_class_example.py`](../tests/physics/test_half_alpha_antisymmetric_remesh_class_example.py)
  checks the public facade and example
  [`178_half_alpha_antisymmetric_remesh_class.py`](../examples/02_physics_regimes/178_half_alpha_antisymmetric_remesh_class.py).
- [`test_binary64_p2_reception_stability.py`](../tests/physics/test_binary64_p2_reception_stability.py)
  verifies global numeric consensus and `q=0` for the exact-half P2 kernel,
  its `eta=q_eff=0` composition and finite extinction horizon, extremes,
  signed-zero scope, the canonical-factor gain-four witness and hostile seals;
  [`test_binary64_p2_reception_stability_example.py`](../tests/physics/test_binary64_p2_reception_stability_example.py)
  checks the public facade and example 174.
- [`test_runtime_p2_reception_stage.py`](../tests/physics/test_runtime_p2_reception_stage.py)
  verifies real event/stage provenance, exact P2 support, mix, interval, metric,
  endpoint replay, underflow and signed-zero behavior, default-factor rejection,
  nested reseal resistance and explicit negative scope;
  [`test_runtime_p2_reception_stage_example.py`](../tests/physics/test_runtime_p2_reception_stage_example.py)
  checks the public facade and example 175.
- [`test_runtime_p2_reception_remesh_sequence.py`](../tests/physics/test_runtime_p2_reception_remesh_sequence.py)
  verifies per-cycle EN and REMESH identity, fixed P2 configuration, the active
  suffix boundary, finite post-horizon extinction and fail-closed negative
  scope;
  [`test_runtime_p2_reception_remesh_sequence_example.py`](../tests/physics/test_runtime_p2_reception_remesh_sequence_example.py)
  checks the public facade, stub and example
  [`176_runtime_p2_reception_remesh_sequence.py`](../examples/02_physics_regimes/176_runtime_p2_reception_remesh_sequence.py)
  with its same-invocation two-cycle witness.
- [`test_runtime_p2_reception_remesh_policy.py`](../tests/physics/test_runtime_p2_reception_remesh_policy.py)
  verifies per-call preflight, two independent successful invocations, complete
  rollback after invalid or graph-mutating post-certification, live U1a
  derivation, active-history scope and directed preflight rejection;
  [`test_runtime_p2_reception_remesh_policy_example.py`](../tests/physics/test_runtime_p2_reception_remesh_policy_example.py)
  checks the public facade, stub and example
  [`177_runtime_p2_reception_remesh_policy.py`](../examples/02_physics_regimes/177_runtime_p2_reception_remesh_policy.py).
- [`test_runtime_remesh_history_stability.py`](../tests/physics/test_runtime_remesh_history_stability.py)
  verifies bit-exact replay, signed residual and energy identities, hard-clip
  nonexpansiveness, the soft-clip counterexample, inactive-delay handling and
  fail-closed derivation seals.
- [`test_remesh_schedule_stability.py`](../tests/physics/test_remesh_schedule_stability.py)
  verifies the exact defect telescope, schedule-gain slack and lower bound,
  expansive and contracting cases, barycenter drift and fail-closed seals.
- [`test_runtime_remesh_schedule_stability.py`](../tests/physics/test_runtime_remesh_schedule_stability.py)
  verifies adjacent represented schedule endpoints, the recorded history
  advance, common normalized metrics, finite energy telescoping and fail-closed
  nested proof bindings.
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
