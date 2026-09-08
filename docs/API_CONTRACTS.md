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
zero-duration events. The result does not certify solver accuracy, equivalent
timestep refinement, jump gains or adaptive U2/U4.

[`execute_event_remesh_cycle`](../src/tnfr/operators/event_remesh_runtime.py)
composes one such schedule with exactly one canonical pre-REMESH `_epi_hist`
sample and one separately invoked delayed map. The outer transaction rejects
ordered node-support changes or schedule-owned delayed-history writes. It also
binds the immutable schedule result to the live event log and freezes the
endpoint clock, phase, pressure-hook identity and deterministic REMESH controls.
Edges may change during the schedule; the EPI-only delayed map and its ON_REMESH
observers must preserve the resulting edge state and all non-EPI channels.

The supplied metric is materialized once, exposed even for the uniform default,
and reused for cycle-level weighted EPI observations and optional REMESH
evidence. Legacy REMESH metadata retains unweighted summaries. Exact means,
drifts and disagreements remain authoritative; an unrepresentable binary64
display is `None`. Capacity vectors, pressure vectors, schedule refreshes and
the optional post-REMESH pressure callback remain distinct. The latter runs
only when REMESH applies and is counted only after returning. Delay `tau` reads
`_epi_hist[-(tau + 1)]`, with no post-jump delayed-history duplicate. A committed
jump separately records its same-time `epi_time_history` endpoint for Mutation.
The result does not certify mixed runtime gain or repeated stability with
evolving history. External callback and integrator effects remain outside
rollback.

## Contract verification

- [`test_operator_contracts.py`](../tests/operators/test_operator_contracts.py)
  verifies catalog coverage and direct effects.
- [`test_u3_hard_invariant.py`](../tests/operators/test_u3_hard_invariant.py)
  verifies rejection before Coupling or Resonance mutation.
- [`test_canonical_operators_modern.py`](../tests/operators/test_canonical_operators_modern.py)
  covers operator behavior and latency.
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
