# TNFR core dynamics: S1-S16 evidence map

**Current scope: 2026-09-19.** This is a technical map, not an execution queue.
The [research plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns
priority and resumption. Each linked result states its own hypotheses; a line
with implemented tools is not thereby a completed scientific objective.
The preceding long delivery ledger is preserved in the
[research archive](research/archive/README.md).

## Current restricted results

| Line | Retained result | Open boundary | Principal owner |
| --- | --- | --- | --- |
| S1: stability | Fixed/restricted time-varying diffusion, conditional affine/reset and delayed-history bounds | General multichannel and future complete-runtime stability | [Diffusion](TNFR_DIFFUSION_STABILITY_THEOREM.md), [REMESH](REMESH_INFINITY_DERIVATION.md) |
| S2: Lyapunov structure | Model-specific Dirichlet and augmented-history balances | A common full-engine/tetrad Lyapunov theorem | [Variational](TNFR_VARIATIONAL_PRINCIPLE.md), [conservation diagnostics](STRUCTURAL_CONSERVATION_THEOREM.md) |
| S3: observability | Tetrad summaries can lose dynamics; affine and nonlinear reflection observations have explicit scope | A universal minimal complete nodal state | [Scale/fields](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md), [minimality boundary](MINIMAL_STRUCTURAL_DEGREES.md) |
| S4: heterogeneous capacity | Positive reversible metric, fixed/generalized decay rates and exact eigenmode reference | Broad directed/time-dependent certificates | [Diffusion](TNFR_DIFFUSION_STABILITY_THEOREM.md) |
| S5: solver/grammar adaptation | Finite pressure-refreshed execution evidence and restricted exact convergence | Generic runtime mesh convergence and a derived adaptive grammar | [Diffusion](TNFR_DIFFUSION_STABILITY_THEOREM.md), [grammar scope](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) |
| S6: transitions | Scoped diagnostics and finite-size comparison protocol | An admitted dynamical transition/universality claim | [Stability diagnostics](STRUCTURAL_STABILITY_AND_DYNAMICS.md) |
| S7: topology changes | Geometric classification and endpoint/event observations | Autonomous topology selection and predictive transition law | [Scale/geometry](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) |
| S8: coarse dynamics | Exact EPI quotients, derived memory and conditional counted-support joint laws | General nonlinear/changing-support closure | [Scale/geometry](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md), [memory](DERIVED_EPI_MEMORY.md) |
| S9: operator scale flow | Restricted intertwining and inherited observation kernels | Catalog-wide renormalization and same-family closure | [Scale/geometry](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) |
| S10: catalog completeness | Registered contracts and finite conformance probes | Admissible transformation space and generation/completeness theorem | [Operators](STRUCTURAL_OPERATORS.md) |
| S11: coherence levels | Exact geometry of the specified reciprocal diagnostic and fixed-capacity slices | A unique physical coherence metric or general temporal law | [Scale/geometry](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) |
| S12: dissipative/conservative relation | Decoupled product and restricted read-out realizability results | Derivation of the full nodal law from an auxiliary Hamiltonian | [Variational](TNFR_VARIATIONAL_PRINCIPLE.md) |
| S13: non-normal growth | Fixed linear transient criteria with numerical abstention | A general directed nonlinear/U2 stability metric | [Directed dynamics](TNFR_DIRECTED_NONNORMAL_DYNAMICS.md) |
| S14: information geometry | Declared structural metrics on restricted graph/state classes | Cross-topology/nesting/history identification | [Parameter foundations](NODAL_PARAMETER_FOUNDATIONS.md) |
| S15: inverse identification | Finite known-target signatures, mutation records and supplied-word evidence | Unknown-target, aggregate, arbitrary-state and complete-word inversion | [Operator/API contracts](../docs/API_CONTRACTS.md) |
| S16: persistent effective NFR | Conditional state/field/trajectory and finite causal certificates | Autonomous formation, restoration, general persistence and physical identity | [Scale/geometry](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md), [THOL evidence](THOL_BIRTH_AND_TRANSPORT.md) |

## Implementation map

| Mechanism | Shared source | Representative verification |
| --- | --- | --- |
| Diffusion and metric | [structural_diffusion.py](../src/tnfr/physics/structural_diffusion.py) | [coarse-graining tests](../tests/physics/test_epi_coarse_graining.py) |
| Held source and pressure | [forced_support.py](../src/tnfr/physics/forced_support.py), [forcing_realization.py](../src/tnfr/physics/forcing_realization.py) | [forcing tests](../tests/physics/test_forcing_realization.py) |
| Exact affine observation | [epi_memory.py](../src/tnfr/physics/epi_memory.py) | [affine observation tests](../tests/physics/test_affine_observation_realization.py) |
| Inherited support and channels | [quotient_structure.py](../src/tnfr/physics/quotient_structure.py), [joint_quotient.py](../src/tnfr/physics/joint_quotient.py) | [joint contract tests](../tests/physics/test_joint_quotient_contract.py) |
| Geometric dependencies | [geometry_realization.py](../src/tnfr/physics/geometry_realization.py) | [tetrad dependency tests](../tests/physics/test_tetrad_geometry_scope.py) |
| Reflection invariants | [p5_reduction.py](../src/tnfr/physics/p5_reduction.py) | [reflection tests](../tests/physics/test_p5_reflection_invariants.py) |
| Canonical fields | [canonical.py](../src/tnfr/physics/canonical.py) | [field consistency](../tests/physics/test_field_readout_consistency.py) |
| Network stages/events | [network_stage.py](../src/tnfr/operators/network_stage.py), [event_runtime.py](../src/tnfr/operators/event_runtime.py) | [event tests](../tests/operators/test_operator_event_runtime.py) |
| Delayed history | [remesh_history_stability.py](../src/tnfr/physics/remesh_history_stability.py) | [history tests](../tests/physics/test_remesh_history_stability.py) |

## Falsification ledger

- Stored or instantaneous coherence is not a universal predictor of future evolution.
- A finite residual, successful test or sealed invocation is not an asymptotic theorem.
- Exact mean closure can omit geometry; potential closure can still omit xi.
- Capturing a source or supplying a periodic phase does not derive its autonomous law.
- Symmetry of a graph does not automatically extend to a selected operator word,
  source, capacity field, metric or numerical execution order.
- C6's remaining 15 labels are open; no indefinite boundedness or instability
  conclusion follows from the current finite exclusions.

## Completed runtime and causal-binding milestones

The event/history owners distinguish abstract kernels, represented arithmetic,
one executed transition, a finite compatible causal sequence and future behavior.
The restricted P2 half-Reception/REMESH results belong to their explicit domains;
they are not catalog-wide stability. Detailed constants, schedules, proofs and
historical checkpoints remain in [REMESH](REMESH_INFINITY_DERIVATION.md),
[diffusion](TNFR_DIFFUSION_STABILITY_THEOREM.md) and the preserved ledger.
Do not copy their changing lists of witnesses into this index.
