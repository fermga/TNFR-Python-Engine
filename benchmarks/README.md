# TNFR benchmark guide

Benchmarks are instruments, not a second research plan. Consult the
[theory index](../theory/README.md) for claim status and the
[execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
for the active queue. A filename containing `emergent`, `conservation`,
`unification` or `universality` does not establish that claim.

## Instrument families and owners

| Family | Instruments | Status and contribution |
| --- | --- | --- |
| Nodal support and regional identity | `thol_*.py`, [forced support](forced_support_balance.py), [capacity](capacity_localization.py), [memory](derived_epi_memory.py), [selection](selection_birth_closure.py) | Finite trajectories, exact declared-model calculations or read-only retained-record audits; distinguish these in each module |
| C6 winding | `c6_winding_*.py` | Parked carried-map and regional controls; retain evidence without automatically continuing campaigns |
| Directed transport | [dynamics](directed_nonnormal_dynamics.py), [metrics](directed_u2_metrics.py), [transients](directed_transient_u2.py), [time](directed_structural_time.py), [capacity boundary](heterogeneous_vf_boundary.py) | Conditional matrix/metric results and finite controls; [owner](../theory/TNFR_DIRECTED_NONNORMAL_DYNAMICS.md) |
| Field diagnostics | [circular methods](field_methods_battery.py), [potential](phi_s_confinement_investigation.py) | Diagnostics and historical heuristic protocols; [field scope](../docs/STRUCTURAL_FIELDS_TETRAD.md) and [balance scope](../theory/STRUCTURAL_CONSERVATION_THEOREM.md) |
| Arithmetic and spectra | `arithmetic_*.py`, [commutant](commutant_bridge.py), [Paley](paley_bridge.py), residue/prime/word controls | Supplied encodings and conditional algebraic comparisons; [arithmetic owner](../theory/TNFR_NUMBER_THEORY.md) and [Riemann scope](../theory/TNFR_RIEMANN_RESEARCH_NOTES.md) |
| REMESH/Riemann projections | `remesh_infinity_riemann_*.py` | Parked finite projection/spectral controls, not proof of RH or generic runtime stability |
| Auxiliary geometry and physical comparisons | `emergent_*.py` | Explicit graph, mode, wave or potential constructions; [ontology scope](../theory/EMERGENT_ONTOLOGY.md); no automatic physical identification |
| Observation and applications | [static interface](structural_interface_benchmark.py), [temporal](temporal_interface_benchmark.py), [multichannel](multichannel_interface_benchmark.py), [Volts](volts_fixed_reference_exploration.py), [external phase gate](external_phase_gate_validation.py), [U2 comparison](u2_destabilization_irreversibility.py) | Measurement mappings and evaluation controls; [empirical protocol](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md) |
| Performance and harness utilities | [optimization](benchmark_optimization_tracks.py), [lookup](nodal_lookup_scaling.py), [batching](stable_node_offset_scaling.py), [aggregation](tetrad_results_aggregate.py), [utilities](benchmark_utils.py) | Machine/input-dependent measurements and passive aggregation, not physical scaling laws |

The source inventory is discoverable without starting any experiment:

```bash
rg --files benchmarks -g '*.py'
```

## Reuse the mechanism appropriate to the question

- Use the shared [scale bridge](../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) and
  [memory owner](../theory/DERIVED_EPI_MEMORY.md) for closure or hidden-state
  questions. Spectral analogies do not replace their exact admission criteria.
- Use the retained-record THOL audits when the required evidence already exists.
  A producer can regenerate a different causal run; validating an existing
  record's contents is a different operation. A content hash does not establish
  the identity of a producer or the truth of declared acquisition conditions.
- [Wave/particle comparisons](emergent_wave_particle_correspondence.py) retain
  declared graph-wave models and negative controls. Supplied waves, standing
  modes or winding numbers do not by themselves identify physical particles.
- `emergent_simplex_dimension.py`, `emergent_fractal_simplex_dimension.py` and
  `emergent_generation_count.py` retain useful graph/spectral comparisons. Their
  clique size, gluing, scale ratio or potential well are inputs. They do not
  select physical dimension or particle generations.

The retired particle/shell, chemical lookup and PDE recipe producers added
potential wells, screening, shell order or fluid equations without deriving
those laws from the nodal product. Shared graph waves, diffusion and winding
remain available through their mathematical owners; removing the unsupported
identifications does not retire those scoped calculations.
The [scope record](../theory/research/archive/README.md#foundation-reassessment-2026-09-20)
owns the retirement list and API migrations.

The obsolete magnitude-only, out-of-range curvature-zone and orphan timing-
clustering campaigns are removed. Timing clusters and in-sample field
correlations do not establish physical universality or held-out prediction.
The legacy fast coherence-length sweep is also retired: its nominal intensity
never entered preparation or dynamics, so repairing its obsolete API calls
would not make the comparison an intensity-response experiment.
The unused universality aggregator, integrated force-regime producer and
nodal-position/prime comparison are also retired. Their missing-data pairing,
invented events, cancelling scores or disconnected supplied formulas did not
provide a predictive nodal study. Use shared study execution and diagnostics
for actual operator responses; retained winding, graph-wave and arithmetic
comparisons keep their explicit auxiliary scope.
Current form and field-domain controls are the
[signed-form admission tests](../tests/test_nodal_solver_epi_scope.py),
[storage/entropy scope tests](../tests/test_epi_type_signature_scope.py),
[wrapped tetrad bounds](../tests/physics/test_tetrad_bounds.py) and
[circular-resultant controls](../tests/physics/test_phase_curvature_resultant.py).
The corresponding definitions live in the
[form foundations](../theory/FUNDAMENTAL_THEORY.md) and field owner linked above.

## Running and reporting

Install the repository in editable mode, then choose one instrument whose
assumptions and dependencies match the task:

```bash
python -m pip install -e ".[dev-minimal]"
python benchmarks/directed_nonnormal_dynamics.py
```

Use `--help` only where the script exposes a CLI. Optional libraries and retained
artifacts differ by instrument; do not assume every script runs after the same
minimal installation. This directory is not a batch execution manifest.

A reusable record identifies the revision, environment/backend, graph and source
construction, parameters, seed, explicit laws or operator sequence, raw artifact
and hash. Include the available state/field observations and their provenance;
pure algebraic or timing instruments need not invent a trajectory or tetrad.
For data comparisons, separate calibration from evaluation and disclose supplied
labels, known factors and retrospective reconstruction.

Compare performance on identical inputs, semantics and warm-up policy. Report
absolute times alongside ratios. A finite benchmark supports only its measured
domain and does not establish a universal speedup, canonical threshold, general
stability theorem or physical law.
