# Benchmark and research instrument guide

This directory contains performance measurements, conditional model probes,
reserved-response producers and read-only record audits. These roles are
not interchangeable. The [theory catalog](../theory/README.md) owns mathematical
claims; the [execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md)
is the sole task queue. A retained instrument is not automatically an active
research campaign, and a filename containing `emergent` proves no emergence.

## Instrument families and owners

### Engine, response and evidence

| Question | Instruments | Owner and scope |
| --- | --- | --- |
| Does a held source retain information absent from regional contrast/rate? | [Source-relative response](source_relative_form_response.py) | [Derived form](../theory/nodal/DERIVED_FORM_PHASE.md#source-relative-future-response); supplied affine law and zero-source ablation |
| How do supplied joint form/phase laws differ? | [Constitutive probes](phase_form_exchange_comparison.py), [cotangent C8](cotangent_phase_exchange.py) | [Relational exchange](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#finite-modal-exchange-discriminator), [cotangent scope](../theory/TNFR_VARIATIONAL_PRINCIPLE.md#cotangent-c8-finite-response); probes and trajectories are distinct |
| How does capacity change the response? | [Capacity producer](relational_capacity_response.py), [record audit](relational_capacity_audit.py) | [Capacity response](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#finite-capacity-intervention-response); the audit does not rerun trajectories |
| Can a prepared temporal readout identify chi with bounded error? | [P2 acquisition](relational_coefficient_acquisition.py), [read-only audit](../src/tnfr/research/relational_acquisition.py) | [Temporal admission](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#coefficient-temporal-acquisition); known-source software control, prior-derived nonlinear bounds and native Euler defects, not blind physical calibration |
| Does a prepared pattern recover or transmit deformation? | [Local recovery](relational_local_recovery.py), [regional interaction](relational_region_interaction.py) | [Joint-law owner](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md); supplied support and reference geometry |
| Which observations close, and what hidden state carries memory? | [Local composition](relational_local_composition.py), [memory coefficients](relational_memory_prediction.py), [reserved response](relational_memory_response.py) | [Composition](../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md), [memory](../theory/nodal/RELATIONAL_PATTERN_MEMORY.md); fixed rational probes and finite/continuous error have separate scopes |
| Does an intermediary transmit a capacity-dependent response? | [Mediated response](relational_mediation_response.py) | [Finite mediator protocol](../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#finite-mediated-response); shared tangent memory versus omitted or instantaneous memory, with a frozen-capacity control |
| Does a return path distinguish regional orientations? | [Static return geometry](relational_return_geometry.py) | [Equilibrium admission](../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#return-path-equilibrium); exact cycle reconstruction, certified scalar root and native residuals, with no trajectory or event |
| Does the native law have a causally shared oscillatory mode? | [Static collective pulse](relational_collective_pulse.py) | [Pulse admission](../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#shared-collective-pulse); ideal trace-sign certificate, native tangent poles/residues and no-communication control, without a sustained-pulse claim |
| Does a finite trajectory change sector or enter a sufficient basin? | [Formation response](relational_formation_response.py), [capture response](relational_capture_response.py), [endpoint audit](relational_capture_audit.py) | [Capture scope](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-upper-corner-response); endpoint evidence is not a whole-path error bound |
| Is an ideal continuous transit enclosed rigorously? | [Transit proof](relational_transit_proof.py) | [Validated transit](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-validated-transit); also zero-form and reversed-form controls, with their own frozen protocols |
| What holds for directed transport? | [Dynamics](directed_nonnormal_dynamics.py), [U2 metrics](directed_u2_metrics.py), [transients](directed_transient_u2.py), [time](directed_structural_time.py), [capacity boundary](heterogeneous_vf_boundary.py) | [Directed transport](../theory/TNFR_DIRECTED_NONNORMAL_DYNAMICS.md); conditional matrix/metric comparisons |
| What do support changes and capacity actually do? | [Birth/transport](thol_birth_transport.py), [pressure feedback](thol_pressure_feedback.py), [forced support](forced_support_balance.py), [capacity](capacity_localization.py), [selection](selection_birth_closure.py) | [THOL](../theory/THOL_BIRTH_AND_TRANSPORT.md), [capacity](../theory/CAPACITY_LOCALIZATION_BALANCE.md), [support balance](../theory/FORCED_SUPPORT_BALANCE.md); declared operators and laws |
| Can prior THOL or C6 evidence answer the question without another run? | `thol_*audit.py`, regional THOL instruments, `c6_winding_*.py` | [THOL feedback](../theory/THOL_PRESSURE_FEEDBACK.md), [C6 audit](../theory/C6_RESEARCH_MECHANISM_AUDIT.md); retained chains of producers, exact controls and audits, not independent new campaigns |

### Observation, auxiliary models and performance

| Role | Instruments | Boundary |
| --- | --- | --- |
| Static and temporal observation | [Static interface](structural_interface_benchmark.py), [temporal](temporal_interface_benchmark.py), [multichannel](multichannel_interface_benchmark.py), [external phase gate](external_phase_gate_validation.py) | [Interface guide](../docs/STRUCTURAL_INTERFACE_THEORY.md) owns measurement preparation, label availability, estimator limits and fair comparisons |
| Terrestrial transport comparisons | [TCLab](tclab_nodal_exploration.py), [Volts comparison](volts_fixed_reference_exploration.py), [Volts loader](volts_data.py) | [TCLab protocol](../theory/research/TCLAB_EXPLORATORY_PROTOCOL.md), [passive transport](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md); exploratory maps are not admitted physical variables |
| Field diagnostics | [Method battery](field_methods_battery.py), [curvature example](../examples/08_emergent_geometry/k_phi_safety_demo.py) | [Tetrad](../docs/STRUCTURAL_FIELDS_TETRAD.md); declared diagnostic scope, not autonomous dynamics or universal safety thresholds |
| Arithmetic and spectral comparisons | `arithmetic_*.py`, [composition](composition_arithmetic.py), [commutant](commutant_bridge.py), [Paley](paley_bridge.py), residue/prime/word controls | [Arithmetic](../theory/TNFR_NUMBER_THEORY.md); supplied encodings, exact algebra or finite comparisons |
| REMESH/Riemann projections | `remesh_infinity_riemann_*.py` | [Riemann scope](../theory/TNFR_RIEMANN_RESEARCH_NOTES.md); parked spectral controls, not RH or complete-runtime stability proofs |
| Auxiliary geometry | `emergent_*.py`, pulse and symmetry probes | [Ontology boundaries](../theory/EMERGENT_ONTOLOGY.md); supplied graph/wave/simplex models, not identified particles or physical dimensions |
| Actual performance | [Node lookup](nodal_lookup_scaling.py), [offset batching](stable_node_offset_scaling.py) | Source, seed, workload, timings and trajectory hashes; no universal speedup claim |

The complete source inventory can be read without importing or running scripts:

```bash
rg --files benchmarks -g '*.py'
```

Paths remain stable where tests, imports and frozen source manifests consume
them. Grouping the instruments here avoids moving a scientific producer merely
to make a directory look tidier. Inspect the source and its owner before using
an unlisted family member; do not batch-execute this inventory.

## Reuse the mechanism appropriate to the question

Use the [SDK](../docs/CLI_AND_SDK.md) for operator studies and its
[regional/relational guide](../docs/guides/REGIONAL_AND_RELATIONAL.md) for shared
observations, explicit joint evolution and capture admission. Benchmarks add
preparations, comparisons and evidence, not a second production solver.

Prefer a read-only record audit when the needed trajectory already exists.
An audit separates record consistency, the original decision and compatibility
with current source/runtime. Some THOL/C6 audits require local records under
`artifacts/research/` that are not versioned or supplied by cloning the repo;
missing records make those checks unavailable. Sparse checkpoints cannot authenticate omitted
steps or reconstruct a continuous integral. A content digest does not establish
physical acquisition, trustworthy chronology or a complete transitive dependency
archive unless that archive is explicitly included.

Use [derived memory](../theory/DERIVED_EPI_MEMORY.md) and the
[scale bridge](../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) for closure questions.
The auxiliary gasket instruments reuse the pure
[graph preparation](graph_fixtures.py), including its ordered outer corners.
That supplied construction is not a THOL event or a selected physical dimension.
The Bell instrument now uses those corners at every level; earlier level-two
and higher fixtures used the first inserted nodes instead. The CHSH bound is
unchanged, while those graph fixtures are corrected; historical outputs were
not recomputed.

A rank calculation for rational surrogate coefficients is not a nonlinear
closure proof for actual irrational coefficients. An observation of winding,
a standing mode or a supplied potential does not identify a physical particle.

## Running and reporting

Install the repository in editable mode, then run the selected instrument from
the repository root. Dependencies and command-line support differ by module:

```bash
python -m pip install -e ".[dev-minimal]"
python -m benchmarks.nodal_lookup_scaling --sizes 10 --repeats 2
python -m benchmarks.stable_node_offset_scaling --sizes 6 --repeats 2 --scoped
```

These small examples exercise timing/report plumbing; meaningful performance
comparisons require identical input, source, hardware, backend, warm-up and
measurement scope. Retain absolute times and trajectory agreement, not just a
speed ratio. `--wheel` selects a separate baseline without installing it.
`pytest benchmarks --benchmark-only` is not a supported directory-wide suite;
scientific regression tests live under `tests/`. The optional `test-performance`
extra supplies a third-party plugin only when explicitly requested.

### Freeze before evaluating a reserved response

The following producers accept `--prepare --output PATH`; evaluation uses the
same command and path without `--prepare`:

| Producer module under `benchmarks` | Role |
| --- | --- |
| `source_relative_form_response` | Held-source prediction and ablation |
| `phase_form_exchange_comparison` | Prepared constitutive probes, not a trajectory |
| `relational_capacity_response` | Capacity intervention |
| `relational_coefficient_acquisition` | One known-source P2 acquisition; full retained state/error chain and source archive |
| `relational_local_recovery` | Prepared local recovery |
| `relational_region_interaction` | Transmitted regional response |
| `relational_formation_response`, `relational_capture_response` | Finite sector/endpoint responses |
| `relational_memory_response` | Matched-grid memory prediction and full-engine comparison |
| `relational_mediation_response` | Matched-grid mediator-capacity intervention; retained memory versus instantaneous and omitted-memory controls |
| `relational_transit_proof` | Continuous proof audit; `--zero-form` and `--reverse-form` are mutually exclusive controls |

For example, use a fresh path for a current-source regression:

```bash
python -m benchmarks.relational_capacity_response --prepare --output artifacts/relational-capacity/result.json
python -m benchmarks.relational_capacity_response --output artifacts/relational-capacity/result.json
```

Preparation and evaluation are separate invocations. Keep sibling prediction
or protocol files and any source archives together. Admission binds declared
inputs and recorded source/runtime fingerprints; changing source bytes, line
endings or dependency versions can invalidate a replay. Prepare a new pair
rather than rewrite the first experiment. Use the CLI rather than feeding raw
JSON into producer functions that expect native exact-rational mappings.

`docs/assets/` contains immutable first predictions, responses and source
archives linked by each theorem owner. New output belongs in `artifacts/`.
Failed decisions must remain visible; an inconclusive certificate is not a
pass. The later continuous transit certificate does not replace the earlier
finite-executor verdict. Consensus (`target_sector=0`) and unavailable evidence
are different results; do not interpret the sector as a Boolean.

For every new numerical claim, retain source/configuration, input construction,
seed, backend/precision, law or operator sequence, raw artifact and provenance.
Separate calibration from evaluation and disclose supplied labels or factors.
Finite measurements establish their measured scope, not universal physical laws.

### Read-only temporal-acquisition audit

Inspect the existing P2 response without regenerating it:

```bash
python -m tnfr.research.relational_acquisition artifacts/research/relational_coefficient_acquisition/result.json
```

This command reads the sibling protocol and archive, then reconstructs saved
contrast/error evidence through the [shared auditor](../src/tnfr/research/relational_acquisition.py).
It has no prepare/evaluate mode and writes no artifacts. Exit 0 means record
consistency, **not** that the experiment passed; `recorded_passed` and
`reconstructed_passed` retain the original verdict separately. Exit 1 denotes
inconsistent evidence and 2 missing/unavailable files. Local artifacts may be
absent after cloning; their absence is not permission to replay the producer.
The [contract](../docs/contracts/RELATIONAL_DYNAMICS.md#relational-acquisition-audit)
owns admission, work limits and the authentication boundary.

### Multichannel observation records

The multichannel instrument computes one window series for both ranking and
block summaries. Malformed ARFF rows reject the record rather than silently
removing samples; binary labels must remain aligned with signal samples.
The byte limit covers cached payloads and expanded ZIP members as well as the
download. Unavailable class means serialize as `null`; reports use the shared
strict JSON and atomic writer.

Check `auc_available` and `auc_unavailable_reason` before interpreting scores.
When windows or a label class are absent, retained legacy AUC values of `0.5`
are compatibility placeholders, not measured chance performance. Available
orientation-free rankings remain same-sample descriptions, not reserved
forecasts; the [interface guide](../docs/STRUCTURAL_INTERFACE_THEORY.md)
owns the methodological boundary.

## Retired instruments

The [retirement record](../theory/research/archive/README.md#benchmark-instrument-cleanup-2026-09-27)
and [recovery manifest](../theory/research/archive/BENCHMARK_CLEANUP_2026-09-27.json)
identify unsupported timing comparisons, obsolete threshold/U2 investigations
and orphan utilities removed in this audit. Useful mathematical results,
negative boundaries, maintained tests and frozen evidence remain with their
owners; retirement does not assert that every alternative model is impossible.
