# TNFR examples

Examples demonstrate declared constructions, APIs and conditional results. Their
numbers and historical filenames are discovery aids, not a ranking of scientific
validity. The [theory index](../theory/README.md) owns claim status; the
[execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the active research queue. Running an example does not reopen a parked branch.

## Start with the intended task

| Directory | Python files | Use and authority | Interpretation |
| --- | --- | --- | --- |
| `01_foundations` | 5 | SDK and [grammar scope](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) | Supplied preparations, words and state ensembles |
| `02_physics_regimes` | 37 | [Diffusion certificates](../theory/TNFR_DIFFUSION_STABILITY_THEOREM.md), diagnostics and auxiliary models | Read each model's hypotheses; a diagnostic decrease is not general stability |
| `03_riemann_zeta` | 19 | Finite instruments in the [Riemann notebook](../theory/TNFR_RIEMANN_RESEARCH_NOTES.md) | Parked comparisons; disclose supplied zeros/primes |
| `04_riemann_L_twisted` | 18 | Character/L-function instruments in the same [notebook](../theory/TNFR_RIEMANN_RESEARCH_NOTES.md) | Supplied arithmetic data and finite comparisons, not generalized RH |
| `05_type_hygiene` | 4 | [Catalog and state-type controls](../theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md) | Finite delay projection, actual storage, event-count and registry observations |
| `07_number_theory` | 14 | [Arithmetic definitions](../theory/TNFR_NUMBER_THEORY.md), residues and prime structure | Disclose factorization, sieves and other construction inputs |
| `08_emergent_geometry` | 47 | [Scale/geometry bridge](../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md), spectra and auxiliary models | Separate prescribed geometry, observed structure and autonomous generation |
| `10_applications` | 6 | [Measurement protocol](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md), adapters and backend provenance | Data admission and reserved prediction remain separate obligations |

The inventory contains 149 executable demonstrations and one shared support
module, `_flat_grammar_model.py`. Counts describe files, not independent research
lines. The grammar automaton examples reuse that module; arithmetic and physical
examples reuse their package owners. A finite fixture may recur as a controlled
comparison without constituting a second implementation of its governing law.

Install the repository before running a selected entry point:

```bash
python -m pip install -e .
python examples/01_foundations/01_hello_world.py
python examples/01_foundations/04_operator_sequences.py
python examples/01_foundations/10_simplified_sdk_showcase.py
```

Example 01 is the minimal SDK entry point. Example 04 separates flat grammar
admission from a deliberately illustrative pressure proxy. Example 10 surveys
the topology builders and configured SDK operations. Example 07 is a larger
prepared-state ensemble, not part of this introductory command list.

For a retained declaration and finite execution report, use
[reproducible_study.py](01_foundations/reproducible_study.py) and the
[shared CLI/SDK guide](../docs/CLI_AND_SDK.md). The same declaration can be run
through either interface; its export is not a complete resumable checkpoint.

Optional dependencies vary by script. Plotting examples require `viz-basic`;
examples 91 and 92 additionally require `scikit-learn`, and 92 downloads and
caches UCI data. The Torch demonstration uses `compute-torch`. Read module
docstrings and available `--help` before an explicit run; the directory is not an
instruction to execute every file. Importing a demonstration does not start its
command-line workflow. Some older examples still configure imports or plotting
defaults at module scope, so this is not a promise of side-effect-free imports.
Package ownership belongs to [Architecture](../ARCHITECTURE.md), and verification
requirements to [Testing](../TESTING.md).

## Retained controls and shared owners

The type-hygiene directory retains four scoped entry points:

| Entry | What it observes |
| --- | --- |
| [77](05_type_hygiene/77_remesh_infinity_residue_split_demo.py) | Finite fixed-delay Fourier projection and window sensitivity |
| [79](05_type_hygiene/79_epi_type_signature_demo.py) | Actual scalar-chart storage membership and descriptive temporal entropy |
| [82](05_type_hygiene/82_remesh_window_type_signature_demo.py) | Selected finite REMESH event/window comparisons |
| [89](05_type_hygiene/89_operator_catalog_discipline_signature_demo.py) | The implemented registry and its idempotency |

Entropy thresholds and scalar-only fixtures do not prove that a richer state
type is necessary or impossible. The retired type-necessity and synthetic
closure campaigns are replaced by the actual domain and execution contracts:
[signed-form admission](../tests/test_nodal_solver_epi_scope.py),
[circular U3 admission](../tests/operators/test_u3_hard_invariant.py),
[field availability](../tests/sdk/test_nfr_observation_scope.py) and
[grammar observations](../tests/operators/test_grammar_observations.py).

For supplied topology construction use [10](01_foundations/10_simplified_sdk_showcase.py);
for declared diffusion use [99](08_emergent_geometry/99_structural_diffusion.py).
Prescribed phase/form response belongs to [179](08_emergent_geometry/179_phase_form_driven_response.py).
Removed chemistry and Millennium demonstrations supplied target laws or
encodings without deriving them from nodal dynamics. Their removal does not
affect the scoped diffusion, graph-wave, winding and arithmetic owners.
Retirement reasons and API migrations have one
[scope record](../theory/research/archive/README.md#foundation-reassessment-2026-09-20).

## Diffusion and runtime evidence

These examples reuse the
[diffusion stability owner](../theory/TNFR_DIFFUSION_STABILITY_THEOREM.md).
The table distinguishes an exact declared model from finite engine evidence.
Detailed bounds and assumptions remain in that owner instead of being duplicated
as a second theorem ledger here.

| Entries in `02_physics_regimes/` | What they demonstrate | Boundary |
| --- | --- | --- |
| [160](02_physics_regimes/160_core_research_integration.py), [161](02_physics_regimes/161_core_research_trajectory.py), [162](02_physics_regimes/162_hybrid_epi_stability.py) | Restricted S16 and affine-reset controls | Declared pure-EPI/hybrid models |
| [163](02_physics_regimes/163_reception_runtime_bridge.py), [164](02_physics_regimes/164_resonance_runtime_bridge.py), [165](02_physics_regimes/165_operator_event_relaxation.py) | Local operator and event-time binding | Captured runtime maps, not arbitrary operators |
| [166](02_physics_regimes/166_event_remesh_reference_family.py), [167](02_physics_regimes/167_reversible_eigenmode_reference.py), [168](02_physics_regimes/168_runtime_reversible_eigenmode_reference.py) | Modal reference, Euler refinement and finite binary64 defects | Conditional exact-real refinement differs from runtime convergence |
| [169](02_physics_regimes/169_event_remesh_causal_runtime.py), [170](02_physics_regimes/170_runtime_remesh_block_margin.py) | Causal cycle receipts and finite block margins | One invocation/block does not certify future stability |
| [171](02_physics_regimes/171_remesh_schedule_policy_stability.py), [172](02_physics_regimes/172_runtime_remesh_relative_defect.py), [173](02_physics_regimes/173_binary64_remesh_relative_defect.py) | Conditional history envelopes and represented-number defects | Required gain/defect hypotheses are separate from their finite measurements |
| [174](02_physics_regimes/174_binary64_p2_reception_remesh_stability.py), [175](02_physics_regimes/175_runtime_p2_reception_stage.py), [176](02_physics_regimes/176_runtime_p2_reception_remesh_sequence.py), [177](02_physics_regimes/177_runtime_p2_reception_remesh_policy.py) | Restricted P2 half-Reception and alpha-one REMESH composition | Numeric kernels, one stage, finite sequence and revalidated policy have distinct scopes |
| [178](02_physics_regimes/178_half_alpha_antisymmetric_remesh_class.py) | Restricted antisymmetric binary64 REMESH class | REMESH invariance alone is not complete-runtime invariance |

## Geometry and phase/form response

The [scale bridge](../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) owns effective
geometry and observation closure. The tetrad is a required diagnostic interface,
not a proven complete state; reflection equivalence and loss of hidden state are
separate questions. Auxiliary symplectic/U(2) demonstrations do not establish
that engine operators are Hamiltonian or generate particles.

[The unified-field showcase](08_emergent_geometry/unified_fields_showcase.py)
compares seeded supplied states on a path, cycle and barbell using the shared
snapshot readout. It executes no operators or evolution. Its optional plot
reports descriptive field statistics and explicitly leaves temporal conservation
unavailable; the former handwritten operator substitutes and physical-domain
validation claims have been removed.

[Example 179](08_emergent_geometry/179_phase_form_driven_response.py) checks a
prescribed rotating phase contrast and its derived EPI response on the six-node
prism. It evaluates detached analytic snapshots rather than a native trajectory.
The [phase/form owner](../theory/NODAL_PARAMETER_FOUNDATIONS.md#16-phase-and-form-directed-exchange-frames-and-the-moving-mean)
states the exact assumptions, moving mean and same-input contraction scope.
The imposed phase clock is not a derived autonomous maintenance mechanism.

```bash
python examples/08_emergent_geometry/179_phase_form_driven_response.py --output-dir docs/assets/phase_form_driven_response
```

Optional plots require the `viz-basic` extra. Retained outputs are the
[figure](../docs/assets/phase_form_driven_response/phase_form_driven_response.png),
[JSON](../docs/assets/phase_form_driven_response/phase_form_driven_response.json)
and [CSV](../docs/assets/phase_form_driven_response/phase_form_driven_response.csv).
Their recorded residuals and refinements are finite evidence, not autonomous
formation or a physical identification.
