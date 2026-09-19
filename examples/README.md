# TNFR examples

Examples demonstrate declared constructions, APIs and conditional results. Their
numbers and historical filenames are discovery aids, not a ranking of scientific
validity. The [theory index](../theory/README.md) owns claim status; the
[execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the active research queue. Running an example does not reopen a parked branch.

## Start with the intended task

| Directory | Python files | Use and authority | Interpretation |
| --- | --- | --- | --- |
| `01_foundations` | 4 | SDK and [grammar scope](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) | Supplied preparations, words and state ensembles |
| `02_physics_regimes` | 37 | [Diffusion certificates](../theory/TNFR_DIFFUSION_STABILITY_THEOREM.md), diagnostics and auxiliary models | Read each model's hypotheses; a diagnostic decrease is not general stability |
| `03_riemann_zeta` | 19 | Finite instruments in the [Riemann notebook](../theory/TNFR_RIEMANN_RESEARCH_NOTES.md) | Parked comparisons; disclose supplied zeros/primes |
| `04_riemann_L_twisted` | 18 | Character/L-function instruments in the same [notebook](../theory/TNFR_RIEMANN_RESEARCH_NOTES.md) | Supplied arithmetic data and finite comparisons, not generalized RH |
| `05_type_hygiene` | 13 | [Catalog and state-type controls](../theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md) | Finite representation probes; entropy does not determine state dimension |
| `06_navier_stokes` | 1 | [Selected PDE correspondence](../theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) | Added models and unresolved continuum obligations |
| `07_number_theory` | 15 | [Arithmetic definitions](../theory/TNFR_NUMBER_THEORY.md), residues and prime structure | Disclose factorization, sieves and other construction inputs |
| `08_emergent_geometry` | 47 | [Scale/geometry bridge](../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md), spectra and auxiliary models | Separate prescribed geometry, observed structure and autonomous generation |
| `09_millennium` | 3 | Conditional algebraic comparisons; see the [theory index](../theory/README.md) | No solution to the named open problems is claimed |
| `10_applications` | 6 | [Measurement protocol](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md), adapters and backend provenance | Data admission and reserved prediction remain separate obligations |

The inventory contains 162 executable demonstrations and one shared support
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

Optional dependencies vary by script. Plotting examples require `viz-basic`;
examples 91 and 92 additionally require `scikit-learn`, and 92 downloads and
caches UCI data. The Torch demonstration uses `compute-torch`. Read module
docstrings and available `--help` before an explicit run; the directory is not an
instruction to execute every file. Importing a demonstration does not start its
command-line workflow. Some older examples still configure imports or plotting
defaults at module scope, so this is not a promise of side-effect-free imports.
Package ownership belongs to [Architecture](../ARCHITECTURE.md), and verification
requirements to [Testing](../TESTING.md).

## Consolidated demonstrations

The former handwritten phase updates were removed because they presented
`theta_dot` as the EPI nodal equation, used noncanonical pressure/coherence
proxies, and duplicated the same kernels in several files. Their conceptual
topics now point to the existing owners below; the old numerical outputs are
not reinterpreted as canonical evidence.

| Retired entry | Reason and retained route |
| --- | --- |
| `02_musical_resonance.py` | Handwritten phase synchronization; use [179](08_emergent_geometry/179_phase_form_driven_response.py) for an explicitly prescribed phase/form response |
| `03_network_formation.py` | Handwritten connection rules and a private coherence score; use [10](01_foundations/10_simplified_sdk_showcase.py) for supplied SDK topology construction |
| `05_coherence_evolution.py` | Phase update incorrectly attributed to the EPI law; use [99](08_emergent_geometry/99_structural_diffusion.py) for the declared diffusion model |
| `06_network_topologies.py` | Duplicated private pressure/phase dynamics; use [10](01_foundations/10_simplified_sdk_showcase.py) and [99](08_emergent_geometry/99_structural_diffusion.py) |
| `08_emergent_phenomena.py` | Ad hoc phase/frequency rules and collective-behavior claims; use [07](01_foundations/07_phase_transitions.py) for prepared observations and [179](08_emergent_geometry/179_phase_form_driven_response.py) for conditional response |
| `09_visualization_suite.py` | Plots of the duplicated substitute dynamics; use [179](08_emergent_geometry/179_phase_form_driven_response.py) for exported plots of its declared model |
| `13_quantum_mechanics_demo.py` | Imposed energy targets and handwritten noisy updates, not a derived cavity spectrum; see the [spectral validation boundary](../theory/PHYSICAL_REGIME_CORRESPONDENCES.md#44-validation-boundary) |
| `14_uncertainty_and_interference.py` | Raw-array Fourier widths and a forced wave grid without a canonical bridge or validated bound; see the [Fourier scope](../theory/PHYSICAL_REGIME_CORRESPONDENCES.md#53-validation-boundary) |

The original bytes and retirement hashes are retained in the local audit
artifacts under `artifacts/research/examples_consolidation_originals_2026_09_19/`.
Example 01 now uses the public SDK directly. Examples 10 and 30 keep their
command-line workflows behind `main()` instead of executing them on import.
Example 109 retains its finite classical MAX-CUT baseline, with its supplied
antialignment update distinguished from canonical TNFR phase pressure. None of
these retirements introduces a replacement physical law or research branch.

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
