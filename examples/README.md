# TNFR examples

Examples demonstrate declared constructions, APIs and conditional results. Their
numbers and historical filenames are discovery aids, not a ranking of scientific
validity. The [theory index](../theory/README.md) owns claim status; the
[execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the active research queue. Running an example does not reopen a parked branch.

## Start with the intended task

| Directory | Use | Interpretation |
| --- | --- | --- |
| `01_foundations` | State, operators, topology and SDK | Supplied preparations and schedules; musical/social analogies are not empirical laws |
| `02_physics_regimes` | Diffusion, diagnostics, auxiliary models and runtime certificates | Read each model's hypotheses; a diagnostic decrease is not general stability |
| `03_riemann_zeta` | Finite spectral and zeta instruments | Parked classical-problem comparisons; disclose supplied zeros/primes |
| `04_riemann_L_twisted` | Character and L-function constructions | Supplied arithmetic data and finite comparisons, not a proof of generalized RH |
| `05_type_hygiene` | State-space and catalog countercontrols | Conditional type/representation checks; entropy does not determine state dimension |
| `06_navier_stokes` | Selected PDE correspondences | Added models and unresolved continuum obligations |
| `07_number_theory` | Arithmetic pressure, residues and prime structure | Disclose factorization, sieves and other construction inputs |
| `08_emergent_geometry` | Graph spectra, quotients, auxiliary geometry and phase/form response | Separate prescribed geometry, observed structure and autonomous generation |
| `09_millennium` | Conditional algebraic reformulations | No solution to the named open problems is claimed |
| `10_applications` | Data adapters and backend demonstrations | Observation model, data split and backend provenance remain required |

Install the repository before running a selected entry point:

```bash
python -m pip install -e .
python examples/01_foundations/01_hello_world.py
python examples/01_foundations/10_simplified_sdk_showcase.py
```

Optional dependencies vary by script. Read its imports, module docstring and
available `--help`; the directory is not an instruction to execute every file.
Package ownership belongs to [Architecture](../ARCHITECTURE.md), and verification
requirements to [Testing](../TESTING.md).

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
