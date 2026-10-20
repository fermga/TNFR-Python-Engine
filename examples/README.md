# TNFR examples

Examples exercise organized mathematical objects, their evolution, interaction
and observation under explicit laws. Choose a capability and its assumptions
before running an instrument. The [theory index](../theory/README.md) owns
definitions and mathematical scope; the
[execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns research priorities. File numbers are stable identifiers, not a required
learning order or a sequence of research milestones.

## Begin with an organized mathematical object

Install the repository and run the introductory example:

```bash
python -m pip install -e .
python examples/01_foundations/01_hello_world.py
```

[Example 01](01_foundations/01_hello_world.py) prepares a five-node cyclic
organization with nonuniform signed form, circular phase and held capacity.
Under its explicitly selected native relational law, the SDK certifies a
winding-one recovery class, executes one shared finite step and assesses the
resulting state again. The report retains the full state and law: winding
identifies a geometric class, not everything needed to predict the object's
response.

The initial organization is supplied. Each capture certificate supports ideal
continuous recovery from its own snapshot; the computed finite-step response
and numerical defect are separate evidence. This provides a small working
object whose geometry, dynamics and recovery conditions can be inspected.
Continue with the [recovery-report guide](../docs/guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#read-an-organized-objects-recovery-report)
and [native execution guide](../docs/guides/relational/RELATIONAL_EXECUTION.md#execute-the-conditional-relational-model).

Choose the continuation that answers your question:

| Capability | Entry point | Evidence and scope |
| --- | --- | --- |
| Evolve a prepared organization and account for regional exchange | [Example 180](08_emergent_geometry/180_relational_exchange.py) | Four joint native-law steps, unequal and zero capacities, storage increments and Euler defects; detached two-ring snapshots compare capture hypotheses |
| Inspect conditional formation and response | [Formed-class workflow](../docs/guides/relational/SINE_PATTERNS.md#sine-formed-class-pair) and [response comparison](../docs/guides/relational/SINE_PATTERNS.md#sine-formed-class-response) | The separate normalized-sine law, with its own preparation, uncertainty and certificate |
| Apply a supplied operator word | [Reproducible study](01_foundations/reproducible_study.py) and [CLI/SDK guide](../docs/CLI_AND_SDK.md) | A retained study declaration and finite execution report; grammar, target state and event admission are separate from continuous recovery |

The first route extends the native introductory object. The other routes
select different execution contracts; a theorem or certificate transfers only
when its complete premises match. A study export is not a resumable checkpoint.

## Start with the intended task

| Directory | Use and authority | Interpretation |
| --- | --- | --- |
| `01_foundations` | Organized-object introduction, SDK and [grammar scope](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) | Prepared objects with conditional recovery, supplied words and state ensembles |
| `02_physics_regimes` | [Diffusion certificates](../theory/TNFR_DIFFUSION_STABILITY_THEOREM.md), diagnostics and auxiliary models | Read each model's hypotheses; a diagnostic decrease is not general stability |
| `03_riemann_zeta` | Finite instruments in the [Riemann notebook](../theory/TNFR_RIEMANN_RESEARCH_NOTES.md) | Parked comparisons; disclose supplied zeros/primes |
| `04_riemann_L_twisted` | Character/L-function instruments in the same [notebook](../theory/TNFR_RIEMANN_RESEARCH_NOTES.md) | Supplied arithmetic data and finite comparisons, not generalized RH |
| `05_type_hygiene` | [Operator and state-type scope](../theory/README.md#operators-grammar-and-support-events) | Finite delay projection, actual storage and event/window observations |
| `07_number_theory` | [Arithmetic definitions](../theory/TNFR_NUMBER_THEORY.md), residues and prime structure | Disclose factorization, sieves and other construction inputs |
| `08_emergent_geometry` | [Scale/geometry bridge](../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md), spectra and auxiliary models | Separate prescribed geometry, observed structure and autonomous generation |
| `10_applications` | [Measurement protocol](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md), adapters and backend provenance | Data admission and reserved prediction remain separate obligations |

Use `rg --files examples -g "*.py"` for the current file inventory. The grammar
automaton examples reuse `_flat_grammar_model.py`; arithmetic and physical
examples reuse their package owners. A finite fixture may recur as a controlled
comparison without constituting a second implementation of its governing law.

Optional dependencies vary by script. Plotting examples require `viz-basic`;
examples 91 and 92 additionally require `scikit-learn`, and 92 downloads and
caches UCI data. The Torch demonstration uses `compute-torch`. Read module
docstrings and available `--help` before an explicit run; the directory is not an
instruction to execute every file. Importing a demonstration does not start its
command-line workflow. Some demonstrations configure imports or plotting
defaults at module scope, so this is not a promise of side-effect-free imports.
Package ownership belongs to [Architecture](../ARCHITECTURE.md), and verification
requirements to [Testing](../TESTING.md).

## Retained controls and shared owners

[Example 04](01_foundations/04_operator_sequences.py) is a supplementary
grammar control: it compares flat word admission with an independent,
illustrative pressure proxy. It executes no operators on an evolving object.
When adding operator events, use the [event contracts](../docs/contracts/OPERATOR_EVENTS.md)
and their live-state prerequisites as well as grammar admission.

[Example 10](01_foundations/10_simplified_sdk_showcase.py) surveys supplied
topology builders, templates and configured SDK operations. Use it as a
construction reference. Example 07 is a larger prepared-state ensemble.

The type-hygiene directory retains these scoped entry points:

| Entry | What it observes |
| --- | --- |
| [77](05_type_hygiene/77_remesh_infinity_residue_split_demo.py) | Finite fixed-delay Fourier projection and window sensitivity |
| [79](05_type_hygiene/79_epi_type_signature_demo.py) | Actual scalar-chart storage membership and descriptive temporal entropy |
| [82](05_type_hygiene/82_remesh_window_type_signature_demo.py) | Selected finite REMESH event/window comparisons |

Entropy thresholds and scalar-only fixtures do not prove that a richer state
type is necessary or impossible. Use the domain and execution contracts for
[signed-form admission](../tests/test_nodal_solver_epi_scope.py),
[circular U3 admission](../tests/operators/test_u3_hard_invariant.py),
[field availability](../tests/sdk/test_nfr_observation_scope.py) and
[grammar observations](../tests/operators/test_grammar_observations.py).

For declared diffusion use [99](08_emergent_geometry/99_structural_diffusion.py).
Prescribed phase/form response belongs to [179](08_emergent_geometry/179_phase_form_driven_response.py).
Source recovery for retired examples belongs to the
[recovery index](../theory/research/archive/README.md#recovery-records);
interface changes belong to the [API migrations](../docs/API_CONTRACTS.md#removed-unsupported-helpers).

## Diffusion and runtime evidence

These supplementary controls reuse the
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

Run the native evolution continuation with:

```bash
python examples/08_emergent_geometry/180_relational_exchange.py
```

Its detached certificate comparison distinguishes exact-reflection, full-state
local and sector hypotheses. An unavailable narrower certificate can coexist
with an admitted broader one. Read the [execution guide](../docs/guides/relational/RELATIONAL_EXECUTION.md#execute-the-conditional-relational-model)
for law and storage premises.

The [scale bridge](../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) owns effective
geometry and observation closure. The tetrad is a required diagnostic interface,
not a proven complete state; reflection equivalence and loss of hidden state are
separate questions. Auxiliary symplectic/U(2) demonstrations retain their own
equations and premises; a Hamiltonian interpretation of engine operators
requires a separate argument.

The [curvature safety illustration](08_emergent_geometry/k_phi_safety_demo.py)
compares supplied WS-graph field observations before and after operator events.
Its fit and safety labels are configured advisories, not a stability theorem.

[The unified-field showcase](08_emergent_geometry/unified_fields_showcase.py)
compares seeded supplied states on a path, cycle and barbell using the shared
snapshot readout. It executes no operators or evolution. Its optional plot
reports descriptive field statistics and explicitly leaves temporal conservation
unavailable.

[Example 179](08_emergent_geometry/179_phase_form_driven_response.py) checks a
prescribed rotating phase contrast and its derived EPI response on the six-node
prism. It evaluates detached analytic snapshots rather than a native trajectory.
The [phase/form owner](../theory/nodal/PHASE_FORM_EXCHANGE.md#16-phase-and-form-directed-exchange-frames-and-the-moving-mean)
states the exact assumptions, moving mean and same-input contraction scope.
The imposed phase clock is not a derived autonomous maintenance mechanism.

```bash
python examples/08_emergent_geometry/179_phase_form_driven_response.py --output-dir output/phase_form_driven_response
```

Optional plots require the `viz-basic` extra. Retained outputs are the
[figure](../docs/assets/phase_form_driven_response/phase_form_driven_response.png),
[JSON](../docs/assets/phase_form_driven_response/phase_form_driven_response.json)
and [CSV](../docs/assets/phase_form_driven_response/phase_form_driven_response.csv).
Their recorded residuals and refinements are finite evidence, not autonomous
formation or a physical identification.

The [offline interface example](10_applications/93_structural_interface_demo.py)
compares a planted spatial target and supplied sinusoidal blocks using shared
observation adapters. It executes no coupling law or nodal trajectory. The
signal adapter currently omits coherence-length fit/fallback provenance; the
example exposes that limitation and treats operator suggestions as telemetry.
