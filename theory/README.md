# TNFR theory: reading routes, mathematical owners and implementation

Use this page to find a concept, its derivation, and the code and checks that
implement it. It is the single maintained theory catalog. The
[portfolio](../TNFR_lineas_de_investigacion.txt) classifies research priorities;
the [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns the next task. [AGENTS](../AGENTS.md) supplies working conventions,
and the [documentation map](../docs/README.md) routes engineering questions.

The nodal identity does not uniquely determine all evolution laws. Conditional
formation, maintenance, interaction and memory results now exist for specified
models and preparations. They do not establish a uniquely selected fundamental
law, autonomous support creation, arbitrary nonlinear nesting or physical
identification. Read the hypotheses of the linked owner before reusing a result.

## Choose a question

| I want to understand... | Start here | Then use |
| --- | --- | --- |
| What EPI, capacity, pressure, phase and time mean | [Foundations](FUNDAMENTAL_THEORY.md), [parameter ledger](NODAL_PARAMETER_FOUNDATIONS.md) | [State and laws](#state-and-laws) |
| How form and phase can generate a maintained pattern | [Relational law and formation](nodal/RELATIONAL_EXCHANGE_ADMISSION.md) | [Form, phase and coherent patterns](#form-phase-and-coherent-patterns) |
| What is lost when a region is treated as one node | [Composition](nodal/RELATIONAL_PATTERN_COMPOSITION.md), [joint memory](nodal/RELATIONAL_PATTERN_MEMORY.md) | [Theory to execution](#theory-to-execution) |
| What a prepared response can identify | [Coefficient, memory and initial-jet admission](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#coefficient-synergy-audit) | [Sample/error contracts](../docs/contracts/RELATIONAL_DYNAMICS.md#relational-coefficient-samples); identification is not law selection |
| Which stability or memory theorem applies | [Transport and memory](#transport-memory-and-clocks) | Check fixed/changing support, forcing, phase law and history first |
| What operators and grammar actually guarantee | [Operator and grammar block](#operators-grammar-and-support-events) | [API contracts](../docs/API_CONTRACTS.md), [CLI/SDK guide](../docs/CLI_AND_SDK.md) |
| What a field or diagnostic measures | [Field guide](../docs/STRUCTURAL_FIELDS_TETRAD.md) | [Diagnostics and geometry](#diagnostics-and-geometry) |
| What is physically tested, auxiliary or still open | [Physical comparisons](#auxiliary-models-and-physical-comparisons) | [Research and measurement](#research-measurement-and-parked-evidence) |
| What the arithmetic applications contribute | [Arithmetic and applications](#arithmetic-and-applications) | Their supplied inputs and arithmetic checks, not a physical identification |

**How to read status:** a definition or contract specifies a model/interface;
a conditional theorem needs its stated hypotheses; finite evidence records
particular tests or executions; an auxiliary construction adds premises; an
open question lacks a derivation or adequate evidence. These statuses can
coexist in one document. “Parked” concerns research priority, not invalidity.
A filename containing “canonical”, “emergent” or “unification” is not evidence.

## Directory responsibilities

| Location | Responsibility |
| --- | --- |
| `theory/*.md` | Cross-cutting foundations and established topical owners, grouped below; filenames remain stable for citations |
| `theory/nodal/` | Detailed pressure, form/phase, relational-pattern and reduction derivations |
| `theory/research/` | One execution plan and supporting measurement protocols |
| `theory/research/archive/` | Retirement records, superseded plans and evidence provenance; historical instructions are inactive |

The catalog below lists every maintained theory document once. Reading routes
and implementation links may point to the same owner without copying its proof
or its changing research status. `scripts/check_documentation.py` checks catalog
coverage and the generated website menu; link checks separately verify
destinations and anchors. After changing catalog rows or groups, run
`python scripts/check_documentation.py --write-generated` to update the menu.

<!-- BEGIN THEORY CATALOG -->

## State and laws

| Mathematical owner | What it defines or establishes |
| --- | --- |
| [Fundamental theory](FUNDAMENTAL_THEORY.md) | EPI state space, nodal identity, chart/equivalence admission and the held S0/derived D boundary |
| [Parameter foundations](NODAL_PARAMETER_FOUNDATIONS.md) | Units, zero limits, cross-channel dependencies, clocks and the threshold/constant ledger |
| [Relation foundations](nodal/RELATION_FOUNDATIONS.md) | Support, conductance, distance and causal dependence; zero-relation limits, autonomous contact countermodels and undetermined relation storage |
| [Pressure premises](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md) | Locality, covariance, phase-domain boundaries and what does not select a unique pressure law |
| [Joint parameter response](nodal/JOINT_PARAMETER_RESPONSE.md) | Signed form, pressure derivatives, pressure-state closure and finite phase/capacity/source compatibility |
| [Glossary](GLOSSARY.md) | Uniform concept cards with checked classifications, dependencies, evidence links and a generated index; linked owners retain definitions and proofs |

## Form, phase and coherent patterns

| Mathematical owner | What it establishes and what remains separate |
| --- | --- |
| [Inherited form dynamics](nodal/INHERITED_FORM_DYNAMICS.md) | Fine-to-coarse pushforwards, inherited pressure/metric, field reconstruction and changing support |
| [Phase-to-form exchange](nodal/PHASE_FORM_EXCHANGE.md) | Causal-source matching, regional frames and response to a prescribed phase input |
| [Primitive phase closure](nodal/PRIMITIVE_PHASE_CLOSURE.md) | Symmetry and closure obstructions, oriented response and the actual runtime phase-writer inventory |
| [Derived form phase](nodal/DERIVED_FORM_PHASE.md) | Regional Cartesian/Gram geometry, fixed affine closure, derived angles and source-relative predictions |
| [Relational exchange, recovery and formation](nodal/RELATIONAL_EXCHANGE_ADMISSION.md) | Explicit joint law, recovery/capture, coefficient nonselection and prepared identification with bounded software acquisition; support, constitutive premises and physical admission remain separate |
| [Relational pattern composition](nodal/RELATIONAL_PATTERN_COMPOSITION.md) | Tangent closure, nonlinear state/rate obstructions, support-event budgets/nonselection, and conditional restoring effective links |
| [Relational pattern memory](nodal/RELATIONAL_PATTERN_MEMORY.md) | Hidden-state memory, controlled fast limits, inherited series composition and collective three-port interaction, frozen response, return geometry and damped modes; supplied fine support and scoped approximation bounds |
| [Emergence and ontology](EMERGENT_ONTOLOGY.md) | Which formation/identity claims are supported, which mechanisms can be reused, and which ontological claims remain open |

## Transport, memory and clocks

| Mathematical owner | Domain and reuse boundary |
| --- | --- |
| [EPI diffusion stability](TNFR_DIFFUSION_STABILITY_THEOREM.md) | Reversible and restricted time-varying diffusion, metric/solver bounds and finite executor evidence; not full-runtime stability |
| [Directed nonnormal dynamics](TNFR_DIRECTED_NONNORMAL_DYNAMICS.md) | Directed transport, transient amplification and clock changes (historical R9); eigenvalue decay alone is insufficient |
| [Derived EPI memory](DERIVED_EPI_MEMORY.md) | Linear/affine hidden-state elimination, initial sources, observation closure and error propagation |
| [Capacity and localization](CAPACITY_LOCALIZATION_BALANCE.md) | Held-capacity balance, release and admission of proposed form/capacity laws |
| [Cycle support dynamics](CYCLE_SUPPORT_DYNAMICS.md) | Joint cycle budgets and finite reorganization clocks under the stated phase/capacity laws |
| [Cycle memory relaxation](CYCLE_MEMORY_RELAXATION.md) | Restricted delayed REMESH/history contraction; distinct from eliminating hidden phase/form state |
| [Forced support balance](FORCED_SUPPORT_BALANCE.md) | Held-source restoration, reference changes, event work and clock boundaries |
| [REMESH fixed-delay models](REMESH_INFINITY_DERIVATION.md) | History companion systems, filters and runtime defects; no unrestricted infinite-runtime theorem |
| [Scale, geometry and bridge](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) | Quotients, shared realizations, reflection invariants, field dependencies and coherence-length limits |

## Operators, grammar and support events

| Mathematical owner | Domain and reuse boundary |
| --- | --- |
| [Structural operators](STRUCTURAL_OPERATORS.md) | Named transformations and their interpretation; registered catalog size does not prove mathematical completeness |
| [Unified grammar](UNIFIED_GRAMMAR_RULES.md) | U1–U6 word and live-state contracts, with configured policy premises |
| [Diagnostic and grammar scope](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) | Counterexamples, constitutive closure audit and limits of grammar/diagnostic claims |
| [Coupling and winding](COUPLING_WINDING_PERSISTENCE.md) | Restricted cycle gap diffusion, winding retention and event/runtime exclusions |
| [Coherent pattern contact](COHERENT_PATTERN_CONTACT.md) | Prepared contact, two-port recovery and formation barriers under its stated event/phase laws |
| [THOL pressure feedback](THOL_PRESSURE_FEEDBACK.md) | History-driven pressure, refresh order and disconnected-child limitations |
| [THOL birth and transport](THOL_BIRTH_AND_TRANSPORT.md) | Configured child creation, admitted transport and source/event accounting; not spontaneous substrate creation |
| [Child coupling feedback](CHILD_COUPLING_FEEDBACK.md) | Child-target Coupling, changes of reference and comparison with the original profile |

## Diagnostics and geometry

| Mathematical owner | Domain and reuse boundary |
| --- | --- |
| [Minimal structural degrees](MINIMAL_STRUCTURAL_DEGREES.md) | Complementary tetrad observations and counterexamples to complete-state reconstruction |
| [Structural conservation diagnostics](STRUCTURAL_CONSERVATION_THEOREM.md) | Actual temporal balance, currents and residuals; a field energy needs its own monotonicity proof |
| [Stability and dynamics diagnostics](STRUCTURAL_STABILITY_AND_DYNAMICS.md) | Operational stability/lifecycle observations and their implementation boundaries |
| [Extended fields](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) | Algebraic derived quantities, availability and emergent-property audit; static statistics are not temporal laws |

## Auxiliary models and physical comparisons

| Mathematical owner | Added premises and open bridge |
| --- | --- |
| [Variational and exchange models](TNFR_VARIATIONAL_PRINCIPLE.md) | Auxiliary action/Hamiltonian read-outs, exact restricted diffusion bridges and conditional cotangent closure |
| [Dissipative/open-system models](DISSIPATIVE_AND_OPEN_SYSTEMS.md) | Specified forcing and dissipative constructions with their neutral-mode boundaries |
| [Gauge and polarization constructions](GAUGE_SYMMETRY_AND_UNIFICATION.md) | Classical auxiliary symmetries; no engine-wide gauge or particle-unification theorem |
| [Physical phenomena atlas](PHYSICAL_REGIME_CORRESPONDENCES.md) | Conditional reductions, supplied analogues, measurement prerequisites and unresolved physical correspondences |

## Arithmetic and applications

These are scoped mathematical constructions and application references, not
parallel active campaigns or evidence that arithmetic nodes are physical NFRs.
Historical R identifiers remain searchable here; their proofs own the claims.

| Mathematical owner | Contribution and boundary |
| --- | --- |
| [Number theory](TNFR_NUMBER_THEORY.md) | Arithmetic encodings, pressure and primality criterion; supplied divisor information is explicit |
| [Structural observability — R1](TNFR_STRUCTURAL_OBSERVABILITY.md) | Symmetry-sector observations and finite operator/selector scope |
| [Arithmetic dynamics — R2](TNFR_ARITHMETIC_DYNAMICS.md) | Pulse recurrence for the declared residue-network construction |
| [CRT composition — R3](TNFR_CRT_FRACTALITY.md) | Chinese-remainder transport/product synthesis |
| [p-adic dynamics — R4](TNFR_PADIC_DYNAMICS.md) | Projective transport; a static lift is not a temporal REMESH echo |
| [Algebraic number fields — R5](TNFR_ALGEBRAIC_NUMBER_FIELDS.md) | Finite/algebraic-field extensions and classification limits |
| [Additive dynamics — R6](TNFR_ADDITIVE_DYNAMICS.md) | Controlled Fourier construction; no demonstrated TNFR-specific excess |
| [Arithmetic pressure — R7](TNFR_ARITHMETIC_PRESSURE.md) | Channel independence and redundancy for primality; completeness remains open |
| [Arithmetic operators — R8](TNFR_ARITHMETIC_OPERATORS.md) | Synthetic effect comparisons, not actual glyph/grammar execution |
| [Riemann scope](TNFR_RIEMANN_RESEARCH_NOTES.md) | Current arithmetic/spectral evidence and unresolved analytic bridge; not a proof of RH |
| [Prime-ladder atlas](NUCLEUS_A_PRIME_LADDER_ATLAS.md) | Internal reproduction of supplied spectra and classical identities |
| [Equivariance obstructions](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md) | Conditional negative algebraic results, without an exhaustion theorem |
| [Applied structural analysis](APPLIED_STRUCTURAL_ANALYSIS.md) | Factorization heuristics, arithmetic verification and fallback/evaluation provenance |

## Research, measurement and parked evidence

| Owner | Responsibility |
| --- | --- |
| [Execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) | Sole active F1–F4 task queue; supporting P1–P5 admission gates |
| [Research strategy](NODAL_RESEARCH_STRATEGY.md) | Scientific rationale, premise review and current/historical reuse audit |
| [Passive transport protocol](research/PASSIVE_TRANSPORT_PROTOCOL.md) | Independent measurement and clock mapping, calibration/evaluation separation |
| [Regional phase/amplitude protocol](research/PHASE_AMPLITUDE_MEASUREMENT_PROTOCOL.md) | Collective observation map, uncertainty and data-source admission |
| [TCLab exploratory protocol](research/TCLAB_EXPLORATORY_PROTOCOL.md) | Frozen driven thermal comparison and unresolved physical admission |
| [C6 mechanism audit](C6_RESEARCH_MECHANISM_AUDIT.md) | Parked finite evidence: 41/56 labels excluded, 15 open; global stability is not closed |

<!-- END THEORY CATALOG -->

## Theory to execution

Follow the actual path before transferring a theorem. These are representative
integration checks, not a claim that every module or theorem has been verified.
[Architecture](../ARCHITECTURE.md) owns dispatch; [Testing](../TESTING.md) owns
selection and commands; the [SDK guide](../docs/CLI_AND_SDK.md) owns usage.

| Mechanism | Shared implementation | Representative checks | Public use and limit |
| --- | --- | --- | --- |
| Configured nodal pressure | [DNFR](../src/tnfr/dynamics/dnfr.py), [nodal row](../src/tnfr/dynamics/canonical.py) | [Core contracts](../tests/core_physics/README.md) | `runtime.step` also includes configured phase/capacity/events; diffusion hypotheses cannot be assumed |
| Operator-word studies | [Study runner](../src/tnfr/sdk/study.py), [CLI adapter](../src/tnfr/cli/study.py) | [SDK studies](../tests/sdk/test_study.py), [CLI wiring](../tests/cli/test_study_commands.py) | `run_study` / `tnfr network`; cycles count words, not seconds |
| Regional form and held-source response | [Form geometry](../src/tnfr/physics/form_geometry.py), [source-relative form](../src/tnfr/physics/source_relative_form.py) | [Form controls](../tests/physics/test_form_geometry.py), [source controls](../tests/physics/test_source_relative_form_response.py) | `Network.regional_form` / `source_relative_form`; stored-pressure observations, not a new solver |
| Conditional joint evolution | [Relational engine](../src/tnfr/dynamics/relational.py) | [Execution](../tests/test_relational_exchange_execution.py), [regular domain](../tests/test_relational_regular_execution.py) | `relational_exchange` / `step_relational`; opt-in held-capacity law, separate from CLI word studies |
| Prepared coefficient response and uncertainty | [Jet/sample observations](../src/tnfr/physics/relational_observations.py), [shared rational intervals](../src/tnfr/mathematics/_rational_interval.py), [exact report export](../src/tnfr/sdk/relational_reports.py) | [Routine admission/export](../tests/test_relational_coefficient_identification.py), [mathematical controls](../tests/physics/test_relational_coefficient_scope.py) | Module-level `bound_relational_coefficient_from_jet` / `bound_relational_coefficient_from_samples`; supplied preparation, affine clock and errors; no graph mutation, automatic precision pass or universal chi |
| Retained temporal acquisition evidence | [Read-only record auditor](../src/tnfr/research/relational_acquisition.py), [frozen P2 producer](../benchmarks/relational_coefficient_acquisition.py) | [Audit admission](../tests/test_relational_acquisition_audit.py), [original record and protocol controls](../tests/physics/test_relational_coefficient_acquisition.py) | `audit_relational_coefficient_acquisition`; reconstructs saved errors and original verdict without producer replay; known-source software evidence, unavailable if artifacts are absent |
| Pressure and environmental state | [Same-law closure and P2 chart](nodal/JOINT_PARAMETER_RESPONSE.md#pressure-state-closure), [native evaluator](../src/tnfr/dynamics/relational.py) | [Zero-pressure and sufficient-coordinate controls](../tests/physics/test_relational_pressure_state.py) | Existing field evaluation; pressure alone loses state, while the restricted chart or its memory retains it; no physical-vacuum identification |
| Pattern geometry and exchange budgets | [Relational observations](../src/tnfr/physics/relational_observations.py), [support transport](../src/tnfr/physics/support_transport.py) | [Pattern observations](../tests/test_relational_pattern_observation.py), [phase response](../tests/test_relational_phase_response.py) | `relational_pattern`; fresh detached field, supplied regions/frame, retained complete centered state |
| Supplied attachment and event supply | [Relational observations](../src/tnfr/physics/relational_observations.py) | [Static interface/budget controls](../tests/test_relational_attachment.py), [SDK reports](../tests/sdk/test_relational_reports.py) | `relational_attachment`, `assess_supply`; represented budget, separate component and joined fields, no live event or autonomous selector |
| Identity-preserving bridge relocation | [Relocation and recovery theorem](nodal/RELATIONAL_PATTERN_COMPOSITION.md#identity-preserving-bridge-relocation), [shared observations](../src/tnfr/physics/relational_observations.py) | [Static event controls](../tests/test_relational_attachment.py), [SDK reports](../tests/sdk/test_relational_reports.py) | `relational_relocation`; before/after fields and support budget; snapshot comparison is not automatic theorem admission or event selection |
| Event choice and clock restrictions | [Support-law closure](nodal/RELATIONAL_PATTERN_COMPOSITION.md#support-law-choice-and-clock), [exact finite-action owner](../src/tnfr/physics/selector_symmetry.py) | [Event-action controls](../tests/physics/test_relational_event_selection.py), [whole-law clock controls](../tests/physics/test_relational_event_clock.py) | Existing observers and exact incidence declarations; no autonomous selector, stochastic execution or event timer |
| Complete operator actions and connection | [Joint reset/contact derivation](nodal/RELATIONAL_PATTERN_COMPOSITION.md#nodal-reorganization-and-contact), [all-operator mechanism map](STRUCTURAL_OPERATORS.md#operator-mechanism-and-activation-audit) | [Actual UM/RA budgets](../tests/physics/test_coupling_attachment_budget.py), [AL/EN and phase admission](../tests/physics/test_relational_contact_admission.py), [SDK reset controls](../tests/sdk/test_relational_reset.py) | `Network.relational_reset` and shared exact export; no activation law or post-event recovery certificate |
| Native rhythm and precontact locking | [Pulse scope](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-pulse-scope), [locking admission](nodal/RELATIONAL_PATTERN_COMPOSITION.md#precontact-rhythm-and-locking) | [Existing joint-mode controls](../tests/physics/test_relational_local_recovery.py), [native contact counterexamples](../tests/physics/test_relational_contact_admission.py) | Reuses `relational_exchange` / `step_relational`; no new pulse variable, auxiliary clock or synchronization-triggered event |
| Protected basin and continuous transit | [Capture](../src/tnfr/physics/relational_capture.py), [transit](../src/tnfr/physics/relational_transit.py) | [Capture](../tests/test_relational_capture.py), [SDK reports](../tests/sdk/test_relational_reports.py) | `relational_*capture`; sufficient scoped certificates, distinct from stepping or an automatic pattern detector |
| Linear reduction and nonlinear memory | [Invariant-row and coordinate-memory algebra](../src/tnfr/mathematics/linear_observation.py), [memory prediction](../benchmarks/relational_memory_prediction.py), [mediator intervention](../benchmarks/relational_mediation_response.py) | [Exact closure/split](../tests/test_linear_observation.py), [joint mediation](../tests/physics/test_relational_mediation.py), [frozen memory response](../tests/physics/test_relational_memory_response.py), [mediator protocol](../tests/physics/test_relational_mediation_response.py) | Supplied linear models and scoped research instruments; no reduced nonlinear SDK evolution or fitted memory law |
| Collective return geometry | [Cycle reconstruction](../src/tnfr/physics/phase_cycle_geometry.py), [static root admission](../benchmarks/relational_return_geometry.py) | [Cycle contracts](../tests/physics/test_phase_cycle_geometry.py), [return geometry](../tests/physics/test_relational_return_geometry.py) | Supplied return edge; exact periods, certified root bracket and represented field residuals remain distinct; no event, trajectory or sustained pulse |
| Native causal oscillatory response | [Shared uniform-form tangent](../src/tnfr/dynamics/relational.py), [static collective modes](../benchmarks/relational_collective_pulse.py) | [Tangent contracts](../tests/test_relational_uniform_tangent.py), [causal mode controls](../tests/physics/test_relational_collective_pulse.py), [SDK/export](../tests/sdk/test_relational_uniform_tangent.py) | `relational_uniform_tangent`; ideal trace-sign proof and numerical poles/residues have distinct scopes; no auxiliary oscillator or finite nonlinear pulse certificate |
| Restoring effective links and environment | [Conditional path/capture theorem](nodal/RELATIONAL_PATTERN_COMPOSITION.md#effective-link-admission), [environmental capture](nodal/RELATIONAL_PATTERN_COMPOSITION.md#environmental-capture-domain), [nonlinear fast limit](nodal/RELATIONAL_PATTERN_MEMORY.md#fast-mediator-reduction), [series composition](nodal/RELATIONAL_PATTERN_MEMORY.md#two-mediator-composition), [collective three-port law](nodal/RELATIONAL_PATTERN_MEMORY.md#three-port-collective-interaction), [native field/tangent](../src/tnfr/dynamics/relational.py) | [Paths, cuts, capture and pressure insufficiency](../tests/physics/test_relational_effective_link.py), [moving-boundary, fast-field, series and collective controls](../tests/physics/test_relational_mediation.py) | Controlled local fast limits retain inherited port geometry; three-port phase interaction is not an independent-pair sum; static native integration, no new reduced executor or fixed-capacity scalar closure |
| Fields and diagnostic provenance | [Fields](../src/tnfr/physics/fields.py), [study diagnostics](../src/tnfr/sdk/study.py) | [Read-out consistency](../tests/physics/test_field_readout_consistency.py), [SDK diagnostics](../tests/sdk/test_study.py) | `tetrad` / `diagnose_network`; unavailable fields and estimator provenance remain explicit |
| Named events and grammar | [Registry](../src/tnfr/operators/operator_contracts.py), [atomic stages](../src/tnfr/operators/network_stage.py) | [Event execution](../tests/operators/test_operator_event_runtime.py) | [API contracts](../docs/API_CONTRACTS.md); operator events are not finite-duration nodal solutions |
| Connection state, zero boundary and autonomous contact | [Relation foundations](nodal/RELATION_FOUNDATIONS.md), [neighbor differences](../src/tnfr/mathematics/_neighbor_differences.py), [pressure](../src/tnfr/dynamics/dnfr.py), [edge semantics](../src/tnfr/physics/_edge_semantics.py) | [Absent, zero and positive conductance](../tests/core_physics/test_stable_neighbor_pressure.py), [support accounting](../tests/physics/test_support_transport.py), [conditional contact controls](../tests/physics/test_constitutive_support_scope.py) | Existing channel semantics are preserved; weighted autonomous completions and relation storage are unselected hypotheses, not an edge-birth executor |
| Arithmetic applications | [Factorization guide](../applications/factorization-lab/README.md), [primality guide](../applications/primality-test/README.md) | Each application's declared arithmetic verification | Supplied encodings, candidate heuristics and fallbacks; separate from physical generative research |

The SDK delegates to the shared owners; it is not another model implementation.
A public adapter is warranted for a stable contract, not for every exploratory
coefficient calculation. Detached reports are evidence projections, not full
resumable checkpoints. The single-bridge local memory preparation and two-bridge
formation/capture supports have different hypotheses despite sharing a law.

## Historical aliases and maintenance

The former S1–S16 map referred to stability/Lyapunov/solver questions (S1, S2,
S4, S5, S13), state/geometry/reduction (S3, S8, S9, S11, S14), transitions/support
(S6, S7), catalog completeness (S10), inverse observation (S15) and effective
NFR identity (S16). Their topical owners are cataloged above. These aliases
are not completion labels or a second active queue. R1–R9 are listed with their
owners; the old B0–B11 catalog audit was finite implementation evidence, not a
proof of universal operator completeness.

The [archive](research/archive/README.md) owns retirement and recovery records.
[Retired programme boundaries](research/archive/RETIRED_PROGRAMME_BOUNDARIES.md)
preserve useful negative conclusions from removed physical-programme wrappers.
Do not treat an archived instruction as current work or rewrite frozen results
when code evolves. Preserved numerical records retain model, inputs, source,
precision, path and outcome; tests do not turn finite evidence into a theorem.

Update a derivation and its affected implementation/tests together. Update this
catalog only when ownership, scope or navigation changes, and the execution
plan only when a task or priority changes. Do not append the same result to
every guide or restore a parallel status index.
