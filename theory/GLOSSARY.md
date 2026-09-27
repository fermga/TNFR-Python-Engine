# TNFR concept glossary

This compact vocabulary maps concepts to their maintained definitions, proofs
and execution owners. It is not another theory owner or research queue. Use the
[theory catalog](README.md) and the authority hierarchy in [AGENTS.md](../AGENTS.md)
to resolve conflicts against explicit premises, mathematics and actual execution.

**Canonical** means a maintained definition or execution contract here; it does
not mean uniquely derived or physically established. **Fundamental** names a
primitive or an identity of a declared model, whose premises remain revisable.
**Emergence** requires a dynamical result under stated assumptions. Computing a
quantity, assigning a label or configuring a controller is not that result.

Each card separates its basis from its emergence claim. Primitive state,
identities, constitutive premises, derived consequences, diagnostics, policies,
auxiliary models and hypotheses have different roles. **Maintained** means the
scoped declaration is maintained, not that nature has validated it. **Candidate**
retains an unresolved proposal. Conditional emergence is reserved for derived
dynamical results with explicit hypotheses. Exact identities, conditional proofs,
represented arithmetic and finite numerical observations remain distinct evidence.

Dependencies are prerequisites of definitions or justifications, not the
interaction edges of the dynamical system. Their acyclic declaration does not
exclude coupled phase/form feedback. Related-reading links need not be dependencies.

The cards are the only editable concept catalog. Generate their index with
`python scripts/check_documentation.py --write-generated`.

<!-- BEGIN GENERATED CONCEPT INDEX -->

| Concept | Basis | Emergence | Status |
| --- | --- | --- | --- |
| [Form (EPI)](#epi) | primitive | not-claimed | maintained |
| [Reorganization capacity (nu_f)](#capacity) | primitive | not-claimed | maintained |
| [Primitive phase (phi, theta)](#phase) | primitive | not-claimed | maintained |
| [Support, conductance and distance](#support) | primitive | not-claimed | maintained |
| [Structural clock](#clock) | primitive | not-claimed | maintained |
| [Fractal-Resonant Node (NFR)](#nfr) | primitive | not-claimed | maintained |
| [Structural pressure (DeltaNFR)](#pressure) | constitutive | not-claimed | maintained |
| [Nodal equation and complete closure](#nodal-equation) | identity | not-claimed | maintained |
| [Conditional relational exchange law](#relational-law) | constitutive | not-claimed | maintained |
| [Relative resultant, phase metric and mobility](#phase-metric) | derived | not-claimed | maintained |
| [Joint storage and signed exchange work](#storage-work) | derived | not-claimed | maintained |
| [Pure-EPI diffusion and relaxation](#diffusion) | derived | conditional | maintained |
| [Conditional pattern formation and recovery](#pattern-recovery) | derived | conditional | maintained |
| [Derived form geometry](#form-geometry) | derived | not-claimed | maintained |
| [Coarse closure and hidden-state obstruction](#closure) | derived | not-claimed | maintained |
| [Derived hidden-state memory](#hidden-memory) | derived | not-claimed | maintained |
| [Supplied bridge interfaces](#attachment) | derived | not-claimed | maintained |
| [Cycle winding](#winding) | derived | not-claimed | maintained |
| [Coherence read-out (C)](#coherence) | diagnostic | not-claimed | maintained |
| [Pressure/rate equilibrium test](#equilibrium) | diagnostic | not-claimed | maintained |
| [Sense Index (Si)](#sense-index) | diagnostic | not-claimed | maintained |
| [Structural-field tetrad](#tetrad) | diagnostic | not-claimed | maintained |
| [Structural potential (Phi_s)](#potential) | diagnostic | not-claimed | maintained |
| [Phase gradient magnitude](#phase-gradient) | diagnostic | not-claimed | maintained |
| [Circular phase curvature (K_phi)](#phase-curvature) | diagnostic | not-claimed | maintained |
| [Coherence range (xi_C)](#coherence-length) | diagnostic | not-claimed | maintained |
| [Structural currents and balance diagnostic](#balance-diagnostic) | diagnostic | not-claimed | maintained |
| [Named structural operators](#operators) | constitutive | not-claimed | maintained |
| [Grammar and automatic selection](#grammar) | policy | not-claimed | maintained |
| [Thresholds, gains and numerical tolerances](#thresholds) | policy | not-claimed | maintained |
| [Spectral pulse and modal rhythm](#pulse) | auxiliary | not-claimed | maintained |
| [Auxiliary symplectic substrate](#symplectic) | auxiliary | not-claimed | maintained |
| [Arithmetic and Riemann constructions](#arithmetic) | auxiliary | not-claimed | maintained |
| [Physical measurement bridge](#measurement-bridge) | constitutive | not-claimed | candidate |
| [Physical constituents from TNFR](#physical-emergence) | hypothesis | not-claimed | candidate |

<!-- END GENERATED CONCEPT INDEX -->

<!-- BEGIN CONCEPT CARDS -->

## Supplied state and complete models

<a id="epi"></a>

### Form (EPI)

- **Status:** maintained
- **Basis:** primitive
- **Emergence:** not-claimed
- **Definition:** Structural configuration x in a declared form space; the graph scalar chart is signed real.
- **Domain:** Declared form units X; finite real values or uniform-real BEPI for scalar-only consumers.
- **Premises:** Choose the state space and chart before specifying evolution.
- **Dependencies:** none
- **Owner:** [Form foundations](FUNDAMENTAL_THEORY.md#foundational-state-admission).
- **Evidence:** [Types and implementation](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation).
- **Implementation:** [Scalar admission](../src/tnfr/types.py).
- **Tests:** [EPI domain](../tests/operators/test_epi_domain_consistency.py).
- **Limits:** Magnitude cannot substitute for the signed structure of richer BEPI; EPI is not automatically voltage, energy or matter.

<a id="capacity"></a>

### Reorganization capacity (nu_f)

- **Status:** maintained
- **Basis:** primitive
- **Emergence:** not-claimed
- **Definition:** Nonnegative multiplier of pressure in the nodal form rate, relative to the declared clock.
- **Domain:** Finite nonnegative rates in inverse structural-clock units; zero suppresses the unforced form row.
- **Premises:** Supply capacity or independently admit its evolution; Hz_str does not identify laboratory seconds.
- **Dependencies:** clock
- **Owner:** [Capacity foundations](NODAL_PARAMETER_FOUNDATIONS.md#31-capacity-clock-and-positivity-require-compatible-laws).
- **Evidence:** [Capacity-law admission](CAPACITY_LOCALIZATION_BALANCE.md#capacity-law-admission).
- **Implementation:** [Nodal arithmetic](../src/tnfr/dynamics/canonical.py).
- **Tests:** [Nodal contracts](../tests/core_physics/test_canonical_nodal_equation.py).
- **Limits:** Capacity is neither the actual form derivative nor angular frequency without another law; zero does not imply full equilibrium.

<a id="phase"></a>

### Primitive phase (phi, theta)

- **Status:** maintained
- **Basis:** primitive
- **Emergence:** not-claimed
- **Definition:** Circular synchronization coordinate in S¹, using a declared real lift where required.
- **Domain:** Finite represented angles in radians; wrapped differences require a branch convention.
- **Premises:** Phase evolution requires an independent supplied or derived row.
- **Dependencies:** none
- **Owner:** [Parameter ledger](NODAL_PARAMETER_FOUNDATIONS.md#2-parameter-and-dependency-ledger).
- **Evidence:** [Primitive phase closure](nodal/PRIMITIVE_PHASE_CLOSURE.md#17-primitive-phase-origin-symmetry-retained-state-and-the-missing-row).
- **Implementation:** [Circular admission](../src/tnfr/operators/_phase_gate.py).
- **Tests:** [Circular semantics](../tests/operators/test_circular_phase_semantics.py).
- **Limits:** Phase is not itself a monotone clock or an angle derived from EPI contrasts.

<a id="support"></a>

### Support, conductance and distance

- **Status:** maintained
- **Basis:** primitive
- **Emergence:** not-claimed
- **Definition:** Support names neighbors; conductance weights form transport; path distance supplies a separate observation geometry.
- **Domain:** The graph model declares direction, multiplicity, isolates and admissible weights.
- **Premises:** Initial support and any support evolution or intervention must be stated.
- **Dependencies:** none
- **Owner:** [Pressure dependencies](NODAL_PARAMETER_FOUNDATIONS.md#21-the-implemented-pressure-is-a-specified-relational-map).
- **Evidence:** [Representation contract](NODAL_PARAMETER_FOUNDATIONS.md#22-pressure-execution-and-representation-contract).
- **Implementation:** [Support transport](../src/tnfr/physics/support_transport.py).
- **Tests:** [Regional cut](../tests/test_regional_support_cut.py).
- **Limits:** A zero-conductance edge can remain a phase neighbor. Transport weights and diagnostic lengths need not coincide.

<a id="clock"></a>

### Structural clock

- **Status:** maintained
- **Basis:** primitive
- **Emergence:** not-claimed
- **Definition:** Declared parameter t relative to which continuous rates and timestamps are interpreted.
- **Domain:** Declared time units T and finite admitted intervals; Boolean durations and lost nonzero values are invalid.
- **Premises:** Clock changes transform the complete law, including capacity evolution.
- **Dependencies:** none
- **Owner:** [Clock covariance](NODAL_PARAMETER_FOUNDATIONS.md#pressure-clock-full-state-closure).
- **Evidence:** [Joint unit changes](NODAL_PARAMETER_FOUNDATIONS.md#3-joint-changes-of-form-and-time-units).
- **Implementation:** [Time admission](../src/tnfr/_exact_time.py).
- **Tests:** [Parameter covariance](../tests/physics/test_nodal_parameter_covariance.py).
- **Limits:** Operator-word counts are not elapsed seconds; synchronization alone supplies no physical clock.

<a id="nfr"></a>

### Fractal-Resonant Node (NFR)

- **Status:** maintained
- **Basis:** primitive
- **Emergence:** not-claimed
- **Definition:** Modeled region of structural coherence carrying form, capacity and phase within a declared network.
- **Domain:** A supplied node or region and its stated identity; nesting supplies operational multiscale structure.
- **Premises:** Specify membership, internal state and coupling before testing persistence or formation.
- **Dependencies:** epi, capacity, phase, support
- **Owner:** [Substrate and scale](FUNDAMENTAL_THEORY.md#29-assumed-substrate-and-emergence-between-scales).
- **Evidence:** [Identity obligations](NODAL_RESEARCH_STRATEGY.md#closure-audit-equations-events-and-identity).
- **Implementation:** [SDK NFR observation](../src/tnfr/sdk/simple.py).
- **Tests:** [NFR observation scope](../tests/sdk/test_nfr_observation_scope.py).
- **Limits:** A graph node, nested child or topology label does not establish autopoiesis, persistent identity or a physical constituent.

<a id="pressure"></a>

### Structural pressure (DeltaNFR)

- **Status:** maintained
- **Basis:** constitutive
- **Emergence:** not-claimed
- **Definition:** Response p driving form change; its realization is selected independently of the written nodal product.
- **Domain:** Tangent response in the chosen form chart, with units [x]/([nu_f][t]).
- **Premises:** The native mixture specifies form, phase, capacity and topology channels and coefficients.
- **Dependencies:** epi, capacity, phase, support, clock
- **Owner:** [Pressure scope](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md).
- **Evidence:** [Specified relational map](NODAL_PARAMETER_FOUNDATIONS.md#21-the-implemented-pressure-is-a-specified-relational-map).
- **Implementation:** [Native pressure](../src/tnfr/dynamics/dnfr.py).
- **Tests:** [Pressure orientation](../tests/physics/test_dnfr_canonical_orientation.py).
- **Limits:** Stored and refreshed pressure differ. The full mixture is not generally a fixed-metric gradient; retrospective derivative reconstruction is not prediction.

<a id="nodal-equation"></a>

### Nodal equation and complete closure

- **Status:** maintained
- **Basis:** identity
- **Emergence:** not-claimed
- **Definition:** The unforced continuous form row is dx/dt = nu_f * p.
- **Domain:** An admitted form chart, pressure, capacity and clock.
- **Premises:** This model identity requires additional laws for all consumed state and events to determine a full evolution.
- **Dependencies:** epi, capacity, pressure, clock
- **Owner:** [Nodal equation](FUNDAMENTAL_THEORY.md#21-nodal-equation).
- **Evidence:** [Integrated form](FUNDAMENTAL_THEORY.md#23-integrated-form-and-stability-criterion).
- **Implementation:** [Canonical derivative](../src/tnfr/dynamics/canonical.py), [integrators](../src/tnfr/dynamics/integrators.py).
- **Tests:** [Nodal equation](../tests/core_physics/test_canonical_nodal_equation.py), [forcing scope](../tests/test_nodal_forcing_scope.py).
- **Limits:** Declared Gamma adds an independent rate and can move form at zero capacity. Hybrid jumps are separate; the identity does not select their occurrence.

<a id="relational-law"></a>

### Conditional relational exchange law

- **Status:** maintained
- **Basis:** constitutive
- **Emergence:** not-claimed
- **Definition:** Joint rows xdot = nu_f*(-e*q/d + w*g) and thetadot = (w/beta)*nu_f*q/H, where q_i=sum_neighbors(x_i-x_j) and d_i is the degree.
- **Domain:** Fixed connected simple reciprocal unit support with at least two nodes, held nonnegative capacities, e≥0, w,beta>0 and admitted phase chamber.
- **Premises:** Native pressure, joint storage and local capacity separability select this conditional completion.
- **Dependencies:** nodal-equation, phase-metric
- **Owner:** [Capacity-separable law](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-separable-exchange).
- **Evidence:** [Alternative capacity premise](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-mediated-exchange-counterexample).
- **Implementation:** [Relational engine](../src/tnfr/dynamics/relational.py).
- **Tests:** [Exchange execution](../tests/test_relational_exchange_execution.py).
- **Limits:** This opt-in law differs from default operator runtime; the nodal identity does not uniquely select its premises or beta.

## Derived structure and scoped dynamical results

<a id="phase-metric"></a>

### Relative resultant, phase metric and mobility

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** z_i = sum_j exp(i*(theta_j-theta_i)); g_i = Arg(z_i)/pi; H_i = pi*abs(z_i)*sinc(Arg(z_i)); phase mobility is nu_i/H_i.
- **Domain:** Reciprocal unit support without isolates, nonzero z_i and Arg(z_i) in (-pi,pi); H_i is positive on this regular chart. Acute edges suffice.
- **Premises:** Use the unweighted neighbor resultant and sinc(a)=sin(a)/a, continuously extended to 1 at zero.
- **Dependencies:** phase, support, capacity
- **Owner:** [Relational geometry](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#1-state-inherited-geometry-and-the-independent-premise).
- **Evidence:** [Phase row and domain](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#2-forced-phase-row-and-its-domain).
- **Implementation:** [Captured relational field](../src/tnfr/dynamics/relational.py).
- **Tests:** [Regular execution](../tests/test_relational_regular_execution.py).
- **Limits:** Resultant/branch boundaries are not successful zero values. Materialized trigonometric sums carry no automatic exact-real error enclosure.

<a id="storage-work"></a>

### Joint storage and signed exchange work

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** For S = E_D + beta*V, with E_D = sum_edges (x_i-x_j)^2/2 and V = sum_edges [1-cos(theta_j-theta_i)], the ideal balances are Sdot=-sum D_i and E_D_dot=-sum D_i+sum J_i. Count unordered edges once.
- **Domain:** The held-support relational law; q=Bx, loss D_i=e*nu_i*q_i²/d_i and signed exchange J_i=w*nu_i*q_i*g_i.
- **Premises:** Use the admitted held-capacity relational law without forcing or support events; the choice of storage and beta remains supplied.
- **Dependencies:** relational-law
- **Owner:** [Shared signed work](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-work-integration).
- **Evidence:** [Joint storage balances](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#3-complete-source-storage-and-mean-balances).
- **Implementation:** [Field work](../src/tnfr/dynamics/relational.py), [regional observations](../src/tnfr/physics/relational_observations.py).
- **Tests:** [Relational work](../tests/test_relational_work_observation.py).
- **Limits:** This is not measured physical energy or the tetrad energy diagnostic. Numerical residuals, regional source terms and support-event costs remain explicit.

<a id="diffusion"></a>

### Pure-EPI diffusion and relaxation

- **Status:** maintained
- **Basis:** derived
- **Emergence:** conditional
- **Definition:** Isolated form pressure is -L_rw*x; fixed positive capacities and reciprocal nonnegative conductance give consensus when the positive-conductance graph is connected.
- **Domain:** Source-free form channel, fixed conductance/support and held positive capacities; disconnected components and isolates have separate conclusions.
- **Premises:** Select only the isolated EPI channel without forcing or events. The resulting generator conserves the d_i/nu_i-weighted form total; conservation is a consequence, not an extra assumption.
- **Dependencies:** nodal-equation, support
- **Owner:** [Diffusion stability](TNFR_DIFFUSION_STABILITY_THEOREM.md).
- **Evidence:** [Ideal theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md#ideal-real-arithmetic-theorem).
- **Implementation:** [Structural diffusion](../src/tnfr/physics/structural_diffusion.py).
- **Tests:** [Diffusion controls](../tests/physics/test_structural_diffusion.py).
- **Limits:** Directed, forced, changing-capacity and full-runtime systems require separate hypotheses; this result does not establish autonomous NFR formation.

<a id="pattern-recovery"></a>

### Conditional pattern formation and recovery

- **Status:** maintained
- **Basis:** derived
- **Emergence:** conditional
- **Definition:** A stated full-state geometric identity can recover locally or be reached from a different winding preparation under the relational law.
- **Domain:** Proved acute equilibrium neighborhoods or the separately certified supplied two-ring support and preparation.
- **Premises:** Positive held capacities, admitted coefficients, regularity and the particular recovery/capture hypotheses.
- **Dependencies:** relational-law, storage-work, winding, nfr
- **Owner:** [Local recovery](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery).
- **Evidence:** [Validated continuous transit](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-validated-transit).
- **Implementation:** [Capture](../src/tnfr/physics/relational_capture.py), [transit](../src/tnfr/physics/relational_transit.py).
- **Tests:** [Local recovery](../tests/physics/test_relational_local_recovery.py), [transit evidence](../tests/physics/test_relational_transit_proof.py).
- **Limits:** Recovery, finite lifetime and formation are different obligations. Support and preparation remain supplied; arbitrary formation and physical identity are unproved.

<a id="form-geometry"></a>

### Derived form geometry

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** Regional EPI contrasts determine Cartesian/Gram observations and, away from zero contrast, an orientation angle.
- **Domain:** Supplied ordered regions, declared form chart and any independent held source used as an orientation reference.
- **Premises:** Project and differentiate the admitted fine law; retain source-relative information when a source distinguishes orientations.
- **Dependencies:** epi, nodal-equation
- **Owner:** [Derived form phase](nodal/DERIVED_FORM_PHASE.md).
- **Evidence:** [Inherited observations](nodal/DERIVED_FORM_PHASE.md#regional-form-engine-integration), [source-relative response](nodal/DERIVED_FORM_PHASE.md#source-relative-future-response).
- **Implementation:** [Form geometry](../src/tnfr/physics/form_geometry.py), [source-relative form](../src/tnfr/physics/source_relative_form.py).
- **Tests:** [Form geometry](../tests/physics/test_form_geometry.py), [source response](../tests/physics/test_source_relative_form_response.py).
- **Limits:** The derived angle is unavailable at zero contrast and is not automatically the primitive phase consumed by pressure.

<a id="closure"></a>

### Coarse closure and hidden-state obstruction

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** A retained observation closes only when its rate is determined by the retained state under the specified law.
- **Domain:** Exact linear realizations and the separately analyzed nonlinear paired-C5 observation.
- **Premises:** Fix the fine law, observation and state domain before proving sufficiency or giving a counterexample.
- **Dependencies:** nodal-equation, relational-law
- **Owner:** [Pattern composition](nodal/RELATIONAL_PATTERN_COMPOSITION.md).
- **Evidence:** [State and current rates fail to close](nodal/RELATIONAL_PATTERN_COMPOSITION.md#state-rate-predictivity).
- **Implementation:** [Invariant-row realization](../src/tnfr/mathematics/linear_observation.py).
- **Tests:** [Linear observation](../tests/test_linear_observation.py), [nonlinear obstruction](../tests/physics/test_relational_local_composition.py).
- **Limits:** Ten-coordinate tangent closure does not establish nonlinear closure; failure of one observation is not impossibility of every reduction.

<a id="hidden-memory"></a>

### Derived hidden-state memory

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** Eliminating unobserved coordinates retains their propagated initialization and history-dependent influence in the visible equation.
- **Domain:** A complete fine law and fixed observation; diffusion and nonlinear relational results have different hypotheses.
- **Premises:** Retain hidden initial state, nonlinear generation and forcing; prove approximation bounds separately.
- **Dependencies:** closure
- **Owner:** [EPI memory](DERIVED_EPI_MEMORY.md), [relational memory](nodal/RELATIONAL_PATTERN_MEMORY.md).
- **Evidence:** [Exact elimination](DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state), [cubic forecast](nodal/RELATIONAL_PATTERN_MEMORY.md#derived-cubic-memory-forecast).
- **Implementation:** [Linear memory owner](../src/tnfr/physics/epi_memory.py).
- **Tests:** [Linear memory](../tests/physics/test_epi_memory.py), [relational memory](../tests/physics/test_relational_pattern_memory.py).
- **Limits:** Memory does not choose a missing microscopic law. The finite forecast is not a universal finite-history closure, fitted kernel or certified ODE error bound.

<a id="attachment"></a>

### Supplied bridge interfaces

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** Adding or relocating a supplied unit bridge changes incident degree, form-gradient sums and phase resultants. Fresh fields and signed edge costs describe the instantaneous response and storage jump.
- **Domain:** Acute simple unit graphs with unchanged nodal state: two disjoint connected components for attachment, or two nontrivial components separated by the removed bridge for relocation.
- **Premises:** Retain internal state and relative frames; supply the candidate and evaluate both supports through the same law. Event passivity is an additional premise; recovery requires the separate theorem's hypotheses.
- **Dependencies:** relational-law, phase-metric, storage-work
- **Owner:** [Composition and support-event interfaces](nodal/RELATIONAL_PATTERN_COMPOSITION.md).
- **Evidence:** [Attachment identity](nodal/RELATIONAL_PATTERN_COMPOSITION.md#one-bridge-interface-admission), [event-budget and nonselection result](nodal/RELATIONAL_PATTERN_COMPOSITION.md#support-event-premise-admission), [conditional relocation and recovery](nodal/RELATIONAL_PATTERN_COMPOSITION.md#identity-preserving-bridge-relocation).
- **Implementation:** [Shared attachment and relocation observers](../src/tnfr/physics/relational_observations.py).
- **Tests:** [Static attachment, relocation and budget controls](../tests/test_relational_attachment.py).
- **Limits:** Reports do not change live support, authenticate available work or automatically certify recovery. Represented budgets are not transcendental error enclosures. Admission selects neither occurrence nor timing.

<a id="winding"></a>

### Cycle winding

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** Integer cycle count from consistently wrapped oriented phase gaps divided by 2*pi.
- **Domain:** A supplied oriented cycle and declared branch/availability conditions.
- **Premises:** Constancy along a path requires exclusion of relevant branch crossings and accounting for support changes.
- **Dependencies:** phase, support
- **Owner:** [Cycle boundary](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-cycle-resultant-obstruction).
- **Evidence:** [Cycle invariant and scoped obstruction](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-cycle-resultant-obstruction).
- **Implementation:** [Winding certificates](../src/tnfr/physics/winding_certificates.py).
- **Tests:** [Winding admission](../tests/physics/test_phase_winding_certificates.py).
- **Limits:** Snapshot winding proves neither formation nor physical charge, spin or particle species.

## Diagnostics and their availability

<a id="coherence"></a>

### Coherence read-out (C)

- **Status:** maintained
- **Basis:** diagnostic
- **Emergence:** not-claimed
- **Definition:** Configured map C=1/(1+mean(abs(p))+mean(abs(xdot))) in declared pressure/rate scales.
- **Domain:** Admitted finite stored arguments; local and aggregate reductions have different scopes.
- **Premises:** The reciprocal observation is selected, not uniquely derived from the nodal equation.
- **Dependencies:** pressure, nodal-equation
- **Owner:** [Telemetry scope](NODAL_PARAMETER_FOUNDATIONS.md#6-telemetry-time-and-energy-are-not-interchangeable).
- **Evidence:** [Core metric definitions](FUNDAMENTAL_THEORY.md#5-core-structural-metrics).
- **Implementation:** [Shared kernel](../src/tnfr/metrics/common.py).
- **Tests:** [Coherence reduction](../tests/sdk/test_coherence_reduction_semantics.py).
- **Limits:** High C proves neither complete equilibrium nor stability, identity or formation. Stored rates need not be fresh nodal predictions.

<a id="equilibrium"></a>

### Pressure/rate equilibrium test

- **Status:** maintained
- **Basis:** diagnostic
- **Emergence:** not-claimed
- **Definition:** A tolerance predicate on pressure and form-rate magnitudes, exposed by is_structural_equilibrium.
- **Domain:** Declared finite arguments and nonnegative tolerances.
- **Premises:** Supplied quantities and tolerances determine the diagnostic verdict.
- **Dependencies:** pressure, nodal-equation
- **Owner:** [Diagnostic scope](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).
- **Evidence:** [Equilibrium boundary](NODAL_PARAMETER_FOUNDATIONS.md#6-telemetry-time-and-energy-are-not-interchangeable).
- **Implementation:** [Shared predicate](../src/tnfr/metrics/common.py).
- **Tests:** [Equilibrium scope](../tests/sdk/test_coherence_reduction_semantics.py).
- **Limits:** A tolerated zero form rate is not a full phase/capacity/support fixed point, particularly at zero capacity or with forcing.

<a id="sense-index"></a>

### Sense Index (Si)

- **Status:** maintained
- **Basis:** diagnostic
- **Emergence:** not-claimed
- **Definition:** Configured clipped mixture of normalized capacity, phase dispersion and pressure magnitude.
- **Domain:** Admitted inputs and current normalization policy.
- **Premises:** Selected weights and reductions define the score; controllers may explicitly consume it.
- **Dependencies:** capacity, phase, pressure
- **Owner:** [Si definition and consumers](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#sense-index-definition-information-loss-and-consumers).
- **Evidence:** [Diagnostic closure limits](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#7-derived-observables-and-dynamical-closure).
- **Implementation:** [Sense Index](../src/tnfr/metrics/sense_index.py).
- **Tests:** [Study diagnostics](../tests/sdk/test_study.py).
- **Limits:** Controller use does not promote Si to an emergent law, complete state or proved stability margin.

<a id="tetrad"></a>

### Structural-field tetrad

- **Status:** maintained
- **Basis:** diagnostic
- **Emergence:** not-claimed
- **Definition:** Interface (Phi_s, abs(grad phi), K_phi, xi_C) observing pressure aggregation, local phase structure and coherence range.
- **Domain:** Each constituent retains its independent data and availability conditions.
- **Premises:** Four complementary questions motivate the interface; completeness is a separate mathematical obligation.
- **Dependencies:** potential, phase-gradient, phase-curvature, coherence-length
- **Owner:** [Diagnostic degrees and reconstruction](MINIMAL_STRUCTURAL_DEGREES.md).
- **Evidence:** [Dynamical non-reconstruction](MINIMAL_STRUCTURAL_DEGREES.md#33-dynamical-non-reconstruction).
- **Implementation:** [Shared fields](../src/tnfr/physics/fields.py).
- **Tests:** [Read-out consistency](../tests/physics/test_field_readout_consistency.py).
- **Limits:** The tetrad does not reconstruct general form, capacity, support, history or future evolution; unavailable fields cannot certify a complete safety verdict.

<a id="potential"></a>

### Structural potential (Phi_s)

- **Status:** maintained
- **Basis:** diagnostic
- **Emergence:** not-claimed
- **Definition:** Nonlocal aggregation Phi_s(i)=sum_j p_j/d(i,j)^2 over other nodes under the declared distance convention.
- **Domain:** Admitted pressure and lengths, with specified unreachable/zero-distance handling.
- **Premises:** The inverse-square aggregation is a selected observation kernel.
- **Dependencies:** pressure, support
- **Owner:** [Structural fields](../docs/STRUCTURAL_FIELDS_TETRAD.md).
- **Evidence:** [Topology-dependent bounds](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#4-structural-potential-and-topology-dependent-bounds).
- **Implementation:** [Potential field](../src/tnfr/physics/fields.py).
- **Tests:** [Field observations](../tests/physics/test_field_readout_consistency.py).
- **Limits:** Potential is not universally bounded by pi, the relational storage or an independently derived physical interaction potential.

<a id="phase-gradient"></a>

### Phase gradient magnitude

- **Status:** maintained
- **Basis:** diagnostic
- **Emergence:** not-claimed
- **Definition:** Mean absolute wrapped separation from the node's declared neighbors.
- **Domain:** Finite phases and supported neighbor conventions; circular bound pi.
- **Premises:** Mean separation is the selected local diagnostic.
- **Dependencies:** phase, support
- **Owner:** [Structural fields](../docs/STRUCTURAL_FIELDS_TETRAD.md).
- **Evidence:** [Gradient scope](MINIMAL_STRUCTURAL_DEGREES.md#42-phase-gradient).
- **Implementation:** [Phase gradient](../src/tnfr/physics/fields.py).
- **Tests:** [Field observations](../tests/physics/test_field_readout_consistency.py).
- **Limits:** A warning cut is not a derived instability threshold or a future-transition proof.

<a id="phase-curvature"></a>

### Circular phase curvature (K_phi)

- **Status:** maintained
- **Basis:** diagnostic
- **Emergence:** not-claimed
- **Definition:** Wrapped displacement from the neighbor-resultant direction.
- **Domain:** Defined nonempty-neighborhood direction, with an explicit isolate convention; magnitude bounded by pi.
- **Premises:** Exactly joint-zero represented resultants are unavailable, not assigned an invented direction.
- **Dependencies:** phase, support
- **Owner:** [Curvature contract](../docs/STRUCTURAL_FIELDS_TETRAD.md).
- **Evidence:** [Circular curvature scope](MINIMAL_STRUCTURAL_DEGREES.md#43-circular-curvature).
- **Implementation:** [Curvature observation](../src/tnfr/physics/fields.py).
- **Tests:** [Resultant availability](../tests/physics/test_phase_curvature_resultant.py).
- **Limits:** Curvature alone proves neither confinement nor Mutation admission; Laplacian comparisons require additional small-spread hypotheses.

<a id="coherence-length"></a>

### Coherence range (xi_C)

- **Status:** maintained
- **Basis:** diagnostic
- **Emergence:** not-claimed
- **Definition:** Distance-binned static coherence-product fit, with a separately tagged spectral fallback.
- **Domain:** Declared sampling, fit acceptance, graph and distance units.
- **Premises:** Pressure-derived products and fallback rules specify the observation.
- **Dependencies:** pressure, support
- **Owner:** [Shared fit definition](NODAL_PARAMETER_FOUNDATIONS.md#52-one-coherence-fit-definition-across-implementations).
- **Evidence:** [Correlation scope](MINIMAL_STRUCTURAL_DEGREES.md#44-correlation-length).
- **Implementation:** [Estimator](../src/tnfr/physics/fields.py).
- **Tests:** [Estimator provenance](../tests/sdk/test_nfr_observation_scope.py).
- **Limits:** An uncentered static fit is neither a temporal relaxation measurement nor proof of criticality; its fallback is a different observable.

<a id="balance-diagnostic"></a>

### Structural currents and balance diagnostic

- **Status:** maintained
- **Basis:** diagnostic
- **Emergence:** not-claimed
- **Definition:** Declared phase/pressure currents and density rho=Phi_s+K_phi supply a trajectory balance with measured residual.
- **Domain:** Matching captured support, positive finite intervals and the stated divergence convention.
- **Premises:** Temporal changes and both sector divergences come from actual captured evidence.
- **Dependencies:** potential, phase-curvature, clock
- **Owner:** [Conservation diagnostics](STRUCTURAL_CONSERVATION_THEOREM.md).
- **Evidence:** [Measured balance](STRUCTURAL_CONSERVATION_THEOREM.md#45-measured-conservation-residual-and-conditional-bounds).
- **Implementation:** [Conservation observations](../src/tnfr/physics/conservation.py).
- **Tests:** [Observation admission](../tests/physics/test_conservation_observation_admission.py).
- **Limits:** Single snapshots cannot establish temporal conservation. The field sum-of-squares energy is a Lyapunov candidate, distinct from relational storage.

## Declared events, policies and open bridges

<a id="operators"></a>

### Named structural operators

- **Status:** maintained
- **Basis:** constitutive
- **Emergence:** not-claimed
- **Definition:** Thirteen registered transformations define declared jump contracts on nodal/graph state.
- **Domain:** The actual glyph, public operator, atomic stage or word-execution path.
- **Premises:** Identify primary and secondary writes and preserve their execution provenance.
- **Dependencies:** nodal-equation, phase, support
- **Owner:** [Structural operators](STRUCTURAL_OPERATORS.md).
- **Evidence:** [Execution contracts](../docs/API_CONTRACTS.md#contract-model).
- **Implementation:** [Operator registry](../src/tnfr/operators/operator_contracts.py).
- **Tests:** [Operator contracts](../tests/operators/test_operator_contracts.py).
- **Limits:** A named jump is not a finite-duration nodal solution. Defining a transformation does not derive when it occurs or prove catalog completeness.

<a id="grammar"></a>

### Grammar and automatic selection

- **Status:** maintained
- **Basis:** policy
- **Emergence:** not-claimed
- **Definition:** Configured U1-U6 word/live-state admission and separate controller choices govern supported execution.
- **Domain:** Declared history, node state, grammar profile and execution path.
- **Premises:** Role membership, thresholds and selector inputs remain explicit configuration.
- **Dependencies:** operators
- **Owner:** [Unified grammar](UNIFIED_GRAMMAR_RULES.md).
- **Evidence:** [Premises and trajectories](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#8-grammar-derivation-premises-language-and-trajectories).
- **Implementation:** [Grammar bases](../src/tnfr/operators/grammar_canon.py), [selectors](../src/tnfr/dynamics/selectors.py).
- **Tests:** [Grammar bases](../tests/operators/test_grammar_canon.py).
- **Limits:** Grammar admission is neither autonomous selection nor a stability theorem. Mutation needs observed signed temporal evidence, distinct from predicted nu_f*p.

<a id="thresholds"></a>

### Thresholds, gains and numerical tolerances

- **Status:** maintained
- **Basis:** policy
- **Emergence:** not-claimed
- **Definition:** Settings used by named consumers to admit actions, label telemetry or control computation.
- **Domain:** Each value retains its units, normalization, consumer and overrides.
- **Premises:** Distinguish exact geometric bounds from policy cuts and arithmetic tolerances.
- **Dependencies:** clock, grammar
- **Owner:** [Threshold ledger](NODAL_PARAMETER_FOUNDATIONS.md#11-thresholds-constants-and-numerical-settings-have-different-duties).
- **Evidence:** [Assumption-explicit thresholds](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#3-assumption-explicit-model-thresholds).
- **Implementation:** [Defaults](../src/tnfr/config/defaults_core.py), [compatibility constants](../src/tnfr/constants/canonical.py).
- **Tests:** [Threshold resolution](../tests/operators/test_thol_threshold_configuration.py).
- **Limits:** A CANONICAL name or pi-based formula does not prove universality. The exact radian wrap bound is a different claim.

<a id="pulse"></a>

### Spectral pulse and modal rhythm

- **Status:** maintained
- **Basis:** auxiliary
- **Emergence:** not-claimed
- **Definition:** A specified graph-wave comparison assigns frequencies sqrt(lambda_k); nodal pulse reports separately read stored capacity and phase.
- **Domain:** The declared graph operator, spectrum and supplied wave model.
- **Premises:** The wave law and units are additional choices, distinct from first-order diffusion.
- **Dependencies:** support, capacity, phase
- **Owner:** [Wave correspondence](PHYSICAL_REGIME_CORRESPONDENCES.md#4-finite-spectral-and-wave-correspondence).
- **Evidence:** [Finite mode calculation](PHYSICAL_REGIME_CORRESPONDENCES.md#43-retained-graph-mode-calculation).
- **Implementation:** [Pulse read-outs](../src/tnfr/physics/structural_diffusion.py).
- **Tests:** [Pulse scope](../tests/physics/test_nodal_pulse_scope.py).
- **Limits:** Modal rhythm and synchronization differ; neither derives a primordial pulse, universal clock or maintained engine wave.

<a id="symplectic"></a>

### Auxiliary symplectic substrate

- **Status:** maintained
- **Basis:** auxiliary
- **Emergence:** not-claimed
- **Definition:** An ambient model assigns canonical pairs to selected graph read-outs and a declared quadratic Hamiltonian.
- **Domain:** That auxiliary phase space, not arbitrary complete nodal execution.
- **Premises:** Canonical brackets, conjugate partners and Hamiltonian are supplied construction choices.
- **Dependencies:** tetrad, balance-diagnostic
- **Owner:** [Auxiliary substrate](FUNDAMENTAL_THEORY.md#72-auxiliary-symplectic-substrate).
- **Evidence:** [Nodal correspondence boundary](TNFR_VARIATIONAL_PRINCIPLE.md#3-harmonic-model-and-the-unresolved-nodal-correspondence).
- **Implementation:** [Symplectic substrate](../src/tnfr/physics/symplectic_substrate.py).
- **Tests:** [Auxiliary model](../tests/physics/test_symplectic_substrate.py), [graph realizability](../tests/physics/test_symplectic_graph_realizability.py).
- **Limits:** Its harmonic flow and classical U(2) constructions do not establish quantum states, particle emergence or a Hamiltonian full-engine law.

<a id="arithmetic"></a>

### Arithmetic and Riemann constructions

- **Status:** maintained
- **Basis:** auxiliary
- **Emergence:** not-claimed
- **Definition:** Supplied arithmetic encodings and finite spectral constructions support scoped identities, diagnostics and comparisons.
- **Domain:** The particular inputs, graph family, cutoff and numerical evaluator.
- **Premises:** Prime/divisor data, logarithmic frequencies and known-zero targets remain identified when supplied.
- **Dependencies:** support
- **Owner:** [Number theory](TNFR_NUMBER_THEORY.md), [Riemann scope](TNFR_RIEMANN_RESEARCH_NOTES.md).
- **Evidence:** [Construction inventory](TNFR_RIEMANN_RESEARCH_NOTES.md#2-reusable-construction-and-evidence-inventory).
- **Implementation:** [Arithmetic model](../src/tnfr/mathematics/number_theory.py), [finite pulse](../src/tnfr/riemann/nodal_pulse.py).
- **Tests:** [Arithmetic contracts](../tests/mathematics/test_number_theory_canonical.py), [finite pulse](../tests/mathematics/test_riemann_nodal_pulse.py).
- **Limits:** Shared telemetry does not identify arithmetic and physical dynamics. Finite scans or assigned spectra establish no RH proof or unproved asymptotic limit.

<a id="measurement-bridge"></a>

### Physical measurement bridge

- **Status:** candidate
- **Basis:** constitutive
- **Emergence:** not-claimed
- **Definition:** Independently justified map from model state or collective observations to measurements, preparation and laboratory time.
- **Domain:** A named terrestrial system with declared data/instruments, uncertainty, calibration and reserved evaluation.
- **Premises:** Specify observables and pressure inputs before observing the response; latent primitive coordinates need not each have a direct sensor.
- **Dependencies:** epi, capacity, phase, clock, pressure
- **Owner:** [Measurement admission](research/PASSIVE_TRANSPORT_PROTOCOL.md#state-clock-and-measurement-admission).
- **Evidence:** [Physical atlas status](PHYSICAL_REGIME_CORRESPONDENCES.md#verification-status).
- **Implementation:** none: protocol-specific evaluators do not establish an admitted general physical bridge.
- **Tests:** none: no general physical identification has passed independent measurement admission.
- **Limits:** Retrospective pressure reconstruction is circular. Compatibility with known physics need not distinguish a unique explanatory mechanism.

<a id="physical-emergence"></a>

### Physical constituents from TNFR

- **Status:** candidate
- **Basis:** hypothesis
- **Emergence:** not-claimed
- **Definition:** Hypothesis that justified nodal structure and dynamics generate patterns explaining physical constituents and collective properties.
- **Domain:** A future identified model with independent observable predictions and terrestrial evidence.
- **Premises:** Admit complete state, constitutive laws, support/capacity evolution and the physical measurement bridge.
- **Dependencies:** nfr, nodal-equation, measurement-bridge
- **Owner:** [Research strategy](NODAL_RESEARCH_STRATEGY.md).
- **Evidence:** [Foundation origin ledger](FUNDAMENTAL_THEORY.md#foundational-state-admission).
- **Implementation:** none: the engine implements scoped models, not an established physical ontology.
- **Tests:** none: model tests cannot replace independent physical validation.
- **Limits:** Conditional formation, winding, symmetry and memory results do not themselves derive matter, spin, quantum mechanics or substrate creation from nothing.

<!-- END CONCEPT CARDS -->

## Card template and admission checks

Revise a card and its responsible owner together, retaining a stable lowercase
ID. Use the exact field order below and replace instructional values with scoped
statements or references. Dependencies contains existing comma-separated card IDs
or `none`; it is a declared prerequisite graph, not a mathematical proof.

```markdown
<a id="new-concept"></a>

### Concept title

- **Status:** maintained or candidate
- **Basis:** primitive, identity, constitutive, derived, diagnostic, policy, auxiliary or hypothesis
- **Emergence:** not-claimed or conditional
- **Definition:** State what the concept means.
- **Domain:** State where it is defined and executable.
- **Premises:** Name supplied assumptions and choices.
- **Dependencies:** existing-id, another-id
- **Owner:** Link the maintained theory/docs definition owner.
- **Evidence:** Link the evidence; a derived concept needs a theory derivation anchor.
- **Implementation:** Link its src owner, or write none: followed by a reason.
- **Tests:** Link relevant test files, or write none: followed by a reason.
- **Limits:** State what does not follow from the declaration.
```

The [glossary checker](../scripts/check_glossary.py) validates card structure,
labels, local references, declared dependencies and index freshness. A maintained
implemented concept needs linked tests; derived claims need a derivation section
and declared dependencies; hypotheses retain candidate status. These checks
cannot verify a proof, test adequacy or physical truth, and never promote a
concept automatically.

Reproducibility also requires fixed source, configuration, inputs, ordering,
precision and execution path; a seed alone is insufficient. Use
[TESTING.md](../TESTING.md) for validation and the
[sole execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
for the next research task.
