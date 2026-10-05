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
| [Conditional smooth-sine exchange law](#sine-law) | constitutive | not-claimed | maintained |
| [Relative resultant, phase metric and mobility](#phase-metric) | derived | not-claimed | maintained |
| [Joint storage and signed exchange work](#storage-work) | derived | not-claimed | maintained |
| [Pure-EPI diffusion and relaxation](#diffusion) | derived | conditional | maintained |
| [Conditional pattern formation and recovery](#pattern-recovery) | derived | conditional | maintained |
| [Conditional structural resonance](#resonance) | derived | conditional | maintained |
| [Conditional autonomous nonlinear pulse](#autonomous-pulse) | derived | conditional | maintained |
| [Conditional parametric pulse instability](#parametric-instability) | derived | conditional | maintained |
| [Conditional nonlinear recurrence](#nonlinear-recurrence) | derived | conditional | maintained |
| [Derived form geometry](#form-geometry) | derived | not-claimed | maintained |
| [Conditional dynamical scale inheritance](#scale-inheritance) | derived | conditional | maintained |
| [Coarse closure and hidden-state obstruction](#closure) | derived | not-claimed | maintained |
| [Derived hidden-state memory](#hidden-memory) | derived | not-claimed | maintained |
| [Retained relational phase offset](#retained-phase-offset) | derived | conditional | maintained |
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
- **Owner:** [Relation foundations](nodal/RELATION_FOUNDATIONS.md).
- **Evidence:** [Zero-relation boundary](nodal/RELATION_FOUNDATIONS.md#zero-relation-boundary); [pressure contract](NODAL_PARAMETER_FOUNDATIONS.md#21-the-implemented-pressure-is-a-specified-relational-map).
- **Implementation:** [Support transport](../src/tnfr/physics/support_transport.py).
- **Tests:** [Regional cut](../tests/test_regional_support_cut.py).
- **Limits:** A zero-conductance edge can remain a phase neighbor. Normalized transport weight is not an absolute interaction rate; distance is separate. No autonomous relation-birth law is derived.

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
- **Evidence:** [Identity obligations](NODAL_RESEARCH_STRATEGY.md#closure-audit-equations-events-and-identity), [retained constituent influence](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-inheritance), [exact unordered-pair state](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-unordered-state).
- **Implementation:** [SDK NFR observation](../src/tnfr/sdk/simple.py), [detached fine-to-collective assessment](../src/tnfr/physics/relational_sine_scale.py).
- **Tests:** [NFR observation scope](../tests/sdk/test_nfr_observation_scope.py), [retained fine-state and symmetry controls](../tests/physics/test_relational_sine_replica.py).
- **Limits:** A graph node, nested child or topology label does not establish autopoiesis, persistent identity or a physical constituent. A collective description retains its constituents and their dynamics; the exact pair quotient removes interchangeable labels, not internal continuous coordinates. Nesting alone does not derive a closed scale law.

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
- **Premises:** Native pressure and joint storage; either own-capacity independence with zero-row freezing, or uniform primitive-neighborhood locality across admitted supports with the zero common-phase-clock convention.
- **Dependencies:** nodal-equation, phase-metric
- **Owner:** [Capacity-separable law](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-separable-exchange).
- **Evidence:** [Relative-phase uniqueness and common-clock classification](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#primitive-locality-phase-clock); [joint-cost classification under prescribed loss](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#joint-storage-locality-classification); [alternative when locality is relaxed](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-mediated-exchange-counterexample).
- **Implementation:** [Relational engine](../src/tnfr/dynamics/relational.py).
- **Tests:** [Exchange execution](../tests/test_relational_exchange_execution.py).
- **Limits:** This opt-in law differs from default operator runtime; the nodal identity does not uniquely select its premises or beta. [Passive alternatives](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-passive-loss-completion) and a [nonlinear countermodel](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-nonlinear-passive-completion) share equilibrium/recovery properties while changing response. Signed degree-one form response is a sufficient additional premise, not unit covariance or a consequence of matching the equilibrium tangent.

<a id="sine-law"></a>

### Conditional smooth-sine exchange law

- **Status:** maintained
- **Basis:** constitutive
- **Emergence:** not-claimed
- **Definition:** With K=diag(nu_i/d_i), unit graph Laplacian L and S_i=sum_neighbors sin(theta_j-theta_i), the baseline rows are xdot=-e*K*L*x+(w/pi)*K*S and thetadot=(w/(beta*pi))*K*L*x.
- **Domain:** Fixed finite connected simple reciprocal unit support with at least two nodes, held nonnegative capacities, e>=0, w,beta>0 and finite signed form/circular phase.
- **Premises:** Select the complete normalized-sine law, cosine storage, fixed clock and no forcing/events. Conditional selection retains the owner's conservation/composition, information, storage and capacity premises; the nodal identity alone does not select them.
- **Dependencies:** nodal-equation, phase, capacity, support
- **Owner:** [Complete smooth-law comparison](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#global-closure-pressure-comparison).
- **Evidence:** [Balance and global continuation](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#complete-balance-continuation-and-inherited-geometry), [conditional restrictions and nonselection](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#global-closure-pressure-comparison), [form-balance selection of pressure with separate phase-row premises](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#closed-form-balance-and-source-selection).
- **Implementation:** [Detached comparison and shared rates](../src/tnfr/physics/relational_sine_comparison.py), [primitive admission](../src/tnfr/physics/_sine_admission.py).
- **Tests:** [Independent complete-law controls](../tests/physics/test_relational_sine_comparison.py), [source and theorem-domain admission](../tests/physics/test_sine_admission.py), [balance, composition and branching controls](../tests/physics/test_relational_pressure_composition.py).
- **Limits:** This changes both native form/phase rows, not just the native phase-domain setting. Smoothness and shared storage do not uniquely select the law; alternative mobilities remain separate models. [Changing the phase-storage premise](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#phase-storage-selection-boundary) also permits different nonlinear laws with the same form balance and consensus tangent. Individual proofs may require strictly positive loss/capacity. Detached readers and scoped forecasts do not replace Network.step_relational.

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
- **Evidence:** [Phase row and domain](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#2-forced-phase-row-and-its-domain); [aligned edge-cost classification and metric continuation](TNFR_VARIATIONAL_PRINCIPLE.md#native-phase-storage-classification); [regular point/chord admission](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-full-regular-execution).
- **Implementation:** [Captured relational field](../src/tnfr/dynamics/relational.py).
- **Tests:** [Regular execution](../tests/test_relational_regular_execution.py).
- **Limits:** Resultant/branch boundaries are not successful zero values. Certified regular-domain bounds are separate from materialized sums and work defects; they do not enclose rates or the continuous trajectory. Numerically unresolved regular states can be rejected.

<a id="storage-work"></a>

### Joint storage and signed exchange work

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** For S = E_D + beta*V, with E_D = sum_edges (x_i-x_j)^2/2 and V = sum_edges [1-cos(theta_j-theta_i)], the ideal balances are Sdot=-sum D_i and E_D_dot=-sum D_i+sum J_i. Count unordered edges once.
- **Domain:** The selected native or baseline smooth-sine law on held unit support. With q=L*x, both have D_i=e*nu_i*q_i^2/d_i; native exchange is J_i=w*nu_i*q_i*g_i, while sine exchange is J_i=(w/pi)*(nu_i/d_i)*q_i*S_i.
- **Premises:** Use one admitted complete law without forcing or support events. Native joint-cost classification additionally assumes its own local-law and prescribed-loss class; sine storage is retained as a separate model premise.
- **Dependencies:** relational-law, sine-law
- **Owner:** [Native signed work](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-work-integration), [sine balance](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#complete-balance-continuation-and-inherited-geometry).
- **Evidence:** [Joint storage balances](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#3-complete-source-storage-and-mean-balances); [joint-cost classification](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#joint-storage-locality-classification).
- **Implementation:** [Native field work](../src/tnfr/dynamics/relational.py), [sine rate/work kernel](../src/tnfr/physics/relational_sine_comparison.py).
- **Tests:** [Native work](../tests/test_relational_work_observation.py), [sine complete-law controls](../tests/physics/test_relational_sine_comparison.py).
- **Limits:** This is not measured physical energy or tetrad energy. Shared total storage/loss does not imply identical exchange, trajectories or conserved means; alternative reciprocal mobilities retain their own loss/exchange rows. Numerical residuals and support-event costs remain explicit.

<a id="diffusion"></a>

### Pure-EPI diffusion and relaxation

- **Status:** maintained
- **Basis:** derived
- **Emergence:** conditional
- **Definition:** Isolated form pressure is -L_rw*x; fixed positive capacities and reciprocal nonnegative conductance give consensus when the positive-conductance graph is connected.
- **Domain:** Source-free form channel, fixed conductance/support and held positive capacities; disconnected components and isolates have separate conclusions.
- **Premises:** Select only the isolated EPI channel without forcing or events. With d_i=sum_j W_ij denoting conductance strength, the resulting generator conserves the d_i/nu_i-weighted form total; conservation is a consequence, not an extra assumption.
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
- **Definition:** A declared geometric identity can recover locally, or be reached from a preparation initially lacking that identity, under a specified complete law.
- **Domain:** Native acute recovery and certified two-ring transit; separately, the baseline smooth-sine law with its whole-sector capture and prepared-entry hypotheses.
- **Premises:** Retain each theorem's supplied support, complete law, positive capacities/loss, phase domain, initial form and uncertainty. Recovery and formation have different initial sets.
- **Dependencies:** relational-law, sine-law, storage-work, winding, nfr
- **Owner:** [Native recovery](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery), [sine pattern dynamics](nodal/SINE_PATTERN_DYNAMICS.md).
- **Evidence:** [Native validated transit](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-validated-transit), [sine sector capture](nodal/SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture), [prepared acquisition and uncertainty](nodal/SINE_PATTERN_DYNAMICS.md#sine-prepared-sector-entry).
- **Implementation:** [Native capture](../src/tnfr/physics/relational_capture.py), [native transit](../src/tnfr/physics/relational_transit.py), [sine recovery](../src/tnfr/physics/relational_sine_recovery.py), [sine entry](../src/tnfr/physics/relational_sine_entry.py).
- **Tests:** [Native transit](../tests/physics/test_relational_transit_proof.py), [whole-sector capture](../tests/physics/test_relational_sine_sector_capture.py), [prepared entry](../tests/physics/test_relational_sine_entry.py).
- **Limits:** Recovery, finite lifetime and formation remain different obligations. Adequate storage does not guarantee reachability or select preparation; [budget and symmetry controls](nodal/SINE_PATTERN_DYNAMICS.md#sine-budget-consensus) retain their own domains. A failed sufficient certificate is unavailable, not impossibility. Support selection and physical identity remain open.

<a id="resonance"></a>

### Conditional structural resonance

- **Status:** maintained
- **Basis:** derived
- **Emergence:** conditional
- **Definition:** A positive finite-frequency maximum of a declared stable input/output gain, distinguished from reciprocal exchange, complex poles and maintained motion.
- **Domain:** Tangent dynamics at an exact critical pattern of the complete normalized-sine law, with connected unit support, positive held capacity, positive loss/coupling and a positive phase Hessian modulo common origin.
- **Premises:** Specify the form-rate probe and observation; retain all support and environmental coordinates. The work-conjugate theorem, cycle template and two-C5/intermediary remote phase theorem have distinct observation hypotheses.
- **Dependencies:** sine-law, phase, capacity, support, clock
- **Owner:** [Resonance foundations](nodal/RESONANCE_FOUNDATIONS.md).
- **Evidence:** [Full-network gain theorem](nodal/RESONANCE_FOUNDATIONS.md#collocated-resonance), [observation-dependent modal controls](nodal/RESONANCE_FOUNDATIONS.md#modal-resonance), [mediated response, memory and autonomous tangent reversal](nodal/RESONANCE_FOUNDATIONS.md#mediated-resonance).
- **Implementation:** [Sine resonance assessments](../src/tnfr/physics/relational_sine_resonance.py).
- **Tests:** [Independent tangent and response controls](../tests/physics/test_relational_sine_resonance.py).
- **Limits:** The theorem does not supply the probe, create support, invoke RA, guarantee every remote response or establish nonlinear formation or physical resonance. A response gain peak proves neither an autonomous periodic orbit nor its parametric instability. The native law and auxiliary graph-wave rhythm have separate generators.

<a id="autonomous-pulse"></a>

### Conditional autonomous nonlinear pulse

- **Status:** maintained
- **Basis:** derived
- **Emergence:** conditional
- **Definition:** Nonstationary periodic full-state form/phase exchange under a declared autonomous law, distinguished from a spectral diagnostic, transient rebound, approximate recurrence or prescribed periodic input.
- **Domain:** The complete normalized-sine law at e=0: isolated unit P2 with held nonnegative capacities of positive sum and 0<E<2*beta; separately, the exact doubled-C5 winding-one family with common positive capacity, identical signed internal coordinates and 0<H<beta*cos(2*pi/5).
- **Premises:** Supplied support, nonzero preparation, positive beta and coupling, structural clock and exactly zero loss. The doubled-C5 result requires its exact symbolic invariant family; no rounded graph membership is inferred. These premises do not follow from the demand for permanence.
- **Dependencies:** sine-law, phase, capacity, support, clock
- **Owner:** [Permanent-pulse admission](nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission).
- **Evidence:** [P2 libration and dissipative obstruction](nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission), [exact internal pulse and observation-dependent period](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-internal-pulse).
- **Implementation:** [Detached pair-pulse assessment](../src/tnfr/physics/relational_sine_resonance.py), [prepared replica-pulse assessment](../src/tnfr/physics/relational_sine_scale.py).
- **Tests:** [P2 field, period and boundaries](../tests/physics/test_relational_sine_resonance.py), [complete replica field and elliptic-solution controls](../tests/physics/test_relational_sine_replica.py).
- **Limits:** The period belongs to the exact continuous law, not an Euler orbit. In the replica family the labeled period is twice the unordered-pair period while collective means remain constant; every constituent remains present. Existence does not prove stability: sufficiently small nonzero replica pulses are transversely unstable, without a certified numerical amplitude threshold. Conservation does not select an attracting amplitude, justify zero loss or create activity from equilibrium.

<a id="parametric-instability"></a>

### Conditional parametric pulse instability

- **Status:** maintained
- **Basis:** derived
- **Emergence:** conditional
- **Definition:** Exponential transverse growth generated by periodic coefficients of a full autonomous variational law, established by an unstable return multiplier.
- **Domain:** Sufficiently small positive internal amplitude in the exact conservative doubled-C5 pulse family, with the full twenty real state directions retained.
- **Premises:** The same support, law, common positive capacity and exact preparation as the pulse theorem; collective feedback and the amplitude-dependent period remain in the transverse calculation.
- **Dependencies:** autonomous-pulse, storage-work
- **Owner:** [Complete variation and symmetry-correct return](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-pulse-variation).
- **Evidence:** [Exact small-amplitude splitting and analytic remainder](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-pulse-splitting).
- **Implementation:** [Detached variation and asymptotic splitting assessments](../src/tnfr/physics/relational_sine_scale.py).
- **Tests:** [Full fine Jacobian, harmonic solvability and return-scope controls](../tests/physics/test_relational_sine_replica.py).
- **Limits:** This is an existential small-amplitude instability interval, not a computed monodromy, numerical threshold or verdict for a chosen nonzero amplitude. Neutral time-amplitude shear is distinct from transverse exponential growth. Leaving one periodic waveform need not destroy a trapped geometric identity; no fundamental pulse, external drive or particle identification follows.

<a id="nonlinear-recurrence"></a>

### Conditional nonlinear recurrence

- **Status:** maintained
- **Basis:** derived
- **Emergence:** conditional
- **Definition:** The complete form/circular-phase state returns arbitrarily close to its initial state along unbounded positive times; an exact period is not required.
- **Domain:** The complete normalized-sine law on finite connected unit support with e=0 and strictly positive held capacities, restricted to a positive energy sublevel and a nonzero-width interval of conserved weighted form mean.
- **Premises:** Fixed support, positive beta and coupling, supplied finite state, declared clock and no input, clipping or events; circular phases and ambient product volume are essential.
- **Dependencies:** sine-law, phase, capacity, support, clock
- **Owner:** [Full nonlinear recurrence](nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence).
- **Evidence:** [Compactness, volume preservation, recurrence and separatrix exception](nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence); [additional acute-cycle identity barrier](nodal/RELATIONAL_PATTERN_MEMORY.md#sine-conservative-identity).
- **Implementation:** [Family-level recurrence assessment](../src/tnfr/physics/relational_sine_resonance.py), [shared cycle identity/barrier assessment](../src/tnfr/physics/relational_sine_recovery.py).
- **Tests:** [Invariant, admission and exception controls](../tests/physics/test_relational_sine_resonance.py), [cycle geometry and recovery controls](../tests/physics/test_relational_sine_recovery.py).
- **Limits:** The nonstationary result holds almost everywhere in the stated ambient measure. It does not certify a chosen state, finite grid or fixed-energy surface, give a return time, establish one frequency or derive the loss-free law. Winding preservation requires the separate acute barrier; it does not follow from recurrence alone.

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

<a id="scale-inheritance"></a>

### Conditional dynamical scale inheritance

- **Status:** maintained
- **Basis:** derived
- **Emergence:** conditional
- **Definition:** A specified map from fine nodal states to collective states carries the admitted fine evolution to the same declared law family, with explicit state restrictions and unit changes.
- **Domain:** The supplied synchronized replica submanifold of the native or separately declared sine law; retained internal coordinates are required away from synchronization.
- **Premises:** Fixed replicated support, equal capacity within each fiber, compatible state preparation and a complete law; iterated maps must compose.
- **Dependencies:** relational-law, support, capacity, phase, clock
- **Owner:** [Scale inheritance](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-inheritance).
- **Evidence:** [Native exact replication](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#4-origin-units-and-exact-replication), [replica composition and constitutive freedom](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-constitutive-nonselection).
- **Implementation:** [Native field](../src/tnfr/dynamics/relational.py), [sine comparison and mobility](../src/tnfr/physics/relational_sine_comparison.py), [retained replica state](../src/tnfr/physics/relational_sine_scale.py).
- **Tests:** [Complete replica rows](../tests/physics/test_relational_sine_replica.py), [counterfamily inheritance](../tests/physics/test_relational_sine_comparison.py).
- **Limits:** This conditional result does not establish generic fractality, spontaneous hierarchy formation, a universal fractal dimension or unique microscopic dynamics. Losing internal coordinates requires a closure or memory proof; a repeated graph is insufficient.

<a id="closure"></a>

### Coarse closure and hidden-state obstruction

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** A retained observation closes only when its rate is determined by the retained state under the specified law.
- **Domain:** Exact linear realizations and separately admitted nonlinear pair observations; the complete bipartite replica support and matched within-pair capacities define the sine reduction.
- **Premises:** Fix the fine law, observation and state domain before proving sufficiency or giving a counterexample.
- **Dependencies:** nodal-equation, relational-law
- **Owner:** [Pattern composition](nodal/RELATIONAL_PATTERN_COMPOSITION.md), [exact replica scale and symmetry](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-inheritance).
- **Evidence:** [State and current rates fail to close](nodal/RELATIONAL_PATTERN_COMPOSITION.md#state-rate-predictivity), [closed unordered state with retained internals](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-unordered-state).
- **Implementation:** [Invariant-row realization](../src/tnfr/mathematics/linear_observation.py), [retained replica observation](../src/tnfr/physics/relational_sine_scale.py).
- **Tests:** [Linear observation](../tests/test_linear_observation.py), [nonlinear obstruction](../tests/physics/test_relational_local_composition.py), [exact symmetry and closure](../tests/physics/test_relational_sine_replica.py).
- **Limits:** Ten-coordinate tangent closure does not establish nonlinear closure. Replica means alone fail to close outside their exact synchronized sector; the complete unordered state closes without deleting internal degrees of freedom. A zero snapshot defect is not proof of future closure; failure of one observation is not impossibility of every reduction.

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
- **Evidence:** [Exact elimination](DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state), [cubic forecast](nodal/RELATIONAL_PATTERN_MEMORY.md#derived-cubic-memory-forecast), [conservative transfer and tangent memory](nodal/RESONANCE_FOUNDATIONS.md#finite-conservative-memory).
- **Implementation:** [EPI memory](../src/tnfr/physics/epi_memory.py), [exact coordinate elimination](../src/tnfr/mathematics/linear_observation.py), [conservative path adapter](../src/tnfr/physics/relational_sine_resonance.py).
- **Tests:** [Linear memory](../tests/physics/test_epi_memory.py), [relational memory](../tests/physics/test_relational_pattern_memory.py), [conservative path controls](../tests/physics/test_relational_sine_resonance.py).
- **Limits:** Memory does not choose a missing microscopic law. The finite forecast is not a universal finite-history closure, fitted kernel or certified ODE error bound.

<a id="retained-phase-offset"></a>

### Retained relational phase offset

- **Status:** maintained
- **Basis:** derived
- **Emergence:** conditional
- **Definition:** Internal recovery can leave a preparation-dependent limiting common phase relative to a retained reference, despite restoring the same internal twist.
- **Domain:** The ideal native isolated unit C5 law near an acute winding-one twist, with centered perturbations and a common phase frame.
- **Premises:** Positive homogeneous held capacity, positive coefficients, no input/event during recovery, and a separately declared reference/readout.
- **Dependencies:** phase, relational-law, pattern-recovery
- **Owner:** [Retained collective phase](nodal/RELATIONAL_PATTERN_MEMORY.md#relational-retained-phase-memory).
- **Evidence:** [Integrated skew pairing](nodal/RELATIONAL_PATTERN_MEMORY.md#the-first-nonzero-retained-offset), [fixed finite-amplitude error bound](nodal/RELATIONAL_PATTERN_MEMORY.md#relational-finite-phase-memory), [finite-clock readout](nodal/RELATIONAL_PATTERN_MEMORY.md#relational-finite-time-memory), [finite-duration contact](nodal/RELATIONAL_PATTERN_MEMORY.md#relational-finite-contact-memory) and [retained receiver mean](nodal/RELATIONAL_PATTERN_MEMORY.md#relational-retained-receiver-record); contact cost alone cannot distinguish opposite offsets.
- **Implementation:** [Cycle-memory coefficients and finite certificate](../src/tnfr/physics/relational_cycle_memory.py), [accumulated contact and retention](../src/tnfr/physics/relational_memory_contact.py), existing regional/attachment observations.
- **Tests:** [Coefficient bounds](../tests/test_relational_cycle_memory.py), [independent mechanism](../tests/physics/test_relational_cycle_memory_mechanism.py), [finite certificate](../tests/test_relational_finite_memory.py), [finite-clock readout](../tests/test_relational_memory_readout.py), [finite contact](../tests/test_relational_memory_contact.py), [exact error-budget controls](../tests/physics/test_relational_finite_memory_proof.py).
- **Limits:** A common phase is not an intrinsic isolated-ring label. The cubic, finite-clock and contact/removal bounds apply to their stated preparations. A lasting receiver mean additionally requires regional response and isolated capture/conservation; a port signal alone is insufficient. Persistence under later interventions, nondestructive measurement and physical identification are not established; no contact is executed.

<a id="attachment"></a>

### Supplied bridge interfaces

- **Status:** maintained
- **Basis:** derived
- **Emergence:** not-claimed
- **Definition:** Adding or relocating a supplied unit bridge changes incident degree, form-gradient sums and phase resultants. Fresh fields and signed edge costs describe the instantaneous response and storage jump.
- **Domain:** Simple unit graphs admitted independently before and after the event under one selected native phase domain (acute, positive_resultant or regular); unchanged nodal state, disjoint connected components for attachment, or two nontrivial components separated by the removed bridge for relocation.
- **Premises:** Retain internal state and relative frames; supply the candidate and evaluate both supports through the same law. Event passivity is an additional premise; recovery requires the separate theorem's hypotheses.
- **Dependencies:** relational-law, phase-metric, storage-work
- **Owner:** [Composition and support-event interfaces](nodal/RELATIONAL_PATTERN_COMPOSITION.md).
- **Evidence:** [Attachment identity](nodal/RELATIONAL_PATTERN_COMPOSITION.md#one-bridge-interface-admission), [event-budget and nonselection result](nodal/RELATIONAL_PATTERN_COMPOSITION.md#support-event-premise-admission), [conditional relocation and recovery](nodal/RELATIONAL_PATTERN_COMPOSITION.md#identity-preserving-bridge-relocation).
- **Implementation:** [Shared attachment and relocation observers](../src/tnfr/physics/relational_observations.py).
- **Tests:** [Static attachment, relocation and budget controls](../tests/test_relational_attachment.py).
- **Limits:** Reports do not change live support, authenticate work or automatically certify recovery; acute recovery theorems keep their stronger phase hypotheses. Represented budgets are not transcendental enclosures. Admission selects neither occurrence nor timing.

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
- **Limits:** Modal rhythm and synchronization differ; neither derives a primordial pulse, universal clock or maintained engine wave. Same-law periodic states require the separate [autonomous-pulse theorem](#autonomous-pulse), not this auxiliary spectrum.

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
