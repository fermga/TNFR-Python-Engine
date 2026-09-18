# Nodal research strategy: closure, emergence and laboratory tests

Review dates: 2026-09-17 to 2026-09-18. Scope: source, theory, retained research evidence,
empirical interfaces and selected primary experimental sources. This is a
strategic audit, not an exhaustive software verification or a new stability
theorem. The original review used the historical B75 snapshot
`sha256:f84290315bc1cee1e01f6c71ee1d086a7b56a68d5673fa02c260e04963ea5b92`.
Historical tests and certificates were inspected, not rerun for that review.
The subsequent P1 implementation repairs the engineering defects identified
below and has its own contract tests and source checkpoint. Current delivery
status belongs to the plan; old source line numbers describe the audit snapshot.
The
[research portfolio](../TNFR_lineas_de_investigacion.txt) owns the main/supporting/
deferred classification; the [five-stage plan](research/FIVE_STAGE_EXECUTION_PLAN.md)
alone owns execution order and stage status. This review owns the scientific
rationale and audit findings, not a second task queue.

## 1. Assessment and priority decision

TNFR currently has a useful mathematical and computational core for declared
graph dynamics. Its strongest evidence consists of restricted derivations,
executable contracts, reproducible finite trajectories and counterexamples.
It does not yet supply a closed autonomous multichannel theory or a verified
laboratory correspondence for its complete dynamics. These are separate gaps.

The principal research objective is now **generative structural emergence**:
derive formation, persistence, interaction and observable properties from
TNFR coherence dynamics. The user explicitly selected this priority over
further immediate measurement fitting. The P1-P5 measurement bridge remains
the supporting route for physical tests; it is not the whole programme or
a prerequisite for internal mathematical mechanism studies. C6 indefinite
boundedness remains open and parked. Its binary64 first-exit proof would not
by itself explain autonomous selection or physical emergence.

| Area | Established or implemented content | Boundary |
| --- | --- | --- |
| Pure EPI transport | Graph diffusion, weighted balances, restricted stability and exact-mode reference results | Fixed/declared support and capacity assumptions; not all channels or schedules |
| Canonical events | Thirteen operator contracts, simultaneous stages and finite causal execution records | Admission of a supplied word does not derive its autonomous selection |
| Coarse state | Exact linear projection memory, hidden-initial-state source and restricted quotients | General nonlinear/changing-support closure remains open |
| Pattern support | Forced relative profiles, phase response, winding and capacity-feedback controls | Prepared contrast, freezing and self-restoration are different mechanisms |
| Delayed history | Restricted REMESH disagreement theorems and finite runtime bridges | Not control of every mean, auxiliary state or future call |
| C6 | B75 retains 41/56 excluded first-exit labels, with 15 pending | No indefinite boundedness certificate or instability proof |
| Observations | Signal-processing interfaces, synthetic controls and externally reported empirical summaries | Current APIs and retained evidence do not establish held-out full-nodal physical validation |

These conclusions reuse [core research](CORE_RESEARCH_PROGRAM.md),
[derived memory](DERIVED_EPI_MEMORY.md),
[forced support](FORCED_SUPPORT_BALANCE.md),
[C6](COUPLING_WINDING_PERSISTENCE.md) and the
[scoped emergence review](EMERGENT_ONTOLOGY.md). They do not create replacement
theorem owners or new O/S research identifiers.

### Why this order, and what "optimal" can mean here

The revised order prioritizes the requested generative hypothesis and reuses
existing derivations and negative controls. It is not a proven global optimum.
Physical experiments still require P2 admission; internal structural studies
can proceed with explicitly stated mathematical premises.

| Alternative next focus | Useful contribution | Reason for its present position |
| --- | --- | --- |
| Extend the C6 search | A restricted binary64 first-exit proof | Does not establish measurement semantics; existing unresolved proof search is not a prerequisite for prediction |
| Resolve canonical generation/selection and maintenance | Directly addresses the primary hypothesis | First identify a closed mechanism or scoped obstruction using existing birth, feedback and symmetry tools; avoid unsupported autonomy claims |
| Fit the available oscillator/EEG signals | Descriptive patterns and possible candidate rejection | Availability does not resolve hidden state, clock or same-record fitting; no signal is admitted merely because it fits |
| Calibrated restricted transport, then observation/recovery extensions | Reuses exact dynamics, observability, memory and fixed-target accounting | Supporting physical-test route; current missing data admission does not block primary mathematical work |

Early read-only observability and apparatus checks may run alongside the
diagnostic repair. The single delivery queue is a resource/traceability
policy, not a mathematical requirement that all reasoning be serial.

## 2. What must actually follow from the nodal structure

The [all-parameter foundation audit](NODAL_PARAMETER_FOUNDATIONS.md) now owns
the cross-channel type/unit ledger, form/time covariance and conditional
locality-to-diffusion derivation. It includes the corrected shared distance fit
and metric normalization scope, and distinguishes source-free evolution from
additive forcing. Reuse it with the existing variational/closure owners;
neither the audit nor centralized coefficients select an autonomous completion.
The subsequent signed-polar representation test provides a concrete closure
obstruction without prescribing phase speed: identical complex observations
can have different squared-modulus rates. Retaining phase at zero form and
the sign of EPI avoids that information loss but does not determine their
laws. Timestamped capacity diagnostics and explicit phase-resultant availability
support a shared conditional joint-response identity. It retains both the
pressure derivative and the capacity product term in EPI acceleration.
Phase/capacity compensation is possible only under a support-weighted source
compatibility condition; even strict U3 does not guarantee it. Conversely an
exact finite P2 family keeps nonuniform form with coordinated phase/capacity
changes. This is compatibility evidence, not selection or stability of that
family. The plan alone owns the next finite compatibility gate.

Write `x=EPI` and `p=DeltaNFR`. The law `xdot=diag(nu)*p` determines a rate
only after the state and pressure are specified. A predictive model also
needs its phase/capacity evolution, support, history, event selection and
boundary rules. The equation alone does not select these laws. Defining
`p=observed_rate/nu` after measurement would encode the answer.

Every proposed study must declare, separately:

1. Canonical state and invariants.
2. Derived pressure/operator laws and their domains.
3. Constitutive choices still awaiting derivation.
4. Preparation, imposed inputs and event policy.
5. Numerical representation and computational budgets.
6. Sensor map, physical clock, calibration and uncertainty.

The current pressure owner is [dnfr.py](../src/tnfr/dynamics/dnfr.py), with
weighted EPI transport but arithmetic/support-based other neighbor channels.
The optional `Gamma` path in
[integrators.py](../src/tnfr/dynamics/integrators.py) augments the rate as
`nu*p+Gamma`; it requires separate admission in a strict nodal derivation.
Absorbing it into pressure by dividing by capacity is invalid at zero
capacity and is not a derivation. The optional extended phase law in
[canonical.py](../src/tnfr/dynamics/canonical.py) explicitly uses operational
sensitivities. Neither optional path is evidence of a defect in the C6
continuation that disables or excludes it. Neither becomes a consequence
of the EPI equation merely because its implementation exists.

For this programme, unproved additions stay outside the explanatory model.
Independent measurements may calibrate an observation map or an admitted
state quantity; such calibration is not a first-principles derivation.
External laws, fitted damping, noise or potentials remain comparison models
unless their TNFR derivation is supplied. Measurement uncertainty is an
observation property, not an extra force silently added to the dynamics.

### Measurement correspondence versus generative emergence

The target is not to rename measured voltage, temperature or other recorded
coordinates as EPI. A deeper structural account is a research hypothesis;
neither the nodal factorization nor a successful measurement fit establishes
that TNFR is physically prior to the observed system. "Underlying" here means
a proposed explanatory state, not a demonstrated earlier physical time.
The motivating hypothesis is stronger than interpreting each already known
physical particle as a graph node: observable entities and their properties
would arise as persistent, interacting coherence patterns of an underlying
nodal dynamics. This is an open physical hypothesis, not an established
consequence of the repository. Mathematical projection memory is a useful
example of derived behavior, but does not by itself establish this stronger
claim about physical constituents.

Such a claim must identify a canonically generated family of patterns, its
formation and persistence conditions, its interaction law and the observation
rule producing measurable properties. A stable graph pattern alone is not a
physical particle identification. Physical particle or quantum claims also
require their corresponding quantitative measurement predictions; classical
wave polarization or a discrete graph spectrum cannot silently supply them.
If physical time is itself claimed to emerge, the present structural evolution
parameter and its relation to measured time must be accounted for, not treated
as an already derived physical clock. Graph, partition and operator-policy
selection likewise remain explicit premises wherever they are imposed.

Keep the following three claims separate:

| Claim | What would establish it within a declared scope | Present boundary |
| --- | --- | --- |
| Measurement correspondence | A fixed observation map, admitted clock/state and independent prediction | P2/P3 target; Volts was a within-acquisition exploration with unverified mapping |
| Generative mathematical reduction | Derive an observed law or effect from a specified finer nodal dynamics and observation rule, without fitting a new effective force or memory law | Exact restricted projection/memory results exist; their graph, capacities and partition are hypotheses |
| Physical emergence from that mechanism | The same admitted fine model predicts observable responses across reserved preparations or observation scales, including a contrast that can reject the proposed reduction | Not established; even success would not prove unique or universal physical fundamentality |

A measurement has the form `y=H(z)` in a proposed model: `z` is the declared
TNFR state and `H` the observation rule. A direct scalar chart such as nominal
voltage-to-EPI is a measurement hypothesis, not a derivation of voltage from
an unobserved substrate. A generative reduction instead derives the evolution
of `H(z)` from an independently specified nodal dynamics. The observation rule
must represent what an instrument or aggregation actually reads; it must not
be an arbitrary function adjusted separately to encode each response. Hidden
initial conditions must be known, bounded or observable from allowed data.

The existing [five-node memory witness](DERIVED_EPI_MEMORY.md#6-exact-nonzero-memory-witness-p5)
is a concrete reuse target. On its fixed unit path and declared partition,
states `(0,1,1,1,0)` and `(0,0,3,0,0)` have the same two macro coordinates
`(0,1)` but different initial macro rates, `(1,-1/3)` and `(0,0)`.
The shared nodal generator derives both the hidden-state source and the
subsequent memory kernel; no new delay parameter is fitted. These are existing
exact-model results, not new experiments or observations of physical emergence.
They also do not derive the physical selection of that graph or partition.

For experiment selection, prefer an admitted source that can test parameter
transfer between preparations, a predicted modal-rate ratio, or an existing
projection/closure contrast. One exponential continuation can exercise the
measurement bridge but cannot by itself discriminate microscopic structure.
Different models with identical observable predictions remain indistinguishable
by that experiment. Do not demand a unique TNFR signature from the pure-EPI
sector, which is graph diffusion under the stated hypotheses, or interpret
baseline accuracy alone as proof of an underlying ontology. The sole execution
plan prioritizes the generative mechanism and retains P2 as supporting work;
this does not bypass P3 before a physical P4 study.

### Two immediate identifiability constraints

**Capacity versus pressure.** Rate observations alone constrain their product.
At that observation level, replacing positive `nu_i` by `a_i*nu_i` and
`p_i` by `p_i/a_i` preserves the product. This is not necessarily a symmetry
of the full canonical pressure law, which itself can depend on capacity.
A declared pressure realization, independent calibration or discriminating
intervention is required to identify the factors.

**Relative conductance versus absolute coupling.** In the exact-real,
fixed-support EPI channel, for `kappa>0`,

`L_rw(kappa*W)=I-(kappa*D)^(-1)*(kappa*W)=L_rw(W)`.

At fixed capacity, uniformly multiplying all edge weights therefore changes
no EPI-channel trajectory. A laboratory coupling-strength sweep cannot be
represented by this weight multiplication alone. The clock/capacity map
must independently explain the rate change; fitting it separately to each
test outcome removes that prediction. Binary64 materialization can perturb
the identity numerically, but is not an explanation of a physical coupling
response. The identity does not cover the singular `kappa=0` support boundary
or arbitrary multichannel dynamics.

These constraints suggest useful tests instead of new explanatory constants.
For example, on a fixed symmetric graph with common positive capacity,
nonuniform pure-EPI modal rates obey `r_k/r_l=lambda_k/lambda_l` for positive
eigenvalues. A common time-scale calibration cancels from the ratio. This is
a restricted, dimensionless prediction to test after admitting the sensor
map, not a unique prediction distinguishing TNFR from ordinary diffusion.

### Cross-domain reuse for the generative hypothesis

This source review reuses the older ontology, Riemann and Millennium work as
mechanism libraries, not as established solutions or one unified physical
theory. A result counts as generative only relative to its explicit inputs:
an operator built using primes, a chosen winding ring, a supplied potential
well or an imported differential equation must not be reported as generating
that same structure without those premises. Exact properties of such models
can still supply valuable tools and counterexamples.

| Source family / existing owners | Reusable structural content | Input or missing bridge that prevents promotion |
| --- | --- | --- |
| Riemann and [arithmetic pulse](../src/tnfr/mathematics/arithmetic_pulse.py), [Krylov/Hankel tools](../src/tnfr/mathematics/krylov.py) | Finite observable recurrences and exact ranks can test how many nodal degrees a readout distinguishes | Arithmetic graph/seed construction is supplied. General Hankel output rank need not equal reachable Krylov dimension without observability; finite prime-frequency sums are not an unregularized zeta series on the critical line |
| [Pointed symmetry](../src/tnfr/physics/pointed_symmetry.py), [operator](../src/tnfr/physics/operator_equivariance.py) and [word](../src/tnfr/physics/word_equivariance.py) audits | Separate label-invariant structural behavior from a chosen node, seed or observer; useful for parent selection and pattern identity | Existing comparisons of old-node channels do not compare newborn nodes, changed edges or hierarchy; graph automorphisms alone do not establish symmetry of phase, preparation and grammar history |
| [Navier-Stokes notes](TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) and [variational bridge](TNFR_VARIATIONAL_PRINCIPLE.md) | Energy/transport budgets and restricted diffusion provide controls for persistence and growth claims | The simulated fluid PDE is supplied, not derived from TNFR. Fixed-viscosity 3D regularity is not established by finite runs; an inviscid-limit uniformity question is a separate obligation |
| [P versus NP notes](TNFR_P_VS_NP_RESEARCH_NOTES.md), [variational audit](../src/tnfr/physics/variational.py) | Finite trapping and descent obstructions help test whether a purported restoring scalar actually decreases | General nodal pressure is not minus the tetrad-potential gradient. Only the stated Dirichlet/mobility model has that proven gradient structure; trap diagnostics do not prove complexity separation |
| [Yang-Mills notes](TNFR_YANG_MILLS_RESEARCH_NOTES.md), [gauge readout](../src/tnfr/physics/gauge.py) | Gauge covariance and obstruction checks distinguish independent relational degrees of freedom from a coordinate change | The implemented phase-difference connection is pure gauge; finite structural spectral gaps do not construct non-Abelian physical gauge dynamics or a continuum mass gap |
| [Hodge notes](TNFR_HODGE_RESEARCH_NOTES.md) | Exact chain/cochain identities and harmonic sectors are possible controls for global structure versus local gradients | The simplicial complex and cochains are supplied; the TNFR tetrad-to-cochain evolution bridge and the algebraic-cycle claim remain unproved |
| [BSD notes](TNFR_BSD_RESEARCH_NOTES.md) | Arithmetic/rank diagnostics illustrate how observable degeneracy depends on the supplied construction | Arithmetic inputs and GL(2), rank/order and analytic bridges must remain explicit; no physical pattern, particle identity or BSD theorem follows from a numerical analogy |
| [Scoped ontology](EMERGENT_ONTOLOGY.md), [winding persistence](COUPLING_WINDING_PERSISTENCE.md), [derived memory](DERIVED_EPI_MEMORY.md) | Conditional discrete winding, protected phase evolution, coarse memory and sufficient-state results are already nodal mechanisms or exact graph facts | Winding spreads under the protected update; it is not localized matter. Selected graph spectra, U(2) polarization, external wells and vortex models do not supply physical particles, measurement probabilities or autonomous interactions |
| [THOL birth](THOL_BIRTH_AND_TRANSPORT.md), [support balance](FORCED_SUPPORT_BALANCE.md), [feedback](CHILD_COUPLING_FEEDBACK.md) | The strongest executed generative chain: nodal history, actual threshold, child, edge and subsequent pressure/transport | Event/target selection remains supplied; held capacity/phase support, mean drift, freezing and a changed target cannot be relabeled active self-restoration |

Two concrete synergies now have a bounded executable audit. First, transfer
the pointed-symmetry question from arithmetic
observation to **who selects the parent of a structural birth**. For a declared
state action and equivariant deterministic selector `s(gX)=g*s(X)`, every
stabilizer of `X` must fix the selected node. An orbit without a fixed candidate
therefore forbids an unmarked unique selection. This elementary implication
is not a theorem that the existing C8 runtime state has that stabilizer:
its winding phases and pointed IL/OZ history must be included in the state.
The exact finite-action implementation and a new bounded test including
newborn correspondence are centralized in
[the birth result](THOL_BIRTH_AND_TRANSPORT.md#8-selection-actual-dispatch-and-the-unique-parent-obstruction).
They keep declared model symmetry separate from marked runtime relabeling.

Second, connect **selection to the operation actually executed**. The default
[selector](../src/tnfr/dynamics/selectors.py) does not choose THOL. Its parametric
variant can choose that glyph, but primitive pressure dispatch is distinct
from the public hierarchy-creation operation used by the birth study. A stored
THOL pressure increment can also be erased by a subsequent default refresh.
The completed audit traces these links: both real built-in policies select
IL on the above-threshold prepared C8 source; separate explicit interventions
distinguish primitive pressure from public birth and held from refreshed
pressure consumption. Trigger sufficiency fails in that scope. The symmetry
argument permits investigating candidate sets rather than an unmarked single
parent, but it neither derives their activation law nor justifies simultaneous
public births. The execution plan owns that continuation. No heuristic score
or corrected software dispatch should be promoted to a derived physical law.

The cross-domain review found stale scope claims in the old number-theory,
P-versus-NP and fluid notes. Corrections preserve historical constructions
while distinguishing inserted critical-line shifts, conditional equivariance,
restricted gradient flow and the actual open regularity problem. Their old
wording is not evidence for the primary hypothesis. The precise next task,
completion condition and parked branches live only in the execution plan.

### Module reuse for set-valued generation

The focused follow-up inventories the package layout (586 Python files at this
checkpoint) and traces the interacting owners below, their callers and tests.
It is a cross-module audit of the current generative dependency chain, not a
claim that every repository file or research theorem was reverified. The older
domain review above remains useful; no arithmetic/Millennium campaign is
restarted to answer a structural admission question.

| Responsibility | Reuse owner | Required boundary |
| --- | --- | --- |
| Physical/legacy acceleration evidence | [nodal_equation.py](../src/tnfr/operators/nodal_equation.py), `observe_structural_acceleration` | One detached observation carries source, time basis, last-three samples and availability. Physical history has priority and must match the current EPI endpoint. The numeric compatibility wrapper still returns zero for insufficient history; use executor receipts for causal provenance. |
| Grammar admission | [grammar_dynamics.py](../src/tnfr/operators/grammar_dynamics.py), `validate_candidate` | Admission is not operator selection, optional precondition satisfaction, or a viable birth. A validated future handler supplies no past history. |
| Optional THOL gate | [preconditions/self_organization.py](../src/tnfr/operators/preconditions/self_organization.py) | One read-only gate for public and legacy callers. Respect whether the configured public gate is enabled; do not impose positive pressure on a configuration where that gate is disabled. |
| Threshold configuration | [_thol_config.py](../src/tnfr/operators/_thol_config.py) | Explicit tau, generic alias, THOL alias, core default. A configured threshold is not a derived constant. OZ/ZHIR and advisory scores retain their separate meanings. |
| Actual birth viability | [self_organization.py](../src/tnfr/operators/self_organization.py), `_prepare_execution` | Reuse its proposal, depth, hierarchy, metabolic and metric validation. `abs(acceleration)>tau` alone does not establish a materializable child. |
| Joint birth, conflicts and rollback | [network_stage.py](../src/tnfr/operators/network_stage.py), `execute_self_organization_stage` and `GraphTransactionSnapshot` | Public births already use a common detached snapshot, merged child IDs and whole-support validation. Check the returned schedule: subclass/grammar fallback is sequential. |
| Operator routing and actual connection | [word_execution.py](../src/tnfr/operators/word_execution.py), [_coupling_stage_kernel.py](../src/tnfr/operators/_coupling_stage_kernel.py), [sampling.py](../src/tnfr/dynamics/sampling.py) | Reuse dispatch and UM; explicitly refresh the candidate sample after birth. Newborns are not automatically later operator targets. |
| Parent symmetry and whole eligible sets | [selector_symmetry.py](../src/tnfr/physics/selector_symmetry.py), [symmetry_sectors.py](../src/tnfr/physics/symmetry_sectors.py) | Exact supplied-group obstruction differs from a numerical invariant-sector projection. Declare phase, history, marks, support and algorithmic ordering. |
| EPI transport, connection budgets and targets | [support_transport.py](../src/tnfr/physics/support_transport.py), [forced_support.py](../src/tnfr/physics/forced_support.py) | Birth changes dimension; an edge reset compares the same postbirth nodes. Keep the original target and separate relative shape from mean drift. |
| Reduction and hidden state | [epi_memory.py](../src/tnfr/physics/epi_memory.py), [observability.py](../src/tnfr/physics/observability.py), [phase_quotient.py](../src/tnfr/physics/phase_quotient.py) | Fixed-model memory/quotient results detect missing information; they do not establish closure of the nonlinear changing-support birth process. |
| Prescribed metabolic controls | [dynamics/metabolism.py](../src/tnfr/dynamics/metabolism.py), [operators/metabolism.py](../src/tnfr/operators/metabolism.py), [bifurcation.py](../src/tnfr/dynamics/bifurcation.py) | Word construction, stress sensitivity, amplitude maps and route scores are supplied policies/diagnostics, not derived autonomous selection. |
| Provenance and evidence scope | [core_manifests.py](../src/tnfr/research/core_manifests.py), [event_runtime.py](../src/tnfr/operators/event_runtime.py), [stage_contracts.py](../src/tnfr/operators/stage_contracts.py) | A detached observation is not an executor seal; finite stage provenance is not indefinite stability or physical identification. |

The audit identifies concrete consolidation boundaries. The duplicate legacy
THOL validator used permissive casts and wrote context/bifurcation metadata
during validation, while the public gate was strict and read-only. Both now
delegate to the public gate's shared implementation. Telemetry belongs to
successful execution; compatibility callers no longer mutate graph state as a
side effect of a readiness check. Public gate activation remains configurable.
The public operator and metabolic wrapper also share threshold resolution,
including finite-value validation and exact precedence. Existing mathematical
and runtime owners are retained rather than replaced by a second birth engine.

The advisory bifurcation score also had an incorrect threshold label. Its
configured combination can give `0.54` with zero acceleration, or `0.46` at
or beyond its acceleration scale with the other contributions zero. Thus
`score>=0.5` is not the public THOL birth condition. The
[advisory counterexamples](../tests/operators/test_bifurcation_advisory_scope.py)
also show that a suggested THOL can fail its public gate. Documentation now
describes that helper and metabolic words as configured diagnostics/policies;
their algorithms and coefficients have not been tuned to obtain births.

The old symmetry projector averaged whatever automorphisms were returned,
including a cap-truncated collection. This is not generally a projector: for
`R^3=I`, `Q=(I+R)/2` has `Q^2-Q=(R^2-I)/4`, which is nonzero. The shared orbit
partition now constructs the invariant-sector projector by orbit averaging.
Explicit supplied permutations are generators of the represented subgroup;
they need not enumerate a closed group. Automatic enumeration must finish
within its cap or reject; it no longer silently promotes a partial list to
`Aut(G)`. Requested edge weights use exact matching, and unsupported parallel-
edge enumeration fails explicitly. An explicitly supplied action still does
not authenticate those permutations as graph or complete-state symmetries.
The exact selector theorem separately keeps its closed-group validation.

For a complete declared state `X`, a covariant eligible set obeys
`E(gX)=gE(X)`. If `gX=X`, then `E(X)=gE(X)`, so it must be a union of
stabilizer orbits. This reuses the exact selection obstruction without
inventing a unique representative. The empty set is valid and must be
handled before the nonempty-candidate theorem helper. Nodal labels are
colors: node IDs stored inside labels are not automatically transported.
Encode parent/child links and birth order as directed relations, or use an
explicit typed bijection like the retained newborn test. Neither a covariant
set nor removal of a naming artifact derives the decision to execute it.

The [birth energy boundary](THOL_BIRTH_AND_TRANSPORT.md#3-birth-creates-a-node-coupling-must-supply-an-actual-edge)
is another useful reuse result: isolated birth extends the Dirichlet form by
a zero block and leaves its energy unchanged. This is diagnostic blindness,
not free physical creation. The child's zero degree also excludes the
positive-metric connected-profile theorem before attachment. The existing
same-support reset then accounts for the actual UM edge. No additional
energy functional is required; proof and executable control stay with the
birth result owner.

Three orchestration details constrain reuse. First, THOL already resolves
cross-parent child-ID collisions in snapshot node rank, while telemetry keeps
requested target order. Its structural target-order result is not a theorem
of complete insertion-order or relabeling invariance. Second, network words
and event invocations freeze their original targets; sampled candidates and
target sets must be distinguished after growth. Fixed-support flow evidence
cannot silently include a changing-node jump. Complete the birth stage and
start a fresh declared flow invocation on its resulting support. Third, an
empty stage can still refresh pressure and write stage metadata. A relation
observer must not dispatch it accidentally while claiming a read-only check.
The eligibility adapter's outer transaction now covers input materialization,
fresh observation and the reused stage commit; empty dispatch skips the stage.

These findings led to the implemented
[eligible-set/dispatch contract](THOL_BIRTH_AND_TRANSPORT.md#9-set-valued-eligibility-and-explicit-finite-dispatch).
It preserves separate fields for history availability, grammar admission,
optional gates, threshold crossing and viable public proposal. The existing
simultaneous birth stage and shared merge validation perform the action;
coupling remains separate. The completed
[preparation/policy comparison](THOL_BIRTH_AND_TRANSPORT.md#10-preparation-dependence-and-distributed-explicit-birth)
finds 0/1/8 eligible parents for no/single/all-node real prefixes. A common
prefix and explicit simultaneous dispatch create eight isolated children;
both built-in policies still choose IL. Supplied preparation and policy are
therefore separate explanatory premises. The finite rotation control retains
phase, histories and neighbor order; it is not a complete symmetry theorem.
The [distributed continuation](THOL_BIRTH_AND_TRANSPORT.md#11-distributed-generated-support-and-pressure-feedback)
now closes the finite birth/support/refreshed-pressure path: actual UM adds
16 edges and all eight children evolve; two controls retain isolated,
unchanged children. Exact energy accounting and the independent forcing
capture reveal both transport and a negative capacity-channel weighted source.
Child growth and collective drift coexist. This does not establish maintenance.
The connected generated support meets the existing forced-support reference
domain. The [fixed-target continuation](THOL_BIRTH_AND_TRANSPORT.md#12-fixed-original-model-profile-under-a-later-um-stage)
now reuses its captured channels and event/target observers. A later all-node
UM adds eight edges and changes the compatible relative profile while both
finite branches approach the frozen original target. The UM branch approaches
less; signed same-state rate accounting separates generator change from target
forcing and numerical realization. Its changed model is incompatible with the
old profile, so short error decrease cannot establish its restoration.
The original derived profile is not the actual born shape, and original-metric
mean change is not the changed model's drift coefficient. Disconnected controls
do not enter this theorem by silently dropping children. The completed
[lineage comparison](THOL_BIRTH_AND_TRANSPORT.md#13-lineage-scoped-coordination-on-the-generated-support)
uses actual ancestry without selecting labels by outcome. Parent-only UM adds
no edges; child-only UM forms the eight-edge child ring. Neither held model
preserves the original profile, despite finite error reduction. Its target-local
proposals match those in the retained all-node control; their joint effects
are not additive channel ablations. This closes the declared dispatch-scale
comparison and motivates examining the state needed to describe each family,
rather than searching more target subsets. No second score,
snapshot system, pressure law or stability ledger is needed. The execution
plan alone owns the next task and its acceptance conditions.

The resulting support also makes the older quotient/memory tools relevant to
the generative objective. Actual THOL ancestry supplies candidate parent-child
families, but a hierarchy label does not prove that each family evolves as one
closed macro node. On a fixed captured affine model `x'=-A*x+b`, the shared
reversible projection gives `y'= -R*A*P*y-R*A*Q*x+R*b`, with `Q=I-P*R`.
All-state affine closure requires `R*A*Q=0`; reversibility equates this with
the lift condition `Q*A*P=0`. This is the existing
[projection/memory criterion](DERIVED_EPI_MEMORY.md#4-kernel-positivity-and-the-exact-closure-criterion),
with the independently captured constant source retained, not a new force.
The completed [family analysis](THOL_BIRTH_AND_TRANSPORT.md#14-state-sufficiency-of-actual-parent-child-families)
finds that hidden-state term nonzero in all four retained held models, with
exact same-observation/different-rate witnesses. Ancestry alone is insufficient
for the chosen observation; this does not reject the fine TNFR model.

For a memory calculation in homogeneous coordinates, use each current held
model's `u=x-m_H(x)*1-z` and its exact `A=e*diag(nu)*L_rw`. Applying the
pure-EPI observer directly to the forced raw EPI would erase a real source.
The frozen original `z0/H0` used to evaluate retention remains separate from
these coordinates used for elimination. A support/capacity event can change
the projection itself: `(R1-R0)*x` is an observation-coordinate change even
when EPI has not moved. Exact rational closure checks, numerical residuals and
actual runtime observations must keep their existing distinct scopes.

The completed minimal-state study reuses the
[P5 minimality argument](DERIVED_EPI_MEMORY.md#93-minimality-of-the-retained-linear-state),
the rectangular rational matrix product and
[exact rank](../src/tnfr/mathematics/krylov.py) to derive the smallest invariant
row space containing these actual family observations. Its observation
`s=C*x` retains `R=D*C` and obeys `s'=-G*s+C*b` with `C*A=G*C`.
The captured source remains explicit. Extra coordinates can be signed global
contrasts, not newly established macro nodes. All four retained models have
exact ranks `8,15,16,16`: eight extra linear coordinates are necessary and
sufficient, so there is no proper all-state linear compression preserving
these means. Results and evidence belong to
[THOL Section 15](THOL_BIRTH_AND_TRANSPORT.md#15-minimal-sufficient-linear-state-for-the-retained-family-observations).
The available sixteen-coordinate realization already supports a full EPI
response study. Quantifying the cost of discarding its internal coordinates
is optional, so that approximation question is parked while the primary
route tests actual canonical mechanisms. The executor evolves native
EPI; the invertible `C` chart preserves its information and prediction.
Exact full rank does not establish autonomous persistence, nor does it create
a need to approximate before testing a mechanism.

The completed [full-state Emission response](THOL_BIRTH_AND_TRANSPORT.md#16-full-epi-response-to-one-child-emission)
uses these coordinates without loss. Its fixed finite prediction accounts
for runtime numerical defects; paired spatial difference energy decreases
while a mean offset persists. AL improves the original-target score at the
event, so this is not evidence of recovery from target damage. This resolves
a held-model response question and makes the distinction from endogenous
maintenance sharper. The existing composite runtime can test which mechanisms
actually respond without inventing a new pressure law; its policies and their
phase/capacity/history effects must remain explicit.

The completed [native step](THOL_BIRTH_AND_TRANSPORT.md#17-one-native-composite-runtime-response)
now locates actual all-node IL pressure feedback and later phase evolution;
capacity adaptation is inactive under its unchanged gate. The two branches
retain common phase/forcing, permitting a conditional paired nodal recurrence,
while the original individual relative profile becomes incompatible with the
new source. This separates incremental attenuation from preservation of a
fixed shape. No moving target may be fitted afterward and labeled recovered
identity.

The completed [five-step window](THOL_BIRTH_AND_TRANSPORT.md#18-five-additional-native-policy-steps)
retains shared coefficients but encounters the actual mixed policy
`IL/IL/EN/IL/AL`. The common-IL map remains valid on three steps; EN and AL
write EPI using the existing sequential runtime while integration consumes
pressure generated before those writes. Their lag-counter forcing is a
configured scheduling mechanism, not an emergent transition. Paired spatial
energy falls 75.74646%, the EN step changes the mean offset, and the original
target remains incompatible. Capacity adaptation never qualifies.

There is also a separate [phase-direction conditioning boundary](THOL_BIRTH_AND_TRANSPORT.md#global-phase-direction-near-the-symmetric-input):
the ideal regular twist has no global phasor direction, whereas the exact
represented angles have a rigorously enclosed tiny nonzero resultant. The
argument map is sensitive there. An absolute direction difference alone
cannot reject structural persistence because a common rotation may preserve
relative phase; comparisons must retain node/phase alignment and invariant
readouts. The completed
[retained-mechanism audit](THOL_BIRTH_AND_TRANSPORT.md#19-retained-reset-accounting-and-phase-conditioning)
closes the EN/AL accounting with the actual sequential resets and
pre-generated pressure. Pure relabeling with transported enumeration preserves
the phase output exactly. A change of node enumeration at the same named
input changes its relative phase pattern, not only its absolute orientation.
This rules out numerical robustness for that comparison; it neither rejects
all phase mechanisms nor supplies an alternative canonical force.

This supporting block is closed. The main remaining distinction is between
a canonical operator's **action** and the structural law selecting its
**activation, targets and order**. Individual contracts constrain permitted
transformations; they do not by themselves determine the full event history.
The nodal equation fixes the EPI rate once capacity and pressure are supplied,
but does not uniquely fix every pressure realization, capacity/phase law or
selector. The registry's completeness is also a separate open S10 question.
Calling these obligations open does not discard the canonical operators.
An operator describes a transformation of NFRs or their relations, not a
separate entity acting on them. The implemented AL requires existing basal
capacity on an existing node, and THOL requires a parent. Their execution
does not derive the initial nodal substrate. A new graph vertex is also not
by itself evidence that a new coherence region forms and persists. Selection
must be studied within this configuration-to-transformation mechanism, without
attributing intentional agency to nodes or searching for a preferred word.

Configured lag counters can form part of an autonomous augmented-state
program. What remains unproved is that this particular scheduling rule follows
from nodal structure. The next principal question concerns that derivation,
using the already available acceleration, grammar, eligibility and symmetry
results. Existing fixed-capacity diffusion and forced-support results already
separate passive relaxation from a supported differentiated profile; the P2
capacity-feedback theorem already supplies one restricted feedback control.
Repeating them or extending the same native horizon would not answer the
selection question. The plan owns the sole bounded task and its stop rule.

### Shared observations before further generation research

The additional cross-module review follows history through THOL, the SDK,
propagation diagnostics and the nodal integrator, and compares approximate
diffusion readouts with the existing exact support budgets. It does not rerun
every theorem or certify the repository as free of defects. The useful
unifications preserve the following distinctions rather than assigning one
meaning to every value named acceleration or energy.

| Quantity | Shared owner and integration | Scope retained by eligibility observation |
| --- | --- | --- |
| Three-sample EPI acceleration | `observe_structural_acceleration`, consumed by THOL, SDK and propagation diagnostics | Missing/short history is unavailable, not observed zero. Physical and unit-operator-step time bases remain explicit; nonfinite derived spans/rates reject. |
| Two-sample signed EPI rate | [mutation_trigger.py](../src/tnfr/physics/mutation_trigger.py) | A valid Mutation secant can coexist with unavailable acceleration. Do not reuse its availability flag for a three-sample question. |
| Cached integrator acceleration | [integrators.py](../src/tnfr/dynamics/integrators.py) | This is a difference of represented RHS rates, potentially before EPI clipping. It is not automatically the finite difference of the realized history. |
| Propagation association | [propagation.py](../src/tnfr/dynamics/propagation.py), `detect_bifurcation_cascade` | Require a positive record naming the requested source and available acceleration. Records lack timing and a before/after acceleration comparison; the output cannot identify the cause of a crossing. |
| Network SDK summaries | [simple.py](../src/tnfr/sdk/simple.py) with existing sense, circular-mean and scalar-EPI readers | Aggregate the entire network without writing live sense telemetry; respect circular phase and signed scalar representation. An undefined direction or invalid tetrad field is not a safe measured value. |
| Approximate diffusion and exact support budgets | [structural_diffusion.py](../src/tnfr/physics/structural_diffusion.py), [support_transport.py](../src/tnfr/physics/support_transport.py) | Do not promote underflow to equilibrium. Approximate readouts reject nonrepresentable nonzero terms; detached rational budgets retain their separate exact domain. |
| Changed-support profile/reset accounting | [forced_support.py](../src/tnfr/physics/forced_support.py) | Public inputs are reconstructed and checked once per call. A private reset core reuses those validated profiles inside the same event observation, avoiding two repeated exact Poisson solves. |

The numeric failure has a direct nodal witness. For a two-node edge with
`w=2^-600`, `EPI=(2^-600,0)` and `nu=(1,1)`, the random-walk diffusion rate is
the representable pair `(-2^-600,+2^-600)`. Multiplying conductance by the
EPI difference first can underflow, returning a false zero gradient and rate.
Even with `w=1`, the same EPI has nonzero exact Dirichlet energy `2^-1201`
and derivative `-2^-1199`, below the binary64 range. Refusing that approximate
balance prevents a false equilibrium claim; it does not change the nodal law
or substitute an unreported exact solver. Likewise, finite sample timestamps
can have a nonfinite total span: accepting that span previously allowed a
nonzero acceleration to appear as zero.

These repairs supply one shared evidence observation to the implemented
eligible-set relation without inventing another history estimator. SDK reports
retain the old numeric acceleration field for compatibility and expose the
independent observation. A supplied history remains supplied evidence, not a
causal runtime seal. The propagation source check removes a reproduced false
attribution; the retained compatibility metadata explicitly disclaims causation.

The exact forced-support optimization is confined to one invocation. It does
not cache results across mutable graphs or trust fields of a supplied public
record. It keeps changing dimension, isolated newborns, mean drift and target
replacement under their existing separate contracts. This is the reusable
bridge from birth/coupling to a later transport budget; it is not an automatic
generation or restoration mechanism.

The legacy optional `validate_nodal_equation` helper remains a post-state,
held-step comparison, not a general operator canonicity certificate. In
particular, its standalone argument handling and numerical clipping convention
still need a separate compatibility review before it can be reused as an
admission boundary. Eligibility uses the public proposal/grammar boundaries
above instead. Neither the shared observations nor
the SDK safety flags replace them.

### Region formation: integrated reuse and missing mechanisms

This follow-up source/theory/test review addresses **formation and persistence
of an NFR as a coherent region**, rather than creation of a graph vertex alone.
The [ontology map](EMERGENT_ONTOLOGY.md#11-nfr-formation-definition-birth-and-collective-dynamics)
retains that distinction. The review found one missing mathematical bridge
worth implementing: a regional budget which retains the environment. Existing
selection witnesses already settle the narrower permission-versus-choice
question; another selector sweep would repeat that evidence.

The highest-value reuse is the following connected chain. The rows are
dependencies within O1/O3/O4/S16, not independent new research programmes.

| Connection and existing owners | What can be reused now | Required scope or missing bridge |
| --- | --- | --- |
| Region and environment: [support transport](../src/tnfr/physics/support_transport.py), [forcing capture](../src/tnfr/physics/forcing_realization.py), [forced targets](../src/tnfr/physics/forced_support.py) | Exact internal dissipation, signed boundary exchange, independent phase/capacity/topology sources and pressure defects | New regional observer preserves full-graph strengths. Fixed-region instantaneous balance does not prove sustained regional identity |
| Temporal identity and sufficient state: [observability](../src/tnfr/physics/observability.py), [structural distance](../src/tnfr/physics/structural_state_distance.py), retained lineage and native receipts | Separate mean, relative form, phase, capacity, support and history; compare actual descendant sets under transported labels | The sixteen-coordinate result closes a held EPI model, not the full runtime. Structural distance has a fixed topology/domain and declared channel scales; it supplies neither birth correspondence nor history identity |
| Environmental memory and scale: [EPI memory](../src/tnfr/physics/epi_memory.py), [quotient/reduction certificates](../src/tnfr/physics/reduction_certificates.py), [spectral projectors](../src/tnfr/physics/spectral_projectors.py) | Eliminating the environment derives a hidden-source and memory term. Exact quotient tests can reject autonomous macro evolution. Projectors avoid choosing a privileged vector within a degenerate normal eigenspace | Static Kron resistance preservation is not dynamic closure; its stiffness is not the heterogeneous nodal generator. Use the reversible H metric before transferring orthogonal spectral results. The eight-family closure failure is already known |
| Sources and restoration: [capacity localization](../src/tnfr/physics/capacity_localization.py), [capacity feedback](../src/tnfr/physics/capacity_feedback.py), [phase response](../src/tnfr/physics/phase_response.py), [hybrid gains](../src/tnfr/physics/hybrid_operator_stability.py) | Distinguish a held contrast-supported profile from dynamics maintaining that contrast. Check source compatibility, phase sensitivity and mean drift separately from disagreement decay | Restricted P2/C6 theorems do not transfer to the sixteen-node multichannel system. The original profile is incompatible after native phase evolution; zero instantaneous error slope is not compatibility |
| Birth and activation: [acceleration](THOL_BIRTH_AND_TRANSPORT.md), [eligibility](../src/tnfr/operators/self_organization_selection.py), [grammar](../src/tnfr/operators/grammar_dynamics.py), [selector symmetry](../src/tnfr/physics/selector_symmetry.py) | Reuse the actual history, allowed target relation, complete proposals and atomic dispatch. Existing witnesses distinguish an allowed action from a uniquely derived one | Stable diffusion can cross the configured acceleration trigger. Neither that crossing nor a successful supplied dispatch derives autonomous initiation or a phase transition |
| Persistence controls: [REMESH history](../src/tnfr/physics/remesh_history_stability.py), [schedule envelopes](../src/tnfr/physics/remesh_schedule_policy_stability.py), existing zero-capacity and fixed-support controls | Test whether an apparent survivor is a delayed copy, frozen state, maintained source, drifting consensus or restoring coupled pattern | Alpha-one delay can cycle; spatial contraction does not control the mean; conditional schedule gains do not derive the schedule |

The new regional derivation and implementation live in
[forced-support balance, section 7](FORCED_SUPPORT_BALANCE.md#7-a-region-and-its-environment-on-the-same-nodal-support).
It resolves three terms in a region's centered EPI-variance rate: internal
dissipation, boundary work and independently captured non-EPI forcing. Stored
pressure defects remain separate. The complementary weighted-total boundary
currents cancel exactly, while regional variance need not decrease. Thus a
region can be maintained by its relations without being a closed subsystem.
This is a direct consequence of the existing nodal model, with no fitted
force, new clock or new operator.

The children in the first attached THOL snapshot provide an additional
conditional calculation: with no child-child edges, their held-environment
response block is diagonal. Its equilibrium is the actual weighted parent
input plus `F_i/e`, with rate `e*nu_i`. This does not require a new solver or
discarding the eight hidden coordinates. It identifies what the environment
would have to maintain; moving parents or phase/capacity invalidate a frozen
target. Boundary dependence is compatible with the relational NFR definition.

The [finite regional identity](FORCED_SUPPORT_BALANCE.md#9-finite-regional-observation-with-a-held-nodal-rate)
extends the same owner to an observed endpoint. It separates initial held-rate
internal/boundary/source terms, the Euler quadratic term and endpoint defects.
These first-order terms are not time-integrated measured fluxes. In native
records, generated pressure, intentional operator writes and later phase
updates must be distinguished; an updated source cannot be assigned
retroactively to a completed integration step. Lineage continuity and exact
relative-form equality remain separate observations, neither an autonomous
maintenance criterion by itself.

The [retained temporal audit](FORCED_SUPPORT_BALANCE.md#10-retained-temporal-regional-identity-audit)
now closes these finite budgets on the original control interval `1.5-1.75`.
Membership, support and capacity persist while all nine relative EPI forms
change. Child variance increases by approximately `0.00126775466`: parent
boundary work drives the contrast and IL partly attenuates that drive.
Variance is not the canonical coherence score or a universal identity test.
The later phase update changes the canonical source, so neither source
maintenance nor future identity follows from this accounted increment.
The known phase-enumeration sensitivity remains a gate for interpreting
subsequent source responses; the execution plan owns the next bounded task.

The [phase-to-source gate](FORCED_SUPPORT_BALANCE.md#11-phase-source-relevance-at-a-fixed-regional-state)
now establishes that this ambiguity affects the canonical drive at the
retained endpoint. Two archived phase outputs, evaluated with the same
remaining state and enumeration, give a maximum phase-source difference of
approximately `0.04914324`; pair 3's model variance rate changes sign.
Historical stored pressure and its nodal rate remain unchanged. Thus the
ordering issue cannot be dismissed as an unrelated diagnostic display.
The next repair must make the specified numerical reduction reproducible,
while separating exact sums of represented phasors from a certified
transcendental resultant of the represented angles. Existing exact dyadic
reduction and rational phase bounds are reusable; no extra force, fitted
epsilon or longer trajectory follows from this result. This is a numerical
gate toward interpreting sustained coherent regions, not a replacement for
the underived state/history selection and maintenance mechanisms.

The [exact represented-component reducer](FORCED_SUPPORT_BALANCE.md#12-exact-reduction-of-represented-phase-components)
is now implemented through the existing dyadic sum owner. Its derived
resultant is permutation invariant, its common scaling avoids joint-zero
underflow/overflow, and exact cancellation supplies no angle. Independent
tests preserve minor-component underflow as an explicit defect. The caller inventory finds
distinct global, local, thresholded circular-mean and pressure fallbacks;
unifying arithmetic must not silently unify their different policies.
The [versioned global coordinator](FORCED_SUPPORT_BALANCE.md#versioned-global-coordination-integration)
now consumes the reducer when explicitly selected and restores graph-owned
state on failure. Its legacy default, local kernels and native runtime policy
remain unchanged. The controlled retained-phase-vector fixture gives identical
aligned proposals across three enumerations with fixed local inputs/gains;
this does not establish whole-runtime invariance or true-angle accuracy.
The [pinned source comparison](FORCED_SUPPORT_BALANCE.md#13-versioned-phase-correction-at-the-retained-regional-state)
now finds identical exact-version outputs under the frozen reversal and a
maximum fresh-pressure change of `2^-53` versus legacy, without any regional
rate-sign change. This closes the bounded numerical gate.

The [retained paired regional response](FORCED_SUPPORT_BALANCE.md#14-regional-recovery-versus-loss-of-form-in-the-retained-paired-window)
then returns directly to the primary question. Eight ancestry pairs reduce
raw perturbation error, but only four improve it relative to the control's
remaining contrast. The child cohort gains spatial error while its mean
offset relaxes; every control centered vector changes. Whole-network
attenuation therefore does not establish uniform regional recovery. The
[retained distortion ledger](FORCED_SUPPORT_BALANCE.md#15-child-cohort-distortion-and-regional-mean-to-shape-transfer)
now attributes 84.9442% of the interval's increase to Reception and 15.0558%
to integration. Held-pressure lag attenuates part of the increase. The same
map converts a child-parent mean contrast into within-child spatial error,
with essential cross terms. This complements the earlier closure obstruction
without identifying its partition or transfer direction with this one.

The [localized regional response](FORCED_SUPPORT_BALANCE.md#16-localized-regional-form-damage-and-finite-configured-restoration)
now passes its finite configured criterion: actual form damage is followed
by a 62.07099% error-energy reduction, with increasing control contrast.
Its exact ledger attributes the improvement to Reception and ordinary nodal
transport; it retains parent input, IL pressure corrections and control-form
drift. This is a stronger witness than mean relaxation or reduced error during
flattening, but supplied timing and source preparation still preclude an
autonomous-maintenance claim.

Reception has opposite regional-error signs in the two retained perturbations.
The [conditional response criterion](FORCED_SUPPORT_BALANCE.md#17-conditional-regional-response-and-environmental-input)
now explains both with exactly matched declared coefficients. Isolated
child-shape images attenuate in both directions; regional-mean and parent
input overwhelms the available margin only in the earlier witness. The
same map can turn zero regional shape into nonzero shape, so an unconditional
regional error bound cannot ignore the environment. These are exact map
statements with separately verified retained realization defects, not a
uniform future-runtime theorem.

This connects the generative hypothesis to a precise geometric question:
which shape directions does the derived input map reach, and which read-outs,
if any, are protected from it? The next bounded task uses its existing
nullspace images and full H metric. It does not repeat the family-closure
calculation or require another trajectory. Here geometry means the structure
of nodal relations and their induced maps; no external spatial geometry is
inserted. The support in this calculation is already formed. Demonstrating
its emergence, its feedback on persistent identity and a correspondence to
measured particles are distinct remaining obligations. The execution plan
owns the single active queue.

The [environmental-input geometry](FORCED_SUPPORT_BALANCE.md#18-environmental-input-geometry-and-protected-read-outs)
now has exact rank seven, already from parent inputs alone, for both
Reception and the held-pressure step. Its H-centered protected-readout
space is zero-dimensional. This excludes absolute input protection for the
current lineage-defined child cohort, while preserving the earlier
conditional-recovery result. The [exact map-symmetry audit](FORCED_SUPPORT_BALANCE.md#19-support-symmetry-versus-the-admitted-nodal-and-reset-maps)
now finds eight weighted-support symmetries, all respected by A, but only
the identity for sequential Reception S and T=S-hA. Each local EN row family
is covariant under all eight; its fixed ordered composition breaks that
symmetry. Thus the common matrix group gives no environmental restriction.
The captured canonical source and generation EPI/phase are separately
asymmetric. This does not refute conditional recovery or identify order as
the only obstruction to autonomous maintenance.

The [same-snapshot comparison](FORCED_SUPPORT_BALANCE.md#20-same-snapshot-reception-and-the-limit-of-geometric-protection)
is now complete with unchanged nodal coefficients and geometry. J and J-hA
respect all eight support symmetries and remove the regional-mean-to-shape
leak. Orbit-invariant environmental inputs have a rank-one centered image,
leaving six conditional protected directions. Both actual paired environments
violate this input restriction; unrestricted rank remains seven. No source
or state is averaged to force symmetry. Missing full stage state/history
prevents an authenticated historical runtime reconstruction.

This result closes the geometric-input and update-order detour. It supplies
a conditional mechanism and its explicit failed premise, not autonomous
pattern selection or maintenance. Further symmetry/rank sweeps would not
resolve that missing dynamics. Return to a closed endogenous-feedback
question using existing pressure, phase, capacity and state/history owners;
the execution plan defines the next bounded task. Exact symmetry is not
assumed necessary for a persistent NFR.

The [relaxed-source classification](FORCED_SUPPORT_BALANCE.md#21-closing-the-relaxed-phase-capacity-source)
now proves that common fixed fields of attractive phase relaxation inside a
shared semicircle, positive capacity adaptation and zero-topology-channel
pressure have uniform EPI and capacity. Fresh Si strengthens the result:
capacity maxima eventually enter the existing adaptation gate, so merely
leaving some lower-capacity gates inactive cannot maintain a heterogeneous
zero-pressure equilibrium. This is an exact-real equilibrium result, not a
convergence theorem or a proof about arbitrary composite runtime cycles.

Si is a diagnostic, not a fundamental nodal mechanism. The gate proof above
concerns the implemented controller that reads it, together with the specified
phase and capacity laws. It must not be promoted to a general TNFR obstruction
or used to reject held-capacity profiles under a different justified closure.
The proposed functional source-regeneration test is therefore secondary and
parked. Primary work requires structural evolution relations with their
derivations and remaining hypotheses explicit, independently of telemetry-based
control. Existing elimination-derived memory adds no new sustaining drive,
and configured REMESH/target controllers must not silently replace the missing
closure. The full triad matters: regular winding can retain phase structure
even when EPI is uniform. The execution plan retains the bounded controller
comparison as an optional specification and excludes another diagnostic audit,
passive-memory or symmetry campaign.

The [source-tangency result](FORCED_SUPPORT_BALANCE.md#22-source-tangency-without-a-telemetry-controller)
now turns this gap into an explicit equation: at zero pressure on fixed support,
`(w_phi/pi)*(R-I)*theta_dot-v*L_U*nu_dot=0`. The existing circular-mean response
provides R; no new telemetry controller is needed. An instantaneous equality
is not indefinite maintenance, and the equality does not select the two rates.
It permits heterogeneous initial capacity while constraining its changes.
The [capacity/phase gate](FORCED_SUPPORT_BALANCE.md#23-capacity-exposure-does-not-determine-a-phase-clock)
is now closed with an independence result: structural capacity does not by
itself determine phase speed. The original TNFR source distinguishes
reorganization capacity from periodic rhythm; the implemented oscillator
proposal makes an additional constitutive identification. Two analytic
completions of the same initial triad give different relative phase and EPI
futures. Accumulated capacity is derived, but is not a circular clock.
The [held-source geometry](FORCED_SUPPORT_BALANCE.md#24-rigidity-and-flexibility-of-a-held-phase-source)
now has both an analytic rigidity theorem and an exact flexible family.
Nonnegative irreducible mean response permits only common rotation locally
and along regular constant-source paths. Connected support alone is weaker:
the cube's antipodal-pair cancellation preserves g=0 while relative phase
changes. The resulting differentiated EPI still relies on held heterogeneous
capacity, and its nontrivial phase deformation violates the all-edge pi/2
gate. It demonstrates geometric freedom, not formation of a cube or an NFR.
Strict all-edge separation below pi/2 restores rigidity when g=0. The
[grammar audit](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#8-grammar-derivation-premises-language-and-trajectories)
now refutes the analogous full-rank implication at nonzero source: an exact
double-star inside strict U3 has an extra tangent, obstructed at second order.
Its local finite level set still allows only common rotation. An independent
exact U5 quotient also refutes a universal parent-versus-average-child
coherence inequality. These results require separate word, geometric,
coarse-dynamics and trajectory claims. The
[joint grammar refactor](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#13-structural-grammar-refactor-and-the-full-nodal-system)
now exposes rule premises, legacy preferences and exact finite execution
evidence separately, reusing the shared validators, executor and gain owner.
It supplies an explicit no-substitution rejection mode, not an autonomous
operator selector. All four tetrad fields participate in the dependency map;
none is promoted from a diagnostic definition into an unexplained force.
The sole execution plan prioritizes the joint constitutive-closure review
(G3) before further dynamics: identify what independently determines phase,
capacity, support and relevant history, or establish the missing implication
with a counterexample. Differentiating EPI alone cannot close that system.
Finite strict-U3 geometry is a bounded supporting question within this review.
The [first constitutive audit](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#14-constitutive-closure-audit-from-the-nodal-law)
now supplies a capacity-independence witness even under dissipation, normalized
conductance scale freedom, a moving-geometry energy balance and necessary
reciprocal couplings for a specified variational completion. None chooses an
autonomous law. The user has prioritized a return to physical/mathematical
bases within G3: distinguish EPI form, its observation/chart, directed pressure
and structural time before treating an implementation as a foundational axiom.
[FUNDAMENTAL_THEORY](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation)
now owns that review, including the source's dimensional conflicts and the
test for a closed reduced state. Existing source/quotient results constrain
new derivations; they do not forbid a better justified representation.
Compatibility, a chosen dynamics and autonomous formation remain separate;
no source-response run is launched.
The [diagnostic scope note](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#7-derived-observables-and-dynamical-closure)
owns the Si distinction: a derived observable can appear in a justified law,
but its definition alone does not derive that law. Existing controllers retain
their engineering scope; a blanket ban on all derived observables would also
exclude legitimate structural read-outs without a mathematical reason.

**Polyhedral geometry as a candidate comparison.** The user's proposed
Platonic-solid connection belongs to this same geometry/identity question.
Here a form's function means a specified nodal response or persistent read-out;
an observed association of form and function is motivation, not a derivation
of a preferred geometry or an intrinsic purpose assigned to that form.
Reuse the existing [symmetry sectors](../src/tnfr/physics/symmetry_sectors.py),
[structural morphisms](../src/tnfr/physics/structural_morphism.py) and
[inverse spectrum/symmetry examples](../benchmarks/inverse_spectrum_to_symmetry.py).
The latter explicitly constructs icosahedral/dodecahedral graphs; the
[simplex example](../benchmarks/emergent_simplex_dimension.py) selects complete
graphs, and the [nested tower](../benchmarks/emergent_resonant_pattern_tower.py)
selects a tetrahedral/Sierpinski construction. Their supplied geometry and
spectral read-outs do not demonstrate native formation of that geometry.
Legacy language equating those constructions with emergent physical space,
particle families or universal persistence is not evidence for this route.

Distinguish a symmetry-enforced eigenvalue multiplicity from a regional
read-out insensitive to environmental input. They are different mathematical
claims. Graph symmetry must also survive the actual capacity, source and
operator-map realization before it constrains the dynamics. In particular,
the known sequential-EN equivariance obstruction below prevents transferring
a Laplacian symmetry to an ordered reset without checking commutation.
The existing [NetworkX graph generators](https://networkx.org/documentation/stable/reference/generators.html)
already provide the Platonic graph constructors, so no parallel topology
generator is needed. Platonic graphs are candidate supplied controls, not preferred TNFR forms
by assumption. A generative claim would additionally have to derive their
formation and persistence from admissible nodal evolution, with no shape
template inserted as the purported conclusion. No polyhedral campaign is
started here; the execution plan retains one primary task.

#### What the existing activation evidence already decides

The nodal derivative is defined at zero EPI. On a two-node pure-EPI support,
`x=(0,1)` and positive capacity give positive first-node evolution without an
AL event. Existing state/support are premises; this is not creation of the
substrate. [Physics derivation](../src/tnfr/config/physics_derivation.py)
already separates nodal activation from initialization/grammar contracts.

The [grammar tests](../tests/operators/test_grammar_dynamics.py) admit IL, RA
and SHA in one context. Admission therefore does not select a unique next
operator. The fallback priority is a separate implemented policy. Likewise,
[contract signature tests](../tests/operators/test_operator_contracts.py)
retain a Silence/Contraction ambiguity for four instantaneous features.
These witnesses refute uniqueness from those specified inputs; they do not
prove that every possible extension using nodal history is impossible.

[Validated-sequence tests](../tests/operators/test_validated_sequence_execution.py)
already distinguish prior IL, recent destabilization, phase admissibility and
future handler obligations. Such obligations restrict a requested word;
they do not derive every target, timing and reduction order. The finite-action
symmetry owner supplies another restriction: a deterministic equivariant
selection must be fixed by the state stabilizer. Multiple fixed candidates
still leave a choice. Existing stage reductions remain declared policy.

The unresolved constitutive premise is therefore explicit: a state/history
relation determining when an admitted transformation actually occurs, on
which targets and with what schedule, or a proof that the remaining choices
give the same relevant outcome. A caller-supplied choice can define an
autonomous augmented algorithm when its counters are included; that does not
derive its choice law from the nodal equation. No priority reassignment or
threshold fit closes this gap. Reuse these obstructions and continue with a
scoped regional mechanism rather than treating selection as an endless
search for one preferred word.

#### Legacy claims excluded from the present evidence chain

Two concrete implementation/documentation contradictions were found during
the review. Their legacy helpers remain outside the regional balance; these
findings are not claims that the helpers were repaired in this delivery.

- [Operator algebra](../src/tnfr/operators/algebra.py) describes SHA as
  idempotent/freezing. Its `validate_idempotence` compares different words
  containing one SHA each, not `SHA(SHA(X))` against `SHA(X)`. The actual
  default unclipped capacity update multiplies by
  `r=0.9204225284540524`; positive capacity gives `r^2*nu != r*nu`.
  Its NUL dimension-reduction wording and the similar
  [grammar-canon wording](../src/tnfr/operators/grammar_canon.py) also do not
  describe the current capacity/pressure kernel and its optional edge-aware
  EPI write. Use current contracts and kernels, not those algebraic labels,
  when classifying freezing or restoration.
- [Word equivariance](../src/tnfr/physics/word_equivariance.py) labels an
  ordered node loop an equivariant sweep. On two nodes, sequential EN with
  mix `1/2` has EPI matrix `S=[[1/2,1/2],[1/4,3/4]]`. Node swapping gives
  `P*S*P=[[3/4,1/4],[1/2,1/2]]`, different from `S`. Uniform-input invariance
  is weaker than equivariance of the complete map. Transport the target
  order too, or establish commutation/simultaneity. The abstract composition
  result remains conditional on genuinely equivariant factor maps.

Several older modules have useful readouts but different premises:

- [Pattern constructors](../src/tnfr/physics/patterns.py) initialize waves,
  vortices and bumps. [NFR location](../benchmarks/emergent_nfr_where.py),
  [geometry](../benchmarks/emergent_nfr_geometry.py) and the
  [resonant tower](../benchmarks/emergent_resonant_pattern_tower.py)
  inspect supplied spectral or geometric constructions. A standing-wave
  zero, selected spectrum or constructed winding is not a demonstrated
  trajectory forming a persistent NFR. `Network.nfr()` is a whole-network
  diagnostic, not a derived regional segmentation law.
- [Cell](../src/tnfr/physics/cell.py), [life](../src/tnfr/physics/life.py),
  [interaction words](../src/tnfr/physics/interactions.py),
  [centralization](../src/tnfr/dynamics/emergent_centralization.py) and the
  [integration engine](../src/tnfr/dynamics/emergent_integration_engine.py)
  contain supplied boundaries, models, words or computational heuristics.
  Their names do not establish derived self-maintenance or interactions.
  U5 assessments likewise use a declared parent/child partition and scale;
  they cannot select a physical boundary from an arbitrary graph.
- [Arithmetic pulse](../src/tnfr/riemann/nodal_pulse.py) explicitly supplies
  logarithmic frequencies and a time-dependent phase law. Reuse its exact
  observability/Krylov techniques, not those arithmetic inputs as a hidden
  pressure source. Millennium notes remain scoped tool libraries: signed
  energy budgets and finite algebra are reusable; imported fluid laws,
  pure-gauge constructions and supplied complexes do not fill the missing
  generative law.

The EPI cut current also differs from the inward neighbor-mean field current
in [conservation diagnostics](../src/tnfr/physics/conservation.py), whose
density is `Phi_s+K_phi`. Mixing their orientation, normalization or density
would give the wrong regional balance. Keep one owner for each quantity.

#### Meaning of progress toward a persistent region

A useful regional claim must identify its nodes/lineage and the allowed
equivalence of its form before scoring a response. Uniform mean drift,
common phase rotation, changed capacity and changed membership cannot be
silently discarded. Tetrad equality is insufficient: equal diagnostics can
hide distinct rates. The runtime state must retain physical history,
configuration and lag/adaptation counters as well as glyph history.

The available tools can then ask whether a declared relational pattern
survives or returns under its actual source and boundary dynamics. Internal
variance decrease is only one term; source compatibility and the state
maintaining that source are separate obligations. A loss of compatibility
with a frozen profile may indicate an evolving pattern, but it does not
authorize selecting a better target after observing the outcome. Likewise,
the retained near-zero phase-resultant sensitivity must stay visible in any
claim whose identity depends on that relative phase pattern.

The review therefore favors the existing full-state regional mechanism over
a new reduction, longer blind continuation, independent C6 campaign or
search for particle analogies. The sole
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) owns the next bounded
identity/source test and its acceptance conditions.

## 3. Synergies worth testing next

The following combinations reuse existing mechanisms. Their proposed
extensions are research questions, not completed discoveries.

| Combined question | Reusable owners | Next discriminator and rejection condition |
| --- | --- | --- |
| Can a macro-node have a sufficient state? | [epi_memory.py](../src/tnfr/physics/epi_memory.py), [operator_quotient.py](../src/tnfr/physics/operator_quotient.py), [p5_reduction.py](../src/tnfr/physics/p5_reduction.py) | Prepare equal observed states with different hidden states. If futures differ, retain the derived memory/source or a sufficient extra coordinate; reject instantaneous closure. |
| What maintains a pattern rather than merely moving it? | [forced_support.py](../src/tnfr/physics/forced_support.py), [capacity_feedback.py](../src/tnfr/physics/capacity_feedback.py), [CHILD_COUPLING_FEEDBACK.md](CHILD_COUPLING_FEEDBACK.md) | Predict mean drift and shape response to one admitted intervention against the original target. Improvement only against a changed target does not establish restoration. |
| When does local phase stability survive composition? | [phase_response.py](../src/tnfr/physics/phase_response.py), [phase_quotient.py](../src/tnfr/physics/phase_quotient.py), [coupling_winding.py](../src/tnfr/physics/coupling_winding.py) | Carry chart, nonzero-resultant and U3/support margins through one complete word. A branch crossing or hidden-fiber rate difference blocks local-to-global promotion. |
| Does structural birth create useful feedback? | [THOL_PRESSURE_FEEDBACK.md](THOL_PRESSURE_FEEDBACK.md), [THOL_BIRTH_AND_TRANSPORT.md](THOL_BIRTH_AND_TRANSPORT.md), event/profile budgets | Separate birth, attachment, refreshed pressure and recovery. A disconnected child or overwritten pressure increment is not a self-maintaining structure. |
| Can history stabilize a sufficient reduced model? | [DERIVED_EPI_MEMORY.md](DERIVED_EPI_MEMORY.md), [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) | Test reduction/transition commutation and retain the mean and hidden source. Do not replace an eliminated-state memory kernel with a configured echo without proof. |
| Can observed transitions be predicted from nodal mechanisms? | [phase_transition.py](../src/tnfr/physics/phase_transition.py), [phase_scaling.py](../src/tnfr/physics/phase_scaling.py), phase and event budgets | Produce observations through one declared canonical protocol. Test alternative preparations, sweep directions, durations and sizes; threshold labels alone cannot certify a physical transition. |

### Emergence as sufficient coarse dynamics

The strongest existing example of something new appearing at another scale
is derived memory. For fixed pure-EPI evolution `xdot=-A*x`, a projection
`R`, lift `P` with `RP=I`, and `Q=I-PR`, exact elimination gives

`ydot=-RAP*y + integral_0^t K(t-s)*y(s) ds + f(t)`,

`K(t)=RA*exp(-QAQ*t)*QAP`,
`f(t)=-RA*exp(-QAQ*t)*Q*x0`.

The microscopic state can be instantaneous while its partial observation
has memory. No external delay law is needed. The initial hidden-state term
is indispensable in general. Existing four-/five-node path (P4/P5) controls separate exact
closure, finite-history error and invalid direct-echo substitution. A useful
next experiment is to observe only part of a controlled network and predict
the resulting memory from the admitted full model. Choosing a partition does
not itself derive how a physical macro-node forms or selects its boundary.

For the generated families, an optional future compression study could use
the existing weighted Gram identity and reversible contraction to bound
`||R*u(t)-v(t)||_Hbar^2 <= t^2*trace(K0)*||u0||_H^2`, where each current held
reference supplies `u=x-m_H(x)*1-z`, `u'=-A*u`, and the approximation is
`v'=-R*A*P*v`, `v(0)=R*u0`. Reconstructing raw EPI observations retains the
derived profile and mean drift. This is a continuous held-model upper bound,
not measured accuracy or a runtime Euler error certificate. It remains
parked until a concrete computational or observational constraint requires
discarding coordinates; it is not a gate for the primary mechanism study.

### Emergence as active restoration

On fixed connected symmetric support with positive fixed capacities, pure
EPI diffusion relaxes nonuniformity. Passive disagreement contraction cannot
alone explain a permanently differentiated active pattern. Existing held
capacity and phase profiles can support differentiation; freezing can retain
it; neither establishes autonomous restoration after a perturbation.

The next formation study should use one existing, fully declared canonical
policy and ask: what state supplies the restoring pressure, how is it
maintained, and does the same rule remain admitted after its own changes?
Measure original-target error, mean motion, capacity activity and consumed
pressure separately. If the policy is externally chosen, report controlled
pattern maintenance. Deriving endogenous selection is the further O1.b task.

### Three meanings of phase

The circular coordinate `phi`, a dynamical regime change, and a thermodynamic
phase transition are different concepts. Existing `LIFE`, `NON_LIFE` and
`CRITICAL` labels in `phase_transition.py` are operational classifications,
not findings about biological life or thermodynamic criticality. The
Landau and Stuart-Landau examples in [EMERGENT_ONTOLOGY.md](EMERGENT_ONTOLOGY.md)
declare auxiliary equations; their transitions cannot be imported as TNFR
operator theorems.

A nodal regime study should distinguish at least: loss of a U3/chart domain,
change of linearized stability, externally triggered support change,
capacity exhaustion/freezing, and numerical threshold crossing. Hysteresis
would require forward/reverse protocols and checks against transients and
sampling artifacts. A thermodynamic claim additionally requires a defined
family and limiting evidence. Do not insert an effective temperature or a
critical exponent to force a desired transition.

## 4. Findings in the existing empirical path

The detailed source audit is retained in
`artifacts/research/strategic_empirical_readiness_audit.md`.
These findings concern the pre-P1 interfaces, not the truth of every
externally reported measurement. They are retained as the rationale for the
implemented repair, not as a claim that the repaired failure paths still exist.

| Finding | Evidence | Required response before physical use |
| --- | --- | --- |
| Failed modal analysis becomes a favorable classification | `signal_confrontation.py`, lines 266-270: exception gives wave fraction zero, then diffusive-valid true | Return explicit unresolved/error status; a failed measurement must not become evidence |
| Prediction score is fitted and evaluated on the same window | `_nodal_skill_from_graph` and `nodal_prediction_skill`, lines 318-388 | Separate calibration and held-out runs; freeze graph, normalization and coefficients |
| Positive-capacity premise is not enforced | The fitted diffusion step is unconstrained | Reject/label inadmissible negative capacity; do not silently refit the physical interpretation |
| Modal classifier is stronger in name than in mathematics | AR(2) real/complex roots and an energy-majority rule | Report a statistical diagnostic; real roots may grow and complex roots need not conserve energy |
| Observed graph is not a full canonical state | `build_coupling_graph` uses PLV neighbors and an amplitude-deviation pressure proxy, without EPI/capacity or the U3 gate | Preserve proxy labels; use known wiring when available; derive signed pressure from admitted state |
| Empirical provenance is incomplete here | EEG/thermal summaries lack a pinned local replay envelope | Preserve numbers as reported claims; require dataset/run hashes, preprocessing, splits and uncertainty before reuse |
| Clock information can be lost | Grid-frequency benchmark discards timestamps/nonfinite entries before subsampling | Preserve times and missing-data masks for derivative/rate claims |

One limitation is algebraically decisive. With observed increments `r` and
the selected diffusion direction `d`, the unconstrained same-window fit is
`c=(r dot d)/(d dot d)`. When both norms are nonzero, its improvement over
persistence is

`skill=(r dot d)^2 / ((r dot r)*(d dot d)) >= 0`.

Thus positive training skill is built into this least-squares fit; it is not
independent evidence of predictive success. The fit remains useful as a
descriptive diagnostic. Separate-run prediction and admissibility checks
are required for falsification.

The [empirical record](../docs/EMPIRICAL_CONFRONTATION_EEG.md) is qualified
accordingly. The subsequent P1 delivery implements explicit modal abstention,
frozen nodal calibration/forecast/scoring, prospective-prefix descriptors,
timestamp-preserving ingestion and strict file-backed evidence admission.
See the [implementation contract](research/FIVE_STAGE_EXECUTION_PLAN.md#implemented-contract-and-migration).
Legacy training fits and observed PLV graphs remain descriptive. External classification
baselines and statistical error measures are evaluation tools, not TNFR
forces or constitutive laws.

### Repository-reuse audit of the organized plan

The subsequent source/test review found additional constraints below.
They are incorporated in the sole execution plan, not a second queue.
These are pre-implementation source-level findings; no empirical trajectory or
runtime reproduction was executed for that review. The first three rows now
have tested engineering repairs; the model-specific physical conditions remain.

| Finding and existing evidence | Consequence for reuse |
| --- | --- |
| [Multichannel preprocessing](../src/tnfr/validation/multichannel_interface.py) centers/transforms full records; [temporal_interface.py](../src/tnfr/validation/temporal_interface.py) computes features before masking pre-event windows and selects the best channel on the scored record | Whole-run splits alone do not prevent look-ahead. Freeze feature selection and enforce information available at forecast time; retain full-record results as retrospective descriptions |
| [EvidenceSidecar](../src/tnfr/research/evidence_sidecar.py) has useful model/clock/observation/cost fields but is tied to arithmetic manifests, checks only nonempty artifact hashes and does not reject nonfinite nonnegative horizons | Adapt this owner to core graph/measurement semantics and verify artifact bytes; do not create a competing evidence ledger or treat admission metadata as proof |
| [NumericalCertificate.validate_for_admission](../src/tnfr/research/certificates.py) does not check that the stored verdict equals its declared strict tolerance predicate | A manually constructed inconsistent verdict can pass metadata checks. Harden consistency before using the certificate as a gate; numerical context is not measurement uncertainty |
| [units.py](../src/tnfr/units.py) supplies a configurable bridge, while [physics/calibration.py](../src/tnfr/physics/calibration.py) supplies topology-dependent coherence expectations | Reuse conversion only with an independently justified finite scale. Neither the default bridge nor topology calibration measures a physical clock or sensor |
| [Observability](../src/tnfr/physics/observability.py) already checks declared linear sensors; [temporal identifiability](../src/tnfr/physics/temporal_identifiability.py) already bounds finite-signature separation | Reject inadequate proposed measurements before acquisition. Reuse the scoped theorems, with actual sensor/generator and uncertainty; full-state recovery is not required for every observable forecast |
| [Derived memory](../src/tnfr/physics/epi_memory.py) retains a hidden-initial-state source but only diagnostic propagator residuals; [P5 history bounds](../src/tnfr/physics/p5_memory_truncation.py) provide an exact restricted reference | Do not rederive completed kernels, omit the hidden source or promote general residuals into error bounds. The new contribution must name its observation question and accuracy domain |
| [Forced-support compatibility](../src/tnfr/physics/forced_support.py) and [child-coupling controls](CHILD_COUPLING_FEEDBACK.md) separate compatible targets from decreasing original-metric error; [perturbation tests](../tests/physics/test_structural_perturbation_response_runtime.py) retain a no-damage case | Reuse signed old-target budgets and matched branches; neither a new decreasing energy nor the retained no-damage control establishes recovery. Fixed-node telescopes do not directly cover birth |
| [Binary64 capacity-feedback tests](../tests/physics/test_binary64_capacity_feedback.py) retain nonzero fixed gaps; [phase scaling](../src/tnfr/physics/phase_scaling.py) consumes a balanced grid and [topology transitions](../src/tnfr/physics/topology_transitions.py) read graph geometry | Respect known resolution obstructions and diagnostic domains. Do not reopen exact-consensus counterexamples, equate birth with size replication or infer a fixed-geometry state transition from morphology |

The resulting work is a bounded extension of existing interfaces: prospective
evaluation, a measurement annex and one admitted test. Passive diffusion,
memory formulas, event execution and target comparison retain their current
owners. Extending to a new explanatory channel requires new admission and
reserved prediction evidence; the protocol can be reused, the earlier
physical conclusion cannot simply be transferred.

## 5. A feasible terrestrial empirical programme

### First data candidate: known wiring and repeated circuit measurements

[Zenodo record 3521009](https://zenodo.org/records/3521009) supplies electronic
oscillator networks, repeated recordings and a coupling sweep. The
[authors' data article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6961064/)
describes 28 circuits, 20 topologies, three repetitions and 30,000 samples
per trace. Only one of each oscillator's three state variables is recorded,
making hidden-state closure a substantive issue. The article's introductory
count of 100 coupling values differs from the methods' 101 indexed settings;
inspect actual archive contents before constructing a sweep.

This review downloaded metadata and only `Structure.zip` (9,991 bytes),
verified its published checksum, and read its edge lists without extraction.
The local inspection found 20 connected 28-node, 42-edge graphs with the same
labeled degree vector. Those are topology facts, not dynamical results.
The record declares CC-BY-4.0; the complete archive totals 20,611,525,626 bytes.
No time series was downloaded, fitted or admitted as TNFR evidence.

Reproducible acquisition/inspection:
`artifacts/research/inspect_zenodo_3521009.py`; retained metadata, archive and
inspection: `artifacts/research/zenodo_3521009_admission/`.
Sampling-clock verification, channel calibration, uncertainty and a TNFR
measurement map remain open. Start with a bounded subset only after these
gates, rather than acquiring the whole archive.

**Why this candidate is useful:** known support separates wiring from
functional correlation; topology changes at fixed degree can test whether
an admitted mechanism predicts more than degree alone. Repeated traces can
separate calibration from evaluation. A coupling sweep tests the normalized-
weight obstruction above. Partial observation connects directly to derived
memory. These are proposed uses, not findings from the signals. Replications
do not guarantee matched initial conditions or establish causal isolation
without checking the acquisition protocol. The circuits' Rössler equations
are not imported into the TNFR explanatory model.

### First controlled apparatus: transport before autonomous oscillation

A small resistor-capacitor network or thermal network is a candidate for
testing the already-derived EPI transport channel. Published
[experimental RC diffusion work](https://bigwww.epfl.ch/publications/sierociuk1501.pdf)
supports the practicality of a circuit diffusion apparatus. Its fractional
or electrical model is not a TNFR derivation and is not added to the engine.
Apparatus selection still needs an inventory of available instruments and a
measurement-map specification. No apparatus was selected or purchased here.

Prepare two independent patterns on known support; calibrate observation
scales and the common clock on separate runs; then predict modal rate ratios,
weighted-mean balance and response to one changed connection without
refitting. The simple Laplacian rate ratios require independently admitted
common positive capacity; for heterogeneous capacity use the spectrum of
`diag(nu)*L_rw` with independently calibrated capacities. Include a
disconnected-support control and measured instrument
drift. A success would establish a restricted transport correspondence,
not uniquely establish TNFR over other diffusion descriptions.

Mechanical metronomes are a possible later phase experiment:
[a primary laboratory study](https://www.nature.com/articles/srep17008)
measured synchronization using spring-driven metronomes and optical
tracking. Their energy supply and moving base introduce state and inputs
that must be accounted for. A shared base is not automatically a nearest-
neighbor TNFR ring, and synchronization alone does not validate the nodal
equation. This makes them less suitable than passive transport as the first
full correspondence test.

All selected data must originate on Earth's surface with workstation or
ordinary research-laboratory resources. No astronomical, satellite,
accelerator-scale or mandatory supercomputer data is part of this plan.

### Measurement and rejection contract

Extend the existing [CoreExperimentManifest](../src/tnfr/research/core_manifests.py)
with a measurement annex, retaining its source/claim metadata. The annex
needs raw hashes and license; sensor identities and units; timestamps and
gaps; known support and weight meaning; observation map and calibration;
pressure/capacity/phase availability; run-level splits; interventions;
uncertainty; exclusion rules; and declared predictions.

Do not equate dominant signal frequency with structural mobility. Do not
create a phase channel for a nonoscillatory measurement merely to fill a
tetrad field. Mark unavailable quantities explicitly. Per-window independent
normalizations can alter physical balance and may use test information;
freeze admissible measurement transformations before evaluation.

Reserve whole runs, preparations or topologies; overlapping windows must
not cross calibration/evaluation boundaries. Predict increments and finite
trajectories through existing nodal owners. Compare persistence, calibrated
AR-1 and the corresponding restricted graph model, keeping all as explicit
comparators. Wrong-edge and disconnected-edge controls must test the claimed
structural dependence, rather than serving only as easier competitors.

Reject the proposed correspondence when held-out observations systematically
contradict its predeclared uncertainty/solver bounds, an intervention has the
wrong response, or an admissible positive-capacity map cannot be maintained.
Report insufficient sampling or missing state as inconclusive. Thresholds
must be justified by instrument/calibration evidence before inspecting the
evaluation outcome. Finite agreement establishes only the tested bridge.

## 6. Execution and portfolio ownership

The [nodal execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md)
now prioritizes generative coherence research and preserves the five requested
measurement outcomes as its supporting bridge. Physical partial-observation
and recovery studies still follow an admitted prediction. It contains the only active
status board, numbered tasks, dependencies, finite budgets and acceptance
conditions. Consult it for the next executable action; do not reconstruct
another queue from the prospective questions in this review.

The [portfolio](../TNFR_lineas_de_investigacion.txt) separates the primary
generative axis, supporting theory/measurement and deferred domain campaigns. C6 is
preserved at B75 as supporting work, with no campaign running by default.
The exact previous roadmap is kept in the
[historical archive](research/archive/README.md). Mathematical derivations
remain in their existing owners rather than being copied into planning files.

The macro target is a common nodal description that predicts formation,
persistence, reorganization and loss of patterns across independently tested
conditions. The next scoped criterion is a closed generation mechanism or
an explicit obstruction for one declared family, with physical predictions
requiring their own admitted bridge. Neither C6 completion, a finite generated
pattern nor one successful apparatus proves universal physical emergence.

## 7. Review evidence and reproducibility boundary

Source-level subaudits are retained locally under `artifacts/research/`:
`strategic_nodal_emergence_audit.md`, `strategic_c6_transfer_audit.md` and
`strategic_empirical_readiness_audit.md`. These local ignored artifacts are
not guaranteed to exist in a fresh clone. The present tracked theory note
and roadmap retain the substantive findings and references.

The initial strategic review performed metadata/topology acquisition and
documentation verification only. The later plan-reuse audit inspected
source/tests and revised documentation without further acquisition.
Neither executed a new C6 optimizer, exact guard scan,
physical time-series fit, laboratory run or runtime dynamics change.
The new planning priority does not rewrite historical B75 evidence or its
source/document hashes. The later P1 delivery includes targeted contract and
integration tests with a fresh checkpoint at
`artifacts/research/p1_validation_2026_09_17_final.json`. Its synthetic forecast and
evidence round trip establish software behavior, not physical correspondence.
The [P2 annex](research/PASSIVE_TRANSPORT_PROTOCOL.md) records the additional
metadata-only terrestrial search and proposed passive two-node experiment.
Neither inspected candidate is admitted; actual state/clock calibration and
uncertainty evidence are still missing. No physical time series was evaluated.

The subsequent observation/transport integration audit passed 601 targeted
tests across history/THOL, propagation, birth/transport integration, exact
support accounting and SDK diagnostics. Changed Python files pass flake8;
179 local documentation links/anchors pass the focused check. Its scoped
working-source digest and per-run file lists are retained locally in
`artifacts/research/observation_transport_audit_validation_2026_09_18.json`.
This is software regression and finite mathematical-control evidence, not a
new physical experiment, autonomous-generation result or whole-repository
verification. The preceding audit artifacts retain their original hashes.

The later regional reuse review adds one detached shared observer and one
offline retained-state audit, documented in
[forced-support balance, sections 7-8](FORCED_SUPPORT_BALANCE.md#7-a-region-and-its-environment-on-the-same-nodal-support).
Its 33 observer tests and 19 benchmark tests are included in a 313-test
targeted validation with the reused support/forcing/target and
grammar/contract/symmetry owners. Changed Python files pass flake8. The
local checkpoint is
`artifacts/research/regional_reuse_validation_2026_09_18.json`.
This delivery adds no engine evolution path, new trajectory, fitted
parameter or physical dataset. Its one-snapshot regional rates do not
certify temporal identity, active maintenance or physical emergence.

The temporal regional follow-up adds the shared finite Euler observer and
one offline reader, with 24 observer tests and 34 reader tests. A combined
156-test run passes, including the reused instantaneous regional and
support/forcing tests; four changed Python files pass flake8. The checkpoint
is `artifacts/research/regional_identity_validation_2026_09_18.json`.
Independent exact recounting uses the original primitive records, without
the new observer. No engine evolution or historical artifact is changed.

The subsequent phase-source relevance study passes 151 targeted tests,
including 43 new portable tests, and an independent 175-check exact recount.
Its two fresh forcing captures are detached readings of the pinned endpoint;
no phase coordination or native trajectory runs. Both new Python files pass
flake8. The scoped checkpoint is
`artifacts/research/phase_source_relevance_validation_2026_09_18.json`.

The represented-phasor owner adds 52 independent tests. Its combined
225-test validation includes existing midpoint, circular/numerical and stable
pressure controls; both new Python files pass flake8. The scoped record is
`artifacts/research/phasor_resultant_validation_2026_09_18.json`.
That standalone delivery changed no caller. The subsequent opt-in global
integration adds 24 independent tests and passes 175 combined cases, including
the existing graph-transaction owner. The source, matching type stub and
tests pass flake8. Its checkpoint is
`artifacts/research/exact_phase_coordination_validation_2026_09_18.json`.
The retained primitive phase vector is used only in a controlled synthetic
fixture; no historical trajectory is replayed or extended, and no fresh
pressure is captured in this integration delivery.
