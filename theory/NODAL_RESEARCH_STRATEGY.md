# Nodal research strategy: closure, emergence and laboratory tests

Review date: 2026-09-17. Scope: source, theory, retained research evidence,
empirical interfaces and selected primary experimental sources. This is a
strategic audit, not an exhaustive software verification or a new stability
theorem. The scientific source remains the B75 snapshot
`sha256:f84290315bc1cee1e01f6c71ee1d086a7b56a68d5673fa02c260e04963ea5b92`.
Historical tests and certificates were inspected, not rerun. The
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

The next major deliverable should be **one independently calibrated,
falsifiable nodal prediction**, coupled to a small study of sufficient state
and emergent memory. C6 indefinite boundedness remains a legitimate proof
question, but is no longer a prerequisite for this work. Completing its
binary64 first-exit proof would not determine the measurement map, explain
autonomous operator selection or establish physical emergence.

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

The plan is justified by reuse, identifiable predictions, finite cost and
ability to detect a failed hypothesis. It is not a proven global optimum:
candidate apparatus, costs and scientific utility have not been exhaustively
specified or compared. Selection remains conditional on P2 admission.

| Alternative next focus | Useful contribution | Reason for its present position |
| --- | --- | --- |
| Extend the C6 search | A restricted binary64 first-exit proof | Does not establish measurement semantics; existing unresolved proof search is not a prerequisite for prediction |
| Start multichannel autonomous maintenance | Directly addresses persistence | Adds channel, policy and observation obligations before the simpler measurement bridge is established |
| Fit the available oscillator/EEG signals | Descriptive patterns and possible candidate rejection | Availability does not resolve hidden state, clock or same-record fitting; no signal is admitted merely because it fits |
| Calibrated restricted transport, then observation/recovery extensions | Reuses exact dynamics, observability, memory and fixed-target accounting | Smallest currently identified route to a falsifiable measurement bridge; success establishes only that restricted correspondence |

Early read-only observability and apparatus checks may run alongside the
diagnostic repair. The single delivery queue is a resource/traceability
policy, not a mathematical requirement that all reasoning be serial.

## 2. What must actually follow from the nodal structure

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
These findings concern the current interfaces, not the truth of every
externally reported measurement.

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
accordingly. This strategic delivery does not repair the runtime APIs;
those fixes are the first implementation gate. External classification
baselines and statistical error measures are evaluation tools, not TNFR
forces or constitutive laws.

### Repository-reuse audit of the organized plan

The subsequent source/test review found additional constraints below.
They are incorporated in the sole execution plan, not a second queue.
These are source-level findings; no empirical trajectory or runtime
reproduction was executed for this review.

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

The [five-stage execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md)
replaces this review's earlier six-delivery proposal. It follows the user's
five requested outcomes, evaluating a reserved prediction before extending
the partial-observation and recovery studies. It contains the only active
status board, numbered tasks, dependencies, finite budgets and acceptance
conditions. Consult it for the next executable action; do not reconstruct
another queue from the prospective questions in this review.

The [portfolio](../TNFR_lineas_de_investigacion.txt) separates three primary
axes, their supporting mechanisms and deferred domain programmes. C6 is
preserved at B75 as supporting work, with no campaign running by default.
The exact previous roadmap is kept in the
[historical archive](research/archive/README.md). Mathematical derivations
remain in their existing owners rather than being copied into planning files.

The macro target is a common nodal description that predicts formation,
persistence, reorganization and loss of patterns across independently tested
conditions. The near-term success criterion is deliberately smaller: a
reproducible prediction of an unexamined observation, with a known reason it
could fail. Neither C6 completion nor one successful apparatus proves that
all physical reality emerges from TNFR.

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
source/document hashes. Future empirical API changes will require targeted
contract tests and a fresh source-bound checkpoint.
