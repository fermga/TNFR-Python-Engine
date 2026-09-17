# Five-stage nodal research execution plan

Planning and repository-reuse review: 2026-09-17. **Status: documented;
implementation has not started. Next executable task: P1.1.**
P1-P5 are delivery stages of one
programme, not new research lines. These stage labels are distinct from the
four- and five-node path graphs also named P4/P5 in existing mathematics. The
[portfolio](../../TNFR_lineas_de_investigacion.txt) identifies primary,
supporting and deferred work. The
[strategic review](../NODAL_RESEARCH_STRATEGY.md) owns the scientific rationale.
This file alone owns the active task order, stage status and acceptance gates.

## 1. Target, boundaries and completion

Deliver a reproducible route from canonical nodal dynamics to independently
specified measurements, held-out predictions, sufficient partial observation,
and controlled studies of pattern maintenance, recovery and regime changes.
All explanatory dynamics must trace to `dEPI/dt=nu_f*DeltaNFR`, admitted
structural state and derived canonical transformations. Do not prescribe
pressure retrospectively from the measured derivative.

Separate a derived law, a declared constitutive choice, preparation/input,
numerical settings and measurement calibration. Calibration of an existing
state or observation map is not derivation of a new physical law. External
models and statistical diagnostics may be controls, never hidden additions
to TNFR pressure. All observations must be feasible on Earth's surface with
a workstation or ordinary research laboratory; astronomical, satellite and
large-facility data remain excluded.

Completing a **study** means producing its reproducible decision and evidence.
It does not guarantee confirming its hypothesis. Keep delivery state
`planned / executing / complete / blocked` separate from scientific outcome
`supported_in_scope / rejected_in_scope / inconclusive / not_tested`.
An unresolved mandatory gate stays blocked; it is not marked complete merely
because its budget ended. A completed rejection closes that particular test,
while its positive scientific objective remains unmet. These proposed workflow
labels do not replace the existing mathematical `ClaimStatus` enum.

The five scientific objectives are not all fulfilled until their stated
positive acceptance conditions are met. No plan can promise those outcomes.
Full autonomous pattern maintenance, thermodynamic criticality and universal
physical emergence require stronger results than this finite programme.

## 2. One status board and dependency chain

| Stage | Existing research ownership | Current state | Required output |
| --- | --- | --- | --- |
| P1. Correct diagnostics and separate calibration/evaluation | O6.a with O1.a; S3/S15 | Planned; first task P1.1 | Reliable instrument semantics and independent evaluation API |
| P2. Define EPI, capacity, phase and time measurements | O6.a/O1.a; S3/S14/S15 | Planned; consumes P1 | One admitted observation/model contract and bounded protocol |
| P3. Predict a reserved response without refitting | O6.a with S1/S4/S15/S16 | Planned; requires P1-P2 | Frozen prediction, held-out decision and reproducible evidence |
| P4. Study partial observation and derived memory | O1.a/O4.a; S3/S8/S9 | Planned; consumes P3's classified result | Sufficient-state or explicit closure-obstruction study |
| P5. Study maintenance, recovery and transitions | O3.a/O4.c; S1/S2/S6/S7/S16 | Planned; consumes admitted state and prediction evidence | Fixed-target recovery study, then a bounded regime study |

Default execution is P1 -> P2 -> P3 -> P4 -> P5. One delivery is active at a
time; supporting modules are reused inside it. Read-only preparation may
overlap, but must not start another campaign or consume reserved outcomes.
In particular, P2's model-specific observability/hidden-source veto and
apparatus/license inventory can be prepared during P1. State sufficiency
needed for the first prediction is a P2 gate; P4 studies a further reduction
or observation question, not a prerequisite postponed until after scoring.

If P3 rejects the laboratory mapping, its physical P4/P5 branch stops.
P4 may still close an explicitly internal mathematical question using the
existing exact model, with no physical promotion. A revised mapping returns
to P2, receives a new protocol version and fresh held-out data; the old test
set becomes exploratory. Inconclusive P3 evidence requires its missing-data
or uncertainty gate to be resolved, not relabeled as support. This keeps a
negative experiment useful without creating an endless fitting loop.

## 3. Shared evidence contract and resource discipline

Compose a measurement/protocol annex with the existing
[CoreExperimentManifest](../../src/tnfr/research/core_manifests.py), adapting
the [evidence envelope](../../src/tnfr/research/evidence_sidecar.py) rather
than creating another ledger. Its current manifest/provenance fields are
arithmetic-specific; do not satisfy them with invented factorization data.
Reuse [claim tracking](../../src/tnfr/research/claims.py) and
[readout semantics](../../src/tnfr/metrics/observations.py) without treating
a status label or metadata envelope as a verified measurement. The annex
must retain:

| Contract element | Required content |
| --- | --- |
| Question and scope | Existing O/S owner, falsifiable claim, model hypotheses and excluded interpretations |
| Raw observations | Source/license, immutable content hashes, apparatus, channel identities, units, timestamps, gaps and repeat IDs |
| State map | Sensor-to-EPI coordinates; independently constrained capacity; phase definition or explicit unavailability; support/weight meaning |
| Evolution | Canonical pressure owner/channel configuration, phase/capacity policy, physical-time conversion, operator word and grammar context |
| Calibration | Run IDs, permissible fitted quantities, uncertainty, identifiability argument and frozen transform/model hash |
| Evaluation | Disjoint run IDs, initialization allowance, horizon, predicted quantities, interventions and controls |
| Decision | Numerical and measurement error budget, predeclared primary criterion, secondary diagnostics and abstention/rejection rules |
| Resources | Exact run grid, seeds, duration, solver limits, download/storage limits and stop condition |
| Result | Complete planned/failed/skipped run ledger, predictions saved before scoring, observed outcomes, scoped decision and source revision |

Save each version's manifest, predictions and result in one campaign bundle;
the manifest lists filenames and checksums. Future bundles must be replayable
from versioned code plus accessible raw data. Local ignored artifacts alone
are not a publication package. Keep historical evidence unchanged.

Source hashes cover declared nonignored source, not ignored raw data or the
chronology of a forecast. Bind raw data, calibration and predictions by
separate verified digests and retain the pre-evaluation record. Reuse strict
serialization and atomic writing from [utils/io.py](../../src/tnfr/utils/io.py).
Before trusting admission, check finite context values, certificate-verdict
consistency, digest-to-file agreement and frozen nested calibration content.
An exact certificate, a numerical residual and measurement uncertainty have
different meanings; none supplies the others. Arithmetic
[circularity checks](../../src/tnfr/research/circularity.py) need explicit
measurement questions about future samples, outcome-derived wiring, labels
and post-selection; their default answers are not a physical leakage audit.

Before each execution, pin numeric tolerances and budgets from the admitted
model, instrument calibration and resource inventory. A blank tolerance or
unset acquisition limit blocks the run. Resource caps are engineering
choices, not TNFR physical constants. No automatic increase in run count,
time horizon, solver budget or model complexity follows an inconclusive result.

Use the relevant existing test suites and adversarial controls after changes;
broaden only for changed shared callers or a concrete failure. No research
trajectory is rerun merely to refresh an unchanged historical test total.

### Reuse map: existing owner, boundary and necessary new work

This map allocates work within P1-P5; it is not an additional task queue.
The technical theorem inventory remains in
[CORE_RESEARCH_PROGRAM.md](../CORE_RESEARCH_PROGRAM.md).

| Existing owner | Reuse in this programme | Boundary / necessary delta |
| --- | --- | --- |
| [Signal confrontation](../../src/tnfr/validation/signal_confrontation.py), [multichannel](../../src/tnfr/validation/multichannel_interface.py), [temporal interface](../../src/tnfr/validation/temporal_interface.py) | P1 signal/readout pipeline and synthetic controls | Repair failure semantics, isolate fit/evaluate and information available at forecast time; observational PLV graphs are not canonical wiring certificates |
| [Core manifests](../../src/tnfr/research/core_manifests.py), [sidecars](../../src/tnfr/research/evidence_sidecar.py), [numerical certificates](../../src/tnfr/research/certificates.py), [observations](../../src/tnfr/metrics/observations.py) | P1-P5 provenance and scoped readout vocabulary | Adapt the envelope to core models and harden admission; add a small measurement annex, not another infrastructure stack |
| [Unit bridge](../../src/tnfr/units.py), [exact time](../../src/tnfr/_exact_time.py) | P1/P2 configured conversion and represented time checks | Supply independently justified finite bridge, timestamps and gap policy; default bridge one is not physical calibration |
| [Nonnormal prediction](../../src/tnfr/physics/nonnormal_prediction.py) | P1/P3 predeclared split identity, abstention and separate result reporting | A scoped synthetic benchmark; its even/odd fixture split is not a safe time-series split or a physical fit/evaluate API |
| [Observability](../../src/tnfr/physics/observability.py), [temporal identifiability](../../src/tnfr/physics/temporal_identifiability.py) | P2 early sensor/observable checks and optional finite-response discrimination; P4 reduction design | Bind the actual generator, observation map, conditioning and noise bound; finite signatures do not identify arbitrary unseen states |
| [Structural diffusion](../../src/tnfr/physics/structural_diffusion.py), [single-mode reference](../../src/tnfr/physics/reversible_eigenmode_reference.py), [integrators](../../src/tnfr/dynamics/integrators.py) | P2/P3 transport law, admissible exact reference and shared runtime path | Exact single-mode bounds do not cover mixed modes or changed channels; bind numerical error for the selected model |
| [Derived EPI memory](../../src/tnfr/physics/epi_memory.py), [P5 reduction](../../src/tnfr/physics/p5_reduction.py), [history bounds](../../src/tnfr/physics/p5_memory_truncation.py) | P2 hidden-source veto; P4 kernel, source and established positive/negative fixtures | Add one new observation question; general floating propagator residuals are not certified error bounds |
| [Forced support](../../src/tnfr/physics/forced_support.py), [capacity feedback](../../src/tnfr/physics/capacity_feedback.py), [perturbation comparison](../../benchmarks/structural_perturbation_response.py) | P5 original-target, damage/benefit and event accounting | Respect ordered support, old metric, restricted feedback domain and retained no-damage/binary64-floor controls |
| [Event runtime](../../src/tnfr/operators/event_runtime.py), [network stages](../../src/tnfr/operators/network_stage.py), [contracts](../../src/tnfr/operators/operator_contracts.py) | P3/P5 admitted named transformations and finite causal provenance | Reuse the executor; neither a supplied policy nor a finite receipt proves autonomous selection or future stability |
| [Phase scaling](../../src/tnfr/physics/phase_scaling.py), [topology transitions](../../src/tnfr/physics/topology_transitions.py) | P5.2 optional scoped regime readouts | Scaling needs balanced grids; hysteresis needs chronological sweeps; graph morphology cannot diagnose a fixed-geometry state transition |

[physics/calibration.py](../../src/tnfr/physics/calibration.py) calibrates
topology-dependent `T_C`/`xi_C` expectations, not sensor scales or physical
clocks. [dynamics/sampling.py](../../src/tnfr/dynamics/sampling.py) samples
nodes, not measurement timestamps. Neither is a substitute for P2 admission.
Use only the rows required by the first declared claim; a reusable owner is
not a reason to activate every mechanism or build a universal framework.

## 4. P1 — Correct diagnostics and isolate evaluation

**Entry:** retain the source audit and review current callers/tests. Real
measurements are unnecessary to close this engineering stage.

- **P1.1, abstention:** replace the exception-to-favorable-classification
  path with explicit failure/unresolved status and a reason. Audit callers
  and serialization so an unavailable result cannot become either physical
  regime through a Boolean fallback. Define migration behavior for the
  existing `diffusive_face_valid` field rather than silently breaking callers.
  Include `SignalConfrontation.summary`, public exports/serialization and
  `examples/10_applications/159_empirical_confrontation_pipeline.py`; an
  unresolved result encoded as false or missing must not fall through to
  the wave label.
- **P1.2, diagnostic scope:** distinguish modal root classification from
  stable relaxation, conservation and a nodal certificate. Handle growing,
  degenerate, nonfinite, constant and short signals explicitly. Synthetic
  AR/wave/noise fixtures remain diagnostic controls, not TNFR laws.
- **P1.3, fit versus evaluate:** retain legacy same-window skill only as a
  descriptive fit. Introduce a frozen calibration result and an evaluator
  that cannot update graph, normalization, coefficients or thresholds using
  held-out samples. Permit independently known graph support. Split whole
  runs/subjects/preparations, with no overlap from windows, filters or smoothing.
  Freeze channel selection too. Full-record centering, FFT/Hilbert transforms
  and centered differences may use future samples even within a separate
  evaluation run. Forecast initialization/features must use only information
  available at issue time, with declared warmup/latency. Retrospective
  descriptors remain separately labeled; masking pre-event windows after a
  full-record transform does not make an early-warning result prospective.
- **P1.4, admissibility:** require positive capacity/time for the selected
  positive-capacity diffusion claim. Negative fitted capacity is an explicit
  model-domain failure; zero is a separately declared inactive boundary.
  Never clip or refit an inadmissible value silently. Preserve sample gaps
  and durations. Audit any Euler-step stability premise separately from
  the continuous model's positive capacity.
  Before acquisition reuse, preserve timestamps through parsing/subsampling,
  validate the clock bridge as finite and positive, and apply content/hash
  and size checks to cached files and expanded archive members as well as
  downloads. A finite time scalar alone does not certify sampling regularity.
- **P1.5, contract tests and documentation:** add meaningful failure and
  leakage controls, migrate affected consumers, and update the empirical
  record to match executable behavior. In the shared admission owners,
  reject inconsistent numerical verdicts, nonfinite horizons and unmatched
  artifact digests before using their results as gates. Freeze or detect
  nested calibration mutations; a frozen dataclass with mutable mappings
  is insufficient. Keep this a small extension of existing interfaces.

**Owners:** `src/tnfr/validation/signal_confrontation.py`,
`multichannel_interface.py`, `temporal_interface.py`, and the shared evidence
owners in the reuse map; inspect
`benchmarks/temporal_interface_benchmark.py` before reusing its time ingestion.
Relevant tests: `tests/test_signal_confrontation.py`,
`tests/test_multichannel_interface.py`, `tests/test_temporal_interface.py`,
`tests/research/test_research_infrastructure.py`,
`tests/test_structural_observation_metadata.py` and affected unit/ingestion
tests. Existing observability/identifiability tests supply P2 controls.

**Acceptance:** forced diagnostic errors never yield a favorable result;
perturbing evaluation data cannot alter frozen calibration; reserved
observations cannot leak through preprocessing; changing a suffix after a
forecast origin leaves its already issued forecast/features unchanged;
inadmissible capacity and insufficient data stay distinguishable from
rejection; relevant caller and
serialization tests pass. Output: implemented instrument contract, tests,
updated docs and a source-bound validation record. Empirical status remains
`not_tested`.

## 5. P2 — Admit one measurement and dynamical model

**P2.1, specify the first prediction before choosing a convenient fit.**
Prefer the restricted passive EPI transport question: predict relaxation or
the effect of one independently known connection change. Phase/capacity
feedback is not presumed absent in physical data; it must be excluded or
accounted for by the admitted measurement/model domain.

A commanded apparatus intervention is independently recorded input, not
automatically a canonical glyph. If modeled as a glyph, bind its actual
contract, including U3 where required. Otherwise compare separately admitted
fixed-support preparations or derive the explicit input path; do not assign
nodal state ad hoc to manufacture the desired connection response.

**P2.2, complete the measurement table.**

| Quantity | Required independent specification | Failure that blocks admission |
| --- | --- | --- |
| EPI | Fixed sensor coordinate, scale/offset calibration and uncertainty | Per-window redefinition selected to improve the result |
| Capacity `nu_f` | Identifiable positive reorganization rate under the declared pressure and clock | Only the product of unknown pressure and capacity is observed |
| Pressure | Computed from admitted nodal state and canonical channel implementation | Defined as the observed derivative divided by fitted capacity |
| Phase | Declared observable/reference, sampling adequacy and U3 domain, or inactive/unavailable channel | Oscillation phase assumed for a nonoscillatory signal or PLV treated as a U3 certificate |
| Time | Recorded timestamps, physical-to-structural conversion and gap handling | Instrument Hz silently equated to Hz_str or gaps removed from a derivative |
| Support/weights | Known wiring or independently calibrated structure and weight meaning | Outcome-window correlation used as independent causal wiring |

Uniform positive scaling of all conductances leaves exact `L_rw` unchanged;
it cannot alone model a coupling-strength sweep at fixed capacity. Simple
Laplacian modal-rate ratios require common positive capacity; otherwise use
the independently specified generator `diag(nu_f)*L_rw`. Freeze these
assumptions before defining the evaluation set.

**Pre-acquisition sufficiency gate:** apply existing linear-observability,
channel-ablation and hidden-source results to the proposed generator and
sensor map. Require enough initial information, or a justified bound, for
the declared observable forecast; full-state reconstruction is unnecessary
when that narrower forecast closes. Numerical rank alone is insufficient
without conditioning and measurement resolution. If distinguishing finite
response prototypes, reuse the temporal-signature separation/noise margin
with independently fixed feature scales. A finite-prototype result is not
general capacity/pressure inversion. Resolve this minimum check in P2;
do not acquire data first and leave a known obstruction for P4.

**P2.3, select one eligible acquisition route.** A modest passive circuit or
thermal network is a candidate, pending apparatus and calibration inventory.
The already-inspected Zenodo 3521009 topology archive is a candidate resource,
not an admitted passive diffusion dataset. Its oscillators have hidden
variables and supplied drive; one recorded channel does not establish full
state sufficiency. Use it only for a question whose map can be justified.
If it fails that gate, record the reason and choose the simpler apparatus
without importing the circuits' governing equations into TNFR.

**P2.4, reserve evidence and register predictions.** Specify independent
calibration and evaluation runs plus a known intervention if that is the
claim. Previously inspected signals cannot be newly called blind. Before
download, specify exact archive files, compressed and expanded byte limits;
do not fetch the entire 20.6 GB collection. First admission concerns one
apparatus/topology and one claim, not a survey of every dataset.

**Acceptance:** a reviewer can compute the prediction from frozen
calibration, preparation and admitted TNFR inputs without using evaluation
outcomes. Units, clock, identifiability, uncertainty, provenance and all
resource/decision fields are complete. No apparatus is selected merely by
analogy. Output: admitted model/measurement annex and preregistered P3
protocol, or explicit non-admission with its missing dependency.
Use deterministic error enclosures where justified; for stochastic
measurement uncertainty, declare coverage assumptions, run-level inference
and any multiple-endpoint correction separately. Pointwise intervals do not
automatically provide joint trajectory coverage. Statistical observation
assumptions are not added forces in TNFR dynamics.

## 6. P3 — Make and evaluate a reserved prediction

**P3.1:** save content-hashed predictions before opening the reserved response.
Choose one primary endpoint (relaxation trajectory/rate or edge-change
response), a horizon and an error criterion justified in P2. Available
initialization samples must be identified separately; later test states may
not refresh an open-loop forecast. Report any one-step, measurement-fed
score separately from a multi-step prediction.

**P3.2:** execute the admitted canonical integrator/channel or the exact
reference within its proved domain. Record pressure refresh, numerical
defects, mean and shape. Use existing `structural_diffusion.py`,
`forced_support.py`, `forcing_realization.py` and relevant event owners;
do not create a second nodal solver in a benchmark.

**P3.3:** evaluate all reserved runs with unchanged settings. Use equally
informed persistence, AR-1 and the corresponding restricted graph-model
controls where applicable; they are comparisons, not explanatory ingredients.
Include one wrong-support or disconnected-support control that tests the
specific structural claim when one is asserted. For a model-equivalence
test, specify that narrower claim rather than imposing unique-mechanism
discrimination. A common diffusion limit may produce identical predictions
to its classical comparator; report equivalence honestly.

**Initial feasibility-pilot cap:** one apparatus/topology, one preparation
contrast, one intervention when claimed, and at most three independent repeats per
evaluation condition. P2 must establish whether this cap can resolve the
proposed effect; otherwise the protocol is not admitted. Repeated samples
inside one trajectory are not independent experimental repetitions.
Three repeats are a resource policy, not a universal confirmation standard.
A larger confirmatory design needs an explicitly revised budget and sample
size rationale before evaluation is opened; an inconclusive result does not
authorize automatic extra repeats or reuse of its outcomes as blind data.

**Acceptance:** deterministic defects meet their declared bounds, and the
measurement result meets the separately preregistered decision/coverage
criterion. The intervention has the predicted response where claimed; use
controls to discriminate a mechanism only when that is the assertion.
An unbracketed effect, insufficient power or missing state is inconclusive;
a systematic contradiction rejects that mapping/model claim. No favorable
training score substitutes for this gate. Output: frozen forecast bundle,
run-level results and a separate statement of physical scope.

## 7. P4 — Partial observation and derived memory

**P4.1:** declare what is hidden and what is observed under the admitted
model. Freeze a projection `R`, lift `P` with `RP=I`, and the initial-state
information legitimately available to the predictor. If hidden state cannot
be known, derive bounds or a sufficient observable extension; never feed
the unobserved test state into a purported observable-only forecast.

**P4.2:** reuse, rather than reprove, the fixed pure-EPI result for
`xdot=-A*x`, `Q=I-PR`: the reduced model has kernel
`K(t)=RA*exp(-QAQ*t)*QAP` and initial source
`f(t)=-RA*exp(-QAQ*t)*Q*x0`. Preserve both. An imposed REMESH echo is not this
eliminated-state kernel. Label the positive fixed-capacity/support premises;
extension to nonlinear channels requires a new derivation.

**P4.3:** compare full-state projection, an instantaneous reduced model and
the derived memory/source model on reserved preparations. Start from the
existing four-/five-node path (P4/P5) positive/negative controls, then address one named new
observation or measurement question. Do not rerun the completed theorem and
call it new research. Select established fixtures from their retained
evidence/tests, rerunning only what a changed implementation requires.
Use at most six predetermined preparations: two pairs
with equal coarse states but differing hidden states, plus two controls,
with one fixed observation window per preparation and no partition search.

**Acceptance:** the proposed sufficient reduced state predicts the reserved
observed evolution within its stated bounds without refitting the kernel.
If equal observations have different futures, reject instantaneous closure;
that finding does not reject the full TNFR model. A formal memory formula
alone does not supply a usable predictor if its hidden source is unavailable.
Output: sufficient-state contract or explicit obstruction, error envelope
and comparison record. Physical claims require a supported P3 mapping;
otherwise retain an internal-only result and leave the physical gate open.

**Owners:** `epi_memory.py`, `p5_memory_truncation.py`, `p5_reduction.py`,
`operator_quotient.py`, `structural_morphism.py` and
[DERIVED_EPI_MEMORY.md](../DERIVED_EPI_MEMORY.md). Reuse their existing
physics tests and `benchmarks/derived_epi_memory.py` when its domain applies.
The Taylor propagator checks in `epi_memory.py` are residual diagnostics,
not certified solution-error bounds. `p5_memory_truncation.py` supplies
rational enclosures only for its fixed unit five-node path, common capacity
and declared partition. Outside that domain, justify the new error envelope
before accepting a memory prediction; do not relabel a residual as a bound.

## 8. P5 — Maintenance, recovery and transitions

P5 has two dependent studies in the same programme. It does not launch
independent THOL, REMESH, winding or criticality campaigns.

**Model-extension gate:** pure-EPI diffusion on fixed connected symmetric
nonnegative conductance with fixed positive capacities relaxes to a uniform
field; it cannot alone sustain a stationary differentiated pattern.
If P5 needs phase/capacity feedback, forcing, REMESH
or other events, derive and admit those channels/state/input mappings in a
versioned P2 contract and validate the relevant new predictions on fresh
reserved evidence under P3 rules. Passive-transport support does not transfer
to the extension. Reuse the protocol and unchanged components, not the old
test outcome as evidence for a new mechanism. An internal-only extension
remains explicitly mathematical until its physical mapping passes these gates.

### P5.1. Fixed-target maintenance and recovery

Fix the original target, comparison coordinates/metric, admissible domain,
perturbation, canonical response word and finite horizon before execution.
Explain which derived capacity/phase/support mechanism supplies recovery.
A supplied response policy is a controlled intervention, not an autonomous
selection rule. Any metric/support change needs explicit reset accounting.
Keep the first comparison on identical ordered nodes. Existing fixed-support
reset telescopes do not compare a THOL birth across changed node count;
that would require a separately justified state/target correspondence.
Check the signed original-metric rate and event budget: compatibility of
the target with the new generator does not imply monotone old-target error.

Use three matched branches: undisturbed reference, disturbed without response,
and disturbed with response. Use the same state before feedback in the two
disturbed branches. Allow at most three independent pilot repeats per
branch, one predetermined word and horizon; no word search or extension
until recovery appears. Measure mean drift, original-target shape error,
capacity activity, consumed pressure, event resets and operator admission.
Freeze success thresholds from the model and measurement resolution.
With target errors `R_u`, `R_p`, `R_f`, reuse damage `R_p-R_u`, benefit
`R_p-R_f` and remaining gap `R_f-R_u`. Require positive damage by the
predeclared criterion before calling the outcome recovery. The retained
perturbation campaign is a no-damage control, not a successful restoration
fixture; a no-damage result ends that recovery test without extending it.

**Acceptance:** an actual disturbance is resolved within the predicted
domain/horizon, improves against the disturbed control and fixed original
target by the declared margin, and preserves the required state/grammar
conditions. Freezing, disappearance of amplitude, a changed comparison
target, disconnected child creation or overwritten pressure does not meet
the active-restoration claim. Finite maintenance is reported only for its
observed horizon. Autonomous maintenance additionally needs derived endogenous
policy selection. Indefinite maintenance, controlled or autonomous, requires
a preserved full-state domain and a proof covering future evolution under
the declared policy. Those stronger claims remain open unless separately proved.

Reuse `forced_support.py`, `capacity_feedback.py`, `support_transport.py`,
`phase_response.py` and existing causal event budgets. The existing
`benchmarks/structural_perturbation_response.py`, THOL birth/attachment and
child-coupling campaigns provide controls; activate only those necessary for
the admitted mechanism.
The capacity-feedback sufficient box concerns synchronized, equal-conductance
two-node support and includes `h*e*(c+rho*A)<=1`, plus its model-domain
conditions. Runtime/grammar admission must be verified separately; this
model theorem does not certify it. It is not a general restoration theorem.
Reuse the known binary64 nonzero fixed-gap controls to set represented
resolution limits. Do not require exact capacity consensus for those proved
fixed-gap cases or reopen that completed obstruction; this does not rule out
EPI consensus or consensus under other admitted kernels.

### P5.2. A causal regime study

Proceed only after P5.1 classifies a mechanism or an informative obstruction
and the proposed regime protocol has its own admitted scope. Specify whether
the predicted change is a chart/U3 loss, stability change, support event,
capacity exhaustion or another derived mechanism. Circular phase, a finite
regime transition and thermodynamic criticality remain distinct.

Use `phase_transition.py` and, when needed, `phase_scaling.py` on actual
canonical trajectories. Do not import the transitions or exponents of an
auxiliary Landau/Stuart-Landau equation. Vary one already admitted quantity
or preparation and specify the predicted onset or absence of change.
Start with one size and the bounded control/replicate grid justified by the
mechanism. Finite-size and hysteresis studies are optional, not compulsory
to establish a finite regime change. If both are necessary, the initial
pilot ceiling is three sizes x three control values x three repeats x two
directions = 54 runs, not a mandatory run count or adequate-sampling theorem.
Use a balanced size/control/replicate grid for `phase_scaling.py`; forward
and reverse sweeps are separate inputs with declared chronological
continuation/reset rules. A node-birth trajectory is not a set of independent
size replicates. Correlated nodes are not independent repeats. Apply the
same pre-evaluation budget/precision rule as P3.

`topology_transitions.py` reads graph geometry. Use it only for a changing
geometry/support question; fixed-geometry EPI/phase changes require their
own admitted state diagnostics. A morphology label is not a general regime
detector, and a finite grid is not evidence of a thermodynamic limit.

**Acceptance:** the observed finite regime behavior agrees with the
predeclared nodal mechanism and survives the relevant duration, numerical
resolution and control checks. Boundary peaks, plateaus, unresolved
transients or arbitrary classification thresholds remain inconclusive;
they do not authorize automatic grid growth. Report no transition when
that is the supported result. A thermodynamic or universal-exponent claim
requires a further derivation and is outside this finite delivery.

**Combined output:** recovery and regime results share the P2 measurement
contract, P3 freeze/evaluation rules and P4 state-sufficiency boundary.
Distinguish controlled recovery, finite maintenance, autonomous maintenance
and physical correspondence in the final evidence table; none substitutes
for the others.

## 9. Resumption, archival work and final delivery

At each completed delivery, update this status board and the portfolio's
programme-level summary. Retain the exact inputs, results, failed gates and
one next executable task. If the model changes after evaluation, version
the protocol and reserve fresh evidence; do not rewrite previous outcomes.

C6 stays preserved at B75: 41/56 excluded labels, 15 pending. No C6 LP,
history expansion or stability search belongs to P1-P5 by default. Its
existing bounded solver comparison is archived optional work, activated
only for a named dependency or an explicit reprioritization. The earlier
roadmap snapshot and B75 evidence are not rewritten.

The final programme delivery must contain: the repaired instrument contract,
admitted observation model, frozen held-out prediction result, reduced-state
memory result, maintenance/recovery/regime results, and a single evidence
table separating exact mathematics, implementation validation and physical
support. List rejected and unresolved claims alongside positive results.
The next action after this planning checkpoint is **P1.1**, followed by the
remaining P1 tasks. No implementation or experiment ran to produce this plan.
