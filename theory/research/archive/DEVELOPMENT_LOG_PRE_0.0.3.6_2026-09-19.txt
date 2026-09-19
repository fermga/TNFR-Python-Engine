# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed — 2026-09-11 Reception temporal evidence

- Centralized graph-backed Reception reads in one owner-bound, value-sealed
  pre-EN snapshot. Direct operator execution, low-level graph dispatch and the
  all-target Jacobi stage now use the same target, neighbour, kind and optional
  source inputs, and reject a stale or cross-graph prepared snapshot before any
  structural write.
- Added one sealed `ReceptionStageObservation` per accepted two-phase target.
  It records pre/post EPI and kind plus the exact pre-EN neighbour/source read;
  tracked source metadata is checked again after metrics, monitors, any
  requested pressure refresh and schedule recording, before final warning
  publication. Late EPI,
  kind or tracked-source mutation rejects the stage and rolls back graph-owned
  state without leaking that warning. Disabled source tracking preserves opaque
  legacy metadata without claiming its contents.
- Carried Reception observations through `NetworkStageResult`,
  `ExecutedGlyphStage` and the enclosing event-execution seal. This is finite
  provenance evidence, not a proof of auxiliary-state stability.
- Made Reception metrics state their read boundaries explicitly, separated
  stored-metadata presence from the pre-EN source snapshot, and changed the
  coarse effectiveness predicate to `abs(DeltaNFR) < 0.1`, sharing that policy
  constant with Coherence. Corrected the low-level EN doctest to the canonical
  half-mix result `0.45`.
- Aligned the public strict precondition with the canonical validator and made
  empty-source advice a single non-gating warning. Standalone metrics now reuse
  the same empty-neighbour and missing-EPI policy as execution.
- Defined directed Reception input as incoming `source -> receiver` arcs across
  graph dispatch, object dispatch, domain preflight, metrics and optional source
  telemetry. Source discovery remains diagnostic and does not select or gate the
  direct neighbours used by the numeric blend.
- Corrected EN's postcondition boundary: the jump leaves stored `DeltaNFR` and
  `dEPI` unchanged, so operator-local `C(t)` is immediately unchanged. Pressure
  refresh is a later observation and may change coherence.

### Added — 2026-09-11 Binary64 REMESH boundary and P2 causal binding

- Centralized the production delayed-REMESH scalar evaluation in
  `_delayed_remesh_kernel.py` and added an exact pairwise observer. It records
  the Jensen denominator, ideal/raw/bounded separations and signed rounding,
  clipping and total defects without promoting one pair to a uniform class.
  A normal-valued `alpha=1/2` witness requires exactly `eta=2^210-1/4`, so
  strict robust-envelope contraction requires `q<4/(2^212+3)`; a bounded hard-
  clipped interval alone is insufficient for the existing `q=9/16` policy.
- Added the sealed `alpha=1` hard-clip REMESH class. On fixed ordered support,
  fixed positive metric and sufficient finite represented history inside one
  interval, the runtime recurrence copies the global delayed row numerically,
  has uniform relative defect `eta=0`, and preserves the class under REMESH.
  Signed-zero bits, schedules, repeated event execution and future runtime
  behavior remain outside this certificate.
- Added the first useful uniform represented class with `0<alpha<1`. On
  ordered P2 support, any fixed positive diagonal metric, fixed delays,
  `alpha=0.5`, a symmetric hard interval `[-B,B]` with
  `B>=4*2^-1074`, and sufficient antisymmetric history rows `(a,-a)`, the
  production REMESH map preserves the class and has sharp uniform relative
  defect `eta*=135/124`. The proof combines an analytic IEEE error bound for
  `sqrt(D)>=11*2^-1074` with exact enumeration of 6,615 integer-core candidates,
  3,890 of them admissible; `(-3,-2,-3)*2^-1074` attains equality. Robust
  composition is strict for `q<124/259`: `q=4/9` gives `q_eff=259/279` and
  margin `20/279`, while equality at `q=124/259` has zero margin and `q=9/16`
  is rejected. REMESH-only forward invariance does not extend to arbitrary
  metric-centered rows, unrestricted fixed lattices, graph execution or future
  complete-runtime behavior.
- Added the restricted global binary64 P2 half-Reception composition. With two
  mutual singleton neighbors, an immutable all-target EPI snapshot, the exact
  configured mix `0.5` and one common hard clamp, both proposals are
  numerically equal for every finite represented pair in the source interval.
  The numeric kernel gain
  is therefore `q=0`; combined with `alpha=1`, active-history spatial
  disagreement is exactly zero after `tau_global+1` cycles. The certificate
  covers arbitrary finite repetition of these numeric EPI kernels while
  withholding complete Reception-stage, grammar, live graph and solver claims.
- Added a finite causal adapter for one executed P2 half-Reception stage. It
  binds a sealed event result, two-phase EN evidence, ordered graph support,
  runtime singleton neighbors, exact half mix, hard interval, metric ray and
  captured endpoints to the global `q=0` kernel by bit-exact replay. REMESH
  graph binding, auxiliary Reception state and repeated/future runtime claims
  remain outside its scope.
- Added a sealed finite causal P2 Reception/REMESH sequence certificate. It
  binds every selected executor-owned EN stage to the global `q=0` kernel and
  every same-cycle `alpha=1` hard-clipped REMESH to its represented global-delay
  row with `eta=0`. For an observed sequence with
  `N >= tau_global+1`, it verifies the active history suffix and exact
  post-horizon spatial-disagreement extinction. Stale retained rows before that
  suffix, future or unobserved repetition, auxiliary state, solver properties
  and full TNFR stability remain outside its scope.
- Added a reusable transactional P2 runtime-policy entry point. Every call
  preflights the intact
  kernel, undirected P2 support and metric, exact half-Reception factor,
  hard-clipped `alpha=1` REMESH controls, active history, clock and canonical
  zero-flow cycle specifications. It rederives U1a admission from the live EPI
  pair at every cycle start. Causal execution and finite post-certification
  occur inside one outer graph transaction, so a zero pair or any other failure
  restores graph-owned state. Success certifies only that invocation; future,
  unobserved-repetition and auxiliary-state claims remain false.
- Added examples 173–178, including
  `176_runtime_p2_reception_remesh_sequence.py`, exact public stubs, facade
  exports and adversarial tests covering normal/subnormal extremes, signed
  zero, inactive and active clipping, hostile inputs, private reseals and the
  exact energy-gain-four witness for the canonical default EN factor. Example
  177 demonstrates two independently revalidated policy invocations. Example
  178 records the sharp half-alpha class, exact proof partition, robust gain
  boundaries and the metric-centering and fixed-lattice falsifiers.

### Added — 2026-09-11 Relative-defect REMESH/schedule stability

- Added `remesh_schedule_relative_defect_stability.py` and its public stub. For
  one intact common-`q` policy certificate, a declared signed pre-schedule
  defect bound `E_H(z)-E_H(y) <= eta*J`, with
  `J=sum_d c_d E_H(x[k-d])`, yields the exact effective head gain
  `q_eff=q*(1+eta)`. The theorem reuses the existing companion envelope,
  rejects `q_eff>1`, gives nonincrease at `q_eff=1`, and gives block margin
  `1-q_eff` plus geometric spatial-disagreement decay when `q_eff<1`.
- Added `runtime_remesh_schedule_relative_defect.py` and focused causal tests.
  Its finite observer binds an intact executed cycle sequence to the theorem,
  verifies every selected signed defect and represented schedule gain, checks
  the full history-energy vector envelope, and applies the complete-block
  endpoint bound. `J=0` is checked algebraically without forming a ratio.
  This finite evidence does not establish a forward-invariant runtime class,
  repeated or future binary64 stability, solver properties, or full TNFR
  stability.
- Re-exported both robust APIs from `tnfr.physics` and added
  `172_runtime_remesh_relative_defect.py`, which records an exact-zero defect
  complete block and a positive binary64 defect accepted at its exact minimum
  `eta`.
- Centralized same-call integrity validation in the causal block adapters. The
  relative-defect constructor and each public integrity query now perform one
  deep source, certificate and nested-block validation apiece; every later
  query validates afresh, so no mutable trust cache is retained.

### Added — 2026-09-11 Uniform exact REMESH/schedule policy stability

- Added `remesh_schedule_policy_stability.py`, its public stub, focused tests
  and `171_remesh_schedule_policy_stability.py`. For one fixed exact REMESH
  companion and any sequence of consensus-preserving exact schedule maps with
  common fixed-metric disagreement gain at most `q` in `[0,1]`, the sealed
  theorem verifies `B_q=diag(q,1,...,1)P`, prefix gain at most one and block
  gain at most `q` over the sufficient universal horizon
  `L=active_max_delay+1`. Thus `q<1` gives uniform normalized margin `1-q` and
  repeated geometric spatial-disagreement decay. `q=1` supplies only
  nonincrease and a zero certified margin. Schedule-map verification, binary64
  runtime promotion, rounding/clipping control, solver claims, adaptive grammar
  and full TNFR stability remain outside scope.
- Centralized strict exact square-matrix multiplication and nonnegative integer
  powers in `_exact_linear_algebra.py`; rectangular proof kernels retain their
  separate contracts.

### Added — 2026-09-11 Causal event/REMESH execution

- Added `runtime_remesh_schedule_block_margin.py`, its exact public stub,
  focused tests and `170_runtime_remesh_block_margin.py`. The sealed observer
  selects a nonempty contiguous boundary block from one intact causal
  execution and verifies `D=K+S` with nonnegative schedule slack. At positive
  initial energy it records the block-specific normalized lower margin
  `kappa=K/V_before` and endpoint gain bound `1-kappa`. The public witnesses
  give `kappa=139/256` and the lag-one `alpha=1` boundary `kappa=0`.
  Equilibrium and amplitude scaling exclude a positive uniform absolute drop;
  a uniform positive normalized block margin over a declared forward-invariant
  runtime class, intrablock runtime prefix control, repetition and future stability remain
  unproved.
- Added `event_remesh_causal_runtime.py`, its exact public stub, focused tests
  and `169_event_remesh_causal_runtime.py`. The outer executor runs ordered
  `EventRemeshCycleExecutionSpec` values on one graph in one transaction and
  seals ordinal/spec/result receipts, including exact schedule identity. Its
  result always retains the ordinary offline cycle observation and, by default,
  the compatible runtime schedule/history telescope. The explicit
  `require_runtime_telescope=False` branch always stores `None`, so causal words
  without a common affine metric still retain same-invocation order, common
  graph identity and finite graph-owned atomicity; telescope-specific claims
  remain false. It does not compose
  schedule and REMESH gains or prove a uniform positive normalized block
  margin over a declared forward-invariant runtime class, intrablock runtime prefix control,
  solver accuracy/order, mesh convergence, repetition or future stability.
  Emitted I/O, warnings, external resources and external-only aliases remain
  outside rollback.

### Changed — 2026-09-11 Causal proof validation

- Reused authoritative nested validation results within each single deep-proof
  query, removing repeated telescope and boundary walks without caching trust
  between calls.
- Hardened event/REMESH graph admission to inspect the concrete runtime type
  hierarchy without reading a caller-controlled virtual `__class__` attribute
  before the transaction snapshot exists.

### Added — 2026-09-10 Executed refinement and transactional runtime boundary

- Added `observe_executed_event_local_zhir_physical_prejump`. Starting from one
  intact stage-certified `OperatorEventExecutionResult`, it binds the scheduled
  and committed ZHIR event, preceding physical partition, terminal segment,
  glyph stage, ordered Mutation decisions and trigger certificates to the same
  execution. The existing physical pre-jump observer remains an offline
  coordinate pairing without common-execution provenance.
- Added `event_remesh_refinement.py` and
  `test_event_remesh_refinement.py`. The new pure observer compares three
  already committed event/REMESH cycles on strictly nested coarse,
  intermediate and fine physical partitions. It records exact represented EPI
  checkpoints and persistent-node pairwise `L_inf` errors, executor-linked ZHIR
  rates and gates, and modal products only under a common captured generator.
  Integrity or decreasing finite errors do not certify solver accuracy/order,
  mesh convergence, Lyapunov decrease, a combined schedule/REMESH gain,
  whole-three-mesh atomicity or future behavior.
- Added `reversible_eigenmode_reference.py`, its exact public stub, focused
  tests and `167_reversible_eigenmode_reference.py`. For any fixed connected
  symmetric nonnegative rational conductance with positive rational capacity,
  the sealed pure kernel verifies one exact nonuniform eigenmode in
  `H=diag(d_i/nu_i)`, encloses its exponential solution rationally, constructs
  pressure-refreshed Euler products, and proves exact `L_inf` and
  `H`-error-energy bounds. Proper positive subdivision strictly improves the
  factor and quadratic bounds; the `h_max` bound yields conditional exact-real
  convergence for fixed-data admissible partition families. The `mu*T <= 4096`
  limit caps only the enclosure's integer-power exponent and does not bound
  arbitrary rational bit size. Binary64 asymptotics, solver order, mixed
  modes, directed or changing generators, glyph/REMESH dynamics and full TNFR
  stability remain outside the certificate. Example 167 checks both exact
  nonuniform modes of nonregular `P3` through the public `tnfr.physics` facade.
- Added `runtime_eigenmode_reference.py`, its exact public stub, focused tests
  and `168_runtime_reversible_eigenmode_reference.py`. The sealed adapter
  derives one exact reversible single-mode reference from finite intact
  executor-owned pressure-refreshed partitions. For every segment it separates
  the rationalized binary64 pressure residual `rho`, held-input execution
  residual `eta` and combined defect `epsilon`, then propagates general off-mode
  errors through the complete Euler matrices. It also reports signed
  represented-minus-continuous endpoint intervals and exact rational `L_inf`
  and `H`-error-energy bounds. The supplied family remains an offline ordering
  of individually certified executions; binary64/runtime mesh convergence,
  solver accuracy/order, common causal provenance, glyph/REMESH dynamics,
  repetition and future/full TNFR stability remain outside the observation.
- Added `event_remesh_reference.py`, its exact public stub, focused tests and
  `166_event_remesh_reference_family.py`. The sealed effective-P2 family binds
  three event-free, fixed-conductance, homogeneous-capacity,
  pressure-refreshed Euler executions of one nonuniform initial mode, with
  `0 < lambda*h < 1` and two proper positive subdivisions. Rational
  exponential enclosures prove
  the finite continuous/Euler error bounds and strict subdivision improvement;
  unit-delay REMESH scales the ideal error by `beta=(1-alpha)^2`, while the
  committed endpoint bound adds the signed rounding-plus-clipping residual
  norm. Hard clipping must use one common scalar interval. Solver order,
  generic or binary64 asymptotic convergence, arbitrary glyphs or mixed
  channels, soft clipping, changing support or metric, repetition and future
  behavior remain outside the certificate. Its continuous/Euler proof is now
  centralized as the two-node specialization of the general reversible
  eigenmode theorem. Its three physical partitions now pass once through the
  common runtime binding, with stronger exact-affine and zero-defect checks;
  the REMESH derivation retains only its scale and runtime residual layer.
- Added `remesh_history_stability.py` and
  `test_remesh_history_stability.py`. The exact finite companion certificate
  derives the invariant temporal distribution for uniform unclipped delayed
  REMESH and verifies a stationary-weighted Jensen disagreement identity.
  `0 < alpha < 1` gives primitive temporal mixing toward the preserved history
  barycenter; `alpha = 1` is a pure-delay permutation with conserved augmented
  energy and possible periodic orbits. Binary64 clipping, changing parameters,
  spatial consensus, zero pressure and schedule/REMESH composition remain
  outside the theorem. The certificate, transition observation and both
  constructors are re-exported from `tnfr.physics`.
- Added `runtime_remesh_history_stability.py` and its exact public stub. The
  one-transition bridge accepts an applied executor-sealed event/REMESH cycle,
  reverses the retained runtime history into companion order, and replays the
  raw affine and canonical clipping evaluations bit for bit. It retains signed
  exact rounding and clipping residuals, their augmented-energy defects,
  stationary-barycenter drift and a posteriori sufficient Lyapunov margins.
  Hard clipping is nonexpansive for the observed common-interval step; the soft
  knee counterexample and a binary64 rounding counterexample remain explicit.
  Live history advance, repetition, schedule composition and future stability
  are not promoted by this isolated bridge.
- Delayed-REMESH stability evidence now rejects a nonzero exact diagnostic that
  would underflow to displayed binary64 zero, before planning or execution can
  write graph state.
- Added `remesh_schedule_stability.py` and its public stub. The sealed pure
  observation telescopes ideal REMESH dissipation with signed raw, clipping and
  schedule disagreement-energy defects, retains the exact schedule-gain slack,
  a sufficient one-step lower bound and stationary-history barycenter drift.
  Caller-supplied heads remain algebraic evidence without executor provenance,
  repeated-runtime or future-stability promotion.
- Added `runtime_remesh_schedule_stability.py` and its public stub. The
  adjacent-cycle adapter requires exact recorded boundaries, one fixed REMESH
  configuration and a common normalized schedule metric. It binds each applied
  binary64 REMESH result to the next represented schedule and verifies the
  resulting history append. Per-boundary balances telescope to one exact finite
  energy drop and summed gain-based lower bound without multiplying
  fixed-history REMESH factors. Shared graph provenance, cross-call atomicity,
  global executable gain, repetition and future stability remain outside the
  result.
- Exact REMESH companion certificates and transitions now reconstruct every
  derived field during seal validation. Privately re-sealed inconsistencies,
  duplicate or unhashable node orders and hostile equality payloads fail closed.
- Required custom event-schedule integrators to modify only EPI, `dEPI_dt`,
  `d2EPI_dt2` and the runtime clock and to realize exactly, in rationalized
  represented values, `EPI_right-EPI_left = dt*nu_f_left*DeltaNFR_left`.
  This held-input identity does not certify solver accuracy or order.
- Made graph transaction snapshots owner-bound and preserved NetworkX
  structural mapping identities, alias topology and capturable callback-owned
  state. Exact `CallbackSpec` carriers and replayable one-dimensional object
  history arrays retain their aliases while their reachable mutable state is
  covered. Snapshot preflight rejects mutable structural keys with non-identity
  hash/equality, custom `__deepcopy__` hooks and unmodelled opaque C state while
  accepting common immutable atoms. Exact standard locks and loggers, including
  nested occurrences, are treated as external resources without invoking their
  copy reducers. Rollback covers capturable graph-reachable state; emitted I/O
  and warnings, external-resource state and external-only aliases remain outside
  it.
- Centralized proof-stamp comparison on a closed immutable token grammar with
  bit-exact float and complex handling. Runtime, refinement, diffusion,
  pointwise, hybrid, Mutation and REMESH evidence now authenticates its own
  outer seal before nested semantic validation and fails closed without calling
  truth, equality or descriptor protocols on replaced stamp values.
- Compacted parent runtime seals through exact-type references to independently
  validated child seals. Event/REMESH proof growth is now linear in the number
  of segments while deep child tampering still invalidates every enclosing
  record. Graph snapshots also preseed all direct graph-owned identities before
  capture and admit only the canonical `NodeCache` serialization binding and
  its exact weak owner reference.
- Made standalone REMESH planning observationally pure and materialized live
  configuration, runtime controls, EPI channels and delayed history once per
  guarded boundary. A conversion or history read that changes graph state now
  aborts and restores the enclosing transaction instead of becoming a later
  baseline. `ON_REMESH` callbacks are graph-read-only observers; the public
  history append helper still derives its snapshot and capacity from the live
  canonical graph configuration.
- Materialized `physical_flow_partitions` exactly once inside the outer
  `execute_event_remesh_cycle` transaction and forwarded that tuple unchanged
  to event execution.
- Centralized the SDK graph-copy resource policy on the transaction classifier.
  Measurement, cloning and pulse trajectories now discard stale runtime caches,
  detach ordinary mutable graph data, and preserve exact external resources by
  identity under the documented external-state boundary. Alias relationships
  crossing those resources and explicit graph-container back-references remain
  outside the SDK data-copy contract. Ordinary stored Python bound methods are
  rebound to copied receivers, including receivers that own nested external
  resources; built-in methods retain Python's atomic deepcopy behavior.
- Updated S1, S2, S5 and S16 to distinguish the implemented finite three-mesh
  observation from convergence and to record the separate uniform exact
  augmented-history theorem. Subsequent entries above record the completed
  one-transition binary64 bridge and finite adjacent schedule/history
  telescope and the first compatible finite P2 reference-family theorem.

### Added — 2026-09-09 Physical pressure-refreshed event flow

- Added immutable `PhysicalFlowPartition` declarations with exact rationalized
  binary64 coverage, additive/subtractive clock checks and at least two
  positive physical segments.
- Extended `execute_operator_event_schedule` with
  `physical_flow_partitions=`. It refreshes `DeltaNFR` before every segment and
  at the terminal boundary, captures all segment flows, and rolls back the
  whole graph if a boundary callback or later operation fails.
- Added sealed `PressureRefreshBoundaryObservation`,
  `PhysicalEulerModalObservation` and
  `ExecutedPressureRefreshedFlowPartition` evidence. Pressure callbacks must
  preserve all non-pressure nodal state, full edge state, persistent graph
  configuration, histories, conductance and clock; existing non-`None` cached
  pressure weights cannot change, while the canonical default may initialize a
  missing or `None` cache.
- Applied the same restricted-write policy to operator-stage pressure refreshes,
  froze hook presence and identity for the whole schedule, rejected integrator
  history writes even on newly added nodes, and preserved capturable callable
  state. Violations now fail before replacement hooks can run and roll back all
  graph-owned state.
- Hardened graph transactions for custom mapping factories, owner-qualified
  slots, NetworkX internal mappings and ordinary lock-bearing metadata while
  preserving their identities across rollback.
- Centralized runtime pressure callback dispatch in `_refresh_delta_nfr`,
  retaining legacy callbacks without retrying an internal `TypeError`.
- Added exact common-metric segment gain composition and kept solver accuracy,
  order, mesh convergence, adaptive U2/U4 and future behavior explicitly
  uncertified.
- Added `EventLocalZHIRPhysicalPrejumpObservation` and
  `EventLocalZHIRPhysicalRefinementComparison`. They separate Mutation's actual
  terminal-segment secant from an offline whole-interval secant and compare
  ordinary binary64 evaluations of the exact-real held factor `1-T*mu` and
  refreshed product `product_k(1-h_k*mu)` without identifying runtime endpoints
  with those maps or assuming equivalent decisions. Modal comparison requires
  trusted segment replays and one fixed generator; matching spectra alone cannot
  promote an isospectral, noncommuting sequence. Exact endpoint-map promotion
  remains gated by the separate exact-affine certificate.

### Added — 2026-09-09 Event-local ZHIR refinement evidence

- Rejected a scheduled Mutation before graph writes when the binary64
  subtraction of its immediately preceding flow endpoints differs from that
  flow's authoritative represented duration. This prevents ZHIR from measuring
  a secant on an unintended time interval at large absolute clock origins.
- Extended observed-flow certificates with sequential built-in Euler replay for
  any positive internal substep count whose represented substep durations sum
  exactly to the declared interval. This held-pressure result remains separate
  from the one-step affine map and from pressure-reevaluated physical refinement.
- Value-sealed executor flow wrappers and their nested proof payloads, preserving
  binary64 bit distinctions and structural snapshots of mutable node labels.
  Direct construction, replacement and post-capture mutation fail closed.
- Value-sealed complete executed glyph stages and required one intact stage per
  committed event. ZHIR decision order is bound to execution targets; an
  abstaining adjacent flow remains observable without entering the represented
  schedule product.
- Added `EventLocalZHIRPrejumpObservation` for offline coordinate-paired flow and
  event records. It keeps exact rational endpoint secants separate from the
  actual binary64 Mutation subtraction/division and retains both passing and
  rejected strict-threshold outcomes.
- Added `EventLocalZHIRHeldPressureComparison`. Under identical state, pressure,
  capacity, conductance, support, duration, event coordinate and threshold, it
  certifies a gate result across different internal substep counts only when
  observed decisions agree and every exact rate perturbation lies strictly
  inside the baseline threshold margin.
- Added one sealed `MutationStageDecisionObservation` per accepted two-phase
  ZHIR target and carried it into opt-in executed glyph stages independently of
  EPI-map certification. The record freezes trigger evidence, capacity, phase
  and regime decisions, acceleration, bifurcation read-out and U4 context.
- Kept pressure-reevaluated refinement, modal equivalence, solver accuracy/order,
  shared flow/event execution provenance, adaptive U2/U4 and future behavior
  explicitly outside these results.
### Added — 2026-09-09 Ordered event/REMESH cycle observations

- Added sealed `RemeshHistoryTransitionObservation` evidence to each
  `EventRemeshCycleResult`. It records incoming and outgoing exact represented
  histories, the canonical bounded append, and independently available local
  and global delayed vectors.
- Added `EventRemeshCycleBoundaryObservation`,
  `ObservedEventRemeshCycleSequence` and
  `compose_event_remesh_cycle_observations` for at least two sealed cycle
  results supplied in caller order. The pure composer checks exact recorded EPI,
  capacity, pressure, phase, clock and full REMESH-history continuity at every
  adjacent supplied-result boundary. Its indices are local ordinals, not runtime
  call identifiers.
- Separated exact raw metric equality from normalized metric-ray compatibility
  and added tri-state alignment with each nested represented schedule metric.
  Exact boundary continuity remains separate; the stronger common-metric
  sequence result requires every nested schedule to expose the shared ray.
  Nested schedule compositions and REMESH results remain independent evidence.
- Kept the repetition boundary explicit. An `alpha=1`, lag-one two-state trace
  alternates between its current and delayed EPI even though each fixed-history
  REMESH result has zero current-state coefficient. No schedule/REMESH gain
  product, evolving-history repetition, runtime-global gain, whole-sequence
  atomicity, full graph/grammar-history continuity, solver-accuracy result or future
  theorem is claimed.
- Rejected reuse of the identical cycle-result object as two observations. This
  guards trivial self-pairing but does not establish causal order or shared-graph
  provenance; distinct value-equal records remain observationally indistinct.

### Added — 2026-09-09 Observed represented EPI schedule composition

- Added opt-in executor-owned EPI jump evidence to the shared all-target stage
  dispatcher. AL/SHA/VAL/NUL/ZHIR/NAV reuse their frozen pointwise proposals;
  EN/RA certify one simultaneous neighbour stage. Unsupported glyphs and
  rejected domains return explicit abstention, while a certificate/proposal
  mismatch aborts before commit and pressure refresh.
- Added `include_stage_certificates` to operator-event execution and the
  event/REMESH cycle. It implies positive-flow capture and returns one
  `ExecutedGlyphStage` per accepted event, bound to the actual captured EPI
  endpoints, exact pre/post metric rays and adjacent positive-flow evidence.
- Added sealed `RepresentedEPIScheduleOperation` records and
  `ObservedRepresentedEPIScheduleComposition`. A complete finite observed trace
  publishes its exact rational represented-map gain product only under exact
  operation cardinality, node order, endpoint continuity and one normalized
  positive metric. Ineligible traces retain per-operation reasons and publish
  no partial product.
- Kept the global theorem boundary explicit: the product is global for the
  represented affine maps and bounds this observed trace through exact endpoint
  binding. `runtime_schedule_global_gain_certified` remains false; solver
  accuracy, refinement, full-multichannel behavior, delayed REMESH, future
  schedules and repeated execution are not certified.
- Sealed `NodalFlowIntervalCertificate` proof fields and made wrapper and
  composition properties revalidate them fail-closed, so replaced factors or
  Boolean claims cannot promote runtime or represented-map conclusions.
- Centralized exact positive-metric normalization and proportionality checks in
  `physics/_exact_metric.py` for pointwise, neighbour, resonance and hybrid
  composition certificates.

### Added — 2026-09-09 Runtime-observed nodal-flow evidence

- Added detached one-interval snapshots and certificates that report the exact
  rational held-pressure nodal identity, the pure-EPI pressure/affine/quotient
  result and the same-operation binary64 Euler replay as separate claims.
  Standalone endpoint agreement never infers runtime provenance.
- Bound `include_flow_certificates` to actual positive calls made by
  operator-event execution and event/REMESH cycles. Each
  `ExecutedNodalFlowInterval` derives built-in integrator provenance. Custom or
  subclassed integrators, RK4, multiple substeps, a live Gamma type other than
  `none`, clipping and requested extended dynamics block trusted binary64
  identification. Stale pure-EPI pressure can still pass that held-pressure
  level while blocking exact affine and quotient promotion. Changed support,
  conductance, capacity or pressure blocks the corresponding claim.
- Kept interval evidence read-only and opt-in; by itself it provides no
  solver-accuracy, glyph-gain or future/repeated-schedule theorem.
- Centralized scalar Gamma cache refresh through the live Gamma specification,
  aligning cached and vectorized paths after configuration replacement or
  in-place mutation.

### Added — 2026-09-08 Dissonance, Coupling, event-time and proof boundaries

- Promoted Recursivity as the thirteenth atomic all-target glyph stage. One
  immutable snapshot now preflights all targets and produces one shared
  advisory that is deduplicated per telemetry step. The stage commits node
  histories, provenance, metrics, monitors and pressure refresh atomically
  while leaving EPI, nu_f, phase, DeltaNFR and support unchanged. Explicit
  delayed EPI mixing remains the separate apply_network_remesh operation;
  grammar replacements and execution overrides retain the transactional
  Gauss-Seidel fallback.
- Promoted Self-organization as the twelfth atomic all-target two-phase stage.
  Every THOL target now reads one detached stage-start graph. Cross-parent child
  identifier collisions and structural commits resolve in snapshot-node rank;
  the complete child/`sub_nodes`/`sub_epis`/`hierarchy` merge is validated
  on a detached graph before live mutation. Direct and staged execution share
  the prepared nodal action and amplitude-alignment kernel, with full rollback
  through late monitor, metric and pressure-refresh failures.
- Promoted Dissonance as the eleventh atomic all-target two-phase stage.
  Local OZ actions and outgoing propagation now read one immutable snapshot;
  overlapping incoming pressure increments use a deterministic `math.fsum`
  reduction in snapshot-node order. Local pressure amplification, per-node RNG
  progress, ordered events, late warnings and full-stage rollback have explicit
  contracts and tests across graph variants.
- Promoted Coupling as the tenth atomic all-target two-phase stage. One
  immutable snapshot now feeds a deterministic circular-displacement merge,
  final phase normalization, U3 revalidation and functional-link coalescing.
  THOL and the final REMESH advisory promotion complete the thirteen-stage
  immutable proposal boundary.
- Added a finite operator-event schedule with zero-duration jumps and exactly
  `m + 1` declared flow intervals for `m` events. Exact rationalized binary64
  durations and offsets own physical ordering; float timestamps are display
  values. A companion exact-bound diagnostic gives a sufficient fixed-flow
  relaxation duration without executing operators or adapting U2/U4.
- Added `execute_operator_event_schedule`, which binds a valid schedule to the
  configured nodal integrator and shared atomic glyph stages. It freezes initial
  targets, validates every live clock boundary, rejects collapsed and
  nonadditive binary64 intervals, separates zero-duration jumps from EPI
  secants, and rolls back graph-owned flow/jump state as one transaction.
- Added `execute_event_remesh_cycle` as an atomic one-cycle bridge from an
  operator-event schedule through one canonical pre-REMESH history sample to
  the separately invoked delayed map. It freezes one positive diagonal metric,
  rejects schedule changes to ordered node support or incoming REMESH history,
  exposes weighted-consensus drift and capacity separately, and makes the
  optional post-map pressure refresh explicit. It claims no mixed or repeated
  evolving-history gain.
- Applied delayed REMESH now records its same-time `epi_time_history` right
  endpoint and treats `ON_REMESH` callbacks as observers of the EPI-only map.
  Mutations of structural channels, topology, delayed or physical histories,
  clock, event log, hook/config provenance or canonical telemetry fail
  atomically. Structural-memory EPI propagation records a further endpoint.
- Event/REMESH execution now preflights and freezes all deterministic delayed-map
  controls before schedule flow, binds immutable events to the live event log,
  and protects endpoint traces through the optional pressure refresh. Exact
  rational observations remain authoritative when a derived float display is
  unrepresentable and therefore `None`.
- Canonical delayed history now rebuilds `deque` subclasses before append and
  verifies the expected length transition, preventing overridden methods from
  inserting hidden samples.
- Centralized the runtime and delayed-planner validation of strict positive
  REMESH delays and protected bounded-history allocation from oversized
  `deque.maxlen` values. `_epi_hist` now retains its container identity during
  graph-owned rollback when in-place restoration is possible.
- Contracted the separately invoked delayed REMESH map with immutable plans and
  results, exact-support temporal snapshots, explicit insufficient-history and
  empty-support no-ops, exact recurrence/rounding/clipping separation and
  graph-state atomic commit. Optional one-step evidence separates convex
  three-input bounds, mean drift and conditional fixed-history gains; extreme
  diagnostics outside binary64 reporting range now fail explicitly.
- Added fixed-branch phase coarse-graining. The pairwise phase realization
  inherits the reversible diffusion quotient; the canonical mean-of-phasors
  channel has a restricted lifted-subspace closure and a `K3,3`
  same-macro-state counterexample to global projected autonomy. Canonical phase
  support remains unweighted even across zero-conductance edges.
- Completed fixed-`N` coherence-level geometry: canonical network levels are
  stratified `2N`-dimensional cross-polytope boundaries, and fixed positive
  capacities induce exact weighted `N`-dimensional pressure sections.
- Added a conditional pointwise certificate for frozen
  AL/SHA/VAL/NUL/ZHIR/NAV proposals. A successful network-stage result carries
  evidence computed from the executor's own detached snapshot and frozen
  proposals. It separates exact runtime realization, affine quotient gain and
  aligned pre/post diffusion metrics; histories, pressure refresh, mixed words
  and repetition remain open.
- Integrated the structural-affinity boundary throughout telemetry and the
  auxiliary Hamiltonian: `coherence_matrix` remains a bounded, potentially
  indefinite pairwise affinity and never substitutes for canonical `C(t)`.
  Read-only nodal-pulse diagnostics no longer append `W_sparse`, `W_i` or
  `W_stats` history while sampling that affinity.

### Changed — 2026-09-08 structural-affinity boundary

- Corrected the historical `coherence_matrix` documentation: it is an
  auxiliary bounded structural-affinity matrix, not the constitutive `C(t)`
  kernel and not positive semidefinite in general. The three-node path with
  identical node attributes gives the exact counterexample
  `W = I + A_path`, whose spectrum contains `1 - sqrt(2) < 0`; its normalized
  unit-diagonal trace is also independent of canonical coherence.
- Centralized conversion of dense and sparse affinity payloads for the legacy
  Hamiltonian builders. Dense 3-by-3 payloads are no longer mistaken for
  sparse triples, caller-supplied node order is now honored and validated,
  disabled affinity contributes a zero matrix, and the class and standalone
  builders share one finite strength and ordering contract.

### Changed — 2026-09-08 Coherence stage contract

- Promoted Coherence as the ninth atomic all-target two-phase stage. Direct IL
  and the network runner now share one pure snapshot proposal for
  sign-preserving pressure-magnitude contraction and circular phase locking.
  The canonical word path is target-order invariant for committed target
  `DeltaNFR` and phase before the opaque pressure refresh; the earlier
  multi-target operator-major Gauss-Seidel behavior remains visible only through
  the explicit legacy runner and its schedule-mismatch diagnostic.
- Committed direct IL phase and telemetry before monitor and metrics hooks.
  Integrity now compares bound before/after snapshots, including `|DeltaNFR|`,
  and never consults an unbound latest telemetry record. Stage telemetry adds
  canonical global and radius-local structural `C(t)` fields while preserving
  historical `C_global_*` and `C_local_*` pressure-dispersion values with
  explicit auxiliary aliases and deprecation metadata.
- Corrected negative-pressure reduction metrics, removed EPI-headroom and
  signed-positive-pressure admission assumptions, and deferred IL precondition
  warnings until all other fallible stage effects succeed. Warning-as-error,
  monitor and pressure-refresh failures restore the complete stage; grammar
  replacement still uses the transactional Gauss-Seidel fallback. The
  unpromoted set is now OZ, UM, THOL and REMESH.

### Added — 2026-09-07 all-target stability and stage contracts

- Promoted Emission, Silence, Expansion and Contraction to the shared
  all-target two-phase scheduler alongside Reception and Resonance. Their
  frozen pointwise proposals are built from one stage-start snapshot, globally
  validated and atomically merged; AL/SHA use one common stage timestamp.
  Primary structural channels are target-order invariant before the opaque
  pressure refresh. Ordered lifecycle, audit/telemetry and monitor effects
  retain requested target order, while identity-bearing caches remain tied to
  their live graph and node objects. Operators not yet promoted at that point
  remained transactional Gauss-Seidel stages.
- Promoted Mutation as the seventh two-phase stage. Its immutable all-target
  proposal binds each target's temporal threshold evidence, phase result,
  structural acceleration and U4 context to the stage snapshot, then commits
  phase and acceleration before merging ordered histories, bifurcation events,
  metrics and monitor effects. The opaque pressure refresh and relabeling
  equivariance remain outside the target-order result. The impossible legacy
  `ZHIR_BIFURCATION_MODE="variant_creation"` branch is now rejected before any
  write: ZHIR remains phase-only plus detection, while topology and sub-EPI
  creation belong to THOL.
- Promoted Transition as the eighth two-phase stage. Its immutable proposal
  binds `nu_f`, phase, `DeltaNFR`, latency state and each node's jitter draw/RNG
  progress to one stage snapshot. A missing graph seed is resolved inside the
  transaction, copied into the proposal snapshot, rolled back on rejection and
  persisted on success; one shared stage instant supplies every latency
  calculation. Ordered warnings, histories, transition events, metrics and
  monitor effects retain requested target order. Cache state, the opaque
  pressure refresh and relabeling equivariance remain outside the target-order
  result.
- Centralized VAL/NUL capacity, EPI-boundary and NUL reciprocal-pressure
  proposals. NUL audit events now carry target identity, and contraction
  metrics bind the captured pre-operation pressure or the matching node event,
  report signed and magnitude changes separately, and do not label unchanged
  zero pressure as densified. NAV transition telemetry now distinguishes the
  true pre-handler pressure, handler output and final retained pressure.
- Extended the EN/RA runtime-to-theorem bridge from one target to the complete
  two-phase Jacobi stage. The new read-only certificate assembles every ideal
  and represented row and replays finite repeated structural traces; regression
  tests compare every step with the public runtime. It proves convex hard-bound
  forward invariance in its declared domain and reports clipping, U3-set
  changes, identity rejection, consensus drift and
  RA diffusion-metric drift separately. It does not promote observed binary64
  agreement to global runtime affinity.
- Added the first direct composition from one certified all-target EN/RA stage
  to a strictly positive-duration fixed post-stage pure-EPI flow. The bridge
  rebuilds local domain conditions and validates node order, represented maps,
  rational row algebra, proof stamps and the post-stage metric before using the
  generic hybrid composer. It reports finite-horizon disagreement contraction,
  weighted-mean drift, RA metric changes and the aggregate pure-EPI pressure
  defect separately. Stored-pressure refresh, global binary64 runtime affinity
  and repetition of the stage-flow schedule remain explicitly uncertified.
- Replaced the dimension-dependent weighted-Frobenius composition penalty with
  a tighter exact rational quotient-gain bound for scalar and small quotient
  maps, retaining Frobenius as the conservative fallback. Identity now
  certifies gain one; declared bounds and proof-stamp integrity use the same
  decisive bound. Log-space composition retains the precise rational product
  and uses visible upward 32-bit-significand dyadic factors to prevent
  denominator growth without weakening the upper-bound proof.
- Added centralized network-stage contracts for all 13 operators, covering
  conservative read/write footprints, cross-target overlap, merge status,
  direct and shared rollback scope, structural-state target-order behavior,
  the retained target order of lifecycle and telemetry streams, relabeling
  scope and promotion blockers. Live schedule diagnostics now derive from this
  registry.
- Made every non-EN/RA operator-major stage in the shared word runner a complete
  graph transaction. Late target or pressure-refresh failures restore the
  stage-start graph while successful execution retains the established
  Gauss-Seidel reads. Unrelated synchronization primitives in graph metadata
  are preserved by identity across both successful execution and rollback.
- Centralized the dominant-neighbour EPI-kind rule shared by EN and RA and
  exposed EN kind before/after in its local realization certificate. EN now
  uses its historical pre-clipping proposal consistently in direct, staged and
  certified paths, including soft clipping.

### Added — 2026-09-07 temporal Mutation and transactional operators

- Added one pure Mutation trigger certificate that separates the instantaneous
  nodal prediction `nu_f * DeltaNFR`, the observed signed two-sample EPI secant
  and the three-sample structural acceleration. Timestamped histories use their
  physical intervals and require a fresh endpoint; legacy histories retain an
  explicit unit-operator-step basis. Missing, invalid or stale observations
  abstain without becoming false non-crossings.
- Recorded bounded timestamped EPI histories in the runtime, reset them across
  same-time hybrid jumps, made autonomous ZHIR selection fall back to IL with
  provenance, and added atomic whole-word SDK preflight across every target.
  Fluent builders and templates now request evidence-gated Mutation, record an
  explicit exploration abstention when the gate cannot be certified and never
  manufacture threshold evidence.
- Centralized structural acceleration in `compute_d2epi_dt2`, including the
  unequal-timestep second difference, canonical BEPI scalarization and a
  read-only mode used by Mutation and Self-organization.
- Made propagated Dissonance and Self-organization full graph transactions:
  all factors, histories, propagation proposals, hierarchy changes, telemetry
  and monitoring sinks are validated before commit and restored on a late
  failure. Common operator preflight now also clears failed integrity-monitor
  proposals.
- Synchronized the public runtime and typing facades for factor contracts,
  Mutation evidence reports and SDK entry points. Pulse trajectories now use
  within-interval threshold interpolation so distinct local and global crossings
  in the same sampled step retain their observed order.
- Unified the writable EPI boundary for AL, EN, RA, VAL, NUL, THOL and REMESH:
  raw real scalars and uniform-real BEPI embeddings retain their sign, while
  nonuniform or complex payloads are rejected before mutation. Yang-Mills graph
  construction now starts from the same scalar structural vacuum without
  consuming an extra random draw.
- Routed legacy Self-organization and Mutation validators through the shared
  physical/legacy history precedence and acceleration kernel. Multiscale EPI
  evolution now composes local and cross-scale pressure first, then advances
  every graph once through the nodal integrator, removing scale-order-dependent
  direct increments.
- Corrected the spectral-coordinate engine to evaluate canonical `-L_rw*EPI`
  on the node field and multiply heterogeneous `nu_f` pointwise before
  projection. Mutating simulations are no longer result-cached; reconstruction
  refreshes pressure/rate telemetry and restarts histories from one truthful
  endpoint rather than fabricating Mutation evidence.
- Replaced direct membrane EPI injection by an additive, U3-gated membrane
  `DeltaNFR` channel with explicit `dt`, simultaneous proposals, nodal residuals
  and atomic commit. GPU AL/RA strategies now use the public canonical operators
  with block rollback, and metabolic workflows execute validated OZ-THOL-IL
  words with atomic failure recovery; the duplicate propagation helper is now
  a read-only view of committed THOL telemetry.
- Replaced insertion-order-dependent SDK Reception/Resonance and GPU Resonance
  stages with one shared two-phase Jacobi scheduler: every target reads the
  same immutable snapshot, global validation precedes writes, and state,
  histories, telemetry and runtime attachments commit or roll back atomically.
  The neutral word executor and field signatures now use this shared semantic
  layer rather than depending on the SDK. GPU Resonance now runs the same
  pressure-refresh callback and reports amplification from observed frequency
  changes instead of assuming that amplification occurred.
- Corrected Self-organization so signed structural acceleration drives signed
  pressure, nested child EPI does not add mass to the parent coordinate,
  configured depth limits are effective, and network propagation requires an
  explicit grammar-valid Resonance stage. Membrane flux now advances every
  graph node through one shared nodal-integration clock while adding membrane
  pressure only at the boundary.
- Hardened spectral and optimization infrastructure: graph Fourier reads use
  the canonical signed scalar EPI chart; bases and cache results are
  authenticated and detached; cache budgets use stored bytes; sequence FFT
  precision is applied; normalized cross-power is scale invariant; and direct
  FFT/Euler routes expose when convergence or integration stability is not a
  certified property.
- Removed fabricated GPU, cache, memory and speedup evidence. Canonical graph
  pressure retains explicit CPU provenance until an accelerated kernel has
  full weighted/directed parity, fallback requires a distinct declared CPU
  callable, and orchestration learns only from finite measured comparisons.
  Structural clipping now has one validated scalar/vector soft-knee contract
  and no longer borrows an optimizer speedup constant.
- Extended the shared graph transaction boundary to preserve runtime object
  identity while restoring mutable mappings, sequences, sets, deques, NumPy
  arrays, inherited slots, custom graph attributes and the graph-owned cache
  manager. Rollback also restores exact node, neighbour, predecessor,
  multiedge-key and graph-attribute insertion order, preserving later
  Gauss--Seidel semantics. Unsupported mutable attachments now fail before the
  first write.
- Centralized finite signed scalar-EPI admission for diagnostics and manifests.
  Centralization, pattern discovery and self-optimization now distinguish
  measured evidence from unavailable values instead of inventing load,
  prediction, compression, memory, cache or speedup observations. Advisory
  integration routes now abstain until an executable adapter exists, validate
  canonical state with a dependency signature and retain performance values only
  with finite measured evidence; obsolete synthetic baselines were removed.
- Stabilized coherence contrast at the finite binary64 extremes by sharing a
  scaled distance kernel across NumPy and Python paths, so opposite maximal
  finite values produce maximal finite contrast without overflow warnings.
- Corrected parallel/distributed execution contracts: workers use the public
  Sense-Index computation, graph direction and multiedges survive transport,
  result merges reject incomplete or duplicate outputs, simulations accept an
  explicit seed, and execution/backend claims remain unmeasured unless observed.
- Made operator registry probes explicitly opt out through `__register__ = False`,
  preventing test or extension classes from contaminating the canonical
  13-operator catalog and its discipline signature.

### Added — 2026-09-06 core stability program

- Added the S1–S16 core dynamics research map and completed its first restricted
  result: fixed connected symmetric pure-EPI diffusion with positive
  heterogeneous structural frequencies converges exponentially in the
  `diag(d_i/nu_i)` metric.
- Added a read-only stability certificate reporting the conserved weighted mean,
  Lyapunov balance, generalized spectral gap and exponential decay bound.
- Derived a common Dirichlet-energy bound for arbitrary time-varying capacities
  inside declared positive finite bounds, with a counterexample showing that
  changing capacity ratios make the final consensus schedule-dependent.
- Hardened that time-varying result into a rational exact-real theorem induced
  by the effective binary64 conductances and capacity bounds. Exact theorem
  status, availability of a positive downward-rounded binary64 rate, ordinary
  spectral diagnostics and runtime-integration verification are now separate;
  underflow and diagnostic overflow cause safe operational abstention without
  erasing a valid exact-real proof.
- Corrected the earlier claim that U2 compliance alone proves Lyapunov stability
  for every operator sequence; that general result remains open.
- Added a graph-specific EPI reconstruction certificate: full nodal `Phi_s`
  recovers EPI modulo constants when `rank(-K L_rw)=N-1`, and one conserved
  zero-mode scalar completes the state. The rank condition is measured, without
  universal promotion, on 142 connected graph-atlas cases through six nodes;
  extreme weighted stars expose the separate numerical-conditioning limit.
- Added a graph-specific explicit-Euler relaxation diagnostic that reports the
  heterogeneous modal factors, stability limit and solver-step count while
  preserving the distinction from U4's operator-position policy.
- Derived a common-metric stability theorem for arbitrary switching among a
  finite family of connected symmetric topologies on fixed node support when
  all normalized `d_i/nu_i` metric vectors are exactly common. The certificate
  now separates exact equality from caller-tolerance proximity.
- Derived the exact finite-gain criterion for declared affine EPI resets in a
  common diffusion metric: the linear map and offset must preserve the
  consensus subspace. Added exact zero-energy counterexamples for failure, an
  upward-rounded rational weighted-Frobenius gain bound computed on the
  represented binary64 coefficients, and an exact rational log/exp enclosure
  for the hybrid flow/reset budget that separates finite-horizon contraction,
  repeated-word disagreement decay and preservation of the initial weighted
  consensus. Added a reproducible heterogeneous two-node example covering an
  absorbed amplification, uniform consensus drift and an exact local-offset
  obstruction. Operator names remain metadata rather than assumed gains.
- Hardened heterogeneous and switching diffusion certificates against
  binary64 reciprocal, accumulation and Laplacian-row residuals. The executable
  proof now rationalizes the materialized generator and displayed metric,
  requires the canonical uniform fixed-point identity in addition to the weaker
  consensus-subspace condition, proves quotient dissipation by rational
  LDL/inverse-norm bounds, and keeps eigensolver rates and exact weighted-mean
  preservation separate. Hybrid composition consumes only the downward-rounded
  certified rate and rebuilds affine jump proofs from their declared inputs.
- Unified scalar EPI semantics across raw numbers, live `BEPIElement` values and
  canonical or JSON mappings. Uniform finite real BEPI embeddings retain their
  signed scalar coordinate; genuinely nonuniform or complex payloads retain a
  nonnegative magnitude projection for generic read-outs and are rejected by
  pure-EPI scalar certificates. Empty BEPI values now have a defined zero
  magnitude without inventing a signed representative.
- Added the first two runtime-to-theorem operator bridges, for local Reception
  and Resonance. One neutral kernel now fixes their unweighted neighbour mean
  and blend arithmetic. The Reception certificate distinguishes the ideal-real
  map, represented coefficient map, and current runtime result, abstains outside
  convex inactive-hard-clip scope, and proves that every nontrivial local EPI
  update requires pure-EPI pressure refresh. Partial graph `GLYPH_FACTORS` now
  merge with canonical defaults, and Reception's fallback uses the centralized
  `EN_MIX_FACTOR`.
- Added the read-only Resonance realization certificate and aligned the RA
  runtime with its U3 and identity contracts. Only individually phase-compatible
  neighbours contribute to EPI, phase, or the frequency trigger. RA validates
  the convex EPI mix, nonnegative frequency boost, bounded phase coupling, and a
  phase gate no weaker than the canonical `pi/2` limit before mutation. Its four
  reported layers are the ideal-real blend, represented binary64 map, two-stage
  binary64 proposal, and accepted identity-gated snapshot. Scalar EPI may change,
  but a strict negative/positive crossing and replacement of an established
  nonempty `epi_kind` are rejected independently; exact zero is neutral and an
  absent kind may be initialized. Rejection is atomic for state and operator
  tracking.
- Added fixed post-RA diffusion and hybrid-recovery diagnostics. A local
  frequency boost generally changes `h_i=d_i/nu_i`, so the certificate retains
  the fixed post-RA theorem but abstains from a pre/post common-metric switching
  claim unless the represented metrics are exactly proportional. It reports the
  pressure-refresh defect after an accepted EPI change and explicitly declines
  global binary64 runtime affinity; recovery estimates remain display-only while
  Boolean recovery decisions use the exact hybrid composer.
- Hardened the sampled S16 trajectory certificate so forward-Euler residuals
  and local and cumulative common-Lyapunov variation decisions compare exact
  rational quantities with the represented caller tolerance. Rounded floats
  remain display diagnostics and can no longer promote a boundary failure.
- Added an exact reversible pure-EPI coarse-graining certificate. It constructs
  the quotient nodal generator, measures unresolved within-block dynamics and
  reuses the structural-morphism classification instead of adding an operator.
- Derived the local coherence levels as nonsmooth L1 diamonds, separating the
  exact constitutive distance from any unchosen global information metric.
- Added an exact decoupled metriplectic product for harmonic substrate pulse and
  EPI relaxation; nonzero physical cross-coupling remains open.
- Added a contract-identifiability certificate and proved that instantaneous
  channel/direction/scale/context features cannot distinguish Silence from
  Contraction, preserving temporal sequence identification as an open problem.
- Corrected phase-transition classification to use `abs(mean(S))` and
  `abs(mean(chi))`. The previous mean-absolute inputs could label cancelling
  local chirality as global homochirality; local magnitudes remain available as
  separate telemetry. Removed universal-divergence language from its scope.
- Reclassified the legacy phase `symmetry_zscore` as an operational
  standardized spatial imbalance because coupled graph nodes do not supply an
  independent-sample significance test. Added strict time-series validation
  and exposed node-count and classifier-input trajectories.
- Added a balanced finite-size phase-scaling diagnostic with replicate standard
  errors and slopes against node count. It deliberately leaves conversion to
  thermodynamic exponent ratios and universality open.
- Added descriptive radial/annular/multinodal transition certificates with
  exact tetrad, flow, coherence and Si endpoint deltas. Label changes are
  sample-bracketed observations, not interpolated bifurcations or predictors.
- Added a generic operator-quotient certificate separating projected autonomy
  from lift invariance. Matrices receive global finite-dimensional residuals
  with explicitly tolerance-conditioned decisions; nonlinear callables receive
  sampled fiber-dependence and repeatability evidence. Graph/history/nesting
  mutations are rejected as outside that model.
- Proved the fixed directed pure-EPI identity `p'=-Lp` for pressure and exposed
  the logarithmic-norm sign as the exact Euclidean transient-growth criterion.
  A deterministic 16-graph calibration/holdout benchmark separates it from
  stable-spectrum prediction while keeping all magnitude claims finite-family.
- Added an exact seven-channel structural-state quotient metric for finite simple
  graphs within a declared topology/label class. It is invariant under node
  relabeling and circular phase wrapping, includes effective edge conductance
  and structural length, requires positive channel scales (with a documented
  compatibility fallback for the length scale), and reports nodal-equation
  residuals without silently repairing snapshots.
- Separated algebraic EPI observability from numerically reliable reconstruction:
  the potential and augmented observers now use independent scale-aware SVD
  thresholds, and reconstruction success also requires a small relative residual.
- Added quantitative one-step operator signatures. All 13 canonical operators
  are distinct on the declared deterministic probes, resolving the categorical
  Silence/Contraction collision. Added the exact finite-prototype noise margin:
  nearest-row recovery is unique for additive error strictly below half the
  minimum scaled pairwise separation. Unseen-state, stochastic and word-level
  inverse identification remain open.
- Replaced the phase-transition example with a reproducible finite-size
  diagnostic and corrected the synchronization example and ontology text so
  finite crossovers, selected warning margins and auxiliary normal forms are
  not presented as universal TNFR critical laws.
- Corrected the Yang–Mills Y1–Y4 scope: `Phi_s²/(pi/2)²` is a
  single-snapshot magnitude penalty `V_Phi`, and the legacy `u6_*` sweep fields
  are compatibility names for that magnitude coordinate. They now expose
  `u6_drift_assessed=False`; a canonical U6 verdict requires a matching
  reference snapshot. Grammar counts now include only applicable checks.
- Established that the implemented connection `A=d(arg Psi)` is pure gauge:
  its oriented cycle sums vanish analytically and its covariant Laplacian is
  unitarily equivalent to the ordinary weighted graph Laplacian. Reclassified
  curvature, vortex, confinement and dynamical gauge-language as unsupported by
  that construction; any non-zero cycle value is a numerical closure residual.

### Changed — 2026-09-06 research integration and scope hardening

- Added the preferred phase-topology vocabulary `WindingSector`,
  `classify_winding_sector()` and `Network.winding()` while retaining the
  historical particle-named aliases. Winding now requires an explicit finite
  phase on every declared cycle node, derives an order automatically only when
  the whole graph is a simple cycle, reports unavailable whole-graph telemetry
  as `None` with a reason, and keeps integer winding separate from the legacy
  continuous `Q` bilinear.
- Made `Network.nfr()` refuse to fabricate an equilibrium from incomplete
  state. It now records whether `dEPI/dt` came from node telemetry or was
  reconstructed from the nodal equation, and returns an unavailable read-out
  when the required finite fields are absent.
- Hardened primality, counting, coherence and tolerance entry points against
  booleans, fractional counts, strings and non-finite values. Public counting
  APIs no longer truncate numeric inputs silently.
- Recast the finite shell classifier as an explicit auxiliary graph model.
  Shell capacities, orbital degeneracies, spin factor, closure sequence and
  label map are declared assumptions; even spectral clusters expose no angular
  index, and the preferred `shell_closure_distance` is the single source for
  closure and legacy reactivity metadata.
- Reclassified both N-body implementations as explicit adapters: one embeds a
  Newtonian central-force law and the other uses a selected phase-coupled pair
  law. Added finite-input validation and corrected history allocation so a
  final state is retained when `store_interval` does not divide the step count.
- Replaced universal operator, Noether, gauge, variational, particle, chemical
  and cross-domain claims by the exact finite certificate or auxiliary-model
  scope actually implemented. Historical public names remain as compatibility
  aliases where removal would break callers.
- Corrected the fixed-delay REMESH study to a finite cyclic DFT diagnostic:
  fixed modes are selected by the delay gcd, the lcm sets sample alignment, and
  changes between finite windows are leakage measurements rather than an
  infinity-limit kernel, convergence rate or catalog-completeness result.
- Changed the operator-contract audit verdict to finite deterministic probe
  coverage. Passing all probes confirms those declared fixtures and contracts;
  it does not establish universal operator behavior or full U1-U6 compliance.
- Separated structural geometry from transport conductance: an explicit edge
  `length` now controls structural-potential paths, while `weight` remains
  conductance and the legacy length fallback. Both channels invalidate caches.
- Hardened the research certificates against false numerical promotion. Exact
  switching metrics are compared before floating normalization, logarithmic-norm
  signs inside backward error abstain, and nodal topology labels accept only the
  calibrated inverse-square kernel.
- Corrected structural morphism classification so automorphisms must actually
  intertwine the generator and rank is invariant under global matrix scaling.
  Coarse-graining now rejects disconnected or zero-capacity macro quotients;
  the metriplectic product is independent of graph insertion order and reports
  stored-pressure consistency separately.
- Added an executable S16 endpoint certificate that composes stability, EPI
  reconstruction, reversible quotient closure, structural distance, pure-EPI
  pressure and nodal-equation consistency under one shared hypothesis set.
  Added `CoreExperimentManifest` as domain-neutral research provenance, with a
  validated Git revision, immutable version map, explicit clean/dirty state and
  a required SHA-256 source digest for dirty working snapshots. Arithmetic
  factor/bit semantics remain in the historical manifest.
- Extended S16 from frozen endpoints to finite sampled pure-EPI trajectories.
  Every interval now combines the endpoint evidence with a persistent-node
  forward-Euler defect and its modal stability limit; the full path requires an
  exact common switching metric and non-increase of the observed common
  Lyapunov functional. First failures remain chronologically inspectable.
- Separated modal spectral resolution from the dimensionless EPI residual
  tolerance, made the eigenvalue cutoff relative to the fastest decay rate,
  and added a whole-path positive-variation budget so small local energy
  increases cannot accumulate silently.
- Added a strict coarse/fine comparison with independent path and agreement
  tolerances. It requires nested common times and a smaller fine mesh width,
  an explicit caller declaration that both meshes represent the same dynamics,
  uses direct persistent-id EPI errors for the verdict, and leaves the
  relabeling-quotient distance as a diagnostic. The reproducible fixed-generator
  example additionally checks both meshes against the exact linear semigroup;
  it reports finite two-mesh evidence without claiming convergence or order.
- Centralized scoped Git/source hashing in `current_git_source_provenance()` so
  research examples share one clean/dirty provenance implementation.
- Prevented boolean state values from becoming physical zero/one scalars and
  made coherence means stable at finite extreme magnitudes. Winding-word
  observation now rejects absent or invalid phases before mutation. The
  transient U2 certificate preserves its serialized legacy fields while
  rejecting unimplemented metric labels.
- Made both N-body adapters reach a requested off-grid final time with a short
  last step. Newtonian pair distances now use scaled norms so representable
  forces at extreme separations are not erased by intermediate overflow, and
  the TNFR adapter validates body counts and mass-vector shape consistently.
- Centralized circular phase comparisons across operator telemetry, lifecycle,
  structural identity, U3 validation and transition postconditions. Shortest-
  arc distance now handles wrap boundaries and arbitrary complete-turn
  representatives; requested directed phase shifts remain separate telemetry.

### Added — 2026-09-05 post-review roadmap (R1–R9 continuation, N00–N13)

Twelve PRs implementing the external post-review canonical plan on top of the
R1–R9 research program. Every PR carries an exact/invariance test suite, a
benchmark with a C5 reproducibility manifest, a theory-doc section and an honest
claim-ledger transition. **U2/U6 in AGENTS.md were not modified** (the canonical
gate was never met), **no 14th operator** was invented, and no classical open
problem is claimed. Full ledger:
[theory/STRUCTURAL_RESEARCH_PROGRAM.md](theory/STRUCTURAL_RESEARCH_PROGRAM.md).

- **R9 directed non-normal dynamics** (`src/tnfr/physics/directed_diffusion.py`,
  new `transient_u2.py`, `heterogeneous_vf.py`): the stationary `L²(π)` metric
  layer and signed-vs-total U2 integral readings (N03); the scalar-`ν_f`
  **structural-time theorem** `x(t) = e^{−s(t)L}x₀` with clock-invariant
  reorganization and a finite bound `J ≤ M‖LQ‖‖x₀‖/ω` (N04); the transient U2/U6
  certificate showing the non-consensus dynamics **contracts** in the Euclidean
  per-node energy (`peak = 1`) — the naive ambient `>1` is exactly the oblique
  projection factor `‖Q‖` (N05); and the heterogeneous-`ν_f` boundary where the
  clock-change theorem stops (non-commuting generators, N13).
- **R1 symmetry-sector observability** (`operator_equivariance.py`, new
  `word_equivariance.py`, `pointed_symmetry.py`): per-copy graph-cache isolation
  fixing a cross-operator leak (N01); the **word composition-closure theorem**
  (equivariant operators compose to equivariant words, N06); and the **pointed
  selector** structure — a localized action performs a declared reduction
  `Aut(G) → Γ_v` (orbit–stabilizer, residual sectors, origin conjugation, N07).
- **R4/R8 structural morphisms** (new `structural_morphism.py`, `remesh_audit.py`):
  a **morphism taxonomy derived from the nodal equation** — an intertwiner
  `M L_src = L_tgt M` is exactly a nodal-flow transport, and 6 of 7 kinds emerge
  from `∂EPI/∂t = ν_f·ΔNFR` (N08); and the REMESH contract audit closing R4b as an
  honest negative — the p-adic tower lift is a projection morphism that fails only
  the temporal echo (N09).
- **R5 trace collisions** (new `trace_collisions.py`): loss of observability under
  the field trace, with an exact character formula (Fourier inversion) and a
  Galois-invariant collision histogram (N10).
- **R2 pulse amplitudes** (new `pulse_amplitudes.py`): the pointed-circulant pulse
  amplitudes are the normalized spectral multiplicities `a_λ = m_λ/n` (N12).

### Changed — 2026-09-05

- **R7 arithmetic-pressure canon** (`src/tnfr/mathematics/arithmetic_pressure.py`):
  removed the "minimal and complete" claim for the three pressure channels; added
  `ArithmeticPressureVector` and an exact functional-independence proof over ℚ.
  The scalar `delta_nfr_value` aggregation and API are preserved (N02).
- Documentation resynced: `theory/README.md` R1–R9 index, `AGENTS.md` §12/§13 and
  its verbatim mirror, and the per-line theory notes now reflect the N00–N13
  results and claim ledger.

### Fixed — 2026-09-04 canonicity audit (invariants #1, #2, #4, #5, #6)

- **Directed ΔNFR orientation (invariant #1).** The fused/vectorized EPI channel
  now realizes the canonical *outgoing* random-walk Laplacian `L_out = I − D⁻¹W`
  (node *i* receives the weighted mean of the nodes it points to), matching
  `structural_diffusion_operator`. The fused kernel previously accumulated
  predecessors (`L_in = L_outᵀ`), so directed graphs diffused the wrong way.
  Undirected graphs are unchanged (both orientations coincide).
- **Edge weights in the EPI channel (invariant #1).** Edge weights now reach the
  fused ΔNFR EPI channel (weighted neighbour mean + weighted degree). Unit
  weights reproduce the unweighted result bitwise; a live per-call weight read
  makes weight mutations take effect on the next computation.
- **U3 is a hard invariant for UM and RA (invariant #2).** The phase gate
  `|φ_i − φ_j| ≤ Δφ_max = π/2` now runs on every Coupling/Resonance application,
  independent of `VALIDATE_OPERATOR_PRECONDITIONS`, and raises before any state
  mutation (it was previously skipped by default; RA treated it as a 1.0-rad
  warning). Quality preconditions remain configurable.
- **Single ξ_C kernel (invariant #5).** `telemetry()` and `tetrad()` now both use
  `estimate_coherence_length` (with the spectral-gap fallback), so they agree and
  never return NaN on a connected graph.
- **Example 122 basis-invariance (invariants #4, #6).** The phase-sector
  factor-coset read now uses a basis-invariant eigenspace-projector score
  `‖P_d Π_λ‖²` plus a derived-tolerance invariant-subspace residual certificate
  (`τ = √ε·‖L‖`); the non-canonical `η² > 0.9` rule (which mis-fires on
  n = 209, 253, 299) is withdrawn. Complexity is stated in input bits `log₂ n`;
  no factoring speedup or cryptographic claim.
- **Dependencies / docs.** `cachetools` range widened to `>=5.0,<8.0` (only the
  stable `cached`/`LRUCache` APIs are used). The factorization-lab README labels
  `[UM, RA, IL]` as a grammar *fragment* and shows the complete
  `[AL, UM, RA, IL, SHA]` word.

### Added — the emergent pulse read-outs (the rhythm the substrate plays)

- **The pulse, surfaced at both scales.** The conservative face of the nodal
  dynamics is a *sustained vibration*; it is now a first-class read-out.
  `compute_emergent_pulse` / SDK `net.rhythm()` give the **collective** network
  pulse (resonances `ω_k = √λ_k`, fundamental, dominant beat, vibration energy);
  `compute_nodal_pulse` / SDK `net.resonance()` give the **per-NFR** pulse — every
  NFR a phase oscillator at its own `νf` and phase `φ`, coupled by *resonance*
  (`local_phase_sync` per NFR, the Kuramoto order `R`, gate `Δφ_max = π/2`). The
  collective pulse emerges as the per-NFR pulses lock (`R → 1`).
- **The pulse in motion.** `net.pulse_trajectory(steps)` evolves a copy and records
  the rhythm forming over time — `R(t)`, `C(t)`, the per-NFR local resonance —
  surfacing the **local-before-global** synchronization cascade (clusters lock
  before the global rhythm). `net.evolve(record=True)` + `net.history()` surface the
  engine's own canonical per-step series (`kuramoto_R`, `C_steps`, `phase_sync`,
  `Si_mean`).
- **Dual-face telemetry.** `compute_unified_telemetry` now carries both the
  dissipative read-out (canonical tetrad + coherence, relaxes to `ΔNFR = 0`) and the
  conservative `pulse` + `resonance` blocks (which do not saturate).
- **The pulse beyond the physics network (benchmarks).** `emergent_fractal_pulse.py`
  — resonance locks **scale by scale** on a self-similar network (the temporal face
  of U5); `emergent_arithmetic_pulse.py` — the residue-NFR pulse **tone-count is the
  proved cyclotomy law** `s_k(p) = gcd(k, p−1) + 1`, so a prime is its most
  degenerate chord and the factorization type is the chord size. The pulse is woven
  into `EMERGENT_ONTOLOGY.md` (§2.1/§2.2/§5.5) and `TNFR_NUMBER_THEORY.md` (§9.12).
- **The music of the NFR (music as a lens on structural frequency).**
  `emergent_musical_nfr.py` reads the **structural-frequency** spectrum (`νf`,
  `ω_k = √λ_k`, `Hz_str`) through music — *not* audio. The **dynamical regime
  follows the dimension**: a **1D thread** (a string) is *harmonic* and its just
  consonances (octave/fifth/fourth) **are** emergent (pitched); a **2D+** form (a
  drum) is *inharmonic* (unpitched), where consonance is the U3 phase gate;
  **polyphony = primes** (the Euler product); and the **Kac wall** — isospectral
  NFRs share the pulse, so it hears the *type*, not the identity (`Fix(G)^⊥`). Only
  **equal temperament** / the chosen scale are imposed. Woven into
  `EMERGENT_ONTOLOGY.md` §5.5 + the `form → dimension → dynamics` synthesis (§2.1).

### Changed — TNFR–Navier–Stokes re-founded on the two-face reading

- **The NS program was re-founded on the current paradigm.** The previous
  diffusive-face enstrophy-budget program — the `u2_compliance` structural-time
  benchmark, the N1–N14 milestone examples (77–86, 104, 105) and the old
  `operator.py` — was retired. New foundation `src/tnfr/navier_stokes/`: a faithful
  lean pseudo-spectral 3D integrator (`TNFRNavierStokes`, rotational form, exact
  Leray projection, integrating-factor RK2) + `conservative_face.py`
  (`verify_diffusive_face`, `face_of_flow`, `vorticity_modal_spectrum`,
  `measure_cascade_frontier`).
- **The honest two-face reading.** Incompressible NS is first order, so its *linear*
  part is the **diffusive (over-damped) projection** of the substrate wave
  (`ν_f = ν`; `verify_diffusive_face` VALID for every physical viscosity, recovering
  `ν_f = ν`). Blow-up is therefore a purely **nonlinear `K_φ` cascade** (the
  vortex-stretching VAL source), not a linear resonance — unlike oscillatory data
  (EEG), which sits on the under-damped conservative face.
- **The blow-up frontier, measured** (`measure_cascade_frontier`, example 158): at
  matched structural time `τ_str = ν·t` every run saturates at fixed Re (the
  diffusive face regularises) and the peak enstrophy debt grows with Re
  (1.00 → 1.04 → 1.68 at Re 126/314/628). The `Re → ∞` cascade bound = Clay, **open**.
- **HONEST SCOPE:** closes nothing; global 3D NS regularity stays open.

### Added — the empirical-confrontation pipeline (TNFR-IA → engine)

- **`examples/10_applications/159_empirical_confrontation_pipeline.py`** packages the
  empirical arm's workflow with engine primitives: map a multichannel signal onto the
  emergent phase-locking graph, read the canonical magnitudes (the pulse, the tetrad,
  `ξ_C`, Kuramoto `R`), and **diagnose its face** with the engine's own
  `verify_overdamped_projection` — making the theory falsifiable against data (the face
  is *measured*, not assumed). Cross-program face map: oscillatory data → conservative
  face; linear NS → diffusive face.

### Changed (emergent derivation — the grammar temporal windows from the pulse)

- **The U4b/U2 grammar windows are now derived, not assumed.** A destabilizer's
  `|ΔNFR|` perturbation relaxes geometrically under the discrete nodal step
  (`q = 1 − νf·dt·ρ`, with `ρ = trace(L_rw)/N = 1`, exact). Read two ways from the
  same `q`: the **relaxation time** to the coherence band `1/(π+1)` is the **U4b
  recency window** (and the `GRAMMAR` repeat-avoidance window) = **3**; the
  **geometric absorption capacity** `⌊1/(1−q)⌋` is the **U2 debt threshold** = **2**.
  New `derive_bifurcation_window_from_physics` / `derive_u2_debt_capacity_from_physics`
  (`config/physics_derivation.py`) replace the literal `3`/`2` — with **no `e`** (the
  canonical relaxation is the *discrete* geometric decay `qⁿ`, not the continuous
  exponential `e^{−νf λ t}`, which is only the `dt → 0` limit the engine never takes).
  The earlier **graduated destabilizer split** (strong = 4 / moderate = 2) was a
  heuristic the dynamics does not support and has been **dropped** — one emergent
  window for every destabilizer (the relaxation rate `ρ = trace/N = 1` is
  topology-independent, so the window is a topology-independent constant).

### Performance

- **Topology-keyed spectral cache.** `structural_eigenmodes` and `relaxation_spectrum`
  now share one memoized eigendecomposition of the symmetric normalized Laplacian,
  keyed on the graph topology (self-invalidating when nodes/edges/weights change).
  The spectrum is invariant under evolution on a fixed graph, so the O(N³) `eigh`
  runs once per topology instead of once per pulse/spectrum read-out.

### Changed (historical derivation — channel weights and nominal operator coefficients from π)

- **Replaced the residual magic numbers on the nodal-physics paths with values
  derived from π** (the sole structural scale), enforced by
  `tests/core_physics/test_emergent_constants_guard.py`.
  The φ/γ/e purge had left two load-bearing weight sets **frozen at their literal
  φ/γ decimals** (`DNFR_WEIGHTS`/`SI_WEIGHTS` = `{0.737, 0.155, 0.09}` where
  `0.737 = φ/(φ+γ)`) and had replaced the nominal operator policy
  coefficients with arbitrary
  "operational" decimals (`IL=0.75, OZ=2.0, SHA/NUL=0.9, VAL=1.05`). These are
  used *numerically* in every ΔNFR and Sense-Index evaluation, hence in every
  recorded result. They are now emergent:
  - **Channel-mixing weights → the coherence-band hierarchy.** Each
    structurally-active channel takes the high-coherence share `π/(π+1)` of the
    remainder: `(π/(π+1), π/(π+1)², 1/(π+1)²)` — which **normalises to exactly 1**
    (`π/(π+1) + π/(π+1)² + 1/(π+1)² = (π+1)²/(π+1)² = 1`). Ordering by structural
    primacy (phase ≻ EPI ≻ νf; topo inactive). `SI_WEIGHTS` takes the same hierarchy.
  - **Nominal operator policy coefficients → the coherence band and the π-fraction ladder.** Pressure
    lever (ΔNFR): `IL = π/(π+1)`, `OZ = (π+1)/π` (a balanced `IL∘OZ` is **exactly
    isometric**). Capacity lever (νf, slow): the gentle π-step `δ = 1/(4π)` —
    `SHA/NUL = 1−δ`, `VAL = 1+δ`, `NUL_densification = 1/(1−δ)` (volume
    conservation). Secondary couplings on the π-fraction ladder (`1/(4π), 1/(2π),
    1/(8π)`); ZHIR θ-shift `1/π`; `NAV_eta`/`REMESH_alpha` = the unit midpoint `0.5`.
  - **Selection, feedback & adaptation → π/band (no more operational decimals on
    the coherence paths).** `SELECTOR_WEIGHTS` takes the same coherence-band
    hierarchy; the coherence triggers are the high-coherence gate `π/(π+1)`, the
    new rectified-mean level `2/π`, and the unit midpoint/quarter `0.5`/`0.25`;
    `AU_CURVATURE` is the exact midpoint `(0.9π+π)/2` of the strict K_φ gate and the
    π wrap; the phase couplings, `FEEDBACK` tolerances/rates, `OZ` noise and `THOL`
    metabolic weights are π-fractions (`1/(2π), 1/(4π), 1/(8π)`); the `get_factor`
    safety fallbacks reference the emergent constants. The selector *magnitude*
    thresholds (`dnfr_hi/lo`, `accel_hi/lo`) are honestly left **operational**
    (|ΔNFR|/∂²EPI scale, not coherence — π-flavouring them would repeat the φ/γ/e
    naming-convention error).
  - New single-source constants in `constants/canonical.py`:
    `CHANNEL_WEIGHT_PRIMARY/SECONDARY/TERTIARY`, `COHERENCE_RETENTION`,
    `DISSONANCE_AMPLIFICATION`, `COUPLING_GENTLE/MODERATE/FINE`,
    `MID_COHERENCE_THRESHOLD`. Full suite green (`2201 passed`) after each stage;
    U2 grammar checks remained passing, which did not establish a universal
    dynamical boundedness theorem. **Recorded research results computed with the
    old constants still require recomputation (planned Stage 5).**
  - **Benchmark/example φ/γ/e input purge.** Fixed a broken example
    (`examples/02_physics_regimes/37_operator_tetrad_synergy.py` imported the purged
    `GAMMA`/`PHI` from `constants/canonical` → `ImportError`; examples aren't in the
    test suite so it had slipped through) — it now runs. Updated the
    `coherence_projector_sense_index` benchmark `SI_WEIGHTS` to the band hierarchy,
    removed dead `PHI/GAMMA/E` constants from `boundary_vibration`, de-refuted the
    `phase_wall` correspondence comments (its TEST-4 obstruction result — building
    `φA+γL+πL²+eK` to *prove* the four constants are insufficient — is kept), and
    replaced ~27 stale "(φ,γ,π,e) remain the assumed substrate" claims with "π"
    across 14 benchmark files. The legitimate emergent-*object* studies are kept
    (the Kuramoto φ-as-Fibonacci-limit, the golden-angle sphere sampling, Euler
    products, the Γ chirality matrix, tetrahedral symmetry groups).
  - **Recomputation & robustness (the canonical-emergence proof).** Re-running the
    paradigm results under the emergent engine changed **no headline verdict** —
    because each is *structural*, not an artifact of the magic numbers: primality
    (`ΔNFR=0`), Riemann σ_c/GUE and exact S_n equivariance (`‖[L, P_σ⊗P_τ]‖ = 0`),
    Navier–Stokes (a **pseudo-spectral** solver that never reads the operator
    gains), conservation, the scoped phase linearization
    (`K_φ ≈ L_rw·φ`) and the `1/√λ₂` spectral comparison/fallback for fitted
    `ξ_C`, and
    the Yang–Mills finite potential-magnitude diagnostic derives from graph and
    structural-potential data,
    S_n symmetry, or unit arithmetic. Only the dynamic *trajectories* (C(t)/Si
    curves, network-optimization outcomes) shift, with their qualitative attractors
    invariant. The φ/γ/e and arbitrary operational decimals were therefore **never
    load-bearing**: the refactor both cleans the foundation and *proves* the results
    are genuinely emergent under the stated formulas.
  - **Constitution guard (Stage 0) + the last comment straggler.** A new regression
    guard (`tests/core_physics/test_emergent_constants_guard.py`) pins the channel
    weights, operator gains and coupling ladder to their exact π-formulas and
    asserts that **no nodal-physics constant equals a removed frozen φ/γ/e decimal**
    — so the obsolete constants cannot creep back. Cleaned the final φ/γ/e *comment*
    straggler the input purge had missed in `src/`: the extended nodal system's
    `_compute_phase_transport_derivative` (`dynamics/canonical.py`) no longer cites
    the dead `≈ 0.618 = 1/φ` / `0.155` / `0.135` origins, nor the false "RECALIBRATED
    from canonical constants" / dead "Import canonical constants" comments (values
    unchanged — honest operational magnitudes on the optional J_φ-transport path).
    The constants regression guard records the completed migration.

### Changed (documentation aligned to emergent π-derived canonicity)

- **Centralized the documented π-scaled monitoring policies**
  across `AGENTS.md` (+ the `.github/agents/my-agent.md` mirror),
  `ARCHITECTURE.md`, `CONTRIBUTING.md`, `theory/`, `docs/grammar/`, examples, and
  code docstrings: the selected Φ_s policies use π as a reference scale — drift
  `Δ Φ_s < π/2 ≈ 1.571` (half phase-wrap) and per-node `|Φ_s| < π/4 ≈ 0.785`
  (quarter phase-wrap). These are policies rather than consequences of phase
  wrapping, and they replace the old φ ≈ 1.618 / empirical 0.7711 framing;
  the strong-coherence cut is the emergent band gate `π/(π+1) ≈ 0.7585`
  (replacing the frozen `(e·φ)/(π+e) ≈ 0.7506`). Corrected a propagated arithmetic
  error: `π/(π+1)` is **0.7585**, not 0.7616 (it must complement `1/(π+1)=0.2415`).
  The SDK `COHERENCE_STRONG` now aliases the emergent `HIGH_COHERENCE_THRESHOLD`
  (π/(π+1)); `MIN_BUSINESS_COHERENCE` (0.75) stays the separate operational
  business-health knob.
- **Removed `theory/SPIRAL_ATTRACTORS_AND_LOGARITHMIC_DYNAMICS.md` and its demo
  (`examples/02_physics_regimes/32_spiral_attractors_demo.py`)** — a false
  φ/γ/e-era claim ("golden ratio as dynamical attractor", "fourth constant γ").
  The demo imported the purged φ/γ/e constants (so it could not run, making the
  document's "Validated" status false) and the golden-attractor check was circular
  (it set `b = 2·ln(φ)/π` by hand, then "verified" quarter-turn ratios = φ). Only
  the trivial, non-distinctive kernel (log spirals appear in a rotation + growth
  regime, with a free `b = νf·k/ω`) was true; φ is not selected by the dynamics.
  References cleaned from `theory/README.md`, `FUNDAMENTAL_THEORY.md`, and
  `examples/README.md`.
- **Completed a full `theory/` document audit** (every `theory/*.md`) for residual
  φ/γ/e false claims, with **no further deletions needed** — SPIRAL_ATTRACTORS was
  the only doc with a false *thesis*; the rest are genuinely emergent and carried
  only scattered stale refs (now fixed). The most significant correction purges the
  refuted **"Universal Tetrahedral Correspondence"** (the φ↔Φ_s, γ↔|∇φ|, π↔K_φ,
  e↔ξ_C mapping) from `TNFR_RIEMANN_RESEARCH_NOTES.md` (20 references) — the explicit
  mapping becomes the canonical **structural-field diagnostic tetrad** (only π is structural), the
  three inter-prime coupling kernels are relabeled *exploratory, not canonical*, and
  the stale `DNFR_/SI_/SELECTOR_WEIGHTS` derivation claims/anchors are corrected to
  the operational `defaults_core.py` values. Operator-gain tables across
  `STRUCTURAL_OPERATORS`, `STRUCTURAL_CONSERVATION_THEOREM`,
  `STRUCTURAL_STABILITY_AND_DYNAMICS`, `TNFR_VARIATIONAL_PRINCIPLE`,
  `TNFR_YANG_MILLS_RESEARCH_NOTES` (+ 2 `yang_mills/structural_gap.py` docstrings),
  `CATALOG_TYPE_HYGIENE_PROGRAMME`, and `TNFR_NUMBER_THEORY` were updated from frozen
  φ/γ/e formulas (e.g. IL `φ/(φ+γ)≈0.737`→`0.75`, OZ `φ/γ≈2.803`→`2.0`, NUL
  densification `2.803`→`1/λ≈1.111`, U6 `Δ Φ_s < φ`→`π/2`, `|∇φ|` heuristic
  `γ/π`→`π/16`) to the operational engine values. No engine code changed (the code
  was already purged; only doc text and 2 cosmetic docstrings).

### Changed (operational-knob relocation — `canonical.py` is now pure physics)

- **Split the ~150 operational engine-tuning knobs out of
  `constants/canonical.py` into a new dedicated module
  `constants/operational.py`** (explicitly *engine tuning, NOT TNFR physics*).
  `canonical.py` now holds **89 numeric constants**, all genuine structural /
  physics quantities (π phase-wrap bounds, spectral-gap ξ_C, the coherence band,
  operator gains, tetrad / phase / νf / EPI / KL / DT scales); the **150** moved
  knobs (caches, FFT tuning, optimization speedup/performance estimates,
  pattern-discovery confidence, integration baselines, operator scoring weights)
  live in `operational.py`. The new module imports only `PI` from canonical
  (one-way dependency; canonical never imports operational), and a parallel
  `engines/constants/operational.py` star-shim mirrors the existing `canonical`
  shim. The `canonical ∪ operational` union reproduces the pre-split constant set
  **exactly** (verified name→value, 0 leaks / 0 drift). 26 consumer modules were
  redirected; mixed importers were split to preserve their structural imports.

### Removed (φ/γ/e purge — only π remains a genuine structural scale)

- **Removed the obsolete constants φ (golden ratio), γ (Euler–Mascheroni), and
  e (Napier) from the engine.** They are no longer canonical constants, appear in
  no calculation, weight, threshold, or comment, and the "(φ,γ,π,e) notational
  vertex / four-constants / assumed-substrate" framing is retired. **Only π is a
  genuine structural scale** (the phase-wrap bound of the phase sector:
  `|∇φ| ≤ π`, `|K_φ| ≤ π`; `0.9·π` is a warning margin). The fitted coherence
  length has `1/√λ₂` as a connected-graph spectral comparison/fallback; every
  other parameter is derived under stated hypotheses or is operational.
- **Φ_s monitoring policies now use centralized π-scaled values**: per-node
  `PHI_S_VON_KOCH_THRESHOLD = π/4 ≈ 0.785` (quarter phase-wrap) and drift
  `U6_STRUCTURAL_POTENTIAL_LIMIT = π/2 ≈ 1.571` (selected before/after drift
  policy), replacing the empirical `0.7711` / golden-ratio (`φ ≈ 1.618`)
  framing. Neither is a graph-independent potential bound.
- **Removed `derive_tetrad_threshold_values`** and the `φ/γ/e` accumulation-law
  threshold-derivation machinery (`ThresholdDerivation`). Operator gain magnitudes
  are now plain operational parameters — the theory fixes each operator's channel
  and sign via its contract, not its magnitude.
- **Re-derived the live physics constants** from π / nodal / spectral quantities,
  de-dressed the engine-configuration tier (cache, FFT, optimization, performance
  knobs) to plain operational values, and purged the `φ/γ/e` references from
  source comments, docstrings, and the documentation set (`ARCHITECTURE.md`,
  `README.md`, `CHANGELOG.md`, `.zenodo.json`, `CONTRIBUTING.md`,
  `benchmarks/README.md`, and the `theory/` + `docs/` notes).

### Changed (emergent-canon consolidation — frozen φ/γ/e values re-derived)

- **Audited every constant** for emergent grounding.
  The purge had left the numeric *values* frozen (e.g. `K_TOP_FALLBACK` still held
  `2.803171 = φ/γ`); those magic numbers are now re-derived or eliminated so the
  canonical base is genuinely emergent.
- **Genuine emergent derivation** — the prime-detection threshold
  `MATH_DELTA_NFR_THRESHOLD = 0.5` is the unit-gap midpoint: with unit arithmetic
  ΔNFR coefficients, `prime ⟺ ΔNFR = 0` exactly and every composite has
  `ΔNFR > 1`, so any cut in `(0, 1)` separates them.
- **π-derived**: `MAX_STRUCTURAL_FREQUENCY = 2π`, `MIN_STRUCTURAL_FREQUENCY = 1/(2π)`,
  `AU_CURVATURE_PERMISSIVE = 0.96·π`, `CRITICAL_EXPONENT = GRAD_PHI_CANONICAL_THRESHOLD = π/16`,
  `DYNAMICS_SI_HI = π/(π+1)`, the `K_TOP` clamp `1/(8π) … 1.0` and fallback `π`.
- **Removed the non-physical / vestigial** arithmetic-recalibrated trio
  (`PHI_S_THRESHOLD`, `GRAD_PHI_THRESHOLD`, and `K_PHI_THRESHOLD = 3.2275`, which
  *exceeded* the π phase-wrap bound and was therefore an unreachable no-op check).
- **Eliminated the dead domain constants** (`MEDICAL_*`, `BUSINESS_*`, `EXAMPLE_*`,
  `VIZ_*`, `CLI_*`, `THERAP_*`, `SCRIPT_*`, `TOOL_*`, `UTILS_*`) and the dead
  `CANONICAL_CONSTANTS` registry; relocated the SDK builder defaults into
  `sdk/builders.py`. The remaining ~180 operational engine knobs were rounded to
  plain ≤2-decimal values (dropping the false φ/γ/e precision). `constants/canonical.py`
  shrank from ~770 to ~565 lines.
- Reconciled inline operator gains (`operators/__init__.py`) to the canonical
  `SHA_VF_FACTOR` / `NUL_SCALE_FACTOR` / `VAL_SCALE_FACTOR`, and removed residual
  inline artifacts (`10·φ`, `e`, `4/(e+φ)`) in `bifurcation.py`, `variational.py`,
  `cycle_detection.py`, and `signatures.py`.

## [0.0.3.5] - 2026-06-24 — Tetrad correspondence audit & emergent redesign

A computational audit of the "Universal Tetrahedral Correspondence" found that
only **π** is a genuine structural scale; the four-constant correspondence
(φ↔Φ_s, γ↔|∇φ|, e↔ξ_C) is mostly an **organizing overlay**. Several thresholds
asserted as "derived" were empirical, inert (magic), or measured false. This
work corrects the claims and replaces magic thresholds with emergent,
system-measured quantities. The nodal equation, the 13 operators, and grammar
U1–U6 are unchanged.

### Corrected (canonicity claims)

- **Only π is a genuine structural scale** — the phase-wrap bound shared by BOTH
  `|∇φ|` and `K_φ` (both are means of wrapped angles, ≤ π). γ, e, φ are
  recoverable as mathematical identities but are NOT the structural scales of
  their tetrad fields. `K_φ` matches `L_rw·φ` only on a smooth, consistent
  unwrapped branch with matching normalization; the reported correlation is a
  finite-protocol measurement. The exact wrapped computation is nonlinear;
  fitted `ξ_C` uses `1/√λ₂` as a spectral comparison/fallback (not base e).
- **`|∇φ|` bound corrected** from `γ/π ≈ 0.1837` to the phase-wrap bound `0.9π`
  in `physics/variational.py`, symmetric with `K_φ`. The measured synchronization
  onset is ≈ 0.29 and σ-dependent, NOT the constant γ/π; γ/π is retained
  elsewhere only as a heuristic early-warning level, explicitly labelled
  non-derived.
- **`derive_tetrad_threshold_values`** rows re-statused: π `geometric`; φ, γ, e
  `overlay` (recoverable identities, not structural scales).
- **`ARCHITECTURE.md`, `.zenodo.json`, `CONTRIBUTING.md`, `theory/README.md`,
  `docs/STRUCTURAL_FIELDS_TETRAD.md`, `constants/canonical.py`** — removed the
  "Universal Tetrahedral Correspondence foundation / 100% derived / zero
  empirical tuning / verified to machine precision" claims; replaced with the
  honest tiering (π genuine; γ/e/φ notational overlay).
- **Second audit pass (repo-wide)** — removed the remaining "Universal
  Tetrahedral Correspondence / canonical derivation / Kuramoto critical
  coupling" claims and internal contradictions across `ARCHITECTURE.md`
  (the "Mathematical Purity / 497 magic numbers eliminated / zero empirical"
  sections), `README.md`, `theory/FUNDAMENTAL_THEORY.md`, `theory/GLOSSARY.md`,
  `theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md`, `AGENTS.md` (+ mirror),
  `docs/grammar/PHYSICS_VERIFICATION.md`, and the central constants modules
  (`telemetry/constants.py`, `mathematics/unified_numerical.py`,
  `config/defaults_core.py`, `physics/signatures.py`,
  `operators/grammar_telemetry.py`, `physics/emergent_chemistry.py`) plus four
  benchmarks. Exposed cosmetic "derivations" (e.g. `MIN_BUSINESS_SENSE_INDEX =
  1/φ + 0.082 ≈ 0.700`, `⌊φ×10⌋ = 16`) as calibrated/notational values.

### Changed (emergent replacement of magic thresholds)

- **`physics/phase_transition.py` redesigned** around a standardized signed-mean
  ratio. The "universal critical exponent γ_c = γ/π" was **measured false**
  (the fitted exponent is protocol-dependent), and the classification noise
  floor `(γ/π)²` was **proven inert** (it sat in a two-order-of-magnitude gap;
  sweeping it changed no classification). Removed the magic constants `GAMMA_C`,
  `ORDER_PARAMETER_NOISE_FLOOR`, `CHIRALITY_THRESHOLD` and the
  `theoretical_exponent` field; added `symmetry_zscore(mean, var, n) =
  |mean|/√(Var/N)` and the selected cut `Z_SIGNIFICANCE = 1`. The later S6
  audit recorded above corrects its interpretation: coupled graph nodes do not
  justify an independent-sample significance test, so the ratio is an
  operational spatial-imbalance diagnostic.
- **Named γ/π constants relabelled** as heuristic / non-derived in
  `constants/canonical.py` (`CRITICAL_EXPONENT`, `GRAD_PHI_CANONICAL_THRESHOLD`,
  `PHASE_GRADIENT_THRESHOLD_CANONICAL`) and `mathematics/unified_numerical.py`;
  `gauge.py`, `emergent_chemistry.py`, `interactions.py` regime thresholds
  marked calibrated/heuristic, not derived.

### Notes

- **Third audit pass (repo-wide, exhaustive)** — removed the remaining
  "Universal Tetrahedral Correspondence / canonical derivation / zero empirical
  fitting" claims across the two subprojects (`primality-test/`,
  `factorization-lab/`), examples, benchmarks, the `mathematical_purity` tests
  (now check genuine bounds, not the refuted mapping), all theory docs
  (`FUNDAMENTAL_THEORY.md §4` and `GLOSSARY` renamed to "structural-field
  tetrad"), and ~40 in-code combo comments (`defaults_core.py`, `bifurcation.py`,
  `cycle_detection.py`, `number_theory.py`, etc.) now marked notational. Exposed
  the primality coefficients (ζ=φγ, η=(γ/φ)π, θ=1/φ) as combos chosen to
  approximate the original empirical values (ζ=1.0, η=0.8, θ=0.6).
- The gauge force-regime classification and emergent-chemistry excitation scale
  were flagged here for full emergent rework; that rework is now complete — see
  "Emergent redesign" below.
- Full test suite green (2196 passed) after all three passes.

### Consequences audit (computational impact, 2026-06-21)

A measure-first review of whether *calculations* (not just narrative) depended
on the refuted tetrad values. Main finding: **the significant results are
robust**, because they emerge from STRUCTURE (integer orderings, exact zeros,
relative scores) rather than the scale values (γ/π, etc.):

- **Emergent chemistry** (periodic table, magic numbers 2/10/18/36/54/86,
  octet): the aufbau filling order uses only the integers (n+l, n); the octet
  is the exact ΔNFR=0 zero — both independent of any scale coefficient. The
  `nu_excitation` (=γ/π), `nu_0`, `coherence_gap` fields were DEAD (defined,
  never consumed) and were removed; only `theta_valence` enters, as a positive
  scale (the zero is robust to its value).
- **Number-theory primality** (n prime ⟺ ΔNFR=0): each pressure term vanishes
  individually for primes (Ω−1, τ−2, σ/n−(1+1/n) are all 0), so the result is
  independent of the coefficients ζ, η, θ.
- **Gauge interaction regimes**: `dominant_regime` is decided by relative
  scores, NOT the γ/π threshold; `above_threshold` (which used γ/π) is metadata.
- **Riemann ζ-bridge buffer** γ/π: a regularisation shift whose exact value is
  immaterial. **K_φ asymptotic exponent** α≈2.76: a measured fit, unrelated.

Real consequences corrected:
- `PHASE_CURVATURE_ABS_THRESHOLD = φ×π ≈ 5.083` was a **non-physical** K_φ bound
  (|K_φ| ≤ π by phase wrap, so any check using it was a no-op); it was dead code,
  corrected to `0.9π ≈ 2.827`.
- `STRUCTURAL_STABILITY_AND_DYNAMICS.md §2.2` still described the old γ_c
  classification table; updated to the emergent z-score rule.
- Removed the dead chemistry scale parameters; fixed a residual
  `MATHEMATICAL_DYNAMICS_BASIS.md` |∇φ| claim. Full suite green (2196 passed).

### Emergent redesign (gauge regimes + chemistry excitation, 2026-06-21)

Completed the full emergent rework of the two studies whose conceptual base was
the (now-refuted) four-constant overlay. Both were rebuilt to rest on STRUCTURE
alone (measure-first; no value replaced by another magic value):

- **Gauge interaction-regime classification** (`physics/gauge.py`): removed the
  three overlay threshold constants `REGIME_DOMINANCE_THRESHOLD` (1/φ),
  `REGIME_STRONG_THRESHOLD` (γ/π, "Kuramoto critical coupling in gauge") and the
  unused `REGIME_SECONDARY_THRESHOLD` (γ/(π+γ)). The per-sector `above_threshold`
  activity flags now use a single parameter-free criterion: a sector is *active*
  when its normalised score exceeds the equipartition share `1/N_REGIMES = 0.25`
  (the maximum-entropy reference, derived from the number of gauge sectors — the
  four structural channels of the tetrad). Uniform across all four sectors; no
  overlay constant. New public symbols `N_REGIMES`, `REGIME_ACTIVITY_SHARE`
  replace the removed thresholds. `dominant_regime` (relative `max` of scores)
  is unchanged — it was already robust. Measured: the criterion always flags the
  dominant sector and additionally marks genuine co-active secondaries.
- **Emergent-chemistry valence scale** (`physics/emergent_chemistry.py`): removed
  the last free scale parameter (`theta_valence = 1/φ`) and the now-trivial
  `EmergentChemistryParameters` dataclass. `ΔNFR_chem(Z)` is now the **integer**
  structural distance of the outer shell to a closed configuration, in natural
  units (one subshell step = 1) — the exact chemical analogue of primality
  `ΔNFR(n)=0`. Noble gases (2,10,18,36,54,86) → ΔNFR=0; halogens/alkali → 1;
  oxygen → 2; carbon → 4. Magic numbers and the octet are unchanged (they always
  emerged from the integer (n+l) ordering and the exact zero).
- **Measured (chemistry):** tested whether the (n+l) filling order could emerge
  from the raw Laplacian spectrum of a concentric multi-shell ("onion") manifold.
  It does NOT — Madelung ordering reflects electron-electron screening absent
  from a free graph Laplacian. (n+l) is therefore documented honestly as an
  integer excitation-count rule (total radial+angular quanta), not a spectral
  derivation and not a constant correspondence.
- Tests updated (`test_gauge.py`: `TestRegimeActivityCriterion`, equipartition
  consistency). Full suite green (2195 passed, 2 skipped).

## [0.0.3.4] - 2026-06-17

This release consolidates the emergent-geometry program, centralizes the
operator/grammar/contract layer onto single canonical sources, opens three new
TNFR-native Millennium-problem programs, and refactors the documentation to the
current engine state. The 13-operator catalog, grammar U1–U6, and the nodal
equation are unchanged; everything below either *measures* structure the nodal
equation already contains or removes duplication. Full suite: 2043 passed, 2
skipped.

### Symplectic Substrate (historical entry; current scope corrected)

This release introduced an auxiliary ambient **symplectic phase space**
initialized from graph-field snapshots. Later audits established that its exact
harmonic-flow identities do not derive the full nodal equation or certify the
engine operators.

- **New module**: `src/tnfr/physics/symplectic_substrate.py` — phase space
  `P = ℝ^{4N}` with conjugate pairs `(K_φ, J_φ)` (geometric) and `(Φ_s, J_ΔNFR)`
  (potential); symplectic 2-form `ω` (antisymmetric, non-degenerate, closed);
  canonical Poisson brackets; `H_sub = ½Σ(K_φ²+J_φ²+Φ_s²+J_ΔNFR²)` equal to the
  structural-energy snapshot exactly after adding the held-fixed gradient
  background; Liouville `div(X_H)=0` for the specified harmonic flow. Each
  engine operator would require a separate Jacobian pullback test.
- **Auxiliary model structure tower** (each measured to machine precision):
  Noether charges (time-translation → `H_sub`; geometric U(1) → `E_geo = ½Σ|Ψ|²`;
  potential U(1) → `E_pot`); the compatible Hermitian / flat-Kähler triple
  `(ω, J, g)` with `J = −ω` — so the `i` in `Ψ = K_φ + i·J_φ` *is* the complex
  structure the substrate induces; complete integrability (action–angle,
  Liouville–Arnold); Poincaré–Cartan integral invariants; Marsden–Weinstein
  symplectic reduction; and the hidden **U(2) polarization symmetry** whose
  SU(2) part supplies three conserved **Stokes parameters** on the per-node
  Poincaré sphere (classical wave polarization — Stokes 1852 / Poincaré 1892 —
  not isospin or qubits).
- **Historical threshold overlay**: `derive_tetrad_threshold_values` reconstructs
  φ, γ, and e from chosen mathematical identities. They are organizational
  overlays rather than derived TNFR field scales; π remains the phase primitive.
- **Consolidated entry point**: `verify_substrate_geometry(G)` bundles all
  certificates into a `SubstrateGeometryReport`.
- **SDK**: `Network.symplectic_substrate()` + `SymplecticReport`, in
  `TNFR.analyze()`.
- **Honest scope**: a flat, constant-coefficient linear Kähler backbone — a
  consolidation of geometry already implied by `conservation.py` +
  `variational.py`; it does not resolve any open program.
- **Demonstrations**: `examples/08_emergent_geometry/98`, `106`, `114`.

### Emergent Geometry — Structural Diffusion (transport layer)

The EPI channel of the canonical ΔNFR is the random-walk graph Laplacian
`−L_rw·EPI` (verified to residual ~1e-16), so the nodal equation is literally a
discrete diffusion equation with diffusivity `νf`. From this single identity the
engine measures, in TNFR's own variables, a tower of empirically-established
transport phenomena.

- **New module**: `src/tnfr/physics/structural_diffusion.py` — six transport
  layers: diffusion/synchronization (Fourier/Fick/Kuramoto), overdamped drift
  (`q̇ = νf·F`, Stokes/Einstein mobility — corrects the prior "Newton's second
  law" reading: the bare first-order nodal equation is overdamped, νf is mobility
  not inverse mass), discrete standing-wave modes (bounded-manifold Laplacian
  eigenmodes), structural-stability dispersion relation (`σ_k = r − νf·λ_k`, the
  spectral form of U2), random walk + effective resistance (Ohm/Kirchhoff), and
  structural flow (current, Kirchhoff continuity, Ohm).
- **Overdamped-projection bridge**: the nodal equation is the strong-damping
  limit of the substrate wave `q̈ + γq̇ + Lq = 0` with `νf = 1/γ`; the γ-dial
  spans diffusion (γ→∞) to standing waves (γ→0).
- **Honest scope**: the EPI-channel ↔ Laplacian identity is exact; the full ΔNFR
  is multi-channel; `λ_2` is purely topological and does not encode any canonical
  constant (measured negative result).
- **Demonstrations**: `examples/08_emergent_geometry/99`, `113`, `134`, `135`.

### Operator Contracts & Energy — Centralization and Emergence

- **Canonical contract layer**: new `src/tnfr/operators/operator_contracts.py` —
  the single source of truth for what each operator does to node state, anchored
  to the direct `_op_*` effect (TNFR.pdf §2.2.1). Each `OperatorContract` records
  the public English name, the `primary_channel` (one nodal-equation channel:
  EPI / νf / θ / ΔNFR), the `scale` (NODE for twelve operators, NETWORK for the
  U5 operator REMESH), and a verifiable postcondition. The proactive audit
  (`audit_operator_contracts`), the reactive integrity monitor (`POSTCONDITIONS`),
  and the introspection metadata now all derive from this spec — eliminating the
  historical drift where scattered copies disagreed (e.g. AL claiming "positive
  ΔNFR" though `_op_AL` only raises EPI; RA checked for EPI increase though it
  preserves identity; VAL/NUL checked |EPI| though they scale νf).
- **Public English names**: the structural-operator name (Emission, Reception, …)
  is canonical at the public level; the glyph code (AL, EN, …) is the internal
  symbol.
- **Energy/coherence are emergent**: the structural candidate energy
  `E = ½Σ(Φ_s²+|∇φ|²+K_φ²+J_φ²+J_ΔNFR²)` contains no EPI or νf term (measured:
  scaling EPI or νf leaves E unchanged). The per-operator Lyapunov role in
  `physics/lyapunov.py` is therefore re-derived from the canonical grammar U2 role
  (`config.physics_derivation`), not from a hardcoded energy algebra:
  stabilisers {IL, THOL}, destabilisers {OZ, ZHIR, VAL}, the rest neutral. The
  form-channel operators (AL, EN, RA, REMESH) leave a same-snapshot evaluation
  unchanged when all derived fields are held fixed because EPI is absent from
  the formula. This is a dependency statement, not an isometry or a conservation
  law after pressure and fields are recomputed.
- **Dual-lever clarified**: the two levers are the two right-hand-side factors of
  the nodal equation — νf (capacity) and ΔNFR (pressure); operators that write
  the form EPI (the LHS) sit on neither lever.
- **Demonstrations**: `examples/08_emergent_geometry/152`,
  `examples/02_physics_regimes/115`.

### Grammar — Single Canonical Source & Formal-Language Characterization

- **Centralization**: the operator-classification sets (generators, closures,
  stabilizers, destabilizers, transformers, bifurcation triggers/handlers) are
  derived once in `config.physics_derivation` and re-exported by
  `operators/grammar_types.py`. Every grammar consumer — the U1–U6 validator,
  the secondary sequence validator, grammar_dynamics, the runtime preconditions,
  the error factory, and the operator metadata — now reads the single source.
  Parallel hardcoded copies (including a secondary validator that wrongly listed
  NUL as a U2 destabilizer) were removed and pinned by
  `tests/operators/test_grammar_canonical_consistency.py`.
- **Canonical grammar spec**: new `operators/grammar_canon.py` materializes the
  U1–U6 role table, the five-type structural typology, and the canonical glyphic
  macros (anchored to TNFR.pdf §2.3), with a self-consistency check.
- **Formal-language thread** (characterization, demos only): the flat,
  non-nested default-depth history projection is represented by 320 reachable
  states and a 52-state minimal complete DFA. Its numerical growth radius is
  `10.9560791442`; both U2 and U4b reduce the radius. The 944-element syntactic
  monoid is aperiodic (index 4), so this constructed flat language is star-free
  / first-order definable. Nested `THOL[...]` syntax requires a context-free,
  stack-like model, while runtime U3 and reference-dependent U6 remain outside
  the DFA. The Parry process is a selected maximum-entropy Markov policy on the
  dominant component, not a TNFR physical equilibrium or grammar validator.
- **Demonstrations**: `examples/08_emergent_geometry/139`–`152`.

### Number Theory & the Dual-Lever

- Prime families as orbits on the zero-pressure set `{ΔNFR = 0}`; numbers as a
  coupled network (Ω-graded centrality, primes as the transport periphery); the
  nodal flow on numbers (primes as equilibria, not attractors); primality as
  grammatical inertness; numbers as free-monoid words with the dual-lever as the
  two additive gradings (count Ω → ΔNFR pressure, size log → νf capacity); the
  capacity arm carries von Mangoldt and the prime-ladder Hamiltonian P14 is the
  capacity-arm operator — locating the Riemann oscillatory obstruction on the
  capacity axis the per-node substrate is blind to.
- **Honest scope**: these restate classical multiplicative number theory through
  the grammar/dual-lever lens; they close no open problem.
- **Demonstrations**: `examples/07_number_theory/94`–`97`, `100`–`102`,
  `116`, `146`–`149`.

### Millennium Problem Programs (TNFR-native reformulations)

Three new programs join Riemann / Navier–Stokes / Yang–Mills. **None claims a
solution** — each carries an explicit honest-scope statement and classified
obstruction.

- **P vs NP (PNP-1)** — the nodal equation is a gradient flow, so verifying a
  configuration's coherence is `O(|E|)` but synthesizing a globally coherent one
  by relaxation traps in dissonance basins (measured global-optimum hit rate
  drops monotonically with problem size on frustrated MAX-CUT). Mirrors P≠NP;
  Branch B open. `theory/TNFR_P_VS_NP_RESEARCH_NOTES.md`,
  `examples/09_millennium/109`.
- **Birch–Swinnerton-Dyer (BSD-1)** — `a_p = p+1−#E(F_p)` as structural pressure;
  the accumulated product reproduces the original 1965 empirical rank separation
  by brute-force point counting. GL(1)→GL(2) gap open; Branch B.
  `theory/TNFR_BSD_RESEARCH_NOTES.md`, `examples/09_millennium/110`.
- **Hodge (HC-1, scope corrected under Unreleased)** — an auxiliary simplicial
  complex reproduces the standard finite Hodge decomposition and expected Betti
  numbers. Canonical `|∇φ|` and `K_φ` are node summaries, not oriented edge and
  face cochains, so no tetrad cochain tower has been derived. The `(p,p)` and
  algebraicity gap remains outside this baseline. `theory/TNFR_HODGE_RESEARCH_NOTES.md`,
  `examples/09_millennium/111`.

### Documentation, Examples & Repository Hygiene

- **README + core theory docs** refactored to the current engine state: corrected
  a real API note (operators are callable, there is no `.apply()`), added the
  emergent-geometry section to `theory/FUNDAMENTAL_THEORY.md`, rewrote the energy
  classification in `theory/STRUCTURAL_OPERATORS.md` /
  `theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md` to the emergent/grammar-U2 frame,
  and updated counts.
- **Examples reorganized** into 10 thematic subfolders (`01_foundations` …
  `10_applications`), resolving the prior 77–86 numbering collision; each file
  keeps a stable global number. Foundational examples refactored to the canonical
  Kuramoto phase-synchrony physics.
- **Documentation-integrity pass**: repaired all dangling example/source/`.md`
  links repo-wide, pruned 9 obsolete `docs/` files, rebuilt the `theory/` hub,
  and resynced the derived `.github/agents/my-agent.md` mirror.
- **Deep repo cleanup**: removed foreign GraphQL scratch JSONs, a CUDA debug
  script, a backup test, an empty dead CLI module, and a stale task tracker; fixed
  a dangling `tnfr-validate` console-script entry point in `pyproject.toml`.
- **Registry consolidation + lint**: the SDK fluent glyph→operator map and the
  lyapunov operator table now derive from the canonical registries; cleared a
  small set of dead-code lint findings.
- **SDK fixes**: repaired silent no-ops in `auto_optimize` and
  `evolve_grammar_aware`; added a proactive measured operator-contract fidelity
  audit (`net.audit_operators()`).

### Research Program Milestones (Yang–Mills Y1–Y5, REMESH-∞ N15, Navier–Stokes N16–N17)

The Yang–Mills (Y1–Y5) and REMESH-∞ / Navier–Stokes (N15–N17) program
milestones below were developed earlier in the cycle and are part of this
release. Each carries an explicit honest-scope statement; none resolves a Clay
Millennium Problem.

#### Y5 — TNFR–Yang–Mills Closure / Obstruction Classification

- **Verdict**: `BRANCH_B_OBSTRUCTION_CLASSIFIED` — Y1–Y4 establish a finite TNFR `U(1)` structural gauge diagnostic surface, but Clay-strength closure requires a new canonical non-Abelian derivation plus a continuum / thermodynamic lower-bound theorem.
- **New API**: `classify_yang_mills_closure()` in `src/tnfr/yang_mills/closure.py`, exported from `tnfr.yang_mills` with `YangMillsClosureReport`.
- **Finite TNFR branch**: `A_FINITE_U1_DIAGNOSTIC_SURFACE` when Y4 reports stable finite positive gaps.
- **Clay-strength branch**: `B_REQUIRES_NEW_CANONICAL_NONABELIAN_DERIVATION` because Y3 remains `OPEN_DERIVABILITY_GAP`.
- **Scope discipline**: `clay_problem_resolved = False`; the obstruction is localized, not removed.
- **Validation**: 4 new tests in `tests/physics/test_yang_mills_closure.py` cover Branch-B classification, report reuse, sampled collapse handling, and package-root import. Y1–Y5 focused run: `33 passed`.
- **Next target**: Y6 / Branch-B derivation search for a TNFR-native non-Abelian connection and non-commuting generator algebra. If no derivation exists without external group labels, the programme should pause at Branch B.

#### Y4 — TNFR–Yang–Mills Finite Scaling Diagnostic

- **Verdict surface**: `FINITE_SCALING_EVIDENCE` or `GAP_COLLAPSE_OBSERVED` depending on sampled finite graph families. This is a finite diagnostic only, not a continuum theorem.
- **New API**: `run_finite_scaling_study()` in `src/tnfr/yang_mills/scaling.py`, exported from `tnfr.yang_mills` with `FiniteScalingPoint` and `FiniteScalingReport`.
- **Scaling coordinate (historical name corrected under Unreleased)**: graph node count `n` under fixed single-snapshot magnitude ratios `ρ_U6 = max_i |Φ_s(i)|/(π/2)`; this does not assess U6 drift. Grouped reports fit finite log-log slopes of mean gap versus `n`.
- **Scope discipline**: Y4 runs while YMG-4 remains open. Therefore finite positive scaling evidence cannot be promoted to a Clay-strength Yang–Mills mass-gap claim.
- **Validation**: 6 new tests in `tests/physics/test_yang_mills_scaling.py` cover report shape/scope, grouped finite scaling, reproducibility, sampled collapse classification, invalid input rejection, and package-root import. Y1–Y4 focused run: `29 passed`.
- **Next target**: Y5 closure / obstruction classification, likely Branch B unless a later TNFR-native non-Abelian connection and generator algebra are derived.

#### Y3 — TNFR–Yang–Mills Non-Abelian Derivability Audit

- **Verdict**: `OPEN_DERIVABILITY_GAP` — audited candidate routes for deriving a non-Abelian / multi-channel gauge sector from TNFR-internal data only; no route is promoted to canonical status.
- **New API**: `audit_nonabelian_derivability()` in `src/tnfr/yang_mills/derivability.py`, exported from `tnfr.yang_mills` with `NonAbelianCandidateAudit` and `NonAbelianDerivabilityReport`.
- **Routes audited**: U5 nested-EPI multiplets, THOL/REMESH operator-history internal spaces, and graph cycle-basis bundles.
- **Obstruction**: current canonical `Ψ = K_φ + i·J_φ` gauge structure supplies a scalar local `U(1)` connection. Nested EPI or operator-history data do not yet derive component-mixing parallel transport or non-commuting generator algebra; cycle-basis routes require non-canonical basis/orientation selection.
- **Validation**: 5 new tests in `tests/physics/test_yang_mills_derivability.py` cover baseline `U(1)` confirmation, nested-EPI obstruction, cycle-bundle rejection, unsupported route errors, and package-root import. Y1+Y2+Y3 focused run: `23 passed`.
- **Open boundary**: YMG-4 remains open. Y4 scaling can proceed only as a conditional finite diagnostic; it cannot become a Clay-strength claim while non-Abelian derivability is unresolved.

#### Y2 — TNFR–Yang–Mills U6 Confinement Sweep

- **Verdict**: `EMPIRICAL_FINITE_GRAPH_ONLY` — finite sweep surface created for testing how the Y1 structural gauge gap behaves across two groups of the single-snapshot potential-magnitude coordinate; the original U6 wording was corrected under Unreleased.
- **New API**: `run_u6_confinement_sweep()` in `src/tnfr/yang_mills/u6_sweep.py`, exported from `tnfr.yang_mills`.
- **Sweep coordinate**: legacy `ρ_U6 = max_i |Φ_s(i)|/(π/2)`; values below one mean below that magnitude scale, not U6-confined. A U6 drift verdict requires a reference snapshot.
- **Telemetry recorded**: gap statistics, self-adjointness, seeded local-U(1)
  spectral invariance, pure-gauge cycle-closure residuals, applicable snapshot
  checks, legacy magnitude ratios, and finite-scope metadata. The cycle residual
  is numerical error, not curvature activity.
- **Validation**: 5 new tests in `tests/physics/test_yang_mills_u6_sweep.py` cover report shape/scope, U6 target tracking, gap contracts, reproducibility, invalid input rejection, and package-root import. Y1+Y2 focused run: `18 passed`.
- **Open boundary**: Y2 does not prove a U6 lower-bound theorem and does not address non-Abelian derivability (YMG-4) or continuum scaling (YMG-5). Next target: Y3 derivability audit.

#### Y1 — TNFR–Yang–Mills Finite Structural Gauge Gap Diagnostic

- **Verdict**: `DIAGNOSTIC_SURFACE_CREATED` — first TNFR-native Yang–Mills / structural mass-gap attack surface implemented as a finite-graph diagnostic, not a Clay-strength proof.
- **New package**: `src/tnfr/yang_mills/` with `build_structural_gauge_graph()`, `build_structural_gauge_gap_operator()`, and `compute_structural_gauge_gap()`.
- **Operator (current notation)**: `H_structural = L_A + V_F + V_Phi`, where
  `A=d(arg Psi)` makes `L_A` unitarily equivalent to the ordinary weighted graph
  Laplacian, `V_F` is analytically zero apart from numerical cycle-closure
  residuals, and `V_Phi` is the selected single-snapshot magnitude penalty
  `Phi_s²/(pi/2)²`; it is not a U6 test.
- **TNFR scope discipline**: the reported number is the spectral gap of this
  selected finite matrix. It is not evidence of dynamical gauge curvature,
  confinement, a physical mass, or a continuum Yang–Mills mass gap.
- **Validation**: 13 new tests in `tests/physics/test_yang_mills_structural_gap.py` cover graph construction, self-adjointness, non-negative finite gap reporting, seeded local-U(1) spectral invariance, reproducibility, package imports, and no EPI/phase mutation. Focused run: `13 passed`.
- **Open boundaries**: non-Abelian derivability (YMG-4) and continuum / thermodynamic scaling (YMG-5) remain open.
- **Documentation**: `theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md` records the Y-series gap ledger and updates the next target to Y2 (U6 confinement sweep).

#### N17-A — U3+U5 → K41: Analytical Cascade Locality (ANALYTICAL_CONSISTENT_CONDITIONAL)

- **Verdict**: `ANALYTICAL_CONSISTENT_CONDITIONAL` — K41 $k^{-5/3}$ spectrum derived conditionally from TNFR grammar rules U2+U3+U5+CDC; algebraically closed given the Cascade Development Condition.
- **Lemma U5-SS** (U5 + U2 → scale self-similarity): U5-uniformity (same canonical operators and constants at every hierarchy level) + U2 force $u_\ell = C(\varepsilon r_\ell)^{1/3}$ in the inertial range. The K41 scaling emerges from grammar structure (U5 collapses the dimensionless ratio to a level-independent constant), not from external dimensional analysis.
- **Lemma U3-CL** (U3 → cascade locality, conditional): Under Lemma U5-SS, U3 (phase-gated coupling, $|\phi_i - \phi_j| \le \Delta\phi_{\max}$) blocks all inter-level interactions **if and only if** the Cascade Development Condition (CDC) holds → constant energy flux $\Pi_\ell = u_\ell^3 / r_\ell = \varepsilon$ across scales.
- **Theorem** (U2 + U3 + U5 + CDC → K41): $E(k_\ell) \sim \varepsilon^{2/3} k_\ell^{-5/3}$ — proof: $E_\ell \sim u_\ell^2 \sim \varepsilon^{2/3} r_\ell^{2/3}$, $E(k_\ell) = E_\ell / \Delta k_\ell$ with $\Delta k_\ell \sim k_\ell$ (log bands) → $E(k_\ell) \sim \varepsilon^{2/3} k_\ell^{-5/3}$. □
- **CDC (irreducible gap)**: CDC (adjacent cascade levels have $|\phi_{\ell,i} - \phi_{\ell+1,j}| \ge \Delta\phi_{\max}$ for all $i,j$) is not derivable from U3, U5, or the nodal equation. It is the K41 locality hypothesis restated in TNFR language, and the structural analogue of $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$ in the Riemann programme — reachable only by a sufficiently developed turbulent cascade, not from the canonical operator catalog alone.
- **N17-A does not close NS-G1..G4** — those gaps concern continuum-limit, uniform bounds, BKM criterion, and vortex stretching; not cascade locality.
- **N17-B pre-registered** (deferred): empirical energy spectrum via `energy_spectrum_3d()` (to be implemented in `src/tnfr/navier_stokes/operator.py`), n ∈ {32, 48}, ν ∈ {0.01, 0.005}, T = 2.0. Expected verdict: `STEEPER_THAN_K41` (CDC not satisfied at Re_eff ≤ 500).
- **Documentation**: `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §20 (full lemmas, theorem, CDC gap analysis, verdict table, N17-B pre-registration spec).

#### N16 — NS-G5 Closure: 2D-Embedding Lemma

- **Verdict**: NS-G5 **CLOSED** at the discrete-operator level via the **2D-Embedding Lemma (Theorem NS-G5-TNFR)**.
- **Algebraic proof** (three steps using existing `TNFRNavierStokesOperator` methods on z-independent u = (u₀(x,y), u₁(x,y), 0)):
  1. `vorticity_3d`: ω₀ = ω₁ = 0, ω₂ = ∂_x(v) − ∂_y(u)
  2. `vortex_stretching_field`: S_a = ω₀·∂_x(u_a) + ω₁·∂_y(u_a) + ω₂·∂_z(u_a) = ω₂·0 = 0 for all a
  3. `stretching_production`: = 0.0 exactly in IEEE 754
- **TNFR reading**: z-channel decoupling → no cross-channel ΔNFR → enstrophy ≤ viscous dissipation (monotonically non-increasing) → discrete TNFR analogue of 2D NS global regularity.
- **Contrast with 3D**: ∂_z(u_a) ≠ 0 activates cross-channel ΔNFR coupling → stretching production generically positive → U2 (convergence/boundedness) is not guaranteed → vortex stretching amplification is structurally active.
- **Empirical corroborator**: `examples/85_navier_stokes_dimensional_asymmetry.py` — z-independence → `stretching_production` ≈ 0 at machine precision across all tested configurations (commit `1fac358b`).
- **Scope**: NS-G5 closure does NOT affect NS-G1..G4 and does NOT address the Clay Millennium Problem (3D global regularity).
- **Documentation**: `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §19.

#### N15 REMESH fixed-delay surrogate — corrected historical record

- **Historical milestones**: commits `a1f298fd`, `badac156`, and `48b0574a`
  introduced a history-space projection, projected charge/energy claims, and
  spectral comparisons under the label “REMESH-∞ closure.” The corrected
  analysis is maintained in
  [theory/REMESH_INFINITY_DERIVATION.md](theory/REMESH_INFINITY_DERIVATION.md)
  §§1–23.
- **Valid restricted result**: on a finite cyclic history window, the unclipped
  fixed-coefficient filter
  $F=\beta I+\gamma S^{\tau_l}+\delta S^{\tau_g}$ is a normal contraction for
  $0<\alpha<1$. Its Cesàro averages converge to the orthogonal projection onto
  $\ker(I-F)$. The common fixed modes have period
  $\gcd(\tau_l,\tau_g)$; the former `lcm` lattice statement was incorrect.
- **Runtime boundary**: `apply_network_remesh` is history-gated, configurable,
  structurally clipped, and uses a snapshot whose location changes with
  $\tau_g$. The finite cyclic result does not establish its literal
  $\tau_g\to\infty$ limit.
- **Invariant correction**: convex EPI mixing is not an isometry. Orthogonal
  projection contracts the surrogate history norm, but it neither proves
  conservation of TNFR structural charge nor monotonicity of the structural
  candidate energy. The former exact-conservation and universal $O(1/n)$ rate
  claims are superseded.
- **Registry scope**: the surrogate projection can be computed without adding
  an implementation registry entry. This does not prove that the 13 registered
  operators exhaust all admissible TNFR transformations; catalog completeness
  remains open.
- **External-program scope**: the finite surrogate does not advance RH,
  Navier–Stokes regularity, K41, or an RMT correspondence. Historical branch
  labels now apply only to comparisons made inside that declared surrogate.

## [0.0.3.3] - 2026-03-07

### Documentation Audit (Sessions 1-4)

- **Comprehensive tone audit**: Removed speculative/grandiose language across 25+ files
- **TNFR_RIEMANN_RESEARCH_NOTES.md**: Reduced from 2679 to 1499 lines (removed unfounded claims)
- **AGENTS.md**: Fixed 'inevitability' → 'derivation strength', updated conservation test count (62 → 88), verified all 40+ cross-reference links
- **Synced .github/agents/my-agent.md** with AGENTS.md corrections
- **Updated test counts** to 1,655 across 8 files
- **Removed orphaned file**: src/train_gmx_optimizer.py
- **Fixed contradictions** between AGENTS.md and theory/ documents
- **Validated**: 1653 passed, 2 skipped

## [0.0.3.2] - 2026-03-06

### Documentation & Consistency Fixes

- **Corrected false Γ(4/3)/Γ(1/3) derivation** in MINIMAL_STRUCTURAL_DEGREES.md and FUNDAMENTAL_THEORY.md (Γ(4/3)/Γ(1/3) = 1/3, not 0.7711)
- **Synchronized .github/agents/my-agent.md** with AGENTS.md (K_φ threshold, MIN_BUSINESS_COHERENCE, THOL_MIN values)
- **Fixed CHANGELOG version** to match pyproject.toml (0.0.3.2)
- **Fixed MIN_BUSINESS_COHERENCE precision** in ARCHITECTURE.md and CONTRIBUTING.md (0.751 → 0.7506)
- **Resolved phantom docs/TNFR_FORCES_EMERGENCE.md** references across 8+ files
- **Removed dead code** src/tnfr/config.py (shadowed by config/ package)
- **Cleaned unused imports** in sdk/simple.py

## [0.0.3] - 2026-03-05

### Structural Conservation Theorem

- **conservation.py**: Introduced a finite-trajectory structural-balance residual
  and Noether-like vocabulary. Later audits established that U1-U6 do not imply
  a vanishing residual or a universal conservation theorem.
- Added charge-density/current, structural-energy, Ward-residual, stability-policy,
  and spectral diagnostics; these names do not by themselves establish conserved
  charges, Noether hypotheses, or Lyapunov monotonicity.
- Two-sector structure: Potential (Φ_s ↔ J_ΔNFR) and Geometric (K_φ ↔ J_φ) coupled through Ψ = K_φ + i·J_φ
- 62 validation tests, charge drift < 0.03% across topologies

### Dissipative Conservation

- **dissipative_conservation.py**: GPU-accelerated dissipative conservation analysis with PyTorch backend
- Phase field computation, dissipation rate tracking, and energy budget monitoring

### Closed-Loop Integrity Monitor

- **integrity.py**: `StructuralIntegrityMonitor` with complete postconditions for all 13 canonical operators
- Each operator (AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH) has verified pre/postcondition contracts
- Automatic violation detection and reporting

### Grammar-Aware Dynamics

- **grammar_dynamics.py**: Bridge between grammar validation (U1-U6) and dynamic operator selection
- Incremental U1-U6 checks: `validate_candidate()`, `filter_candidates()`, `suggest_alternative()`, `enforce_grammar_on_glyph()`
- Priority-based operator substitution with fallback logic
- **grammar_application.py**: Pre-validation in `apply_glyph_with_grammar()` for grammar enforcement before operator application
- **selectors.py**: `_soft_grammar_prefilter()` wired with grammar_dynamics for operator filtering

### Simple SDK — Research-Grade Access

- **simple.py**: Upgraded with full Structural Field Tetrad, conservation laws, and unified telemetry access
- **TetradSnapshot** dataclass: phi_s, grad_phi, k_phi, xi_c, j_phi, j_dnfr with `is_safe()` and `summary()`
- **ConservationReport** dataclass: noether_charge, energy, lyapunov_stable, lyapunov_derivative, conservation_quality with `summary()`
- **10 new Network methods**: `tetrad()`, `fields()`, `conservation()`, `telemetry()`, `tensor_invariants()`, `emergent_fields()`, `evolve_grammar_aware()`, `integrity_check()`, upgraded `results()` and `info()`
- **TNFR.analyze()**: One-shot comprehensive analysis (coherence, tetrad, conservation, tensor invariants, emergent fields, integrity)
- Feature-gated imports: `_HAS_FIELDS`, `_HAS_CONSERVATION`, `_HAS_INTEGRITY`, `_HAS_GRAMMAR_DYNAMICS`
- 29 new tests in `tests/sdk/test_simple_advanced.py`

### Shared Test Infrastructure

- **tests/conftest.py**: Centralized test fixtures (`make_ring_graph`, `make_node_data`, `ring3`, `ring5`, `small_graph`)
- DRY reduction across 16+ test files that previously duplicated `_make_graph` helpers

### Code Quality

- Fixed bare `except:` clauses in grammar_dynamics.py (now `except Exception:`)
- NAV bypass fix for grammar validation edge case
- Redundancy elimination across physics helpers
- Rich operator postconditions (13/13 coverage)

### Cross-Codebase Constant Unification (Round 1)

- **grammar_types.py**: Eliminated duplicate operator sets (single canonical definition)
- **THOL_MIN_COLLECTIVE_COHERENCE**: Unified to canonical 0.2413 (was 0.3)
- **MIN_BUSINESS_COHERENCE**: Centralized to canonical formula (e×φ)/(π+e) ≈ 0.7506
- **health_analyzer.py / self_organization.py**: Aligned fallback values to canonical

### Phase Gradient Threshold Unification

- **Canonical value**: γ/π ≈ 0.1837 (Kuramoto critical coupling in TNFR units)
- **Unified across 9 code files**: Replaced competing values (0.2904, 0.2886, 0.2915, 0.38) with single canonical derivation
- **Updated 8 documentation files**: Consistent threshold references throughout

### Cross-Codebase Constant Unification (Round 2)

- **compute_structural_potential_field**: Added alias in physics/fields.py (was silently missing, imported in 2 files)
- **SHA_VF_FACTOR comment**: Fixed from ≈ 0.8476 to correct ≈ 0.9015 in defaults_core.py
- **Operator fallback values**: SHA (0.85→0.9015), NUL (0.85→0.9015), VAL (1.05→1.0676) aligned to canonical
- **K_φ hotspot formula**: Fixed in conservation.py from 2π/√5 ≈ 2.8099 to canonical 0.9×π ≈ 2.8274
- **grammar_core.py K_φ default**: Fixed from 3.0 to canonical 2.8274
- **telemetry/constants.py**: Removed dead try/except ImportError fallback; direct canonical imports
- **config.py**: Structural field thresholds now derive from constants.canonical (was hardcoded)
- **pyproject.toml**: Added mpmath to core dependencies (was required but unlisted)
- **Documentation sync**: Updated 7 doc files with correct threshold values and test counts

### Test Suite

- **1,655 tests** (1,646 passing, 9 skipped), 0 failing
- Coverage spans operators, physics, dynamics, grammar, conservation, integrity, SDK, and factorization

## [0.0.2] - 2025-11-29

### TNFR Development Doctrine Establishment

- **Foundational Principle**: Added TNFR Development Doctrine as core methodological commitment
- **Theoretical Integrity**: Commitment to follow mathematics objectively from nodal equation ∂EPI/∂t = νf · ΔNFR(t)
- **Scientific Independence**: Defend conclusions emerging rigorously from TNFR principles regardless of external paradigm alignment
- **Validation Criteria**: Established 4-point validation framework (Derivable, Testable, Reproducible, Coherent)

### Complete Framework Expansion

- **29 New Examples**: Comprehensive examples (11-39) covering physics, biology, cosmology, consciousness studies
- **TNFR-Riemann Program**: Complete theoretical framework connecting discrete operators to Riemann Hypothesis
- **Advanced Physics Modules**: Classical mechanics, quantum mechanics, symplectic integration implementations
- **Extensive Theory Documentation**: 25+ specialized theoretical documents in theory/ directory

### Documentation Academic Modernization

- **Unified Academic Tone**: Systematic elimination of grandilocuent language across all documentation
- **README Gateway**: Transformed main README into coherent documentation entry point
- **Consistent Terminology**: Standardized "Primary theoretical reference" replacing "SINGLE SOURCE OF TRUTH"
- **Professional Presentation**: Enhanced credibility through formal academic language standards

### Test Suite Optimization

- **Major Cleanup**: Removed 58 obsolete test files (82 → ~30 files)
- **100% Pass Rate**: Achieved 173 passing, 7 skipped, 0 failing tests
- **Focused Validation**: Retained only tests validating TNFR theoretical foundations
- **Core Coverage**: Mathematics, operators, physics, validation maintained

### Technical Enhancements

- **Enhanced N-body Dynamics**: Improved TNFR integration with classical mechanics
- **Riemann Operator**: Complete implementation with eigenvalue analysis capabilities
- **Type System**: Enhanced type definitions and structural validation
- **Code Quality**: Significant cleanup removing outdated components

## [9.7.0] - 2025-11-29

### Major Theoretical Enhancements

- **Universal Tetrahedral Correspondence**: Complete mathematical framework establishing exact mapping between four universal constants (φ, γ, π, e) and four structural fields (Φ_s, |∇φ|, K_φ, ξ_C) *(later superseded — see the φ/γ/e purge under [Unreleased]: only π is a genuine structural scale)*
- **Unified Field Framework**: Mathematical unification discovering complex geometric field Ψ = K_φ + i·J_φ with emergent invariants
- **Self-Optimizing Engine**: Self-optimization capabilities with unified field telemetry for automated structural optimization
- **Complete Academic Documentation**: Comprehensive conversion to formal academic tone across entire documentation ecosystem

### Canonical Invariants Optimization

- Consolidated from 10 to 6 canonical invariants based on mathematical derivation from nodal equation
- Optimized invariants: Nodal Equation Integrity, Phase-Coherent Coupling, Multi-Scale Fractality, Grammar Compliance, Structural Metrology, Reproducible Dynamics
- Enhanced theoretical consistency and reduced redundancy

### Documentation Modernization

- **AGENTS.md**: Complete academic conversion maintaining single source of truth status
- **README.md**: Restructured with new Getting Started section and clear learning paths
- **GLOSSARY.md**: Comprehensive expansion with Universal Tetrahedral Correspondence coverage
- Eliminated promotional language and emojis across entire ecosystem
- Updated all version references to 9.7.0

### Structural Field Tetrad

- **Complete Mathematical Foundations**: All four canonical fields now have rigorous mathematical derivations
- **CANONICAL Status**: Φ_s, |∇φ|, K_φ, ξ_C all promoted to canonical status with theoretical validation
- **Unified Complex Geometry**: Integration of curvature and transport via complex field Ψ

### Development Infrastructure

- Updated pyproject.toml to v9.7.0 with current dependency structure
- Modernized CONTRIBUTING.md with academic tone and current 6 invariants
- Enhanced TESTING.md with updated invariant validation framework
- Complete English-only policy implementation

## [9.1.0] - 2025-11-14

### Added

- Phase 3 structural instrumentation:
  - `run_structural_validation` aggregator (grammar U1-U3 + field thresholds Φ_s, |∇φ|, K_φ, ξ_C, optional ΔΦ_s drift).
  - `compute_structural_health` with risk levels and recommendations.
  - `TelemetryEmitter` integration example (`examples/structural_health_demo.py`).
  - Performance guardrails: `PerformanceRegistry`, `perf_guard`, `compare_overhead`.
  - CLI: `scripts/structural_health_report.py` (on-demand health summaries).
  - Docs: README Phase 3 section, CONTRIBUTING instrumentation notes, `docs/STRUCTURAL_HEALTH.md`.
- Glyph-aware grammar error factory (operator glyph → canonical name mapping).

### Tests

- Added unit tests for validation, health, grammar error factory, telemetry emitter, performance guardrails.

### Performance

- Validation instrumentation overhead ~5.8% (moderate workload) below 8% guardrail.

### Internal

- Optional `perf_registry` parameter in `run_structural_validation` (read-only timing).
- Canonical operator registry frozen (removed dynamic auto-registration, cache
  invalidation, metaclass telemetry, reload script). Attempting dynamic
  registration now raises. Ensures strict adherence to unified grammar (U1-U4)
  and prevents non-canonical transformations.

### Deferred

- U4 bifurcation validation excluded pending dedicated handler reintroduction.

### Integrity

- All changes preserve TNFR canonical invariants (no EPI mutation; phase verification intact; read-only telemetry/validation).
- Registry immutability strengthens invariants #1 (EPI only via operators), #4
  (operator closure) and #5 (phase verification untouched). Tests updated:
  removed dynamic registration tests; added `test_canonical_operator_set`.

## [9.0.2]

Previous release (see repository history) with foundational operators, unified grammar, metrics, and canonical field tetrad.

---
