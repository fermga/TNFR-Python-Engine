# Structural Stability and Dynamics

This document collects implemented stability diagnostics, phase-transition
read-outs, life telemetry, lifecycle classification, an auxiliary Hamiltonian
model, and integrity tools anchored to the nodal equation
$\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$. Exact
restricted results, selected policies, finite observations, and compatibility
models are identified separately.

**Status**: Mixed — exact restricted diffusion results, operational operator
classifications and measured diagnostics are identified separately.

---

## 1. Lyapunov Stability Analysis

### 1.1 Tetrad-plus-current structural energy

`compute_energy_functional` implements the non-negative quadratic diagnostic

$$
E[G] = \frac{1}{2}\sum_i \left[\Phi_s(i)^2 + |\nabla\phi|(i)^2 + K_\phi(i)^2 + J_\phi(i)^2 + J_{\Delta\mathrm{NFR}}(i)^2\right]
$$

This is the **tetrad-plus-current structural energy**: it contains three tetrad
fields $(\Phi_s,|\nabla\phi|,K_\phi)$ and the two currents
$(J_\phi,J_{\Delta\mathrm{NFR}})$. It omits the fourth tetrad field $\xi_C$, so
calling it the “tetrad energy” is inaccurate. It also contains no explicit EPI
or $\nu_f$ term, although recomputing the fields after a state change can alter
the value indirectly.

For general engine trajectories, $E$ is a **Lyapunov candidate**. The finite
difference returned by `compute_lyapunov_derivative` only reports whether two
supplied snapshots show non-increase within its tolerance. Grammar compliance
does not prove $dE/dt\leq0$. A proved Lyapunov result exists for the different,
weighted centered energy of restricted pure EPI diffusion; see
[the heterogeneous diffusion theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md).

The historical `coherence_matrix` is also outside this Lyapunov claim. It is
an auxiliary entrywise-bounded affinity built from phase, EPI, frequency and
Si, and it is distinct from canonical $C(t)$. Entrywise nonnegativity does not
imply a nonnegative spectrum after the graph-support mask: three identical
nodes on a path give $W=I+A_{P_3}$ with eigenvalue $1-\sqrt{2}<0$. The matrix
can supply a Hermitian term to the auxiliary Hamiltonian because it is real
symmetric; it is not a positive-semidefinite coherence observable.

### 1.2 Canonical channels and nominal U2 roles

The canonical operator channel is defined by
[`operator_contracts.py`](../src/tnfr/operators/operator_contracts.py). It is
distinct from the operator's U2 composition role:

| Primary nodal channel | Operators | Contract scope |
|-----------------------|-----------|----------------|
| **EPI (form)** | AL, EN, RA, REMESH | Write or propagate form |
| **$\nu_f$ (capacity)** | SHA, VAL, NUL | Decrease, increase, or remove capacity; NUL also densifies pressure |
| **$\Delta\mathrm{NFR}$ (pressure)** | IL, OZ, THOL, NAV | Stabilize, perturb, reorganize, or control pressure |
| **$\theta$ (phase)** | UM, ZHIR | Synchronize or transform phase |

Grammar U2 instead classifies IL and THOL as stabilizers; OZ, ZHIR, and VAL as
destabilizers; and the other eight operators as neutral with respect to U2
debt. Thus a phase-channel operator such as ZHIR can be a U2 destabilizer, and
a pressure-channel operator such as NAV can remain U2-neutral.

[`physics.lyapunov`](../src/tnfr/physics/lyapunov.py) maps those U2 roles to a
legacy compatibility model. Its public objects retain names such as
`OperatorLyapunovBound`, but the multipliers are **nominal policy values**, not
proved bounds on the structural energy above or on $C(t)$. The preferred API
names are `OperatorPolicyMultiplier`, `OPERATOR_POLICY_MULTIPLIERS`,
`U2PolicyRole`, `evaluate_sequence_policy`, and
`compare_operator_energy_to_policy`; the legacy names remain source-compatible
wrappers.

#### Stabilizer policy entries

| Operator | Canonical default input | Policy multiplier |
|----------|-------------------------|-------------------|
| **IL** (Coherence) | pressure retention $\pi/(\pi+1)\approx0.7585$ | $0.7585$ |
| **THOL** (Self-organization) | acceleration $1/(4\pi)\approx0.0796$ | $1-1/(4\pi)\approx0.9204$ |

#### Destabilizer policy entries

| Operator | Canonical default input | Policy multiplier |
|----------|-------------------------|-------------------|
| **OZ** (Dissonance) | pressure scale $(\pi+1)/\pi\approx1.3183$ | $1.3183$ |
| **ZHIR** (Mutation) | phase-shift factor $1/\pi\approx0.3183$ | $1+1/\pi\approx1.3183$ |
| **VAL** (Expansion) | capacity scale $1+1/(4\pi)\approx1.0796$ | $1.0796$ |

#### U2-neutral policy entries

AL, EN, UM, RA, SHA, NUL, NAV, and REMESH receive multiplier one because they
carry no U2 stabilizer/destabilizer debt. This value does not predict zero
observed change in phase-dependent or pressure-dependent energy fields.

The legacy `compute_operator_energy_bound` and `compute_sequence_energy_bound`
functions interpret their numeric input as an abstract policy score. Their
names do not confer energy-bound semantics. The legacy `within_bound` result of
`verify_operator_lyapunov` is a one-sided screen against that model and carries
`is_lyapunov_certificate=False`. Actual field-energy and coherence changes
require before/after telemetry, including the actual glyph executed after
grammar selection. See
[example 39](../examples/02_physics_regimes/39_nodal_equation_decomposition.py).

### 1.3 U2 stability scope

Grammar rule U2 requires destabilizers to be accompanied by stabilizers and
limits uncompensated debt. This is an operator-composition policy. It does not
by itself imply

$$
\sum_{\text{ops}} \Delta E_{\text{op}} \le 0
$$

because the grammar does not supply a common state functional, elapsed times, or
proved gain bounds for arbitrary operator realizations. Even a nominal
multiplier product at most one remains a compatibility-model result; analytic
contractivity requires a separate dynamical proof.

Examples 29 and 38 report the nominal product beside measured energy changes.
Agreement or disagreement in a finite run measures the compatibility model; it
cannot establish a grammar-wide guarantee.

`analyze_operator_policy_context` also reports a U2 multiplier beside the
normalized graph diffusion gap. It deliberately leaves the historical
`effective_convergence_rate` field as `NaN`: an operator-position multiplier and
a continuous-time pure-EPI eigenvalue cannot be combined without an explicit
operator-time model. Its `policy_half_steps` is a score-model statistic; its
`diffusion_relaxation_time` is the separate unit-capacity pure-EPI scale.

### 1.4 Restricted affine EPI gain theorem

One restricted operator result now supplies real gain semantics without
changing U2. In a positive common diffusion metric `H=diag(h)`, let
`Q=I-1h^T/(h^T1)` and `V(x)=||Qx||_H^2/2`. A declared affine EPI reset
`x+=Ax+b` has a finite global multiplicative `V` gain exactly when `A1` and
`b` are uniform. Otherwise a uniform input gives the decisive
`V_before=0<V_after` counterexample.

For a passing reset, the sharp gain is the squared weighted induced norm of
`QAQ`. The engine computes a rational quotient-gain upper bound exactly on the
represented binary64 coefficients and retains the weighted-Frobenius value as a
fallback for larger non-scalar maps. Exact scalar quotient actions avoid its
dimension penalty. Valid flow and reset proofs then compose this bound with
continuous diffusion to yield a finite-horizon multiplicative bound. The exact
gain product remains certificate data; log-space composition uses separate
upward 32-bit-significand dyadic gain factors to bound rational growth while
preserving a conservative upper enclosure. The flow contribution uses only its
rationally certified quotient-rate lower bound, never its eigensolver estimate.
The strict contraction decision uses a rational upper enclosure of the net
log-energy budget; caller tolerance does not decide its sign, and the reported
multiplier is rounded upward. Asymptotic disagreement decay is asserted only
when the caller declares repetition of the same word with positive flow
duration. Exact
weighted-mean preservation identifies convergence to the initial weighted
consensus only together with that repeated contraction and exact preservation
of the same mean by the represented flow.

The first two catalog realizations are now explicit for local and all-target
Reception (EN) and Resonance (RA). Their runtime kernels use one centralized unweighted
neighbour mean and two-stage binary64 blend. Under fixed connected undirected
support, scalar or uniform-real BEPI, convex mixing, and inactive hard clipping,
the ideal-real update is affine and preserves constants, but a nontrivial local
blend does not preserve a positive weighted mean as a global functional. A
represented coefficient matrix is nested into the affine theorem only when its
consensus identity passes exactly; runtime/matrix agreement remains a snapshot
diagnostic.

At the all-target boundary, every row is assembled from one immutable stage
snapshot. Convex ideal-real mixing preserves configured hard bounds under
repetition. Fixed-map promotion checks support, represented row sums and, for
RA, fixed U3 sets plus a forward-invariant sign/kind domain. The finite runtime
trace, weighted-mean drift and pre/post diffusion metrics remain independent
diagnostics; the certificate does not claim global binary64 affinity.

The shared execution boundary now also covers target-local AL, SHA, VAL,
NUL, ZHIR and NAV, IL's snapshot-bound pressure and phase proposal, UM's
overlapping phase/topology proposal, OZ's overlapping local/propagated pressure
reduction, and THOL's child-support and hierarchy merge. Together with EN/RA and REMESH,
all thirteen stages preflight and freeze every target proposal before
committing. Their contracted
structural channels are target-order invariant before the opaque
pressure-refresh callback: AL writes EPI; IL writes `DeltaNFR` and phase; OZ
writes reduced `DeltaNFR` and per-node RNG progress; SHA
writes `nu_f`; VAL writes EPI and `nu_f`; NUL writes EPI, `nu_f` and
`DeltaNFR`; THOL writes `d2EPI`, `DeltaNFR`, child nodes,
`sub_nodes`, `sub_epis` and graph `hierarchy` after collision-safe
snapshot-rank allocation and detached validation; and ZHIR writes phase, with
structural-acceleration telemetry bound to the same snapshot evidence and U4
context. NAV writes `nu_f`, phase,
`DeltaNFR` and per-node RNG progress from the same snapshot. REMESH leaves structural channels unchanged and merges one graph advisory per telemetry step. UM merges
shortest-arc phase displacements in stable snapshot order, normalizes final
phases, revalidates U3 and coalesces accepted functional links
deterministically. This merge is an engine policy, not a phase-dynamics or
Lyapunov theorem.

IL reports canonical stage/global and radius-local structural `C(t)` separately
from its retained pressure-dispersion telemetry. A missing NAV random seed is
resolved inside the transaction and survives only a successful commit; stable
node offsets and draw counts preserve each random stream across target orders.
AL/SHA bind all targets to one stage timestamp, while NAV uses one shared
latency-observation instant. Ordered lifecycle, audit/telemetry and monitor
streams retain requested target order. IL warnings occur only after all other
fallible stage effects succeed and remain inside the rollback boundary. Cache
state and the opaque pressure refresh are excluded from the invariance result;
identity-bearing caches remain tied to their live graph and node objects.

This execution and atomicity result alone supplies no affine gain. For
AL/SHA/VAL/NUL/ZHIR/NAV, the shared pointwise stage executor can opt into a
certificate computed from its own detached snapshot and frozen proposals.
A successful result distinguishes exact represented EPI realization, affine
gain in the pre-flow metric and an aligned pre/post diffusion metric. The
request rejects unsupported, empty or grammar-replaced stages before live
writes. By itself it does not certify IL, UM, OZ, THOL, histories, pressure
refresh, mixed words or repeated runtime execution. The event runtime can
compose supported pointwise and EN/RA represented maps only after binding all
observed endpoints and one exact common metric.

OZ now derives every local action and outgoing propagation increment from one
immutable snapshot. Incoming increments are summed with `math.fsum` in
snapshot-node rank; its local pressure-magnitude postcondition remains separate
from the final signed field because positive propagation can partially cancel a
negative pressure. REMESH uses an immutable advisory proposal, deduplicates its
graph event and leaves structural channels unchanged; explicit delayed EPI
mixing remains outside the glyph stage. The separate delayed operation has an
immutable exact-support plan, side-effect-free insufficient-history and
empty-support no-ops, and graph-state atomic commit. Its opt-in one-step
evidence separates the exact three-input recurrence, binary64 rounding,
clipping, weighted-mean drift, a convex disagreement bound and a conditional
fixed-history gain. It does not certify repetition with evolving history.
Multi-target IL and UM historically
used the sequential schedule; their canonical word paths now use the shared
snapshot. A grammar replacement still executes through the transactional
Gauss-Seidel fallback.

RA additionally filters every contributing neighbour through circular U3 with
a configured limit in `[0,pi/2]`. Its EPI mix and phase coupling lie in `[0,1]`,
and its frequency amplification is nonnegative. Scalar EPI can change: the
identity gate independently forbids a strict negative/positive crossing and a
change to an established nonempty `epi_kind`; exact zero is neutral and an
absent kind may be initialized. The certificate separates the ideal-real blend,
represented binary64 map, two-stage proposal, and accepted identity-gated
snapshot. A local frequency boost generally changes `h_i=d_i/nu_i`, so a fixed
post-RA diffusion theorem remains available while pre/post switching abstains
unless the metrics are exactly proportional. For the proved nonisolated
single-target EN/RA boundary, a nontrivial EPI write creates the exact defect
`delta L_rw e_i`. In an all-target stage, the relevant quantity is instead the
aggregated defect `L_rw delta`; simultaneous row changes may cancel into a
uniform shift even when individual EPI values changed. The configured refresh
still runs transactionally, and RA phase or capacity changes may independently
require a full multichannel refresh when the aggregate EPI defect vanishes.
These gates, rounding stages, and multichannel changes preclude a global
binary64 runtime-affinity claim.

The generic theorem otherwise applies only to the supplied affine map. The
canonical operator name is metadata, and the nominal policy table above
contributes no coefficient. See the centralized
[affine-reset and hybrid-word theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md#affine-reset-gain-and-hybrid-word-theorem)
and the implementations in
[`hybrid_operator_stability.py`](../src/tnfr/physics/hybrid_operator_stability.py)
[`reception_realization.py`](../src/tnfr/physics/reception_realization.py), and
[`resonance_realization.py`](../src/tnfr/physics/resonance_realization.py).

### 1.5 Operator-event physical-time boundary

The finite event model gives canonical operators zero duration and assigns
elapsed physical time only to nodal-flow intervals. A schedule with `m`
operator events has `m+1` intervals: before the first jump, between successive
jumps and after the last. The schedule is finite and immutable; it does not
execute an operator. See
[`event_timing.py`](../src/tnfr/operators/event_timing.py).

Each accepted duration is first materialized as binary64 and then retained as
the exact `Fraction` represented by those bits. Exact prefix sums define flow
offsets, event offsets, total duration and physical ordering. Float start,
event and end timestamps are display representations of the corresponding
exact absolute times. At large origins a positive interval can leave its
display endpoints equal, or timestamp subtraction can disagree with the
declared duration. Consequently, schedule logic never reconstructs duration
from timestamps, and the schedule builder never populates `epi_time_history`.
Coincident jumps instead share an exact event coordinate in the hybrid log and
are ordered by `event_index`.

For one declared interval,
[`event_duration.py`](../src/tnfr/physics/event_duration.py) maps physical
duration to the fixed pure-EPI disagreement theorem. If `r_lower` is its exact
rational energy-decay-rate lower bound and `q` is the represented target
fraction, the diagnostic encloses `log(1/q)` above by `ell_upper` and certifies
the target only when

`r_lower * exact_duration >= ell_upper`.

The sufficient duration display is rounded toward positive infinity. The
decay-factor result is an upper bound. Ordinary libm duration/factor values and
the source eigensolver rate are estimates and never decide the theorem; the
spectral estimate is explicitly unsealed and is omitted when malformed. A
positive rational rate can still decide the theorem when its downward-rounded
binary64 display is zero.

The separate `execute_operator_event_schedule` runtime now binds these records
to the graph's configured nodal integrator and the shared atomic network-stage
dispatcher. It freezes the initial targets, checks the live binary64 clock at
each boundary and rejects positive intervals whose endpoints collapse or whose
direct float addition misses the scheduled endpoint. Because ZHIR currently
reads a secant from absolute binary64 timestamps, its pre-flow is also rejected
when endpoint subtraction differs from the authoritative declared duration.
Flow boundaries populate timestamped EPI evidence; a same-time jump restarts
that history and remains in `hybrid_event_log`. One graph transaction covers
flow, jump, history, cache and event-log state, while external emitted effects
remain outside rollback.

Within each positive call the integrator may modify only EPI, `dEPI_dt`,
`d2EPI_dt2` and the runtime clock. A custom integrator must also satisfy, in
exact rationalized represented values,

$$
x_i^+-x_i^-=h\,\nu_{f,i}^-\,\Delta\mathrm{NFR}_i^-.
$$

This is a finite held-input nodal-equation identity and supplies no accuracy or
order claim. The transaction is owner-bound to the graph that created it and
preserves NetworkX structural mapping identities, their alias topology and
capturable callback-owned state. Mutable structural keys with non-identity
hash/equality, custom `__deepcopy__` hooks and unmodelled opaque C state are
rejected before execution; common immutable atoms remain valid metadata.
Graph-reachable capturable state is restored, while emitted I/O or warnings,
external resources and references held only by external aliases remain outside
rollback.

Opt-in interval evidence separates two binary64 runtime claims. The original
single-step replay and exact rational Euler map still require one Euler substep.
The broader sequential held-pressure replay admits one or more substeps only
when their represented duration sum equals the declared interval and the exact
built-in integrator, Euler method, absent Gamma, inactive clipping, disabled
extended dynamics, stable support and unchanged capacity/pressure are all
observed. Those internal substeps retain the interval-start pressure; they are
not a pressure-reevaluated physical mesh and have no diffusion modal decision.

An explicit `PhysicalFlowPartition` changes this execution contract. Let

$$
A=\operatorname{diag}(\nu_f)L_{\mathrm{rw}},\qquad
T=\sum_{k=0}^{q-1}h_k .
$$

If exact-real Euler evolution holds the initial pressure throughout duration
$T$, its pure-EPI outer model is

$$
\widehat{x}_H=(I-TA)x_0,
\qquad g_H(\mu)=1-T\mu .
$$

Splitting that model into internal substeps whose exact represented durations
sum to $T$ does not change $\widehat{x}_H$ because the pressure vector remains
fixed. By contrast, the exact-real model for explicit pressure refresh is

$$
\widehat{x}_{k+1}=(I-h_kA)\widehat{x}_k,
\qquad
\widehat{x}_R=\left[\prod_{k=0}^{q-1}(I-h_kA)\right]x_0,
\qquad
g_R(\mu)=\prod_{k=0}^{q-1}(1-h_k\mu).
$$

These formulae describe real-arithmetic model maps. The executor instead records
binary64 endpoints, which a modal diagnostic alone does not identify with
$\widehat{x}_H$ or $\widehat{x}_R$. A trusted held-pressure replay identifies
the actual binary64 operations. The stronger exact-affine certificate identifies
the rationalized observed endpoints with the corresponding exact-real map; a
physical parent needs that certificate for every segment before its exact gain
bounds compose. Fixed conductance and capacity plus verified pure-EPI pressure
at every boundary are necessary for these promotions. Outside that domain the
executor still records the trajectory, while affine and modal claims abstain as
their respective conditions require. In general $\widehat{x}_H\ne
\widehat{x}_R$, and the two model modal-stability decisions can differ. For
example, on the nonuniform mode of unit-capacity $K_2$, $\mu=2$, $T=1.25$ gives
$g_H=-1.5$, whereas two refreshed segments of length $0.625$ give
$g_R=0.0625$.

`build_physical_flow_partition` rejects a declaration unless at least two
positive represented segments cover the parent exactly in rationalized
binary64 time and every displayed boundary is additive and subtractive.
`execute_operator_event_schedule(..., physical_flow_partitions=...)` invokes
the centralized pressure reader at all $q+1$ boundaries. Each callback must
preserve all non-pressure nodal data, full edge state, effective conductance,
persistent graph configuration, history and clock. Existing non-`None` cached
pressure weights are immutable; only the canonical default pressure
implementation may initialize a missing or `None` cache. The complete graph
transaction rolls back on any
violation or later failure. Executor-owned evidence binds the boundary
snapshots, callback identity, every segment flow certificate, segment-start
modal diagnostics and any exact common-metric gain product.

At a terminal ZHIR boundary, the last two live `epi_time_history` samples span
the final physical segment. That terminal secant is therefore the only observed
rate available to Mutation. An offline whole-parent secant is retained
separately for comparison. `compare_event_local_zhir_physical_refinement`
pairs this physical observation with a matching held-pressure baseline and
reports physical-minus-baseline rates, endpoints and gate decisions. It also
compares ordinary binary64 evaluations of $g_H$ with the segment-factor product
only when trusted held-pressure segment replays share the baseline's exact node
order, capacity and conductance. These are model diagnostics rather than
endpoint-map certificates. Matching sorted spectra alone are insufficient because
isospectral generators need not share an eigenbasis or commute. Agreement is
not assumed and disagreement is a valid measurement. Neither record establishes
solver order, three-mesh convergence, U4 readiness, adaptive grammar or future
behavior.

`observe_executed_event_local_zhir_physical_prejump` strengthens only the
provenance axis. Given one intact stage-certified execution and a committed ZHIR
index, it binds the scheduled event, executed event, preceding physical parent,
terminal segment, glyph stage, ordered decisions and trigger certificates by
identity. Its common-execution claim is absent from the reusable offline
coordinate pairing. It still proves no solver property, convergence, general
U4 readiness or future behavior.

The narrower event-local comparison in
[`event_refinement.py`](../src/tnfr/physics/event_refinement.py) starts from that
executor-sealed held-pressure evidence. One pre-jump observation pairs a flow
offline with a scheduled ZHIR event at the same exact and represented endpoint;
common schedule-execution provenance is not inferred. It preserves two distinct
secants: the rational quotient of exact represented endpoints and duration, and
the binary64 subtraction/division performed by the actual Mutation gate. The
second quantity alone controls the strict threshold and may record either an
accepted or rejected gate.

Two observations with the same support, initial EPI, capacity, pressure,
conductance, duration, event coordinate and `xi`, but different internal
substep counts, have a certified identical represented gate outcome only if the
observed decisions agree and each exact binary64 rate perturbation is strictly
smaller than the baseline threshold margin. Contact with the margin is not a
certificate. Since the substeps hold the interval-start pressure, this result
does not establish the separately executed pressure-reevaluated partition,
modal equivalence, solver accuracy or order, U4 readiness, adaptive U2/U4 or
future behavior.

With `include_stage_certificates=True`, interval capture is implied and each
accepted event produces an `ExecutedGlyphStage`. Pointwise
AL/SHA/VAL/NUL/ZHIR/NAV and neighbour-reading EN/RA reuse the certificates
computed from their executor-owned snapshots and proposals; other glyphs or
failed domains abstain explicitly. Each represented certificate is checked
against the actual EPI endpoints and adjacent positive-flow evidence. Accepted
two-phase ZHIR stages additionally seal one complete decision observation per
target before commit metadata can change; the event-stage record retains these
observations independently of whether its EPI map is certifiable.
Accepted two-phase EN stages similarly seal one observation per target. One
owner-bound pre-EN snapshot supplies neighbour integration, semantic-kind
selection, optional source detection and metrics. The record then verifies the
committed EPI, kind and tracked source list after all graph-mutating stage checks
and before warning publication.
This closes temporal provenance for those finite fields while leaving the
complete auxiliary trajectory and all future executions outside the claim.
The complete `ExecutedGlyphStage` is value-sealed, and the enclosing execution
result checks stage cardinality, event order and ordered ZHIR/EN target support.
An adjacent flow that explicitly abstains remains observable while its failed
inner proof cannot enter the represented schedule product.
`ObservedRepresentedEPIScheduleComposition` then retains a complete
chronological operation record and multiplies exact rational gain factors only
if every represented affine map is intact, all node orders and consecutive EPI
endpoints match, and one normalized metric spans the finite trace. This global
gain belongs to the represented maps and bounds the observed trace through its
endpoint bindings. The hard-false `runtime_schedule_global_gain_certified`
field prevents interpreting it as a global executable binary64 map.

One `execute_event_remesh_cycle` then appends exactly one pre-REMESH state and
applies the delayed map inside an individually atomic graph transaction. The
delayed map preserves the post-schedule edge state and every stored non-EPI
alias. Its sealed `RemeshHistoryTransitionObservation` verifies, for capacity
`M`, `H_out = tail_M(tail_M(H_in) || (x_pre,))`, and records local/global lag
availability independently. The optional pressure callback is operationally
complete only after one successful return; this does not prove that its output
satisfies `DeltaNFR = -L_rw EPI`.
The cycle materializes a supplied `physical_flow_partitions` iterable once
inside the outer transaction and forwards that same tuple to schedule
execution, so iterator failure and plan selection share the cycle rollback.

For at least two sealed cycle results supplied in caller order,
`compose_event_remesh_cycle_observations` constructs sealed adjacent
`EventRemeshCycleBoundaryObservation` records and an
`ObservedEventRemeshCycleSequence`. These compare exact recorded EPI, capacity,
pressure, phase, clock and full delayed history. Raw metric equality is kept
separate from compatibility of exact normalized metric rays: proportional raw
weights define the same disagreement geometry at different energy scales.
Tri-state nested schedule metric alignment is reported without merging each
`ObservedRepresentedEPIScheduleComposition` with its REMESH result.
The zero-based indices are local ordinals. Repeated object identity is rejected,
while distinct copies still cannot prove causal order or shared-graph execution.
Exact recorded-boundary continuity is independent of those metric conditions.
The stronger common-metric sequence result requires one normalized ray and a
matching non-absent metric from every nested schedule; raw equality is not
required. A `None` or `False` alignment blocks only this stronger result, not
exact recorded-boundary continuity.

No schedule/REMESH gain product follows across evolving history. In an explicit
sequential lag-one execution with `alpha=1`, EPI `(2,0)` and delayed row `(0,2)`
map first to `(0,2)` and then back to `(2,0)`, even though both fixed-history
REMESH records have zero current-state coefficient. The sequence
therefore withholds evolving-history gain and repetition, runtime-global gain,
whole-sequence atomicity, full graph/grammar-history continuity, solver
accuracy, shared execution provenance and future stability.

The S5/S16 finite three-mesh observation is implemented in
[`event_remesh_refinement.py`](../src/tnfr/physics/event_remesh_refinement.py).
It compares three already executed sealed cycles under strict coarse-to-
intermediate-to-fine nesting and explicit compatibility of schedule, ordered
support, initial nodal channels, captured generator, metric, runtime metadata,
incoming history and REMESH controls. Exact represented EPI checkpoints cover
the pre-schedule state, physical boundaries and both sides of REMESH; pairwise
exact `L_inf` errors use persistent node identifiers. ZHIR rows require the
executor-linked observer in all three runs; they are collected whenever that
evidence exists, while optional `zhir_xi` only validates the executed threshold.
Modal rows require a common captured generator. Detached results omit the full
pre-schedule graph namespace, callback closure and RNG state, node metadata and
sub-EPI state. Accordingly, `complete_reference_problem_certified` and
`epi_differences_attributable_only_to_mesh_certified` remain false.
`three_mesh_observation_certified` establishes only this finite construction.
Even a decreasing intermediate/fine error diagnostic does not prove solver
order, mesh convergence, Lyapunov decrease or a mixed gain.

The general reversible exact-mode runtime adapter now binds any finite ordered
family of compatible, individually executor-certified
`ExecutedPressureRefreshedFlowPartition` records. It derives the exact
rationalized conductance, capacity, initial field, reversible metric and
partitions from captured evidence and rejects changes among those sources. For
each segment it separates pressure realization
`rho=p64-(-L_rw*z)`, held-input execution
`eta=z_next-z-h*diag(nu_f)*p64`, and
`epsilon=h*diag(nu_f)*rho+eta`. Since represented defects need not stay in the
reference eigenspace, their endpoint contribution follows the complete
recurrence `r_next=(I-h*A)r+epsilon`. Signed coordinate enclosures then give
exact rational lower and upper bounds for represented-minus-continuous
`L_inf` error and `H`-error energy. This is a finite offline family: the
individual partitions retain executor provenance, while their caller ordering
does not prove common causal provenance, runtime mesh convergence, solver
accuracy/order, repetition or future stability. It contains no glyph or REMESH
claim. The derivation is centralized in
[`TNFR_DIFFUSION_STABILITY_THEOREM.md`](TNFR_DIFFUSION_STABILITY_THEOREM.md#finite-executor-binding-of-the-exact-mode).

The compatible effective-P2 reference family closes one finite runtime-linked
subcase of the general pure reversible single-eigenmode theorem. It routes all
three physical partitions through the common runtime observer once, then
requires exact-affine identification, zero pressure-realization and held-input
residuals, and zero local and endpoint defects before adding the REMESH-specific
result. It requires an
event-free two-node path, a nonuniform initial mode, fixed positive
conductance, homogeneous positive capacity, pure-EPI pressure refreshed at
every boundary, exact represented Euler
segments with `0 < lambda*h < 1`, and two proper positive subdivisions.
For unit REMESH delays with the exact initial field as delayed data and hard
clipping on one common scalar interval, rational exponential enclosures prove
the finite continuous/Euler bound and strict subdivision improvement. The
ideal post-REMESH error is exactly `beta=(1-alpha)^2` times the pre-REMESH
error, while the committed runtime bound adds the signed rounding-plus-clipping
residual norm. This does not establish solver order, generic or binary64
asymptotic convergence, glyph or mixed-channel dynamics, soft clipping,
changing support or metric, repetition or future behavior. The complete
continuous/Euler proof is in
[`TNFR_DIFFUSION_STABILITY_THEOREM.md`](TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem);
the additional REMESH equations and runtime boundary are in
[`REMESH_INFINITY_DERIVATION.md`](REMESH_INFINITY_DERIVATION.md#28-effective-p2-three-mesh-reference-family).

The restricted augmented-history REMESH problem is now solved for a distinct
exact recurrence with fixed uniform `alpha`, fixed positive diagonal spatial
metric, fixed delays and ordered support, and no clipping. Let the active delay
coefficients be `c_r`, let `P` be the finite newest-first companion matrix, and
let `pi` be its exact invariant temporal distribution. Then

$$
V(X_k)=\sum_j\pi_j\frac12\lVert Q_Hx_{k-j}\rVert_H^2
$$

obeys

$$
V(X_k)-V(X_{k+1})
=\frac{\pi_0}{2}\sum_{a<b}c_ac_b
  \lVert Q_Hx_{k-a}-Q_Hx_{k-b}\rVert_H^2\geq0.
$$

Equality holds exactly when all active centered fields agree. For
`0 < alpha < 1`, the companion is primitive and every spatial coordinate
converges to its preserved stationary history barycenter. For `alpha=1`, it is
a pure-delay permutation: the functional is conserved and periodic histories
remain possible, including the alternating witness above. The theorem therefore
resolves the earlier obstruction without claiming strict decay in that regime.
It proves neither spatial consensus nor zero pressure.

One further bridge now identifies a single applied, sealed runtime cycle with
that exact companion. It converts the retained oldest-first runtime window to
the theorem's newest-first order, replays the raw affine and clipping calls bit
for bit, and keeps the signed exact decomposition

$$z=y+r_{\rm round}+r_{\rm clip}$$

between ideal companion head `y`, binary64 raw head and committed bounded head
`z`. The bridge gives exact lifted augmented-energy defects for rounding and
clipping, plus a posteriori sufficient lower bounds derived in the same fixed
metric. A hard common-interval clamp is disagreement-nonexpansive; the soft
knee can expand disagreement, and exact binary64 rounding can dominate an
arbitrarily small ideal Jensen dissipation. Both cases are retained as finite
observations instead of being hidden by the ideal theorem. A nonzero rational
diagnostic that would display as binary64 zero is rejected before commit.

The post-REMESH head is not immediately appended to the runtime history, so the
bridge constructs a lifted companion state and does not transfer repetition or
temporal convergence to the live executor. Changing coefficients, delays,
metric or support and composition with the next event schedule remain outside
this single-transition result.

The separate exact REMESH-head/schedule-head layer supplies the algebra needed
at that next boundary. In one fixed metric it decomposes the augmented energy
drop into ideal Jensen dissipation minus signed raw, clipping and schedule
defects. If the supplied scheduled head satisfies `E(s) <= q E(z)`, the exact
drop is the sum of a computable lower bound and the nonnegative schedule-gain
slack. This yields a sufficient finite-step test without multiplying a
fixed-history REMESH factor by `q`. It also retains the exact change in the
stationary history barycenter. The pure layer does not itself identify the
heads or gain with one runtime trace.

The adjacent-cycle adapter supplies that finite identification when a sealed
cycle sequence has exact recorded boundaries, fixed REMESH configuration and
one common normalized schedule metric. It requires the next represented
schedule to map the bounded REMESH head to the next pre-REMESH head and checks
that this endpoint becomes the next recorded history head. Consecutive
augmented energies then telescope exactly, and the total drop is the sum of
the per-boundary lower bounds and the augmented schedule-gain slacks
`pi[0] * schedule_gain_slack`. The input order still
does not prove shared graph provenance or causal execution, so no cross-call
atomicity, global executable gain, repetition or future theorem follows. See
[`remesh_history_stability.py`](../src/tnfr/physics/remesh_history_stability.py),
[`runtime_remesh_history_stability.py`](../src/tnfr/physics/runtime_remesh_history_stability.py),
[`remesh_schedule_stability.py`](../src/tnfr/physics/remesh_schedule_stability.py),
[`runtime_remesh_schedule_stability.py`](../src/tnfr/physics/runtime_remesh_schedule_stability.py),
and the centralized derivation in
[`REMESH_INFINITY_DERIVATION.md`](REMESH_INFINITY_DERIVATION.md#24-exact-finite-companion-history-stability).

The outer `execute_event_remesh_cycle_sequence` wrapper supplies the missing
finite execution provenance. It accepts ordered
`EventRemeshCycleExecutionSpec` values, executes every cycle on one graph under
one graph-owned transaction and seals a `CausalEventRemeshCycleReceipt` for
each ordinal. The resulting `ExecutedEventRemeshCycleSequence` binds the exact
specs, cycle results and offline cycle observation from that same invocation.
By default it additionally requires and retains the compatible runtime
schedule/history telescope. The explicit
`require_runtime_telescope=False` branch always records
`runtime_telescope=None`, while preserving causal order, common graph identity
and atomic graph rollback for that finite block. Telescope-specific claims are
false on that branch. The offline
observers retain their negative provenance scope when called independently.
The wrapper still proves no global schedule/REMESH gain, uniform positive
normalized block margin over a declared forward-invariant runtime class, intrablock runtime
prefix-amplification bound, solver accuracy/order, mesh convergence, repetition
or future stability; emitted I/O, warnings, external resources and
external-only aliases remain outside rollback.

The next finite layer is
[`observe_executed_event_remesh_block_margin`](../src/tnfr/physics/runtime_remesh_schedule_block_margin.py).
It selects a nonempty contiguous block from a causal execution that retained an
exact runtime telescope and preserves every selected boundary by identity. It
rejects the no-telescope branch. Summing
their balances gives

$$
D_B=V_{\rm before}-V_{\rm after}=K_B+S_B,
\qquad S_B\geq0.
$$

For $V_{\rm before}>0$, the block-specific normalized lower margin and endpoint
bound are

$$
\kappa_B=\frac{K_B}{V_{\rm before}},
\qquad
\frac{V_{\rm after}}{V_{\rm before}}\leq1-\kappa_B.
$$

The normalized fields are undefined at zero initial energy. One exact fixture
has $\kappa_B=139/256$; the lag-one `alpha=1` identity-schedule orbit has
$\kappa_B=0$. A uniform positive absolute drop is impossible across equilibrium
and amplitude-scaled copies because this quadratic energy and its drop scale to
zero. A positive margin on one observed block proves no uniform class
coercivity. Repeated stability would additionally require a uniform positive
normalized block margin over a declared forward-invariant runtime class and a
finite runtime intrablock prefix-amplification bound. Both remain open for the binary64
runtime class observed here.

A distinct exact-model theorem now supplies both bounds under explicit policy
hypotheses. Index histories immediately after each schedule, retain one fixed
REMESH companion `P` and one fixed positive spatial metric, and require every
possibly varying exact schedule to preserve spatial consensus and have
disagreement-energy gain at most the same `q` in `[0,1]`. The componentwise
energy envelope is

$$B_q=\operatorname{diag}(q,1,\ldots,1)P.$$

Nonnegative-matrix domination gives prefix gain at most one. Every companion
path visits the head within the sufficient universal horizon
$L=\texttt{active\_max\_delay}+1$, so $B_q^L\leq qP^L$. Stationarity of the
temporal weights then yields

$$V_{k+n}\leq q^{\lfloor n/L\rfloor}V_k.$$

Consequently, `q<1` gives uniform normalized block-margin lower bound `1-q`
and geometric decay of spatial disagreement in every retained history row.
This seminorm does not control their spatially uniform temporal means. At
`q=1`, the exact result proves nonincrease and zero certified margin only. The
sealed implementation is
[`remesh_schedule_policy_stability.py`](../src/tnfr/physics/remesh_schedule_policy_stability.py).
It assumes rather than inspects the schedule family and does not absorb
binary64 rounding or clipping defects; runtime forward invariance and a
strictly positive net runtime margin remain open.

The signed-defect extension closes the corresponding conditional robust
calculus. Let `y` be the ideal REMESH head, `z` the bounded head and

$$
J_k=\sum_d c_dE_H(x_{k-d}),\qquad
\delta_k=E_H(z_k)-E_H(y_k)\leq\eta J_k.
$$

Jensen gives `E_H(y_k)<=J_k`; a consensus-preserving schedule gain `q`
therefore yields the effective head gain
`q_eff=q*(1+eta)`. The existing companion envelope applies with `q_eff`.
Values above one are rejected, equality gives nonincrease with zero margin,
and `q_eff<1` gives block margin `1-q_eff` and geometric decay of spatial
disagreement. The signed defect is not the energy or norm of `z-y`.
[`remesh_schedule_relative_defect_stability.py`](../src/tnfr/physics/remesh_schedule_relative_defect_stability.py)
seals this exact conditional result without duplicating the matrix-power
kernel.

The finite
[`observe_executed_event_remesh_relative_defect_block`](../src/tnfr/physics/runtime_remesh_schedule_relative_defect.py)
adapter binds it to a contiguous block of one intact causal execution that
retained its compatible runtime telescope. It
checks the exact `J`, defect and slack at every boundary, requires each
represented gain `q_j<=q`, verifies the full energy-vector envelope and checks
the endpoint factor `q_eff^floor(N/L)`. At `J=0` the inequality is tested
directly and its ratio is undefined. This recorded block does not prove a
uniform runtime class or forward invariance. Extending the restricted
half-alpha class below to broader represented states, or promoting the finite
P2 observation to future complete-runtime execution, remains the next
repeated-runtime boundary.

The represented-number analysis now resolves two narrower boundaries. The
exact pairwise identity lifts a scalar REMESH defect ratio to arbitrary finite
support and positive diagonal metric. A normal-valued `alpha=1/2` pair requires
`eta=2^210-1/4`, so a bounded hard-clipped box alone is not a useful uniform
class for the `q=9/16` witness policy. At `alpha=1`, sufficient finite history
inside one fixed hard interval instead gives an exact global-delay numeric copy,
uniform `eta=0`, and REMESH-only forward invariance.

The first useful `0<alpha<1` subclass is now exact. Fix ordered P2 support,
any positive diagonal metric, positive delays, `alpha=1/2`, a symmetric hard
interval `[-B,B]` with `B>=4s`, `s=2^-1074`, and sufficient represented
history with every row `(a,-a)`. IEEE round-to-nearest-even and symmetric
clipping preserve numeric antisymmetry, so REMESH alone maps the class into
itself. For scalar amplitudes `c,l,g`, set
`D=c^2+l^2+2g^2`, `x=sqrt(D)` and let `r` and `y` be the production and ideal
heads. With `u=2^-53`, direct error propagation gives

```text
|r-y| <= A*x + C*s,
A = 3*u/2 + u^2/2,
C = 9/4 + 9*u/4 + u^2/2.
```

For `x>=11s`, the resulting exact relative-defect tail is strictly below
`135/124`. For `x<11s`, all inputs reduce to bounded integer multiples of
`s`; exact enumeration covers 6,615 candidates, 3,890 with `0<D/s^2<121`, and
finds the sharp maximum `eta*=135/124` at `(-3,-2,-3)s`. The strict robust
threshold is therefore `q<124/259`. In particular, `q=4/9` gives
`q_eff=259/279` and normalized block margin `20/279`; `q=124/259` gives only
nonincrease with zero margin, and `q=9/16` is inadmissible. This theorem does
not include general positive-metric-centered rows: one exact centered input
triple acquires represented center `-2^-84`. Nor does it include an
unrestricted fixed lattice, since `(h,0,0)` maps to `h/4`. It certifies only
the REMESH map on the declared class, without schedule, graph/event, repeated
complete-runtime or future-execution claims.

On P2, a configured half-Reception numeric EPI kernel supplies the first
restricted global numeric EPI-kernel family for that boundary. Mutual
singleton neighbor sets and one immutable Jacobi snapshot make both binary64
proposals the same reversed sum
of half-scaled operands. A common hard clamp preserves their numeric equality,
so `q=0` in every positive diagonal metric. The common robust theorem then
gives exact active-history spatial-disagreement extinction after
`tau_global+1` cycles of the restricted repeated kernels. Underflow may change
the consensus value and signed-zero bits may differ; neither affects centered
energy. This result does not certify an actual graph, complete Reception stage,
grammar admission, event ownership, graph transaction or solver.

A finite causal adapter now closes part of that execution boundary. From one
intact `OperatorEventExecutionResult`, it selects a graph-owned EN event and
deeply revalidates its two-phase neighbor-stage evidence, ordered P2 targets,
mutual singleton runtime neighbors, exact half mix, hard interval, unchanged
capacity/conductance, diffusion metric and captured endpoint replay. The
observed EPI stage therefore inherits `q=0`, finite grammar admission and
whole-schedule graph atomicity. The source REMESH history/configuration,
auxiliary Reception state and repeated/future runtime remain unbound.

A second sealed adapter closes the same-graph boundary for one completed finite
causal sequence. Every observed cycle must bind a P2 half-Reception stage to the
whole schedule EPI transition and its same-cycle hard-clipped `alpha=1` REMESH
to an exact `eta=0` global-delay copy. Ordered support, the normalized positive
metric ray, hard interval, delays, history capacity and runtime alpha source are
fixed across the invocation. If `N >= L = tau_global+1`, every row in each
post-horizon active history suffix of length `L` has zero spatial-disagreement
energy, as do the corresponding observed post-REMESH endpoints. Older retained
history rows are outside the source-class claim. The result is finite and
retrospective: it does not certify Reception auxiliary state, the current live
graph after observation, future or unobserved repetition, solver accuracy or
full TNFR stability. See
[`runtime_p2_reception_remesh_sequence.py`](../src/tnfr/physics/runtime_p2_reception_remesh_sequence.py).

The reusable
[`execute_p2_half_reception_remesh_policy_invocation`](../src/tnfr/physics/runtime_p2_reception_remesh_policy.py)
closes the per-call transaction boundary. It snapshots the graph before
materializing caller inputs, preflights the current restricted P2/REMESH
conditions, rederives U1a admission from the live EPI pair at every cycle start,
executes the zero-flow `EN -> IL -> REMESH` sequence without an affine telescope,
and constructs the finite certificate before commit. A zero cycle-start pair or
any other failure, including invalid or graph-mutating post-certification,
restores graph-owned state. Two successful calls are therefore independently
validated finite observations. The first does not certify the second in advance,
and neither establishes auxiliary Reception-state stability, a forward-invariant
complete-runtime class, solver properties or full TNFR stability.

This execution contract does not prove solver accuracy or invariance under an
equivalent timestep refinement, assign an affine gain to every jump, establish
full-multichannel or repeated stability, or convert the continuous duration or
Euler modal count into adaptive U2 debt or U4 recency. The executable
[`165_operator_event_relaxation.py`](../examples/02_physics_regimes/165_operator_event_relaxation.py)
records the schedule-only distinctions with two coincident jumps.

### 1.6 Spectral Gap Characterisation

For connected fixed symmetric pure EPI diffusion, the first positive
generalized eigenvalue $\lambda_*$ of $Bv=\lambda Hv$, with
$H=\operatorname{diag}(d_i/\nu_i)$, controls the weighted-energy decay:

| Quantity | Expression | Physical meaning |
|----------|-----------|------------------|
| Energy decay | $V(t) \le e^{-2\lambda_*t}V(0)$ | Exact upper bound |
| Relaxation scale | $1/\lambda_*$ | Heterogeneous structural time |
| Homogeneous reduction | $\lambda_*=\nu_f\lambda_2(L_{sym})$ | Common-capacity case |

The corresponding exact single-mode Euler theorem is now available for every
fixed connected symmetric rational conductance with positive rational
capacity. If `v=x_0-mean_H(x_0)*1` is nonzero and satisfies the exact identity
`A*v=mu*v` for `A=diag(nu_f)L_rw`, then

```text
x(T) = mean_H(x_0)*1 + exp(-mu*T)*v,
x_P(T) = mean_H(x_0)*1 + product_j(1-mu*h_j)*v.
```

For a positive partition with `0 < mu*h_j < 1`, the factor error is bounded by
`(mu^2/2) sum_j h_j^2 <= (mu^2/2) T h_max`. Multiplication by
`||v||_inf` gives the exact `L_inf` endpoint error; multiplication by
`E_H(v)` times the squared factor error gives its `H`-error energy. Proper
positive subdivision strictly improves the factor and quadratic bound, and the `h_max` estimate
proves conditional exact-real convergence for fixed-data admissible families.
The complete derivation and runtime-scope boundary are centralized in the
[`reversible single-eigenmode theorem`](TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem).
The pure certificate does not inspect binary64 execution, glyphs or REMESH.

This table is the exact real-arithmetic theorem. The executable binary64
certificate separately rationalizes its materialized generator and displayed
metric. It first requires exact invariance of the consensus subspace, then
proves positivity of the symmetrized dissipation on the metric-orthogonal
quotient and returns a downward-rounded certified rate. Its generalized
eigenvalue remains an estimate. Exact conservation of the displayed weighted
mean is a further independent Boolean; without it, the snapshot projection
center is not promoted as the final consensus value.

For time-varying capacities satisfying positive finite per-node bounds, the
Dirichlet energy is a common Lyapunov function with decay rate at least
`2 lambda_2(B) min_i(lower_i/d_i)`. The executable theorem treats the effective
binary64 conductances and capacity bounds as exact real coefficients, then forms
degrees, the Laplacian, the quotient gap and the rate in rational arithmetic.
Ordinary binary64 eigengaps, products and energies remain labelled estimates;
whether `diag(strength)-adjacency` happens to annihilate constants after floating
accumulation is a separate diagnostic. A positive exact rate that underflows on
publication still proves the exact-real theorem but supplies no operational
binary64 rate. The routine neither observes the future capacity schedule nor
certifies a numerical integrator. Within the conditional exact-real model the
field converges to consensus, but the consensus value is schedule-dependent
unless the capacity ratios remain fixed.

For a finite family of changing symmetric topologies on fixed node support,
the normalized metric `d_i/nu_i` is a common Lyapunov metric when the raw
represented metric vectors are exactly proportional across every regime.
Caller-tolerance proximity is diagnostic and cannot promote the theorem. The
minimum certified rational quotient bound then controls arbitrary switching.
Every represented regime must fix the uniform EPI field exactly; the weaker
consensus-subspace condition and preservation of the displayed weighted mean are
reported independently and cannot replace that canonical fixed-point identity.
The object's equilibrium, Lyapunov-value and derivative fields are binary64
diagnostics for its first snapshot and displayed normalized metric. Composition
instead uses the rationalized reference metric and certified rate; a sampled
trajectory must recenter that common metric at every state.
This is a restricted
topology-change theorem; it does not cover node creation/removal or nodal-type
transitions.

**Implementation**:
`physics.structural_diffusion.verify_heterogeneous_diffusion_stability` provides
the exact fixed-capacity certificate;
`derive_time_varying_diffusion_stability_bound` supplies the conditional common
bound; `verify_switching_diffusion_stability` checks the common-metric switching
theorem. `diagnose_euler_relaxation_window` resolves the frozen graph's actual
explicit-Euler modal factors and solver-step relaxation count, using a
dimensionless zero-mode tolerance relative to the fastest decay rate. That count is a
numerical-integration quantity, not the U4 operator-position window.
`physics.lyapunov` retains nominal operator-role diagnostics and must not be read
as a universal convergence prover.

---

## 2. Phase Transitions

### 2.1 Order-parameter candidate

The symmetry-breaking field $\mathcal{S}$ is the implemented order-parameter
candidate for controlled phase-transition sweeps:

$$
\mathcal{S}(i) = \left(|\nabla\phi|^2 - K_\phi^2\right) + \left(J_\phi^2 - J_{\Delta\mathrm{NFR}}^2\right)
$$

### 2.2 Phase Classification

The phase is decided by the standardized spatial imbalance of the signed global mean,
$z = |\langle\mathcal{S}\rangle| / \sqrt{\mathrm{Var}(\mathcal{S})/N}$, and likewise for
signed chirality. Because graph nodes are coupled, this ratio is not a
hypothesis-test z-score without an independent sampling or effective-sample-size
model. The engine retains the historical function name `symmetry_zscore`, but
uses the ratio only as a deterministic classifier input. The selected operational
cut is $z = 1$. The local magnitudes $\langle|\mathcal{S}|\rangle$ and
$\langle|\chi|\rangle$ remain separate telemetry and cannot establish global
symmetry breaking because opposite signs may cancel.

| Phase | Condition | Operational meaning |
|-------|-----------|------------------|
| **NON_LIFE** | $z \le 1$ | signed mean below the selected standardized-imbalance cut |
| **LIFE** | $z > 1$ AND $z_\chi > 1$ | both signed-imbalance ratios exceed the selected cut |
| **CRITICAL** | $z > 1$ AND $z_\chi \le 1$ | order imbalance without a chirality-ratio crossing |

### 2.3 Effective time-series exponent fit

`fit_critical_exponent` and `detect_phase_transition` fit the diagnostic model

$$
|\langle\mathcal{S}\rangle| \sim |t-t_c|^{p_{\mathrm{fit}}}
$$

by log-log regression. In the time-series detector, $t_c$ is the time of the
largest sampled susceptibility, when that maximum is positive. The fit uses
only positive samples strictly after $t_c$; it never substitutes earlier
samples. It returns `exponent`/`measured_exponent` and $R^2$, or `None` when fewer
than three eligible post-$t_c$ samples exist.

The neutral symbol $p_{\mathrm{fit}}$ avoids assigning competing $\beta$ and
$\gamma$ names to the same implemented regression. The result is
protocol-dependent. It becomes a conventional critical exponent only when the
time coordinate is mapped to a declared control-parameter distance and a
finite-size protocol supports that interpretation. No universal value follows
from the nodal equation.

### 2.4 Classification scale

The classifier uses only `Z_SIGNIFICANCE = 1`, a selected standardized-spread
policy. It is neither a p-value cut nor a graph-independent critical constant.
Legacy constants such as $\pi/16$, `0.034` and `0.155` do not enter this phase
classification and must not be presented as universal transition thresholds.

### 2.5 Susceptibility

The finite-sample structural susceptibility diagnostic is

$$
\chi_{\mathcal{S}}(t) = N \cdot \operatorname{Var}(\mathcal{S})
$$

The implementation records its maximum along the supplied sequence. A sampled
peak is not a divergence theorem.

### 2.6 Finite-size protocol

`analyze_phase_finite_size_scaling` accepts one shared control grid at three or
more node counts, with a balanced replicate axis. It reports the sampled
pseudocritical control, replicate standard errors and power-law slopes of peak
susceptibility, order magnitude and coherence length against node count `N`.
The use of `N` is explicit: converting these slopes to conventional exponent
ratios requires an independently justified linear-size or dimension map.
Replicating one graph family does not establish universality. A susceptibility
maximum on the first or last sampled control value is flagged as unbracketed
instead of being presented as a located critical point. Every exact maximizer
is retained, so plateaus are marked ambiguous and any boundary contact makes
the sampled peak unbracketed.

**Implementation**: `src/tnfr/physics/phase_transition.py` provides snapshot and
time-series classification; `src/tnfr/physics/phase_scaling.py` provides the
balanced finite-size diagnostic.

---

## 3. Life-telemetry diagnostics

### 3.1 Implemented formulas

[`physics.life`](../src/tnfr/physics/life.py) consumes supplied time series; it
does not evolve the graph. It is an assumption-explicit logistic diagnostic,
not a biological classifier or an operator implementation. Every channel must
be a finite numeric one-dimensional series; Boolean, multidimensional, NaN and
infinite inputs are rejected. `detect_life_emergence` additionally requires a
nonempty nonnegative EPI series, matching channel shapes and strictly increasing
sample times. It requires $0\leq\varepsilon\leq1$, $\gamma\geq0$, and
$\mathrm{EPI}_{\max}>0$. No input is clipped or broadcast implicitly.

Let $x(t)\geq0$ be the supplied EPI-magnitude series. The declared model computes

$$
G(t)=\gamma x(t)\left(1-\frac{x(t)}{\mathrm{EPI}_{\max}}\right)
$$

and the **time-local** autopoietic coefficient

$$
A(t)=\frac{G(t)\,\dot x(t)}{|\Delta\mathrm{NFR}_{\mathrm{ext}}(t)|^2+\epsilon_{\mathrm{num}}}.
$$

The numerator and denominator are evaluated element by element. The
implementation does not take the ensemble or time averages shown in older
versions of this note, and `dEPI_dt` supplies $\dot x$ rather than the function
estimating it internally. The small $\epsilon_{\mathrm{num}}$ is the shared
safe division guard.

The remaining returned series are

$$
V_i(t)=\frac{|\varepsilon G(t)|}
{|\varepsilon G(t)|+|\Delta\mathrm{NFR}_{\mathrm{ext}}(t)|+\epsilon_{\mathrm{num}}},
$$

$$
S(t)=\frac{\varepsilon\,|\gamma(1-2x(t)/\mathrm{EPI}_{\max})|}
{|\partial_t\Delta\mathrm{NFR}_{\mathrm{ext}}(t)|+\delta+\epsilon_{\mathrm{num}}},
\qquad
M(t)=\frac{x(t)-\mathrm{EPI}_{\max}/2}{\mathrm{EPI}_{\max}}.
$$

`LifeTelemetry.vitality_index` is the internal-versus-total pressure ratio
$V_i$ above. It does not include $C(t)$; callers may combine the two as a
separate analysis.

### 3.2 Operational threshold detection

`detect_life_emergence` uses $A(t)>1$ as a selected operational event. It
linearly interpolates the first transition from $A\leq1$ to $A>1$, returns the
first supplied time when the series already starts above one, and otherwise
returns `None`. This classifier records a telemetry crossing; it does not prove
future self-sustenance or a biological classification.

The returned `LifeTelemetry` contains the supplied times, $V_i$, $A$, $S$, $M$,
and `life_threshold_time`.

---

## 4. Node lifecycle classifier

### 4.1 Implemented state priority

`get_lifecycle_state` is an instantaneous rule-based classifier. It does not
estimate whether $\nu_f$ or $\Delta\mathrm{NFR}$ is increasing. With the default
parameters, it evaluates conditions in this order:

| Returned state | Implemented condition |
|----------------|-----------------------|
| **COLLAPSING** | $\nu_f<0.01$, or $\lvert\Delta\mathrm{NFR}\rvert>10$, or a non-isolated node has coupling $<0.1$ |
| **MUTATION** | $\lvert\Delta\mathrm{NFR}\rvert>5$ and $\nu_f>0.1$ |
| **PROPAGATION** | coupling $>0.7$ and $\nu_f>0.1$ |
| **STABILIZATION** | $\lvert\Delta\mathrm{NFR}\rvert<1$ and scalar EPI $>0.8$ |
| **ACTIVATION** | $\nu_f\geq0.1$ after the earlier checks |
| **DORMANT** | all remaining states above the collapse-frequency cut |

The stabilization test uses scalar EPI as a proxy; it does not call the
canonical $C(t)$ kernel. `LifecycleState.COLLAPSED` exists in the enum but
`get_lifecycle_state` currently returns `COLLAPSING` for every collapse trigger
and never returns `COLLAPSED`.

For a node with neighbors, the current coupling approximation is

$$
c_i=1-\frac{\min\left(|\theta_i-\operatorname{mean}_{j\in\mathcal N(i)}
\theta_j|,\pi\right)}{\pi}.
$$

This is an arithmetic neighbor-phase mean followed by a capped absolute
difference, rather than a circular mean. Isolates receive $c_i=0$, but the
network-decoupling collapse check is applied only when neighbors exist.

### 4.2 Collapse-reason check

`check_collapse_conditions` is a separate predicate. It returns the first
matching reason in this order:

| Collapse reason | Default condition |
|-----------------|-------------------|
| **Frequency failure** | $\nu_f<0.01$ |
| **Extreme dissonance** | $\lvert\Delta\mathrm{NFR}\rvert>10$ |
| **Network decoupling** | non-isolated node with $c_i<0.1$ |
| **EPI dissolution** | scalar EPI $<0.01$ |

The EPI-dissolution condition belongs to this separate predicate and is not
consulted by `get_lifecycle_state`. Configuration values override graph values,
which in turn override these operational defaults.

**Implementation**:
[`operators.lifecycle`](../src/tnfr/operators/lifecycle.py) provides
`LifecycleState`, `CollapseReason`, `get_lifecycle_state`,
`check_collapse_conditions`, and `should_collapse`.

---

## 5. Auxiliary internal Hamiltonian

### 5.1 Implemented matrix

[`operators.hamiltonian`](../src/tnfr/operators/hamiltonian.py) constructs the
finite matrix

$$
H_{\mathrm{int}}=H_{\mathrm{coh}}+H_{\mathrm{freq}}+H_{\mathrm{coupling}},
$$

with the literal implementation

$$
H_{\mathrm{coh}}=C_0 W,\qquad
H_{\mathrm{freq}}=\operatorname{diag}(\nu_{f,1},\ldots,\nu_{f,N}),\qquad
H_{\mathrm{coupling}}=J_0 A_{\mathrm{sym}}.
$$

$W$ is the matrix returned by `coherence_matrix`. The constructor default is
$C_0=-1$, so the coherence term is $-W$; writing an additional leading minus
sign reverses the implemented sign. The default coupling is $J_0=0.1$, and the
builder writes both matrix directions for every graph edge. The constructor
checks every component and their sum for Hermiticity and raises when the check
fails.

This matrix supplies an auxiliary linear model. The repository does not derive
the general engine trajectory or the canonical graph $\Delta\mathrm{NFR}$ from
it.

### 5.2 Unitary flow and eigenmodes

For a constructor-accepted Hermitian matrix, `time_evolution_operator` computes

$$
U(t)=\exp\left(-\frac{iH_{\mathrm{int}}t}{\hbar_{\mathrm{str}}}\right).
$$

`get_spectrum` uses a Hermitian eigensolver and returns ascending real
eigenvalues and eigenvectors satisfying

$$
H_{\mathrm{int}}|\phi_n\rangle=E_n|\phi_n\rangle.
$$

An eigenvector evolves only by the phase
$e^{-iE_nt/\hbar_{\mathrm{str}}}$ in this auxiliary unitary flow. That makes it
a stationary ray of this model; it does not make it a maximally stable TNFR
configuration or establish dissipative attraction.

### 5.3 Compatibility helpers and sign scope

Despite its name, `compute_delta_nfr_operator()` does not compute a commutator.
It literally returns

$$
G_+=\frac{i}{\hbar_{\mathrm{str}}}H_{\mathrm{int}},
$$

which is anti-Hermitian and is the negative of the ket-state generator
$-iH_{\mathrm{int}}/\hbar_{\mathrm{str}}$ used by $U(t)$.

`compute_node_delta_nfr(n)` separately constructs
$\rho_n=|n\rangle\langle n|$ and returns the real part of

$$
\frac{i}{\hbar_{\mathrm{str}}}
\langle n|[H_{\mathrm{int}},\rho_n]|n\rangle.
$$

For this localized projector, the displayed diagonal commutator is exactly
zero: $[H,\rho_n]_{nn}=H_{nn}-H_{nn}=0$. Under the implemented unitary
$U\rho U^\dagger$, the density-matrix derivative would instead carry the sign
$-i[H,\rho]/\hbar_{\mathrm{str}}$. These helpers therefore do not reconstruct
the engine's node-local structural pressure; canonical $\Delta\mathrm{NFR}$ is
computed by
[`dynamics.dnfr`](../src/tnfr/dynamics/dnfr.py).

---

## 6. Optional structural integrity tools

### 6.1 Reactive monitor

The reactive monitor is opt-in. `enable_integrity_monitor(G, mode=...)` creates
a `StructuralIntegrityMonitor` and stores it in
`G.graph["integrity_monitor"]`. Calls through the operator-class pipeline then
invoke `before_operator` and `after_operator`. Without an attached monitor,
ordinary operator calls do not run these diagnostics.

For each monitored call, the implementation attempts to compare conservation
snapshots, the finite change of the structural-energy candidate, heuristic
grammar-violation labels, Noether-charge drift, and an operator-specific
postcondition. These are runtime diagnostics. They do not turn the
five-term energy into a general Lyapunov theorem. `ENFORCE` raises after a
reported unhealthy result; it does not roll back an operator mutation.

An `IntegrityReport` is healthy only when its conservation quality is above
`0.7`, its sampled energy change is classified stable, no grammar diagnostics
are present, and its postcondition check passes. Charge drift is reported but
is not part of that property.

### 6.2 Implemented postcondition registry

`POSTCONDITIONS` has one entry for every canonical operator name, but several
entries check only a measurable proxy and REMESH is explicitly advisory:

| Operator | Check currently performed by the reactive registry |
|----------|----------------------------------------------------|
| **AL** | EPI does not decrease; $\nu_f$, phase and $\Delta\mathrm{NFR}$ do not change |
| **EN** | immediate operator-local $C(t)$ is unchanged before pressure refresh |
| **IL** | $C(t)$ does not decrease and $\lvert\Delta\mathrm{NFR}\rvert$ does not increase |
| **OZ** | $\lvert\Delta\mathrm{NFR}\rvert$ does not decrease |
| **UM** | $\lvert\Delta\mathrm{NFR}\rvert$ does not increase |
| **RA** | nonzero EPI sign is preserved and $\nu_f$ does not decrease |
| **SHA** | EPI is unchanged and $\nu_f$ does not increase |
| **VAL** | $\nu_f$ does not decrease |
| **NUL** | $\nu_f$ does not increase |
| **THOL** | $C(t)$ does not fall by more than 10% |
| **ZHIR** | delegates phase, identity, and bifurcation checks to the mutation postcondition module |
| **NAV** | at least one of $\nu_f$, $\theta$, or $\Delta\mathrm{NFR}$ changes |
| **REMESH** | advisory entry; returns success without a network-remesh check |

The U3 phase-compatibility gate for UM and RA is a hard precondition in their
operator pipeline, separate from the reactive postcondition table. Registry
lookup uses the lower-case English function name (`"coherence"`,
`"self_organization"`, and so on); a glyph string such as `"IL"` does not
select the corresponding registry checker.

#### ZHIR temporal evidence and execution boundary

The nodal equation gives the instantaneous prediction
`predicted_depi_dt = nu_f * DeltaNFR`. Its signed strict comparison with `xi`
is useful as a current-state prediction, but it is not a measurement of a
realized trajectory and cannot satisfy the ZHIR gate by itself. The SDK keeps
the historical name `near_bifurcation` as an alias of this predicted crossing.

ZHIR has a separate non-disableable admission gate based on an **observed signed
secant**. Timestamped `epi_time_history` supplies
`(EPI[k] - EPI[k-1]) / (t[k] - t[k-1])`; both samples and the interval must be
finite, the interval must be strictly positive, and the final EPI sample must
represent the current endpoint. The runtime default requires an exact endpoint
match. Once physical history is supplied it is authoritative, so an invalid or
stale record does not fall back to an untimestamped source. Event-schedule
execution writes such evidence only when a ZHIR pre-flow's binary64 timestamp
difference exactly recovers the schedule's authoritative declared duration;
otherwise it rejects the schedule before graph writes.

The compatibility channels `epi_history` and `_epi_history` use two finite
scalar samples separated by one operator step. Their difference is therefore a
legacy unit-step rate, explicitly reported as
`physical_time_resolved=False`; it makes no claim about physical elapsed time or
endpoint provenance. In both evidence modes the gate requires the signed strict
inequality `observed_depi_dt > xi`. Equality, contraction, invalid or missing
history, non-increasing physical time, and a stale physical endpoint reject
direct execution. The default `ZHIR_THRESHOLD_XI = 0.1` is an operational
calibration. Direct execution also requires active finite capacity
(`nu_f > 0`), while `ZHIR_MIN_VF` can only tighten that condition.

Dynamic selection treats unavailable, invalid, or non-crossing Mutation
evidence as an abstention: it substitutes IL before ordinary grammar enforcement
and records the requested and applied glyphs with the reason in
`mutation_abstentions`. The SDK whole-word runner preflights every target node
before any operator in a word containing ZHIR. It also rejects timestamped
evidence when an earlier EPI-channel operator in the same word would make its
endpoint stale before Mutation; legacy histories retain their unit-operator-step
compatibility semantics. Missing or invalid evidence leaves the certificate's
`observed_crossed` value unknown (`None`), rather than treating absence as an
observed non-crossing. The observed-predicted `rate_gap` is available only for
valid physical evidence because legacy evidence has no physical-time basis.

`compute_d2epi_dt2` is a separate three-sample estimate of structural
acceleration: it compares adjacent timestamped secants for physical histories
and uses the unit-step second difference for legacy histories. It may support
bifurcation-potential telemetry; it is not the two-sample ZHIR threshold gate.
Likewise,
`ZHIR_BIFURCATION_VF_THRESHOLD = 0.5` controls whether a branch selector
proposes ZHIR and is not an admission precondition. The pure
`MutationTriggerCertificate` and SDK `nodal_state()` expose prediction and
evidence without evaluating U4b. Prior IL and a recent destabilizer remain
separate U4b context requirements enforced by the grammar or strict
precondition route; neither read-only interface certifies execution readiness.

Mutation postconditions read `epi_kind` as structural identity. Successful
dispatch records `source_glyph = "ZHIR"` as operator provenance without changing
that identity; `source_glyph` and `epi_kind` are distinct metadata channels.

Registry coverage is therefore not a proof that every full operator contract
has been verified. For a reproducible catalog-level measurement, use
`audit_operator_contracts`. It builds controlled graphs, places each request in
its intended context, checks the actually appended glyph so a grammar fallback
cannot certify the request, and returns an `OperatorContractAudit`. Its REMESH
case remains an advisory network-level result.

### 6.3 Monitor modes

| Mode | Behaviour |
|------|-----------|
| **OFF** | Hook methods return default data without metric computation |
| **OBSERVE** | Record reports and violations without raising |
| **ENFORCE** | Raise `StructuralIntegrityViolation` on failure |

### 6.4 Suggestions and SDK scope

Corrective suggestions are generated for recognized conservation-derived
grammar labels, an increasing energy-candidate observation, or charge drift
above the internal alert. A postcondition failure alone need not produce a
suggestion.

The SDK exposes two different dictionary-returning conveniences:

- `Network.integrity_check(operator_name)` calls `after_operator` directly for
  at most ten current nodes and returns `operator`, `nodes_checked`, `passed`,
  `failed`, `pass_rate`, and per-node `reports`. It does not execute an
  operator, capture a matching before snapshot, attach the monitor, or audit
  all 13 operators. Use an English function name to activate a registry check.
- `Network.audit_operators()` runs the independent controlled
  `audit_operator_contracts` protocol and returns a dictionary with the 13
  contextual results and its summary. It audits the operator implementation,
  rather than the current `Network` instance's trajectory.

**Implementation**:
[`physics.integrity`](../src/tnfr/physics/integrity.py) provides
`IntegrityReport`, `IntegritySummary`, `MonitorMode`,
`StructuralIntegrityViolation`, `POSTCONDITIONS`, and
`audit_operator_contracts`.

**Tests**:
[`tests/physics/test_structural_integrity.py`](../tests/physics/test_structural_integrity.py)
and
[`tests/sdk/test_simple_advanced.py`](../tests/sdk/test_simple_advanced.py).

---

## Implementation and examples

| Module | Content |
|--------|---------|
| `src/tnfr/physics/lyapunov.py` | Nominal U2-role multipliers and spectral diagnostics |
| `src/tnfr/physics/structural_diffusion.py` | Fixed, time-varying and exact-common-metric pure-EPI flow certificates |
| `src/tnfr/physics/hybrid_operator_stability.py` | Declared affine-reset gains and hybrid flow/reset budgets |
| `src/tnfr/physics/reception_realization.py` | Read-only EN runtime-to-affine-flow boundary |
| `src/tnfr/physics/resonance_realization.py` | Read-only identity-gated RA runtime-to-affine-flow boundary and post-RA metric audit |
| `src/tnfr/physics/network_stage_stability.py` | All-target EN/RA certificates and the validated one-stage positive-duration post-flow bridge |
| `src/tnfr/physics/pointwise_stage_stability.py` | Executor-bound pointwise affine realization and gain levels |
| `src/tnfr/operators/event_timing.py` | Exact finite flow/jump schedules and binary64 clock readiness |
| `src/tnfr/operators/event_runtime.py` | Atomic observed flow/glyph binding and finite represented EPI-map composition |
| `src/tnfr/operators/event_remesh_runtime.py` | Atomic schedule/delayed-REMESH cycle with separate evidence channels |
| `src/tnfr/operators/event_remesh_sequence.py` | Exact continuity across ordered supplied cycle results |
| `src/tnfr/operators/event_remesh_causal_runtime.py` / `src/tnfr/operators/event_remesh_causal_runtime.pyi` | One graph-owned finite causal cycle sequence and exact public interface |
| `src/tnfr/physics/event_refinement.py` | Offline and executor-linked event-local ZHIR evidence |
| `src/tnfr/physics/event_remesh_refinement.py` | Finite strict three-mesh event/REMESH observations |
| `src/tnfr/physics/event_remesh_reference.py` / `src/tnfr/physics/event_remesh_reference.pyi` | Effective-P2 finite reference-family certificate and exact public interface |
| `src/tnfr/physics/reversible_eigenmode_reference.py` / `src/tnfr/physics/reversible_eigenmode_reference.pyi` | General exact-rational reversible single-eigenmode Euler theorem and public interface |
| `src/tnfr/physics/runtime_eigenmode_reference.py` / `src/tnfr/physics/runtime_eigenmode_reference.pyi` | Finite executor binding with exact pressure/execution defects and full-matrix propagation |
| `src/tnfr/physics/remesh_history_stability.py` | Exact uniform finite companion-history Lyapunov certificate |
| `src/tnfr/physics/remesh_schedule_policy_stability.py` / `src/tnfr/physics/remesh_schedule_policy_stability.pyi` | Conditional exact uniform REMESH/schedule spatial-disagreement theorem |
| `src/tnfr/physics/remesh_schedule_relative_defect_stability.py` / `src/tnfr/physics/remesh_schedule_relative_defect_stability.pyi` | Conditional exact `q_eff=q*(1+eta)` robust policy theorem |
| `src/tnfr/physics/binary64_remesh_relative_defect.py` / `src/tnfr/physics/binary64_remesh_relative_defect.pyi` | Exact pairwise REMESH boundary; uniform `alpha=1`, `eta=0` class; and sharp `alpha=1/2` antisymmetric P2 class with `eta=135/124` |
| `src/tnfr/physics/binary64_p2_reception_stability.py` / `src/tnfr/physics/binary64_p2_reception_stability.pyi` | Global `q=0` P2 half-Reception kernel composed with the `alpha=1` REMESH class |
| `src/tnfr/physics/runtime_p2_reception_stage.py` / `src/tnfr/physics/runtime_p2_reception_stage.pyi` | Finite executor binding of one P2 two-phase EN EPI stage to the global `q=0` kernel |
| `src/tnfr/physics/runtime_p2_reception_remesh_sequence.py` / `src/tnfr/physics/runtime_p2_reception_remesh_sequence.pyi` | Finite causal binding of executed P2 EN/REMESH cycles and observed active-suffix extinction |
| `src/tnfr/physics/runtime_p2_reception_remesh_policy.py` / `src/tnfr/physics/runtime_p2_reception_remesh_policy.pyi` | Transactional per-invocation P2 preflight, execution and finite certification |
| `src/tnfr/physics/runtime_remesh_history_stability.py` | One-transition executed binary64 REMESH/companion bridge |
| `src/tnfr/physics/remesh_schedule_stability.py` | Exact REMESH-head/schedule-head augmented-energy balance |
| `src/tnfr/physics/runtime_remesh_schedule_stability.py` | Adjacent-cycle runtime/history energy telescope |
| `src/tnfr/physics/runtime_remesh_schedule_block_margin.py` / `src/tnfr/physics/runtime_remesh_schedule_block_margin.pyi` | Exact normalized margin for a contiguous causally executed finite block |
| `src/tnfr/physics/runtime_remesh_schedule_relative_defect.py` / `src/tnfr/physics/runtime_remesh_schedule_relative_defect.pyi` | Finite causal verification of signed defects and the robust policy envelope |
| `src/tnfr/operators/_delayed_remesh_kernel.py` | Immutable delayed REMESH proposals and one-step evidence |
| `src/tnfr/physics/phase_quotient.py` | Fixed-branch pairwise quotient, restricted canonical phase lift and counterexample |
| `src/tnfr/physics/coherence_geometry.py` | Local, fixed-network and fixed-capacity coherence strata |
| `src/tnfr/physics/phase_transition.py` | Order parameter, operational phase classification, effective exponent fit |
| `src/tnfr/physics/life.py` | Strict supplied-series logistic diagnostics and selected $A(t)>1$ event |
| `src/tnfr/operators/lifecycle.py` | Instantaneous node-state and collapse predicates |
| `src/tnfr/operators/hamiltonian.py` | Auxiliary matrix, unitary flow, spectrum, compatibility helpers |
| `src/tnfr/physics/integrity.py` | Optional reactive monitor and contextual operator audit |

The finite three-mesh contracts are falsified and sealed by
[`test_event_remesh_refinement.py`](../tests/physics/test_event_remesh_refinement.py).
The effective-P2 reference family and public example are checked by
[`test_event_remesh_reference.py`](../tests/physics/test_event_remesh_reference.py)
and
[`test_event_remesh_reference_example.py`](../tests/physics/test_event_remesh_reference_example.py).
The general reversible exact-mode theorem, sealing and public `P3` example are
checked by
[`test_reversible_eigenmode_reference.py`](../tests/physics/test_reversible_eigenmode_reference.py)
and
[`test_reversible_eigenmode_reference_example.py`](../tests/physics/test_reversible_eigenmode_reference_example.py).
The finite executor binding, proof seals and public executed `P3` example are
checked by
[`test_runtime_eigenmode_reference.py`](../tests/physics/test_runtime_eigenmode_reference.py)
and
[`test_runtime_eigenmode_reference_example.py`](../tests/physics/test_runtime_eigenmode_reference_example.py).
The finite companion-history theorem is checked exactly by
[`test_remesh_history_stability.py`](../tests/physics/test_remesh_history_stability.py).
The common-`q` exact policy theorem, public facade and example are checked by
[`test_remesh_schedule_policy_stability.py`](../tests/physics/test_remesh_schedule_policy_stability.py)
and
[`test_remesh_schedule_policy_stability_example.py`](../tests/physics/test_remesh_schedule_policy_stability_example.py).
The robust signed-defect theorem and its finite causal adapter are checked by
[`test_remesh_schedule_relative_defect_stability.py`](../tests/physics/test_remesh_schedule_relative_defect_stability.py),
[`test_runtime_remesh_schedule_relative_defect.py`](../tests/physics/test_runtime_remesh_schedule_relative_defect.py)
and
[`test_runtime_remesh_schedule_relative_defect_example.py`](../tests/physics/test_runtime_remesh_schedule_relative_defect_example.py).
The binary64 REMESH boundary, P2 numeric EPI-kernel composition and public
examples are checked by
[`test_binary64_remesh_relative_defect.py`](../tests/physics/test_binary64_remesh_relative_defect.py),
[`test_half_alpha_antisymmetric_remesh_class.py`](../tests/physics/test_half_alpha_antisymmetric_remesh_class.py),
[`test_binary64_p2_reception_stability.py`](../tests/physics/test_binary64_p2_reception_stability.py),
[`test_binary64_remesh_relative_defect_example.py`](../tests/physics/test_binary64_remesh_relative_defect_example.py),
[`test_half_alpha_antisymmetric_remesh_class_example.py`](../tests/physics/test_half_alpha_antisymmetric_remesh_class_example.py)
and
[`test_binary64_p2_reception_stability_example.py`](../tests/physics/test_binary64_p2_reception_stability_example.py).
The finite executed-stage binding and example are checked by
[`test_runtime_p2_reception_stage.py`](../tests/physics/test_runtime_p2_reception_stage.py)
and
[`test_runtime_p2_reception_stage_example.py`](../tests/physics/test_runtime_p2_reception_stage_example.py).
The finite causal P2 EN/REMESH sequence, active-suffix boundary and public
example are checked by
[`test_runtime_p2_reception_remesh_sequence.py`](../tests/physics/test_runtime_p2_reception_remesh_sequence.py)
and
[`test_runtime_p2_reception_remesh_sequence_example.py`](../tests/physics/test_runtime_p2_reception_remesh_sequence_example.py).
The reusable policy, its rollback boundaries and two-call example are checked by
[`test_runtime_p2_reception_remesh_policy.py`](../tests/physics/test_runtime_p2_reception_remesh_policy.py)
and
[`test_runtime_p2_reception_remesh_policy_example.py`](../tests/physics/test_runtime_p2_reception_remesh_policy_example.py).
The runtime residual and lifted-energy bridge is tested in
[`test_runtime_remesh_history_stability.py`](../tests/physics/test_runtime_remesh_history_stability.py).
The pure schedule-head telescope and gain-based lower bound are tested in
[`test_remesh_schedule_stability.py`](../tests/physics/test_remesh_schedule_stability.py).
The adjacent runtime schedule binding and finite history telescope are tested in
[`test_runtime_remesh_schedule_stability.py`](../tests/physics/test_runtime_remesh_schedule_stability.py).
The finite same-invocation causal wrapper, receipt bindings, graph rollback and
public example are tested in
[`test_event_remesh_causal_runtime.py`](../tests/operators/test_event_remesh_causal_runtime.py)
and
[`test_event_remesh_causal_runtime_example.py`](../tests/operators/test_event_remesh_causal_runtime_example.py).
The exact finite-block identities, normalized diagnostics, public facade and
`139/256` versus zero-margin examples are checked by
[`test_runtime_remesh_schedule_block_margin.py`](../tests/physics/test_runtime_remesh_schedule_block_margin.py)
and
[`test_runtime_remesh_schedule_block_margin_example.py`](../tests/physics/test_runtime_remesh_schedule_block_margin_example.py).

### SDK Entry Points

```python
from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)
monitor = enable_integrity_monitor(net.G, mode=MonitorMode.OBSERVE)
# Subsequent operator-class calls append IntegrityReport objects to monitor.summary.

snapshot = net.integrity_check("coherence")  # dict; up to ten current nodes
catalog = net.audit_operators()               # dict; 13 controlled probes
```

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [29_lyapunov_stability_demo.py](../examples/02_physics_regimes/29_lyapunov_stability_demo.py) | Nominal operator-role multipliers, measured energy diagnostics, spectral read-outs, and life telemetry |
| [161_core_research_trajectory.py](../examples/02_physics_regimes/161_core_research_trajectory.py) | Sampled pure-EPI path, modal limit, common energy budget and two-mesh comparison |
| [162_hybrid_epi_stability.py](../examples/02_physics_regimes/162_hybrid_epi_stability.py) | Affine amplification absorbed by diffusion, consensus drift, and the infinite-gain local-offset witness |
| [163_reception_runtime_bridge.py](../examples/02_physics_regimes/163_reception_runtime_bridge.py) | EN ideal-real, represented, runtime-snapshot and pressure-refresh boundary |
| [164_resonance_runtime_bridge.py](../examples/02_physics_regimes/164_resonance_runtime_bridge.py) | U3-filtered RA identity gate, four realization layers, post-RA flow certificate and switching abstention |
| [166_event_remesh_reference_family.py](../examples/02_physics_regimes/166_event_remesh_reference_family.py) | Effective-P2 `2/4/8`-segment finite Euler/REMESH reference family and explicit false scope |
| [167_reversible_eigenmode_reference.py](../examples/02_physics_regimes/167_reversible_eigenmode_reference.py) | Both exact nonuniform modes of nonregular `P3`, with exact-real Euler refinement bounds and explicit runtime abstention |
| [168_runtime_reversible_eigenmode_reference.py](../examples/02_physics_regimes/168_runtime_reversible_eigenmode_reference.py) | Finite `2/4/8`-segment executed nonregular-`P3` binding with nonzero `rho`, `eta`, `epsilon` and explicit false runtime-convergence scope |
| [169_event_remesh_causal_runtime.py](../examples/02_physics_regimes/169_event_remesh_causal_runtime.py) | One finite same-invocation event/REMESH cycle sequence with causal receipts, graph-owned atomicity, and the lag-one `alpha=1` stability boundary |
| [170_runtime_remesh_block_margin.py](../examples/02_physics_regimes/170_runtime_remesh_block_margin.py) | Exact `kappa=139/256` finite-block lower margin and the causal `alpha=1`, `kappa=0` boundary, with uniform-class and prefix claims withheld |
| [171_remesh_schedule_policy_stability.py](../examples/02_physics_regimes/171_remesh_schedule_policy_stability.py) | Conditional exact common-`q` policy theorem with prefix gain upper bound one, uniform block margin `1-q`, strict pure-delay disagreement decay and the zero-margin `q=1` boundary |
| [172_runtime_remesh_relative_defect.py](../examples/02_physics_regimes/172_runtime_remesh_relative_defect.py) | Exact `q_eff=q*(1+eta)` theorem bound to zero-defect and positive-binary64-defect finite causal blocks, with forward invariance and future stability withheld |
| [173_binary64_remesh_relative_defect.py](../examples/02_physics_regimes/173_binary64_remesh_relative_defect.py) | Exact pairwise `alpha=1/2` obstruction and the forward-invariant REMESH-only `alpha=1`, `eta=0` hard-clip class |
| [174_binary64_p2_reception_remesh_stability.py](../examples/02_physics_regimes/174_binary64_p2_reception_remesh_stability.py) | Global `q=0` P2 half-Reception numeric kernel composed with `alpha=1`, giving active-history extinction after `tau_global+1` restricted cycles |
| [175_runtime_p2_reception_stage.py](../examples/02_physics_regimes/175_runtime_p2_reception_stage.py) | One executed two-phase P2 EN EPI stage bound to the global `q=0` kernel, with REMESH graph binding and repeated runtime withheld |
| [176_runtime_p2_reception_remesh_sequence.py](../examples/02_physics_regimes/176_runtime_p2_reception_remesh_sequence.py) | One completed same-invocation P2 EN/REMESH sequence with `N >= tau_global+1` and finite observed active-history extinction |
| [177_runtime_p2_reception_remesh_policy.py](../examples/02_physics_regimes/177_runtime_p2_reception_remesh_policy.py) | Two successive independently validated finite P2 policy invocations with future and auxiliary-state claims withheld |
| [178_half_alpha_antisymmetric_remesh_class.py](../examples/02_physics_regimes/178_half_alpha_antisymmetric_remesh_class.py) | Sharp `eta=135/124` for the REMESH-only `alpha=1/2` antisymmetric P2 class, exact IEEE tail/core proof, strict and zero-margin `q` boundaries, and excluded generalizations |

## Cross-References

- Structural-energy candidate and conservation diagnostics: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) §8
- Grammar U2 (convergence): [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Hamiltonian/Lagrangian formulation: [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md)
- Order parameter $\mathcal{S}$: [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) §3.2
- Dissipative extensions: [DISSIPATIVE_AND_OPEN_SYSTEMS.md](DISSIPATIVE_AND_OPEN_SYSTEMS.md)
- Gauge structure: [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md)
