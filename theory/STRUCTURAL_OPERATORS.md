# TNFR Structural Operators

## Complete Specification of the 13 Canonical Operators

**Status**: CANONICAL registry specification; generative completeness over an
independently defined admissible-transformation space remains open
**Date**: March 2026  
**Version**: 0.0.3.5
**Prerequisite**: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §2 (Nodal Equation), [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) (Grammar U1–U6)

---

## Table of Contents

1. [Scope and Motivation](#1-scope-and-motivation)
2. [Operator Algebra from the Nodal Equation](#2-operator-algebra-from-the-nodal-equation)
3. [The Operator Taxonomy](#3-the-operator-taxonomy)
4. [Generators](#4-generators)
5. [Integrator](#5-integrator)
6. [Stabilizers](#6-stabilizers)
7. [Destabilizers](#7-destabilizers)
8. [Coupling and Propagation](#8-coupling-and-propagation)
9. [Transformers](#9-transformers)
10. [Closure and Regime Operators](#10-closure-and-regime-operators)
11. [Fragments and Complete Words](#11-fragments-and-complete-words)
12. [Operator Roles and Energy Diagnostics](#12-operator-roles-and-energy-diagnostics)
13. [Postcondition Contracts](#13-postcondition-contracts)
14. [Operator Constants Reference](#14-operator-constants-reference)
15. [Implementation Reference](#15-implementation-reference)
16. [Summary](#16-summary)
17. [Experimental Operator-Tetrad Synergies](#17-experimental-operator-tetrad-synergies)

---

## 1. Scope and Motivation

Registered structural operators are the canonical semantic interface for named
node and network transformations. Declared numerical solvers are a separate,
explicit evolution path: they may advance EPI only through the shared nodal
integrator from finite `nu_f` and `DeltaNFR`, while recording the associated
provenance or residual. Ad hoc state assignment outside these two declared
paths is forbidden. Both paths remain constrained by the nodal equation:

$$
\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t) \tag{NE}
$$

Each registered operator implements a specific transformation tied to (NE). The
13-operator catalog covers the engine's declared transformation categories:
creation, integration, stabilization, destabilization, coupling, propagation,
freezing, dimensional change, self-organization, phase transformation, regime
transition, and multi-scale recursion. Coverage of this declared catalog does
not prove that it generates every independently admissible TNFR transformation.

### 1.1 Why the registered catalog has 13 members

The registry contains 13 public semantic transformations satisfying:

1. **Nodal equation compatibility**: Every transformation must declare its effect on
   EPI, $\nu_f$, $\Delta\text{NFR}$, phase, or the coupling structure.
2. **Grammar closure**: The set must include generators (U1a), closures (U1b), stabilizers (U2), destabilizers (U2), coupling operators (U3), bifurcation triggers and handlers (U4), and multi-scale operators (U5).
3. **Semantic distinction**: Each member has a named physical contract. The
   finite temporal-signature experiment separates their implementations on its
   declared probes, but global algebraic irreducibility remains open.

A proof of completeness first requires a transformation space defined without
reference to this catalog. That is research line S10 in
[CORE_RESEARCH_PROGRAM.md](CORE_RESEARCH_PROGRAM.md).

### 1.2 Conventions

Throughout this document:
- Glyph codes (AL, EN, IL, ...) reference the structural symbols.
- Lowercase English tokens (`emission`, `reception`, ...) are executable API
  identifiers; title-case names are public display/class names.
- Operator gain magnitudes are operational parameters (only $\pi$ is a genuine structural scale); the engine-configuration tier in `canonical.py` is calibrated, not derived.
- Grammar roles reference rules U1–U6 from [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md).
- Energy diagnostics and their proof boundaries are discussed in [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md).

---

## 2. Operator Algebra from the Nodal Equation

### 2.1 Structural Triad

Each node $i$ carries three irreducible attributes:

| Attribute | Symbol | Domain | Units |
|-----------|--------|--------|-------|
| Form | $\text{EPI}_i$ | $\mathcal{B}_{\text{EPI}}$ (Banach space) | — |
| Frequency | $\nu_{f,i}$ | $\mathbb{R}^+$ | Hz_str |
| Phase | $\phi_i$ (or $\theta_i$) | $[0, 2\pi)$ | rad |

The derived quantity $\Delta\text{NFR}_i$ (structural pressure) drives evolution.
Every operator has exactly one registered **primary** channel among
$(\text{EPI}, \nu_f, \phi, \Delta\text{NFR})$; a concrete implementation may
also update declared secondary channels.

### 2.2 Operator as Transformation

An operator $\hat{O}$ maps the node state $\sigma_i = (\text{EPI}_i, \nu_{f,i}, \phi_i, \Delta\text{NFR}_i)$ to a new state:

$$
\hat{O}: \sigma_i \mapsto \sigma_i' = (\text{EPI}_i', \nu_{f,i}', \phi_i', \Delta\text{NFR}_i')
$$

subject to:
1. **Nodal equation**: The resulting state must be consistent with $\partial\text{EPI}/\partial t = \nu_f \cdot \Delta\text{NFR}$.
2. **Grammar constraints**: The operator must satisfy its declared U1--U5 role;
   U6 is evaluated from before/after $\Phi_s$ telemetry.
3. **Contracts**: Pre-conditions and post-conditions specific to each operator.

### 2.3 Composition

Operators compose into sequences $[\hat{O}_1, \hat{O}_2, \ldots, \hat{O}_n]$ applied left-to-right. Grammar validation operates on the full sequence. The grammar is not commutative: the order of operators affects validity and outcome.

SDK execution is operator-major across sequence positions, as are the
supported GPU blocks. All thirteen SDK positions use a shared atomic two-phase
Jacobi stage when grammar accepts the requested glyph for every target:
EN/IL/RA build neighbour-reading proposals, AL/SHA/VAL/NUL/ZHIR/NAV build
pointwise proposals, UM merges overlapping phase/topology proposals, OZ reduces
overlapping local and propagated pressure proposals, and THOL merges child
support and hierarchy proposals, and REMESH merges one shared advisory per telemetry step. IL contracts each target's
pressure magnitude and computes circular
phase locking from the immutable stage snapshot; canonical structural `C(t)`
telemetry and auxiliary pressure dispersion remain separate.
Supported GPU AL/RA blocks reuse the corresponding path. Every target reads one
immutable stage-start snapshot before any proposal commits. The committed
primary channels are invariant to target iteration order before the opaque
pressure-refresh callback; AL and SHA additionally use one shared timestamp
per stage, while NAV uses one shared latency-observation instant. A random NAV
stage binds each node's draw and RNG progress to the snapshot; a missing graph
seed is resolved inside the transaction and persists only after successful
commit. Ordered lifecycle, audit/telemetry and monitor streams retain target
order. UM averages shortest-arc phase displacements in immutable snapshot-rank
order, normalizes the result, revalidates U3 after the merge and deterministically
coalesces accepted functional links. This reducer is an engine policy, not a
stability theorem. OZ derives local actions and outgoing propagation from the
same snapshot, then sums incoming increments with `math.fsum` in snapshot-node
rank. Its local pressure-magnitude postcondition precedes signed accumulation:
positive incoming increments can partially cancel a negative pressure. THOL
allocates colliding child identifiers in snapshot-node rank and validates all
child nodes, `sub_nodes`, `sub_epis` and graph `hierarchy` writes on a
detached candidate before committing `d2EPI`, `DeltaNFR` and support. IL
warnings are emitted as the final transactional effect. Cache state
and the opaque pressure refresh are excluded from the invariance claim;
identity-bearing caches remain tied to their live graph and node objects. No
relabeling-equivariance result follows.

The REMESH glyph stage is advisory-only: it leaves structural channels and
support unchanged and deterministically deduplicates one graph event per
telemetry step. The separate apply_network_remesh operation owns delayed EPI
mixing. Multi-target IL previously used the sequential schedule; its canonical
word runner now uses the snapshot rule. A grammar replacement or Recursivity
execution override still follows and reports the transactional Gauss-Seidel
path. Thus the guarantee is stage-local and does not make a mixed word
simultaneous.

The hybrid time contract treats each operator position as a zero-duration jump
and assigns physical time only to the `m + 1` declared nodal-flow intervals
around `m` events. Exact rational values of materialized binary64 durations and
their offsets are authoritative; absolute timestamps are display fields, and
coincident jumps remain ordered by event index. The schedule builder itself
does not execute the word or manufacture timestamped EPI secants.

`execute_operator_event_schedule` supplies the separate runtime binding. It uses
the configured nodal integrator once for each positive unpartitioned interval.
An explicit `PhysicalFlowPartition` instead invokes it once per segment and
refreshes pressure at every segment boundary, including the terminal boundary.
The shared all-target dispatcher executes each jump with the initial target tuple
frozen. Collapsed or nonadditive binary64 intervals are rejected before writes.
Flow boundaries become timestamped samples, while a same-time EPI jump restarts
that history and is recorded only as a zero-duration event. One graph transaction
covers flow, jump, history, cache and event-log state; emitted external effects
remain outside rollback.

An integrator may write only EPI, `dEPI_dt`, `d2EPI_dt2` and the runtime clock
during one positive flow. A custom integrator must satisfy the exact
represented-rational held-input nodal identity
`EPI_right-EPI_left = dt*nu_f_left*DeltaNFR_left` for every target; this is not
a solver-accuracy or order certificate. `GraphTransactionSnapshot` is bound to
its originating graph. It restores NetworkX mapping identities and alias
relations plus graph-reachable capturable callback state. Mutable structural
keys with custom value hash/equality, custom `__deepcopy__` hooks and unmodelled
opaque C state are rejected before writes; common immutable atoms are accepted.
I/O, emitted warnings, external resources and aliases held only outside the
graph are not rollback state.

For physical ZHIR, `observe_event_local_zhir_physical_prejump` remains an
offline coordinate pairing. The stricter
`observe_executed_event_local_zhir_physical_prejump` starts from one intact
stage-certified execution and binds its scheduled and committed event,
preceding physical partition, terminal pre-flow, glyph stage, ordered decisions
and trigger certificates by identity. It certifies common execution provenance
for that finite window only; solver order, convergence, general U4 readiness
and future behavior remain outside it.

With `include_stage_certificates=True`, flow capture is implied. The runtime
binds executor-owned pointwise AL/SHA/VAL/NUL/ZHIR/NAV or all-target EN/RA
certificates to the EPI endpoints captured around each actual stage and to its
adjacent flow evidence. Unpartitioned parents use their direct interval record;
physical parents use the terminal pre-jump or initial post-jump segment record.
Unsupported glyphs and rejected certificate domains abstain explicitly. A
complete `ObservedRepresentedEPIScheduleComposition` orders every positive
scheduled parent and glyph as a `RepresentedEPIScheduleOperation`. A physical
parent can contribute the product of its segment gains only when every exact
affine map is intact and one normalized rational metric spans the segments. The
complete trace additionally requires exact node order, endpoint continuity and
one common metric. This globally bounds the represented maps and, through
endpoint binding, the observed trace. `runtime_schedule_global_gain_certified` remains
false: no global executable binary64 map, solver accuracy, refinement
invariance, full-multichannel result, future schedule, repetition or adaptive
U2/U4 policy is certified.

`execute_event_remesh_cycle` supplies a narrower executable composition. It
runs one event schedule, appends its endpoint through the ordinary pre-REMESH
`_epi_hist` helper, and invokes the separate delayed map inside an outer graph
transaction. Ordered node support and incoming delayed history are fixed across
the schedule; edges may change there. The endpoint clock, event log, phase,
pressure hook and deterministic delayed-map controls remain bound to the
schedule result. The EPI-only map treats ON_REMESH callbacks as observers and
requires them to leave all graph-owned topology, metadata, histories and stored
non-EPI aliases unchanged. Capturable state reachable through a graph-owned
observer belongs to the same rollback boundary.

One frozen positive diagonal metric measures the cycle-level pre-schedule,
pre-REMESH and post-REMESH EPI observations and is passed unchanged to
stability evidence; REMESH metadata retains legacy unweighted summaries. Exact
weighted-mean drift and disagreement remain authoritative when their optional
binary64 display is `None`. Capacity vectors, stage refresh counts and an
explicitly requested post-REMESH pressure refresh stay separate. That refresh
runs only after an applied map. Delay `tau` reads `_epi_hist[-(tau + 1)]`, and no
post-jump delayed-history sample is appended. The applied jump instead restarts
or appends the same-time `epi_time_history` right endpoint, so Mutation cannot
read it as a finite flow secant. The cycle may retain the finite represented
flow/glyph composition in its event result, but the delayed map and its one-step
evidence remain separate. Atomic execution does not provide a global binary64
runtime gain, solver-accuracy or history-updated repetition theorem. Capturable
graph-reachable integrator and callback state is covered; emitted I/O or warnings,
external resources and external-only aliases remain outside rollback.
The cycle materializes `physical_flow_partitions` exactly once inside this outer
transaction and passes the resulting tuple unchanged to the event executor.

Every cycle also seals a `RemeshHistoryTransitionObservation`. With bounded
history size `M`, incoming exact history `H_in` and exact pre-REMESH EPI vector
`x_pre`, it verifies `H_out = tail_M(tail_M(H_in) || (x_pre,))`. Rebuild
truncation, append eviction and the local and global selections
`H_out[-(tau + 1)]` are explicit; the two declared lags are tested
independently. Completion of a requested post-REMESH pressure callback is an
operational fact and does not establish `DeltaNFR = -L_rw EPI`.

The pure `compose_event_remesh_cycle_observations` observer binds at least two
sealed results supplied in caller order. Its sealed
`EventRemeshCycleBoundaryObservation`
objects compare exact recorded EPI, capacity, pressure, phase, clock and full
delayed history at adjacent supplied-result boundaries. Sequence indices are
local ordinals. Identical-object reuse is rejected, but distinct copies do not
prove consecutive calls or shared-graph provenance.
`ObservedEventRemeshCycleSequence` distinguishes identical raw metric weights
from exact normalized-ray compatibility and reports tri-state alignment with
each nested represented schedule metric. It retains nested schedule
compositions and REMESH results as separate evidence.
Its exact recorded-boundary result is independent of metric compatibility. A
separate stronger sequence result requires one common normalized ray and a
matching exposed metric from every nested schedule; raw metric equality remains
diagnostic. A `None` or `False` nested alignment blocks only the stronger result,
not exact recorded-boundary continuity.

This boundary supplies no mixed gain product, evolving-history repetition,
runtime-global gain, whole-sequence atomicity, full graph or grammar-history
continuity, solver or future theorem. The omission is material: for lag one,
`alpha=1`, EPI `(2,0)` and prior delayed row `(0,2)`, one explicit sequential
empty-schedule execution alternates `(2,0) -> (0,2) -> (2,0)`. Each fixed-history
REMESH record has zero current-state coefficient, so its one-step factor cannot
be multiplied across the changing history.

`execute_event_remesh_cycle_sequence` adds one graph-mutating outer boundary
without changing that observer. It executes ordered
`EventRemeshCycleExecutionSpec` values on one graph in one transaction. Every
sealed `CausalEventRemeshCycleReceipt` binds an execution ordinal, the exact
spec and the resulting cycle, including identity of the declared and executed
schedule. The sealed `ExecutedEventRemeshCycleSequence` always retains the
ordinary offline cycle observation. By default it requires and retains a
compatible runtime schedule/history telescope. The explicit
`require_runtime_telescope=False` branch always records
`runtime_telescope=None`, thereby admitting grammar-valid words that have no
common represented affine metric. In both branches only the outer result proves same-invocation
causal order, common graph identity and finite graph-owned atomicity. The
no-telescope branch makes every telescope-specific claim false. The wrapper
supplies no global schedule/REMESH gain,
uniform repeated margin, solver accuracy/order, mesh convergence, repetition or
future stability. Emitted I/O, warnings, external resources and external-only
aliases remain outside rollback.

`observe_event_remesh_three_mesh_refinement` compares three already executed
and sealed cycles with strictly nested coarse, intermediate and fine physical
boundaries. A common schedule, ordered full support, initial nodal channels,
captured conductance, normalized metric, execution metadata, callback metadata,
incoming history and REMESH configuration are required. The observer records
exact represented EPI at pre-schedule, physical-boundary, pre-REMESH and
post-REMESH checkpoints and exact pairwise `L_inf` errors on persistent node
identifiers. ZHIR rows use executor-linked decisions whenever the evidence is
present; optional `zhir_xi` only validates the executed threshold. Modal rows
are available only for a common captured generator. Detached cycles omit the
complete pre-schedule graph namespace, callback closure and RNG state, node
metadata and sub-EPI state, so the observer withholds both complete-reference
and mesh-only causal claims. This completes a finite S5/S16 observation. Neither
its integrity nor decreasing measured errors proves solver order, mesh
convergence, Lyapunov decrease or a combined schedule/REMESH gain.

The pure `certify_reversible_single_eigenmode_euler_reference` kernel now
supplies the mathematical reference independently of any operator. On fixed
connected symmetric rational conductance with positive capacity, it verifies
one exact centered eigenmode in `H=diag(d_i/nu_i)`, its exponential solution,
pressure-refreshed Euler products, `L_inf` and `H`-error-energy bounds, strict
proper-subdivision improvement, and conditional exact-real convergence as
`h_max -> 0`. The
[`complete proof`](TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem)
also records that `mu*T <= 4096` caps only the rational enclosure's
integer-power exponent, not arbitrary rational bit size. The certificate has
no executor or glyph provenance.

`observe_p2_event_remesh_reference_family` closes one compatible finite
runtime-linked specialization: an event-free effective two-node path with a
nonuniform initial mode,
fixed positive conductance, homogeneous positive capacity, pure-EPI pressure
refreshed at every boundary, exact represented
Euler segments satisfying `0 < lambda*h < 1`, and two proper positive
subdivisions. Unit REMESH delays, the exact initial field as delayed data and
hard clipping on one common scalar interval yield rational finite error bounds,
strict subdivision improvement, exact ideal scaling by
`beta=(1-alpha)^2`, and a runtime bound with the signed
rounding-plus-clipping residual norm. It does not establish solver order,
generic or binary64 asymptotic convergence, arbitrary glyphs or mixed channels,
soft clipping, changing support or metric, repetition or future behavior. The
continuous/Euler equations and proof are centralized in
[`TNFR_DIFFUSION_STABILITY_THEOREM.md`](TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem);
the REMESH specialization is in
[`REMESH_INFINITY_DERIVATION.md`](REMESH_INFINITY_DERIVATION.md#28-effective-p2-three-mesh-reference-family);
the implementation and exact typed interface are
[`event_remesh_reference.py`](../src/tnfr/physics/event_remesh_reference.py) and
[`event_remesh_reference.pyi`](../src/tnfr/physics/event_remesh_reference.pyi),
verified by
[`test_event_remesh_reference.py`](../tests/physics/test_event_remesh_reference.py)
and
[`test_event_remesh_reference_example.py`](../tests/physics/test_event_remesh_reference_example.py),
with the public `2/4/8` witness in
[`166_event_remesh_reference_family.py`](../examples/02_physics_regimes/166_event_remesh_reference_family.py).
The pure nonregular-`P3` witness is
[`167_reversible_eigenmode_reference.py`](../examples/02_physics_regimes/167_reversible_eigenmode_reference.py).

The separate uniform exact companion recurrence now has an augmented-history
result. `certify_uniform_remesh_history_stability` derives a row-stochastic
history matrix and its invariant temporal measure. For any one fixed positive
diagonal spatial metric, `observe_uniform_remesh_history_transition` verifies
an exact Jensen identity showing that the stationary-weighted disagreement
energy is nonincreasing. `0 < alpha < 1` gives primitive temporal mixing and
pointwise convergence to a preserved history barycenter; `alpha=1` gives a
pure-delay permutation with conserved energy and possible periodic histories.
The result is outside the named REMESH stage and does not identify the clipped
binary64 runtime, imply spatial consensus or zero pressure, or combine its map
with a schedule gain. See
[`REMESH_INFINITY_DERIVATION.md`](REMESH_INFINITY_DERIVATION.md#24-exact-finite-companion-history-stability).

The conditional exact composition theorem
`certify_uniform_remesh_schedule_policy_stability` indexes histories after each
schedule. For a fixed companion and metric, any sequence of exact schedules
that preserves spatial consensus and has one common disagreement gain bound
`q` is dominated by `B_q=diag(q,1,...,1)P`. Every prefix has gain at most one;
over the sufficient universal horizon `L=active_max_delay+1`, the gain is at
most `q`. Thus `q<1` gives uniform normalized block margin `1-q` and repeated
geometric spatial-disagreement decay, including for pure-delay `alpha=1`.
This result does not control spatially uniform temporal means, verify the
schedule hypotheses or promote the binary64 runtime and its defects. See
[`REMESH_INFINITY_DERIVATION.md`](REMESH_INFINITY_DERIVATION.md#211-uniform-exact-remeshschedule-policy-stability).

`certify_uniform_remesh_schedule_relative_defect_stability` adds the exact
robust budget needed between the binary64 REMESH realization and that policy
theorem. For Jensen input energy `J`, ideal head `y` and bounded head `z`, the
declared signed defect `E_H(z)-E_H(y)<=eta*J` changes the common head gain to
`q_eff=q*(1+eta)`. The certificate rejects `q_eff>1`; equality gives a zero-
margin nonincrease result, while strict inequality gives normalized block
margin `1-q_eff` and repeated exact-model spatial-disagreement decay. It does
not identify a runtime class or derive `eta` from rounding/clipping semantics.

`observe_executed_event_remesh_relative_defect_block` performs the missing
finite check on one contiguous part of an intact causal execution that retained
its compatible runtime telescope. It verifies
each exact defect and schedule gain, the componentwise `D_qeff P` history-
energy envelope, continuity and the complete-block endpoint factor. Its
runtime claims stop at the selected block: a forward-invariant class,
repetition, future binary64 behavior and full TNFR stability remain open.
See
[`REMESH_INFINITY_DERIVATION.md`](REMESH_INFINITY_DERIVATION.md#212-relative-signed-defect-envelope-and-finite-causal-verification).

The exact represented-number boundary narrows that open statement. The
pairwise observer replays the shared REMESH kernel and computes the exact local
relative defect. A normal-valued `alpha=1/2` witness requires
`eta=2^210-1/4`, which refutes a useful uniform bound over a general bounded
hard-clipped box for the `q=9/16` policy. The separate `alpha=1` certificate
fixes support, metric, delays and interval; sufficient represented history then
has `eta=0` and is forward invariant under REMESH alone.

On an abstract P2 support, the configured `EN_mix=0.5` Reception EPI kernel
closes the first restricted global numeric EPI-kernel composition. Mutual
singleton neighbor sets and one immutable Jacobi snapshot give two reversed
sums of the same half-scaled operands, hence numeric consensus for every finite
represented pair in the source interval.
The centered-energy gain is globally `q=0`; with `alpha=1`, active-history
spatial disagreement vanishes after `tau_global+1` restricted kernel cycles.
This does not certify a graph or complete EN stage. Grammar replacement,
preconditions, semantic writes, callbacks, event ownership and transactions
remain outside the theorem. See
[`REMESH_INFINITY_DERIVATION.md`](REMESH_INFINITY_DERIVATION.md#214-a-global-binary64-p_2-half-reception-kernel-family).

`certify_executed_p2_half_reception_stage` now binds that numeric theorem to
one graph-owned event. It requires an intact event result and deeply
revalidated neighbor-stage certificate, then checks the ordered P2 targets,
mutual singleton runtime neighbors, exact half mix, matching hard interval and
diffusion metric, preserved capacity/conductance and bit-exact captured
endpoint replay. This establishes one grammar-admitted two-phase EN EPI stage
with observed `q=0` and finite schedule atomicity. It does not bind REMESH to
that graph, certify all auxiliary Reception state or prove repeated execution.

`certify_executed_p2_half_reception_remesh_sequence` closes that same-graph
boundary for one completed finite invocation. It takes the abstract kernel
certificate and one intact `ExecutedEventRemeshCycleSequence`. In every cycle it
requires an executor-owned P2 half-Reception stage whose endpoints span the
complete schedule EPI transition, and a same-cycle hard-clipped `alpha=1`
REMESH bridge with the common runtime alpha source and exact `eta=0` global-delay
copy. Support, metric, interval, delays and history capacity remain fixed.
For `N >= L = tau_global+1` observed cycles it certifies zero spatial energy on
the active history suffix of length `L` and on the observed post-horizon REMESH
endpoints. Older retained history rows are outside the claim. The certificate
does not cover Reception auxiliary writes, the current live graph after
observation, future or unobserved repetition, solver accuracy or full TNFR
stability. See
[`runtime_p2_reception_remesh_sequence.py`](../src/tnfr/physics/runtime_p2_reception_remesh_sequence.py),
[`test_runtime_p2_reception_remesh_sequence.py`](../tests/physics/test_runtime_p2_reception_remesh_sequence.py)
and example
[`176_runtime_p2_reception_remesh_sequence.py`](../examples/02_physics_regimes/176_runtime_p2_reception_remesh_sequence.py).

`execute_p2_half_reception_remesh_policy_invocation` adds the reusable
operational boundary. Each call snapshots the graph before materializing caller
inputs, preflights the current P2 support, metric, exact half-Reception factor,
`alpha=1` hard-clipped REMESH controls, active history, clock and canonical
zero-flow `EN -> IL -> REMESH` specifications, rederives U1a admission from the
live pair at every cycle start, then executes and constructs the finite
certificate before the outer transaction can commit. A zero cycle-start pair or
any failed preflight, execution or post-certification restores graph-owned state.
This is orchestration
of existing canonical operators, not a fourteenth operator or a new grammar
rule. Every successful call certifies only its completed finite trace; later
calls are checked independently, and auxiliary Reception-state stability remains
outside the result. See
[`runtime_p2_reception_remesh_policy.py`](../src/tnfr/physics/runtime_p2_reception_remesh_policy.py),
[`test_runtime_p2_reception_remesh_policy.py`](../tests/physics/test_runtime_p2_reception_remesh_policy.py)
and example
[`177_runtime_p2_reception_remesh_policy.py`](../examples/02_physics_regimes/177_runtime_p2_reception_remesh_policy.py).

`observe_runtime_remesh_history_bridge` identifies one applied,
executor-sealed REMESH result with that companion while retaining exact signed
binary64 rounding and clipping residuals. It lifts the runtime head into the
companion history and separates those defects from the ideal Jensen
dissipation. Hard clipping onto one common scalar interval is
disagreement-nonexpansive in the fixed positive diagonal metric; soft clipping
and rounding can block unconditional decrease.
The runtime does not immediately append the post-REMESH head, so this isolated
bridge proves neither live history advance nor repetition.

`observe_remesh_schedule_history_transition` adds one supplied scheduled head
to the exact balance. It keeps the raw, clipping and schedule energy defects,
verifies the declared schedule gain against the supplied heads, and publishes
a sufficient lower bound plus its exact gain slack. The pure observation has no
runtime provenance. `observe_runtime_remesh_schedule_sequence` supplies the
stricter adjacent-cycle binding: in one common normalized metric and fixed
REMESH configuration it requires the next represented schedule to start at the
bounded REMESH head, finish at the next pre-REMESH head, and become the next
recorded history head. Compatible boundary balances telescope additively over
the finite caller-ordered sequence. Shared causal graph provenance, cross-call
atomicity, a global executable gain, repeated stability and future behavior
remain open.

The shared pointwise stage executor can opt into a conditional certificate for
AL/SHA/VAL/NUL/ZHIR/NAV. Its successful `NetworkStageResult` carries evidence
computed from the same detached snapshot and frozen proposals used by the
commit. The three levels separate exact realization, pre-flow affine gain and
an aligned pre/post diffusion metric. Requests reject unsupported, empty or
grammar-replaced stages before live writes. By itself the result supplies no
IL, mixed-word, pressure-refresh or repeated-runtime theorem; the event runtime
can compose it with EN/RA and flow evidence only after exact endpoint and
common-metric gates pass.

---

## 3. The Operator Taxonomy

The 13 operators partition into functional classes defined by their effect on the nodal equation:

| Class | Operators | Declared structural role | Grammar Roles |
|-------|-----------|--------------------------|---------------|
| **Generators** | AL, NAV, REMESH | Create or activate EPI | U1a |
| **Integrator** | EN | Integrates external input | — |
| **Stabilizers** | IL, THOL | Direct pressure reduction (IL) or stabilizing reorganization (THOL) | U2 |
| **Destabilizers** | OZ, ZHIR, VAL | Incur pressure, phase, or capacity debt | U2 |
| **Coupling** | UM, RA | Phase synchronization | U3 |
| **Transformers** | ZHIR, THOL | Bifurcation-driven change | U4a, U4b |
| **Closure** | SHA, NAV, REMESH, OZ | Terminate sequences | U1b |
| **Simplifier** | NUL | Reduces dimensionality | — |

Some operators appear in multiple classes. THOL is simultaneously a stabilizer (U2) and a transformer (U4b). NAV and REMESH serve as both generators (U1a) and closures (U1b). OZ is both a destabilizer (U2) and closure (U1b). This multiplicity reflects the richness of their physics.

---

## 4. Generators

Generators declare creation or latent-state activation. Grammar rule U1a requires
that a standalone sequence beginning from $\text{EPI}=0$ start with a registered
generator.

**Scope**: The nodal derivative $\nu_f\Delta\text{NFR}$ is defined at
$\text{EPI}=0$ whenever its factors are finite. U1a is an operator-history
contract that makes the origin or reactivation of form explicit; it is not a
repair for a mathematical singularity.

### 4.1 Emission (AL)

**Physics**: Foundational activation of nodal resonance. Creates EPI from vacuum via resonant emission.

**Primary-channel transformation**:

$$
\text{EPI}' = \operatorname{clip}(\text{EPI} + b)
$$

where $b>0$ is the configured `AL_boost`. The direct glyph handler does not
also write $\nu_f$, phase, or $\Delta\text{NFR}$.

**Activation threshold**: $\text{EPI} < \text{EPI}_{\text{threshold}}$ where the
default $\text{EPI}_{\text{threshold}}=0.8$ is configurable through
`AL_MAX_EPI_FOR_EMISSION` (an operational gate; the AL contract fixes the
channel and sign, not this threshold).

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Emission amplitude $b$ | configured positive gain | Operational parameter; channel and direction are contractual |
| Activation threshold | $0.8$ by default | Configurable operational gate |

**Properties**:
- **Irreversible**: Sets an immutable activation flag. Re-emission increments an activation counter but preserves the original timestamp.
- **Genealogical**: Maintains structural lineage tracking (origin timestamp, parent references, derived node list).
- **Latency-aware**: Detects and clears silence (SHA) latency state on reactivation.
- **All-target time basis**: One accepted AL stage assigns one common timestamp
  to all newly emitted targets; repeated targets retain their original origin.

**Grammar**: Generator (U1a).

**Contract**:
- Pre: $\text{EPI} < 0.8$ and $\nu_f$ at or above the configured basal
  threshold when strict preconditions are enabled.
- Post: EPI does not decrease; $\nu_f$, phase and $\Delta\text{NFR}$ remain
  unchanged; the activation flag is set.

### 4.2 Transition (NAV)

**Physics**: Controlled regime shift. Navigates between attractor states (dormant → active → resonant) with regime-specific parameter adjustment.

**Transformation** (regime-dependent):

| Regime | $\nu_f$ change | $\theta$ shift | $\Delta\text{NFR}$ reduction |
|--------|---------------|----------------|--------------------------|
| Latent → Active | +20% | $+0.1$ | $-30\%$ |
| Active → Active | configurable | $+0.2$ | $-20\%$ |
| Resonant → Active | $-5\%$ | $+0.15$ | $-10\%$ |

**Regime detection**:

$$
\text{regime} = \begin{cases}
\text{latent} & \text{if } \nu_f < 0.05 \text{ or latent flag set} \\
\text{resonant} & \text{if } \text{EPI} > 0.5 \text{ and } \nu_f > 0.8 \\
\text{active} & \text{otherwise}
\end{cases}
$$

**Properties**:
- **Latency recovery**: When transitioning from latent state, verifies EPI drift against preserved snapshot (tolerance: 1% for established nodes, $0.330$ for initial nodes).
- **Regime traceability**: Records origin regime, before/after state, and phase
  shift in telemetry. Pressure telemetry distinguishes the true pre-handler
  value, the low-level handler result, and the final regime-retained value.
- **All-target stage**: One immutable proposal binds each target's `nu_f`, phase,
  `DeltaNFR`, latency state and optional jitter draw/RNG progress before any
  write. A missing graph seed is resolved inside the outer transaction, copied
  into the detached snapshot, restored on rejection and retained on success.
  Stable per-node offsets and draw counts make the committed RNG progress
  independent of target iteration order.
- **Shared latency clock**: Every target computes silence duration from one
  stage instant. Ordered warnings, histories, transition events, metrics and
  monitor callbacks still follow requested target order. Cache state, the
  opaque pressure-refresh result and relabeling equivariance remain outside the
  structural target-order claim.

**Grammar**: Generator (U1a), Closure (U1b).

**Contract**:
- Pre: Valid regime state detectable.
- Post: Smooth transition without coherence collapse; latency attributes cleared if applicable.

### 4.3 Recursivity (REMESH)

**Physics**: Propagates fractal pattern echoes across nested EPIs. Tracks
multi-scale identity by linking current structure to prior states; whether a
particular trajectory preserves identity is checked from the resulting state.

**Transformation**:

$$
\text{EPI}_{\rm raw}(t) = (1-\alpha)^2\text{EPI}(t)
 + \alpha(1-\alpha)\text{EPI}(t-\tau_l)
 + \alpha\text{EPI}(t-\tau_g),
\qquad
\text{EPI}_{\rm new}=\operatorname{clip}(\text{EPI}_{\rm raw})
$$

where the default is alpha = 0.5, but graph configuration can override it.
The runtime validates 0 < alpha <= 1 rather than clamping, so the three raw
coefficients are a convex partition and sum to one. Both delays must be strict
positive integers; neither booleans nor fractional values are coerced.

The operation is guarded by max(tau_l, tau_g) + 1 stored snapshots. Before that
point it returns an immutable insufficient-history no-op. Empty live support is
a separate side-effect-free no-op. Once the guard passes, the selected
local-lag and global-lag entries must each be a node-to-EPI mapping
whose support equals the current graph support. Both are temporal per-node
snapshots; the global lag is not a spatial network mean. Hard or soft structural
clipping acts only after the raw recurrence and can make the committed map
nonlinear.

**Properties**:
- **Depth parameter**: Recursion depth $\geq 1$ (validated at construction; raises error if $< 1$).
- **U5 assessment boundary**: Declared depth records recursive structure but
  does not prove a parent/child coherence target. For a concrete hierarchy,
  `assess_u5_parent_child_coherence(..., alpha=...)` evaluates
  $C_{\rm parent}\geq\alpha\sum C_{\rm child}$ from canonical per-node $C$;
  $alpha$ is explicit and has no graph-independent default.
- **Constant-history fixed point**: identical current, local-delay and
  global-delay EPI snapshots are preserved before clipping.
- **Energy scope**: for one positive diagonal metric h, let V_h denote
  weighted disagreement from the h-weighted mean. Convexity gives

  $$
  V_h(x_{\rm raw})\leq
  \beta V_h(x_0)+\gamma V_h(x_l)+\delta V_h(x_g).
  $$

  This bound uses all three inputs. It is not a gain bound relative to the
  current state alone when delayed histories are free. With the two delayed
  vectors fixed, write b = gamma x_l + delta x_g. The raw map of the current
  vector preserves the consensus subspace, and has disagreement gain at most
  beta squared, when b is uniform. Otherwise a consensus current vector can be
  sent to positive disagreement, so no finite global multiplicative gain
  exists. Weighted-mean preservation is reported separately from disagreement.
  Clipping intervention is reported separately from the raw recurrence. None
  of these one-step facts establishes repeated-map stability, pressure closure,
  a structural-charge conservation law or U2 convergence.

- **Execution boundary**: plan_network_remesh returns the immutable proposal;
  apply_network_remesh commits it and returns an immutable result. Strict
  validation precedes writes. A commit, metadata, history or propagated
  callback failure restores graph-owned state, topology, caches and capturable
  callback state. External effects already emitted by callbacks are outside
  this rollback boundary. Exact evidence values that exceed the finite
  binary64 diagnostic range are rejected explicitly before commit. The
  function reads the EPI history but does not append or shift it.

**Grammar**: Generator (U1a), Closure (U1b).

**Contract**:
- Pre: Parent EPI properly formed; depth $\geq 1$.
- Post: Nested structure maintained; parent identity preserved.

**Fixed-delay Cesàro surrogate (historical N15 programme)**:

For a finite cyclic history window, remove clipping, fix
$(\tau_l,\tau_g,\alpha)$ with $0<\alpha<1$, and let $S$ be the unitary cyclic
shift. The auxiliary filter

$$F=\beta I+\gamma S^{\tau_l}+\delta S^{\tau_g}$$

is a normal contraction. Its Cesàro averages converge to the orthogonal
projection onto $\ker(I-F)$. If the window length is compatible with the
delays, those fixed modes satisfy both $z^{\tau_l}=1$ and $z^{\tau_g}=1$;
their periods therefore divide $\gcd(\tau_l,\tau_g)$; they are not
determined by the historically stated $\operatorname{lcm}(\tau_l,\tau_g)$.

This finite cyclic filter is not the runtime history-update map. In particular,
it does not prove a literal $\tau_g\to\infty$ limit for
`apply_network_remesh`, whose required history, selected snapshot and clipping
change with $\tau_g$. Orthogonality gives contraction only in the surrogate's
declared history norm; it does not conserve the TNFR structural charge, make
the structural candidate energy monotone, or provide a universal convergence
rate.

The projection can be computed without registering another engine operator.
That is a software-reuse observation, not a catalog-completeness theorem. A
proof that the 13 operators exhaust admissible TNFR transformations remains an
open inverse problem requiring an independently defined transformation space.

**Full corrected analysis**:
[REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) §§1–23. It
retains the historical N15 milestones and commit anchors (`a1f298fd`,
`badac156`, `48b0574a`) while separating them from the current result.

---

## 5. Integrator

### 5.1 Reception (EN)

**Physics**: Captures and integrates incoming neighbour resonance from the
network environment. EN does not write $\Delta\text{NFR}$; a later pressure
refresh is a separate realization boundary.

On directed support, an arc $j\to i$ makes $j$ an incoming EN input for receiver
$i$: the numeric snapshot reads predecessors and source discovery follows the
same source-to-receiver path orientation. The current affine realization
certificate remains restricted to undirected support.

**Runtime transformation**: let `r(EPI)` denote the centralized real-scalar
glyph reader. It is defined for raw finite real values and uniform-real BEPI
embeddings, retaining their signed value. Genuinely nonuniform or complex BEPI
values have a maximum-component magnitude for generic read-only diagnostics,
but Reception rejects them before proposing a blend. Let
`B_i=ensure_bepi(EPI_i)` be the accepted target operand. Reception first
forms the unweighted arithmetic mean of the runtime neighbours,

$$
\bar r_i = \frac{1}{|N_i|}\sum_{j\in N_i}r(\mathrm{EPI}_j),
$$

then evaluates and structurally clips

$$
x_i' = \operatorname{clip}\!\left(\operatorname{float}
\left((1-m)B_i+m\bar r_i\right)\right).
$$

Edge `weight` does not enter this neighbour average. For every uniform-real
scalar embedding, including negative EPI, `float(B_i)=x_i`; the runtime and
pure-EPI scalar readings therefore share the same signed state. With
`0<=m<=1`, fixed support and values inside the declared EPI interval, the ideal
real update is affine and its hard clip is inactive by convexity:

$$
\text{EPI}' = (1 - m) \cdot \text{EPI} + m \cdot \text{EPI}_{\text{in}}
$$

where $m$ is the configured reception mixing fraction (the standard graph
configuration uses the selected $1/(\pi+1)$ policy value).

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Mixing fraction $m$ | configured; standard policy $1/(\pi+1)$ | Operational interpolation weight |
| Convexity term | $m(1-m)$ | Algebraic mixing diagnostic, not a full-energy contraction theorem |

**Properties**:
- **Source detection**: Detects emission sources within configurable distance (default: 2 hops) via `detect_emission_sources()`.
- **Runtime/transport separation**: Reception's unweighted neighbour mean can
  differ from the conductance-weighted mean defining pure-EPI diffusion.
- **Affine certificate boundary**: `certify_reception_epi_realization()` keeps
  the ideal real map, represented binary64 coefficient matrix and actual
  two-stage runtime value separate. It promotes the represented map only when
  the declared scalar chart and clipping conditions pass and its consensus-row
  identity is exact. Snapshot agreement with the runtime is diagnostic, not a
  global floating-point linearity proof.
- **Pressure refresh**: If a nontrivial local update changes the target by
  `delta`, retaining the pre-update pure-EPI pressure leaves the exact defect
  `delta L_rw e_i`. On connected positive conductance this is nonzero, so the
  pressure must be recomputed before subsequent evolution is treated as the
  pure-EPI channel.
- **Immediate coherence contract**: The EN jump writes EPI, semantic kind and
  optional source metadata while leaving stored $\Delta\text{NFR}$ and
  $d\mathrm{EPI}$ unchanged. Because canonical $C(t)$ reads only those two
  channels, its operator-local value is unchanged at that boundary. This says
  nothing about $C(t)$ after pressure refresh or along a later trajectory.
- **Source telemetry**: Records detected sources in metadata for analysis.

**Grammar**: Integrator (no active destabilizer/stabilizer role).

**Contract**:
- Pre: Target EPI and signed pressure lie below their configured Reception
  admission ceilings. Isolation and an empty detected-source set are admitted
  rather than treated as hard failures; graph-backed execution with source
  tracking enabled emits one advisory for the latter.
- Post: Neighbour EPI is integrated; stored pressure and change rate are
  unchanged, hence immediate operator-local $C(t)$ is unchanged.

---

## 6. Stabilizers

Stabilizers are assigned negative-feedback roles intended to reduce U2 debt.
Grammar rule U2 requires that every destabilizer ({OZ, ZHIR, VAL}) be
compensated by a stabilizer ({IL, THOL}); this finite-word condition does not by
itself prove that $\int \nu_f\Delta\mathrm{NFR}\,dt$ is bounded for every runtime
realization.

### 6.1 Coherence (IL)

**Physics**: Stabilizes structural form through direct negative feedback on
$|\Delta\text{NFR}|$. Bounded evolution still requires a specified trajectory,
gains, timing and pressure law.

**Transformation**:

$$
\Delta\text{NFR}' = \Delta\text{NFR} \cdot (1 - \rho)
$$

where the default retention is
$1-\rho=\pi/(\pi+1)\approx0.7585$, hence
$\rho=1/(\pi+1)\approx0.2415$. The gain remains configurable.

**Phase locking** (optional):

$$
\theta' = \theta + \lambda \cdot \text{wrap}\!\left(\bar{\theta}_{\mathcal{N}} - \theta\right)
$$

where $\lambda \approx 0.3$ is the phase locking coefficient and $\bar{\theta}_{\mathcal{N}}$ is the circular mean of neighbor phases.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| ΔNFR retention factor $f$ | $\pi/(\pi+1)\approx0.7585$ | Default `IL_dnfr_factor`; configurable gain |
| Pressure-square reduction | $1-f^2\approx0.425$ | Isolated $\Delta\mathrm{NFR}^2$ term at the default; not a full-energy bound |
| Phase locking $\lambda$ | $\approx 0.3$ | Configurable coupling strength |

**Properties**:
- **Monotonic $C(t)$**: Global coherence must not decrease (except within explicit dissonance tests).
- **Phase alignment**: Optional circular averaging drives neighborhood synchronization.
- **Telemetry**: Records $C(t)$ before/after, ΔNFR reduction factors, and phase locking events.

**Grammar**: Stabilizer (U2); Bifurcation Handler (U4a).

**Contract**:
- Pre: Active structure exists.
- Post: $|\Delta\text{NFR}|$ reduced; $C(t)$ non-decreasing.

### 6.2 Self-Organization (THOL)

**Physics**: Autonomous emergence via bifurcation. Creates sub-EPIs when the structural acceleration exceeds the bifurcation threshold, implementing operational fractality.

Its primary contract channel is $\Delta\text{NFR}$ with direction
**reorganize**:

$$
\Delta\text{NFR}'=\Delta\text{NFR}+a\,\frac{\partial^2\text{EPI}}{\partial t^2}.
$$

The instantaneous magnitude can rise or fall with the signed acceleration.
THOL's U2 stabilizer role comes from organizing a bifurcation while preserving
global form; it is not a universal one-step pressure contraction.

**Bifurcation detection**:

$$
\frac{\partial^2 \text{EPI}}{\partial t^2} = \text{EPI}(t) - 2\,\text{EPI}(t-1) + \text{EPI}(t-2) \tag{finite difference}
$$

When $|\partial^2\text{EPI}/\partial t^2| > \tau$ (bifurcation threshold), sub-EPIs are spawned.

**Sub-EPI creation**:

$$
\text{EPI}_{\text{sub}} = \text{EPI}_{\text{parent}} \cdot 0.3 + \text{contribution}_{\text{metabolic}}
$$

where $0.3$ is the operational fractal scaling factor (a free parameter).
The child is a nested coordinate with an explicit parent reference; it is not
an additive mass term. THOL leaves the parent's scalar EPI coordinate and all
neighbour EPI coordinates unchanged. Propagation of form belongs to Resonance
(RA) and must appear explicitly in a grammar-valid word.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Fractal scale | $\approx 0.3$ | Operational fractal nesting (free parameter) |
| U5 hierarchy coefficient $\alpha$ | explicit per assessment | No graph-independent default |
| Sub-$\nu_f$ damping | $0.95$ | Child inherits 95% of parent frequency |
| Bifurcation threshold $\tau$ | configurable (default $0.1$) | From graph configuration |
| Pressure-reorganization gain $a$ | $1/(4\pi)\approx0.0796$ | Default `THOL_accel`; operational gain |

`THOL_MIN_COLLECTIVE_COHERENCE` is a deprecated, inert compatibility
alias for the general fragmentation-risk cut. It is not read by THOL and does
not supply the explicit U5 coefficient.

**Properties**:
- **Autopoietic**: Creates independent sub-nodes with hierarchy metadata (bifurcation level, hierarchy path, parent reference).
- **Metabolic integration**: When enabled, captures network signals and metabolizes them into sub-EPI values.
- **Amplitude-alignment telemetry**: `compute_subepi_amplitude_alignment`
  returns $1/(1+\operatorname{var}(a_{\rm child}))$ for stored child EPI
  amplitudes (or zero when fewer than two exist). It is neither canonical
  $C(t)$ nor a U5 decision. The historical
  `compute_subepi_collective_coherence` name is only a compatibility alias.
- **Explicit U5 telemetry**: `assess_u5_parent_child_coherence` evaluates
  $C_{\rm parent}\geq\alpha\sum C_{\rm child}$ only when the hierarchy and
  $\alpha$ are supplied. THOL does not enforce a hidden collective threshold.
- **Depth-limited**: Maximum nesting depth prevents unbounded recursion
  (default: 5 levels). At the limit THOL still applies its signed pressure
  reorganization but records that no child was created.
- **Channel-pure**: The parent and its neighbours keep their EPI coordinates;
  any subsequent form propagation is an explicit RA stage.

**Grammar**: Stabilizer (U2); Bifurcation Handler (U4a); Transformer (U4b).

**Contract**:
- Pre: Sufficient EPI history ($\geq 3$ points); $\nu_f > 0$; elevated $\Delta\text{NFR}$.
- Post: Sub-EPIs spawned (if bifurcation); parent identity and hierarchy metadata preserved.

---

## 7. Destabilizers

Destabilizers increase pressure or capacity stress under their contracts.
Grammar rule U2 requires compensation by a stabilizer as a syntactic safety
policy. Integral convergence additionally needs a declared pressure law,
operator gains, timing, and state-space bounds.

### 7.1 Dissonance (OZ)

**Physics**: Injects controlled instability by amplifying structural pressure. Probes bifurcation readiness by elevating $|\Delta\text{NFR}|$.

**Transformation**:

$$
\Delta\text{NFR}' = f \cdot \Delta\text{NFR}
$$

where the default amplification is
$f=(\pi+1)/\pi\approx1.3183$, the reciprocal of the default IL retention.
It remains an operational gain; the contract fixes only $f>1$.

**Bifurcation trigger**: When $\partial^2\text{EPI}/\partial t^2 > \tau$, the system enters a bifurcation-active state requiring a handler (IL or THOL per U4a).

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Amplification factor $f$ | $(\pi+1)/\pi\approx1.3183$ | Default `OZ_dnfr_factor`; configurable gain |
| Pressure-square increase | $f^2-1\approx0.738$ | Isolated $\Delta\mathrm{NFR}^2$ term at the default; not a full-energy bound |

**Properties**:
- **Network propagation**: Optional cascading to neighbors via phase-weighted, uniform, or frequency-weighted modes.
- **Bifurcation detection**: Monitors $\partial^2\text{EPI}/\partial t^2$ against threshold $\tau$.
- **Telemetry**: Records propagation events, affected nodes, and bifurcation flags.

**Grammar**: Destabilizer (U2); Bifurcation Trigger (U4a); Closure (U1b).

**Contract**:
- Pre: Sufficient EPI/$\nu_f$; $\Delta\text{NFR}$ below critical.
- Post: $|\Delta\text{NFR}|$ increased; bifurcation flag set if acceleration exceeds $\tau$.

### 7.2 Expansion (VAL)

**Physics**: Raises the $\nu_f$ capacity channel. This increases the response
rate to any nonzero pressure without asserting an instantaneous $\Delta\text{NFR}$
or EPI change.

**Transformation**:

$$
\nu_f' = f_{\text{VAL}}\nu_f, \qquad
f_{\text{VAL}}=\text{VAL\_scale}>1
$$

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Default scale factor $f_{\text{VAL}}$ | $1+1/(4\pi)\approx1.0796$ | Operational capacity step |
| Nominal square-factor change | $f^2-1\approx0.166$ at the default | Algebraic $\nu_f^2$ diagnostic; the structural candidate energy has no explicit $\nu_f$ term |
| Min EPI | $1/(2\pi) \approx 0.159$ | minimum structural base (π-fraction, tunable) |
| Min coherence | $\sin(\pi/3) \approx 0.866$ | 60° harmonic coherence |
| Bifurcation threshold | $1/(\pi + 1) \approx 0.2415$ | Detection threshold |

**Grammar**: Destabilizer (U2).

**Contract**:
- Pre: EPI above minimum; coherence above 0.866; bounded $\Delta\text{NFR}$.
- Post: Dimensionality increased; requires IL/THOL compensation. Avoid VAL$\to$VAL chaining.

---

## 8. Coupling and Propagation

Coupling operators establish and utilize phase-synchronized links between nodes. Grammar rule U3 requires phase compatibility verification: $|\phi_i - \phi_j| \leq \Delta\phi_{\max}$.

### 8.1 Coupling (UM)

**Physics**: Synchronizes phases across neighbors, establishing structural links for resonance exchange.

**Transformation**:

$$
\phi_i' \to \phi_j', \qquad |\phi_i - \phi_j| \leq \Delta\phi_{\max}
$$

**Compatibility threshold**: $\pi/(\pi+1) \approx 0.7585$ (the high-coherence gate, complement of the fragmentation threshold $1/(\pi+1)$).

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Compatibility threshold | $\pi/(\pi+1) \approx 0.7585$ | high-coherence gate |
| Phase push | $1/(\pi + 1) \approx 0.241$ | Same physics as EN mixing |
| $\Delta\text{NFR}$ reduction | $1/(2\pi)\approx0.1592$ | Default phase-alignment pressure-relief gain |

**Properties**:
- **Phase verification mandatory**: Antiphase ($|\phi_i - \phi_j| > \Delta\phi_{\max}$) produces destructive interference; coupling is forbidden.
- **EPI identity preserving**: Form is not modified; only phase alignment changes.
- **$\nu_f$ synchronization**: Optional frequency alignment across coupled nodes.

**Grammar**: Coupling (U3); requires phase verification.

**Contract**:
- Pre: Active EPI and $\nu_f$ above thresholds; $|\phi_i - \phi_j| \leq \Delta\phi_{\max}$; network edges exist.
- Post: Phase spread narrowed; EPI identity preserved; links established.

### 8.2 Resonance (RA)

**Physics**: Propagates coherent patterns through phase-aligned nodes. It may
change the target's scalar EPI through neighbour mixing while preserving the
pattern's signed/kind identity, and it conditionally amplifies local structural
frequency.

Let the active RA neighbourhood be the individually U3-compatible subset

$$
N_i^{\mathrm{U3}}=
\{j\in N_i:|\operatorname{wrap}(\theta_i-\theta_j)|\leq\Delta\phi_{\max}\}.
$$

The runtime rejects an isolate or an empty compatible subset. Its configured
phase limit must satisfy $0\leq\Delta\phi_{\max}\leq\pi/2$; configuration may
tighten but cannot relax the canonical gate. Only this subset contributes to
the EPI mean, circular phase mean, and frequency trigger. With
$r=\texttt{RA_epi_diff}$ and $c=\texttt{RA_phase_coupling}$,

$$
\bar x_i=\frac{1}{|N_i^{\mathrm{U3}}|}\sum_{j\in N_i^{\mathrm{U3}}}x_j,
\qquad
x_i^*=\operatorname{clip}\!\left((1-r)x_i+r\bar x_i\right),
$$

$$
\theta_i^*=\operatorname{wrap}_{[0,2\pi)}
\left(\theta_i+c\,\operatorname{wrap}(\bar\theta_i-\theta_i)\right).
$$

Here $\bar\theta_i$ is the unweighted circular mean of compatible phases. If
the runtime's exact antipodal-pair guard or exactly zero represented resultant
marks that mean undefined, phase remains unchanged; this operational guard is
sufficient for those cancellation classes and is not a universal symbolic
zero-resultant test. When $|\bar x_i|>10^{-9}$, the selected runtime trigger,

$$
\nu_{f,i}^*=(1+a)\nu_{f,i},\qquad
a=\texttt{RA_vf_amplification}.
$$

The $10^{-9}$ trigger is operational binary64 policy, not a derived structural
constant. Before EPI, phase, $\nu_f$, history, or RA metadata changes, the
runtime requires

$$
0\leq r\leq1,\qquad a\geq0,\qquad0\leq c\leq1,
$$

finite scalar or uniform-real BEPI operands, a finite nondecreasing proposed
frequency, and the identity gate below.

**Key constants and domains**:

| Quantity | Default | Runtime domain / status |
|----------|---------|-------------------------|
| EPI mix $r$ | $1/(2\pi)\approx0.1592$ | `[0,1]`; convex propagation |
| Frequency amplification $a$ | $1/(8\pi)\approx0.0398$ | `[0,+infinity)`; nondecreasing capacity |
| Phase coupling $c$ | $1/(4\pi)\approx0.0796$ | `[0,1]`; shortest-arc interpolation |
| U3 phase limit | $\pi/2$ | `[0,pi/2]`; may only be tightened |
| Capacity trigger | $10^{-9}$ | Operational binary64 threshold on $\lvert\bar x_i\rvert$ |
| Resonance detection threshold | $\approx0.198$ | Separate operational detection sensitivity |

**Identity contract**:

- **Scalar change is allowed**: accepted RA replaces the target scalar by the
  neighbour blend; preservation does not mean “EPI without alteration.”
- **Sign identity**: a strict negative-to-positive or positive-to-negative
  proposal is rejected. Exact zero is neutral: arriving at or departing from
  zero does not invert a pre-existing nonzero sign.
- **Kind identity**: an established nonempty `epi_kind` must remain exactly the
  same. `None` and the empty string mean absent kind and may be initialized by
  the existing neighbour/fallback convention.
- **Independent gates**: sign compatibility cannot compensate for a kind
  violation, or conversely. A failure is atomic for nodal state, glyph history,
  and RA telemetry.

**Runtime-to-theorem boundary**:

- EN and RA are the first two catalog bridges. They reuse one neutral
  unweighted-neighbour EPI kernel even when pure-EPI diffusion uses a
  conductance-weighted mean.
- `certify_resonance_epi_realization()` separates four layers: the ideal-real
  convex EPI blend, the affine matrix assembled from represented binary64
  coefficients, the actual two-stage binary64 proposal, and the accepted
  identity-gated runtime snapshot. Only a represented map whose exact
  consensus identity and declared scalar/clipping domain pass enters the affine
  jump theorem.
- A successful local frequency boost generally changes the post-RA metric
  $h_i=d_i/\nu_{f,i}$. The fixed post-RA pure-EPI flow can still be certified;
  arbitrary switching between pre/post generators is withheld unless their
  represented metrics are exactly proportional. A displayed recovery time is
  only an estimate; the Boolean fixed-post-flow recovery verdict comes from the
  exact hybrid composer.
- If accepted RA changes target EPI by $\delta$, retaining the prior pure-EPI
  pressure creates the exact defect $\delta L_{rw}e_i$. Pressure must be
  refreshed before subsequent evolution is interpreted as that isolated
  channel. This does not identify or certify stored multichannel $\Delta NFR$.
- Two-stage rounding, structural clipping, U3/identity gates, and the phase and
  frequency channels prevent a global binary64 runtime-affinity claim.

**Global $C(t)$ telemetry**: Optional before/after measurements report the
realized network coherence response; propagation alone does not fix its sign.

**Grammar**: Propagation (U3); requires phase verification.

**Contract**:
- Pre: Scalar or uniform-real coherent EPI; active edge; at least one
  U3-compatible neighbour; factor, phase-limit, identity, and finite-state gates
  pass.
- Post: Scalar EPI may move within its preserved sign/kind identity class;
  $\nu_f$ is nondecreasing and is amplified when the compatible neighbour field
  crosses the operational trigger; any $C(t)$ change is measured rather than
  assumed.

---

## 9. Transformers

Transformers execute structural bifurcations — qualitative state changes that
require recent destabilizing context and, for ZHIR, a prior coherent base.
Grammar rule U4b uses the configured three-operation recency window; this is
finite-word bookkeeping, not a measured energy threshold or universal
relaxation time.

### 9.1 Mutation (ZHIR)

**Physics**: Controlled phase transformation. The nodal equation supplies the
instantaneous prediction

$$
(\partial_t\mathrm{EPI})_{\mathrm{pred}}
= \nu_f\,\Delta\mathrm{NFR}.
$$

This prediction describes the current capacity-pressure product. Its strict
comparison with $\xi$ is a predictive diagnostic; it is not evidence that a
growth rate has been observed and does not admit ZHIR by itself.

**Observed temporal evidence**: the non-disableable runtime trigger instead
uses the last two finite scalar EPI samples to form the signed secant

$$
\widehat{\partial_t\mathrm{EPI}}_{\mathrm{obs}}
=\frac{\mathrm{EPI}_{k}-\mathrm{EPI}_{k-1}}{t_k-t_{k-1}}.
$$

Timestamped `epi_time_history` is physical-time evidence. It requires finite
samples, a finite strictly positive interval, and a final sample that represents
the current EPI endpoint (exactly under the runtime default; the pure
certificate can declare a finite nonnegative absolute endpoint tolerance).
When this channel is supplied it is authoritative: invalid or stale physical
evidence does not fall back to an untimestamped history.

For compatibility, `epi_history` and then `_epi_history` can provide legacy
evidence. Their two samples are separated by one **operator step**, so the
calculation reduces to `EPI[k] - EPI[k-1]`. Such evidence is explicitly marked
`physical_time_resolved=False` and carries no physical-time or current-endpoint
claim.

The observed comparison is signed and strict:
$\widehat{\partial_t\mathrm{EPI}}_{\mathrm{obs}}>\xi$. Equality, contraction,
sub-threshold growth, missing evidence, invalid samples, a non-increasing
physical timestamp, or a stale physical endpoint reject direct execution
before its phase mutation. The gate also requires finite active capacity
$\nu_f>0$; an explicitly configured `ZHIR_MIN_VF` may tighten that condition.
The default `ZHIR_THRESHOLD_XI = 0.1` is an operational calibration, not a
structural constant; a configured replacement must be finite and nonnegative.

The pure `MutationTriggerCertificate` and the SDK `nodal_state()` read-out keep
the predicted and observed rates and crossings separate. In particular,
`near_bifurcation` is a legacy SDK alias of the **predicted** crossing, while
`mutation_threshold_satisfied` reports only the observed strict threshold.
Unavailable or invalid evidence leaves `observed_crossed=None` instead of
inventing a negative observation. `rate_gap = observed - predicted` is reported
only for valid physical-time evidence, where those rates share a time basis;
it remains unavailable for a legacy operator-step difference.
Neither interface evaluates the prior-IL/recent-destabilizer requirements and
neither certifies U4b execution readiness.

When a dynamic selector proposes ZHIR without valid crossing evidence or active
capacity, it substitutes IL before ordinary grammar enforcement and records the
requested and applied glyphs with the reason in `mutation_abstentions`.
The SDK whole-word runner applies a stronger atomic boundary: before executing
the first operator of a word containing ZHIR, it checks the temporal gate for
every target node. It also rejects timestamped evidence when an earlier
EPI-channel operator in that word would make the endpoint stale before ZHIR;
legacy unit-operator-step evidence retains its compatibility interpretation.
This preflight does not replace grammar validation or U4b. The fluent SDK's
`apply_evidence_gated_mutation()` adds an explicitly experimental policy on top:
when evidence is absent, stale, non-crossing or attached to inactive capacity,
it executes a declared complete word without ZHIR and records the per-node
evidence and abstention reason. It never constructs history; malformed evidence
still raises, and direct sequence execution remains strict.

**Phase transformation**:

$$
\theta'=\operatorname{wrap}_{[0,2\pi)}\!\left(
\theta+s\,\operatorname{sign}(\Delta\text{NFR})\right)
$$

where the default calibrated magnitude is
$s=(1/\pi)(\pi/4)=1/4$ radians unless an explicit fixed shift is supplied.
The runtime preconditions and U4b history determine whether the mutation is
admissible; $|\Delta\text{NFR}|$ does not scale this phase step.

The phase calculation and its branch-specific `_zhir_*` telemetry payload use
one RNG-free immutable kernel. At the all-target boundary, the complete frozen
proposal also binds the temporal threshold evidence, structural acceleration
and U4 context to one stage snapshot. The stage commits phase and acceleration
before merging histories, provenance, metrics and bifurcation events in
requested target order. Its primary structural result is target-order invariant
before the opaque pressure refresh; the ordered auxiliary streams and
relabeling equivariance remain outside that result.

**Bifurcation monitoring**: `compute_d2epi_dt2` estimates structural
acceleration from three EPI-history samples. Timestamped histories use adjacent
secant slopes over their physical intervals; legacy histories use the unit-step
second difference. The result can flag bifurcation potential when it exceeds
$\tau$. This acceleration diagnostic is distinct from both the instantaneous
nodal prediction and the two-sample observed ZHIR gate.

`ZHIR_BIFURCATION_MODE="detection"` is the only supported mode. The legacy
`"variant_creation"` value is rejected before any write. ZHIR therefore remains
a phase-only transformation with bifurcation detection and telemetry; topology
or sub-EPI creation belongs to THOL and must be expressed through that operator.

**Key parameters**:

| Parameter | Default | Runtime role / status |
|-----------|---------|-----------------------|
| Signed observed-growth threshold $\xi$ | `ZHIR_THRESHOLD_XI = 0.1` | Strict operator-admission calibration; operational, not structural |
| Direct activity condition | $\nu_f>0$ | Zero capacity cannot execute the phase transformation |
| Optional direct minimum | absent (`ZHIR_MIN_VF = 0`) | Tightens the activity condition only when configured explicitly |
| Branch-selection threshold | `ZHIR_BIFURCATION_VF_THRESHOLD = 0.5` | Selects whether the bifurcation router proposes ZHIR; it is not a ZHIR operator precondition |
| Default phase shift | $1/4$ rad | Calibrated magnitude; $\Delta\text{NFR}$ supplies direction only |
| Sampled $\lvert\Delta E\rvert$ | $\leq 0.056$ per node in the referenced protocol | Finite observation, not an operator contract |

**Grammar**: Transformer (U4b); Bifurcation Trigger (U4a).

**Contract**:
- Pre: active finite $\nu_f>0$ (and `ZHIR_MIN_VF`, only when explicitly
  configured); valid two-sample temporal evidence satisfying the signed strict
  inequality $\widehat{\partial_t\mathrm{EPI}}_{\mathrm{obs}}>\xi$; prior IL
  for the stable base; and a recent destabilizer inside the configured U4b
  window. Temporal evidence is either fresh timestamped physical evidence or
  legacy unit-operator-step evidence under the compatibility contract. The
  normal grammar route or the strict precondition route enforces the two U4b
  context requirements.
- Post: phase $\theta$ shifted; the structural identity `epi_kind` preserved;
  bifurcation potential flagged if $\partial^2\text{EPI}/\partial t^2>\tau$.
  After successful dispatch, `source_glyph` records `ZHIR` as provenance. That
  provenance field does not define or overwrite `epi_kind`.

---

## 10. Closure and Regime Operators

### 10.1 Silence (SHA)

**Physics**: Freezes structural evolution by suppressing $\nu_f$. With $\nu_f \to 0$, the nodal equation yields $\partial\text{EPI}/\partial t \approx 0$ regardless of $\Delta\text{NFR}$.

**Transformation**:

$$
\nu_f'=f_{\text{SHA}}\nu_f,\qquad
f_{\text{SHA}}=1-1/(4\pi)\approx0.9204
$$

EPI is preserved via latency snapshot.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| $\nu_f$ suppression factor | $1-1/(4\pi)\approx0.9204$ | Operational `SHA_VF_FACTOR` |
| Sampled $\lvert\Delta E\rvert$ | $\leq 0.187$ in the referenced protocol | Finite observation, not an operator contract |

**Properties**:
- **Latency state**: Activates a latent flag with timestamped EPI snapshot.
- **EPI preservation**: Drift tolerance of 1% for established nodes, $0.330$ for initial nodes.
- **Reactivation protocol**: AL or NAV recovery verifies silence duration and EPI drift, then clears latency attributes.
- **All-target time basis**: One accepted SHA stage gives every target the same
  latency-start timestamp while preserving a target-specific EPI snapshot.

**Grammar**: Closure (U1b).

**Contract**:
- Pre: Existing EPI; $\Delta\text{NFR}$ not at critical levels.
- Post: $\nu_f \to 0$; EPI remains invariant; latent flag set with snapshot.

### 10.2 Contraction (NUL)

**Physics**: Densifies and consolidates structural form by reducing dimensionality. Compresses $\nu_f$ while increasing local $\Delta\text{NFR}$ density.

**Transformation**:

$$
\nu_f'=f_{\text{NUL}}\nu_f,\qquad
f_{\text{NUL}}=1-1/(4\pi)\approx0.9204
$$

Local $\Delta\text{NFR}$ density increases due to compression:

$$
\Delta\text{NFR}'=\frac{1}{f_{\text{NUL}}}\Delta\text{NFR}
\approx1.0865\,\Delta\text{NFR}
$$

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Scale factor | $1-1/(4\pi)\approx0.9204$ | Same operational $\nu_f$ step as SHA |
| Densification factor | $1/f_{\text{NUL}}\approx1.0865$ | Reciprocal configured capacity factor |

The immutable scale proposal binds the requested capacity factor, its exact
binary64 reciprocal, the pre/post pressure and any EPI-boundary adaptation
before commit. Each densification audit event carries its target identifier.
Contraction metrics use their captured pre-operation pressure, or the latest
matching target event for compatibility, and report signed pressure change
separately from magnitude increase. Unchanged zero pressure is therefore not
misreported as densification.

**Grammar**: Simplifier (no active grammar role; supports VAL reversals).

**Contract**:
- Pre: Non-trivial EPI (not $\approx 0$).
- Post: Dimensionality reduced; pressure density increased. Avoid NUL$\to$NUL chaining.

---

## 11. Fragments and Complete Words

Operators compose left-to-right. A named fragment is reusable syntax; a
complete word is a concrete sequence evaluated against its execution context.
Canonical U6 additionally needs before/after $\Phi_s$ snapshots and therefore
cannot be certified from operator labels alone.

### 11.1 Named workflow fragments

| Fragment | Sequence | Structural intent | Missing standalone context |
|----------|----------|-------------------|----------------------------|
| **Bootstrap** | [AL, UM, IL] | Create → Couple → Stabilize | U1b closure; U3 phase values |
| **Stabilize** | [IL, SHA] | Stabilize → Close | U1a generator when starting from null |
| **Explore** | [OZ, ZHIR, IL] | Destabilize → Transform → Stabilize | U1a/U1b glue and prior IL for ZHIR |
| **Propagate** | [RA, UM] | Resonate → Couple | U1a/U1b glue; U3 phase values |

These fragments are not asserted to be standalone valid words. For example,
`[AL, IL, OZ, ZHIR, IL, SHA]` supplies the generator, prior stable base,
recent destabilizer, handler/stabilizer and closure needed by the Explore body.

### 11.2 Complete-word examples

| Name | Sequence | Description |
|------|----------|-------------|
| Bifurcated base | [AL, EN, IL, OZ, ZHIR, IL, SHA] | Exploration with mutation and stabilization |
| Bifurcated collapse | [AL, OZ, NUL, IL, SHA] | Stress testing with contraction recovery |
| Theory system | [AL, NAV, UM, RA, IL, SHA] | Cognitive consolidation via coupling and resonance |
| Full deployment | [AL, UM, RA, IL, OZ, ZHIR, IL, SHA] | Integration pipeline with the prior IL required by ZHIR |
| Minimal stabilizer | [AL, IL, SHA] | Shortest valid bootstrap-stabilize-close |
| Contained crisis | [AL, EN, IL, OZ, SHA] | Crisis containment through intervention |
| Phase lock | [AL, EN, IL, OZ, ZHIR, SHA] | Synchronization through mutation |
| Resonance peak hold | [AL, EN, IL, RA, SHA] | Peak detection and maintenance |

### 11.3 Validation boundary

The sequence validator checks U1 initiation/closure, U2 debt coverage, U3
operator awareness, U4 context, and declared U5 Recursivity depth. Actual U3
phase compatibility is checked by operator preconditions. Canonical U6 is a
separate snapshot observation. Thus a sequence-level pass is not full U1--U6
trajectory certification and does not prove convergence.

---

## 12. Operator roles and energy diagnostics

The structural energy diagnostic
$E = \frac{1}{2}\sum_i [\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 +
J_\phi^2 + J_{\Delta\text{NFR}}^2]$ is nonnegative and can serve as a
Lyapunov candidate for a specified dynamics. It contains no explicit EPI or
$\nu_f$ term, although an operator can still change $E$ indirectly by updating
phase, pressure, topology or coupled state. The grammar U2 role therefore does
not determine the sign of $\Delta E$.

### 12.1 Grammar U2 roles

| Registry role | Operators | Structural intent |
|---------------|-----------|-------------------|
| **Stabiliser** | IL, THOL | Supply negative feedback or reorganize a perturbed state |
| **Destabiliser** | OZ, ZHIR, VAL | Introduce pressure, phase or capacity perturbations that incur U2 debt |
| **Neither U2 class** | AL, EN, RA, REMESH, UM, SHA, NUL, NAV | Their other grammar and operator contracts still apply |

These sets are derived from the declared nodal-channel predicates in
`config.physics_derivation`. They are policy classifications, not measured
energy inequalities.

### 12.2 Nominal operator parameters

| Operator | U2 role | Nominal parameter | What it does not prove |
|----------|---------|-------------------|------------------------|
| IL | Stabiliser | pressure factor $f=\pi/(\pi+1)\approx0.7585$ | A universal $\Delta E\leq0$ theorem |
| THOL | Stabiliser | pressure-reorganization gain $a=1/(4\pi)\approx0.0796$ | A universal contraction rate |
| OZ | Destabiliser | pressure factor $f=(\pi+1)/\pi\approx1.3183$ | A state-independent energy multiplier |
| ZHIR | Destabiliser | default signed phase step $=1/4$ rad | A curvature or energy growth rate |
| VAL | Destabiliser | $\nu_f$ scale $1+1/(4\pi)\approx1.0796$ | An instantaneous energy sign |

**Dual-lever note**: the U2 role is distinct from the dual-lever channel
(§17.1). VAL engages the capacity lever yet is a U2 destabiliser; NAV engages
the pressure lever but is outside the U2 destabiliser set because its declared
trajectory is controlled. Actual energy change must be measured on the
executed state transition.

### 12.3 Grammar U2 stability boundary

U2 requires destabilizers to be compensated by stabilizers, but it does not
prove for every compliant sequence that

$$
\sum_{\text{ops}} \Delta E_{\text{op}} \leq 0
$$

The operator registry supplies local role and nominal parameter metadata rather than a common
Lyapunov proof for all realizations. Such a proof additionally needs a specified
state functional, pressure law, timing and gain bounds. The fixed symmetric pure
EPI channel is one class where this has now been established, including
heterogeneous positive capacities. A declared affine EPI reset can also be
bounded in the same metric exactly when it preserves the consensus subspace;
the resulting rational quotient-gain bound, with a weighted-Frobenius fallback,
composes with elapsed diffusion time.
This generic theorem does not certify any catalog operator until its runtime
action is derived as that affine map on a declared domain.

**Exact restricted result**: see
[TNFR_DIFFUSION_STABILITY_THEOREM.md](TNFR_DIFFUSION_STABILITY_THEOREM.md).

---

## 13. Postcondition Contracts

Every operator has a postcondition contract anchored to the **direct effect on node state** (the nodal dynamics $\partial\text{EPI}/\partial t = \nu_f\cdot\Delta\text{NFR}$, anchored to TNFR.pdf §2.2.1). The canonical contract layer `src/tnfr/operators/operator_contracts.py` is the **single source of truth** — it records each operator's `primary_channel` (one nodal-equation channel: $\text{EPI}$ / $\nu_f$ / $\theta$ / $\Delta\text{NFR}$), `scale` (NODE for twelve operators, NETWORK for the U5 operator REMESH), and `postcondition`. The proactive audit (`audit_operator_contracts`), the reactive integrity monitor (`POSTCONDITIONS`, `src/tnfr/physics/integrity.py`), and the introspection metadata all derive from this spec. The monitor supports three modes: OFF (production), OBSERVE (log violations), ENFORCE (raise exceptions).

| # | Operator | Glyph | Channel | Postcondition |
|---|----------|-------|---------|---------------|
| 1 | Emission | AL | EPI | EPI not decreased; $\nu_f$, phase and $\Delta\text{NFR}$ unchanged |
| 2 | Reception | EN | EPI | Immediate operator-local $C(t)$, $\Delta\mathrm{NFR}$ and $d\mathrm{EPI}$ unchanged |
| 3 | Coherence | IL | $\Delta\text{NFR}$ | $C(t)$ non-decreasing; $\lvert\Delta\text{NFR}\rvert$ reduced |
| 4 | Dissonance | OZ | $\Delta\text{NFR}$ | $\lvert\Delta\text{NFR}\rvert$ not decreased |
| 5 | Coupling | UM | $\theta$ | Phase compatibility $\lvert\phi_i - \phi_j\rvert \le \Delta\phi_{\max}$ |
| 6 | Resonance | RA | EPI | EPI structural identity (sign/kind) preserved |
| 7 | Silence | SHA | $\nu_f$ | EPI preserved over time; $\nu_f$ frozen |
| 8 | Expansion | VAL | $\nu_f$ | $\nu_f$ not decreased (capacity added) |
| 9 | Contraction | NUL | $\nu_f$ | $\nu_f$ not increased (capacity removed) |
| 10 | Self-Organization | THOL | $\Delta\text{NFR}$ | Global form preserved; sub-EPIs created (if bifurcation) |
| 11 | Mutation | ZHIR | $\theta$ | Phase $\theta$ changed when $\Delta\text{EPI}/\Delta t > \xi$ |
| 12 | Transition | NAV | $\Delta\text{NFR}$ | Controlled trajectory; no coherence collapse |
| 13 | Recursivity | REMESH | EPI (network) | Nested structure maintained; parent identity preserved |

The contract catalog is complete as a specification but is not instantaneously
identifiable from its categorical fields. The exact certificate
`contract_identifiability_certificate()` shows that channel, direction, scale
and context leave Silence and Contraction in the same class. Their distinct
semantics require a temporal or quantitative postcondition observation. This
negative inverse result neither removes an operator nor proves completeness of
the catalog over every admissible TNFR transformation.

---

## 14. Operator Constants Reference

Operator gain magnitudes are **free operational parameters**: each operator's
contract fixes its primary channel and qualitative direction or intent, not its
numeric magnitude. The values below are operational calibrations. Only **$\pi$**
is a genuine structural phase scale. The authoritative current values live in
[`src/tnfr/constants/canonical.py`](../src/tnfr/constants/canonical.py) and the
contracts in [`operators/operator_contracts.py`](../src/tnfr/operators/operator_contracts.py).

### 14.1 The structural scale

| Symbol | Name | Value | Role |
|--------|------|-------|------|
| $\pi$ | Pi | $3.141592653589793$ | the one genuine structural scale: bounds the phase sector ($\lvert\nabla\phi\rvert \le \pi$, $\lvert K_\phi\rvert \le \pi$) |

### 14.2 Operator gain magnitudes (operational)

Representative operational parameters (the contract fixes the channel and
qualitative direction or intent, not every magnitude):

| Constant | Value | Used by |
|----------|-------|---------|
| EN / THOL collective-coherence fraction | $1/(\pi + 1) \approx 0.241$ | EN mixing, UM phase push, THOL/VAL threshold (π-derived) |
| SHA / NUL frequency factor | $1-1/(4\pi)\approx0.9204$ | SHA suppression and NUL compression |
| NUL densification factor | $1/(1-1/(4\pi))\approx1.0865$ | Reciprocal configured capacity factor |
| VAL scale factor | $1+1/(4\pi)\approx1.0796$ | VAL capacity expansion |
| AL emission boost | configured positive gain | AL creation; direct-handler fallback is `COUPLING_GENTLE` |
| THOL fractal scale | $0.3$ | Sub-EPI scaling |

The complete, authoritative set lives in `src/tnfr/constants/canonical.py`; among them the only genuine structural scale is the phase scale $\pi$ (the $1/(\pi+1)$ entry above is the one π-derived value).

### 14.3 Constant-Operator-Grammar Traceability

The contract records each operator's **primary channel and qualitative
direction or intent**. The gains above are operational parameters; the channel
and grammar mapping is:

```
OZ   (ΔNFR ↑, amplification)   →  U2 (destabilizer)
IL   (ΔNFR ↓, reduction)       →  U2 (stabilizer), U4a (handler)
EN   (EPI, mixing)             →  integrator
VAL  (νf ↑, expansion)         →  U2 (destabilizer)
SHA  (νf → 0, suppression)     →  U1b (closure)
ZHIR (θ, mutation)             →  U4b (transformer)
THOL (sub-EPI, self-org)       →  U2 (stabilizer), U4b (transformer)
AL   (EPI from vacuum)         →  U1a (generator)
```

**Source**: `src/tnfr/constants/canonical.py` and
`src/tnfr/operators/operator_contracts.py` (the contract source of truth:
channel, direction, scale and postcondition).

---

## 15. Implementation Reference

### 15.1 Source Modules

| Module | Content |
|--------|---------|
| `src/tnfr/operators/definitions.py` | Facade: imports all 13 operator classes |
| `src/tnfr/operators/definitions_base.py` | `Operator` abstract base class with `__call__` workflow |
| `src/tnfr/operators/emission.py` | AL implementation |
| `src/tnfr/operators/al_sha_stage_proposals.py` | Immutable AL/SHA structural and lifecycle proposals |
| `src/tnfr/operators/reception.py` | EN implementation |
| `src/tnfr/operators/_neighbor_epi_kernel.py` | Shared unweighted EN/RA scalar-mean and blend kernel |
| `src/tnfr/operators/coherence.py` | IL implementation |
| `src/tnfr/operators/dissonance.py` | OZ implementation |
| `src/tnfr/operators/coupling.py` | UM implementation |
| `src/tnfr/operators/resonance.py` | RA implementation |
| `src/tnfr/operators/_resonance_identity.py` | Shared RA factor, U3 phase-limit, sign/kind identity, and circular-mean gates |
| `src/tnfr/operators/silence.py` | SHA implementation |
| `src/tnfr/operators/expansion.py` | VAL implementation |
| `src/tnfr/operators/contraction.py` | NUL implementation |
| `src/tnfr/operators/_scale_operator_kernel.py` | Shared immutable VAL/NUL capacity, pressure and EPI-boundary proposal |
| `src/tnfr/operators/self_organization.py` | THOL implementation |
| `src/tnfr/operators/mutation.py` | ZHIR implementation |
| `src/tnfr/operators/_mutation_stage_kernel.py` | Pure immutable ZHIR phase proposal |
| `src/tnfr/operators/transition.py` | NAV implementation |
| `src/tnfr/operators/jitter.py` | Reproducible jitter proposal, progress validation and atomic commit |
| `src/tnfr/operators/recursivity.py` | REMESH implementation |
| `src/tnfr/operators/network_stage.py` | Shared transactional Jacobi and Gauss-Seidel stage executors |
| `src/tnfr/operators/event_timing.py` | Exact finite flow/jump schedules and clock boundaries |
| `src/tnfr/operators/event_runtime.py` | Observed flow/glyph binding and represented EPI-map composition |
| `src/tnfr/operators/event_remesh_runtime.py` | Atomic event-schedule/delayed-REMESH cycle with separate evidence |
| `src/tnfr/operators/event_remesh_sequence.py` | Exact continuity across ordered supplied event/REMESH cycle observations |
| `src/tnfr/operators/event_remesh_causal_runtime.py` / `src/tnfr/operators/event_remesh_causal_runtime.pyi` | One graph-owned finite causal cycle sequence and exact public interface |
| `src/tnfr/physics/event_refinement.py` | Offline and executor-linked event-local ZHIR observations |
| `src/tnfr/physics/event_remesh_refinement.py` | Strict finite coarse/intermediate/fine event/REMESH observations |
| `src/tnfr/physics/event_remesh_reference.py` / `src/tnfr/physics/event_remesh_reference.pyi` | Effective-P2 finite reference-family certificate and exact public interface |
| `src/tnfr/physics/reversible_eigenmode_reference.py` / `src/tnfr/physics/reversible_eigenmode_reference.pyi` | Pure exact-rational reversible single-eigenmode Euler theorem and public interface |
| `src/tnfr/physics/remesh_history_stability.py` | Exact uniform finite companion-history stability certificate |
| `src/tnfr/physics/remesh_schedule_policy_stability.py` / `src/tnfr/physics/remesh_schedule_policy_stability.pyi` | Conditional exact common-`q` REMESH/schedule spatial-disagreement theorem |
| `src/tnfr/physics/remesh_schedule_relative_defect_stability.py` / `src/tnfr/physics/remesh_schedule_relative_defect_stability.pyi` | Conditional exact robust policy theorem with `q_eff=q*(1+eta)` |
| `src/tnfr/physics/binary64_remesh_relative_defect.py` / `src/tnfr/physics/binary64_remesh_relative_defect.pyi` | Pairwise REMESH defect boundary and `alpha=1`, `eta=0` hard-clip class |
| `src/tnfr/physics/binary64_p2_reception_stability.py` / `src/tnfr/physics/binary64_p2_reception_stability.pyi` | Global `q=0` P2 half-Reception EPI-kernel composition |
| `src/tnfr/physics/runtime_p2_reception_stage.py` / `src/tnfr/physics/runtime_p2_reception_stage.pyi` | One executed two-phase P2 EN EPI stage bound to the global `q=0` kernel |
| `src/tnfr/physics/runtime_p2_reception_remesh_sequence.py` / `src/tnfr/physics/runtime_p2_reception_remesh_sequence.pyi` | One completed causal P2 EN/REMESH sequence with observed active-suffix extinction |
| `src/tnfr/physics/runtime_p2_reception_remesh_policy.py` / `src/tnfr/physics/runtime_p2_reception_remesh_policy.pyi` | Per-invocation P2 preflight, causal execution, finite post-certification and outer rollback |
| `src/tnfr/physics/runtime_remesh_history_stability.py` | One-transition runtime REMESH/companion bridge with signed residuals |
| `src/tnfr/physics/remesh_schedule_stability.py` | Exact REMESH-head/schedule-head augmented-energy balance |
| `src/tnfr/physics/runtime_remesh_schedule_stability.py` | Adjacent-cycle runtime/history energy telescope |
| `src/tnfr/physics/runtime_remesh_schedule_block_margin.py` / `src/tnfr/physics/runtime_remesh_schedule_block_margin.pyi` | Exact normalized margin for one finite causal boundary block |
| `src/tnfr/physics/runtime_remesh_schedule_relative_defect.py` / `src/tnfr/physics/runtime_remesh_schedule_relative_defect.pyi` | Finite causal verification of signed defects and the robust energy envelope |
| `src/tnfr/operators/nodal_equation.py` | Nodal equation validation |
| `src/tnfr/operators/canonical_patterns.py` | Canonical sequence definitions |
| `src/tnfr/operators/introspection.py` | `OperatorMeta` metadata registry |
| `src/tnfr/operators/operator_contracts.py` | **Canonical contract layer** (single source of truth: channel × scale × postcondition) |
| `src/tnfr/operators/stage_contracts.py` | All-target schedule, footprint, merge, rollback and invariance contracts |
| `src/tnfr/operators/grammar_canon.py` | Canonical grammar spec (U1–U6 role table, structural typology, glyphic macros) |
| `src/tnfr/operators/grammar.py` | Grammar validation (public API facade) |
| `src/tnfr/operators/grammar_dynamics.py` | Incremental grammar-aware dynamics |
| `src/tnfr/operators/grammar_application.py` | Pre-validated operator application |
| `src/tnfr/physics/integrity.py` | Sampled postcondition audit and runtime monitor |
| `src/tnfr/physics/lyapunov.py` | Configured operator-role energy diagnostics |
| `src/tnfr/physics/reception_realization.py` | Read-only EN runtime-to-affine-flow certificate |
| `src/tnfr/physics/resonance_realization.py` | Read-only RA four-layer realization, identity, post-metric, and pressure audit |
| `src/tnfr/physics/network_stage_stability.py` | Executor-bound all-target EN/RA represented-map certificates |
| `src/tnfr/physics/pointwise_stage_stability.py` | Executor-bound pointwise represented-map certificates |
| `src/tnfr/physics/runtime_flow_stability.py` | Detached observed-flow certificates and proof sealing |
| `src/tnfr/physics/_exact_metric.py` | Shared exact positive-metric normalization and proportionality |
| `src/tnfr/constants/canonical.py` | All derived constants |

### 15.2 Base Operator Workflow

The `Operator.__call__(G, node, **kw)` method implements the canonical execution pipeline:

1. **Precondition validation**: Operator-specific checks via `_validate_preconditions()`.
2. **State capture**: Records $(\text{EPI}, \nu_f, \Delta\text{NFR}, \theta)$ before application.
3. **Integrity snapshot** (pre): `_integrity_monitor.before_operator()`.
4. **Grammar-aware application**: `apply_glyph_with_grammar()` enforces the
   incremental operator-history checks available at execution time. Canonical
   U6 requires a separate before/after field observation.
5. **Integrity evaluation** (post): `_integrity_monitor.after_operator()`
   evaluates the configured postcondition when monitoring is enabled.
6. **Nodal equation validation** (optional): Checks $|\partial\text{EPI}/\partial t_{\text{measured}} - \nu_f \cdot \Delta\text{NFR}| \leq \epsilon$.
7. **Metrics collection**: Operator-specific telemetry via `_collect_metrics()`.

### 15.3 Executable Demonstrations

| Example | Operators demonstrated |
|---------|----------------------|
| [04_operator_sequences.py](../examples/01_foundations/04_operator_sequences.py) | All 13 operators, canonical compositions |
| [10_simplified_sdk_showcase.py](../examples/01_foundations/10_simplified_sdk_showcase.py) | SDK-level operator usage |
| [29_lyapunov_stability_demo.py](../examples/02_physics_regimes/29_lyapunov_stability_demo.py) | Configured per-operator energy-bound diagnostics and their scope |
| [36_grammar_violation_detector.py](../examples/02_physics_regimes/36_grammar_violation_detector.py) | Grammar enforcement across sequences |
| [163_reception_runtime_bridge.py](../examples/02_physics_regimes/163_reception_runtime_bridge.py) | EN ideal-real, represented, runtime-snapshot, and pressure-refresh boundary |
| [164_resonance_runtime_bridge.py](../examples/02_physics_regimes/164_resonance_runtime_bridge.py) | RA U3 filter, identity gate, four realization layers, post-flow certificate, and switching abstention |
| [167_reversible_eigenmode_reference.py](../examples/02_physics_regimes/167_reversible_eigenmode_reference.py) | No glyph execution: pure exact-real references for both nonuniform modes of nonregular `P3` |
| [169_event_remesh_causal_runtime.py](../examples/02_physics_regimes/169_event_remesh_causal_runtime.py) | One finite same-invocation event/REMESH cycle sequence with causal receipts, outer graph atomicity, and an explicit stability boundary |
| [171_remesh_schedule_policy_stability.py](../examples/02_physics_regimes/171_remesh_schedule_policy_stability.py) | Conditional exact common-`q` REMESH/schedule theorem, including strict pure-delay disagreement decay and the zero-margin `q=1` boundary |
| [172_runtime_remesh_relative_defect.py](../examples/02_physics_regimes/172_runtime_remesh_relative_defect.py) | Robust `q_eff=q*(1+eta)` envelope verified on exact-zero and positive-binary64-defect finite causal blocks |
| [173_binary64_remesh_relative_defect.py](../examples/02_physics_regimes/173_binary64_remesh_relative_defect.py) | Exact pairwise REMESH obstruction and the bounded `alpha=1`, `eta=0` REMESH-only class |
| [174_binary64_p2_reception_remesh_stability.py](../examples/02_physics_regimes/174_binary64_p2_reception_remesh_stability.py) | Global `q=0` half-Reception EPI kernel and finite-horizon disagreement extinction in the restricted repeated model |
| [175_runtime_p2_reception_stage.py](../examples/02_physics_regimes/175_runtime_p2_reception_stage.py) | One grammar-admitted graph-owned P2 EN EPI stage bound to the global `q=0` kernel, with REMESH and future repetition withheld |
| [176_runtime_p2_reception_remesh_sequence.py](../examples/02_physics_regimes/176_runtime_p2_reception_remesh_sequence.py) | One completed same-invocation P2 EN/REMESH sequence with an active suffix of length `tau_global+1` and finite observed spatial-disagreement extinction |
| [177_runtime_p2_reception_remesh_policy.py](../examples/02_physics_regimes/177_runtime_p2_reception_remesh_policy.py) | Two successive P2 policy invocations, each independently preflighted, executed and finite-certified inside its own outer transaction |

### 15.4 SDK Entry Points

```python
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)

# Grammar-aware evolution (incremental operator-history checks)
net.evolve_grammar_aware(steps=10)

# Sampled contract audit; inspect evaluated coverage and individual outcomes
report = net.integrity_check()

# One-line self-optimization (auto operator selection)
from tnfr.sdk.fluent import TNFRNetwork
TNFRNetwork(G).focus(node).auto_optimize().execute()
```

---

## 16. Summary

The canonical registry contains 13 operators with executable contracts and the
operator roles used by U1--U5. U6 consumes before/after structural-potential
telemetry rather than adding another operator role. The registry is complete as
the engine's registered specification. Generative completeness and
irreducibility over every admissible TNFR transformation have not been proved.

**Key results**:

1. **Registry coverage**: All 13 declared transformations have implementations,
   contracts and grammar metadata.
2. **Identification boundary**: Contract categories alone merge Silence and
   Contraction; quantitative one-step signatures separate all 13 only on the
   declared deterministic probes.
3. **Grammar-role coverage**: The operator set supplies the roles used by
   U1--U5; U6 is evaluated from $\Phi_s$ snapshots.
4. **Constants**: Operator gain magnitudes in `canonical.py` are operational parameters (only $\pi$ is a genuine structural scale); the engine-configuration tier is calibrated, not derived.
5. **Stability boundary**: Exact Lyapunov decrease is proved for restricted
   pure-EPI diffusion classes; U2 compliance alone is not a global Lyapunov
   theorem for arbitrary operator sequences.
6. **Contract instrumentation**: The integrity subsystem can sample all 13
   registered contracts and can observe or enforce evaluated runtime
   postconditions. Coverage and outcomes must be reported separately.
7. **Named fragments**: Bootstrap, Stabilize, Explore and Propagate encode
   reusable workflow bodies that need grammar context to become complete words.
8. **Four-channel structure**: Every contract has one primary channel among
   EPI, $\nu_f$, phase, and $\Delta\text{NFR}$; only capacity and pressure are
   the explicit multiplicative levers of the nodal rate.
9. **Operator-tetrad coupling**: Finite experiments measure different tetrad
   responses across operators. They do not prove tetrad completeness.
10. **Scoped $\Phi_s$ linearity**: At fixed topology, structural potential is a
    linear aggregation of the $\Delta\text{NFR}$ source. The reported
    correlation coefficient belongs to one finite sweep.

---

## 17. Experimental Operator-Tetrad Synergies

Computational experiments (Examples 37–39, seed 42, $n=20$, Erdos-Renyi
$p=0.25$) report how executed operator requests change selected structural
fields. The response tables are finite-protocol measurements; they do not
derive universal causal fingerprints from the nodal equation.

### 17.1 Centralized Operator-Channel Classification

The contract source of truth partitions operators by the primary state channel
they write or control:

| Primary channel | Operators | Mechanism |
|-------|-----------|-----------|
| EPI (form) | AL, EN, RA, REMESH | Write, integrate, propagate, or echo form |
| $\nu_f$ (capacity) | SHA, VAL, NUL | Freeze, raise, or reduce reorganization capacity |
| $\theta$ (phase) | UM, ZHIR | Synchronize or transform phase |
| $\Delta\mathrm{NFR}$ (pressure) | IL, OZ, THOL, NAV | Stabilize, perturb, organize, or shift structural pressure |

The nodal rate contains the capacity and pressure levers explicitly. EPI and
phase operators affect the future rate through state and coupling updates. U2
and U4 roles are independently derived from contract predicates; they must not
be inferred solely from the primary channel.

**Example**: NAV (Transition) produces the largest single $\Delta\text{NFR}$ change ($d = -0.444$), consistent with its role as a regime-shift operator.

### 17.2 Operator-Tetrad Fingerprint Matrix

One finite protocol measured percentage changes after single requested operator
applications on a random 20-node network. These rows are sampled response
profiles, not unique or universal operator fingerprints:

| Operator | $\Phi_s$ (%) | $\lvert\nabla\varphi\rvert$ (%) | $K_\varphi$ (%) | $\xi_C$ (%) |
|----------|-------------|----------------------|-----------------|-------------|
| IL (Coherence) | +7.8 | $-0.5$ | 0.0 | $-2.1$ |
| OZ (Dissonance) | +7.8 | $-0.5$ | 0.0 | $-2.1$ |
| UM (Coupling) | $-73.7$ | +2.5 | $-8.1$ | +31.4 |
| NAV (Transition) | $-331.9$ | +45.1 | 0.0 | +187.4 |
| SHA (Silence) | 0.0 | 0.0 | 0.0 | 0.0 |

**Key findings**:
1. **UM (Coupling) has the richest tetrad coupling**: it modifies all four fields simultaneously, consistent with its role as a phase-synchronization operator (U3).
2. **NAV (Transition) dominates $\Phi_s$**: its $-332\%$ structural potential change is the largest single-operator perturbation, matching its physics as a regime-shift operator.
3. **SHA (Silence) is tetrad-neutral**: $\nu_f \to 0$ freezes evolution without affecting field state, confirming its closure role (U1b).
4. **IL and OZ produce identical tetrad signatures**: this is analyzed in §17.3.

### 17.3 IL-OZ Tetrad Symmetry

**Observation**: Coherence (IL) and Dissonance (OZ) produce identical energy functional changes ($dE = -0.011$) and identical tetrad field perturbations when applied to the same initial state.

**Interpretation**: Both operate exclusively via the $\Delta\text{NFR}$ lever with the same magnitude $|d(\Delta\text{NFR})| = 0.0096$, but with different physical semantics:
- **IL** reduces $|\Delta\text{NFR}|$ via negative feedback (stabilizer contract).
- **OZ** increases $|\Delta\text{NFR}|$ via positive feedback (destabilizer contract).

The identical sampled response follows from this initial state and the absolute-
value summaries used by the protocol. Repeated IL and OZ requests can separate,
but U2 role labels alone do not determine convergence or divergence.

### 17.4 Structural Potential Linear Response

**Finite sweep**: $\Phi_s$ is linear in the source at fixed topology. The six
sampled $\xi_C$ values are strongly nonlinear and state-dependent; no linear
$\xi_C$ law is claimed.

| $\Delta\text{NFR}_{\text{init}}$ | $d(\Phi_s)$ | $d(\xi_C)$ |
|----------------------------------|-------------|------------|
| 0.01 | $-0.001$ | $-0.16$ |
| 0.05 | $-0.007$ | $-0.85$ |
| 0.10 | $-0.014$ | $-1.91$ |
| 0.30 | $-0.042$ | $-9.83$ |
| 0.50 | $-0.071$ | $-35.1$ |
| 0.80 | $-0.113$ | $-1464$ |

$\Phi_s$ response is linear in this sweep, as expected from its source-aggregation
definition. The reported $\xi_C$ nonlinearity above the sampled
$\Delta\text{NFR}$ range is protocol-specific finite telemetry; it does not
establish a correlation-length divergence or critical transition.

### 17.5 Diagnostic dependency chain

The implementation has a one-way read-out dependency:

$$
\text{Operator} \to \text{updated nodal/graph state} \to
\text{tetrad fields} \to (E,Q)
$$

The fields are deterministic diagnostics of a fully specified state, not
independent variables and not a complete observer. The numerical deltas reported
by example 37 belong to one graph, node, seed and realized operator history;
fallback substitutions are recorded explicitly.

### 17.6 Grammar-Energy Landscape

The example evaluates energy along several finite sequences. Its observed signs
are properties of that protocol. U2 constrains operator-word roles but does not
bound this quadratic energy for every pressure law or realization.

**Finite diagnostic** (seed 42): the nominal cumulative multiplier for one
Bootstrap+Explore+Stabilize sequence is $\lambda = 1.288$, while the observed
net change is $dE=-9.59$. This shows only that this nominal multiplier does not
predict the sign of that sampled transition. It supplies no stronger guarantee
from grammar compliance.

### 17.7 Executable Demonstrations

| Example | Experiment | Key metric |
|---------|------------|------------|
| `examples/02_physics_regimes/37_operator_tetrad_synergy.py` | Sampled response matrix, energy observations, policy checks and balance telemetry | Per-request realized operator and tetrad delta |
| `examples/02_physics_regimes/38_grammar_energy_landscape.py` | Energy trajectory and nominal multiplier comparison | Observed $E(t)$ for the declared sequence |
| `examples/02_physics_regimes/39_nodal_equation_decomposition.py` | Lever classification, causal chain, waveform trajectory, response functions | $\nu_f$ vs $\Delta\text{NFR}$ per operator |

All experiments use seed 42 for reproducibility (Invariant #6).

---

## Cross-References

- Nodal equation derivation: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §2
- Grammar rules U1–U6: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Energy diagnostics and scoped stability results: [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md) §1
- Conservation laws: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)
- Variational formulation: [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md)
- Structural field tetrad: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §3
- Canonical constants: `src/tnfr/constants/canonical.py`
- Operator-tetrad synergy experiment: `examples/02_physics_regimes/37_operator_tetrad_synergy.py`
- Grammar-energy landscape experiment: `examples/02_physics_regimes/38_grammar_energy_landscape.py`
- Nodal equation decomposition experiment: `examples/02_physics_regimes/39_nodal_equation_decomposition.py`
- Glossary: [GLOSSARY.md](GLOSSARY.md)
