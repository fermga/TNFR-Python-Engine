# TNFR Structural Operators

## Complete Specification of the 13 Canonical Operators

**Status**: CANONICAL registry specification; generative completeness over an
independently defined admissible-transformation space remains open
**Version source**: [pyproject.toml](../pyproject.toml)
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
18. [Mechanism and Activation Audit](#operator-mechanism-and-activation-audit)

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
capacity attenuation/expansion, self-organization, phase transformation, regime
transition, and multi-scale recursion. Coverage of this declared catalog does
not prove that it generates every independently admissible TNFR transformation.

### 1.1 Why the registered catalog has 13 members

The registry contains 13 public semantic transformations satisfying:

1. **Nodal equation compatibility**: Every transformation must declare its effect on
   EPI, $\nu_f$, $\Delta\text{NFR}$, phase, or the coupling structure.
2. **Grammar closure**: The set must include generators (U1a), closures (U1b), stabilizers (U2), destabilizers (U2), coupling operators (U3), bifurcation triggers and handlers (U4), and multi-scale operators (U5).
3. **Semantic distinction**: Each member has a named executable contract. The
   [temporal-signature owner](../src/tnfr/physics/temporal_identifiability.py)
   can compare implementations on declared probes; any separation result
   retains those probes and observations. Global algebraic irreducibility
   remains open.

A proof of completeness first requires a transformation space defined without
reference to this catalog. This remains the open catalog-completeness boundary
(historical S10); the [theory catalog](README.md#operators-grammar-and-support-events)
routes related contract and scope questions to their owners.

### 1.2 Conventions

Throughout this document:
- Glyph codes (AL, EN, IL, ...) reference the structural symbols.
- Lowercase English tokens (`emission`, `reception`, ...) are executable API
  identifiers; title-case names are public display/class names.
- Operator gains and thresholds are configured parameters. $\pi$ is the half-turn
  in the chosen radian phase chart, not a derivation of those parameters.
- Grammar roles reference rules U1–U6 from [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md).
- Energy diagnostics and their proof boundaries are discussed in [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md).

---

## 2. Operator Algebra from the Nodal Equation

### 2.1 Structural Triad

Each node $i$ carries the declared structural triad. This representation is
not a proof that the three attributes remain independent under every richer
form representation or specified quotient:

| Attribute | Symbol | Domain | Units |
|-----------|--------|--------|-------|
| Form | $\text{EPI}_i$ | Declared form space; scalar glyphs require a signed real chart or uniform-real BEPI embedding | Declared form unit X |
| Frequency | $\nu_{f,i}$ | Nonnegative reals; individual operators may require strict positivity | Hz_str |
| Phase | $\phi_i$ (or $\theta_i$) | $[0, 2\pi)$ | rad |

The derived quantity $\Delta\text{NFR}_i$ (structural pressure) drives evolution.
Every operator has exactly one registered **primary** channel among
$(\text{EPI}, \nu_f, \phi, \Delta\text{NFR})$; a concrete implementation may
also update declared secondary channels.

### 2.2 Operator as Transformation

An operator is an event map on its declared full input state $z$, which can
include neighbors, graph data and history. Its local channel projection is
$\sigma_i=(\text{EPI}_i,\nu_{f,i},\phi_i,\Delta\text{NFR}_i)$:

$$
\hat{O}: z\mapsto z',\qquad
\sigma_i'=\operatorname{proj}_i(z').
$$

subject to:
1. **Flow/event distinction**: Continuous segments obey the declared nodal
   law; direct EPI-writing events contribute separate jumps. Stored pressure
   after an event need not equal freshly evaluated constitutive pressure.
   A local channel tuple alone need not determine an operator's result.
2. **Grammar constraints**: The operator must satisfy its declared U1--U5 role;
   U6 is evaluated from before/after $\Phi_s$ telemetry.
3. **Contracts**: Pre-conditions and post-conditions specific to each operator.

### 2.3 Composition

Operators compose left-to-right. A word's grammar acceptance does not replace
live state admission, and order can change both acceptance and outcome.

The execution path determines the composition:

- A public single-node operator selects its admitted action before subclass
  execution; its generic call is not a graph transaction (§15.2).
- A supported all-target stage builds proposals from one immutable snapshot,
  validates them and commits graph-owned state together. A fallback or explicit
  execution override can instead use the declared sequential stage. A word
  remains ordered even when its individual stages are simultaneous.
- A hybrid event schedule treats each glyph as a zero-duration jump and places
  elapsed time only in the declared continuous intervals. A same-time form jump
  is not a measured finite-time secant for Mutation.
- The REMESH glyph is advisory. Delayed EPI mixing is a separate invocation
  whose state includes the retained history; its one-step gain cannot be
  multiplied across changing histories without the corresponding lifted proof.

These are execution contracts, not an autonomous invocation law. The
[operator-event contract](../docs/contracts/OPERATOR_EVENTS.md) owns schedules,
transaction boundaries, pressure refresh, stage evidence and causal sequences.
Use its [atomic-stage section](../docs/contracts/OPERATOR_EVENTS.md#atomic-stage-observations-and-finite-composition)
for snapshot/target-order guarantees and its
[history section](../docs/contracts/OPERATOR_EVENTS.md#exact-remesh-memory-and-schedule-certificates)
for the exact and represented REMESH distinctions. Captured graph-owned state
and emitted external effects have different rollback boundaries.

Finite evidence must bind the actual endpoints, ordered support, declared
clock and common metric. It does not automatically prove a global binary64
runtime gain, future repetition, solver accuracy or full-channel stability.
The mathematical owners are the
[diffusion and affine-reset theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md) and
[REMESH history derivation](REMESH_INFINITY_DERIVATION.md); their hypotheses
remain necessary when composing their results.

---

## 3. The Operator Taxonomy

The sections below group operators by their declared function. Groups can
overlap: for example, an allowed word closure need not be an equilibrium,
and a U2 stabilizer need not decrease a chosen diagnostic energy.

The [grammar role registry](UNIFIED_GRAMMAR_RULES.md#1-canonical-operator-roles)
owns the role assignments. The
[generated contract table](../docs/API_CONTRACTS.md#canonical-contracts) owns
tokens, primary channels, measurement contexts and postconditions. Neither
classification is a list of every secondary write. The mechanism map in
[§18](#operator-mechanism-and-activation-audit) distinguishes those writes
from supplied selection policies and possible collective interpretations.

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

**Physics**: Declared activation by a positive EPI increment on an existing
node. Zero EPI is a coordinate value, not absence of the nodal substrate;
the map does not derive creation from a vacuum.

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

**Physics**: Configured adjustments selected by latent/active/resonant state
labels. These operational labels do not establish dynamical attractors or
prove transitions between them.

**Public/staged transformation** (regime-dependent):

| Regime | $\nu_f$ change | $\theta$ shift | $\Delta\text{NFR}$ reduction |
|--------|---------------|----------------|--------------------------|
| Latent → Active | +20% | $+0.1$ | $-30\%$ |
| Active → Active | configurable | $+0.2$ | $-20\%$ |
| Resonant → Active | $-5\%$ | $+0.15$ | $-10\%$ |

The direct NAV glyph only performs its declared pressure action. Regime and
latency effects belong to the public/staged path. `NAV_STRICT` and `NAV_RANDOM`
require actual Boolean values on all paths; see the shared option admission
in [§15.2](#152-base-operator-workflow).

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
- Post: Configured capacity, phase and pressure proposals; applicable latency
  metadata is cleared. No general future-coherence bound follows.

### 4.3 Recursivity (REMESH)

**Execution**: The named REMESH glyph records a network advisory with a
declared recursion depth. It does not execute delayed mixing or create nested
structure. The separately requested network history map below mixes supplied
current and past form; it does not derive fractality or preserve an arbitrary
pattern identity.

**Separately requested network history transformation**:

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
- Pre: Declared depth $\geq 1$, configured minimum support and applicable U5
  context. The network history map has its additional admission above.
- Post: The node-level call records an advisory. Explicit network mixing has
  the stated conditional bounds, not a universal nested-identity guarantee.

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

**Implemented map**: Contracts stored $|\Delta\text{NFR}|$. The low-level
glyph preserves EPI, capacity and phase; the public class and shared stage
also apply the configured phase proposal below. Bounded future evolution
still requires a specified law, gains and event schedule.

**Transformation**:

$$
\Delta\text{NFR}' = \Delta\text{NFR} \cdot (1 - \rho)
$$

where the default retention is
$1-\rho=\pi/(\pi+1)\approx0.7585$, hence
$\rho=1/(\pi+1)\approx0.2415$. The gain remains configurable.

**Public/staged phase locking**:

$$
\theta' = \theta + \lambda \cdot \text{wrap}\!\left(\bar{\theta}_{\mathcal{N}} - \theta\right)
$$

where the default is $\lambda=0.3$, with $0\leq\lambda\leq1$, and
$\bar{\theta}_{\mathcal{N}}$ is the unweighted neighbor circular mean.
Setting the coefficient to zero disables angular displacement. Isolates
have no aligning direction; a joint-zero resultant uses the declared
current-phase fallback. The same kernel owns direct public and staged
proposals; an all-target stage reads an immutable snapshot.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| ΔNFR retention factor $f$ | $\pi/(\pi+1)\approx0.7585$ | Default `IL_dnfr_factor`; configurable gain |
| Pressure-square reduction | $1-f^2\approx0.425$ | Isolated $\Delta\mathrm{NFR}^2$ term at the default; not a full-energy bound |
| Phase locking $\lambda$ | $\approx 0.3$ | Configurable coupling strength |

**Storage and observations**: The pressure-only glyph leaves
$E_D+\beta V$ unchanged because that functional does not contain stored
pressure. For one public IL target on simple undirected support, with all
neighbor phases held fixed, write $R$ for their resultant magnitude and
$a\in[-\pi,\pi]$ for the target's wrapped displacement from its direction.
The ideal phase contribution is

$$\Delta V=R\bigl[\cos a-\cos((1-\lambda)a)\bigr]\leq0.$$

This single-target argument does not certify simultaneous phase changes,
subsequent flow or represented transcendental rounding. Contracting stored
pressure with unchanged stored EPI rates cannot decrease the configured
structural $C$ read-out. An ensuing pressure refresh is a separate operation
and need not preserve that comparison. Telemetry retains structural $C$,
the separately named legacy pressure-dispersion statistic, pressure reduction
and phase-locking observations.

**Grammar**: Stabilizer (U2); Bifurcation Handler (U4a).

**Contract**:
- Pre: Active structure exists.
- Post: Stored $|\Delta\text{NFR}|$ is not increased at the contraction;
  EPI and capacity are preserved. No future-flow or autonomous-trigger claim.

Implementation: [shared IL kernel](../src/tnfr/operators/_coherence_stage_kernel.py),
[public lifecycle](../src/tnfr/operators/coherence.py);
[stage and signed-pressure controls](../tests/operators/test_coherence_jacobi_stage.py).

### 6.2 Self-Organization (THOL)

**Physics**: An invoked pressure-reorganization map with conditional nested
child creation. Its acceleration threshold, invocation and hierarchy policy
are supplied inputs; executing THOL does not derive autonomous emergence.

Its primary contract channel is $\Delta\text{NFR}$ with direction
**reorganize**:

$$
\Delta\text{NFR}'=\Delta\text{NFR}+a\,\frac{\partial^2\text{EPI}}{\partial t^2}.
$$

The instantaneous magnitude can rise or fall with the signed acceleration.
THOL's U2 stabilizer role comes from organizing a bifurcation while preserving
global form; it is not a universal one-step pressure contraction.

**Bifurcation detection and history**:

$$
\widehat A=\frac{2}{h_1+h_2}
\left(\frac{x_2-x_1}{h_2}-\frac{x_1-x_0}{h_1}\right),
\qquad h_j=t_j-t_{j-1}>0.
$$

The shared `compute_d2epi_dt2` reads the latest three active samples. An
available timestamped history is authoritative and must end at the current
EPI. Legacy untimed histories retain the unit-step difference
`x2-2*x1+x0`; fewer than three active samples mean unavailable acceleration,
represented by zero. Cached curvature does not replace this history in
graph-backed THOL. The estimate compares prior nodal rates; it does not add
an independent acceleration equation.

When $|\widehat A|>\tau$, the public operator proposes sub-EPIs subject to
hierarchy-depth and domain checks. The node-protocol primitive implements
only the pressure channel; graphless nodes supply their own acceleration.
Both paths use the shared checked pressure proposal in
[`_thol_pressure.py`](../src/tnfr/operators/_thol_pressure.py).

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
- **Nested creation**: Creates child coordinates with hierarchy metadata
  (bifurcation level, hierarchy path, parent reference). Independent persistence
  and autonomous maintenance do not follow from this event.
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

The child has no transport edge until a later admissible coupling action.
Default structural pressure reads actual neighbors, not hierarchy membership.
Likewise, a refresh before integration overwrites THOL's direct pressure
increment; holding that increment through a physical interval instead gives
the additional exact reference change `h*nu_f*THOL_accel*A_hat`. See
[THOL pressure and feedback](THOL_PRESSURE_FEEDBACK.md) for the execution-order
budget, disconnected zero modes and finite implementation evidence.

**Grammar**: Stabilizer (U2); Bifurcation Handler (U4a); Transformer (U4b).

**Contract**:
- Pre: Finite supported state and valid configuration. When enabled, the
  optional strict gate additionally checks configured EPI/capacity minima,
  positive stored pressure, connectivity and at least three active history
  samples. Exceeding the acceleration threshold is not required to execute
  the pressure action. See the shared
  [THOL precondition owner](../src/tnfr/operators/preconditions/self_organization.py).
- Post: Parent EPI, capacity and phase are preserved; pressure follows the
  signed acceleration proposal. Child creation additionally requires its
  threshold, depth and proposal checks; a successful THOL may create no child.

---

## 7. Destabilizers

Destabilizers increase pressure or capacity stress under their contracts.
Grammar rule U2 requires compensation by a stabilizer as a syntactic safety
policy. Integral convergence additionally needs a declared pressure law,
operator gains, timing, and state-space bounds.

### 7.1 Dissonance (OZ)

**Implemented map**: Changes stored pressure while preserving EPI, capacity,
phase and support. Deterministic amplification, fixed near-zero seeding,
additive noise and network propagation are distinct branches.

**Transformation**:

$$
\Delta\text{NFR}' =
\begin{cases}
f\,\Delta\text{NFR},&|\Delta\text{NFR}|>10^{-9},\\
0.1,&|\Delta\text{NFR}|\leq10^{-9}.
\end{cases}
$$

where the default amplification is
$f=(\pi+1)/\pi\approx1.3183$, the reciprocal of the default IL retention.
It remains an operational gain with $f>1$. The fixed cutoff and nonzero seed
are implementation choices, not a derived spontaneous source.
With `OZ_NOISE_MODE=True`, positive `OZ_SIGMA` instead gives
$p'=p+\mathrm{jitter}$; nonpositive sigma leaves the local pressure unchanged.
An additive sample can decrease $|p|$, so amplification's local magnitude
postcondition is not a theorem for the noise branch.
`OZ_NOISE_MODE` uses the shared Boolean parser on direct, public and stage
paths; a supported false string selects the deterministic branch. Unknown
Boolean strings reject before branch effects (§15.2).

**Bifurcation read-out**: The implementation compares the magnitude of its
observed structural acceleration with the configured threshold $\tau$.
A flag records that comparison, not a bifurcation theorem for an underlying
continuous law. U4a separately requires handler context for the trigger token.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Amplification factor $f$ | $(\pi+1)/\pi\approx1.3183$ | Default `OZ_dnfr_factor`; configurable gain |
| Pressure-square increase | $f^2-1\approx0.738$ | Isolated $\Delta\mathrm{NFR}^2$ term at the default; not a full-energy bound |

**Properties**:
- **Network propagation**: The public class and shared stage can add positive
  increments to neighbors through phase-weighted, uniform or frequency-weighted
  policies. Positive incoming increments can cancel a negative local pressure;
  the local proposal and final accumulated pressure have separate contracts.
- **Bifurcation detection**: Monitors $\partial^2\text{EPI}/\partial t^2$ against threshold $\tau$.
- **Telemetry**: Records propagation events, affected nodes, and bifurcation flags.

**Grammar**: Destabilizer (U2); Bifurcation Trigger (U4a); Closure (U1b).

**Contract**:
- Pre: Consumed finite pressure and the active branch's admitted factors.
  Enabled strict preconditions additionally impose configured EPI, capacity,
  pressure and overload policies; these do not prove absence of a dynamical
  bifurcation or future instability.
- Post: The deterministic local branch increases pressure magnitude. Noise and
  incoming propagation require their separately retained observations. An
  acceleration-threshold flag is a configured diagnostic, not a bifurcation
  proof or an endogenous instruction to execute another operator.

Changing stored pressure alone leaves $E_D+\beta V$ unchanged. Recomputing
pressure from unchanged constitutive inputs can erase the event's pressure
write; persistent forcing or pressure memory must be supplied explicitly.
Implementation: [local glyph](../src/tnfr/operators/__init__.py),
[public propagation](../src/tnfr/operators/dissonance.py),
[shared stage](../src/tnfr/operators/_dissonance_stage_kernel.py);
[signed cancellation and noise controls](../tests/operators/test_dissonance_jacobi_stage.py).

### 7.2 Expansion (VAL)

**Implemented map**: Multiplies capacity by the configured factor. At held
pressure this multiplies the nodal product by that factor. The default
`EDGE_AWARE_ENABLED=True` branch also scales and projects EPI; disabling it
leaves EPI untouched. Neither branch directly writes pressure, phase or support.

**Transformation**:

$$
\nu_f' = f_{\text{VAL}}\nu_f, \qquad
f_{\text{VAL}}=\text{VAL\_scale}>1
$$

With edge awareness enabled, the form factor is limited by the configured
sign-specific EPI boundary, then the selected hard/soft projection is applied.
Thus a positive factor does not guarantee an increase in signed EPI: negative
form, saturation and projection must retain their actual endpoint. This is
an event map, not an increase of graph or state-space dimension. With edge
awareness disabled, $E_D+\beta V$ is unchanged at the event; with it enabled,
the actual EPI reset determines the storage change.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Default scale factor $f_{\text{VAL}}$ | $1+1/(4\pi)\approx1.0796$ | Operational capacity step |
| Nominal square-factor change | $f^2-1\approx0.166$ at the default | Algebraic $\nu_f^2$ diagnostic; the structural candidate energy has no explicit $\nu_f$ term |
| Optional strict minimum EPI | $1/(2\pi) \approx 0.159$ | Supplied `VAL_MIN_EPI` admission policy |

**Grammar**: Destabilizer (U2).

**Contract**:
- Pre: When strict preconditions are enabled, configured capacity saturation,
  minimum stored pressure and EPI, and optional network-size checks apply.
  These policies are not consequences of the nodal identity.
- Post: Capacity is not decreased; no EPI, graph or state-space dimension
  increase follows from multiplying $\nu_f$. U2 compensation and prefix-debt
  limits still apply.

Implementation: [shared VAL/NUL kernel](../src/tnfr/operators/_scale_operator_kernel.py);
[enabled/disabled boundary and signed-form controls](../tests/operators/test_scale_operator_kernel.py).

**Optional VAL telemetry.** `expansion_metrics` reports signed relative form
and capacity changes only when their denominators are nonzero. Missing stored
acceleration and undefined growth ratios remain `None`; present invalid values
and nonfinite or unordered thresholds reject the observation. Stored acceleration
has unverified provenance and is not a temporal bifurcation certificate.
`coherence_above_threshold` reads the current immediate-neighbor proxy, without
a pre-event baseline. `growth_ratio_within_policy` applies a configured numeric
band, not a derived fractality test. The legacy `coherence_preserved` and
`fractal_preserved` keys alias these respective policies. `expansion_healthy`
is `False` when a known policy fails, `None` when no failure is known but
required evidence is unavailable, and `True` only when every policy passes;
`assessment_status`, `failed_policies` and `unavailable_policies` separate those
cases. This is optional post-event reporting, not a new VAL precondition,
rollback promise or future-health theorem. The
[focused telemetry controls](../tests/operators/test_expansion_metrics_scope.py)
cover signed form, invalid authoritative aliases, policy boundaries and absence.

---

## 8. Coupling and Propagation

Coupling operators propose or use phase-compatible interactions between nodes.
Grammar rule U3 verifies circular separation:
$|\operatorname{wrap}(\phi_i-\phi_j)|\leq\Delta\phi_{\max}$.

### 8.1 Coupling (UM)

**Execution**: Proposes configured circular phase alignment across admitted
neighbors and, when requested, functional links for subsequent interaction.

**Transformation**:

$$
\phi_i' \to \phi_j', \qquad |\operatorname{wrap}(\phi_i - \phi_j)| \leq \Delta\phi_{\max}
$$

**Two separate gates**: U3 compares wrapped phase gaps with the admitted
circular limit, $\pi/2$ by default. When functional links are enabled,
`UM_COMPAT_THRESHOLD` separately gates the proposed link's configured
form/Si/phase score; its default is $\pi/(\pi+1)\approx0.7585$.
That score threshold is not a phase angle or an autonomous attachment law.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Functional-link score threshold | $\pi/(\pi+1) \approx 0.7585$ | Configured `UM_COMPAT_THRESHOLD`; only the enabled link branch consumes it |
| Phase push | $1/(\pi + 1) \approx 0.241$ | Configured circular-alignment gain; shared numeric value with EN mixing |
| $\Delta\text{NFR}$ reduction | $1/(2\pi)\approx0.1592$ | Default phase-alignment pressure-relief gain |

**Properties**:
- **Phase verification mandatory**: Circular separation above the configured
  U3 limit rejects coupling. This gate alone supplies no wave-amplitude or
  destructive-interference calculation.
- **EPI identity preserving**: Form is not modified; phase alignment, optional
  capacity synchronization, pressure relief and functional-link proposals
  have separate configured effects.
- **$\nu_f$ synchronization**: Optional frequency alignment across coupled nodes.

`UM_BIDIRECTIONAL`, `UM_SYNC_VF`, `UM_STABILIZE_DNFR` and
`UM_FUNCTIONAL_LINKS` control separate optional effects. Their shared admission
is in [§15.2](#152-base-operator-workflow); disabling an effect cannot bypass U3.

**Grammar**: Coupling (U3); requires phase verification.

**Contract**:
- Pre: The non-disableable U3 gate requires an existing phase-compatible
  neighbor. Enabled strict preconditions additionally check configured EPI
  and capacity thresholds. Link candidates retain their separate score and
  post-merge phase admission.
- Post: Configured circular phase proposals; EPI preserved. Optional functional
  proposals may add admitted links. Neither global phase-spread decrease nor
  the creation of at least one edge follows from a successful invocation.

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

Transformers execute declared state reorganizations that require recent
destabilizing context and, for ZHIR, a prior coherent base. A phase reset or
threshold flag is not by itself a mathematical bifurcation of a continuous law.
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
The displayed sign formula concerns nonzero pressure. The represented kernel
uses `copysign(1.0, pressure)`: `+0.0` selects the positive shift and `-0.0`
the negative shift. Zero current pressure therefore does not suppress an
otherwise admitted event whose preceding observed growth crossed the threshold.
The runtime preconditions and U4b history determine whether the mutation is
admissible; $|\Delta\text{NFR}|$ does not scale this phase step.

Native occurrence additionally depends on selection. The
[heterogeneous-capacity admission study](nodal/FORCED_WINDING_AND_WRITERS.md#35-heterogeneous-capacity-opens-a-conditional-mutation-admission-gate)
separates fresh diagnostic compatibility, declared histories, an executed
default prefix and the direction of the resulting phase action.

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

**Implemented map**: Attenuates capacity without changing EPI, stored pressure,
phase or support. Finite attenuation need not stop later evolution. Even along
a family with $\nu_f\to0$, vanishing nodal rate requires control of the product
$\nu_f\Delta\mathrm{NFR}$, not of capacity alone. A subsequent pressure refresh
may depend on the changed capacity; nonzero declared forcing is separate.

**Transformation**:

$$
\nu_f'=f_{\text{SHA}}\nu_f,\qquad
f_{\text{SHA}}=1-1/(4\pi)\approx0.9204
$$

EPI is preserved by leaving it untouched. The public class and shared stage
record a latency snapshot; this metadata does not enforce future preservation.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| $\nu_f$ suppression factor | $1-1/(4\pi)\approx0.9204$ | Operational `SHA_VF_FACTOR` |

**Properties**:
- **Latency state**: Activates a latent flag with timestamped EPI snapshot.
- **Reactivation bookkeeping**: Later lifecycle code can inspect and clear
  latency attributes. A drift tolerance or wall-clock timestamp is not a
  structural-time evolution law or a physical preservation theorem.
- **All-target time basis**: One accepted SHA stage gives every target the same
  latency-start timestamp while preserving a target-specific EPI snapshot.

**Grammar**: Closure (U1b).

**Contract**:
- Pre: Consumed capacity is finite and nonnegative. When strict preconditions
  are enabled, capacity must meet `SHA_MIN_VF`; no critical-pressure gate or
  nonzero-EPI requirement is imposed by that precondition.
- Post: Capacity is not increased; EPI, pressure and phase remain fixed at
  the event, with the declared latency metadata. Finite attenuation need not
  produce zero capacity or freeze subsequent nodal evolution.

Since the joint storage $E_D+\beta V$ has no capacity term, SHA alone has zero
instantaneous change in this functional. It changes subsequent mobility under
an admitted law rather than supplying event work or deciding when UM occurs.
Implementation: [SHA proposal and lifecycle](../src/tnfr/operators/al_sha_stage_proposals.py);
[shared lifecycle controls](../tests/operators/test_al_sha_stage_proposals.py).

### 10.2 Contraction (NUL)

**Physics**: Attenuates capacity and rescales stored pressure by the reciprocal
configured factor. This changes neither graph dimension nor the dimension of
the EPI state space; “contraction” names this declared map.

**Transformation**:

$$
\nu_f'=f_{\text{NUL}}\nu_f,\qquad
f_{\text{NUL}}=1-1/(4\pi)\approx0.9204
$$

Stored pressure is multiplied by the reciprocal capacity factor:

$$
\Delta\text{NFR}'=\frac{1}{f_{\text{NUL}}}\Delta\text{NFR}
\approx1.0865\,\Delta\text{NFR}
$$

The ideal product $\nu_f'p'=\nu_fp$ is preserved for this stored-pressure
map. It is not a statement about pressure after constitutive refresh. As with
VAL, `EDGE_AWARE_ENABLED=True` additionally multiplies EPI by $f_{\mathrm{NUL}}$
and applies the configured projection; the disabled branch does not consume
or change EPI. Capacity-only contraction leaves $E_D+\beta V$ unchanged;
the enabled form reset has its own storage budget. Local contraction toward
zero need not reduce differences from neighboring forms.

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
- Pre: Admitted capacity, pressure and any consumed scalar form. Enabled
  strict preconditions additionally impose configured minimum capacity and
  EPI plus a maximum density policy; these are not universal safety bounds.
- Post: Capacity is not increased; stored pressure follows the checked
  reciprocal proposal, with zero pressure remaining zero. Subsequent pressure
  refresh is a separate constitutive operation.

Implementation: [shared scale proposal](../src/tnfr/operators/_scale_operator_kernel.py);
[reciprocal coefficient and branch controls](../tests/operators/test_scale_operator_kernel.py).

---

## 11. Fragments and Complete Words

Operators compose left-to-right. A named fragment is reusable syntax; a
complete word is a concrete sequence evaluated against its execution context.
Canonical U6 additionally needs before/after $\Phi_s$ snapshots and therefore
cannot be certified from operator labels alone.

### 11.1 Named workflow fragments

The [grammar composition section](UNIFIED_GRAMMAR_RULES.md#8-composition-and-implementation)
defines the reusable Bootstrap, Stabilize, Explore and Propagate fragments.
They require the missing initialization, closure, history or phase context
before becoming executable words. For example, `[AL, IL, OZ, ZHIR, IL, SHA]`
supplies a generator, prior Coherence, recent destabilizer, handler and closure
around the Explore body; Mutation still requires its live growth evidence.

### 11.2 Complete-word examples

Read executable recipes through `list_sequences()` or the fluent SDK's
`list_canonical_sequences()` rather than copying a second name/word table.
The [recipe registry](../src/tnfr/operators/canonical_patterns.py) owns named
canonical patterns; the [study API](../src/tnfr/sdk/study.py) exposes the
registered study recipes. Names such as `phase_lock`, `resonance_peak_hold`
and `therapeutic_protocol` are labels, not proofs of locking, peak detection,
maintenance or a medical effect. Each execution retains the selected word,
actual glyph trace and live admission outcome.

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

Use the [U2 role and debt contract](UNIFIED_GRAMMAR_RULES.md#3-u2-stabilization-and-boundedness-policy).
It classifies operator obligations; it is not a measured energy inequality.
Primary channel, grammatical role and diagnostic response are distinct.

### 12.2 Nominal operator parameters

The maps in §§4–10 state their effective factors and branches. Their configured
magnitudes do not establish universal response multipliers: an IL pressure
factor is not a full-state energy factor, a ZHIR phase step is not a curvature
growth rate, and a VAL capacity increase has no unconditional energy sign.
Read the effective factors from the actual invocation and use the
[configuration owners](#14-operator-constants-reference) for their defaults.

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

[operator_contracts.py](../src/tnfr/operators/operator_contracts.py) owns each
operator's primary channel, qualitative direction, scope, measurement context
and postcondition. The [API contract table](../docs/API_CONTRACTS.md#canonical-contracts)
is generated from that registry. Introspection and the integrity monitor reuse
the same records instead of maintaining another operator catalog here.

The [integrity owner](../src/tnfr/physics/integrity.py) provides sampled audits
and runtime monitoring with OFF, OBSERVE and ENFORCE modes. Report evaluated
coverage, unavailable context and individual outcomes separately. A monitor's
post-event exception does not by itself roll back the event; the execution
path must supply any transaction guarantee. Direct glyphs, public classes and
atomic stages can have different secondary effects and measurement boundaries.

The contract catalog is complete as a specification but is not instantaneously
identifiable from its categorical fields. The exact certificate
`contract_identifiability_certificate()` shows that channel, direction, scale
and context leave Silence and Contraction in the same class. Their distinct
semantics require a temporal or quantitative postcondition observation. This
negative inverse result neither removes an operator nor proves completeness of
the catalog over every admissible TNFR transformation.

---

## 14. Operator Constants Reference

The contract constrains a map's channel and postcondition, not a unique numeric
gain. Effective factors must be admitted on their active branch before writes.
Their owners are [canonical constants](../src/tnfr/constants/canonical.py),
[operator factor contracts](../src/tnfr/operators/factor_contracts.py) and the
individual map kernels linked in §§4–10.

### 14.1 The structural scale

In radians, $pi$ is a half-turn and sets the range of a principal phase
difference. The resulting [circular field bounds](../docs/STRUCTURAL_FIELDS_TETRAD.md)
do not select a pressure gain, coherence threshold or physical clock.

### 14.2 Operator gain magnitudes (operational)

Mixing fractions, capacity multipliers, emission boosts, phase steps and child
scales are supplied operational choices. An expression containing $pi$ is
still a configured choice unless a stated theorem fixes it under explicit
premises. Reproduction requires the effective configuration, not a copied
table of nominal defaults. The parameter-by-parameter scope belongs to
[NODAL_PARAMETER_FOUNDATIONS.md](NODAL_PARAMETER_FOUNDATIONS.md).

### 14.3 Constant-Operator-Grammar Traceability

Trace an invocation through its registered contract, active factor resolution,
map implementation and [grammar roles](UNIFIED_GRAMMAR_RULES.md#1-canonical-operator-roles).
Neither the primary channel nor the sign of a gain determines a grammatical
role or a full-state Lyapunov inequality. Runtime measurements and analytic
proofs must use the same admitted branch and state representation.

---

## 15. Implementation Reference

### 15.1 Source Modules

| Responsibility | Owner |
| --- | --- |
| Public operator classes and common call boundary | [definitions.py](../src/tnfr/operators/definitions.py), [definitions_base.py](../src/tnfr/operators/definitions_base.py) |
| Registered operator contracts and active factors | [operator_contracts.py](../src/tnfr/operators/operator_contracts.py), [factor_contracts.py](../src/tnfr/operators/factor_contracts.py) |
| Word roles, validation, live selection and observation | [Grammar implementation map](UNIFIED_GRAMMAR_RULES.md#8-composition-and-implementation) |
| Stage, schedule and delayed-history execution/evidence | [Operator-event contract](../docs/contracts/OPERATOR_EVENTS.md) |
| Integrity and diagnostic energy policies | [integrity.py](../src/tnfr/physics/integrity.py), [lyapunov.py](../src/tnfr/physics/lyapunov.py) |
| Declared recipes and shared CLI/SDK studies | [canonical_patterns.py](../src/tnfr/operators/canonical_patterns.py), [study.py](../src/tnfr/sdk/study.py) |

Individual operator sections link their map kernels. The event contract links
the specialized execution and certificate owners; this document does not
maintain a second list of every event/REMESH adapter.

### 15.2 Base Operator Workflow

`Operator.__call__(G, node, **kw)` in
[definitions_base.py](../src/tnfr/operators/definitions_base.py) follows this
order:

1. Admit the execution window, graph seed, replayable history and hard
   invariants. U3 cannot be disabled by turning off optional preconditions.
2. Validate shared arguments, consumed form, requested telemetry sinks and
   monitor interface, then the enabled operator-specific preconditions.
3. Select the grammar-admitted action before subclass writes. If fallback is
   selected, invoke that operator's own public workflow, metadata and metrics.
   Strict rejection supplies no replacement. Admit active factors before
   preparatory writes.
4. Execute the selected map and its declared secondary effects. Capture the
   pre-state only when metrics or the nodal comparison are requested; invoke
   before/after integrity hooks only when a monitor is installed.
5. Perform any requested held-step nodal comparison and metrics collection.
   The comparison uses a supplied interval after the event; it neither measures
   the event's duration nor identifies a hybrid jump with continuous flow.

The generic public call is not a graph transaction. Later hook or comparison
failure need not undo a completed glyph. Transactional atomic stages have the
separate [stage contract](../docs/contracts/OPERATOR_EVENTS.md#atomic-stage-observations-and-finite-composition).
U6 remains a reference-relative observation rather than an implicit proof
supplied by this pipeline.

**Branch controls.** [factor_contracts.py](../src/tnfr/operators/factor_contracts.py)
resolves the following controls before active factors or branch proposals, on
direct glyph, public-class and atomic-stage paths:

| Controls | Admission |
| --- | --- |
| `UM_BIDIRECTIONAL`, `UM_SYNC_VF`, `UM_STABILIZE_DNFR`, `UM_FUNCTIONAL_LINKS`; `OZ_NOISE_MODE` | Shared Boolean parsing: supported false strings disable the branch; unknown strings reject. Non-string values retain the shared parser's Boolean conversion. |
| `NAV_STRICT`, `NAV_RANDOM` | Actual Boolean values only; strings and numeric `0`/`1` reject. |

These different input contracts are intentional. Branch selection is supplied
configuration, not an emergent operator choice or a relaxation of live grammar.

### 15.3 Executable Demonstrations

Use the [examples catalog](../examples/README.md) for runnable entry points,
including operator sequences, grammar rejection and EN/RA realization studies.
The [event contract](../docs/contracts/OPERATOR_EVENTS.md) routes the specialized
flow, REMESH and history examples to their precise hypotheses. A demonstration
is an executable illustration, not an automatic execution or future guarantee.

### 15.4 SDK Entry Points

This read-only example obtains contracts and recipe definitions without
executing or selecting an operator:

```python
from tnfr.sdk import TNFR, list_sequences

emission = TNFR.operators("emission")
assert emission["token"] == "emission"
assert emission["channel"] == "EPI"
recipes = list_sequences()
assert recipes and all(recipe["operators"] for recipe in recipes)
```

For actual execution use the [minimal admitted word](../docs/API_CONTRACTS.md#valid-execution-example)
or the [shared CLI/SDK study workflow](../docs/CLI_AND_SDK.md). Preserve its
preparation, operator cycles, seeds and live rejection outcomes. A configured
automatic selector is a policy, not a derived invocation law.

---

## 16. Summary

Use the registry for **what a named operator promises**, grammar for **word
and live admission**, and the selected execution contract for **what commits
and what evidence is retained**. The [mechanism audit](#operator-mechanism-and-activation-audit)
separates these from collective descent, autonomous occurrence and physical
identification. None follows solely from a catalog label or a diagnostic score.

---

## 17. Experimental Operator-Tetrad Synergies

Examples 37–39 are finite diagnostic instruments. An operator request can be
replaced by grammar fallback; its measured response belongs to the realized
glyph and actual execution path. No numerical response table is asserted here
without a retained preparation, source/configuration and response record.

### 17.1 Centralized Operator-Channel Classification

Read primary channels from the
[generated contract table](../docs/API_CONTRACTS.md#canonical-contracts).
Capacity and pressure enter the nodal rate multiplicatively; changes to form,
phase, support or history can influence their later values under a specified
law. A primary-channel label does not enumerate secondary writes or determine
an energy sign.

### 17.2 Operator-Tetrad Fingerprint Matrix

[Example 37](../examples/02_physics_regimes/37_operator_tetrad_synergy.py)
measures requested/realized glyphs and before/after field changes on a supplied
graph. Such a matrix is a finite response profile, not a universal fingerprint
or a proof that the tetrad reconstructs the complete state. Zero sampled change
does not establish neutrality under subsequent pressure refresh or evolution.

### 17.3 IL-OZ Tetrad Symmetry

Matching sampled IL/OZ read-outs would not establish equality of their maps or
an exact symmetry. IL contracts stored pressure and its public/staged path also
aligns phase. OZ has a different pressure map, near-zero/jitter branches and
optional propagation. Interpret a response only after checking the realized
glyph, auxiliary writes and pressure-refresh boundary.

### 17.4 Structural Potential Linear Response

At fixed support and distance kernel, structural potential is linear in its
pressure source: $Phi_s=Bp$, hence $DeltaPhi_s=BDelta p$. This identity
belongs to the [field definition](../docs/STRUCTURAL_FIELDS_TETRAD.md), not to
a fitted response sweep. A topology or distance change also changes $B$.
The coherence-length estimator has no corresponding general linear law;
its fit/fallback provenance must remain explicit. Nonlinear sampled response
alone does not establish divergence or a critical transition.

### 17.5 Diagnostic dependency chain

The read-out chain is

$$

\text{executed map} \longrightarrow \text{updated state}
\longrightarrow \text{fields} \longrightarrow \text{diagnostics}.

$$

The fields observe a fully specified state; they are neither additional
independent state variables nor a complete future predictor. If a controller
uses a diagnostic, that feedback policy must be declared separately.

### 17.6 Grammar-Energy Landscape

[Example 38](../examples/02_physics_regimes/38_grammar_energy_landscape.py)
compares measured diagnostic energy with configured operator multipliers on
finite sequences. U2 constrains word obligations; it does not bound that energy
for every law or realization. A mismatch between a nominal multiplier and a
measured change is evidence about the specified comparison, not a stronger
grammar theorem.

### 17.7 Executable Demonstrations

The instruments are [operator-field responses](../examples/02_physics_regimes/37_operator_tetrad_synergy.py),
[grammar-energy comparison](../examples/02_physics_regimes/38_grammar_energy_landscape.py)
and [nodal-row decomposition](../examples/02_physics_regimes/39_nodal_equation_decomposition.py).
Retain source, configuration, graph/node order, input, seed, precision,
backend, pressure refresh and realized glyph trace with any evaluated response.
A seed alone cannot bind an older reported value to the current implementation.

---

<a id="operator-mechanism-and-activation-audit"></a>
## 18. Mechanism and activation audit

This map addresses connection formation using the existing 13 operators. It
does not add a second executable catalog: names, primary channels and contracts
remain in [operator_contracts.py](../src/tnfr/operators/operator_contracts.py).
The distinction between primitive glyphs, public classes and snapshot stages
is essential. A primary channel is not a complete list of writes.

For each proposed mechanism distinguish four questions: **what action is
implemented, when it is admissible, who selects it, and what accounts for its
actual state change**. Existing grammar, history gates and controllers answer
parts of the first three under configured policies. They do not yet derive a
unique autonomous law from an NFR's state. The storage `S=E_D+beta*V` below
belongs to the [conditional relational model](nodal/RELATIONAL_EXCHANGE_ADMISSION.md),
not to every operator runtime or a universal physical energy.

### All thirteen actions: useful mechanisms and remaining premises

| Operator | Actual mechanism relevant to interaction | Reusable synergy and boundary |
| --- | --- | --- |
| **AL — Emission** | Adds/projected form on existing support; basic event holds phase, capacity and stored pressure | A local form contrast changes the following relational phase rates. Emission from uniform form costs storage; amplitude, source and occurrence remain supplied. No node or edge creation. |
| **EN — Reception** | Blends unweighted existing incoming-neighbor form; source ranking is telemetry | One unclipped target can reduce form storage and alter later phase compatibility. No U3 source filter or reception from an absent edge is implemented. |
| **IL — Coherence** | Glyph contracts stored pressure; public/staged path also aligns target phase with its neighbor resultant | A single-target phase reset can release phase storage. The pressure-only change may disappear on refresh; no global or simultaneous-stage storage theorem follows from its stabilizer label. |
| **OZ — Dissonance** | Perturbs stored pressure, with near-zero/jitter branches and optional neighbor propagation | Supplies a declared perturbation and grammar history; pressure-only change is neutral in this particular `S`. Refreshed constitutive pressure can erase it. Signed propagation can cancel another node's pressure. |
| **UM — Coupling** | Moves phase, optionally aligns capacity/contracts stored pressure and proposes links to compatible nonneighbors | A phase reset can offset a positive new-edge cost. Requires an existing compatible neighbor at the acting target. Candidate inventory, affinity/Si threshold, tie policy and time are supplied; generated conductance is generally nonunit. |
| **RA — Resonance** | Mixes form over existing U3-compatible neighbors; configured phase interpolation and triggered capacity amplification | RA then UM can reduce total storage while adding a positive-cost edge. RA itself creates no support and does not contract stored pressure. Sign/kind preservation does not imply maintained pattern identity. |
| **SHA — Silence** | Attenuates capacity and records latency information, preserving form at the event | Capacity-only reset is neutral in `S` but changes future rates. A positive retention factor does not produce zero capacity in a finite ideal event; vanishing capacity alone does not prove `nu*p` vanishes. |
| **VAL — Expansion** | Raises capacity; default edge-aware policy also scales/projects EPI | Capacity changes local speed, while the auxiliary form reset can change subsequent phase response and storage. Expansion does not derive a new state dimension, source or clock. |
| **NUL — Contraction** | Lowers capacity and rescales stored pressure; default edge-aware policy also scales/projects EPI | Ideal stored-product preservation is distinct from the freshly evaluated relational rate. Account for the actual form jump; it is not generally a storage-neutral capacity-only event. |
| **THOL — Self-organization** | Primitive signed acceleration/pressure reset; public/staged path can create an isolated nested child after history/U5 gates | Existing eligibility/dispatch and child-feedback tools can be reused. Parent metadata is not an edge. Child form/capacity and timing follow construction rules; absent edge storage does not mean free substrate creation. |
| **ZHIR — Mutation** | Configured phase reset following a live signed EPI-growth test and separate grammar context | Can alter phase geometry without adding support. A valid growth trigger is eligibility, not a derived necessity to mutate. Rich-form magnitude is not signed scalar growth; the full reset may raise storage. |
| **NAV — Transition** | Primitive pressure change; public/staged regimes can also change phase and capacity | A candidate preparation/phase-reset mechanism, with its own budget. Regime factors, optional random draw and invocation remain policies; latency metadata is not a structural event clock. |
| **REMESH — Recursivity** | Advisory glyph, delayed same-support EPI mixing, and explicitly invoked topological replacement are distinct APIs | Fixed-node rewiring can reuse joint reset accounting. Delayed-history disagreement bounds are not a theorem for `S`; community replacement may change nodes and lies outside the same-node observer. |

Execution owners are linked in §15. The particularly relevant shared kernels
are [AL/SHA proposals](../src/tnfr/operators/al_sha_stage_proposals.py),
[reception input](../src/tnfr/operators/_reception_kernel.py),
[UM proposals](../src/tnfr/operators/_coupling_stage_kernel.py),
[atomic stages](../src/tnfr/operators/network_stage.py),
[Mutation admission](../src/tnfr/operators/_mutation_gate.py), and
[THOL dispatch](../src/tnfr/operators/self_organization_selection.py).

### Quantitative synergies, without inventing an activation law

The [full-reset derivation](nodal/RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact)
and shared `observe_relational_reset` separate nodal reorganization on old
support from support work at the new state. They retain the actual endpoints
instead of assigning energy signs from operator names.

- **Form changes influence phase evolution.** For held support, phase and
  capacity, an AL/EN jump `d` changes the next phase row by
  `(w/beta)*diag(nu/H)*B*d`. One unclipped EN target has nonpositive form cost;
  an AL jump from uniform form supplies positive cost. RA/VAL/NUL may also
  change form, but their auxiliary writes and clipping must be included.
- **Nodal relaxation can accompany attachment.** Actual UM and RA-then-UM
  witnesses have positive bridge cost but negative complete storage change.
  This removes the frozen-triad budget obstruction for those declared actions.
  It does not derive their selection, time or subsequent maintenance.
- **Phase-only IL can prepare a budget.** On simple undirected support, one
  target with fixed neighbors and nonzero resultant `R*exp(i*mu)` moves from
  `theta` toward `mu` by a fraction `lambda` in `[0,1]`. With the selected
  shortest displacement `a=wrap(mu-theta)`, the ideal phase cost change is
  `R*(cos(a)-cos((1-lambda)*a))<=0`. This proves a conditional single-target
  result; it does not extend automatically to overlapping simultaneous resets.
- **Capacity controls response, not stored identity.** Held form/phase/support
  makes a capacity-only reset neutral in `S`. The later dynamics still changes.
  VAL/NUL's default EPI effects and post-event pressure refresh prevent a
  blanket capacity-only interpretation. Zero capacity and nonzero Gamma must
  retain their separate contracts.
- **History can gate an event without explaining its origin.** ZHIR uses a
  measured signed secant, THOL uses its configured history and construction,
  REMESH consumes declared delayed state. These are useful causal inputs;
  none turns a diagnostic label into an independently derived event law.

### Hidden assumptions exposed by the map

**Stored pressure versus an independently evaluated law.** Pure stored-pressure
changes from IL/OZ/THOL/NAV affect a held-pressure interval. If every argument
of a state-based constitutive law is unchanged, its next refresh restores the
same pressure. A pressure intervention needs an explicit lifetime or retained
state to have a lasting role. The relational observer does not infer either.

**Continuous evolution versus a finite operator action.** Uniform form gives
`q=0` and zero instantaneous relational phase velocity, even at nonuniform
admitted phase. UM can still move phase in that state. Its finite reset is not
automatically a time step of the current continuous law. Any proposed internal
NFR activation must justify this boundary, not merely pass a storage check.

**Potential contact versus existing support.** UM's search over graph nodes
or a supplied sample assumes candidate access. EN/RA consume existing edges;
they cannot themselves transmit input across a missing edge. THOL children
start isolated. Two isolates cannot bootstrap UM because its initial U3 gate
needs an existing neighbor. The potential-contact relation is therefore a
separate hypothesis from the realized transport graph.

**Symmetry versus a configured score.** UM's absolute-form similarity changes
under a common form offset, while the selected relational law does not. Si
enters its affinity, and rank resolves some ties. Successful execution of that
policy does not derive it from offset-covariant nodal dynamics. The existing
[event-action audit](nodal/RELATIONAL_SUPPORT_EVENTS.md#support-law-choice-and-clock)
supplies the appropriate symmetry/clock controls.

**Phase compatibility versus synchronized rhythms.** A small instantaneous
wrapped gap does not imply equal phase velocities or sustained locking. Equal
velocities can retain a gap outside U3. A possible connection mechanism should
retain both `delta=theta_j-theta_i` and `delta_dot` from the complete law, and
state what makes the candidate observable before connection. Treating rhythm
locking as a prerequisite is a testable hypothesis, not yet an event rule.
The SDK's modal `rhythm()` and snapshot `resonance()` observations do not supply
that missing temporal proof or clock. Synchronization produced by an already
present link cannot circularly establish the cause of that link's birth.

**Representation versus physical evidence.** Native AL/EN serialized
uniform-real BEPI now reaches relational execution through shared signed
scalar admission. Rich or complex form is still outside that scalar model.
Mutation uses that same boundary for live history matching, rather than a
magnitude surrogate. These are integration corrections, not new physics.

<a id="collective-operator-descent"></a>
### When an internal transformation defines a collective operator

Let `z'=f(z)` be one declared complete fine law, `y=R(z)` a proposed NFR
description, and `T` a supplied event map. An autonomous continuous law in `y`
requires `DR(z) f(z)` to agree on every admitted state with the same `R(z)`.
An event on that same description additionally exists precisely when

`R(z1)=R(z2) => R(T(z1))=R(T(z2))`.

Necessity follows by evaluating the proposed collective event at the same
input. Sufficiency follows by defining its output using any admitted lift;
the displayed condition makes that definition independent of the lift.
Admission, target/port choice, retained history and any occurrence rule must
also be well defined on those classes. If a transformation changes the state
space, specify the before and after observation maps separately. Covariance
of an event family with a transformed target does not establish closure after
discarding that target. These are conditional descent criteria, not claims
that the named engine operators already satisfy them.

The [unordered-pair state](nodal/SINE_PAIR_STATE.md#sine-replica-unordered-state)
provides an existing exact continuous quotient under its sine-law premises.
It retains internal coordinates; means alone generally fail. Whether an action
at one constituent descends requires its own target and symmetry test. A
continuous fine flow cannot cause a finite jump in a continuous observation;
an apparent jump in a selected partition instead requires an explicit change
of observation or event law. Neither interpretation creates extra fine nodes.

The [full-form formation obstruction](nodal/RELATIONAL_FORMATION_CONTROLS.md#relational-full-consensus-formation-obstruction)
adds a complementary native-law constraint. Its monotone quantity `W` bounds
phase acquisition during smooth flow, and its reset corollary supplies a
necessary jump budget when the same graph, capacity and law are retained.
`W` is not energy: even a storage-passive reset can increase it. Existing
`observe_relational_reset` retains the endpoint data needed for such a
comparison; its represented phase costs are not ideal trigonometric bounds.
Refreshing pressure can erase a pressure-only intervention; changing capacity
or support requires new theorem admission. None of these endpoint checks
selects an action or proves that it follows from the fine continuous dynamics.

The [AL pair comparison](nodal/SINE_PAIR_INTERACTION.md#sine-pair-emission-descent)
now supplies an exact example: the same admitted scalar form map on both
interchangeable members descends, whereas a fixed singleton target generally
needs a retained port mark. Its two exceptional source cases are a synchronized
tip and two actual no-ops. Equal reset storage costs need not give equal
collective outputs. This result uses actual AL form clipping, without claiming
closure of its complete grammar, history and lifecycle runtime.

The [regional-transfer identity](nodal/SINE_PAIR_INTERACTION.md#sine-autonomous-regional-transfer)
adds a distinct realizability test: the closed normalized-sine law preserves
degree/capacity-weighted form. Any actual positive AL-only increment changes
that invariant, even when the event descends to a collective state. Autonomous
regional gain instead has a compensating change outside the region. The
synchronized-pair control also derives its subsequent phase response.
This is an internal exchange mechanism under
the stated law, not the autonomous execution of the registered AL event.

Thus an NFR's identity, its inherited response, a realizable event and the
event's occurrence are separate proof obligations. A useful operator can be
a supplied intervention while autonomous occurrence remains open. The
[sole research queue](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
selects the next bounded comparison from these obligations.
A justified negative result remains useful;
the map provides mechanisms to test without asserting autonomous connection,
substrate generation or physical identification.

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
