# TNFR Structural Operators

<a id="complete-specification-of-the-13-canonical-operators"></a>
## Implemented contracts and canonical development of the 13 operators

**Status**: CANONICAL registry specification; generative completeness over an
independently defined admissible-transformation space remains open
**Version source**: [pyproject.toml](../pyproject.toml)
**Prerequisite**: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §2 (Nodal Equation), [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) (Grammar U1–U6)

---

## Table of Contents

1. [Scope and Motivation](#1-scope-and-motivation)
2. [State, event maps and composition](#2-state-event-maps-and-composition)
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

Under TNFR's [constitutive premise](FUNDAMENTAL_THEORY.md#operator-grammar-constitutive-premise),
structural operators and grammar ground every canonical transformation. The
registered operators implement named node and network maps; that implementation
does not by itself exhaust or prove the theoretical premise. Declared numerical
solvers are a separate, explicit execution path whose canonical interpretation
requires a justified operator realization. The shared nodal integrator consumes admitted capacity,
pressure and any declared Gamma source, with their provenance and residuals.
The unforced continuous row is

$$
\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t) \tag{NE}
$$

Finite operator events instead contribute declared jumps; (NE) alone neither
derives their maps nor selects their occurrence. Their complete input state,
admission and continued law belong to the contracts below. The 13-operator
catalog covers the engine's declared transformation categories:
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
- Lead explanations with English names, such as Emission (AL), Reception (EN)
  and Coherence (IL). Registry display names and lowercase executable tokens
  remain distinct. Simple glyph codes abbreviate compact words and formulas
  without apostrophes. Write ordered composition with arrows, such as
  `Emission → Coherence → Silence`, abbreviated `AL → IL → SHA`;
  slash alternatives and nested expressions are different structures. The
  [source comparison](UNIFIED_GRAMMAR_RULES.md#source-grammar-and-notation)
  separates manuscript prescriptions, runtime policies and dynamical necessity.
- Lowercase English tokens (`emission`, `reception`, ...) are executable API
  identifiers; title-case names are public display/class names.
- Operator gains and thresholds are configured parameters. $\pi$ is the half-turn
  in the chosen radian phase chart, not a derivation of those parameters.
- Grammar roles reference rules U1–U6 from [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md).
- Energy diagnostics and their proof boundaries are discussed in [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md).

---

<a id="2-operator-algebra-from-the-nodal-equation"></a>
## 2. State, event maps and composition

### 2.1 Structural Triad

Each node $i$ carries the declared structural triad. This representation is
not a proof that the three attributes remain independent under every richer
form representation or specified quotient:

| Attribute | Symbol | Domain | Units |
|-----------|--------|--------|-------|
| Form | $\text{EPI}_i$ | Declared form space; scalar glyphs require a signed real chart or uniform-real BEPI embedding | Declared form unit X |
| Reorganization capacity | $\nu_{f,i}$ | Nonnegative reals; individual operators may require strict positivity | Hz_str |
| Phase | $\phi_i$ (or $\theta_i$) | Circle $\mathbb{R}/(2\pi\mathbb{Z})$; admitted finite real representatives | rad |

A wrapped output chart such as $[0,2\pi)$ is not a universal restriction on
input representatives. The [shared U3 gate](../src/tnfr/operators/_phase_gate.py)
admits represented phases before comparing wrapped differences; declared
continuous lifts remain separate data when a theorem consumes arithmetic
phase means. The [capacity foundation](NODAL_PARAMETER_FOUNDATIONS.md#34-derived-pulse-scales-and-observation-periods)
also separates reorganization capacity from a measured oscillation frequency.

The selected model specifies $\Delta\text{NFR}_i$ (structural pressure), which
drives the form row. The nodal identity alone does not determine this driver.
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

<a id="structural-state-action-specification"></a>
### 2.4 Common structural state/action specification

Operators are the proposed generative mechanisms of TNFR organization: their
actions must change what constituents can do together, and their resulting
state must participate in subsequent admitted actions. The historical source
(TNFR.pdf, PDF pp.22–27) presents emergence and functional recursion in this
sense. This motivates implementation requirements; its illustrative biological,
physical and symbolic claims do not establish a law or empirical validation.
Neither a registered name nor a supplied repeating word proves self-organization.

The THOL passage (pp.112–116) also sketches internal reorganization with
unchanged EPI. This makes changed relations and subsequent response relevant
targets even without birth. Its internal coherence matrix is not thereby
identified with graph conductance. REMESH (pp.124–128) requires recognizable
function across scales; copying nodes alone does not realize that role.
Both need explicit structural state and actual operator mechanisms below.

Use the existing [state/representation distinction](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation)
and [partial-action descent](#collective-grammar-descent), rather than a second
operator registry or a universal scalar state. Each concrete action requires:

1. **Source and successor state.** Declare member identities, realized and
   potential relations when consumed, hierarchy, form representation, capacity,
   circular phase and stored versus refreshed pressure. Retain associated
   temporal samples, persistent grammar context, lifecycle, clocks, policy,
   boundary inputs and reproducibility state that affect admission or output.
   A rollback snapshot is not a proof of minimal sufficient state.
2. **Actual action.** Bind the registered operator, execution path, ordered
   targets, parameters and branch to a partial map and its admitted domain.
   Include context writes, abstention and relevant outputs. A supplied action
   or candidate has supplied provenance; autonomous occurrence requires a
   justified state-dependent mechanism, including its clock and abstention.
3. **Structural result.** Identify preserved, transformed, born and removed
   constituents and relations. Declare the reachable output image, inherited
   constraints and any independently supplied inputs. A hierarchy record alone
   does not make a child a functioning collective NFR.
4. **Continued participation.** Specify the subsequent operators and any
   operator-grounded inter-event rows, successor admission and feedback through
   which constituents influence one another. Prove any claimed reduction,
   robustness or effective identity under that complete mechanism. Justify
   inactive operators; no fixed itinerary of all thirteen is required.

The existing descent theorem owns when an observation preserves admission,
actions and outputs. This specification adds no universal law or general
autonomous selector. It connects the historical roles to the actual shared
engine contracts; justified constitutive revisions remain possible.

Two implemented boundaries make that connection concrete. The
[selector batch](../src/tnfr/dynamics/selectors.py) readmits live policy, grammar,
clock, lag records and the original ordered targets after preparation and
selection hooks. Hook changes cannot authorize a glyph using stale admission;
lag counters advance once. Proposals retain their selection boundary, and
hook effects are not implicitly rolled back. The
[THOL signal reader](../src/tnfr/operators/metabolism.py) admits authoritative
form/phase aliases and finite aggregates before clipping; the public birth
path reuses that reader instead of duplicating neighbor admission. Invalid
causal inputs cannot silently become an apparently valid child construction.
The [execution map](THEORY_TO_EXECUTION.md#operator-audit-execution) links the
direct, public, stage and selector controls.

<a id="operator-contextual-rewriting"></a>
#### Guarded rewriting, context and causal access

The [formal-method comparison](BIBLIOGRAPHIC_METHODS.md#formal-operator-languages)
provides tools for the existing specification, not a replacement interpreter.
The following are conditional deductions about marked actions and observations;
they are not claims that the engine already implements a graph grammar or a
bigraphical reactive system.

An action label must retain its operator, path, targets, parameters and consumed
inputs. Write `s --a--> T_a(s)` only when the complete source belongs to `D_a`.
The [partial-action criterion](#collective-grammar-descent) already owns these
domains and the conditions for projecting them. The thirteen names identify
families of transformations, not thirteen fixed maps: parameter values, marks,
history and route matter. A faithful rewrite encoding must preserve admission,
successors and relevant outcomes in both directions, or explicitly claim only
one-way simulation. Encoding a recorded glyph word proves neither property.

**A composite action inherits its prefix guards.** For successful deterministic
actions, the domain of first `a` then `b` is exactly
`D_a intersect inverse_image(T_a,D_b)`. Iterating gives the domain of a finite
word. This follows directly from when composition of partial maps is defined.
Transport the actual successor predicate, including history and support,
through the earlier reset; concatenating token names cannot replace it.
Abstention and non-atomic failure need their explicit outcome transitions.
For example, the [birth/contact control](THOL_PRESSURE_FEEDBACK.md#thol-acquired-birth-participation)
has parent pressure 3/4 without UM and 9/16 with UM. Two IL contractions by 1/2
leave 3/16 above EN's 0.15 ceiling in the first branch but 9/64 below it in the
second; three contractions admit both pressure gates. This derives that part
of the shared preparation without a scan; it does not replace EN's other gates.

**Independent actions require independent reads.** Consider deterministic
partial actions `a,b` on a fixed, explicitly addressed state domain. Let `R_a`
include every value on which `a`'s guard or written values depend, and let `W_a`
bound its writes; require a frame property outside `W_a`. Include support and
absence tests, histories, grammar, configuration and allocation state where
consumed. Assume both actions are admitted initially, their read/write bounds
remain valid after either action, and

```text
W_a intersect (R_b union W_b) = empty,
W_b intersect (R_a union W_a) = empty.
```

Then both orders remain admitted and `T_b(T_a(s))=T_a(T_b(s))`. Proof: each
action leaves the other's complete guard and proposal inputs unchanged; their
disjoint write patches agree in either order, and the frame property preserves
everything else. This is a sufficient criterion, not a necessary one: overlapping
actions can commute for other proved reasons. Dynamic support or allocation
needs stable declared addresses, reserved fresh identities or a separate proof
up to consistent renaming. It cannot silently use the fixed-domain argument.

For runtime equality, include RNG progress, refresh callbacks, audit streams
and lifecycle effects. Ordered logs usually break full-state commutation. A
weaker structural projection requires an explicit equivalence and proof that
discarded effects cannot change the compared continuation. Reuse the existing
[conservative stage resources](../src/tnfr/operators/stage_contracts.py);
their abstract unions are not concrete independence certificates. No new
scheduler, target reordering or selective cache invalidation is authorized by
this criterion alone.

**Distinct targets do not imply that criterion.** In the ideal, unclipped scalar
Reception restriction, suppose both own contexts admit their actions, mixing
is 1/2, and two connected targets initially have forms `(1/4,3/4)`. A common
pre-state update gives `(1/2,1/2)`. Updating the first then the second gives
`(1/2,5/8)`; reversing the order gives `(3/8,1/2)`. Each target writes its own
form but reads the other. This exact arithmetic is a form-channel counterexample
to schedule equivalence, not a new executed trajectory, a complete-state
comparison or a proof of global nonconfluence. The existing
[snapshot stages](../src/tnfr/operators/network_stage.py) deliberately define
their own proposal/merge semantics. Parallel derivation does not justify
changing a sequential word or mixing different operator kinds in one stage.

Support guards can conflict as well: if two targets request the same isolated
potential partner, a successful first UM makes its degree nonzero. The second
request then fails the existing potential-pair guard. This conditional
produce/forbid example needs no edge deletion and shows why a name-only
operator-pair table cannot decide independence.

**An identity must retain the admitted environment.** Let `Q` satisfy the
existing partial-action criterion for a family `A`. Let `I_u` describe each
allowed boundary intervention, including its admission, state changes and
observable outcome. If the same criterion holds for every `I_u`, and the
compared outputs of both families also factor through `Q`, and the domain
is closed under the compared successors, induction on words in
`A union {I_u}` proves equal projected outcomes and admission at every prefix.
This is a corollary of the existing theorem, not an independent universal
bisimulation claim. Conversely, disagreement for one allowed boundary action
already prevents this contextual replacement. Autonomous timed replacement
additionally retains the law selecting actions and times.

Consequently, a proposed collective NFR must declare its ports and allowed
contexts. Equal current EPI, internal shape or one isolated response is
insufficient. The actual [birth/contact discriminator](THOL_PRESSURE_FEEDBACK.md#thol-acquired-birth-participation)
already shows that ancestry without realized input cannot substitute for
contact. The [retained-member interface](#retained-member-scale-actions)
preserves information for subsequent actions; it does not establish a smaller
state or invariance under arbitrary new environments.

<a id="operator-access-separation"></a>
**Hierarchy, access and transport have distinct roles.** Denote declared
ancestry by `H`, realized support by `E`, and the candidate
relation available to an occurrence policy by `A`. This notation defines no
new engine storage. Retained collective membership is another association;
it need not be exclusive, and ancestry need not mean spatial containment.
Actual THOL can extend `H` while leaving the child isolated
in `E`. A policy may consume the returned birth association or live hierarchy
as evidence for `A`; that is an explicit access premise, not an automatic UM
edge or a requirement that coupling occurs. UM must still pass its actual
grammar, U3, compatibility, state and successor checks. EN/RA continue to use
their declared realized support.

There is a precise information obstruction to a transport-only policy. If its
entire observation omits external node identities, birth/ancestry associations
and global candidate enumeration, adding or permuting an isolated outside
child leaves that observation unchanged. A deterministic function of it cannot
distinguish those possibilities; it cannot uniquely return the actual unseen
child in a way equivariant under such permutations. This does not apply when
the parent's `sub_nodes`, hierarchy or returned birth association is consumed:
those supply additional information. The engine already retains such data.
The missing implication is a justified access policy, not a lack of a possible
representation or a need to scan every graph node.

Candidate discovery can therefore be studied independently of physical distance
or a global word: declare the available association, re-admit its current
endpoints, and distinguish no candidate, ambiguous access, rejected UM and
executed UM. Equal access must not be confused with equal affinity or priority.
Node labels and insertion order cannot select among equivalent candidates
without an explicit tie-breaking premise. A fixed attempt clock still remains
a supplied occurrence policy even if the request is generated from NFR state.

**Reachability does not fix timing.** For an auxiliary finite-state continuous-time
Markov jump model with nonnegative rates `lambda_a(s)` and finite total rates,
scaling all rates by `c>0` preserves enabled transitions and relative choice
probabilities, while scaling holding times by `1/c` at nonabsorbing states.
Absorbing states remain absorbing. Thus one untimed language admits different
clocks; no stochastic TNFR law is imported. A timed quotient cannot merely
forget multiplicity: if `m` distinct matches each have rate `lambda` and lead
to the same different observed successor class, they contribute `m*lambda`
from that fine source to the observed successor class. This alone establishes
no Markov quotient. Within-class event intensity is not a visible
quotient jump rate. Match symmetries and indistinguishability must be fixed before
counting. A target/partner UM request is oriented even on undirected support;
swapping its roles need not be the same action. Discrete generations, operator
steps and elapsed structural time retain their existing separate contracts.

These deductions support guarded operator composition, explicit candidate
access and context-sensitive identity. They do not choose a pressure law,
inheritance coefficient, occurrence rate, universal normal form or collective
scale. Complete expressive sufficiency would require a target transformation
class, an equivalence and a constructive generation theorem under live grammar.

<a id="operator-reachable-image"></a>
### 2.5 Reachable image of a structural action

More constituents can support new collective behavior without adding new
independent preparation variables. To distinguish these claims, fix a source
chart `U` open in `R^d`, a discrete branch, parameters, event times and inputs.
For an ideal real `C^1` reset `J:U -> R^m`, `rank DJ <= d`. Constant rank `r`
on a neighborhood gives a local `r`-dimensional image by the constant-rank
theorem; rank at one point does not identify the whole image.

If `m>d`, the image has zero ambient m-dimensional measure. Indeed, on a compact
chart neighborhood J is Lipschitz. A mesh of source cubes of side epsilon uses
`O(epsilon^-d)` cubes, each with image diameter `O(epsilon)`, giving total
ambient covering volume `O(epsilon^(m-d)) -> 0`. A countable cover of U gives
the claim, hence no open ambient output set. This does not prohibit birth,
new response functions or collective organization.

At fixed elapsed time, an independently justified `C^1` local flow `Phi_t`
is a local diffeomorphism. The chain rule gives
`rank D(Phi_t o J) = rank DJ`; a noninvertible regular continuation only gives
an upper bound. Independent input `u in R^k` augments the source bound to
`d+k`, without guaranteeing an increase. Variable elapsed time is another
parameter. No conclusion here concerns infinite-time closures, nonregular or
nonunique continuation, arbitrary input histories or infinite event cascades.
Clipping and branch boundaries need their own strata; this is not a dimension
claim about discrete binary64 states.

The [THOL birth instance](nodal/SINE_COLLECTIVE_HIERARCHY.md#thol-birth-reachable-image)
gives a three-parameter inherited child image, its actual associated histories
and fresh child grammar context. Its controls also show when neighbor state
must remain an argument. Continued coupling and organization are separate
operator obligations, not consequences of the image's dimension.

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

**Activation policy**: With strict preconditions enabled, public and staged AL
require signed EPI below `EPI_LATENT_MAX` and capacity at least
`VF_BASAL_THRESHOLD`; both defaults are `0.5`. These supplied thresholds do
not follow from the nodal equation. The separately exported legacy
`validate_emission` checks only `AL_MAX_EPI_FOR_EMISSION` (default `0.8`);
current public/staged AL does not call that legacy policy.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Emission amplitude $b$ | configured positive gain | Operational parameter; channel and direction are contractual |
| Strict activation ceiling / capacity floor | $0.5$ / $0.5$ by default | Separate configured admission policies |

**Properties**:
- **Irreversible**: Sets an immutable activation flag. Re-emission increments an activation counter but preserves the original timestamp.
- **Genealogical**: Maintains structural lineage tracking (origin timestamp, parent references, derived node list).
- **Latency-aware**: Detects and clears silence (SHA) latency state on reactivation.
- **All-target time basis**: One accepted AL stage assigns one common timestamp
  to all newly emitted targets; repeated targets retain their original origin.

**Grammar**: Generator (U1a).

**Contract**:
- Pre: When strict preconditions are enabled, signed scalar EPI is below
  `EPI_LATENT_MAX` and nonnegative capacity meets `VF_BASAL_THRESHOLD`.
  Consumed state and thresholds are admitted before conversion/comparison;
  invalid authoritative aliases cannot fall through. The nonnegative integer
  `MIN_NETWORK_DEGREE_COUPLING` controls an isolation warning, not rejection.
- Post: EPI does not decrease; $\nu_f$, phase and $\Delta\text{NFR}$ remain
  unchanged; the activation flag is set.

### 4.2 Transition (NAV)

**Physics**: Configured adjustments selected by latent/active/resonant state
labels. These operational labels do not establish dynamical attractors or
prove transitions between them.

**Public/staged transformation** (regime-dependent):

| Regime | $\nu_f$ change | $\theta$ shift | Handler-pressure retention |
|--------|---------------|----------------|--------------------------|
| Latent → Active | +20% | $+0.1$ | $0.7$ |
| Active → Active | configurable | $+0.2$ | $0.8$ |
| Resonant → Active | $-5\%$ | $+0.15$ | $0.9$ |

These retention factors multiply the low-level handler output, after its
configured pressure/capacity mixture and jitter, not the incoming stored
pressure alone. `NAV_RANDOM=False` selects signed deterministic jitter;
it does not remove a nonzero configured jitter amplitude. The numeric table
describes the standard policy values; represented arithmetic retains the
actual configured scalars.

Capacity multiplication admits the exact product of the represented operands
before returning binary64. A positive product rounded to zero rejects before
the public/staged or compatibility transition commits; an intentional zero
factor, zero initial capacity and representable subnormal result remain valid.
No positive floor or reactivation law is introduced by this admission rule.

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
- **Latency observation**: Compares current EPI with the preserved scalar and
  warns when absolute drift exceeds `0.01*abs(preserved_epi)`, including a
  zero preserved value. NAV does not use AL's separate initial-node tolerance
  or repair the drift. Successful execution clears its consumed latency record.
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

The [collective Transition contract](nodal/SINE_COLLECTIVE_TRANSITION.md)
applies this path to the marked contact after EN/IL/SHA. The latent flag
selects a different phase/capacity reset from an otherwise identical physical
state without that flag. Immediate sine-pressure refresh removes raw jitter
from the continuous state while retaining execution context. A valid NAV
word can leave the old acute sector despite passing finite U6; preservation
has its own explicit phase-margin condition.

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
  $alpha$ is explicit and has no graph-independent default. Every selected
  node must supply stored pressure and `dEPI`; missing inputs reject rather
  than certifying the target with assumed zeros.
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
  context. When enabled, the optional application precondition admits
  `REMESH_MIN_NODES` as a strict nonnegative integer (default 2), without
  Boolean, textual or fractional coercion. The network history map has its
  additional admission above.
- Post: The node-level call records an advisory. Explicit network mixing has
  the stated conditional bounds, not a universal nested-identity guarantee.

The [collective sine specialization](nodal/SINE_COLLECTIVE_RECURSIVITY.md)
retains associated histories, actual form-reset work and both continued rows.
The [structure and memory owner](REMESH_STRUCTURE_AND_MEMORY.md) separates the
historical cross-scale role from current temporal lags and local/collective
memory. Its [post-birth composition](REMESH_STRUCTURE_AND_MEMORY.md#remesh-postbirth-response)
uses actual THOL/UM, acquired sampling and advisory/map controls followed by
EN/RA; it does not complete structural recursion or create an inherited past.

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

**Model and proof owner**:
[REMESH fixed-delay models](REMESH_INFINITY_DERIVATION.md) separates the runtime
map, finite companion recurrence and cyclic filter, with their respective
stability, numerical and composition hypotheses.

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
`B_i=ensure_bepi(EPI_i)` be the accepted target operand. With
`source_nodes=None`, Reception forms the unweighted arithmetic mean of all
runtime incoming neighbours,

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
- **Weighted collective action**: The
  [marked-contact contract](nodal/SINE_COLLECTIVE_RECEPTION.md) retains the
  actual neighbor multiplicities and complete EN/IL/SHA context. Convex EN
  mixing need not reduce weighted form storage, including on that contact's
  twin stratum. In the selected sine law, fresh pressure is unchanged while
  the subsequent phase row and form acceleration can change.
- **Affine certificate boundary**: `certify_reception_epi_realization()` keeps
  the ideal real map, represented binary64 coefficient matrix and actual
  two-stage runtime value separate. It promotes the represented map only when
  the declared scalar chart and clipping conditions pass and its consensus-row
  identity is exact. Snapshot agreement with the runtime is diagnostic, not a
  global floating-point linearity proof. This existing certificate covers the
  all-input action. An explicit `source_nodes` request, even one listing every
  input, remains outside that promotion; stage evidence abstains rather than
  attaching the old certificate to a different requested action.
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
  admission ceilings. Admit finite signed scalar EPI, finite signed pressure
  and finite non-Boolean ceilings before conversion or comparison; a ceiling
  remains an upper bound, not an absolute-value test. Isolation and an empty
  detected-source set are admitted
  under the default all-input action rather than treated as hard failures;
  graph-backed execution with source tracking enabled emits one advisory for
  the latter. An explicit causal-source request has the separate nonempty
  source domain below; a diagnostic source list is not that request.
- Post: Neighbour EPI is integrated; stored pressure and change rate are
  unchanged, hence immediate operator-local $C(t)$ is unchanged.

<a id="reception-causal-sources"></a>
#### Explicit causal sources and selective reception

**Constitutive action decision.** Public Reception, its atomic stage and the
stage dispatcher accept `source_nodes` as either `None` or a nonempty tuple of
unique realized incoming sources. `None` retains the existing all-input action.
The explicit public request uses the single-target atomic stage; a multi-target
stage applies the same tuple to each target and re-admits its incoming arcs.
An explicit tuple determines which source forms enter the mean and which source
kinds can influence the kind proposal. Every selected source must exist, belong
to the receiver's current incoming neighborhood, and supply explicit admitted
scalar EPI. Supplied order is retained, including the existing first-source
tie rule for equal dominant magnitudes. An empty tuple rejects before writes;
it does not fall back to all neighbors or manufacture a self-source.

This is supplied source access within an actual relation, not an autonomous
compatibility filter. An outgoing-only arc, a distant detected emitter or a
merely potential UM partner does not become an input. Existing self-loop and
unique-neighbor conventions remain separate from source selection. The shared
[Reception snapshot](../src/tnfr/operators/_reception_kernel.py) validates the
selection against available incoming support and binds the actual selected
sources, values, kinds and selection origin at the pre-write boundary; the
blend, kind and observation must describe that same read. Source-ranking
telemetry does not decide it. The
[execution contract](../docs/API_CONTRACTS.md) owns request validation and
public/stage evidence, including the legacy certificate's abstention.

Let $N_i$ be the nonempty available incoming set and $S_i\subseteq N_i$ the
nonempty selected set. In the ideal real scalar chart, with fixed sources,
$0<m\leq1$ and inactive clipping, the selected action is

$$
J_{S_i}(x)_i=(1-m)x_i+m\mu_{S_i},\qquad
\mu_{S_i}=\frac{1}{|S_i|}\sum_{j\in S_i}x_j,
$$

with other form coordinates fixed. Its difference from the all-input action is

$$
\bigl(J_{S_i}(x)-J_{N_i}(x)\bigr)_i
=m(\mu_{S_i}-\mu_{N_i}).
$$

For a proper subset, this difference has coefficient $-m/|N_i|$ in any omitted
source coordinate on an open independent-form domain. The common target term
cancels, also when a retained self-loop makes that source the target itself.
Thus selection changes the map; it is not just a restriction of the old map's
admitted domain. With target $1/4$, sources $3/4,-1/4$ and $m=1/2$, all-input
reception gives $1/4$, while the two singleton selections give $1/2$ and zero.
The same selected-source association must govern semantic kind; an excluded
large donor cannot determine the received kind. Coincident outputs on one
snapshot do not establish equal actions or full identity preservation.

The [finite EN-to-RA control](../tests/operators/test_reception_source_integration.py)
executes selected or legacy Reception from matched prepared states, then actual
Resonance at another node reading the receiver. The changed received form
changes that later RA response while support stays fixed. Public and staged
EN paths give the same algebraic predictions. This establishes finite
participation under supplied requests, not autonomous compatibility, source
acquisition or continuing organization.

For fixed selection the ideal scalar action preserves constants and the
common convex EPI interval. It is covariant under affine form-chart changes
when rails are transformed consistently. These identities concern the ideal
map, separately from rounded coefficients, actual binary64 evaluation,
clipping and semantic-kind thresholds. The kind rule compares absolute form
magnitudes and need not be invariant under a form translation, even without
ties; scalar covariance is not a symmetry of the complete semantic action.
These identities do not establish passivity: on a
three-node star with degree weights $(2,1,1)$, target/leaf forms $(0,1,-1)$
and $m=1/2$, selecting the positive leaf gives $(1/2,1,-1)$. The squared
degree-weighted centered norm rises from $2$ to $9/4$, although every form
remains in $[-1,1]$. The pure-EPI pressure-defect identity above still applies
to the actual local form change on its original connected diffusion support;
excluding an EN input does not delete a transport edge or refresh pressure.

**Historical scope and open mechanism.** The retained May manuscript's EN
section (pp83–86; [source audit](UNIFIED_GRAMMAR_RULES.md#source-grammar-and-notation))
motivates filtering by the receiver's phase and compatibility in phase and
$\nu_f$. It gives no quantitative capacity test and does not require equal
capacities. Page 84's exact-phase pseudocode does not derive a robust circular
tolerance or a universal law. The current explicit-source action therefore
does not silently impose U3, phase alignment, a capacity ratio, prior AL
history or form similarity. Under held forms and selection its scalar response
is unchanged by phase/capacity changes; that boundary remains deliberate.

The bounded state-dependent family below supplies one compatibility and
selection/abstention policy. It does not select a unique law or identify an
observed oscillation period: the
[capacity and pulse distinction](NODAL_PARAMETER_FOUNDATIONS.md#34-derived-pulse-scales-and-observation-periods)
still applies. Autonomous attempt scheduling, retention of richer identity and
a complete historical EN realization remain open.

<a id="reception-compatibility-policy"></a>
#### A declared compatibility family and conditional occurrence

`ReceptionCompatibilityPolicy(max_phase_gap, max_relative_capacity_gap, mode)`
requires two supplied thresholds: $\delta\in[0,\pi]$ for circular phase
separation and $\eta\in[0,1]$ for relative capacity contrast. The
[shared compatibility owner](../src/tnfr/operators/_reception_compatibility.py)
admits explicit finite phase and nonnegative capacity for the target and every
realized incoming candidate. A source $j$ is compatible with receiver $i$ when

$$
|\operatorname{wrap}(\theta_j-\theta_i)|\leq\delta,
\qquad
|\nu_{f,j}-\nu_{f,i}|
\leq\eta\max(\nu_{f,i},\nu_{f,j}).
$$

This is an opt-in constitutive family, not U3 or a uniquely derived threshold.
Equality is included. The phase test reuses the shared circular-difference
implementation, with its numerical domain. The capacity inequality compares
exact rational products of admitted represented inputs, avoiding a rounded
ratio or epsilon denominator. Missing or malformed consumed state rejects;
an invalid candidate cannot be hidden by first failing its other comparison.
EPI is not a compatibility input. An actual selected EN action must still admit
its consumed forms, grammar and all other existing execution conditions.
An existing self-loop retains the incoming-source convention; its equal
phase/capacity pair is compatible. A nonempty selection alone therefore does
not prove that form arrived from a distinct NFR.

The capacity test explicitly admits $(0,0)$; a pair with exactly one zero is
admitted only at $\eta=1$. This states equality or allowed disparity of supplied
capacities, not the presence of an active emission. $\eta=0$ requires equality,
while $\eta=1$ permits any admitted capacity pair. A zero capacity freezes the
unforced continuous form row, not all possible hybrid EN jumps. Requiring
strictly positive receiving capacity would be a different additional premise.

The [conditional dispatch](../src/tnfr/operators/reception_selection.py) uses
one receiver and the incoming support order. Let $N_i$ be that candidate tuple
and $C_i$ its compatible subsequence. Its two modes distinguish the earlier
mathematical alternatives:

- `select` invokes EN with $C_i$ when it is nonempty; this generally changes
  the form/kind action relative to the all-input map.
- `require_all` invokes EN with $N_i$ only when $N_i$ is nonempty and every
  source is compatible; this restricts the all-input action's admitted domain.

`execute_eligible_reception_stage` returns compatibility evidence and
`stage_result=None` when the selected condition is unmet. No EN history,
accepted-stage result or refresh callback is fabricated by this abstention.
The empty neighborhood cannot pass through a vacuous all-source test. Once a
nonempty tuple is selected, canonical EN executes through its existing atomic
stage. Invalid selected EPI, grammar rejection, a failed precondition or glyph
substitution remains a rejection, not a disguised compatibility abstention.
The helper supplies conditional occurrence at an explicitly requested attempt;
it derives neither attempt times nor inter-attempt dynamics.

Compatibility evidence must retain associated target and **all candidate**
phase/capacity values, identities/order and policy, including rejected sources.
A source previously outside the threshold can become eligible without changing
any old selected value. Binding only the selected tuple would therefore lose
the cause of selection. Actual execution associates this decision with the
same pre-write read as the EN form/kind action. Source telemetry retains its
independent read domain. The existing all-input affine certificates remain
unavailable for policy-derived explicit-source actions, including permissive
thresholds that select every input.

**Invariances and alternatives.** In the ideal coordinates the predicate is
invariant under common phase rotation, exchange of the two phase/capacity
values and common positive capacity scaling. Incoming graph orientation does
not become reciprocal. Under a common clock change with positive instantaneous
rate $\alpha$, replacing every capacity by $\nu_f/\alpha$ preserves selection;
equivalence of a complete dynamics still requires transforming every row and
schedule. Independent node-wise capacity rescaling generally changes it.

For positive capacities let $d_{\max}=|\nu_i-\nu_j|/\max(\nu_i,\nu_j)$.
The alternative sum-normalized contrast satisfies
$d_{\Sigma}=d_{\max}/(2-d_{\max})$. A threshold $\eta$ therefore corresponds
to $\eta/(2-\eta)$; these are the same positive-domain threshold family in
different parameters. A logarithmic ratio is likewise monotone in this
contrast on positive capacities. An absolute capacity tolerance is another
possible model but carries units that must transform. None is selected by the
nodal product alone, and exact real invariance is separate from represented
input rounding and shared phase arithmetic.

**Selection boundaries.** Strict margins away from both thresholds make the
selected set locally constant. At a boundary the hybrid action need not be
continuous: with target form zero, one always-compatible zero-form source and
another unit-form source crossing the phase threshold, the selected EN output
changes between zero and $m/2$. Both selected sets are nonempty. The normalized
capacity contrast also has no continuous extension at $(0,0)$: along
$(\epsilon,\epsilon)$ it is zero and along $(\epsilon,0)$ it is one. The
division-free inequality is defined there, but for $\eta<1$ its accepted origin
is not robust to arbitrary small unequal perturbations. Smooth weights or
hysteresis would change the action or add memory; neither follows silently.

**Cross-operator participation.** A successful EN form reset itself preserves
phase, capacity and support. At held policy its compatible tuple is therefore
unchanged: EN alone cannot alternate this filter. This is selection invariance,
not an involution, continuing EN admissibility or a proof of stable identity.
IL's public phase adjustment, UM, RA and ZHIR can change phase eligibility;
SHA, VAL, NUL and NAV can change capacity eligibility under their own contracts.
UM can change realized access, while THOL's lineage alone supplies no source
arc. Delayed REMESH form mixing changes content but not this predicate when
support/phase/capacity are held; an actual support/identity replacement needs
new source admission. AL changes content, and OZ's primary pressure action
affects separate admission without itself changing this predicate.

The [kernel controls](../tests/operators/test_reception_compatibility_kernel.py)
separate policy boundaries, state association and arithmetic; the
[finite integration controls](../tests/operators/test_reception_compatibility_integration.py)
test actual occurrence/abstention and continued canonical participation.
These bounded supplied attempts do not establish autonomous source acquisition,
universal compatibility, continuing organization or physical identification.

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
Canonical retention admission is `0 <= IL_dnfr_factor < 1`; the identity
factor one is excluded by the shared factor contract.

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
subsequent flow or represented transcendental rounding. The
[maintained-contact IL foundation](nodal/SINE_COLLECTIVE_PRESSURE_HANDOFF.md)
separately accounts for the edge shared by its two simultaneous targets and
proves conditional nonincrease and geometric retention on that model's tube.
It does not substitute for actual word admission or a represented endpoint
check. Contracting stored
pressure with unchanged stored EPI rates cannot decrease the configured
structural $C$ read-out. An ensuing pressure refresh is a separate operation
and need not preserve that comparison. Telemetry retains structural $C$,
the separately named legacy pressure-dispersion statistic, pressure reduction
and phase-locking observations.

**Grammar**: Stabilizer (U2); Bifurcation Handler (U4a).

**Contract**:
- Pre: An existing target has admitted state, event parameters, selected
  admission policy and independent live grammar/context.
- Post: Stored $|\Delta\text{NFR}|$ is not increased at the contraction;
  EPI and capacity are preserved. No future-flow or autonomous-trigger claim.

The default optional strict runtime policy implements admission as signed
`EPI > min_epi` and `nu_f > min_vf`, with both default lower bounds zero.
This is stronger than finite signed-form admission. In particular, a negative
scalar remains valid form even when this policy rejects it; rejection does
not establish absence of structure. The
[reflected organized-source pair](nodal/ORGANIZED_SOURCE_PORT.md#organized-source-coherence-domain)
has a domain-preserving ideal IL phase map but cannot admit both opposite
forms under that default gate. Mathematical map compatibility, configured
readiness and grammar admission remain distinct.

Implementation: [shared IL kernel](../src/tnfr/operators/_coherence_stage_kernel.py),
[public lifecycle](../src/tnfr/operators/coherence.py);
[stage and signed-pressure controls](../tests/operators/test_coherence_jacobi_stage.py).

<a id="coherence-activity-admission"></a>
#### Signed-form event domain and activity policies

**Source audit.** The identified [May manuscript](UNIFIED_GRAMMAR_RULES.md#source-grammar-and-notation),
section 2.2.1, pages 86–89, describes Coherence as identity-preserving
reorganization. Page 86 proposes a regularizable oscillation, an input below
a coherence threshold and capacity to reorganize without dissonance. Page 87
holds capacity constant while a coherence matrix settles. These qualitative
conditions do not specify `EPI > 0`, `EPI != 0` or the engine's numerical
thresholds. They also do not prove that the implemented map preserves every
identity or selects its own occurrence. The earlier validator's attribution
of its positive-form inequality to that passage was unsupported.

**Domain derivation.** On the retained graph state the primary event is

$$J_f^\lambda(x,\nu,p,\theta,G)
  =(x,\nu,fp,T_\lambda(\theta;G),G).$$

Here `T` is the circular proposal above, including its isolated and
zero-resultant conventions. Neither changed channel consumes the sign or
magnitude of `x`, nor divides by `nu`. Thus finite signed scalar form,
nonnegative finite capacity, valid pressure, phase, support and event
parameters supply the primary map's domain; represented output arithmetic
still has to be admitted. Form reversal commutes with this primary map
when the other coordinates are held. This statement concerns the map, not
all telemetry, a continued law or a simultaneous-event storage theorem.

The [form foundation](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation)
already distinguishes zero scalar form from absence of the node. The
[parameter ledger](NODAL_PARAMETER_FOUNDATIONS.md#2-parameter-and-dependency-ledger)
likewise makes zero capacity freeze the unforced form row, not every possible
instantaneous jump. Neither premise justifies replacing the signed threshold
by a nonzero-magnitude test.

Two controls independent of the organized-source preparation distinguish
domain from effect. On a two-node edge, a target with `x=0`, `nu=0`, stored
`p=1`, `theta=0` and neighbor phase `pi/4` has `p'=f` and
`theta'=lambda*pi/4`; both changed channels are nontrivial when `lambda>0`.
The same formulas hold at either sign of form. Conversely positive form and
capacity with zero pressure and aligned phases give no primary change.
Positive form/capacity are therefore neither necessary for this map nor
sufficient for nontrivial action. These controls establish no physical
particle, autonomous trigger or general identity-maintenance theorem.

**Explicit compatibility contract.** Strict IL uses the shared
[admission owner](../src/tnfr/operators/preconditions/coherence.py).
`IL_PRECONDITIONS.activity_policy` selects:

- `signed_threshold` (unchanged default): `x > min_epi` and
  `nu > min_vf`, with both defaults zero. A finite negative `min_epi`
  retains its signed lower-bound meaning; `min_vf` is nonnegative.
- `event_domain`: admit valid signed form, including zero, and nonnegative
  capacity without additional activity thresholds. Supplying `min_epi` or
  `min_vf` with this policy rejects instead of silently ignoring them.

This additive choice changes admission, not the event map, grammar or
occurrence law. Existing configurations and the frozen default-policy pair
rejection retain their meaning. A new use of `event_domain` must record that
choice; it cannot relabel an earlier rejected experiment as successful.
Default/explicit threshold controls, direct helpers, readiness, public and
atomic consumers share normalized primitive admission. Boolean physical
scalars, nonfinite/unrepresentable values, invalid authoritative aliases and
nonuniform/complex BEPI reject before comparison or advisory publication.
Uniform-real BEPI retains its signed scalar meaning.

Readiness reports only the selected policy on admitted target state. It does
not check a complete neighbor proposal, live grammar, stage atomicity or
future stability. Its retained `epi_active` and `vf_active` keys mean that
the respective policy conditions pass, not existence or measured activity.
Zero form does not satisfy U1's nonzero-initial-form predicate: independent
initialization/history is still required. Zero stored pressure quiets only
the contraction channel; phase may still change. The capacity-zero boundary
concerns an instantaneous event, not a finite-duration reorganization law.
See the [activity and admission controls](../tests/operators/test_coherence_activity_policy.py)
and [execution contract](../docs/contracts/OPERATOR_EVENTS.md#coherence-activity-policy).

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

Signed scalar admission is shared with the form foundation: uniform-real
BEPI retains its sign, while complex/nonuniform form cannot substitute a
magnitude history. Nonzero primitive samples, times, secants or acceleration
lost to represented zero reject before telemetry or event writes. Invalid
authoritative timestamped evidence never falls back to a legacy channel.
This corrects earlier permissive materialization; ordinary finite signed
histories and the explicit short-history unavailability contract are retained.

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
  $\alpha$ are supplied. Every selected parent/child must also provide finite
  stored pressure and `dEPI`; missing channels reject instead of becoming
  zero. Birth initialization supplies no observed child rate. Requesting this
  assessment through THOL metrics therefore requires additional explicit
  evidence. THOL does not enforce a hidden collective threshold.
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
The [actual AL continuation](THOL_PRESSURE_FEEDBACK.md#thol-emission-history-continuation)
uses history produced by IL/OZ/AL to test signed THOL feedback and the next
admitted AL action. Its observation window and no-child domain remain explicit.
The [collective hierarchy contract](nodal/SINE_COLLECTIVE_HIERARCHY.md)
specializes these obligations to the selected sine law, including the
isolated child's unsupported connected-law domain and separately supplied
continuation alternatives.

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

**Bifurcation read-out**: The implementation compares an acceleration scalar
with the configured threshold $\tau$. Its compatibility reader returns zero
when temporal evidence is unavailable; use the typed
[`observe_structural_acceleration`](../src/tnfr/operators/nodal_equation.py)
observation to distinguish that case from a measured finite-difference zero.
A flag records the scalar comparison, not a bifurcation theorem for an
underlying continuous law. U4a separately requires handler context for the
trigger token.

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
  In the current implementation, `uniform` is a compatibility alias for
  `phase_weighted`: both use conductance and phase affinity. Only the
  `frequency_weighted` branch adds capacity weighting; these names do not
  identify three distinct propagation formulas.
- **Acceleration diagnostic**: Compares the finite-difference scalar with
  threshold $\tau$; observation availability remains a separate obligation.
- **Telemetry**: Records propagation events, affected nodes, and bifurcation flags.

**Grammar**: Destabilizer (U2); Bifurcation Trigger (U4a); Closure (U1b).

**Contract**:
- Pre: Consumed finite pressure and the active branch's admitted factors.
  Enabled strict preconditions additionally impose configured EPI, capacity,
  pressure and overload policies; these do not prove absence of a dynamical
  bifurcation or future instability.
  Admit authoritative pressure, strict thresholds, propagation primitives and
  active `OZ_SIGMA` before float conversion. Reject Boolean/nonfinite values
  and nonzero inputs lost during materialization. The EPI minimum stays signed;
  capacity/pressure bounds and the integer degree minimum are nonnegative.
  Inactive sigma is not consumed; finite nonpositive active sigma retains its
  no-op. Input admission does not certify all subsequent floating products.
- Post: The deterministic local branch increases pressure magnitude. Noise and
  incoming propagation require their separately retained observations. An
  acceleration-threshold flag is a configured diagnostic, not a bifurcation
  proof or an endogenous instruction to execute another operator.

Changing stored pressure alone leaves $E_D+\beta V$ unchanged. Recomputing
pressure from unchanged constitutive inputs can erase the event's pressure
write; persistent forcing or pressure memory must be supplied explicitly.
The [collective handoff foundation](nodal/SINE_COLLECTIVE_PRESSURE_HANDOFF.md)
formalizes this continuous-state equivalence while retaining the changed
execution context. Reciprocal default IL/OZ pressure gains do not make the
complete actions inverses.
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
The [scale configuration contract](../docs/contracts/OPERATOR_EVENTS.md#scale-stage-configuration)
requires Boolean flags, admitted active clipping and finite strict policy
thresholds. Configured soft steepness reaches both direct and atomic proposals.
The [complete contact action](nodal/SINE_FORM_CAPACITY_ACTION.md) derives
the refreshed rows, event work, changed means and common-clock boundary;
a local scale also needs the retained form origin for collective descent.

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
The [retained-member scale interface](#retained-member-scale-actions) applies
the same actual VAL/NUL maps to declared members without replacing their
state or ports by a scalar aggregate.

<a id="derived-expansion-member"></a>
### 7.3 Explicit family with one derived Expansion member

The historical VAL role includes compatible derived forms retaining a core
(TNFR.pdf pp.105–108); its clone/link example does not determine a complete
state, context or port law. The
[derived-member action](../src/tnfr/operators/derived_expansion.py) implements
one declared constitutive family beyond the ordinary scalar map. It is an
explicit opt-in action associated with VAL, not a fourteenth token, a unique
canonical proliferation law or an autonomous selection mechanism.

`DerivedExpansionPolicy` requires `source_order="pre_val"` or `"post_val"`
and explicit `form_ratio=alpha`, `capacity_ratio=beta`, both in `(0,1]`.
The bounded positive ratios are this family's restriction, not a necessity
for every NFR or every possible Expansion. A current singleton retained-source
record and a supplied fresh child ID identify the parent and successor.
`execute_derived_expansion_stage` executes the existing canonical scalar VAL
on that parent with its own grammar, preconditions and exact-stage admission.
No replacement glyph or inherited sequence context can stand in for VAL.

Write its actual primitive source and result as

```
S^- = (x,nu,theta,p)
S^+ = (F(x),f*nu,theta,p),  f>1.
```

Here F is the configured VAL form map, including projection or the identity
when edge awareness is disabled. Both inherited coordinates come from the
same requested before/after boundary. The isolated child is initialized by

```
x_child = alpha*x_source
nu_child = beta*nu_source
theta_child = theta_source
p_child = 0.
```

Signed form and semantic kind are inherited. The proposed child form must
lie inside configured EPI bounds; rejection preserves the stated formula
instead of clipping it. Nonfinite and lost nonzero products reject, while
zero source form or capacity may legitimately produce zero. Pressure zero
is a declared initial condition, not a conserved-resource result. The child
has its own fresh context and one initial legacy EPI sample, without a
timestamped past, measured acceleration or an independently executed VAL.
Its `derived_from` association does not create THOL hierarchy or authenticate
an event history.

Only [fresh-node materialization](../src/tnfr/operators/_birth_materialization.py)
is shared with THOL. THOL's acceleration, metabolism, nonnegative child-form
clamp, hierarchy and occurrence remain its own mechanisms. Reusing that
numerical proposal here would lose signed derived forms. The parent's identity
and old ports are retained while its actual VAL changes state. No edge or
external-port copy is part of birth. A later UM must separately admit and
create a relation; subsequent EN/RA still read actual directed support.
Failure within the compound parent-action/birth transaction restores the
graph-owned source. Failure of a later separate UM does not undo a previously
successful birth.

#### Ordering, response and parameter identification

At fixed source and ratios, the final parent is identical for both orders,
but the child differences, post minus pre, are

```
Delta x_child = alpha*(F(x)-x)
Delta nu_child = beta*(f-1)*nu.
```

With inactive clipping and `F(x)=f*x`, the form difference is
`alpha*(f-1)*x`, retaining its sign. With no form change, forms can agree
while capacities differ for positive nu. Successor admission can itself
distinguish the orders; a response comparison requires both to be admitted.

If both later UM requests have k identical compatible old neighbors, their
phase proposals agree. Optional target capacity blending with gain gamma
instead gives target-capacity difference `gamma*beta*(f-1)*nu/(k+1)`.
Their new edge scores can differ through form; equal support does not imply
equal weighted-flow continuation. On the undirected one-old-peer domain,
actual parent EN with factor m followed by peer RA with factor r, same-kind
form and inactive clipping, gives peer-response difference
`r*m*alpha*(F(x)-x)/2`. This is a finite operator response, not an observed
oscillation-frequency or physical identification.

Primitive state and EN/RA responses do not identify ordering if ratios can be
freely refitted. In the linear branch, post inheritance with
`(alpha/f,beta/f)` produces the same ideal primitive child coordinates as pre
inheritance with `(alpha,beta)`; the declared policy/provenance records remain
distinct. Both satisfy the
bounded ratio domain; the reverse conversion requires the corresponding
multiplied ratios to remain at most one. Saturation can make the conversion
state dependent, and the no-form-change branch only needs the capacity-ratio
conversion. No historical or current premise uniquely selects one order.

Phase inheritance commutes with a common circular phase rotation in the ideal
model. Positive ratios preserve form sign and capacity nonnegativity; a
common form translation is not automatically a symmetry. Neither total EPI
nor total capacity is assumed conserved: their coordinate-sum changes at this
event are `(F(x)-x)+alpha*x_source` and `(f-1)*nu+beta*nu_source`. These are
bookkeeping identities, not universal resources or physical energies. The
event changes arity and is not a finite-duration unforced nodal flow. A later
complete law must admit the new state and supply its own balance and identity
conditions. Fixed inheritance adds constrained coordinates, not by itself
new independent preparation freedoms.

The [family contract controls](../tests/operators/test_derived_expansion.py)
check policy, identity, source admission, unchanged ordinary VAL and atomic
failure boundaries. The [actual continuation controls](../tests/operators/test_derived_expansion_continuation.py)
compare both orders, signed and degenerate sources, reparameterized primitive
agreement, and UM followed by EN/RA. These are supplied finite interventions;
neither spontaneous source acquisition nor persistent organization follows.

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
- **Phase verification mandatory**: The ordinary action requires an existing
  compatible neighbor. The explicit potential-pair action below supplies its
  isolated partner alongside compatible old neighbors without changing U3.
  Ordinary candidate search excludes snapshot-incompatible candidates and
  omits a new candidate outside final U3. A selected old neighbor or explicitly
  requested partner outside final U3 instead rejects the common stage.
  These gates supply no wave-amplitude or
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
  neighbor, or the explicit isolated-partner request below. Enabled strict
  preconditions additionally check configured EPI
  and capacity thresholds. Link candidates retain their separate score and
  post-merge phase admission.
- Post: Configured circular phase proposals; EPI preserved. Optional functional
  proposals may add admitted links. Neither global phase-spread decrease nor
  the creation of at least one edge follows from an ordinary invocation. The
  explicit creating request instead requires its one final edge or rejects.

<a id="coupling-potential-pair"></a>
#### Creating a relation to an explicit isolated partner

The public call `Coupling()(G, i, potential_partner=j)` and shared UM stage
admit one target i and one supplied, already existing partner j with total
graph degree zero. The target may have realized neighbors. Directed incoming
edges, self-loops and zero-weight edges also violate j's isolation premise;
a partner already connected elsewhere remains outside this domain. Potential
access is consumed request state; no temporary edge is inserted to pass a
gate. This declared constitutive extension serves UM's historical creating
and iterated-union roles (TNFR.pdf pp.93–97); it does not derive autonomous
partner discovery or merge constituent identities.

The [shared pair reader](../src/tnfr/operators/_coupling_potential.py) admits
both endpoint signed forms, nonnegative capacities and authoritative phases.
It checks j against the target separately: another compatible neighbor cannot
hide an incompatible requested partner. It also reads the target's actual
outgoing neighbors in their existing order, admits their phases and retains
the ordinary U3-compatible tuple K. The participant tuple is K followed
by j, including when K is empty. Functional links must be enabled; the
configured activity policy retains its current admission. The
[existing proposal kernel](../src/tnfr/operators/_coupling_stage_kernel.py)
uses that augmented tuple for the same circular map, optional target capacity
synchronization and stored-pressure relief. Bidirectional phase movement
includes the target in the circular mean and moves the participating old
neighbors as well as j. All forms and old edge records/weights remain
unchanged. A phase write on another participant does not execute UM on it
or append to its operator history.

For the single-target bidirectional action, lift the target to zero and all
participant phases `phi_k` into `[-L,L]`, where `L <= pi/2`. The resultant
`1 + sum_k exp(i*phi_k)` has real part at least one, so its argument M exists
in the same sector. With `lambda in [0,1]`, the consistent lifted proposals
are `theta_i'=lambda*M` and `phi_k'=(1-lambda)*phi_k+lambda*M`. Thus every
selected target-participant gap contracts by `1-lambda` in exact arithmetic.
The former two-isolate case remains a subdomain: both unidirectional and
bidirectional maps leave its relative gap `(1-lambda)*delta`.

Unidirectional movement on the augmented domain has no such general guarantee.
For target and requested child at zero, old neighbors at `(-L,L,L)` and
`L=pi/4`, the partner-inclusive mean has positive argument M. Moving only the
target by `lambda*M>0` increases its gap from the neighbor at `-L` beyond L.
Final U3 must reject that entire action. The represented stage rechecks every
selected target-participant relation; the real proof does not bound all
floating-point trigonometric errors. Edges from a moved old neighbor to other
nodes, global phase spread and overlapping multi-target requests are outside
this guarantee.

The supplied partner bypasses candidate sampling, candidate-count policy and
random draws; it does not bypass the configured form/Si/phase affinity score.
A failed final affinity rejects the entire creating action. The final edge's
conductance is the existing UM score, not an assumed unit coupling. A directed
graph receives i→j only. A zero score can create support if the supplied
threshold admits zero; support and nonzero conductance remain distinct.

Public creation delegates to the existing atomic stage. One actual UM request
must pass grammar, whose live U3 read includes the same supplied partner even
if no old neighbor is compatible. Other grammar conditions are retained;
request-free prospective selection keeps its ordinary domain. A fallback
glyph cannot count as successful creation. The
stage commits phase, optional channels, link and target context together, with
its existing graph-owned rollback on failure. Optional refresh is an explicit
callback, not an inferred continuation law. The existing last-stage telemetry
records the supplied pair and its request origin; it is not an authenticated
event history. Customized execution subclasses are outside this bounded path.

Omitting the partner retains the ordinary API and its numerical policy. The
raw glyph API, graphless protocol path and all-node word runner do not infer
potential access. RA still consumes realized outgoing neighbors: RA on i can
read j through i→j, while RA on j still requires its own outgoing support.
Reception likewise uses its actual direction
convention. Subsequent actions retain their own state/history admission. The
[pair controls](../tests/operators/test_coupling_potential_relation.py) and
[pure proposal controls](../tests/operators/test_coupling_potential_kernel.py)
retain the isolated subdomain; the [connected-target controls](../tests/operators/test_connected_coupling_kernel.py)
exercise augmented selection, represented final admission and old-weight
preservation. Creating
this relation enables interaction; it proves neither persistent organization
nor a collective identity or autonomous occurrence.

The separate [disjoint-component request](COUPLING_COMPOSITION.md#connected-coupling-request)
admits a connected partner through the same UM map, protecting all old edges
incident to phase participants. Its [collateral counterexample](COUPLING_COMPOSITION.md#connected-coupling-collateral-boundary)
explains why removing the isolation check alone is insufficient. The default
isolated domain above keeps its original boundary.

<a id="coupling-birth-association-access"></a>
#### Deriving a potential request from live birth associations

The [access consumer](../src/tnfr/operators/birth_access_coupling.py) implements
the declared policy `unique_isolated_birth_child`. At a supplied parent and
occasion, let H(i) be its consistent direct `sub_nodes` list and define
`A(i) = {j in H(i): degree(j) = 0}`. Degree means total support degree,
including incoming edges in directed graphs. Every association is validated
before filtering: duplicate, missing, self or contradictory child references
are state errors. A present parent entry in the redundant hierarchy index must
agree; missing entries retain THOL's legacy node-local authority. Historical
`sub_epis` values are not a current access authority or proof of executed birth.

If A(i) is empty or has more than one member, the consumer abstains without an
operator event. Only a singleton supplies the partner to the existing strict
atomic UM stage. Numerical compatibility does not resolve ambiguous access;
a unique incompatible child raises through UM rather than causing a fallback
search. Access, actual admission and committed transport remain separate.
The passive report cannot activate execution: every invocation reads current
state. Enabled canonical grammar, rejection mode `raise` and requested strict
preconditions are mandatory, including before abstention.

Under a consistent identity relabeling, H and degree transport with the graph,
so A transports and a unique selection is equivariant. Order supplies no
winner for a larger set. This is a conditional property of the declared
relation, not authentication of genealogy or an autonomous occurrence theorem.
Actual UM effects equal an explicit request for the same child under the same
admitted state and options. [Acquired-source controls](THOL_PRESSURE_FEEDBACK.md#thol-state-derived-access)
connect that equivalence to subsequent EN/RA participation.

Successful contact removes the child's isolation. A later declared disconnection
can restore access: no consumed-birth receipt or age cutoff is introduced.
Discovery uses direct associations, not global candidate enumeration; the
existing transaction can still traverse the graph. Parent choice, invocation
timing, ambiguous competition and persistent organization remain unresolved.

<a id="operator-birth-contact-composition"></a>
#### Finite birth, contact and response through actual operators

The [birth/contact controls](../tests/operators/test_operator_birth_contact_chain.py)
compose actual THOL, explicit-pair UM, RA, IL and SHA on one initially isolated
prepared parent. Its associated temporal samples and IL/OZ context are supplied
admission evidence, not an acquired trajectory. THOL creates the child recorded
in its own birth association; both endpoints are still isolated before UM.
The caller supplies that child as the potential partner, then RA on the parent
reads the child's form over the newly realized relation.

With parent form x, the admitted nonmetabolic child form `s*x` and supplied RA
mixing coefficient r, the ideal parent response is `(1-r+r*s)*x`. The finite
control checks the represented result against an independent affine expression.
The same preparation and birth without UM leaves RA inadmissible for lack of
support. This establishes a concrete causal role for the new relation in this
supplied sequence; it does not establish spontaneous selection or maintenance.

Raw public/stage RA does not own temporal sampling. Its form jump leaves the
old parent endpoint inconsistent with the live EPI, so that old history cannot
authorize a new measured acceleration. The control explicitly uses the existing
runtime boundary recorder at the unchanged declared clock. That observation
step retains one current parent sample and the child's own first timestamped
sample; neither has a measured acceleration. No elapsed flow, inherited parent
history or artificial child action is inserted. IL/SHA then retain form and
support while applying their actual coherence/lifecycle effects. The child
does not inherit the parent's operator history or stabilization credit.

This is a finite composition and a history-ownership control on public and
shared-stage paths. It is not a new all-operator itinerary, an autonomous
organization theorem, or evidence about the connected-parent domain below.

<a id="coupling-connected-parent"></a>
#### Connected-parent birth, contact and continued participation

The [connected birth/contact controls](../tests/operators/test_connected_parent_birth_contact.py)
prepare a parent with an old realized peer, associated temporal samples and
IL/OZ context. Actual THOL creates an isolated child; its recorded identity
supplies the requested UM partner. No ancestry metadata is used as an extra
UM gate. The common proposal now includes both the old compatible peer and
the child, retaining all old edge records, parent form history and the child's
fresh context. Public, atomic and dispatcher controls cover directed and
undirected support, including parallel edges with the existing unique-neighbor
convention.

Ordinary UM can create that same child edge through its candidate policy.
It nevertheless reads only the old phase/capacity neighborhood for its reset;
the explicit request makes the child participate in those reads. For target
and child phase zero and old-peer phase `alpha in (0,L]`, the ideal
bidirectional means are respectively `alpha/2` and `arg(2+exp(i*alpha))`.
The latter lies strictly between zero and `alpha/2`. The finite controls
distinguish the actual phase and capacity results rather than treating equal
edge sets as equal actions.

Subsequent actual EN on the parent and RA on its old peer distinguish the
connected child from the matched no-UM control: the child's form enters the
unweighted donor mean only after contact. The no-UM case still has the old
peer, so it remains a valid response control. Directed controls separately
check incoming EN versus outgoing RA; creation does not silently add the
reverse arc. Neither UM participation nor neighboring EN/RA grants the child
an operator event, inherited grammar or measured acceleration. Invocation
times and source samples remain supplied; this finite composition establishes
causal participation without deriving source acquisition, autonomous partner
access, continued stability or a collective NFR identity.

The [acquired-history extension](THOL_PRESSURE_FEEDBACK.md#thol-acquired-birth-participation)
instead produces the source samples and grammar through actual IL/OZ/AL,
then invokes eligible THOL and the existing UM/EN/RA paths. A third branch
retains UM but excludes the child from EN's inputs, isolating its causal
contribution. A separate branch executes the child's own EN/RA word. These
finite controls retain supplied timing, access and inheritance coefficients;
they do not establish autonomous occurrence or universal operator sufficiency.

Link existence and stored conductance are separate. UM assigns its calculated
compatibility score to a newly created edge; this path does not continuously
evolve that weight or rewrite existing-edge weights. A present zero-weight
edge can still enter support-based phase neighborhoods. The
[collective Coupling result](nodal/SINE_PAIR_COUPLING.md) distinguishes a valid
AL/UM/SHA word from admission to a subsequent unit-conductance sine law:
generic outputs leave that law's domain, while a synchronized selected pair
admits a unit internal edge with rederived degree-five dynamics. A separately
admitted conservative conductance/support completion accepts the generated
weights and retains identity on the declared source. Its complete law and
held-weight premise are explicit; UM is not changed to fit the old restriction.

The [complete retained-rim contact audit](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#organized-contact-coupling-realization)
gives a complementary obstruction: retaining nine strict-acute winding-one
C5 rims forces at least nine of 45 requested spokes beyond the exact U3
limit. This applies at the common final phases. Concrete represented proposals
require their own numerical association; the detached controls certify those
calls without asserting a universal trigonometric error bound. A supplied
attachment theorem therefore need not be realizable by this registered action.

The [marked-port admission](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#organized-contact-marked-coupling-port)
instead retains one visible candidate on that acquired source. Its complete
observation history supplies a sharper form bound. With explicitly equal
stored sense indices and admitted runtime context, the exact map proposes
one link while retaining all old cycles; a different sense-index completion
of the same primary source omits it. Strict activity validation rejects the
small-form donor separately. Detached finite controls check their actual
represented proposals. These are conditional admission results, not an
occurrence law or admission to a subsequent continuous flow.

An explicitly supplied weighted continuation then gives a
[finite matched-port exchange](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#organized-contact-marked-port-response).
Comparing the same phase reset with and without the actual new edge separates
the connection's donor-mean response from the reset itself. The declared
whole-window bounds retain the old cycles and resolve the sign after four
reading errors. Both reset and attachment inject storage; their work stays
separate from subsequent continuous loss. This successor-law result does
not turn the operator into an autonomous occurrence or maintenance mechanism.

The [general finite bridge interface](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#sine-finite-bridge-interface)
extends the response calculation across finite internal supports under that
same held unit-capacity law. Raw port intervals, global form and weighted-degree
bounds, donor count, horizon and four reading-error radii supply a sufficient
signed contrast; associated old-chart margins supply a separate retention
decision. The [shared preparation reader](../src/tnfr/physics/_sine_preparation.py)
rebuilds the bounds without authenticating a source, graph or operator event.
The [positive-capacity extension](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#sine-capacity-bridge-interface)
admits a separately declared shared-capacity successor law, including endpoints
that may follow SHA, NUL or VAL. Its arithmetic contrast retains internal motion
in both matched branches; the inverse-capacity observation has its own mass
normalization and reading errors. This predicts a conditional continuation
after an independently admitted endpoint. It does not select a capacity law,
execute those actions or carry unit-capacity mean cancellation over unchanged.
The [matched-state refinement](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#sine-matched-bridge-interface)
retains common initialization to bound the interaction itself. Its exact
deterministic response enclosure vanishes with bridge conductance in this
weighted-both-rows law; each reading keeps its own observation-error floor.
This continuity does not transfer to native bare-neighbor phase support,
select a Coupling occurrence or establish a new joint identity.
For [two simultaneous contacts](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/theory/nodal/SINE_CONTACT_COMPOSITION.md#sine-contact-composition),
a full-state mixed sensitivity bounds the error of adding their separate
responses by a conductance-product allowance. Exact linear controls show
that shared and separate endpoints can give different joint responses despite
identical separate finite readings. Old internal paths can also couple distinct
ports. This is a conditional prediction contract;
the contacts' admission, occurrence and resulting joint identity remain
separate obligations. A nonzero composition error does not identify a nonlinear
constitutive law, because changing contacts also changes a linear generator.
The [support-associated refinement](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/theory/nodal/SINE_CONTACT_COMPOSITION.md#sine-contact-influence)
retains actual conductances, positive capacities, marked endpoints and the
observed constituent. Separate invariant contact blocks have exactly zero
composition error; a connecting path enters the finite allowance with its
actual weights. A positive allowance alone establishes neither a nonzero
response nor an autonomous law for the constituent.
The [retained-boundary predictor](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/theory/nodal/SINE_CONTACT_COMPOSITION.md#sine-retained-boundary-prediction)
evolves a proper subset of the form and phase coordinates while bounding
omitted initialization, hidden drift and later boundary feedback. The same
constituent observation must lie inside the retained region. This is an
approximate continuous prediction with a stated error, not an admission
state for Coupling or another registered action. Event occurrence, consumed
history and successor-law obligations still require their own retained data.

For [two separately maintained organizations](nodal/SINE_ORGANIZATION_CONTACT.md),
the same unrestricted candidate policy creates a multi-edge contact, including
neighbors of the aligned receiver pair. The actual weighted boundary must stay
associated with its constituent state. Initial geometric retention and later
departure from the receiver's original identity tube follow from the declared
continuous law; the event does not by itself certify a new joint organization.

The [three-candidate proximity comparison](nodal/SINE_PROXIMITY_CONTACT.md)
keeps that source and word, changing only the existing candidate-count policy.
It creates four cross edges and the internal edge; a separate full-source
energy proof retains a common identity including relative constituent origins.
Candidate selection precedes final link checks, and ties within the separated
three-member group do not change the selected set. Neither policy defines a
universal coordination number or an autonomous cause of contact. The proof
retains actual boundary weights rather than imposing pair symmetry.

Its [relative-registration extension](nodal/SINE_PROXIMITY_CONTACT.md#sine-registration-protocol)
admits independent bounded form and phase origins while retaining the same
full internal source and actual word. A's old-neighbor reset is unchanged;
the shifted receiver changes cross scores, attachment work and conserved
mean values. Recompute those quantities from the actual endpoint. The
explicit maintenance domain does not select contact or require exact
initial origin matching.

The [finite exchange theorem](nodal/SINE_PROXIMITY_CONTACT.md#sine-contact-finite-exchange)
then compares matched opposite phase preparations under their own actual
weighted laws. Exact cut-current balances and a full-law remainder establish
a signed form contrast throughout a finite window; a separately admitted
disconnected model has zero paired contrast. This is continuous response after
the supplied UM word, not a repeated operator, a closed two-coordinate
oscillator or evidence of autonomous contact occurrence.

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
0<r\leq1,\qquad a>0,\qquad0<c\leq1,
$$

finite scalar or uniform-real BEPI operands, a finite nondecreasing proposed
frequency, and the identity gate below.

The EPI proposal uses the shared operator-assignment boundary adapter. Admit
raw finite real rails before conversion, including rejection of Boolean rails
and nonzero underflow. This adapter retains the operator path's existing
hard-mode fallback and default soft stiffness; it does not consume the
integrator's configurable `CLIP_SOFT_K`. Detached realization readers use the
same rails and mode, without promoting soft proposals to affine certificates.

**Key constants and domains**:

| Quantity | Default | Runtime domain / status |
|----------|---------|-------------------------|
| EPI mix $r$ | $1/(2\pi)\approx0.1592$ | `(0,1]`; convex propagation |
| Frequency amplification $a$ | $1/(8\pi)\approx0.0398$ | `(0,+infinity)`; nondecreasing capacity |
| Phase coupling $c$ | $1/(4\pi)\approx0.0796$ | `(0,1]`; shortest-arc interpolation |
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
- The [native split-phase chart](NATIVE_COLLECTIVE_CALCULUS.md#native-split-phase-resonance)
  retains separate U3 subsets and proposals for each target constituent.
  Independent sign/kind admission permits an increase in internal phase
  separation; a paired target's common-subset contraction is not universal.
- The [aggregate inheritance domain](NATIVE_AGGREGATE_OPERATORS.md#native-resonance-inheritance-domain)
  proves one same-factor coarse RA for complete twin actions with paired
  phases, common strict sign/kind and duplicated consumed grammar. Unequal
  forms and capacities are allowed; the separate capacity obstruction rules
  out a universal mean-capacity claim. Event inheritance does not close flow.

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

**Default observed temporal evidence**: the non-disableable runtime trigger instead
uses the last two finite scalar EPI samples to form the signed secant

$$
\widehat{\partial_t\mathrm{EPI}}_{\mathrm{obs}}
=\frac{\mathrm{EPI}_{k}-\mathrm{EPI}_{k-1}}{t_k-t_{k-1}}.
$$

Timestamped `epi_time_history` declares evidence on a supplied time basis.
It requires finite samples, a finite strictly positive interval, and a final sample that represents
the current EPI endpoint (exactly under the runtime default; the pure
certificate can declare a finite nonnegative absolute endpoint tolerance).
When this channel is supplied it is authoritative: invalid or stale physical
evidence does not fall back to an untimestamped history.
The pure reader checks endpoint value, not an independently authoritative
live clock or source provenance. The observation/execution protocol must bind
that sample to the present structural time and intervening events; the
`physical_time_resolved` label alone supplies no laboratory calibration.
Shared represented-real admission rejects nonzero inputs lost to zero.
A nonzero nodal product lost to zero raises an input error; an underflowed
observed rate remains invalid evidence, not a zero-growth observation.

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

The separate opt-in [operator-growth consumer](NATIVE_OPERATOR_DYNAMICS.md#operator-hybrid-mutation-consumer)
executes an actual source schedule and immediately admits the same phase-only
ZHIR stage from its exact signed hybrid growth. It retains independent grammar,
active capacity and disjoint consumed windows without fabricating continuous
history. Its supplied trigger law does not change the default interfaces below
or establish autonomous timing, richer mutation or a unique canonical threshold.

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

The shared phase proposal admits raw phase and every active scalar before
binary64 conversion. Boolean/nonreal inputs and nonzero values lost to zero
reject; an invalid authoritative phase cannot fall through to another alias.
Direct graph dispatch checks phase before adapter caching. Signed-zero
pressure direction is preserved, and the fixed branch does not consume
dynamic proposal arguments. This strengthens input admission without changing
the phase map for valid values or waiving temporal/grammar requirements.

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
The [selected sine-law compatibility contract](nodal/SINE_COLLECTIVE_MUTATION_ADMISSION.md)
separates general continuous-rate possibility from a particular contact's
history, grammar and whole-interval restrictions. Its existence theorem does
not supply the temporal evidence or occurrence law for an executed event.

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
- Post: phase $\theta$ shifted; the semantic label `epi_kind` preserved;
  bifurcation potential flagged if $\partial^2\text{EPI}/\partial t^2>\tau$.
  After successful dispatch, `source_glyph` records `ZHIR` as provenance. That
  provenance field does not define or overwrite `epi_kind`; retaining a label
  alone does not establish organizational identity.

<a id="retained-mutation-participation"></a>
### 9.2 Retained Mutation and relational participation

The [retained Mutation consumer](../src/tnfr/operators/retained_mutation.py)
applies actual ZHIR to a supplied nonempty, unique, ordered tuple of members
inside a current [retained organization](#retained-member-scale-actions).
It reuses the canonical atomic stage and each target's own temporal evidence
and grammar. No history, prior coherence or trigger is pooled or inherited
from a parent. Timestamped evidence retains its current-form endpoint
requirement; legacy evidence retains its distinct operator-step contract.
Neither is authenticated as an acquired history by the retained observation.

The graph remains the full state. One transaction covers live re-admission,
execution and the resulting observation. Actual target phase, context and
lifecycle can change; raw target form, kind, capacity and pressure remain
held, with support/edge data and non-target state protected. The returned
`RetainedMutationDispatch` associates `before`, `after`, `targets` and the
actual stage result. Its per-target sealed decisions retain the consumed
evidence and event boundary; consistency is not invocation authentication,
a portable checkpoint or a collective-identity theorem. Recapture observations
after the event even when their form mean and contrast happen to be unchanged.

**Indirect form response.** Let S and S' be the nonempty incoming source
tuples selected for a core before and after a local phase reset. ZHIR holds
capacities, so the [EN compatibility family's](#reception-compatibility-policy)
relative-capacity predicates are unchanged. Only its circular phase predicates
can change the tuple. On an admitted same-kind, unclipped real form branch,
with EN mix m and source means `mu_S`, `mu_S'`,

```text
x_core_EN = (1-m)*x_core + m*mu_S,
Delta x_core_EN = m*(mu_S' - mu_S).
```

If the tuple stays fixed, EN's immediate form output stays fixed. Strict
eligibility margins provide such a region; crossing a source boundary can
produce a discontinuous mean change. Empty selection is explicit abstention,
not an EN event with a fabricated mean. Rejected state/grammar and invalid
selected form remain separate from that abstention. For a later admitted RA
with the core as its sole compatible donor and mix r, its form difference is
`r*m*(mu_S' - mu_S)` on the same unclipped branch. Binary64 rounding and the
actual sign/kind/projection gates retain their own scope.

The current ZHIR event thus need not change scalar EPI to alter a later real
operator response. Even the invertible form pair `(q,d)` is insufficient when
associated phase/ports are omitted. This is a consequence of ZHIR followed
by EN/RA, not a claim that ZHIR directly realizes the manuscript's nonzero
form change or its unspecified topology-changing mutation.

**RA has a separate phase response.** RA uses outgoing U3-compatible neighbors,
not EN's incoming selection. A changed tuple can alter its form mean. With a
fixed tuple, unchanged forms and the same branch, its immediate form result
is unchanged, but its circular-resultant phase alignment can still change.
For example, two donors initially at one phase, one shifted by s with `|s|<pi`,
have resultant angle shifted by `s/2` in a consistent chart. If U3 stays
admitted, an RA phase gain kappa gives a target phase difference `kappa*s/2`.
The nonzero-resultant and represented arithmetic conditions are essential.

**Symmetry is scoped.** A common rotation of every consumed phase preserves
ideal circular distances, source tuples and form means; a defined resultant
and the RA phase result rotate covariantly. Rotating a pair while leaving an
external port fixed is a relative change, not that symmetry. Absolute
quadrant telemetry, raw phase associations and floating-point threshold
decisions need separate treatment. Recapture sealed observations, use margins
for finite controls, and do not infer full-state or bitwise symmetry from
the ideal circular identity.

The [finite continuation controls](../tests/operators/test_retained_mutation_continuation.py)
compare actual ZHIR and no-ZHIR with unchanged ports, selected EN and later
RA, plus unchanged-source, common-rotation and unavailable-own-history
controls. Their supplied timestamped secant can exceed the trigger while the
instantaneous capacity-pressure product does not. Actual IL/OZ prepare own
grammar while holding form; they do not generate the supplied past samples.
The [transaction controls](../tests/operators/test_retained_mutation.py) cover
member association, own admission and protected-state boundaries. These are
finite interventions, not an autonomous origin or persistence claim.

**Occurrence must come from the NFR.** The canonical requirement separates
action maps from the request generated by an NFR's state, history and
relations, its admission or abstention, and its timing. A supplied target/word
tests an action but does not close this causal requirement. It does not mean
that an NFR invents an arbitrary operator law. The
[native selector](NATIVE_OPERATOR_DYNAMICS.md#native-selector-memory-and-occurrence)
is an existing candidate mechanism: diagnostic proposals, lag priority,
temporal gate and live grammar jointly determine an applied action. Its
normalization can consume disconnected context, and outer calls supply its
decision schedule. State dependence alone therefore establishes neither
locality nor an autonomous clock. A richer form/identity realization and
endogenous occurrence must retain these unresolved premises; the
[research program](RESEARCH_PROGRAM.md#research-current-question) owns their
next bounded discriminator.

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
  are enabled, capacity must meet the admitted finite `SHA_MIN_VF` policy
  (default `0.01`); equality is allowed. Boolean, nonfinite or unrepresentable
  thresholds reject. No critical-pressure gate or nonzero-EPI requirement is
  imposed by that precondition.
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
and applies the configured projection. The disabled pure numerical branch
does not consume or change EPI; public execution retains its scalar EPI
preflight. Capacity-only contraction leaves $E_D+\beta V$ unchanged;
the enabled form reset has its own storage budget. Local contraction toward
zero need not reduce differences from neighboring forms.

**Key constants**:

| Constant | Value | Derivation |
|----------|-------|------------|
| Scale factor | $1-1/(4\pi)\approx0.9204$ | Same operational $\nu_f$ step as SHA |
| Densification factor | $1/f_{\text{NUL}}\approx1.0865$ | Reciprocal configured capacity factor |

The immutable scale proposal binds the requested capacity factor, its rounded
binary64 reciprocal coefficient, the pre/post pressure and any EPI-boundary
adaptation before commit. Each densification audit event carries its target identifier.
Contraction metrics use their captured pre-operation pressure, or the latest
matching target event for compatibility, and report signed pressure change
separately from magnitude increase. Unchanged zero pressure is therefore not
misreported as densification.

**Grammar**: Simplifier with no active U1/U2 role; it neither closes a word
nor supplies U2 stabilization. Default VAL/NUL factors are not reciprocals,
and even reciprocal factors do not invert the full pressure/history and
projected-form action.

**Contract**:
- Pre: Admitted capacity, pressure and any consumed scalar form. Enabled
  strict preconditions additionally impose configured minimum capacity and
  EPI plus a maximum density policy; these are not universal safety bounds.
- Post: Capacity is not increased; stored pressure follows the checked
  reciprocal proposal, with zero pressure remaining zero. Subsequent pressure
  refresh is a separate constitutive operation.

Implementation: [shared scale proposal](../src/tnfr/operators/_scale_operator_kernel.py);
[reciprocal coefficient and branch controls](../tests/operators/test_scale_operator_kernel.py).

<a id="retained-member-scale-actions"></a>
### 10.3 Retained members, ports and actual VAL/NUL participation

**Constitutive scope.** The historical Expansion passage (TNFR.pdf pp.105–108)
describes compatible derived forms and reorganization retaining a core; its
clone/link example does not define inherited state, context or ports. The
Contraction passage (pp.109–112) describes synthesis retaining diversity and
memory. Its sum-to-core arrow is not an equality or a conservation law, and
its merge example leaves the successor state unspecified. These roles are
broader than the current scalar maps. A graph clone or average cannot silently
supply the missing realization.

The [retained organization interface](../src/tnfr/operators/retained_organization.py)
captures a supplied nonempty ordered tuple of existing members. It retains
their primitive state, own context/history associations, ordered incoming and
outgoing neighbor identities, and incident keyed edge data. Internal and
boundary ports stay associated with actual member identities; describing a
port does not consume its boundary node's form. The sealed record belongs to
its source graph. The full graph remains the runtime state, and the record
is neither a portable checkpoint nor a minimal sufficient collective state.
Graph policies and boundary-node coordinates remain live inputs rather than
frozen record contents. Temporal signatures retain present raw evidence,
including stale samples, without certifying endpoint currency or acceleration.

`execute_retained_scale_stage` re-admits that association and executes the
existing atomic Expansion or Contraction stage on those members. It requires
the requested two-phase action; a fallback glyph cannot satisfy the request.
No refresh callback or alternate sequence context is admitted by this bounded
consumer. Before/after observations accompany the actual stage result. Member
identities, phases, support, edge records and nonmember state are preserved;
each member receives its own admitted scale action and context update. Failure
of the guard or subsequent observation rolls back the graph-owned transaction.
No new scalar law, macro-node, birth, merge or autonomous occurrence is added.

The child's own context matters without imposing a new initiation rule. In
the current incremental grammar, U1a requires a generator only when EPI is
zero and history is empty; a nonzero THOL child is not blocked merely by its
empty history. Its zero stored pressure separately fails default strict VAL
minimum-pressure admission. Selecting `VAL_MIN_DNFR=0` is an explicit policy
choice, not inherited stabilization credit. A child EN can supply an actual
history entry and participation, but is not a universal VAL/NUL prerequisite.
Neither execution nor grouping transfers the parent's prior IL/OZ history,
debt, coherence credit or timestamped past to the child. Form jumps still
require their own temporal observation boundary before derivative claims.

#### Why a scalar average loses continued response

Consider the undirected path `v--parent--child`, with forms v, xp and xc.
Fix support, same-kind scalar form, compatible phases and a region where
clipping is inactive. Each action retains its own grammar and preconditions.
From the same prestate, actual RA at the peer with factor r and EN at the
parent with factor m give

```
Y_RA = (1-r)*v + r*xp
Y_EN = (1-m)*xp + (m/2)*(v+xc).
```

Their derivative with respect to `(xp,xc)` is
`[[r,0],[1-m,m/2]]`, with determinant `r*m/2`. For positive factors, these
responses cannot both factor through one `C^1` scalar observation with `C^1`
response maps on an open two-form domain: the chain rule would give rank at
most one. In particular, `(xp+epsilon,xc-epsilon)` preserves the member mean
but changes the peer RA response by `r*epsilon`. This is a regular predictive
reduction obstruction, not a claim about arbitrary scalar encodings.

The actual sequential word RA at the peer followed by EN at the parent has
the same obstruction. Its second output is
`(1-m+m*r/2)*xp + (m/2)*xc + m*(1-r)*v/2`, because EN reads the already changed
peer. The joint derivative still has determinant `r*m/2`. Separate-prestate
and sequential probes therefore retain their different schedules. Neither
argument certifies a full-state macro-NFR; phase, capacity, pressure, history
and admission remain additional obligations.

#### The initial inherited form image does not close under EN

Nonmetabolic THOL initially gives `xc=s*xp`; UM preserves these forms. This
one-dimensional form image can be represented by one scalar initially. EN
on the parent with factor m instead gives

```
xp' = (1-m)*xp + (m/2)*(v+s*xp)
xc' = s*xp
xc' - s*xp' = s*m*((1-s/2)*xp-v/2).
```

The defect is generally nonzero. Its special zero set needs its own invariant
domain; initial inheritance alone does not supply one. This statement is
about form proportionality, since optional UM capacity blending can already
change the full initial THOL capacity relation. Varying the supplied initial
xp and a prior EN factor m gives a two-form map with determinant
`s*((1-s/2)*xp-v/2)`. Where nonzero it has a locally open form image. The
extra preparation variable is a supplied input; deterministic birth alone
has not created a new independent degree of freedom.

#### Replication multiplicity is part of a port contract

For n independent donor forms, the current unweighted mean is
`mu=sum_j(x_j)/n`. Replacing each donor j by `k_j` copies gives
`mu_copy=sum_j(k_j*x_j)/K`, where `K=sum_j(k_j)`. Equality for all forms in an
open domain requires and is implied by `k_j/K=1/n` for every j: equal
replication multiplicity in that read row. One particular vector can agree
by cancellation without satisfying this all-state condition. Duplicating one
donor a k times changes the mean by `(k-1)*(x_a-mu)/(n+k-1)`; donors `1,-1`
therefore have mean zero, while `1,1,-1` have mean `1/3`.

Equal means alone preserve neither ordered semantic-kind selection nor the
phase, capacity and context channels of the full operator. Existing parallel
edges do not manufacture repeated donor identities. Conductance weights do
not generally repair an unweighted EN/RA read. Restricted quotients can remain
valid when admission, retained observations and successor actions intertwine
on their declared domain. A new organization may instead have a changed
response; it must not be called an invisible representation refinement.
The [addressed-port construction](REMESH_STRUCTURE_AND_MEMORY.md#remesh-addressed-port-reexpression)
retains unequal members while preserving a declared EN-input/output/RA family.
Its [output interface](../docs/API_CONTRACTS.md#retained-addressed-output) reads
only an actually shared receiver through canonical EN; it does not turn an
unweighted neighbor list into an automatically grouped collective state.
The [collective Emission realization](REMESH_STRUCTURE_AND_MEMORY.md#collective-emission-port-realization)
uses one admitted member to realize a declared port gain through actual AL.
Its local boost depends on member count; target admission and own activation
remain constituent information. Equal mean alone cannot authorize a broadcast
or repeated collective word.
The [interface controls](../tests/operators/test_retained_organization.py)
check association, admission, unchanged ports and transaction boundaries;
the [finite continuation controls](../tests/operators/test_retained_scale_continuation.py)
use an actual THOL child and UM contact before scale actions and EN/RA
participation, with default-pressure rejection kept separate. Their sources
and invocation schedule are supplied. The retained-member interface preserves
the distinction above. The separate [single-derived-member VAL family](#derived-expansion-member)
now supplies one explicit birth contract; uniquely selected or general
proliferation and collective identity remain open. The concentration fragment
below adds actual internal Reception; it does not establish memory-preserving
reabsorption of a member.

<a id="retained-synthesis-concentration"></a>
### 10.4 Retained NUL/EN concentration

The [synthesis owner](NUL_SYNTHESIS.md#retained-synthesis-concentration) retains
the complete concentration derivation and its integration controls. Actual
NUL on both members followed by selected-source EN at the core changes form
response while retaining identities, ports and own contexts. It is an explicit
fragment, not a new glyph or a complete historical reabsorption mechanism.
The ideal unclipped map is `xp'=f*((1-m)*xp+m*xc), xc'=f*xc`; full mixing can
erase old form contrast. The original endpoint, pressure/admission and rollback
results retain their premises in that owner.

Historical TNFR.pdf pp109-112 motivates synthesis retaining diversity and
memory; its unspecified merge operation does not require graph-node deletion.
The [future-response and admission analysis](NUL_SYNTHESIS.md#nul-future-response)
distinguishes concentration, response-sufficient encapsulation and information
loss. A shared present form or one equal boundary response does not establish
that the interior can be forgotten.

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

The [mechanism and activation audit](#operator-mechanism-and-activation-audit)
separates registered promises, live admission and committed execution from
collective descent, autonomous occurrence and physical identification.

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

This map addresses collective action and interaction using the existing 13 operators. It
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

<a id="nfr-operator-realization-and-occurrence"></a>
### NFR realization, source proposals and event occurrence

In the engine, an operator denotes a transformation realized in an NFR's state
and relations; registering that map introduces no additional acting entity.
The target identifies where the transformation is evaluated. The source of
its occurrence is a separate question: a supplied word, an external input or
a declared state-dependent selector may request the same admitted map.
Request provenance describes the software/model boundary, not an independently
established physical cause. An additional intentional actor is not required.
These distinctions preserve
the [NFR formation contract](EMERGENT_ONTOLOGY.md#11-nfr-formation-definition-birth-and-collective-dynamics).

<a id="operator-causal-role"></a>
#### Constitutive role and implementation role

The foundational investigation must declare which claim is being made about
each consumed transformation:

| Role in a proposed model | Additional obligation |
| --- | --- |
| Basic constitutive mechanism, continuous or discrete | Declare its independent state, local dependence, complete evolution or occurrence law, units and balances. Its primitive status is a stated hypothesis. |
| Effective action realized by an organization | Derive the observed transformation from that organization's admitted interaction, with domains, timing, error if approximate, and the reaction of the remaining state. |
| Supplied intervention or execution policy | Retain the actual request source, parameters, target and admission; do not infer endogenous occurrence from execution. |

The same named action can be studied in more than one role, but a claim must
not switch roles during its argument. Fundamental discrete events are a
legitimate model choice; an infinitesimal generator is another, requiring its
own law or limit. A finite registered map supplies neither by its name alone.
Conversely, the engine's separation of continuous rows and jumps does not
exclude a structural realization in which an organized interaction produces
an effective event.

The [operator-pulse audit](nodal/OPERATOR_PULSE_CLOSURE.md#operator-pulse-causal-closure)
constrains its declared maps and retained channels. It does not select which
role is fundamental, nor prove that all thirteen actions are necessary or
sufficient at every scale. A scoped causal account must include abstention,
admission, consumed history and any replenishment, without explaining pulse
and operator timing solely by referring to each other. Research priority and
closure criteria belong to the [four pillars](RESEARCH_PROGRAM.md#foundational-pillars).

The identified [May manuscript](UNIFIED_GRAMMAR_RULES.md#source-grammar-and-notation)
supports this structural reading, while retaining unproved activation proposals:

| Historical source | Current interpretation and reusable mechanism |
| --- | --- |
| Pages 38–40 describe relational NFR state and internal/external topology; pages 78–80 describe nodal transformations and AL from latent conditions without intention | Retain the actual target, admitted state and boundary. Implemented AL changes form on existing support; it does not derive the substrate or create a node from nothing. |
| Page 79 also describes a glyph as a condensed node and an active form | This is a stronger realization proposal. A constructed NFR would need an admitted interaction whose induced transformation realizes the named map; an operator class or glyph string is not that constructed organization. |
| Pages 112–114 describe endogenous THOL, active form, capacity and an internal loop; page 124 proposes automatic NAV activation | Live history and state gates supply eligibility. THOL's configured construction and native state-dependent selectors are executable policies; neither the source wording nor their implementation proves a uniquely derived occurrence law. |
| Pages 129, 133 and 135 retain phase, capacity, history and inherited activation | Shared live U3, causal grammar debt, observed Mutation evidence and retained hierarchy are useful existing state. An inherited label cannot replace missing target-specific evidence. |
| Pages 143–147 describe stateful operations, conditional activation and a containing node regulating a sequence | Scoped program targets preserve software composition. A collective NFR additionally needs the action and grammar descent criteria below, including sufficient boundary, memory and any claimed selector state. Brackets alone supply none of these proofs. |

The source is not internally uniform: page 41 describes UM as collapse or
deactivation, whereas pages 93–96 and 131 describe coupling. The latter agrees
with the registered coupling contract; the isolated earlier label is not an
additional UM action. Likewise, an endogenous-reorganization claim on page 112
and a threshold algorithm on pages 114/145 require distinct evidence: supplying
the algorithm defines a model, not a derivation of its thresholds or selection.
An operator mechanism may itself have a realization in another NFR or in a
larger organization. The existing collective-action criteria make that proposal
testable; no universal identification between glyphs and formed NFRs is assumed.
The [organized-source Emission test](nodal/ORGANIZED_SOURCE_PORT.md#organized-source-emission-realization)
provides a concrete realization obstruction: the existing 45-node source's
all-duration form envelope excludes the default nonclipped increment at all
nine constituents of its contacted child. The proof allows donor reaction
and relaxes collateral endpoint conditions. It neither executes a word nor
excludes smaller gains, and separates the direct event's held pressure from
the same-law refresh used in a continuous endpoint comparison.

<a id="continuous-operator-realization-audit"></a>
### Operator participation in the continuous contact studies

**Audit result:** the fully weighted autonomous contact completion in
[Relation foundations](nodal/RELATION_FOUNDATIONS.md#weighted-contact-operator-boundary)
does not execute the registered operators or their grammar. No reduction,
continuous-generator limit or full-state realization of that completion from
those operators has been established. Its acquisition and recovery theorems
remain conditional results for independently postulated equations. They cannot
yet be claimed as consequences of TNFR's fundamental operator mechanisms.
The engine's permission to evaluate continuous laws is not a mathematical
justification for omitting those mechanisms from a foundational claim.

| Mechanism | Actual operator participation | Boundary of the weighted completion |
| --- | --- | --- |
| Contact / union | Coupling (UM) proposes U3-compatible new edges with a configured compatibility weight; existing edge weights are preserved. Actual reset work and subsequent-law admission have separate owners. | The two positive loss-funded support rates continuously change supplied candidate weights. Neither rate is derived from UM, and neither checks primitive phase, U3 context or the compatibility score. |
| Form propagation / resonance | Resonance (RA) mixes form over admitted neighbors and has declared phase/capacity effects. | The weighted form/phase exchange is an independently specified field. Agreement of one diffusion channel would not identify all of RA, its occurrence or its grammar. Oscillation, synchronization and the RA event are distinct claims. |
| Self-organization | Self-organization (THOL) uses its admitted acceleration/pressure mechanism; the public/staged path can additionally create an isolated child subject to history/hierarchy gates. | Attraction to an organized state under the auxiliary ODE does not execute THOL. That ODE retains neither THOL's temporal evidence nor its child-creation state. |
| Ordered structural dynamics | The native hybrid path executes selected glyphs with live admission and retained grammar/history. Its decision schedule is part of the law. | The contact field has no word, event trace, selector memory or grammar state. Smoothness, storage loss and a regular resultant chart do not establish U1-U6. |

Existing partial bridges remain useful. For example,
[the RA realization reader](../src/tnfr/physics/resonance_realization.py)
relates an admitted local EPI blend to a represented affine jump and, under
its additional premises, fixed post-event pure-EPI recovery. The collective
operator and grammar descent results likewise retain their established scope.
The missing result is the realization of the **complete weighted contact law**,
not the absence of every mathematical connection between operators and flows.

**A concrete admission obstruction.** Take two unit P2 components with forms
`(0,epsilon,0,0)`, positive held unit capacities, phases `(0,0,delta,delta)`,
and absent candidate contact `{1,2}`. Choose `epsilon!=0` and
`pi/2<delta<pi`. Each component's relative resultant is one, so the auxiliary
field is regular. At zero contact, `q=(-epsilon,epsilon,0,0)` and the full
endpoint allocation gives `B=e*epsilon^2>0`. The candidate cost is finite:

```text
c = epsilon^2/2 + beta*(1-cos(delta)).
```

Both postulated contact rates are therefore strictly positive. Actual UM with
the default U3 limit excludes this cross-component candidate, even when its
separate score threshold is relaxed to zero. Its existing same-phase neighbor
does admit the target's UM event. Thus this is a candidate mismatch, not an
invalid target or a contradiction in either separate model. It disproves
unqualified identification of the support law with registered UM over its
regular domain. It does not disprove an explicitly restricted or otherwise
derived operator realization. Inside the small phase-compatible acquisition
ball this particular obstruction is absent, but the missing derivation remains.

There is also a compatible-domain distinction: UM does not reinforce an
already present edge, whereas either support row can have positive rate there.
For RA, equal phases with unequal form can give nonzero relational phase
velocity while its configured phase-alignment event leaves those phases equal.
Matching form mixing alone consequently misses a consumed row.
The [cross-path controls](../tests/operators/test_passive_contact_operator_boundary.py)
use actual finite UM/RA events and native initial fields against detached
support-rate accounting. They install no weighted trajectory or new operator.

**Required bridge.** A foundational use must provide at least one explicit
construction, with every claimed channel and its domain covered:

1. A hybrid law that actually executes the registered maps, declares their
   selection/abstention, grammar state, elapsed schedule and inter-event flow,
   and includes each jump's work and its successor domain. Every consumed
   inter-event row also needs an operator-based constitutive justification;
   attaching events to an arbitrary auxiliary flow does not close the bridge; or
2. A proved restriction, observation/reduction or scaled operator limit,
   retaining the state/history required by admission and reaction. Specify
   its map, units, clock, gain scaling, error/convergence and time interval,
   including support birth and any switching boundaries; or
3. An explicitly proposed revision of the constitutive operator mechanisms,
   with independent equations, justification and comparison to existing
   contracts. A new label or a post-hoc decomposition is not that derivation.

Not every member of the catalog must act on every trajectory. An inactive
operator or eliminated channel needs the stated regime and a proof of the
claimed invariance, restriction or approximation. A finite fixed-gain event
cannot simply be divided by an arbitrarily small time to define a finite
continuous generator. Inserting occasional UM/RA/THOL events into the existing
ODE would change its complete law and require rechecking its theorems.
The [current program](RESEARCH_PROGRAM.md#research-current-question) prioritizes
this causal obligation; the earlier scoped pillar resolutions are not a bridge
for the later weighted contact extension.

<a id="analytic-fixed-duration-realization"></a>
### Analytic obstruction to fixed-duration constant increments

**Conditional realization criterion.** Let `M` be a connected finite-dimensional
real-analytic state manifold carrying a declared autonomous law. Suppose its flow `Phi_T`
exists and is real analytic in initial state throughout `M` for the fixed
finite duration `T`, and there is an equilibrium `z_star` in `M`. Let `O`
be a real-analytic scalar observation. No nonzero constant `b` can satisfy

```text
O(Phi_T(z))-O(z)=b
```

on a nonempty relatively open subset of `M`. Indeed, the left-hand side
minus `b` is real analytic. If it vanishes on that subset, the identity
theorem extends the equality across connected `M`. At the equilibrium the
observed difference is zero, contradicting `b!=0`. This is a restriction
on a specified realization class, not on every canonical operator or NFR.

The result tests a necessary observed component of a proposed map. The
remaining coordinates may react freely; excluding the component excludes
the stronger complete transformation. It does not follow merely from
conservation of a whole-network total. Fixed `T` may be chosen separately
for each fixed `b`, but both must then be independent of the initial state
throughout the proposed open domain.

**Countable sampling corollary.** Fix the same nonzero gain and any prescribed
countable set of finite durations `T_n` with these flow properties. Each
realization set `Z_n={z: O(Phi_T_n(z))-O(z)=b}` is closed with empty interior,
hence nowhere dense. Such a countable union cannot cover a nonempty open
subset of `M`. One direct Baire argument works in a coordinate ball inside
that subset: choose nested closed balls of shrinking radius, each avoiding
the next `Z_n`, with each ball inside the previous interior. Completeness
of the initial compact coordinate ball supplies a limit point outside
every `Z_n`, a contradiction. Thus even an arbitrary state-dependent choice
among those prescribed durations cannot realize the exact fixed gain on
an open full-state domain. The union need not be empty, closed or nondense;
the conclusion is that it contains no nonempty open domain.

This criterion needs analyticity, an equilibrium in the connected flow
domain, fixed duration and a genuinely open full-state domain. Smoothness
alone is insufficient. A positive increment on a constrained actuator
manifold, an isolated endpoint match, a state-dependent stopping time or an
approximate map is not excluded. A zero increment also gives no contradiction.
Changing the observation requires re-admitting its regularity and meaning;
thresholded labels are not analytic observations automatically.

The [organized-source application](nodal/ORGANIZED_SOURCE_PORT.md#organized-source-open-domain-emission)
verifies these hypotheses for the existing lifted 45-node law and its
child-form mean. It does not impose this continuous realization on the
native hybrid engine. Grammar and event maps retain their own admission;
neither an operator registry nor this obstruction supplies an occurrence law.

<a id="regular-stopping-realization"></a>
### Tangent criterion for regular near-identity stopping

**Conditional realization criterion.** Let `F` be a continuously
differentiable autonomous field with local flow `Phi_t`, and let `O` be a
differentiable vector observation. At a fixed initial state `z`, suppose

```text
O(Phi_{T(z,b)}(z)) = O(z) + b*v(z),    v(z) != 0,
T(z,0) = 0,    T(z,b) > 0 for small b > 0.
```

Here `T(z,b)` is the elapsed duration, with a finite right derivative
`a(z)` at zero. The chain rule gives

```text
a(z) * DO(z) * F(z) = v(z),    a(z) > 0.
```

Positivity follows because `T(z,b)/b>=0`, and `a=0` contradicts `v!=0`.
Thus choosing a duration can rescale the observed velocity, but cannot
change its direction. A nonzero prescribed form increment with unchanged
target phases requires equal positive target form velocities and zero
target phase velocities. Collateral coordinates remain free. No derivative
of `T` with respect to the initial state, analyticity or equilibrium is
needed for this necessary condition.

Tangent agreement does not suffice for a finite map. For a held linear
observation `P`, if `P*F(z)=0`, the next necessary condition is
`P*DF(z)*F(z)=0`: Taylor expansion at `T=a*b+o(b)` otherwise gives a nonzero
coefficient of `b^2/2` in the held output. This argument needs no second
gain derivative of `T`. If `T` is C1 in gain with `a>0`, its local inverse
also shows that the held output must remain constant throughout a short
time interval, rather than only at isolated selected endpoints.

A regular positive clock rescaling multiplies both observed rows by the
same factor and rescales `a` inversely; it preserves these collinearity
conditions. Singular stopping, such as `T=sqrt(b)`, is outside the finite
derivative premise. A finite-gain hit or a long excursion has its own
realization question. The zero-gain identity here is a formal extension of
an ideal primary-channel map, not an admitted zero-gain invocation or an
identity of the public operator's lifecycle and metadata effects.

The [organized-source application](nodal/ORGANIZED_SOURCE_PORT.md#organized-source-regular-stopping)
derives the exact tangent locus for the actual two-row law and a constrained
state that passes it but fails at second order. It supplies no event selector
or alteration of the registered Emission contract.
Its [local carrier classification](nodal/ORGANIZED_SOURCE_PORT.md#organized-source-carrier-persistence)
then identifies all nearby phase-stationary states of that exact invariant
restriction: an antisymmetric family can move internally while its target
form stays fixed. A six-derivative rank certificate and the implicit-function
theorem establish nonlinear persistence, rather than assuming it from tangent
agreement. The conclusion is local to that carrier and supplies no event.

### Existing execution responsibilities

The existing implementation separates these responsibilities:

- `structural.run_sequence` binds a supplied word to one existing target and
  observes its signed initial form. The network word runner fixes the supplied
  population at invocation; each stage admits the actual state and history.
- `dynamics.selectors` proposes actions from configured state metrics, history,
  thresholds and possibly random draws. Shared grammar/preconditions can reject
  them or apply the declared standalone replacement policy. State dependence
  does not make that policy a consequence of the nodal identity.
- Scoped programs retain ordered targets and distinguish requested glyphs from
  subsequent step-function calls. The default step may select further actions;
  a requested-word trace is not their complete history. The
  [program contract](../docs/API_CONTRACTS.md#program-compilation-contract)
  owns these provenance and failure boundaries.

The native selector route also has AL/EN lag policies and configurable
incremental grammar. It executes primitive glyphs, not the full public-class
lifecycle or a prevalidated complete word. Its realized glyph history does not
necessarily retain every proposal or selection reason. Matching operator names
across these paths cannot substitute for their different execution contracts.

The default runtime separately applies nodal integration, phase coordination,
capacity adaptation and an automatic REMESH gate. Enabling glyph selection
does not derive these inter-event rows or their schedule; disabling selection
does not remove them. The opt-in relational executor advances supplied form
and phase laws without invoking structural operators or ordered grammar.
Both routes therefore require a complete operator realization for a canonical
claim. The nodal product alone does not supply that missing derivation.

The strongest existing synergies are already conditional results: retained
histories distinguish admissibility at equal aggregate state; the maintained
parent supplies genuine pre-birth history; actual support events carry separate
work; and continuous regional exchange can change a region's form without an
invoked AL event. These mechanisms help test an occurrence model, but must not
be renamed as proof that one has been derived. The
[collective grammar criterion](#collective-grammar-descent) gives the appropriate
obstruction when the proposed NFR state discards information needed by a gate.

<a id="operator-canonical-coverage-audit"></a>
### All thirteen actions: implementation and canonical realization

**Coverage decision.** All thirteen operators have registered contracts,
primitive dispatch, public classes and declared snapshot-stage contracts.
This is implemented coverage, not a complete realization of their fundamental
roles. The retained manuscript's broader formation, filtering, topology,
mutation and multiscale claims have unresolved bridges to the present scalar
and graph maps. Neither the historical descriptions nor passing software
contracts establishes that those bridges exist.

The source comparison uses the identified 378-page May manuscript in the
[grammar source audit](UNIFIED_GRAMMAR_RULES.md#source-grammar-and-notation).
Complete relevant operator sections, surrounding grammar and selected rendered
equations were checked. Page locators below refer to that source, not the
different March manuscript. The source contains incompatible prescriptions;
we retain its functional objectives without silently imposing every example
or numerical policy on the engine.

The direction of justification is **structural operators and grammar, then
complete dynamics, then organized behavior and observations**. Operators may
be postulated as primitive mechanisms; they need not be circularly inferred
from an already chosen ODE. Their complete maps, domains, causal occurrence
and composition still need mathematical definitions. A primary-channel entry
is a description of current implementation, not a proof that a fundamental
operator can affect no other part of a richer form.

| Operator | Actual mechanism relevant to interaction | Reusable synergy and boundary | Historical role still to justify |
| --- | --- | --- | --- |
| **AL — Emission** | Adds/projected form on existing support; the separate [finite-duration proposal](EMISSION_EPISODES.md#emission-episode-status) integrates held form flow with one admitted AL lifecycle | [Collective descent](nodal/SINE_PAIR_INTERACTION.md#sine-pair-emission-descent) and the [complete AL/SHA word](nodal/SINE_PAIR_WORD.md) retain targets, source, context and work. A local form contrast changes phase rates. The [episode observation](EMISSION_EPISODES.md#emission-selector-observation) separates law-derived activity from solver diagnostics for actual native selection. No node/edge creation or autonomous input timing. | pp80-82: endogenous activation of a latent organization is broader than a supplied finite boost on an existing node; its source and occurrence remain to be grounded. |
| **EN — Reception** | Blends realized incoming form from all neighbors, an [explicit tuple](#reception-causal-sources), or a [declared compatibility policy](#reception-compatibility-policy) at a supplied attempt | The [weighted contact action](nodal/SINE_COLLECTIVE_RECEPTION.md) retains its original all-input EN/IL/SHA and work premises. Selected mean/kind and compatibility evidence retain their associated sources; changed actions do not inherit the old affine certificate. | pp83-86: one conditional phase/capacity family does not derive unique compatibility, autonomous attempt timing or richer identity retention. It is not U3; capacity is not an observed oscillation frequency. |
| **IL — Coherence** | Glyph contracts stored pressure; public/staged path also aligns target phase with its neighbor resultant | The [collective handoff](nodal/SINE_COLLECTIVE_PRESSURE_HANDOFF.md) accounts for actual adjacent-target phase work and refresh; its [identity domain](nodal/SINE_COLLECTIVE_IDENTITY_TRANSITION.md) retains context. The stabilizer label supplies no universal storage theorem. | pp86-89: persistent functional coherence and an asymptotic coherence-matrix condition do not follow from one stored-pressure contraction. The richer form/coherence representation and continued law are required. |
| **OZ — Dissonance** | Perturbs stored pressure, with near-zero/jitter branches and optional neighbor propagation | The [handoff](nodal/SINE_COLLECTIVE_PRESSURE_HANDOFF.md) separates changed context from pressure erased on constitutive refresh. A [persistent pressure extension](nodal/SINE_PRESSURE_COMPLETION.md) needs its own mean, storage and law premises. Signed propagation can cancel pressure. | pp89-93: opening a controlled reorganization is not proof of a bifurcation. Amplification, signed jitter, incoming propagation and pressure persistence have distinct contracts. |
| **UM — Coupling** | Moves phase, optionally aligns capacity/contracts stored pressure and proposes links; explicit [isolated-partner](#coupling-potential-pair) and [disjoint-component](COUPLING_COMPOSITION.md#connected-coupling-request) requests retain distinct domains | The [actual birth/contact](#coupling-connected-parent), [weighted support](nodal/SINE_PAIR_COUPLING.md), [maintained contact](nodal/SINE_PROXIMITY_CONTACT.md) and [one-way return domain](REMESH_STRUCTURE_AND_MEMORY.md#operator-maintenance-return) retain different laws and participation premises. The [reciprocal boundary result](REMESH_STRUCTURE_AND_MEMORY.md#reciprocal-maintenance-boundary) motivates optional [incident-relation preservation](COUPLING_COMPOSITION.md#ordinary-coupling-relation-preservation): actual merged writers retain their original U3 boundary through atomic return. Legacy behavior, affinity and access remain explicit; bounded capacity is separate. | pp93-97: functional union does not automatically merge identities or acquire a collective NFR. Reciprocal persistence, endogenous candidate access and overlapping organizations remain open. |
| **RA — Resonance** | Mixes form over existing U3-compatible neighbors; configured phase interpolation and triggered capacity amplification | The [collective AL/RA/SHA contract](nodal/SINE_PAIR_RESONANCE.md) proves a conditional complete action and retention domain. The [reciprocal return](REMESH_STRUCTURE_AND_MEMORY.md#reciprocal-maintenance-boundary) exposes capacity growth; [RA/NUL composition and live renewal](NUL_SYNTHESIS.md#amplification-contraction-maintenance) distinguish neutral capacity, pressure growth and participant-supported numerical retention. RA creates no support and does not contract stored pressure; passivity and identity require their own premises. | pp97-100: propagation preserving organization is broader than a local scalar blend. Existing EPI realization and collective-action proofs are partial bridges, not a derivation of autonomous sustained resonance. |
| **SHA — Silence** | Attenuates capacity and records latency information, preserving form at the event | The [complete word](nodal/SINE_PAIR_WORD.md), [lifecycle-sensitive NAV action](nodal/SINE_COLLECTIVE_TRANSITION.md) and [retained reactivation prefix](REMESH_STRUCTURE_AND_MEMORY.md#retained-silence-reactivation) retain own memory and successor domains. Capacity-only work is zero, but future rows change; finite attenuation need not freeze flow. | pp101-105: a continued law must sustain recovery beyond the admitted finite prefix. Finite attenuation is not total silence; zero capacity does not recover through multiplicative VAL/NAV alone. |
| **VAL — Expansion** | Ordinary VAL scales capacity and optionally form; the separate [derived-member family](#derived-expansion-member) adds one isolated successor with explicit inheritance and fresh context | The [retained-member action](#retained-member-scale-actions) preserves individual ports/context; the opt-in birth composes actual scalar VAL with shared materialization and separate UM/EN/RA participation. The [full form-capacity action](nodal/SINE_FORM_CAPACITY_ACTION.md) retains its own work/flow premises. | pp105-108: one bounded supplied family does not select unique inheritance, arbitrary branching, autonomous proliferation or persistent identity. Existing scalar VAL remains unchanged; a graph clone alone does not realize the historical role. |
| **NUL — Contraction** | Lowers capacity and reciprocally rescales stored pressure; optional form scaling remains separate from the [NUL/EN concentration fragment](#retained-synthesis-concentration) | The fragment retains actual members/ports and adds internal Reception; the [full reset owner](nodal/SINE_FORM_CAPACITY_ACTION.md) and [directional continuation](nodal/SINE_DIRECTIONAL_MAINTENANCE.md) retain their distinct work and flow premises. The [amplification/renewal analysis](NUL_SYNTHESIS.md#amplification-contraction-maintenance) retains NUL's pressure cost and the closed-population capacity obstruction. Form agreement, context and identity are not interchangeable. | pp109-112: elementary retained concentration does not derive memory-preserving reabsorption, a unique synthesis map or persistent identity. No member is merged/deleted, and NUL still supplies no live U1 closure or U2 stabilization credit. |
| **THOL — Self-organization** | Primitive signed acceleration/pressure reset; public/staged path can create an isolated nested child after history/hierarchy gates | The [collective hierarchy contract](nodal/SINE_COLLECTIVE_HIERARCHY.md) separates no-child refresh equivalence from a changed domain. [Maintained-parent birth](nodal/SINE_MAINTAINED_PARENT_THOL.md) supplies one actual event with genuine parent history and a declared frozen-isolate continuation. The [episode selection boundary](EMISSION_EPISODES.md#emission-selector-observation) distinguishes cached acceleration requests from THOL's actual history-derived pressure action. Parent metadata is neither an edge nor child temporal evidence. | pp112-116: a history-gated nested child is a construction, not yet a proof of autonomous self-organization, its connectivity or its inherited dynamics. Acceleration gain and clock units remain declared. |
| **ZHIR — Mutation** | Configured phase reset following a live signed EPI-growth test and separate grammar context; [retained participation](#retained-mutation-participation) can change later actual EN/RA response | [Temporal admission](nodal/SINE_COLLECTIVE_MUTATION_ADMISSION.md) and the [complete phase action](nodal/SINE_COLLECTIVE_MUTATION_ACTION.md) retain separate obligations, work and successor-domain conditions. [Actual episode/native competition](EMISSION_EPISODES.md#emission-native-source-competition) retains own grammar and shared consumption of matching physical intervals; selected sources and phase/port association remain necessary. | pp116-121: indirect form response does not complete richer-form mutation or organizational identity. NFR-generated request and timing remain distinct from supplied invocation; a retained kind label is insufficient. |
| **NAV — Transition** | Primitive pressure change; public/staged regimes can also change phase and capacity | The [contextual Transition contract](nodal/SINE_COLLECTIVE_TRANSITION.md) separates lasting phase/capacity changes from jitter erased on refresh. Its admitted word can leave the original acute sector; regimes and invocation remain policies. | pp121-124: activation of a prepared regime is not node creation from nothing. Regime labels and warnings are configured policies, not a proved bifurcation or source-acquisition law. |
| **REMESH — Recursivity** | Advisory glyph, delayed same-support EPI mixing, and explicitly invoked topological replacement are distinct APIs | The [collective delayed-state contract](nodal/SINE_COLLECTIVE_RECURSIVITY.md) retains associated histories, actual mixing and reset work. The advisory does not mix form; topology replacement requires its own before/after state and law admission. | pp124-128: structural recursion across scales needs a state/identity map. Advisory, delayed mixing and topological reconstruction are distinct partial realizations, not interchangeable operations. |

Execution owners are linked in §15. The particularly relevant shared kernels
are [AL/SHA proposals](../src/tnfr/operators/al_sha_stage_proposals.py),
[reception input](../src/tnfr/operators/_reception_kernel.py),
[UM proposals](../src/tnfr/operators/_coupling_stage_kernel.py),
[atomic stages](../src/tnfr/operators/network_stage.py),
[Mutation admission](../src/tnfr/operators/_mutation_gate.py), and
[THOL dispatch](../src/tnfr/operators/self_organization_selection.py).

These links constitute scoped coverage of the registered mechanisms, not
one universal execution domain. Their source families, complete laws and
clock/pressure conventions must match before combining results. The
[partial-action criterion](#collective-grammar-descent) additionally requires
admission and successor state at every prefix. THOL birth can leave the
connected-support law domain, and a REMESH advisory cannot stand in for
delayed mixing. Preserving such explicit boundaries is part of integration.
The [viability and ordered-return analysis](NUL_SYNTHESIS.md#operator-viability-and-ordered-return)
separates actual admission, reference preservation and continued response:
IL can open a blocked RA pressure gate, while NUL's position changes which
live form input is contracted. A selector proposal certifies none of those
obligations. Support/type-changing roles require their own successor domains.
A compatibility classification does not establish that all thirteen full
public actions preserve one identity, that their words realize every TNFR
transformation, or that the continuous law selects their occurrence.

#### Execution integrity and the remaining constitutive decision

The audit corrected concrete inconsistencies within the existing finite-action
contracts, without promoting their policies to fundamental laws:

- AL treats an explicit inactive flag as first activation, validates its type
  and rejects conflicting retained AL provenance before writing lifecycle state.
- The optional OZ overload policy counts unresolved perturbations after the
  latest admitted resolver in its existing window. Earlier stabilization does
  not resolve a later perturbation; persistent grammar debt remains separate.
  Its monitor binds amplification versus jitter to the pre-event policy and
  rejects invalid pressure endpoints. Finite jitter may reduce pressure
  magnitude; local amplification is checked before incoming stage propagation.
- ZHIR rejects a phase proposal with no represented circular effect before
  committing. Its postcondition uses the same comparison instead of an
  arbitrary minimum amplitude. The represented period is `math.tau`, not a
  symbolic exact value of pi.
- Capacity and nonzero form products lost to represented zero are rejected by
  the shared scale proposal before commit. Intentional zeros and declared
  clipping retain their own semantics.
- Direct and public NAV admission share finite threshold and strict flag
  validation, preserving signed form/pressure and the optional history scope.
  Its monitor recognizes admitted represented changes without a minimum-size
  cutoff. THOL monitoring checks finite endpoints and the unchanged parent
  form, capacity and phase; its signed pressure reset need not keep the
  instantaneous coherence diagnostic within an arbitrary ten-percent drop.
  These are partial postconditions, not full reset or hierarchy certificates.
- Automatic REMESH requires a positive evidence window and admitted consumed
  samples. Missing or short history abstains; an empty window cannot trigger
  mixing. Its configured thresholds still do not derive autonomous occurrence.
- Atomic stages accept the engine's exact integrity-monitor modes while
  retaining manual state capture and rollback. External or modified copy
  protocols remain rejected; monitor identity and mutable reports are restored
  when a later event/refresh fails.

These are execution guarantees, not proof of productive dissonance, permanent
coherence, a new identity or cross-scale recursion. The live implementations,
regression owners and broader historical roles have distinct evidence.
The [execution map](THEORY_TO_EXECUTION.md#operator-audit-execution) routes the
corrected boundaries to their shared kernels and controls.

The [common state/action specification](#structural-state-action-specification)
must now be applied to each proposed realization. Distinguish richer form from its scalar
EPI observation; retain support, phase, capacity, history and identity data
actually consumed; and specify how birth, synthesis, mutation and scale maps
change that domain. Each historical role needs an explicit choice: existing
implementation, justified restriction, proposed constitutive revision or open
obligation. Current primary-channel metadata cannot decide that choice.
Then a complete interacting mechanism must justify occurrence, abstention,
grammar and every inter-event row. The
[research program](RESEARCH_PROGRAM.md#research-current-question) owns this
sequence. It does not require every operator on every trajectory, or discard
the existing partial realization and collective-action theorems.

All thirteen glyphs also have implemented snapshot stages in the shared
[stage contracts](../src/tnfr/operators/stage_contracts.py). This execution
coverage includes REMESH's advisory, not its separate delayed mixing API.
For an organized object, complete-word integration still consumes the actual
fine state, member-to-history and boundary associations, live eligibility,
reset work and the successor law at every prefix. A valid word can leave a
declared identity, as the [directional counterexample](nodal/SINE_DIRECTIONAL_MAINTENANCE.md#directional-maintenance-result)
shows. Unsupported observations must remain unavailable rather than being
filled from a parent, a previous support or a diagnostic label.

The following completed studies supply distinct operator-to-law handoffs.
Their coefficients and source domains must remain attached to each result.

| Technical owner | Admitted result and continuing boundary |
| --- | --- |
| [Maintained-parent THOL](nodal/SINE_MAINTAINED_PARENT_THOL.md) and [committed UM](nodal/SINE_MAINTAINED_PARENT_UM.md#maintained-um-committed-result) | Genuine parent history, actual birth and a connected endpoint admitted by primitive-state review into the [finite motion family](nodal/SINE_SUCCESSOR_NONSTATIONARY.md#successor-nonstationary-result). The [original failed transaction](nodal/SINE_MAINTAINED_PARENT_UM.md#maintained-um-transaction-result) and first assessment remain immutable evidence, distinct from the corrected committed result. |
| [Weighted-star handoff](nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-constitutive-handoff-result) | The actual no-refresh AL/UM/SHA output admits two complete continuations with different fresh pressure/form rates, each with its own storage balance and constituent maintenance. This selects neither law and covers no unexamined operator. |
| [Maintained-parent SHA](nodal/SINE_MAINTAINED_PARENT_CLOSURE.md#maintained-parent-sha-result) and [changed-law retention](nodal/SINE_MAINTAINED_PARENT_CLOSURE.md#maintained-sha-retention-result) | Zero storage jump, changed capacities and nonzero parent rates; all 53 principal branches are separately certified through elapsed `[0,1/2]`. The earlier UM certificate is not transferred through SHA. |

SHA is one capacity-only closure control, not a preferred ending for every
organization. Another closing action needs its own full word, live context
and successor law. These examples establish neither a universal collective
executor nor an obligation to repeat all thirteen actions.

A reusable complete-law handoff must associate the requested and executed
action with its source, raw endpoint, refreshed pressure, both continued
rates, event work and retained or unavailable history. The shared
[word dispatcher](../src/tnfr/operators/word_execution.py) and atomic stage
result supply execution mechanisms; the
[event runtime](../src/tnfr/operators/event_runtime.py) supplies configured
nodal EPI integration. Neither alone admits the selected complete weighted
sine form/phase law after every stage. Existing research refresh callbacks
compose these responsibilities for particular models; their shared field,
integrator and state-admission owners are the reusable basis for that handoff.
`StudySpec` remains a construction recipe, not a checkpoint restoring an
organization's complete law, history and context.

Population scheduling is another explicit composition boundary. The shared
word runner fixes targets at invocation, across all stages and cycles, so a
THOL-born child is not a later target in that invocation. A subsequent call
selects the population again. SDK trajectory recording currently separates
cycles into calls, while unrecorded evolution batches them; its current named
words contain no THOL. Extending those words to birth requires a declared
target-population policy and checks that recording preserves that policy.
This scheduling question does not invalidate the retained single-target
birth or hypothetical successor certificates.

The [hierarchy-to-contact composition](nodal/SINE_HIERARCHY_CONTACT.md)
admits one concrete boundary under an explicit frozen-isolate intermediate
law. Actual parent-target UM connects the new child, with complete support
work and nonzero child response on a declared source family. This does not
supply observed birth from a maintained organization. The birth changes
node support, so the ordinary U6 before/after reference is unavailable there;
same-support stage observations do not become a full-word U6 certificate.

### Quantitative synergies, without inventing an activation law

The [full-reset derivation](nodal/RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact)
and shared `observe_relational_reset` separate nodal reorganization on old
support from support work at the new state. They retain the actual endpoints
instead of assigning energy signs from operator names.

- **Form changes influence phase evolution.** For held support, phase and
  capacity, an AL/EN jump `d` changes the next phase row by
  `(w/beta)*diag(nu/H)*B*d`. One unclipped EN target has nonpositive form cost
  on this native unit support; the [weighted-contact result](nodal/SINE_COLLECTIVE_RECEPTION.md#3-exact-event-work-and-its-sign)
  instead requires the full event sign criterion and simultaneous cross terms;
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
  blanket capacity-only interpretation. The [full contact analysis](nodal/SINE_FORM_CAPACITY_ACTION.md)
  distinguishes both fresh rows, mean leaves, form-origin dependence and
  an explicit scalar-storage certificate obstruction. Zero capacity and
  nonzero Gamma retain their separate contracts.
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
The [pressure-completion analysis](nodal/SINE_PRESSURE_COMPLETION.md)
derives the resulting mean/power constraints and a conditional smooth-storage
obstruction. Its alternative models show which premises can be changed; the
result does not require every continuous model to retain every pressure write.

**Continuous evolution versus a finite operator action.** Uniform form gives
`q=0` and zero instantaneous relational phase velocity, even at nonuniform
admitted phase. UM can still move phase in that state. Its finite reset is not
automatically a time step of the current continuous law. Any proposed internal
NFR activation must justify this boundary, not merely pass a storage check.

**Potential contact versus existing support.** UM's search over graph nodes
or a supplied sample assumes candidate access. EN/RA consume existing edges;
they cannot themselves transmit input across a missing edge. THOL children
start isolated. Ordinary UM cannot bootstrap two isolates from its sampled
candidate list because its initial gate needs an existing neighbor. The
[explicit potential-pair extension](#coupling-potential-pair) now admits one
supplied isolated partner, including for a target with old neighbors, through
the same circular gate and augmented participant map. This
does not derive access: potential contact remains distinct from the realized
transport graph and from autonomous occurrence.

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

**Marked occurrence corollary.** Fix supplied inputs and configuration `u`,
including any consumed clock or random draws. Let `S(s,u)` be the *realized*
admitted action after proposal and live admission, or explicit abstention;
let `m(S)` be its retained collective mark. A deterministic marked selector
and event descend through `y=Q(s)` precisely when, for every admitted pair
with equal `Q(s)`, both `m(S(s,u))` and the corresponding projected successor
agree. An abstention retains any selector bookkeeping; its successor is the
unchanged state only under an explicit no-write convention.
Necessity follows by evaluating the proposed marked action at the same `y,u`;
sufficiency follows by choosing any lift, since both outputs are then
independent of it. Before/after maps must be declared if support changes.
This specializes the partial-transition criterion below; it supplies no selector.
Interchangeable fine targets may share one collective mark. If marks are
discarded, this stronger marked criterion is not necessary for an unmarked
state transition alone.

Existing equal-aggregate/different-history controls obstruct marked descent
for a policy that attempts the affected action and distinguishes successful
execution from rejection/abstention in its retained mark. Eligibility alone
does not establish an occurrence difference. A policy choosing a common
admitted action or abstaining on both states may still descend. A continuous
closed quotient alone decides neither case. Retain or derive the required
memory before attributing such selection to the collective NFR.

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
[research program](RESEARCH_PROGRAM.md#research-current-question)
selects the next bounded comparison from these obligations.
A justified negative result remains useful;
the map provides mechanisms to test without asserting autonomous connection,
substrate generation or physical identification.

### Native selector and collective dynamics

Native selector premises, bounded collective predictions and admission
counterexamples are maintained in [Native operator dynamics](NATIVE_OPERATOR_DYNAMICS.md).
The original section links remain available below.

<a id="native-selector-memory-and-occurrence"></a>
<a id="native-selector-memory-and-realized-occurrence"></a>
[Native selector memory and realized occurrence](NATIVE_OPERATOR_DYNAMICS.md#native-selector-memory-and-occurrence).

<a id="native-selector-clock-admission"></a>
<a id="decision-schedule-and-structural-time"></a>
[Decision schedule and structural time](NATIVE_OPERATOR_DYNAMICS.md#native-selector-clock-admission).

<a id="native-selector-environment-inputs"></a>
<a id="disconnected-context-and-the-native-occurrence-boundary"></a>
[Disconnected context and the native occurrence boundary](NATIVE_OPERATOR_DYNAMICS.md#native-selector-environment-inputs).

<a id="native-selector-collective-batch"></a>
<a id="a-restricted-collective-map-of-one-native-selector-batch"></a>
[A restricted collective map of one native selector batch](NATIVE_OPERATOR_DYNAMICS.md#native-selector-collective-batch).

<a id="native-collective-step"></a>
<a id="one-complete-native-step-on-the-synchronized-collective-family"></a>
[One complete native step on the synchronized collective family](NATIVE_OPERATOR_DYNAMICS.md#native-collective-step).

<a id="native-nonuniform-collective-step"></a>
<a id="nonuniform-collective-phase-and-capacity-closure-and-a-selector-boundary"></a>
[Nonuniform collective phase and capacity: closure and a selector boundary](NATIVE_OPERATOR_DYNAMICS.md#native-nonuniform-collective-step).

<a id="native-collective-mean-prediction"></a>
<a id="a-next-call-collective-form-mean-prediction-after-transition"></a>
[A next-call collective form-mean prediction after Transition](NATIVE_OPERATOR_DYNAMICS.md#native-collective-mean-prediction).

<a id="native-reception-envelope"></a>
<a id="propagated-admission-through-the-first-lag-forced-reception"></a>
[Propagated admission through the first lag-forced Reception](NATIVE_OPERATOR_DYNAMICS.md#native-reception-envelope).

<a id="native-capacity-feedback"></a>
<a id="active-capacity-adaptation-exposes-a-discarded-internal-association"></a>
[Active capacity adaptation exposes a discarded internal association](NATIVE_OPERATOR_DYNAMICS.md#native-capacity-feedback).

<a id="native-joint-admission"></a>
<a id="joint-enclosures-distinguish-certified-decisions-from-unresolved-branches"></a>
[Joint enclosures distinguish certified decisions from unresolved branches](NATIVE_OPERATOR_DYNAMICS.md#native-joint-admission).

<a id="native-collective-resonance"></a>
<a id="native-resonance-with-primitive-phase-dispersion"></a>
[Native Resonance with primitive phase dispersion](NATIVE_OPERATOR_DYNAMICS.md#native-collective-resonance).

<a id="native-resonance-u3-boundary"></a>
<a id="fresh-resonance-selection-does-not-imply-live-u3-admission"></a>
[Fresh Resonance selection does not imply live U3 admission](NATIVE_OPERATOR_DYNAMICS.md#native-resonance-u3-boundary).

<a id="collective-grammar-descent"></a>
### From a collective action to an admitted ordered language

The event criterion above concerns a projected state change. Inheriting the
engine's grammar requires a quotient of the information consumed by its
admission and execution, as well as that change. Let `s` be a declared complete
execution state and `y=Q(s)` its collective description. Besides mathematical
state, `s` can include consumed boundary ports, ordered histories, clocks,
grammar context, baselines and configuration. Retain only what the selected
actions consume, but prove that discarding the rest preserves their decisions
and responses. Configuration fixed across the comparison is an explicit
premise. A reproducible random action also retains its generator state or
supplied draws; a seed alone does not determine an arbitrary continuation.

**Partial-transition criterion.** Fix a family of marked actions `a`; the mark
specifies any externally supplied target, parameters and execution path. On a
declared execution domain `E`, let `D_a` be the admission domain and let
`T_a:D_a -> E` include all consumed context updates. Assume `E` contains the
successors needed by every finite continuation being compared. There exists
an exact collective partial action with the same admission decisions and
projected successors if and only if, for every `s1,s2` in `E` with
`Q(s1)=Q(s2)`,

```text
s1 in D_a  <=>  s2 in D_a,
s1,s2 in D_a  =>  Q(T_a(s1)) = Q(T_a(s2)).
```

The first condition says that admission is constant on each collective class;
the second is event descent on its admitted part. Necessity follows by
evaluating the same collective input. For sufficiency, declare its admission
and successor using any lift. Both choices are independent of that lift.
Induction over the actual left-to-right order then gives identical admission
prefixes and collective states for every finite word in the declared family.
The hypotheses must hold at every reachable prefix, not just the initial
pattern. Restricting the claim to one source class therefore still requires
its continuation domain. Conversely, inheritance of all length-one actions
already requires both conditions. If support or state type changes, use the
declared before/after observation maps and corresponding typed domains.

**Outputs are separate obligations.** An exact collective monitor, status or
event cost also requires its consumed output to agree on the same classes.
This includes availability and the declared distinctions between rejection,
unsupported evidence and execution failure; equal final form does not suffice.
For an action with a reference-relative monitor, include its reference and
geometry in the comparison. A cost can instead have a proved class-wide bound,
but that bound is not a uniquely determined cost or an occurrence law. U2's
unit debt cannot substitute for actual reset work. If a failed attempt can
leave writes or completed prefixes, model its outcome and resulting state as
an explicit transition; the partial-map theorem supplies no rollback. These
boundaries follow the selected [event execution contract](../docs/contracts/OPERATOR_EVENTS.md).

For the current grammar this issue is already concrete. Identical nodal
channels and identical bounded glyph traces can have different persistent U2
debt. Identical recent context can also conceal whether an earlier IL occurred,
changing Mutation's lifetime stable-base requirement. The shared
[context owner](../src/tnfr/operators/grammar_debt.py) and its
[eviction controls](../tests/operators/test_grammar_debt_and_depth.py) retain
both facts. Mutation's measured signed EPI history, U3's live phase/neighbor
gate, Recursivity's actual depth and U6's reference require their own consumed
information. A validated future handler does not manufacture executed history.
The [grammar owner](UNIFIED_GRAMMAR_RULES.md#8-composition-and-implementation)
specifies these existing contracts; no sum or average of constituent histories
is established here as a collective grammar state.

The [weighted phase-offset family](nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-capacity-descent)
supplies another precise primary-map boundary: on its connected positive
support, the post-reset law preserves the entire family exactly when actual
capacity ratios are constant inside each block. This
includes nonuniform initial capacities. A singleton reset can retain current
family membership while breaking future tangency. This classification
does not replace the context obligations above or admit a complete SHA
event from collective form and phase alone.

A pointwise pair quotient, or an AL map that descends on it, is consequently
not yet a quotient of the full operator runtime. Conversely, a continuous
collective model need not carry unused runtime history merely to define its
mathematical dynamics. The claim determines the information obligation.
The [weighted-star sufficient chart](nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-star-state-sufficiency)
is a concrete continuous example: retained internal coordinates and a dynamic
boundary reconstruct all consumed flow rows, while its detached report
supplies no measured histories, debt, hierarchy or lifecycle. Such a report
cannot by itself admit any registered operator or select its occurrence.
Retaining every constituent and its context is a valid description, but does
not by itself establish a smaller autonomous state, formed collective identity
or generative completeness. The criterion inherits specified operations and
words; it does not select which operation occurs, derive their policies from
the nodal row, or prove that the 13 registered actions generate every possible
TNFR transformation.

The [uniform-capacity native sector](NATIVE_AGGREGATE_OPERATORS.md#native-uniform-capacity-transition)
provides a restricted positive instance with actual default IL/AL/EN selection:
retained grammar, lag and controller state admit an ideal projected primary
transition without retaining fine form contrast. Aggregate margins define
its stopping boundary. Its completed native controls preserve unequal fine
stability counts; the result does not identify complete histories or outputs.
The [horizon proof](NATIVE_AGGREGATE_OPERATORS.md#native-uniform-capacity-horizon)
uses the actual six-call Emission recurrence to bound that affine domain.
This recurrence belongs to the configured selector, not a derived occurrence law.
At [hard clipping](NATIVE_AGGREGATE_OPERATORS.md#native-uniform-capacity-clipping),
the extended pair map retains internal spread and the incoming held-pressure
association. The zero-spread slice inherits ordinary coarse execution, while
a specified Emission/Reception clipping branch excludes universal mean-only inheritance.
The [limit theorem](NATIVE_AGGREGATE_OPERATORS.md#native-uniform-capacity-limit)
combines these rows with actual lag recurrence to exclude persistent
nonuniform form/phase in that sector; accepted glyph occurrences and context
updates continue even when the primary coordinates are stationary at saturation.
Its [capacity robustness extension](NATIVE_AGGREGATE_OPERATORS.md#native-capacity-robustness)
retains that conclusion with unequal fine capacities in a derived nonzero
neighborhood. Hull preservation first keeps selection in AL/EN/IL; a strict
pressure bound then proves permanent adaptation qualification and consensus.
The event policy remains supplied, and the proof does not extend the smaller
uniform-capacity aggregate state to arbitrary capacity heterogeneity.

**Retain associations, not independent inventories.** If a pair swap removes
member labels, it must simultaneously transport each member's state, ports,
history and consumed lifecycle data. Two separate unordered lists of physical
states and histories lose their association. In the
[live-history control](../tests/operators/test_collective_pair_stage_reuse.py),
reassigning the same measured histories between unequal current forms leaves
both independent inventories unchanged but makes their endpoints stale.
The correct comparison is the orbit of the jointly associated data. This
retains information without asserting that a smaller minimal state exists.
Configuration, externally supplied clocks and any consumed node-order policy
must also transform or be fixed under the declared symmetry. A timestamp
record is not the structural evolution clock.

The [complete selected-pair AL/SHA word](nodal/SINE_PAIR_WORD.md) verifies
these obligations on one explicit positive-width physical source domain and
associated lifecycle stratum. Live admission holds at both prefixes; the
primary resets have an exact binary64 subdomain, and the postword sine law
retains the existing geometric identity. This is a supplied two-step action,
not closure under arbitrary words, histories or repeated interventions.

The [AL/RA/SHA extension](nodal/SINE_PAIR_RESONANCE.md) adds a relational
action whose neighbor resultant depends on retained internal phase geometry.
It proves exact-real descent, conditional RA passivity in the sine storage,
and continued identity on its declared all-neighbor domain. Full represented
stages have separate finite controls and arithmetic defects; this result
does not assert general floating-point permutation invariance or transfer
passivity to arbitrary U3 subsets and adjacent target sets.

The [same-law relative-phase response](nodal/SINE_PAIR_INTERACTION.md#sine-pair-autonomous-relative-phase)
also distinguishes the internal preparation without executing any operator.
Its finite-window bound retains the complete constituent dynamics and small
preparation/readout errors. This connects a state dependency of RA with an
autonomous observable; it does not derive the event's blend, capacity change,
grammar history or occurrence from elapsed continuous time.

An existing positive precedent is
[P5 history composition](DERIVED_EPI_MEMORY.md#95-exact-composition-with-a-declared-temporal-echo):
uniform delayed mixing, history shifts and specified linear flow maps commute
with its retained spatial projection. Its exact-real theorem, represented
rounding defects and clipping counterexample separate three different claims.
The proof method transfers; its linear coefficients and delay assumptions do
not establish a nonlinear pair word. Conversely, clipping need not obstruct
the lossless pair orbit: the existing AL theorem admits any shared point map
on its declared domain. Generic quotient diagnostics remain
[sampled or linear observations](../src/tnfr/physics/operator_quotient.py),
not certificates for graph mutation or complete runtime grammar.

The [marked-contact NUL/IL/SHA transition](nodal/SINE_COLLECTIVE_IDENTITY_TRANSITION.md)
instantiates the criterion on an exact eleven-cell physical stratum with
retained ordered constituent context. Its one-word domains distinguish
organizational class from predictive state: the ideal word preserves an
explicit acute winding sector and finite U6 bounds while changing form and
capacity. Represented controls show why a simple eleven-node graph, centered
form or bounded history without persistent debt is insufficient. The output
has its own domain; neither repetition nor subsequent sector maintenance
follows. This does not alter the earlier frozen joint-radius departure.

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
