# TNFR API and execution contracts

**Status:** Active normative view
**Version source:** [package metadata](../pyproject.toml)

This hub owns shared API admission and execution boundaries and the generated
operator-contract view. The [operator registry](../src/tnfr/operators/operator_contracts.py)
owns operator metadata; the linked engine, observation and validation modules
own their respective execution paths. A registry change alone does not revise
an integrator, diagnostic or certificate contract.

| Responsibility | Contract owner | Implementation boundary |
| --- | --- | --- |
| Common admission, solvers and named operator contracts | This document | Shared scalar/clock owners, nodal integrators and operator registry |
| Conditional relational dynamics and observations | [Relational dynamics](contracts/RELATIONAL_DYNAMICS.md) | Selected joint law, snapshot/step reports and capture/transit admission |
| Hybrid events and finite execution evidence | [Operator events](contracts/OPERATOR_EVENTS.md) | Declared schedules, transactions, runtime provenance and REMESH certificate APIs |

For executable calls and examples, use the [CLI and SDK guide](CLI_AND_SDK.md).
Mathematical definitions, hypotheses and proofs remain with the linked theory
owners. A serialized declaration or diagnostic report does not bypass live
preconditions or certify future stability.

## Contract model

Every canonical operator has:

- a lowercase English execution token, a public display/class name, and an
  internal glyph;
- one primary nodal-equation channel;
- node or network scale;
- a direct-effect direction;
- a verifiable postcondition;
- independent grammar and state preconditions.

The channel classification does not replace grammar roles. For example,
Mutation acts primarily on phase while also being a U2 destabilizer and U4
transformer.

`Scale` is the U5 fractality axis: `node` acts at the current fiber/level and
`network` denotes the multi-scale REMESH echo. It is not an execution-footprint
field. Coupling remains node-scale on this axis even though one application may
write neighbouring phases and edge support; those overlaps belong to the
separate stage contract.

### Nodal arithmetic and input admission

The optional `validate_nodal_equation` helper compares supplied EPI endpoints
with one unforced Euler step holding the current capacity and pressure fixed.
THOL proposal validation reuses this same arithmetic and configuration owner.
It neither authenticates an executed interval nor converts a named operator
jump into continuous nodal flow. A boundary-clipped match can occur with no EPI
motion despite a nonzero unprojected rate. The shared default tolerance is
`NODAL_EQUATION_TOLERANCE`; compatibility units are EPI for clip-aware comparison
and EPI/time for rate comparison. Invalid time, coefficients or active policy
raise errors. See the [parameter foundations](../theory/NODAL_PARAMETER_FOUNDATIONS.md#11-thresholds-constants-and-numerical-settings-have-different-duties)
for numerical-policy scope and ownership.
The generic public-operator comparison runs after its event and supplies no
rollback guarantee; THOL checks its detached proposal before committing it.

Positive-duration default and extended integration resolve the same clipping
policy through `dynamics.structural_clip.resolve_clip_policy`. Invalid modes,
bounds or soft-knee gain reject before rate evaluation, including on frozen
rows; an invalid mode never silently selects hard clipping. Zero-duration
execution retains its no-op contract.

The detached `TNFRUnifiedBackend` nodal proposal uses the shared rate/Euler
arithmetic. It reports `epi_step_scope="unforced_unclipped_proposal"` and
`stability_not_certified=True`: it does not execute Gamma, clipping or history
writes. Stored-state fields must retain their scalar numeric types; Boolean
and textual values are rejected before coercion. Finite inputs with a
nonrepresentable rate or endpoint reject instead of returning a successful
infinite state. A temporal integration request has its own full integrator
contract; it is not this detached proposal.

The backend's temporal request validates the raw step, integral step count and
Boolean trajectory flag before dispatch. Stored-pressure execution preflights
the entire supplied clock grid, including internal subdivisions, through the
integrator's own preparation and clock checks. A later inevitable clock overflow
or rounded nonadvance therefore rejects before the first state write. Each step
still admits live state/configuration; the preflight neither invokes sources nor
makes future callbacks or state-dependent failures globally transactional.

Validated nodal scalar inputs use the shared represented-real admission:
Boolean/text values and nonzero values that disappear during binary64
materialization reject. Capacity admits zero and has no arbitrary finite upper
cutoff; the supplied product and returned derivatives must still be finite.
The extended model reuses the same EPI derivative owner. Validation certifies
these numeric domains, not the model's physical derivation; an explicitly
unchecked call does not acquire that certificate.

Scalar operator inputs use the same admission before coercion, including
incoming form consumed from a neighbor. Runtime Mutation-history recording
also requires signed scalar EPI; a rich BEPI magnitude cannot become observed
scalar motion. Every node's sample and retained history are admitted before
that recording boundary writes any node's history.

Default pressure and optional EPI/capacity hooks reuse nodal capacity admission;
phase pressure admits the authoritative finite phase before trigonometric
conversion. Malformed primary aliases cannot be replaced by a valid secondary
spelling. Invalid state rejects before stored pressure is written, including
after cache reuse and on isolates. Preparation caches are not a rollback
boundary. The phase-only hook requires only its consumed phase coordinates.
Registered pressure hooks and runtime refresh share one optional-`n_jobs`
dispatcher. Legacy hooks without that keyword remain supported; a `TypeError`
inside a Python callback body propagates without replaying its side effects.
Uninspectable callables retain the traceback-based argument-binding fallback.
This dispatch boundary does not promise rollback of writes already made by the hook.
The finite forcing observer uses the runtime's own vectorization predicate to
exclude every disabled NumPy route; alternative accumulation paths are not
silently identified with its certified numerical kernel.

Default pressure mixtures and selector score weights use the same strict
nonnegative represented-real normalization. Invalid coefficient types or
negative values reject rather than becoming zero or a different default mix.
Large finite coefficients retain a finite normalized mixture even when their
raw sum overflows. An all-zero mix still selects uniform weights by explicit
compatibility policy; it is not a request to disable the corresponding model.

### Stored observations and validation

Global and local structural coherence share stored-alias admission, finite mean
reduction and the reciprocal diagnostic in `metrics.common`. The radius-local
observation includes its center; the legacy immediate-neighbor observation
excludes it and returns zero at isolates. Invalid consumed aliases reject
without becoming zero or falling through to another spelling. These are
different declared observation scopes, not different coherence laws; neither
refreshes pressure or reconstructs an observed EPI rate.

Local structural and phase observations share one center-inclusive support-ball
reader: the center must exist and radius must be a nonnegative integer, excluding
Booleans. Phase order reuses admitted trigonometric values and the shared
Kuramoto magnitude; malformed present phases cannot be skipped to report perfect
alignment. Its legacy empty-graph value of one is a convention, not evidence.
A zero resultant has zero magnitude but no direction. Pairwise linear phase
affinity is a configured score, distinct from structural C and live U3 admission.

The shared coherence-value boundary checks the original real input in `[0, 1]`
before accepting its finite binary64 representation. Rounding a value above one
to one, or a nonzero value to zero, cannot make it admissible. The SDK and
configured validation reports reuse this boundary. `TNFRUnifiedValidationSystem`
memoizes only typed immutable inputs together with the consumed policy, and
returns detached results. Text cannot reuse a numeric result; changed bounds or
string patterns require their own evaluation. Both caching switches must be
enabled for reads and writes. Its frequency cap and phase-range warnings are
application policy, not additional constraints on the nodal identity.

`TNFRValidator.validate_inputs()` delegates to the existing typed input helpers
and returns admitted values for supplied form, capacity, phase, pressure, node,
glyph and graph inputs. Failures raise `ValidationError` or populate `error`
when `raise_on_error=False`; the aggregate `validate()` consumes that failure.
Its legacy `config` argument does not override the shared adapter's policy.
Graph-interface admission does not validate an entire trajectory or operator word.

The aggregate validator admits Boolean check flags and operator graph/target
context before dispatch; missing context cannot produce a successful skipped
operator check. Runtime validation explicitly opts into the existing mutating
clamp pass. Disabled passes are tagged `skipped`; reported runtime/invariant
failure respects `raise_on_error`. Live graph checks always run afresh: the legacy
cache switches remain compatibility arguments, since graph identity cannot
capture changing state, selected checks or arbitrary custom-validator inputs.
Returned violation records are detached while preserving node-label identity.

`run_structural_validation()` and its health summary distinguish field
availability from threshold flags. Unavailable comparisons have no passing
Boolean; empty support, undefined curvature or invalid thresholds cannot yield
low risk. Independent valid alerts remain active. Reports retain comparison
status/reasons and xi estimator provenance. These observers do not advance nodal
state; shared field and geometry owners may maintain rebuildable graph caches.

Coherence level-set certificates additionally reject a non-equilibrium input
that rounds to the exact equilibrium level. Their fixed capacities use the
shared represented-real boundary. Retrospective learning efficiency is absolute
scalar EPI change per retained operator count, not evidence of learning quality;
its initial state and count must refer to the same observation window.

OZ topology telemetry compares explicitly captured before/after heterogeneity
scores. Without a preceding observation, its delta and change flags are `None`.
The shared direct/stage lifecycle captures this diagnostic only when metrics
are requested; nodal validation alone does not require topology telemetry.
The native pressure operation preserves support; an asymmetric snapshot alone
does not establish an operator-induced change or mathematical symmetry breaking.

### Configured feedback and adaptation

`StructuralFeedbackLoop` reads signed scalar form and radius-local coherence
through their shared owners. Invalid state or consumed mutable policy rejects
before operator selection. Its adaptation accepts a finite signed performance
measure, checks the represented update, then applies the existing threshold
clamp; NaN or overflow cannot become a successful clamp. Each cycle measures
one pre-action and one post-action state without initializing an unused backend.
This remains a supplied controller, not an emergent operator-selection law or
a transaction covering arbitrary later operator/callback failures.

Phase coordination validates every proposed phase before its first phase
write on NumPy, scalar and worker paths. Only its exact-reduction transaction
also restores graph-owned caches/history after a late failure. Capacity
adaptation applies raw represented-real admission before eligibility checks;
neither change selects a phase law or derives its configured diagnostic gates.
Missing Si or pressure makes the capacity gate unavailable: its consecutive
stability count resets and that node's capacity is retained. Missing capacity
rejects before mutation because the neighbor snapshot consumes it. Explicit
invalid values, including `None`, still reject; absence is not a zero-valued
measurement. The policy reads stored inputs and does not refresh them.
It is a per-invocation event, not a continuous capacity law. Preservation of
a supplied relation `nu=g(x)` has separate
[flow and event obligations](../theory/CAPACITY_LOCALIZATION_BALANCE.md#capacity-law-admission).

Coherence and equilibrium observations apply this admission before scalar or
array coercion. Stored pressure/rate/capacity readers honor the first present
alias: an invalid authoritative value cannot fall through to a later alias or
become zero. Missing attributes retain their documented defaults. Stability
tolerances must be finite nonnegative real values even on empty support;
their positive cuts describe a diagnostic neighborhood, not exact equilibrium.
Sense-index normalization shares these reads on Python and NumPy paths,
including cache refresh. This changes invalid-input handling, not the Si law
or its status as a configured diagnostic used by existing controllers.

### Continuous integration paths

The older `integrate_canonical_nodal_equation` convenience API also uses the
shared derivative/Euler arithmetic but holds pressure and capacity throughout
its loop. Both accepted method names describe that same constant-slope map;
its step-change stopping criterion is not equilibrium. Its legacy missing-value
defaults (unit capacity, zero form/pressure) are initialization policies, and
it writes neither the runtime clock nor derivative history. Invalid candidate
states reject before commit. The default runtime's `rk4` path, in contrast,
samples the declared Gamma forcing at the quadrature times while holding the
nodal base fixed. Shared quadrature handles exceptional floating-point ranges;
this does not establish fourth-order accuracy for arbitrary changing pressure.

The default additive-forcing integrator evaluates declared Gamma sources strictly
on positive-duration calls:
an invalid source cannot silently become the unforced model on one backend.
The live Gamma registry owns dispatch. Built-in array formulas apply only to
their unchanged registered implementations; custom or replaced entries use
the staged scalar path, once per node and quadrature sample. These callbacks
read the live graph, not hypothetical intermediate EPI states. Kuramoto-based
forcing refreshes its phase-dependent cache even when the requested time is
unchanged. The permissive public `eval_gamma(..., strict=False)` read retains
its explicit zero-on-error compatibility behavior; it is not solver admission.
Callback effects outside the integrator's documented write boundary are not
made transactional by source dispatch.
The opt-in extended EPI/phase/pressure model is a separate unforced law; these
Gamma dispatch rules do not add a source to that model.

## Canonical contracts

Direct and staged Emission/Silence share structural and lifecycle proposals.
Emission rejects a configured clipped result that would decrease the supplied
EPI before writing form or lifecycle metadata. For example, soft clipping of
`-0.99 + 0.001` produces approximately `-0.995113`, outside AL's nondecrease
contract; hard clipping admits the corresponding `-0.989` result. The selected
clipping law itself is unchanged. Silence validates nonnegative raw capacity
even when its attenuation factor is zero. Expansion/Contraction likewise
validate consumed raw state and any active telemetry sink before primary
writes. These preflights do not make arbitrary later callback failures atomic.

This table is generated from the registry, including its declared measurement
context. Refresh it with `python scripts/check_documentation.py --write-generated`;
the normal documentation gate checks exact agreement. It specifies metadata and
contracts, not a proof that every execution path satisfies a global theorem.

<!-- BEGIN GENERATED OPERATOR CONTRACTS -->

| Operator | Token | Glyph | Primary channel | Scale | Context | Registered postcondition |
| --- | --- | --- | --- | --- | --- | --- |
| Emission | emission | AL | EPI | node | network | EPI not decreased; νf, phase and ΔNFR unchanged |
| Reception | reception | EN | EPI | node | network | Immediate operator-local C(t) unchanged; ΔNFR and dEPI unchanged |
| Resonance | resonance | RA | EPI | node | identity | EPI structural identity (sign/kind) preserved |
| Silence | silence | SHA | nu_f | node | network | νf not increased; EPI, ΔNFR and phase unchanged during SHA |
| Expansion | expansion | VAL | nu_f | node | network | νf not decreased (capacity added) |
| Contraction | contraction | NUL | nu_f | node | network | νf not increased (capacity removed) |
| Coupling | coupling | UM | theta | node | network | &#124;ΔNFR&#124; not increased (mutual stabilization) |
| Mutation | mutation | ZHIR | theta | node | phase | θ transformed (θ → θ') |
| Coherence | coherence | IL | delta_nfr | node | network | &#124;ΔNFR&#124; not increased and C(t) not decreased |
| Dissonance | dissonance | OZ | delta_nfr | node | node | &#124;ΔNFR&#124; not decreased |
| Self-organization | self_organization | THOL | delta_nfr | node | network | parent EPI, nu_f and phase fixed; DeltaNFR follows signed acceleration; nested child creation preserves parent identity |
| Transition | transition | NAV | delta_nfr | node | state | state changed (νf, θ, or ΔNFR) |
| Recursivity | recursivity | REMESH | EPI | network | advisory | node-level advisory; network effect = EPI mixed toward temporal/multi-scale history |

<!-- END GENERATED OPERATOR CONTRACTS -->

The REMESH row distinguishes the advisory glyph from the separately invoked network operation
apply_network_remesh. Its public planner, plan_network_remesh, returns an
immutable all-node proposal. Insufficient history or empty live support is an
explicit no-op; after the history guard passes, each selected temporal snapshot
must be a node mapping with
exactly the live support. Missing values are never replaced by current EPI.

The executor returns an immutable result that exposes raw affine and bounded
values separately. Optional evidence reports the three-snapshot convex
disagreement bound, weighted-mean drift and only sufficient fixed-history gain
conditions. These fields do not infer stability from the operator name. The
commit is atomic over graph-owned state, topology, metadata, history, caches
and capturable callback state. Effects already emitted to external systems by a
callback cannot be rolled back.

A direct node-level Recursivity glyph and its shared word stage are
advisory-only: the stage derives one immutable advisory from its snapshot,
deduplicates one graph event per telemetry step and leaves structural channels
unchanged. It never triggers delayed EPI mixing implicitly.

The optional memory helper `propagate_structural_identity` is a separate
hybrid jump. It interpolates signed scalar form and nonnegative capacity and
uses the shortest circular phase arc, with the shared signed antipodal tie
convention. It does not solve continuous nodal flow or certify persistence.
Strength must be a finite represented real in `[0,1]`; zero is a no-op.
Every unique non-origin target is validated before writes, including consumed
triad coordinates, clipping policy and lineage sink. Missing coordinates and
invalid strengths no longer acquire zero/default interpretations; invalid
clipping modes reject. Materialized uniform-real BEPI is admitted in the scalar
chart; richer or serialized form payloads are not. Convex form/capacity
proposals round once from represented inputs and reject nonzero underflow.
Lineage records actual endpoints and the step ordinal, while stored pressure
and the physical clock are retained. Input preflight does not promise rollback
of arbitrary custom mapping/setter failures or the entire memory wrapper.

For exact wording and measured context, inspect
[`OPERATOR_CONTRACTS`](../src/tnfr/operators/operator_contracts.py).

## Six global invariants

The single working definition is [AGENTS.md, canonical invariants](../AGENTS.md#8-canonical-invariants).
This API reference states path-specific preconditions, effects and evidence;
it does not maintain a second list or strengthen those invariants.

## Valid execution example

```python
from tnfr.operators.definitions import Coherence, Emission, Silence
from tnfr.structural import create_nfr, run_sequence

G, node = create_nfr("seed", epi=0.1, vf=1.0, theta=0.0)
run_sequence(G, node, [Emission(), Coherence(), Silence()])
```

The word starts with a U1 generator, contains a stabilizer, and ends with a U1
closure. State-dependent operator preconditions remain mandatory during
execution. Coupling and Resonance additionally enforce U3 before mutation.

Resonance's optional strict preconditions and readiness report share admitted
state and finite nonnegative policy thresholds. Invalid thresholds reject
before direct or atomic-stage writes. The readiness flag covers the configured
form-magnitude, capacity, pressure-magnitude and connectivity checks; it does
not replace U3 or authorize execution. Its neighbor-mean phase warning retains
an unavailable direction at an exactly zero represented resultant, while invalid
consumed phases raise. These reads do not write graph caches.

## Nodal solver input, clock and output boundaries

The shared [nodal integrator](../src/tnfr/dynamics/integrators.py) reads the
authoritative capacity, pressure and previous derivative aliases: invalid
provided values cannot fall through to a later alias or a zero default.
Capacity must be finite and nonnegative; pressure and retained derivatives
must be finite represented reals. Missing values retain the documented defaults.
The step, initial time and minimum step use the same raw scalar admission:
Boolean/text inputs and nonzero values lost to zero during conversion reject.
This retires the previous Boolean-timestep compatibility; use explicit numeric
`0.0` or `1.0` when those durations are intended. Admitted zero-duration calls
remain no-ops.

Every positive internal substep must advance a finite represented clock;
checking only the requested final duration is insufficient under subdivision.
Empty and populated graphs use the same repeated-add time convention.
Projected EPI and stored rate/acceleration must be finite before commit.
The scalar and optional extended paths restore solver-owned node outputs if
a later substep fails; callback side effects and diagnostic caches are outside
this local restoration. The [event executor](contracts/OPERATOR_EVENTS.md#schedules-and-transaction-boundaries)
supplies its broader transaction.
Clipping still separates the stored unconstrained rate from the realized EPI
secant. These checks establish neither solver accuracy nor a physical clock.
Controls: [integrator numerics](../tests/test_integrator_numerics.py).

Active built-in Gamma sources validate authoritative phase aliases before
trigonometric caching or array conversion. A vector source requires one phase
per node; a malformed primary alias cannot be hidden by a valid secondary one.
Scalar and array Gamma APIs validate the supplied clock. The unforced `none`
source does not consume phase, and custom registry entries retain their declared
input dependencies. The shared trigonometric cache validates current raw phase
values on every read, including when they would coerce to a cached value.

U3 phase/limit admission, Coupling coefficients and consumed known numeric glyph
factors use the shared represented-real boundary. A tiny nonzero factor cannot
silently select a zero-valued policy; unknown extension keys remain outside
the canonical factor schema. Direct and simultaneous Resonance reuse one capacity proposal: finite
nonnegative capacity is required even when amplification is inactive. Graphless
unidirectional Coupling updates only the target phase. On graph paths, UM/RA
means use unique outgoing support neighbors, including zero-weight edges;
parallel multiplicity and incoming arcs do not supply additional samples.
These remain operator policies, distinct from weighted EPI transport. The
[coupled-state controls](../tests/operators/test_coupled_state_admission.py)
also verify support-cache invalidation after functional-link creation.

## Conditional relational execution

The maintained contract is [Conditional relational execution](contracts/RELATIONAL_DYNAMICS.md#conditional-relational-execution).

### Exact linear observations of a supplied generator

See [Exact linear observations](contracts/RELATIONAL_DYNAMICS.md#exact-linear-observations-of-a-supplied-generator).

<a id="relational-pattern-observation"></a>

### Prepared relational pattern observations

See [Prepared relational pattern observations](contracts/RELATIONAL_DYNAMICS.md#prepared-relational-pattern-observations).

<a id="relational-attachment-observation"></a>

### Supplied relational attachment observation

See [Supplied relational attachment observation](contracts/RELATIONAL_DYNAMICS.md#supplied-relational-attachment-observation).

### Supplied joint state and support reset

See [Joint reset accounting](contracts/RELATIONAL_DYNAMICS.md#relational-reset-observation).

### Conditional relational capture

See [Conditional relational capture](contracts/RELATIONAL_DYNAMICS.md#conditional-relational-capture).

### Validated conditional relational transit

See [Validated conditional relational transit](contracts/RELATIONAL_DYNAMICS.md#validated-conditional-relational-transit).

### Exact cycle resultant sectors

See [Exact cycle resultant sectors](contracts/RELATIONAL_DYNAMICS.md#exact-cycle-resultant-sectors).

## Operator-event timeline

The maintained [operator-event and finite-evidence contract](contracts/OPERATOR_EVENTS.md#operator-event-timeline)
owns schedules, runtime transactions, flow/stage observations, REMESH composition
and the scoped certificate APIs. The headings above retain earlier incoming links;
full contracts are maintained only in the linked chapters.

## Contract verification

The [testing guide](../TESTING.md) owns selection and execution commands.
Core operator regression entry points are:

- [Registry and direct effects](../tests/operators/test_operator_contracts.py).
- [Circular U3 rejection before mutation](../tests/operators/test_u3_hard_invariant.py).
- [Public operator behavior](../tests/operators/test_canonical_operators_modern.py).

Use the [grammar verification map](../theory/UNIFIED_GRAMMAR_RULES.md#9-verification-and-reporting)
for U1–U6, the [REMESH proof owner's direct checks](../theory/REMESH_INFINITY_DERIVATION.md#22-reproducibility-and-direct-checks)
for history/schedule/certificate controls, and the
[theory-to-implementation map](../theory/README.md) for other model-specific
controls. Those owners retain the detailed test obligations; this hub does
not duplicate their inventories. Finite regression checks are not unrestricted
stability proofs or physical validation.

## Extension rule

The catalog is fixed at 13 canonical operators. A domain feature should compose
existing operators or remain a diagnostic/morphism outside the catalog. Any
proposal to alter the catalog requires a nodal-equation channel, scale,
postcondition, grammar classification, tests, and an update to the canonical
synthesis; adding a class or registry entry alone does not establish
canonicity.

## Auxiliary spectral-expectation contract

`SpectralExpectationOperator` evaluates the Hermitian quadratic form
`<psi|A|psi>`. For a unit-norm state, including the default
`expectation(..., normalise=True)` path, its exact-real value lies in the
spectral interval of `A`; numerical evaluation retains floating-point error.
With `normalise=False`, an unnormalized input instead scales that interval by
`||psi||^2`: `A=diag(2,3)` and `psi=(2,0)` give `8`, not a value in `[2,3]`.
Values above one are valid in either mode. This auxiliary value is never the
structural coherence `C(t)`, never inherits a `[0,1]` bound, and never enters
`C_steps`. Payload metadata `range="unbounded_real"` and `bounded=False`
describe the family of supplied observables; they do not negate the conditional
spectral bound for a fixed Hermitian operator and normalized state.

`NodeNX`, `create_math_nfr`, the dynamics runtime and the CLI expose the
canonical names `spectral_operator`, `spectral_expectation_threshold` and
`spectral_operator_expectation`. Every result payload contains `value`,
`threshold`, `passed`, `metric_kind`, `range`, `bounded`,
`provenance`, `canonical_coherence_certified=False` and
`records_to_C_steps=False`.

Historical `coherence_operator`, `coherence_threshold`, `coherence_value`,
`coherence_passed` and runtime history keys remain explicit compatibility
aliases. They mirror the same auxiliary spectral value and carry no `C(t)`
meaning. Supplying contradictory canonical and historical inputs is rejected.

The CLI accepts `--math-spectral-expectation-spectrum`,
`--math-spectral-expectation-floor` and
`--math-spectral-expectation-threshold`. The former
`--math-coherence-*` spellings remain accepted as aliases.
