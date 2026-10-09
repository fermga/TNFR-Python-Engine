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
For calculations that consume conditional sine reports, the
[chained-report contract](contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission)
owns primitive re-admission, derived-field rebuilding and source-association
limits. Public adapters delegate to those shared consumers.
The [relational contract index](contracts/RELATIONAL_DYNAMICS.md) distinguishes
complete-state observations, exact collective families, finite maintenance
bounds and observations of saved forecasts. Sharing a storage identity or
export format does not make their support, law or evidence interchangeable.
The [conservative source-admission readers](contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-cycle-sector-barrier)
can exclude formation by a geometric storage barrier or by finite rapid-contact
averaging. They assess the declared source without evolving it; an unavailable
exclusion is not a formation certificate.
The [directed saddle-corridor reader](contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-directed-saddle-corridor)
instead certifies a finite winding passage from strict nonlinear storage and
momentum bounds on an exact symmetric source. It supplies neither an acute
retention certificate nor an independently uncertain source box.
The [same-orbit formation reader](contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-conservative-formation-retention)
combines both nonlinear passages with a retained band and full-state error
bounds. It certifies conditional existence; its event-defined centers remain
unavailable as numerical preparations and its captured-state flags stay false.
The [retained-metric forecast](contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-saddle-metric-forecast)
instead advances an actual admitted source with full-coordinate uncertainty
and checked whole-time phase domains. Its finite enclosure and any partial
coverage remain distinct from a formation or retention verdict.

## Contract model

### Word and operator controls

Word admission rejects unregistered operator identifiers before role checks;
operator instances retain their metadata, including Recursivity depth. Grammar
context flags use the shared Boolean parser: `"false"` does not grant initialized
form or diagnostic permission, and unknown Boolean strings reject. Context is
copied, leaving the caller's mapping unchanged. See the
[grammar contract](../theory/UNIFIED_GRAMMAR_RULES.md#8-composition-and-implementation).

`GrammarValidator.validate` and `collect_grammar_errors` consume the same
immutable outcomes from `GrammarValidator.validate_checks`; diagnostic text
does not decide whether a check passed. Strings use shared identifier normalization; supplied
operator objects retain their metadata, so unknown identifiers report `SYNTAX`
and Recursivity depth remains subject to U5. `run_structural_validation` uses
these errors for its word status. Omitting the sequence skips word validation;
live U3 admission and temporal U6 evidence remain separate obligations.

Coupling's four `UM_*` branch flags and `OZ_NOISE_MODE` use the same Boolean
parser before active-factor resolution and proposals. Transition's `NAV_STRICT`
and `NAV_RANDOM` retain their stricter Boolean-only domain across public classes,
direct glyphs and stages; strings are rejected. Disabling a branch removes only
that branch's factor requirements. It does not waive live U3 or other operator
preconditions. The [operator maps](../theory/STRUCTURAL_OPERATORS.md) and
[event contracts](contracts/OPERATOR_EVENTS.md) specify the execution effects.

### Removed unsupported helpers

These removals have explicit replacements or scope boundaries; no compatibility
alias substitutes a different mathematical model. The
[recovery index](../theory/research/archive/README.md#recovery-records)
retains the removed source identities.

| Removed interface | Migration or retained contract |
| --- | --- |
| `validate_grammar(..., collect_unified_telemetry=...)` | Remove the argument; word validation has no observed graph. Use `diagnose_network(actual_graph)` or `physics.fields.compute_unified_telemetry(actual_graph)` with explicit state. The removed option reported fields of a fixed demonstration graph, not the validated word or its execution. |
| `tnfr.operators.algebra.validate_identity_property`, `validate_idempotence`, `validate_commutativity_nul` | Use the actual [operator event contracts](contracts/OPERATOR_EVENTS.md) and compare the same admitted transformations. The retired routines compared different words or unsupported words; they did not prove the advertised algebra. Silence attenuates capacity and is not generally idempotent. |
| `tnfr.dynamics.DynamicLimits`, `DynamicLimitsConfig`, `compute_dynamic_limits` and `tnfr.dynamics.dynamic_limits` | No replacement emergent-limit law. The unused score-based proposal mixed configured coefficients and a fabricated missing-Si baseline. Declare application bounds explicitly through the model's existing configuration and admission. |
| `tnfr.compat` and `tnfr._compat` | Import `dataclass` from `dataclasses` and `TypeAlias` from `typing` on supported Python versions. NumPy is required; optional dependencies use their actual libraries and the engine's existing dependency admission. Unused fake-library stubs are removed. |
| `tnfr.secure_config` | Import the same implementations from `tnfr.config.security`. |
| `evaluate_bifurcation_risk` from `tnfr`, `tnfr.math`, `tnfr.mathematics` and `tnfr.math.symbolic` | No bifurcation detector replaces it. A threshold on supplied acceleration neither proves stability nor justifies automatic operator selection. `compute_second_derivative_symbolic()` retains the product-rule identity; actual event admission belongs to the operator contracts. |
| `tnfr.operators.discover_operators`, `tnfr.operators.registry.discover_operators` and `structural_operator` | Remove the no-op discovery calls and decorator. `get_operator_class()` retains lazy class lookup and explicit subclass registration; `TNFR.operators()` owns the canonical contract inventory. Registering a class does not create a canonical operator contract. |
| `tnfr.operators.registry.invalidate_operator_cache` and `get_operator_cache_stats` | Removed unused telemetry; the invalidation function did not clear a cache. Operator class lookup and actual graph/state cache invalidation keep their separate owners. |
| `tnfr.math.grammar_validators`, `tnfr.math.optimizer` and their facade exports | Use `tnfr.operators.grammar` for word admission. Removed duplicate role scores and a greedy score-based word extension; neither certified dynamical stability nor an optimum. Application objectives remain supplied policies. |
| `tnfr.math.fields_symbolic` and its facade exports | Use `tnfr.physics.fields` for the implemented graph diagnostics. The removed continuum expressions were a different model, not symbolic versions of those fields. Nodal calculus in `tnfr.math.symbolic` remains available. |
| `tnfr.riemann.operator_catalog_discipline_signature` and its facade exports | Use `TNFR.operators()` and `operator_contracts.verify_contract_consistency()` for canonical inventory checks. Those checks do not prove ontological completeness; compatibility class registration remains separate. |
| `tnfr.operators.grammar.validate_sequence_cached`, `clear_validation_cache`, `get_validation_cache_stats` and `get_grammar_cache_stats` | Use `validate_grammar()` for operator objects, `validate_sequence()`/`parse_sequence()` for canonical names, and `GrammarValidator` for detailed configured checks. The retired cache wrapper did not validate live graph context; node/edge counts cannot replace live preconditions. |
| `tnfr.utils.clear_orjson_param_warnings` (also in `tnfr.utils.io`) and `tnfr.config.presets.legacy_preset_guidance` | Remove these no-op calls. Shared JSON admission/encoding and `get_preset()` retain their existing behavior; unknown presets still raise `KeyError`. |
| Global `tnfr.validation.config.ValidationConfig` fields `epi_range`, `vf_range`, `phase_coupling_threshold`, `cache_validation_results`, `max_validation_time_ms` | Removed unconsumed settings. Scalar domains, live U3 admission and the separate cached-validator configuration retain their own owners. Construction rejects the removed keywords; `configure_validation()` rejects them without committing other supplied settings. |

### Supplied-law symbolic calculus

`tnfr.math.symbolic.solve_nodal_equation_constant_params` constructs
`EPI_0 + nu_f_val * delta_nfr_val * (t - t0)` for supplied scalar expressions
held independent of its clock `t`. Each argument enters SymPy before
arithmetic; supplied symbols are preserved without placeholder substitution
or an integration-constant naming convention. The increment remains factored
to preserve the initial form when approximate coefficients and the time
origin have widely different scales. Approximate inputs retain their SymPy
precision; later expansion or numerical evaluation can still round values.
Zero capacity or pressure yields the initial form. These conditional symbolic
calculations do not admit a runtime state or supply a complete evolution law.
The same helper remains exported by `tnfr.math`, `tnfr.mathematics` and `tnfr`.
The numerical `tnfr.mathematics` facade derives its symbolic exports from
`tnfr.math` and omits them only when SymPy itself is missing. Unexpected
symbolic import failures propagate rather than appearing as unavailable helpers.

The retained `check_convergence_exponential(growth_rate, time_horizon)` uses
the specified unit-amplitude, unit-capacity exponential pressure law. Its
Boolean describes infinite-horizon integral convergence, which requires a
strictly negative rate. At zero rate it returns `False` and the finite integral
is the supplied horizon: constant pressure produces linear form growth, not
equilibrium. This is a corrected boundary, not a general U2 or stability result.

### Shared input-validation facade

The public `tnfr.validation` input validators re-export the implementations in
`validation/input_validation.py`; no parallel validation rules are introduced.
This includes signed form/pressure, nonnegative capacity, circular phase,
node/glyph, graph-interface and operator-parameter adapters. Their configured
input scope is distinct from live operator admission: a graph-interface check
does not establish a complete nodal state, and an operator-parameter check does
not authenticate an event or certify its trajectory.

The global `tnfr.validation.config.ValidationConfig`, also exported as
`tnfr.validation.StructuralValidationConfig`, supplies only the flags and
minimum severity consumed by `structural.run_sequence`. The existing facade
name `tnfr.validation.ValidationConfig` retains its separate meaning: the
configuration of `TNFRUnifiedValidationSystem`, including its cache policy.
These are different consumers, not interchangeable configurations.
`StructuralValidationConfig` construction and `configure_validation()` use
shared Boolean parsing and `InvariantSeverity`
members or their exact string values. Updates admit all keys and values before
mutating the shared object, so an invalid batch cannot partially disable its
checks. Keys come from declared dataclass fields, not arbitrary attributes.
Direct Python attribute assignment is outside that update contract. Scalar
domains, live U3 admission and other validator configurations have separate
owners; a successful audit is not a certificate for every nodal model.

### Shared operator requirements

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

Shared physics scalar and series readers also use represented-real admission:
nonzero values lost in binary64 conversion reject before a diagnostic can
interpret them as zero. Their `ValueError` policy and signed-zero convention
remain intact. Series return owned, writable float64 arrays; ordinary numeric
arrays and plain lists/tuples of binary16/32/64 floating scalars use equivalent
vector checks. Other source types retain scalar admission before conversion,
and invalid elements retain their indexed errors. Model-specific sign and shape
constraints still apply.

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
The opt-in extended EPI/phase/pressure integrator uses this same live source
registry once per node at each synchronous Euler substep. Its form row is
`dEPI/dt=nu_f*DeltaNFR+Gamma`, including at zero capacity; its independently
configured phase and pressure rows are unchanged. The scalar helper
`compute_extended_nodal_system` evaluates the unforced local rows and does
not itself consume a graph registry. Zero-duration calls evaluate no source.
Restoration after failure covers solver-owned outputs, not callback side
effects or diagnostic caches.

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
The separate [grammar role table](../theory/UNIFIED_GRAMMAR_RULES.md#1-canonical-operator-roles)
is generated from the grammar registry; a role does not replace the operator's
live preconditions or measured postcondition.

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
the canonical factor schema. The conservation/gauge snapshot mapper applies the
same scalar admission to consumed U3 phases and U6 pressure/reference fields;
nonzero underflow makes that diagnostic unavailable, not a passing zero.
Direct and simultaneous Resonance reuse one capacity proposal: finite
nonnegative capacity is required even when amplification is inactive. Graphless
unidirectional Coupling updates only the target phase. On graph paths, UM/RA
means use unique outgoing support neighbors, including zero-weight edges;
parallel multiplicity and incoming arcs do not supply additional samples.
These remain operator policies, distinct from weighted EPI transport. The
[coupled-state controls](../tests/operators/test_coupled_state_admission.py)
also verify support-cache invalidation after functional-link creation.

## Conditional relational execution

The maintained contract is [Conditional relational execution](contracts/relational/RELATIONAL_EXECUTION.md#conditional-relational-execution).
The separate [sine report admission](contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission)
also covers exact supplied primitives, captured binary64 coordinates and
derived-field reconstruction. The [regional guide](guides/REGIONAL_AND_RELATIONAL.md)
routes those readers by task; none is an implicit sine mode of the native
relational executor.

### Exact linear observations of a supplied generator

See [Exact linear observations](contracts/relational/OBSERVATION_AND_INFORMATION.md#exact-linear-observations-of-a-supplied-generator).

<a id="relational-pattern-observation"></a>

### Prepared relational pattern observations

See [Prepared relational pattern observations](contracts/relational/RELATIONAL_EXECUTION.md#prepared-relational-pattern-observations).

<a id="relational-attachment-observation"></a>

### Supplied relational attachment observation

See [Supplied relational attachment observation](contracts/relational/RELATIONAL_EXECUTION.md#supplied-relational-attachment-observation).

### Supplied joint state and support reset

See [Joint reset accounting](contracts/relational/RELATIONAL_EXECUTION.md#relational-reset-observation).

### Conditional relational capture

See [Conditional relational capture](contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#conditional-relational-capture).

### Validated conditional relational transit

See [Validated conditional relational transit](contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#validated-conditional-relational-transit).

### Exact cycle resultant sectors

See [Exact cycle resultant sectors](contracts/relational/OBSERVATION_AND_INFORMATION.md#exact-cycle-resultant-sectors).

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

## Mathematics backend selection

[`get_backend`](../src/tnfr/mathematics/backend.py) resolves an explicit name,
then `TNFR_MATH_BACKEND`, then
`tnfr.backend_config.get_config().math_backend`. The service default is `auto`.
Automatic selection tries GPU-capable adapters in JAX, PyTorch, NumPy order,
then repeats that order for any available adapter, including CPU adapters.
An explicit `numpy` request selects NumPy. A registered but unavailable named
adapter falls back to NumPy; unknown names raise `LookupError`.

`MathematicsBackend` includes `is_gpu_available()`, `get_device_name()` and
`get_backend_info()` alongside its array operations. Its
[typing declarations](../src/tnfr/mathematics/backend.pyi) retain that interface
and the canonical `core.exceptions.BackendUnavailableError` identity.
Backend selection supplies a numerical implementation, not an evolution law;
reproducibility records must identify the adapter actually used.
The PyTorch adapter copies read-only NumPy arrays before conversion because
tensors cannot preserve their write protection. Writable arrays retain the
existing sharing behavior when dtype/device permit; native tensor conversion
preserves identity and autograd when no conversion is needed.
The NumPy adapter delegates general matrix exponentials to the required SciPy
dependency. If SciPy is unavailable, that operation raises
`BackendUnavailableError`; a possibly defective eigenvector decomposition is
not substituted for the exponential. Other NumPy array operations remain
available.

`register_backend` validates a complete registration before changing factories,
aliases or instantiated adapters. Canonical names and aliases cannot shadow
one another, including with `override=True`; empty names and the reserved
selection name `auto` reject. A canonical override replaces its
factory and invalidates its instantiated adapter; unmentioned aliases remain.
An alias override may retarget an existing alias. A rejected proposal leaves
the registry and instance cache unchanged.

## Mathematical numerical boundaries

`BEPIElement` admits original finite real/imaginary components before array
conversion, using the shared represented-real boundary for each channel.
Boolean/text coercion and nonzero materialization loss reject, including in
`ensure_bepi` scalar/serialized routes. Complex and nonuniform BEPI remain
valid auxiliary elements; the shared signed-scalar reader still governs
whether an element can enter the nodal form row.
The sampling grid likewise admits original represented-real entries before
conversion: Boolean/text coordinates, complex coordinates and nonzero
materialization loss reject.
The stored BEPI arrays are owned, writable copies. Numeric-array fast paths
retain the same primitive admission and zero normalization; mutable arrays
are re-admitted by consuming calculations. The standard Banach-space element
factory delegates construction admission to `BEPIElement`, while customized
validation hooks retain their dispatch.

`HilbertSpace` represents finite coordinate vectors with a positive integer
dimension. Its floating or complex storage dtype cannot discard a nonzero
real/imaginary channel: real storage rejects nonzero imaginary coordinates,
and narrowing that overflows or erases a nonzero channel rejects. Norms use
the shared range-safe Euclidean observer. Inner products, Gram checks and
projections accumulate in complex128 and reject nonfinite outputs; ordinary
floating rounding remains, including possible underflow in products. A
supplied partial orthonormal family returns only its projection coefficients.

Composite EPI regularity trends admit finite represented-real observations,
including those computed from BEPI, and nonnegative represented tolerances.
Their threshold comparisons use the exact represented values, so an
overflowed or rounded comparison cannot turn a violation into a pass. A
reported drop must itself be representable. The plateau policy uses shared
Boolean parsing. These trends remain auxiliary regularity diagnostics, not
canonical `C(t)` measurements or stability certificates. The generic isometry
factory and norm-preservation checker remain explicitly unimplemented.

`BanachSpaceEPI.derivative_regularity` cancels a common large amplitude before
evaluating its sampled derivative-energy quotient. Its quadrature arithmetic
and result must be finite, with a positive denominator; a nonconstant field's
positive quotient lost to zero rejects. Exact constant fields retain zero.
Composite regularity uses a range-safe discrete norm and admits its strictly positive weights
before conversion; an unrepresentable composite value rejects.
`evaluate_composite_epi_regularity_transform` and its historical
`evaluate_coherence_transform` alias admit nonnegative finite policies and
both observed regularities, including results from a custom space. The
lower-bound comparison uses their exact represented values; the reported
requirement, deficit and ratio with a positive baseline must be representable.
A zero baseline followed by a value above the supplied tolerance retains
the explicit infinite-ratio convention. That ratio does not decide the
inequality or identify the auxiliary functional with canonical coherence.

Spectral state and operator readers admit original complex components before
materialization. Concrete backend tensors are checked through observations
while native calculations retain their automatic-differentiation path. A JAX
symbolic trace has no observable concrete finiteness verdict; its numeric
dtype/shape admission does not certify future values. Spectral expectations,
weighted angles and normalized mathematical evolution scale the vector before
taking its norm, so a representable normalized result is not erased by
overflow in the unscaled squared norm.
The mathematical runtime uses that same admission and normalization in
`stable_unitary`. Its `normalized` and `stable_unitary` reports observe the
standard Euclidean norm without squaring the unscaled amplitudes; an
unrepresentable norm rejects instead of returning infinity. Their tolerances
must be finite nonnegative represented reals. `stable_unitary` reports whether
the final norm is one within tolerance; with `normalise=False`, a preserved
non-unit input norm therefore does not pass this unit-norm check.
`MathematicalDynamicsEngine` and `ContractiveDynamicsEngine` likewise own their
admitted generators and return detached `generator` observations. Reconstruct
the engine to change that law; editing the original input or returned array
does not change its evolution.
Both engines admit signed finite represented-real time increments, including
when `evolve` requests zero steps; step counts are nonnegative integers.
Concrete native real scalar clocks retain their gradient path after admission.
Symbolic JAX clocks receive only scalar-shape and real-dtype checks, with no
concrete finiteness certificate. Exponential arguments and evolved concrete
states must be finite; a nonfinite observed density trace rejects before
normalization. Signed time permits reverse auxiliary evolution; it
does not extend a forward dissipative-semigroup theorem to negative time.
For the built-in engines, one `evolve` invocation reuses the exponential of
its fixed generator and time increment. State admission, normalization and
monitoring still run at each step. The propagator is local to the invocation:
zero steps compute none, and subsequent calls build it anew, including native
gradient graphs. Subclasses and replaced public `step` methods retain their
per-step dispatch.

The public mathematical phase readers apply the same raw-value admission to
scalar, list and NumPy inputs. `normalize_phase` returns the represented
half-open interval `[0, 2*pi)`; a modulo result rounded to its upper endpoint
is canonicalized to zero. `compute_phase_difference` retains its signed
`atan2(sin(delta), cos(delta))` convention and rejects unrepresentable
subtractions. These numerical readers do not replace live U3 admission.
The circular mean uses this same represented phase boundary.
`angle_diff_array` checks selected results in the requested output dtype
before writing: narrowing that erases a nonzero difference rejects and leaves
the output unchanged, including its unselected entries.

`BasicStateProjector` reuses the shared spectral-state normalization, scaling
finite amplitudes before evaluating their norm. Its null threshold must be a
finite nonnegative represented real, matching the other normalization consumers.
A representable large-amplitude vector therefore remains normalized instead
of collapsing to zero after norm overflow. Nonfinite intermediate amplitudes
reject; finite inputs alone do not promise representable output arithmetic.

`ContractiveDynamicsEngine.step(..., raise_on_violation=True)` propagates an
explicit monitored-norm violation and retains the measured gap. Failure to
observe that gap remains distinct from a measured violation. This auxiliary
monitor does not establish contraction in every norm: valid amplitude damping
can increase distance from the maximally mixed state.
Its public Frobenius observer and concrete centered monitoring norms use the
shared range-safe norm. Nonfinite or unrepresentable concrete observations
raise instead of becoming an unavailable gap that bypasses the monitor.
A live JAX trace still has no concrete norm observation and retains an
unavailable gap; native propagation retains its gradient path.

`get_laplacian_spectrum` rebuilds the consumed operator before cache lookup.
Cached diagonalizations use immutable matrix bytes in the current node order,
including the selected combinatorial weight channel; returned arrays are
detached. Changing edge weights or node iteration order cannot reuse a
different matrix's eigenbasis. `heat_diffusion` uses shared nonnegative
represented-time admission before evaluating the supplied spectral semigroup.
The eigensolver follows exact symmetry of the admitted real operator matrix,
not the graph container's directed flag. Reciprocal directed support can have
a symmetric operator and therefore an orthonormal eigenbasis; an asymmetric
operator retains the general right-basis solver, including for partial requests.
For nonnegative reversible graph operators, component support determines the
stationary subspace (constant vectors for combinatorial/random-walk charts,
square-root-strength vectors for the symmetric chart). Identified stationary
modes retain exact represented zero rates at long horizons. A small eigenvalue
alone is never clipped: unresolved mixing with slow modes rejects the chart.
Very weak bridges between otherwise strongly connected components can reach
this numerical resolution limit even with strictly positive finite weights.
This floating subspace check is not a validated spectral enclosure. Signed
combinatorial and asymmetric operators retain their generic numerical spectrum;
arbitrary supplied spectra receive no stationary-mode correction. Nonfinite
heat-result arithmetic rejects explicitly.

Multi-field spectral diagnostics with at most 100 nodes can reuse an
invocation-local basis validation across individual GFTs. Larger diagnostics
retain their sequential backend dispatch and field reductions. No basis-validity
cache survives the call; generic inputs and replaced transform hooks retain
their individual dispatch. The public `spectral_filter` still revalidates its
basis after the supplied filter callback.
`physics.spectral_conservation` aligns consumed snapshot maps with the graph's
node iteration order and requires matching nonempty support. Its real Parseval
diagnostics additionally require a full finite real orthonormal basis; the
general GFT's invertible right-basis domain alone is insufficient. Consumed
snapshot values use shared real admission before array conversion, including
Boolean, nonfinite and nonzero-materialization-loss rejection. Time intervals
must be finite and strictly positive; supplied policy thresholds must be finite
and nonnegative. Invalid observations, charts or unrepresentable report arithmetic
reject before classification. The explicit `sector_ratio` infinity convention
for a geometric denominator below its existing floor remains separate from
arithmetic failure.

The two-snapshot modal source is the GFT of `delta_rho / dt + mean_divergence`;
stored divergence is not multiplied by Laplacian eigenvalues again. The
`mode_transport_rates` classification field contains the signed divergence
spectrum itself, with the non-strict threshold boundary including zero at a
zero threshold. Band scores describe the supplied observations; neither their
ordering nor a future stability guarantee follows from an operator label.
In `conservation_quality_by_band`, a band without observed modes has value
`None`; consumers must handle that absence instead of interpreting it as the
former perfect score of `1.0`.
These are scoped observations under the
[spectral balance contract](../theory/STRUCTURAL_CONSERVATION_THEOREM.md#92-spectral-decomposition).

Liouvillian spectral readers require finite original real/imaginary components
and finite nonnegative represented tolerances. Invalid spectra reject before
selection or metadata mutation. Numeric-array admission returns owned complex
storage with canonical positive zeros; invalid arrays retain the scalar reader's
first failing component and exception. The compatibility `validate_contractivity`
flag checks only eigenvalue real parts against its tolerance; it does not
certify a Lindblad representation or contraction in an arbitrary norm. Stored
spectra retain no independently authenticated generator or clock provenance.
The slow-mode reader selects the first eigenvalue with the least negative real
part strictly below `-tolerance`. Its decay timescale alone does not establish
whole-state convergence, measured recovery or U6 admission.
`build_lindblad_delta_nfr` uses the shared original-component reader for its
Hamiltonian and collapse matrices, admitting each supplied matrix once. Its
trace-preservation check is `vec(I)* L = 0`, with `*` denoting the adjoint;
it does not require `L vec(I) = 0` (unitality). Amplitude damping is an example
that preserves trace while changing the identity.
Both generator factories admit finite real scaling factors and a representable
product, plus positive integer dimensions. They reject nonfinite constructed
matrices. These are supplied auxiliary matrix multipliers; a signed multiplier
does not redefine the nonnegative nodal-capacity domain. Forward GKSL
interpretation requires a nonnegative overall generator factor and its other
[model hypotheses](../theory/DISSIPATIVE_AND_OPEN_SYSTEMS.md#1-gksl-generator).

The auxiliary dissipative diagnostics require representable snapshot purity,
computed actions, bounds, rates and comparison thresholds as well as finite
primitive inputs.
They reject nonfinite results before issuing bound or unitality verdicts;
nonzero collapse-norm squares lost to underflow also reject. Frobenius norms
use the shared scaled fallback when direct squaring exceeds the numeric range,
and observed balance rates use the shared signed secant reader. These checks
preserve the [diagnostic availability conventions](../theory/DISSIPATIVE_AND_OPEN_SYSTEMS.md#6-validation-and-reproducibility):
absent collapse or reference data retain explicit NaNs and unevaluated flags.
The trace-distance ratio remains infinite when the initial distance is at or below
its configured floor but the final distance is not; that observation fails
the contraction check and is distinct from arithmetic overflow.
Generator spectral scales, eigenspectra and residuals must also be finite
before stationarity, trace-preservation or mode classifications.
`DissipativeConservationTracker.record` commits its snapshot history and
diagnostic series only after snapshot admission and all diagnostics succeed.

The optimized arithmetic sieve uses Python integers for divisor sums before
evaluating the supplied floating pressure formula. `FiniteField` admits an
integer prime characteristic, supported positive integer extension degree and
integer modulus coefficients; a composite-characteristic ring is not admitted
as a field. These are premises of the
[arithmetic models](../theory/TNFR_NUMBER_THEORY.md), not physical identifications.
`cayley_diffusion_action` preserves support-derived exact stationary Fourier
modes, including disconnected supports. Unrepresentable positive clock
products or nonfinite Fourier intermediates reject explicitly; the calculation
remains a floating evaluation of a supplied fixed circulant law.

Finite-field arithmetic normalizes integer/index elements to Python integers
before operations and requires `0 <= element < q`; powers require nonnegative
integer exponents, and power-set orders are positive integers.
For built-in field arithmetic, normalized periods can reuse character values
within one invocation while preserving the original summation order. Later
calls read the current presentation anew; customized arithmetic and character
hooks retain their direct dispatch.
`OptimizedTNFRPrimality` applies the same strict `abs(delta_nfr) < threshold`
policy with and without sieve coverage. The cut must be finite, positive and
representable; the default `0.5` separates the prime zero set for its configured
weights. Larger cuts may admit composites and do not alter the exact theorem.
Returned reports and nested metrics are detached from the cache; subsequent
calls do not change previous `cache_hit` observations.

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

`is_positive_semidefinite` requires Hermiticity as well as nonnegative
eigenvalues within its tolerance. `spectral_weighted_angle` uses that shared
gate; positive eigenvalues of a non-Hermitian matrix cannot admit a Hermitian
angle geometry.
The weighted-angle ratio combines binary exponents separately rather than
squaring overlaps and multiplying expectations at their original scale.
This retains finite scale-invariant angles when those intermediate products
would overflow or underflow. Consumed overlaps and positive expectations must
still be representable; this does not repair precision lost in dot products
or remove the configured absolute null threshold.
Operator factories reuse the constructor's Hermiticity and PSD gates and
admit original components before backend conversion. The spectral factory's
real projection within its imaginary-part tolerance retains the native
gradient path. Factory dimensions use the same positive-integer admission
as their SDK/CLI consumers; Boolean and floating dimensions reject.
Explicit comparison floors and thresholds use the shared represented-real
boundary, including rejection of nonzero values lost to materialization.
A comparison floor may be negative independently of the supplied PSD spectrum.
Construction owns a detached matrix and its matching spectrum. The public
`matrix` and `eigenvalues` properties, and `spectrum()`, return detached arrays;
editing these observations or the original input cannot change the operator
or its PSD verdict. Direct property assignment is no longer supported: create
a new operator to change the observable and recompute its spectrum together.
Construction reuses its admitted matrix snapshot for Hermiticity checks,
avoiding redundant backend transfers; native spectral arithmetic retains its
gradient path.

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
