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
## Contract model

### Word and operator controls

Word admission rejects unregistered operator identifiers before role checks;
operator instances retain their metadata, including Recursivity depth. Grammar
context flags use the shared Boolean parser: `"false"` does not grant initialized
form or diagnostic permission, and unknown Boolean strings reject. Context is
copied, leaving the caller's mapping unchanged. See the
[grammar contract](../theory/UNIFIED_GRAMMAR_RULES.md#8-composition-and-implementation).

<a id="word-profile-migration"></a>
`validate_sequence`, `parse_sequence` and `ValidatedSequence` default to
the shared `core` word contracts. String validation delegates to the same
`GrammarValidator` checks used by typed execution, including causal debt and
prior Coherence; a different input representation cannot bypass these checks.
String words carry default operator metadata only. Use actual operator
instances when, for example, Recursivity depth matters.

The explicit context `{"compatibility_profile": "legacy"}` adds the historical
adjacency and THOL-terminal preferences. `validate_sequence` and
`parse_sequence` also accept that keyword directly, taking precedence over the
context value. Both profiles retain the shared core checks; neither establishes
live admission, trajectory stability or every manuscript sentence prescription.
Callers requiring those additional preferences must select `legacy` explicitly.
Both profiles reject violations of the shared causal checks. `parse_sequence`
raises on rejection; `validate_sequence` returns its structured outcome.

`TNFR.explain_sequence(operators, context=...)` reports this same word policy,
ordered English display names in `operators`, its selected `compatibility_profile`
and compact glyph `notation`, such as `AL → IL → SHA`. Lead reader-facing
explanations with the English names. It accepts registered names or simple
glyph codes, not arrow
expressions, apostrophe spellings or bracket syntax as individual tokens.
Invalid context yields `valid=False` with no selected profile; unknown operator
identifiers raise. The report explicitly leaves manuscript syntax, live state
and trajectory checks unassessed. Context is a declaration, not observed
initialization or trigger evidence.

`GrammarValidator.validate` and `collect_grammar_errors` consume the same
immutable outcomes from `GrammarValidator.validate_checks`; diagnostic text
does not decide whether a check passed. Strings use shared identifier normalization; supplied
operator objects retain their metadata, so unknown identifiers report `SYNTAX`
and Recursivity depth remains subject to U5. `run_structural_validation` uses
these errors for its word status. Omitting the sequence skips word validation;
live U3 admission and temporal U6 evidence remain separate obligations.
`structural.run_sequence` observes the target's actual admitted signed form
when determining initialization context; a caller flag cannot promote zero
or invalid form into permission. It records the last completed operator only
after that invocation succeeds.

<a id="grammar-history-and-selector-admission"></a>
**Persistent grammar state.** Shared `require_grammar_state` admits stored
`glyph_history` as an ordered list, tuple or deque of registered identifiers.
Strings, mappings, sets, unknown tokens and explicitly stored `None` reject;
one-shot iterators reject without consumption. Omit an absent history key,
or materialize a valid owned sequence before execution. Present U2 debt must
be a nonnegative integer excluding Booleans, and prior-Coherence state must
be Boolean. Invalid present state cannot fall back to a shorter trace.
The stored operator-step index likewise requires a nonnegative non-Boolean
integer; only a missing index falls back to the retained trace length.
Only absent bookkeeping is reconstructed from retained history; this cannot
recover evicted facts. Deliberate full-trace replacement uses the shared reset
helper; restoring a snapshot preserves its valid persistent context.
`on_applied_glyph` delegates to the shared occurrence recorder: it normalizes
glyph codes and updates the index, debt and prior IL while preserving deque
bounds or unbounded list retention. Tuple traces become unbounded lists.
This helper records a caller-supplied occurrence; it neither executes nor
authenticates an operator. Do not record an already runtime-recorded action twice.

`read_grammar_state` builds a fresh immutable view of canonical history codes,
debt, prior Coherence and operator index. Recording admits each retained
identifier once; it does not cache permission across mutations. History readers,
repetition scores, hysteresis and direct Mutation/Dissonance context readers
use shared admission, so equivalent supported names and codes give the same
decisions. Malformed authoritative history cannot be hidden by a diagnostic
shortcut. Graph validation retains its `TNFRValueError` error envelope.

Native batches pre-admit consumed grammar context and lag counters before
incrementing counters. Lag limits/counts reject Boolean, fractional or invalid
values, and the hysteresis window uses shared integer admission. Grammar's
`enabled` uses shared Boolean parsing. Lag resets follow the glyph actually
applied, including grammar substitution; a blocked AL proposal replaced by IL
does not reset `since_AL`. `_choose_glyph` applies the same grammar to forced
candidates but neither increments counters nor executes them. Native batches
remain sequential: a later operator failure can retain earlier completed
operations and policy bookkeeping; this is not the atomic-stage contract.
Mutation proposal screening and execution share raw authoritative-alias and
signed-form certificate admission. Explicit invalid EPI cannot be replaced by
a secondary alias or turned into missing temporal evidence. Recorded abstention
clocks must be valid represented real values.
Mutation admits raw authoritative phase and active proposal scalars before
conversion. Boolean/nonreal values, nonfinite values and nonzero values lost
to zero reject; direct graph rejection precedes adapter caching. The shared
proposal preserves the consumed pressure's signed zero after admission.
An explicit fixed shift still leaves dynamic proposal arguments unconsumed;
the independent temporal/capacity gate remains required. Valid phase wrapping
and public/direct/staged execution boundaries are unchanged.
Native batch proposals resolve supported English names, tokens and glyph codes
through the shared identifier owner before Mutation screening and lag accounting.
Emission/Reception counters and Mutation abstention records therefore agree
across those spellings, including with canonical grammar disabled. Unknown
proposals retain the selected grammar rejection/substitution policy; this is
not whole-batch prevalidation or rollback of an earlier executed prefix.

With glyph execution enabled, `runtime.step` runs the same batch preflight
before creating history, reserving an invocation ordinal or invoking callbacks.
It re-admits at consumption after callbacks and pressure refresh. Disabled
glyph execution does not consume lag limits or grammar policy. This covers
the supplied batch configuration/context, not arbitrary callbacks or a full-step
transaction. A late failure can retain observations and its invocation ordinal.

Selector preparation and selection hooks also have a live admission boundary.
The batch readmits policy, clock, grammar and lag records after preparation
and after collecting proposals, before executing a glyph. Both boundaries
require the original ordered target population. Preparation may change which
lag choices are forced; collected proposals retain their selection boundary
and are not recomputed. Lag counters advance once. Replaced lag mappings and
history-window settings are consumed from the live context. A rejected hook
result does not roll back the hook's own changes or imply an atomic batch.
The [hook controls](../tests/test_selector_hook_readmission.py) exercise this
boundary through actual glyph execution and rejected requests.

`tnfr.dynamics.apply_native_operator_batch(G, history, decision_trace=None, epi_trace=None)`
exposes that same native all-node batch. It resolves the graph's configured
selector, returns `None` and advances no time. It does not refresh pressure or
diagnostics, supply past samples, select a target subset or install a scheduler.
Default and parametric policies retain their current candidate sets and
normalizers; exposing this boundary does not make them generate all thirteen
operators. Direct glyph execution retains its existing distinction from public
operator classes and atomic stages.

Optional `decision_trace` must be an exact caller-owned list. The batch appends
frozen `NativeOperatorDecision` reports without replaying selection or grammar.
Their `selection` describes the evaluated builtin branch (base, hysteresis,
score, override and soft filter where consumed), or marks custom selection as
opaque. Forced lag choices do not invent an evaluated selector branch.
Requests retain their preparation boundary; temporal and grammar admission use
the live sequential execution state. The reports preserve that distinction.

The report records lag origin/counts, the actual Mutation gate certificate or
abstention reason, the observed grammar decision and the direct dispatch result.
The shared grammar observer distinguishes acceptance, fallback, rejection and
alternative-selection failure, including fallback to the same code. Invalid
inputs before a decision yield no invented approval. `applied_glyph` is recorded
only after dispatch returns; later bookkeeping may still fail.

This is a sequential batch: a later failure preserves completed earlier actions
and propagates the same exception. The caller's list retains the captured
prefix, terminal failure and already computed but unreached proposals where
available. A failed dispatch may itself have partial effects; absent
`applied_glyph` does not certify rollback. Preflight failure before admitted
targets can leave the trace empty. Reports are neither checkpoints nor
authenticated provenance, and custom values retain their opaque scope. See
the [native participation analysis](../theory/NATIVE_OPERATOR_DYNAMICS.md#native-mutation-request-participation)
for the supplied policy, diagnostic and decision-clock boundaries.

<a id="native-epi-source"></a>
**Actual native EPI source observation.** The optional `epi_trace` must be an
exact caller-owned list distinct from `decision_trace`. It receives one frozen
`NativeEPIBatchObservation` around the actual public batch call, including an
incomplete call that raises. Return value, selector preparation, lag advancement,
grammar, dispatch order and sequential failure semantics remain unchanged. The
default performs no full-population EPI captures. These observations are exposed
through `tnfr.dynamics` and defined in `tnfr.dynamics.native_epi_source`.

`NativeEPIEndpoint` retains ordered `nodes`, `node_storage_ids`, `exact_epi`,
`exact_time` and `unavailability_reasons`. EPI uses the shared signed scalar
binary64 embedding, including uniform-real BEPI; fractions encode that admitted
representation, not arbitrary-precision raw input. Missing, invalid or richer
non-scalar form makes observation unavailable without adding a native action
precondition. Storage identities associate this local capture; they are not
portable node identity or authenticated provenance.

Each `NativeEPIDispatch` records the actual dispatch index, target and glyph,
whole-population `before`/`after`, `returned` and `failure_type`. Its
`exact_increment` is the represented endpoint difference where available. An
abstained or None request creates no dispatch entry; an action preserving EPI
creates a zero increment, even if it changes other structural channels. A raised
dispatch can retain measured partial effects and `returned=False`.

The batch contains `initial`, `final`, `dispatches`, `completed`, failure metadata,
`available` and `clock_preserved`. When all endpoints retain the same ordered
population/storage and captured clock, it supplies `exact_net_increment`,
`exact_dispatch_increment`, `exact_dispatch_variation`,
`exact_outside_dispatch_increment`, `exact_outside_dispatch_variation` and the
individual `exact_outside_dispatch_increments`. These are vector sums of exact
represented differences. Failed-prefix arithmetic can be available while
`completed=False`; successful execution can have unavailable observations.
Changed population/storage, changed or unreadable clock, or unavailable EPI
leaves whole-batch totals unavailable rather than inserting zeros.

An unexpected failure of post-dispatch recording marks capture incomplete. If
the primitive returned, its native history cleanup, interval receipt and waiting
bookkeeping still complete before the observation error propagates; later targets
are not executed. If the primitive or its existing bookkeeping fails, that
original error takes precedence. Missing capture cannot be relabelled as an
outside-dispatch change or a complete source.

Outside-dispatch changes include selector/grammar/hook effects between captured
boundaries; they are not attributed to an operator. Zero net residual can hide
cancelling gaps, and endpoint variation misses an internal round trip. Mutable
dispatch registrations and glyph names do not authenticate a canonical map.
The record contains no rate, creates no temporal samples or receipt, and cannot
activate Mutation. The restricted producer below owns its own positive interval
and immediate live admission. See the [source analysis](../theory/NATIVE_OPERATOR_DYNAMICS.md#native-epi-source).

<a id="native-positive-window"></a>
**Restricted positive-window native source.** The opt-in
`tnfr.operators.execute_native_positive_window(graph, *, duration, candidates=None,
method=None, n_jobs=None, mutation_kwargs=None)` executes a real positive interval,
one terminal native batch and eligible canonical ZHIR responses. It accepts no
detached source/report/certificate input. The supplied order differs from the
ordinary runtime's batch-before-flow order; the existing native API is unchanged.

The interval requires the canonical default nodal integrator, zero stored
pressure, no Gamma or extended dynamics, and fixed ordered node/storage identity.
Form, phase, capacity and support are held during that interval. The actual flow
executor advances its declared clock and records its boundaries; a timestamp
assignment cannot substitute for it. The terminal batch requires the admitted
builtin selector, enabled grammar with rejection mode `"raise"`, and unchanged
direct dispatch bindings. Its live grammar,
AL/EN waiting priority, sequential order and histories remain in force.
Native history is the graph-owned exact dictionary `graph.graph["history"]`,
retained by identity; absence creates an empty dictionary. No external history
argument or dictionary subclass is admitted by this restricted route.
Source acceptance requires complete scalar capture, unchanged terminal clock,
contiguous endpoints and zero outside-dispatch variation, not just zero residual.

The source retains the actual flow execution and native observations separately;
native actions are not represented as atomic schedule stages. Growth records exact
represented signed endpoint changes divided by the declared positive duration,
with separate flow, jump and variation accounting. The interval is (s,t], including
the terminal jumps. These values are historical reports; only the private current
production scope can bind a fresh response snapshot. They create no continuous
secant. Form-writing native actions retain their ordinary history invalidation.

The returned `NativePositiveWindowDispatch` exposes `source_result`, `growth`,
`candidates`, `decisions`, `selected_targets` and optional `stage_result`;
`executed` means a supplementary phase stage occurred. `NativePositiveWindowSource`
retains `flow_result`, `native_batch`, `native_decisions` and the initial/flow
endpoints. `NativePositiveWindowGrowth` carries the exact interval and vector
balance. The growth source is `native_positive_window_net_growth`; trigger
telemetry uses `native_positive_window_time`, distinct from schedule growth.

Candidates default to all initial nodes in order; an explicit set is a nonempty
unique exact tuple. Each candidate must independently pass strict exact growth,
capacity/floor, live grammar and unused-window admission. The shared hybrid
`_operator_growth_mutation_end` watermark retains successful consumed endpoints
across this route and the supplied-schedule consumer. Native ZHIR, if actually
returned, has precedence over a supplementary response on that target. It cannot
acquire a crossing continuous secant in this restricted one-batch zero-flow source;
the precedence guard is not evidence of general source competition. Empty selection
retains the source and adds no empty phase stage, ZHIR event or hybrid receipt.

Initial request and domain admission precede source execution; later readmission
can fail after actual source effects. The flow executor retains its own failure
boundary; native failure preserves elapsed flow
and the completed/partial native prefix. An incomplete or unavailable source grants
no response. After source completion, eligibility and response have a separate
graph transaction: their failure restores the completed source state. External
callback effects are outside graph rollback. This is not the source-and-response
atomicity of the supplied-schedule consumer. See the [restricted source analysis](../theory/NATIVE_OPERATOR_DYNAMICS.md#native-positive-window).

<a id="emission-episode"></a>
**Finite-duration Emission proposal.** The opt-in
`execute_emission_episode(graph, targets, *, duration, method=None, n_jobs=None,
window=None)` executes a distinct AL episode. It is an explicit constitutive
extension: the primary coordinate action follows the AL translation family,
while counting one duration-bearing episode as one occurrence is a declared
lifecycle choice. Ordinary instantaneous AL and the zero-flow native interface
retain their contracts. See the [proposal and its assumptions](../theory/EMISSION_EPISODES.md#emission-episode-status).

Targets are a nonempty unique exact tuple. Each has positive admitted capacity
and pressure, strict AL entry/completion state, and its own enabled grammar in
rejection mode `"raise"`. Duration is positive and uses the live structural clock.
An already activated target must retain a countable AL lineage; missing lineage
cannot certify a new counted episode. Completion checks preserve that lineage.
Every nonzero form row must belong to a target; nonparticipants have zero row.
The canonical nodal integrator holds pressure, capacity, phase and support, with
Gamma absent. A prospective target gain `duration * nu_f * DeltaNFR` is compared
against the shared AL proposal. Clipping or a represented endpoint mismatch
rejects the episode; a measured increment cannot be used to fit its own gain.

The integrator supplies the EPI change and real physical boundary samples.
Completion commits shared AL activation lineage, source-glyph provenance and
exactly one grammar occurrence, without another AL form write. Physical history
therefore remains valid. Native waiting counters are unchanged, matching the
separate public-operator semantics. The episode is atomic on graph-owned state;
failure restores its entry state. It is not an interruptible persistent action,
and external callback effects are outside graph rollback.

`EmissionEpisodeResult` retains ordered population/targets, the actual
`flow_result`, prospective rates/gains and represented EPI endpoints. Numerical
substeps count within one episode; separately invoked episodes count separately.
Actual solver derivative diagnostics remain observable and may depend on the
subdivision. Neither a report nor equal endpoints authenticate another execution.

Successive episodes retain actual physical history across pressure-only IL/OZ
actions. A subsequent graph-backed THOL reads active history. Public/staged/native
routes retain their own grammar admission and the optional strict public/staged
gate remains separate; direct glyph dispatch alone is not full grammar admission.
The selected finite window remains part of the observation. Its signed pressure
result need not admit another
positive-rate AL episode. Rejection of that later episode restores its own entry
state, preserving completed THOL. See the [causal composition and boundaries](../theory/THOL_PRESSURE_FEEDBACK.md#thol-emission-history-continuation).

The same actual source can feed the existing eligible THOL dispatcher and
admitted child creation. Its returned parent/child association can supply an
explicit UM request, followed by actual EN/RA participation. Birth supplies
neither contact nor inherited child grammar/physical history. The
[three-branch integration](../theory/THOL_PRESSURE_FEEDBACK.md#thol-acquired-birth-participation)
separates child input from UM's other effects and tests the child's own later
actions; invocation times and source selection remain supplied policies.

`execute_native_emission_window(graph, emission_targets, *, duration,
candidates=None, method=None, n_jobs=None, episode_window=None,
mutation_kwargs=None, selector_acceleration_source="solver_diagnostic")` owns an
actual episode, one sequential native batch and
eligible supplementary ZHIR responses. It requires the unchanged builtin native
dispatch family, exact graph-owned history and
`NATIVE_ZHIR_OCCURRENCE_POLICY="consume_timestamped_interval"`. Its source result
retains the episode and native capture separately. The growth source is
`native_emission_window_net_growth`, with trigger time basis
`native_emission_window_time`; the episode flow and terminal jumps each enter the
whole-window balance once. All rows, endpoint joins and outside-dispatch coverage
are re-admitted by the private live producer.

An actual native ZHIR takes precedence over a supplementary response. For other
candidates, exact growth, capacity, grammar and unused evidence remain independent
requirements. A successful supplementary response also consumes any retained valid
physical interval matching this window and live EPI, through the shared native
occurrence owner. Previously consumed matching evidence blocks admission; a form
jump that cleared physical history creates no invented physical receipt. Phase,
grammar and those receipts commit in the same response transaction. Native failure
retains the completed episode/prefix; response failure restores the post-source
state. No detached source, episode result or growth report is an activation input.

`observe_emission_episode(result)` returns a passive `EmissionEpisodeObservation`
from the actual sealed episode. It separates `exact_held_rates`, actual endpoint
`exact_observed_rates`, law-derived `exact_interior_accelerations`, and the
`solver_rates`/`solver_accelerations` retained at completion. Endpoint secants
need not equal the represented held rate after rounding. The episode's two
endpoints alone leave `episode_sampled_acceleration_available` and
`boundary_transition_available` false; earlier live history is a separate source.
See the [derivation and selection scope](../theory/EMISSION_EPISODES.md#emission-selector-observation).

The native wrapper's opt-in `selector_acceleration_source="emission_interior"`
requires the unchanged builtin parametric selector and the producer's own live
episode. Its call-local channel covers the complete produced population and
supplies acceleration consistently to normalization, base choice, hysteresis,
score and the soft repetition/force filter. It creates no derivative-cache or
history writes. `source_result.selector_observation` retains the passive basis;
`source_result.selector_acceleration_source` and the decision trace identify the
selected channel and episode window. A returned observation cannot activate a
later or foreign batch. Cleanup retires the channel on success and failure.
Ordinary selector calls retain the stored-diagnostic policy. THOL's actual
history-derived pressure action remains distinct; the episode policy does not
reinterpret missing temporal curvature as measured zero or supply a child birth.
Later actions retain their own diagnostic effects: supplementary staged ZHIR
writes its history-derived acceleration cache, while direct native ZHIR does
not. The retained episode report still records the earlier completion values.

Native dispatch also owns a target-local continuous-history boundary. For an
actual scalar-form writer (AL, EN, RA, VAL or NUL), an existing non-None
`epi_time_history` is invalidated to an empty tuple if the final form differs
from its admitted pre-dispatch value. An unreadable final form invalidates
conservatively. This applies to the final dispatched glyph and to visible
partial effects after a failed dispatch; the original exception is preserved.
The optional `physical_history_invalidated` trace field distinguishes a completed
boundary from an unavailable observation. This is not a rollback guarantee or
an audit of arbitrary transient or non-target writes by external replacements.

The boundary creates no timestamps, samples or elapsed duration. It does not
read history contents, alter legacy histories or populate absent/None physical
histories. Phase-only actions and numerically unchanged forms retain the exact
history binding. Empty physical history remains authoritative, preventing silent
fallback to an older legacy series. A later form round trip cannot restore the
discarded continuous interval. The runtime recorder can subsequently begin a
new history at its actual right endpoint; acquiring another growth interval
requires its own admitted temporal mechanism.

This automatic invalidation belongs to native batches and their runtime
consumer. Explicit direct/class/staged interventions retain their separate
observation responsibilities; the event runtime records individual stage
boundaries and the network-word preflight retains its conservative Mutation
guard for earlier primary-EPI writers. See the [operator source boundary](../theory/NATIVE_OPERATOR_DYNAMICS.md#native-operator-growth-source-boundary)
for the difference between finite jumps, a primitive continuous limit and
admissible temporal evidence.

The native graph setting `NATIVE_ZHIR_OCCURRENCE_POLICY` defaults to `"level"`:
qualifying own growth evidence can support repeated Mutation requests. The
explicit alternative `"consume_timestamped_interval"` retains each successfully
dispatched native ZHIR interval in that node's
`_native_zhir_consumed_intervals`. Its identity is the evidence source and two
represented binary64 timestamps, independently of sampled EPI values, threshold,
phase or invocation count. The exact tuple ledger is re-admitted before use;
malformed entries reject. It has no automatic eviction. Revising values or
replaying an older interval does not renew a consumed identity. Changing the
ledger externally resets the model's memory; it is not authenticated provenance.

The alternative retains live capacity/growth checks and configured grammar admission.
An otherwise valid legacy step history lacks timestamped interval identity and
abstains; a consumed interval also abstains. Both requests use the existing
IL substitution by default, or the explicit no-action policy below. A final ZHIR dispatch must satisfy
live occurrence admission even if selected by a grammar fallback. Rejection,
substitution to another glyph and a dispatch that raises do not consume an
interval. A successful ZHIR return commits consumption before later bookkeeping
or another target can fail. Mutation of the ledger during dispatch rejects the
commit; arbitrary external writes are outside the retained-memory guarantee.
Existing partial-effect limitations still apply.

The optional decision trace reports the occurrence policy, interval, prior
consumption and completed consumption separately. This rule applies to native
selector batches, including their existing runtime consumer. Explicit direct
glyphs, public operator classes and atomic stages retain their own contracts;
they neither consult nor populate this native ledger. The level policy ignores
the ledger and does not add receipts. Switching policies does not reconstruct
earlier executions. This is an opt-in occurrence law with growing retained
memory, not a derived clock or a new default canonical law. See the
[interval occurrence analysis](../theory/NATIVE_OPERATOR_DYNAMICS.md#native-mutation-interval-occurrence).

<a id="native-mutation-abstention"></a>
**Native Mutation abstention policy.**

`NATIVE_ZHIR_ABSTENTION_POLICY` accepts exactly `"coherence"` (the default)
or `"skip"`. It affects an actual native Mutation request whose own trigger or
interval policy reports an abstention reason. The default substitutes Coherence
and then applies configured grammar. With `"skip"`, that target dispatches no
operator and does not search for a grammar alternative. Other requests, live
admission of eligible Mutation, and explicit direct/public/staged actions keep
their separate contracts. Neither policy selects hybrid operator-growth evidence.

The policy is admitted before selector construction/preparation, after hooks and
at use. Every actual Mutation request admits explicit capacity and its configured
`ZHIR_MIN_VF` through the shared reader, even when earlier trigger conditions
would exclude execution. Missing/malformed capacity, invalid floor, threshold or
other consumed primary scalars remain errors. Valid inactivity, noncrossing and
the existing certificate's unavailable/invalid temporal-evidence classification
remain distinct reported abstention conditions. This is not approval of those
historical samples. An unrelated request does not consume the Mutation floor.

A skipped Mutation occasion still increments native AL/EN lag counters once.
It adds no glyph, operator step, grammar debt/coherence update, continuous-history
cleanup or interval receipt. The existing `mutation_abstentions` sink is admitted
and records `applied_glyph=None`. The optional decision trace records the consumed
`mutation_abstention_policy`, reason, `status="abstained"`, and no post-gate or
applied glyph. This differs from `status="skipped"` for an opaque selector's
`None` proposal. Tracing itself changes no selection or execution.

No-action applies to the target's dispatch, not the whole sequential batch:
other nodes can still modify it through their real neighborhood actions. A later
error retains the completed prefix and lag writes. Skipping cannot suppress an
eligible action's grammar rejection or a failed actual dispatch. The
[selection analysis](../theory/NATIVE_OPERATOR_DYNAMICS.md#native-mutation-abstention)
states the pressure/memory consequences and remaining source-selection boundary.

Soft repetition thresholds `GRAMMAR.force_dnfr` and `force_accel` and consumed
normalized scores admit original represented-real values in `[0,1]`; Boolean,
text, nonfinite or lost nonzero values reject. Explicit malformed fallback
operators reject. Context parameters must be mappings: optional constructor
`None` remains absent, whereas explicitly malformed graph-owned configurations
reject before defaults. Empty graph mappings retain the documented default
overlay. These checks preserve the supplied normalization/selection policies;
they do not infer global maxima from local input or derive a controller.
Repeat checking and its error context reuse one fresh admitted history view,
without a second permissive interpretation of the retained identifiers.

Selector pressure/acceleration ratios use shared checked division on scalar,
vector and worker paths. Nonfinite quotients and nonzero quotients rounded to
zero reject before threshold comparisons. Valid represented ratios remain
unclipped; a zero supplied normalization maximum retains its unit-divisor
convention. This rejects unrepresentable observations rather than selecting
an operator from an invented zero.

The native selector runs at an outer call's left boundary. Lag counts measure
decision batches, internal integration substeps add none, and `dt=0` still
allows operator events. The [clock comparison](../theory/NATIVE_OPERATOR_DYNAMICS.md#native-selector-clock-admission)
shows that additional outer decisions can change the physical endpoint at
the same final time. Numerical comparisons must retain their declared event
schedule and other per-call policies; `apply_glyphs=False` skips only the glyph
batch, not phase/capacity policies or all runtime bookkeeping.

Native execution consumes complete associated state: form, phase, capacity,
stored and refreshed pressure, grammar and lag records, adaptation counters,
prior rates, temporal observations and target-associated random-stream state.
Default selection additionally uses graph-wide capacity and absolute-pressure
maxima through Si and normalized pressure. Matching a component's local state
or selected operator labels therefore does not establish a closed component
law. Reception consumes live neighboring form; Resonance consumes the live
U3-compatible subset and its associated form/phase data. Earlier target events
can change a later target's branch. Integration consumes event capacities and
held prepared pressure; later policies have their own observation boundaries.

The [native dynamics owner](../theory/NATIVE_OPERATOR_DYNAMICS.md) derives the
scoped selector, ordered-event and complete-call reductions and counterexamples.
The [collective calculus](../theory/NATIVE_COLLECTIVE_CALCULUS.md) owns sufficient
retained state, correlated enclosures and their composition requirements.
The [aggregate operator owner](../theory/NATIVE_AGGREGATE_OPERATORS.md) states
when an actual registered operator descends to a coarse representation and
when it does not. These conditional results neither create a public collective
executor nor let a summary report replace the source required by a later call.

A collective prediction must retain the actual request origin, target order,
live branches, state associations and relevant memory. Marginal bounds or
matching means do not establish their joint realizability. A retained prefix
is not a completed call; refreshed diagnostics, lag increments and invocation
ordinals may survive a later failure on the sequential native path. Distinguish
native rejection, observer interruption, unavailable temporal evidence and
atomic-stage rollback. Ideal reductions and asymptotic theorems also retain
their own arithmetic premises; they are not bitwise runtime guarantees.

The following names are not supported `GRAMMAR_CANON` controls or
`--grammar.*` CLI options: `zhir_requires_oz_window`,
`zhir_dnfr_min`, `thol_min_len`, `thol_max_len`, `thol_close_dnfr`, and `si_high`.
Remove these options from CLI invocations; arbitrary mapping extensions retain
no admission meaning. The shared U4 context and measured Mutation gate own
their actual requirements. Words supply their closure explicitly; no sense-index
threshold chooses a THOL closure. There is no one-to-one threshold replacement.

Coupling's four `UM_*` branch flags and `OZ_NOISE_MODE` use the same Boolean
parser before active-factor resolution and proposals. Transition's `NAV_STRICT`
and `NAV_RANDOM` retain their stricter Boolean-only domain across public classes,
direct glyphs and stages; strings are rejected. Disabling a branch removes only
that branch's factor requirements. It does not waive live U3 or other operator
preconditions. The [operator maps](../theory/STRUCTURAL_OPERATORS.md) and
[event contracts](contracts/OPERATOR_EVENTS.md) specify the execution effects.

<a id="program-compilation-contract"></a>
### Scoped program compilation and execution

The existing [typed tokens](../src/tnfr/tokens.py),
[parser/compiler](../src/tnfr/flatten.py) and
[program executor](../src/tnfr/execution.py) own the `THOL`, `WAIT` and `TARGET`
program syntax used by the specialized CLI sequence route. This is separate
from complete-word validation and the atomic operator-stage route. Compilation
does not admit a child NFR, choose a branch or install a continuous law. The
[source comparison](../theory/UNIFIED_GRAMMAR_RULES.md#source-grammar-and-notation)
retains the wider hierarchical and prosodic proposals with their open premises.

`parse_program_tokens` returns owned typed bodies for explicit THOL blocks;
plain iterable grouping remains sequential. `compile_sequence` returns ordered
`(OpTag, payload)` operations. `BLOCK_ENTER` and `BLOCK_EXIT` carry immutable
`BlockBoundary(path, iteration, repeat)` metadata identifying nested occurrences
and zero-based body repetitions. These markers neither execute a structural
operator nor advance time. Consumers of compiled streams must retain them;
filtering only glyphs loses program scope. `THOLEvaluator` retains its iterator
interface as an unscoped token view, not the execution representation.

The THOL opening executes once per block invocation. Each body repetition
inherits the enclosing targets; TARGET changes are local to that repetition
and are restored at its exit. There is no automatic child-target selection.
Before any explicit TARGET, the target is the current graph population at each
glyph; `TARGET(None)` instead captures the population when that token executes.
Ordered explicit target collections preserve their supplied order.

Program construction follows these explicit contracts:

| Input | Contract |
| --- | --- |
| Omitted THOL closure | `force_close=None`, or `block(..., close=None)`, adds no glyph. Supply `Glyph.NUL` explicitly when that action is intended. |
| Explicit closure glyph | Every supplied registered glyph is emitted after each repetition and remains subject to live admission. Its position does not make it a valid grammatical closure. |
| Nested TARGET assignment | Targets restore at each repetition exit. Put a TARGET after the block when subsequent operations require that selection. |
| WAIT/repeat count | Counts must implement the integer-index protocol, be positive and be non-Boolean. Floats, fractions and strings reject, including integer-valued ones. |
| Nested or reused group | Cyclic ancestry rejects; repeated occurrences of an ordered group remain present. Unordered sets, unknown token fields and malformed glyphs reject. |

`max_materialize` is a nonnegative integer or `None`; the default is 1000.
The finite bound covers parsed input and expanded operations, including scope
markers. `play` completes syntax admission and compilation before changing graph
history. Live targets, grammar and preconditions are still checked during
execution; earlier accepted operations and callback effects are not rolled back
if a later operation fails.

Requested program glyphs use the native direct-glyph path and reject blocked
requests without substitution, even when the graph's ordinary rejection policy
is `fallback`. That original policy is restored before the step function runs.
No future-handler context is invented across changing targets or scopes.
The supplied `step_fn`, or the existing native runtime by default, runs after
each actual glyph, including THOL and an explicit final glyph. WAIT requests
additional step-function calls. The default step can select more operators and
update phase/capacity: WAIT is not an event-free interval, and its count is
neither seconds nor a newly derived clock.

The bounded `history["program_trace"]` distinguishes supplied program requests
from subsequent step-function execution. A glyph/THOL row is appended after
successful glyph dispatch and before stepping, with `selection_origin` set to
`supplied_program`, ordered pre-dispatch `targets`, `event_time`, and
`event_boundary="after_requested_glyph"`. A later-created node cannot become a
recorded target of that earlier dispatch. TARGET rows retain ordered node IDs;
WAIT rows describe requested step calls and make no glyph-event claim.

Rows begin with `step_status="pending"` and `record_boundary="before_step"`.
They become `completed`/`after_step` on success, preserving the existing `t`
field as the post-step clock. Failures retain a `failed`/`step_failed` row and
the original exception; there is no rollback. Requested, started and normally
returned call counts are `step_calls_requested`, `step_calls_started` and
`step_calls_completed`. A returned call can still fail clock admission.
`step_origin` distinguishes `native_runtime` from `supplied_step_function`;
neither value enumerates operators selected internally by that function.

The initial clock and each recorded clock sample use shared finite represented
real admission. Missing `_t` uses the explicit zero convention. Boolean,
nonfinite or nonzero unrepresentable clocks reject. If a callback leaves an
invalid clock, its effects remain, the row retains its last admitted `t`, and
`clock_sample_available=False` marks the unavailable current sample. A rejected
glyph dispatch creates no completed-glyph row; earlier target effects can still
exist on this non-atomic route. Trace truncation and callback effects prevent
using missing rows as proof of no event or as a complete execution checkpoint.

The `canonical_example` program preset requests Emission → Coherence → Silence
(`AL → IL → SHA`). It is distinct from the SDK's registered `basic_activation`
word. Presets retain their declared actions and contextual admission needs;
`contained_mutation` and `coupling_exploration` request Contraction (NUL)
explicitly. Controls:
[program admission and execution](../tests/test_program_execution.py).

<a id="removed-unsupported-helpers"></a>
### Unsupported calls and current replacements

The following interfaces are not part of the current API. Use the declared
replacement where one exists; no compatibility alias substitutes a different
mathematical model. Source recovery belongs to the
[recovery index](../theory/research/archive/README.md#recovery-records).

| Unsupported interface | Current replacement or scope boundary |
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
| Private duplicate start/end/U2 checks in `grammar_patterns` | Use the shared public word validators. Their core checks delegate to `GrammarValidator`; explicit `legacy` retains only additional compatibility preferences. Private helper names are not independent admission contracts. |
| `tnfr.utils.clear_orjson_param_warnings` (also in `tnfr.utils.io`) and `tnfr.config.presets.legacy_preset_guidance` | Remove these no-op calls. Shared JSON admission/encoding and `get_preset()` retain their existing behavior; unknown presets still raise `KeyError`. |
| Global `tnfr.validation.config.ValidationConfig` fields `epi_range`, `vf_range`, `phase_coupling_threshold`, `cache_validation_results`, `max_validation_time_ms` | Removed unconsumed settings. Scalar domains, live U3 admission and the separate cached-validator configuration retain their own owners. Construction rejects the removed keywords; `configure_validation()` rejects them without committing other supplied settings. |

<a id="arithmetic-application-boundaries"></a>
### Arithmetic application boundaries

The [basic primality utility](../applications/primality-test/README.md) retains
static arithmetic descriptors and a numerical predicate. Its CLI and
`python -m tnfr_primality` use the same basic implementation; advanced
adapters, advanced/legacy console names and the `full` extra are retired.
Core arithmetic pressure and residue-graph theorems retain their own premises
and implementations.

`tnfr.factorization`, `TNFR.factorize` and `Network.factorize` are deprecated
compatibility bridges for a separately supplied legacy `tnfr_factorization`
package. They do not discover or activate the
[archived lab](../applications/factorization-lab/README.md).
An absent or archived implementation raises retirement guidance; an admitted
external package emits `DeprecationWarning`. Historical factor-candidate,
acceptance and completeness labels are not independent arithmetic certificates.

The SDK self-optimization helpers keep generic manifest orchestration in the
engine. They require the checkout scripts and an explicit graph payload;
modulus-only lab reconstruction is no longer supported. Wheel-only calls
without those drivers fail explicitly. These helpers propose and assess a
dry run, do not dispatch operators, and return no promotable recommendations.
The default operation-to-regression mapping is empty: unmapped operations
remain pending, rather than inheriting an unrelated arithmetic suite's result.
The [runner contract](../scripts/README.md) owns serialization and observations.

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
`tnfr.validation.StructuralValidationConfig`, supplies invariant flags and
minimum severity consumed by `structural.run_sequence`. Its
`enable_semantic_validation` and `allow_semantic_warnings` fields remain accepted
compatibility settings; neither disables the shared grammar or activates an
independent heuristic gate. `SequenceSemanticValidator` retains its report
interface by delegating to shared word admission. The existing facade
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

Closed scalar configuration intervals use `bounded_represented_real` from
`_exact_time.py` to check the original admitted source against their bounds.
An exact value outside the interval cannot enter by rounding to an endpoint;
floating sources and bounds exposing exact integer ratios use those ratios
for comparison, retaining precision beyond binary64 when available.
Accepted values retain the consumer's binary64 model. This boundary applies
to selector/adaptation controls, stability fractions and active glyph factors.
It does not prohibit rounded arithmetic already declared by a numerical model.

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

`metrics.export_metrics` exports one detached snapshot of retained history;
it does not initialize or repair history on the graph. CSV and JSON admit the
same scalar values before any destination replacement. Missing samples,
clocks and glyph entries remain JSON `null` or empty CSV cells; an observed
zero remains zero. Sigma columns align by retained position, without borrowing
another stream's timestamp or substituting a row index. Present Boolean,
nonreal, nonfinite or unrepresentable numbers reject. This replaces the earlier
zero-filling and inferred-clock behavior; consumers must handle availability.
Each file uses the shared atomic writer, but the CSV file set is not one
transaction if a later filesystem write fails.

`Network.history()` shares retained-sample admission with the metric exporter.
It returns detached `kuramoto_R`, `C_steps`, `phase_sync` and `Si_mean` series.
An absent series is empty; an explicit missing sample remains `None`, so callers
must retain that availability when analyzing the series. Lists, tuples, deques,
other non-text sequences and one-dimensional NumPy arrays preserve stored order.
Unordered collections, one-shot iterators and multidimensional arrays reject;
readers do not consume an iterator, flatten coordinates or repair the graph.
Present samples reject Boolean, nonreal, nonfinite and underflowed values
before conversion. These series have no implied clock or formation certificate.

SDK scalar and raw-map readers raise `ImportError` when their field or balance
implementation is unavailable. In particular, `structural_charge()` and
`candidate_energy()` cannot report an unavailable calculation as zero.
The structured `conservation()` report retains its explicit availability
envelope; missing capability and an observed zero are distinct outcomes.

`NuFWindow` admits nonnegative integer counts, positive represented durations
and each supplied finite timestamp before conversion. Booleans, fractional
counts, malformed lone bounds and nonzero clock underflow reject. Two bounds
must describe a nonempty window consistent with its duration. The
`NuFTelemetryAccumulator` retains exact sums of represented durations, so
evicting a large window cannot erase smaller retained windows. Counts, clock
totals, estimates and unit conversions are admitted before a recording changes
the window buffer or graph history; unrepresentable results reject.
Configuration is validated even when an accumulator already exists. Explicit
empty graph/mapping sinks take precedence over an attached graph.

These are observed count-rate estimates under a supplied Poisson model, with
approximate normal confidence intervals. Independence and nonoverlap are
observation premises, not reducer guarantees. The estimate does not replace
the capacity in a nodal law; the configured `HZ_STR_BRIDGE` is a conversion
convention until independently calibrated. Empty evidence yields unavailable
estimates. Controls: [count-rate telemetry](../tests/test_nu_f_telemetry.py).

The standard observer supplies an unstable-node count proxy, derived from the
configured pressure/rate thresholds. It is neither a count of operator events
nor a measurement of primitive capacity. A matching `BEFORE_STEP`/`AFTER_STEP`
pair requires a fresh stability observation and positive actual elapsed time;
requested `dt` does not replace that interval. The runtime's initial zero clock
is an explicit convention. Missing, stale or invalid stability evidence marks
the observation unavailable without adding a zero or all-node count. A second
`AFTER_STEP` cannot reuse the same interval. Metrics registration places its
producer before this consumer regardless of attachment order, preserving other
callbacks' relative order and existing telemetry configuration. Controls:
[standard observer evidence](../tests/test_standard_observer_telemetry.py).
The completed window publishes its existing snapshot and updates only the
current observation's metric-history entry, without a second append. An
unavailable observation preserves prior successful telemetry.
Repeated diagnostic registration is idempotent, including strict callback
registration; worker counts are resolved from live configuration when observed.
Only advanced glyph-duration diagnostics consume the configured `DT`, admitting
it before metric writes. That configured operation-time basis is distinct from
the standard observer's measured elapsed interval.

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
All-node pulse and dissonance observations share one live trigonometric capture
across their weighted local rows. Sparse affinity entries are partitioned once
per call, checking every index and weight before reducing rows. Entry order,
duplicates and diagonal admission are retained.
Each new observation detects raw phase/support edits; the public one-node reader
still revalidates live state independently. No public input can supply a trusted
snapshot, and the shared capture is not a persistent observation. Weights reject
negative, nonfinite and nonzero-underflowing inputs before acceptance. Ordinary
row summation order is unchanged; extreme finite weights use common positive
rescaling to avoid overflow and subnormal-product loss in the weighted mean.

The [declared-cycle winding reader](../src/tnfr/physics/winding_certificates.py)
uses shared represented-real admission before circular reduction. A nonzero
phase that would materialize as zero makes the observation undefined; an
invalid authoritative alias cannot fall through to a secondary phase. Invalid
or underflowing `branch_tolerance` and `phase_gate` raise `ValueError`; exact
zero remains permitted. The word observer rejects invalid initial phase
before executing a step. A defined sampled winding does not certify identity
between observations or transport of a localized organization.

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

The ten historical invariant-check identifiers retain API compatibility; they
do not replace the six canonical invariants. Scalar checks admit authoritative
aliases through the shared domain owners: uniform signed-real BEPI is valid
form, while complex/nonuniform form cannot be projected into admission.
Nonnegative capacity includes zero even at nonzero pressure. Configured bounds,
edge-phase thresholds and low-capacity/high-pressure alerts remain diagnostic
policies, separate from live U3 admission or NFR existence. A false optional
operator-history marker with changed form produces a warning; that marker
neither forbids declared continuous flow nor identifies the change's cause.
See the [validator controls](../tests/test_legacy_invariant_admission.py).

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

`TNFRSelfOptimizingEngine` admits consumed graph scalars before binary64
conversion. Boolean/text/nonfinite values and nonzero values lost to zero
raise `TNFRValueError`. Invalid optional performance measurements remain
unavailable. Pressure magnitude and measured speedup, memory, charge-drift
and Si means use shared finite reductions, retaining representable means
even when a naive sum would overflow. EPI variance uses translated/scaled
dispersion and becomes `None` when its nonzero square cannot be represented;
that case does not trigger the low-variance policy. Exact zero remains valid.
These are numerical and policy-admission rules, not new nodal dynamics.
The [manifest workflow](../scripts/README.md#reproducibility-and-research)
owns input-file admission and report handling.

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

The shared scalar setters rebuild a cold absolute-maximum cache, or one
without a live witness, from the whole consumed channel. Untargeted nodes
therefore remain included after a selected-node write. Rebuilds admit
authoritative real capacity/pressure values and signed scalar EPI through
their respective shared owners; missing values retain zero, while invalid
present values reject. A tied maximum witness is an implementation choice,
not structural identity. Warm incremental caches require mutations to stay
within their declared setter scope; arbitrary raw writes require rebuilding.
Generic setters have no failure-atomicity promise. Snapshot operator stages
retain their separate graph-owned rollback contract if a rebuild fails.

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
| Dissonance | dissonance | OZ | delta_nfr | node | node | local &#124;ΔNFR&#124; not decreased in amplification mode; finite pressure in jitter mode |
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

<a id="remesh-postbirth-history"></a>
For participation after membership growth, use the existing live sampler
`tnfr.dynamics.remesh_history.append_remesh_epi_history_snapshot` only after
the new members exist. Explicitly starting observation there does not reset
any earlier records. The existing `execute_event_remesh_cycle` composes a
fixed-node schedule, one pre-map sample and the delayed map; a schedule that
creates a node is outside its domain. `apply_network_remesh` itself appends
no delayed row and supplies no advisory occurrence or grammar credit.

The direct map requires its two selected lag rows to match current support;
the event cycle requires every retained row to do so before the schedule.
A pre-birth row can therefore block the cycle even after newer rows admit the
direct map. Neither path fills a missing child's past. An insufficient-history
result is an explicit no-op, not successful recursion. Index delays do not
specify elapsed time without a separate sampling-clock law.

Both `REMESH_TAU_LOCAL` and `REMESH_TAU_GLOBAL` select past values of the same
node from the same population snapshot history. They are not constituent and
collective-memory stores, nor spatial means. The
[structure/memory analysis and actual EN/RA continuation](../theory/REMESH_STRUCTURE_AND_MEMORY.md)
explain these distinctions and the remaining historical cross-scale role.

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

Coupling also supports `Coupling()(G, i, potential_partner=j)` and the
`potential_partner` keyword of `execute_network_operator_stage`. This request
defaults to `potential_scope="isolated"`: the shared atomic UM stage with one target and a totally isolated
partner. The target may already have support: its U3-compatible outgoing
neighbors in their existing order, followed by the requested partner, enter
the same phase, optional capacity and pressure map. The partner must itself
pass U3; another compatible neighbor cannot admit an incompatible request.
Only the requested new link is proposed. Existing edge attributes are
preserved; bidirectional phase writes also affect participating old neighbors
without executing UM or adding history on those neighbors. Final U3 covers
target-participant relations, not every other relation incident to a moved
neighbor. Incoming-only neighbors do not participate in this outgoing map.
Initial/final U3, enabled functional links, configured affinity
and actual UM grammar admission are mandatory; a fallback action does not
satisfy a creating request. The
[potential-pair contract](../theory/STRUCTURAL_OPERATORS.md#coupling-potential-pair)
owns its complete state and continuation limits. Omitting the keyword preserves
ordinary execution. Other operators reject it at the network dispatcher;
the all-node word and hybrid schedule APIs do not select potential partners.
An actual THOL child ID can supply the partner, but ancestry metadata neither
selects the invocation nor authenticates its provenance. No temporary edge is
inserted for admission, and two already-connected endpoints remain outside
this default domain. In directed graphs the new edge is target to
partner: subsequent RA at the target can read the partner, while EN there
still uses only incoming sources. Each continuation has its own admission.

<a id="connected-organization-coupling"></a>
#### Coupling between disjoint connected organizations

`Coupling()(G, i, potential_partner=j, potential_scope="disjoint_components")`
uses the same atomic stage and numerical UM map. The shared
`execute_coupling_stage` and `execute_network_operator_stage` accept the same
keywords. Scope accepts only `isolated` (default) or `disjoint_components`;
the latter requires a partner. An explicit request must execute one canonical
UM target, with its own live grammar; fallback is not successful creation.

The new scope requires an undirected non-multigraph, distinct whole connected
components containing at least two identities each, an absent requested edge
and enabled functional links. Zero-weight support counts. A self-loop alone
does not establish a second constituent. Compatible old target neighbors and
the mandatory partner enter the shared phase/capacity/pressure map. Only the
target receives the operator occurrence; other phase participants keep their
own history and counters.

All old edges incident to the original phase-write set, plus the requested
relation, must satisfy effective U3 initially and after the represented map.
Additional boundary endpoints do not become consensus participants. After
postchecks and callbacks, the same original write set and admitted limit
(or a stricter live limit) remain binding. Node identities, old support and
old edge attributes are preserved, with exactly the requested new edge.
Failure rolls back graph-owned invocation state. External callback effects
are not transactional. Distant unrelated phases/edges are not consumed.

The [derivation and continuation controls](../theory/COUPLING_COMPOSITION.md)
distinguish these guarantees from ordinary UM's existing candidate-link
creation and the default isolated-partner request. They do not supply partner
discovery, collective identity or autonomous occurrence. New-scope stage
telemetry adds `scope` to `potential_pair`; default telemetry is unchanged.

<a id="ordinary-coupling-relation-preservation"></a>
#### Preserving existing relations in ordinary Coupling

`Coupling()(G, i, preserve_relations=True)` routes the ordinary request through
the atomic singleton UM stage. `execute_coupling_stage` and
`execute_network_operator_stage` accept the same Boolean keyword; the latter
admits an enabled request only for UM. The public call still returns `None`;
the stage and dispatcher retain `NetworkStageResult`. Omitted or `False`
requests retain legacy behavior. Strings, integers and `None` are invalid
request values, including on an empty-target branch.

The atomic stage retains its existing transaction metadata domain. Builtin
container cycles are supported; embedding a NetworkX Graph object in edge
metadata is rejected before this request executes. The option does not extend
that admission domain or assign it to the legacy direct public path.

The enabled domain requires canonical Coupling, a nonempty target tuple, an
undirected non-multigraph and `UM_FUNCTIONAL_LINKS=False`. Self-loops and
zero-weight edges are allowed. No potential partner or alternative potential
scope may be combined with it. Multiple targets use the existing common
snapshot and phase merger. Live grammar still applies; a substituted action
cannot satisfy the explicit UM request.

The request preserves all existing edges incident to every original proposed
phase writer, including secondary neighbors, zero displacements and writes
cancelled by the merger. It admits that boundary initially, after the joint
proposal and after postchecks/refresh, retaining the original or stricter live
U3 bound. Extra boundary endpoints do not become consensus inputs. Node
identities, support and edge attributes remain fixed. A preflight change cannot
replace the original source with a more permissive baseline.

Failure restores the graph-owned invocation, including its histories and
telemetry; external callback effects are outside rollback. The guarantee is
about the returned relation boundary, not arbitrary callback metadata,
continuous-time safety, bounded capacity or future compatibility. Normal own
UM events belong only to the actual targets. The
[derivation and controls](../theory/COUPLING_COMPOSITION.md#ordinary-coupling-relation-preservation)
distinguish this fixed-support request from disjoint-component link creation.

<a id="birth-association-coupling"></a>
### State-derived birth-association access to Coupling

`observe_birth_association_access(G, parent)` returns a passive
`BirthAssociationAccess` with the parent, ordered associated, isolated and
connected children, and policy `unique_isolated_birth_child`. Status is
`no_associated_child`, `no_isolated_child`, `ambiguous` or `selected`;
`selected_partner` is available only for a singleton isolated set.
Missing `sub_nodes` means empty; present containers and every referenced
child/parent relation must be valid. Missing hierarchy/index entries retain
THOL's legacy behavior, while a present conflicting index rejects. No global
genealogy audit or historical-origin authentication is implied.

`execute_birth_access_coupling(G, parent, execution_kwargs=None)` re-reads
current access and returns `BirthAccessCouplingDispatch(access, stage_result)`;
`executed` distinguishes actual UM from a side-effect-free abstention. It
accepts neither a partner ID nor a previously returned report. A unique child
still passes the actual atomic UM stage; multiple children abstain before
numerical eligibility, with no scoring or fallback. Directed incoming support
prevents isolation. Unlisted compatible nodes are outside this access domain.

The optional mapping accepts only `window`, `validate_preconditions`,
`collect_metrics`, `validate_nodal_equation`, `validate_postconditions` and
`dt`, using existing common admission. Preconditions must be requested,
canonical grammar enabled and rejection mode `raise`, even on abstention.
Unused UM numerical fields are admitted only if a candidate reaches the stage.
The full invocation rolls back graph-owned changes on failure, preserving
any birth completed beforehand; external effects are outside that guarantee.
The outer result records association-derived access; the inner stage retains
`potential_pair.origin="explicit_request"` at its concrete-argument boundary.
No scheduler, event clock or birth callback is installed. The
[theory contract](../theory/STRUCTURAL_OPERATORS.md#coupling-birth-association-access)
owns the policy and its relabeling/lifetime limits.

### Reception source requests

Reception also supports `Reception()(G, i, source_nodes=(j, k))` and the
`source_nodes` keyword of `execute_neighbor_stage` and
`execute_network_operator_stage`. The tuple is ordered, nonempty and unique;
each selected source must be an existing incoming neighbor of every target
and provide explicit admitted signed scalar EPI. Directed support uses
predecessors, including unique sources on multigraphs. The selected forms
determine the unweighted mean and the supplied order resolves semantic-kind
ties. Missing selected form, duplicate sources and empty selections reject;
there is no implicit fallback to all neighbors.

An explicit public request runs the shared atomic stage and returns `None`.
The same tuple is independently validated for each stage target, and all
proposals read one pre-state. Canonical Reception and actual EN grammar
admission are mandatory. Replacement by another glyph cannot satisfy the
request. The stage rolls back graph-owned state on failure; opaque external
callback effects are outside that guarantee. Omitting the tuple or using
`None` preserves ordinary execution, including its missing-form policy.

This is a supplied causal source selection, not an autonomous phase/capacity
filter. `track_sources` retains its separate telemetry read set and may report
sources that the blend did not consume. Snapshots and stage observations
retain `source_nodes`, ordered associated forms/kinds and
`source_selection_origin`; executor metrics additionally expose
`input_neighbors`. Standalone metrics only observe the live legacy read set
and identify it as `live_legacy_observation`, without reconstructing a past
selection. Existing auxiliary all-neighbor certificates abstain for every
explicit tuple, including a tuple equal to the full neighbor set. The
[causal-source contract](../theory/STRUCTURAL_OPERATORS.md#reception-causal-sources)
owns the mathematics and remaining compatibility question. Other operators
reject the keyword at the stage dispatcher; word and hybrid schedule APIs
do not choose source tuples.

For state-dependent source choice, use
`ReceptionCompatibilityPolicy(max_phase_gap, max_relative_capacity_gap,
mode="select")` and `execute_eligible_reception_stage(G, i, policy)`, exported
from `tnfr.operators`. Both numerical bounds are required. The policy uses
the shared circular phase distance and the dimensionless capacity inequality
`abs(nu_j-nu_i) <= eta * max(nu_i, nu_j)` on admitted represented capacities.
It supplies no measured-frequency interpretation or UM/RA U3 constraint.
`mode="require_all"` instead abstains unless the entire nonempty incoming
set is compatible. The [family and numerical scope](../theory/STRUCTURAL_OPERATORS.md#reception-compatibility-policy)
own the threshold, zero-capacity and symmetry boundaries.

The pure `observe_reception_compatibility` reads explicit phase/capacity for
the receiver and **every** incoming candidate, including excluded sources.
Invalid consumed state raises. It returns a sealed decision with policy,
ordered candidate coordinates and selected IDs; it does not admit EPI or
grammar. Existing self-loops remain candidates, so a nonempty decision does
not itself establish input from another NFR.

The conditional dispatcher handles one receiver per supplied invocation.
Its `EligibleReceptionDispatch` contains `compatibility` and either the
existing canonical `stage_result` or `None`; `executed` distinguishes them.
A source-rule abstention runs no EN, increments no operator history, writes
no stage schedule and calls no pressure refresh. A nonempty decision still
passes actual EN form admission, grammar and configured preconditions;
failures raise and a substituted glyph cannot satisfy the request. The stage
rechecks the complete decision against its own pre-state before commit.
Snapshots and observations retain the sealed `compatibility_decision`.
Executor metrics retain its primitive report projection under the same key,
without a proof seal; it is not execution evidence. These readers report
`source_selection_origin="compatibility_policy"`. Standalone metrics cannot
reconstruct such a past decision.

`execution_kwargs` supports `window`, `validate_preconditions`,
`collect_metrics`, `validate_nodal_equation`, `dt`, `track_sources` and
`max_distance`. Separate keyword arguments accept an optional
`compute_delta_nfr` callback and `include_epi_jump_certificate`; the latter
reports the existing selected-source certificate abstention on an executed
stage. An empty attempt has no stage certificate. All graph-owned failures
roll back, including callback failure; external effects are outside that
guarantee. The dispatcher does not install a recurring selector or event
clock, and it does not extend the fixed-target word/event APIs to partial
stages. The legacy and explicit-source APIs retain their original domains.

Resonance's optional strict preconditions and readiness report share admitted
state and finite nonnegative policy thresholds. Invalid thresholds reject
before direct or atomic-stage writes. The readiness flag covers the configured
form-magnitude, capacity, pressure-magnitude and connectivity checks; it does
not replace U3 or authorize execution. Its neighbor-mean phase warning retains
an unavailable direction at an exactly zero represented resultant, while invalid
consumed phases raise. These reads do not write graph caches.

### Retained members and scale actions

`capture_retained_organization(G, members)` accepts a nonempty exact tuple of
unique existing member IDs. It returns an owner-bound `RetainedOrganization`
with ordered `member_states`, incoming/outgoing neighbor tuples and incident
edge records, including multigraph keys and attribute signatures. Those port
tuples include internal neighbors as well as the external boundary. Parallel
edges retain their records but do not duplicate EN/RA neighbors. Boundary-node
coordinates are not admitted merely to describe a port.

Members must supply admitted signed scalar EPI, finite phase and pressure,
nonnegative capacity, and valid own grammar context. Kind may be absent or
empty. Raw member metadata and present temporal histories retain their
associations through immutable signatures; no missing history is fabricated,
and stale temporal samples are not promoted to current derivative evidence.
The complete graph remains the runtime state. The observation is neither a
portable checkpoint nor a scalar macro-NFR or a collective identity proof.

`retained_organization_matches_graph(G, record)` re-admits the member/port
association. A foreign graph, changed member state/context, changed incident
edge or changed neighbor order invalidates the record. Unrelated boundary
coordinates may change; the actual subsequent operator always reads the live
state it consumes.

`execute_retained_scale_stage(G, record, operator, execution_kwargs=None)`
accepts the canonical `Expansion()` or `Contraction()` and delegates to the
existing atomic pointwise stage on exactly those members. It requires the
requested two-phase action before fallback execution. Each member must pass
its own grammar and configured preconditions; grouping supplies no inherited
history or admission credit. The default scalar maps remain unchanged.

The `execution_kwargs` mapping accepts `window`, `validate_preconditions`,
`collect_metrics`, `validate_nodal_equation` and `dt`. This bounded consumer
accepts no refresh callback, sequence-context override or graph edit. It
checks that all node identities/support/edge attributes, nonmember node state
and member phases remain intact. Existing graph telemetry and the members'
own event context may change. Failure at admission, execution or the returned
observation restores graph-owned state; external effects are outside that
transaction guarantee.

The returned `RetainedScaleDispatch` contains `before`, `after` and the actual
`stage_result`. These are associated observations and a transaction summary,
not an authenticated execution certificate. Subsequent actions continue on
the actual graph and member IDs; a retained request must match its live
source again. The [mathematical scope](../theory/STRUCTURAL_OPERATORS.md#retained-member-scale-actions)
explains why averaging members or cloning donor ports cannot silently replace
this representation, and why proliferation/synthesis remain separate roles.

The shared `execute_pointwise_stage(..., require_exact_operator=True)` exposes
the same opt-in rejection of grammar replacement independently of any EPI
certificate. It accepts an actual Boolean; the default `False` preserves
ordinary fallback behavior. It requires the requested implemented two-phase
path, without supplying a theorem about organization or changing grammar.

### One explicitly derived Expansion member

`DerivedExpansionPolicy(source_order=..., form_ratio=..., capacity_ratio=...)`
requires all three keyword arguments. The order is `pre_val` or `post_val`;
both ratios must be admitted finite represented reals in `(0,1]`. This is a
bounded constitutive family, not a uniquely derived proliferation law or an
additional glyph. Ordinary `Expansion()` keeps its existing scalar semantics.

`execute_derived_expansion_stage(G, source, child, policy, execution_kwargs=None)`
requires a current singleton `RetainedOrganization` for the parent and a fresh
supplied child identity. It executes the actual canonical retained VAL stage
on that parent, then derives one isolated member from its admitted pre-VAL or
post-VAL state. Form and capacity use the same selected boundary and their
respective ratios. The child inherits the source phase and semantic kind;
stored pressure starts at zero by the declared birth law. Nonfinite or nonzero
underflowed products and child EPI outside configured bounds reject; birth
does not silently clip the formula. The ordinary parent VAL may still apply
its configured edge-aware projection.

The child has empty own glyph history and one initial EPI sample. It inherits
no parent debt, stabilization credit, elapsed time or temporal evidence. Its
lineage identifies the parent and selected order; this is not THOL hierarchy,
an acceleration-triggered THOL event or a claim of collective identity. No
parent port or edge is copied. Connect it separately through actual Coupling
(UM), with live U3, affinity and grammar admission; birth does not authorize
that later action. A failed later UM leaves the already committed birth intact.

The executor reuses the retained scale option whitelist and its rejection of
fallback replacement. An outer transaction covers parent VAL, child admission,
materialization and final observation: a failure restores graph-owned state,
not external Python side effects. Existing parent identity, support and
nonmember state remain intact. The result `DerivedExpansionDispatch` contains
the normalized policy, parent and child IDs, `inheritance_source`, actual
`scale_result`, and an `after` observation of `(parent, child)`. It is an
execution summary, not an authenticated invocation certificate or checkpoint.
The [constitutive analysis](../theory/STRUCTURAL_OPERATORS.md#derived-expansion-member)
owns order dependence, balance limits and the distinction from THOL and NUL.

<a id="retained-addressed-input"></a>
### Addressed Reception into a retained organization

`retained_input_ports(record)` derives a tuple of `RetainedInputPort` records
with `ordinal`, `member` and `source`. It requires an intact exact
`RetainedOrganization`. Addresses follow retained member order and incoming
unique-neighbor order, excluding all internal members. Undirected edges supply
inputs; directed reverse-only arcs do not. Parallel edges give one donor
address while their separate attributes remain retained. Empty tables are
valid passive observations. A table is not current execution authorization.

`execute_retained_reception_stage(G, record, port_index, execution_kwargs=None)`
re-admits the owner-bound organization and resolves the exact nonnegative
built-in integer address in that fresh table. Booleans, invalid indices,
foreign/tampered records and changed member state/context or incident support
reject. There is no implicit refresh, fallback or empty successful action.
External donor coordinates and kind are live: the canonical EN snapshot reads
them at execution. Generic retained kind normalization does not replace EN's
actual kind semantics. One selected external donor supplies one actual EN at
the resolved member; the caller passes no separate target or source IDs.

The only execution options are `window`, `validate_preconditions`,
`collect_metrics`, `validate_nodal_equation` and `dt`. Local precondition
validation defaults to and must remain true. Canonical grammar must be enabled
with rejection mode `raise`; the existing global optional precondition policy
retains its own meaning. Source tracking is fixed off. Source/context overrides,
callbacks, pressure refresh and alternative glyphs are unsupported. The inner
EN observation keeps `explicit_request` provenance; the outer resolved port
records organization-relative addressing.

The graph-owned transaction begins before options materialization. Failure
restores the invocation, including post-execution observation and protection
checks. Every non-target node, support and edge attribute, all phases, and the
target's capacity/pressure are preserved. Target form/kind, own event context
and ordinary graph telemetry may change. External effects are not transactional.
`RetainedReceptionDispatch` retains `before`, `after`, `port` and `stage_result`;
these associated summaries are not authenticated receipts. The after record
can be consumed again if fresh; a before record with changed member context
cannot. Subsequent RA is separate, with its own admission and transaction.
The [conditional continuation](../theory/NUL_SYNTHESIS.md#retained-addressed-input)
distinguishes retained interior state from the live external input.

<a id="retained-addressed-output"></a>
### Addressed Reception from a retained source group

`retained_output_ports(record)` passively enumerates `RetainedOutputPort`
records with `ordinal`, `receiver` and `sources`. Every source is an actual
member of the intact retained organization, in its declared order. An external
receiver appears only when it is an outgoing neighbor of **every** member.
Ordering follows the first member's retained outgoing unique-neighbor order.
Directed arcs must point from member to receiver, the incoming direction used
by EN. Internal members and self-loops cannot create external output ports;
parallel edges do not duplicate sources. An empty table is a valid observation.

`execute_retained_output_reception_stage(G, record, port_index,
execution_kwargs=None)` re-admits the source record and resolves that exact
nonnegative built-in integer ordinal. It executes one actual EN at the external
receiver with `source_nodes=record.members` and source tracking disabled. The
caller supplies neither an alternate receiver nor a replacement source list.
Missing, foreign, stale or unsealed associations reject without fallback.
This reuses the ordinary unweighted mean over unique selected sources; edge
weights do not become averaging mass. It is not a new macro-node or a new glyph.

Source member state, own context/history and incident support remain bound to
the retained record. Capture again after a source changes. External receiver
form, kind and own context remain live until invocation; changing them alone
does not stale the source group. Actual EN semantic kinds retain their exact
string-valued interpretation rather than the generic record's normalization.
The conservation guard admits finite nonnegative receiver capacity and finite
pressure through shared primitive readers, with their zero defaults. It adds
no finite-phase check or retained-member kind restriction to the receiver.
The options, enabled grammar/raise requirement, local precondition validation
and existing global policy match the [input consumer](#retained-addressed-input).
Callbacks, source/context overrides and pressure refresh remain unsupported.

One graph-owned transaction covers option materialization, live admission,
execution and final checks. All source members, other non-target nodes, support,
edge attributes and phases remain unchanged, as do receiver capacity and
pressure. Receiver form/kind, own event context and ordinary graph telemetry
may change. A failure restores this invocation; external effects and a later
separate RA are outside that transaction. `RetainedOutputReceptionDispatch`
contains `before`, `after`, `port` and the actual `stage_result`. Its unchanged
source observations and resolved port are consistency-checked summaries, not
authenticated invocation receipts or authorization to reuse a stale record.
The [structural re-expression owner](../theory/REMESH_STRUCTURE_AND_MEMORY.md)
keeps this interface distinct from a complete cross-scale REMESH law.

<a id="retained-collective-emission"></a>
### One member realizing a declared collective Emission gain

`execute_retained_emission_stage(G, record, member_index, mean_gain=b,
execution_kwargs=None)` resolves one member of a freshly re-admitted
`RetainedOrganization`. The ordinal is an exact nonnegative Python integer,
excluding Boolean, within the ordered member tuple. The caller supplies that
address and the occasion; this interface does not search for another member.

The gain must be a finite positive represented real. With N retained members,
the existing resolved `AL_boost` must equal `N*b` as represented rationals.
The consumer checks this relation and never changes factor configuration.
The shared prospective AL proposal must realize that exact local increment;
clipping, overflow or rounded addition that breaks it rejects this exact-gain
interface. This is a restricted realization contract. Ordinary Emission keeps
its existing projection/rounding behavior, and a supplied rational gain is
interpreted through the shared binary64 materialization contract.

The chosen member must pass actual strict Emission entry admission even when
the graph's optional global precondition switch is disabled. Enabled grammar
with rejection `raise` and local `validate_preconditions=True` are mandatory.
The common options are `window`, `validate_preconditions`, `collect_metrics`,
`validate_nodal_equation` and `dt`; target, context, callback, refresh and factor
overrides are unsupported. The stage executes one actual canonical AL, using
its shared proposal, lifecycle and own grammar. An already activated target
must have valid counted lineage; another member's origin cannot supply it.
Other members need not satisfy AL's latent threshold because they do not act.

The graph-owned transaction starts before option materialization and covers
live admission, execution and final observation. It preserves all non-target
nodes, support/edge attributes and phases, target capacity/pressure/kind aliases,
factor configuration and runtime clock. The selected member's actual AL form,
activation, latency clearing and grammar/history updates remain its own. Final
checks verify the declared mean increment and the admitted lifecycle patch;
a late failure restores the invocation. External effects and later EN/RA calls
are outside this transaction.

`RetainedEmissionDispatch` exposes `before`, `after`, `member_index`, `member`,
`mean_gain`, `local_boost`, `realized_mean_gain` and `stage_result`. The gains
are exact `Fraction` observations of represented values. The checked summary
does not authenticate an invocation or prove complete collective AL descent.
The [port realization and its admission boundary](../theory/REMESH_STRUCTURE_AND_MEMORY.md#collective-emission-port-realization)
explain the actual acquired-source comparison and why a mean is insufficient.

<a id="retained-synthesis"></a>
### Retained pair concentration through NUL and EN

`capture_retained_pair(G, (core, member))` requires two distinct existing
members in an exact ordered tuple. Its `RetainedPairState` contains the
underlying owner-bound `organization` and exact `Fraction` values `form_mean`
and `form_contrast`, computed from the admitted binary64 forms. Both member
states, own contexts and ordered ports remain retained. Properties `members`,
`core` and `member` expose that supplied role order. Capture alone requires
no edge and changes no state. Fractions preserve represented real arithmetic,
including halves below the smallest float; they are not a bitwise signed-zero
encoding or a complete state/checkpoint. Runtime actions use actual members,
never a rounded reconstruction from these two observations.
The additional `reception_kinds` tuple retains EN's actual string-valued kind
read: an absent kind yields `""`, while a stored `None` yields `"None"` under
that existing reader. This differs from the generic retained member's empty
kind normalization. Execution re-admits both source associations instead of
silently treating those observations as interchangeable.

`execute_retained_synthesis_stage(G, pair, contraction_kwargs=None,
reception_kwargs=None)` re-admits the pair and requires an existing incoming
`member -> core` arc or their undirected edge. It executes actual simultaneous
Contraction (NUL) on both members, then Reception (EN) on the core using only
that member as its explicit source. A lone `core -> member` arc is insufficient;
the wrapper creates no edge and does not substitute for an admitted UM action.
This is a declared concentration fragment. It does not modify ordinary NUL,
add a glyph, remove a constituent or prove historical reabsorption/synthesis.

Each substage accepts only `window`, `validate_preconditions`, `collect_metrics`,
`validate_nodal_equation` and `dt` through its corresponding mapping. Its own
grammar and factor/precondition owners remain authoritative. EN source tracking
is fixed off; arbitrary source, context, callback and fallback overrides are
unsupported. Neither operation supplies U2 stabilization credit. The existing
factor domains remain `0 < NUL_scale < 1` and `0 < EN_mix <= 1`; a comparison
without reception uses ordinary NUL, not an invalid zero EN coefficient.
After `inject_defaults`, changing `NUL_scale` also requires updating or removing
the explicit `NUL_densification_factor` entry. An omitted entry is derived as
`1/NUL_scale`; a contradictory retained value rejects rather than overriding
that relation. Densification is not an independent factor.
NUL's increase of stored positive pressure can itself block the subsequent
EN at its configured ceiling. Valid NUL admission does not promise that the
whole fragment is admitted; selecting another ceiling is an explicit policy.

The outer transaction includes both stages, preservation checks and final
observation. Failure at the second stage restores the pre-NUL graph as well.
Support, edge attributes, member phases and nonmember state remain intact;
EN also preserves the member's post-NUL state/context. Graph telemetry and
each actual target's event bookkeeping retain their ordinary semantics.
External side effects remain outside graph-owned rollback.

`RetainedSynthesisDispatch` retains `before`, `contracted` and `after` pair
observations plus the actual `contraction_result` and `reception_result`.
These are associated transaction summaries, not authenticated invocations.
Its form chart does not replace capacities, pressure, phase or grammar state.
At full EN mixing, old form contrast may be lost from current forms; a
before-observation does not make the executed action reversible. The
[mathematical owner](../theory/NUL_SYNTHESIS.md#retained-synthesis-concentration)
states the unclipped concentration law, numerical limits and open roles.
The [continuation controls](../theory/NUL_SYNTHESIS.md#nul-future-response)
also distinguish a first matching output from sufficient future response.
The actual graph remains necessary: retained child form can affect a later
RA/EN/RA word, and own grammar can change OZ admission even when primitive
coordinates agree. Neither the core form nor a returned observation authorizes
pooling member histories, removing a constituent or executing a macro-node.

### Retained Mutation and relational participation

`execute_retained_mutation_stage(G, organization, targets,
execution_kwargs=None)` re-admits the existing owner-bound
`RetainedOrganization`. The explicit `targets` must be a nonempty tuple of
distinct retained members. It executes actual canonical Mutation (ZHIR) through
the shared atomic pointwise stage; it supplies no trigger, copied history,
pooled context or replacement glyph. A requested grammar fallback rejects.

Each target needs its own active capacity, admitted signed-growth evidence and
grammar. The existing temporal contract remains authoritative: timestamped
evidence ends at the live EPI value, while legacy histories retain their
unit-operator-step basis without a live-endpoint guarantee. Neither evidence
format authenticates measured provenance or derives an occurrence law. Prior
births and zero-duration jumps do not supply elapsed-time samples.

The option mapping admits `window`, `validate_preconditions`, `collect_metrics`,
`validate_nodal_equation`, `dt`, `tau` and `validate_postconditions`, as consumed
by the existing stage. It admits no pressure callback, source or context
override. The hard Mutation trigger and actual grammar are not replaced by
optional activity or postcondition checks.

The transaction preserves support, edge attributes, all non-target state and
the target's raw form, kind, capacity and stored-pressure aliases. Target phase,
own event history and ordinary lifecycle telemetry may change. Capture,
execution, final observation and preservation checks share one graph rollback
boundary; external side effects retain the shared transaction limitation.

`RetainedMutationDispatch` contains `before`, `after`, ordered `targets` and
`stage_result`. The latter's sealed `mutation_decision_observations` retain the
actual per-target decision, temporal certificate and represented phase map.
The summary checks target coordinates/context; it does not reconstruct a
certificate from the retained observation's opaque history signatures. A
publicly assembled summary cannot authenticate which history produced it.
It is neither a checkpoint nor proof of collective identity. Mutation keeps
the scalar EPI unchanged, but
its phase change can alter a subsequent compatibility-selected EN source set
and therefore the next EN/RA form response. This is an indirect consequence
through existing operators; richer-form mutation and autonomous occurrence
retain their own open obligations. See the
[retained Mutation analysis](../theory/STRUCTURAL_OPERATORS.md#retained-mutation-participation).

## Nodal solver input, clock and output boundaries

The shared [nodal integrator](../src/tnfr/dynamics/integrators.py) reads the
authoritative capacity, pressure and previous derivative aliases: invalid
provided values cannot fall through to a later alias or a zero default.
Capacity must be finite and nonnegative; pressure and retained derivatives
must be finite represented reals. Missing values retain the documented defaults.
The step, initial time and minimum step use the same raw scalar admission:
Boolean/text inputs and nonzero values lost to zero during conversion reject.
Use explicit numeric
`0.0` or `1.0` when those durations are intended. Admitted zero-duration calls
remain no-ops.

Runtime consumption retains the authoritative timestep until shared
represented-real admission, including values changed by `BEFORE_STEP`
callbacks or pressure hooks. Earlier callback/pressure effects are not rolled
back by this check. The auxiliary-engine handoff retains the duration supplied
to the nodal integrator, rather than rereading a mutable `DT` afterward.
The separate mechanical symplectic steppers use the same
raw admission for signed timesteps, retaining negative reversible stages and
zero-step identity.

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

The separate `rk4_coupled_step(state, rhs, *, t, dt)` accepts a nonempty tuple
of complete real coordinates and a callback returning the same-size tuple of
rates. It refreshes that callback at every immutable full-state RK4 stage,
reusing the existing Euler and quadrature arithmetic. Every coordinate, rate,
time and output receives represented-real admission; the positive duration
must retain strictly ordered start, midpoint and endpoint clocks. Invalid
dimensions, nonfinite results and nonzero results lost to zero reject.
This explicit helper supplies no model, clipping, phase wrapping, graph writes,
history or accuracy certificate. The caller retains law, Gamma, clock and
observation association; callback effects are outside rollback. The ordinary
graph integrator's held-pressure semantics remain unchanged. Independent
[coupled-step controls](../tests/test_coupled_rk4_step.py) exercise its numerical
and admission boundaries.

The extended scalar flux divergence retains its unique outgoing-neighbor law
`sqrt(k) * (J_i - mean(J_j))`, including self-loops and excluding parallel-edge
multiplicity. It reuses shared neighbor-difference arithmetic, so finite common
offsets and mixed-sign cancellation do not create spurious overflow or
divergence. Invalid consumed inputs and unrepresentable final values reject.
This scalar contrast is not an incidence divergence or a conservation theorem.

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
nonnegative capacity is required even when amplification is inactive. Transition
also rejects a positive capacity product lost to zero, preserving intentional
zero factors and representable subnormals through shared scalar admission.
Graphless unidirectional Coupling updates only the target phase. On ordinary graph paths, UM/RA
means use unique outgoing support neighbors, including zero-weight edges;
parallel multiplicity and incoming arcs do not supply additional samples.
Default isolated potential-pair UM instead uses its one declared partner and creates
the directed target-to-partner arc when applicable, while retaining compatible
outgoing neighbors in its shared synchronization map. Live grammar admission
receives this request; `validate_candidate(..., potential_partner=j)` can
check the same domain for UM. It supplies no U1/U2/U4 credit and does not
select a partner for ordinary proactive selectors.
The [disjoint-component scope](#connected-organization-coupling) also reaches
request-aware admission and retains its separate collateral-edge guarantee.
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

The nodal runtime's optional `MATH_ENGINE` path forwards authoritative raw
coordinates to its projector; a rejected node is not silently omitted from
the preparation. Tolerance and generator admission precede auxiliary evolution.
Retained state uses the engine's declared backend, with explicit export for
observation adapters. These handoff checks do not promise full-step rollback.

The public mathematical phase readers apply the same raw-value admission to
scalar, list and NumPy inputs. `normalize_phase` returns the represented
half-open interval `[0, 2*pi)`; a modulo result rounded to its upper endpoint
is canonicalized to zero. `compute_phase_difference` retains its signed
`atan2(sin(delta), cos(delta))` convention and rejects unrepresentable
subtractions. These numerical readers do not replace live U3 admission.
The circular mean uses this same represented phase boundary.
The shared `safe_divide` and `clamp_value` helpers reuse raw represented-real
admission for values, bounds and fallback. Boolean, text, nonfinite and
nonzero materialization-loss inputs reject on NumPy and scalar paths before
division or clipping. A zero denominator selects the admitted fallback;
overflow or nonzero underflow of a quotient rejects instead. NumPy
broadcasting and the scalar path's one-dimensional broadcasting remain
available. The `clamp`/`clamp01` adapters inherit this boundary, so clipping
cannot turn an invalid observation into an apparently valid bounded score.
Bounded pressure, capacity and grammar normalizations compare admitted
magnitudes before dividing, so a finite ratio above the clipping bound can
saturate without an overflowing intermediate. Nonzero interior ratios must
remain representable; zero-normalizer conventions remain with each caller.
Range and similarity helpers use exact represented differences when finite
endpoints span more than binary64 can hold. These are numerical diagnostic
policies, not stability certificates.
`angle_diff_array` checks selected results in the requested output dtype
before writing: narrowing that erases a nonzero difference rejects and leaves
the output unchanged, including its unselected entries.

`BasicStateProjector` admits signed scalar EPI, nonnegative capacity, finite
phase and a positive integer dimension before array construction or random
excitation. It requires a complex dtype and admits its null tolerance before
random draws. Boolean/text values and nonzero materialization loss reject,
including nonzero components erased when narrowing the output dtype.
The CLI math summary reads authoritative aliases under these same domains;
invalid primary state cannot fall through to a secondary alias, and legacy
phase storage is retained. This auxiliary map does not identify physical state.
The summary rebuilds its validator from current admitted operators and
thresholds; a cached verdict cannot substitute for those live premises.
The projector reuses the shared spectral-state normalization, scaling
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
