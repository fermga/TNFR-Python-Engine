# Finite-duration Emission episodes

<a id="emission-episode-status"></a>
## Constitutive status and scope

This chapter specifies an **opt-in constitutive proposal**, not a proof that
ordinary instantaneous Emission (AL) already has a duration. The
[operator-realization criterion](STRUCTURAL_OPERATORS.md#continuous-operator-realization-audit)
permits an explicitly stated revision with its own complete law and comparison
to existing contracts. Ordinary AL, the general runtime and the
[zero-flow native window](NATIVE_OPERATOR_DYNAMICS.md#native-positive-window)
retain their separate behavior.

The existing [AL/IL coordinate limit](NATIVE_OPERATOR_DYNAMICS.md#native-operator-growth-source-boundary)
does not lift its finite-jump histories into continuous evidence. Each finite AL
restarts the continuous history, and activation counts do not converge when the
number of events diverges. Preparing pressure with Dissonance (OZ) likewise does
not derive every subsequent dynamical row. These obstructions motivate an
explicit episode model rather than an implicit reinterpretation of either path.

The [episode executor](../src/tnfr/operators/emission_episode.py) and
[native consumer](../src/tnfr/operators/native_emission_window.py) implement this
proposal. The [API contract](../docs/API_CONTRACTS.md#emission-episode) owns their
admission and failure interfaces. The [research program](RESEARCH_PROGRAM.md#research-current-question)
alone owns the next question; this chapter installs no separate campaign.

<a id="emission-episode-semigroup"></a>
## Primary action and its finite-duration lift

On an admitted unclipped scalar branch, the primary AL map is the translation
`A_b(x)=x+b`, with positive gain b. Whenever all intermediate states remain in
that branch, `A_b(A_c(x))=A_(b+c)(x)`. Choose held capacity nu>0, pressure p>0 and
rate r=nu*p independently of the response. For a supplied duration T>0 define

```text
S_t(x) = x+r*t,        0 <= t <= T,
x_dot = nu*p,         p_dot = nu_dot = theta_dot = 0.
```

Support is fixed and Gamma is absent. Then `S_0` is the identity,
`S_(t+s)=S_t o S_s` on the admitted local domain, and the coordinate endpoint is
the AL map with gain `b=r*T`. Differentiating the explicit expression gives the
stated form row. Conversely, that constant-rate initial-value problem has this
unique form path. This derives the coordinate action for the **chosen** rate;
neither the rate law, pressure persistence nor episode duration follows from the
AL name or translation composition alone.

The units are `[r]=[EPI]/[time]` and `[p]=[EPI]/([nu]*[time])`; the finite AL gain
has EPI units. A supplied schedule sets structural time, not laboratory time.
Initial and prospective final states must remain inside the strict AL latent
and basal-capacity domain and an unprojected scalar branch. Positive monotonic form
then stays in that chart throughout the ideal episode. Negative-rate translation,
clipping, evolving capacity/phase/support, forcing and refreshed pressure are
outside this proposal. Every nonzero form-flow row must belong to an episode
target; a nonparticipant has zero form row, even if its stored pressure is nonzero
at zero capacity.

**The additional lifecycle premise.** One admitted episode counts as one AL
occurrence. Its own grammar is checked before motion; shared activation lineage,
source-glyph provenance and one grammar mark commit at completion. There is no
second AL form jump at that boundary. This convention defines a finite-duration
action and its memory; it is not the limit of infinitely many recorded AL events.
The episode is a synchronous atomic call, without mid-episode operator decisions.
Interrupted/resumable actions would need another execution and memory contract.

Existing public-operator waiting semantics are retained. Episode completion
updates AL grammar/lifecycle, while native `since_AL`/`since_EN` counters remain
owned by native batches. An episode does not silently reset that separate memory.
Thus native decision occasions and waiting inputs remain declared policy.

<a id="emission-episode-numerics"></a>
## Represented execution and observation

The actual canonical integrator supplies the form evolution and its physical
boundary samples. The effective gain is computed **before execution** from
admitted capacity, pressure and duration. The shared AL proposal supplies the
independent endpoint comparator. The bounded implementation rejects an unowned
row, clipping or an endpoint that does not match that proposal, rather than
fitting the gain from the observed change. This is a represented execution
contract; a binary64 pass is not exact real arithmetic on arbitrary inputs.

The episode preserves phase, capacity, pressure and support through the shared
flow contract. Its real `epi_time_history` is retained because the episode has
no instantaneous form jump. Those observed samples can support ordinary native
Mutation (ZHIR); a predicted derivative is never inserted as measured history.
The timestamp of AL lifecycle provenance remains separate from structural time.

Numerical substeps inside one episode do not add grammar or activation counts.
Two separately invoked episodes add two occurrences even if their total duration
and final form agree. Solver derivative/acceleration diagnostics retain their
actual updates: subdivision can change those diagnostics and the most recent
sample interval when separate episodes are used. A selector consuming them can
therefore distinguish executions. No complete-state or selector invariance under
arbitrary temporal refinement is claimed.

For the chosen constant row, the ideal interior rate is r and its interior
acceleration is zero. An initial derivative cache is not evidence of the true
pre-entry rate. Its difference from r therefore does not establish an entry
impulse. The typed observation below distinguishes certified interior quantities,
solver diagnostics and unproved boundary transitions.

<a id="emission-selector-observation"></a>
## Operator-grounded observation for one native decision

For an admitted episode on `(s,t)`, distinguish three channels:

| Channel | Meaning and evidence | Limitation |
| --- | --- | --- |
| Held generator rate r and interior acceleration 0 | Derived from the declared AL episode law and retained complete row | Conditional on held pressure/capacity/support and absent forcing; no boundary-transition theorem |
| Endpoint secant `(x(t)-x(s))/(t-s)` | Exact accounting of represented endpoints from actual execution | Two endpoints do not determine curvature; rounding can separate this secant from the represented generator rate |
| Solver rate and acceleration caches | Actual diagnostics retained at episode completion | Their initialization and numerical subdivision are not observations of the pre-entry derivative |

Two endpoint samples admit infinitely many interpolants. For example, adding
`c*(u-s)*(u-t)` with small c on an interior chart leaves both endpoints fixed
while changing the interior second
derivative by `2*c`. Thus neither the endpoint secant nor its agreement with r
measures zero acceleration. Zero interior acceleration follows instead from
the explicitly selected constant-row law. Its units are `[EPI]/[time]^2`.
Earlier live samples may support a separate temporal-curvature estimate; the
episode-only two-endpoint observation does not erase or reinterpret that history.

The opt-in `emission_interior` selector policy consumes the law-derived channel
for one actual native batch immediately following its own episode. It applies
to the produced complete population, including admitted zero-flow rows. The
ordinary `solver_diagnostic` policy continues to consume stored diagnostics.
The explicit choice is an observation policy, not a uniquely derived law of
operator occurrence or a claim that the episode boundary has zero activity.

One selected channel supplies normalization, the parametric base choice,
hysteresis, score and the soft repetition/force filter. Changing only the base
choice would leave a second acceleration consumer reading incompatible data.
No temporary EPI derivative or acceleration write is needed. The private
producer binds the observation to the actual graph, source, clock and prepared
batch; a detached report cannot activate selection. The native grammar and
each operator's own state admission remain independent requirements.

This distinction has an actual operator consequence in the declared P2 control.
After the same AL episode, one Euler step leaves core cache 2 and two leave 0.
The raw policy requests THOL in the first case and ZHIR in the second. Actual
graph-backed THOL reads active temporal history, not that cache. Its existing
two-sample compatibility result is zero; pressure stays fixed, but its grammar
occurrence pushes OZ outside the core's recent context. The supplementary ZHIR
is excluded by its grammar gate, leaving core phase zero. The second execution performs native
ZHIR, changes phase by 1/4 and consumes its physical interval. Direct native
THOL creates no child in this control.

Supplementary staged ZHIR records its own history-derived acceleration in the
live cache; with these two samples its existing compatibility value is zero.
Direct native ZHIR does not perform that stage diagnostic write. The passive
episode report retains the original completion diagnostics independently of
those later actual actions. This is not measured zero curvature or a selector
overwrite. Pre-response and post-response live caches must be distinguished.

With `emission_interior`, both subdivisions instead request the same admitted
ZHIR/IL pair and retain the same supplementary peer response. The prospective
control compares form, phase, pressure, grammar and consumption while retaining
the differing solver caches. This is scoped decision/action agreement for the same
episode, preparation and policies, not complete-state refinement invariance.
Other durations, policies, boundary histories and operator rows retain their
own obligations. In particular, overlooking a real rate change between two
episodes is not justified by their separate zero interior accelerations.

The [actual THOL continuation](THOL_PRESSURE_FEEDBACK.md#thol-emission-history-continuation)
composes pressure-changing IL/OZ, successive AL episodes, history-derived THOL
and a further AL. It distinguishes signed feedback, successor-domain rejection
and extra episode boundaries from numerical substeps. Its temporal estimate
does not extend the `emission_interior` selection policy to boundary activity.

<a id="emission-native-source-competition"></a>
## Actual native source and competing responses

The new consumer declares this order: complete the AL episode, execute one real
native batch, then consider a supplementary hybrid-growth ZHIR stage. It retains
the episode result and native capture separately, with actual ordered targets,
storage identity, clock and complete endpoint coverage. There is no fabricated
atomic schedule containing the native glyphs. Outside-dispatch form variation
must vanish; cancelling unexplained changes cannot supply operator growth.

For the right-included window (s,t], let F be the observed episode flow increment
and J the sum of terminal native form jumps. Exact represented accounting gives

```text
D = x(t+)-x(s) = F+J,          G = D/(t-s).
```

Captured jump variation is distinct from signed growth. The episode contributes
F once; its effective AL gain is not added again to J. The growth source and
time-basis labels distinguish this proposal from both supplied schedules and
the zero-flow native window. Only the private producer's live graph binding can
authorize a fresh response snapshot; returned reports grant no permission.

An actually completed native ZHIR wins over a supplementary response on its own
target. Other candidates still need their own exact G>xi, active capacity/floor,
live grammar and unused hybrid window. Pressure-normalized selection and supplied
waiting memory are retained, so this is a declared competition policy rather
than an autonomous occurrence theorem.

The consumer requires the existing timestamped native consumption policy.
When a successful supplementary ZHIR leaves valid physical evidence for the
same window and live EPI, it also consumes that identified physical interval
through the shared occurrence owner. An already consumed matching interval
blocks both selection and strict admission. This prevents a later ordinary
native call from reusing those same retained samples. If a terminal native form
jump has invalidated physical history, no physical receipt is invented. The
ordinary level policy and unrelated standalone APIs are unchanged; external
clock/history/ledger rewrites remain outside the intact-memory guarantee.

The episode is atomic. After it completes, a native failure retains the episode
and successful/partial native prefix. The supplementary response has a separate
transaction: its phase, grammar and consumption updates either commit together
or restore the completed post-source graph. No source is replayed or erased to
repair a later response failure.

<a id="emission-episode-discriminator"></a>
## Prospective finite discriminator and remaining questions

The [participation controls](../tests/operators/test_emission_episode.py) use a
unit-weight P2, forms `(1/4,1/4)`, capacities `(8,1)`, phase zero and initial
pressure `(1/8,1/2)`. Actual IL on both (retention 1/2), OZ on both (factor 2),
then IL only on the core give pressure `(1/16,1/2)`. The core has IL/OZ/IL and
zero debt; the peer has IL/OZ and one debt. This asymmetric history is necessary:
an extra peer IL would later push OZ outside the recent-context window.

A duration 1/4 episode has rate 1/2 and effective gain 1/8 on each node, ending at
form 3/8 with one new AL mark. With declared stored Si `(0,1)` and no forced lag,
the core's normalized pressure 1/8 selects native ZHIR and the peer selects IL.
At xi 1/16 the newly observed growth admits the core's actual native action.
The peer's IL restores its debt while retaining a recent OZ, allowing its
supplementary ZHIR. A configured phase increment 1/4 gives each one response;
the core must not receive a second one merely because hybrid growth also crosses.

For a same-interval reuse control, actual IL with zero pressure retention and
zero phase locking on both preserves EPI samples and structural time. Declared
Si zero then makes both ordinary native requests ZHIR. Under consume-and-skip
policy, both must abstain because the shared physical intervals were consumed.
No artificial history or interval is installed for this comparison.

These finite contracts distinguish operator participation, source identity,
response competition and consumption. They do not establish spontaneous episode
timing, a unique pressure law, signed/nonlinear continuous realizations of other
operators, acquired organizations or empirical physics. Generalizing the episode
requires its own complete state, observation and admission argument. The
[independent boundary controls](../tests/operators/test_emission_episode_admission.py)
retain rejected domains and the distinct failure boundaries.
