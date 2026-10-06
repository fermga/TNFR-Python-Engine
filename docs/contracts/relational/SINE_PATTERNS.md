# Sine pattern geometry, formation and recovery

Relative patterns and origins, composition, equilibrium and symmetry, prepared formation, recovery/capture, controlled slow references and budget exclusions.

Part of [Relational dynamics contract index](../RELATIONAL_DYNAMICS.md). Section links remain stable; hypotheses and model changes remain local to each result.

### Relative sine patterns and moving references

`bound_relational_sine_pattern(graph, *, reference_node, reference_model,
form_error_bounds, phase_error_bounds)` in
[`relational_sine_pattern.py`](../../../src/tnfr/physics/relational_sine_pattern.py)
admits a complete supplied network under the same separate smooth sine law.
Shared sine staging validates signed represented form, phase, exact held
nonnegative capacities, simple connected unit support and absent Gamma.
It does not execute the native Arg law.

The two error inputs are required ordered sequences of nonnegative radii,
one per captured node. Exact rationals are retained; Booleans and nonfinite
values reject. The observation premise is
`actual_i = nominal_i + common_offset + residual_i` at one actual time,
with each residual bounded by its supplied radius. Phase centers and errors
must use supplied consistent real lifts. No phase unwrapping, sensor
calibration, missing-node inference or clock authentication is performed.
Common offsets can be arbitrarily unknown, but cannot be assumed common
across asynchronous samples without another observation law.

All nodes, including intermediaries, remain. Relative state boxes enclose
`x_i-x_ref` and `theta_i-theta_ref`; the reference rows are identically zero.
For every other row the radius is the sum of its own radius and the
reference's. Direct edge bounds use only their two endpoint radii.
The shared sine rate/work kernel supplies full rates, reference rates,
relative rates, storage and loss. Relative rates subtract the reference
rate; this is not an intervention that freezes the reference node.

The returned `SineRelativePattern` retains both uncertainty descriptions.
Its static edge, rate and storage bounds apply to the original residual
class. The Cartesian relative box forgets the correlation from the shared
reference error and can contain additional states. The tighter static
bounds must not be used to certify every point of that larger box.
The [mathematical owner](../../../theory/nodal/SINE_PATTERN_RECOVERY.md#sine-relative-pattern-state)
defines pattern identity, winding limits and a sufficient analytic recovery
criterion. This report does not itself certify a recovery basin.

`pattern.forecast(observation_time=..., end_time=..., time_step=..., order=6)`
uses [chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission) and reconstructs
the initial relative box from nominal coordinates, the reference node and
original residual radii. The returned pattern retains those rebuilt boxes.
The method selects an initial zero-reference
representative and passes the full outer box to `bound_sine_flow`.
Every node then evolves, including the reference;
only the returned endpoint is projected. The solver's final held-capacity
coordinate is a storage layout and does not designate a physically hidden
node in this adapter. The same size, time and numerical-budget restrictions
apply. Its `2*n+1<=24` layout limits this forecast adapter to eleven nodes;
static reports are not restricted to the solver's state dimension.

`SineRelativeForecast` retains the source, entire `full_forecast`,
relative endpoint intervals and reference displacement from its initial
representative. Its status forwards the solver's status. Endpoint bounds
refer to `full_forecast.validated_end_time`, even if the requested horizon
is unavailable. Initial absolute offsets remain unknown. The method neither
fits an observation history nor proves that relaxed sample bounds came
from a common trajectory.

Both reports use shared exact projection and label admission through
`to_dict()`, with schemas `tnfr.relational-sine-relative-pattern.v1` and
`tnfr.relational-sine-relative-forecast.v1`, suitable for the SDK atomic
JSON writer. The existing hidden inverse consumes absolute nodal rates;
feeding it moving-reference differences is unsupported and can produce an
incorrect hidden form even with exact, full-rank observations.

### Joining relative sine patterns with declared origins

<a id="sine-relative-frame-composition"></a>

`assess_sine_pattern_composition(left, right, *, bridge, observation_time,
edge_turn_offsets, form_origin_difference=None, phase_origin_difference=None,
work_allowance=None)` in
[`relational_sine_composition.py`](../../../src/tnfr/physics/relational_sine_composition.py)
accepts two `SineRelativePattern` sources. The equivalent
`left.compose_with(right, ...)` delegates to that owner. This is a detached
preparation and capture assessment; it does not add a live edge or choose an
event time.

Both sources retain disjoint ordered node labels, complete connected simple
unit supports, positive held capacities and identical admitted effective
loss/exchange/storage coefficients under the regular sine reference model.
Positive form loss is required. Shared primitive admission runs before any
derived field is consumed. Cached source storage, rates and verdicts are
rebuilt. The joint support retains the existing geometry budget of 32 nodes
and 50 edges. `bridge` is one ordered pair, left node then right node.

The two origin differences are exact or represented finite real scalars,
defined as **right common additive origin minus left common additive origin**
in the source's `nominal + common + residual` representation. They are not
measured reference-node gaps. The phase difference belongs to the declared
consistent real lifts. Interval origin inputs are unsupported here; no
midpoint or zero replaces an uncertain or absent origin. Original per-node
residual radii remain in the whole joined family. Its single absolute common
origin remains unknown.

`observation_time` declares the nonnegative synchronous structural time of
both sources and the hypothetical event; the reports do not authenticate
that declaration. Model equality does not prove clock calibration or sample
synchrony. Boolean, nonfinite or malformed authoritative scalars reject.
Missing origin differences return an explicitly unavailable preparation,
with no joined state or capture certificate; unrelated malformed declarations
still reject instead of being hidden by that unavailability.

When both differences are supplied, the shared exact-row pattern builder
shifts right nominal coordinates and retains every original residual radius.
It recomputes joined degrees, mobility, rates, storage and weights without a
graph roundtrip that would round admitted rationals. The existing
`certify_sine_sector_capture` checks the actual joined family and all full
support faces in the independently supplied integer edge sector. Internal
certificates and the bridge cost alone cannot replace this full-state check.

`SinePatternComposition.status` describes preparation availability; the
separate `capture.admitted` describes sufficient maintenance after the
hypothetical event. A failed sufficient bound is not a destruction verdict.
Representative weighted-mean bounds are relative to the left source's
unknown common origin; the capture's absolute weighted means remain `None`.
Support reweighting is distinct from continuous weighted-form transport.

The report separates internal storage, the bridge storage jump and complete
joined storage. Optional nonnegative `work_allowance` is a declared event
budget: `within_allowance`, `exceeds_allowance` and `unresolved` distinguish
its interval comparison, while `not_supplied` leaves it unevaluated. Neither
capture nor continuous loss supplies an allowance or an occurrence law.
The [mathematical owner](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-relative-frame-composition)
owns the exact family map, event-weight identity and missing-frame obstruction.

The owner and generic SDK `relational_report_to_dict` use shared exact JSON
projection and node-label admission. Export preserves independent availability,
source declarations and nested capture evidence; it does not authenticate them.

### Composing analytically acquired patterns

<a id="sine-prepared-composition"></a>

`assess_sine_prepared_composition(left, right, *, bridge, left_initial_time,
right_initial_time, edge_turn_offsets, form_origin_difference=None,
phase_origin_difference=None, work_allowance=None)` in the
[composition owner](../../../src/tnfr/physics/relational_sine_composition.py)
accepts two `SinePreparedEntry` reports whose original sources are
`SineRelativePattern`, including zero-radius families. The equivalent
`left.compose_with(right, ...)` delegates to this owner. Standalone absolute
comparison sources are outside this handoff contract; no implicit conversion
changes their frame interpretation.

The source primitives, scaled durations and original integer sector
declarations are re-admitted. Both entries are then rebuilt by their owner;
cached endpoint intervals, horizons, means, status and storage are not evidence.
The sources need disjoint labels, positive held capacities and identical
admitted effective coefficients. The shared joint 32-node/50-edge budget is
checked before recomputing the analytic bounds.

Each finite nonnegative initial time is explicit. The endpoint time is
`initial_time + scaled_time / effective_epi_weight`; the two exact results
must agree. No caller-supplied observation-time override or silent waiting is
accepted. Different durations are allowed only when the declared starting
times give the same endpoint. The report records these preparation declarations,
not authenticated timestamps or measurements.

Exact relative common origins keep the meaning of the
[static composition contract](#sine-relative-frame-composition). Original
residual mean intervals and memberwise conserved means propagate those origins
to the analytic endpoints. Tighter internal edge bounds are retained from each
rebuilt entry. Bridge bounds use the centered port enclosures and the conserved
mean intervals. The prefix uses isolated degrees; post-event dynamics uses
joined degrees. The post-event representative mean uses conserved component
inventories plus the two port reweightings.

`SinePreparedCompositionSource` records the actual analytic endpoint provenance.
It is not a synthetic observation or an input accepted by the generic
relative-state reader. The shared sector core receives the corresponding
whole-state enclosure, preserving its internal correlations. Replacing it by
independent endpoint-box corners would require recomputing the bounds for
that larger set.

`SinePreparedComposition.status` describes frame availability. Missing origins
leave `capture` and joined edge/storage bounds unavailable. The separate
`acquisition_and_capture_certified` property requires both rebuilt entries to
certify acquisition and the new joint capture to be admitted. Joint capture
alone does not establish earlier formation. `budget_status` independently
compares the bridge-work enclosure with an optional supplied allowance.
An unavailable sufficient bound does not assert destruction or impossibility.

Direct and generic SDK export retain exact fractions, independent availability
and nested source/capture evidence, with shared label admission. Neither export
nor rebuilding authenticates the preparation. No live edge, reset, solver
trajectory or occurrence law is installed. The
[mathematical owner](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-prepared-composition)
provides the containment, clock and charge identities and the frozen control.

### Whole-set sine cycle recovery

`certify_sine_cycle_recovery(source, *, cycle, winding, radius,
phase_turns=None)` in
[`relational_sine_recovery.py`](../../../src/tnfr/physics/relational_sine_recovery.py)
accepts a `SineRelativePattern` or `SineRelativeForecast`. Both expose the
same operation through `certify_cycle_recovery(...)`. It observes the supplied
state set without running a solver or changing a graph.

Cycle recovery, general-target recovery, conservative identity and sector
capture share [chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission).
Specialized cycle certificates retain their larger support domain;
general-target and sector consumers keep their declared geometry budgets.

The ordered cycle must cover every node and every edge of the complete
simple unit support, with at least three nodes. Extra chords, attached
environmental nodes and a selected cycle inside a larger graph reject.
The exact target has phase turns `winding*j/n` in this cycle order and uniform
form. Integer `winding` must satisfy `4*abs(winding)<n`; consensus is zero.
This analytically specifies a strictly acute critical phase, rather than
recognizing an equilibrium from a small floating residual.

`radius` is a finite strictly positive proof-domain parameter in the declared
form/phase coordinates. It is neither a physical threshold nor a tolerance
for replacing equality. Optional `phase_turns` are integers in captured node
order, added to the observed phase lifts. Their default is zero. Booleans,
noninteger turns and malformed ordered inputs reject. Lifts are not guessed
from the observations to obtain a successful result.

The certificate uses the
[whole-set proof](../../../theory/nodal/SINE_PATTERN_RECOVERY.md#sine-cycle-recovery):
pairwise differences bound the centered form/phase-deviation norm; exact
criticality cancels the linear phase-storage term before its quadratic
remainder is enclosed. The cycle spectral gap, radius cosine margin and
storage barrier use shared rational pi, square-root and cosine bounds.
The declared phase target and its storage retain mathematical pi.

The two input scopes remain distinct:

- An original pattern uses its nominal differences and the two endpoint
  residual radii. Common origins cancel before interval evaluation.
- A forecast uses differences of the **full** endpoint coordinates at
  `validated_end_time`. It consumes neither the source's tighter static
  bounds nor the twice-projected relative-coordinate box.

Positive dissipation and strictly positive held capacities are separate
theorem premises. `exact_held_capacity` retains exact scalar constraints
where present, including positive values smaller than the interval grid;
`capacity_bounds` also records outward arithmetic enclosures. A forecast's
augmented final capacity has no such exact constraint and its entire interval
must have a positive lower endpoint. Admission supplies no uniform decay
rate as capacities approach zero.

`SineCycleRecovery` retains the source, exact target turns, ordered support,
input uncertainty scope, norm/storage bounds and both strict margins.
`hypothesis_failures` identifies missing capacity/dissipation premises;
`unresolved_conditions` identifies unproved geometric or numerical bounds.
Only an empty pair gives `status="admitted"`. Otherwise status is
`"unavailable"`, which is not an instability verdict or proof that the state
cannot recover. Invalid target, support and input contracts raise.

For original observations, `observation_time` is unavailable (`None`), not
invented as zero. For forecasts it is the actual validated endpoint time.
`input_forecast_admitted` and `input_forecast_requested_end_time` retain the
original solver result. A partial endpoint can pass the recovery theorem
without changing that solver's unavailable requested-horizon verdict.

Admission guarantees recovery to the declared twist modulo common origins
for every state in the stated set, under the supplied unforced sine law,
fixed support and held positive capacities. It does not establish formation
from a different winding, survive arbitrary future inputs or support events,
select the law, or identify matter. `to_dict()` uses shared exact projection
and label admission with schema `tnfr.relational-sine-cycle-recovery.v1`,
suitable for the SDK atomic JSON writer. A projected public dataclass does
not authenticate its source or proof history.

### Whole-set sine recovery with a retained environment

`certify_sine_pattern_recovery(source, *, target_phase_turns, radius,
phase_turns=None)` extends the same recovery owner to the full connected
simple unit support. `SineRelativePattern` and `SineRelativeForecast` expose
it through `certify_pattern_recovery(...)`. The cycle method above retains
its more specific target and spectral formula.

`target_phase_turns` supplies one exact integer or `Fraction` per node in
captured order, in units of mathematical `2*pi`. Booleans and floating-point
target turns reject. The target has uniform form, with a free common form
and phase origin. No target is fitted to the observed response.

The owner reuses exact phase-cycle reconstruction on **every supplied edge**.
Principal target edge turns must be strictly acute. Opposite signed sine
currents must cancel symbolically at every node. This is a sufficient exact
criticality check, not a complete decision procedure for trigonometric
identities. A target failing that check rejects; a near-zero floating residual
or an interval containing zero cannot certify its equilibrium. The geometry
owner's 32-node/50-edge evaluation budget remains explicit. It is a software
budget, not a restriction derived for the underlying law.

The spectral gap is bounded on the full exact combinatorial Laplacian by the
existing rational quotient owner. Common-origin removal, source uncertainty,
capacity admission, pairwise centered norm, quadratic storage remainder and
strict barrier checks share the cycle certificate's implementation. Target
edge angles may differ. Adding an environmental edge changes criticality and
the spectral problem; two isolated cycle certificates cannot be composed.

The [complete-network proof](../../../theory/nodal/SINE_PATTERN_RECOVERY.md#sine-interacting-recovery)
admits two C5 patterns and one intermediary with nonzero observation errors.
A positive-capacity intermediary transmits a donor perturbation while the
whole network remains in a common recovery domain. A zero-capacity
intermediary is a separate no-influence control and does not pass the
positive-capacity recovery theorem.

The returned `SinePatternRecovery` retains exact target and geometric
evidence, full-support spectral provenance and the source/time distinctions
specified above. Its schema is `tnfr.relational-sine-pattern-recovery.v1`.
An unavailable radius or storage inequality is not an instability verdict.
No trajectory, node deletion, support event or native-law dispatch occurs.
In particular, static admission of eleven nodes does not extend the existing
validated forecast solver's coordinate budget or certify an unevaluated
forecast horizon. Public dataclass export does not authenticate evidence.

### Sine formation eligibility and timed exclusion

`assess_sine_mediated_formation(*, model, amplitude, capacity_contrast,
exclusion_time=None, profile="localized", donor_epi=None)` in
[`relational_sine_formation.py`](../../../src/tnfr/physics/relational_sine_formation.py)
assesses declared eleven-node preparations, without executing a trajectory.
Its donor is an exact winding-one C5, its receiver has uniform phase and
form, and the intermediary has form `amplitude` and phase zero. The default
`localized` profile sets every other form to zero. The `balanced` profile
instead sets every donor form to `2*amplitude`, retaining receiver form zero.
Both profiles cost `amplitude**2` in form storage; balanced initial loss is
one quarter of localized loss. This restricted comparison is not an optimum
over arbitrary preparations. `profile="explicit"` instead requires exactly
five ordered signed `donor_epi` coordinates for donor nodes `0..4`, with
`amplitude` still supplying intermediary form. Receiver forms remain zero.
The complete edge differences determine storage; it need not equal
`amplitude**2`. A donor vector with either named profile is rejected rather
than silently overriding that profile. Capacities are one except receiver nodes `6` and `9`, which
have capacities `1+capacity_contrast` and `1-capacity_contrast`.
Both contact edges and every state coordinate remain supplied premises.

An explicit regular reference model with positive dissipation is required.
Amplitude, donor coordinates and contrast use shared represented-real admission;
`abs(capacity_contrast)<1` preserves positive capacities. Optional
`exclusion_time` must be strictly positive and denotes a proof horizon,
in the declared sine-model clock. It is not a new dynamics parameter,
detection threshold, numerical time step or laboratory duration.
Invalid domains raise. Phases are retained as exact rational turns; a
materialized radian approximation is not substituted for the preparation.

The [proof](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-formation-eligibility)
distinguishes three obligations:

- Exact symmetry and equilibrium controls: zero contrast preserves receiver
  reflection; an identically zero full form gradient gives equilibrium.
  Zero intermediary form alone does not establish equilibrium for an explicit
  donor. `silent_donor_subspace` identifies intermediary form and donor port
  zero with `D1=-D4,D2=-D3`: donor sign/reflection symmetry holds the receiver
  fixed even with nonzero capacity contrast. Its exclusion reason remains
  separate from equilibrium. A nonzero reflection-odd initial acceleration
  is proved from an exact coefficient, without a numerical cutoff; its
  absence does not exclude a later causal response.
- Storage eligibility: entering the joint acute target requires strictly
  more storage than its entry-face barrier, which is higher than the final
  target's minimum storage. The donor need not retain its winding earlier
  in the path for this argument to apply.
- Timed entry exclusion: either global phase-speed bounds or the necessary
  phase-action time exclude entry through a supplied horizon. A lower bound
  on accumulated full-network dissipation must also exhaust the entry budget.
  Initial loss alone is not a finite-time proof.

`phase_action_bound` retains the actual maximum of `nu_i/d_i+nu_j/d_j`
over receiver edges, the allowable-loss interval `F-beta*B5`, and a rational
`necessary_entry_time_lower_bound`. When that allowance is resolved strictly
positive, the latter bounds `e*beta**2*pi**4/(w**2*k_max*(F-beta*B5))`
from below using the shared outward constants. A nonpositive allowance is
already a storage obstruction; an unresolved sign does not permit division.
Both leave this finite time bound unavailable with explicit reasons.
`entry_through_horizon_excluded` requires the supplied horizon to be strictly
below the certified lower bound. The bound is conditional on target entry,
not a predicted event or a global speed estimate after exhausting the budget.

Nodal affine bounds use each profile's actual initial form gradients and
clip at zero before integrating. `early_loss_lower_bound` retains the
intermediary-only bound for compatibility; `full_early_loss_lower_bound`
sums disjoint nodal contributions. The balanced intermediary contribution
is zero, although the full loss is positive for nonzero amplitude.
The [directional proof](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-balanced-formation)
additionally retains exact weighted-Laplacian moments of the initial
gradient. `directional_loss_bound` exposes those moments, its polynomial
norm bound and explicit availability. A missing horizon, zero gradient,
excessive hyperbolic argument or nonpositive polynomial endpoint makes
that sufficient certificate unavailable. Its squared bound is never used
outside the admitted positive window. The certificate retains the short-window
`cosh<=11/10` majorant and admits wider windows through
`cosh<=1/(1-v/2)` for squared argument bound `v<2`. The reported
`cosh_upper_bound` and `cosh_bound_method` retain which proof was consumed;
these are bounds, not a replacement evolution law. `combined_early_loss_lower_bound`
takes the maximum of the nodal and directional bounds, never their sum:
both bound the same total dissipation.

The separate `maintained_target_obstruction` applies the
[global auxiliary-function theorem](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-maintained-target-obstruction).
It requires the exact effective ratio `0<w/e<3/2`, storage scale one and
positive capacity contrast `1/2` on the existing eleven-node preparation. Its
`form_storage_coefficient=4/5` and mobility spectrum enclosure `[1/44,3]`
belong to the proof, not the evolution law. The mixed coefficient is
`w/(2*pi*e)`, which recovers `1/(2*pi)` at equal weights. It retains initial/target
functional bounds and the correlated gap `V5-(4/5)*F`, rather than
subtracting independent enclosures of the shared `V5` term.

`exchange_to_loss_ratio` uses exact rationals of the model's stored normalized
coefficients; `sufficient_ratio_upper_bound` records `3/2`. These metadata remain
available when the ratio fails admission, without fabricated functional bounds.
The open interval is a conservative sufficient proof domain, not a physical
transition threshold or a classification of every ratio. At or above its
endpoint this certificate is unavailable, not evidence of formation or
instability. The proof identifier is `eleven_node_sine_ratio_mixed_lyapunov`.
The ratio fields append optional defaults so older positional construction
remains valid; a missing field does not establish the extended proof's premises.

For the analytical clock change `tau=k*t`, `k>0`, dividing both complete
evolution rows by `k` preserves the ratio, mixed functional and target verdict;
necessary entry times multiply by `k`. This is not an extra clock setting in
the report. `RelationalExchangeModel` normalizes supplied channel weights:
ideal common rescaling of those raw weights is removed by normalization, not
converted into a time change. Rounded inputs can alter the represented ratio.
Admission always follows the represented effective pair, including near the
ratio boundary; raw decimal labels are not exact normalized coefficients.

A positive gap sets `maintained_target_excluded` and adds the distinct
`maintained_target_lyapunov_obstruction` reason to the outer report. This
excludes convergence to the maintained target and entry into a valid basin
that guarantees it. It does not set `early_loss_exclusion_certified` or
exclude every transient visit to the joint acute region. The nested status
is `excluded`, `not_excluded`, `unresolved` or `unavailable`; the latter
retains law/capacity mismatch reasons and no invented functional bounds.
Neither an insufficient gap nor an unavailable certificate proves formation.
No artificial `F<=4` admission cutoff is used: the actual source gap decides
the certificate, while the theorem proves it positive for the whole class
with `F<=4` under the stated premises.

Both new nested observations have default `None` for older or manually
constructed reports; the assessment always supplies them. `None` means
absent evidence. The outer `tnfr.relational-sine-mediated-formation.v1`
schema, existing constructor positions and shared exact JSON projection
are retained. No frozen response archive is regenerated by these readers.

The assessment separates `excluded`, `passes_necessary_conditions` and
`unresolved_bound`. Passing these checks does not establish formation,
capture or existence of a successful trajectory. An absent or insufficient
timed bound cannot certify a future crossing. The target is its geometric
orbit modulo common origins; it does not fix an incompatible final mean.
The held-capacity weights `degree/capacity` define conserved form and
continuously lifted phase sums, retained separately from circular means.
The report retains the complete preparation, law coefficients, exact and
outward bounded quantities, proof time and reasons. Shared JSON projection
supports SDK export without authenticating provenance.

With the plan's default coefficients, analytic proofs exclude both profiles
throughout `abs(amplitude)<=2`, `abs(capacity_contrast)<1`. At `amplitude=2`,
`capacity_contrast=1/2`, computational exclusion passes with proof horizon
`1/5` for localized and `3/5` for balanced. These are restrictions on the
supplied preparation families, not a general impossibility of pattern formation.
The theory's constant-donor corollary also covers arbitrary constant donor
form `D` and intermediary form `H` at `((D-H)**2+H**2)/2<=4` under those
coefficients. Explicit donor vectors extend preparation admission without
changing the target, phase state, support or receiver initialization. The
[internal-form analysis](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-internal-form-geometry)
derives the full six-coordinate quadratic forms and causal response maps.
Its nonuniform example has donor forms `(10,13,14,14,13)/3`, intermediary
`4/3` and contrast `1/2`. It passes initial and short-window necessary checks,
but the wider certificate excludes its joint entry by `6/5`. Independently,
the global functional excludes maintained-target convergence for every
six-coordinate preparation with `F<=4` throughout the sufficient ratio domain,
at unit storage scale and positive half contrast. The earlier entry exclusions retain
their own scopes; transient acute-region visits are not generally excluded.

### Source geometry and the necessary nonlinear receiver correction

`SineMediatedFormation.receiver_excitation()` returns `SineReceiverExcitation`
from the existing formation owner. Under
[chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission), it rebuilds the
preparation before decomposing full form into donor-reflection-even and odd
parts. Both eleven-coordinate vectors are retained; their storage and
dissipative norms use the actual support, degrees and capacities.
`source_coordinates` has order
`(H, D0-H, s1, s2, a1, a2)`, as defined in the
[source theorem](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-source-receiver-excitation).
The decomposition remains available for valid preparations outside the more
restricted tangent theorem.

For effective half weights, unit storage scale and capacity contrast `+1/2`,
the report bounds the receiver's **tangent** quadratic phase cost for every
future time by `3*even_form_storage/4`. The inequality is strict for positive
even storage; zero even storage gives exactly zero tangent receiver output.
This is the full damped tangent about the prepared phase equilibrium, with
its noncommuting matrices retained. It is not an upper bound on the nonlinear
receiver potential or a forecast of a particular time.

If the actual receiver reaches potential `7/2`, its phase-edge deviation from
the matching tangent must have norm at least
`sqrt(7)-sqrt(3*even_form_storage/2)`. The interval field encloses this analytic
threshold, not an observed correction. A strictly positive outward lower
endpoint gives `nonlinear_correction_status="positive_bound"`; an unresolved
interval or nonpositive analytic margin leaves the positive bound unavailable.
Unsupported laws give `status="unavailable"` with reasons and `None` tangent
quantities. Corrupted consumed source data reject rather than certify.

For the same law and **total** initial form storage `F<=4`, the separate
`full_phase_budget_status="available"` certifies that the actual full-network
phase potential stays strictly below
`actual_full_phase_storage_upper_bound=139/20` for all future time. Any receiver
barrier crossing must follow a donor potential-barrier crossing, and requires
a donor potential decrease greater than `V5-69/20`. The report exposes the
conditional order flag and a fresh outward lower bound on that decrease.
This decrease is not identified receiver work or a reusable energy reserve.
Above the total form budget the status is `outside_original_form_budget`;
outside the admitted law it is `unavailable_law`. In both cases the nonlinear
fields remain `None`, independently of any available tangent evidence.

The odd component is silent in the tangent and when prepared alone under the
nonlinear law. It cannot be removed from nonlinear mixtures: equal even
coordinates, energy and quadratic spectral moments do not fix receiver work.
No value in this report establishes passage or installs an excitation law.
`to_dict()` uses the shared exact SDK projection with schema
`tnfr.relational-sine-receiver-excitation.v1`.

### Nonlinear receiver localization from regional storage

`SineMediatedFormation.receiver_localization()` returns `SineReceiverLocalization`.
It reuses source admission and the original half-weight sine law, unit storage
scale and positive-half capacity contrast. The auxiliary function
`U=F+V_D+7*V_R/5+7*V_bridges/6-q.T*K*S/3` is globally decreasing along that
complete nonlinear law. Its coefficients are proof weights, not configured
evolution parameters. The [regional proof](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-weighted-receiver-exclusion)
uses exact positive-definite matrices and full-state asymptotic convergence.

For total initial form storage `F<=(5-sqrt(5))/2`, every nonflat receiver
equilibrium is excluded and relative receiver phase converges to consensus.
The report exposes `receiver_nonflat_equilibria_excluded` and
`relative_receiver_consensus_certified`. It does not select the donor's final
state, prohibit transient winding or barrier passage, or supply a recovery
deadline. This is an endpoint result; `receiver_excitation()` owns the distinct
tangent and whole-network phase bounds.

`status` is `certified`, `not_certified` or `unavailable`. Admission uses the
exact conjunction `5-2*F>=0` and `(5-2*F)**2-5>=0`; a positive squared margin
alone cannot admit the wrong algebraic branch. Fresh shared cycle constants
produce `critical_form_storage_bounds` and
`target_minus_initial_margin_bounds`. Those intervals display the threshold
and comparison, even when too narrow a distinction is unresolved numerically;
they do not control the exact decision. Unsupported laws leave the functional
coefficients and endpoint bounds `None`, with explicit reasons. Invalid
consumed source data reject. Above the sufficient threshold there is no
formation verdict.

`to_dict()` uses shared exact projection with schema
`tnfr.relational-sine-receiver-localization.v1`. `receiver_transfer()` reuses
one validated preparation, retains this report in its optional trailing
`receiver_localization` field and adds `weighted_receiver_localization` as
an endpoint-exclusion reason when certified. Necessary passage/work fields
retain their conditional meaning; legacy reports leave the new field `None`.

### Receiver identity transfer with donor unwinding

`SineMediatedFormation.receiver_transfer()` assesses a different endpoint:
the donor becomes flat, the receiver retains winding `+1`, both bridge phase
gaps vanish and form becomes uniform. It retains the original formation
report as its source and does not reinterpret its coexistence verdict.
Under [chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission), static
symmetry flags, conserved means, mobility and phase-storage constants are
reconstructed from the admitted preparation and shared mathematical owners.
Cached equilibrium, silence or barrier fields cannot supply an exclusion.
The [transfer theorem](../../../theory/research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-receiver-transfer-admission)
fixes effective `e=w=1/2`, `beta=1`, capacity contrast `+1/2` and the same
eleven-node preparation. Other coefficients or contrasts leave this
transfer certificate unavailable; no alternative law is installed.
`SineReceiverTransferAdmission.status` separates `excluded`,
`passes_necessary_conditions`, `unresolved_bound` and `unavailable`.
The latter two distinguish unresolved numerical budget signs from failed
theorem premises. Its export schema is `tnfr.relational-sine-receiver-transfer.v1`.

The exact `target_phase_turns` and `target_geometry` reuse shared phase
reconstruction. The target is an acute full-support critical point. Its
local basin can be checked with `certify_sine_pattern_recovery` on a supplied
full-state observation or validated forecast. Target criticality alone
does not put the original preparation, or its later evolution, in that basin.
The compatible uniform form equals the original conserved weighted form
mean. The reported common phase offset selects a representative with no
additional integer turns; actual lifted history can require a weighted
integer correction. It is not a prediction of the final absolute phase.

At this target, storage and the mixed functional both equal `V5`.
Their source-minus-target margins are respectively `F` and `(4/5)*F`:
the shared phase term cancels exactly. These are endpoint budget margins,
not positive certificates of transfer. Zero form and the proved silent
donor subspace remain distinct exclusions.

First entry into the joint acute donor-zero/receiver-one region requires
storage at least `B5`. The remaining loss allowance is therefore
`A=F+V5-B5`, not the coexistence allowance `F-B5`. Both rings must cross an
antipodal edge before that entry. The donor's minimum angular excursion is
`3*pi/5`, the receiver's is `pi`; their disjoint nodal loss sums give the
combined action cost `29/25` at the admitted capacities. For resolved `A>0`,
the necessary entry time satisfies `T>58*pi**4/(25*A)` in the original
structural clock. The report provides a conservative rational lower bound.
Nonpositive or unresolved allowance cannot be divided by, and no entry time
is predicted. The supplied source horizon is used only for a strict
necessary-time comparison.

Any reused full-flow early-loss bound is compared with this transfer's own
allowance; the coexistence deficit and its exclusion flag are not copied.
Its horizon must be finite and strictly positive, and its retained loss bound
finite and nonnegative. This optional bound remains conditional evidence:
the caller must retain its justification for the same preparation, law and
clock through that horizon. The reader does not regenerate or authenticate
the timed-loss certificate.
An early-loss exclusion needs both the new entry-time condition and a
strictly positive recomputed deficit. Passing static conditions leaves
accessibility unresolved. Export retains the source, target, proof scope
and unavailable evidence through the shared exact JSON projection.

The optional receiver-passage fields retain a separate potential barrier:
`receiver_first_barrier_phase_storage=7/2`. This is the minimax cost between
the flat and twisted receiver wells, not the `B5` acute face or an antipodal
edge test. If transfer occurs, a finite first crossing must occur and the
signed accumulated incoming port work there must strictly exceed
`necessary_receiver_running_work_strict_lower_bound=7/2`.
`receiver_barrier_time_loss_product_lower_bound` is a conservative rational
lower bound for `14*pi**2/3`: crossing at time `tau` requires
`tau*D_receiver(tau)` at least this value. It uses full nodal gradients and
original incidence degrees in the receiver's share of continuous loss.
Neither field evaluates an integral or predicts a crossing time.

For initial form storage `F<=4`, `donor_barrier_first_required=True`
means that any such receiver crossing requires an earlier donor potential
barrier crossing. It does not assert an actual crossing, a winding change
at that instant or a donor endpoint. The distinct
`simultaneous_barrier_passage_excluded` flag excludes both ring potentials
being at least `7/2` together. Both flags reuse the full nonlinear phase
budget from `receiver_excitation()`, strengthening the energy-only sufficient
conditions `F<=7/2` and `F<=7-V5`. Admission uses exact source storage, not
a rounded display interval. A false flag leaves the condition unresolved;
it does not predict simultaneous passage or reverse the order. These fields
are `None` outside this reader's existing theorem domain or in legacy
records. They never become general transfer-exclusion reasons. The
[port-work and order proof](../../../theory/research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-receiver-port-passage)
has broader explicitly stated premises; this adapter retains its own scope.

<a id="sine-donor-well-retention"></a>
### Frozen receiver barrier exclusion

[`relational_receiver_barrier.py`](../../../src/tnfr/research/relational_receiver_barrier.py)
prepares and assesses one full eleven-node `H=2` source under the smooth sine
law. `prepare_receiver_barrier()` declares its exact phase turns, held
capacities, original clock, `T=32`, step `1/8` and Taylor order 16.
`evaluate_receiver_barrier()` delegates once to `bound_sine_flow`; it supplies
no independent solver or altered pressure. All 23 augmented coordinates are
retained. The held last capacity is an uncertainty coordinate, not another
evolving physical variable.

`assess_receiver_barrier_forecast()` checks the frozen declaration and complete
step coverage. Its [conditional theorem](../../../theory/research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-localized-receiver-exclusion)
requires receiver potential strictly below `7/2` in **every whole-time tube**
and full twelve-edge endpoint storage strictly below `7/2`. Together, these
exclude the receiver potential barrier for all future time. Safe endpoints
alone, receiver-only endpoint storage or an unfinished horizon cannot pass.
An upper bound failing either comparison is unresolved, not a proved passage.
No all-time winding constancy or selected donor endpoint follows.

The [producer](../../../benchmarks/relational_receiver_barrier.py) freezes a
protocol and source archive before evaluation, refuses overwrites, and retains
evaluation errors and unsuccessful responses. Post-evaluation source, runtime
and archive checks guard consistency. Public forecast records and archive
hashes do not independently authenticate execution or chronology. This is a
source-specific research certificate, not a native runtime event or new law.

### Donor-well recovery from the full prepared form state

`SineMediatedFormation.donor_well_retention()` returns the separate
`SineDonorWellRetention`, reusing the original exact donor-twist/flat-receiver
preparation and the [auxiliary-well theorem](../../../theory/research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-donor-well-retention).
It validates the consumed preparation, support, phase and capacity data and
rederives form storage rather than trusting an altered report's budget or
verdict. It does not recompute a trajectory or a timed loss producer.

The law domain is the existing `beta=1`, `delta=1/2`, `e>0`, `0<w/e<3/2`
domain of the global auxiliary function. On that domain, the exact critical
level `7/2` separates the donor's sublevel component from the other relative
equilibria. The source belongs to that component, or immediately enters it at
the boundary, whenever `F<=F_c=(25*sqrt(5)-55)/16`. The complete-law convergence
theorem then selects donor winding `+1` and receiver winding `0`, modulo
common phase origin. This is a terminal identity certificate; it does not
assert an unchanged winding or acute chart at every intermediate time.

Admission uses the exact rational sign of
`exact_retention_polynomial_margin=3125-(16*F+55)**2`, with nonnegative `F`.
The reported threshold and `escape_margin_bounds` are outward intervals for
display; the latter is **critical-minus-source**, `7/2-W(0)`. Interval overlap
or a rounded threshold cannot override the polynomial decision.

Within the law domain, `status="certified"` sets
`relative_donor_pattern_convergence_certified` and
`receiver_only_targets_excluded`. A larger form budget gives
`status="not_certified"` with both flags false; it proves neither escape nor
receiver formation. Unsupported law premises give `status="unavailable"`
and explicit reasons. There is no research ceiling `F<=4` in the reader's
admission; the actual sufficient condition is checked against the source.

`receiver_transfer()` reuses this certificate in its optional
`donor_well_retention` field and adds a distinct exclusion reason when
certified. Its older acute-face allowance `F+V5-B5`, phase-action time and
early-loss evidence retain their own definitions: the minimax level is not
substituted into a bound derived for another passage. The original coexistence
report is unchanged. The new export schema is
`tnfr.relational-sine-donor-well-retention.v1`, using shared exact JSON
projection and the SDK atomic writer. A public report remains distinct from
authenticated provenance or a checkpoint.

<a id="sine-donor-dissipative-capture"></a>
### Dissipative capture above the initial donor-well barrier

`SineMediatedFormation.donor_dissipative_capture()` returns
`SineDonorDissipativeCapture`. It reuses the source validation of the donor-well
reader and recomputes the full initial gradient `q=Lx` and
`N=q.T*K*q`, with `K=diag(nu_i/d_i)`. Stored gradient or loss verdicts do not
replace that calculation. Its [proof](../../../theory/research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-donor-dissipative-capture)
requires the original effective half weights, `beta=1` and `delta=1/2`.

This is an analytic finite-window bound, not a propagated trajectory. At the
fixed structural horizon `T=1/4`, the proof gives

```text
W(T) <= V5 + A,             A = (4/5)*F - N/25
V(theta0 + s*(theta(T)-theta0)) <= V5 + B,  B = 121*N/38400,  0<=s<=1
```

The sufficient admission is `A < (5*sqrt(5)-11)/4`, evaluated exactly through
`endpoint_exact_polynomial_margin=125-(11+4*A)**2 > 0`. On this admitted
support, `N<=6*F` proves `A>=14*F/25>=0`; it also makes the phase-path
inequality automatic whenever the endpoint condition passes. The retained
phase-path margin explains how the endpoint is connected to the donor's
component; it is not an independent adjustable gate. Outward display
intervals do not decide either exact algebraic comparison.

`donor_component_entry_certified` means the actual state at this horizon
belongs to the donor component below `W=7/2`.
`relative_donor_pattern_convergence_certified` and
`receiver_only_targets_excluded` then express its asymptotic consequences.
The certificate supplies no point-valued future state, convergence deadline,
claim of earlier escape, all-time winding history or physical identification.
Its fixed horizon and conservative coefficients are proof choices, not new
terms in the law or empirical constants. Strict admission includes an open
neighborhood of any certified preparation within the six-form source slice.

An admitted source outside the sufficient bound is `not_certified`; it is not
predicted to transfer. Unsupported law premises are `unavailable`, and invalid
consumed source data reject. `receiver_transfer()` appends this report in its
optional `donor_dissipative_capture` field and uses a distinct exclusion reason
when certified, preserving the earlier static and action-time contracts.
The export schema is `tnfr.relational-sine-donor-dissipative-capture.v1`,
using shared exact projection and the SDK atomic writer.

<a id="sine-asymptotic-equilibria"></a>
### Complete sine equilibria on the two-C5/intermediary support

[`classify_c5_sine_critical_set`](../../../src/tnfr/physics/phase_cycle_geometry.py)
accepts exact support geometry and two ordered, labeled C5 cycles. The graph
must consist of those disjoint rings and one remaining intermediary joined
to one node of each ring, with no additional edge. It retains the supplied
node order and ring orientations; only a common phase origin is removed.

The [completeness proof](../../../theory/research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-eleven-node-asymptotic-equilibria)
gives 30 exact configurations per labeled ring and two choices on each bridge,
or 3,600 full relative geometries. The catalog is factorized: it stores local
options and reconstructs an individual requested combination. Nonacute cycle
branches and antipodal bridges remain present. Its rational turns denote
exact multiples of mathematical `2*pi`, not rounded radian measurements.
`cycle_choices` follows the two declared cycles; `bridge_turns` follows the
catalog's `bridge_edge_indices` in full geometry order. The reconstructed
nodal representative fixes the first stored node's phase to zero.

The separate `CircularPhaseState` and `reconstruct_circular_phase_state`
share exact cycle closure and nodal reconstruction with the existing geometry
owner. They do not weaken `PhaseCycleState` or the strict acute admission of
`reconstruct_phase_cycle_state` and recovery readers. An antipodal branch
requires its declared turn representative; no branch-independent winding is
inferred from it. Geometric criticality alone is not a stability certificate.

`SineExchangeComparison.asymptotic_equilibria(cycles=...)` delegates to
[`assess_sine_asymptotic_equilibria`](../../../src/tnfr/physics/relational_sine_equilibria.py).
It retains the captured complete sine source, its model, support, state and
held capacities. With positive loss, exchange, storage scale and every
capacity positive, the theorem certifies convergence to one catalog member
and uniform form fixed by the conserved weighted form mean. It also proves
finite limits for continuous phase lifts, while their terminal integer turns
and the selected relative geometry remain unknown. No initial form budget or
acute-phase restriction is needed.

Zero loss or a frozen node leaves this stronger theorem unavailable; it does
not prove nonconvergence. Invalid source or support rejects. This detached
reader executes no trajectory, assigns no convergence rate or deadline and
installs no new runtime law. In particular, one exact sine equilibrium has
a zero native neighbor resultant, so this catalog cannot establish native
Arg-law continuation. Source provenance and unavailability remain explicit
in the shared exact JSON projection; an exported report is not authenticated
proof history or a checkpoint.
The report schema is `tnfr.relational-sine-asymptotic-equilibria.v1`.

#### Exact geometric inertia and full-law stability

The general `compose_bridge_tree_hessian_inertia(component_inertias, bridges)`
owner returns `BridgeTreeHessianInertia`. Each component triple is an
independently justified relative `(positive, negative, zero)` Hessian inertia.
Each bridge triple contains two component indices and sign `+1` or `-1`.
Ordered collections, nonnegative non-Boolean integer counts, valid distinct
indices, unique pairs and a connected tree are required. An intercomponent
cycle is rejected. Singleton components use `(0,0,0)`; additional relative
zero modes remain in the output rather than becoming an attraction verdict.

The shared algebra retains bridge order/orientation, `total_nodes`, relative
inertia and one removed common-phase direction. Its export schema is
`tnfr.bridge-tree-hessian-inertia.v1`. These are conditional geometric results:
the reader does not infer component inertias or bridge signs from a live
state, authenticate their premises, classify a dynamical law or execute an
attachment. A one-node report is valid algebra, but is outside the nonisolated
row-normalized dynamical theorem. The
[composition proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-bridge-tree-composition)
states the full-law consequences and the additional equilibrium premises.

`C5SineCriticalSet.phase_hessian_inertia(cycle_choices=..., bridge_turns=...)`
returns `C5PhaseHessianInertia` for one revalidated exact catalog member.
It derives both component inertias and actual bridge signs before calling
that shared composition owner; its existing report and schema remain unchanged.
`relative_inertia` records `(positive, negative, zero)` dimensions of the
cosine-weighted phase Hessian after removing one common phase origin.
`common_phase_nullity=1` retains that removed direction separately. The report
includes the exact reconstructed state, ring negative-edge counts and bridge
signs. It uses constrained cycle algebra, not a floating eigenvalue threshold.
Its schema is `tnfr.c5-phase-hessian-inertia.v1`.

`phase_hessian_index_counts[m]` counts catalog members with `m` negative
relative directions. This compact histogram comes from the local branch
counts and two independent bridge signs. No full Cartesian list is stored.
The [stability proof](../../../theory/research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-eleven-node-equilibrium-stability)
shows that every relative Hessian is nonsingular; a global neutral origin
must not be reported as an unresolved relative degeneracy.

`SineAsymptoticEquilibria.classify_equilibrium(cycle_choices=..., bridge_turns=...)`
returns a separate `SineEquilibriumStability`. It revalidates the captured
source through the shared asymptotic admission and checks the supplied catalog
against that source before applying the full reciprocal-law theorem. Altered
public flags cannot override the actual loss or capacity premises.

For positive loss, exchange, storage scale and held capacities, a negative
Hessian index `m` gives `relative_unstable_modes=m`,
`relative_stable_modes=20-m` and `relative_center_modes=0` on the two
conserved-mean tangent spaces. The full 22-dimensional tangent retains two
neutral common-origin directions. These counts follow from the complete
form-phase pencil; they do not assume commuting form and phase Hessians.
For `m=0`, status is `locally_exponentially_attracting` and
`local_exponential_attraction_certified` is true. Otherwise status is
`nonlinearly_unstable` and `nonlinear_instability_certified` is true.
Missing law hypotheses give `unavailable`, false certification flags and
`None` mode counts, while exact geometry remains available.

There are nine locally attracting relative geometries under these premises:
each ring independently has winding `-1`, `0` or `+1`, with both bridge gaps
zero. Strictly positive coefficient or capacity changes preserve the local
classification, but may change rates, responses and basins. Classification
of a named equilibrium does not assert that the captured source is near it,
that it will reach it, or that its basin has a supplied numerical radius.
The export schema is `tnfr.relational-sine-equilibrium-stability.v1` and uses
the same exact projection and source-label controls as the parent report.

### Joint acute cycle-period obstruction

`assess_acute_cycle_periods(geometry, cycle_periods=..., cycle_combination=...)`
in [`phase_cycle_geometry.py`](../../../src/tnfr/physics/phase_cycle_geometry.py)
returns `AcuteCyclePeriodAssessment`. The reader rederives and validates the
supplied `PhaseCycleGeometry`; its existing support and evaluation budgets
remain unchanged. Ordered periods must be non-Boolean integers in the stored
fundamental-cycle order. The ordered combination must have that same length
and be nonzero. Rational coefficients remain exact; other supported real
coefficients follow shared represented-real admission. A tree has no nonzero
cycle combination and cannot supply this witness.

The reader combines the actual signed edge chains before taking their absolute
sum. `combined_period` is the same linear combination of supplied periods;
`strict_period_bound` is one quarter of the combined chain's L1 norm in turns.
`strict_bound_margin` subtracts the absolute combined period from that bound.
A nonpositive margin gives `status="obstructed"` and
`obstruction_certified=True`: no strictly acute circular phase configuration
has those periods. Equality is excluded because the phase domain is open.

A positive margin gives `status="necessary_bound_passed"` and no obstruction
certificate. It does not establish phase feasibility, sine balance, equilibrium
or recovery. The function tests one supplied witness; it neither enumerates
all separating directions nor solves a nonlinear equilibrium. Periods and
chains retain their specified cycle basis. No live graph or state is changed.
The export schema is `tnfr.acute-cycle-period-assessment.v1`, with shared exact
projection and node-label validation. The
[joint-compatibility theorem](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-cycle-sector-compatibility)
owns the full criterion, complete-law transfer and counterexamples.

### Target-free acute-sector capture

`certify_sine_sector_capture(source, edge_turn_offsets=...)` in the shared
[`relational_sine_recovery.py`](../../../src/tnfr/physics/relational_sine_recovery.py)
returns `SineSectorCapture`. It accepts a captured `SineExchangeComparison`,
`SineRelativePattern` or `SineRelativeForecast`. Pattern and forecast reports
also expose `.certify_sector_capture(...)`. No equilibrium target, local
target radius or reference phase profile is supplied.

Offsets are exact non-Boolean integers in the canonical `PhaseCycleGeometry`
edge order. Each declares `delta_e=theta_head-theta_tail+2*pi*offset_e`;
the resulting periods are the fundamental-cycle sums of those integers.
Independent edge offsets can express nonzero winding. They are not the
node-wise `phase_turns` argument of the target-centered recovery readers.
Support retains the shared 32-node/50-edge evaluation budget, all nodes and
all unit edges. Supplied graph, coefficients, capacity and sector remain
premises rather than autonomous selections.

The reader uses [chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission) and
recomputes form and phase storage from source coordinates and uncertainty. It
requires positive held capacities and positive loss, exchange and storage
scale. A zero loss or frozen node cannot receive the convergence theorem.

The complete source set must be strictly acute in its declared cell.
`boundary_face_lower_bounds` covers every canonical edge with signs `-1,+1`,
including possibly empty faces. Convex supporting planes, shared exact linear
algebra and outward rational trigonometry provide each bound. The minimum is
`boundary_phase_storage_lower_bound`; multiplication by the storage scale gives
`boundary_storage_lower_bound`. The strict `energy_margin` subtracts
`storage_upper_bound` from this barrier. These are sufficient bounds, not
claims that a numerical optimizer found the exact boundary minimum.

With all premises and both strict comparisons satisfied, `status="admitted"`
certifies existence of one interior equilibrium phase geometry, uniqueness in
the sector, forward sector retention and convergence of every represented
initial state under the complete sine law. Its coordinates and a convergence
time are not produced. Otherwise `status="unavailable"` retains separate
`hypothesis_failures` and `unresolved_conditions`; no instability or absence
of equilibrium follows. The
[theorem](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture)
owns the compactness, invariance and full nodal dissipation proof.

Comparison sources retain exact captured degree/capacity-weighted form and
declared lifted-phase means. Relative observations leave absolute origins
unobserved. Forecasts follow the full-endpoint and held-capacity requirements
of shared admission. The schema is `tnfr.relational-sine-sector-capture.v1` and
uses shared exact JSON projection and label checks. No graph, event or
trajectory is executed. Capture within this supplied sector does not certify
formation of the sector from another preparation or admission of a preceding
support/reset event.

### Analytic prepared entry into a captured sector

`certify_sine_prepared_entry(source, scaled_time=..., edge_turn_offsets=...)`
in [`relational_sine_entry.py`](../../../src/tnfr/physics/relational_sine_entry.py)
returns `SinePreparedEntry`. An exact captured `SineExchangeComparison` retains
its identical-real-phase admission contract. A `SineRelativePattern` instead
supplies its original per-node form and phase residual radii, consistent real
lifts and arbitrary common origins; its `.certify_prepared_entry(...)` adapter
uses this same owner. Both paths require strictly positive exact held
capacities and positive loss. The complete normalized sine law, all
support and all coefficients remain unchanged through transit and capture.
Unsupported source domains raise. The supplied nonnegative `scaled_time=tau`
corresponds to `horizon=tau/e` in the declared original clock. The shared
support budget is 32 nodes/50 edges; the exact exponential work budget is
`lambda*tau<=4096`. The reader never adjusts the supplied horizon or law.

The [transit theorem](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-prepared-sector-entry)
uses the exact degree/capacity-weighted mean, reversible diffusion gap and a
global bound on the nonlinear sine current. It retains both complete evolution
rows through nonacute and antipodal passage. Weighted centering contracts the
initial form residual norm. Initial phase and form errors remain in the phase
endpoint; `phase_radius_upper_bound` describes only its dynamic correction.
Cached source storage, projected relative rectangles and nominal means cannot
replace the original residual set. Initial storage is recomputed with both
form and phase uncertainty.

`centered_endpoint_form_bounds` and `centered_endpoint_phase_bounds` enclose
every node after subtracting that member's own conserved weighted means.
Tighter edge intervals retain the same correlations. For an exact source,
`endpoint_form_bounds` and `endpoint_phase_bounds` also enclose absolute
coordinates. For a relative pattern they remain `None`, as do the absolute
weighted means, even at zero residual error. Nominal mean fields describe
only the supplied centers. The family is a union of distinct conserved mean
leaves; arbitrary corners of the outer node rectangle need not satisfy the
correlated edge constraints. No forecast or observation is fabricated.

The existing sector-capture interval kernel assesses that enclosed endpoint at
the actual `horizon`, using the same complete law. Its source field retains
the initial source; `uncertainty_scope` explicitly identifies the analytic
endpoint. `input_forecast_admitted` and `input_forecast_requested_end_time`
remain unavailable. `initial_zero_winding_certified` requires every original
raw phase-edge interval to be strictly acute, with no automatic unwrapping.
Their differences then telescope, proving zero initial periods. If this
sufficient initial comparison fails, `initial_cycle_periods` is `None` and
acquisition is unavailable. Admission additionally requires endpoint capture
and a nonzero final period vector. An early,
loose or wrong-sector comparison remains unavailable, not proof of failed
formation. No exact crossing time or acuteness during transit is claimed.

`initial_form_storage` is exact only without form uncertainty; otherwise it
is `None`. The additive `initial_form_storage_bounds`,
`initial_phase_storage_bounds` and `initial_storage_bounds` retain the full
preparation budget. A structured form preparation can be very
costly, and the small-ratio existence family has diverging initial storage.
This is conversion of supplied form information into maintained phase winding,
not autonomous selection of a preparation, support birth or physical identity.
The exact export schema is `tnfr.relational-sine-prepared-entry.v1` and uses
the shared JSON projection and label admission.

<a id="sine-slow-phase-bound"></a>
### Controlled slow-phase comparison

`bound_sine_slow_phase(source, slow_time=...)` in
[`relational_sine_reduction.py`](../../../src/tnfr/physics/relational_sine_reduction.py)
returns `SineSlowPhaseBound`; `SineRelativePattern.bound_slow_phase(...)`
delegates to the same owner. Sources are exact `SineExchangeComparison`
preparations or original `SineRelativePattern` residual families, with all
primitive data revalidated by the shared preparation owner. Forecast reports
are not accepted. The fixed connected simple unit support, positive exact
held capacities, positive loss, exchange and storage scale belong to the
complete normalized sine law. Initial phases may be nonuniform and nonacute
on their supplied real lifts. The identical-phase restriction of exact
prepared entry does not apply to this comparison.

The caller supplies a finite nonnegative `slow_time=sigma` through shared
exact/represented-real admission; Booleans and nonfinite inputs reject.
With `alpha=w/(beta*pi*e)`, `eta=beta*alpha^2` and `tau=e*t`, the
comparison instant satisfies `sigma=eta*tau`. The report retains the supplied
exact `slow_time` and outward intervals
`scaled_time_bounds=sigma*beta*e^2*pi^2/w^2` and
`horizon_bounds=sigma*beta*e*pi^2/w^2`. These generally irrational clock
conversions enclose one declared instant; they are not independently uncertain
times or midpoint substitutes. No horizon is selected or shortened.

The [comparison theorem, Section 29](../../../theory/nodal/SINE_FORM_PHASE_REDUCTION.md#sine-controlled-slow-phase)
uses `K=diag(nu_i/d_i)`, `M=K^-1`, `A=KL` and
`z=alpha*P_M*x`, where `P_M` subtracts the weighted mean. Its reference obeys
`dpsi/dsigma=K*S(psi)` with `psi(0)=theta(0)+z(0)`. Initial form information
therefore remains in the reference preparation. Write
`D(tau)=exp(-A*tau)`. The reported bounds distinguish these weighted
`M`-norm comparisons at the declared instant:

| Report field | Quantity bounded |
| --- | --- |
| `composite_phase_error_upper_bound` | `theta(tau)+D(tau)*z(0)-psi(sigma)` |
| `phase_error_upper_bound` | `theta(tau)-psi(sigma)`, including the fast initial transient |
| `joint_error_upper_bound` | `theta(tau)+z(tau)-psi(sigma)` |
| `scaled_form_remainder_upper_bound` | `z(tau)-D(tau)*z(0)` |
| `form_remainder_upper_bound` | `P_M*x(t)-D(tau)*P_M*x(0)` in original form units |

`uniform_composite_phase_error_upper_bound` covers the corrected comparison
over the entire interval from zero through the supplied slow horizon.
The uncorrected phase has initial discrepancy `-z(0)`; uniform small error
from time zero does not follow for a fixed nonzero scaled preparation.
`scaled_form_norm_upper_bound` and `form_norm_upper_bound` also retain their
homogeneous transients. The scalar `exponential_decay_bounds` controls the
matrix semigroup's norm; it is not a componentwise replacement for
`D(tau)*z(0)`. No matrix exponential or reference trajectory is evaluated.

`phase_edge_error_upper_bounds` follow the canonical edge order and bound
differences between actual and reference edge gaps. The separate
`phase_potential_error_upper_bound` concerns
`V(theta)=sum_edges(1-cos(theta_j-theta_i))`, without a storage-scale factor.
`initial_form_norm_upper_bound` and `initial_storage_bounds` preserve the
original preparation cost. Small feedback strength with fixed scaled form
does not imply a small original form or storage budget.

Every member of a relative residual family is compared with its own reference
initialized from that member's actual phase and form. The common error bound
does not compare the entire family to one nominal reference trajectory.
`reference_initial_phase_bounds`, `weighted_form_mean` and
`weighted_phase_mean` are `None` for relative sources, including zero-error
sources with unobserved origins. `centered_reference_initial_phase_bounds`
retains correlated outer bounds after subtracting each member's own weighted
phase mean. `reference_scope` records this distinction. Cached storage and
projected relative rectangles do not replace the original residual set.

Shared geometry admission retains the 32-node/50-edge budget. The global
Lipschitz bound is `ell=2*max(nu_i)`; the rational growth enclosure requires
`ell*slow_time<=4096`. Unsupported domains and larger growth exponents raise.
Decay uses the shared exact exponential owner through exponent 4096 and a
monotone tail enclosure beyond it, recorded by `decay_tail_enclosure_used`.
This preserves the requested comparison time. The tail enclosure and outward
128-bit dyadic rounding can impose a numerical floor on very small errors;
they do not weaken the analytic theorem or certify a rounded value as zero.

The report supplies error bounds without an accuracy threshold or an
`admitted` verdict. A large bound remains a bound. It supplies no solver
execution, endpoint capture, basin match, infinite-time limit exchange or
event selection. Those claims require their own full-state evidence. Its
schema is `tnfr.relational-sine-slow-phase.v1`. Direct `to_dict()` export and
`relational_report_to_dict()` delegation retain the same exact report body and
node-label admission, with their respective schema envelopes. The latter also
supports `SinePreparedEntry` and `SineSlowCapture`; neither route authenticates
source provenance.

<a id="sine-slow-capture"></a>
### Full-state capture from a controlled phase reference

`certify_sine_slow_capture(source, slow_time=..., target_phase_turns=...)`
in [`relational_sine_reduction.py`](../../../src/tnfr/physics/relational_sine_reduction.py)
returns `SineSlowCapture`. `SineRelativePattern.certify_slow_capture(...)`
delegates to this owner. An existing `SineSlowPhaseBound` also exposes
`.certify_capture(target_phase_turns=...)`, which recomputes preparation and
comparison bounds from its `source` and `slow_time` under
[chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission). The same
exact/relative source, positive complete-law coefficients, held capacities,
finite slow clock and
numerical budgets apply as in the preceding comparison contract.

`target_phase_turns` supplies one exact integer or `Fraction` per captured
node, in source node order. Floats, Booleans, incomplete vectors and unsupported
targets reject. The shared target owner reconstructs the circular geometry,
requires strictly acute principal edge turns and proves exact sine criticality
by its symbolic odd-cancellation test. An unresolved symbolic identity is an
unsupported target, not a proved absence of equilibrium. `target_geometry`,
`target_phase_turns` and `target_edge_turns` retain this evidence; the edge
fields use the canonical geometry order. No numerical equilibrium solve or
target search is performed.

The [handoff theorem, Section 30](../../../theory/nodal/SINE_FORM_PHASE_REDUCTION.md#sine-slow-phase-capture-handoff)
first establishes an independent enclosure of the reference phase flow.
`target_centered_phase_bounds` subtracts the target's degree/capacity-weighted
mean before enclosing mathematical pi. Each actual source member compares
with the stationary target shifted to that member's conserved phase mean.
`nominal_initial_reference_distance_bounds` and
`initial_reference_uncertainty_upper_bound` retain the initial mismatch and
the original form/phase residuals. Their combined bound propagates with the
global Lipschitz growth factor into `reference_distance_upper_bound`.
This bounds each member's own reference; it does not replace the family by
one nominal flow or assign the target to an actual state.

`actual_phase_distance_upper_bound` adds the complete law's uncorrected
phase error, including its fast initial transient. The remaining original
form norm is retained from `slow_phase.form_norm_upper_bound`. These two
norm bounds supply correlated outer edge intervals for the actual endpoint.
The target's supplied real lifts determine integer edge offsets exactly;
the reader does not unwrap the source to reduce a residual. Arbitrary common
origins cancel, while unknown absolute means of relative sources remain
unavailable.

For `ell=2*max(nu_i)`, phase distance bound `r` and original form norm bound
`X`, the report also proves
`correlated_form_storage_upper_bound=ell*X^2/2` and
`correlated_phase_storage_upper_bound=V(target)_upper+ell*r^2/2`.
The phase expression uses criticality to cancel the linear term and bounds
the unscaled potential `V`; the complete storage includes its factor `beta`.
The shared capture core intersects these producer-proved upper bounds with
its edgewise storage intervals, preserving their lower endpoints. These
private intersections are not public caller-supplied certificate inputs.
The enclosed endpoint retains both edge boxes and the proved correlations;
arbitrary corners of the larger Cartesian edge box need not satisfy its
storage upper bounds.

The nested `capture` requires every enclosed endpoint phase-edge interval to
be strictly acute and total form-plus-phase storage to lie strictly below all
certified sector boundary faces. `SineSlowCapture.admitted` and `.status`
delegate to this full-state result. Good phase geometry with excessive form
storage, or an unresolved acute margin, remains unavailable. Such a failed
sufficient comparison does not prove instability or impossibility. Passing
certifies the actual source family's subsequent retention and convergence in
that sector under the unchanged complete law.

`source` remains the original preparation. The analytic endpoint time is
enclosed by `slow_phase.horizon_bounds`; nested `capture.observation_time`
and forecast metadata remain `None`. No rounded timestamp, observed endpoint,
supplied response or solver trajectory is inserted. Initial winding is not
assumed or certified, and a zero target period is allowed: this capture
verdict alone is not an acquisition or winding-creation certificate. The
target remains a proof reference, without a state assignment, changed law or
event. Exact export uses `tnfr.relational-sine-slow-capture.v1` and the shared
SDK projection and label admission, retaining the nested preparation,
comparison, target and capture provenance.

<a id="sine-budget-consensus"></a>
### Consensus for a phase-flat preparation budget

`certify_sine_budget_consensus(geometry, reference_model=..., capacity=...,
form_storage_budget=...)` in
[`relational_sine_budget.py`](../../../src/tnfr/physics/relational_sine_budget.py)
returns `SineBudgetConsensus`. Its `family` is a `SinePhaseFlatBudgetFamily`:
every signed form vector with `x.T*L*x/2<=B`, initially identical real phase
lifts, and arbitrary independent common form and phase origins. No particular
EPI vector, phase record, reference response or representative state is
consumed. Independent nonzero phase errors are outside this family.

The supplied `PhaseCycleGeometry` is rebuilt from its primitive support;
forged derived fields reject. The connected simple support declares unit
sine coupling and retains the 32-node/50-edge budget. `capacity` contains one
strictly positive exact/represented scalar per geometry node, held throughout
the complete unforced sine law. The regular reference model supplies positive
loss, exchange and storage scale. `form_storage_budget=B` must be finite and
nonnegative. Shared scalar admission rejects Booleans and nonfinite values.
This reader has no horizon argument or trajectory solver.

The [all-time theorem](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-budget-consensus)
uses the shared weighted quotient gap `lambda`, `K=diag(nu_i/d_i)`, `M=K^-1`
and `alpha=w/(beta*pi*e)`, with `eta=beta*alpha^2`. It bounds the initial
auxiliary function
`W=(||P_M*theta+z||_M^2+||z||_M^2)/2`, `z=alpha*P_M*x`, by
`Wbar=2*eta*B/(beta*lambda)`. This function is distinct from the model's
original form-plus-phase storage. For `B>0`, the sufficient premises are
`eta<=1` and, on every edge, `2*sqrt(K_i+K_j)*sqrt(Wbar)<pi`.
`feedback_margin` and `branch_margins` retain their outward interval checks.
The feedback premise is a sufficient proof condition, not an instability
threshold when it fails.

Passing proves `consensus_certified` and `zero_winding_certified` for every
family member and all nonnegative structural times (`tau=e*t`). Each member
converges to its own conserved weighted form mean and common phase lift.
No convergence rate or common absolute origin is supplied. The natural
invariant strip is `|theta_j-theta_i|<pi`, so consensus and zero winding do
not require every edge to remain acute. `acute_margins` and the stronger
`acute_trapping_certified` flag separately test the `pi/2` condition.
If `B=0`, connectedness forces uniform form as well as flat phase: the whole
family is exactly stationary for every admitted positive coefficient ratio,
independently of the `eta<=1` premise.

`initial_form_norm_upper_bound` and `initial_lyapunov_upper_bound` retain
preparation bounds. `candidate_phase_edge_upper_bounds` describes the proposed
invariant sublevel and is not a trajectory bound unless the certificate
passes. Then `lyapunov_upper_bound`, `phase_norm_upper_bound`,
`scaled_form_norm_upper_bound`, `form_norm_upper_bound` and
`phase_edge_upper_bounds` supply uniform all-time bounds on the centered
coordinates and canonical edge gaps. When sufficient inequalities remain
unresolved, `status="unavailable"`, `unresolved` states why, and those
certified-bound fields are `None`. Initial and candidate fields remain visible;
their availability does not certify motion, formation or instability. Fixed
dyadic rounding can leave a mathematically valid strict condition unresolved.

When the feedback premise is certified,
`nonzero_winding_budget_lower_bound` gives a conservative necessary original
form cost for nonzero winding acquisition from exactly flat phases. With
`r=w/e` and `kappa=max_edges(K_i+K_j)`, it encloses from below
`beta^2*lambda*pi^4/(8*kappa*r^2)`. This inverse-square requirement is neither
sufficient for formation nor a sharp threshold. The field is `None` when its
feedback premise is unavailable. The result covers the declared fixed-budget
family; it does not extend to arbitrary initial phase records or select a
preparation, constitutive law or support.

The exact schema is `tnfr.relational-sine-budget-consensus.v1`. `to_dict()`
and the generic SDK exporter reuse the shared rational projection and label
admission. Generic export wraps the same report payload in the standard
`tnfr.relational-report.v1` envelope; SDK file export uses the shared atomic
writer. They retain the explicit family and theorem scope without
authenticating a physical preparation. See the
[C5 usage example](../../guides/relational/SINE_PATTERNS.md#certify-an-entire-phase-flat-budget-family).

<a id="sine-cycle-symmetry"></a>
### Full-state symmetry and an oriented cycle

`assess_sine_cycle_symmetry(source, permutation_indices=..., cycle=...)` in
[`relational_sine_symmetry.py`](../../../src/tnfr/physics/relational_sine_symmetry.py)
returns `SineCycleSymmetryAssessment` for one supplied node permutation under
the complete unforced sine law.
`source.assess_cycle_symmetry(...)` delegates to the same owner. The source
must be an exact `SineExchangeComparison` and uses
[chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission).
Nonnegative form loss and capacities are allowed; exchange and
storage scale must be positive. No inverse capacity, spectral gap or
phase-geometry solver budget is needed.

`permutation_indices[i]` gives the destination index of source node `i`.
The ordered sequence must contain each index exactly once, with nonboolean
integer entries. `cycle` gives at least three distinct source labels in
orientation order, without repeating its first vertex at the end. Every
successive edge, including the closing edge, must belong to the support.
The cycle may cover only part of a larger graph, but the permutation must
preserve the complete support and all capacities. A symmetry of the cycle
alone does not establish equivariance of the full law.

`support_automorphism`, `capacity_preserved`, `form_preserved` and
`phase_lift_preserved` separately record the support and exact full-state
checks, including `x[p(i)]=x[i]` and `theta[p(i)]=theta[i]`.
`cycle_orientation_reversed` compares the mapped signed cycle chain with
the negative original chain. Phase equalities concern the
captured real lifts, without a tolerance or an inferred nodewise wrapping.
The [proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-equal-budget-preparation)
uses full-field equivariance and uniqueness to preserve a fixed initial
state: `trajectory_symmetry_certified` does not itself require cycle reversal.
Reversal additionally certifies `zero_winding_when_nonantipodal`: the cycle's
principal winding is zero at every time when none of its edges is antipodal.
`nonzero_winding_limit_excluded` also rules out a circular phase limit with
nonzero winding and no antipodal cycle edges. In particular, the source cannot
acquire a strict acute nonzero winding on that cycle.

The full obstruction gives `status="certified"`; otherwise
`status="unavailable"` and `unresolved` identifies each failed premise.
Malformed source, permutation or cycle inputs raise. Exact form, phase-lift
and capacity residuals remain visible. `initial_form_storage` is recomputed
from primitive edge differences, independently of cached source storage.

This conditional exclusion does not certify avoidance of antipodal edges,
assign a principal period on that boundary, identify a limiting equilibrium
or prove consensus. A failed symmetry check does not prove acquisition.
The reader supplies no trajectory, horizon or event and does not enumerate
an automorphism group. Common form and phase shifts preserve the tested
equalities. A permutation-invariant uncertainty box need not have invariant
individual members, so relative patterns and forecasts are unsupported.
Combined transformations such as a permutation followed by global sign
reversal are also outside this pure-permutation test.

`to_dict()` uses schema `tnfr.relational-sine-cycle-symmetry.v1` with the
shared SDK rational projection and label admission. Generic SDK export wraps
the same report payload in `tnfr.relational-report.v1`; file export uses the
shared atomic writer. These exports retain the revalidated source, supplied
permutation and oriented cycle without authenticating a physical preparation. The
[equal-budget example](../../guides/relational/SINE_PATTERNS.md#distinguish-equal-budgets-by-preparation-symmetry)
compares this obstruction with the separately certified acquisition source.

<a id="sine-conservative-identity"></a>
### Conservative cycle identity and recurrent families

`assess_sine_cycle_identity(source, cycle=..., winding=..., radius=...,
excess_ceiling=..., form_mean_bounds=(lower, upper), phase_turns=None)` in the
[shared recovery/geometry owner](../../../src/tnfr/physics/relational_sine_recovery.py)
assesses the complete conservative sine law on an acute cycle. `source` is
a `SineRelativePattern` or `SineRelativeForecast`. The same source admission,
exact target, centered norm, Taylor remainder and coercive barrier serve
the existing recovery certificate. Positive-loss recovery still requires
`e>0`; this new assessment requires exactly `e=0` and positive held capacities.
No model coefficient is changed to make either theorem apply.

The target has integer winding `k` with `4*abs(k)<n`. The supplied radius
must certify `abs(2*pi*k/n)+sqrt(2)*radius<pi/2`. With the shared lower
coercivity bound `kappa`, the family is

```text
Z² < radius²,
E - E_target < excess_ceiling < kappa*radius²,
lower < sum_i(d_i/nu_i)*x_i / sum_i(d_i/nu_i) < upper.
```

`excess_ceiling` is strictly positive and the mean interval has positive
width. These are caller-declared analysis restrictions, not new parameters
in the evolution law. Shared scalar admission rejects Booleans, nonfinite
values and malformed bounds. The family is open, invariant in both time
directions and has finite positive ambient volume; its closure is compact.
Every member retains its acute cycle geometry and winding. Almost every
member is nonstationary and recurrent in full form/circular-phase state.

`SineCycleIdentityAssessment` keeps distinct conclusions:

- `family_admitted` and `family_almost_everywhere_recurrence_certified` concern
  the invariant family, not the supplied observation's individual future.
- `source_set_trapping_certified` concerns every state consistent with the
  source's relative uncertainty and the declared held law. It requires the
  strict radius and barrier tests, independently of a smaller family ceiling.
- `relative_source_family_membership` is `certified_inside` only when the
  source bounds also satisfy the family ceiling. Otherwise it is `unresolved`:
  failure of a sufficient upper bound is not an outside-state proof.
- Absolute family membership remains unavailable because these relative
  sources discard an arbitrary common form origin. The supplied mean slab
  restricts a family; it does not measure that missing origin. Recurrence of
  a chosen state remains `unavailable_for_chosen_state`.

Forecast reports retain their full endpoint box, actual validated time and
original admission status. When a held capacity is interval-valued, the
theorem applies separately to each admitted fixed positive realization,
using that realization's weighted mean. It does not define a flow of uncertain
or changing capacities. A source set may be trapped even when a requested
family ceiling fails admission; an admitted family may coexist with unresolved
source trapping. Their margins and unresolved conditions are kept separate.

The [proof](../../../theory/nodal/SINE_PATTERN_RECOVERY.md#sine-conservative-identity)
uses conserved excess storage and the shared finite-volume recurrence
argument. It supplies no return time, single frequency, attraction, pattern
formation, coarse-scale closure or physical identification. A finite grid,
singleton or fixed-energy surface does not inherit an ambient
almost-everywhere result automatically.

`to_dict()` uses `tnfr.relational-sine-cycle-identity.v1`; the generic SDK
export delegates to the same exact projection and label admission. This
detached assessment evolves no graph and installs no new runtime law.
