# Normalized-sine comparison, inference and forecasts

Shared sine-law report admission, captured comparison, prepared transport, source and interaction budgets, retained mediation, sampling, hidden state/capacity inference and prior-only forecasting.

Part of [Relational dynamics contract index](../RELATIONAL_DYNAMICS.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="sine-chained-report-admission"></a>
### Admission when chaining sine reports

Report-consuming readers that link here re-admit the declared inputs needed
by their calculation, reconstruct consumed derived quantities and apply their
own theorem hypotheses. Cached bounds, degrees, rates or success flags cannot
replace those inputs. Each reader's section identifies its reconstruction
inputs and any stronger domain or numerical budget.

For complete captured or relative sources,
[`_sine_admission.py`](../../../src/tnfr/physics/_sine_admission.py) checks the
declared sine law, simple connected unit support, node-coordinate order,
degree/neighbor consistency, finite form/phase/capacity vectors and nonnegative
residual radii. Neighbor-row order does not change support equivalence.
Authoritative stored model coefficients are admitted without a second
normalization: exact rationals retain their values; negative loss, Boolean
or nonfinite coefficients reject. Admission precedes coefficient equality,
including comparisons between a pattern and its full forecast.

Graph capture and exact detached declarations are different input paths.
`bound_relational_sine_exchange` first captures graph coordinates through the
shared binary64 boundary; the resulting fractions represent those captured
values, not necessarily an originally supplied `Fraction`. A downstream
reader that admits exact rational primitives can instead consume an explicitly
declared detached source in its retained node order. That declaration is a
new mathematical preparation, not an exact reconstruction of a rounded graph.
Rebuilding rates and bounds is required after such a change. Neither rounded
radians nor a small residual can stand for a separately declared exact turn
target, branch history or invariant family.

A consumed `SineRelativeForecast` must retain the pattern's model, support and
held capacities, a complete endpoint layout and a valid actual time within
its declared horizon. Visible capacities equal the pattern's exact values;
the initial final-capacity coordinate uses the interval owner's outward
rounding of the pattern's final exact capacity,
and its nonnegative endpoint interval must still contain the held value.
A wider endpoint enclosure is permitted. The changed-capacity `freeze_hidden`
control is unsupported by these pattern readers. They assess the full endpoint
at `validated_end_time`, retaining the solver's status and requested horizon;
they do not replace it with projected rows or original static bounds.

The hidden-state/capacity/prior chain instead reconstructs its inverse from
retained visible state, unit incidence, first-rate observations and prior
metadata in the [observation owner](../../../src/tnfr/physics/relational_sine_observation.py).
Capacity inference also uses its supplied acceleration channels, with omitted
channels unavailable. The [forecast owner](../../../src/tnfr/physics/relational_sine_forecast.py)
rebuilds joint witness admission and the complete initial box before propagation.
These detached stages preserve exact values without recapturing a graph or
accessing the live Gamma registry.

Calculations use the admitted, normalized inputs. Source association is
separate: complete-source readers may retain the supplied report, while
inverse/prior adapters preserve an original only when it equals the rebuilt
report after validation. Neither association supplies raw inputs to subsequent
arithmetic. Other unconsumed snapshot fields may remain attached; their
presence or object identity is not evidence that they were recomputed.
Only the requested calculation's inputs are checked.
This boundary does not authenticate acquisition, source chronology or public
dataclasses, and does not replay a forecast or prove that observations share
a trajectory. Export preserves the report under its existing schema.

Shared coefficient and source admission imposes no global geometry or solver
work cap. Positive loss/capacity, exact symmetry, target criticality and each
finite work budget remain requirements of the individual consumer.

### Detached normalized-sine complete-law comparison

`bound_relational_sine_exchange(graph, reference_model=...)` in
[`relational_sine_comparison.py`](../../../src/tnfr/physics/relational_sine_comparison.py)
captures the same signed scalar, phase, capacity and unit-support contract,
but computes the explicitly distinct law
`normalized_sine_reciprocal_exchange`. The reference model must be explicitly
regular; it supplies the shared effective coefficients and storage scale,
not the candidate's phase-domain admission. The comparison uses
`s_i=sum_j sin(theta_j-theta_i)/(pi*d_i)` in form pressure and
`theta_dot_i=w*nu_i*(Bx)_i/(beta*pi*d_i)`. It is defined at zero and branch
resultants, including states rejected by the native model.

The [mathematical owner](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#global-closure-pressure-comparison)
declares the changed pressure-superposition premise and derives the full
phase row, balance and continuation. No new `phase_domain` value or native
dispatch setting is introduced. The existing mean-sine diagnostic does not
select this dynamical interpretation by itself.

`SineExchangeComparison` stores captured represented coordinates as exact
rationals and encloses ideal trigonometric sources, pressure, both rates and
storage. Form gradients and dissipation are exact rational values. Nodal
and global work residuals are computed interval sums, not assigned zero;
enclosure of zero is a consistency check, not the all-state proof.
`to_dict()` uses shared exact projection with schema
`tnfr.relational-sine-comparison.v1` and retains the separate law identifier.
Node and edge labels pass the shared SDK label validator before projection:
JSON scalar/tuple labels are supported; arbitrary objects or dataclasses
cannot silently become exported node identities.

`comparison.resultant_kinematics()` optionally derives a
`SineResultantKinematics` report from that same detached capture. It retains
the source comparison and exact rational phase-rate numerators
`a_i=(w/beta)*nu_i*(Bx)_i/d_i`, so `theta_dot_i=a_i/pi`. The existing
relative-resultant chain rule consumes these rational directions before one
common mathematical-pi division. This preserves exact cancellation of common
rates rather than subtracting independently enclosed phase rates.
Both this reader and `phase_rate_numerators()` re-admit primitive form, phase,
capacity, support and law before calculation. Stored gradients or rate arrays
are not evidence: the shared comparison owner reconstructs the fields consumed
by the calculation. This is detached validation, not graph recapture or source
authentication.

`resultant_rate_bounds` contains real/imaginary derivative intervals in
source node order for each node-relative neighbor resultant, not a normalized
pair or regional phasor. `speed_upper_bounds` encloses
`sum_j abs(theta_dot_j-theta_dot_i)`, an upper bound on the resultant speed;
its lower endpoint is not a lower bound on actual speed. No positive
resultant is required. These are ideal instantaneous comparison rates, not
captured native floating-point rates, observations over time, future speed
bounds or a trajectory certificate. The method performs no graph reread,
native evaluation or mutation and adds no cost to the base report unless
requested. Its `to_dict()` validates labels through the same owner and uses
`tnfr.relational-sine-resultant-kinematics.v1` with the full source comparison.
Projection preserves supplied evidence; it does not authenticate a manually
constructed dataclass.

The observer performs no trajectory, mutation, boundary continuation or
automatic model selection. Preserved native equilibrium shapes do not imply
an identical equilibrium set, recovery rate, basin, memory or frozen response.
The supplied coefficients, initialization and alternative-law provenance
remain explicit even when the two laws agree at consensus.

<a id="localized-identity-transport-budget"></a>
### Private finite localized-transport budget

`_bound_localized_identity_transport` in
[`_localized_identity_transport.py`](../../../src/tnfr/physics/_localized_identity_transport.py)
evaluates the [prepared transport proof's](../../../theory/nodal/LOCALIZED_IDENTITY_TRANSPORT.md)
scalar sufficient bounds. Its supplied model is the actual positive-loss
normalized-sine law. The conservative comparison uses `omega(k)=1-cos(k)`;
the actual loss remains in both the declared model and its error budget.
The preparation already contains a localized Fejer envelope and directional
carrier. Identity compares signed form and real phase lifts jointly against
the complete moving reference, including its internal rotation. The resulting
finite tube is conditional on that preparation.

The reader re-admits the explicitly regular `reference_model` coefficients
through the shared sine owner, requiring positive loss and exchange and unit
storage scale, without renormalizing the supplied mix. `node_count`,
`fejer_order` and `horizon` are ordinary Python integers: `N>=8` is divisible
by four, `Q>=2`, `Q-1<N/4` and
`0<T<=N/2`. `amplitude`, `initial_state_error_bound` and `localization_radius`
use shared exact or represented-real admission before arithmetic, requiring
`a>0`, `epsilon>=0` and `1<R<N/2`. Boolean and nonfinite values reject. Unit
cycle conductances, geometric lengths and held capacities, no input or event,
and the clock `tau=w*t/pi` in both rows are mathematical premises. This
function accepts no graph or observed state with which to verify them.

Exact rational arithmetic and outward intervals retain initial error, actual
loss, conservative dispersion and nonlinear remainder contributions, the
localization tail and endpoint-window separation. The report separates strict
bootstrap, identity tolerance, concentration and window conditions. A failed
sufficient condition yields `not_certified` with `unavailable_reasons`, not
impossibility of transport; failed bootstrap leaves actual-state norm and tail
bounds unavailable. No bound is clipped to manufacture a passing result.
Clock conversions divide exact endpoints before outward materialization.
A zero lower endpoint for an unresolved tiny clock rate does not turn its
retained positive exchange coefficient into a frozen law.

This private calculator authenticates no source, evaluates no trajectory and
installs no native solver or execution law. It is not a graph observer or a
public forecast API. The existing shared sine rows remain the actual field
owner. The result supplies no formation, permanent-motion or physical
identification certificate.

### Private phase-uniform source obstruction

`_bound_phase_flat_directional_source` in the same private module evaluates
the [source obstruction](../../../theory/nodal/LOCALIZED_TRANSPORT_SOURCES.md#phase-flat-source-obstruction).
It shares the cycle, band, horizon and actual positive-loss model admission
above. `source_form_norm_bound=M` and `target_amplitude=a` must be positive;
`initial_state_error_bound=epsilon` must be nonnegative, under shared exact or
represented-real admission. The supplied source is within joint error epsilon
of `(u,0)` for arbitrary real `||u||<=M`. Spatial asymmetry, localization and
nonzero form mean are permitted; no initial source or target is reconstructed.

The actual damped tangent preserves equal norms in opposite carrier bands.
Its distance from every norm-a target confined to either band is at least
`a/sqrt(2)`. This argument requires neither a conservative replacement nor
spatial reflection symmetry. The report evaluates the nonlinear candidate
`delta=epsilon+(32/3)*T*M^3`. Strict `delta<M` closes the bootstrap and admits
the whole-interval distance bound `a/sqrt(2)-delta`, with outward arithmetic.
`status="excluded"` requires that lower bound to exceed the target tube radius
`a/25` strictly. Otherwise the result is `not_certified` with explicit reasons;
failed bootstrap leaves actual distance and exclusion margin unavailable.

This is a sufficient exclusion from a declared source class, not evidence of
formation when the bound fails. It leaves other mobile organizations open and
evaluates no trajectory. The separate
[exact reflection obstruction](../../../theory/nodal/LOCALIZED_TRANSPORT_SOURCES.md#reflection-source-obstruction)
concerns disjoint reflected localization windows under the full law; it is not
a consequence of phase uniformity alone or another output of this calculator.

### Private mobile-pair identity and discriminator budgets

`_bound_localized_mobile_interaction` in
[`_localized_mobile_interaction.py`](../../../src/tnfr/physics/_localized_mobile_interaction.py)
evaluates scalar budgets for the
[joint band-identity proof](../../../theory/nodal/LOCALIZED_MOBILE_INTERACTION.md#mobile-pair-identity)
and its prospective interaction discriminator. It shares the actual positive-loss
model, cycle, amplitude, error, radius and horizon admission of the prepared
transport reader. The horizon must also be even: opposite-band reference
centers start at `+T/2` and `-T/2`, meet at `T/2` and separate by the endpoint `T`.
Each preparation has its own joint form/phase error ball, and the combined
initial state is exactly the sum of those same two preparations. Three
independently perturbed histories do not satisfy this correlation premise.

`prepared_identity_certified` requires the joint full-state reference error and
each projected component's tail to meet their strict tolerances, with separated
endpoint windows. The band
projections retain the declared identity labels through overlap; they do not
close two nonlinear subsystems, partition the actual state spatially or
prescribe exact constituent trajectories.
`cubic_remainder_certified` separately admits the nominal fifth-order remainder
and correlated-preparation sensitivity for the
[cubic Duhamel instrument](../../../theory/nodal/LOCALIZED_MOBILE_INTERACTION.md#mobile-pair-defect).
Its tangent is the actual damped tangent. The joint-minus-separate defect
cancels all linear evolution, including reused errors; intensity cross terms
are not substitutes for that full-state observable.

The helper computes no cubic proxy or paired trajectory. It returns
`interaction_norm_bounds=None`, with no interaction-detection verdict; a
positive lower bound would require separation of an evaluated proxy from its
complete error budget. `status="certified"` records sufficient identity and
cubic-error budgets only. Failed
bootstraps leave the corresponding actual-state and remainder bounds
unavailable. The [frozen endpoint protocol](../../../theory/nodal/LOCALIZED_MOBILE_INTERACTION.md#mobile-pair-protocol)
defines that separate evaluation. These private scalar bounds authenticate no
source, install no solver and establish no resolved deformation or collision.

### Private certified mobile endpoint observation

`_evaluate_localized_mobile_response` in
[`_localized_mobile_response.py`](../../../src/tnfr/physics/_localized_mobile_response.py)
rebuilds those primitive budgets and encloses the nominal cubic endpoint with
the [finite spectral and quadrature method](../../../theory/nodal/LOCALIZED_MOBILE_RESPONSE.md#mobile-response-numerics).
The spectral factory additionally requires `N<=4096`, `Q<=32`,
`3(Q-1)<N/4` and `sigma*T<=1`; it retains both real channels, all cubic
modes and the declared source reflection. These are instrument domains.
The returned closure consumes only freshly admitted time, not a cached
external preparation report.

Gauss order 12 and deterministic prior-bound panel doubling implement the
frozen calculation. Root/weight intervals, source materialization, propagators,
convolution and Parseval reduction retain outward error. A failed numerical
budget cannot certify a positive interaction even when its computed lower
bound is positive. `nominal_interaction_certified` and
`uniform_interaction_certified` separately require the original identity
contract and the corresponding complete error budget. A zero lower bound
remains unresolved sufficiency, not a declaration that the response vanishes.
This calculation evaluates an analytic tangent integral, not the full
nonlinear trajectory and not a physical observation.

The [retained endpoint result](../../../theory/nodal/LOCALIZED_MOBILE_RESPONSE.md#mobile-response-result)
passes the numerical budget and certifies a nonzero complete-flow nominal
defect while retaining both prepared band identities. The original
shared-source-family bound remains unresolved. The
[read-only evidence controls](../../../tests/physics/test_localized_mobile_response_evidence.py)
rebuild budgets and verdicts from primitive inputs and retained spectral boxes
without replaying the producer; byte associations do not authenticate execution.
Nonadditivity establishes no common composite NFR identity or capture. That
claim needs an initially absent actual-state relation and joint retention
beyond linear overlap, with constituent and boundary state retained. A
prospective composite identity may allow shared motion and changing profiles;
it is not supplied by the separated reference windows of this endpoint test.

### Private cycle consensus-basin bound

`_bound_cycle_consensus_basin(*, reference_model, node_count,
initial_state_norm_upper_bound)` in the shared
[`_localized_identity_transport.py`](../../../src/tnfr/physics/_localized_identity_transport.py)
evaluates the [zero-sector corollary](../../../theory/nodal/LOCALIZED_ORGANIZATION_GEOMETRY.md#mobile-zero-sector-basin).
It admits an ordinary integer `N>=3`, a nonnegative exact or represented norm
bound `M`, and the regular positive-loss normalized-sine model with `beta=1`.
The caller supplies the fixed unit cycle, held unit capacities, no inputs or
events, and a joint signed-form/real-phase-lift norm about constant backgrounds.
A storage observation or wrapped edge gaps cannot replace that lift premise.

The reader computes `initial_storage_upper_bound=2*M**2` and
`acute_barrier_margin=1-2*M**2`. A strictly positive margin also guarantees
the initial zero-winding acute chart under the supplied norm premise;
`consensus_basin_certified=True` and `status="certified"` then retain the
all-time cosine lower bound and convergence to the actual conserved form and
lifted-phase means. Otherwise `edge_cosine_lower_bound=None` and
`status="not_certified"` leave the conclusion unavailable with an explicit
reason. This is failed sufficiency, not evidence of a different outcome.

The scalar reader inspects no state, authenticates no source and evaluates no
trajectory. It supplies neither a convergence deadline nor a mobile-template
extension, and excludes no finite-lived organization. The
[geometry owner](../../../theory/nodal/LOCALIZED_ORGANIZATION_GEOMETRY.md#geometry-and-formation)
separates consumed support/conductance and phase geometry from unused embedding
or independent edge lengths, and retains the stationary winding-one control
showing why small storage alone does not certify consensus.

### Sine-law prospective sampling budgets

`bound_sine_sampling_smoothness(*, reference_model, form_diameter_bound,
capacity_ceiling, window_start, window_end)` in
[`relational_sine_sampling.py`](../../../src/tnfr/physics/relational_sine_sampling.py)
derives uniform first, second and third derivative bounds from a declared
class of full sine-law networks. It requires the same explicit regular
reference model, fixed simple unit support without isolates, held
nonnegative capacities and no forcing/events. The supplied diameter is the
full-network form range at `window_start`; the capacity ceiling includes
every hidden node. Neither premise is inferred from the observations.
Common form and phase shifts do not affect these bounds.

The immutable `SineSamplingSmoothness` retains the model, class premises,
window and derived derivative bounds. Its `sample_budget` method accepts
`sample_step`, separate `form_sample_error_bounds` and
`phase_sample_error_bounds`, shared `timestamp_error_bounds` and
`observation_time`. It composes two shared sample-jet budgets and rejects
sampling windows outside the independently bounded window. One-shot ordered
iterables are materialized once. The returned `SineSamplingBudget` retains
both budgets and their smoothness provenance.
Under [chained-report admission](#sine-chained-report-admission), composition
rebuilds derivative bounds from the model, initial diameter, capacity ceiling
and window. The rebuilt smoothness report is retained; a changed window
requires new whole-window bounds.

`phase_increment_margins` bounds the distance from the nearest-increment
ambiguity at pi, including timing and observation error.
`phase_increments_resolved` is true only for two strictly positive margins;
failure means the sufficient condition is unresolved. This certificate
still needs actual circular samples and an initial reference to construct
a lift; it neither unwraps a record nor supplies an absolute phase origin.

The reports' `to_dict()` methods use exact SDK projection with schemas
`tnfr.sine-sampling-smoothness.v1` and `tnfr.sine-sampling-budget.v1`.
These APIs compute prospective bounds only: no trajectory, acquisition,
inverse, law selection or reserved response is performed.

### Retained-state sine environmental pressure

`comparison.mediated_pressure(mediator=...)` derives a
`SineMediatedPressure` through
[`relational_sine_mediation.py`](../../../src/tnfr/physics/relational_sine_mediation.py).
It uses the already captured `SineExchangeComparison` without reading the
graph again. The supplied mediator must exist and have at least two ports.
Primitive state and law are re-admitted and the complete field is rebuilt before
decomposing pressure, rates, work or memory coefficients. A valid unchanged
capture may retain its original object association only after the calculation
uses normalized rebuilt values; modified cached arrays cannot change the result.
Its actual form, phase and capacity remain in the source comparison; no
minimum replacement or node removal occurs. Zero hidden capacity and zero
form-dissipation coefficient are admitted, as are zero or negative-real
resultants. The law remains `normalized_sine_reciprocal_exchange`.

`ports` follows the captured support order and `port_degrees` retains full
original incidence. `environmental_pressures` and `internal_pressures`
split the captured ports' full pressure; `environmental_phase_rates` and
`internal_phase_rates` split their phase rates. Pressure contributions
require port capacity to obtain form rates. Computed reconstruction residuals
retain interval uncertainty rather than being assigned zero.

The hidden rows, form contrast and its rate, port mean form/rate and hidden
phase acceleration expose the same two-coordinate realization. The exact
rational `memory_decay=mu*e` and enclosed `phase_gain=mu*w/(beta*pi)` and
`memory_phase_gain=mu^2*w^2/(beta*pi^2*k)` belong to the
[causal memory identity](../../../theory/nodal/SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure).
They are derived coefficients, not fitted kernels. The capture can serve as
initial state only at a separately declared time origin. The report does not
evaluate history integrals, approximate a delay or authenticate initial data.

`incident_form_storage` and the unscaled `incident_phase_cost` supply
`incident_storage` with the reference beta. `boundary_form_work` and
`boundary_phase_work` use full port rates, including visible internal
neighbors. `incident_storage_rate` is computed by differentiating incident
edges; `balance_residual` compares that rate with `boundary_work-hidden_loss`.
The exact `hidden_loss=mu*e*k*(x_h-X)^2` covers the hidden node only.
Splitting pressures does not split visible losses into separately squared
gradients or imply that incident storage decreases.

The optional `port_boundary_form_work`, `port_boundary_phase_work` and
`port_boundary_work` arrays retain these contributions in `ports` order and
own their corresponding aggregate sums. Their orientation is **into the
incident star**. These fields are instantaneous work rates, not accumulated
work. For a region whose only cut edge joins this mediator to one port, its
incoming work rate is the negative of that port's contribution; other cuts
must also be retained when present. See the
[receiver passage balance](../../../theory/research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-receiver-port-passage).
Legacy manually constructed records may leave the arrays `None`; missing
decompositions are not zero work.

`to_dict()` uses the shared exact projection and validates source, mediator
and port labels before exporting schema
`tnfr.relational-sine-mediated-pressure.v1`. It can be passed to the SDK's
atomic `export_to_json`. A manually constructed report is not authenticated
by projection. This opt-in derivation adds no solver, native dispatch,
stationary elimination, support event or new constitutive choice.

### Hidden-state inference from prior visible rates

`infer_relational_sine_hidden_state(visible_graph, *, ports,
form_rate_bounds, phase_rate_bounds, reference_model, source_id, clock_id,
observation_time, evidence_window, forecast_start)` in
[`relational_sine_observation.py`](../../../src/tnfr/physics/relational_sine_observation.py)
implements the
[conditional observability admission](../../../theory/nodal/SINE_ENVIRONMENTAL_MEMORY.md#sine-hidden-state-observability).
It receives a visible-only graph, never a full-state comparison or hidden
coordinates. `ports` declares at least two distinct visible nodes coupled to
one absent intermediary. Every visible component must touch a port. Shared
relational admission retains simple unit support, authoritative scalar
aliases, finite phase, nonnegative capacity and absent Gamma; only the
visible graph's connectivity requirement differs from full execution.
Its original port degrees are visible degrees plus one.

Both rate mappings must contain exactly the declared ports and supply
ordered finite endpoint pairs through the existing interval reader.
Visible state and model parameters are exact captured represented inputs;
rate intervals must include all claimed measurement, derivative and clock
error. This API does not propagate uncertainty in the visible state,
support or coefficients. Zero-capacity rates must enclose zero, and those
ports contribute no observation row. Missing rates are not zero observations.
The regular reference model identifies supplied coefficients; native
resultant admission is not applied to this separate sine law.

`SineHiddenStateInference` retains source and clock identifiers, supplied
rate bounds, visible support/state, original port degrees, per-port form
and phase-projection bounds, selected pair, rank evidence and computed
residuals. `hidden_form_bounds` intersects active phase-row reconstructions.
The combined form/phase-rate formula removes the diffusive contribution
before phase inversion. `hidden_unit_phase_relative_to_anchor_bounds`,
when available, encloses cosine/sine relative to `phase_anchor`. It is not
normalized onto the unit circle or materialized as a branch angle.

`phase_gram_determinant_bounds` encloses the sum of squared pairwise phase
sines over active observation rows. It is a derived conditioning diagnostic,
not the ordinary resultant, a hidden-state existence certificate or a
dynamic control signal. A selected pair must have its sine determinant
certified away from zero. An unresolved determinant does not prove rank
one; exact equal captured phases or a single active row can establish that
rank. This implementation abstains from rank-one phase inversion, including
exceptional exact unique boundary cases covered by the mathematical owner.

The statuses have deliberately different meanings:

- `bounded_candidate`: a certified pair bounds every hidden state consistent
  with the declared data; the necessary consistency checks do not exclude
  it. Existence, unique realizability with uncertain inputs and provenance
  authentication are not established.
- `inconsistent`: a necessary form, zero-capacity, phase-projection, circle
  or additional-row condition is excluded by the supplied intervals.
- `unavailable`: sufficient active/rank information is missing or unresolved.
  Form information may still be available separately.

`observation_time` lies within the nonnegative `evidence_window`, whose end
must be strictly earlier than `forecast_start`. These are exact/represented
time inputs; a point window is allowed for declared instantaneous evidence.
The inferred state belongs to `observation_time`, not automatically to the
window end or future forecast start. A rate extracted with the existing
forward three-sample observer belongs to its first sample. Input metadata
does not authenticate an acquisition or certify its derivative error budget.

Hidden capacity is absent from the instantaneous inverse equations;
`hidden_capacity_identified` remains false. A future response additionally
requires that capacity or a justified identification of it. The report
executes no trajectory and supplies no physical sensor map.
`to_dict()` validates node/edge/port labels and uses shared exact projection
with schema `tnfr.relational-sine-hidden-state.v1`, suitable for the existing
atomic SDK JSON writer.

### Hidden-capacity inference from prior accelerations

`state_inference.infer_capacity(*, form_acceleration_bounds,
phase_acceleration_bounds, source_id, clock_id, observation_time,
evidence_window)` reuses the same
[sine observation owner](../../../src/tnfr/physics/relational_sine_observation.py).
It reconstructs the retained first-rate inference under
[chained-report admission](#sine-chained-report-admission).
Each acceleration mapping may contain a subset of the state's ports; at
least one channel is required. Missing channels are absent, not zero.
The clock identifier and observation time must equal those of the retained
state report. The new evidence window contains that time and ends strictly
before the inherited forecast start. Both windows and their combined
extent remain in the result; metadata does not authenticate an acquisition.

`SineHiddenCapacityInference` retains its `state_inference` without graph
recapture. `visible_form_rate_bounds` and `visible_phase_rate_bounds`
use supplied first-rate observations at ports and rates derived from the
known sine law at visible nonports. `visible_rate_provenance` distinguishes
these sources. Nonports have no hidden incidence; no missing port
observation is replaced by a computed rate. Original degrees and every
structural port, including zero-capacity ports, enter the hidden response.

The [acceleration identity](../../../theory/nodal/SINE_ENVIRONMENTAL_MEMORY.md#sine-hidden-capacity-observability)
supplies hidden form and phase responses per unit capacity and affine
acceleration channels. Each `SineCapacityChannel` retains its port,
kind (`phase_acceleration`, `form_acceleration` or
`exchange_acceleration`), observed bounds, baseline and capacity
sensitivity. When both accelerations are supplied, the exchange combination
cancels the form-gradient derivative directly. This derived channel shares
its input uncertainty; it is not an independent measurement.

A channel's `capacity_bounds` stores its raw quotient only when sensitivity
is separated from zero. The report intersects all such quotients with
nonnegative capacity and checks all observed channels against the common
result. Raw negative quotients remain visible as evidence rather than
being silently clamped to a valid estimate. Computed residuals are not set
to zero. Zero-containing sensitivity does not certify exact blindness;
the exact reflected counterexample belongs to the mathematical owner.

Known form and phase-projection bounds can sometimes bound capacity while
the full phase remains unresolved. Unknown sine/cosine projections retain
their proved `[-1,1]` ranges, and `hidden_phase_available` stays false.
No circular direction is invented. Missing form prevents this inference;
inconsistent source evidence remains inconsistent.

`bounded_candidate` means a conditional capacity enclosure under all
declared premises, not an existence or exact-identification certificate.
`inconsistent` marks an excluded necessary condition; `unavailable` marks
insufficient resolved information. A bounded capacity alone does not
provide the phase or propagated state required for a future prediction.
There is no capacity evolution law, numerical differentiation, trajectory
or solver in this method.

`to_dict()` validates retained source and channel labels and uses shared
exact projection with schema `tnfr.relational-sine-hidden-capacity.v1`.
The public dataclasses and their metadata do not authenticate prior
observations or manually constructed reports.

### Joint prior admission and sine forecasts

The inverse and forecast stages use the
[chained-report admission contract](#sine-chained-report-admission).

`physics.relational_sine_forecast.admit_sine_prior(capacity_inference)`
provides a separate joint-admission boundary after the two inverse reports.
It requires a bounded hidden form/capacity and a hidden relative phase
rectangle with strictly positive cosine lower bound. Shared interval
`arg` lifts that rectangle relative to its retained visible anchor.
Equivalent phases differing by `2*pi` use this same chart; no midpoint is
normalized into an authoritative phase.

One candidate is chosen by rounding the inferred hidden form, lifted angle
and capacity to the fixed `2^-20` rational grid. Its coordinates and exact
unit-phase functions must be contained in the prior enclosures. Shared
full-field jets must then enclose every candidate first/second derivative
strictly within or on the corresponding supplied observation interval.
This containment certifies a joint witness under the declared model and
evidence. Overlapping separate intervals or residuals containing zero would
not suffice. Failure is `unavailable`, not proof of nonexistence.

`SinePriorAdmission` retains the validated capacity/state reports, incidence,
complete initial box, phase lift, witness and its forward derivative bounds.
The witness proves nonemptiness; it does not replace the initial uncertainty
box. The observation source and error bounds remain supplied premises.

`forecast_sine_prior(admission, *, end_time, time_step, order=6,
freeze_hidden=False)` propagates that complete box from the actual prior
observation time, including the gap before `forecast_start`. The coordinate
order is visible forms, hidden form, visible phase lifts, hidden phase lift,
and hidden capacity. Its final row is `mu'=0`, so capacity uncertainty enters
the shared Jacobian and propagated radii. `freeze_hidden=True` changes that
initial capacity to zero as an explicit counterfactual intervention; it
does not fit the original acceleration observations.

`bound_sine_flow` is the supplied-full-state adapter for this same solver.
It requires an explicit regular reference model, connected simple undirected
unit incidence, exact nonnegative held capacities and at most eleven nodes.
Its final node is the supplied hidden node. Time is exact rational, with a
fixed step budget of at most 256 and Taylor order from 1 through 16.
The `2*n+1` layout shares `MAX_COMPARISON_DIMENSION=24` with the Picard/Taylor
and comparison-flow kernels. This is a numerical work limit; it does not
guarantee that inclusion, uncertainty or remainder bounds resolve for an
admitted dimension. Frozen protocols retain their declared original budgets.
The sine law itself is globally smooth in real phase lifts; it does not
inherit the native Arg law's excluded resultants.

Both paths reuse `_sine_rates` and `mathematics/_validated_taylor.py`.
`dynamics/integrators.py` was inspected: its stored-pressure quadrature
does not evolve this coupled refreshed full field. It is therefore not
silently substituted for the existing validated full-state integration.
Every step retains strict whole-time Picard inclusion, comparison-propagated
initial radii, Taylor remainder, endpoint and time. Known held coordinates
are intersected with their exact initial enclosures after each step.
Unresolved inclusion retains the partial chain and failed tube without
adaptive retries.

`to_dict()` uses the existing exact SDK projection with schemas
`tnfr.relational-sine-prior-admission.v1` and
`tnfr.relational-sine-forecast.v1`. The
[prospective protocol](../../../theory/nodal/SINE_ENVIRONMENTAL_MEMORY.md#sine-prior-reserved-forecast)
separates prior derivatives, joint inference, issued prediction and reserved
source response. Its frozen-mediator control is a counterfactual, and its
synthetic derivative evidence is not a sampled laboratory measurement.

### Detached stationary sine mediation

`bound_relational_sine_mediation(graph, mediator=..., reference_model=...)`
in [`relational_sine_mediation.py`](../../../src/tnfr/physics/relational_sine_mediation.py)
uses the comparison's shared admission and rate/work kernels. The supplied
node must have at least two distinct ports and strictly positive capacity.
Its reference model supplies coefficients and storage scale as above.
Zero visible capacities are admitted. The complete source state is validated
and retained, including the hidden coordinates that the hypothetical
minimum replaces.

The observer requires a certified nonzero port resultant. It evaluates the
conditional hidden form mean and circular direction `Z/|Z|` without turning
an approximate angle into an authoritative phase. A zero-containing magnitude
enclosure raises `ValueError`; this is failed reduction admission and does
not imply that the full smooth field is undefined. Neither native resultant
admission nor the existence of only one stationary hidden branch is claimed.
Magnitude bounds use the proved triangle bound `R<=k`, and normalized
phasor components are intersected with their proved range `[-1,1]`.
This reduces interval dependency inflation near cancellation without
assigning an angle, imposing a dynamical threshold or declaring an exact zero.

`SineMediationComparison` retains source node/edge order, visible coordinates,
ports, the minimum's form, complex direction relative to the first port,
resultant magnitude and positive phase curvature. Visible `degrees` retain
the mediator incidence even though `edges` lists only visible internal edges;
the report is not a reconstructed ordinary graph. Its visible gradients,
pressure, rates, inherited storage, exact loss and computed work residuals
belong to the conditional reduced equation.

`hidden_form_tracking_defect` and `hidden_phase_tracking_defect` enclose
the negative velocities of the moving conditional minimum when the hidden
rows vanish there. Excluding zero proves instantaneous non-tangency. Enclosing
zero alone proves neither tangency nor future invariance. These defects do
not depend on the positive hidden capacity; actual relaxation time does.

The law is explicitly
`stationary_minimum_reduction_of_normalized_sine_exchange`. `to_dict()`
uses shared exact projection with schema `tnfr.relational-sine-mediation.v1`
and can be passed to the SDK's atomic `export_to_json`. It is a detached
comparison whose source/visible/port/mediator and edge labels are validated
before generic dataclass projection. It is not a
checkpoint, live elimination, ODE solution, fast-limit
certificate or new primitive law. The
[mathematical owner](../../../theory/nodal/SINE_ENVIRONMENTAL_MEMORY.md#sine-mediator-common-geometry)
separates conditional storage, finite-capacity memory, the hidden oscillator
and the hypotheses of a controlled approximation.
The [finite-state audit](../../../theory/nodal/SINE_ENVIRONMENTAL_MEMORY.md#finite-environment-reuse-audit)
explains why an unavailable stationary direction need not invalidate the
uneliminated sine field, and why the default hidden tangent does not oscillate.
The [autonomous path proof](../../../theory/nodal/SINE_ENVIRONMENTAL_MEMORY.md#autonomous-path-cancellation)
also gives a finite cancellation crossing with all three nodes active:
zero tracking defect before the crossing and any finite hidden capacity do
not guarantee that the hidden state continues to minimize storage afterward.
This is a conditional continuous theorem, not a result inferred from the
stationary report or an executed trajectory.
