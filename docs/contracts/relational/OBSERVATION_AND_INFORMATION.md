# Shared observations, information and finite-sample bounds

Regional and source-relative observations, exact linear and phase-moment information, finite-sample evidence and shared circular geometry.

Part of [Relational dynamics contract index](../RELATIONAL_DYNAMICS.md). Section links remain stable; hypotheses and model changes remain local to each result.

### Exact linear observations of a supplied generator

`tnfr.mathematics.linear_observation.derive_linear_observation(J, O, *,
max_rank_calls=4096)` returns an immutable minimal row-space realization for
`z'=Jz`, `y=Oz`. It admits square finite exact/represented real `J` and ordered
nonempty output rows, including dependent and zero rows. Rational values
remain exact; floating values define their represented rational model.
Its `observation`, `right_inverse`, `reduced_generator` and `output_map`
verify `CT=I`, `CJ=GC`, `O=DC`, with explicit rank progression and a strict
rank-call budget. A zero output has no state; budget exhaustion returns no
partial certificate. This is not graph, stability, nonlinear-law or exact
transcendental admission. The old affine diffusion API retains its independent
model checks and negative generator sign; see the
[composition scope](../../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md).

`derive_coordinate_memory(generator, visible_indices)` in the same module
derives the exact coordinate split `y_dot=A*y+B*h`, `h_dot=C*y+D*h` for a
supplied fixed generator with sign `z_dot=J*z`. Visible indices must be an
ordered, nonempty proper subset of distinct nonboolean integers; hidden
indices retain the original complementary order. Scalar admission and
detached immutable tuples follow the same exact/represented-real boundary.
The report retains `generator`, both index tuples, `visible_generator`,
`hidden_to_visible`, `visible_to_hidden`, `hidden_generator` and
`kernel_at_zero=B*C`. The corresponding memory is `B*exp(D*t)*C`, with the
separate initial-state source `B*exp(D*t)*h(0)`. No exponential sampler,
stability/positivity assertion, graph authentication or nonlinear closure is
provided. Zero `kernel_at_zero` does not imply zero memory at later lags.
See the [joint mediator derivation](../../../theory/nodal/RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction)
for an admitted use with the relational tangent; it does not replace full
`Network.step_relational` evolution with a fitted memory law.

<a id="form-loss-observation-bound"></a>

### Conservative observation versus form-only loss

`tnfr.mathematics.linear_observation.bound_form_loss_observation_error(J, O, *,
damping, exchange, horizon)` returns an immutable `FormLossObservationBound`.
It admits an even-dimensional exact/represented real skew generator, ordered
by equally sized form and phase blocks, commuting with `R_n=((0,-I),(I,0))`.
Exactly two ordered orthonormal output rows must satisfy `O R_n=R_1 O`.
The preparation is fixed to `O^T`; a different or correlated hidden lift is
not supplied implicitly. Invalid hypotheses raise rather than producing a
zero or passing bound.

The nonnegative `damping=gamma`, positive `exchange=omega` and nonnegative
`horizon=T` declare the target `G=((-gamma,-omega),(omega,0))` in the same
clock and energy chart as `J`. The report retains all primitive matrices and
coefficients, `preparation`, `target_generator`, `witness_time` and
`uniform_error_lower_bound`. The latter bounds the supremum Euclidean response
error over the unit visible ball and the **whole interval** `[0,T]`, with
zero hidden preparation, or any independent hidden family containing that
subset. The witness time can be earlier than `T`. Zero damping/horizon gives
zero, which does not certify equality or an accurate approximation.

The [proof and exact bound](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#effective-loss-covariance-bound)
use rational arithmetic without an exponential evaluation. Exclusion requires
the bound to exceed the independently declared tolerance strictly. This
detached reader authenticates the supplied matrix identities, not a TNFR
graph, clock conversion, ideal irrational energy chart, effective coefficient,
nonlinear approximation or physical identification. It installs no loss law.
The scalar and matrix admission uses the existing exact/represented owner;
booleans, nonfinite values and nonzero values lost during materialization
reject before the calculation.

`bound_orthogonal_form_loss_observation_error(J, O, *, damping, exchange,
horizon)` shares the same scalar, skew-generator, orthonormal-output and
transpose-preparation admission, but **does not require channel covariance**.
Its `method="orthogonal_initial_response"` retains the exact row-sum bound
`generator_norm_upper_bound=N`. With `M=gamma+omega`, it evaluates
`gamma*s-(N^2+M^2)*s^2/2` at `s=min(T,gamma/(N^2+M^2))`. This follows from
the skew initial observed derivative and contractive Taylor remainders;
see the [geometry-independent obstruction](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#structural-channel-distinction).
The original reader keeps `method="quadrature_covariance"` and its stronger
symmetry-specific bound, with no required generator norm. Both methods
retain their full-response, zero-hidden-subset and conditional scopes.
Failure of the original symmetry admission is not evidence of effective loss.

<a id="phase-moment-information"></a>

### Exact first/third moment information comparison

`assess_phase_moment_information(left_phasors, right_phasors, *, epsilon=1)`
in [phase_response.py](../../../src/tnfr/physics/phase_response.py) returns a
detached `PhaseMomentInformationAssessment`. It compares the sine source and
cosine potential with the already declared cubic-sine alternative. The
[information theorem](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency)
owns the conditional classification and its independent assumptions; this
reader checks two supplied finite multisets rather than proving sufficiency
for an arbitrary source law.

Both inputs must be ordered, nonempty collections of equal size `d<=128`.
Each entry is a relative-gap phasor `(c,s)` satisfying **exactly**
`c*c+s*s==1` after shared scalar admission. Repeated entries are retained as
distinct incidences. Use rational `Fraction` coefficients for exact
nontrivial unit phasors. Floating inputs denote their exact represented
coefficients and are accepted only if those coefficients satisfy the identity;
rounded trigonometric pairs are not normalized or accepted with a tolerance.
Boolean, text and nonfinite scalars reject; nonrational materialization must
not lose a nonzero input to underflow. Exact rational values need not fit
binary64. Mappings, sets, malformed pairs, unequal degrees and oversized
collections reject. The
128-entry bound is a computational budget, not a physical degree limit.

`epsilon` is an admitted finite nonnegative coefficient of the supplied
countermodel; zero is the equal-law control. It installs no runtime pressure
or companion phase row. The reader neither receives nor mutates a graph,
reconstructs phase angles, nor authenticates measured phase preparation.
Zero resultants and nonacute phasors remain valid inputs to these algebraic
sources; `nonzero_first_resultants` and `strictly_acute` report those
properties separately without assigning a native resultant direction.

`first_resultants` and `third_resultants` retain exact sums of the first and
third complex phasor powers, in left/right order. Exact equality of the first
moments sets `first_resultants_equal`; overlap of numerical intervals cannot
substitute for it. Each `*_source_pi_numerators` field stores **pi times**
the corresponding normalized root source: `sum(s)/d` for sine and
`sum(s+epsilon*s**3)/d` for the cubic alternative. These do not include a
capacity or a separate evolution-law coefficient.

All signed differences are **right minus left**.
`cubic_source_difference_pi_numerator` remains rational, while
`cubic_source_difference_bounds` encloses its division by mathematical pi.
The exact obstruction can remain true even when this enclosure contains zero.
`cosine_storage_sums` and `cubic_storage_sums` retain unscaled potential sums
over the declared root incidences; they are not whole-graph energy or its
time derivative. Their corresponding difference fields follow the same sign
convention. `sufficiency_obstruction` is true precisely when the equal-degree
pair has exactly matching first moments and unequal cubic sources. A false
flag is not a universal sufficiency certificate, and matching first moments
does not assert equality of full state, storage or future evolution.

Direct `to_dict()` uses schema `tnfr.phase-moment-information.v1`. The generic
SDK report envelope and atomic JSON writer also retain exact fractions and
scope; export is not provenance authentication. See the
[standalone guide](../../guides/relational/OBSERVATION_AND_INFORMATION.md#compare-phase-moment-information)
and [admission controls](../../../tests/physics/test_phase_moment_information.py).

<a id="phase-information-response"></a>

### Finite equal-information star response

`assess_phase_information_response(left_phasors, right_phasors, *, duration,
epsilon=1, preparation_error=0, observation_error=0)` in
`physics.phase_response` returns `PhaseInformationResponse`.
It rebuilds the preceding exact source comparison from primitive phasors.
Both lists must have the same degree and exactly equal first resultants.
The positive `epsilon` declares the existing cubic current and its own
potential; zero is not admitted as a distinct alternative.

The report predicts two endpoint root-form differences on fixed simple unit
stars, with zero ideal forms, ideal root phase zero, unit held capacity,
`e=0,w=beta=1`, no input/events and common clock `tau=t/pi`. Both the form
and phase rows evolve. Model-indexed tuples put sine first and the supplied
cubic member second; preparation-indexed rate pairs put left first.
The observable is right root form minus left root form at `duration`.

- `duration` is strictly positive; both error bounds are nonnegative.
  All scalar arguments use shared exact/represented-real admission.
- `preparation_error` bounds each initial form and **lifted radian** phase
  coordinate around its ideal preparation, independently at every node and
  in each trial. It is not a cosine/sine coefficient error. Numerical angle
  conversion must be included in this budget.
- `observation_error` bounds each endpoint root reading independently.
  It contributes twice that value to a difference.
- Capacity, law coefficients, support and clock are held exact. The error
  budgets do not admit uncertainty in those premises or laboratory units.

Exact centers and error components are retained separately. The
[full nonlinear estimate](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#finite-phase-information-response)
includes future leaf motion and the initial root error; it never reconstructs
pressure from a response. Prediction intervals are materialized outwards.
`discrimination_certified=True` and `status="certified"` mean those intervals
are strictly separated, in either direction. Otherwise the status is
`"unavailable"` with `finite_response_intervals_not_separated`; overlap does
not establish physical equivalence or rejection. Equal initial sine sources
do not imply equal future sine readings.

The direct schema is `tnfr.phase-information-response.v1`; generic SDK export
uses `tnfr.relational-report.v1` with report type `PhaseInformationResponse`.
This is a detached conditional forecast, not a graph step, installed law,
physical measurement bridge or uniquely selected constitutive law. See the
[usage](../../guides/relational/OBSERVATION_AND_INFORMATION.md#phase-information-response) and
[independent controls](../../../tests/physics/test_phase_information_response.py).

<a id="phase-moment-motion"></a>

### Motion retained by paired phase and rate information

`derive_phase_moment_motion(phasors, gap_rates, *, epsilon=1)` in the same
phase-response owner returns `PhaseMomentMotion`. It shares exact unit-phasor
admission and first/third moment and source/storage calculations with the
static comparison. `gap_rates` is an ordered list of the same length, containing
each gap's own signed angular velocity in radians per declared time unit.
Boolean, nonfinite, unordered or incorrectly sized rates reject. No phase
angle is reconstructed and no measured phasor is renormalized.

For `Z_m=sum(exp(i*m*delta))` and the derived rate-weighted moment
`M_m=sum(delta_dot*exp(i*m*delta))`, the reader exposes `Z_m_dot=i*m*M_m`
for `m=1,3`. It reports both source numerators **pi*p** and their derivatives
**pi*p_dot**, with the same `p=sum(j)/(pi*d)` normalization as above.
`cosine_storage_rate` and `cubic_storage_rate` are the incident potential work
`sum(j*delta_dot)`, without degree normalization. They are not whole-network
energy rates; capacity and other law coefficients have not been added.
Incidence and `epsilon` are fixed on the differentiated segment.

The rates must come from a separately justified complete law or kinematic
source, in a common declared clock. In the unit conservative star example,
rates derived from `theta'=K*L*x` use `tau=t/pi`; only under that complete law
does the reported unscaled source derivative equal the root's acceleration
in `tau`. A sampled secant is not automatically an admitted instantaneous
rate. No error-free observation or closed future state is inferred here.

The [proof and paired-state counterexample](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#phase-motion-information)
show why the same phase distribution and the same rate distribution can give
different current derivatives when their association changes. These are
derived observations of existing state, not new free parameters, a pulse law
or an autonomous reduced model. The existing graph-based
`SineExchangeComparison.resultant_kinematics()` instead derives the first
resultant rate from that comparison's complete phase row.

Direct export uses `tnfr.phase-moment-motion.v1`; generic SDK export retains
exact fractions, association and scope. See the
[example](../../guides/relational/OBSERVATION_AND_INFORMATION.md#phase-moment-motion) and
[independent controls](../../../tests/physics/test_phase_moment_motion.py).

<a id="sine-star-moment-chart"></a>

### Exact phase/rate coordinates of the conservative star

`derive_sine_star_moment_closure(first_resultant, first_rate_weighted_resultant)`
in the shared phase-response owner returns `SineStarMomentClosure`. Each input
is one ordered `(real, imaginary)` pair, admitted through the shared finite
represented-real reader; booleans, nonfinite values, mappings, sets and other
lengths reject. No graph or report cache is consumed.

The complete model is the three-node star, fixed unit support and held unit
capacities, conservative normalized sine, `e=0,w=beta=1`, no inputs or events.
`M=sum(delta_j'*exp(i*delta_j))` and all returned rates use **tau=t/pi**;
passing rates from another clock without transformation describes a different
input. The consumer must establish these model and observation premises.
The function does not authenticate them from two complex numbers.

Require **0<|Z|^2<4** exactly. This domain admits a real fine lift for every
complex `M`; a lift may have irrational phasors. It is not the exact rational
unit-phasor input domain of `derive_phase_moment_motion`. The chart returns
relative mean form, squared internal form difference, the leaf phase product,
both moment derivatives and full star storage, all as exact fractions. It
evaluates the inherited vector field, without integrating or changing a graph.

The [proof](../../../theory/nodal/SINE_PAIR_STATE.md#sine-star-moment-chart)
reuses the existing unordered pair state. Four real coordinates remain;
only common origins and simultaneous leaf labels are removed. Singular
coincident and antipodal observations reject rather than filling missing
state with zero. The fine law is still regular there, and its full state or
a separately admitted representation is needed for continuation. Near the
boundaries exact arithmetic does not supply measurement-error control or
an all-time chart guarantee. Other support, capacities and laws need their
own pushforward; closure does not select this law as fundamental.

Direct export uses `tnfr.sine-star-moment-closure.v1`; the generic SDK envelope
retains exact fractions and scope. See the
[standalone use](../../guides/relational/OBSERVATION_AND_INFORMATION.md#sine-star-moment-chart)
and [full-law and boundary controls](../../../tests/physics/test_sine_star_moment_closure.py).

<a id="relational-coefficient-jet"></a>

### Coefficient bounds from a declared initial response jet

`bound_relational_coefficient_from_jet(*, form_bounds, rate_bounds,
acceleration_bounds)` in the [observation owner](../../../src/tnfr/physics/relational_observations.py)
encloses `chi=m1^2/[pi^2*(m1^2-m0*m2)]`. Its
[preparation theorem](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#prepared-coefficient-identification)
requires the specified unforced two-channel law, held common positive capacity,
a nonzero observed spatial mode and initial phase consensus. The function does
not inspect or authenticate these conditions, estimate derivatives from samples,
read a graph or change state. It does not take a supplied beta as its answer.

Each argument is an ordered pair of finite real interval endpoints, lower then
upper. Shared admission preserves exact rational values and otherwise retains
represented real values; Booleans, text, nonfinite values, reversed endpoints,
unordered containers and wrong lengths reject. One-shot iterables are accepted.
The existing 128-bit outward dyadic interval owner encloses all arithmetic and
mathematical pi. Very small exact intervals can widen across zero on this grid;
that loss of resolution produces unavailability, not a zero-signal certificate.

The frozen `RelationalCoefficientJetBounds` retains:

- `form_bounds`, `rate_bounds`, `acceleration_bounds`: admitted, outward-rounded
  jet intervals in one observation's value/time units;
- `squared_rate_bounds` and `restoring_gap_bounds`: the intervals for
  `m1^2` and `m1^2-m0*m2`, with squared rate units;
- `coefficient_bounds`: an enclosing endpoint pair, or `None`;
- `unavailable_reasons`, `arithmetic_method` and explicit conditional `scope`.

No coefficient is returned if the form interval contains zero, the form/rate
product is strictly positive (incompatible with the prepared `e>=0` decay), or
the restoring gap is not proved positive. Reasons distinguish
`initial_form_not_separated_from_zero`, `incompatible_initial_decay`,
`nonpositive_restoring_gap` and `unresolved_restoring_gap`. Multiple reasons can
coexist. A zero-rate interval with negative restoring acceleration can identify
chi zero; critical repeated poles do not make this jet formula singular.

An available interval is not an automatic precision pass or proof that the
model fits the data. The caller must supply uncertainty that actually encloses
the initial derivatives, baseline/gain and clock errors, and declare a useful
width threshold separately. The
[three-sample bounds](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-jet-uncertainty)
require an independent third-derivative bound on the whole sampling interval;
neither three observations nor a solver work residual establishes it.
Misprepared phase or unresolved modal mixtures can yield a plausible but
inapplicable coefficient even when arithmetic succeeds.

For example, exact declared intervals `(1,1)`, `(-3,-3)`, `(7,7)` give restoring
gap 2 and enclose `9/(2*pi^2)`. These numbers illustrate the arithmetic, not a
physical acquisition. `relational_report_to_dict` exports the report's bounds
as exact fraction records and preserves `None` and all reasons. The usual
atomic SDK writer can save that projection; it authenticates no preparation
or calibration. [Routine controls](../../../tests/test_relational_coefficient_identification.py)
exercise input rejection, resolution boundaries, gain/clock covariance and export.

<a id="relational-coefficient-samples"></a>

### Coefficient bounds from three uniformly timed samples

`bound_relational_coefficient_from_samples(samples, *, sample_step,
sample_error_bound, third_derivative_bound)` in the same observation owner
implements the [three-sample theorem](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-jet-uncertainty).
Supply exactly three ordered finite real values at relative times `0,h,2h`,
positive h, nonnegative uniform sample error epsilon and an independently
justified whole-window noiseless C3 bound M3. Shared exact/represented-real
admission rejects Booleans, text, nonfinite values, unordered containers,
wrong sample counts and invalid bound signs before estimation.

The immutable `RelationalCoefficientSampleBounds` retains `samples`,
`sample_step`, `sample_error_bound`, `third_derivative_bound`, exact rational
`rate_estimate` and `acceleration_estimate`, and their `rate_error_bound`
and `acceleration_error_bound`. It delegates the enclosing initial-value and
derivative intervals to `bound_relational_coefficient_from_jet` and retains
the full result as `jet`. Mathematical-pi and final outward-rounding provenance
remain in that nested report; exact rational stencils add no hidden float step.
The shared exporter supports this report, including nested unavailability.

This helper infers neither timing accuracy nor C3 regularity/noise from three
samples. It does not read a graph, fit beta, authenticate preparation or choose
a precision threshold. Known-source computational use is demonstrated by the
[P2 acquisition](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-temporal-acquisition),
whose independent nonlinear and numerical bounds are not general guarantees
for arbitrary samples, graphs or physical measurements. The API changes no
evolution law and adds no `Network` method for this graph-independent arithmetic.

### Normalized contrast between two declared rate intervals

<a id="relational-rate-contrast"></a>

`bound_relational_rate_contrast(*, before_bounds, after_bounds)` in the
[observation owner](../../../src/tnfr/physics/relational_observations.py) bounds
the rate change and `Q=after/before-1`. It consumes two declared intervals,
not a graph or a pair of live states. It installs no law or intervention.
The [K3 protocol](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#normalized-capacity-discriminator)
uses it for a relative-phase rate, but the arithmetic does not authenticate
that preparation or identify which candidate law generated the rates.

Both inputs use the coefficient observer's shared admission: exactly two
ordered finite real endpoints, preserving exact or represented values,
rejecting Booleans, text, unordered containers and reversed endpoints.
The same outward dyadic interval owner supplies arithmetic. The immutable
`RelationalRateContrastBounds` retains `before_bounds`, `after_bounds`,
`change_bounds`, `normalized_change_bounds`, `unavailable_reasons`,
`arithmetic_method` and explicit `scope`. The change retains rate units;
the normalized change is dimensionless. Positive or negative resolved
baselines are allowed. A baseline interval containing zero returns `None`
for the normalized change and `baseline_rate_not_separated_from_zero`;
the undivided change remains available. Lost grid resolution has the same
abstention behavior, not an exact-zero interpretation.

The quotient uses `after/before-1`, avoiding the unnecessary dependency loss
in `(after-before)/before`. Identical nondegenerate intervals nevertheless
do not prove identical rates. The result preserves uncertainty rather than
assuming correlated errors. A common nonzero gain and positive constant
clock conversion cancel in the ideal quotient; different gains/clocks or
additive rate offsets do not. Error and arithmetic budgets remain caller
obligations; extreme scales can widen the fixed-grid enclosure materially.

`relational_report_to_dict` and the shared atomic SDK writer retain exact
fraction endpoints and explicit abstention. There is no automatic precision
verdict or model classification, no derivative estimator in this helper,
and no `Network` wrapper for graph-independent arithmetic. The
[routine controls](../../../tests/test_relational_rate_contrast.py) check
admission, corner containment, signs/scales, dependency and export.

### Rate bounds from three samples, independently of coefficient identification

<a id="relational-rate-samples"></a>

`bound_relational_rate_from_samples(samples, *, sample_step,
sample_error_bound, third_derivative_bound)` supplies the shared three-sample
admission and forward initial-rate stencil. It requires exactly three ordered
finite samples at `0,h,2h`, `h>0`, a nonnegative uniform sample error epsilon,
and an independently justified whole-window bound `|y'''|<=M3`. The initial
rate estimate is `(-3*y0+4*y1-y2)/(2*h)` with error
`4*epsilon/h+M3*h^2/3`. Admission preserves exact or represented real values
and rejects the same invalid domains as the coefficient-sample helper.

The immutable `RelationalRateSampleBounds` retains `samples`, `sample_step`,
`sample_error_bound`, `third_derivative_bound`, `rate_estimate`,
`rate_error_bound`, exact rational `rate_bounds` and conditional `scope`.
Unlike coefficient identification, this stencil requires no phase-consensus
or pure-form-mode preparation. For a phase readout, the caller must supply
a consistent real lift: three wrapped values across a branch cut are not a
smooth signal. The helper authenticates neither this lift nor the clock,
errors, smoothness or preparation. A sample error includes every uncertainty
between the supplied value and the declared ideal observation, including
initialization and numerical error when applicable.

The rate and coefficient sample helpers delegate to the joint sample-jet
owner below, retaining their existing schemas and exact numerical values.
To compare two preparations, pass each sample-rate report's `rate_bounds` to
`bound_relational_rate_contrast`; its own denominator and precision duties
remain applicable. The shared SDK exporter retains exact sample-rate reports.
No coefficient law, graph execution or model classifier is introduced.

### Joint value, rate and acceleration from finite samples

`bound_relational_sample_jet_budget(*, sample_step, sample_error_bounds,
third_derivative_bound, timestamp_error_bounds=(0,0,0),
first_derivative_bound=None, observation_time=0)` in
[the shared observation owner](../../../src/tnfr/physics/relational_observations.py)
prepares a `RelationalSampleJetBudget` without consuming sample values.
The error arguments contain three ordered nonnegative exact/represented
reals; booleans and nonfinite inputs reject. Sample spacing is positive.
The nominal times are `observation_time+(0,h,2h)`. The enlarged evidence
window encloses every possible acquisition time and must be nonnegative.
Both derivatives refer to the first nominal time.

Nonzero timestamp uncertainty requires an independently supplied speed
bound on that entire enlarged window. A C3 bound alone cannot control
timing error. The C3 bound covers at least the nominal stencil window.
These are clock-jitter bounds in one declared clock, not an unknown
clock-scale calibration. The report retains the sample/timing contributions
and differentiation remainder separately, their totals and the uncertainty
of the initial value. Missing speed is valid only for exact timestamps.

`bound_relational_jet_from_samples(samples, **budget_arguments)` admits
three ordered values and returns `RelationalJetSampleBounds` with the
budget, both exact rational estimates, and `value_bounds`, `rate_bounds`
and `acceleration_bounds`. It reuses one set of stencil arithmetic for
both legacy adapters. The shared SDK exporter supports both new reports
through `relational_report_to_dict` and the existing atomic JSON writer.
See the [joint error proof](../../../theory/nodal/SINE_ENVIRONMENTAL_MEMORY.md#sine-finite-sample-admission)
for the unequal-error coefficients and error/spacing tradeoff.

Angular values must describe one supplied continuous real lift. No
automatic unwrapping or smoothness estimate is inferred from the samples.
The resulting boxes are conservative, correlated observations; compatibility
with them does not establish a trajectory fitting the entire raw record.
In particular, a nonzero `value_bounds` width cannot be silently collapsed
into the point-valued visible state of `infer_relational_sine_hidden_state`.

### Static K3 phase-sampling certificate

<a id="relational-capacity-sampling"></a>

`certify_relational_capacity_sampling()` in
[`tnfr.research.relational_capacity_discriminator`](../../../src/tnfr/research/relational_capacity_discriminator.py)
recomputes the fixed [K3 temporal admission](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#capacity-discriminator-temporal-admission).
It has no response inputs. Its immutable report retains the exact prior,
mathematical-pi preparation enclosures, six-coordinate box, chart/metric
bounds, four candidate/arm derivative and first-exit results, and declared
sample/rate error budgets. Unresolved fixed admissions raise rather than
returning an unsupported certificate. `to_dict()` reuses the exact dataclass
projection; the usual atomic writer can save it without adding this research
certificate to the engine's SDK report registry.

The private candidate-field expressions are interval verification algebra
for this fixed K3 reference. They install no countermodel into native
execution and do not evolve a graph. The certificate proves a common
whole-window derivative bound, not that any instrument or numerical sample
source meets the supplied absolute-error ceiling. Exact preparation, affine
timing and supplied coefficients retain the theorem's stated scope.

### Bounded K3 phase-sampling response

<a id="relational-capacity-response"></a>

`prepare_relational_capacity_response()` in the same research owner returns
a fixed immutable protocol. It recomputes prior-domain and seventh-derivative
remainder bounds without evaluating sampled endpoints. The declaration fixes
source beta one, polynomial order six, the two laws and two capacity arms,
three sample times, prediction intervals and all precision gates.
`evaluate_relational_capacity_response(protocol)` rejects a changed protocol
before explicitly evaluating the bounded software source. These Python calls
do not authenticate a prior file freeze; the
[producer](../../../benchmarks/relational_capacity_sampling_response.py) supplies
separate `--prepare` and evaluation invocations with checked source/runtime
and archive hashes, exclusive output and no refinement or retry.

The exact response retains each case's six-coordinate initial coefficients,
whole-box seventh coefficients, polynomial/remainder/state sample enclosures,
phase-contrast midpoints/radii and shared sample-rate report. Per-law analysis
retains normalized contrast, positive baseline, exact corner hull, arithmetic
inflation and individual acceptance verdicts. Partial cases and reasons
remain available on failure. `completed` means all cases produced admitted
samples; `passed` additionally requires all frozen discrimination checks.
Exact `to_dict()` projection preserves the evidence and its separate scope.

The [mathematical owner](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#capacity-discriminator-sampled-response)
proves the enclosure and defines the decision. This verification instrument
does not change native network evolution or infer a physical law. Its known
source labels are used for software acceptance, not supplied to the neutral
sample-rate or contrast observers as fitted answers.

<a id="relational-acquisition-audit"></a>

### Read-only retained coefficient-acquisition audit

`audit_relational_coefficient_record(record, *, protocol)` in
[`tnfr.research.relational_acquisition`](../../../src/tnfr/research/relational_acquisition.py)
checks the retained known-source P2 v1 record against an independently supplied
frozen protocol. It verifies the supported preparation/model/observation and
clock declarations, whole-window bounds, each saved state/rate contrast and
numerical defect, accumulated error, sample report and original decisions.
Malformed or contradictory mappings raise; a partial acquisition or failed
analysis retains no coefficient. It neither constructs a graph nor imports a
producer or evolves state.

`audit_relational_coefficient_acquisition(response_path)` additionally reads
the sibling `.protocol.json` and `.source.zip`. It uses the shared strict JSON
reader, binds the embedded protocol to the original, verifies its digest and
the source archive's inventory/member digests, and delegates reconstruction
to the same owner. The archive is read without extraction; encrypted/unsupported
compression rejects. Work budgets allow at most 4096 steps, 32 MiB per stored
file and 128 MiB of expanded source content. These are audit policies, not
physical constants. Current installed source/runtime is not required to match
the historical archive; no replay compatibility verdict is inferred.

The frozen `RelationalAcquisitionAudit` exposes `status`, `consistent`,
`completed_steps`, `acquisition_complete`, `recorded_passed`,
`reconstructed_passed`, optional `sample_bounds`, `unavailable_reasons` and
`scope`. `status="consistent"` reports internal agreement, not experiment
success: an original negative decision remains false. Missing files return
`"unavailable"` and invalid records return `"inconsistent"`, with unavailable
conclusions as `None`. A complete frame window can still have failed or
unavailable coefficient analysis. `to_dict()` projects nested sample evidence
through the existing exact exporter for the usual atomic writer.

The [proof and original evidence](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-temporal-acquisition)
retain their own scope. Consistent bytes/arithmetic do not authenticate an
executed history, trusted chronology, dependency binaries, parameter-blind
uncertainty or physical admission. The audit never overwrites artifacts or
changes the original prediction/precision rule. See
[usage](../../guides/relational/OBSERVATION_AND_INFORMATION.md#audit-saved-acquisition-evidence-without-replay)
and [routine negative controls](../../../tests/test_relational_acquisition_audit.py).

## Shared circular geometry

### Exact cycle resultant sectors

[`derive_cycle_resultant_sector`](../../../src/tnfr/physics/phase_resultant_sectors.py)
accepts a supplied pure cycle, its full oriented `cycle_nodes` order and an
exact full-support `phase_turns` mapping. Turns mean angles divided by
mathematical `2*pi`; inputs must be rational numbers, excluding Booleans and
floats. Live graph phases, capacity and conductance weights are not consumed.
The shared topology budget applies; at least five nodes are required.

The detached report separates ordinary `support_winding`, the windings of
derived skip-two cycles, `zero_resultant_nodes` and
`negative_real_resultant_nodes`. Antipodal edges make the corresponding
winding unavailable. `resultant_sector_available` only requires nonzero
neighbor resultants; `regular_phase_chart` also excludes the local negative-real
Arg branch. Neither flag admits a runtime step. The auxiliary cycles add no
coupling edges and are not another state variable. Exact classifications of
declared turns do not certify rounded radian measurements or a whole path.
The [formation-domain theorem](../../../theory/nodal/RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-formation-domain)
owns the conditional invariant and its degree-two scope; extra neighbors can
invalidate that obstruction. This exact-turn observer does not select or
extend any of the executor's separately declared phase domains.
