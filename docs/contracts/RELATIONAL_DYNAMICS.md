# Relational dynamics API contracts

This chapter owns admission, execution and report contracts for the native
relational law, its observations, and the separately declared normalized-sine
and reciprocal-mobility comparison laws. The linked source modules own their
APIs; definitions and proofs remain with the linked theory owners.
Research priorities belong only to the
[execution plan](../../theory/research/FIVE_STAGE_EXECUTION_PLAN.md).

| Contract family | Read this section | Model boundary |
| --- | --- | --- |
| Native field and phase admission | [Conditional relational execution](#conditional-relational-execution) | `RelationalExchangeModel` with `dynamics/relational.py`; its three phase domains do not change the law. |
| Supplied linear observations and finite samples | [Observation bounds](#shared-observation-and-finite-sample-bounds) | A supplied generator or preparation/error model, never inferred execution. |
| Native stepping and hypothetical support changes | [Steps and pattern reports](#native-relational-stepping-and-pattern-reports) | Atomic Euler execution is distinct from detached reports and event budgets. |
| Native capture, retained memory and continuous proofs | [Native proof tools](#native-relational-capture-memory-and-proof-tools) | Each theorem retains its stated support, clock, preparation and domain. |
| Smooth sine field, inference, recovery, response and scale | [Sine proof tools](#normalized-sine-comparison-and-proof-tools) | `normalized_sine_reciprocal_exchange`; reusing a regular reference model does not enable a native execution mode. |
| Alternative mobility and protected relative geometry | [Declared mobility](#sine-pairing-mobility), [relative balance and protection](#sine-mobility-geometry) | Both exchange rows change; shared storage does not transfer the old mean, pulse or recurrence claims. |
| Exact supplied phase turns | [Circular geometry](#shared-circular-geometry) | Geometry alone does not admit either complete law. |

See the [common contract hub](../API_CONTRACTS.md) for shared scalar/solver
boundaries and the [regional and relational guide](../guides/REGIONAL_AND_RELATIONAL.md)
for executable SDK examples. The general operator runtime, including RA,
remains a separate execution contract.

## Conditional relational execution

The opt-in owner [dynamics/relational.py](../../src/tnfr/dynamics/relational.py)
implements `RelationalExchangeModel`, `evaluate_relational_exchange` and
`step_relational_exchange`. `Network.relational_exchange(model)` and
`Network.step_relational(model, dt=..., t=...)` are thin SDK delegates.
The [theory owner](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-separable-exchange)
states the joint-storage, independent-capacity and zero-phase-activity premises.
No runtime flag implicitly substitutes this model for operator execution.

The [resonance contract](#sine-resonance) separately assesses the normalized-sine
comparison law. It does not interpret this native executor's spectrum or the
configured RA operator as that law's frequency response.

The [uniform-locality classification](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#primitive-locality-phase-clock)
fixes the same relative phase evolution under explicit graph-family and storage
premises, without assuming individual phase freezing. The only remaining term
is a state-independent common angular rate. This executor uses its zero-rate
representative and retains both rows frozen at zero capacity. Whole-system
rest or same-model capacity homogeneity are sufficient reference conditions;
clock-unit conversion alone is not. No supplied common clock is inferred.
Its ideal rates consume own capacity, incident primitive form/phase and fixed
model coefficients; total neighbor degrees and remote states are not row
inputs. This does not bypass whole-graph validation: an invalid remote node
still rejects evaluation. Global storage aggregation and backend-dependent
floating arithmetic remain separate from mathematical locality. The result
justifies the existing relative dynamics within that class; it adds no mode,
automatic law selector or new public API.

The [joint-storage classification](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#joint-storage-locality-classification)
also selects this executor's quadratic/cosine storage from a larger periodic
edge-cost class when e,w>0, uniform locality, capacity homogeneity and its exact
continuous loss are imposed across the full regular graph family. It does not
derive that loss from passivity, fix beta or extend the executor's phase domain.
Mixed candidate costs are rejected by mathematical work identities, not by
silently evaluating them with this executor's already selected phase row.

The [passive-loss comparison](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-passive-loss-completion)
adds `rho*N*g` only in its declared mathematical comparison law. It retains
the same equilibria and native form row but has additional phase loss and a
different response. `RelationalExchangeModel` exposes no `rho` option: this
executor and its work reports implement the exact-loss reference. Shared field
and tangent readers supply detached baseline geometry for the comparison;
their output is not an execution or certificate of the alternative. Existing
finite capture/recovery records retain their original law and bounds.
The [nonlinear passive countermodel](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-nonlinear-passive-completion)
likewise introduces no executor parameter. Its unchanged equilibrium tangent
and newly proved common local energy sublevel do not authenticate a reference
capture report for the alternative or transfer a finite transit enclosure.
Its comparison work residual can have either sign; `continuous_loss` in the
reference report remains the exact-loss reference quantity.

`storage_scale` is a required finite positive beta. Nonnegative EPI weight
and positive phase weight use the native coefficient normalization once;
`model.effective_weights` retains that normalized pair. Native staging and
detached sine capture revalidate authoritative stored coefficients through the
shared relational owner before field arithmetic or writes. Revalidation rejects
Boolean, nonfinite and out-of-domain values without normalizing again. Exact
rational proof coefficients retain their values; native outputs also require
binary64 representability. Reconstructing a model or report does not authenticate
its provenance or exempt its fields from admission. Graph-level pressure
mixes, custom pressure hooks, clipping rails, histories and automatic policies
are outside this selected model, and their configuration is preserved.
Active/unknown Gamma and a simultaneous independent-pressure extension reject
without invoking their callbacks; an overridden registry entry for `none`
cannot hide forcing.

The [phase-storage classification](../../theory/TNFR_VARIATIONAL_PRINCIPLE.md#native-phase-storage-classification)
keeps the existing unit cosine-cost normalization. A mathematical choice
`U_phi=kappa*V_phi` with coefficient `beta_U` is represented by the single
`storage_scale=beta_U*kappa`; the API does not add a redundant cost multiplier.
Changing that product changes storage and phase rates while leaving the
specified form-pressure row unchanged. The positive metric at zero phase
pressure uses the existing continuous sinc limit, not an inferred ratio of
two zeros. None of these contracts identifies storage with physical energy.

The executor admits connected, simple, loopless, undirected
support with at least two nodes and unit conductance. Each node must supply
finite signed real scalar EPI, primitive phase and nonnegative capacity.
Uniform-real BEPI, materialized or in its canonical serialized mapping, is
admitted through the shared signed-scalar validator and, like the scalar nodal
solver, commits as its signed scalar. Richer, complex or malformed form rejects.
Raw Boolean/text values and nonzero inputs lost during materialization reject.
This admission does not include isolated-node evolution or a continuous ramp
from zero to unit conductance. Generic pressure has different channel support:
a zero-weight graph edge can still contribute phase, capacity and topology.
The [relation foundation](../../theory/nodal/RELATION_FOUNDATIONS.md#zero-relation-boundary)
separates those cases from true absence; its weighted comparisons are not a
new execution mode of this API.
`model.phase_domain` selects one of three admission paths for the same joint law:

- `"acute"` is the default. Every represented edge gap in the shared signed
  `atan2(sin(delta),cos(delta))` chart must be strictly acute; its positive
  metric check is numerical, not an exact transcendental certificate.
  The same chart admits the whole Euler segment. Reduction modulo binary64
  `2*pi` is not used: at large raw lifts it can disagree substantially with
  the trigonometric phase used by the law. Raw phases remain in the report.
- `"positive_resultant"` requires a certified positive real part of every
  relative neighbor resultant `z_i = sum_j exp(i*(theta_j-theta_i))`. The
  [shared rational enclosure](../../src/tnfr/mathematics/_phase_resultant_chamber.py)
  computes lower bounds `L_i <= Re(z_i)` from exact represented raw radians,
  using the existing mathematical-pi enclosure and a bounded cosine series.
  All `L_i` must be positive; the materialized resultant real part and metric
  must also pass their positivity checks. Individual obtuse or antipodal edges
  can be admitted when their full neighborhood meets these conditions.
- `"regular"` admits relative resultants in the principal-argument domain
  $\mathbb{C}\setminus(-\infty,0]$. The same enclosure owner computes exact rational
  rectangles `([C_lower,C_upper],[S_lower,S_upper])` for each mathematical
  cosine/sine sum at the exact represented raw phases. The quantity
  `m_i=max(0,C_lower,S_lower,-S_upper)` lower-bounds distance to the excluded
  nonpositive-real ray. Every `m_i` must be strictly positive. Negative real
  parts are allowed when the imaginary part is separated from zero. Zero
  resultants and negative-real branch-cut resultants remain excluded.

The second path is a sufficient right-half-plane chamber, not the entire
regular slit-plane domain. The third path targets that larger domain but its
finite-work certificates remain sufficient tests: an unresolved rectangle or
nonpositive lower margin need not prove an actual singularity or branch cut.
Work limits and outward rounding are numerical policies, not physical
thresholds. The options keep the same specified continuous pressure and phase
laws and do not alter default runtime dispatch. A regular snapshot or admitted
proposal does not establish all-future regularity, and it does not transfer
acute stability or winding-protection theorems to a nonacute state.

Regular execution also checks its materialized cosine/sine sums against any
certified component signs and requires a represented argument strictly between
`-pi` and `pi`. An unresolved represented branch, nonpositive metric or lost
nonzero derived value rejects evaluation. With `a=atan2(S,C)`, the regular
metric is evaluated as `H=pi*S/a` when `S` is nonzero, with `H=pi*C` at
`S=a=0,C>0`. This uses the captured sine sum directly rather than evaluating
`sin(a)` again near the branch cut. It is the same ideal metric
`pi*|z|*sinc(a)`, with a numerical realization suitable for this domain.

All three relational modes use the private shared pressure adapter in
[the pressure owner](../../src/tnfr/dynamics/dnfr.py). It combines `g=a/pi`
from each mode's captured relative sums with the existing stable EPI
neighbor reduction and configured two-channel weights. It checks live node,
phase and support order, finite sources in `(-1,1)` and representable pressure
before writing its detached state. It does not reconstruct the phase source
by subtracting a separately rounded global neighbor angle. Field
`pressure_path="relative_resultant_canonical"` identifies this realization;
a committed relational step records the same value in `_DNFR_META.hook` and
`_dnfr_hook_name`. The ordinary default, fused and phase-only pressure paths
keep their existing contracts. A private supplied-source call does not itself
authenticate the caller's derivation of the sources. Reusing one captured
source removes a second global-angle reconstruction from the relational
field; pressure and phase work still retain their actual rounding residuals.
This numerical correction does not re-evaluate or reinterpret frozen response
artifacts produced through earlier arithmetic paths.

Evaluation uses a detached preparation and actual native pressure, with no
live graph mutation. Its immutable `RelationalExchangeField` retains node
order, edges, state, source, metric, both rates and execution path. Storage,
continuous loss, actual work, balance residual and pressure/product defects
use exact fractions of represented values. Tiny exact derived storage is
retained even when it has no nonzero binary64 display. `phase_storage` is the
unscaled `V_phi`; `storage` includes its beta factor.
For `"positive_resultant"`, `resultant_real_lower_bounds` retains the exact
rational `L_i` values in field node order. For `"regular"`, `resultant_bounds`
retains the component rectangles and
`resultant_regular_margin_lower_bounds` retains the corresponding `m_i`.
Unused domain evidence remains `None`; all three are `None` in acute mode.
These explicit enclosures differ from exact arithmetic on rounded storage
or work values: they certify mathematical trigonometric sums and domain
separation, not errors in the separately computed native pressure, phase
metric or rates. Measured pressure-split, rate and work residuals are retained
in every domain. The model and `scope` retain the selected chamber and
enclosure-method provenance. [Regular pressure controls](../../tests/test_relational_regular_pressure.py)
cover captured-source coherence, branch preservation and prewrite rejection.

### Conditional and fast-mediator observations

The [nonlinear fast-mediator limit](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#fast-mediator-reduction)
has a separate continuous approximation contract. Its instantaneous reduced
field is observed by evaluating a detached full graph with the mediator at
the declared local midpoint, then selecting the visible rows. This does not
make midpoint replacement an exact finite-capacity step or a supported
ten-node direct-edge execution mode. The proof retains an initial transient,
a regular neighborhood and a fixed finite horizon; it supplies no fixed-step
Euler error guarantee as mediator capacity grows. Existing field/work reports
suffice for its static controls; no reduced solver or infinite capacity is
admitted by this API.

The [two-intermediary and series result](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#two-mediator-composition)
uses the same detached evaluation after the declared path reconstruction.
Keep its selected phase lift and original port degrees. Replacing a reduced
segment by an ordinary unit edge, or choosing the principal endpoint phase
difference without retaining the path's phase information, can change the
response. Stationary composition is separate from the error bounds for a
simultaneous fast limit; no sequential projection is a finite-capacity step.

The [three-port reduction](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#three-port-collective-interaction)
uses a detached star reconstruction: mediator form is the mean of its three
port forms, and mediator phase is the argument of their collective resultant.
Admit the full reconstructed graph, including every acute fine edge, before
selecting visible rows. A zero resultant does not define a mediator phase;
nonzero resultant alone does not certify the acute domain. Retain original
port degrees and internal neighbor resultants. The inherited storage and
field generally require all three ports together; independent pair interfaces
cannot replace them on an open neighborhood. The separate local fast bound
retains the moving circular mean and hidden initial state. Reconstruction is
an observation preparation, not an instantaneous reset of a live mediator.

### Uniform-form tangent observation

`evaluate_relational_uniform_tangent(graph, model=...)` and the thin SDK
delegate `Network.relational_uniform_tangent(model)` reuse detached field
admission. They require exactly uniform represented EPI, with no equilibrium
tolerance. Nonzero phase pressure and form rates remain visible in the retained
`field`; uniform form alone does not establish a critical phase geometry.

`RelationalUniformTangent.generator` follows `(all form, all phase)` in field
node order. It differentiates the declared smooth law with held support,
capacities and model coefficients. The phase-source derivative uses the Arg
formula with the captured relative resultants. Since `Bx=0`, derivatives of
the phase metric contribute no phase-to-phase block at this state. This is a
materialized ideal derivative, not the derivative of floating-point rounding
or a certified spectral enclosure. Nonfinite or lost nonzero coefficients
are rejected using shared admission.

`phase_source_jacobian`, exact represented `common_offset_residuals` and the
derived `phase_source_row_sum_residuals` retain numerical defects rather than
projecting them to zero. The shared `relational_report_to_dict` exporter admits
this report and validates its field's node labels. It exports stored dataclass
fields; the derived row-sum property can be recomputed from the stored Jacobian.
No eigenmode classification, event selection, source, time step or live graph
mutation is part of this observer. Its use in a
[collective pulse study](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#shared-collective-pulse)
keeps local modal evidence distinct from a maintained nonlinear oscillation.

At an admitted uniform-form state, multiplying `generator` by the concatenated
`field.form_rate + field.phase_rate` evaluates the ideal law's local second
time derivative, with the same materialization limits. In the
[regular source/receiver audit](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regular-seeded-reachability-audit),
this distinguishes zero initial phase velocity from nonzero phase acceleration.
Independent rational bounds certify that preparation's derivative signs; the
matrix product alone does not bound a Taylor remainder or a future response.

<a id="phase-consensus-tangent-observation"></a>
### Phase-consensus tangent observation

`evaluate_relational_consensus_tangent(graph, model=...)` and
`Network.relational_consensus_tangent(model)` admit exactly equal captured
raw phases with arbitrary signed form. `RelationalConsensusTangent` retains
the same field, generator, phase-source Jacobian and represented offset
residuals as the uniform-form reader. Both use one shared derivative builder
after their own exact admission. The older reader still requires uniform EPI.

At phase consensus, the first phase derivative of the reciprocal metric
vanishes even when `L*x` is nonzero. With `K=diag(nu_i/d_i)` and the
combinatorial Laplacian `L`, the ideal blocks are
`(-e*K*L, -(w/pi)*K*L; (w/(beta*pi))*K*L, 0)`. The
[derivation and full-state comparison](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#native-consensus-full-state-tangent)
retain arbitrary positive capacity for spectral claims. The reader itself
also admits zero capacities, under the same native field contract.

The actual source can have nonzero form and phase rates. This is its local
Jacobian, not a claim that it is an equilibrium or that the nonlinear flow
keeps a constant generator as phases separate. Entries are materialized
derivatives with the existing finite-arithmetic checks, not outward spectral
or trajectory enclosures. There is no eigenvalue classifier, time step,
formation verdict, forcing installation or graph mutation. Exact export
preserves source order and residuals through the shared SDK report writer.

## Shared observation and finite-sample bounds

These readers preserve independently supplied model, preparation and measurement
premises. The general linear algebra is law-independent; coefficient adapters
retain their own relational preparation. None executes a graph.

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
[composition scope](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md).

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
See the [joint mediator derivation](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#mediated-pattern-interaction)
for an admitted use with the relational tangent; it does not replace full
`Network.step_relational` evolution with a fitted memory law.

<a id="relational-coefficient-jet"></a>

### Coefficient bounds from a declared initial response jet

`bound_relational_coefficient_from_jet(*, form_bounds, rate_bounds,
acceleration_bounds)` in the [observation owner](../../src/tnfr/physics/relational_observations.py)
encloses `chi=m1^2/[pi^2*(m1^2-m0*m2)]`. Its
[preparation theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#prepared-coefficient-identification)
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
[three-sample bounds](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#coefficient-jet-uncertainty)
require an independent third-derivative bound on the whole sampling interval;
neither three observations nor a solver work residual establishes it.
Misprepared phase or unresolved modal mixtures can yield a plausible but
inapplicable coefficient even when arithmetic succeeds.

For example, exact declared intervals `(1,1)`, `(-3,-3)`, `(7,7)` give restoring
gap 2 and enclose `9/(2*pi^2)`. These numbers illustrate the arithmetic, not a
physical acquisition. `relational_report_to_dict` exports the report's bounds
as exact fraction records and preserves `None` and all reasons. The usual
atomic SDK writer can save that projection; it authenticates no preparation
or calibration. [Routine controls](../../tests/test_relational_coefficient_identification.py)
exercise input rejection, resolution boundaries, gain/clock covariance and export.

<a id="relational-coefficient-samples"></a>

### Coefficient bounds from three uniformly timed samples

`bound_relational_coefficient_from_samples(samples, *, sample_step,
sample_error_bound, third_derivative_bound)` in the same observation owner
implements the [three-sample theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#coefficient-jet-uncertainty).
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
[P2 acquisition](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#coefficient-temporal-acquisition),
whose independent nonlinear and numerical bounds are not general guarantees
for arbitrary samples, graphs or physical measurements. The API changes no
evolution law and adds no `Network` method for this graph-independent arithmetic.

### Normalized contrast between two declared rate intervals

<a id="relational-rate-contrast"></a>

`bound_relational_rate_contrast(*, before_bounds, after_bounds)` in the
[observation owner](../../src/tnfr/physics/relational_observations.py) bounds
the rate change and `Q=after/before-1`. It consumes two declared intervals,
not a graph or a pair of live states. It installs no law or intervention.
The [K3 protocol](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#normalized-capacity-discriminator)
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
[routine controls](../../tests/test_relational_rate_contrast.py) check
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
[the shared observation owner](../../src/tnfr/physics/relational_observations.py)
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
See the [joint error proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-finite-sample-admission)
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
[`tnfr.research.relational_capacity_discriminator`](../../src/tnfr/research/relational_capacity_discriminator.py)
recomputes the fixed [K3 temporal admission](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-discriminator-temporal-admission).
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
[producer](../../benchmarks/relational_capacity_sampling_response.py) supplies
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

The [mathematical owner](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-discriminator-sampled-response)
proves the enclosure and defines the decision. This verification instrument
does not change native network evolution or infer a physical law. Its known
source labels are used for software acceptance, not supplied to the neutral
sample-rate or contrast observers as fitted answers.

<a id="relational-acquisition-audit"></a>

### Read-only retained coefficient-acquisition audit

`audit_relational_coefficient_record(record, *, protocol)` in
[`tnfr.research.relational_acquisition`](../../src/tnfr/research/relational_acquisition.py)
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

The [proof and original evidence](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#coefficient-temporal-acquisition)
retain their own scope. Consistent bytes/arithmetic do not authenticate an
executed history, trusted chronology, dependency binaries, parameter-blind
uncertainty or physical admission. The audit never overwrites artifacts or
changes the original prediction/precision rule. See
[usage](../guides/REGIONAL_AND_RELATIONAL.md#audit-saved-acquisition-evidence-without-replay)
and [routine negative controls](../../tests/test_relational_acquisition_audit.py).

## Native relational stepping and pattern reports

The following step and graph observers consume the native Arg field from
`dynamics/relational.py`, including its selected phase domain.

### Finite relational steps

A step requires positive `dt` and a finite clock advance. It uses explicit
`t`, or graph `_t` with initial default zero. Both Euler rows use the same
initial snapshot and the shared nodal/Euler arithmetic. In the default mode,
the entire proposed phase segment must remain in its initial acute relative
lift. In the positive-resultant mode, let `h_i` be the exact rational difference
between the represented phase endpoint and its initial represented value.
The sufficient whole-segment condition is
`L_i - sum_j abs(h_j-h_i) > 0`, using the unit Lipschitz bound for cosine.
`segment_resultant_real_lower_bounds` retains these exact margins in node order
only in positive-resultant mode.

Regular mode uses the same exact increment differences and requires
`m_i - sum_j abs(h_j-h_i) > 0`. The complex exponential and distance to a
closed set both have unit Lipschitz bounds, so these margins separate every
resultant along the straight phase chord from the excluded ray.
`segment_resultant_regular_margin_lower_bounds` retains these exact margins
only in regular mode. Unused segment evidence is `None`; both fields are
`None` in acute mode. These sufficient tests can reject a chord that is in
fact regular. They certify the straight represented-endpoint proposal,
including endpoint-rounding effects, not the continuous ODE solution or its
numerical error. Endpoint state and pressure are separately refreshed and
admitted before commit. Endpoint admission alone cannot skip a chart boundary.
No finite solver defect is silently converted into a source.

All admission, arithmetic, endpoint checks and graph-owned cache hooks are
staged before commit to ordinary NetworkX attribute dictionaries. Rejection
leaves live state and metadata unchanged. Success writes form, phase, fresh
endpoint pressure and model rate, advances `_t`, updates pressure/trigonometric
cache metadata, removes stale second-derivative aliases and retains the
immutable report at `_relational_exchange`. Existing operator histories are
not extended; they may consequently be stale for a history-consuming operator.
This is neither a concurrent-access transaction nor an entire-loop rollback.

`RelationalExchangeStep` retains `before`, `after`, `epi_update_defect`,
`phase_update_defect`, `clock_defect` and actual `energy_change`.
`energy_step_defect = energy_change - dt * before.storage_rate` measures the
departure from the initial represented work. It is not a bound on ODE error.
An energy increase is reported honestly; successful admission is not a
monotonicity certificate. Repeat steps explicitly and inspect their evidence.
No physical clock, autonomous pattern formation or universal-law selection is
inferred. [Routine controls](../../tests/test_relational_exchange_execution.py)
exercise this owner and SDK delegation; [usage](../guides/REGIONAL_AND_RELATIONAL.md#execute-the-conditional-relational-model)
provides a minimal preparation.

The [local-recovery theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
requires strictly positive held capacities, positive EPI dissipation and a
strictly acute phase equilibrium. It concerns the continuous law near a
prepared geometry, modulo common form and phase offsets. The executor also
admits zero capacity and zero EPI weight, where that recovery result need not
hold. Neither admission nor a small instantaneous work residual certifies
recovery for an arbitrary Euler step or a complete runtime with other events.

The [paired-region result](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-region-interaction)
uses the same law and shared regional support accounting. Its bridge and phase
sectors are prepared. Transmitted deformation does not derive a new interaction
force, autonomous formation or a closed evolution for regional means/storage.

<a id="relational-pattern-observation"></a>

### Prepared relational pattern observations

[`observe_relational_pattern`](../../src/tnfr/physics/relational_observations.py)
and `Network.relational_pattern(model, *, reference_phase, regions, cycles=())`
evaluate one fresh detached relational field and reuse it for the supplied
observations. They do not mutate the graph, select regions, infer a reference
equilibrium or certify a temporal recovery result. The report's `field`
retains the same model, node order, native pressure and numerical defects as
the execution owner.

`reference_phase` maps every graph node to its explicitly supplied real phase
lift; the report retains it in field node order. Each ordered region retains
its nodes, exact represented form mean, phase-error mean, centered form and
centered phase error. Regions must be nonempty and contain distinct existing
nodes; different regions may overlap and need not partition the graph.
Differences use the supplied lifts without independent
wrapping or an inferred history. Common reference-frame offsets are distinct
from internal deformation; arbitrary per-node `2*pi` changes can alter these
lifted observations. Their separate squared norms use the declared form/phase
coordinates and are not combined into a physical distance.

Supplied oriented `cycles` delegate to the shared winding observer. Regional
`transport` delegates to `observe_regional_support_balance`, using the phase
source `w*g` independently of the form rate and retaining native pressure
split defects. Any zero capacity in the full graph, zero EPI weight or a
whole-support region makes transport unavailable, checked in that order and
reported through `transport_unavailable_reason`;
other available pattern observations remain retained. Neither the weighted
form accounting nor a winding integer closes future regional dynamics.

The fresh field's `work` is a `RelationalWorkBalance` aligned with `field.nodes`.
Its rational `form_gradient` retains the exact represented `q=Bx`, before the
older floating `field.form_gradient` is rounded. It separates nonnegative
`dissipation`, signed `exchange` (positive toward form storage), `form_work`,
`phase_work`, `form_residual`, `phase_residual` and their `balance_residual`.
The existing global loss, storage rate and balance residual sum these same
contributions. They are gradient-work contributions to total storage, not
derivatives of an independently assigned nodal energy.

Each `region.work` sums those contributions over its supplied nodes. Only a
partition sums to the global quantities; overlapping regions double count
shared nodes. `region.boundary` uses a shared `RegionalSupportCut`, projected
from admitted `transport` or obtained through `observe_regional_support_cut`.
That cut observer revalidates captured primitives and permits full support,
zero capacities and zero EPI weight without dividing by any of them. Existing
`RegionalSupportBalance.cut` is a read-only projection of its retained data,
not a separate provenance check. All cut indices refer to the full node order.

Writing its outward current as `Q_R`, the boundary report retains:

| Report quantity | Exact arithmetic on the represented field |
| --- | --- |
| `form_weighted_rate` | `sum_R (d_i/nu_i)*field.form_rate_i` |
| `form_boundary_rate` | `-e*Q_R` |
| `form_source_rate` | `w*sum_R d_i*g_i` |
| `form_pressure_defect_rate` | `sum_R d_i*pressure_split_residual_i` |
| `form_rounding_defect_rate` | `sum_R (d_i/nu_i)*nodal_rate_rounding_defect_i` |
| `form_identity_residual` | Actual weighted form rate minus the four terms above; exact zero |
| `phase_weighted_rate` | `sum_R (H_i/nu_i)*field.phase_rate_i` |
| `phase_boundary_rate` | `(w/beta)*Q_R` |
| `phase_rate_residual` | Actual weighted phase rate minus its boundary term; retained, not assigned zero |

Here `d_i` is the full-graph degree and `H_i` the captured phase metric.
The phase rate is not the derivative of a weighted phase total because `H`
changes. Unlike legacy `transport`, these balances admit `e=0`, full-support
regions (empty cut), and zero capacity outside the region. A zero capacity
inside it sets `weighted_rate_unavailable_reason="zero_capacity_in_region"`;
both weighted rates, form rounding/identity residuals and phase rate residual
are then `None`. Nodal/regional work, cut and undivided model terms remain
available. No division through zero or inference of equilibrium is performed.
The [derivation and scope](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-work-integration)
distinguish these represented balances from exact-real storage identities.

The fresh field additionally retains exact rational `phase_mobility`,
`a_i=nu_i/H_i`, and `phase_rate_rounding_defect`, the represented phase rate
minus `(w/beta)*a_i*q_i`. These use the captured positive metric and exact
`work.form_gradient`; the older floating gradient can lose information.

`region.phase_response` is a `RegionalPhaseResponse` computed from this same
field and cut. For `n` selected nodes and `k=w/beta`, it retains:

| Report quantity | Exact arithmetic on the represented field |
| --- | --- |
| `mean_mobility`, `mean_form_gradient` | Arithmetic regional means of `a` and `q` |
| `mobility_variance`, `form_gradient_variance`, `mobility_gradient_covariance` | Population variances and covariance, with denominator `n` |
| `mean_mobility_boundary_rate` | `k*mean_mobility*Q_R` |
| `covariance_rate` | `k*n*Cov_R(a,q)` |
| `covariance_rate_squared_bound` | `k^2*n^2*Var_R(a)*Var_R(q)` |
| `model_total_rate` | `k*sum_R a_i*q_i`; equals the two contributions above |
| `rounding_residual` | Sum of captured `phase_rate_rounding_defect` |
| `total_rate`, `mean_rate` | Sum and arithmetic mean of actual captured phase rates |
| `identity_residual` | Actual total minus boundary, covariance and rounding terms; exact zero |

The squared bound bounds `covariance_rate**2`, without a tolerance or square
root. This unweighted rate admits zero capacity, a singleton, full support
and zero EPI weight under the field's existing admission. It is a model-rate
observation, not a measured time derivative. A zero cut need not imply zero
response. Covariance includes boundary mobility variation on a proper region;
it need not arise solely from internal edges or from phase geometry when
capacity is heterogeneous. Partitioned totals add, but their mean-mobility
boundary terms need not cancel. See the
[nonlinear response theorem](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regional-phase-mobility-balance).

The actual engine always supplies field `work`, `phase_mobility` and
`phase_rate_rounding_defect`, and regional `work`, `boundary` and
`phase_response`. Optional `None` defaults preserve earlier manually
constructed report records. These additional observations alter no evolution
law and certify no autonomous regional closure or monotone clock.
The [state-and-rate counterexample](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#state-rate-predictivity)
shows why: a ten-coordinate coarse projection and its predicted rate can agree
while coarse acceleration differs. This does not identify the full reports:
their retained centered form vectors distinguish the omitted internal
contrast. No acceleration observer or reduced evolution law is implied by
the existing snapshot interface.
The separate [joint memory theorem](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md)
uses those full coordinates to justify a prepared-even approximation with
conditional finite-horizon error bounds. Its neighborhood and derivative
constants are not part of an executable admission certificate; the SDK still
executes and reports the full nonlinear state.

`tnfr.sdk.relational_report_to_dict(report)` projects native relational reports
and explicitly registered owner-managed reports. The
[shared dispatcher](../../src/tnfr/sdk/relational_reports.py) owns the accepted
types; unsupported values reject. Each report's subsection states its scientific
scope. Having a `to_dict()` method alone does not register an arbitrary object.
Its detached JSON-compatible envelope has
`schema="tnfr.relational-report.v1"`, `report_type` and recursively projected
`report` fields. Exact fractions use
`{"numerator": ..., "denominator": ...}` records; tuples become ordered arrays,
including tuple node labels. JSON scalar labels are supported; opaque objects
are rejected, including labels in nested supports, regions and cuts, rather
than replaced with `repr` or serialized as dataclass contents. Save the mapping with
`export_to_json(relational_report_to_dict(report), path)`.

This projection is not the `StudyResult` schema, a restorable checkpoint, a
source fingerprint or authentication of a caller-supplied report. Node-type
reconstruction is not promised. It supplies no new CLI execution mode.
[Usage](../guides/REGIONAL_AND_RELATIONAL.md#observe-a-prepared-relational-pattern) belongs to the
shared SDK guide.

<a id="relational-attachment-observation"></a>
### Supplied relational attachment observation

The shared attachment observer in
[relational observations](../../src/tnfr/physics/relational_observations.py)
and the SDK delegate compare two disjoint, separately admitted connected
components with one supplied ordered unit bridge. The model must select the
same phase domain for both components and the joined field: `"acute"`,
`"positive_resultant"` or `"regular"`. Every field is admitted independently;
admitting two components does not establish admission of their joined support.
Component forcing, invalid primitive data, nonunit edges, overlapping labels
or invalid cross-component endpoints reject the comparison.

The report retains both complete component fields, the fresh joined field and
the two before/after port cards. Each card retains primitive form/phase/capacity,
degree, exact represented form gradient, relative cosine/sine resultant,
pressure, metric and both rates. The resultant is captured from the sums
already used by the engine; no second pressure implementation is introduced.
The new field observation has a compatibility default of None for older
manually constructed reports; current native evaluations always populate it.

Rational pressure, metric and rate differences follow the joined node order
and subtract the matching component row. Storage changes subtract both
component totals; phase storage remains the unscaled cosine cost, while total
storage includes the model's beta. Shared support-reset accounting supplies
the form-energy identity, and the cut is outward from the full left component.
Zero capacity retains the native frozen-row semantics.

The derived property `continuous_loss_change` subtracts the component loss
rates from the joined rate. It is not event work: a rate change cannot fund an
instantaneous storage jump. `represented_zero_supply_passive` tests whether
the captured `storage_change` is nonpositive. For the ideal state-preserving
unit-edge addition, the required supply is
`(x_a-x_b)**2/2 + beta*(1-cos(theta_b-theta_a))`; the EPI dissipation coefficient
does not multiply this storage increment.

`attachment.assess_supply(supplied_work)` checks the **additional** event-passivity
premise `storage_change <= supplied_work` against the captured represented
storage. Work is required explicitly and has the same structural-storage units.
It may be signed: positive supplies storage and negative extracts it. Exact
rational inputs remain exact, including values too small for binary64; other
real inputs use shared represented-real admission. Booleans, nonfinite values
and non-real inputs reject. No graph is read, refreshed or modified.

The frozen `RelationalAttachmentSupplyAssessment` retains `required_supply`
(the existing storage change), `supplied_work`, `supply_margin` and
`represented_balance_satisfied`, with explicit scope. The common relational
exporter accepts this assessment and retains exact fractions; attachment exports
also include the two derived properties. The caller's work is not authenticated,
and arithmetic on a publicly constructed report is not proof of its provenance.
A represented balance is not an exact-real trigonometric passivity certificate.
Even a zero-cost permitted attachment can change rates; neither that balance nor
the continuous loss selects an occurrence time or requires an event.

Both live graphs remain untouched. The disconnected union is used only by
the support-transport observer, never by the connected relational evaluator.
This is a hypothetical support comparison, not an executed event, a selector,
a recovery certificate or a closed coarse-state model. Exact arithmetic on
captured binary64 values is not a bound on ideal trigonometric evaluation.
The [interface derivation](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#one-bridge-interface-admission)
owns the result; [SDK usage](../guides/REGIONAL_AND_RELATIONAL.md#compare-a-supplied-connection)
owns the call and report field names. The common relational exporter also
accepts this report, validating labels in its nested component/joined fields,
bridge and port cards and retaining detached exact differences.

<a id="relational-relocation-observation"></a>
### Supplied relational bridge relocation

`observe_relational_relocation(graph, *, model, remove_bridge, add_bridge)`
in the same [observation owner](../../src/tnfr/physics/relational_observations.py)
and `Network.relational_relocation(model, *, remove_bridge, add_bridge)` compare
one supplied support exchange without changing primitive form, phase or held
capacity. The original graph must pass the native connected, simple, unit-edge,
unforced admission in the selected phase domain. Removing the supplied existing
edge must leave exactly two connected components, each containing at least two
nodes. The first old endpoint identifies the first component. The new ordered endpoints must
cross those components in the same order, and the new edge must be absent from
the original support. No-op replacements, internal new edges and removal of a
non-bridge or leaf bridge reject. The final connected field passes the same
selected phase-domain admission independently. Nonnegative capacities,
including zero, retain their normal execution meaning.

The frozen `RelationalRelocationObservation` retains:

- `before` and `after`: both fresh complete fields in the original node order.
- `components`: the two ordered node partitions after removal, **not** component
  fields; no relational evaluation runs on the disconnected intermediate graph.
- `remove_bridge`, `add_bridge` and `ports`: supplied edges and before/after
  cards for their unique endpoints in field order. The existing
  `RelationalAttachmentPort` card is reused, including for a shared endpoint.
- Exact represented pressure, metric and rate differences, form/phase/total
  storage changes, and the shared form-only `transport_reset`.
- `cut_before` and `cut_after`, both directed outward from the first component.

All internal edges and the primitive state are retained. This preserves existing
internal cycles and their instantaneous phase data; it does not establish their
future winding, attraction or physical identity. The generic graph observer is
distinct from the conditional two-C5 recovery argument in the
[relocation theorem](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#identity-preserving-bridge-relocation).

The attachment and relocation reports share `assess_supply(supplied_work)` and
`represented_zero_supply_passive`. `continuous_loss_change` is the new minus
old loss rate, which supplies no instantaneous event work. The reused
`RelationalAttachmentSupplyAssessment.required_supply` is a **signed lower
bound on net supplied work**, namely the captured `storage_change`; it can be
negative when relocation releases storage. Declared extraction is admitted by
the budget exactly when `supplied_work >= storage_change`. This comparison
does not authenticate a supply, select a support event or derive its time.

The common exporter retains both fields, both cuts, the node partitions and
exact changes with the same label checks and rational representation. All
live attributes, histories and support remain unchanged. Floating phase
evaluation supplies represented evidence, not an ideal trigonometric enclosure
or an executable recovery certificate. See
[usage](../guides/REGIONAL_AND_RELATIONAL.md#compare-a-supplied-bridge-relocation).

<a id="relational-reset-observation"></a>
### Supplied joint state and support reset

`observe_relational_reset(before_graph, after_graph, *, storage_scale)` and
`Network.relational_reset(after, *, storage_scale)` compare supplied endpoints
without mutating either graph. They admit the same nonempty ordered nodes,
simple undirected loopless support, symmetric nonnegative conductances and
explicit finite phase at every node. Shared transport admission validates
signed scalar form, nonnegative capacity and stored pressure. It retains its
zero defaults when capacity or stored pressure is absent; these defaults are
not evidence of explicit or measured zero values. Disconnected
support, isolates and zero/nonunit weights are allowed **for observation**;
neither endpoint is thereby admitted to the relational evolution law.

The positive finite `storage_scale` is beta in `S=E_D+beta*V`. Form storage
uses conductance, while phase storage uses every support edge, including zero
conductance, consistently with the native phase neighborhood. The shared
half-sine evaluation avoids cancellation in small phase costs. Its signed
atan2 chart matches the acute native field, including large raw lifts;
binary64 `2*pi` is not treated as an exact trigonometric period. Raw phase
subtraction must still materialize finitely before wrapping. Reports use exact
fractions of represented primitives, not ideal trigonometric enclosures.

`RelationalResetObservation` retains:

- `before`, `after`: detached transport snapshots, including capacity and stored
  pressure with the inherited defaults above, and actual conductance;
  `phase_before/after`, `edges_before/after`.
- `form_state_change`, `phase_state_change`: changes on the old support.
- `form_support_change`, `phase_support_change`: changes of support at the
  new nodal state. `transport_reset.before` is this algebraic intermediate,
  not another observed event.
- `form_storage_change`, `phase_storage_change`, `phase_storage_before/after`,
  `storage_before/after`, `storage_change` and `identity_residual`. Phase terms
  are unscaled `V`; total storage applies beta. The exact decomposition is
  `Delta S = Delta E_state + Delta E_support + beta*(Delta V_state + Delta V_support)`.

The same budget mixin as attachment/relocation provides
`represented_zero_supply_passive` and `assess_supply(supplied_work)`. The latter
retains the signed `required_supply=storage_change` without clipping. These
compare a declared work budget; they do not infer a reservoir, authenticate
an actual event or assign an activation time. A sequence's net storage decline
does not prove each constituent event was separately passive. Birth, deletion
or relabeling of nodes lies outside this same-node interface.

`relational_report_to_dict` retains these fields and exact fractions through
the existing detached exporter; unsupported opaque node labels reject. See
the [joint reset guide](../guides/REGIONAL_AND_RELATIONAL.md#compare-a-joint-state-and-support-reset),
[theory](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#nodal-reorganization-and-contact)
and [actual operator controls](../../tests/physics/test_coupling_attachment_budget.py).

## Native relational capture, memory and proof tools

These certificates retain the native law and the preparation/domain specified
by each API. An observed state, a proposed event and a validated trajectory
are not interchangeable evidence.

### Conditional relational capture

[`certify_relational_capture`](../../src/tnfr/physics/relational_capture.py) and
`Network.relational_capture(model, *, cycles)` apply the
[protected-capture theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-protected-capture)
to one fresh detached field. They add no solver, projection or controller.
The existing model/graph/phase-domain admission runs first. Supply exactly
two ordered five-node cycles covering the full graph; its only other edges
must join matching positions zero and one. Unsupported topology or malformed
inputs raise. An admitted engine snapshot need not satisfy this theorem.

Each ring must have exactly identical form and phase coordinates
`x=m+(A,-A,-B,0,B)` and `theta=c+(a,-a,-b,0,b)`, with held unit capacities
and positive EPI, phase and storage coefficients. Fractions of the captured
values retain both offsets and every copy/reflection defect. Nonzero defects
are not rounded away, even when small. The theorem's exact symmetry is a
sufficient premise; its failure is not proof that recovery is impossible.

Mathematical-pi enclosures test three disjoint open rectangles in `(a,b)`:

| Rectangle | Phase coordinates | Ideal limiting phase sector |
| --- | --- | --- |
| Positive twist | `(2*pi/3,pi) x (0,pi/2)` | `+1` |
| Consensus | `(-2*pi/3,2*pi/3) x (-pi/2,pi/2)` | `0` |
| Negative twist | `(-pi,-2*pi/3) x (-pi/2,0)` | `-1` |

The same rational cosine owner encloses full-support phase storage at the
exact represented raw radians. `storage_bounds` adds exact form storage and
the declared beta factor; its upper bound must be strictly below `7*beta`.
This threshold is derived for this support, symmetry and storage model, not a
universal physical constant or a configurable acceptance tolerance. The
field's floating `phase_storage` is retained separately and is not used as
an exact energy enclosure. Chosen real phase lifts are not silently replaced.

`RelationalCaptureCertificate` retains all candidate rectangle margins, the
selected `rectangle_kind`, exact symmetry defects, coefficient/capacity
admission, storage bounds and enclosure provenance. It sets `target_sector`
only when every premise passes; otherwise `status="unavailable"`,
`target_sector=None` and explicit `unavailable_reasons` are retained.
Its `admitted` property concerns the ideal continuous conditional law from
this exact represented initial state. Current winding is computed separately
by the shared observer and need not match the predicted limiting sector.

The result does not certify future Euler steps, authenticate a caller-created
dataclass, establish an error bound for a numerical trajectory, or demonstrate
formation from winding zero. Exact capture implies an open full-state basin
qualitatively, but no asymmetry tolerance is computed or admitted here.
The shared `relational_report_to_dict` exporter preserves the exact evidence.
[Usage](../guides/REGIONAL_AND_RELATIONAL.md#check-a-protected-relational-basin) and
[routine controls](../../tests/test_relational_capture.py) exercise the same owner.

`certify_relational_local_capture(graph, *, model, cycles, target_sector=1)`
and `Network.relational_local_capture` apply the existing full-state local
energy theorem to the same supplied support. They require unit capacity and
storage scale, positive coefficients and a declared sector in `{-1,0,1}`.
They do not require reflection symmetry. The ideal reference in each ring is
`sector*pi*(4/5,-4/5,-2/5,0,2/5)`; common form and phase offsets are removed
exactly, without changing individual phase lifts. Rational pi and cosine
enclosures must prove squared quotient distance below `9/800` and excess
storage above the target below `1/100000`. These conservative constants come
from the local energy barrier, not fitting the evaluated response.

`RelationalLocalCaptureCertificate` retains the declared target, affine-pi
phase errors, distance/energy bounds and individual admission reasons. Its
target is available only when all premises hold. A negative lower bound on
the enclosed excess is allowed: interval uncertainty can straddle zero,
while the admitted local theorem supplies nonnegativity of the exact excess.
This certifies ideal continuation from the precise represented snapshot.
Applying it to a numerical endpoint supplies no error enclosure connecting
that endpoint to the original continuous initial-value problem. Both capture
reports use the same exact exporter; neither controls the dynamics.

`observe_relational_sector_geometry(graph, *, storage_scale, cycles,
target_sector=1)` and `Network.relational_sector_geometry` expose the shared
geometric evidence without a dynamics model. They consume the supplied simple
unit support, signed form, primitive phase and positive storage scale. The
support must be the two five-node rings with matching bridges at positions
zero and one. Capacity, pressure, forcing and reference rates are not consumed.
The detached `RelationalSectorGeometry` retains nodes, edges, coordinates,
exact form storage, gap lifts, cycle periods and rigorous energy/barrier bounds.
Its `admitted` property means that acute geometry, the declared sector and the
strict energy barrier all pass; it makes no claim about a future trajectory.

Its `sublevel_*` bounds concern states in the same sector with storage no
greater than the reported upper bound. They are unavailable if any geometric
premise fails. The report contains no `future_*` bounds or limiting-target
prediction. Turning sublevel evidence into a future guarantee requires a
separate complete-law proof, such as the
[strict-loss class theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-sector-law-class).
That theorem covers specified autonomous smooth phase laws with the native
form row, positive held capacities/coefficient scales, common-offset symmetry,
rest and a uniform pointwise strict form-loss bound. Its conclusions include
sector retention, recovery of the twist and eventual exponential convergence;
the rates remain law-dependent. A snapshot residual or supplied Boolean flag
cannot admit a law to that class. Geometry can be read even when a graph has
forcing or lacks capacity; this does not certify that graph's execution.

`certify_relational_sector_capture(graph, *, model, cycles, target_sector=1)`
and `Network.relational_sector_capture` apply the
[full-state acute-sector theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-acute-sector-capture).
On the same support they require strictly positive held capacities, positive
storage scale and coefficients, strictly acute gaps and both ring periods equal to the declared
`+1` or `-1` sector. Exact affine-pi inequalities validate integer full-turn
candidates for each raw edge difference. The three independent cycle periods
include the four-edge bridge cycle, which must have period zero. Floating
remainders and numerical winding telemetry do not establish these premises.

`RelationalSectorCaptureCertificate` retains these gap margins and periods,
full energy bounds and a certified interval for the sufficient barrier
`beta*[10-5*cos(2*pi/5)-4*cos(3*pi/8)]`. The energy upper bound must be strictly below
the barrier lower bound. Exact reflection and a small Euclidean distance to a
target are unnecessary. The limiting target is circular, modulo a common
offset and fixed nodewise full-turn representatives. This is a sufficient
barrier, not a claimed sharp bound or a numerical tolerance. Unresolved
premises remain unavailable with evidence. As with the other certificates,
admission concerns ideal continuation from the captured state; applying a
new theorem after an experiment does not revise its frozen decision criteria.

`geometric_barrier_bounds` encloses the unscaled geometric cost and
`capture_barrier_bounds` includes beta. The retained `unit_capacity` and
`unit_storage_scale` flags are descriptive; `positive_capacity` supplies
the capacity admission. Zero capacity is not admitted by this recovery theorem.
For full admission, `normalized_energy_margin_lower_bound=eta_lower>0`
supplies three conditional all-future bounds: acute edge margin `13*eta_lower`,
nodewise resultant real part `26*degree*eta_lower/pi_upper`, and phase metric
`26*degree*eta_lower`. The resultants/metrics follow `field.nodes`. All three
future-bound fields are `None` when any premise fails, even if energy alone
passes. They describe the ideal continuation, not Euler error or a solver step.
The [derivation](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-sector-consolidation)
connects those margins to the same Jensen barrier without adding a threshold policy.

| Certificate | Distinguishing sufficient premises | Scope |
| --- | --- | --- |
| `relational_capture` | Exact reflected state, unit capacity, positive beta, strict rectangle and energy below `7*beta` | Can admit nonacute states and consensus |
| `relational_consensus_capture` | Exact reflected form, equal initial phases, unit capacity/beta, effective `e=w=1/2`, form storage at most 9 | Analytic entry into consensus by time one, then convergence; no evaluated endpoint |
| `relational_consensus_formation_obstruction` | Arbitrary full form on the same two-ring support, equal initial phases, unit capacity/beta, effective `e=w=1/2`, form storage at most 9 | Excludes aligned acute unit winding throughout regular existence; no global continuation or consensus verdict |
| `relational_local_capture` | Unit capacity/beta, small full-state quotient distance and excess energy | No reflection; declared target `-1`, `0` or `1` |
| `relational_sector_capture` | Positive held capacities/beta, exact acute sector and scaled barrier | No reflection or local-radius gate; targets `-1` or `1`; quantitative regularity |

These independent sufficient theorems share one capture owner and exact
exporter. The sector certificate uses the same geometric kernel as the
law-neutral observer while preserving its reference-law admission and flat
report fields. The rho and eta comparison laws are mathematical comparisons,
not runtime modes of this certificate. No ordering of verdicts or automatic
policy is implied.

<a id="phase-consensus-capture"></a>
### Analytic capture from phase consensus

`certify_relational_consensus_capture(graph, *, model, cycles)` and
`Network.relational_consensus_capture(model, *, cycles)` apply the
[phase-consensus obstruction](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-consensus-preparation-obstruction).
They reuse one existing native capture, including full field and support
admission. Both rings have the same exact represented form
`m+(A,-A,-B,0,B)` and every captured raw phase equals one common value.
Held capacities and storage scale are one; normalized effective weights
are exactly `e=w=1/2`. The sufficient initial form budget is
`F=6*A**2-4*A*B+4*B**2<=9`. These fixed constants belong to this theorem
and structural clock, not a configurable policy or physical scale.

`RelationalConsensusCaptureCertificate.initial` retains the ordinary capture
report unchanged. Its initial `energy_admitted` may be false: the new result
proves a later entry, rather than relaxing that report's strict `E<7` test.
The new report retains exact `initial_form_storage`, `form_storage_margin`,
phase/coefficient admission and separate analytic bounds:

- Through `bootstrap_time=1`, the reflected phase coordinates obey
  `|a|<=sqrt(3*F/2)/(3*pi)` and `|b|<=sqrt(F)/(2*pi)`, with strict
  positive margins below `1/2` and outward rational enclosures.
- At time one, phase storage is at most `F/4` and total storage at most
  `2*F/3<=6`. The report retains consensus rectangle margins and
  `capture_margin=7-2*F/3`. These certify subsequent ideal consensus.

On full admission, `target_sector=0` and `status="admitted"`. Missing theorem
premises give explicit reasons, `target_sector=None` and unavailable dynamic
estimates; they do not prove that formation or instability occurs. Malformed
inputs or unsupported native support still raise through shared admission.
The analytic bounds are not an endpoint state, a sampled trajectory or a
guarantee that future finite steps pass the configured phase-domain guard.
The conclusion excludes maintained unit winding only in this stated class.
The shared exporter validates the nested initial labels and preserves exact
fractions and unavailable values. See the
[usage example](../guides/REGIONAL_AND_RELATIONAL.md#check-relaxation-from-phase-consensus).

<a id="full-form-consensus-obstruction"></a>
### Full-form phase-consensus obstruction

`certify_relational_consensus_formation_obstruction(graph, *, model, cycles)`
and `Network.relational_consensus_formation_obstruction` reuse one fresh native
field and the existing exact two-C5/adjacent-matching-bridge support admission.
They retain arbitrary form, including all bridge costs, without imposing or
projecting copy/reflection symmetry. Exactly equal captured raw phases,
unit held capacities and beta, effective `e=w=1/2`, and `F(0)<=9` admit the
[nonlinear theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-full-consensus-formation-obstruction).

The report exposes `all_regular_time_phase_storage_upper_bound=7*F(0)/10`,
the outward target-storage bounds and their strictly positive exclusion margin.
`excluded_target_sectors=(-1,1)` excludes entry into the aligned acute
unit-winding targets or convergence to them on the ideal regular path. It does
not forbid every transient winding change or classify all other outcomes.
`continuation_status="not_certified"` remains explicit: this theorem proves
neither indefinite regular existence nor convergence to consensus.

Failed theorem premises return `status="unavailable"`, named reasons and no
prospective bounds or excluded sectors. Invalid live state/support still
rejects through shared admission. The detached observer executes no trajectory,
installs no law and does not revise frozen evidence. Export preserves exact
rationals, full source state, cycles and unavailable fields, validating labels
before projection. Publicly constructing a report does not authenticate it.
The earlier reflected capture is retained because it proves the stronger
consensus conclusion on its smaller class.

<a id="relational-detachment-observation"></a>
### Isolated-cycle capture and hypothetical detachment

`certify_relational_cycle_capture(graph, *, model, cycle, target_sector=1)`
and `Network.relational_cycle_capture` apply the
[isolated-ring theorem](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#relational-pattern-detachment)
through the shared capture owner. The graph must be exactly the supplied
ordered unit C5. Native connected-field admission runs first. The sufficient
capture conditions require positive held capacities and coefficients, strictly
acute true-pi gaps, oriented winding equal to the declared `+1` or `-1`, and
mathematical storage strictly below `beta*(5-4*cos(3*pi/8))`.

`RelationalCycleCaptureCertificate` retains the fresh `field`, `cycle`, exact
gap/period evidence, phase and total `storage_bounds`, geometric and scaled
barrier bounds, and `storage_margin_lower_bound`. The gap evidence follows
`field.edges`; `cycle_winding` follows the declared orientation. `admitted`
and `target_sector` require every premise. An unmet sufficient condition
returns `status="unavailable"`, `target_sector=None` and reasons; invalid
field, topology or input admission raises. No reflection or equal-capacity
premise is imposed. Recovery means ideal convergence to uniform form and
the cycle twist modulo this component's own limiting offsets.

`observe_relational_detachment(graph, *, model, cycles, target_sector=1)`
and `Network.relational_detachment` compare removal of the two matching bridges
at positions zero and one of the supplied C5 rings. The original graph must
be exactly this connected unit support. The observer retains form, phase and
capacity and recomputes each component's degree, pressure, metric and rates.
It evaluates the two post-cut connected components separately. A disconnected
union is used only for shared reset accounting, never relational execution.
Live nodal attributes, graph metadata, histories and support remain unchanged.

`RelationalDetachmentObservation` retains:

- `before`, the connected field; `components`, the two full C5 capture
  certificates; and `removed_bridges`, the supplied matching pairs.
- `reset`, the shared `RelationalResetObservation`. Its pressure snapshots
  contain freshly evaluated pre/post-cut pressure, not stale stored pressure.
  It owns represented event storage and `assess_supply`; continuous loss is
  not an event reserve.
- Exact represented `form_rate_change`, `phase_rate_change`, `pressure_change`
  and `phase_metric_change`, each after minus before in `before.nodes` order.
- `field_form_storage_change`, `field_phase_storage_change` and
  `field_storage_change`, the sums of component field values minus the
  corresponding original field value. Phase storage is unscaled.
- `storage_reconciliation_residual`, equal to `field_storage_change` minus
  `reset.storage_change`; `capture_admitted` is the conjunction of the two
  component admissions.

The reset uses wrapped represented half-sine costs; a positive-resultant
native field uses raw represented gaps. Their floating arithmetic need not
agree, particularly for large phase representatives. Neither substitutes for
the rational cosine enclosures used by the capture certificate. The exporter
retains all three evidence types separately, exact fractions, nested labels
and the derived `capture_admitted` flag.

The joined acute-sector theorem implies both ideal component barriers;
direct component checks can also succeed outside that sufficient joined
barrier. Zero bridge storage does not imply unchanged rates or safe timing.
The early formation seed has a regular post-cut field but a topological
obstruction to the target twist; formed-state convergence establishes a
sufficiently late window, without certifying a particular event time.
No observer installs a removal event, selects its time, certifies prior
formation error or widens connected-support execution. A different phase law
requires its own post-cut admission to the theorem; the SDK certifies the
declared native reference model. See the
[usage](../guides/REGIONAL_AND_RELATIONAL.md#inspect-hypothetical-pattern-detachment),
[engine controls](../../tests/test_relational_detachment.py) and
[SDK/export controls](../../tests/sdk/test_relational_detachment.py).

<a id="relational-seeded-formation-obstruction"></a>
### Storage obstruction for a uniform receiver

[`certify_relational_seeded_formation_obstruction()`](../../src/tnfr/physics/relational_capture.py)
evaluates the [fixed source/receiver storage theorem](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#induced-formation-storage-obstruction).
The source is an exact unit-C5 winding-one twist; the receiver has uniform
phase on its own supplied C5. Both have the same constant form, held unit
capacity, `beta=1` and `e=w=1/2`. The two cases add either the position-zero
bridge or both matching bridges at positions zero and one. Subsequent support
is held fixed without forcing or additional events.

`RelationalSeededFormationObstruction` encloses the source storage and the
necessary two-pattern target storage. Each `RelationalSeededFormationCase`
retains its matching port pairs, bridge-cost ceiling, strict initial-storage
upper bound and `additional_storage_gap_lower_bound`. It covers every relative
phase admitted by the **initial ideal positive-resultant chamber**, without
sampling that phase or evaluating a graph. It does not assert that every
represented approximation will pass the engine's numerical admission.

Writing `c=cos(2*pi/5)`, the source storage is `V*=5*(1-c)`. Positive real
source-port resultants imply each added bridge costs less than `1+2*c`.
Both acute winding-one rings require total storage at least `2*V*`, so the
one- and two-bridge deficits exceed `4-7*c` and `3-9*c`, respectively. The
shared exact cosine geometry certifies positive lower bounds on both deficits.
The target is two acute winding-one sectors or their recovered twists;
ordinary winding one in a nonacute configuration is insufficient for this
storage lower bound.

`status="obstructed"` and `obstruction_certified=True` mean the specified target
is excluded under storage-nonincreasing continuation. There is no `admitted`
alias and no failed numerical trajectory. If the bounds cannot separate,
`status="unavailable"` retains the intervals and reasons. The deficit is only
a necessary missing storage amount; supplying a work number without changing
the state or law cannot repair it, and exceeding it would not prove formation.

Initial states in the wider mathematical regular domain are outside this
certificate. The same theorem documents a two-port state with negative real
but nonzero imaginary source resultants, positive native metrics and enough
total storage to avoid this particular obstruction. The positive-resultant
option rejects that state; the regular option admits its represented
preparation when the resultant rectangles, branch and materialization checks
pass. This changes executable coverage, not the obstruction's hypotheses.
Initial regularity and sufficient storage do not establish a regular future
route, target reachability or eventual formation.

The separate [geometric and initial-rate audit](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regular-seeded-reachability-audit)
provides a regular common-sublevel path, an affordable internal cancellation
crossing and an exact eight-coordinate reflection restriction. These do not
alter this obstruction certificate's scope or admit the existing copied-ring
transit solver. The native full field and uniform tangent supply its static
controls; no new event or autonomous selection API is required.

The [guide](../guides/REGIONAL_AND_RELATIONAL.md#check-a-uniform-receivers-formation-budget),
[capture controls](../../tests/test_relational_capture.py),
[independent proof/domain controls](../../tests/physics/test_relational_seeded_formation.py)
and [SDK export controls](../../tests/sdk/test_relational_reports.py) distinguish
this conditional exclusion from successful capture and physical evidence.

<a id="relational-cycle-memory"></a>
### Asymptotic collective memory on an isolated cycle

`bound_relational_cycle_memory(*, model, form_direction, phase_direction,
capacity, amplitude_radius, target_sector=1)` in the
[cycle-memory owner](../../src/tnfr/physics/relational_cycle_memory.py)
encloses the quadratic coefficient of a limiting common-phase shift. Its
[theorem](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#relational-retained-phase-memory)
concerns the ideal native law on a unit C5 with positive **homogeneous held**
capacity, `e,w,beta>0`, and no inputs or further support events.

The supplied ordered directions `u,v` each contain exactly five finite real
coordinates with exact sum zero. They specify the ideal family
`x=epsilon*u`, `theta_i=target_sector*2*pi*i/5+epsilon*v_i`, in a common
declared lift, with dimensionless epsilon. Directions carry their nodal
coordinate units. Common initial offsets can be supplied identically without
changing the coefficient. Exact rational inputs are preserved; other reals
use shared represented-real admission. Booleans, nonfinite values, unordered
containers, wrong cardinality and noncentered directions reject. The model
uses its admitted effective coefficients; zero EPI weight, nonpositive capacity
or radius, and sectors other than nonboolean integer `+1` or `-1` reject.
This is an ideal family declaration, not admission of a materialized graph.

For `D*v = v_(i+1)-v_(i-1)` and `kappa=target_sector*2*pi/5`, the exact
`cyclic_pairing=u.T*D*v` supplies
`C=w*sin(kappa)*cyclic_pairing/(10*e*beta*pi*cos(kappa)^2)`.
The limiting lifted phase shift is `C*epsilon^2+O(epsilon^3)`.
The positive common capacity cancels from C; it still affects the clock
and the response to an added connection. A zero pairing makes this quadratic
term zero, without proving the absence of all higher-order memory.

The immutable `RelationalCycleMemoryBounds` retains:

- The model, directions, common capacity, declared radius and sector;
  exact `cyclic_pairing` and `quadratic_phase_shift_coefficient_bounds`.
- `quadratic_left_port_form_rate_coefficient_bounds`: C multiplied by
  `-capacity*w/[pi*(1+2*cos(kappa))]`. This is the leading initial form-rate
  response after hypothetical one-port contact with an untouched, aligned,
  equal-form and equal-capacity reference C5. It is not a measured response.
- True-pi/trigonometric bounds, `reference_storage_bounds`,
  `storage_upper_bound`, `capture_barrier_bounds`, and strict lower bounds
  on the family acute and storage margins. These establish a sufficient
  recovery basin for every `abs(epsilon)<=amplitude_radius` when
  `basin_admitted` is true; failures retain `unavailable_reasons`.
- `remainder_order=3` and **`remainder_bound=None`**. The coefficient remains
  available when a requested family radius fails the sufficient basin test,
  since it concerns the local limit as epsilon tends to zero.

The family storage bound uses the existing cycle Dirichlet owner and cosine
barrier, with no trajectory or fitted response. Certified coefficient bounds
and an admitted recovery radius do **not** enclose the finite-amplitude final
offset or certify its sign at a chosen nonzero amplitude. The missing remainder
must not be replaced by zero. Exact scaling preserves tiny nonzero input
directions and model denominators without first rounding them into the
interval grid.

The observable distinction requires a retained reference and a declared
readout; an isolated common phase rotation is a symmetry. Hypothetical contact
cost is even in that phase difference, while the signed form-rate response
distinguishes its sign. Contact can require supplied work and change the future
state. Neither an actual event, a nondestructive measurement nor a new clock
is installed. The SDK exporter retains this static report; graph-dependent
mean/covariance and contact readouts remain with `relational_pattern` and
`relational_attachment`, rather than a duplicate `Network` execution method.
See [usage](../guides/REGIONAL_AND_RELATIONAL.md#bound-an-ideal-pattern-memory-family),
[coefficient admission](../../tests/test_relational_cycle_memory.py),
[independent mechanism controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
and [SDK export](../../tests/sdk/test_relational_cycle_memory.py).

<a id="relational-finite-memory"></a>
### Fixed finite-amplitude limiting memory

`certify_relational_cycle_memory(*, amplitude=Fraction(1, 2**20))` reuses the
general coefficient/capture owner with the narrower
[integrated error proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#relational-finite-phase-memory).
Its premises are fixed: ideal unit C5, winding +1, `e=w=1/2`, `beta=nu=1`,
zero means, `x(0)=+/-epsilon*(1,-1,0,0,0)` and
`theta(0)=theta_*+epsilon*(0,1,-1,0,0)`. The supplied law remains unforced,
with fixed support/capacity throughout recovery. An untouched aligned C5
supplies the reference for hypothetical matching-port contact.

Shared exact/represented admission requires `0 < amplitude <= 1/1024`;
invalid inputs raise. The immutable `RelationalFiniteMemoryCertificate`
retains the preparation, radius `1/100`, strict first-exit storage margin,
state/integral bounds and checked constants. Its
`phase_remainder_bound=2**15*amplitude**3` bounds the discrepancy between
the actual limiting phase and its enclosed quadratic approximation.
Each `RelationalFiniteMemoryCase` retains its asymptotic report and the
full `limiting_phase_shift_bounds`.

The nonlinear hypothetical readout is evaluated directly over those bounds:
the right-port form rate is
`atan(sin(delta)/(2*cos(2*pi/5)+cos(delta)))/(2*pi)` and the left rate is
its negative. `contact_storage_bounds` enclose `2*sin(delta/2)**2`, avoiding
subtraction near zero. This work is separate from continuous dissipation;
no reservoir, passive event or actual attachment is supplied.

`phase_signs_separated` and `contact_signs_separated` retain independent
verdicts; `status="admitted"` and `admitted=True` require both. At the
default amplitude, the positive-arm phase is between approximately
`5.48241e-13` and `6.05085e-13`, and its left readout lies between
`-5.95181e-14` and `-5.39267e-14`. The negative arm has the opposite signs.
The exact rational interval endpoints in the report are authoritative;
symmetric bounds do not prove exact nonlinear antisymmetry of the two arms.
The error bound is exactly `2**-45` at this amplitude.

An inconclusive sign bound returns `status="unavailable"` with explicit
reasons. For example, `1/1024` admits recovery but not phase-sign separation;
at `2**-200`, exact scaled phase signs survive while fixed interval precision
cannot resolve the contact signs. Neither result becomes an exact zero.
Failure of the fixed proof constants raises `ArithmeticError`.

This is a limiting-state certificate without a trajectory, finite recovery
time, graph initialization error bound or physical measurement. The generic
family report's `remainder_bound=None` remains unchanged. The shared SDK
exporter preserves all bounds, nested scopes and the `admitted` property.
See [usage](../guides/REGIONAL_AND_RELATIONAL.md#certify-one-finite-memory-preparation),
[admission/readout tests](../../tests/test_relational_finite_memory.py),
[independent proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
and [export tests](../../tests/sdk/test_relational_cycle_memory.py).

<a id="relational-memory-readout"></a>
### Finite-clock memory readout

`certify_relational_cycle_memory_readout(*, decay_blocks=40)` reuses the fixed
amplitude `2**-20` certificate and the
[continuous decay proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#relational-finite-time-memory).
Require a nonnegative integer number of blocks; booleans and nonintegers
raise `TypeError`, negative integers raise `ValueError`. Each block is 4128
units of the declared structural clock. This is a conservative proof schedule,
not laboratory seconds or an earliest recovery time.

The `RelationalMemoryReadoutCertificate` retains the nested limiting-memory
certificate, exact horizon, modified-storage equivalence/initial bounds,
centered-state radius `R=8*epsilon/2**decay_blocks` and mean-phase tail
`Q=4096*R**2`. The auxiliary modified storage proves decay; it does not
replace the native storage or install a new dynamics. At the default horizon
165120, `R=2**-57` and `Q=2**-102`.

Each `RelationalMemoryReadoutCase` includes mean and port-phase intervals,
acute margins, resultant real-part bounds and three separate rate pairs:

- `*_no_contact_form_rate_bounds` enclose the residual isolated-ring motion;
  only the untouched right reference has an exact zero background.
- `*_contact_form_rate_bounds` evaluate the full proposed degree-three port
  law, including residual form and phase. Left and right rates need not be
  exact negatives, and immediate phase rates need not vanish.
- `*_contact_form_rate_change_bounds` subtract the no-contact baseline,
  enclosing the change caused by the proposed connection at the same state.

`contact_storage_bounds` enclose the independent event work
`u[0]**2/2 + 2*sin((m+v[0])/2)**2`. The default bounds are strictly positive;
continuous loss is not credited as a reservoir. The separate analytical
`contact_rate_error_upper_bound=2*R+Q` and
`rate_change_error_upper_bound=4*R+Q` compare the unknown actual finite-time
rates/increments with the unknown actual limiting readout. They are not
error bars around an arbitrary midpoint: the limiting-memory interval has
its own retained uncertainty.

The three phase/contact/increment separation flags remain independent.
`admitted` requires all three, with per-case and overall unavailable reasons.
At 40 blocks both preparations have opposite resolved signs at both ports;
at zero blocks conservative bounds are unavailable, not a zero response.
Resultant or acute-domain arithmetic failures raise `ArithmeticError`.
Intervals deliberately discard correlations and may be wider than necessary.

This reader performs no trajectory, contact or subsequent evolution. Its
certificate concerns a hypothetical one-sided response at a finite clock time
under the fixed ideal preparation, not a binary64 initialization guarantee or
a finite accumulated measurement. The shared SDK exporter preserves exact
time, nested evidence, all three observations and the `admitted` property.
See [usage](../guides/REGIONAL_AND_RELATIONAL.md#bound-memory-readout-at-a-finite-time),
[API tests](../../tests/test_relational_memory_readout.py),
[proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
and [native mechanism controls](../../tests/physics/test_relational_cycle_memory_mechanism.py).

<a id="relational-memory-contact"></a>
### Accumulated response during a finite contact

[`certify_relational_memory_contact(*, duration=Fraction(1, 4096))`](../../src/tnfr/physics/relational_memory_contact.py)
consumes the fixed 40-block readout and the
[continuous short-contact proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#relational-finite-contact-memory).
It assumes one matching-port unit bridge is inserted at structural time
165120 and held throughout the declared duration, with the same native law,
unit capacities and preparation. The alternative keeps both rings separate
and continues their recovery. This reader executes neither alternative.

Shared exact/represented admission requires `0 < duration <= 1/3`; invalid
values raise, including booleans. This domain keeps the shared exact
exponential enclosure within its interval of admission; it is not a universal
maximum connection lifetime. Exact rational durations are not rounded to
binary64 before evaluation.

`RelationalMemoryContactCertificate` retains the nested readout, exact
duration/end time, neighborhood, proof constants, exponential bounds and
per-arm `RelationalMemoryContactCase` results. If the initial distance from
the aligned joined equilibrium is bounded by `rho`, the whole contact state
is bounded by `rho*exp(3*h)` and every coordinate's change by
`rho*(exp(3*h)-1)`. The accumulated form error about `h*initial_rate` is
`rho*(exp(3*h)-1-3*h)`, enclosed with the shared rational series. These are
continuous bounds, not an Euler step, a fitted acceleration or a zero remainder.

Each case preserves three different accumulated observations:

- `*_contact_form_change_bounds`: final-minus-initial port form on the
  joined support, including the continuous remainder.
- `*_no_contact_form_change_bounds`: independently evolving isolated control.
  The left ring retains a bounded residual change; the right reference stays
  exactly at its zero-form equilibrium.
- `*_contact_induced_form_change_bounds`: the difference between the two
  evolved solutions at the same elapsed time. It is not a subtraction of
  two initial rates extrapolated without error.

Positive acute margins apply over the whole interval, preserving both ring
windings. This does not claim an unchanged shape or arbitrary future lifetime.
The inherited `contact_storage_bounds` give the insertion cost;
`required_event_work_upper_bound` is an upper bound sufficient to cover either
arm, not a minimal necessary supply or evidence that a reservoir exists.
`continuous_loss_upper_bound` is a separate bound on subsequent dissipation.
No removal, event selector or credit from earlier loss is supplied.

At the default duration `1/4096`, all accumulated contact and contact-induced
signs separate, and both windings persist. For the positive preparation, the
right reference port change is approximately between `1.30029e-17` and
`1.46936e-17`; exact endpoints in the report are authoritative. At duration
`1/3`, the state/winding bounds still pass while the response signs are
unavailable. All intervals and reasons are retained; an inconclusive bound
does not demonstrate a lost or zero signal. Fixed proof/containment failures
raise `ArithmeticError`. `admitted` requires both sign verdicts and winding
preservation; the SDK exporter retains that property and all nested evidence.

The [guide](../guides/REGIONAL_AND_RELATIONAL.md#certify-accumulated-response-during-contact),
[API controls](../../tests/test_relational_memory_contact.py),
[proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
and [native field controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
retain this fixed ideal-law scope. Physical clock/measurement, numerical
initialization error and lasting memory after detachment are separate duties;
the fixed ideal retention certificate below addresses only the last of these.

<a id="relational-memory-retention"></a>
### Retained regional record after contact removal

[`certify_relational_memory_retention()`](../../src/tnfr/physics/relational_memory_contact.py)
closes the fixed default contact with a supplied state-preserving removal of
its one bridge at `T+h`, where `T=165120` and `h=1/4096`. It consumes the
existing contact enclosure and the
[regional retention proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#relational-retained-receiver-record),
without evaluating a trajectory or constructing a binary64 surrogate for its
ideal interval state. There are no new configurable preparation or event laws.

`RelationalMemoryRetentionCertificate` retains the nested contact, reference
storage and isolated capture barrier, separate receiver-mean and component
capture verdicts, and per-preparation `RelationalMemoryRetentionCase` bounds.
The untouched receiver starts with zero mean. Initially only its contact port
has a nonzero form rate, so its final mean lies in `h*right_initial_rate/5 +/- B`,
where `B` is the existing per-node accumulated form remainder. Averaging five
independent remainders costs `B`, not `B/5`. A single-port change is not the
regional mean, and the joined support does not supply an equal-and-opposite
transfer law.

After removal each unit C5 retains its acute winding. Its storage is at most
the exact twist storage plus `20*rho_tube**2`, using the whole-contact state
tube and cancellation of the phase potential's first variation. A strict
margin below the inherited isolated capture barrier admits recovery of both
rings. With the declared common held capacities, each isolated arithmetic
form mean is conserved exactly. Thus `receiver_persistent_mean_bounds` also
encloses the receiver's eventual uniform form and remains valid at every time
after the cut. Its no-contact control has mean exactly zero.

The two preparation means are separated: approximately
`+[2.47071e-18, 3.06859e-18]` and `-[2.47071e-18, 3.06859e-18]`, with exact report
endpoints authoritative. These symmetric bounds do not prove exact
antisymmetry of the actual outcomes. The record is a change of mean relative
to the declared baseline, not a new winding, intrinsic shape label or physical
particle property.

The endpoint bridge form/phase bounds enclose its storage before removal.
`removal_storage_change_bounds` is the negative of that cost; it is separate
from insertion work and continuous loss. A negative jump neither identifies
where energy goes nor installs a reservoir. The report preserves availability
reasons and requires regional sign separation, both component captures and
nonincreasing removal storage for `admitted`. A successful contact port sign
alone cannot establish retention. No disconnected union is passed to the
connected native executor; future recovery is a componentwise theorem.

The [guide](../guides/REGIONAL_AND_RELATIONAL.md#certify-a-retained-receiver-record),
[API controls](../../tests/test_relational_memory_contact.py),
[proof controls](../../tests/physics/test_relational_finite_memory_proof.py),
[native mean controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
and [SDK export](../../tests/sdk/test_relational_cycle_memory.py) retain this
conditional scope. Neither event occurrence, laboratory precision nor physical
identification is certified.

### Validated conditional relational transit

[`certify_relational_transit_capture`](../../src/tnfr/physics/relational_transit.py)
and `Network.relational_transit_capture(*, model, cycles, horizon, time_step,
order=12, requested_sector=1)` perform a read-only proof computation for the
ideal continuous law.
The [validated-transit derivation](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-validated-transit)
owns the reduction and enclosure argument. This API does not advance the
graph, execute a numerical engine step, select operators or alter the supplied
preparation. A represented Euler chord is not used as an exact ODE tube.

The initial state must have the same exact copied/reflected two-ring support
and coordinates as `relational_capture`, with held unit capacities, positive
coefficients and `model.phase_domain="positive_resultant"`. Existing graph,
scalar and cycle admission remains active. Reflection defects are never
projected away. The initial snapshot need not already pass a protected-basin
rectangle or storage test: its complete detached `RelationalCaptureCertificate`
is retained as `initial`, independently of the transit verdict.
The regular engine option does not broaden this proof: its interval system
still requires the positive-resultant chamber throughout every accepted tube.

`horizon` and `time_step` must be strictly positive exact `Fraction` or integer
values; floats and Booleans are rejected. These times are relative structural
durations from the supplied state, with no interpretation of a stored graph
clock or laboratory units. `order` must be an integer from 4 through 16.
The work policy admits at most 4096 declared proof steps. The last interval is
shortened to reach the exact horizon; there is no automatic retry, preparation
change, adaptive step selection or horizon extension.

Interval coordinates are `(q,r,a,b)`, where `q=3*A-B` and `r=2*B-A`. Every
accepted `TransitStep` retains a strict whole-time Picard inclusion, positive
relative-resultant bounds for all distinct node rows including the central
row, and a rational endpoint enclosure. Centered interval Taylor expansion
bounds analytic truncation on that whole tube. A Metzler matrix comparison
propagates the entering coordinate uncertainty using enclosed Jacobian bounds;
128-bit outward dyadic arithmetic retains rounding uncertainty. Whole-time
regularity, local truncation and propagated error are separate obligations.
The analytic `atan(u)/u` enclosure retains its zero-safe series for constant
intervals inside `[-1/2,1/2]`. Outside that interval, Taylor evaluation requires
the constant interval to exclude zero and encloses the same function through
`atan(u)` and interval division. An interval outside the series domain that
also contains zero is unavailable. This extends the proof arithmetic's
admitted domain without changing the relational pressure law, phase chamber
or evolution equation. It introduces no additional evolution. Frozen source
and result bundles retain their original provenance.

`RelationalTransitCertificate.admitted` requires the entire requested horizon
to be validated and the entire final box to lie in a selected protected
rectangle, with joint-storage upper bound strictly below `7*beta`. The
`requested_sector` policy accepts integer `1`, `0`, `-1` or `None`; Booleans,
floats and other values are rejected. Its default `1` requires the positive
rectangle `(2*pi/3,pi) x (0,pi/2)`. Zero requests the consensus rectangle
`(-2*pi/3,2*pi/3) x (-pi/2,pi/2)`, and `-1` requests the reflection of the
positive rectangle. `None` accepts any one of these disjoint rectangles.
The point and interval capture owners share their affine margin definitions.

This sufficient endpoint gate invokes the existing protected-basin theorem.
Only full admission sets `target_sector` to the limiting sector actually
certified: consensus is the valid integer `0`, not an unavailable result.
`requested_sector` is a caller's admission policy; it neither changes the
trajectory nor establishes its current winding or limiting target. Endpoint
centers, sampled storage or floating winding alone cannot pass the gate.
`initial_winding_zero` separately
certifies a sufficient zero-winding condition on the original lifted phase
state; it is not a gate for a general transit certificate. A claim of entry
from winding zero requires both this flag and complete transit admission.

An unresolved proof step stops the calculation and retains the accepted
`steps`, `validated_horizon`, their final `endpoint`, the attempted
`failed_tube` and explicit `unavailable_reasons`. A fully validated horizon
can also return unavailable when the whole endpoint fails the sufficient
capture gate. Unavailable means the requested proof was not obtained, not that
the ideal trajectory fails to recover. No limiting target is assigned then.
The report preserves `initial_box`, requested clock/order policy,
`endpoint_storage` and enclosure-method scope. All three candidates' interval
margins appear in `candidate_rectangle_margin_bounds`, in sector order
`1,0,-1`. A matching strict rectangle supplies `rectangle_kind` and
`rectangle_margin_bounds`, even if another obligation leaves the overall
report unavailable; without a match they are `None` and empty respectively.
`positive_rectangle_margins` remains the positive candidate's margins for
compatibility, including when consensus or negative capture is requested.
These geometric observations do not override horizon, storage or initial-law
admission failures, which always leave `target_sector=None`.

`relational_report_to_dict` exports the full nested evidence with exact
fraction endpoints and validates node/cycle labels through `initial`.
Projection is detached evidence, not a restart format or provenance
authentication. Successful admission concerns this conditional ideal ODE;
it supplies no guarantee for subsequent finite engine execution and leaves
every original frozen experimental verdict unchanged. See the
[SDK usage](../guides/REGIONAL_AND_RELATIONAL.md#validate-continuous-transit-to-a-protected-basin).

### Unequal reflected source/receiver continuous proof

`certify_relational_reflected_transit(initial_box, model=..., horizon=...,
time_step=..., order=...)` in
[`relational_reflected_transit.py`](../../src/tnfr/physics/relational_reflected_transit.py)
accepts exact interval coordinates `(p,r,P,R,a,b,A,B)`. It declares two unit
C5 rings with matching adjacent bridges, unit held capacities, exact joint
reflection and no forcing or events. It requires the explicit regular model.
The source and receiver remain independent; no live graph is inspected,
projected onto symmetry or advanced. The
[derivation](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regular-seeded-continuous-response)
owns reconstruction, the complete law and the ideal initial-value interpretation.

The private reflected-flow owner evaluates all six independent nodal
resultants, including the central rows, with shared interval/jet principal
arguments. Positive-real resultants use the removable-axis `atan(u)/u`
metric formula; sign-separated imaginary charts support negative real parts.
Zero/branch ambiguity rejects the box. A failed sufficient enclosure is not
a proved singularity. The shared Picard/Taylor/comparison kernel retains whole
tubes, domain margins, propagated initial uncertainty and Taylor remainders.
The existing copied-ring certificate retains its separate four-coordinate,
positive-resultant contract.

The certificate's `admitted` flag means only that the requested finite horizon
is enclosed. It separately retains winding flags, integrated loss, storage and
remaining budget relative to the two acute twists. A positive remaining budget
does not admit capture. A negative upper budget excludes that future target
under the same storage-loss law. Interval loss integration and direct storage
are intersected; inconsistent enclosures reject rather than being clamped.
The first unresolved tube is retained without automatic step refinement.
`to_dict()` reuses exact SDK dataclass projection, preserving rational bounds;
it does not authenticate publicly constructed certificates.

The [frozen producer](../../benchmarks/relational_seeded_response.py) separates
preparation from response, retains source archives and never overwrites a
record. A separately identified numerical correction retains its prior-record
hash and cannot be described as an independent blind replication. This proof
API is not a new operator, event law or SDK graph-projection workflow.

Its `target-budget` study retains the same ideal initial state and law, starting
at zero with default total horizon one. An explicit exact rational `horizon`
can declare a separate bounded continuation (at most 256 steps); it must match
at preparation and evaluation. The short study's horizon is fixed.
The protocol declares the strict sign test
before evaluation. `passed` means full-horizon admission, the declared endpoint
width and a resolved sign, whether positive or negative; it is not a formation
verdict. `target_budget_verdict` describes the declared endpoint, while
`budget_sign_at_validated_horizon` preserves partial evidence. The first
certified negative budget remains a scoped future exclusion even if a later
tube or endpoint is unresolved. False winding flags mean preservation was not
certified; they do not prove a change. No retry or horizon extension is implicit.

`certify_relational_reflected_barrier(initial_box, model=...)` in the same
proof owner is a separate static observer. It admits the regular reflected
lift, computes storage from the state and checks the sharp separating surface
`a+A=4*pi/3`, whose storage is at least `7*beta`. Strictly lower storage on
the lower side excludes subsequent two-acute-winding-`+1` entry under the
same fixed-support unforced law. Its `obstructed` flag is a sufficient
conditional theorem; `unavailable` does not establish reachability. It exposes
energy and separator margins, domain evidence and exact JSON projection.
The [collective energy proof](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#reflected-collective-energy-barrier)
distinguishes this transition barrier from the smaller target minimum. The
observer neither changes live state nor supplies convergence or future
regularity. Applying it retrospectively does not rewrite a frozen verdict.

### Ideal reflected equilibria and full-network stability

`certify_relational_reflected_equilibrium(family, model=..., orientation=1)` in
[`relational_reflected_equilibria.py`](../../src/tnfr/physics/relational_reflected_equilibria.py)
encloses one named analytic state: `consensus`, `aligned_twist`,
`aligned_saddle` or `opposite_twist`. The latter three have orientations `+1`
and `-1`; consensus has one representative. This stability certificate requires
the explicit regular model and positive EPI weight, phase weight and storage
scale. Zero dissipation is outside its recovery contract.

Coordinates enclose the named ideal point, not a box whose every state is an
equilibrium. The opposite branch uses the unique cubic root in `(1/8,1/7)`
and bounded cosine-based angle isolation. Source rates enclosing zero are a
consistency check; the [analytic classification](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#reflected-regular-equilibria)
establishes existence and completeness within the stated reflection lift.
No graph is recognized, rounded or projected into an equilibrium.

The report includes exact-coordinate bounds, storage, three edge-stiffness
classes, winding, domain margins, full phase-Hessian inertia and joint quotient
mode counts. Inertia is ordered positive/negative/zero, and quotient modes
stable/unstable/neutral. The full ten-node phase space has one common phase
zero mode; the full twenty-coordinate state has two common-offset zero modes.
The quotient counts exclude those two offsets. Stability is proved using the
full Hessian and symmetric quadratic pencil, including perturbations outside
reflection. `to_dict()` preserves exact bounds through shared SDK projection.
Existence and local recovery do not certify formation or global regularity.

### Global regularity evidence and boundary-access limits

`observe_relational_regularity(graph, model=...)` in
[`relational_regularity.py`](../../src/tnfr/physics/relational_regularity.py)
requires an explicit regular model and captures one detached admitted field.
It encloses storage from exact represented forms and certified trigonometry.
If its upper bound is strictly below `beta*max(2,minimum_degree)`, the
[continuous theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#regular-domain-continuation-and-boundary-access)
guarantees future regularity under held unit support, capacities and model,
without forcing/events. The report gives conservative positive metric and
excluded-ray-distance lower bounds. Otherwise status is
`storage_test_unresolved` and those future margins are `None`; this does not
establish a singularity, instability or failure of a different certificate.

The report separately bounds instantaneous resultant derivatives using the
captured numerical phase rates. Those are exact supplied directions with
outward trigonometric evaluation, not enclosures of ideal ODE rates. The
instantaneous speed bounds cannot be multiplied by an arbitrary duration to
certify a future trajectory. Global regular continuation itself promises
neither convergence, continued acute admission nor finite Euler stability.

`certify_relational_reflected_boundary_exit(model=..., form_amplitude=1)` in
[`relational_reflected_boundary.py`](../../src/tnfr/physics/relational_reflected_boundary.py)
instead evaluates a named ideal boundary family. The amplitude is an exact
positive integer or Fraction; the model is explicitly regular. Its endpoint
is not admitted for native execution. The report encloses limiting consumed
rates, resultants and derivatives, storage/loss, the exact `7*beta` gap and
the path-dependent full-state phase-rate limit. A positive gap proves the
counterexample also lies below that barrier.

The [local backward-flow argument](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#reflected-boundary-exit)
proves existence of nearby acute initial states reaching zero in finite time.
It supplies no chosen initial sample, exit-time estimate or prediction for
the frozen formation experiment. The smooth auxiliary reflected field is
not a full-state continuation rule. Six-resultant admission remains enforced;
the full vector field has no continuous extension at this endpoint.
Both reports use shared exact SDK projection through `to_dict()` and execute
no trajectory, event, reset or model change.

### Retained formation robustness

[`audit_relational_formation_robustness`](../../src/tnfr/research/relational_formation_robustness.py)
reads the original continuous-transit report and its sibling protocol/source
archive. It checks their retained digests and archived sources before using
the whole-time tubes. It does not run the producer or evolve a graph.
The physical-flow formulas retain their archived syntax. An audit-only
four-coordinate comparison oracle is also checked against the archived syntax;
the current shared comparison kernel must match its exact matrix on every
consumed inflated tube. A refactored delegate can therefore retain the frozen
premise, while a changed comparison formula is inconsistent. The audit records
the shared comparison owner's source digest without changing the archive.

The [formation-robustness theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-formation-law-robustness)
uses the exact same preparation, support, held unit capacities, coefficients
and structural clock. Each retained four-coordinate tube is enlarged by
`1/1024`. Shared interval geometry and reference Jacobians establish a regular
corridor, a cumulative logarithmic norm bound and a protected endpoint. The
comparison controls a changed law throughout the corridor, rather than
relabelling the original tube as its trajectory enclosure.

`RelationalFormationRobustnessAudit` retains reference provenance, exact
arithmetic evidence and separate `RelationalFormationLawBound` records for
the rho and eta comparisons. Its `admitted` property requires all evidence
checks to pass. `to_dict()` preserves exact fractions; it does not authenticate
a publicly constructed report. Missing records and unresolved proof bounds
remain unavailable; inconsistent record identities reject the evidence before
derivative calculation. Neither case reports failed physical formation.

The two parameter caps apply separately, with the other correction absent.
They are conservative sufficient mathematical bounds, not optimal tolerances,
physical constants or new modes of `RelationalExchangeModel`. Capture uses
the reflected `R+` region with storage below `7*beta`; the saved endpoint is
not admitted to the stricter full-state acute-energy region. Exact symmetry
preservation by the compared laws is essential to this quantitative result.
The original reference certificate and all frozen verdicts remain unchanged.

## Normalized-sine comparison and proof tools

All APIs in this section concern the separately identified smooth sine law.
They share its capture and rate/work kernels; none selects it in the native
executor or the operator runtime. Loss, capacity and preparation hypotheses
remain specific to each assessment.

<a id="sine-chained-report-admission"></a>
### Admission when chaining sine reports

Report-consuming readers that link here re-admit the declared inputs needed
by their calculation, reconstruct consumed derived quantities and apply their
own theorem hypotheses. Cached bounds, degrees, rates or success flags cannot
replace those inputs. Each reader's section identifies its reconstruction
inputs and any stronger domain or numerical budget.

For complete captured or relative sources,
[`_sine_admission.py`](../../src/tnfr/physics/_sine_admission.py) checks the
declared sine law, simple connected unit support, node-coordinate order,
degree/neighbor consistency, finite form/phase/capacity vectors and nonnegative
residual radii. Neighbor-row order does not change support equivalence.
Authoritative stored model coefficients are admitted without a second
normalization: exact rationals retain their values; negative loss, Boolean
or nonfinite coefficients reject. Admission precedes coefficient equality,
including comparisons between a pattern and its full forecast.

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
metadata in the [observation owner](../../src/tnfr/physics/relational_sine_observation.py).
Capacity inference also uses its supplied acceleration channels, with omitted
channels unavailable. The [forecast owner](../../src/tnfr/physics/relational_sine_forecast.py)
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
[`relational_sine_comparison.py`](../../src/tnfr/physics/relational_sine_comparison.py)
captures the same signed scalar, phase, capacity and unit-support contract,
but computes the explicitly distinct law
`normalized_sine_reciprocal_exchange`. The reference model must be explicitly
regular; it supplies the shared effective coefficients and storage scale,
not the candidate's phase-domain admission. The comparison uses
`s_i=sum_j sin(theta_j-theta_i)/(pi*d_i)` in form pressure and
`theta_dot_i=w*nu_i*(Bx)_i/(beta*pi*d_i)`. It is defined at zero and branch
resultants, including states rejected by the native model.

The [mathematical owner](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#global-closure-pressure-comparison)
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

<a id="sine-regional-transfer"></a>
### Regional form transfer and supplied increments

`comparison.regional_transfer(region=...)` reuses a captured
`SineExchangeComparison`, without a graph reread. All held capacities must be
strictly positive. The fixed-support weights are `rho_i=d_i/nu_i`; the report
uses weighted form sums, not arithmetic means or physical mass. The
[boundary identity](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-autonomous-regional-transfer)
holds for both zero and positive form loss in this law. It does not transfer
to native Arg pressure or state-dependent reciprocal mobility.
Both readers use [chained-report admission](#sine-chained-report-admission).
Regional accounting rebuilds rates and currents from the exact state; increment
accounting needs only admitted forms, weights and supplied signed increments.

`SineRegionalTransfer` retains the source, supplied region and its complement,
weights, weighted form sums and the inward cut current. The diffusion and sine
contributions are separate. Internal edges cancel algebraically; each cut
edge contributes `e*(x_out-x_in)+(w/pi)*sin(theta_out-theta_in)` to the region.
Direct weighted sums of the rebuilt nodal rates and their cut residuals are
computed independently. `global_weighted_form_conserved` records the conditional
identity; an interval residual containing zero is a numerical consistency
check, not its proof. A regional current does not close the region's evolution
without the consumed boundary state or certify persistent identity.

No pair midpoint or nonzero regional resultant is required by this cut
reader. The [zero-pair-resultant theorem](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-zero-resultant-restoration)
shows why cancelled collective form transport can coexist with internal
phase motion. Its exact antipodal preparation is symbolic; a graph storing
represented radians remains its own source. The cut reader reports current,
not a future activation time. With the theorem's zero loss, node-relative
`resultant_kinematics()` bounds nodal form acceleration after the declared
capacity/degree and phase-weight scaling; it does not
report that pair's phasor derivative or supply a midpoint at cancellation.

`comparison.assess_form_increment(increments=...)` accepts a complete ordered
vector of finite signed increments in captured node order. It does not infer
unobserved changes or silently fill missing nodes. `SineFormIncrementAssessment`
retains exact weights, increments and before/after weighted form sums.
`closed_flow_endpoint_obstructed` is true precisely when the weighted change
is nonzero. Zero change leaves reachability unresolved;
`endpoint_reachability_certified` remains false. No phase endpoint, storage
budget, runtime event or trajectory is supplied by this necessary condition.
The test concerns a full endpoint in the same form chart; it does not establish
an obstruction on a quotient that discards common form origins.

The pair-Emission report's `form_increment(outcome=...)` accepts
`first_member`, `second_member` or `whole_pair` and delegates those actual
clipped/rounded proposals to the same increment owner. Other nodes have zero
increment because these three declared actions leave them unchanged. A
positive AL-only increment fails the global invariant; an increase in one
region with compensating change elsewhere need not fail it. Saturated or
rounded no-ops are not reported as positive injections.

Owner schemas are `tnfr.relational-sine-regional-transfer.v1` and
`tnfr.relational-sine-form-increment.v1`. Generic SDK export delegates their
payloads through `tnfr.relational-report.v1`. Source provenance and label
validation are retained. No observation mutates the graph, installs an
activation rule or identifies form with an independently measured quantity.

### Sine-law prospective sampling budgets

`bound_sine_sampling_smoothness(*, reference_model, form_diameter_bound,
capacity_ceiling, window_start, window_end)` in
[`relational_sine_sampling.py`](../../src/tnfr/physics/relational_sine_sampling.py)
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
[`relational_sine_mediation.py`](../../src/tnfr/physics/relational_sine_mediation.py).
It uses the already captured `SineExchangeComparison` without reading the
graph again. The supplied mediator must exist and have at least two ports.
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
[causal memory identity](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#causal-sine-environmental-pressure).
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
[receiver passage balance](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-receiver-port-passage).
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
[`relational_sine_observation.py`](../../src/tnfr/physics/relational_sine_observation.py)
implements the
[conditional observability admission](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-hidden-state-observability).
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
[sine observation owner](../../src/tnfr/physics/relational_sine_observation.py).
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

The [acceleration identity](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-hidden-capacity-observability)
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
[prospective protocol](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-prior-reserved-forecast)
separates prior derivatives, joint inference, issued prediction and reserved
source response. Its frozen-mediator control is a counterfactual, and its
synthetic derivative evidence is not a sampled laboratory measurement.

### Relative sine patterns and moving references

`bound_relational_sine_pattern(graph, *, reference_node, reference_model,
form_error_bounds, phase_error_bounds)` in
[`relational_sine_pattern.py`](../../src/tnfr/physics/relational_sine_pattern.py)
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
The [mathematical owner](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-relative-pattern-state)
defines pattern identity, winding limits and a sufficient analytic recovery
criterion. This report does not itself certify a recovery basin.

`pattern.forecast(observation_time=..., end_time=..., time_step=..., order=6)`
uses [chained-report admission](#sine-chained-report-admission) and reconstructs
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

### Whole-set sine cycle recovery

`certify_sine_cycle_recovery(source, *, cycle, winding, radius,
phase_turns=None)` in
[`relational_sine_recovery.py`](../../src/tnfr/physics/relational_sine_recovery.py)
accepts a `SineRelativePattern` or `SineRelativeForecast`. Both expose the
same operation through `certify_cycle_recovery(...)`. It observes the supplied
state set without running a solver or changing a graph.

Cycle recovery, general-target recovery, conservative identity and sector
capture share [chained-report admission](#sine-chained-report-admission).
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
[whole-set proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-cycle-recovery):
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

The [complete-network proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-interacting-recovery)
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
[`relational_sine_formation.py`](../../src/tnfr/physics/relational_sine_formation.py)
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

The [proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-formation-eligibility)
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
The [directional proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-balanced-formation)
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
[global auxiliary-function theorem](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-maintained-target-obstruction).
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
[internal-form analysis](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-internal-form-geometry)
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
[chained-report admission](#sine-chained-report-admission), it rebuilds the
preparation before decomposing full form into donor-reflection-even and odd
parts. Both eleven-coordinate vectors are retained; their storage and
dissipative norms use the actual support, degrees and capacities.
`source_coordinates` has order
`(H, D0-H, s1, s2, a1, a2)`, as defined in the
[source theorem](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-source-receiver-excitation).
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
evolution parameters. The [regional proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-weighted-receiver-exclusion)
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
Under [chained-report admission](#sine-chained-report-admission), static
symmetry flags, conserved means, mobility and phase-storage constants are
reconstructed from the admitted preparation and shared mathematical owners.
Cached equilibrium, silence or barrier fields cannot supply an exclusion.
The [transfer theorem](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-receiver-transfer-admission)
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
[port-work and order proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-receiver-port-passage)
has broader explicitly stated premises; this adapter retains its own scope.

<a id="sine-donor-well-retention"></a>
### Frozen receiver barrier exclusion

[`relational_receiver_barrier.py`](../../src/tnfr/research/relational_receiver_barrier.py)
prepares and assesses one full eleven-node `H=2` source under the smooth sine
law. `prepare_receiver_barrier()` declares its exact phase turns, held
capacities, original clock, `T=32`, step `1/8` and Taylor order 16.
`evaluate_receiver_barrier()` delegates once to `bound_sine_flow`; it supplies
no independent solver or altered pressure. All 23 augmented coordinates are
retained. The held last capacity is an uncertainty coordinate, not another
evolving physical variable.

`assess_receiver_barrier_forecast()` checks the frozen declaration and complete
step coverage. Its [conditional theorem](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-localized-receiver-exclusion)
requires receiver potential strictly below `7/2` in **every whole-time tube**
and full twelve-edge endpoint storage strictly below `7/2`. Together, these
exclude the receiver potential barrier for all future time. Safe endpoints
alone, receiver-only endpoint storage or an unfinished horizon cannot pass.
An upper bound failing either comparison is unresolved, not a proved passage.
No all-time winding constancy or selected donor endpoint follows.

The [producer](../../benchmarks/relational_receiver_barrier.py) freezes a
protocol and source archive before evaluation, refuses overwrites, and retains
evaluation errors and unsuccessful responses. Post-evaluation source, runtime
and archive checks guard consistency. Public forecast records and archive
hashes do not independently authenticate execution or chronology. This is a
source-specific research certificate, not a native runtime event or new law.

### Donor-well recovery from the full prepared form state

`SineMediatedFormation.donor_well_retention()` returns the separate
`SineDonorWellRetention`, reusing the original exact donor-twist/flat-receiver
preparation and the [auxiliary-well theorem](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-donor-well-retention).
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
replace that calculation. Its [proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-donor-dissipative-capture)
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

[`classify_c5_sine_critical_set`](../../src/tnfr/physics/phase_cycle_geometry.py)
accepts exact support geometry and two ordered, labeled C5 cycles. The graph
must consist of those disjoint rings and one remaining intermediary joined
to one node of each ring, with no additional edge. It retains the supplied
node order and ring orientations; only a common phase origin is removed.

The [completeness proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-eleven-node-asymptotic-equilibria)
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
[`assess_sine_asymptotic_equilibria`](../../src/tnfr/physics/relational_sine_equilibria.py).
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
[composition proof](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-bridge-tree-composition)
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
The [stability proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-eleven-node-equilibrium-stability)
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
in [`phase_cycle_geometry.py`](../../src/tnfr/physics/phase_cycle_geometry.py)
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
[joint-compatibility theorem](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-cycle-sector-compatibility)
owns the full criterion, complete-law transfer and counterexamples.

### Target-free acute-sector capture

`certify_sine_sector_capture(source, edge_turn_offsets=...)` in the shared
[`relational_sine_recovery.py`](../../src/tnfr/physics/relational_sine_recovery.py)
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

The reader uses [chained-report admission](#sine-chained-report-admission) and
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
[theorem](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture)
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
in [`relational_sine_entry.py`](../../src/tnfr/physics/relational_sine_entry.py)
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

The [transit theorem](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-prepared-sector-entry)
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
[`relational_sine_reduction.py`](../../src/tnfr/physics/relational_sine_reduction.py)
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

The [comparison theorem, Section 29](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-controlled-slow-phase)
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
in [`relational_sine_reduction.py`](../../src/tnfr/physics/relational_sine_reduction.py)
returns `SineSlowCapture`. `SineRelativePattern.certify_slow_capture(...)`
delegates to this owner. An existing `SineSlowPhaseBound` also exposes
`.certify_capture(target_phase_turns=...)`, which recomputes preparation and
comparison bounds from its `source` and `slow_time` under
[chained-report admission](#sine-chained-report-admission). The same
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

The [handoff theorem, Section 30](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-slow-phase-capture-handoff)
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
[`relational_sine_budget.py`](../../src/tnfr/physics/relational_sine_budget.py)
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

The [all-time theorem](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-budget-consensus)
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
[C5 usage example](../guides/REGIONAL_AND_RELATIONAL.md#certify-an-entire-phase-flat-budget-family).

<a id="sine-cycle-symmetry"></a>
### Full-state symmetry and an oriented cycle

`assess_sine_cycle_symmetry(source, permutation_indices=..., cycle=...)` in
[`relational_sine_symmetry.py`](../../src/tnfr/physics/relational_sine_symmetry.py)
returns `SineCycleSymmetryAssessment` for one supplied node permutation under
the complete unforced sine law.
`source.assess_cycle_symmetry(...)` delegates to the same owner. The source
must be an exact `SineExchangeComparison` and uses
[chained-report admission](#sine-chained-report-admission).
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
The [proof](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-equal-budget-preparation)
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
[equal-budget example](../guides/REGIONAL_AND_RELATIONAL.md#distinguish-equal-budgets-by-preparation-symmetry)
compares this obstruction with the separately certified acquisition source.

### Detached stationary sine mediation

`bound_relational_sine_mediation(graph, mediator=..., reference_model=...)`
in [`relational_sine_mediation.py`](../../src/tnfr/physics/relational_sine_mediation.py)
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
[mathematical owner](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-mediator-common-geometry)
separates conditional storage, finite-capacity memory, the hidden oscillator
and the hypotheses of a controlled approximation.
The [finite-state audit](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#finite-environment-reuse-audit)
explains why an unavailable stationary direction need not invalidate the
uneliminated sine field, and why the default hidden tangent does not oscillate.
The [autonomous path proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#autonomous-path-cancellation)
also gives a finite cancellation crossing with all three nodes active:
zero tracking defect before the crossing and any finite hidden capacity do
not guarantee that the hidden state continues to minimize storage afterward.
This is a conditional continuous theorem, not a result inferred from the
stationary report or an executed trajectory.

<a id="sine-resonance"></a>
### Sine-law resonance and declared input/output response

[`relational_sine_resonance.py`](../../src/tnfr/physics/relational_sine_resonance.py)
owns static tangent assessments for the complete normalized-sine law.
`assess_sine_cycle_resonance(model=..., node_count=..., mode_index=...,
capacity=..., winding=0)` declares a complete unit cycle with uniform form,
exact target phase turns `winding*j/node_count`, common held positive capacity
and an explicit regular reference model with positive EPI loss weight.
The reference model supplies normalized coefficients, not native Arg rates.

Counts, winding and mode must be exact nonboolean integers, with `n>=3`,
`4*abs(winding)<n` and `1<=mode_index<n`. Shared represented-real admission
retains exact positive rational capacity. The analytic template allocates no
graph and performs no mode search or floating eigensolve. Its modal rows
differentiate the shared sine field; bounds use mathematical pi and outward
interval arithmetic. This is not admission of a live observed cycle.

`SineCycleResonance` retains the template, coefficient ratio
`beta*(e/w)^2`, modal matrix, damping ratio and separate pole/phase-peak
classifications. Its input is a unit Euclidean real Fourier mode added to
the form-rate row. Form amplitude has transfer `s/D(s)`, phase amplitude
has `b*r/D(s)`, with `D=s^2+e*r*s+a*b*cos(alpha)*r^2` and
`r=capacity*lambda/2`. The work-conjugate output is `lambda*X`.
Natural frequency equals the form/work gain peak; it is not necessarily
an observable free-oscillation frequency. The exact work peak gain is
`2/(e*capacity)`. Positive-frequency phase gain and complex poles require
different inequalities.

`report.gain(angular_frequency)` admits a finite nonnegative exact or represented
real frequency in the model clock and encloses all three gains separately.
Under [chained-report admission](#sine-chained-report-admission), it rebuilds
and retains the mode from its law, model, cycle size, winding, mode index
and capacity.
Unresolved denominators or classification signs retain unavailable reasons;
an interval containing zero does not establish a zero gain, critical damping
or absent susceptibility. At zero frequency the form/work transfer is zero.
Invalid domains reject rather than being coerced into an admissible mode.

`certify_sine_recovery_resonance(recovery, port=(i,j))` reuses a shared
full-support sine pattern/cycle recovery report through
[chained-report admission](#sine-chained-report-admission).
The target must meet that owner's sufficient acute criticality, positive
capacity, support and recovery hypotheses. With `K=diag(nu_i/d_i)` and
`p=e_i-e_j`, the declared probe is `f=K*p`, the input term is `f*u(t)`
and the conjugate output is `y=f^T*L*delta_x`. The theorem establishes an
interior gain maximum and the bound `(K_ii+K_jj)/e`, without selecting its
frequency. All original environmental coordinates remain in the target;
the statement concerns its tangent, not an observed transient's transfer.
Uncertain held capacities describe a family of systems. If a selected port's
capacity is uncertain, the corresponding probes/readouts also form a family,
rather than one calibrated fixed physical port.

Reports use exact shared JSON projection through `to_dict()` and the SDK
atomic writer for export. `relational_report_to_dict` also accepts the
resonance report types and delegates projection to this owner while preserving
the generic SDK envelope. Export does not authenticate a manually constructed
dataclass. No report installs forcing, executes RA, adds an edge, advances a
solver or demonstrates nonlinear formation, autonomous excitation or physical
resonance. The [foundation](../../theory/nodal/RESONANCE_FOUNDATIONS.md)
owns the full-network proof, gain ceiling, counterexamples and clock scope.

<a id="sine-mediated-response"></a>
### Remote sine response through a retained intermediary

`assess_sine_mediated_response(recovery, donor_cycle=..., receiver_cycle=...,
mediator=...)` in the same [resonance owner](../../src/tnfr/physics/relational_sine_resonance.py)
uses [chained-report admission](#sine-chained-report-admission) for its supplied
recovery and requires exactly two disjoint ordered
five-node cycles and one intermediary. Each cycle's first node is its sole
contact with the intermediary. All eleven nodes and twelve edges must be
present, with no additional edges. Capacities must be exact held positive
scalars; uncertain capacity families are outside this adapter.

Shared recovery admits the exact acute critical target, independently of the
nominal observed perturbation. Both bridge target gaps must be zero modulo a
turn. Those hypotheses force uniform acute cycle twists; winding -1, 0 and 1
are permitted separately on each cycle. No graph mutation or target fitting
occurs. The declared clock and normalized coefficients are retained.

The donor direction is `f=K*p_D`, where `p_D` is one at the donor port,
minus one quarter at its other four nodes, and zero elsewhere. The receiver
contrast `r_R` has the same weights on the receiver. It observes phase or
form relative to the other receiver nodes, using target-relative phase lifts.
It is not a structural-work-conjugate readout. The report concerns the
exact target's tangent, not a time-invariant approximation inferred from the
observed transient.

`SineMediatedResponse` retains `form_markov_bounds` and `phase_markov_bounds`
for derivative orders 0, 1 and 2, and the corresponding full-state derivative
vectors. These are `C*J**j*(f,0)`, without Taylor factorials. They describe
both the autonomous tangent response to initial form `epsilon*f` and the
impulse kernel for hypothetical form-rate input `f*u(t)`. A constant input
switched on from zero has its corresponding first phase effect at order 3,
not order 2. No impulse or other forcing is installed.

Static bridge balance gives exact zero DC contrasts. The phase second
derivative is `phase_order_two_pi_numerator/pi`, which is strictly negative
by exact positive-capacity/coefficient admission. This sign remains distinct
from an outward enclosure that touches zero. The form coefficient can cancel.
`positive_frequency_phase_peak_certified` and
`phase_impulse_sign_reversal_certified` apply the
[remote-response proof](../../theory/nodal/RESONANCE_FOUNDATIONS.md#mediated-resonance):
there is a finite-frequency phase-gain maximum and the autonomous tangent
phase contrast becomes positive after its initial negative response.
Peak frequency, reversal time and a quantitative nonlinear amplitude bound
remain unavailable. Neither positive-realness nor the collocated gain ceiling
is assigned to this remote phase output.

The retained 2-by-2 contact blocks use form-then-phase order. They define the
intermediary generator and the donor-to-intermediary and intermediary-to-receiver
couplings. `memory_kernel_at_zero_bounds` encloses their product. The complete
kernel also depends on the intermediary exponential, hidden initial state and
visible history; no instantaneous direct edge replaces it.

The two `*_silent` flags certify sufficient tangent reflection symmetries by
checking support, exact target cosine geometry and equal capacities on reflected
pairs. A false flag means this proof is unavailable, not that transmission has
been demonstrated. Finitely many zero Markov entries never prove permanent
silence. Invalid support, incompatible recovery or inexact capacities reject.
`to_dict()` uses schema `tnfr.relational-sine-mediated-response.v1`; the generic
SDK export delegates to it and retains the generic envelope. This is a static
theorem report, not a trajectory, formation event or physical observation.

<a id="sine-pair-pulse"></a>
### Autonomous nonlinear pulse on a complete sine pair

`assess_sine_pair_pulse(graph, reference_model=...)` in the shared
[resonance owner](../../src/tnfr/physics/relational_sine_resonance.py)
reuses sine state admission, rates and work. It requires the complete two-node,
single-unit-edge support, held nonnegative capacities and absent inputs/events.
The model supplies the coefficients, storage scale and clock. The reader
never changes a positive loss coefficient to zero. It does not execute the
native argument-pressure runtime or install a new pressure law.

In captured node order, `u=x0-x1`, `delta=theta1-theta0` and
`sigma=nu0+nu1`. Storage is `E=u**2/2+beta*(1-cos(delta))`;
`normalized_energy_bounds` encloses `E/beta`. Equal captured phase values
also give `exact_energy`, preserving exact rational decisions at zero energy
and the separatrix. Other energies use outward enclosures; unresolved signs
remain unavailable rather than being rounded into a certificate.

`SinePairPulseAssessment` separates the following cases:

- `libration_certified`: supplied `e=0`, `sigma>0` and certified
  `0<E<2*beta`. The full nonlinear continuous state is periodic, including
  a pair with exactly one zero capacity.
- `dissipative_recurrence_excluded`: `e>0`. Nonstationary recurrence is
  excluded even if the initial loss happens to be zero.
- `frozen` or `equilibrium`: respectively both capacities vanish, or the
  captured forms and phases coincide. These checks certify stationary cases;
  they do not enumerate every circular representation of an equilibrium.
- `separatrix_out_of_scope`, `rotation_out_of_scope` or
  `energy_classification_unresolved`: no nonstationary libration certificate.

`small_amplitude_period_bounds` encloses `T0=2*pi**2*sqrt(beta)/(w*sigma)`
when `e=0` and `sigma>0`. For certified libration, `period_bounds` uses
the exact elliptic-integral inequalities `T0<T<=T0/sqrt(1-E/(2*beta))`.
The period refers to the model clock, not an identified laboratory unit.
A certificate of period existence can remain true while its numerical bound
is unavailable near the separatrix; `period_unavailable_reasons` retains
that distinction. Relative-rate bounds and `initial_continuous_loss` come
from the shared complete-law reader, not a fitted oscillation.

`to_dict()` uses schema `tnfr.relational-sine-pair-pulse.v1`; the generic
SDK report export delegates to the same projection. The detached assessment
does not advance time, modify a graph, certify an Euler trajectory, select
an attracting amplitude or generate motion from equilibrium. The
[proof](../../theory/nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
also establishes the broader finite closed positive-loss sine obstruction.
Zero loss and a nonzero initial preparation remain explicit premises,
not a derived fundamental selection or physical identification.

<a id="sine-conservative-path-memory"></a>
### Conservative path memory and nonlinear regional transfer

`assess_sine_path_memory(graph, reference_model=..., mediator=...,
consensus_form=0, consensus_phase=0)` uses the same
[resonance owner](../../src/tnfr/physics/relational_sine_resonance.py).
It requires a complete unit P3 path, the correct middle node, common positive
held capacity and an explicitly zero EPI loss weight. Shared capture owns
scalar/alias, support and absent-source admission. Positive loss, unequal or
zero capacity and extra support reject; this adapter never changes the law.

The `SinePathMemoryAssessment` report separates two uses of the captured state:

- `nonlinear_edge_storage_bounds` and `nonlinear_edge_rate_bounds` describe
  actual incident-edge storage and its instantaneous work in the original
  structural clock. Endpoint order follows graph capture; `path` and
  `path_indices` retain left, middle, right. `nonlinear_transfer_bounds`
  encloses the left edge's rate, whose exact negative is the right edge's
  rate. The two `nonlinear_edge_rate_residual_bounds` compare these formulas
  with the independent chain-rule work. Total continuous loss is exactly zero.
- `coordinate_memory` and `linear_observation` concern the separately
  declared uniform-form/uniform-phase reference tangent. The supplied origins
  are admitted through shared exact-or-represented-real admission. The report
  does not infer this target from the graph or linearize about a moving state.
  `initial_tangent_state` contains all form differences from the reference,
  followed by all chosen real phase-lift differences, in captured node order.
  No error estimate identifies its linear evolution with the nonlinear future.

The exact clock is `tau=w*t/pi`; `clock_rate_pi_numerator` and
`clock_rate_bounds` retain that conversion. In this clock, the consensus
generator is `[[0,-M],[M/beta,0]]`, where `M=nu*D^-1*L` uses the complete
path degrees. Its entries are exact rationals for admitted captured values;
mathematical pi has not been replaced by a floating-point coefficient.
The existing coordinate-memory owner partitions endpoint forms then endpoint
phases as visible, retaining middle form and phase as hidden.

With `omega_squared=hidden_frequency_squared=nu**2/beta`, the exact kernel is
`cos(omega*tau)*K0+sin(omega*tau)*K1/omega`. `K0` belongs to
`coordinate_memory.kernel_at_zero`; `K1=kernel_derivative_at_zero` is its
normalized-time derivative. Hidden initialization contributes the analogous
combination of `hidden_initial_forcing=B*h0` and
`hidden_initial_forcing_derivative=B*D*h0`. Omitting that term changes the
preparation. No kernel, exponential or memory convolution is numerically
propagated by this static report.

The shared minimal-observation construction requires all six tangent
coordinates to close the four endpoint observations. This is an all-state
linear statement, not a physical count of hidden constituents. The full
tangent has a return period enclosed by `full_tangent_period_bounds`,
`2*pi**2*sqrt(beta)/(w*nu)` in structural time. A particular state can be
stationary or have a shorter fundamental period; the exact
`tangent_motion_nonstationary` flag checks its generator action. Neither
that flag nor the period bound certifies a nonlinear pulse.

`to_dict()` uses schema `tnfr.relational-sine-path-memory.v1`; generic SDK
export delegates to this same projection. The
[proof](../../theory/nodal/RESONANCE_FOUNDATIONS.md#finite-conservative-memory)
derives regional transfer, the periodic memory and the obstruction to exact
irreversible finite tangent reduction. A finite-window damping approximation,
microscopic zero-loss selection and universal fractality remain separate
questions. No graph mutation, external drive or runtime-default change occurs.

<a id="sine-nonlinear-recurrence"></a>
### Full nonlinear recurrence: family admission and individual limits

`assess_sine_recurrence(graph, reference_model=..., energy_ceiling=...,
form_mean_bounds=(lower, upper))` uses the shared
[resonance owner](../../src/tnfr/physics/relational_sine_resonance.py).
It admits complete finite connected simple unit support with at least two
nodes, strictly positive held capacities, explicit zero EPI loss weight and
the shared absent-input/event premises. It applies to the full nonlinear
sine law, with real form and circular phase, without an equilibrium target
or small-amplitude assumption.

The caller supplies a strictly positive energy ceiling and a strictly
positive-width mean interval. Shared exact-or-represented-real admission
rejects Booleans, nonfinite values and unordered/invalid bounds. These values
select the analyzed family; they do not add parameters to the dynamics.
The family contains every state with `E<=energy_ceiling` and weighted form
mean in that interval, at the captured support, capacity and law coefficients.
It is not a Cartesian product of independently bounded phase and form data.

`SineRecurrenceAssessment` retains exact weights `rho_i=d_i/nu_i`, their
normalization and `weighted_form_mean`. Its
`weighted_mean_rate_residual_bounds` is computed from the shared form rates,
and `comparison.storage_rate` retains the computed work enclosure. The exact
divergence identity is `-e*sum(nu_i)`, hence zero in the admitted law. Interval
residuals checking zero do not by themselves prove either invariant.

The report uses `path_length_upper_bound=n-1` to enclose
`R=sqrt(2*(n-1)*energy_ceiling)` in `form_radius_bounds` and supplies the
family coordinate box `[lower-R,upper+R]` in `form_coordinate_bounds`.
This conservative connected-path bound needs no spectral computation or
all-pairs distance search. Compactness includes the phase torus, not arbitrary
unwrapped real phase lifts. Tiny exact positive capacities and slab widths
remain positive in admission even if an outward projection touches zero.

The report keeps three conclusions separate:

- `almost_everywhere_recurrence_certified` and
  `finite_positive_family_volume_certified` apply to the declared family in
  ambient form Lebesgue measure times phase-circle Haar measure.
- `snapshot_family_membership` is `inside`, `outside` or `unresolved`, with
  explicit reasons. Exact mean comparison and, for identical captured phases,
  `captured_energy_exact` can resolve boundaries that projected energy cannot.
  Otherwise overlapping energy bounds remain unresolved. An outside snapshot
  does not invalidate the separately admitted family theorem or inherit its
  form box.
- `snapshot_motion_status` is `stationary`, `nonstationary` or `unresolved`.
  A nonzero exact form Laplacian or a sign-resolved sine current proves motion.
  Uniform captured phases and zero form Laplacian certify a stationary case;
  these are sufficient checks, not an exhaustive equilibrium classifier.
  `individual_recurrence_status` is `trivial_stationary_recurrence` for that
  stationary case and `unavailable_for_chosen_state` otherwise.

In particular, confirmed family membership and known motion do not imply an
individual recurrence certificate. The
[proof and exact separatrix counterexample](../../theory/nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
explain why this distinction is necessary. No return deadline, exact period,
attracting amplitude or preserved pattern identity is reported. A finite
numeric grid or seed distribution does not automatically inherit a theorem
for absolutely continuous ambient preparations; fixed-energy and fixed-mean
surfaces need their own measure argument.

The [joint-episode theorem](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-joint-recurrent-episodes)
combines this family result with a separately proved open acquisition region
on the same doubled-C5 law. It establishes repeated finite matching episodes
for almost every preparation there. This reader alone does not admit that
region, infer an identity predicate, or certify any captured state's episodes.
Leaving a collective-coordinate chart is not itself loss of joint matching.

`to_dict()` uses schema `tnfr.relational-sine-recurrence.v1`; generic SDK
export uses the same projection. The reader executes no trajectory, changes
no runtime law and does not reinterpret a dissipative recovery report.

<a id="sine-conservative-identity"></a>
### Conservative cycle identity and recurrent families

`assess_sine_cycle_identity(source, cycle=..., winding=..., radius=...,
excess_ceiling=..., form_mean_bounds=(lower, upper), phase_turns=None)` in the
[shared recovery/geometry owner](../../src/tnfr/physics/relational_sine_recovery.py)
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

The [proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-conservative-identity)
uses conserved excess storage and the shared finite-volume recurrence
argument. It supplies no return time, single frequency, attraction, pattern
formation, coarse-scale closure or physical identification. A finite grid,
singleton or fixed-energy surface does not inherit an ambient
almost-everywhere result automatically.

`to_dict()` uses `tnfr.relational-sine-cycle-identity.v1`; the generic SDK
export delegates to the same exact projection and label admission. This
detached assessment evolves no graph and installs no new runtime law.

<a id="sine-replica-scale"></a>
### Inherited sine dynamics with both internal nodes retained

`assess_sine_replica_scale(graph, reference_model=..., pairs=...,
phase_turns=None)` in
[`relational_sine_scale.py`](../../src/tnfr/physics/relational_sine_scale.py)
is a detached observer of the complete conservative sine law. It consumes an
ordered partition into two-node pairs. Every fine node must occur exactly
once; there are no edges inside a pair, and each adjacent pair of blocks
must retain all four unit edges. The resulting base graph is connected.
Capacities must be held, strictly positive and equal within each pair;
different pairs may have different capacities. The model explicitly has
zero EPI loss weight and the existing absent-input/event premises.

The caller supplies consistent phase lifts, optionally corrected by integer
`phase_turns` in captured fine-node order. Each pair must admit a certified
half-difference of magnitude less than `pi/2`. This selects its nonantipodal
circular midpoint; no phase unwrapping is guessed. Common turn offsets cancel
before mathematical-pi enclosure. A failed chart test does not mean that the
underlying smooth sine field is undefined, nor is initial chart admission a
future trapping certificate.

At an antipodal pair the global phasor is zero and its angle is unavailable,
while the full sine law remains regular. Its magnitude can have a cusp
through the zero, so extending signed `cos(delta)` does not extend this
nonnegative midpoint-magnitude contract. Use the complete fine comparison
and regional current observation there; do not relax chart admission.

For each pair the report retains mean form/phase and their half-differences,
along with the complete captured fine comparison. These four coordinates
retain both original nodes. The internal phase half-difference `delta_i`
gives a resultant amplitude `R_i=cos(delta_i)`. The inherited phase-driven
mean form response contains `R_i*R_j`, while the mean phase row keeps the
original base Laplacian. The amplitude's derivative additionally consumes
internal form contrast. These are calculated effects of the retained state,
not tunable coupling strengths or controller scores.

The report compares averaged full fine rates, their factorized mean/internal
identities and the naive base-law rates. It also retains the complete storage
decomposition. Computed residual intervals check the implementation; the
[proof owner](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-inheritance)
establishes the all-state identities. The internal phase storage correction
need not be positive outside an acute coarse chart.

Exact synchronization requires zero internal form **and** phase differences.
That supplied invariant submanifold inherits the original nodal rows, capacity
and clock, with fine storage four times coarse storage. It is not attracting
synchronization or autonomous formation. A zero instantaneous coarse-rate
defect does not prove this membership or general closure: internal form can
subsequently create phase dispersion. Nearby equal-mean preparations can
already have different mean rates.

`SineReplicaScaleAssessment` separates `synchronized_submanifold_invariant`
from `source_in_synchronized_submanifold` and
`same_law_reduced_flow_certified_for_source`. The latter certifies the supplied
source's inherited reduced flow only on exact synchrony. The
`all_state_coarse_closure_obstructed` flag concerns a description using only
block means across the full admitted family, not the impossibility of every
specially prepared coarse trajectory or an augmented collective state.
`resultant_magnitude_bounds`, `resultant_magnitude_rate_bounds` and
`phase_current_factors` retain the derived internal effect separately from
`coarse_form_defect` and `coarse_phase_defect`. Intervals containing zero are
not automatically exact equalities.

The same report also carries an exact collective description of **unordered**
pairs. Alongside mean form `X` and midpoint phase `Theta`, it retains
`R=cos(delta)`, `U=u^2` and `Q=u*sin(delta)`. Here `Q` is a form/phase
correlation, not nodal pressure. Their domains are `0<R<=1`, `U>=0` and
`Q^2=U*(1-R^2)`. These invariants identify the complete pair up to exchanging
its two constituents. They remove a redundant label, not continuous internal
degrees of freedom. The
[unordered-state proof](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-unordered-state)
derives their closed evolution and covers the synchronized singular point.

`internal_form_squared` stores exact `U`; `form_phase_correlation_bounds`
encloses `Q`; the existing resultant field encloses `R`. Their direct rates
are pushforwards of the full fine-node rates. The `closed_*_rates` independently
evaluate the inherited invariant equations, using the derived
`internal_restoring_coefficients` and `internal_phase_coefficients`.
The three rate-closure residuals and `internal_constraint_residual` retain
computed interval evidence. `internal_constraint_rate_residual` differentiates
the constraint with those closed rates; it is not overwritten with zero.
`unordered_storage_bounds` and `unordered_storage_rate_bounds` compute full
fine storage and its derivative from the invariant description, with separate
comparison residuals against the original fine calculation.

The [inherited Poisson identity](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-inherited-poisson)
provides a further independent comparison within this report.
`poisson_coordinate_order` declares `(X,Theta,R,U,Q)`;
`poisson_coefficient_pairs` declares the four independent bracket entries
`(X,Theta)`, `(R,U)`, `(R,Q)`, `(U,Q)`. Each pair's
`poisson_coefficients` encloses those entries; the opposite entries are their
negatives and all others are zero. The coefficient uses the **base** degree:
`kappa=b*nu/(4*d)`. Capacities and support are held as required above.

`unordered_storage_gradients` and `poisson_generated_rates` retain the full
storage gradient and its bracket-generated five rates. `poisson_rate_residuals`
compare these rates with the direct pushforward of the captured fine-node
field. `poisson_casimir_residuals` evaluate the tensor on the gradient of
`Q^2-U*(1-R^2)`; they retain computed interval bounds, not forced zeros.
These fields expose an inherited identity, not an added evolution law,
physical energy, a trajectory or a new arbitrary-state admission API.

The separate [resultant-derivative inverse](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-internal-observability)
is an exact observation theorem with known law, capacities, clock and
neighboring collective state. This forward assessment does not implement an
inverse solver for noisy samples. Independent derivative intervals or a
positive resultant alone do not establish a jointly realizable observation.

`unordered_pair_state_closure_certified` and
`unordered_pair_state_identifies_swap_orbits` apply to realized states in the
declared pair chart. They coexist with the means-only obstruction. The
assessment consumes a captured fine state; it does not admit arbitrary
independent interval tuples as realizable invariant data or authenticate a
manually constructed report. At `R=1,U>0,Q=0`, the correlation rate is nonzero:
equal phases alone do not justify discarding internal form. The polynomial
rows divide by neither `U` nor `1-R^2`, and remain defined at the synchronized
point. At `R=0` the chosen midpoint chart ends; a smooth fine field does not
make this particular collective coordinate chart globally valid.

Generic SDK export delegates exact projection and label admission to the
report owner; `to_dict()` uses `tnfr.relational-sine-replica-scale.v1`.
No node is removed, no edge weight or runtime law is replaced,
and no trajectory is executed. A collective description does not establish
that the supplied pair is an emergent NFR, a universal fractal hierarchy or
a physically identified constituent.

<a id="sine-replica-persistence"></a>
### Collective identity with active internal pairs

`assess_sine_replica_persistence(graph, reference_model=..., pairs=...,
radius=..., excess_ceiling=..., form_mean_bounds=..., winding=1,
phase_turns=None)` reuses the single captured replica comparison and the
shared full-support sine geometry. The five ordered pairs must describe a
complete doubled C5, with common strictly positive held capacity, exactly
zero form loss, and the existing absent-input/event premises. `winding`
accepts exactly the nonboolean integer `1` or `-1`. Both members of pair
`j` have exact target turns `winding*j/5`; the captured phases remain their
actual represented values, with explicit optional lift turns in node order.

The caller declares a positive radius, positive excess-storage ceiling and
finite open form-mean interval of positive width. These choose an analysis
family; they do not modify the law. The
[joint theorem](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-joint-persistence)
requires a strict acute radius and an excess ceiling below the full-state
coercive barrier. The engine reuses the general exact rational Laplacian gap
lower bound; it need not match the sharper symbolic doubled-C5 gap. Failure
of a sufficient numerical bound is explicit abstention, not instability.

`SineReplicaPersistenceAssessment` keeps the following decisions separate:

- `family_admitted` certifies the open invariant family with all five
  synchronized pair tips removed. It has finite positive volume in the
  **fine** form/phase state; `family_almost_everywhere_recurrence_certified`
  applies to that measure, not an arbitrary preparation surface.
- `source_set_trapping_certified` checks the captured whole-state norm and
  excess against the acute barrier. It does not require membership in the
  smaller declared excess/mean family.
- `source_family_membership` additionally checks the captured absolute
  mean, strict family ceiling and every pair's nontip status. It is
  `certified_inside`, `outside` for a proved mean/tip exclusion, or
  `unresolved` when upper bounds are insufficient. An unresolved upper bound
  is not a proved dynamical exclusion.
- `pair_source_status` distinguishes `exact_nontip` from `synchronized_tip`.
  `pair_all_time_activity_certified` and
  `pair_all_time_circulation_certified` require source trapping and that
  pair's exact nontip status. `source_joint_persistence_certified` requires
  trapping and all five nontip pairs. A synchronized pair can coexist with
  protected geometry and activity in the other pairs.
- `individual_recurrence_status` remains `unavailable_for_chosen_state`.
  Family membership supplies neither a selected-state recurrence certificate
  nor a deadline for a full-state return.

The internal angle belongs to `(delta,u/sqrt(beta))`, not to primitive nodal
phase or a new clock. For strict acute geometry, the report retains
`internal_phase_radius_bounds`, `normalized_angular_speed_lower_bound`,
`internal_angular_speed_bounds` and `internal_full_turn_time_bounds`.
The numerical lower factor uses `c_r*cos(D)`, with `D=radius/sqrt(2)`,
which is safely below the theorem's `c_r*sinc(D)`. These are positive-turn
traversal bounds for admitted active sources, not equal periods, phase
locking, amplitude floors or bounds on full-state recurrence.
An unresolved positive numerical speed bound for an extremely small
coefficient does not refute the strict mathematical rotation theorem.
Uncertified geometry withholds the associated numeric bounds.

The report preserves exact mean, divergence, target provenance, computed
geometry/storage bounds and explicit unresolved conditions. Its `to_dict()`
uses `tnfr.relational-sine-replica-persistence.v1`; the generic SDK exporter
delegates to that owner. No simulation, native runtime dispatch, graph write,
event, attracting waveform or autonomous grouping is introduced. The
two-sided invariant family cannot certify capture from outside itself.

<a id="sine-replica-capacity"></a>
### Unequal capacities and sufficient collective state

`assess_sine_replica_capacity(graph, reference_model=..., pairs=...,
phase_turns=None, radius=None, excess_ceiling=None, form_mean_bounds=None,
winding=1)` observes the same conservative sine law with independently held,
strictly positive fine capacities. It captures once and reuses the paired
coordinate and support kernels. Complete bipartite replica support is required
over a connected simple unit base graph, with no within-pair edges. Pair lifts
must admit `|delta|<pi/2`; the reader does not infer a branch. No graph state,
capacity, event or runtime law is changed.

The [derivation](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-capacity-asymmetry)
retains pair mean capacity `v`, half-difference `eta`, and the original
`(X,Theta,u,delta)` coordinates. `ordered_direct_rates` are pushed forward
from the actual fine field; `ordered_generated_rates` use the derived rows.
Their differences appear in `ordered_rate_residuals`. The inherited storage
gradients and two Poisson coefficients expose the same-law cross terms, with
separate `ordered_poisson_rate_residuals`.

The unordered coordinates additionally retain `P=eta*u`, `T=eta*sin(delta)`
and `E=eta**2`, where `E` here is squared capacity contrast, not total storage.
`capacity_aware_coordinate_order` identifies `(X,Theta,R,U,Q,P,T,E)`.
Independent fine-rate pushforwards, generated rows and residuals have that
order. `capacity_aware_constraint_names` identifies the six reported Gram
constraint residuals. These observations come from one realizable fine state;
containing zero in separate interval residuals does not admit arbitrary
invariant tuples. Forward formulas require no division by `eta` and recover
the symmetric case exactly. Swapping whole members, including capacity, is
a relabeling symmetry; swapping states while leaving unequal capacities fixed
generally changes the rates.

`normalized_form_mean_weights` and `weighted_form_mean` use fine
degree-over-capacity weights. Their balance residual is separate from
`arithmetic_mean_rate_bounds`, which need not contain zero. All identities
refer to **captured represented values**: shared staging may round two distinct
raw rational capacities to the same float. A lost raw contrast is not evidence
about an asymmetric captured system. Existing equal-capacity scale and
persistence readers keep their exact captured-equality requirements.

Supply all three family arguments together or leave all absent. Without them,
`family` is `None`. With them, five ordered pairs must form doubled C5 and
`winding` must be the nonboolean integer `1` or `-1`. The nested
`SineReplicaCapacityFamily` reuses the full-state geometry, exact target turns
and rational spectral lower bound of the persistence owner. Its finite-volume
family includes synchronized tips and uses the conserved weighted mean slab.
`family_form_coordinate_bounds` safely expands that slab by `sqrt(2)*radius`.

- `family.family_admitted` and
  `family.family_almost_everywhere_recurrence_certified` concern an open
  invariant fine-state family at the supplied fixed capacities.
- `family.source_set_trapping_certified` concerns the captured whole-state
  norm and excess barrier, independently of the smaller declared mean/ceiling
  family. `family.source_family_membership` additionally checks those strict
  bounds, retaining `outside` or `unresolved` when appropriate.
- `family.individual_recurrence_status` stays `unavailable_for_chosen_state`.
  No selected-state return or return deadline is inferred.

`all_time_internal_activity_status` and `angular_circulation_status` stay
`not_certified_by_this_reader`, even when geometric trapping succeeds. Exact
unequal-capacity nontip stalls and tip crossings prevent transferring the
earlier all-state claim; they do not establish permanent cessation or loss of
collective identity. The report's `to_dict()` schema is
`tnfr.relational-sine-replica-capacity.v1`, supported by the generic SDK exporter.

<a id="sine-state-pairing"></a>
### Phase-only partners and independent collective-law admission

`observe_phase_pairs(nodes=..., phases=...)` in the shared scale owner takes
only ordered labels and their circular phase representatives. Form, capacity,
edges, a model and a proposed partition are not inputs. It compares squared
chords `2*(1-cos(theta_j-theta_i))` using the shared certified cosine bounds.
This is strictly monotone in circular distance on `[0,pi]`, so it needs no
inverse angle, phase unwrapping or fitted proximity threshold. Rational inputs
are retained exactly; other real inputs use shared represented-real admission.
The 64-node limit is an explicit quadratic-work budget, not a physical scale.

`PhasePairObservation` retains the admitted `nodes` and `phases`, pairwise
`distance_indices` and `squared_chord_bounds`, and each node's
`nearest_partner_indices`, `nearest_separation_margins` and `node_statuses`.
A partner is certified only when its distance upper bound is strictly below
every competitor's lower bound. The complete `candidate_pairs` is available
only if every choice is unique and mutual. Otherwise `status` is `unavailable`,
`candidate_pairs` is `None`, and `reasons` preserve the abstention. An overlap
of enclosures does not prove an exact tie. Equal absolute represented gaps
can separately establish an exact nearest tie when its group is strictly
closer than every outside competitor. Node order determines serialization,
never tie-breaking or greedy rematching.

The [separation theorem](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-phase-pairing)
proves that the protected doubled-C5 tube identifies its constituent pairs
this way, including unequal positive held capacities. Common circular shifts,
phase reversal and relabeling preserve the mathematical partition. A rounded
common shift that discards differences before capture is a changed state;
numerically unresolved comparisons still abstain. An instantaneous match on
a generic state has no future guarantee.

`assess_sine_state_pairing(graph, reference_model=..., pair_order=None,
phase_turns=None, radius=None, excess_ceiling=None, form_mean_bounds=None,
winding=1)` captures the sine comparison once, projects its phases into the
same observer, then independently applies the existing paired-support and
capacity contracts to the observed partition. It does not recapture the graph.
`SineStatePairingAssessment` retains `comparison`, `observation`, the declared
`pair_order` and its source, supplied `phase_turns`, optional `capacity`
assessment and explicit `capacity_admission_status` and reasons.

When a partition is observed, optional `pair_order` may only reorder or reverse
members of those **already observed** pairs. If observation abstains, a supplied
order remains an unverified declaration and capacity admission is not attempted.
Without it, capture order supplies serialization for the
instantaneous collective rows. All three family bounds must be supplied
together, and target-dependent family assessment additionally requires
explicit `pair_order`. Neither cycle orientation nor a favorable target fit
is selected automatically. Lifts are forwarded to the existing chart admission:
a valid circular match across a raw branch boundary can require declared
turns before its collective coordinates are admitted.

Known support/law/chart refusal leaves the observed partition intact, with
`capacity=None` and an explicit rejection. Invalid caller declarations remain
errors; numerical or programming failures are not converted into scientific
counterexamples. In particular, a phase-only permutation on fixed support can
yield strict mutual partners that contain live internal edges and fail the
replica contract. The reader never replaces them with graph twin classes.

`all_time_pairing_persistence_certified` requires successful capacity assessment
and its captured-source trapping certificate in the protected doubled-C5 tube.
It concerns the persistence of the identified partition, independently of
membership in the narrower declared mean/excess family. It gives no formation,
internal-circulation, selected-state recurrence or physical-identity certificate.
The standalone schemas are `tnfr.phase-pairs.v1` and
`tnfr.relational-sine-state-pairing.v1`; the generic SDK exporter delegates to
their owners while retaining these distinct verdicts.

<a id="sine-pairing-transition"></a>
### Local onset and loss of observed pairing

`assess_sine_pairing_transition(graph, reference_model=...)` uses one shared
sine comparison capture and its unchanged phase-only observer. It differentiates
the observation under the complete normalized-sine law, without introducing a
pairing force or changing the native runtime. Fixed unit support, held capacity,
the comparison's coefficients and structural clock remain declared premises.

`SineExchangeComparison.phase_rate_numerators()` supplies the exact rational
row `N_i=w*nu_i*(L*x)_i/(beta*d_i)`, so `theta'_i=N_i/pi`. The same owner serves
resultant kinematics without computing unused trigonometric drift bounds when
only the phase row is needed. For the observer's squared chords,
`D'_ij=2*sin(theta_j-theta_i)*(N_j-N_i)/pi`.
`SinePairingTransitionAssessment` retains `comparison`, `observation`, these
`phase_rate_numerators`, and `squared_chord_rate_bounds` in the observation's
`distance_indices` order. These are ideal law rates, not measured secants.

The shared nearest-group admission returns `exact_nearest_groups` and
`outside_group_distance_margins`. A singleton needs strict separation from
every competitor. An exact tie needs equal absolute rational phase gaps and
strict separation from outsiders; overlapping distance enclosures alone
remain unresolved. Within a tie of absolute gap `d`, derivatives share the
factor `2*sin(d)/pi`; the signed numerator differences are exact rational
coefficients. This preserves their dependence instead of subtracting separate
rounded derivative bounds. A zero or unresolved factor, or tied first-order
coefficients, gives no split certificate.

The `backward` and `forward` sides each retain:

- `time_direction` (`-1` or `1`), `nearest_partner_indices` and `node_statuses`;
- `tied_rate_factor_bounds`, `tied_rate_coefficient_margins` and
  `tied_derivative_margin_bounds`, with unavailable quantities explicit;
- `candidate_pairs`, `status` and `reasons`: `certified_matching` requires
  all choices to be strict and mutual; `certified_nonmutual` requires all
  choices to be determined and at least one to be nonmutual; otherwise the
  result is `unavailable`;
- separate `support_admission_status` and reasons, checked on the captured
  fixed graph only when a complete partition is established. Admission uses
  the existing paired-support and positive-capacity contract;
  it supplies no future phase chart, collective closure or protected-family
  membership.

`local_interval_existence_certified` certifies the reported nearest choices
on some sufficiently small one-sided interval by smoothness and strict
transversality. `certified_time_horizon` stays `None`: no numerical endpoint,
uniform finite-precision resolution or lifetime follows. A finite-precision
observer may abstain arbitrarily close to a tie even when the mathematical
choice is strict. Complete grouping can be established or excluded locally
while its separate support admission fails.

The [proof and frozen direction control](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-pairing-transition)
use the same phases with opposite internal forms. The theorem also gives
qualitative robustness on time windows away from the crossing, but the report
does not supply a numerical preparation radius or force perturbed switches to
occur simultaneously. No graph edge, controller, trajectory, asymptotic
capture or physical identification is created. The schema is
`tnfr.relational-sine-pairing-transition.v1`, supported by the generic SDK
exporter.

<a id="sine-pairing-mobility"></a>
### Pairing response under a declared alternative mobility

The [constitutive-scope proof](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-pairing-constitutive-scope)
separates a local grouping mechanism from a uniquely selected microscopic
law. On its fixed doubled-C5 preparation, both initially tied preference
margins have positive derivatives for every finite strictly positive,
phase-only reciprocal mobility with a locally well-defined flow. Reversing
the initial form reverses these derivatives. This supplies a local interval
for each admitted law; it does not transfer the sine law's numerical window.

`SineExchangeComparison.with_current_squared_mobility(epsilon=...)` evaluates
the existing counterfamily on its unchanged source, without another capture:
`f_i=1+epsilon*(S_i/d_i)**2`, `m_i=f_i/(pi*d_i)`,
`x_dot_i=w*nu_i*m_i*S_i`, `theta_dot_i=(w/beta)*nu_i*m_i*q_i`.
Here `q=L*x` and `S_i` is the captured neighbor sine sum. Epsilon must be a
finite nonnegative admitted real scalar; form loss must be explicitly zero.
The graph, signed state, held nonnegative capacity and clock retain the
comparison's admission. No native law is installed or dispatched.

`SineMobilityComparison` retains its original `comparison`, exact `epsilon`,
`mobility_factors`, `mobility_corrections`, changed `inverse_phase_metric`,
phase source, pressure, both full rate rows and computed storage-work residuals.
Its law is `current_squared_reciprocal_mobility`; at epsilon zero it recovers
the reference field. The actual source storage stays in the nested comparison.
The shared kernel applies the factor to both exchange rows, with no derivative
reconstructed from an observed response. Work cancellation follows from their
matched reciprocal structure; interval residuals alone do not prove it.
State-dependent skew exchange does not by itself preserve a Poisson bracket,
invariant volume, recurrence or the prepared sine pulse.

`assess_sine_pairing_mobility(graph, reference_model=..., epsilon=...,
margin_triples=...)` evaluates two caller-declared squared-chord margins.
Each triple is `(observer, alternative, preferred)`, defining
`M=D_observer,alternative-D_observer,preferred` with
`D_ij=2-2*cos(theta_j-theta_i)`. It retains the source and alternative full
fields, the chosen margins and their initial rates. No partner, epsilon or
observation is optimized after evaluating the response.

`SinePairingMobilityAssessment` contains `mobility`, the independent phase
`observation`, `margin_triples` and their captured `margin_indices`.
`initial_exact_ties` distinguishes algebraic equal distances from overlapping
`initial_margin_bounds`. `sine_margin_rate_bounds`,
`mobility_margin_rate_bounds` and `mobility_margin_correction_bounds` retain
the two source rates, changed rates and their separately calculated corrections.
The `sine_` and `mobility_` contrast fields retain numerator, denominator,
normalized bounds and status. Status is `available` or
`denominator_not_separated_from_zero`; unavailable normalized bounds are `None`.
`source_margin_rate_difference_exact_zero` records the exact shared-sine
coefficient cancellation. `certified_time_horizon` remains `None`.

The dimensionless contrast `(M1_dot-M2_dot)/(M1_dot+M2_dot)` uses those declared
rate rows. A denominator not separated from zero leaves it unavailable.
Common positive clock scaling cancels, and common phase-origin motion does
not change chord derivatives. A separated contrast can therefore distinguish
phase-space directions without fitting a clock. Exact zero requires algebraic
cancellation, not an interval that merely contains zero.

The fixed comparison uses triples `((1,2,0), (2,1,3))`, the original twenty-edge
source, and epsilon zero versus one. The baseline contrast is exactly zero;
the alternative is strictly between `1/500` and `1/400`. Both local preference
directions agree. Form reversal reverses the raw rates and retains this ratio.
This is a mathematical discriminator under declared laws, not a physical
measurement or a unique-law selection. Neither a complete future matching,
protected family nor numerical time horizon is inferred by this reader.
The schemas are `tnfr.relational-sine-mobility-comparison.v1` and
`tnfr.relational-sine-pairing-mobility.v1`, supported by the generic SDK exporter.

<a id="sine-mobility-geometry"></a>
### Relative geometry and balance under alternative mobility

The [relative-geometry theorem](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-mobility-relative-geometry)
extends the existing acute storage barrier to the current-squared reciprocal
law. The full field is invariant under common form and phase translations.
Removing these two origins gives a sufficient autonomous relative state;
it does not remove internal coordinates, freeze a moving reference or change
the declared clock. Conservation of storage protects this relative geometry.
It does not by itself establish an invariant measure or recurrence.

`SineMobilityComparison.relative_balance(reference_node=...)` consumes its
existing capture and requires strictly positive held capacities. The node
must belong to that capture. `SineMobilityRelativeBalance` retains `mobility`,
the reference label/index, the nonreference `relative_indices`, and exact
`relative_form` and `relative_phase` differences. Phase coordinates are
circular coordinates supplied as raw real representatives, not a globally
single-valued chart. The corresponding rate intervals subtract the actual
moving reference's rates. No graph is read again.

`normalized_mean_weights` are normalized `d_i/nu_i` weights. The report keeps
the weighted form mean, the local lifted phase mean, their rate bounds both
from full-node sums and after exact baseline cancellation, and the difference
between those evaluations. The lifted phase mean is not a scalar on the phase
torus. `node_divergence_bounds` sum to `divergence_bounds`; the relative
divergence is identical because the discarded origins do not enter the field.
The exact formulas and their scope belong to the linked proof. At epsilon
zero, `constant_mobility_mean_and_volume_identities_certified` is true. A zero
rate or divergence at a single positive-epsilon source proves no identity.

`assess_sine_mobility_geometry(graph, reference_model=..., pairs=..., radius=...,
excess_ceiling=..., epsilon=..., winding=1, phase_turns=None)` uses one capture,
the existing doubled-C5 support admission and the shared exact target/storage
barrier. It requires common positive held capacity, zero form loss, positive
phase exchange and storage scale, and winding `-1` or `1`. Radius, ceiling,
epsilon and phase lifts retain shared real/integer admission; a positive
ceiling is required. Unsupported support or capacity raises. Insufficient
strict interval margins produce explicit unresolved conditions.

`SineMobilityGeometryAssessment` keeps its `balance` and thus both full rate
rows and the original source. The first member of the first pair selects the
relative reference. `phase_turns` independently supplies the barrier chart;
no absolute form-mean bounds are consumed. Its target, spectral bound, norm,
cancelled excess storage and barrier margins retain the original meanings.

- `family_admitted` checks the geometric barrier and strict family ceiling.
  The bounded relative open family excludes synchronized pair tips. Their
  complement is invariant by twin symmetry and uniqueness.
- `source_set_trapping_certified` checks the source's strict norm and barrier
  margins independently of that smaller family's ceiling or tip exclusion.
- `source_relative_family_membership` is `certified_inside`, `outside` for an
  exact pair tip, or `unresolved`; `pair_source_status` and reasons remain
  available. Family admission and represented-source membership are separate.
- `relative_coordinate_deviation_bound` encloses the uniform `sqrt(2)*radius`
  bound for anchored form differences and lifted phase deviations from target
  along the trapped flow. It is not a bound on absolute means or raw phases.
- At epsilon zero, an admitted family has
  `relative_family_recurrence_status="certified_almost_everywhere_in_relative_family"`.
  At positive epsilon it is `unavailable_invariant_measure_unproved`.
  An unadmitted family instead reports `unavailable_family_not_admitted`.

For positive epsilon, the old mean and Euclidean-volume identities fail in
general, including arbitrarily close to the same exact target. To recover
recurrence for almost every relative preparation in the volume sense, one
must supply a finite invariant measure equivalent to relative volume, such
as a positive integrable invariant density. A singular equilibrium measure
does not discharge that obligation. Nonzero divergence does not rule out
such a density or recurrence. No return claim for the selected state or
discarded common origins, internal circulation, pulse period, attraction or
native runtime dispatch follows from this report.

The schemas are `tnfr.relational-sine-mobility-relative-balance.v1` and
`tnfr.relational-sine-mobility-geometry.v1`; both use the generic SDK exporter.
Projection validates nested source, pair and reference labels but does not
authenticate a manually constructed public dataclass.

<a id="sine-pairing-window"></a>
### Whole-window pairing under a full initial uncertainty box

`assess_sine_pairing_window(graph, reference_model=..., form_error_bounds=...,
phase_error_bounds=..., window_start=..., window_end=...)` retains one comparison
capture. Both error arguments are mandatory vectors of nonnegative radii in
captured node order. They describe independent errors in every initial form
and real phase representative at time zero. The captured center, fixed unit
support, held nonnegative capacities and structural clock are retained; the
reader requires exactly zero form loss and no input or event. Time admission
requires `0<=window_start<window_end`. Boolean, nonfinite and malformed inputs
are rejected through shared admission. The 64-node ceiling is a work budget.
The exact captured centers and rational radii define the admitted box.
Displayed interval endpoints may be rounded outward; their enclosing intervals
do not authorize additional initial states beyond the declared exact radii.

Let `a=w/pi`, `b=w/(beta*pi)` and let `avg_i` average the actual neighbors.
The complete conservative law gives global bounds
`M_i=a*nu_i` on `abs(x'_i)` and
`B_i=b*nu_i*(M_i+avg_i(M))` on `abs(theta''_i)`.
Initial form errors give a phase-rate error bound
`V_i=b*nu_i*(r_x_i+avg_i(r_x))`. With the exact captured numerator `N_i`,
every admitted solution satisfies
`abs(theta_i(t)-theta_i(0)_center-N_i*t/pi)<=r_theta_i+V_i*t+B_i*t^2/2`.
The implementation uses rational upper bounds based on the shared lower bound
for mathematical pi. This controls the nonlinear feedback for the entire
window, without a Taylor-step solver or reduced hidden state.

`SinePairingWindowAssessment` preserves:

- The `comparison`, input error vectors, `initial_form_bounds`,
  `initial_phase_bounds` and exact `window` endpoints.
- `phase_rate_numerators`, `form_speed_upper_bounds`,
  `initial_phase_rate_error_bounds`, `phase_acceleration_upper_bounds` and
  `phase_remainder_upper_bounds`; the last quantity uses the window end and
  bounds deviation from each affine center throughout the window.
- `phase_window_bounds` and pairwise `phase_gap_window_bounds` in
  `distance_indices` order. Gaps use the exact numerator difference directly,
  preserving common-rate cancellation rather than subtracting separate phase
  windows.
- `squared_chord_window_bounds` and `chord_bound_methods`. Absolute gap ranges
  proved within `[0,pi]` use monotone cosine endpoint bounds; otherwise the
  shared interval cosine supplies a conservative bound. No phase lift is
  inferred or imposed to improve a verdict.
- `nearest_partner_indices`, `nearest_separation_margins`, `candidate_pairs`,
  `status`, `reasons` and `whole_window_nearest_relation_certified`.

Each nearest choice requires its chord upper bound to lie strictly below
every competitor's lower bound. All nodes must have certified choices before
the report states `certified_matching` or `certified_nonmutual`; the former
also requires mutuality. These claims hold for every source in the full box
and every time in the closed window. Wide or overlapping bounds yield
`unavailable`, not a refuted physical prediction. No exact synchronization of
initially equal-phase pairs is assumed under independent errors.

A complete matching receives the shared, independent
`support_admission_status` and reasons. This checks the existing fixed graph
and positive-capacity paired-support contract; it supplies no future pair
chart, arbitrary collective closure or permanent trapping. The initial box
is a declared analysis premise, not an authenticated measurement uncertainty.
The result does not guarantee that a separately rounded sensor resolves the
mathematical margins.

The [frozen whole-box proof](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-pairing-window)
uses errors `2^-20` in all twenty initial coordinates and the window
`[1/128,1/64]`, with opposite nearest maps for the two form-reversed centers.
The earlier local-transition reader retains its absent numerical horizon;
this certificate has separately admitted finite-time evidence. No graph is
mutated, no integration trajectory is fabricated, and no support birth or
unbounded lifetime is claimed. The schema is
`tnfr.relational-sine-pairing-window.v1`, supported by the generic SDK exporter.

<a id="sine-joint-pairing-projection"></a>
### Compare phase grouping with joint form-phase grouping

`SinePairingWindowAssessment.joint_observation()` returns a
`SineJointPairingProjection` from the same captured comparison, preparation
errors and time window. It does not reread a graph, change the law or replace
the phase-only verdict. This general projection is distinct from the
special synchronized `K2,2` reference certificate below.

`initial_observation` applies the shared joint observer to the exact captured
center. The initial full error box receives separate joint-distance bounds
and strict nearest-partner admission; the center's matching alone does not
certify that box. On the whole future window, the form tube follows
`|x_i(t)-x_i(0,center)| <= form_error_i + form_speed_upper_i*window_end`.
The projection adds its squared pair differences to `beta` times the existing
sharp phase-chord bounds, preserving the original nonlinear phase remainder
and every competing partner. Zero-crossing form intervals must be squared
with a zero lower bound. Joint distances use their known nonnegative domain.

The source joint-distance rates include both terms
`2*dx*dx_dot + 2*beta*sin(dtheta)*dtheta_dot`, using the already captured
complete form row and exact phase-rate numerators. These are source rates,
not measured derivatives or rates throughout the uncertain window.
Initial-center observation, initial-box identification and future-window
identification have different evidence. `same_pairing_as_initial_box` is
unavailable unless both boxes/windows have complete certified matchings;
two absent pairings do not establish equality. Resolved nonmutual choices
remain distinct from unresolved distance bounds. Agreement at preparation
and on a later window does not by itself certify the intervening time gap;
the fixed-source theorem proves that additional interval separately.

`support_admission_status` tests the same strict replica contract after
observation. A certified joint matching can fail that support test. It then
has no right to inherit the strict replica law; rejection does not prove
that every possible collective description fails. The
[unchanged-source comparison](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-joint-grouping-comparison)
establishes different joint partners already present in both original
form-reversed boxes, while their phase-only verdicts differ. It does not
assert that joint proximity defines a unique NFR or physical geometry.

The schema `tnfr.relational-sine-joint-pairing-projection.v1` supports the
generic SDK exporter and retains its nested phase-window evidence. Public
dataclass construction does not authenticate provenance. No support birth,
permanent identity, graph mutation or new trajectory is inferred.

The separate [critical-boundary theorem](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-joint-boundary-acquisition)
uses a symbolically specified trigonometric form amplitude. Its exact ties
are algebraic identities, not overlaps of numerical distance intervals.
The theorem proves a local acquisition of support-compatible joint pairing
from the complete form and phase rates, with no numerical lifetime. Existing
graph readers still capture represented real coordinates; inserting a rounded
critical amplitude changes that mathematical source. Neither a rounded graph
nor the window projection automatically inherits the exact tie or local-time
proof. Shared interval kernels check strict coefficient signs and bound the
symbolic amplitude without installing a new runtime law or preparation type.

<a id="sine-joint-pairing-window"></a>
### Joint form-phase identification during exchange

`observe_joint_pairs(nodes=..., forms=..., phases=..., storage_scale=...)`
uses squared distance `(x_i-x_j)^2+2*beta*(1-cos(theta_i-theta_j))`.
It accepts distinct ordered labels, signed exact/represented real scalars,
circular phase representatives and a positive scale. It does not accept
BEPI containers, project complex forms to magnitudes or infer the scale.
The 64-node limit is a work budget. Strict mutual nearest partners are
inferred from outward distance bounds without a supplied partition, label
tiebreak or matching repair. Resolved nonmutual choices and unresolved
comparisons have separate statuses; neither supplies `candidate_pairs`.
This is an instantaneous observation, not a future-state or support claim.

`assess_sine_joint_pairing_window(graph, reference_model=..., pairs=...,
form_error_bounds=..., phase_error_bounds=..., window_end=...,
phase_turns=None)` instead captures one exact reference through the replica
owner. It requires conservative normalized-sine dynamics, fixed unit `K2,2`
support, two internally synchronized pairs, equal initial forms, common
positive held capacity and a certified interpair lift strictly below pi.
The same law, support, capacity and structural clock hold for all compared
trajectories. Positive `window_end=T` declares the interval `[0,T]`;
both error vectors must supply four finite nonnegative initial radii.
Independent errors in all eight coordinates need not retain synchronization.

The [joint identity theorem](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-joint-identity-window)
derives a constant positive reference separation `J*` from inherited energy,
including when the phases coincide. The full-field bound uses scaled real
phase lifts, with `r0^2=sum_i(form_error_i^2+beta*phase_error_i^2)` and
`L=2*w*nu/(pi*sqrt(beta))`. Circular chord errors are bounded by these
lift errors; no global Lipschitz claim is made for an embedded field.
`identification_budget_margin` is a conservative lower bound on
`J*-8*r0^2*exp(2*L*T)`. Certification requires both its strict positivity
and resolved mutual nearest comparisons of the distance tubes.
`joint_distance_window_bounds` contains **squared** distances, indexed by
`distance_indices`; it is not an evolved snapshot.

`reference_identity_certified` records the exact reference theorem separately
from `whole_window_identity_certified`. Zero amplitude makes the reference
identity unavailable. A positive amplitude too small for interval resolution,
excessive uncertainty or unrepresentable exponential amplification makes the
finite certificate unavailable, without asserting actual identity loss.
Zero error bypasses amplification arithmetic but retains the reference and
distance checks. `reference_period_bounds` reuses the P2 period owner;
`window_covers_reference_period` concerns that reference only, not perturbed
periodicity or a recurrence deadline. The scale is inherited from the declared
storage and coordinate chart, not selected as a unique physical metric.

No graph is advanced or mutated. These readers do not prove attraction,
permanent perturbed identity, autonomous formation, operator occurrence or
physical constituents. Source capture, errors, horizon, bounds and availability
are retained by schemas `tnfr.joint-pairs.v1` and
`tnfr.relational-sine-joint-pairing-window.v1`, including generic SDK export.

<a id="sine-pair-support-symmetry"></a>
### Pair-state interchangeability on fixed support

`assess_sine_pair_support_symmetry(graph, reference_model=..., pairs=...,
witness_pair=None)` checks the
[complete-law symmetry criterion](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-pair-support-symmetry)
on one captured simple connected unit graph. It requires zero form loss and
a common strictly positive held capacity. The supplied ordered pairs must
partition every captured node; unlike strict replica execution, the symmetry
question admits a single pair and within-pair edges. The 64-node limit is a
work budget. This reader neither discovers a partition nor changes the graph.

An independent member swap preserves the full sine field for all states if
and only if the two members have the same external neighbors. Equivalently,
every interpair adjacency block is empty or contains all four edges; internal
pair edges may be present. Necessity follows from the exact normalized phase
row; sufficiency also accounts for the nonlinear sine row. Equal neighbor
counts without equal attachments do not establish this symmetry.

`SinePairSupportSymmetryAssessment` retains the captured `comparison`, declared
`pairs`, their `pair_indices`, `within_pair_edge_indices`, complete
`cross_block_edge_counts` and `incomplete_cross_blocks`. Each entry of
`external_neighbor_difference_indices` names the captured indices that
distinguish that pair's attachments. `pair_swap_symmetry` records each result.
For each failed swap, `phase_row_commutator_witnesses` contains one exact
`(row, column, numerator)` entry of `G[P(i),P(j)]-G[i,j]`, where
`theta'=G*x/pi`; successful swaps have `None`. This is an all-state algebraic
witness, not a residual sampled at the source.

`independent_pair_swaps_equivariant` and `unordered_pair_quotient_status`
distinguish `certified_by_independent_swap_symmetry` from
`obstructed_by_fixed_support`. This concerns the full unordered member-state
orbits, represented by the existing invariants only on their declared chart;
it neither compresses them to pair means nor invents a scalar macro-node law.
Separate `strict_replica_admission_status` and reasons retain the existing
no-internal-edge contract. A graph can satisfy the symmetry theorem while its
internal edges require different inherited rows and reject that older domain.

Optional `witness_pair` is an explicit nonboolean index in the declared pair
list. No failing pair is selected automatically. The nested
`SinePairSwapWitness` exchanges both captured form and phase within that pair,
retaining fixed support and capacities. It records the permutation, swapped
state, exact form gradient, shared phase-rate numerators, bounded full rates
and equivariance residuals. It does not construct or commit a live graph.

The witness also retains source and swapped local pair-mean phase-rate
numerators, their exact differences and bounds after division by pi.
`source_orbit_equal` checks exact unordered raw member-state multisets, not
overlap of trigonometric enclosures. A nonzero mean-rate difference then
certifies `collective_phase_rate_obstruction_certified`. A zero difference
does not repair a failed all-state symmetry test. The reported means refer
to continuous local lifts; they do not define a global circular mean or
identify a regional angle with primitive phase. Interval residuals containing
zero are not proofs of exact field equality.

The fixed one-edge-deletion control yields receiver mean rates `+1/(48*pi)`
and `-1/(48*pi)` from identical unordered pair states. It identifies missing
state-to-attachment information, not a missing force or a support event law.
The schema is `tnfr.relational-sine-pair-support-symmetry.v1`, supported by
the generic SDK exporter. Graph/state relabeling, phase-only observation and
autonomous support selection remain separate questions.

<a id="sine-mixed-pair-state"></a>
### Sufficient pair state on asymmetric support

`assess_sine_mixed_pair_state(graph, reference_model=..., pairs=...,
phase_turns=None)` shares the preceding support-symmetry admission and a single
complete sine capture. Its
[mixed-state theorem](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-mixed-pair-state)
retains signed member-to-attachment information exactly where an independent
pair swap is not a graph symmetry. This removes redundant member labels;
it does not remove internal continuous degrees of freedom.

The same explicit phase-lift convention and strict `|delta|<pi/2` pair chart
as the replica observer apply, independently of strict replica support.
Missing turns mean zero supplied turns, not inferred unwrapping. A failed
chart rejects this representation; it does not invalidate the smooth fine
law. Within-pair edges and incomplete cross-blocks are admitted here without
extending the older complete-replica formulas or persistence certificates.

`SineMixedPairStateAssessment` retains `support_symmetry`, `phase_turns`,
`phase_chart_margins`, `coordinates` and `rates`. Its `comparison` and `pairs`
properties expose the retained source. Each coordinate records `form_mean`
and the exact radian/turn split of its lifted phase mean. Its `mode` is:

- `ordered`: keep `form_half_difference` and the radian/turn split of
  `phase_half_difference`, relative to the declared ordered attachments.
- `unordered`: keep `resultant_magnitude_bounds` for `R=cos(delta)`, exact
  `internal_form_squared=U=u**2`, and `form_phase_correlation_bounds` for
  `Q=u*sin(delta)`. The exact realized invariants obey
  `Q**2=U*(1-R**2)`, `0<R<=1`, `U>=0`.

Fields not retained in a mode are `None`, not zero. `reconstruction_stratum`
distinguishes the ordered chart and the unordered reconstruction cases.
The exact realized coordinates identify precisely the orbits generated by
the individually permitted swaps. Synchronized tips are admitted; their
quotient singularity does not make the fine field singular. No numerical
inverse of arbitrary interval tuples or independent interval realizability
test is provided. The retained ordered fine source is provenance, not an
additional coordinate of the mathematical quotient.

Rate records retain `form_mean_bounds` and exact `phase_mean_numerator`;
the phase rate is that numerator divided by mathematical pi. Ordered pairs
also retain `form_half_difference_bounds` and
`phase_half_difference_numerator`. Unordered pairs instead retain rates of
`R`, `U` and `Q`, obtained by differentiating those invariants along the
shared fine field. These are instantaneous bounds at the realized source,
not a new coarse integrator or a future chart certificate. The proof gives
reconstruction up to allowed symmetries and exact inherited rows, including
the contributions of asymmetric neighbors and internal edges.

On the fixed nineteen-edge control, modes are
`(ordered, unordered, unordered, unordered, ordered)`. Swapping source members
`0,1` now changes retained signed coordinates. The known receiver rates
`+1/(48*pi)` and `-1/(48*pi)` therefore correspond to distinguishable states.
This repairs the prior information loss without changing the law or repeating
the frozen finite-window evaluation. A simultaneous graph/state relabeling
changes reporting labels, not the mechanism. The schema is
`tnfr.relational-sine-mixed-pair-state.v1`, supported by the generic SDK exporter.

<a id="sine-pair-emission"></a>
### Structural Emission on an unordered pair

`assess_sine_pair_emission(graph, reference_model=..., pairs=...,
pair_index=..., boost=..., phase_turns=None)` reuses one mixed-pair source
capture and its conservative normalized-sine admission. The selected pair
must admit an independent swap; other pairs may retain ordered attachments.
The SDK delegates through `Network.relational_sine_pair_emission`.
This is the [structural descent comparison](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-pair-emission-descent),
not a native Arg evolution step or a public AL execution.

The explicit positive finite `boost` uses the registered AL factor contract.
The actual `emission_epi_proposal` owner supplies each candidate's represented
addition, clipping and nondecrease check. Both member proposals must be
admitted before any report is returned; rejection cannot become a no-op.
The shared operator boundary policy retains its existing hard fallback for
unrecognized modes and default soft steepness. It does not consume the
integrator's configurable `CLIP_SOFT_K`. The report records the effective
`clip_policy=(minimum, maximum, mode, steepness)` without changing those rules.

`SinePairEmissionAssessment` retains `source_state`, the selected index,
represented boost and proposals, and three outcomes: `first_member`,
`second_member`, `whole_pair`. Each outcome records target positions within
the selected pair, ordered endpoint forms, effective increments and collective
`X,U,Q`; phase mean and `R` remain those of the source. Ordered endpoints are
fine provenance, not an extra coordinate of the unmarked collective state.
Binary64 AL endpoints are stored as exact represented rationals; bounded
trigonometric projections retain the source's strict phase chart and lifts.

`single_member_source_orbit_equal` compares exact member-state multisets.
Its status distinguishes `equal_at_source_synchronized_tip`,
`equal_at_source_both_noops` and `obstructed_at_source`. Interval overlap,
equal means or equal storage costs cannot certify that equality. A source
exception does not establish all-state single-target closure. In general,
the missing information is a marked constituent relative to the pair.

`whole_pair_structural_descent_certified` concerns the common scalar map on
both members and its invariant admission domain. Clipping can change internal
form/correlation despite this symmetry. `runtime_admission_certified` and
`event_occurrence_derived` remain false: grammar, configured public
preconditions, history, lifecycle, pressure refresh and event timing are not
assessed. No graph state, support or cache is modified; no subsequent flow,
storage passivity, maintained identity or physical interpretation is inferred.

The owner schema is `tnfr.relational-sine-pair-emission.v1`. Generic SDK export
uses the common `tnfr.relational-report.v1` envelope and delegates the same
payload, including source provenance and unavailable runtime claims.
Its optional `form_increment(outcome=...)` uses the separate
[weighted-form ledger](#sine-regional-transfer) to test a necessary condition
for being a closed-flow endpoint. Structural descent and that endpoint test
have different obligations.

<a id="sine-replica-equilibria"></a>
### Exhaustive fully acute equilibrium families

`assess_sine_replica_equilibria(graph, reference_model=..., pairs=...)` reuses
one comparison capture and the shared complete-replica support admission.
The ordered structural pairs must describe the full unit doubled C5; form
loss is exactly zero and every fine capacity is held and strictly positive.
Member capacities need not agree. No pair phase chart, target winding or phase
lift is supplied: the [classification](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-acute-critical)
derives circular pair synchrony only for fully acute equilibria.

`SineReplicaEquilibriaAssessment.targets` lists the integer solutions of
`4*abs(k)<5` in the declared cycle orientation. Each target retains its exact
`target_phase_turns`, `target_edge_turns`, shared `target_geometry`
reconstruction, `edge_cosine_bounds` and `storage_bounds`. The reconstruction
proves circular consistency and sine cancellation; its separate phase-law
interpretations are not imported. Uniform form and arbitrary common form and
phase origins complete each symbolic family. The shared target helper also
serves the earlier trapping readers, whose public winding admission remains
exactly `-1` or `1`.

`acute_equilibrium_classification_certified` concerns the complete family
classification under this law/support, independently of the captured phase
sector. It does not assert that the source is an equilibrium. The source keeps
separate evidence:

- `source_edge_cosine_bounds` and `source_acute_status` distinguish
  `fully_acute`, `outside_fully_acute_sector` and `unresolved`. Strictly positive
  edge cosines certify acuteness; an interval crossing zero is not exclusion.
- `source_form_uniform` and `source_raw_phase_uniform` compare exact captured
  values, without a tolerance. Nonuniform form excludes full equilibrium by
  connectedness and positive capacity, even outside the acute sector.
- `source_equilibrium_status` is `certified_consensus_equilibrium` for uniform
  captured form and phase, `certified_not_equilibrium` for a proved exclusion,
  or `unavailable` when the admitted evidence does not decide it.
  `source_equilibrium_reasons` records the applicable argument;
  `source_equilibrium_winding` is zero only for certified source consensus,
  and otherwise `None`.

Captured radians are exact rational numbers after shared engine staging.
Within the fully acute sector a nonzero equilibrium twist requires irrational
phase differences `2*pi*k/5` modulo full turns, so nonidentical captured
rational phases exclude exact equilibrium. This is a theorem about the
captured representation, not an inference from a small rate residual or an
uncertainty claim about a physical measurement. If the source has uniform form
but nonacute or unresolved phases, this reader makes no global classification.

Consensus remains a valid full equilibrium even when the phase-only partner
observer abstains because of ties. The structural pairing used to classify
the supplied graph never replaces that observer's output. The two nonzero
windings are exchanged by orientation reversal; the report selects neither.
It introduces no graph mutation, attraction, new force, preferred spatial
shape, formation event or physical identification. Its `to_dict()` schema is
`tnfr.relational-sine-replica-equilibria.v1`, supported by the generic SDK
exporter.

<a id="sine-replica-internal-pulse"></a>
### Exact prepared internal pulse

`assess_sine_replica_pulse(reference_model=..., form_half_difference=...,
phase_half_difference=..., capacity=...)` in the same scale owner assesses
an exact mathematical preparation, rather than capturing a graph. The support
is the complete double-replica C5, with uniform mean form, exact mean phase
turns `i/5`, identical signed internal coordinates `u,delta` in every pair,
and one common strictly positive held capacity. Common form and phase origins
are immaterial. The input phase half-difference is in radians and must have
certified magnitude less than `pi/2`. Shared exact/represented real admission
rejects Booleans and nonfinite values; rational inputs retain their exact value.

The reference model must declare `phase_domain="regular"` and zero EPI loss,
as required by the shared sine comparison interface. Its coefficients and storage
scale are reused for the sine comparison law, with no inputs, events or support
changes. Neither the selected native phase chamber nor a small graph residual
certifies this symbolic preparation. No live graph membership is inferred and
no graph or runtime law is changed.

The [proof](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-internal-pulse)
derives the invariant family directly from the full retained rows. The report
preserves the input scalars and exact target turns, enclosures of the internal
storage `H=u^2+beta*cos(2*pi/5)*sin(delta)^2`, the elliptic parameter
`m=H/(beta*cos(2*pi/5))`, rates and period bounds. A proved `0<m<1`
certifies a nonstationary libration and permanent validity of the pair chart.
For zero internal form and nonzero admitted phase, the chart proves this
strict inequality even if the projected energy interval touches the barrier;
numerical period availability remains a separate calculation.
The full labeled state returns after `T`; the unordered state returns after
`T/2` because the constituents swap. A stationary state has no nonstationary
period. Above-barrier states leave this chart; an unresolved boundary receives
no libration certificate. This is not a theorem excluding all circular-state
rotations.

All fine edges remain strictly acute throughout the orbit only when
`m<sin(pi/20)^2`. This stronger test is separate from chart retention and
the fixed collective winding. The small-amplitude limit and nonlinear period
enclosures use the shared elliptic-integrand inequality, not an elliptic solver
or a measured numerical return. Unresolved numerical bounds remain unavailable
even when a qualitative theorem applies. SDK export preserves those distinctions.
`period_bounds` describes labeled return; `unordered_period_bounds` is its half.
`nonlinear_periodic_exchange_certified` and `all_fine_edges_acute_status`
report different tests, while `graph_membership_certified` remains false.
The report's `to_dict()` schema is `tnfr.relational-sine-replica-pulse.v1`.
No exact period, internal locking or attraction is certified for transverse
perturbations of this lower-dimensional preparation.

<a id="sine-replica-pulse-variation"></a>
### Full perturbations of the prepared pulse

`assess_sine_replica_pulse_variation(...)` accepts the same keyword arguments
and delegates preparation, model and scalar admission to the pulse owner.
Its `SineReplicaPulseVariation` report retains that assessment in `pulse`.
It differentiates the complete fine law with fixed support, coefficients and
capacity; it does not restrict disturbances to identical internal pairs.
Changing those held parameters is a different perturbation problem.

`mode_blocks` contains three real four-by-four coefficient matrices, indexed
by the C5 Fourier representatives `mode_indices=(0,1,2)`. The convention is
`exp(+2*pi*i*k*j/5)`. For nonzero representatives, the coordinates are
`(X_hat,Theta_hat,i*u_hat,i*delta_hat)`; their real and imaginary parts each
follow the displayed real block. The zero mode uses ordinary real collective
and internal perturbations. Thus `mode_multiplicities=(1,2,2)` accounts for
all twenty real fine-state directions. This is an invertible symmetry
decomposition, not discarded state or a fitted low-dimensional model.

These matrices are evaluated at the supplied initial internal state. Their
formulas apply along its exact reference evolution; they are not finite-time
transition maps. `dimensionless_mode_blocks` uses form perturbations divided
by `sqrt(beta*cos(2*pi/5))` and the clock `tau=Omega*t`, with both scales
enclosed separately. The instantaneous blocks depend on internal phase.
Only after following a complete libration does the return spectrum depend on
the dimensionless energy `m` and geometry alone, independently of initial
orbit phase up to conjugacy. Holding raw form fixed while changing `beta`
generally changes `m`.

`variation_identity_certified` concerns the exact differentiation and
decomposition. `periodic_reference_certified` separately requires the pulse
owner's nonstationary libration certificate. Equilibrium, unresolved energy
and above-barrier preparations still have an instantaneous variation law;
they do not receive that periodic-reference flag. The zero-amplitude frequency
bounds describe the stationary target limit, not the finite-amplitude spectrum.

The constant `symplectic_form` and computed `hamiltonian_structure_residuals`
check the inherited linear Hamiltonian identity. They introduce no additional
physical law or energy. `half_return_swap` is `diag(1,1,-1,-1)`: when the pulse
is certified, half-period coefficients are conjugated by this swap. The
unordered return derivative must include it; a raw half-period propagator
cannot be treated as the return map. The
[proof owner](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-pulse-variation)
derives these statements, the uniform amplitude/time shear and the remaining
transverse stability criterion.

No propagator or Floquet multiplier is evaluated here, and
`orbital_stability_status` remains `"not_assessed"`. Static trapping,
instantaneous eigenvalues and interval residuals alone cannot supply that
verdict. No nonlinear finite-perturbation error bound is claimed. Exact export
uses `tnfr.relational-sine-replica-pulse-variation.v1`, with the shared SDK
dispatcher preserving the same scope and symbolic-source provenance.

<a id="sine-replica-pulse-splitting"></a>
### Small-amplitude return splitting

`assess_sine_replica_pulse_splitting(reference_model=..., capacity=...)`
reuses the exact zero-amplitude variation template. It accepts no finite
amplitude, observed graph or horizon. Its `SineReplicaPulseSplitting` report
assesses the analytic family approaching that reference through `m>0`;
the stationary reference itself receives no instability verdict.

The [proof](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-pulse-splitting)
retains the changing-period clock and the induced collective response before
extracting the internal resonant terms. `p_exact_coefficients` and
`q_exact_coefficients` encode `a+b*sqrt(5)` pairs for the local harmonic
coefficients `P_k,Q_k`, not nodal pressure or internal correlation. Their
interval evaluations generate `slow_generators` in rotating amplitude order
`(A,B)`, with `G_k=[[0,Q_k/2],[-P_k/2,0]]` and squared exponents `-P_k*Q_k/4`.
The effective logarithmic return generator is `m*G_k+O(m^2)` in an analytic
internal basis, not an instantaneous constant-coefficient law for the nodes.

The exact signs give `mode_classifications=("hyperbolic","elliptic")` for
representatives `(1,2)`, each with two real spatial copies. Analytic remainder
control establishes some strictly positive amplitude interval in which mode 1
destabilizes the orbit. Mode 2 is bounded in the transverse linear problem on
a sufficiently small interval; it does not imply nonlinear stability.
`existential_amplitude_interval_certified` and
`sufficiently_small_nonlinear_orbital_instability_certified` express these
conditional theorems. They do not evaluate a chosen amplitude.

`labeled_log_multiplier_slope_magnitude_bounds` encloses `2*pi*sqrt(abs(-P*Q/4))`;
the unordered slope is half that value. These are leading coefficients of
`m`, real for the hyperbolic case and imaginary for the elliptic case. They
are not finite log-multiplier bounds. `amplitude_upper_bound` and
`finite_amplitude_remainder_bound` remain `None`; `finite_preparation_assessed`
and `return_multipliers_computed` remain false. The API therefore cannot certify
that a caller's small numerical amplitude lies within the instability interval.

Orbital departure is compatible with the separate static winding barrier and
conserved full storage. This reader installs no drive, controller, changed law
or numerical return solver. Its exact projection uses
`tnfr.relational-sine-replica-pulse-splitting.v1`; the generic SDK dispatcher
preserves the same absence of a finite-preparation verdict.

## Shared circular geometry

### Exact cycle resultant sectors

[`derive_cycle_resultant_sector`](../../src/tnfr/physics/phase_resultant_sectors.py)
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
The [formation-domain theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-formation-domain)
owns the conditional invariant and its degree-two scope; extra neighbors can
invalidate that obstruction. This exact-turn observer does not select or
extend any of the executor's separately declared phase domains.
