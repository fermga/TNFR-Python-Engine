# Retained sine pair state, identification and pulses

Sufficient unordered/mixed pair state and cancellation inversion, capacities and support symmetry, pairing observations and windows, emission/transfer and doubled-C5 pulse response.

Part of [Relational dynamics contract index](../RELATIONAL_DYNAMICS.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="sine-moving-pattern-interface"></a>
### Retained interaction state of a moving pair

The [theory contract](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-moving-pattern-interface)
combines existing owners on the fixed conservative unit doubled C5:

- `comparison.regional_transfer(region=pair)` retains per-edge currents and
  the weighted cut sum. Each pair has weighted form `8*X`; divide a selected
  neighboring block's four-edge sum by eight to obtain its contribution to
  mean form. Opposite nonzero block contributions can cancel in the net sum.
- `assess_sine_replica_scale` retains the realized internal `R,U,Q`, means,
  environment and fine source, and derives the exact full-law pushforward
  within its nonantipodal chart. This supplies sufficient state under its
  symmetry premises; the instantaneous current alone does not.
- `assess_sine_mixed_pair_state` retains ordered constituent information where
  support breaks an independent swap symmetry. Its actual attachment-sensitive
  field cannot be replaced by complete-replica formulas.

At zero pair resultant, keep the full comparison and regular cut/kinematics
readers, or use the [global pair coordinates](#sine-global-pair-state) under
their fixed support and unit-law premises. The midpoint scale reader still
rejects its chart boundary. Node-relative
`resultant_kinematics` is not a pair midpoint or normalized pair phasor; its
rates and the regional ledger retain their own normalizations. None of these
observations drops constituents, advances a graph or supplies a boundary law.

A fully retained joint network is closed under its supplied law. A pattern
considered alone still consumes its surroundings; a held environment is a
new premise, and hidden-state elimination must retain initialization and
causal memory. Prepared phase-offset families and tangent memory kernels
do not provide an all-state replacement. The existing SDK delegates/exporters
remain the implementation boundary; no duplicate interface report is needed.
The [guide](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-moving-pattern-interface)
connects the current and retained-state workflows.

<a id="sine-global-pair-state"></a>
### Global unordered pair state, including phase cancellation

The [pair owner](../../../src/tnfr/physics/relational_sine_pair.py) exposes
`SineGlobalPairState` through two exact evaluators:

- `derive_sine_global_pair_state(forms, phasors)` consumes ten signed forms
  and ten Cartesian unit phasors in fine-node order `0,...,9`.
- `evaluate_sine_global_pair_state(*, form_means, resultants, phase_products,
  internal_form_squared, form_phase_moments)` consumes five entries in each
  primitive vector, in base-cycle order `0,...,4`.

The pair-state, cancellation and finite receiver APIs are defined in
`tnfr.physics.relational_sine_pair`. Their established imports from
`tnfr.physics.relational_sine_scale` remain aliases to the same objects;
signatures, report fields and JSON schemas are unchanged.

The support is fixed: consecutive pairs `(0,1),...,(8,9)`, all four unit
cross edges between adjacent pairs around C5, and no within-pair edges.
The complete normalized-sine law has `e=0`, `w=beta=nu=1`, held capacities,
original structural time `t`, and no forcing or events. These evaluators
neither capture an arbitrary graph nor select another model.

For unit phasors `z_plus,z_minus`, set `u=(x_plus-x_minus)/2`. The five
coordinates are `X=mean(x)`, `Z=mean(z)`, `P=z_plus*z_minus`, `U=u**2`
and `W=u*(z_plus-z_minus)/2`. Complex coordinates use `(real, imaginary)`
pairs, not Python complex values or complex EPI. Shared exact/represented-real
admission normalizes scalars to fractions and rejects Boolean, nonfinite,
unordered or incorrectly sized inputs. Unit norms and realizability are
exact tests; represented decimal phasors are not silently normalized.

The invariant evaluator requires `abs(P)**2=1`, `P*conj(Z)=Z`,
`abs(Z)**2<=1`, `U>=0`, and `W**2=U*(Z**2-P)`, and checks the implied
`abs(W)**2=U*(1-abs(Z)**2)`. These conditions admit a fine representative
up to whole-member exchange, including at zero resultant or zero variance.
Rational invariant coordinates may require irrational fine representatives.
The [global-state proof](../../../theory/nodal/SINE_PAIR_STATE.md#sine-global-pair-state)
owns reconstruction, closure and continuation; arbitrary independent boxes
or unconstrained Euler updates are not admitted by this API.

Every `*_rate_pi_numerators` coordinate field contains **pi times** its
original-clock derivative. `directed_block_edges` orders `(receiver, source)`;
`block_current_pi_numerators` contains pi times each contribution to receiver
mean-form rate, while `block_current_rate_pi_squared_numerators` contains
**pi squared times** its derivative. Weighted cut contributions are eight
times the mean-form contributions. `form_storage`, `phase_storage`, `storage`
and the three `*_storage_rate_pi_numerator` fields retain independently
computed component work and its total. `pair_strata` distinguishes
`split_phase`, `coincident_phase` and `antipodal_phase`.

No cached report rate, storage or verdict is an evaluator input. Reevaluate
the five primitive vectors to obtain a fresh report; export alone does not
authenticate edited dataclasses. The direct schema is
`tnfr.sine-global-pair-state.v1`, also supported by generic SDK projection
and atomic export. This is a global symmetry representation and instantaneous
field evaluation, not a new integrator, native execution mode, grouping rule,
free parameter or physical identification. Existing midpoint adapters keep
their chart restrictions. See the [standalone example](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-global-pair-state)
and [full-node controls](../../../tests/physics/test_sine_global_pair_state.py).

<a id="sine-pair-cancellation-observability"></a>
### Recover internal information from an exactly cancelled pair

`observe_sine_pair_cancellation(*, form_means, resultants, pair_index,
resultant_first_tau_derivative, resultant_second_tau_derivative)` in the
[pair owner](../../../src/tnfr/physics/relational_sine_pair.py) returns
`SinePairCancellationObservation`. It retains the fixed support and complete
unit-coefficient conservative law of the global pair state above. Its clock
is explicitly **`tau=t/pi`**; input derivatives are `pi*d_t Z` and
`pi**2*d_t**2 Z`, respectively, not both multiplied by the same factor.

Supply all five signed form means and Cartesian resultants in base-cycle
order, an integer index `0..4`, and the two exact Cartesian derivatives of
that selected resultant. Shared exact/represented-real admission rejects
Boolean, nonfinite, unordered and incorrectly sized inputs. Every resultant
must lie in the unit disk, and the selected resultant must be exactly zero.
The function derives `h` and `B` from the supplied means and adjacent
resultants using the same support owner as the forward evaluator. It accepts
no hidden phase products, internal variances or cached verdicts as evidence.

The report always reconstructs `form_phase_moment=-i*Z'` and
`internal_form_squared=abs(Z')**2`. Its `phase_product` is:

- `Z'**2/abs(Z')**2` when `Z'!=0`, with reconstruction order 1;
- `(2*Z''-B)/conj(B)` when `Z'=0,B!=0`, with order 2;
- `None` when `Z'=B=Z''=0`, with order `None` and
  `unavailable_reason="higher_order_or_persistent_environment_evidence_required"`.

Recovered products must have exact unit norm and satisfy
`Z''=2*i*h*Z'+(B+P*conj(B))/2`. Contradictory evidence, including nonzero
`Z''` with `Z'=B=0`, raises `ValueError`. No tolerance, derivative fitting,
phase repair or guessed hidden neighboring state is supplied. The report
retains admitted input evidence, `form_contrast=h`, `neighbor_resultant=B`
and `phase_product_reconstruction_order` alongside the reconstructed state.

This is exact local two-derivative compatibility and conditional state
recovery, not authentication of a time series or independent validation of
the supplied law. A finite zero record does not certify permanent silence;
the [proof](../../../theory/nodal/SINE_PAIR_STATE.md#sine-pair-cancellation-observability)
separates delayed revelation from genuine environmental indistinguishability.
No noisy-data admission, trajectory integration, graph mutation or physical
identification follows. Direct schema
`tnfr.sine-pair-cancellation-observation.v1` and generic SDK export preserve
exact fractions and the unavailable `None` value. See the
[example](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-cancellation-observability)
and [fine-row tests](../../../tests/physics/test_sine_pair_cancellation_observation.py).

<a id="sine-pair-finite-exchange"></a>
### Finite directed exchange near an invisible pair family

`assess_sine_pair_finite_exchange(*, phase_rotation, horizon_tau)` in the
[pair owner](../../../src/tnfr/physics/relational_sine_pair.py) returns
`SinePairFiniteExchange`. It evaluates the
[analytic finite-exchange theorem](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-pair-finite-exchange)
on the fixed conservative unit doubled C5. The law, held capacities and support
are those of the global pair state above; there are no inputs or events.

Supply an exact Cartesian unit phasor `phase_rotation=(a,b)` and a strictly
positive finite real `horizon_tau`. Shared admission retains represented-real
values as fractions and rejects Boolean scalars, malformed/unordered phasors,
nonfinite values and a norm different from exactly one. No phase normalization
or horizon adjustment is performed. The clock is `tau=t/pi`; the original-clock
endpoint is `T=pi*horizon_tau`.

All ten initial forms are zero. In `orientation_order`, selected pair 0 has
phasors `(1,-1)` or `(i,-i)`. Both members of each surrounding pair share its
phasor. The comparison uses surrounding pairs `(q,1,-1,-1)`, whereas its
symmetry control uses `(q,q,-q,-q)`, with `q=a+i*b`. These are separately
prepared autonomous networks. The report retains their full initial phasors
and fresh `SineGlobalPairState` assessments, including component storage and
internal rates. The control is an exact stationary family; its selected-edge
integrated currents and their difference are zero for every admitted horizon.

`directed_block_edge=(0,1)` uses `(receiver, source)` order. The observable is
the difference, real-antipodal minus imaginary-antipodal, of
`integral_0^T I_(1->0)(t) dt`, where
`I_(1->0)=Im(conj(Z_0)*Z_1)/(2*pi)`. It is a contribution to mean form;
the corresponding weighted-cut integral is eight times larger. The report
does not substitute the net change in pair 0, which also consumes the other
neighbor's current, or interpret form exchange as energy transfer.

The report retains the exact rational
`integrated_current_difference_leading_term` and
`integrated_current_difference_remainder_upper_bound`. Its
`integrated_current_difference_bounds` encloses their difference and sum
using the shared outward rational interval method. The bounds hold for the
evolving complete nonlinear law, including environmental backreaction.
Strictly negative/positive reported intervals yield `certified_negative` or
`certified_positive`; otherwise `status="unavailable"` retains the unresolved
interval and its reason. A zero leading coefficient or an interval containing
zero does not certify equality of the two responses. The evaluator neither
searches for a shorter horizon nor retries with a different rotation.

This is an analytic finite-time certificate, not an executed trajectory,
an inverse observation, a new propagator or a noisy-data inference. The direct
schema `tnfr.sine-pair-finite-exchange.v1` and generic SDK projection retain
exact inputs, source states, bounds and unavailable values; export does not
authenticate an edited report. See the
[usage example](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-finite-exchange)
and [independent fine-law controls](../../../tests/physics/test_sine_pair_finite_exchange.py).

<a id="sine-pair-receiver-readout"></a>
### One receiver endpoint with preparation and readout uncertainty

`assess_sine_pair_receiver_readout(*, phase_rotation, horizon_tau,
form_error_bound, phase_error_bound, readout_error_bound)` in the
[pair owner](../../../src/tnfr/physics/relational_sine_pair.py) returns
`SinePairReceiverReadout`. It reuses the finite-exchange preparation family
above, with the same complete conservative sine law and all ten fine nodes.
Its new observation is the **absolute mean form of pair 1**, fine nodes
`(2,3)`, at `T=pi*horizon_tau`. Both incoming connections and the complete
environment evolve. No phase derivative, hidden-state reading, baseline
sample or selected-edge integral is an observation input.

The phase rotation is an exactly unit Cartesian phasor. The horizon must
satisfy `0<horizon_tau<1/2`. This upper restriction belongs to the rational
uncertainty majorant, not a physical lifetime or loss of existence of the
fine flow. All three error bounds are required finite nonnegative scalars;
zero is admitted, Boolean/nonfinite/negative values reject. Shared exact and
represented-real admission occurs before arithmetic, with no normalization,
float materialization of exact fractions, missing-budget default or adaptive
horizon selection.

`form_error_bound` bounds each fine node's initial deviation from its zero
nominal form. `phase_error_bound` bounds each initial circular phase distance
from its nominal phase, in radians. The budgets cover the hidden pair,
receiver and environment independently. The proof chooses initial lifts
within that circular bound and follows the actual periodic field; no arbitrary
unwrapping is inferred from data. The form budget also bounds the common
form origin. An additional unbounded offset is outside the contract.
Support, capacities, law coefficients and clock are held exact. The final
scalar mean-form reading has additive error at most `readout_error_bound`.

For each orientation, `nominal_readout_centers` combines the common linear
term and its own cubic coefficient. `nominal_readout_bounds` includes the
complete-law remainder. The two nonnegative amplification coefficients
propagate separate form/phase preparation errors into
`propagated_preparation_error_upper_bound`; adding final reading error gives
`readout_uncertainty_radius`. `expanded_readout_bounds` encloses the complete
recorded reading in `orientation_order=("real_antipodal","imaginary_antipodal")`.
These are outward intervals; all primitive budgets, coefficients, nominal
phasors and freshly derived full pair states are retained as exact fractions.

`readout_gap_lower_bound` is the larger of the two ordered differences of
one interval's lower endpoint and the other's upper endpoint. Strictly
positive gap yields `status="certified_disjoint"`. Overlap or a shared
endpoint yields `status="unavailable"`, retaining the intervals and reason
`receiver_readout_intervals_overlap_or_touch`. An unresolved sufficient bound
does not prove that the actual responses coincide. Membership in an enclosing
interval alone does not prove existence of a compatible complete trajectory,
identify an arbitrary hidden state or authenticate an observed sample.

The separate symmetry controls have exact nominal receiver center zero.
Their `symmetry_control_readout_bounds` retain preparation and readout errors:
perturbed controls need not remain stationary or give an exact zero reading.
Every admitted trajectory conserves its own complete storage; perturbed
preparations need not share the nominal storage value or exact antipodality.

This is a forward analytic discrimination certificate for two declared
preparation families. It contains no measured sample, fitted pressure or
executed numerical trajectory. Direct schema
`tnfr.sine-pair-receiver-readout.v1`, generic SDK projection and the shared
atomic writer retain its uncertainty and unavailable evidence. Export does
not revalidate an edited report. The
[theorem](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-pair-receiver-readout),
[usage](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-readout)
and [independent controls](../../../tests/physics/test_sine_pair_receiver_readout.py)
own the proof, execution example and tests.

<a id="sine-pair-receiver-constitutive-confounding"></a>
### Joint law and orientation ambiguity of the receiver endpoint

`assess_sine_pair_receiver_confounding(*, phase_rotation, horizon_tau,
epsilon_upper)` in the
[pair owner](../../../src/tnfr/physics/relational_sine_pair.py) returns
`SinePairReceiverConfounding`. It retains the doubled-C5 nominal preparations
and the sole observation `Y=(x_2+x_3)/2` at `t=pi*horizon_tau`. Orientation A
now evolves under the supplied family
`j_epsilon(delta)=sin(delta)+epsilon*sin(delta)^3`, with constant
`0<=epsilon<=epsilon_upper`; orientation B evolves under `epsilon=0`.
Both complete models retain the same phase row, support, held unit capacities,
zero loss and exact structural clock. Each conserves its own declared storage.
This is a comparison of possible law/state pairs, not an executor or a fitted
correction to the observed pressure.

The exactly unit Cartesian `phase_rotation` uses shared scalar admission.
`horizon_tau` and `epsilon_upper` must be finite and strictly positive;
Boolean, nonfinite and nonpositive inputs reject. Exact fractions remain
exact. There is no normalization, numerical trajectory, root search, adaptive
coefficient range or fitted observation. The finite Taylor bound has no
`horizon_tau<1/2` restriction: that restriction belongs to the separate
preparation-error majorant of the preceding report.

The report reconstructs the nominal primitive preparations and evaluates
`Y_A^epsilon-Y_B^0` at the two declared coefficient endpoints. Its intervals
include the separate nonlinear remainders from both complete laws and round
outward. `coefficient_endpoints=(0,epsilon_upper)` orders the evidence in
`endpoint_difference_bounds`; `endpoint_difference_signs` uses `1`, `-1`
or `0` for positive, negative or unresolved enclosures. Strictly opposite
endpoint signs yield
`status="certified_collision_exists"`: continuous dependence on epsilon
then proves that at least one coefficient strictly between zero and the upper
endpoint gives exactly equal actual receiver readings. The certificate does
not supply a root value, uniqueness or an observed collision. An interval
containing zero, a shared endpoint or unresolved signs yield `"unavailable"`;
these outcomes neither prove nor exclude a collision.
`collision_parameter_open_interval` is the exact pair `(0,epsilon_upper)`
only when certified, and otherwise `None`; both endpoints are excluded.

The preparations and readings here are exact. Their equal-output witness is
also admissible when a containing preparation/readout error family permits
zero error. That fact does not transfer the earlier fixed-law separation
certificate to an unknown law. The stationary cancellation controls and the
consensus linearization remain shared by this whole constitutive family;
neither resolves the ambiguity. Different laws may have different storage
values even for the same preparation, and no conservation of the reference
sine storage is imposed on the cubic law.

The cubic alternative consumes phase information beyond the first circular
moment. An independently justified
[first-moment sufficiency premise](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency)
excludes it. The collision does not contradict that conditional sine-selection
theorem or disprove the sine law; this one endpoint and the shared controls
cannot independently establish the missing selection premise.

Direct schema `tnfr.sine-pair-receiver-confounding.v1` and the generic SDK
projection retain the inputs, bounds and unavailable evidence. Export does
not authenticate an edited report. The
[proof](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-pair-receiver-constitutive-confounding),
[usage](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-constitutive-confounding)
and [independent controls](../../../tests/physics/test_sine_pair_receiver_confounding.py)
own the result and its checks. No physical constitutive range, measurement
bridge, formation claim or universal law selection is established.

<a id="sine-pair-receiver-two-time"></a>
### Two receiver readings with one constant coefficient

`assess_sine_pair_receiver_two_time(*, phase_rotation, horizons_tau,
epsilon_upper, form_error_bound, phase_error_bound, readout_error_bound)`
in the [pair owner](../../../src/tnfr/physics/relational_sine_pair.py) returns
`SinePairReceiverTwoTime`. It compares the preceding A-epsilon and B-zero
families at two times using the same receiver `Y=(x_2+x_3)/2`. Each branch
has one initial state throughout; A has one constant coefficient in the
closed interval `[0,epsilon_upper]`. The full doubled-C5 support, unit
capacities, phase row, own-law storage, exact clock and zero inputs, loss and
events retain the preceding contract's meaning.

The exact unit Cartesian rotation `q=a+ib` must satisfy `a>1/2` and `b>0`.
The ordered pair `horizons_tau` must satisfy `0<h0<h1`; `epsilon_upper` is
positive and all three error bounds are nonnegative. Shared scalar admission
rejects Boolean and nonfinite inputs before arithmetic and preserves exact
fractions. The sufficient preparation bound requires
`4*(1+3*epsilon_upper)*h1**2<1`; violation rejects the domain, without asserting
a physical instability. Every initial fine-node form and phase may vary
independently within its budget, including the absolute form origin and
environment. Phase errors mean initial continuous-lift offsets in radians;
no phase-mean reconstruction is performed. Each scalar reading of each
branch has the declared readout error. No independence of errors over time
is assumed.

The report rebuilds the nominal primitive preparations, cubic contrast and
full-law Taylor remainder. A coupled form/phase comparison propagates each
branch's initial uncertainty. Convex coefficient chords bound these errors
and yield affine inequalities `N-A*epsilon <= recorded_A-recorded_B <=
U-B*epsilon` at each time. Strictly positive slopes give the necessary
coefficient interval `[N/A,U/B]`. `necessary_coefficient_bounds` retains both
outward dyadic intervals in time order, **without clipping** to the separately
reported coefficient domain. Matching at both times requires membership in
both intervals with the same coefficient. Interval membership alone proves
neither a realizable collision nor an estimate of the coefficient.

Strictly disjoint outward intervals give `status="certified_disjoint"`.
`coefficient_gap_lower_bound` is the maximum of the two directed endpoint
gaps; a positive value certifies separation. Overlap or touching yields
`status="unavailable"` and
`unavailable_reason="necessary_coefficient_intervals_overlap_or_touch"`.
Nonpositive affine slopes yield `"coefficient_interval_slope_not_positive"`,
with coefficient bounds and gap unavailable (`None`). These sufficient
bounds need not resolve every distinguishable family.

On certification, `joint_readout_weights` gives exact signed weights in time
order, with absolute values summing to one. They eliminate the common
coefficient between the two affine bounds. The outward
`joint_readout_gap_lower_bound` bounds their weighted A-minus-B record
difference in form units, including all declared errors. An extremely small
positive exact readout margin may round down to zero; coefficient-interval
separation still determines status. Weights and margin are `None` when the
coefficient test is unavailable. No response, fitted parameter, numerical
trajectory or time-varying coefficient is consumed.

Direct schema `tnfr.sine-pair-receiver-two-time.v1` and generic SDK projection
retain primitives, coefficient endpoints, error bounds, affine evidence and
unavailable fields. Export does not authenticate a report. The
[frozen protocol and proof](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-pair-receiver-two-time),
[usage](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-two-time)
and [independent controls](../../../tests/physics/test_sine_pair_receiver_two_time.py)
own this conditional separation. It resolves the named two-family ambiguity;
general state recovery, physical law selection and sensor/clock calibration
remain outside its scope.

<a id="sine-pair-receiver-two-law"></a>
### Two receiver readings with an unknown coefficient in either hypothesis

`assess_sine_pair_receiver_two_law(*, phase_rotation, horizons_tau,
epsilon_upper, form_error_bound, phase_error_bound, readout_error_bound)`
in the [pair owner](../../../src/tnfr/physics/relational_sine_pair.py) returns
`SinePairReceiverTwoLaw`. Both nominal orientations now evolve under the
complete cubic-sine family. A and B may use different coefficients, each
in `[0,epsilon_upper]`; each branch retains one coefficient and one initial
state throughout both readings. The complete support, phase row, capacities,
clock and error semantics are those of the preceding two-time contract.
Neither branch imposes the reference sine law or its storage on the other.
`phase_exchange_beta=1` records the held coefficient of the phase row;
`coefficient_endpoints` bounds the unknown cubic-current coefficient of
either hypothesis.

Primitive admission retains the exact unit Cartesian rotation `a+ib` with
`a>1/2,b>0`, two strictly increasing positive horizons, a positive coefficient
upper bound, and three nonnegative error bounds. Boolean and nonfinite
inputs reject before arithmetic; fractions remain exact. The sufficient
preparation domain is `4*(1+3*epsilon_upper)*h1**2<1`. Initial phase errors
are bounded lift offsets in radians. All ten form and phase coordinates,
including the environment and absolute form origin, may be perturbed.
Readout errors may be correlated across times and hypotheses.

The fixed `readout_weights=(-h1,h0)/(h0+h1)` have absolute values summing
to one. Applied to the two receiver values, they cancel each nominal law's
linear Taylor term separately. `cubic_time_factor=h0*h1*(h1-h0)` multiplies
the remaining cubic coefficient. `nominal_cubic_coefficients` stores each
orientation's quadratic polynomial in its own law coefficient, in ascending
power order. `nominal_cubic_coefficient_ranges` gives its exact extrema on
the declared interval; positivity of the polynomial coefficients makes these
the endpoint values in the admitted rotation domain.

`nominal_initial_rate_bounds` bounds the maximum fine-node form rate at
initialization for each orientation over its whole coefficient family.
The full-law continuation bound then supplies `form_growth_bounds_by_time`,
`form_rate_bounds_by_time` and `remainder_upper_bounds_by_time`, with time
outermost and orientation innermost. These retain the evolving environment;
they do not hold the initial pressure fixed. The cubic remainder applies to
nominal zero-form trajectories. Separate `preparation_error_bounds_by_time`
propagate every admitted nonzero initial error under its own law.

`joint_error_radii` combines the complete-law remainder, preparation error
and scalar readout error using the absolute weights, separately for A and B.
`joint_difference_bounds` is an outward dyadic enclosure of every admitted
weighted record difference `Z_A-Z_B` over the entire two-coefficient square.
A strictly positive lower endpoint gives `status="certified_disjoint"`.
Otherwise the report returns `status="unavailable"` with
`unavailable_reason="joint_readout_difference_not_strictly_positive"`.
A zero endpoint cannot certify; an interval containing zero does not prove
a realizable collision. This is a sufficient test of positive separation,
not a complete classifier of all admitted inputs.

The report rebuilds all consumed quantities from primitives; it consumes
neither a measured record nor another report's verdict. The weights do not
cancel an arbitrary unknown constant form origin, which remains bounded by
the preparation budget. Coefficients varying between readings, unknown
clocks and continuous model defects require separate bounds. Certification
distinguishes these orientation families without identifying either
coefficient or selecting a unique physical law.

Direct schema `tnfr.sine-pair-receiver-two-law.v1` and the generic SDK retain
exact rational evidence and unavailable verdicts. Export is not authentication.
The [frozen protocol and proof](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-pair-receiver-two-law),
[usage](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-two-law)
and [independent controls](../../../tests/physics/test_sine_pair_receiver_two_law.py)
own the result. Earlier single-reading and fixed-reference certificates keep
their original inputs, remainder bounds and verdicts.

<a id="sine-pair-receiver-defect"></a>
### Continuous defects in both complete evolution rows

`assess_sine_pair_receiver_defect` in the
[pair owner](../../../src/tnfr/physics/relational_sine_pair.py) takes the same
primitive arguments as `assess_sine_pair_receiver_two_law` plus mandatory
`form_rate_defect_bound` and `phase_rate_defect_bound`. It returns
`SinePairReceiverDefect`. The added primitives are finite nonnegative bounds
on the absolute additive residual in every fine-node form or phase row,
respectively, almost everywhere over the entire observation window. Their
units are form per unit `tau` and radians per unit `tau`. Boolean, nonfinite,
non-real and negative inputs reject through shared original-scalar admission;
exact fractions remain exact. The reference rotation, horizon and preparation
admission domains remain unchanged.

Each hypothesis retains one initial state, one constant coefficient in the
declared family and one residual history across both readings. Histories may
differ between hypotheses and may be arbitrarily correlated across time and
nodes. The theorem concerns absolutely continuous paths satisfying the full
rows almost everywhere. It needs no derivatives of the residuals and installs
no runtime input, pressure law, event or controller.

`reference_certificate` is a fresh two-law report rebuilt internally from
the supplied primitives; this assessor accepts no report argument. Its
conservative-model statements apply to the reference comparison. Residuals
in the expanded family generally change total form and own-law storage;
the [proof](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-pair-receiver-defect)
owns their explicit balances. Perturbed controls need not be stationary.

For `L=1+3*epsilon_upper`, the separate response factors are
`h/(1-4*L*h**2)` and `L*h**2/(1-4*L*h**2)`.
`form_defect_response_factors_by_time` and
`phase_defect_response_factors_by_time` retain these in reference time order.
`defect_error_bounds_by_time` combines them with their respective residual
budgets. The result is additional form uncertainty, separate from the
reference preparation, Taylor remainder and readout budgets.

`joint_defect_radius` is the absolute-weighted defect error for **each**
branch; `joint_error_radii` adds it to the two reference radii in orientation
order. `joint_difference_bounds` combines those expanded errors with the
exact reference cubic ranges before outward materialization. It does not
reuse a cached verdict or inflate already-rounded endpoints. Zero residual
budgets reproduce the reference interval and verdict exactly.

Status is `"certified_disjoint"` only if the expanded interval's lower
endpoint is strictly positive. Otherwise it is `"unavailable"`, with
`unavailable_reason="joint_readout_difference_not_strictly_positive"`.
Touching zero or losing a positive margin to outward rounding cannot certify.
Unavailable evidence does not prove a realizable collision. Arbitrary
time-dependent residuals are not canceled by the fixed nominal time weights.

Direct schema `tnfr.sine-pair-receiver-defect.v1` and generic SDK projection
retain exact inputs, nested reference evidence and the new verdict. Export
does not authenticate a source or verify that an actual residual history
satisfies its declared budget. Physical use needs independent whole-window
defect bounds and a justified state, sensor and clock mapping. The
[frozen protocol](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-pair-receiver-defect),
[usage](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-defect)
and [independent controls](../../../tests/physics/test_sine_pair_receiver_defect.py)
retain this conditional scope.

<a id="sine-pair-persistent-response"></a>
### Internal allocation, finite response and persistent geometric identity

`assess_sine_pair_persistent_response(delta_bounds=..., horizon_tau=...,
form_error_bound=..., phase_error_bound=..., readout_error_bound=...,
radius=...)` in the [pair owner](../../../src/tnfr/physics/relational_sine_pair.py)
assesses the [declared comparison](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-pair-persistent-response).
It is a detached analytic family certificate, not a captured graph, a numerical
trajectory or an executor. `relational_sine_scale` retains a compatibility
re-export of the same implementation.

The fixed model is the ten-node, degree-four, complete doubled C5 with unit
held capacities, unit exchange/storage scales, zero form loss and clock
`tau=t/pi`. Mathematical target phases are `a*2*pi/5`. In preparation order
`phase_split` then `form_split`, pair 0 has either phase offsets `(+delta,-delta)` or
form offsets `(+u,-u)`, with `u^2=2*cos(2*pi/5)*(1-cos(delta))`.
All other nominal coordinates retain the target. Both nominal centers have
the same collective means and storage for the same delta. The independently
perturbed full-state boxes need not be energy matched. Neither symbolic pi
nor the square root is replaced by a captured binary64 source coordinate.

Shared represented-real admission precedes arithmetic. `delta_bounds` contains
exactly two scalars satisfying `0<lower<=upper<=1` in radians. The horizon is
strictly positive and satisfies `2*horizon_tau**2<1`, the domain of this
rational comparison bound. Radius is strictly positive; all three error
budgets are nonnegative. Boolean and nonfinite physical values reject.
These restrictions describe this assessor, not a stability limit of the law.
Positive form and phase widths give full-dimensional open box interiors;
zero widths remain valid lower-dimensional conditional preparations.

The scalar reading is the absolute mean form of nodes 2 and 3 at the declared
horizon. Analytic remainder bounds include the evolving nominal environment.
The nominal synchronized-pair reduction is used only to derive those bounds;
the preparation-error comparison and storage barrier retain all twenty fine
coordinates. Arbitrarily correlated initial errors and independently bounded
additive readout errors are covered.

`SinePairPersistentResponse` retains `nominal_readout_bounds`,
`recorded_readout_bounds` and `recorded_difference_bounds` separately from
`initial_radius_margin_bounds` and `storage_barrier_margin_bounds`.
`persistence_certified_by_preparation` describes the two geometric admissions;
`response_separation_certified` describes only the recorded contrast.
`full_dimensional_preparation` is true exactly when both initial widths are
positive. The symbolic target, support, complete law and clock remain explicit.
`status="certified_persistent_response"` requires both preparations to pass
strict initial-radius and conserved-storage barrier tests inside the acute
target chart, together with a strictly positive outward A-minus-B recorded
contrast. Failure of a sufficient inequality yields `status="unavailable"`
with its reason; it establishes neither a realizable response collision nor
loss of identity. A response certificate alone cannot substitute for the
geometric admission. The frozen protocol's stronger numerical stopping margin
is checked against the resulting bound, rather than hidden in this API status.

Persistence concerns the specified winding/acute geometry for all time under
the unchanged unforced law. It does not promise five internally active pairs,
formation, a selected microscopic law, physical energy units or a particle
identity. No result from the preceding defect or cubic-law certificates is
transferred to this conservative family. The [usage](../../guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-persistent-response)
and [independent controls](../../../tests/physics/test_sine_pair_persistent_response.py)
retain these boundaries. SDK export preserves exact premises and availability;
the direct schema is `tnfr.sine-pair-persistent-response.v1`. Serialization
does not authenticate an actual preparation or observation.

<a id="sine-replica-scale"></a>
### Inherited sine dynamics with both internal nodes retained

`assess_sine_replica_scale(graph, reference_model=..., pairs=...,
phase_turns=None)` in
[`relational_sine_scale.py`](../../../src/tnfr/physics/relational_sine_scale.py)
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
[proof owner](../../../theory/nodal/SINE_PAIR_STATE.md#sine-replica-inheritance)
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
[unordered-state proof](../../../theory/nodal/SINE_PAIR_STATE.md#sine-replica-unordered-state)
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

The [inherited Poisson identity](../../../theory/nodal/SINE_PAIR_STATE.md#sine-replica-inherited-poisson)
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

The separate [resultant-derivative inverse](../../../theory/nodal/SINE_PAIR_STATE.md#sine-replica-internal-observability)
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
[joint theorem](../../../theory/nodal/SINE_REPLICA_PULSE.md#sine-replica-joint-persistence)
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

The [derivation](../../../theory/nodal/SINE_PAIR_STATE.md#sine-replica-capacity-asymmetry)
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

The [separation theorem](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-phase-pairing)
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

The [proof and frozen direction control](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-pairing-transition)
use the same phases with opposite internal forms. The theorem also gives
qualitative robustness on time windows away from the crossing, but the report
does not supply a numerical preparation radius or force perturbed switches to
occur simultaneously. No graph edge, controller, trajectory, asymptotic
capture or physical identification is created. The schema is
`tnfr.relational-sine-pairing-transition.v1`, supported by the generic SDK
exporter.

<a id="sine-pairing-mobility"></a>
### Pairing response under a declared alternative mobility

The [constitutive-scope proof](../../../theory/nodal/SINE_PAIR_MOBILITY.md#sine-pairing-constitutive-scope)
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
The constructor and `relative_balance()` share fresh reconstruction from admitted
primitive state and the declared mobility coefficient. They do not consume cached
gradients, resultants or rate rows as independent evidence.

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

The [relative-geometry theorem](../../../theory/nodal/SINE_PAIR_MOBILITY.md#sine-mobility-relative-geometry)
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

The [frozen whole-box proof](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-pairing-window)
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

The projection re-admits primitive source state, preparation radii and window,
then rebuilds all consumed phase/chord bounds, speed bounds and distance
indices through the same owner as initial window construction. Altered cached
rates, outcomes or verdicts cannot replace those premises. An equivalent
original report association can be retained only after the calculation uses
normalized state and rebuilt evidence; this is not provenance authentication.

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
[unchanged-source comparison](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-joint-grouping-comparison)
establishes different joint partners already present in both original
form-reversed boxes, while their phase-only verdicts differ. It does not
assert that joint proximity defines a unique NFR or physical geometry.

The schema `tnfr.relational-sine-joint-pairing-projection.v1` supports the
generic SDK exporter and retains its nested phase-window evidence. Public
dataclass construction does not authenticate provenance. No support birth,
permanent identity, graph mutation or new trajectory is inferred.

The separate [critical-boundary theorem](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-joint-boundary-acquisition)
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

The [joint identity theorem](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-joint-identity-window)
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
[complete-law symmetry criterion](../../../theory/nodal/SINE_PAIR_STATE.md#sine-pair-support-symmetry)
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
[mixed-state theorem](../../../theory/nodal/SINE_PAIR_STATE.md#sine-mixed-pair-state)
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
This is the [structural descent comparison](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-pair-emission-descent),
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
[weighted-form ledger](SINE_REGIONAL_DYNAMICS.md#sine-regional-transfer) to test a necessary condition
for being a closed-flow endpoint. Structural descent and that endpoint test
have different obligations.

This dependent reader re-admits the source, pair declaration, phase lifts,
boost and effective clip policy and rebuilds symmetry, target indices and
the shared AL proposals before testing the increments. Saved increments and
verdicts are not authoritative inputs. The actual binary64 clipping/rounding,
including legitimate no-ops, remains part of the declared structural action;
the reconstruction neither recaptures a graph nor executes AL.

<a id="sine-replica-equilibria"></a>
### Exhaustive fully acute equilibrium families

`assess_sine_replica_equilibria(graph, reference_model=..., pairs=...)` reuses
one comparison capture and the shared complete-replica support admission.
The ordered structural pairs must describe the full unit doubled C5; form
loss is exactly zero and every fine capacity is held and strictly positive.
Member capacities need not agree. No pair phase chart, target winding or phase
lift is supplied: the [classification](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-replica-acute-critical)
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
phase_half_difference=..., capacity=...)` in the
[`replica pulse owner`](../../../src/tnfr/physics/relational_sine_replica_pulse.py) assesses
an exact mathematical preparation, rather than capturing a graph. The support
is the complete double-replica C5, with uniform mean form, exact mean phase
turns `i/5`, identical signed internal coordinates `u,delta` in every pair,
and one common strictly positive held capacity. Common form and phase origins
are immaterial. The input phase half-difference is in radians and must have
certified magnitude less than `pi/2`. Shared exact/represented real admission
rejects Booleans and nonfinite values; rational inputs retain their exact value.

This owner also supplies the variation, splitting, stiffness-curve and
work-response reports below. The established `relational_sine_scale` imports
remain aliases to the same classes and functions; SDK export schemas are
unchanged.

The reference model must declare `phase_domain="regular"` and zero EPI loss,
as required by the shared sine comparison interface. Its coefficients and storage
scale are reused for the sine comparison law, with no inputs, events or support
changes. Neither the selected native phase chamber nor a small graph residual
certifies this symbolic preparation. No live graph membership is inferred and
no graph or runtime law is changed.

The [proof](../../../theory/nodal/SINE_REPLICA_PULSE.md#sine-replica-internal-pulse)
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
[proof owner](../../../theory/nodal/SINE_REPLICA_PULSE.md#sine-replica-pulse-variation)
derives these statements, the uniform amplitude/time shear and the remaining
transverse stability criterion.

No propagator or Floquet multiplier is evaluated here, and
`orbital_stability_status` remains `"not_assessed"`. Static trapping,
instantaneous eigenvalues and interval residuals alone cannot supply that
verdict. No nonlinear finite-perturbation error bound is claimed. Exact export
uses `tnfr.relational-sine-replica-pulse-variation.v1`, with the shared SDK
dispatcher preserving the same scope and symbolic-source provenance.

<a id="sine-replica-pulse-work-response"></a>
### Finite-time work response along the moving pulse

`assess_sine_replica_pulse_work_response(reference_model=...,
form_half_difference=..., phase_half_difference=..., capacity=...,
scaled_duration=..., order=10)` re-admits the primitive preparation through
the pulse/variation owner. Its symbolic support is the same conservative
doubled C5. No captured graph, cached verdict or externally supplied tangent
is accepted as evidence. Exchange weight, beta and held capacity retain their
admitted values; the duration is strictly positive in **`tau=t/pi`**, distinct
from the variation report's `Omega*t` normalization.

The fixed real `k=1` probes are distributed form masks: collective cosine
and opposite-member sine. Both leave initial phase unchanged and subtract
the same unperturbed moving background. Fine storage-work outputs have
coefficients `(10*lambda_1,20)`, accounting for each mask's squared norm five.
The [proof and port definition](../../../theory/nodal/SINE_REPLICA_PULSE.md#sine-moving-pulse-work-response)
derive the normalization and distinguish reversal from relabeling.

`SineReplicaPulseWorkResponse` retains `reference`, `initial_state_bounds`,
`state_order`, `scaled_duration`, `taylor_order` and work coefficients. The
shared Picard/Taylor kernel encloses the joint ten-coordinate field: the
moving background and both complete four-coordinate variation columns.
The strict whole-time tube must retain the internal pair chart. A successful
`step` gives `response_bounds` (output rows, input columns) and
`antisymmetric_response_bounds=H_CI-H_IC`. `response_sign` is +1 or -1 only
when this interval excludes zero; zero means unresolved direction, not exact
reciprocity. `directional_response_certified` records that distinction.

An invalid primitive raises; a failed enclosure instead returns
`status="unavailable"`, reasons and available failure evidence, with response
fields set to `None`. A successful enclosure has
`status="certified_finite_response"`, even when direction is unresolved.
There is no automatic change of horizon, order or preparation to obtain a sign.
The certificate concerns an infinitesimal probe at a finite delay, not a
finite-amplitude disturbance, Floquet multiplier or orbital stability.
The report installs no runtime law, edge, input or physical interpretation.
Its direct schema is `tnfr.relational-sine-replica-pulse-work-response.v1`;
the generic SDK wrapper preserves the same report and explicit availability.

<a id="sine-replica-pulse-finite-work-response"></a>
### Full nonlinear centered kicks under the fixed pulse protocol

`assess_sine_replica_pulse_finite_work_response(probe_amplitude=..., order=10)`
is a fixed mathematical preparation, not a generic graph reader. The
`SineReplicaPulseFiniteWorkResponse` report declares unit conservative sine
exchange on doubled C5, capacity one, `u_0=delta_0=1/32`, the same two
distributed form masks, and the fixed horizon `tau=1/16`. It admits only
`0<probe_amplitude<=2^-20` through shared scalar admission and validates the
Taylor order before propagation. These supplied experiment parameters do
not become physical constants or a selected law.

The report freshly rebuilds its `tangent` from these primitives. Its
`initial_boxes`, `neighbors`, `edges`, `port_form_directions` and
`preparation_order` declare the four complete fine preparations. Shared sine
rows evolve all twenty coordinates for each positive/negative kick, using
interval-enclosed symbolic phases and masks. Multiplying every original-time
row by outward-enclosed pi applies the clock change `tau=t/pi`. Interval
boxes conservatively enclose the correlated symbolic states; they do not
claim that every independent box point is exactly on the prepared family.

The [uniform flow-error proof](../../../theory/nodal/SINE_REPLICA_PULSE.md#sine-moving-pulse-finite-work-response)
supplies `flow_third_derivative_upper_bound`, `finite_probe_error_upper_bound`,
`analytic_tangent_lower_bound`, `analytic_contrast_lower_bound` and
`fine_edge_margin_lower_bound`. These certify the exact symbolic preparation
independently of numerical success. If a tangent enclosure is available,
`predicted_antisymmetric_response_bounds` widens it by the nonlinear error.

Four independent shared Picard/Taylor steps retain `steps`, `failed_tubes`
and `failure_reasons` in the declared order. They use the globally smooth
sine field; acute identity is covered separately by the analytic margin.
There is no early stopping or retry when one preparation fails. A complete
evaluation gives `response_bounds` and `antisymmetric_response_bounds` from
the centered full-node work readouts. `response_sign=0` means unresolved
direction, not exact zero. If a preparation is unavailable, numerical
response fields are `None`, `status="unavailable"` and named reasons remain;
the independent analytic evidence is retained. Accordingly
`analytic_directional_response_certified` and
`numerical_directional_response_certified` have separate meanings.

The raw four-reading contrast is twice the probe amplitude times the
reported normalized contrast. No measurement precision, laboratory clock,
physical pump, magnetic field or practical signal-to-noise ratio follows.
Direct export uses `tnfr.relational-sine-replica-pulse-finite-work-response.v1`;
the generic SDK wrapper retains all evidence and availability.

<a id="sine-replica-stiffness-trace-curve"></a>
### Necessary stiffness-family screen with observation uncertainty

`assess_sine_replica_stiffness_trace_curve(trace_bounds=...,
determinant_bounds=...)` admits exactly three ordered real scalars or outward
intervals per input through the shared scalar/interval boundary. The inputs
must refer to `tr(M^-1 K)` and `det(M^-1 K)` under independently specified
constant positive-definite mass M and a common clock. This reader does not
identify mass, select a coordinate projection or reconstruct stiffness from
the evaluated response.

The `SineReplicaStiffnessTraceCurve` report retains the primitive boxes,
the conditional geometry coefficient, the trace difference product Q and
cross-multiplied numerator N from the
[proof](../../../theory/nodal/SINE_REPLICA_PULSE.md#sine-pulse-stiffness-discriminator).
`template_residual_bounds=N-a_*Q` tests a necessary template identity;
`affine_obstruction_bounds=(4*N-Q)*Q` tests the symmetric affine one-control
class. The calculation avoids division by small observed trace differences.

`trace_separation_certified` requires Q to exclude zero. Without it both
verdicts are `"unresolved_trace_separation"`. With separated traces,
`template_curve_status="excluded"` requires a residual excluding zero and
`affine_family_status="excluded"` requires strictly positive obstruction.
Otherwise the corresponding verdict is `"not_excluded"`, never a successful
physical admission or exact equality. Wide input boxes may cause abstention.
No threshold, phase fit, time fit, trajectory or runtime mutation is installed.
The coefficient is specific to doubled C5/k1, not a new universal constant.
Direct export uses `tnfr.relational-sine-replica-stiffness-trace-curve.v1`;
the SDK wrapper retains the same primitive evidence and scope.

<a id="sine-replica-pulse-splitting"></a>
### Small-amplitude return splitting

`assess_sine_replica_pulse_splitting(reference_model=..., capacity=...)`
reuses the exact zero-amplitude variation template. It accepts no finite
amplitude, observed graph or horizon. Its `SineReplicaPulseSplitting` report
assesses the analytic family approaching that reference through `m>0`;
the stationary reference itself receives no instability verdict.

The [proof](../../../theory/nodal/SINE_REPLICA_PULSE.md#sine-replica-pulse-splitting)
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
