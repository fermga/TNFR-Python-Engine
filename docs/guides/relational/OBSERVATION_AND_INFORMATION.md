# Shared observations, information and finite-sample bounds

Regional and source-relative observations, exact linear and phase-moment information, finite-sample evidence and shared circular geometry.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

## Observe regional form and its nodal response

Supply an ordered partition into three-node regions. The order fixes their
reporting frames. This example evaluates configured pressure explicitly;
newly created nodes need not carry stored pressure:

```python
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.sdk import TNFR

network = TNFR.create(6, seed=42).ring()
default_compute_delta_nfr(network.G)
regional = network.regional_form(((0, 1, 2), (3, 4, 5)))
print(regional.regions[0].intensity, regional.regions[0].phase_estimate)
```

The detached report retains means, Cartesian contrasts, cross-region Gram data
and instantaneous rates from `nu_f * stored_DeltaNFR`. It does not refresh
pressure, read primitive phase or measure an elapsed interval. Zero contrast
has no form angle; exact Cartesian/Gram data remain available. Scalar EPI
admission and exact-versus-estimated fields follow the
[form observation contract](../../../theory/nodal/DERIVED_FORM_PHASE.md#collective-interaction-closure-and-relational-state).

To test an independently supplied affine law, use
`tnfr.physics.derive_regional_affine_closure(nodes, regions, generator=G, source=b)`.
Here `b` is a **form rate**, not a pressure or a rate reconstructed from the
response. Inspect `all_state_closed`; projected coefficients alone do not
establish autonomous reduced dynamics. See the
[affine-source theorem](../../../theory/nodal/DERIVED_FORM_PHASE.md#held-affine-source-closure).
These exact Python reports contain `Fraction` values and are not `StudyResult`
JSON or resumable checkpoints.

### Retain orientation relative to a held source

Continue the six-node preparation with an independently declared held source:

```python
# Continue the six-node preparation above; b is a declared model input.
b = (1, -1, 0, 0, 0, 0)
relative = network.source_relative_form(((0, 1, 2), (3, 4, 5)), held_source_rate=b)
print(relative.relative_real, relative.contrast_reconstructible)
```

The report embeds `form` and retains orientation relative to the supplied source
contrasts. A source constant inside every region provides no contrast reference;
`contrast_reconstructible=False` preserves that limitation. A changing source
requires a different rate calculation. The
[source-relative owner](../../../theory/nodal/DERIVED_FORM_PHASE.md#source-relative-engine-integration)
defines exact scaling and reconstruction scope; this flag does not identify
primitive phase/capacity or justify the source physically.

## Bound a prepared coefficient response

These graph-independent functions apply the
[prepared-mode identification theorem](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#prepared-coefficient-identification).
They do not infer phase preparation, modal isolation, baseline/gain or an affine
clock from a plausible scalar waveform. They identify a coefficient combination
conditionally, not the correct nonlinear law or a physical constant.

For independently declared initial value/rate/acceleration intervals:

```python
from tnfr.physics.relational_observations import bound_relational_coefficient_from_jet

jet = bound_relational_coefficient_from_jet(
    form_bounds=(1, 1), rate_bounds=(-3, -3), acceleration_bounds=(7, 7)
)
if jet.coefficient_bounds is None:
    print(jet.unavailable_reasons)
else:
    print(jet.coefficient_bounds)  # Exact outward endpoints; not a precision verdict.
```

For samples at `0,h,2h`, supply independent sample-error and whole-window C3
bounds. This next example is an **exact synthetic quadratic**
`y(t)=1-3*t+7*t*t/2`, not an acquired TNFR trajectory. Zero error and zero third
derivative are justified by that supplied polynomial, not inferred from three
samples:

```python
from fractions import Fraction
from tnfr.physics.relational_observations import bound_relational_coefficient_from_samples
from tnfr.sdk import export_to_json, relational_report_to_dict

samples = bound_relational_coefficient_from_samples(
    (1, Fraction(423, 512), Fraction(87, 128)),
    sample_step=Fraction(1, 16),
    sample_error_bound=0,
    third_derivative_bound=0,
)
assert samples.rate_estimate == -3
assert samples.acceleration_estimate == 7
if samples.jet.coefficient_bounds is None:
    print(samples.jet.unavailable_reasons)
else:
    print(samples.jet.coefficient_bounds)
export_to_json(relational_report_to_dict(samples), "coefficient-samples.json")
```

Inspect the nested `jet`, and keep an independently frozen useful-width policy
separate from availability. Actual numerical/acquired samples require their own
error, timing and regularity evidence. The
[jet](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#relational-coefficient-jet) and
[sample contracts](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#relational-coefficient-samples)
own admission and report fields; the
[known-source P2 control](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-temporal-acquisition)
supplies one specific nonlinear/numerical budget, not a general measurement
model. There is no corresponding `tnfr network` execution mode.

### Compare two independently bounded rates

For the [matched capacity test](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#normalized-capacity-discriminator),
normalize the changed rate by its own initial baseline. Supply independently
justified intervals in a common gain and clock. These illustrative bounds
are declared arithmetic inputs, not acquired TNFR data:

```python
from fractions import Fraction
from tnfr.physics.relational_observations import bound_relational_rate_contrast

contrast = bound_relational_rate_contrast(
    before_bounds=(Fraction(99, 100), Fraction(101, 100)),
    after_bounds=(Fraction(91, 100), Fraction(93, 100)),
)
if contrast.normalized_change_bounds is None:
    print(contrast.unavailable_reasons)
else:
    print(contrast.normalized_change_bounds)
```

The [rate-contrast contract](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#relational-rate-contrast)
owns admission, units and resolution limits. The usual
`relational_report_to_dict(contrast)` projection preserves exact endpoints
and availability. Neither this ratio nor a native field snapshot authenticates
the preparation, the rate-error budget or a physical observation.

For temporal observations, first obtain each rate with
`bound_relational_rate_from_samples`. This synthetic cubic demonstrates the
stencil without assuming the coefficient-identification preparation:

```python
from fractions import Fraction
from tnfr.physics.relational_observations import bound_relational_rate_from_samples

h = Fraction(1, 64)
rate = bound_relational_rate_from_samples(
    tuple(1 + 2*t + t**3 for t in (0, h, 2*h)),
    sample_step=h, sample_error_bound=0, third_derivative_bound=6,
)
assert rate.rate_bounds[0] <= 2 <= rate.rate_bounds[1]
```

The cubic supplies its exact C3 bound. Actual phase samples need a consistent
lift and independent bounds for the entire window. Compute one report per
matched preparation, then pass their `rate_bounds` as `before_bounds` and
`after_bounds` to the contrast observer. The
[sample-rate contract](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#relational-rate-samples)
defines these obligations; the coefficient estimator is not a substitute for
a general phase-rate observation.

The fixed K3 derivative/domain budget can be recomputed without acquiring
samples or evolving a graph:

```python
from tnfr.research.relational_capacity_discriminator import (
    certify_relational_capacity_sampling,
)

admission = certify_relational_capacity_sampling()
assert admission.rate_error_bound < admission.rate_error_limit
```

Its [contract](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#relational-capacity-sampling)
distinguishes the four static candidate/arm checks from a response acquisition.
An available admission does not prove that actual samples satisfy its error
ceiling.

### Audit saved acquisition evidence without replay

```python
from tnfr.research.relational_acquisition import audit_relational_coefficient_acquisition

audit = audit_relational_coefficient_acquisition(
    "artifacts/research/relational_coefficient_acquisition/result.json"
)
if audit.consistent:
    print(audit.completed_steps, audit.recorded_passed, audit.reconstructed_passed)
else:
    print(audit.status, audit.unavailable_reasons)
```

The response, sibling protocol and source archive must be retained together.
Missing local files return `unavailable`; conflicting evidence returns
`inconsistent`. A consistent failed acquisition remains failed. The audit reads
saved fields and reconstructs their error chain without invoking the producer,
evolving a graph or rewriting the files. It does not authenticate acquisition
chronology or admit a physical model. The
[audit contract](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#relational-acquisition-audit)
and [instrument guide](../../../benchmarks/README.md#read-only-temporal-acquisition-audit)
own the supported record and command boundary.

<a id="compare-phase-moment-information"></a>

## Compare information retained by phase moments

This standalone comparison asks whether the same degree and first circular
resultant can conceal different source responses. Supply exact unit phasors
for relative phase gaps; no graph, previous example or trajectory is needed.
The nonnegative `epsilon` declares the existing cubic-sine countermodel.

```python
from fractions import Fraction as Q
from tnfr.physics.phase_response import assess_phase_moment_information
from tnfr.sdk import relational_report_to_dict

left = ((Q(1), Q(0)),) * 3 + ((Q(3, 5), Q(4, 5)),) * 3
right = ((Q(4, 5), Q(3, 5)),) * 5 + ((Q(4, 5), -Q(3, 5)),)
information = assess_phase_moment_information(left, right, epsilon=Q(1))

assert information.degree == 6
assert information.first_resultants == ((Q(24, 5), Q(12, 5)),) * 2
assert information.strictly_acute == (True, True)
assert information.first_resultants_equal
assert information.sufficiency_obstruction
assert information.cubic_source_difference_pi_numerator == -Q(14, 125)
assert information.cubic_storage_difference == -Q(24, 125)

information_evidence = relational_report_to_dict(information)
assert information_evidence["report_type"] == "PhaseMomentInformationAssessment"
```

The source difference is **right minus left**, equal to `-14/(125*pi)`;
its `pi_numerator` field stores `-14/125` exactly. The storage difference
is an unscaled sum over root incidences. The sine source and cosine storage
agree for these inputs, while the declared cubic alternative distinguishes
them. No pressure law is installed, and no complete dynamics is selected.

Use `Fraction(3, 5)` rather than a rounded decimal or a numerically computed
cosine/sine pair: unit length must hold exactly for the admitted coefficients.
Inputs have equal degrees between 1 and 128; the
[contract](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#phase-moment-information)
defines scalar admission, exact equality, geometric flags and export.
The [theory](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency)
explains which additional premises make the first moment sufficient for a
primitive additive source. This finite obstruction concerns an instantaneous
root source, not whether the first moment determines the network's future.

<a id="phase-information-response"></a>

### Predict a finite response from equal initial phase summaries

The static comparison can be continued with both form and phase evolving on
the full star. This example fixes two complete laws and an observation time
before any response is evaluated:

```python
from fractions import Fraction as Q
from tnfr.physics.phase_response import assess_phase_information_response
from tnfr.sdk import relational_report_to_dict

left = ((Q(1), Q(0)),) * 3 + ((Q(3, 5), Q(4, 5)),) * 3
right = ((Q(4, 5), Q(3, 5)),) * 5 + ((Q(4, 5), -Q(3, 5)),)
prediction = assess_phase_information_response(
    left, right, duration=Q(1, 16), epsilon=1,
    preparation_error=Q(1, 100000),
    observation_error=Q(1, 100000),
)
assert prediction.discrimination_certified
assert prediction.form_contrast_centers == (Q(0), -Q(7, 1000))
assert prediction.separation_margin_lower_bound > Q(39, 10000)
evidence = relational_report_to_dict(prediction)
assert evidence["report_type"] == "PhaseInformationResponse"
```

Each trial starts near zero form on a unit-capacity star with the supplied
ideal relative phases. `duration` uses `tau=t/pi`. The observation is the
right trial's root form minus the left trial's root form at that time.
Preparation error applies to every initial form and lifted phase coordinate;
readout error applies to each endpoint root reading. Coefficients, capacities,
support and clock remain fixed. These are declared structural error budgets,
not measured laboratory tolerances.

The two prediction intervals are separated despite finite error. The sine
interval has a nonzero radius because equal initial summaries do not give
equal future responses. Larger budgets may return `unavailable`, which means
this sufficient estimate cannot discriminate. The
[contract](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#phase-information-response)
retains all premises; a physical use still needs independent preparation,
measurement and clock admission.

<a id="phase-moment-motion"></a>

### Retain which rate belongs to each phase

Phase geometry and the distribution of rates need not determine their joint
effect. This exact observation keeps the association:

```python
from fractions import Fraction as Q
from tnfr.physics.phase_response import derive_phase_moment_motion
from tnfr.sdk import relational_report_to_dict

phasors = ((Q(3, 5), Q(4, 5)), (Q(4, 5), Q(3, 5)))
left = derive_phase_moment_motion(phasors, (1, 0), epsilon=0)
right = derive_phase_moment_motion(phasors, (0, 1), epsilon=0)
assert left.first_resultant == right.first_resultant
assert left.sine_source_pi_numerator == right.sine_source_pi_numerator
assert left.sine_source_rate_pi_numerator == Q(3, 10)
assert right.sine_source_rate_pi_numerator == Q(2, 5)
assert left.cosine_storage_rate == Q(4, 5)
assert right.cosine_storage_rate == Q(3, 5)
motion_evidence = relational_report_to_dict(left)
assert motion_evidence["report_type"] == "PhaseMomentMotion"
```

The supplied rates are relative angular rates in one declared clock. The
[complete-star witness](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#phase-motion-information)
derives these two choices from two explicit form states under the conservative
sine law. This reader itself supplies no evolution law. It reports instantaneous
source derivatives and incident work; preserving these quantities does not
establish a closed future evolution. The
[contract](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#phase-moment-motion) owns units,
admission and the distinction from a new primitive parameter.

<a id="sine-star-moment-chart"></a>

### Recover the relative state on a regular star chart

For the complete conservative unit three-node star, use gap rates in
`tau=t/pi`. The two complex moments can then encode the existing relative
state, provided `0<|Z|<2`:

```python
from fractions import Fraction as Q
from tnfr.physics.phase_response import (
    derive_phase_moment_motion,
    derive_sine_star_moment_closure,
)
from tnfr.sdk import relational_report_to_dict

# Forms (0, 3/4, -1/4) give these gap rates under the declared star phase row.
motion = derive_phase_moment_motion(
    ((Q(3, 5), Q(4, 5)), (Q(4, 5), Q(3, 5))), (1, 0), epsilon=0
)
state = derive_sine_star_moment_closure(
    motion.first_resultant, motion.first_rate_weighted_resultant
)
assert state.relative_mean_form == Q(1, 4)
assert state.internal_form_squared == Q(1, 4)
assert state.full_storage == Q(73, 80)
assert state.first_resultant_rate == motion.first_resultant_rate
assert relational_report_to_dict(state)["report_type"] == "SineStarMomentClosure"
```

This is a change of coordinates with four real degrees of freedom, not a
new law or a loss of internal motion. Coincident or antipodal phases lose
information and are rejected by this adapter. The
[contract](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#sine-star-moment-chart)
and [existing-state proof](../../../theory/nodal/SINE_PAIR_STATE.md#sine-star-moment-chart)
explain which information is missing and why the fine nodal law remains valid.
