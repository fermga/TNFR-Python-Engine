# Passive transport measurement protocol

**Scope:** a supporting measurement/error contract for two declared pure-EPI
transport models. Physical admission is `not_admitted`; an admitted physical
prediction has not been tested. The [execution plan](FIVE_STAGE_EXECUTION_PLAN.md#supporting-measurement-bridge)
owns P1-P5 and any reopening. No data search, acquisition or refit is active.

The mathematical candidates below remain useful with their own hypotheses.
Their observation maps are proposed realizations, not identification of
primitive EPI with every sensor reading or proof of material emergence.
The user has a computer and no laboratory apparatus; apparatus descriptions
are requirements for a matching public terrestrial source, not purchase tasks.

The [archived source reviews and Volts experiment](archive/measurement/TRANSPORT_SOURCE_REVIEWS.md)
preserve inspected-source metadata, rejected routes, source hashes and the
unfavorable affine-control comparison. The separate
[TCLab record](archive/measurement/TCLAB_EXPLORATORY_RECORD.md) retains its own
model and evaluated data. These observations cannot become unseen again.

## Historical source and experiment routes

<a id="bounded-directed-source-decision-2026-09-26"></a>
<a id="completed-exploratory-record-2026-09-18"></a>
<a id="computer-only-candidates-and-model-selection"></a>
<a id="directed-source-decision-2026-09-26"></a>
<a id="follow-up-admission-search"></a>
<a id="historical-scalar-source-selection-2026-09-18"></a>
<a id="mostr-selected-intake-not-physical-admission"></a>
<a id="multinode-source-review-2026-09-18"></a>
<a id="predeclared-volts-exploratory-delivery-before-decoding-values"></a>
<a id="primary-source-inventory"></a>
<a id="renewed-terrestrial-source-review-2026-09-21"></a>
<a id="terrestrial-source-review-2026-09-21"></a>
<a id="volts-exploratory-record"></a>
<a id="volts-model-boundary"></a>

These earlier anchors refer to the [archived review and experiment](archive/measurement/TRANSPORT_SOURCE_REVIEWS.md), not additional current tasks.

## Derived fixed-reference candidate

On the same fixed symmetric positive P2 support, select **pure EPI pressure**
and capacities `(nu,0)`, with `nu>0`. All other pressure channels are inactive;
otherwise the capacity difference itself could contribute another channel.
The canonical nodal equation gives

```text
p = (r-x, x-r),           x' = nu*(r-x),           r' = 0
d = x-r,                 d' = -nu*d
x(t) = exp(-nu*t)*x(0) + (1-exp(-nu*t))*r.
```

This follows from the nodal law without importing the circuit or cooling
equation. The arithmetic mean is NOT conserved; the reference `r` is.
An inactive node may retain nonzero pressure: stationarity is not structural
equilibrium. The existing diffusion energy owner accepts degenerate mobility;
positive-capacity certificates using `d_i/nu_i` do not cover this boundary.

Physical use requires an independently specified constant reference and
support, adequate state observation and bounded external loading. Flat bath
measurements do not prove that a bath is a zero-capacity TNFR node; an active
controller may instead be an external input. A varying reference requires a
distinct driven model and its complete input history, not a fitted final
asymptote. No held-out value may estimate `r` or the clock conversion.

For this candidate, existing log/exp enclosures can be reused with contrast
`x-r` and decay factor `exp(-nu*t)`. The capacity ratio divides by elapsed
time, not twice elapsed time. There is no constant-mean rejection rule.
Do not fabricate a second observed channel or reuse the equal-capacity
adapter unchanged. The mathematical implementation now lives in
[p2_transport_reference.py](../../src/tnfr/physics/p2_transport_reference.py):
`bound_fixed_reference_capacity` and `bound_fixed_reference_transport` reuse
the same contrast/log, decay/exp and convex interval kernels as the equal-
capacity model. The reference may be a declared interval and the initial
contrast may have either strict sign. The returned outer intervals do not
prove one common latent fit. These functions do not mutate a live graph or
admit empirical observations. Physical admission still requires the source
contract; this remains one candidate within P2.3, not another queue.

## Reference apparatus and canonical equal-capacity prediction

The reference design uses two nominally equivalent capacitive elements, observed
simultaneously, prepared separately and then interconnected through one
fixed passive link. Charging sources are disconnected before the declared
observation interval. Record wiring, switch timing, sensor input loading,
component characterization and ambient conditions. Equivalence and negligible
external forcing over the finite interval require measured bounds; component
labels alone do not establish them. The initial experiment changes only the
prepared contrast, keeping the link and apparatus fixed.

The hardware is a candidate realization, not an imported circuit law.
The proposed TNFR model is the isolated EPI channel on fixed P2 support,
with one common positive capacity and no active phase or capacity-feedback
channel. Each physical voltage is mapped to a signed EPI coordinate `x_i`
by a fixed sensor scale and offset. Both channels must share the same
structural unit after independently calibrated sensor correction. Define
structural time by `t = b * (tau - tau_0)`, where `tau` is the independently
checked acquisition clock and `b > 0` is fixed before capacity estimation.
A declared unit convention for `b` is permitted if labeled as a convention;
it is not evidence of a universal physical-to-structural conversion.

The [canonical diffusion owner](../../src/tnfr/physics/structural_diffusion.py)
then gives, without using a circuit governing law as an axiom,

```text
p_1 = x_2 - x_1,                 p_2 = x_1 - x_2
dx_1/dt = nu * p_1,              dx_2/dt = nu * p_2,       nu > 0
m = (x_1 + x_2)/2,               d = x_1 - x_2
dm/dt = 0,                      dd/dt = -2 * nu * d
d(t) = d(0) * exp(-2 * nu * t)
x_1(t) = m(0) + d(t)/2,          x_2(t) = m(0) - d(t)/2
```

These are predictions of the stated nodal model. Pressure is calculated
from the two observed coordinates, never defined as a measured derivative
divided by fitted capacity. Only the product `nu * b` is identifiable from
a physical-time contrast trace if both are unknown. Fixing the clock bridge
is therefore an admission condition, not a second fit parameter. Estimate
one positive `nu` from calibration runs alone and freeze its uncertainty
set. Nonpositive or unidentifiable capacity blocks this positive-capacity
model; it is not clipped into its domain.

For P2, uniformly scaling the single positive edge weight leaves `L_rw`
unchanged. Changing a resistor or link strength cannot be represented merely
by rescaling that edge at fixed capacity. This first protocol consequently
does not sweep link strength or infer a universal relation between capacity
and hardware component values. An intervention is a recorded preparation,
not automatically a canonical glyph or a U3 certificate.

## State, clock and measurement admission

| Item | Required frozen specification | Current missing evidence |
| --- | --- | --- |
| State map | Two sensor identities; common EPI unit; fixed offsets/scales; range and linearity bounds | Apparatus and sensor calibration absent |
| Time | Raw timestamps, synchronization/skew bounds, finite positive bridge, origin rule and gap policy | Physical clock, sampling period and uncertainty absent |
| Support and input | Known P2 wiring, fixed link, source disconnection, observation interval and loading/leakage bounds | Wiring and external-input characterization absent |
| Capacity | Calibration-only common positive rate and identified uncertainty; no outcome-dependent choice | No calibration runs |
| Phase | Unavailable as an oscillatory observable; phase channel explicitly inactive in the selected model | Physical adequacy of that restriction untested |
| State sufficiency | Actual generator and observation map; distinguish unknown source terms from measurement noise | Model-specific apparatus check pending |
| Reserved observations | Disjoint preparation/run IDs, hashes, initial-sample allowance and fixed forecast horizon | No acquired runs or reserved data |
| Decision | Predeclared joint finite-trajectory acceptance tube, rejection and abstention rules | Instrument, calibration and numerical error budgets unset |

With both EPI coordinates measured, the selected two-state model has
observation matrix `C = I`. Its algebraic state observability is immediate;
this does not prove that the apparatus has only those two relevant states.
With contrast alone, the mean is hidden but the contrast equation closes
under the common-capacity hypotheses. Do not promote that to a full-state
claim. Reuse [observability](../../src/tnfr/physics/observability.py) on the
actual generator and observation map, and the hidden-source distinction in
[derived EPI memory](../../src/tnfr/physics/epi_memory.py) if a reduction is
proposed. An algebraic rank result supplies neither noise conditioning nor
an instrument error bound. A material unmodeled leakage/loading contribution
requires a revised admitted model, not a retrospectively invented pressure.

## Calibration, forecast and finite decision

Use separate whole preparations for calibration and evaluation. Calibration
may estimate only the declared common capacity after sensor and clock
mapping are fixed. Freeze the fit rule, accepted domain, parameter interval,
data identities and code revision. No graph, normalization, filter, channel,
capacity, tolerance or horizon is selected using a reserved outcome.

The evaluation initialization allowance is the first synchronized sensor
pair, with its uncertainty, at a predeclared forecast origin after the
switching interval. Choose this origin from the hardware timing/calibration
rule, not from the later relaxation trace. Forecast all subsequent requested
timestamps before reading their values. No full-record centering, Hilbert
transform, interpolation across unexplained gaps or future-aware smoothing
is needed for this prediction.

Primary observation: the held-out contrast trajectory under the frozen
capacity and clock bridge. The constant-mean prediction is a separately
reported check of the same admitted model, not an ignored nuisance when it
fails. Construct a simultaneous finite-trajectory tube from the declared
initial-state, sensor, time, capacity and numerical uncertainty. A collection
of pointwise confidence intervals is not automatically such a tube. If the
uncertainty cannot be bounded at the declared scope, report `inconclusive`.
Exceeding the frozen acceptance rule rejects this mapping on these runs;
agreement supports only its declared finite domain. Report every reserved
run, including failures, missing samples and exclusions with reasons.

## Implemented continuous-time uncertainty boundary

The numerical part of admission now has an executable owner:
[p2_transport_reference.py](../../src/tnfr/physics/p2_transport_reference.py),
with measurement/run integration in
[validation/p2_transport.py](../../src/tnfr/validation/p2_transport.py).
It uses the existing rational logarithm and negative-exponential enclosures;
it introduces no pressure term, circuit governing law or new nodal solver.
All uncertainty bounds below are supplied hypotheses. Their instrumental
justification is still missing and remains a physical admission gate.

### Continuous capacity, distinct from the Euler control

The earlier P1 finite-increment fit remains an engineering control. Even for
an ideal noiseless P2 trajectory sampled at a common structural step `h`,
its fitted coefficient is

```text
nu_Euler = (1 - exp(-2*nu*h)) / (2*h) < nu,     nu>0, h>0.
```

It cannot be silently relabeled as the continuous physical rate. Irregular
pooled increments do not in general have this single-step conversion.
The new calibration uses each declared run's fixed first and last samples;
interior samples are not fitted, and endpoint consistency does not validate
the whole calibration trajectory. Durations and endpoint selection must not
be changed after opening reserved outcomes.

After fixed sensor conversion, suppose the initial and final contrast have
one resolved common sign. Reverse both signs if negative, giving positive
intervals `D0=[a,b]` and `D1=[c,d]`. For elapsed structural time
`T=[sL,sU]`, `sL>0`, require resolved decay `d/a<1`. Then

```text
r = D1 / D0 is enclosed by [c/b, d/a] = [rL,rU]
nu is enclosed by [-log(rU)/(2*sU), -log(rL)/(2*sL)].
```

Rational log bounds round the interval outward. Initial/final mean intervals
must overlap. Each run must resolve nonzero contrast and positive decay;
incompatible, sign-ambiguous or unidentifiable inputs raise explicit errors.
Intersect all run capacity enclosures. A nonempty intersection is a necessary
outer consistency condition, not proof of one joint latent fit. Finally round
the capacity bounds outward to binary64 rational endpoints to limit arithmetic
growth; a lost positive lower bound or infinite upper bound blocks this path.
This computational enclosure is separate from measurement uncertainty.

### Reserved joint outer tube

`P2MeasurementBounds` freezes two offsets/scales, total EPI-coordinate errors,
timestamp error and a positive structural-time bridge interval. Coordinate
errors include uncertainty in the sensor conversion itself. A bound `eps_t`
on each timestamp gives elapsed-time error `2*eps_t`; time zero is the same
initial event, so its duration interval is exactly zero. Unresolved positive
durations are rejected. The declared sensor/clock bounds must hold jointly
throughout the full finite experiment; no confidence level is inferred.
Both channels in each pair must represent the same latent observation time.
Acquisition skew needs synchronization evidence or an independently derived
coordinate-error allowance for time alignment. Timestamp error alone cannot
align asynchronous channels or justify their constant-mean test.

For capacity in `[nuL,nuU]` and elapsed time in `[tL,tU]`, the decay lies in
`[exp(-2*nuU*tU), exp(-2*nuL*tL)]`, evaluated with rational enclosures.
Coordinates use the positive semigroup directly:

```text
x1(t) = ((1+e)*x1(0) + (1-e)*x2(0))/2
x2(t) = ((1-e)*x1(0) + (1+e)*x2(0))/2,   e=exp(-2*nu*t).
```

Monotonicity in the initial coordinates and corner extrema in `e` give
sound boxes while preserving the mean/contrast dependency. The mean and
contrast are also retained as separate observable tests. The shared exponent
cap is 4096 and the reference accepts at most 10,000 samples; these are
computation guards, not physical constants. Rational output endpoints must
not be converted to ordinary floats and called certified bounds.

`calibrate_p2_transport` checks the known undirected two-node positive edge;
`forecast_p2_transport` consumes only frozen calibration, reserved identities,
initial measurements and timestamps. The latter must be a predeclared schedule
or clock metadata available without inspecting future response values.
`write_p2_transport_forecast` saves exact rational endpoints as hexadecimal
numerator/denominator strings; retain its content hash before opening outcomes.
`score_p2_transport` reuses the whole-acquisition separation and requires that
issued hash, matching units/times and unchanged initialization.

Disjoint coordinate, mean or contrast boxes give
`incompatible_with_declared_bounds`. If every outer box overlaps, the result
is only `not_falsified_by_enclosures`: different boxes may admit different
latent parameters. That is neither a joint-feasibility proof nor statistical
acceptance. The physical status remains `not_admitted_by_this_score` in both
cases. Excessive tube width requires a preregistered resolution/power decision;
it cannot become positive evidence just because all observations overlap.

Independent analytic/Decimal fixtures and wrong-mean, wrong-identity and
changed-forecast controls exercise this boundary. These are software/model
tests, not measured data. Existing manifest/sidecar owners can bind the saved
bytes without acquiring a second evidence ledger. P1's Euler tests and their
historical checkpoint remain unchanged.

## Proposed pilot resources, not instrument tolerances

The following are provisional engineering caps for admission planning.
They do not authorize acquisition with unresolved physical tolerances and
are not TNFR constants or a statistical power calculation.
Under the computer-only constraint these apparatus/preparation rows are
requirements a matching public archive would need to document, not tasks
for the user to build or acquire equipment.

| Resource | Proposed cap and stop rule |
| --- | --- |
| Apparatus | One P2 setup, two synchronized observation channels, one unchanged link |
| Preparations | Two calibration runs at one nonzero prepared contrast; two held-out contrast conditions with three independent preparations each: eight total |
| Record size | At most 10,000 paired samples per run and 100 MiB for the complete raw/calibration/prediction bundle |
| Acquisition duration | At most 30 minutes per run; choose a shorter fixed horizon only from apparatus/calibration evidence before held-out acquisition |
| Data search/download | This metadata review is complete. Before any later signal download, pin exact files and compressed/expanded bounds; proposed pilot caps 100 MiB compressed and 250 MiB expanded |
| Computation | One frozen calibration specification and one prediction per reserved preparation; no automatic refitting or larger campaign after rejection/inconclusiveness |

Sampling interval, voltage/temperature limits, switch settling allowance,
clock accuracy, sensor skew, error bars and acceptance thresholds remain
unset. They must be supplied by the selected instrument and calibration,
not by these sample/storage budgets. If an adequate record cannot fit the
caps, revise and version the protocol before observing reserved outcomes.

## Evidence bundle and current exit gate

Reuse [CoreExperimentManifest](../../src/tnfr/research/core_manifests.py)
and [EvidenceSidecar](../../src/tnfr/research/evidence_sidecar.py). Retain
this annex, exact raw/calibration/evaluation file identities, source revision,
raw timestamps/masks, immutable fit, forecasts and complete result ledger in
one accessible bundle. `validate_metadata()` checks declarations only;
`validate_for_admission(root_dir=...)` additionally checks real file bytes.
Neither establishes physical adequacy, temporal commitment or scientific
truth. Preserve an independently auditable pre-evaluation forecast record;
a later digest alone cannot prove when a prediction was issued.

Declare the five measurement-provenance questions explicitly: future
samples, outcome-derived wiring, fitting on evaluation data, evaluation
labels and post-selection. An affirmative answer can describe exploratory
evidence but cannot silently become prospective validation. Numerical
certificate tolerance is separate from the instrument/statistical budget.

P2 remains `not_admitted` until one accessible public-data choice resolves
the missing table, freezes all tolerances and resources, and passes the
state/clock/identifiability gate. Volts remains a historical exploratory record
of the earlier fixed-reference model/map pair, not an evaluation of the revised
generative candidate and not reserved data for a later redesign. The execution
plan owns when this parked branch resumes;
another fit to this same curve cannot supply missing apparatus evidence or
independent preparations. No apparatus purchase/acquisition is requested.
No admitted physical prediction has yet been tested.
