# Conservative sine saddle preparations and certificates

Signed-involution preparation, the C5/private-leaf saddle, directed passage, same-orbit retention, sensitivity, retained-metric forecasts and constitutive controls.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="signed-involution-reduction"></a>

### Reconstruct an exactly symmetric family without losing its fine nodes

This example checks a combined sign/reflection family of a C5 with one private
leaf per receiver node. Its eight coordinates determine all twenty form and
phase coordinates; this is a conditional exact reduction, not a claim that
every state on this graph needs only eight coordinates.

```python
from fractions import Fraction as Q
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_symmetry import assess_sine_involution_reduction

graph = nx.cycle_graph(5)
graph.add_edges_from((i, i + 5) for i in range(5))
for i in range(5):
    graph.nodes[i].update(EPI=Q(i - 2, 10), theta=Q(i - 2), nu_f=1)
    graph.nodes[i + 5].update(EPI=Q(i - 2, 12), theta=Q(i - 2), nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular"
)
source = bound_relational_sine_exchange(graph, reference_model=model)
reduction = assess_sine_involution_reduction(
    source, permutation_indices=(4, 3, 2, 1, 0, 9, 8, 7, 6, 5), sign=-1
)
assert reduction.family_invariance_certified
assert reduction.source_trajectory_reduction_certified
assert len(reduction.representative_indices) == 4
state = reduction.evaluate(
    reduction.source_form_coordinates, reduction.source_phase_coordinates
)
assert state.comparison.epi == source.epi
assert state.comparison.phase == source.phase
assert state.full_row_equality_certified
```

`state.form_rates` and `state.phase_rates` are the retained representative
rows in original structural time. `state.comparison` keeps all ten nodes,
edges, capacities, storage and full rates. The equations in the
[theory](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-involution-saddle-reduction)
use `tau=t/pi` for this conservative unit law; do not mix these rate units.
The [contract](../../contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-involution-reduction)
separates family invariance from source membership. Symmetry of a nominal
center does not justify reducing every member of an independent error box.

Using the same admitted source, inspect the exact saddle on this support:

```python
from tnfr.physics.relational_sine_resonance import assess_sine_c5_leaf_saddle
from tnfr.sdk import relational_report_to_dict

saddle = assess_sine_c5_leaf_saddle(source, cycle=range(5))
assert saddle.target_storage == Q(7, 2)
assert saddle.relative_hyperbolic_pairs == 1
assert saddle.relative_oscillatory_pairs == 8
assert saddle.positive_root_count == 1
assert saddle.growth_rate_bounds.lo > Q(7, 25)
payload = relational_report_to_dict(saddle)
assert payload["report_type"] == "SineC5LeafSaddle"
```

The saddle's exact turns are separate from the source's observed radians.
The report derives its matrices and isolated root from the complete law; it
does not move the graph to that target. Its direction bounds share one
algebraic root, so do not treat their components as independently chosen
initial conditions. The [contract](../../contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-c5-leaf-saddle)
and [local passage proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-involution-saddle-reduction)
distinguish a directed local transition from acquiring zero-to-unit winding
and from reaching a finite-retention region.

<a id="sine-directed-saddle-corridor"></a>

### Certify a nonlinear winding passage without running a trajectory

Use the exact reconstruction above to declare the full state below. Its
contact phases are initially aligned, but their subsequent feedback stays
in the equations. The forms, phase coordinates and strip are rational:

```python
from tnfr.physics.relational_sine_corridor import assess_sine_saddle_corridor

prepared = reduction.evaluate(
    (Q(1, 6), Q(1, 6), Q(1, 4), Q(5, 24)),
    (-Q(9, 5), -Q(9, 10), -Q(9, 5), -Q(9, 10)),
)
corridor = assess_sine_saddle_corridor(
    prepared.comparison, cycle=range(5), lower_phase=-2, upper_phase=-Q(3, 2)
)
assert corridor.directed_exit_certified
assert corridor.exit_winding == 0
assert corridor.initial_momentum == Q(53, 24)
assert corridor.residence_time_upper_bound > 0
payload = relational_report_to_dict(corridor)
assert payload["report_type"] == "SineSaddleCorridor"
```

The [nonlinear proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-directed-saddle-corridor)
certifies a finite passage from this nonacute winding-one preparation to
winding zero. The reported time is an upper bound, not an observed arrival
time. Reversibility does not turn its nonacute endpoint into a retained acute
pattern. Read the [contract](../../contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-directed-saddle-corridor)
before applying the report to a different family or an uncertain source.

<a id="sine-conservative-formation-retention"></a>

### Inspect the analytical formation and retention construction

Use the same admitted `source` to anchor the complete law and support. This
call verifies a mathematical construction; it does not predict what that
captured snapshot will do:

```python
from tnfr.physics.relational_sine_corridor import assess_sine_saddle_formation

formation = assess_sine_saddle_formation(
    source, cycle=range(5), epsilon=Q(1, 2**32)
)
assert formation.formation_existence_certified
assert formation.independent_source_ball_existence_certified
assert formation.target_radius == Q(1, 2**18)
assert formation.retained_scaled_duration == 1
assert formation.retention_band.checkpoint_offset_from_deep_hit == -Q(1, 2)
assert not formation.captured_source_formation_certified
assert not formation.numerical_source_center_available
assert formation.source_radius_prefactor == Q(1, 2**19)
assert formation.source_radius_exponent == 2**39
payload = relational_report_to_dict(formation)
assert payload["report_type"] == "SineSaddleFormation"
```

The exact source radius means `prefactor * exp(-exponent)`. Keep that
representation: converting this example to a floating radius would underflow
and lose its meaning. The target allows independent errors in every form and
phase coordinate, including all environmental nodes.

The [proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-conservative-formation-retention)
connects the two corridors and the retained band on one orbit. Its source and
target centers are defined through that exact flow; their numerical values
remain unavailable. Preparing a concrete source and reserving its response
requires the separate admission in the
[execution plan](../../../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).
The [contract](../../contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-conservative-formation-retention)
separates conditional existence, captured-state observations and forecasts.

<a id="sine-operational-saddle-preparation"></a>

### Materialize the intermediate preparation and inspect its error geometry

Reuse `source` above to anchor the law and support. The returned preparation
contains explicit rational coordinates; the captured graph is not changed:

```python
from tnfr.physics.relational_sine_corridor import prepare_sine_saddle_state
from tnfr.physics.relational_sine_sensitivity import assess_sine_saddle_sensitivity

preparation = prepare_sine_saddle_state(source, cycle=range(5))
assert preparation.preparation_certified
assert preparation.actual_energy_certified
assert preparation.initial_winding == 1
assert not preparation.numerical_zero_winding_source_available
assert len(preparation.prepared_state.epi) == 10
prepared_payload = relational_report_to_dict(preparation)
assert prepared_payload["report_type"] == "SineSaddlePreparation"

sensitivity = assess_sine_saddle_sensitivity(
    preparation.prepared_state, cycle=range(5), phase_radius=Q(1, 1000)
)
assert sensitivity.metric_positive_definite
assert sensitivity.both_tangent_directions_certified
assert len(sensitivity.full_metric) == 20
assert sensitivity.nonlinear_growth_rate_upper_bound == Q(1, 3) + Q(28, 1000)
assert not sensitivity.captured_source_flow_bound_certified
assert relational_report_to_dict(sensitivity)["report_type"] == "SineSaddleSensitivity"
```

The preparation is near the unstable intermediate configuration, not the
zero-winding source reached along its reversed orbit. The sensitivity bound
keeps all form/phase coordinates, but applies only while both compared flows
stay inside the declared phase neighborhood. Neither call advances state.
The forecast adapter below implements propagation of the metric uncertainty,
Taylor remainder bounds and whole-time domain checks. Read the
[contract](../../contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-operational-saddle-preparation)
and [derivation](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-operational-saddle-preparation)
before treating these static certificates as prospective evidence.

<a id="sine-saddle-metric-forecast"></a>

### Exercise the retained-metric solver on a known equilibrium

This deliberately short software example reconstructs uniform zero form
and phase on the same support. Its exact center is constant. It checks the
forecast interface; it does not evaluate the near-saddle formation preparation:

```python
from tnfr.physics.relational_sine_metric_forecast import forecast_sine_saddle_metric

constant_source = reduction.evaluate((0,) * 4, (0,) * 4).comparison
forecast = forecast_sine_saddle_metric(
    constant_source, cycle=range(5), duration=Q(1, 5000),
    time_step=Q(1, 10000), initial_coordinate_radius=Q(1, 2**30),
    phase_radius=Q(3), order=3, max_steps=2,
)
assert forecast.admitted
assert forecast.validated_duration == forecast.duration
assert len(forecast.endpoint) == 20
assert all(value.contains(0) for value in forecast.endpoint)
assert forecast.steps[1].initial_radius == forecast.steps[0].endpoint_radius
assert relational_report_to_dict(forecast)["report_type"] == "SineSaddleMetricForecast"
```

The broad phase domain here contains the analytic equilibrium; it gives a
loose bound suitable for this small demonstration. A research forecast must
freeze its own source, uncertainty, domain, clock, horizon and budget before
evaluation. An unavailable result preserves partial coverage and never
silently changes those choices. The
[contract](../../contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-saddle-metric-forecast)
explains the retained metric ball, original-time rates and distinct report type.

For the same manufactured equilibrium, `phase_radius=None` can instead
request a growth proof from the actual Jacobian on each full Picard tube:

```python
computed = forecast_sine_saddle_metric(
    constant_source, cycle=range(5), duration=Q(1, 10000),
    time_step=Q(1, 10000), initial_coordinate_radius=Q(1, 2**30),
    phase_radius=None, growth_rate_bounds=(Q(0), Q(1)),
    growth_bisections=8, order=3, max_steps=1,
)
assert computed.admitted
assert computed.growth_mode == "whole_tube_jacobian"
assert computed.steps[0].growth_certificate.certified
```

The bracket and number of refinements are fixed before the call. Failure
does not enlarge the bracket. This option covers regions outside the small
saddle phase tube while keeping the same law and uncertainty metric.

<a id="phase-storage-discriminator"></a>

### Compare local agreement with a different global phase barrier

This static comparison uses the C5/private-leaf support already constructed
above. The captured source anchors support and law; the reader builds the
same named rational intermediate preparation as the mathematical study.
It neither evolves the zero equilibrium below nor replays a frozen response.

```python
from tnfr.physics.relational_phase_storage import assess_saddle_storage_discriminator

discriminator = assess_saddle_storage_discriminator(
    constant_source, cycle=range(5), initial_coordinate_radius=Q(1, 2**100),
)
assert discriminator.whole_local_flow_agreement_certified
assert discriminator.alternative_zero_winding_passage_excluded
assert discriminator.discriminator_certified
assert relational_report_to_dict(discriminator)["report_type"] == "SaddleStorageDiscriminator"
```

The supplied alternative matches the complete sine field throughout a proved
local time window, but its conserved storage excludes a later winding change
from this preparation. Its smooth cutoff is not real analytic; no new default
law is selected. The [contract](../../contracts/relational/SINE_SADDLE_CERTIFICATES.md#phase-storage-discriminator)
keeps nominal directional gates, uncertain-state agreement and all-time
winding exclusion separate. Winding protection does not prove permanent
acute geometry or nonformation from every possible source.

<a id="sine-constitutive-robustness"></a>

### Compare the same formation source under the declared cubic law

Run this read-only audit from the repository checkout, or supply the directory
containing its known frozen evidence bundle. It uses the already evaluated
source `R Phi_sine(237,z)` under both laws and fixes `eta=1/100`; it never
executes a changed-law trajectory or alters the original trial.

```python
from tnfr.research.sine_constitutive_robustness import assess_sine_constitutive_robustness
from tnfr.sdk.relational_reports import relational_report_to_dict

audit = assess_sine_constitutive_robustness("docs/assets/sine_metric_connection")
assert audit.status == "certified"
assert audit.source_winding == 0
assert audit.same_source_acute_formation_excluded
assert audit.perturbed_source_storage_bounds.hi < audit.sector_barrier
assert audit.retained_protocol_passed is False
assert audit.independent_numerical_source_ball_certified is False
assert relational_report_to_dict(audit)["report_type"] == "SineConstitutiveRobustness"
```

Here `certified` means the changed law **excludes** acute unit-winding
acquisition from the retained mapped family. The sine law's positive finite
connection and its incomplete full-horizon verdict remain unchanged. The
comparison does not test an independently chosen point in the enclosing box.
See the [contract](../../contracts/relational/SINE_SADDLE_CERTIFICATES.md#sine-constitutive-robustness)
for source roles, evidence premises and the distinction between an available
equilibrium and an accessible pattern.
