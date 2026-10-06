# Normalized-sine comparison, inference and forecasts

Shared sine-law report admission, captured comparison, retained mediation, sampling, hidden state/capacity inference and prior-only forecasting.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

## Sine-law assessments and forecasts

The following module-level APIs concern `normalized_sine_reciprocal_exchange`.
A regular `RelationalExchangeModel` supplies coefficients and storage scale;
it does not switch `Network.step_relational` to this law. Each example states
whether it consumes a graph, observation bounds or an exact symbolic family.
When passing one report to another reader, inspect the new report's status,
reasons and actual observation time. The
[chained-report contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission)
defines which evidence is read again; retaining the original source object
does not make its cached fields authoritative.

### Inspect the separate smooth-pressure comparison

The mean-sine complete law is a declared constitutive comparison. Its static
observer reuses graph admission and exact trigonometric bounds:

```python
import networkx as nx
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.sdk import RelationalExchangeModel

graph = nx.path_graph(2)
for node in graph:
    graph.nodes[node].update(EPI=node, nu_f=1, theta=node)
comparison = bound_relational_sine_exchange(
    graph, reference_model=RelationalExchangeModel(1, phase_domain="regular")
)
assert comparison.law == "normalized_sine_reciprocal_exchange"
assert comparison.balance_residual.contains(0)
comparison_evidence = comparison.to_dict()
kinematics = comparison.resultant_kinematics()
assert kinematics.comparison is comparison
kinematics_evidence = kinematics.to_dict()
```

This bounds a supplied alternative's instantaneous field. It does not execute
the graph or replace native argument pressure. The
[contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#detached-normalized-sine-complete-law-comparison)
and [selection/obstruction proof](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#global-closure-pressure-comparison)
explain the extra superposition premise and why smoothness alone does not
select a unique law. The optional kinematics report bounds how each complex
neighbor resultant is changing at this captured state. It neither evolves
the graph nor bounds a future observation window.

### Inspect pressure from a retained intermediary

Use the captured smooth comparison to separate the contribution of an
actual intermediary from visible internal interactions:

```python
import networkx as nx
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.sdk import RelationalExchangeModel

graph = nx.path_graph(3)
for node, form, phase, capacity in (
    (0, 1, 0, 1), (1, 0.25, 0.5, 3), (2, 0, 0.5, 2)
):
    graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
comparison = bound_relational_sine_exchange(
    graph, reference_model=RelationalExchangeModel(1, phase_domain="regular")
)
environment = comparison.mediated_pressure(mediator=1)
assert environment.comparison is comparison
assert environment.ports == (0, 2)
assert environment.hidden_form == 0.25
assert environment.balance_residual.contains(0)
assert len(environment.port_boundary_work) == len(environment.ports)
incoming_to_node_zero = -environment.port_boundary_work[0]
environment_evidence = environment.to_dict()
```

This retains the intermediary's actual state, even when it is away from a
minimum or neighboring phases cancel. Its pressure contributions, hidden
rates and boundary work describe the same fine field. The memory
coefficients refer to the exact causal equations, not an evaluated history
or a simulation. Here node zero has only the mediator as an external
neighbor, so `incoming_to_node_zero` is its incoming work rate. Per-port
contributions have the opposite orientation: into the intermediary's star.
Neither is an accumulated energy or a formation verdict. See the
[contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#retained-state-sine-environmental-pressure).

### Prepare a finite-sample error budget

The sample observer estimates an initial value, rate and acceleration from
three values in a declared real chart. Before collecting values, a supplied
sine-law class can provide independent smoothness bounds:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_sampling import bound_sine_sampling_smoothness

h = Fraction(1, 4096)
noise, jitter = Fraction(1, 2**44), Fraction(1, 2**50)
smoothness = bound_sine_sampling_smoothness(
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_diameter_bound=2,
    capacity_ceiling=2,
    window_start=0,
    window_end=2*h + jitter,
)
budget = smoothness.sample_budget(
    sample_step=h,
    form_sample_error_bounds=(0, noise, noise),
    phase_sample_error_bounds=(0, noise, noise),
    timestamp_error_bounds=(0, jitter, jitter),
)
assert budget.phase.acceleration_error_bound < Fraction(1, 1000)
assert budget.phase_increments_resolved
```

These limits cover every node, including the hidden environment. The zero
initial errors declare an independently exact preparation; they are not
deduced from small later errors. No samples or response are generated.
The [sampling contracts](../../contracts/relational/OBSERVATION_AND_INFORMATION.md#joint-value-rate-and-acceleration-from-finite-samples)
describe `bound_relational_jet_from_samples` for actual data, exact SDK
export and the separate noise, timing and differentiation budgets.
Its initial-value interval must be retained. The point-state inverse below
requires an exact visible state, and a derivative-box witness does not
certify that a trajectory reproduces all earlier samples.

### Infer a hidden state from earlier visible observations

Supply only visible nodes, the known ports and earlier rate intervals.
The example uses illustrative prior intervals under the declared sine law;
they are not a laboratory acquisition:

```python
from fractions import Fraction as Q
import networkx as nx
from tnfr.physics.relational_sine_observation import infer_relational_sine_hidden_state
from tnfr.sdk import RelationalExchangeModel

visible = nx.Graph()
visible.add_node("left", EPI=1, theta=0, nu_f=1)
visible.add_node("right", EPI=0, theta=0.5, nu_f=2)
inference = infer_relational_sine_hidden_state(
    visible,
    ports=("left", "right"),
    form_rate_bounds={
        "left": (Q(-336, 1000), Q(-335, 1000)),
        "right": (Q(171, 1000), Q(172, 1000)),
    },
    phase_rate_bounds={
        "left": (Q(119, 1000), Q(120, 1000)),
        "right": (Q(-80, 1000), Q(-79, 1000)),
    },
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    source_id="illustrative-prior-intervals",
    clock_id="declared-structural-clock",
    observation_time=0,
    evidence_window=(0, 0),
    forecast_start=1,
)
assert inference.status == "bounded_candidate"
assert inference.hidden_capacity_identified is False
inference_evidence = inference.to_dict()

# An illustrative earlier phase-acceleration interval adds capacity evidence.
capacity = inference.infer_capacity(
    form_acceleration_bounds={},
    phase_acceleration_bounds={"left": (Q(-95, 1000), Q(-92, 1000))},
    source_id="illustrative-prior-acceleration",
    clock_id="declared-structural-clock",
    observation_time=0,
    evidence_window=(0, 0),
)
assert capacity.status == "bounded_candidate"
assert capacity.capacity_bounds.contains(2)
capacity_evidence = capacity.to_dict()
```

A surviving enclosure does not establish that all uncertain inputs share
one exact solution. Inspect status and reasons; phase alignment can leave
the phase unresolved even when form is bounded. The state refers to time
zero here, and cannot be silently treated as the state at forecast time one.
The additional acceleration bounds constrain capacity under the same model;
they do not propagate either estimate to the forecast time. An actual
prediction requires an admitted joint state/capacity and propagation.
See the [contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#hidden-state-inference-from-prior-visible-rates).
The [capacity contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#hidden-capacity-inference-from-prior-accelerations)
also explains why a capacity bound can survive unresolved phase and why
that does not suffice for predicting the whole state.

### Issue a prior-only sine forecast

After constructing a capacity report as above, call `admit_sine_prior(capacity)`
from `tnfr.physics.relational_sine_forecast` and inspect `admitted` and `reasons`.
For an admitted report, call `forecast_sine_prior` with exact `end_time`,
`time_step` and optional `order`; retain its `validated_end_time` even if the
requested horizon is unavailable.
See the [API contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#joint-prior-admission-and-sine-forecasts)
for exact-time admission, report fields and failure behavior.

The [reserved software instrument](../../../benchmarks/relational_sine_prior_forecast.py)
separates preparation, prediction and evaluation into three invocations:

```sh
python -m benchmarks.relational_sine_prior_forecast --prepare
python -m benchmarks.relational_sine_prior_forecast --predict
python -m benchmarks.relational_sine_prior_forecast
```

Use the working source environment described in the testing guide. The
retained v1 run has already been evaluated; these commands describe its
protocol and are not maintenance checks to rerun. Each stage exclusively
creates its records and refuses to replace earlier evidence, including
failures. Missing archived results remain unavailable.

Preparation writes the public prior/protocol, a separate source-state
record and a complete source archive. Prediction consumes only public
prior evidence; it neither reads that source-state record nor generates a
future source response. Evaluation requires the already issued prediction
and verifies the sealed source, numerical contract and retained hashes.
The frozen-mediator control changes a capacity premise; it is not another
model fitting the full prior acceleration evidence. Synthetic derivatives
and known-source software checks are not laboratory measurements.

### Inspect a conditional intermediary minimum

The smooth comparison can also expose the visible response obtained by
placing one supplied intermediary at its conditional storage minimum:

```python
import networkx as nx
from tnfr.physics.relational_sine_mediation import bound_relational_sine_mediation
from tnfr.sdk import RelationalExchangeModel

graph = nx.path_graph(3)
for node, form, capacity in ((0, 1, 1), (1, 0.5, 4), (2, 0, 2)):
    graph.nodes[node].update(EPI=form, theta=0, nu_f=capacity)
reduction = bound_relational_sine_mediation(
    graph, mediator=1,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
)
assert reduction.nodes == (0, 2)
assert reduction.hidden_form == 0.5
assert reduction.balance_residual.contains(0)
assert reduction.hidden_form_tracking_defect.hi < 0
reduction_evidence = reduction.to_dict()
```

The negative tracking defect matters: although the hidden rates vanish at
the minimum, the visible nodes move its required position. Finite hidden
capacity therefore does not imply an exact instantaneous reduction.
Even zero defect is insufficient near a disappearing resultant: the
[autonomous path example](../../../theory/nodal/SINE_ENVIRONMENTAL_MEMORY.md#autonomous-path-cancellation)
crosses cancellation while the actual intermediary remains fixed and its
conditional minimum changes.
The graph keeps all three nodes. See the
[contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#detached-stationary-sine-mediation)
for source provenance, retained degrees, admission and exact JSON export.
