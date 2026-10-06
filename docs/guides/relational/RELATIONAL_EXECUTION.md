# Native relational execution and observations

Argument-based field and finite-step execution, native tangents, prepared patterns and hypothetical support/state events.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

## Execute the conditional relational model

This preparation is independent of the
[six-node regional observation](OBSERVATION_AND_INFORMATION.md#observe-regional-form-and-its-nodal-response):

```python
from tnfr.sdk import Network, RelationalExchangeModel
import networkx as nx

graph = nx.path_graph(3)
for node, form, capacity in ((0, 0.1, 1.0), (1, 0.0, 0.0), (2, -0.1, 2.0)):
    graph.nodes[node].update(EPI=form, theta=0.0, nu_f=capacity)
network = Network(graph)
model = RelationalExchangeModel(storage_scale=1.0)

field = network.relational_exchange(model)  # detached, no graph mutation
step = network.step_relational(model, dt=0.01)  # one joint Euler step
```

Capacity and support are held; each Euler step validates its full proposal
before committing. Evaluation is read-only. This explicit joint model does not
change `evolve` or `StudySpec` operator-word semantics. Its structural `dt` is
neither a word count nor a calibrated laboratory second.

The default acute phase domain has two explicit wider alternatives. The first
requires certified positive real parts of relative neighbor resultants:

```python
wider_model = RelationalExchangeModel(
    storage_scale=1.0, phase_domain="positive_resultant"
)
wider_field = network.relational_exchange(wider_model)
lower_bounds = wider_field.resultant_real_lower_bounds  # exact fractions
```

The `regular` option also admits negative real parts when the imaginary part
is resolved away from zero. For example, this two-node phase gap is nonacute:

```python
regular_graph = nx.path_graph(2)
for node in regular_graph:
    regular_graph.nodes[node].update(EPI=0.1 * node, theta=2.1 * node, nu_f=1.0)
regular_network = Network(regular_graph)
regular_model = RelationalExchangeModel(1.0, phase_domain="regular")
regular_field = regular_network.relational_exchange(regular_model)
ray_margins = regular_field.resultant_regular_margin_lower_bounds
regular_step = regular_network.step_relational(regular_model, dt=0.01)
segment_margins = regular_step.segment_resultant_regular_margin_lower_bounds
```

Both wider options retain exact admission bounds. The regular option rejects
zero resultants, the nonpositive-real branch ray and unresolved numerical
cases; it uses the same captured relative phase in pressure and metric.
The step additionally checks its entire represented straight proposal chord.

A positive-resultant certificate can admit some nonacute edges. An inconclusive
bound is not proof of a singularity. Successful step admission also does not
promise decreasing storage, exact ODE integration or stability. Inspect retained
balance defects and preserve the supplied coefficients and phase domain. The
[execution contract](../../contracts/relational/RELATIONAL_EXECUTION.md#conditional-relational-execution)
owns state/support admission, segment margins, atomicity and numerical limits;
the [derivation](../../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md)
owns the conditional law and stronger recovery theorems.

### Inspect a moving source at phase consensus

Equal phases do not require equal form or a stationary network. Read the
complete local derivative while keeping the actual initial rates:

```python
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel, relational_report_to_dict

source_graph = nx.path_graph(3)
for node, form, capacity in ((0, 1, 1), (1, -0.5, 2), (2, 0, 0.5)):
    source_graph.nodes[node].update(EPI=form, theta=0, nu_f=capacity)
tangent = Network(source_graph).relational_consensus_tangent(
    RelationalExchangeModel(1)
)
assert any(tangent.field.form_rate) and any(tangent.field.phase_rate)
assert len(tangent.generator) == 6  # all three form and three phase coordinates
tangent_evidence = relational_report_to_dict(tangent)
```

The source is observed without a time step. The Jacobian describes its local
response; it does not keep the later nonlinear evolution at phase consensus.
For exactly uniform form with nonuniform phases, use the separate
`relational_uniform_tangent` admission. Both share their derivative builder
and retain numerical residuals. See the
[tangent contract](../../contracts/relational/RELATIONAL_EXECUTION.md#phase-consensus-tangent-observation)
before using either matrix in a linear approximation.

### Observe a prepared relational pattern

Continue with this `network`, `graph` and `model`. Supply reference phases and
regions rather than interpreting the observer as a pattern detector:

```python
from tnfr.sdk import export_to_json, relational_report_to_dict

pattern = network.relational_pattern(
    model,
    reference_phase={node: 0.0 for node in graph},
    regions=(tuple(graph),),
)
region = pattern.regions[0]
form_deformation = region.form_norm_squared
phase_deformation = region.phase_norm_squared
signed_exchange = region.work.exchange  # Positive: phase storage toward form.
boundary = region.boundary
if boundary.weighted_rate_unavailable_reason is None:
    phase_rate = boundary.phase_weighted_rate
    phase_rounding_residual = boundary.phase_rate_residual
response = region.phase_response  # Unweighted rates, also at zero capacity.
mean_phase_rate = response.mean_rate
mobility_form_contribution = response.covariance_rate
covariance_squared_bound = response.covariance_rate_squared_bound
export_to_json(relational_report_to_dict(pattern), "relational-pattern.json")
```

The report evaluates one fresh field without advancing the graph. Reference
errors use supplied real lifts; keep the same frame when comparing snapshots.
Overlapping regions cannot be summed as a partition. Optional ordered `cycles`
add winding observations through the shared winding owner.

Work, boundary rates and transport have distinct availability conditions. The
zero-capacity node above intentionally makes some quantities unavailable while
form/phase geometry remains observable. Check explicit reasons and `None`
values; do not replace them with zeros. `phase_response` retains unweighted
rates even at zero capacity. The
[report contract](../../contracts/relational/RELATIONAL_EXECUTION.md#prepared-relational-pattern-observations)
centralizes fields, reconstruction, units and support restrictions.

`relational_report_to_dict` supplies the versioned `tnfr.relational-report.v1`
envelope with exact rational data. `export_to_json` writes detached observations;
neither function supplies a restart file or authenticated source record.

### Compare a supplied connection

```python
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel, relational_report_to_dict

left = nx.path_graph(("a", "inside_a"))
right = nx.path_graph(("b", "inside_b"))
for component in (left, right):
    for node in component:
        component.nodes[node].update(EPI=0.0, theta=0.0, nu_f=1.0)
left.nodes["inside_a"]["EPI"] = 1 / 256
attachment = Network(left).relational_attachment(
    Network(right), RelationalExchangeModel(1), bridge=("a", "b")
)
form_rate_change = attachment.form_rate_change  # Exact represented differences.
port_before, port_after = attachment.ports[0].before, attachment.ports[0].after
report = relational_report_to_dict(attachment)
loss_rate_change = attachment.continuous_loss_change
supply = attachment.assess_supply(0)  # Explicitly suppose no event work is supplied.
supply_report = relational_report_to_dict(supply)
```

Both components are evaluated against their hypothetical joined graph; neither
input changes. Labels must be disjoint, and the support/model must meet the
[attachment contract](../../contracts/relational/RELATIONAL_EXECUTION.md#relational-attachment-observation).
The result retains before/after port state, full fields, rate differences and
transport/reset accounting. Equal endpoint form and phase need not imply equal
rates after attachment: degree and phase metric also enter the declared law.
This comparison does not select or create connections autonomously.

`assess_supply` compares declared work with the captured storage increment.
Its `supply_margin` is work minus required supply; a nonnegative margin satisfies
the represented balance under an additional passivity premise. In this example
the endpoint values agree, so zero work covers the zero storage increment even
though the port rates change. Negative work means extraction. No work source or
event time is inferred, and a change in the continuous loss rate is not available
event work. The exact fractions describe captured numerical values; they do not
certify ideal trigonometric storage or physical energy.

### Compare a supplied bridge relocation

```python
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel, relational_report_to_dict

graph = nx.path_graph(4)
for node, form in enumerate((0.0, 2.0, 0.0, 1.0)):
    graph.nodes[node].update(EPI=form, theta=0.0, nu_f=1.0)
relocation = Network(graph).relational_relocation(
    RelationalExchangeModel(1), remove_bridge=(1, 2), add_bridge=(0, 3)
)
assert relocation.components == ((0, 1), (2, 3))
assert relocation.storage_change == -1.5
budget = relocation.assess_supply(0)
assert budget.represented_balance_satisfied
report = relational_report_to_dict(relocation)
```

The graph remains the original path. The comparison retains fresh fields for
the original support and hypothetical exchanged support, both cuts and unique
before/after port cards. The removed bridge defines the ordered components;
the new missing edge must join those same components in that order. All their
internal edges and primitive states remain unchanged. The
[relocation contract](../../contracts/relational/RELATIONAL_EXECUTION.md#relational-relocation-observation)
owns admission and report details.

Here the stored cost decreases by `3/2`. The assessment's `required_supply` is
therefore negative: it is the signed lower bound on net supplied work, rather
than a clipped positive cost. Work `-3/2` would exactly balance declared
extraction; zero work also satisfies the additional passivity premise.
This static comparison does not execute the exchange, choose its time or
certify recovery of a pattern. The conditional two-C5 result has separate
theorem hypotheses; this general graph example does not inherit them.

### Compare a joint state and support reset

Unlike a frozen-state attachment, a complete action can change form or phase
while adding a link. Compare the actual stored endpoints through one owner:

```python
from fractions import Fraction
import networkx as nx
from tnfr.sdk import Network, relational_report_to_dict

before_graph = nx.Graph(((0, 1), (2, 3)))
for node, form in enumerate((0.75, 0.25, 0.25, 0.75)):
    before_graph.nodes[node].update(
        EPI=form, theta=0.0, nu_f=1.0, delta_nfr=0.0
    )
after_graph = before_graph.copy()
for node, form in enumerate((0.625, 0.375, 0.375, 0.625)):
    after_graph.nodes[node]["EPI"] = form
after_graph.add_edge(0, 2, weight=0.5)
reset = Network(before_graph).relational_reset(
    Network(after_graph), storage_scale=1.0
)
assert reset.form_state_change == Fraction(-3, 16)
assert reset.form_support_change == Fraction(1, 64)
assert reset.storage_change == Fraction(-11, 64)
assert reset.identity_residual == 0
payload = relational_report_to_dict(reset)
```

These are supplied snapshots, not execution of an authenticated operator or
an autonomous connection. The positive edge cost is offset by form change
in the complete comparison. The report does not credit earlier dissipation
as stored work or prove separate passivity of intermediate events. Nonunit
support is admissible to this observation but rejects the current unit-support
relational executor. Real UM and RA-then-UM controls are linked from the
[reset contract](../../contracts/relational/RELATIONAL_EXECUTION.md#relational-reset-observation).
