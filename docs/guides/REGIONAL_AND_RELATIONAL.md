# Regional and relational SDK workflows

This guide uses shared engine owners through `Network` and module-level
observers. It complements the
[CLI and SDK introduction](../CLI_AND_SDK.md); these Python routes are not
additional CLI study modes. Supplied partitions, models, references and support
are inputs, not evidence of spontaneous NFR formation.

| Task | API | Detailed owner |
| --- | --- | --- |
| Observe stored form and its rate | `regional_form`, `source_relative_form` | [Form geometry and closure](../../theory/nodal/DERIVED_FORM_PHASE.md) |
| Evaluate or advance a declared joint law | `relational_exchange`, `step_relational` | [Relational execution](../contracts/RELATIONAL_DYNAMICS.md#conditional-relational-execution) |
| Differentiate the joint law at uniform form | `relational_uniform_tangent` | [Tangent contract](../contracts/RELATIONAL_DYNAMICS.md#uniform-form-tangent-observation); detached derivative, not an equilibrium or pulse certificate |
| Observe a prepared region | `relational_pattern` | [Pattern report](../contracts/RELATIONAL_DYNAMICS.md#prepared-relational-pattern-observations) |
| Compare a supplied bridge | `relational_attachment` | [Attachment](../contracts/RELATIONAL_DYNAMICS.md#relational-attachment-observation) |
| Compare a supplied bridge relocation | `relational_relocation` | [Relocation](../contracts/RELATIONAL_DYNAMICS.md#relational-relocation-observation) |
| Test an endpoint basin or a continuous transit | Capture methods below | [Capture scopes](../contracts/RELATIONAL_DYNAMICS.md#conditional-relational-capture) |
| Bound a prepared coefficient response | Jet/sample functions below, not `Network` methods | [Coefficient uncertainty](../contracts/RELATIONAL_DYNAMICS.md#relational-coefficient-samples) |
| Audit a retained P2 acquisition | `tnfr.research.relational_acquisition` | [Read-only record audit](../contracts/RELATIONAL_DYNAMICS.md#relational-acquisition-audit) |

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
[form observation contract](../../theory/nodal/DERIVED_FORM_PHASE.md#collective-interaction-closure-and-relational-state).

To test an independently supplied affine law, use
`tnfr.physics.derive_regional_affine_closure(nodes, regions, generator=G, source=b)`.
Here `b` is a **form rate**, not a pressure or a rate reconstructed from the
response. Inspect `all_state_closed`; projected coefficients alone do not
establish autonomous reduced dynamics. See the
[affine-source theorem](../../theory/nodal/DERIVED_FORM_PHASE.md#held-affine-source-closure).
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
[source-relative owner](../../theory/nodal/DERIVED_FORM_PHASE.md#source-relative-engine-integration)
defines exact scaling and reconstruction scope; this flag does not identify
primitive phase/capacity or justify the source physically.

## Bound a prepared coefficient response

These graph-independent functions apply the
[prepared-mode identification theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#prepared-coefficient-identification).
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
[jet](../contracts/RELATIONAL_DYNAMICS.md#relational-coefficient-jet) and
[sample contracts](../contracts/RELATIONAL_DYNAMICS.md#relational-coefficient-samples)
own admission and report fields; the
[known-source P2 control](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#coefficient-temporal-acquisition)
supplies one specific nonlinear/numerical budget, not a general measurement
model. There is no corresponding `tnfr network` execution mode.

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
[audit contract](../contracts/RELATIONAL_DYNAMICS.md#relational-acquisition-audit)
and [instrument guide](../../benchmarks/README.md#read-only-temporal-acquisition-audit)
own the supported record and command boundary.

## Execute the conditional relational model

This preparation is independent of the six-node observation above:

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

The default acute phase domain has a separate, explicit wider alternative:

```python
wider_model = RelationalExchangeModel(
    storage_scale=1.0, phase_domain="positive_resultant"
)
wider_field = network.relational_exchange(wider_model)
lower_bounds = wider_field.resultant_real_lower_bounds  # exact fractions
```

A positive-resultant certificate can admit some nonacute edges. An inconclusive
bound is not proof of a singularity. Successful step admission also does not
promise decreasing storage, exact ODE integration or stability. Inspect retained
balance defects and preserve the supplied coefficients and phase domain. The
[execution contract](../contracts/RELATIONAL_DYNAMICS.md#conditional-relational-execution)
owns state/support admission, segment margins, atomicity and numerical limits;
the [derivation](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md)
owns the conditional law and stronger recovery theorems.

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
[report contract](../contracts/RELATIONAL_DYNAMICS.md#prepared-relational-pattern-observations)
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
[attachment contract](../contracts/RELATIONAL_DYNAMICS.md#relational-attachment-observation).
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
[relocation contract](../contracts/RELATIONAL_DYNAMICS.md#relational-relocation-observation)
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
[reset contract](../contracts/RELATIONAL_DYNAMICS.md#relational-reset-observation).

### Check a protected relational basin

Initialize two exactly copied/reflected five-node rings with matching
adjacent-port bridges:

```python
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel

graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
graph.add_edges_from(((0, 5), (1, 6)))
phase = (9 / 4, -9 / 4, -9 / 8, 0.0, 9 / 8)
for node in graph:
    graph.nodes[node].update(EPI=0.0, theta=phase[node % 5], nu_f=1.0)
model = RelationalExchangeModel(1.0, phase_domain="positive_resultant")
capture = Network(graph).relational_capture(
    model, cycles=(tuple(range(5)), tuple(range(5, 10)))
)
assert capture.admitted
assert capture.target_sector == 1
energy_interval = capture.storage_bounds  # exact rational enclosure
```

This is a read-only application of a sufficient continuous-law theorem, without
trajectory simulation. `target_sector` describes the ideal limiting state;
`winding` describes the current snapshot. An unavailable certificate keeps its
reasons and no target. It does not guarantee Euler-step stability.

For an actual endpoint near a declared target, a different local certificate
does not require exact reflection:

```python
from tnfr.sdk import relational_report_to_dict

local = Network(graph).relational_local_capture(
    model, cycles=(tuple(range(5)), tuple(range(5, 10))), target_sector=1
)
local_evidence = relational_report_to_dict(local)
```

The local method can return unavailable for the preparation above. A third
method, `Network.relational_sector_capture(model, cycles=..., target_sector=1)`,
uses the full-state acute-sector basin for sectors `+1` and `-1`. These methods
have different capacity, storage, geometry and energy hypotheses; compare the
[capture contracts](../contracts/RELATIONAL_DYNAMICS.md#conditional-relational-capture)
and [maintained example](../../examples/08_emergent_geometry/180_relational_exchange.py)
before choosing one. None certifies numerical path error from an earlier state.

### Validate continuous transit to a protected basin

The transit method attempts to enclose the ideal ODE from a supplied initial
state to a sufficient basin, without advancing the stored network. Continue
with the two-ring graph and positive-resultant model above:

```python
from fractions import Fraction
from tnfr.sdk import export_to_json, relational_report_to_dict

transit = Network(graph).relational_transit_capture(
    model=model,
    cycles=(tuple(range(5)), tuple(range(5, 10))),
    horizon=Fraction(1, 4),
    time_step=Fraction(1, 32),
    order=12,
    requested_sector=None,
)
print(transit.status, transit.validated_horizon, transit.target_sector)
export_to_json(relational_report_to_dict(transit), "relational-transit.json")
```

This illustrates the interface without promising successful admission. Use
positive exact `Fraction` or integer durations; floats and Booleans reject.
`requested_sector=None` accepts any certified limiting sector; the default
requires the positive sector. These are requested proof criteria, not changes
to the dynamics or a prediction of the target.

Inspect `admitted` explicitly: consensus has `target_sector=0`. An unsuccessful
calculation retains its validated prefix and unavailable reasons; it cannot be
reported as successful capture. Whole-time enclosure and numerical settings
are specified by the
[transit contract](../contracts/RELATIONAL_DYNAMICS.md#validated-conditional-relational-transit)
and its [proof owner](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-validated-transit).
A continuous certificate does not alter the verdict of an earlier frozen
finite-executor experiment.
