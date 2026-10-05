# Regional and relational SDK workflows

This guide uses shared engine owners through `Network` and module-level
observers. It complements the
[CLI and SDK introduction](../CLI_AND_SDK.md); these Python routes are not
additional CLI study modes. Supplied partitions, models, references and support
are inputs, not evidence of spontaneous NFR formation.

Choose the model and the kind of evidence before choosing an API. The
[contract index](../contracts/RELATIONAL_DYNAMICS.md) owns detailed admission;
this guide owns usage. Examples are individual preparations, with explicit
links where one reuses another section's state. They are not one sequential
experiment or a research task queue.

| Task family | Examples | Model and evidence |
| --- | --- | --- |
| Read regional form or a supplied source-relative orientation | [Regional observations](#observe-regional-form-and-its-nodal-response) | Stored nodal rate or independently declared source; no pressure refresh by the observer. |
| Bound coefficients or finite-sample rates | [Observation budgets](#bound-a-prepared-coefficient-response) | Module-level bounds, with supplied preparation, clock and errors. |
| Execute or observe native relational dynamics | [Field and step](#execute-the-conditional-relational-model), [patterns](#observe-a-prepared-relational-pattern), [support changes](#compare-a-supplied-connection) | Native Arg law; a detached hypothetical event does not alter support. |
| Assess native capture or retained memory | [Capture](#check-a-protected-relational-basin), [memory and continuous proofs](#native-pattern-memory-and-continuous-proofs) | Separate endpoint, continuous-flow and retained-evidence contracts. |
| Distinguish formation evidence from a necessary budget | [Native transit](#validate-continuous-transit-to-a-protected-basin), [retained proof](#audit-robustness-of-the-retained-formation-proof), [sine eligibility](#check-a-formation-preparation-before-evolving-it), [static retention](#certify-recovery-of-the-prepared-donor), [dissipative capture](#prove-dissipative-recovery-above-the-initial-barrier) | Native validated formation and sine preparation bounds use different laws/supports. Donor recovery proves a terminal identity for its source classes; neither a passing budget nor a prepared protected state proves receiver formation. |
| Assess the smooth sine state and its future | [Comparison](#inspect-the-separate-smooth-pressure-comparison), [relative patterns](#describe-a-whole-pattern-relative-to-a-moving-node), [inference](#infer-a-hidden-state-from-earlier-visible-observations), [forecast](#issue-a-prior-only-sine-forecast) | Explicit comparison law; observations, uncertainty sets and propagated boxes retain different scopes. |
| Assess sine response, pulse or recurrent identity | [Response](#assess-sine-resonance), [pair pulse](#check-whether-a-declared-pair-supports-a-permanent-pulse), [identity](#protect-cycle-identity-during-conservative-motion) | Input/output resonance, nonlinear periodicity and family recurrence are distinct. |
| Retain internal constituents in a collective state | [Replica observation](#observe-a-larger-organization-without-removing-its-constituents), [prepared pulse and perturbations](#assess-an-internal-pulse-and-its-perturbations) | Captured fine state is distinct from a symbolic preparation family and its asymptotic stability result. |
| Compare an internal action with a collective action | [Pair Emission](#compare-emission-on-one-member-and-on-a-whole-pair) | Actual AL form proposals on the sine quotient; no runtime event or occurrence law is executed. |
| Distinguish internal transfer from supplied form injection | [Regional transfer](#distinguish-autonomous-transfer-from-emission) | One captured source supplies reciprocal boundary currents and the full-form endpoint obstruction. |
| Identify interacting pairs when their phases coincide | [Joint form-phase identity](#retain-pair-identity-through-autonomous-exchange) | Storage-induced observation and a finite full-state error bound; exact reference identity is separate from perturbed persistence. |
| Identify constituent pairs from phase | [Phase partners](#identify-pairs-before-checking-their-connections) | Phase-only mutual matching precedes independent support/law admission; only a separate protected-tube certificate gives future persistence. |
| Determine the local direction of a pairing change | [Pairing transition](#predict-a-local-change-in-observed-pairs) | Complete-law chord rates resolve transverse exact ties; an existential interval has no certified numerical horizon. |
| Distinguish mobility choices without fitting a clock | [Constitutive response](#compare-two-declared-mobility-laws) | The same prepared source and two declared chord margins; both changed exchange rows retained, no trajectory or finite-window transfer. |
| Check geometry under changed reciprocal mobility | [Relative protection](#check-relative-protection-under-alternative-mobility) | Reuses the acute storage barrier; relative trapping, moving means and recurrence have separate verdicts. |
| Bound pairing over a finite window and uncertain preparation | [Whole-window certificate](#certify-pairs-throughout-a-declared-time-window) | Full form/phase error boxes and analytic complete-field remainders; every nearest competitor is checked throughout the interval. |
| Compare what phase and joint observations identify | [Same-source comparison](#compare-phase-grouping-with-joint-grouping) | Reuse the admitted window and both form-reversed preparations; observed partners need independent support admission. |
| Check whether member labels can be discarded | [Fixed-support symmetry](#check-which-member-swaps-preserve-the-law) | All-state graph symmetry, strict replica admission and a chosen source swap have distinct verdicts. |
| Retain sufficient state when attachments differ | [Mixed pair state](#retain-state-associated-with-asymmetric-attachments) | Signed internal state at asymmetric pairs and unordered invariants at symmetric pairs; exact-information description, not a macro-node runtime. |
| Classify equilibria without selecting a target winding | [Acute equilibrium families](#classify-the-allowed-acute-equilibria), [complete eleven-node endpoints](#classify-long-time-endpoints-without-selecting-a-basin) | Each supplied support has its own classification. Exact geometry, source-law convergence and basin selection remain distinct. |

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

### Compare two independently bounded rates

For the [matched capacity test](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#normalized-capacity-discriminator),
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

The [rate-contrast contract](../contracts/RELATIONAL_DYNAMICS.md#relational-rate-contrast)
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
[sample-rate contract](../contracts/RELATIONAL_DYNAMICS.md#relational-rate-samples)
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

Its [contract](../contracts/RELATIONAL_DYNAMICS.md#relational-capacity-sampling)
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
[execution contract](../contracts/RELATIONAL_DYNAMICS.md#conditional-relational-execution)
owns state/support admission, segment margins, atomicity and numerical limits;
the [derivation](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md)
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
[tangent contract](../contracts/RELATIONAL_DYNAMICS.md#phase-consensus-tangent-observation)
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

### Check relaxation from phase consensus

For the fixed native law below, copied reflected form with total initial
storage at most nine cannot produce a maintained unit winding from equal
phases. Apply the analytic theorem without integrating a trajectory:

```python
from fractions import Fraction
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel, relational_report_to_dict

consensus_graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
consensus_graph.add_edges_from(((0, 5), (1, 6)))
form = (1, -1, Fraction(-3, 2), 0, Fraction(3, 2))
for node in consensus_graph:
    consensus_graph.nodes[node].update(EPI=form[node % 5], theta=0, nu_f=1)
consensus = Network(consensus_graph).relational_consensus_capture(
    RelationalExchangeModel(1, epi_weight=1, phase_weight=1),
    cycles=(tuple(range(5)), tuple(range(5, 10))),
)
assert consensus.admitted and consensus.target_sector == 0
assert consensus.initial_form_storage == 9
assert not consensus.initial.energy_admitted  # the original E<7 test is unchanged
assert consensus.endpoint_storage_upper_bound == 6  # analytic bound at time one
consensus_evidence = relational_report_to_dict(consensus)
```

The nested initial report remains the actual supplied state. There is no
computed endpoint: loss and phase bounds prove entry into the consensus basin,
then its theorem supplies convergence. Outside these sufficient premises the
report is unavailable, not a prediction of pattern formation. This result
does not cover arbitrary form preparations, other graphs or coefficient laws.
The [contract](../contracts/RELATIONAL_DYNAMICS.md#phase-consensus-capture)
and [proof](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-consensus-preparation-obstruction)
specify that boundary.

### Exclude target formation for arbitrary consensus form

The broader analytic observer admits non-reflected form on the same support:

```python
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel

graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
graph.add_edges_from(((0, 5), (1, 6)))
for node in graph:
    graph.nodes[node].update(EPI=node / 16, theta=0, nu_f=1)
report = Network(graph).relational_consensus_formation_obstruction(
    RelationalExchangeModel(1),
    cycles=(tuple(range(5)), tuple(range(5, 10))),
)
assert report.admitted
assert report.excluded_target_sectors == (-1, 1)
assert report.continuation_status == "not_certified"
```

This excludes the aligned acute unit-winding targets under the fixed law and
budget throughout regular continuous existence. It does not predict a
consensus endpoint or certify indefinite regularity. Read unavailable reasons
when admission fails. The [contract](../contracts/RELATIONAL_DYNAMICS.md#full-form-consensus-obstruction)
keeps this result separate from the stronger reflected consensus theorem.

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

To inspect the sector without choosing a dynamics law, use the shared geometry
reader instead:

```python
geometry = Network(graph).relational_sector_geometry(
    storage_scale=model.storage_scale,
    cycles=(tuple(range(5)), tuple(range(5, 10))),
    target_sector=1,
)
geometry_evidence = relational_report_to_dict(geometry)
```

This consumes support, form, phase and the supplied storage scale, without
evaluating reference rates or requiring capacities. Admission describes the
sector and energy sublevel. Its `sublevel_*` bounds become future bounds only
under an independently admitted evolution law; they do not predict capture,
formation or convergence on their own.

### Inspect hypothetical pattern detachment

Supply a state of the same two-ring network and inspect what removing both
bridges would do. This example is a prepared near-twist snapshot, not a new
formation run or a certified cut time on a previous trajectory:

```python
import math
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel, relational_report_to_dict

graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
graph.add_edges_from(((0, 5), (1, 6)))
for node in graph:
    graph.nodes[node].update(
        EPI=1 / 128 if node % 5 == 0 else 0.0,
        theta=2 * math.pi * (node % 5) / 5,
        nu_f=1.0,
    )
cut = Network(graph).relational_detachment(
    RelationalExchangeModel(1),
    cycles=(tuple(range(5)), tuple(range(5, 10))),
)
assert cut.capture_admitted
assert graph.number_of_edges() == 12
assert cut.reset.storage_change == 0
assert any(cut.phase_rate_change)
cut_evidence = relational_report_to_dict(cut)
```

The certificates in `cut.components` use separately evaluated post-cut fields.
Each includes rigorous storage/gap bounds and an explicit target or unavailable
reasons. For an already isolated C5, call
`Network(component_graph).relational_cycle_capture(model, cycle=ordered_cycle)`.
The native executor still requires a connected graph.

`cut.reset` owns the event budget. Native field storage changes and their
reconciliation residual are reported separately from that budget and from
mathematical capture bounds. Equal bridge endpoints can give zero event cost
while removal changes rates. The
[contract](../contracts/RELATIONAL_DYNAMICS.md#relational-detachment-observation)
and [proof](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#relational-pattern-detachment)
explain why a supplied late cut can preserve identity while an early cut need
not. This observer changes no live edges and chooses no event time.

### Check a uniform receiver's formation budget

A formed source does not guarantee that an initially uniform receiver can
acquire the same pattern. The fixed equal-form source/receiver family has a
storage obstruction under the initial positive-resultant domain:

```python
from tnfr.physics.relational_capture import certify_relational_seeded_formation_obstruction
from tnfr.sdk import relational_report_to_dict

obstruction = certify_relational_seeded_formation_obstruction()
assert obstruction.status == "obstructed"
assert obstruction.obstruction_certified
assert tuple(case.bridge_count for case in obstruction.cases) == (1, 2)
assert all(case.additional_storage_gap_lower_bound > 0 for case in obstruction.cases)
formation_evidence = relational_report_to_dict(obstruction)
```

The supplied connection work is already included. With no further input,
neither interface can reach two maintained acute winding-one patterns from
this preparation. This is a proof-based exclusion, not a failed solver run.
The [contract](../contracts/RELATIONAL_DYNAMICS.md#relational-seeded-formation-obstruction)
specifies the initial and target domains and the wider regular states it does
not exclude. This function neither executes a contact nor changes the phase
admission used by the engine.

For the separate regular two-port candidate, the research API
`certify_relational_reflected_transit` encloses an exact eight-coordinate
source/receiver state, with independent forms and phases for the two rings.
It returns whole-time domain, winding and storage/loss evidence under its
declared fixed-support law. It does not project arbitrary `Network` states
onto reflection symmetry. The [contract](../contracts/RELATIONAL_DYNAMICS.md)
defines admission and report fields; the
[retained response](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regular-seeded-continuous-response)
documents the certified short window and its formation limits. Inspect that
evidence without replaying its frozen producer.

The same proof owner exposes `certify_relational_reflected_barrier` for a
detached exact interval state and model. This static test can exclude the
two-acute-twist target through a geometric transition barrier even while the
weaker target-storage budget remains positive. It does not advance the state
or infer a reflected state from a live graph. See the
[collective bound](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#reflected-collective-energy-barrier)
for the admissible lift and the distinction between transition cost and final
target cost.

### Classify an ideal reflected equilibrium

The same fixed two-ring support admits seven exact regular equilibria in the
inherited reflection lift. Select an analytic family explicitly:

```python
from tnfr.physics.relational_reflected_equilibria import certify_relational_reflected_equilibrium
from tnfr.sdk import RelationalExchangeModel

equilibrium = certify_relational_reflected_equilibrium(
    "opposite_twist",
    model=RelationalExchangeModel(1, phase_domain="regular"),
)
assert equilibrium.winding == (1, -1)
assert equilibrium.quotient_modes == (18, 0, 0)
equilibrium_evidence = equilibrium.to_dict()
```

Its intervals enclose one named ideal equilibrium, not a box of equilibria or
a projected live state. The stability certificate includes all ten nodes'
form and phase perturbations, modulo two common offsets. Local recovery does
not establish formation or global continuation. The
[contract](../contracts/RELATIONAL_DYNAMICS.md#ideal-reflected-equilibria-and-full-network-stability)
defines the families, exact evidence and report fields; the
[classification](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#reflected-regular-equilibria)
owns the proof and its scope.

### Separate protected evolution from a boundary counterexample

The optional domain observer captures the actual graph once:

```python
import networkx as nx
from tnfr.physics.relational_regularity import observe_relational_regularity
from tnfr.physics.relational_reflected_boundary import certify_relational_reflected_boundary_exit
from tnfr.sdk import RelationalExchangeModel

regular_model = RelationalExchangeModel(1, phase_domain="regular")
graph = nx.path_graph(2)
for node in graph:
    graph.nodes[node].update(EPI=0, nu_f=1, theta=2 * node)
domain = observe_relational_regularity(graph, model=regular_model)
assert domain.regularity_certified
domain_evidence = domain.to_dict()

boundary = certify_relational_reflected_boundary_exit(model=regular_model)
assert boundary.below_seven_beta
assert boundary.limiting_resultant_rates[2][0].hi < 0
boundary_evidence = boundary.to_dict()
```

The graph observer's storage test can certify global regular continuation of
the ideal held law. An unresolved result leaves that question open. The named
boundary certificate proves a different, existential result: some nearby
acute states reach a zero resultant. It does not say that the observed graph
will do so. The [contract](../contracts/RELATIONAL_DYNAMICS.md#global-regularity-evidence-and-boundary-access-limits)
distinguishes these statements from captured-rate kinematics and numerical
execution. Neither helper crosses the undefined boundary.

## Native pattern memory and continuous proofs

These examples retain the native Arg law and their declared preparations.
They provide proof reports or inspect retained evidence; they do not execute
the operator runtime or select the smooth sine comparison.

### Bound an ideal pattern-memory family

The cycle-memory helper concerns small centered perturbations about an exact
mathematical C5 twist. It supplies no evolved samples:

```python
from fractions import Fraction
from tnfr.physics.relational_cycle_memory import bound_relational_cycle_memory
from tnfr.sdk import RelationalExchangeModel, relational_report_to_dict

memory = bound_relational_cycle_memory(
    model=RelationalExchangeModel(1),
    form_direction=(1, -1, 0, 0, 0),
    phase_direction=(0, 1, -1, 0, 0),
    capacity=1,
    amplitude_radius=Fraction(1, 64),
)
assert memory.basin_admitted
assert memory.quadratic_phase_shift_coefficient_bounds[0] > 0
assert memory.quadratic_left_port_form_rate_coefficient_bounds[1] < 0
assert memory.remainder_bound is None
memory_evidence = relational_report_to_dict(memory)
```

Here `x=epsilon*form_direction` and the phase is the ideal winding-one twist
plus `epsilon*phase_direction`. The radius certifies recovery of this family;
it does not bound the error in the asymptotic final-phase formula. A positive
quadratic coefficient therefore does not certify the sign at epsilon `1/64`.

The readout compares the recovered pattern with an untouched aligned reference
through a hypothetical port connection. Use the existing `relational_pattern`
mean/covariance report for actual stored states and `relational_attachment`
for proposed contact. Their native floating fields remain separate from this
ideal-family calculation. See the
[contract](../contracts/RELATIONAL_DYNAMICS.md#relational-cycle-memory) for
reference, work, remainder and scope obligations.

### Certify one finite memory preparation

For the fixed unit-capacity native C5 comparison, the separate finite
certificate includes the integrated nonlinear error and both signed readouts:

```python
from fractions import Fraction
from tnfr.physics.relational_cycle_memory import certify_relational_cycle_memory
from tnfr.sdk import relational_report_to_dict

certificate = certify_relational_cycle_memory(amplitude=Fraction(1, 2**20))
assert certificate.admitted
assert certificate.phase_remainder_bound == Fraction(1, 2**45)
positive, negative = certificate.cases
assert positive.limiting_phase_shift_bounds[0] > 0
assert negative.limiting_phase_shift_bounds[1] < 0
assert positive.left_contact_form_rate_bounds[1] < 0
assert negative.left_contact_form_rate_bounds[0] > 0
finite_memory_evidence = relational_report_to_dict(certificate)
```

The two supplied form directions have opposite signs and the same initial
phase perturbation. This proves a finite-amplitude distinction after ideal
recovery, without evolving a graph. It does not say when recovery is complete
or perform a connection. Contact work and the retained reference remain
explicit premises. Valid amplitudes with unresolved signs return unavailable;
inspect the separate phase/readout flags and reasons rather than treating
them as zero. The [contract](../contracts/RELATIONAL_DYNAMICS.md#relational-finite-memory)
states the fixed law, admitted range and precision limits.

### Bound memory readout at a finite time

The fixed preparation also admits a sufficient finite waiting time with
explicit remaining form/phase error:

```python
from fractions import Fraction
from tnfr.physics.relational_cycle_memory import certify_relational_cycle_memory_readout
from tnfr.sdk import relational_report_to_dict

readout = certify_relational_cycle_memory_readout(decay_blocks=40)
assert readout.admitted
assert readout.horizon == 165120
assert readout.state_norm_upper_bound == Fraction(1, 2**57)
assert readout.mean_phase_tail_upper_bound == Fraction(1, 2**102)
positive, negative = readout.cases
assert positive.left_contact_form_rate_change_bounds[1] < 0
assert negative.left_contact_form_rate_change_bounds[0] > 0
assert all(case.contact_storage_bounds[0] > 0 for case in readout.cases)
readout_evidence = relational_report_to_dict(readout)
```

The time is in the model's structural units. Compare the separate
`*_no_contact_form_rate_bounds`, `*_contact_form_rate_bounds` and
`*_contact_form_rate_change_bounds` to distinguish the proposed connection's
effect from ongoing relaxation. A shorter declared time may return
unavailable with retained bounds. This report performs no event or subsequent
measurement; its positive contact work needs an independent event budget.
The [contract](../contracts/RELATIONAL_DYNAMICS.md#relational-memory-readout)
states the proof, exact input domain and observation limits.

### Certify accumulated response during contact

To bound a nonzero-duration response under one supplied held connection:

```python
from fractions import Fraction
from tnfr.physics.relational_memory_contact import certify_relational_memory_contact
from tnfr.sdk import relational_report_to_dict

contact = certify_relational_memory_contact(duration=Fraction(1, 4096))
assert contact.admitted and contact.windings_preserved
assert contact.end_time == Fraction(165120) + Fraction(1, 4096)
positive, negative = contact.cases
assert positive.right_contact_induced_form_change_bounds[0] > 0
assert negative.right_contact_induced_form_change_bounds[1] < 0
assert all(case.accumulated_form_remainder_bound > 0 for case in contact.cases)
assert contact.required_event_work_upper_bound > 0
contact_evidence = relational_report_to_dict(contact)
```

The response is a form change over the interval, with a continuous error
bound. Its no-contact control evolves independently; the previously recovering
left ring is not held artificially fixed. The report keeps changes of state,
cycle winding, required insertion work and continuous loss separate. It
executes no event or trajectory; lasting retention uses the separate certificate
below. See the [contract](../contracts/RELATIONAL_DYNAMICS.md#relational-memory-contact)
for duration admission and unavailable results.

### Certify a retained receiver record

To add a supplied bridge removal and certify the receiver's persistent mean:

```python
from tnfr.physics.relational_memory_contact import certify_relational_memory_retention
from tnfr.sdk import relational_report_to_dict

retention = certify_relational_memory_retention()
assert retention.admitted and retention.both_rings_captured
positive, negative = retention.cases
assert positive.receiver_persistent_mean_bounds[0] > 0
assert negative.receiver_persistent_mean_bounds[1] < 0
assert all(case.no_contact_receiver_mean_bounds == (0, 0) for case in retention.cases)
assert all(case.removal_storage_change_bounds[1] < 0 for case in retention.cases)
retention_evidence = relational_report_to_dict(retention)
```

This fixed certificate reuses the default contact duration and preparation.
It bounds the receiver's five-node mean, admits both isolated rings to recovery,
and applies their conserved means after the state-preserving cut. The record
survives internal recovery under the stated law; it does not require storing
a new memory variable. The bridge removal has its own storage jump, separate
from insertion work and continuous loss. No graph or trajectory is changed.
See the [contract](../contracts/RELATIONAL_DYNAMICS.md#relational-memory-retention)
for the distinction between a regional record, a port response and a physical
identification.

### Validate continuous transit to a protected basin

The transit method attempts to enclose the ideal ODE from a supplied initial
state to a sufficient basin, without advancing the stored network. Continue
with the [two-ring graph and positive-resultant model](#check-a-protected-relational-basin)
from the native capture example:

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

### Audit robustness of the retained formation proof

From a repository checkout, reuse the original proof files without rerunning
their trajectory:

```python
from pathlib import Path
from tnfr.research.relational_formation_robustness import (
    audit_relational_formation_robustness,
)

audit = audit_relational_formation_robustness(
    Path("docs/assets/relational_capture_response/continuous-transit.audit.json")
)
print(audit.status, audit.unavailable_reasons)
exact_evidence = audit.to_dict()
```

The sibling protocol and source archive are required. Admission verifies a
conditional comparison bound for each named phase-law change separately,
while holding the original preparation and structural clock fixed. It creates
no alternate runtime law. The
[contract](../contracts/RELATIONAL_DYNAMICS.md#retained-formation-robustness)
distinguishes the preserved reflected basin, conservative parameter bounds,
missing evidence and inconsistent records.

## Sine-law assessments and forecasts

The following module-level APIs concern `normalized_sine_reciprocal_exchange`.
A regular `RelationalExchangeModel` supplies coefficients and storage scale;
it does not switch `Network.step_relational` to this law. Each example states
whether it consumes a graph, observation bounds or an exact symbolic family.
When passing one report to another reader, inspect the new report's status,
reasons and actual observation time. The
[chained-report contract](../contracts/RELATIONAL_DYNAMICS.md#sine-chained-report-admission)
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
[contract](../contracts/RELATIONAL_DYNAMICS.md#detached-normalized-sine-complete-law-comparison)
and [selection/obstruction proof](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#global-closure-pressure-comparison)
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
[contract](../contracts/RELATIONAL_DYNAMICS.md#retained-state-sine-environmental-pressure).

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
The [sampling contracts](../contracts/RELATIONAL_DYNAMICS.md#joint-value-rate-and-acceleration-from-finite-samples)
describe `bound_relational_jet_from_samples` for actual data, exact SDK
export and the separate noise, timing and differentiation budgets.
Its initial-value interval must be retained. The point-state inverse below
requires an exact visible state, and a derivative-box witness does not
certify that a trajectory reproduces all earlier samples.

### Describe a whole pattern relative to a moving node

For a complete supplied sine-law state, remove unknown common origins while
retaining independent errors in its geometry:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.path_graph(3)
for node, form, phase in ((0, 0, 0), (1, 1, 0.5), (2, 2, 0)):
    graph.nodes[node].update(EPI=form, theta=phase, nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
error = Fraction(1, 1024)
pattern = bound_relational_sine_pattern(
    graph,
    reference_node=0,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(error, error, error),
    phase_error_bounds=(error, error, error),
)
assert pattern.relative_form_bounds[0].width == 0
assert pattern.relative_form_rate_bounds[0].contains(0)
assert pattern.reference_form_rate_bounds.lo > 0
evidence = pattern.to_dict()
```

The zero reference-relative coordinate is an identity, although the actual
reference is moving. The supplied radii concern synchronous residual errors
after a shared offset, in consistently lifted phases; this example is not
measured data. Every intermediary and capacity remains present.

For a prospective full-state enclosure, call `pattern.forecast` with explicit
`observation_time`, `end_time`, `time_step` and `order`.
Inspect `admitted` and `full_forecast.validated_end_time`. Use the returned
forecast bounds for the propagated family; the source's static storage bound
does not cover every point of the solver's larger outer box.
See the [contract](../contracts/RELATIONAL_DYNAMICS.md#relative-sine-patterns-and-moving-references).
This does not infer a missing node or certify recovery. In particular, the
absolute-rate inverse below cannot consume moving-reference rates unchanged.

### Certify recovery of an uncertain cycle pattern

This example supplies a deformed five-node cycle and nonzero residual errors.
The target remains the exact mathematical twist; the nominal phases need not
be an equilibrium:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.cycle_graph(5)
forms = (1, -1, 0, 0, 0)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(forms[node], 1024), theta=Fraction(5*node, 4), nu_f=1,
    )
graph.graph["GAMMA"] = {"type": "none"}
error = Fraction(1, 4096)
pattern = bound_relational_sine_pattern(
    graph,
    reference_node=0,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(error,)*5,
    phase_error_bounds=(error,)*5,
)
recovery = pattern.certify_cycle_recovery(
    cycle=tuple(range(5)), winding=1, radius=Fraction(1, 16),
)
assert recovery.admitted
assert recovery.norm_margin > 0 and recovery.energy_margin > 0
recovery_evidence = recovery.to_dict()
```

The proof covers every state consistent with those residual bounds, including
arbitrary common origins. It uses no trajectory and claims no laboratory
measurement. An explicit analytic check of this same dyadic observation class
is given by the [proof owner](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-cycle-recovery).

The same method on a `SineRelativeForecast` assesses its full endpoint box
at the actual validated time, preserving the solver's original status.
Inspect `hypothesis_failures` and `unresolved_conditions` if admission is
unavailable. This sufficient test does not diagnose instability.
The [contract](../contracts/RELATIONAL_DYNAMICS.md#whole-set-sine-cycle-recovery)
requires the **complete** support to be the cycle: do not remove environmental
connections to make a subregion pass.

<a id="assess-sine-resonance"></a>
### Assess sine resonance with an explicit readout

Declare a complete cycle template, its critical twist and one unit-normalized
nonzero Fourier input. The input is additive **form rate**, not a modulation
of pressure or capacity. Form, phase and structural-work gains are different:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_resonance import assess_sine_cycle_resonance

mode = assess_sine_cycle_resonance(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    node_count=5, mode_index=1, winding=1, capacity=1,
)
assert mode.pole_status == "distinct_real_poles"
assert mode.phase_peak_status == "dc_maximum"
assert mode.form_peak_angular_frequency_bounds.lo > 0
assert mode.work_peak_gain == 4
response = mode.gain(Fraction(1, 10))
assert response.status == "available"
assert response.form_gain_bounds.lo > 0
assert response.work_gain_bounds.hi <= mode.work_peak_gain
resonance_evidence = mode.to_dict()
```

Here free modes decay without oscillating, but form and work gains still
peak at a nonzero frequency. The work readout is the mode's Laplacian
eigenvalue times its form amplitude; its gain normalization matters.
`response` encloses a hypothetical harmonic tangent response. It neither
advances a graph nor applies a sinusoidal input. Frequencies refer to the
declared model clock, with no identification as laboratory hertz.

For a supplied graph, use `certify_sine_recovery_resonance(recovery, port=(i,j))`
with an existing full-support sine recovery certificate. It retains the target
and environment and proves a collocated work-port maximum with a gain ceiling;
it does not compute the maximizing frequency or a remote receiver response.
The [contract](../contracts/RELATIONAL_DYNAMICS.md#sine-resonance) specifies
the capacity-weighted probe. Save either report with
`export_to_json(report.to_dict(), path)` through the SDK atomic writer.

### Certify two patterns with a live intermediary

Keep the two cycles, both contact edges and the intermediary in the same
observation. The target is an exact critical geometry on that complete graph:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.Graph()
graph.add_nodes_from(range(11))
for start in (0, 5):
    graph.add_edges_from(
        (start+j, start+(j+1)%5) for j in range(5)
    )
graph.add_edges_from(((0, 10), (5, 10)))
target = tuple(Fraction(j, 5) for j in range(5))*2 + (Fraction(0),)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(1, 4096) if node == 0 else 0,
        theta=Fraction(1287*(node%5), 1024) if node < 10 else 0,
        nu_f=1,
    )
graph.graph["GAMMA"] = {"type": "none"}
error = Fraction(1, 65536)
pattern = bound_relational_sine_pattern(
    graph,
    reference_node=10,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(error,)*11,
    phase_error_bounds=(error,)*11,
)
recovery = pattern.certify_pattern_recovery(
    target_phase_turns=target, radius=Fraction(1, 16),
)
assert recovery.admitted
assert recovery.norm_margin > 0 and recovery.energy_margin > 0
recovery_evidence = recovery.to_dict()
```

This checks the complete uncertain set without evolving it. Referencing node
10 removes a shared coordinate origin; it does not freeze that node. All
positive capacities remain held model parameters. The
[proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-interacting-recovery)
separately derives donor-to-receiver influence and its frozen-mediator
control. Recovery admission alone is not a measurement of that response.
The [contract](../contracts/RELATIONAL_DYNAMICS.md#whole-set-sine-recovery-with-a-retained-environment)
retains the distinction between a static certificate and a validated forecast.

The same target can certify a declared work-port resonance without deleting
either ring or replacing the intermediary by a static boundary:

```python
from tnfr.physics.relational_sine_resonance import certify_sine_recovery_resonance

port_response = certify_sine_recovery_resonance(recovery, port=(0, 5))
assert port_response.positive_frequency_peak_certified
assert port_response.work_gain_upper_bound == Fraction(4, 3)
assert port_response.peak_angular_frequency_bounds is None
port_evidence = port_response.to_dict()
```

The two selected nodes define the probe `f=K*(e_0-e_5)` and matching work
output; they need not share an edge. This proves that a finite-frequency
maximum exists for that pair of input/output definitions. It does not prove
a maximum in a separately observed receiver coordinate. See
[response normalization](#assess-sine-resonance) before comparing gains.

For a donor-only perturbation and a receiver-only observation, retain the same
complete recovery but use the separate mediated-response assessment:

```python
from tnfr.physics.relational_sine_resonance import assess_sine_mediated_response
from tnfr.sdk import relational_report_to_dict

remote = assess_sine_mediated_response(
    recovery,
    donor_cycle=tuple(range(5)),
    receiver_cycle=tuple(range(5, 10)),
    mediator=10,
)
assert remote.phase_dc_gain == 0
assert remote.phase_order_two_pi_numerator == -Fraction(1, 72)
assert remote.phase_markov_bounds[2].hi < 0
assert remote.positive_frequency_phase_peak_certified
assert remote.phase_impulse_sign_reversal_certified
assert remote.sign_reversal_time_bounds is None
remote_evidence = relational_report_to_dict(remote)
```

The source form direction is the donor port minus the mean of its other four
nodes, multiplied by each node's mobility. The phase readout is the receiver
port relative to its other four nodes. Its static response is zero but its
transient response is nonzero. A donor-only initial disturbance therefore has
a tangent phase response that must reverse sign; a numerical reversal time
or finite nonlinear amplitude is not supplied by this certificate.
The [contract](../contracts/RELATIONAL_DYNAMICS.md#sine-mediated-response)
also retains the intermediary's memory blocks and exact symmetry controls.
This report executes neither a pulse nor a harmonic probe.

### Check whether a declared pair supports a permanent pulse

The existing sine law has exact nonlinear periodic states on its reversible
boundary. Declare that constitutive premise explicitly; the assessment does
not infer it from an observed oscillation or change the model to obtain one.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_resonance import assess_sine_pair_pulse
from tnfr.sdk import relational_report_to_dict

pair = nx.path_graph(2)
for node, form in enumerate((Fraction(1, 4), Fraction(-1, 4))):
    pair.nodes[node].update(EPI=form, nu_f=1, theta=0)
reversible = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular",
)
pulse = assess_sine_pair_pulse(pair, reference_model=reversible)
assert pulse.nonlinear_periodic_exchange_certified
assert pulse.exact_energy == Fraction(1, 8)
assert 9 < pulse.period_bounds.lo <= pulse.period_bounds.hi < 11
pulse_evidence = relational_report_to_dict(pulse)

dissipative = RelationalExchangeModel(1, phase_domain="regular")
control = assess_sine_pair_pulse(pair, reference_model=dissipative)
assert control.nonstationary_recurrence_excluded
assert not control.nonlinear_periodic_exchange_certified
```

These are statements about complete continuous equations, not sampled
trajectories. The first preparation exchanges form and phase indefinitely
without a supplied periodic input. The second loses structural storage and
cannot sustain a recurrent full state. This distinction leaves open which
loss law is justified and where the initial activity comes from. The
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-pair-pulse) describes
zero-capacity, equilibrium and unresolved-period cases; the
[proof](../../theory/nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
derives the nonlinear period and the dissipative obstruction.

### Distinguish regional transfer from total loss

A decreasing local record need not imply irreversible loss. Keep the whole
three-node path and check both incident-edge balances before hiding its middle
node. The same assessment constructs exact memory for a separately declared
consensus tangent:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_resonance import assess_sine_path_memory
from tnfr.sdk import relational_report_to_dict

path = nx.path_graph(3)
for node, (form, phase) in enumerate(
    ((1, 0), (Fraction(1, 4), 0), (0, Fraction(1, 4)))
):
    path.nodes[node].update(EPI=form, theta=phase, nu_f=1)
model = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular",
)
memory = assess_sine_path_memory(
    path, reference_model=model, mediator=1,
    consensus_form=0, consensus_phase=0,
)
assert memory.comparison.continuous_loss == 0
assert memory.nonlinear_edge_rate_bounds[0].hi < 0
assert memory.nonlinear_edge_rate_bounds[1].lo > 0
assert all(bound.contains(0) for bound in memory.nonlinear_edge_rate_residual_bounds)
assert memory.linear_observation.dimension == 6
assert memory.linear_observation.extra_coordinates == 2
assert any(memory.hidden_initial_forcing)
assert 19 < memory.full_tangent_period_bounds.lo < 20
memory_evidence = relational_report_to_dict(memory)
```

The nonlinear instantaneous rates show storage leaving the first edge and
entering the second. The separate linear report retains the middle node's
initial state and its memory feedback; discarding them would change the
endpoint response. Its return period belongs to the consensus tangent, not
to a certified nonlinear continuation of this preparation. The
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-conservative-path-memory)
keeps these scopes and the two clocks explicit. The
[derivation](../../theory/nodal/RESONANCE_FOUNDATIONS.md#finite-conservative-memory)
also gives an analytic tangent example where storage returns to its initial
region instead of being dissipated.

### Admit a nonlinear recurrent family without certifying one orbit

For the explicitly conservative sine law, the full nonlinear recurrence
theorem applies to almost every state in an admitted bounded family. Supply
the energy ceiling and interval of weighted form mean independently; they
describe the family being studied and do not change its evolution law.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_resonance import assess_sine_recurrence
from tnfr.sdk import relational_report_to_dict

cycle = nx.cycle_graph(5)
for node, form in enumerate((Fraction(1, 4), Fraction(-1, 4), 0, 0, 0)):
    cycle.nodes[node].update(EPI=form, theta=0, nu_f=node + 1)
model = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular",
)
recurrence = assess_sine_recurrence(
    cycle, reference_model=model,
    energy_ceiling=1, form_mean_bounds=(-1, 1),
)
assert recurrence.almost_everywhere_recurrence_certified
assert recurrence.snapshot_family_membership == "inside"
assert recurrence.snapshot_motion_status == "nonstationary"
assert recurrence.individual_recurrence_status == "unavailable_for_chosen_state"
assert recurrence.weighted_mean_rate_residual_bounds.contains(0)
assert recurrence.divergence == 0
recurrence_evidence = relational_report_to_dict(recurrence)
```

The two final statuses are compatible: the captured state moves, but the
family theorem does not prove that this particular state returns. The exact
P2 separatrix is a nonrecurrent member of another admitted family, so changing
that unavailable verdict to a passing flag would be incorrect. The report
also exposes an invariant form-coordinate box for the family. Its
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-nonlinear-recurrence)
and [proof](../../theory/nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
distinguish circular-state recurrence, stationary states, uncertain membership
and the absence of a return-time or fixed-period prediction.

The [joint-pattern application](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-joint-recurrent-episodes)
adds an independently proved acquisition neighborhood on doubled C5. Almost
every preparation in that neighborhood repeatedly acquires and loses the
specified matching. The example above does not test membership in that
neighborhood or predict episodes for its captured state.

### Protect cycle identity during conservative motion

The same barrier used for dissipative recovery can instead certify that a
conservative cycle stays near its prepared phase geometry. The family-level
recurrence theorem then applies without asserting a period or an individual
return. Here the support is the **whole** isolated five-node ring:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_recovery import assess_sine_cycle_identity
from tnfr.sdk import relational_report_to_dict

graph = nx.cycle_graph(5)
forms = (1, -1, 0, 0, 0)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(forms[node], 1024), theta=Fraction(5*node, 4), nu_f=1,
    )
model = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular",
)
error = Fraction(1, 4096)
pattern = bound_relational_sine_pattern(
    graph, reference_node=0, reference_model=model,
    form_error_bounds=(error,)*5, phase_error_bounds=(error,)*5,
)
identity = assess_sine_cycle_identity(
    pattern, cycle=tuple(range(5)), winding=1, radius=Fraction(1, 16),
    excess_ceiling=Fraction(1, 2560), form_mean_bounds=(-1, 1),
)
assert identity.family_admitted
assert identity.source_set_trapping_certified
assert identity.relative_source_family_membership == "certified_inside"
assert identity.family_almost_everywhere_recurrence_certified
assert identity.individual_recurrence_status == "unavailable_for_chosen_state"
identity_evidence = relational_report_to_dict(identity)
```

All states consistent with these relative error bounds retain winding one.
The declared mean interval is a separate restriction defining a finite-volume
family: the relative observation cannot measure an absolute common form
origin. Almost every member of that family returns near its starting full
state, but the displayed flags do not certify recurrence of this particular
preparation. They also do not imply that the ring returns to its exact target
or that it formed autonomously. See the
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-conservative-identity)
for independent family, trapping and membership verdicts.

### Observe a larger organization without removing its constituents

This supplied double-replica ring contains ten fine nodes. Compare synchronized
pairs with pairs having the same means but a different internal phase spread:

```python
from fractions import Fraction
from itertools import product
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_scale import assess_sine_replica_scale
from tnfr.sdk import relational_report_to_dict

graph = nx.Graph()
pairs = tuple(((i, 0), (i, 1)) for i in range(5))
for i, pair in enumerate(pairs):
    for node in pair:
        graph.add_node(node, EPI=0, theta=Fraction(5*i, 4), nu_f=1)
for i, j in nx.cycle_graph(5).edges:
    graph.add_edges_from(product(pairs[i], pairs[j]))
model = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular",
)
aligned = assess_sine_replica_scale(graph, reference_model=model, pairs=pairs)
assert aligned.same_law_reduced_flow_certified_for_source

graph.nodes[(0, 0)]["theta"] += Fraction(1, 64)
graph.nodes[(0, 1)]["theta"] -= Fraction(1, 64)
spread = assess_sine_replica_scale(graph, reference_model=model, pairs=pairs)
assert spread.form_means == aligned.form_means
assert spread.phase_mean_radian_parts == aligned.phase_mean_radian_parts
assert spread.resultant_magnitude_bounds[0].hi < 1
assert spread.coarse_form_defect[1].lo > 0
assert not spread.same_law_reduced_flow_certified_for_source
assert len(spread.comparison.nodes) == 10
assert all(bound.contains(0) for bound in spread.mean_form_factorization_residual)
assert spread.storage_decomposition_residual.contains(0)
scale_evidence = relational_report_to_dict(spread)

# Keep internal physics while discarding only interchangeable pair labels.
assert spread.unordered_pair_state_closure_certified
assert spread.unordered_pair_state_identifies_swap_orbits
assert all(bound.contains(0) for bound in spread.internal_constraint_residual)
assert spread.unordered_storage_residual.contains(0)

# Equal phases can hide a form imbalance that starts changing their correlation.
graph.nodes[(0, 0)].update(theta=0, EPI=Fraction(1, 16))
graph.nodes[(0, 1)].update(theta=0, EPI=Fraction(-1, 16))
latent = assess_sine_replica_scale(graph, reference_model=model, pairs=pairs)
assert latent.internal_form_squared[0] == Fraction(1, 256)
assert latent.form_phase_correlation_bounds[0].lo == 0
assert latent.form_phase_correlation_bounds[0].hi == 0
assert latent.closed_form_phase_correlation_rates[0].lo > 0
assert not latent.source_in_synchronized_submanifold
latent_evidence = relational_report_to_dict(latent)
```

The altered internal organization changes a neighboring group's response
although the block means agree. This follows from the fine nodal law, with
no fitted coupling or node deletion. The nominal dyadic phases illustrate
the observer; they are not the exact irrational twist used in the analytic
equilibrium proof. Initial pair-chart admission is not a future trapping
certificate. The [contract](../contracts/RELATIONAL_DYNAMICS.md#sine-replica-scale)
and [proof](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-inheritance)
separate synchronized inheritance, internal influence and means-only nonclosure.
The augmented `(X,Theta,R,U,Q)` state does close in its admitted chart and
identifies the two constituents up to exchange. Their identities as ordered
labels are unnecessary for these collective rates; their internal state is
essential. The report still retains the original source for inspection. Its
intervals are correlated observations, not independently adjustable values.

### Check protected geometry with continuing internal exchange

This detached assessment checks a finite represented preparation against the
conditional all-time theorem. The target remains exact; the source does not
need to be an exact equilibrium or a prepared periodic orbit.

```python
from fractions import Fraction
from itertools import product
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_scale import assess_sine_replica_persistence
from tnfr.sdk import relational_report_to_dict

graph = nx.Graph()
pairs = tuple(((i, 0), (i, 1)) for i in range(5))
for i, (left, right) in enumerate(pairs):
    phase = Fraction(1287*i, 1024)
    graph.add_node(left, EPI=Fraction(1, 1024), theta=phase, nu_f=1)
    graph.add_node(right, EPI=Fraction(-1, 1024), theta=phase, nu_f=1)
for i, j in nx.cycle_graph(5).edges:
    graph.add_edges_from(product(pairs[i], pairs[j]))

model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
persistence = assess_sine_replica_persistence(
    graph, reference_model=model, pairs=pairs,
    radius=Fraction(1, 16), excess_ceiling=Fraction(1, 32768),
    form_mean_bounds=(-1, 1),
)
assert persistence.family_admitted
assert persistence.source_family_membership == "certified_inside"
assert persistence.source_joint_persistence_certified
assert all(persistence.pair_all_time_circulation_certified)
assert persistence.internal_full_turn_time_bounds.lo > 0
assert persistence.individual_recurrence_status == "unavailable_for_chosen_state"
evidence = relational_report_to_dict(persistence)
```

All ten nodes remain present. The collective phase geometry stays protected,
and each pair continues internal exchange, though no common exact period is
certified. The mean slab and excess ceiling specify the family being studied,
not new interaction parameters. The [contract](../contracts/RELATIONAL_DYNAMICS.md#sine-replica-persistence)
separates family recurrence, captured-state trapping, synchronized tips and
angular traversal. This does not demonstrate formation of the supplied graph
or selection of its pair partition.

### Check relative protection under alternative mobility

Reuse `graph`, `pairs` and `model` from the preceding protected-geometry
example, with its unchanged radius and excess ceiling. The alternative
multiplies both exchange rows by a local current-dependent factor. The same
storage barrier still protects relative form/phase organization:

```python
from tnfr.physics.relational_sine_scale import assess_sine_mobility_geometry

geometry = assess_sine_mobility_geometry(
    graph, reference_model=model, pairs=pairs, epsilon=1,
    radius=Fraction(1, 16), excess_ceiling=Fraction(1, 32768),
)
assert geometry.family_admitted
assert geometry.source_set_trapping_certified
assert geometry.source_relative_family_membership == "certified_inside"
assert geometry.relative_family_recurrence_status == (
    "unavailable_invariant_measure_unproved"
)
assert geometry.individual_recurrence_status == "unavailable_for_chosen_state"

balance = geometry.balance
assert balance.relative_field_closure_certified
assert len(balance.relative_form) == len(graph) - 1
assert not balance.constant_mobility_mean_and_volume_identities_certified
relative_rates = balance.relative_form_rates, balance.relative_phase_rates
mean_drift = balance.weighted_form_mean_rate_bounds
divergence = balance.relative_divergence_bounds
evidence = relational_report_to_dict(geometry)
balance_evidence = relational_report_to_dict(balance)
```

The reference origin moves: its rate is subtracted from every retained
relative rate. The report does not impose a fixed mean or reset the clock.
This preparation can have special cancellations; a zero snapshot drift
does not restore a general conservation law. The
[form-balance selection theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#closed-form-balance-and-source-selection)
uses this distinction: its conservation requirement is an additional structural
premise, checked separately from storage protection. The
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-mobility-geometry)
separates the protected relative shape from the unresolved invariant measure
needed for an almost-everywhere return theorem. No internal circulation or
period certificate is transferred from the original sine law.

### Retain unequal constituent capacities

Use the separate capacity assessment when members have different held
capacities. This example keeps collective geometry inside the same protected
region while pair zero has an instantaneous internal stall:

```python
from fractions import Fraction
from itertools import product
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_scale import assess_sine_replica_capacity
from tnfr.sdk import relational_report_to_dict

graph = nx.Graph()
pairs = tuple(((i, 0), (i, 1)) for i in range(5))
h, epsilon = Fraction(1287, 1024), Fraction(1, 1024)
for (left, right), phase in zip(pairs, (0, h, 2*h, -2*h, -h)):
    graph.add_node(left, EPI=0, theta=phase, nu_f=1)
    graph.add_node(right, EPI=0, theta=phase, nu_f=1)
graph.nodes[pairs[0][0]].update(EPI=epsilon/2, nu_f=Fraction(3, 2))
graph.nodes[pairs[0][1]].update(EPI=3*epsilon/2, nu_f=Fraction(1, 2))
for i, j in nx.cycle_graph(5).edges:
    graph.add_edges_from(product(pairs[i], pairs[j]))

model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
capacity = assess_sine_replica_capacity(
    graph, reference_model=model, pairs=pairs,
    phase_turns=(0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
    radius=Fraction(1, 16), excess_ceiling=Fraction(1, 32768),
    form_mean_bounds=(-1, 1),
)
assert capacity.family.family_admitted
assert capacity.family.source_family_membership == "certified_inside"
assert capacity.family.source_set_trapping_certified
assert capacity.ordered_direct_rates[0][1].lo > 0  # Mean phase still moves.
assert capacity.all_time_internal_activity_status == "not_certified_by_this_reader"
assert capacity.family.individual_recurrence_status == "unavailable_for_chosen_state"
evidence = relational_report_to_dict(capacity)
```

The exact stall follows from opposite neighbor phases and the capacity/form
preparation, as derived in the
[proof](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-capacity-asymmetry).
It is not complete equilibrium or a claim that internal motion never resumes.
The report retains the capacity-state correlations needed to predict the
collective rates. The [contract](../contracts/RELATIONAL_DYNAMICS.md#sine-replica-capacity)
explains captured precision, joint relabeling and optional family evidence.
Omit all three family bounds for instantaneous identities on other admitted
complete paired supports; no collective trapping is then assessed.

### Identify pairs before checking their connections

The standalone observer needs no graph or proposed partition:

```python
from fractions import Fraction
from tnfr.physics.relational_sine_scale import observe_phase_pairs

partners = observe_phase_pairs(
    nodes=("a", "b", "c", "d"),
    phases=(0, Fraction(1, 32), 1, Fraction(33, 32)),
)
assert partners.status == "certified"
assert partners.candidate_pairs == (("a", "b"), ("c", "d"))
ambiguous = observe_phase_pairs(nodes=("a", "b", "c"), phases=(0, 0, 0))
assert ambiguous.candidate_pairs is None
```

For the full law, reuse the graph, model and declared lifts from
[the preceding unequal-capacity example](#retain-unequal-constituent-capacities).
This reader discovers the partners first. The supplied `pair_order` only
declares how those observed pairs are ordered for the target-family check;
it cannot replace the observed partition:

```python
from tnfr.physics.relational_sine_scale import assess_sine_state_pairing

pairing = assess_sine_state_pairing(
    graph, reference_model=model, pair_order=pairs,
    phase_turns=(0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
    radius=Fraction(1, 16), excess_ceiling=Fraction(1, 32768),
    form_mean_bounds=(-1, 1),
)
assert pairing.observation.candidate_pairs == pairs
assert pairing.capacity is not None
assert pairing.capacity.family.source_set_trapping_certified
assert pairing.all_time_pairing_persistence_certified
evidence = relational_report_to_dict(pairing)
```

Closeness alone does not establish the required connections. If only phase
attributes are exchanged between two original pairs while their edges stay
fixed, a new mutual phase pairing can fail the support contract. The report
retains that failure and never substitutes the old grouping. Without the
optional family assessment, matching and admitted collective rows remain
instantaneous observations. See the
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-state-pairing) for explicit
ambiguity, phase-chart and target-order requirements.

### Predict a local change in observed pairs

This standalone preparation has two exact nearest-phase ties. Its internal
form predicts their direction of change under the conservative sine law:

```python
import networkx as nx

from tnfr.physics.relational_sine_scale import assess_sine_pairing_transition
from tnfr.sdk import RelationalExchangeModel, relational_report_to_dict

pairs = tuple((2 * i, 2 * i + 1) for i in range(5))
graph = nx.Graph()
phases = (0, 0.125, 0.25, 0.375, 1, 1, 2, 2, 3, 3)
forms = (0.125, -0.125, 0.125, -0.125, 0, 0, 0, 0, 0, 0)
for node, (phase, form) in enumerate(zip(phases, forms)):
    graph.add_node(node, EPI=form, theta=phase, nu_f=1)
for i in range(5):
    graph.add_edges_from(
        (left, right) for left in pairs[i] for right in pairs[(i + 1) % 5]
    )
model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
transition = assess_sine_pairing_transition(graph, reference_model=model)
assert transition.observation.candidate_pairs is None
assert transition.forward.candidate_pairs == pairs
assert transition.forward.support_admission_status == "admitted"
assert transition.backward.status == "certified_nonmutual"
assert transition.forward.certified_time_horizon is None

for node in graph:
    graph.nodes[node]["EPI"] *= -1
control = assess_sine_pairing_transition(graph, reference_model=model)
assert control.observation == transition.observation
assert control.forward.status == "certified_nonmutual"
assert control.backward.candidate_pairs == pairs
evidence = relational_report_to_dict(transition)
```

The two sources have identical phase geometry but opposite phase velocities.
The reports certify local mathematical choices without integrating a
trajectory or stating how long they last. Pairing, its fixed-support admission
and permanent formation remain different claims. See the
[transition contract](../contracts/RELATIONAL_DYNAMICS.md#sine-pairing-transition).

### Compare two declared mobility laws

Reuse the [local transition preparation](#predict-a-local-change-in-observed-pairs)
and restore its original forms. Fix the two margins before evaluating the
alternative. Here the declared epsilon is one; the report also retains the
normalized-sine baseline, equivalent to epsilon zero:

```python
from fractions import Fraction
from tnfr.physics.relational_sine_scale import assess_sine_pairing_mobility

for node, form in enumerate(forms):
    graph.nodes[node]["EPI"] = form
contrast = assess_sine_pairing_mobility(
    graph, reference_model=model, epsilon=1,
    margin_triples=((1, 2, 0), (2, 1, 3)),
)
assert contrast.initial_exact_ties == (True, True)
assert contrast.source_margin_rate_difference_exact_zero
assert contrast.sine_normalized_contrast_bounds.lo == 0
assert contrast.sine_normalized_contrast_bounds.hi == 0
assert contrast.mobility_contrast_status == "available"
relative_rate = contrast.mobility_normalized_contrast_bounds
assert Fraction(1, 500) < relative_rate.lo <= relative_rate.hi < Fraction(1, 400)
assert all(rate.lo > 0 for rate in contrast.mobility_margin_rate_bounds)
assert contrast.certified_time_horizon is None
evidence = relational_report_to_dict(contrast)
```

Each triple names `(observer, alternative, preferred)` for the squared-chord
margin `D(observer,alternative)-D(observer,preferred)`. Both laws favor the
same two partner choices locally, but their relative margin rates differ.
The ratio cancels a common clock factor; its nonzero difference is not just
a different choice of time units. The unchanged source and both alternative
rate rows are available through `contrast.mobility`. This comparison changes
the declared law, not the live graph's runtime dispatch. Its
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-pairing-mobility) distinguishes
the general qualitative theorem, the frozen quantitative discriminator and
unavailable ratios. The original finite-window certificate below retains
its own sine law and is not evidence for the alternative.

### Certify pairs throughout a declared time window

Continue the [local transition preparation](#predict-a-local-change-in-observed-pairs)
above. Restore its original forms, since that example ends with the reversed
control. Every fine form and phase now has an independent initial error; the
capacities, support and law remain fixed:

```python
from fractions import Fraction

from tnfr.physics.relational_sine_scale import assess_sine_pairing_window

for node, form in enumerate(forms):
    graph.nodes[node]["EPI"] = form
budget = dict(
    form_error_bounds=(Fraction(1, 2**20),) * 10,
    phase_error_bounds=(Fraction(1, 2**20),) * 10,
    window_start=Fraction(1, 128),
    window_end=Fraction(1, 64),
)
window = assess_sine_pairing_window(graph, reference_model=model, **budget)
assert window.status == "certified_matching"
assert window.candidate_pairs == pairs
assert window.support_admission_status == "admitted"

for node in graph:
    graph.nodes[node]["EPI"] *= -1
reversed_window = assess_sine_pairing_window(
    graph, reference_model=model, **budget
)
assert reversed_window.status == "certified_nonmutual"
assert reversed_window.candidate_pairs is None
evidence = relational_report_to_dict(window)
```

The verdict concerns every initial state in each full error box and every
time in the closed window, in structural time units. It uses analytic bounds
from the full nonlinear law; it is neither a sampled center trajectory nor
an Euler prediction. The error box does not preserve exact pair synchrony.
A wide enclosure returns `unavailable`, not a failed physical prediction.
The [window contract](../contracts/RELATIONAL_DYNAMICS.md#sine-pairing-window)
keeps this finite guarantee separate from the local reader's unspecified
interval, permanent trapping and a laboratory observation bridge.

### Compare phase grouping with joint grouping

Continue the preceding window example. Its two reports retain their captured
sources even though the live graph now contains the reversed preparation.
Project both windows into the storage-induced form-phase distance without
recapturing or evolving that graph:

```python
joint = window.joint_observation()
reversed_joint = reversed_window.joint_observation()
joint_pairs = ((0, 2), (1, 3), (4, 5), (6, 7), (8, 9))
assert joint.phase_window is window
for result in (joint, reversed_joint):
    assert result.initial_observation.candidate_pairs == joint_pairs
    assert result.initial_box_candidate_pairs == joint_pairs
    assert result.candidate_pairs == joint_pairs
    assert result.same_pairing_as_initial_box is True
    assert result.support_admission_status == "rejected"
assert window.status == "certified_matching"
assert reversed_window.status == "certified_nonmutual"
evidence = relational_report_to_dict(joint)
```

Both full preparation boxes already identify these joint partners, and the
same partners are certified throughout the original future window. The
phase-only transition therefore does not establish their birth. These joint
pairs also differ from the supplied structural pairs: the first two contain
existing edges, so the strict replica support contract rejects them. This
does not erase the observed grouping or exclude every collective description.
See the [projection contract](../contracts/RELATIONAL_DYNAMICS.md#sine-joint-pairing-projection)
for the source-rate calculation, all-coordinate error bounds and availability.

The [critical-boundary theorem](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-joint-boundary-acquisition)
separately defines an exact trigonometric amplitude at which joint partners
are initially tied. Its complete-law derivatives prove local acquisition of
the structural pairing. It is not the represented preparation in this example:
rounding that amplitude does not preserve the exact tie, and the theorem
does not supply a numerical horizon for a graph execution.

### Check which member swaps preserve the law

Reuse the graph, `forms`, `pairs` and `model` from the
[local transition example](#predict-a-local-change-in-observed-pairs). Work on
a detached copy, restore the original forms, and remove the prescribed edge:

```python
from fractions import Fraction

from tnfr.physics.relational_sine_scale import (
    assess_sine_pair_support_symmetry,
    assess_sine_pairing_window,
)

broken = graph.copy()
for node, form in enumerate(forms):
    broken.nodes[node]["EPI"] = form
broken.remove_edge(0, 8)
symmetry = assess_sine_pair_support_symmetry(
    broken, reference_model=model, pairs=pairs, witness_pair=0
)
assert symmetry.pair_swap_symmetry == (False, True, True, True, False)
assert symmetry.unordered_pair_quotient_status == "obstructed_by_fixed_support"
witness = symmetry.witness
assert witness.source_orbit_equal
assert witness.source_pair_mean_phase_rate_numerators[4] == Fraction(1, 48)
assert witness.swapped_pair_mean_phase_rate_numerators[4] == -Fraction(1, 48)
assert witness.collective_phase_rate_obstruction_certified
observed = assess_sine_pairing_window(
    broken,
    reference_model=model,
    form_error_bounds=(Fraction(1, 2**20),) * 10,
    phase_error_bounds=(Fraction(1, 2**20),) * 10,
    window_start=Fraction(1, 128),
    window_end=Fraction(1, 64),
)
assert observed.candidate_pairs == pairs
assert observed.support_admission_status == "rejected"
evidence = relational_report_to_dict(symmetry)
```

Divide those numerators by mathematical pi for rates in the declared clock.
The same unordered pair state has two different receiver phase-mean rates
because the fixed connections distinguish its members. The reader evaluates
the hypothetical state swap without applying it. The unchanged finite-window
budget still certifies the observed pairs, despite the failed replica support.
The edge removal above is
a separate preparation, not an executed event or a support-work certificate.
Internal pair edges can preserve symmetry while requiring different collective
equations; this reader keeps strict replica admission separate. See the
[support-symmetry contract](../contracts/RELATIONAL_DYNAMICS.md#sine-pair-support-symmetry).

### Retain state associated with asymmetric attachments

Continue with `broken`, `model` and `pairs` from the preceding fixed-support
example. The observation below retains which member is attached where only
when that distinction affects the complete law:

```python
from tnfr.physics.relational_sine_scale import assess_sine_mixed_pair_state

mixed = assess_sine_mixed_pair_state(
    broken, reference_model=model, pairs=pairs,
)
assert tuple(item.mode for item in mixed.coordinates) == (
    "ordered", "unordered", "unordered", "unordered", "ordered",
)
assert mixed.coordinates[0].form_half_difference == Fraction(1, 8)
assert mixed.coordinates[1].form_half_difference is None
assert mixed.rates[4].phase_mean_numerator == Fraction(1, 48)
assert mixed.support_symmetry.strict_replica_admission_status == "rejected"
evidence = relational_report_to_dict(mixed)
```

Dividing the receiver numerator by pi gives its phase-mean rate. Exchanging
both source form and phase at nodes `0,1` on this fixed graph changes the
first pair's retained signed coordinates and reverses that receiver rate.
Swaps in the three symmetric pairs leave the mixed description unchanged.
No member or internal continuous coordinate is discarded. The phase lifts
must keep each half-difference strictly inside `(-pi/2, pi/2)`; chart rejection
does not reject the underlying fine dynamics. See the
[mixed-state contract](../contracts/RELATIONAL_DYNAMICS.md#sine-mixed-pair-state)
for the distinction between exact realized coordinates and their interval
projections, synchronized tips and source provenance.

### Compare Emission on one member and on a whole pair

This independent preparation uses two interchangeable pairs. Each candidate
reuses the actual AL form proposal with an explicit boost:

```python
from fractions import Fraction
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel, relational_report_to_dict

graph = nx.complete_bipartite_graph(2, 2)
for node, form in zip(graph, (0.25, 0.75, 0.0, 0.0)):
    graph.nodes[node].update(EPI=form, theta=0.0, nu_f=1.0)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
action = Network(graph).relational_sine_pair_emission(
    model, pairs=((0, 1), (2, 3)), pair_index=0, boost=0.125,
)
assert not action.single_member_source_orbit_equal
assert action.first_member.internal_form_squared == Fraction(9, 256)
assert action.second_member.internal_form_squared == Fraction(25, 256)
assert action.whole_pair_structural_descent_certified
assert action.whole_pair.form_mean == Fraction(5, 8)
assert not action.runtime_admission_certified
assert not action.event_occurrence_derived
evidence = relational_report_to_dict(action)
```

The two single-member targets produce different internal states despite equal
phase and equal collective mean increments. Retaining a target mark resolves
that ambiguity. Applying the same scalar map to both members instead admits
a collective description without that mark. The observer leaves the graph
unchanged and does not execute AL grammar or lifecycle effects. See the
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-pair-emission) for clipping,
saturated no-ops, synchronized tips and the independent occurrence obligation.

### Distinguish autonomous transfer from Emission

Two synchronized pairs with different phases already exchange form under the
declared conservative sine law. This separate preparation compares that
internal current with an AL-only endpoint using one source capture:

```python
from fractions import Fraction
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel, relational_report_to_dict

graph = nx.complete_bipartite_graph(2, 2)
for node in graph:
    graph.nodes[node].update(EPI=0.5, nu_f=1, theta=0 if node < 2 else 0.25)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
action = Network(graph).relational_sine_pair_emission(
    model, pairs=((0, 1), (2, 3)), pair_index=0, boost=0.125,
)
transfer = action.comparison.regional_transfer(region=(0, 1))
assert transfer.regional_form_rate_bounds.lo > 0
assert transfer.complement_form_rate_bounds.hi < 0
assert transfer.regional_balance_residual_bounds.contains(0)
assert transfer.global_weighted_form_conserved

injection = action.form_increment(outcome="whole_pair")
assert injection.comparison is transfer.comparison
assert injection.weighted_form_change == Fraction(1, 2)
assert injection.closed_flow_endpoint_obstructed
balanced = action.comparison.assess_form_increment(
    increments=(Fraction(1, 8), Fraction(1, 8), -Fraction(1, 8), -Fraction(1, 8)),
)
assert not balanced.closed_flow_endpoint_obstructed
assert not balanced.endpoint_reachability_certified
evidence = relational_report_to_dict(transfer)
```

The cut rates use weighted sums, with full-node weights `degree/capacity`.
The positive regional current has a compensating negative current outside;
the hypothetical AL action changes only the first pair and therefore violates
the closed-flow invariant. Compensating increments remove that particular
obstruction, but do not prove an attainable endpoint or supply a final phase.
The [derivation](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-autonomous-regional-transfer)
also gives the phase back-reaction and finite transfer amplitude on this
synchronized submanifold. Neither the observation nor the hypothetical action
advances this graph; [admission and export](../contracts/RELATIONAL_DYNAMICS.md#sine-regional-transfer)
retain the separate model, full source and unresolved endpoint obligations.

Regional currents also remain defined when a pair's mean phasor cancels and
its circular midpoint is unavailable. The
[antipodal-pair result](../../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-zero-resultant-restoration)
uses internal form contrast to restore a current through an isolated zero.
A zero instantaneous current does not imply a disconnected phase channel
or a finite waiting interval. Keep the full comparison and cut reader; the
midpoint-based replica report has a narrower chart. Floating-point `pi`
does not encode the exact antipodal preparation used in that proof.

### Retain pair identity through autonomous exchange

This independent reference has synchronized members within each pair.
Its form and phase differences exchange contributions to one conserved
joint separation. Fix all eight initial error radii and the horizon before
assessing the whole family of perturbed preparations:

```python
from fractions import Fraction
import networkx as nx
from tnfr.sdk import RelationalExchangeModel, relational_report_to_dict
from tnfr.physics.relational_sine_scale import (
    assess_sine_joint_pairing_window,
    observe_joint_pairs,
    observe_phase_pairs,
)

graph = nx.complete_bipartite_graph(2, 2)
for node in graph:
    graph.nodes[node].update(EPI=0.5, nu_f=1, theta=0 if node < 2 else 0.25)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
radius = Fraction(1, 10**6)
window = assess_sine_joint_pairing_window(
    graph, reference_model=model, pairs=((0, 1), (2, 3)),
    form_error_bounds=(radius,) * 4, phase_error_bounds=(radius,) * 4,
    window_end=10,
)
assert window.reference_identity_certified
assert window.whole_window_identity_certified
assert window.candidate_pairs == ((0, 1), (2, 3))
assert window.window_covers_reference_period
evidence = relational_report_to_dict(window)

# Separate static illustration: not a captured point of the above orbit.
phases = (0,) * 4
phase_only = observe_phase_pairs(nodes=range(4), phases=phases)
joint = observe_joint_pairs(
    nodes=range(4), forms=(0.625, 0.625, 0.375, 0.375),
    phases=phases, storage_scale=1,
)
assert phase_only.candidate_pairs is None
assert joint.candidate_pairs == ((0, 1), (2, 3))
```

The phase-only observation cannot distinguish the static equal-phase states;
their form difference distinguishes the pairs in the joint observation.
The window report separately proves that the prepared exchange retains its
joint distinction despite every allowed form and phase perturbation until
time 10. Its period coverage belongs to the exact reference only. Neither
reader advances the graph or changes its law. Larger uncertainty may produce
an unavailable certificate; it need not imply an actual loss of identity.
See the [admission contract](../contracts/RELATIONAL_DYNAMICS.md#sine-joint-pairing-window)
for exact reference versus finite perturbed claims, phase lifts and overflow.

### Classify the allowed acute equilibria

This reader supplies the full list of equilibrium phase families admitted by
the law on the declared support, without selecting a winding in advance:

```python
from itertools import product
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_scale import (
    assess_sine_replica_equilibria,
    observe_phase_pairs,
)
from tnfr.sdk import relational_report_to_dict

graph = nx.Graph()
pairs = tuple(((i, 0), (i, 1)) for i in range(5))
for i, pair in enumerate(pairs):
    for member, node in enumerate(pair):
        graph.add_node(node, EPI=2, theta=0, nu_f=member+1)
for i, j in nx.cycle_graph(5).edges:
    graph.add_edges_from(product(pairs[i], pairs[j]))
model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
equilibria = assess_sine_replica_equilibria(
    graph, reference_model=model, pairs=pairs,
)
assert tuple(target.winding for target in equilibria.targets) == (-1, 0, 1)
assert equilibria.source_equilibrium_status == "certified_consensus_equilibrium"
assert equilibria.source_equilibrium_winding == 0
partners = observe_phase_pairs(
    nodes=equilibria.comparison.nodes, phases=equilibria.comparison.phase,
)
assert partners.candidate_pairs is None  # Consensus gives no unique phase partners.
evidence = relational_report_to_dict(equilibria)
```

All three symbolic families have uniform form; the nonzero windings also
have synchronized structural pairs and uniform increments of one fifth turn
around the base cycle. The captured source above is consensus, and its valid
equilibrium does not make the phase observer's ambiguous grouping unique.
Replacing phases by rounded values near a nonzero target produces a different,
generally moving state. Neither proximity nor tiny rates certify equality.
Read the [contract](../contracts/RELATIONAL_DYNAMICS.md#sine-replica-equilibria)
for the separate source-sector verdicts and represented-radian limits.

### Assess an internal pulse and its perturbations

The same conservative law has an exact prepared periodic family on doubled
C5. The following assessment declares that mathematical family directly;
it does not test whether a rounded graph belongs to it:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_scale import assess_sine_replica_pulse_variation
from tnfr.sdk import relational_report_to_dict

model = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular",
)
variation = assess_sine_replica_pulse_variation(
    reference_model=model,
    form_half_difference=Fraction(1, 64),
    phase_half_difference=0,
    capacity=1,
)
pulse = variation.pulse
assert pulse.target_phase_turns == tuple(Fraction(i, 5) for i in range(5))
assert pulse.nonlinear_periodic_exchange_certified
assert pulse.all_fine_edges_acute_status == "certified"
assert not pulse.graph_membership_certified
assert pulse.period_bounds is not None
assert pulse.unordered_period_bounds == pulse.period_bounds / 2
pulse_evidence = relational_report_to_dict(pulse)
assert variation.mode_multiplicities == (1, 2, 2)
assert 4 * sum(variation.mode_multiplicities) == variation.full_real_dimension == 20
assert variation.periodic_reference_certified
assert variation.orbital_stability_status == "not_assessed"
variation_evidence = relational_report_to_dict(variation)
```

All five pairs share the supplied internal preparation and capacity. Their
means have exact winding-one phases and uniform form. The larger pattern's
means stay fixed while its constituents move. The unordered state repeats
after half the labeled period because every pair exchanges its members.
Bounds use the model's structural clock and an analytic integral inequality;
no trajectory is run. The
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-replica-internal-pulse)
separates exact preparation, pulse, fine-edge acuteness and unavailable
captured-graph membership. General disturbances need a separate stability
analysis; this example does not establish spontaneous preparation or attraction.
The variation report supplies three real block types, covering all twenty
fine-state perturbations. Its matrices describe instantaneous changes, not a
computed return after one period. The
[variation contract](../contracts/RELATIONAL_DYNAMICS.md#sine-replica-pulse-variation)
defines their Fourier coordinates, dimensionless form and required member swap
at half return. Their availability does not mean that stability has passed.

The separate small-amplitude theorem resolves an instability of this family:

```python
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_scale import assess_sine_replica_pulse_splitting
from tnfr.sdk import relational_report_to_dict

model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
splitting = assess_sine_replica_pulse_splitting(reference_model=model, capacity=1)
assert splitting.mode_classifications == ("hyperbolic", "elliptic")
assert splitting.sufficiently_small_nonlinear_orbital_instability_certified
assert splitting.amplitude_upper_bound is None
assert not splitting.finite_preparation_assessed
assert not splitting.return_multipliers_computed
splitting_evidence = relational_report_to_dict(splitting)
```

For sufficiently small nonzero amplitudes, one family of disturbances grows
away from the exactly coordinated pulse. The second is bounded at the linear
level. The theorem proves that a positive interval exists but gives no numerical
endpoint, so this report does not classify the finite preparation in the
preceding example. It also does not imply loss of the collective winding or
total storage. See the
[splitting contract](../contracts/RELATIONAL_DYNAMICS.md#sine-replica-pulse-splitting)
for the analytic coefficients, remainder scope and exact export.

### Check a formation preparation before evolving it

This declared family starts with a twisted donor and a uniform receiver.
A form pulse at the intermediary and unequal receiver capacities produce
an initial directed deformation, but that does not guarantee a new twist:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

model = RelationalExchangeModel(1, phase_domain="regular")
preparation = dict(
    model=model, amplitude=2, capacity_contrast=Fraction(1, 2),
)
initial_checks = assess_sine_mediated_formation(**preparation)
assert initial_checks.initial_form_storage == 4
assert initial_checks.target_budget_status == "passed"
assert initial_checks.entry_budget_status == "passed"
assert initial_checks.phase_action_bound.necessary_entry_time_lower_bound > 240
barrier = initial_checks.maintained_target_obstruction
assert barrier.maintained_target_excluded
assert barrier.target_margin_bounds.lo > Fraction(1, 4)
assert initial_checks.status == "excluded"
assert not initial_checks.early_loss_exclusion_certified
timed = assess_sine_mediated_formation(
    **preparation, exclusion_time=Fraction(1, 5),
)
assert timed.status == "excluded"
assert timed.phase_action_bound.entry_through_horizon_excluded
assert timed.early_loss_exclusion_certified
formation_evidence = timed.to_dict()

# A different exchange/loss balance inside the proved domain.
changed_balance = assess_sine_mediated_formation(
    model=RelationalExchangeModel(
        1, epi_weight=4, phase_weight=5, phase_domain="regular",
    ),
    amplitude=2, capacity_contrast=Fraction(1, 2),
)
changed_barrier = changed_balance.maintained_target_obstruction
assert 1 < changed_barrier.exchange_to_loss_ratio < Fraction(3, 2)
assert changed_barrier.maintained_target_excluded
assert changed_barrier.target_margin_bounds == barrier.target_margin_bounds

# A separate endpoint: the donor unwinds and only the receiver stays twisted.
transfer = timed.receiver_transfer()
assert transfer.source is timed
assert transfer.status == "passes_necessary_conditions"
assert transfer.target_storage_margin == 4
assert transfer.target_functional_margin == Fraction(16, 5)
assert transfer.combined_phase_action_cost == Fraction(29, 25)
assert transfer.compatible_common_phase_turn_offset == Fraction(-11, 190)
assert not transfer.early_loss_exclusion_certified
assert transfer.receiver_first_barrier_phase_storage == Fraction(7, 2)
assert transfer.necessary_receiver_running_work_strict_lower_bound == Fraction(7, 2)
assert transfer.donor_barrier_first_required
assert transfer.simultaneous_barrier_passage_excluded
transfer_evidence = transfer.to_dict()
```

The initial storage checks pass, but the independent mixed-functional
certificate excludes convergence to the maintained two-twist target without
a supplied horizon. A necessary entry time above 240 is conditional on joint
acute entry; it predicts neither entry nor capture. The timed report separately
proves that dissipation exhausts the joint-entry budget before that entry can
occur. Both reports use analytic bounds without evolving a trajectory.

The [whole-class proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-maintained-target-obstruction)
excludes the maintained target for all five donor forms and intermediary form
with initial form storage at most four, under the fixed capacity and storage
premises and the sufficient ratio domain `0<w/e<3/2`. The example calls are
applications of that theorem. This interval is not a sharp physical threshold.
The global
certificate alone does not exclude every transient acute-region visit or
receiver-only winding after losing the donor's identity. Leaving the admitted
ratio domain or changing storage scale or contrast makes this certificate
unavailable; that is not positive formation. Common raw-weight scaling is
removed by ideal normalization; rounded inputs still use their represented
effective ratio. Neither operation implements a new execution clock.

The [contract](../contracts/RELATIONAL_DYNAMICS.md#sine-formation-eligibility-and-timed-exclusion)
documents profile admission, the separate timed bounds and export fields.
Use `profile="explicit"` with five ordered `donor_epi` values to retain internal
donor differences; `amplitude` still supplies intermediary form. No support
event, new loss law or native runtime change is involved.

`receiver_transfer()` keeps this same preparation but asks about a different
final identity. Its passed necessary checks do not prove that the receiver
will acquire the pattern. The original two-pattern target remains excluded.
The [transfer contract](../contracts/RELATIONAL_DYNAMICS.md#receiver-identity-transfer-with-donor-unwinding)
keeps both winding changes, their combined phase-action cost and the new loss
allowance explicit. The reported common phase offset is one compatible lift,
not a predicted final absolute phase. To certify recovery of an actual later
observation, pass `transfer.target_phase_turns` to the existing full-pattern
recovery reader; a positive initial budget does not replace that observation.

The passage flags describe necessary geometry and work **if transfer occurs**.
At form budget at most `4`, the donor must cross its own potential barrier
before the receiver reaches its first one. This is not a statement about
when either winding changes. The receiver must receive accumulated signed
port work exceeding `7/2`, with an additional loss cost for a crossing within
a given time. These are prospective requirements: the initial report does
not evaluate that accumulated work, and a positive instantaneous work rate
does not satisfy them. A false ordering flag leaves the order unknown.

### Distinguish source geometry from a formation prediction

The existing preparation report can retain its communicating and silent
components while bounding the tangent receiver response:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=1, capacity_contrast=Fraction(1, 2),
    profile="explicit", donor_epi=(0, 1, 0, 0, -1),
)
excitation = source.receiver_excitation()
assert excitation.available
assert excitation.even_form_storage == 1
assert excitation.odd_form_storage == 2
assert excitation.tangent_receiver_phase_cost_upper_bound == Fraction(3, 4)
assert excitation.nonlinear_correction_status == "positive_bound"
assert excitation.full_phase_budget_status == "available"
assert excitation.actual_full_phase_storage_upper_bound == Fraction(139, 20)
assert excitation.donor_potential_barrier_first_required
assert excitation.necessary_donor_phase_release_lower_bound > 0
assert tuple(a + b for a, b in zip(
    excitation.even_initial_epi, excitation.odd_initial_epi,
)) == source.initial_epi
excitation_evidence = excitation.to_dict()
```

This example evaluates no trajectory. The tangent ceiling and correction
threshold describe the response needed for nonlinear barrier passage. The
separate full-phase ceiling applies to the actual law at total form budget
at most `4`: receiver passage requires an earlier donor barrier crossing.
Neither bound proves that passage occurs or identifies the donor's potential
decrease as useful receiver work.
Reversing only the odd source retains its energy and tangent receiver response
but changes nonlinear receiver work. Keep both components when studying the
actual law. The [contract](../contracts/RELATIONAL_DYNAMICS.md#source-geometry-and-the-necessary-nonlinear-receiver-correction)
defines unavailable bounds and the [proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-source-receiver-excitation)
states the exact distinction; neither selects this example for a new response.

### Certify receiver consensus without selecting the donor endpoint

A regional nonlinear certificate applies to every six-coordinate preparation
whose total initial form storage is at most `(5-sqrt(5))/2`. It can establish
receiver consensus when the separate donor-recovery checks are inconclusive:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=1, capacity_contrast=Fraction(1, 2),
)
assert source.initial_form_storage == 1
assert not source.donor_well_retention().relative_donor_pattern_convergence_certified
assert not source.donor_dissipative_capture().relative_donor_pattern_convergence_certified
localization = source.receiver_localization()
assert localization.status == "certified"
assert localization.relative_receiver_consensus_certified
assert localization.receiver_nonflat_equilibria_excluded
transfer = source.receiver_transfer()
assert transfer.receiver_localization == localization
assert "weighted_receiver_localization" in transfer.exclusion_reasons
assert transfer.status == "excluded"
localization_evidence = localization.to_dict()
```

This evaluates no trajectory. The proof weights regional potentials and
retains their full nonlinear coupling. It selects the receiver's relative
limit, not a donor limit or a time to recovery. Transient receiver excursions
remain possible; no all-time barrier verdict follows. A `not_certified` result
above the threshold leaves acquisition unresolved. The
[contract](../contracts/RELATIONAL_DYNAMICS.md#nonlinear-receiver-localization-from-regional-storage)
and [proof](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#sine-weighted-receiver-exclusion)
own the exact admission and scope.

### Certify recovery of the prepared donor

The same preparation reader can now select the donor-only relative limit
for a proved interval of form storage. This example is nonsilent and has
enough total storage to pass the bare phase barrier, but the full coupled
dynamics still returns to the original donor pattern:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=Fraction(7, 30), capacity_contrast=Fraction(1, 2),
)
retention = source.donor_well_retention()
assert retention.initial_form_storage == Fraction(49, 900)
assert retention.relative_donor_pattern_convergence_certified
assert retention.receiver_only_targets_excluded
assert retention.exact_retention_polynomial_margin > 0
transfer = source.receiver_transfer()
assert transfer.status == "excluded"
assert transfer.donor_well_retention.relative_donor_pattern_convergence_certified
retention_evidence = retention.to_dict()
```

The certificate applies to all five donor forms and the intermediary form,
not only this localized example. It combines the exact equilibrium geometry
with the existing decreasing auxiliary function. Its threshold is
`F_c=(25*sqrt(5)-55)/16`, approximately `0.056356`, and admission compares
`(16*F+55)**2 <= 3125` exactly. Neither this proof coefficient nor its threshold
is a new physical constant.

This proves the final relative donor identity and excludes either maintained
receiver-only twist. It gives no arrival time or guarantee that winding never
changes during the transient. Above the threshold the certificate does not
decide the outcome; a connected static path is insufficient. The
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-donor-well-retention)
retains the law premises, exact admission and distinct transfer bounds.

### Prove dissipative recovery above the initial barrier

A preparation can exceed the static donor-retention bound and still recover
the donor. This certificate uses the full law's short-time dissipation and
phase displacement to prove entry into the donor component:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=Fraction(1, 4), capacity_contrast=Fraction(1, 2),
)
assert source.initial_form_storage == Fraction(1, 16)
assert source.donor_well_retention().status == "not_certified"
capture = source.donor_dissipative_capture()
assert capture.horizon == Fraction(1, 4)
assert capture.initial_dissipative_norm_squared == Fraction(1, 6)
assert capture.donor_component_entry_certified
assert capture.relative_donor_pattern_convergence_certified
transfer = source.receiver_transfer()
assert transfer.status == "excluded"
assert transfer.donor_dissipative_capture.receiver_only_targets_excluded
capture_evidence = capture.to_dict()
```

The horizon certifies membership in a region that guarantees eventual donor
recovery; it is not the time at which recovery finishes. No trajectory is
numerically evolved. The bound uses both form storage and its full-support
gradient, so equal storage alone does not determine its verdict. A failed
sufficient check leaves the outcome open. This certificate has the original
half-weight law scope; the static donor-well result retains its wider
coefficient-ratio domain. See the
[capture contract](../contracts/RELATIONAL_DYNAMICS.md#sine-donor-dissipative-capture)
for exact admission and retained evidence.

### Compose established component Hessians through a bridge tree

Use this algebra only after deriving each component's relative Hessian inertia
and the bridge signs. Here two C5 components have independently established
positive four-dimensional relative Hessians; a singleton mediates their union:

```python
from tnfr.physics.phase_cycle_geometry import compose_bridge_tree_hessian_inertia

components = ((4, 0, 0), (4, 0, 0), (0, 0, 0))
aligned = compose_bridge_tree_hessian_inertia(
    components, ((0, 2, 1), (2, 1, 1)),
)
assert aligned.total_nodes == 11
assert aligned.relative_inertia == (10, 0, 0)
opposed = compose_bridge_tree_hessian_inertia(
    components, ((0, 2, -1), (2, 1, 1)),
)
assert opposed.relative_inertia == (9, 1, 0)
composition_evidence = aligned.to_dict()
```

This does not reconstruct or check a live phase configuration. The specialized
C5 catalog below supplies those premises from its exact critical states and
delegates to the same algebra. Under the full positive-loss sine theorem,
an aligned bridge preserves positive restoring geometry and an antipodal
bridge adds an unstable direction. Extra connections that close a cycle
require another compatibility argument; neither stability nor the input
inertia triples establish formation. See the
[contract](../contracts/RELATIONAL_DYNAMICS.md#exact-geometric-inertia-and-full-law-stability)
and [proof](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-bridge-tree-composition).

### Reject incompatible cycle periods before seeking an equilibrium

Cycles sharing paths cannot choose their phase periods independently. The
shared exact reader tests one combined-cycle constraint, including cancellation
on shared edges. This example has two fundamental five-edge cycles; each
period individually meets its acute length bound, but their joint assignment
fails on the four-edge cycle formed by their difference.

```python
import networkx as nx
from tnfr.physics.phase_cycle_geometry import (
    assess_acute_cycle_periods,
    derive_phase_cycle_geometry,
)

graph = nx.Graph()
graph.add_nodes_from(range(6))
graph.add_edges_from(((0, 1), (0, 2), (2, 3), (1, 4), (3, 4), (1, 5), (3, 5)))
geometry = derive_phase_cycle_geometry(graph)
assert tuple(map(len, geometry.fundamental_cycles)) == (5, 5)

single = assess_acute_cycle_periods(
    geometry, cycle_periods=(1, -1), cycle_combination=(1, 0)
)
assert single.status == "necessary_bound_passed"
joint = assess_acute_cycle_periods(
    geometry, cycle_periods=(1, -1), cycle_combination=(1, -1)
)
assert joint.combined_period == 2
assert joint.strict_period_bound == 1
assert joint.obstruction_certified
payload = joint.to_dict()
```

An obstruction rules out every strictly acute state with those periods,
not only one proposed phase assignment. Passing a witness leaves existence
unresolved. Even a reconstructed acute phase state can fail the separate
nodal sine-balance condition. The
[proof and controls](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-cycle-sector-compatibility)
distinguish these obligations. This readout creates no connections and does
not establish that a compatible pattern forms.

### Certify sector capture without supplying its final pattern

A whole-sector storage barrier can guarantee convergence without giving the
reader equilibrium coordinates. This source has nonuniform form and a phase
profile with nonzero cycle period. Its future is assessed under the explicitly
supplied complete sine law; the initial state is not declared stationary.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.cycle_graph(5)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(1, 1024) if node == 0 else 0,
        theta=Fraction(5 * node, 4),
        nu_f=2 if node == 1 else 1,
    )
source = bound_relational_sine_pattern(
    graph,
    reference_node=0,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(Fraction(1, 65536),) * 5,
    phase_error_bounds=(Fraction(1, 65536),) * 5,
)
# Canonical edges: (0,1), (0,4), (1,2), (2,3), (3,4).
# Subtract a full turn from the 0->4 phase difference.
capture = source.certify_sector_capture(edge_turn_offsets=(0, -1, 0, 0, 0))
assert capture.cycle_periods == (1,)
assert capture.admitted
assert capture.energy_margin > 0
assert len(capture.boundary_face_lower_bounds) == 10
assert capture.weighted_form_mean is None  # the common origin is unobserved
payload = capture.to_dict()
```

The certificate covers all signed faces of the acute sector and the entire
declared uncertainty set. A positive verdict proves existence, uniqueness and
convergence there without a target solve. An unavailable verdict may reflect
a conservative bound even when an equilibrium exists. This does not generate
the initial winding or support, and a supplied event needs separate state and
storage-jump admission. See the [contract](../contracts/RELATIONAL_DYNAMICS.md#target-free-acute-sector-capture)
and [proof](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture).

### Certify acquisition from initially equal phases

The same sine law can transfer a supplied form profile into nonzero phase
winding. This example declares the profile, coefficients and horizon before
evaluating the analytic certificate. It uses no trajectory fit or supplied
equilibrium coordinates.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_prepared_entry

graph = nx.cycle_graph(5)
for node in graph:
    graph.nodes[node].update(EPI=4092 * (node - 2), theta=0, nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(
    1, epi_weight=Fraction(1023, 1024),
    phase_weight=Fraction(1, 1024), phase_domain="regular",
)
source = bound_relational_sine_exchange(graph, reference_model=model)
entry = certify_sine_prepared_entry(
    source, scaled_time=100, edge_turn_offsets=(0, -1, 0, 0, 0),
)
assert entry.admitted
assert entry.initial_cycle_periods == (0,)
assert entry.capture.cycle_periods == (1,)
assert entry.horizon == Fraction(102400, 1023)
assert entry.initial_form_storage == 167444640
assert entry.capture.energy_margin > Fraction(1, 100)
payload = entry.to_dict()
```

`endpoint_form_bounds` and `endpoint_phase_bounds` enclose all nodes at the
declared horizon. Tighter correlated edge bounds feed the shared capture
kernel; the full Cartesian node box is only an outer projection. The initial
phase winding is absent, but the nonuniform form information and large initial
storage are supplied. The result proves formation and subsequent retention
under this law; it does not select that preparation or identify matter.
Uniform initial form with equal phases instead stays stationary. See the
[contract](../contracts/RELATIONAL_DYNAMICS.md#analytic-prepared-entry-into-a-captured-sector)
and [proof](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-prepared-sector-entry).

The same graph, model and horizon also admit a full preparation family. Keep
the residual radii fixed before evaluating the following extension:

```python
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

preparations = bound_relational_sine_pattern(
    graph, reference_node=0, reference_model=model,
    form_error_bounds=(Fraction(1, 16),) * 5,
    phase_error_bounds=(Fraction(1, 65536),) * 5,
)
family = preparations.certify_prepared_entry(
    scaled_time=100, edge_turn_offsets=(0, -1, 0, 0, 0),
)
assert family.admitted and family.initial_zero_winding_certified
assert family.initial_cycle_periods == (0,)
assert family.capture.cycle_periods == (1,)
assert family.capture.energy_margin > Fraction(12, 1000)
assert family.endpoint_form_bounds is None
assert family.weighted_form_mean is None
assert len(family.centered_endpoint_phase_bounds) == 5
```

Every member starts with zero winding and acquires the captured unit winding.
The relative source leaves common origins unknown: use the centered endpoint
bounds, not invented absolute values. Each member keeps its own conserved
means. Initial form and phase storage are interval bounds, so the nominal
budget does not describe the entire family. With uniform nominal form and
the same small errors, the initial zero-sector capture test instead proves
convergence to consensus; that uncertain control is not stationary.

### Certify an entire phase-flat budget family

Keep the preceding `1023:1` coefficient ratio, but bound the original form
storage by `B=160`. This declares every signed form within that ceiling and
exactly equal initial phases, with arbitrary common origins. The graph
supplies only support; no synthetic nodal state is needed.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.phase_cycle_geometry import derive_phase_cycle_geometry
from tnfr.physics.relational_sine_budget import certify_sine_budget_consensus
from tnfr.sdk import relational_report_to_dict

geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
model = RelationalExchangeModel(
    1, epi_weight=1023, phase_weight=1, phase_domain="regular",
)
budget = certify_sine_budget_consensus(
    geometry, reference_model=model, capacity=(1,) * 5,
    form_storage_budget=160,
)
assert budget.status == "certified"
assert budget.consensus_certified and budget.zero_winding_certified
assert budget.acute_trapping_certified
assert max(budget.phase_edge_upper_bounds) < Fraction(1, 50)
payload = relational_report_to_dict(budget)
```

This certifies consensus and zero winding for the entire family at all future
times. The earlier acquisition source costs `160*1023^2` and lies outside this
family. A zero budget instead makes the whole family stationary. In general,
the theorem protects the natural `(-pi,pi)` edge strip; strict acuteness is a
separate, stronger flag. Its `eta<=1` condition is a sufficient proof premise,
not an instability threshold. When a comparison is unavailable, inspect
`unresolved`: candidate bounds remain visible, but certified trajectory
bounds are `None`.

Under the feedback premise, `nonzero_winding_budget_lower_bound` records a
necessary cost growing as `(w/e)^-2`; exceeding it does not prove formation.
The [contract](../contracts/RELATIONAL_DYNAMICS.md#sine-budget-consensus) and
[proof](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-budget-consensus)
give the family and admission boundaries. The exact `payload` can be saved
with the SDK's `export_to_json`.

### Distinguish equal budgets by preparation symmetry

The acquisition example above uses initial storage `160*1023^2`. The
following reflection-fixed preparation has exactly the same budget under
the same law, but its symmetry excludes nonzero winding on the declared
cycle whenever principal increments are defined.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.sdk import relational_report_to_dict

graph = nx.cycle_graph(5)
for node, value in enumerate((8, 4, -8, -8, 4)):
    graph.nodes[node].update(EPI=1023 * value, theta=0, nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(
    1, epi_weight=Fraction(1023, 1024),
    phase_weight=Fraction(1, 1024), phase_domain="regular",
)
source = bound_relational_sine_exchange(graph, reference_model=model)
symmetry = source.assess_cycle_symmetry(
    permutation_indices=(0, 4, 3, 2, 1), cycle=(0, 1, 2, 3, 4),
)
assert symmetry.initial_form_storage == 160 * 1023**2
assert symmetry.status == "certified"
assert symmetry.trajectory_symmetry_certified
assert symmetry.zero_winding_when_nonantipodal
assert symmetry.nonzero_winding_limit_excluded
payload = relational_report_to_dict(symmetry)
```

Supply an exact comparison and a permutation in its complete node order.
Inspect the symmetry and winding flags separately: the moving source is
neither certified stationary nor guaranteed to avoid antipodal edges.
Relative uncertainty reports are unsupported by this reader.

The [positive acquisition example](#certify-acquisition-from-initially-equal-phases)
already supplies the opposite outcome at this budget. Scalar resources alone
therefore do not determine the acquired geometry. See the
[contract](../contracts/RELATIONAL_DYNAMICS.md#sine-cycle-symmetry) and
[equal-budget proof](../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-equal-budget-preparation)
for the full-state symmetry and orientation conditions.

### Classify long-time endpoints without selecting a basin

For two C5 rings connected through one intermediary, the complete sine law
has a finite exact set of relative equilibria. The following source has
arbitrary finite form and phase; the reader does not require a prepared twist:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

graph = nx.Graph()
graph.add_nodes_from(range(11))
cycles = (tuple(range(5)), tuple(range(5, 10)))
for cycle in cycles:
    nx.add_cycle(graph, cycle)
graph.add_edges_from(((0, 10), (5, 10)))
for node in graph:
    graph.nodes[node].update(EPI=node / 10, theta=node / 7, nu_f=1)
graph.nodes[6]["nu_f"] = 1.5
graph.nodes[9]["nu_f"] = 0.5

source = bound_relational_sine_exchange(
    graph, reference_model=RelationalExchangeModel(1, phase_domain="regular"),
)
result = source.asymptotic_equilibria(cycles=cycles)
assert result.admitted
assert result.single_relative_equilibrium_convergence_certified
assert result.full_lifted_state_convergence_certified
assert result.selected_equilibrium_status == "unavailable_no_basin_selection"

catalog = result.critical_set
assert len(catalog.cycle_edge_turn_options) == 30
assert catalog.relative_state_count == 3600
# Pick one symbolic catalog member, not the predicted endpoint of this source.
member = catalog.reconstruct(
    cycle_choices=(0, 0), bridge_turns=(0, Fraction(1, 2)),
)
assert not any(member.symbolic_sine_coefficients)
endpoint_evidence = result.to_dict()

# Classify a declared target under the source's law, not the source's basin.
twist = catalog.cycle_edge_turn_options.index((Fraction(1, 5),) * 5)
stable = result.classify_equilibrium(
    cycle_choices=(twist, twist), bridge_turns=(0, 0),
)
assert stable.local_exponential_attraction_certified
assert stable.relative_stable_modes == 20
assert stable.relative_unstable_modes == 0
opposed = result.classify_equilibrium(
    cycle_choices=(twist, twist), bridge_turns=(0, Fraction(1, 2)),
)
assert opposed.nonlinear_instability_certified
assert opposed.relative_unstable_modes == 1
assert catalog.phase_hessian_index_counts[0] == 9
assert sum(catalog.phase_hessian_index_counts) == 3600
stability_evidence = stable.to_dict()
```

Each cycle choice indexes its declared oriented ring. The two bridge choices
follow `catalog.bridge_edge_indices` in the stored full geometry; inspect
those indices rather than assuming label sorting or cycle order. Exact turns
in the reconstructed member represent multiples of mathematical `2*pi`.
They are distinct from the captured source's represented radian values.

The convergence theorem does not select the member, its terminal phase turns
or a numerical rate. The separate stability reader classifies each declared
target: nine are locally exponentially attracting modulo common origins;
the other 3,591 are unstable. This does not place the observed source in a
target's basin or supply a certified neighborhood radius. Adding a half-turn
on one bridge in this example creates one unstable direction in the complete
form-phase dynamics, even though both ring geometries remain individually
acute. The catalog keeps all nonacute branches. It stores 30 local
options and independent bridge choices, avoiding a list of 3,600 full states.
Zero loss or any zero capacity makes the stronger convergence verdict
unavailable. The [contract](../contracts/RELATIONAL_DYNAMICS.md#sine-asymptotic-equilibria)
separates that admission from geometric classification and from native Arg
dynamics. Both evidence dictionaries can be saved using the SDK's
`export_to_json`.

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
See the [contract](../contracts/RELATIONAL_DYNAMICS.md#hidden-state-inference-from-prior-visible-rates).
The [capacity contract](../contracts/RELATIONAL_DYNAMICS.md#hidden-capacity-inference-from-prior-accelerations)
also explains why a capacity bound can survive unresolved phase and why
that does not suffice for predicting the whole state.

### Declare the full-state receiver barrier experiment

The source-specific [receiver barrier protocol](../contracts/RELATIONAL_DYNAMICS.md#frozen-receiver-barrier-exclusion)
combines a whole-time initial enclosure with a total-energy tail. Preparation
does not run the trajectory:

```python
from tnfr.research.relational_receiver_barrier import prepare_receiver_barrier

protocol = prepare_receiver_barrier()
assert len(protocol["nodes"]) == 11
assert protocol["maximum_steps"] == 256
```

The [benchmark](../../benchmarks/relational_receiver_barrier.py) archives the
protocol with `--prepare`; the same command without that flag performs its
one fixed-budget evaluation. Use a fresh output path for a separately declared
regression, never overwrite the original response. A partial enclosure is
unavailable; a failed upper-bound comparison is unresolved. Neither establishes
receiver formation. The execution plan owns the retained scientific verdict.

### Issue a prior-only sine forecast

After constructing a capacity report as above, call `admit_sine_prior(capacity)`
from `tnfr.physics.relational_sine_forecast` and inspect `admitted` and `reasons`.
For an admitted report, call `forecast_sine_prior` with exact `end_time`,
`time_step` and optional `order`; retain its `validated_end_time` even if the
requested horizon is unavailable.
See the [API contract](../contracts/RELATIONAL_DYNAMICS.md#joint-prior-admission-and-sine-forecasts)
for exact-time admission, report fields and failure behavior.

The [reserved software instrument](../../benchmarks/relational_sine_prior_forecast.py)
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
[autonomous path example](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md#autonomous-path-cancellation)
crosses cancellation while the actual intermediary remains fixed and its
conditional minimum changes.
The graph keeps all three nodes. See the
[contract](../contracts/RELATIONAL_DYNAMICS.md#detached-stationary-sine-mediation)
for source provenance, retained degrees, admission and exact JSON export.
