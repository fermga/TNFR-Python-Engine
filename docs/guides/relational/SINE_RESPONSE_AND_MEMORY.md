# Sine response, constitutive comparisons and hidden memory

Declared input/output resonance, bridge-law and clock discrimination, return-path geometry, causal hidden memory, autonomous pulses and recurrence distinctions.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

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
The [contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#sine-resonance) specifies
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
[proof](../../../theory/nodal/SINE_PATTERN_RECOVERY.md#sine-interacting-recovery)
separately derives donor-to-receiver influence and its frozen-mediator
control. Recovery admission alone is not a measurement of that response.
The [contract](../../contracts/relational/SINE_PATTERNS.md#whole-set-sine-recovery-with-a-retained-environment)
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
The [contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#sine-mediated-response)
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
[contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#sine-pair-pulse) describes
zero-capacity, equilibrium and unresolved-period cases; the
[proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
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
[contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#sine-conservative-path-memory)
keeps these scopes and the two clocks explicit. The
[derivation](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#finite-conservative-memory)
also gives an analytic tangent example where storage returns to its initial
region instead of being dissipated.

<a id="compare-sine-bridge-channels"></a>

### Compare form and phase at a conservative bridge

Use an admitted critical geometry to compare the two response channels. This
example supplies two winding-one C6 cycles and an aligned bridge; it does not
claim that the geometry formed from the captured consensus state.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_resonance import assess_sine_bridge_channels
from tnfr.sdk import relational_report_to_dict

graph = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))
graph.add_edge(0, 6)
graph.graph["GAMMA"] = {"type": "none"}
for node in graph:
    graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
model = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular",
)
source = bound_relational_sine_exchange(graph, reference_model=model)
channels = assess_sine_bridge_channels(
    source, bridge=(0, 6),
    target_phase_turns=tuple(Fraction(i % 6, 6) for i in range(12)),
)
assert channels.channel_difference_is_positive
assert channels.channel_difference_bounds.contains(Fraction(2, 9))
assert channels.second_jet_diagonal_bounds[0].contains(Fraction(-2, 3))
assert channels.second_jet_diagonal_bounds[1].contains(Fraction(-8, 9))
channel_evidence = relational_report_to_dict(channels)
```

The derivatives use the declared clock `tau=w*t/pi` and energy-normalized
bridge observations. Phase has greater initial response curvature in this
example; this is neither irreversible loss nor universal form damping.
The [contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#sine-bridge-channels)
specifies the admissible nodal preparation, mean modes and hidden-state scope.

<a id="compare-bridge-storage-family"></a>

### Separate pattern identity from its constitutive response

Continue the preceding two-C6 bridge example. Keep its exact target, support,
clock, observation and nodal perturbation fixed; change only the declared
phase potential through the already admitted cubic-sine family.

```python
from tnfr.physics.relational_sine_resonance import assess_bridge_storage_family

storage_family = tuple(
    assess_bridge_storage_family(
        source,
        left_cycle=tuple(range(6)), right_cycle=tuple(range(6, 12)),
        target_phase_turns=channels.target_phase_turns,
        epsilon=coefficient,
    )
    for coefficient in (Fraction(0), Fraction(4, 9), Fraction(1))
)
assert tuple(report.channel_difference for report in storage_family) == (
    Fraction(2, 9), Fraction(0), Fraction(-5, 18),
)
assert all(
    report.nodal_bridge_preparation == storage_family[0].nodal_bridge_preparation
    for report in storage_family
)
assert storage_family[1].phase_hessian == storage_family[1].form_laplacian
family_evidence = relational_report_to_dict(storage_family[1])
assert family_evidence["report_type"] == "BridgeStorageFamilyAssessment"
```

Every member retains this critical winding pattern and conditional local
protection, while the form-minus-phase second response derivative changes
sign. Protection uses each law's own excess storage and an initially admitted
phase neighborhood; this example does not test the captured source against
that condition. At `4/9` the tangent channels have equal restoring geometry,
but the nonlinear law remains distinct. No alternative law is installed by
the comparison. The [contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#bridge-storage-family)
and [proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#storage-family-pattern-robustness)
keep these distinctions explicit.

<a id="finite-bridge-law-discrimination"></a>

### Distinguish the laws through a finite response with bounded errors

Reuse the preceding two-C6 `source` and exact `channels.target_phase_turns`.
Declare two opposite form preparations, the same phase target and a common
positive observation time. The reader bounds the complete nonlinear response;
it does not execute or consume a trajectory.

```python
from tnfr.physics.relational_bridge_discrimination import (
    assess_bridge_finite_law_discrimination,
)

finite_response = assess_bridge_finite_law_discrimination(
    source,
    left_cycle=tuple(range(6)), right_cycle=tuple(range(6, 12)),
    target_phase_turns=channels.target_phase_turns,
    amplitude=Fraction(1, 10), duration=Fraction(1, 50),
    preparation_error=Fraction(1, 10**10),
    observation_error=Fraction(1, 10**8),
)
assert finite_response.discrimination_certified
assert finite_response.phase_chamber_certified
sine_prediction, cubic_prediction = finite_response.curvature_prediction_bounds
assert sine_prediction.hi < cubic_prediction.lo
assert finite_response.separation_margin_lower_bound > Fraction(249, 100000)
assert relational_report_to_dict(finite_response)["report_type"] == "BridgeFiniteLawDiscrimination"
```

The reported intervals concern `C=(2*a-u_plus(h)+u_minus(h))/(a*h**2)`.
Each reading is a full bridge-form difference. Both ideal preparations use
the same magnitude; their independent source errors may break symmetry and
are propagated separately. The finite-amplitude curvature coefficient is
exact, while the finite-time remainder remains explicitly bounded.

The [contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#finite-bridge-law-discrimination)
retains all twenty-four coordinates and distinguishes exact target turns from
their numerical representation. This example predicts which response ranges
would distinguish the models at the declared structural clock. It supplies
no measured response, physical interpretation or unknown-clock cancellation.

<a id="finite-bridge-clock-law-discrimination"></a>

### Distinguish the laws with an uncertain common clock scale

Add the independent phase preparation to the same two-C6 study. Keep the
previous amplitude, sampling duration and error budgets, but allow a common
unknown constant clock scale between `9/10` and `11/10`:

```python
from tnfr.physics.relational_bridge_discrimination import (
    assess_bridge_clock_law_discrimination,
)

clock_response = assess_bridge_clock_law_discrimination(
    source,
    left_cycle=tuple(range(6)), right_cycle=tuple(range(6, 12)),
    target_phase_turns=channels.target_phase_turns,
    amplitude=Fraction(1, 10), sampling_duration=Fraction(1, 50),
    clock_scale_bounds=(Fraction(9, 10), Fraction(11, 10)),
    preparation_error=Fraction(1, 10**10),
    observation_error=Fraction(1, 10**8),
)
assert clock_response.discrimination_certified
assert clock_response.separation_margin_lower_bound > Fraction(1, 2000)
assert relational_report_to_dict(clock_response)["report_type"] == "BridgeClockLawDiscrimination"
```

This predicts a ratio of three positive-time readings: the two opposite form
responses and one response after a phase perturbation. The phase reading is a
signed bridge gap on the target-relative branch. Both law intervals account
for every admitted clock scale and full-coordinate preparation error. The
common clock factor cancels algebraically in the ratio; its effect on finite
time error remains bounded. A nonpositive denominator would leave the ratio
unavailable. The [contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#finite-bridge-clock-law-discrimination)
states those admission rules. No trajectory or physical reading is generated.

<a id="compare-return-path-storage-geometry"></a>

### Compare the geometry selected by different admitted potentials

This standalone example uses the retained two-C5 return-path support. The
named periods are fixed, but the critical phases are solved from each law
rather than supplied as one common target.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.phase_cycle_geometry import assess_return_path_storage_geometry
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.sdk import relational_report_to_dict

return_graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
return_graph.add_edges_from(((0, 10), (10, 5), (1, 6)))
return_graph.graph["GAMMA"] = {"type": "none"}
for node in return_graph:
    return_graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
return_source = bound_relational_sine_exchange(
    return_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
return_geometries = tuple(
    assess_return_path_storage_geometry(
        return_source, left_cycle=tuple(range(5)),
        right_cycle=tuple(range(5, 10)), mediator=10,
        epsilon=coefficient, refinements=40,
    )
    for coefficient in (Fraction(0), Fraction(1))
)
baseline, alternative = return_geometries
assert baseline.named_cycle_periods == alternative.named_cycle_periods == (1, -1, 0)
assert baseline.root_turn_bracket.upper < alternative.root_turn_bracket.lower
assert alternative.minimum_acute_margin_turns_bounds.lo > Fraction(1, 48)
return_evidence = relational_report_to_dict(alternative)
assert return_evidence["report_type"] == "ReturnPathStorageGeometryAssessment"
```

The increase of the special edge angle also increases the connecting gaps
and decreases the bulk ring gaps. Support and periods stay fixed. The
affine coordinates share one unknown exact root; midpoint substitution or
independent sampling from their intervals does not preserve certified
equilibrium. This is a static comparison of declared laws, without installing
either law or claiming that the captured flat phase formed the target. The
[contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#return-path-storage-geometry)
and [derivation](../../../theory/nodal/RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-storage-dependence)
explain the complete-law and numerical boundaries.

<a id="infer-return-path-response"></a>

### Predict a response from a supplied geometric interval

Reuse `return_source` above for its support and independently fixed scales.
The interval below is the independently quantized geometry in the retained
[known-source control](../../../theory/nodal/RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-geometry-response).
The inference function receives no source coefficient and no evaluated response.

```python
from tnfr.physics.phase_cycle_geometry import assess_return_path_geometry_response

geometry_prediction = assess_return_path_geometry_response(
    return_source,
    left_cycle=tuple(range(5)), right_cycle=tuple(range(5, 10)), mediator=10,
    special_turn_bounds=(Fraction(1101, 8000), Fraction(68813, 500000)),
    form_direction=(0,) * 10 + (1,),
    observation_origin="supplied_mathematical_interval",
)
assert geometry_prediction.coefficient_status == "bounded"
assert geometry_prediction.coefficient_lower > 0
assert len(geometry_prediction.response_acceleration_bounds) == 11
prediction_evidence = relational_report_to_dict(geometry_prediction)
assert prediction_evidence["report_type"] == "ReturnPathGeometryResponseAssessment"
```

The output bounds the initial form acceleration after a mediator form
perturbation, with initially unchanged phases. It is conditional on the
declared family, equilibrium and scales; it is not a finite-time forecast.
Always check status first: intervals reaching the limiting geometry can leave
the coefficient unbounded above and `response_acceleration_bounds=None`.
See the [contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#return-path-geometry-response).

The [benchmark lifecycle](../../../benchmarks/README.md#freeze-before-evaluating-a-reserved-response)
owns the separate `prepare`, `predict` and `evaluate` invocations. The retained
known-source record is already evaluated; running this guide is not a request
to regenerate it. A separately declared current-source check writes new local
records and does not authenticate or replace the original evidence.

<a id="retain-sine-bridge-memory"></a>

### Retain the bridge's internal memory

Continue the [two-C6 bridge-channel example](#compare-sine-bridge-channels)
with its `source` and `channels` objects, rather than the intervening
eleven-node return-path preparation. This
reader keeps every linear bridge response in eight shell coordinates, then
retains six of them as hidden memory rather than assigning a fitted loss.

```python
from tnfr.physics.relational_sine_bridge_memory import assess_sine_bridge_memory

bridge_memory = assess_sine_bridge_memory(
    source, left_cycle=tuple(range(6)), right_cycle=tuple(range(6, 12)),
    target_phase_turns=channels.target_phase_turns,
)
assert bridge_memory.visible_observation.dimension == 8
assert bridge_memory.coordinate_memory.visible_indices == (0, 4)
assert bridge_memory.coordinate_memory.kernel_at_zero == (
    (Fraction(-2, 9), 0), (0, Fraction(-4, 9)),
)
assert bridge_memory.static_visible_generator == (
    (0, Fraction(-2, 13)), (Fraction(2, 13), 0),
)
memory_evidence = relational_report_to_dict(bridge_memory)
```

`coordinate_memory` supplies both the convolution kernel and independent
hidden initial-state source. Its matrices act on perturbations of the declared
critical target, not on the captured source as if it had reached that target.
The static matrix is a zero-frequency algebraic coefficient; the oscillatory
memory does not decay, and that coefficient contains no damping. The
[contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#sine-bridge-memory) gives the
projection order, source map and limits of any proposed local approximation.

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
[contract](../../contracts/relational/SINE_RESPONSE_AND_MEMORY.md#sine-nonlinear-recurrence)
and [proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
distinguish circular-state recurrence, stationary states, uncertain membership
and the absence of a return-time or fixed-period prediction.

The [joint-pattern application](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-joint-recurrent-episodes)
adds an independently proved acquisition neighborhood on doubled C5. Almost
every preparation in that neighborhood repeatedly acquires and loses the
specified matching. The example above does not test membership in that
neighborhood or predict episodes for its captured state.
