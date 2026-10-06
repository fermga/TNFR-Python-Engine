# Native relational capture, memory and continuous proofs

Native capture and exclusions, cycle/contact memory, transit, reflected proofs, regularity and retained robustness evidence.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

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
The [contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#phase-consensus-capture)
and [proof](../../../theory/nodal/RELATIONAL_FORMATION_CONTROLS.md#relational-consensus-preparation-obstruction)
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
when admission fails. The [contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#full-form-consensus-obstruction)
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
[capture contracts](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#conditional-relational-capture)
and [maintained example](../../../examples/08_emergent_geometry/180_relational_exchange.py)
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
[contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#relational-detachment-observation)
and [proof](../../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#relational-pattern-detachment)
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
The [contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#relational-seeded-formation-obstruction)
specifies the initial and target domains and the wider regular states it does
not exclude. This function neither executes a contact nor changes the phase
admission used by the engine.

For the separate regular two-port candidate, the research API
`certify_relational_reflected_transit` encloses an exact eight-coordinate
source/receiver state, with independent forms and phases for the two rings.
It returns whole-time domain, winding and storage/loss evidence under its
declared fixed-support law. It does not project arbitrary `Network` states
onto reflection symmetry. The [contract](../../contracts/RELATIONAL_DYNAMICS.md)
defines admission and report fields; the
[retained response](../../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regular-seeded-continuous-response)
documents the certified short window and its formation limits. Inspect that
evidence without replaying its frozen producer.

The same proof owner exposes `certify_relational_reflected_barrier` for a
detached exact interval state and model. This static test can exclude the
two-acute-twist target through a geometric transition barrier even while the
weaker target-storage budget remains positive. It does not advance the state
or infer a reflected state from a live graph. See the
[collective bound](../../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#reflected-collective-energy-barrier)
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
[contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#ideal-reflected-equilibria-and-full-network-stability)
defines the families, exact evidence and report fields; the
[classification](../../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#reflected-regular-equilibria)
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
will do so. The [contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#global-regularity-evidence-and-boundary-access-limits)
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
[contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#relational-cycle-memory) for
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
them as zero. The [contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#relational-finite-memory)
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
The [contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#relational-memory-readout)
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
below. See the [contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#relational-memory-contact)
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
See the [contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#relational-memory-retention)
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
[transit contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#validated-conditional-relational-transit)
and its [proof owner](../../../theory/nodal/RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit).
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
[contract](../../contracts/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#retained-formation-robustness)
distinguishes the preserved reflected basin, conservative parameter bounds,
missing evidence and inconsistent records.
