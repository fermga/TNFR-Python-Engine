# Conservative sine regional and collective dynamics

Regional storage and currents, conservative winding acquisition and retention, compatible collective families, actual contact feedback and forecast-derived organization.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="conservative-regional-winding"></a>

### Certify finite winding from conservative environmental exchange

This standalone example retains two cycles, an intermediary and a supplied
return edge. All phases start equal and the receiver starts with zero form;
only two environmental nodes carry a supplied form contrast.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_conservative_winding_entry
from tnfr.sdk import relational_report_to_dict

conservative_graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
conservative_graph.add_edges_from(((0, 10), (10, 5), (1, 6)))
conservative_graph.graph["GAMMA"] = {"type": "none"}
for node in conservative_graph:
    conservative_graph.nodes[node].update(
        EPI=24 if node == 10 else -24 if node == 1 else 0,
        theta=0, nu_f=1,
    )
conservative_source = bound_relational_sine_exchange(
    conservative_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
regional_entry = certify_sine_conservative_winding_entry(
    conservative_source, cycle=(5, 6, 7, 8, 9),
    scaled_window=(Fraction(1, 4), Fraction(1, 3)),
    edge_turn_offsets=(1, 0, 0, 0, 0),
)
assert regional_entry.acquisition_certified
assert regional_entry.initial_winding == 0
assert regional_entry.certified_winding == -1
assert regional_entry.branch_margin_lower_bound > 0
regional_storage = conservative_source.regional_storage_balance(region=(5, 6, 7, 8, 9))
assert regional_storage.regional_form_storage == 0
assert regional_storage.complement_form_storage == 864
assert regional_storage.boundary_form_storage == 576
entry_export = relational_report_to_dict(regional_entry)
storage_export = relational_report_to_dict(regional_storage)
```

The acquisition interval uses `tau=t/pi`; the snapshot storage ledger uses
original structural `t` for work rates. The certificate encloses the complete
nonlinear phase evolution throughout the interval without running a trajectory.
Its acquired winding is **nonacute**, so it supplies neither equilibrium nor
indefinite maintenance. The nonzero environmental form and connections are
preparation premises. The
[proof and controls](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-winding-entry)
and [entry contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-conservative-winding-entry)
state that scope. The
[storage contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-regional-storage-balance)
distinguishes instantaneous power from integrated receiver work.

`python -m benchmarks.conservative_regional_winding` records a fresh analytic
certificate and source archive under `artifacts/research/`. It preserves
existing records and runs no trajectory or parameter search.

<a id="conservative-source-geometry"></a>

### Inspect a distributed source and certify acute entry

This standalone preparation changes the supplied support and environmental
form, retaining the conservative law. Each receiver node has one private
environmental leaf. The shared map shows which initial phase directions the
environment can supply; the entry certificate then bounds the complete
nonlinear evolution, not a linearized trajectory.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import (
    analyze_sine_conservative_source_geometry,
    certify_sine_conservative_winding_entry,
)
from tnfr.sdk import relational_report_to_dict

distributed_graph = nx.cycle_graph(5)
distributed_graph.add_edges_from((i, 5 + i) for i in range(5))
distributed_graph.graph["GAMMA"] = {"type": "none"}
for node in distributed_graph:
    distributed_graph.nodes[node].update(
        EPI=0 if node < 5 else -30 * (node - 7), theta=0, nu_f=1,
    )
distributed_source = bound_relational_sine_exchange(
    distributed_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
source_geometry = analyze_sine_conservative_source_geometry(
    distributed_source, receiver=(0, 1, 2, 3, 4),
)
assert source_geometry.relative_source_rank == 4
acute_entry = certify_sine_conservative_winding_entry(
    distributed_source, cycle=(0, 1, 2, 3, 4),
    scaled_window=(Fraction(1, 8), Fraction(2, 15)),
    edge_turn_offsets=(0, 0, 0, 0, -1),
)
assert acute_entry.acute_acquisition_certified
assert acute_entry.certified_winding == 1
assert acute_entry.acute_margin_lower_bound > Fraction(1, 5)
geometry_payload = relational_report_to_dict(source_geometry)
```

The [analytic control](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-conservative-source-geometry)
compares this preparation with a full-state reflection at equal storage,
storage partitions and conserved means. The latter cannot acquire acute
nonzero winding. The positive interval lasts `pi/120` in original structural
time; the source and contacts remain supplied, and this witness later exits
the acute sector. It is not a capture or maintenance certificate.

To create a separately named analytic record and source archive, use the same
instrument with the distributed declaration; existing files are protected:

```console
python -m benchmarks.conservative_regional_winding --declaration docs/assets/conservative_source_geometry/declaration.json --output artifacts/research/conservative_source_geometry/certificate-v1.json
```

<a id="conservative-handoff"></a>

### Distinguish acquired geometry from maintained relative motion

Continue the distributed-source example with one algebraic member of the
declared velocity-ramp class. This tests a general inequality; it does not
rerun or retune the earlier frozen response.

```python
from tnfr.physics.relational_sine_entry import assess_sine_conservative_handoff

handoff_graph = distributed_graph.copy()
for node in handoff_graph:
    handoff_graph.nodes[node]["EPI"] = 0 if node < 5 else -192 * (node - 7)
handoff_source = bound_relational_sine_exchange(
    handoff_graph, reference_model=distributed_source.reference_model,
)
handoff = assess_sine_conservative_handoff(handoff_source, cycle=range(5))
assert handoff.omega == 64
assert handoff.entry_acute_certified and handoff.entry_subbarrier_certified
assert handoff.forced_exit_certified
assert handoff.handoff_obstruction_certified
assert handoff.unavoidable_boundary_work_lower_bound > Fraction(1, 100)
handoff_payload = relational_report_to_dict(handoff)
```

The region genuinely acquires acute winding with internal storage below its
acute-face barrier. Nevertheless the retained environment imposes relative
phase velocities that force it out again. The positive work bound concerns
the first exit; it is not a prediction of net work at a later endpoint.
This does not rule out maintenance for another source. Compatible rhythms
mean control of actual relative phase evolution; they do not require all
phases to coincide or identify held capacity with angular frequency. See the
[proof](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-conservative-handoff-obstruction)
and [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-conservative-handoff).

<a id="robust-conservative-passage"></a>

### Retain a brief acquired identity despite full-state preparation errors

Continue the preceding algebraic `handoff_source`, whose nominal form ramp
has `omega=64`. The existing entry reader can admit a nonzero error at every
form and phase coordinate and expose a full-state box at the entry time:

```python
from fractions import Fraction
from tnfr.physics.relational_sine_entry import certify_sine_conservative_winding_entry

passage = certify_sine_conservative_winding_entry(
    handoff_source,
    cycle=range(5),
    scaled_window=(Fraction(11, 512), Fraction(3, 128)),
    edge_turn_offsets=(0, 0, 0, 0, -1),
    source_error_bound=Fraction(1, 4096),
)
assert passage.source_initial_zero_winding_certified
assert passage.acute_acquisition_certified
assert passage.entry_box_contains_source_flow
assert passage.entry_box_acute_retention_certified
assert passage.scaled_retention_duration == Fraction(1, 512)
assert len(passage.entry_form_bounds) == len(passage.entry_phase_bounds) == 10
assert passage.entry_form_radius == Fraction(89, 4096)
assert passage.entry_phase_radius == Fraction(751, 1048576)
assert passage.acute_margin_lower_bound > Fraction(1, 16)
assert passage.entry_box_acute_margin_lower_bound > Fraction(1, 16)
passage_evidence = relational_report_to_dict(passage)
```

All initial states in the declared source box begin with zero winding.
Their actual endpoints lie inside the reported entry box, and **every**
state in that independent box retains acute winding `+1` for the next
`1/512` of scaled time. The environment continues to evolve. The box centers
are bounding references, not replacements for actual trajectories.

This is a robust short passage, consistent with the separate forced-exit
result. Its duration is not extended retention or an entry into the later
moving-family example. `initial_total_storage` describes the nominal center;
`initial_storage_bounds` instead covers the initial source uncertainty. That
source budget does not cover every independent point of the later box.
The [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-conservative-winding-entry)
and [proof](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#robust-conservative-passage)
keep those distinctions explicit. No trajectory is run by this example.

<a id="phase-offset-partition"></a>

### Describe a region's compatible internal and collective motion

This standalone example uses a supplied C5 receiver and one private leaf
per receiver node. It asks whether fixed internal phase offsets can coexist
with moving collective form and phase under the same complete conservative
law. The captured zero state supplies the law and support; it is not the
twisted family and is not evolved into it by this calculation.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import assess_sine_phase_offset_partition

partition_graph = nx.cycle_graph(5)
partition_graph.add_edges_from((i, 5 + i) for i in range(5))
partition_graph.graph["GAMMA"] = {"type": "none"}
for node in partition_graph:
    partition_graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
partition_source = bound_relational_sine_exchange(
    partition_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
offset_turns = tuple(Fraction(i, 5) for i in range(5)) * 2
partition = assess_sine_phase_offset_partition(
    partition_source,
    blocks=(tuple(range(5)), tuple(range(5, 10))),
    phase_offset_turns=offset_turns,
)
assert partition.status == "certified"
assert partition.invariance_certified
collective_state = partition.evaluate(
    collective_form=(1, 0),
    collective_phase_turns=(0, Fraction(1, 8)),
)
assert collective_state.block_phase_rates == (Fraction(1, 3), Fraction(-1))
print(collective_state.block_form_rates)
```

The receiver's constituents retain their phase differences while the two
blocks have their own compensated common motion. These angular rates use
`tau=t/pi`; the declared phase inputs are turns, and held capacity is not
an angular frequency. The returned form rates enclose the existing sine
response at this independently supplied state. No time step is executed.

Along the exact family the receiver's internal storage stays constant and
its net internal regional power is zero, although collective coordinates
and contact dynamics remain active. This is a compatible pattern family,
not autonomous formation, attraction or a robustness certificate. A
phase-flat source outside the family cannot enter it exactly at finite time
under the unchanged smooth law; a finite-width handoff needs its own proof.
For another partition, `unavailable` means the equality checks did not
decide the criterion, whereas `excluded` proves at least one required
condition fails. See the
[contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-phase-offset-partition)
and [theory](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-phase-offset-partition).

<a id="moving-pattern-window"></a>

### Retain compatible motion with errors in every fine node

Continue the [phase-offset partition example](#phase-offset-partition).
Here the reference has nonzero collective exchange and zero conserved
weighted origins. Independent errors affect all receiver and environmental
forms and phases. The reader proves a finite window around this supplied
reference; it does not evolve the earlier zero-state source into the family.

```python
from tnfr.physics.relational_sine_partition import assess_sine_moving_pattern_window

moving_reference = partition.evaluate(
    collective_form=(Fraction(1, 20), Fraction(-3, 20)),
    collective_phase_turns=(Fraction(-2, 5), Fraction(-2, 5)),
)
window_parameters = dict(
    form_error_bounds=(Fraction(1, 1000),) * 10,
    phase_error_bounds=(Fraction(1, 1000),) * 10,
    scaled_horizon=5,
    receiver_phase_radius=Fraction(1, 10),
    contact_phase_radius=Fraction(1, 10),
    reference_contact_bound=Fraction(1, 4),
)
moving_window = assess_sine_moving_pattern_window(
    moving_reference, **window_parameters,
)
assert moving_window.whole_window_retention_certified
assert moving_window.phase_flat_acquisition_status == "not_excluded"
assert moving_window.retention_margin_lower_bound > 0

zero_pulse_reference = partition.evaluate(
    collective_form=(0, 0),
    collective_phase_turns=(Fraction(-2, 5), Fraction(-2, 5)),
)
zero_pulse_window = assess_sine_moving_pattern_window(
    zero_pulse_reference, **window_parameters,
)
assert zero_pulse_window.whole_window_retention_certified
assert zero_pulse_window.phase_flat_acquisition_status == "excluded"
assert zero_pulse_window.initial_full_storage_bounds.hi < Fraction(7, 2)
```

The horizon `5` is in `tau`, or `5pi` in the original structural clock.
Phase uncertainty and all three phase bounds are radians, whereas the
reference phase coordinates above are turns. The certificate retains acute
receiver winding, internal form/phase storage and full contact error bounds
while the environment moves. It also keeps uncertainty in the conserved
origins; relative edge errors alone would discard those coordinates.

Both examples establish maintenance of a prepared neighborhood. The
zero-pulse neighborhood has too little total storage to have been acquired
from phase-flat preparation under this unchanged law. The moving example
avoids that particular obstruction; `not_excluded` does not prove formation.
Its small full-storage interval also excludes a handoff from the earlier
large-energy ramp sources. Their environmental state cannot be discarded
to make the budgets agree.

The same estimate holds for the preceding finite window, so the central
state is not the first appearance of the identity. An acquisition claim
needs an earlier identity-absent preparation, compatible conserved storage
and independent evidence that its actual full state reaches this
neighborhood. See the [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-moving-pattern-window)
and [theory](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-moving-pattern-window).

<a id="collective-pulse-balance"></a>

### Observe transfer between collective contact motion and internal structure

This standalone example starts with flat phase and zero receiver winding.
The supplied form dipole breaks the exact uniform contact-pulse family.
The reader examines its actual full-state response at the initial instant;
it does not substitute a prepared invariant pattern or run a trajectory.

```python
from dataclasses import replace
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import observe_sine_collective_pulse

pulse_graph = nx.cycle_graph(5)
pulse_graph.add_edges_from((i, 5 + i) for i in range(5))
pulse_graph.graph["GAMMA"] = {"type": "none"}
u, epsilon = Fraction(119, 100), Fraction(1, 20)
for node in pulse_graph:
    form = u / 4 if node < 5 else -3 * u / 4
    if node == 0:
        form += epsilon
    elif node == 1:
        form -= epsilon
    pulse_graph.nodes[node].update(EPI=form, theta=0, nu_f=1)
pulse_anchor = bound_relational_sine_exchange(
    pulse_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
pulse_source = replace(
    pulse_anchor,
    epi=tuple(pulse_graph.nodes[node]["EPI"] for node in pulse_anchor.nodes),
)
pulse_balance = observe_sine_collective_pulse(
    pulse_source, cycle=range(5), contact_turn_offsets=(0,) * 5,
)
assert pulse_balance.form_gap == u
assert pulse_balance.flat_phase_jet_available
assert pulse_balance.first_three_energy_derivatives == (0, 0, 0)
assert pulse_balance.contact_rate_variance == Fraction(1, 180)
assert pulse_balance.contact_rate_third_central_moment == 0
assert pulse_balance.fourth_energy_derivative == 4 * u * u / 135
assert pulse_balance.full_storage_bounds.contains(Fraction(14201, 4000))
```

The detached replacement explicitly supplies the rational preparation in
the anchor's node order, avoiding graph capture's floating materialization.
The observer re-admits those primitives and recomputes all consumed rates
and storage; cached derived fields in the anchor are not used as evidence.

The positive fourth derivative shows the first nonzero transfer goes toward
the collective contact pulse. It does not predict later receiver organization.
The collective pulse is not separately conserved away from the exact family:
its feedback depends on the five actual contact-phase deviations. The
reported remainder `H-5h` is signed unless extra convexity is established.

The integer offsets declare continuous contact lifts; independently wrapping
each phase difference before averaging would change this observation. A
snapshot cannot recover a lost branch history. All reported derivatives use
`tau=t/pi`, and no numerical evaluation horizon follows from the local jet.

Although this source's total storage exceeds the necessary phase-flat path
barrier, it is below the sharper lower bound of the preceding moving-pattern
example's fixed error box. That particular handoff is excluded. Testing
finite acute identity and certifying a subsequent full-state maintenance
window remain separate tasks, each retaining the source's actual conserved
budget. See the [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-collective-pulse-transfer)
and [derivation](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-collective-pulse-transfer).

<a id="relative-phase-feedback"></a>

### Distinguish momentary phase compensation from maintained geometry

The same observer reports the internal and contact contributions to receiver
relative phase velocity, together with its acceleration and third derivative.
This exact rational control reuses `pulse_anchor` from the preceding example.
The environment cancels the receiver's initial relative velocity and
acceleration, but the full law produces a nonzero third derivative.

```python
compensated_forms = (1, -1, 0, 0, 0, 4, -4, 1, 0, -1)
compensated_source = replace(
    pulse_anchor,
    epi=tuple(Fraction(compensated_forms[node]) for node in pulse_anchor.nodes),
    phase=(Fraction(0),) * len(pulse_anchor.nodes),
)
compensated = observe_sine_collective_pulse(
    compensated_source, cycle=range(5), contact_turn_offsets=(0,) * 5,
)
assert compensated.receiver_relative_phase_rates == (0,) * 5
assert all(
    internal + contact == 0
    for internal, contact in zip(
        compensated.receiver_internal_phase_rates,
        compensated.receiver_contact_phase_rates,
    )
)
assert all(
    bound.contains(0)
    for bound in compensated.receiver_relative_phase_acceleration_bounds
)
expected_jerk = (Fraction(22, 9), -Fraction(22, 9), 1, 0, -1)
assert all(
    bound.contains(value)
    for bound, value in zip(
        compensated.receiver_relative_phase_jerk_bounds, expected_jerk,
    )
)
assert compensated.receiver_relative_phase_jerk_bounds[0].lo > 0
```

The [algebraic counterexample](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-relative-phase-feedback)
proves the zero derivatives and nonzero cubic drift; interval overlap alone
would not prove either equality. This flat-phase example tests feedback, not
formation. The theorem also gives an exactly twisted symbolic counterpart,
without treating a floating approximation to pi as exact geometry.
The unchanged SDK report export includes the new fields. Neither this local
calculation nor exact rigidity classification certifies the extended
formation-to-retention window; that requires control over the whole interval.

<a id="conservative-source-admission"></a>

### Exclude unsuitable conservative sources before running a trajectory

These two readers assess different necessary conditions. The standalone
example keeps contact phases nonuniform and retains every private leaf.
Exact rational values are explicitly supplied after graph capture.

```python
from dataclasses import replace
from fractions import Fraction as Q
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import assess_sine_contact_averaging
from tnfr.physics.relational_sine_regional import assess_sine_cycle_barrier
from tnfr.sdk import relational_report_to_dict

admission_graph = nx.cycle_graph(5)
admission_graph.add_edges_from((i, i + 5) for i in range(5))
admission_graph.graph["GAMMA"] = {"type": "none"}
for node in admission_graph:
    admission_graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
admission_anchor = bound_relational_sine_exchange(
    admission_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
contact_phases = tuple(Q(1, 4) + Q(sign, 10) for sign in (1, 1, -1, -1, 0))
low_source = replace(
    admission_anchor,
    epi=(Q(1, 4),) * 5 + (Q(-3, 4),) * 5,
    phase=(Q(0),) * 5 + contact_phases,
)
barrier = assess_sine_cycle_barrier(low_source, cycle=range(5))
assert barrier.full_storage_bounds.hi < Q(7, 2)
assert all(sector.acute_acquisition_excluded for sector in barrier.sectors)

rapid_source = replace(low_source, epi=(Q(5),) * 5 + (Q(-15),) * 5)
averaging = assess_sine_contact_averaging(
    rapid_source, cycle=range(5), scaled_horizon=1,
)
assert averaging.initial_full_storage_bounds.lo >= 1000
assert averaging.whole_window_acute_winding_excluded
admission_evidence = relational_report_to_dict(averaging)
```

The first exclusion follows from conservation for all times; the second
covers only `0 <= tau <= 1` (`0 <= t <= pi`). Increasing a uniform form gap
does not by itself create useful directed transfer. Neither reader evolves
the graph, chooses a preparation or rules out every possible coherent
pattern. Their [contracts](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-cycle-sector-barrier)
retain unavailable bounds and distinguish wider-sector protection from
strict acute retention.

<a id="reversible-preparation"></a>

### Connect an absent identity to a retained region

Use `assess_sine_reversible_preparation` when an admitted conservative
checkpoint already has an energy-speed retention window and a validated
complete-law forecast starts from its reversed forms. It consumes the
existing `SineForecast` and `SineExchangeComparison`; it does not run another
solver. Supply the ordered receiver cycle, exact positive source and target
radii, and scaled retention duration. The source center is constructed from
each endpoint, with its enclosure error retained in the forward-inclusion
bound. A reversed midpoint by itself does not certify a preparation.

The [shared instrument](../../../benchmarks/conservative_regional_organization.py)
supports this workflow through a declaration whose `evaluation_kind` is
`reversible_preparation`. Freeze the declaration and source archive with
`--prepare` before the single evaluation. The declaration owns the fixed
grid, numerical budget, target, source-radius rule and first-success rule;
the [research plan](../../../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns permission to schedule a new candidate. Reading a saved response does
not authorize rerunning or tuning it.

Inspect `outcome`, `horizon_complete`, `selected_step_index` and all `steps`.
Only `certified` establishes the conditional source-to-retention inclusion.
`unavailable` and `no_certificate_on_declared_grid` preserve different
coverage limits. These are mathematical preparation certificates, not
measurements or autonomous formation laws. The
[contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-reversible-preparation)
and [proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-reversible-preparation)
own the full hypotheses and interpretation.

<a id="energy-speed-retention"></a>

### Certify a whole retention window without a moving reference

This standalone example supplies a target with independent uncertainty in
all twenty form/phase coordinates. It preserves the complete conservative
C5/private-leaf law. The rational phases below are actual declared centers,
not an assertion that a rounded angle is the exact uniform twist.

```python
from dataclasses import replace
from fractions import Fraction as Q
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_regional import assess_sine_cycle_retention
from tnfr.sdk import relational_report_to_dict

target_graph = nx.cycle_graph(5)
target_graph.add_edges_from((i, i + 5) for i in range(5))
target_graph.graph["GAMMA"] = {"type": "none"}
for node in target_graph:
    target_graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
target_anchor = bound_relational_sine_exchange(
    target_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
u, kappa = Q(7, 50), Q(1, 128)
q = tuple(kappa * value for value in (1, -1, 0, 0, 0))
lq = tuple(kappa * value for value in (3, -3, 1, 0, -1))
forms = tuple(u / 4 + value for value in q) + tuple(
    -3 * u / 4 + value + laplacian for value, laplacian in zip(q, lq)
)
phases = tuple((i - 2) * Q(1256637, 10**6) for i in range(5)) * 2
target = replace(
    target_anchor,
    epi=tuple(forms[node] for node in target_anchor.nodes),
    phase=tuple(phases[node] for node in target_anchor.nodes),
)
retention = assess_sine_cycle_retention(
    target, cycle=range(5), scaled_duration=1,
    source_error_bound=Q(1, 4096),
)
assert retention.initial_winding == 1
assert retention.full_storage_bounds.lo > Q(7, 2)
assert retention.whole_window_retention_certified
assert retention.retention_margin_lower_bound > Q(1, 100)
retention_evidence = relational_report_to_dict(retention)
```

Every state in this exact-radius target retains acute unit winding for
`-1 <= tau <= 1`. The environment remains live and the center has nonuniform
form; no rigid reference is substituted for its trajectory. This does not
show how an earlier zero-winding state reaches the target. Its full energy
avoids one necessary exclusion, while directed entry remains the research
obligation. The [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-cycle-retention)
distinguishes whole-window guarantees from unavailable bounds.

<a id="conservative-regional-phase-transport"></a>

### Inspect propagation into the receiver interior

Continue the [eleven-node winding example](#conservative-regional-winding)
with its `conservative_source`. The contrast below observes the
phase separation between non-port nodes `7` and `9`. Source-node order is
explicit; the coefficients select an observation and do not alter the law.

```python
from tnfr.physics.relational_sine_entry import analyze_sine_conservative_phase_transport

interior_contrast = tuple(
    int(node == 9) - int(node == 7) for node in conservative_source.nodes
)
phase_transport = analyze_sine_conservative_phase_transport(
    conservative_source, contrasts=(interior_contrast,),
)
assert phase_transport.contrast_initial_velocity == (Fraction(0),)
assert phase_transport.contrast_initial_jerk == (Fraction(-80, 3),)
assert phase_transport.contrast_acceleration_bounds == (Fraction(3),)
assert phase_transport.contrast_quadratic_remainder_coefficients == (Fraction(3, 2),)
transport_export = relational_report_to_dict(phase_transport)
```

The interior starts without relative phase velocity but has a nonzero third
phase derivative. This is a local propagation mechanism. The separate global
bound gives `abs(theta_9-theta_7)<=3*tau^2/2` for this preparation; it does not
give a nonzero lower bound or prove later organization. The
[proof](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-phase-transport)
explains why a fully acute unit winding requires `tau^2>pi/3` on this support.
The [regional work criterion](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-work-retention)
adds the form-storage and future boundary-work obligations needed for its
stronger finite-retention conclusion. The early winding witness does not
already satisfy them. See the
[API contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-conservative-phase-transport).

<a id="conservative-regional-forecast"></a>

### Observe a continuous conservative regional forecast

This inexpensive stationary control illustrates the shared solver and observer.
It is independent of the reserved environmental formation experiment.

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_forecast import bound_sine_flow
from tnfr.physics.relational_sine_regional import assess_sine_regional_organization
from tnfr.sdk import relational_report_to_dict

cycle_neighbors = tuple(((i - 1) % 5, (i + 1) % 5) for i in range(5))
stationary_forecast = bound_sine_flow(
    (0,) * 10 + (1,), neighbors=cycle_neighbors,
    visible_capacity=(1,) * 4,
    model=RelationalExchangeModel(1, epi_weight=0, phase_domain="regular"),
    observation_time=0, end_time=Fraction(1, 16),
    time_step=Fraction(1, 16), order=2,
)
regional_observation = assess_sine_regional_organization(
    stationary_forecast, cycle_indices=(0, 1, 2, 3, 4),
    minimum_duration=Fraction(1, 32), acute_margin=Fraction(1, 16),
)
assert regional_observation.horizon_complete
assert regional_observation.outcome == "acute_winding_excluded_on_horizon"
assert regional_observation.endpoint_integrated_regional_work.contains(0)
regional_export = relational_report_to_dict(regional_observation)
```

The observer uses whole-time tubes, original structural `t` and the complete
network rates. The stationary state has zero winding throughout; this control
does not exclude organization from another preparation. A failed enclosure
remains unresolved rather than becoming a negative physical result. See the
[contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-regional-organization).

Reuse that same forecast to separate internal conversion from boundary inputs,
without another solver call:

```python
from tnfr.physics.relational_sine_regional import assess_sine_regional_channels

channel_history = assess_sine_regional_channels(
    stationary_forecast, cycle_indices=(0, 1, 2, 3, 4),
)
assert channel_history.horizon_complete
assert channel_history.initial_phase_flat_certified
assert channel_history.acute_entry_excluded_on_validated_prefix
assert channel_history.cumulative_internal_conversion.contains(0)
channel_export = relational_report_to_dict(channel_history)
```

For a nonstationary source, positive internal conversion transfers phase
storage into form. Boundary inputs to the separate stores differ from the
conjugate work components of the total regional ledger. The running `7/2`
barrier is necessary for acute unit-winding entry from flat receiver phase;
it is not sufficient entry or a ban on every nonacute winding. See the
[channel contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-regional-channel-history).

The [frozen environmental control](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-organization-control)
has a supplied nonuniform environment and no prescribed future input. Its
preparation and evaluation are already closed; the
[benchmark lifecycle](../../../benchmarks/README.md#freeze-before-evaluating-a-reserved-response)
owns any separately declared new evaluation. To analyze its channel transfers under
current analysis code, retain the original response and use a separate output:

```sh
python -m benchmarks.conservative_regional_organization --analyze-channels docs/assets/conservative_regional_organization/response-v1.json --output artifacts/research/conservative_regional_organization/channel-analysis-v1.json
```

This reads and verifies saved evidence without replaying the forecast. It
records the original file hashes, a separate analysis source archive and
retrospective results. It does not alter the original prediction or verdict.
