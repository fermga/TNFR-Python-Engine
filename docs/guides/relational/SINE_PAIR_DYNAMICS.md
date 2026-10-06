# Retained sine pair state, identification and pulses

Sufficient unordered/mixed pair state and cancellation inversion, capacities and support symmetry, pairing observations and windows, emission/transfer and doubled-C5 pulse response.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

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
certificate. The [contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-scale)
and [proof](../../../theory/nodal/SINE_PAIR_STATE.md#sine-replica-inheritance)
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
not new interaction parameters. The [contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-persistence)
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
[form-balance selection theorem](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#closed-form-balance-and-source-selection)
uses this distinction: its conservation requirement is an additional structural
premise, checked separately from storage protection. The
[contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-mobility-geometry)
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
[proof](../../../theory/nodal/SINE_PAIR_STATE.md#sine-replica-capacity-asymmetry).
It is not complete equilibrium or a claim that internal motion never resumes.
The report retains the capacity-state correlations needed to predict the
collective rates. The [contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-capacity)
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
[contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-state-pairing) for explicit
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
[transition contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pairing-transition).

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
[contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pairing-mobility) distinguishes
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
The [window contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pairing-window)
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
See the [projection contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-joint-pairing-projection)
for the source-rate calculation, all-coordinate error bounds and availability.

The [critical-boundary theorem](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-joint-boundary-acquisition)
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
[support-symmetry contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-support-symmetry).

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
[mixed-state contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-mixed-pair-state)
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
[contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-emission) for clipping,
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
The [derivation](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-autonomous-regional-transfer)
also gives the phase back-reaction and finite transfer amplitude on this
synchronized submanifold. Neither the observation nor the hypothetical action
advances this graph; [admission and export](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-regional-transfer)
retain the separate model, full source and unresolved endpoint obligations.

Regional currents also remain defined when a pair's mean phasor cancels and
its circular midpoint is unavailable. The
[antipodal-pair result](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-zero-resultant-restoration)
uses internal form contrast to restore a current through an isolated zero.
A zero instantaneous current does not imply a disconnected phase channel
or a finite waiting interval. Keep the full comparison and cut reader; the
midpoint-based replica report has a narrower chart. Floating-point `pi`
does not encode the exact antipodal preparation used in that proof.
The [global pair example](#sine-global-pair-state) instead supplies exact
Cartesian phasors on the theorem's fixed doubled-C5 support.

<a id="sine-moving-pattern-interface"></a>

### Retain a moving pattern's interaction state

Use the [regional-transfer workflow](#distinguish-autonomous-transfer-from-emission)
to observe present exchange and the
[replica-state workflow](#observe-a-larger-organization-without-removing-its-constituents)
to retain the internal coordinates that govern its evolution. These are
different questions even when they use the same complete law.

On the doubled C5, `boundary_edge_indices` identifies the region's inside and
outside fine nodes; `boundary_form_currents` follows that same order. Grouping
those existing entries by the outside pair exposes its individual contribution.
For unit capacities, each node has degree four, so divide each block's weighted
sum by eight to obtain its contribution to the receiving pair's mean form rate.
Do not discard the individual contributions merely because their sum is zero.
In the [established pulse family](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-moving-pattern-interface),
the means stay fixed while opposite currents vary with the internal motion.

For future response, retain the admitted replica's means, internal `R,U,Q`,
neighbors and relative origins. If attachments break the swap symmetry, use
the [mixed-state workflow](#retain-state-associated-with-asymmetric-attachments).
At exact phase cancellation, preserve the fine state and use the cut reader;
the following global representation is another option under its unit-law and
doubled-C5 premises. Do not create an angle or a zero internal state to make
the midpoint API accept it. The [contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-moving-pattern-interface)
keeps current normalization, chart refusal and environmental information explicit.

<a id="sine-global-pair-state"></a>

### Retain global pair state through phase cancellation

This standalone example uses the fixed doubled C5: ten fine nodes in
consecutive pairs, with all four edges between adjacent pairs. The complete
law is conservative normalized-sine exchange with unit held capacities,
`w=beta=1`, no forcing or events, and original structural time `t`.
The inputs below are exact Cartesian phase phasors, not rounded radian angles.

```python
from fractions import Fraction
from tnfr.physics.relational_sine_scale import (
    derive_sine_global_pair_state,
    evaluate_sine_global_pair_state,
)
from tnfr.sdk import relational_report_to_dict

# Surrounding pairs are aligned. Pair (0,1) is exactly antipodal in both cases.
forms = (0,) * 10
surroundings = ((1, 0),) * 8
real_pair = derive_sine_global_pair_state(
    forms, ((1, 0), (-1, 0)) + surroundings,
)
imaginary_pair = derive_sine_global_pair_state(
    forms, ((0, 1), (0, -1)) + surroundings,
)
for field in (
    "form_means", "resultants", "internal_form_squared", "form_phase_moments",
):
    assert getattr(real_pair, field) == getattr(imaginary_pair, field)
assert real_pair.phase_products[0] == (-1, 0)
assert imaginary_pair.phase_products[0] == (1, 0)
assert real_pair.form_phase_moment_rate_pi_numerators[0] == (0, 0)
assert imaginary_pair.form_phase_moment_rate_pi_numerators[0] == (0, -1)
assert real_pair.storage == imaginary_pair.storage == 8
assert imaginary_pair.full_storage_rate_pi_numerator == 0
payload = relational_report_to_dict(imaginary_pair)
assert payload["report"]["pair_strata"][0] == "antipodal_phase"

# Realizable rational invariants can have irrational constituent coordinates.
state = evaluate_sine_global_pair_state(
    form_means=(0,) * 5,
    resultants=((Fraction(1, 2), 0),) * 5,
    phase_products=((1, 0),) * 5,
    internal_form_squared=(Fraction(1, 3),) * 5,
    form_phase_moments=((0, Fraction(1, 2)),) * 5,
)
assert state.resultant_rate_pi_numerators == ((Fraction(-1, 2), 0),) * 5
assert state.full_storage_rate_pi_numerator == 0
```

In the first comparison, `X=Z=U=W=0` at the selected pair, yet its phase
product `P` distinguishes `pi*W_dot=0` from `-i`. Equal instantaneous
currents therefore do not remove the internal orientation needed for future
response. In the second example, a representative has forms
`+/-1/sqrt(3)` and phasors `1/2 +/- i*sqrt(3)/2`; the evaluator requires
neither square-root materialization nor a midpoint angle.

Coordinate rate numerators multiply derivatives by pi;
`block_current_rate_pi_squared_numerators` instead multiplies mean-form
current derivatives by pi squared. All are instantaneous observations in
the original clock. The functions perform no integration or graph mutation,
and a numerical update of these constrained coordinates is not automatically
realizable. See [admission and export](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-global-pair-state)
and [Section 29's derivation](../../../theory/nodal/SINE_PAIR_STATE.md#sine-global-pair-state).

<a id="sine-pair-cancellation-observability"></a>
### Recover a cancelled pair's orientation from exact derivative evidence

Keep the same fixed doubled C5 and conservative unit-coefficient law. This
inverse accepts collective form/phase means and **supplied exact derivatives**
of the selected mean phasor. It does not read the hidden pair to manufacture
its own observation. The illustrative evidence below is declared in
`tau=t/pi`, not estimated from sampled or laboratory data.

```python
from tnfr.physics.relational_sine_scale import observe_sine_pair_cancellation
from tnfr.sdk import relational_report_to_dict

inputs = dict(
    form_means=(0,) * 5,
    resultants=((0, 0),) + ((1, 0),) * 4,
    pair_index=0,
    resultant_first_tau_derivative=(0, 0),
)
recovered = observe_sine_pair_cancellation(
    **inputs, resultant_second_tau_derivative=(1, 0),
)
assert recovered.phase_product == (1, 0)
assert recovered.internal_form_squared == 0
assert recovered.form_phase_moment == (0, 0)
assert recovered.phase_product_reconstruction_order == 2

# With a nonzero neighborhood resultant, zero acceleration also gives information.
aligned = observe_sine_pair_cancellation(
    **inputs, resultant_second_tau_derivative=(0, 0),
)
assert aligned.phase_product == (-1, 0)

# A zero neighborhood and zero first/second derivatives leave orientation unresolved.
unresolved = observe_sine_pair_cancellation(
    **{**inputs, "resultants": ((0, 0),) * 5},
    resultant_second_tau_derivative=(0, 0),
)
assert unresolved.phase_product is None
assert unresolved.phase_product_reconstruction_order is None
payload = relational_report_to_dict(unresolved)
assert payload["report"]["phase_product"] is None
```

The first two cases distinguish antipodal orientations even though the
selected mean phasor and its first derivative both vanish. The final case
requires higher-order neighborhood evidence or a separately justified
persistence argument; it does not prove the pair will remain invisible.
See the [admission contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-cancellation-observability)
and [conditional reconstruction theorem](../../../theory/nodal/SINE_PAIR_STATE.md#sine-pair-cancellation-observability).

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
See the [admission contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-joint-pairing-window)
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
Read the [contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-equilibria)
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
[contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-internal-pulse)
separates exact preparation, pulse, fine-edge acuteness and unavailable
captured-graph membership. General disturbances need a separate stability
analysis; this example does not establish spontaneous preparation or attraction.
The variation report supplies three real block types, covering all twenty
fine-state perturbations. Its matrices describe instantaneous changes, not a
computed return after one period. The
[variation contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-pulse-variation)
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
[splitting contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-pulse-splitting)
for the analytic coefficients, remainder scope and exact export.

<a id="sine-replica-pulse-work-response"></a>
### Compare matched work responses along a moving pulse

Keep the existing pulse law and declare the background before reading its
response. This example uses two distributed form perturbations on the same
nodes, with their conjugate work outputs:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_scale import assess_sine_replica_pulse_work_response
from tnfr.sdk import relational_report_to_dict

response = assess_sine_replica_pulse_work_response(
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
    form_half_difference=Fraction(1, 32),
    phase_half_difference=Fraction(1, 32),
    capacity=1,
    scaled_duration=Fraction(1, 16),
    order=10,
)
assert response.clock == "tau=t/pi"
assert response.status == "certified_finite_response"
assert response.directional_response_certified
assert response.antisymmetric_response_bounds.lo > Fraction(6, 10**10)
response_evidence = relational_report_to_dict(response)
```

The two probe roles have different responses at this declared delay. The
calculation follows the moving background together with both perturbation
columns; freezing its initial coefficients would remove the effect. It is
linear response to an infinitesimal perturbation, with a validated finite-time
enclosure. It does not certify a finite kick or magnetic Hall transport. Read
the [contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-pulse-work-response)
for port normalization, unavailable results and reversal controls.

For a controlled nonzero kick, the fixed-template reader evolves all ten
fine nodes under the complete nonlinear law. It keeps the preceding source
and horizon and admits a sufficient positive amplitude interval:

```python
from fractions import Fraction
from tnfr.physics.relational_sine_scale import assess_sine_replica_pulse_finite_work_response
from tnfr.sdk import relational_report_to_dict

finite = assess_sine_replica_pulse_finite_work_response(
    probe_amplitude=Fraction(1, 2**20), order=10,
)
assert finite.status == "certified_finite_response"
assert finite.analytic_directional_response_certified
assert finite.numerical_directional_response_certified
assert len(finite.steps) == 4
assert finite.analytic_contrast_lower_bound > Fraction(6, 10**10)
actual = finite.antisymmetric_response_bounds
predicted = finite.predicted_antisymmetric_response_bounds
assert predicted.lo <= actual.lo <= actual.hi <= predicted.hi
finite_evidence = relational_report_to_dict(finite)
```

The positive and negative kicks share the same initial phases. Their centered
difference removes the unperturbed background, and a uniform nonlinear error
bound connects the result to the tangent prediction. The numerical and
analytic certificates remain separate if an enclosure is unavailable.
The [finite-probe contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-pulse-finite-work-response)
records the fixed preparation and work readout. This small structural signal
has no admitted physical measurement or noise scale.

### Screen an observed stiffness family before fitting a response

The selected pulse imposes a necessary relation between the trace and
determinant of a mass-normalized two-mode stiffness. Three separated
observations can exclude a candidate even when its clock scale is unknown.
For example the simple family `J(rho)=rho*I` gives:

```python
from tnfr.physics.relational_sine_scale import assess_sine_replica_stiffness_trace_curve
from tnfr.sdk import relational_report_to_dict

screen = assess_sine_replica_stiffness_trace_curve(
    trace_bounds=(2, 4, 6), determinant_bounds=(1, 4, 9),
)
assert screen.trace_separation_certified
assert screen.template_curve_status == "excluded"
assert screen.affine_family_status == "not_excluded"
screen_evidence = relational_report_to_dict(screen)
```

This is an exact model counterexample, not laboratory data. Supply uncertainty
intervals when observations are uncertain; overlapping traces remain unresolved.
Use `M^-1 K` for a nonidentity kinetic mass. A `not_excluded` result does not
establish a realization, autonomous pulse timing or physical correspondence.
The [contract](../../contracts/relational/SINE_PAIR_DYNAMICS.md#sine-replica-stiffness-trace-curve)
defines both separate exclusion tests.
