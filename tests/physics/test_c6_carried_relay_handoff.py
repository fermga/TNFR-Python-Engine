"""A local corrected coordinate survives changes outside its C6 dependency."""

from dataclasses import replace
from fractions import Fraction as F
import math

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import NodalRemainderState, _validate_nodal_remainder_state
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile
from tnfr.physics.c6_carried_relay import (
    derive_c6_carried_relay, observe_c6_carried_relay_exit,
    derive_c6_carried_local_relay_budget, observe_c6_carried_local_relay_exit,
)
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice, _observe_rebuilt_c6_pressure_lattice,
)

PHASE = tuple(map(float.fromhex, (
    "0x1.0b8fb3e3956cbp-55", "0x1.0c152382d7365p+0", "0x1.0c152382d7365p+1",
    "0x1.921fb54442d18p+1", "0x1.0c152382d7365p+2", "0x1.4f1a6c638d03fp+2",
)))
DELTA, M, GRID = F(1, 2**54), F(1, 2**113), F(1, 2**3222)


@pytest.fixture(scope="module")
def previous():
    # Only B46's proved seven-step conditional prefix is replayed.
    terms = ((-35698968670925291, 110), (229688483940001907, 113),
             (-203177807149452541, 112), (-17846977754793875, 109),
             (-139871603228525, 101), (41950776505050555, 111))
    epi = tuple(float(F(1, 2) + k * DELTA) for k in (-3, 0, 2, 0, 8, -5))
    state = NodalRemainderState(epi, tuple(F(a, 2**b) for a, b in terms), .375, .625)
    lattice = derive_c6_pressure_lattice(
        phase=PHASE, epi_weight=CHANNEL_WEIGHT_SECONDARY, phase_weight=CHANNEL_WEIGHT_PRIMARY,
    )
    closure = derive_c6_carried_closure(derive_c6_carried_profile(lattice), state=state, timestep=.0625)
    return observe_c6_carried_relay_exit(derive_c6_carried_relay(closure), step_budget=10)


@pytest.fixture(scope="module")
def budget(previous):
    return derive_c6_carried_local_relay_budget(previous.relay, state=previous.endpoint)


@pytest.fixture(scope="module")
def passage(budget):
    return observe_c6_carried_local_relay_exit(budget, step_budget=198)


def _state_with_exact(state, changes):
    exact = list(_validate_nodal_remainder_state(state))
    for node, value in changes.items():
        exact[node] = value
    epi = tuple(map(float, exact))
    result = replace(state, epi=epi, remainder=tuple(value - F(v) for value, v in zip(exact, epi)))
    _validate_nodal_remainder_state(result)
    return result


def test_handoff_preserves_exact_endpoint_and_retains_only_local_dependencies(previous, budget):
    assert budget.closure.base_tube.state == previous.endpoint
    assert budget.relay_node == 0 and budget.budget_node == 1
    assert budget.held_nodes == (1, 2, 5) and budget.free_nodes == (3, 4)
    assert budget.dependency_nodes == ((0, 1, 5), (0, 1, 2))
    assert budget.deadline == 198 == previous.relay.exit_deadlines[1] - previous.exit_step
    assert budget.band_covers_deadline
    axis = previous.relay.axes[0]
    assert (budget.decrement, budget.increment, budget.length) == (axis.decrement, axis.increment, axis.length)
    assert budget.budget_correction == axis.correction[1]
    assert budget.budget_drift == previous.relay.drift[1] < 0
    assert budget.initial_offset == previous.endpoint.exact_epi[0] - budget.facet


def test_full_refresh_crosses_free_cells_until_first_held_exit(previous, budget, passage):
    assert passage.exit_step == 197 < budget.deadline and passage.exiting_nodes == (1,)
    assert len(passage.steps) == 197 and len(passage.points) == 198
    assert passage.steps[0].before == previous.endpoint
    assert passage.endpoint == passage.steps[-1].after and not passage.band_failure
    assert passage.endpoint.epi[1] == math.nextafter(previous.endpoint.epi[1], -math.inf)
    assert tuple(sum(step.before.epi[i] != step.after.epi[i] for step in passage.steps)
                 for i in range(6)) == (182, 1, 0, 118, 23, 0)
    # The first continuation enters the old four-cell family again, while
    # later free-node changes still remain inside the local theorem.
    assert passage.steps[0].after.epi == previous.relay.visible_rows[3]
    reference = budget.closure.base_tube.contraction.profile.lattice
    lower_count = 0
    for ordinal, step in enumerate(passage.steps):
        assert step.before == (previous.endpoint if ordinal == 0 else passage.steps[ordinal - 1].after)
        assert all(step.before.epi[node] == previous.endpoint.epi[node] for node in budget.held_nodes)
        assert step.pressure == _observe_rebuilt_c6_pressure_lattice(reference, step.before.epi).pressure
        assert step.capacity == (1.,) * 6 and step.timestep == .0625
        assert step.before.exact_epi == passage.points[ordinal].exact_epi
        assert step.after.exact_epi == passage.points[ordinal + 1].exact_epi
        lower_count += int(step.before.epi[0] == budget.lower_visible)
        point = passage.points[ordinal + 1]
        assert point.lower_visits == lower_count
        assert point.expected_relay == step.after.exact_epi[0]
        assert point.expected_budget == step.after.exact_epi[1]
        assert point.corrected_budget == step.after.exact_epi[1] - budget.budget_correction * step.after.exact_epi[0]
        assert point.corrected_budget == budget.corrected_initial + (ordinal + 1) * budget.budget_drift
        assert point.corrected_budget_residual == 0
        assert not any(step.nodal_balance_residual)


def test_full_vector_area_retains_mean_drift_across_the_handoff(previous, passage):
    area = tuple(sum((step.exact_increment[i] for step in passage.steps), F(0)) for i in range(6))
    assert area == passage.total_nodal_area
    assert tuple(value / M for value in area) == (
        1645227481908860, -502439447874810275, 609821998071731052,
        1133737516365944, 1031924247429912, -111193439442625144,
    )
    assert tuple(end - start for start, end in zip(previous.endpoint.exact_epi, passage.exact_endpoint)) == area
    assert passage.mean_area == sum(area, F(0)) / 6 == F(349, 3 * 2**114)
    total = tuple(a + b for a, b in zip(previous.total_nodal_area, area))
    assert tuple(end - start for start, end in zip(previous.steps[0].before.exact_epi, passage.exact_endpoint)) == total
    assert not any(passage.nodal_balance_residual)
    assert passage.mean_area + previous.mean_area == F(176, 3 * 2**113)


def test_band_certificate_covers_the_deadline_but_not_future_runtime(budget, passage):
    assert budget.band_horizon.maximum_steps == 196713720348826205
    assert budget.band_horizon.maximum_steps >= budget.deadline
    assert budget.initial_membership_verified
    assert not budget.incoming_state_lineage_certified
    assert not budget.free_coordinates_restricted_to_anchor_cells
    assert not budget.future_runtime_certified
    assert not passage.incoming_state_lineage_certified
    assert not passage.future_runtime_certified and not passage.indefinite_trapping_certified


@pytest.mark.parametrize("free", ((.375, .625), (.625, .375), (.5, .5),
                                  (math.nextafter(.375, math.inf), math.nextafter(.625, -math.inf))))
@pytest.mark.parametrize("lower", (False, True))
def test_fresh_canonical_local_rows_ignore_arbitrary_free_visible_values(previous, budget, free, lower):
    reference = budget.closure.base_tube.contraction.profile.lattice
    row = list(previous.endpoint.epi)
    row[0] = budget.lower_visible if lower else budget.upper_visible
    row[3], row[4] = free
    observed = _observe_rebuilt_c6_pressure_lattice(reference, tuple(row))
    area = tuple(F(value) / 16 for value in observed.pressure)
    assert area[0] == (budget.increment if lower else -budget.decrement)
    assert area[1] == budget.budget_drift + budget.budget_correction * area[0]
    # Their exact gradient indices use only the explicit C6 neighborhoods.
    exact = tuple(map(F, row))
    assert observed.gradient_indices[0] * reference.gradient_quantum == (exact[5] + exact[1]) / 2 - exact[0]
    assert observed.gradient_indices[1] * reference.gradient_quantum == (exact[0] + exact[2]) / 2 - exact[1]


def test_eight_cell_control_retains_the_nonzero_binary64_interaction(previous):
    reference = previous.relay.closure.base_tube.contraction.profile.lattice
    base = previous.relay.visible_rows[0]
    rows = []
    for mask in range(8):
        row = list(base)
        for bit, node in enumerate((0, 3, 4)):
            if mask & (1 << bit):
                row[node] = math.nextafter(row[node], -math.inf)
        pressure = _observe_rebuilt_c6_pressure_lattice(reference, tuple(row)).pressure
        rows.append(tuple(F(value) / 16 for value in pressure))
    mixed = tuple(rows[6][i] - rows[2][i] - rows[4][i] + rows[0][i] for i in range(6))
    assert mixed == (F(0), F(0), F(0), -8 * M, -8 * M, F(0))
    assert all(rows[7][i] - rows[3][i] - rows[5][i] - rows[6][i]
               + rows[1][i] + rows[2][i] + rows[4][i] - rows[0][i] == 0 for i in range(6))


@pytest.mark.parametrize("free", ((F(3, 8), F(5, 8)), (F(5, 8), F(3, 8))))
def test_admission_is_not_restricted_to_the_eight_local_cells(previous, budget, free):
    state = _state_with_exact(previous.endpoint, {3: free[0], 4: free[1]})
    altered = derive_c6_carried_local_relay_budget(previous.relay, state=state)
    assert altered.deadline == budget.deadline and altered.budget_drift == budget.budget_drift
    assert altered.closure.base_tube.state == state
    observed = observe_c6_carried_local_relay_exit(altered, step_budget=198)
    assert observed.exit_step == 1


def test_lower_visible_incoming_facet_keeps_the_original_relay_pair(previous, budget):
    state = _state_with_exact(previous.endpoint, {0: budget.facet})
    assert state.epi[0] == budget.lower_visible
    bound = derive_c6_carried_local_relay_budget(previous.relay, state=state)
    assert bound.upper_visible == budget.upper_visible and bound.lower_visible == budget.lower_visible
    assert bound.initial_offset == 0
    observed = observe_c6_carried_local_relay_exit(bound)
    assert observed.steps[0].exact_increment[0] == bound.increment
    assert observed.steps[0].after.exact_epi[0] == bound.facet + bound.increment


def test_open_and_closed_strip_edges_are_checked_on_the_shared_grid(previous, budget):
    included = _state_with_exact(previous.endpoint, {0: budget.facet + budget.increment})
    closed = derive_c6_carried_local_relay_budget(previous.relay, state=included)
    assert closed.initial_offset == budget.increment
    excluded = _state_with_exact(previous.endpoint, {0: budget.facet - budget.decrement})
    with pytest.raises(ValueError):
        derive_c6_carried_local_relay_budget(previous.relay, state=excluded)
    interior = _state_with_exact(previous.endpoint, {0: budget.facet - budget.decrement + GRID})
    admitted = derive_c6_carried_local_relay_budget(previous.relay, state=interior)
    assert admitted.initial_offset == -budget.decrement + GRID


def test_uncovered_band_returns_formal_candidate_without_invalid_after_state(previous):
    original = previous.relay.closure.base_tube.state
    narrow = replace(original, epi_lower=original.epi[5])
    profile = previous.relay.closure.base_tube.contraction.profile
    anchor = derive_c6_carried_relay(derive_c6_carried_closure(profile, state=narrow, timestep=.0625))
    handoff = observe_c6_carried_relay_exit(anchor).endpoint
    state = _state_with_exact(handoff, {5: F(handoff.epi[5]) + GRID})
    bound = derive_c6_carried_local_relay_budget(anchor, state=state)
    assert not bound.band_covers_deadline
    observed = observe_c6_carried_local_relay_exit(bound)
    assert observed.exit_step == 1 and observed.band_failure and observed.exiting_nodes == ()
    assert observed.endpoint is None and observed.steps == () and len(observed.points) == 2
    assert observed.exact_endpoint[5] < F(state.epi_lower)
    assert observed.points[-1].corrected_budget_residual == 0
    assert not any(observed.nodal_balance_residual)


def test_cached_local_coefficients_and_coverage_are_rebuilt(budget, passage):
    forged = replace(budget, local_pressure_rows=((0., 0.), (0., 0.)),
                     budget_drift=F(0), budget_correction=F(0), corrected_initial=F(0),
                     held_nodes=(), free_nodes=tuple(range(6)), deadline=1,
                     band_covers_deadline=False, decrement=F(1), increment=F(1))
    assert observe_c6_carried_local_relay_exit(forged, step_budget=198) == passage
    with pytest.raises(ValueError):
        observe_c6_carried_local_relay_exit(forged, step_budget=197)


@pytest.mark.parametrize("node", (1, 2, 5))
def test_changing_a_held_visible_value_requires_a_new_certificate(previous, node):
    value = math.nextafter(previous.endpoint.epi[node], math.inf)
    changed = _state_with_exact(previous.endpoint, {node: F(value)})
    with pytest.raises(ValueError, match="held anchor values"):
        derive_c6_carried_local_relay_budget(previous.relay, state=changed)


def test_positive_budget_and_upper_owned_tie_use_the_other_signed_deadline(previous):
    anchor = previous.relay
    state = anchor.closure.base_tube.state
    bound = derive_c6_carried_local_relay_budget(anchor, state=state, relay_node=3, budget_node=2)
    assert bound.held_nodes == (1, 2, 4) and bound.free_nodes == (0, 5)
    assert bound.budget_drift > 0 and bound.lower_closed and not bound.upper_closed
    assert bound.deadline == 299 and bound.band_covers_deadline
    observed = observe_c6_carried_local_relay_exit(bound, step_budget=299)
    assert observed.exit_step == 7 and observed.exiting_nodes == (4,)
    assert observed.endpoint == previous.endpoint
    assert all(point.corrected_budget == bound.corrected_initial + point.ordinal * bound.budget_drift
               for point in observed.points)
    assert not any(observed.nodal_balance_residual)


@pytest.mark.parametrize("step_budget", (0, -1, True, 198., F(198)))
def test_step_budget_requires_a_positive_actual_integer(budget, step_budget):
    with pytest.raises(ValueError):
        observe_c6_carried_local_relay_exit(budget, step_budget=step_budget)


def test_work_budget_admission_precedes_any_conditional_nodal_step(budget, monkeypatch):
    import tnfr.physics.c6_carried_relay as owner

    def no_step(*_args, **_kwargs):
        raise AssertionError("insufficient budget must be rejected before execution")

    monkeypatch.setattr(owner, "advance_nodal_remainder", no_step)
    with pytest.raises(ValueError):
        owner.observe_c6_carried_local_relay_exit(budget, step_budget=197)


@pytest.mark.parametrize("relay_node,budget_node", ((False, 1), (0, True), (0, 2), (1, 0), (6, 1)))
def test_node_selection_requires_an_existing_axis_and_adjacent_budget_node(previous, relay_node, budget_node):
    with pytest.raises((TypeError, ValueError)):
        derive_c6_carried_local_relay_budget(
            previous.relay, state=previous.endpoint, relay_node=relay_node, budget_node=budget_node,
        )


def test_nonstate_and_changed_declared_band_are_rejected(previous):
    with pytest.raises((TypeError, ValueError)):
        derive_c6_carried_local_relay_budget(previous.relay, state=None)
    changed = replace(previous.endpoint, epi_lower=.25)
    with pytest.raises((TypeError, ValueError)):
        derive_c6_carried_local_relay_budget(previous.relay, state=changed)
