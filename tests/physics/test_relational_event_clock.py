"""Clock covariance and unselected event rates for supplied bridge exchanges.

These current-source controls use detached fields and one Euler step. The
explicit hazard formulas are logical countermodels, not implemented selectors,
sampled events, ideal trigonometric certificates or recovery proofs.
"""

import math
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel, step_relational_exchange
from tnfr.physics.relational_observations import observe_relational_relocation

MODEL = RelationalExchangeModel(storage_scale=1.0)
EPSILON = Q(1, 256)
CLOCK_SCALE = 4


def _graph(*, capacity_scale=1.0, second_form=False):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for offset in (0, 5):
        graph.add_edges_from(
            (offset + node, offset + (node + 1) % 5) for node in range(5)
        )
    graph.add_edge(0, 5)
    graph.graph["_t"] = 2.0 * capacity_scale
    for node in graph:
        form = EPSILON if node == 0 else Q(0)
        if second_form and node == 1:
            form = EPSILON / 2
        graph.nodes[node].update(
            EPI=float(form),
            theta=2 * math.pi * (node % 5) / 5,
            nu_f=(1.0 + node % 3) / capacity_scale,
            delta_nfr=0.0,
        )
    return graph


def _relocate(graph, bridge=(1, 6)):
    return observe_relational_relocation(
        graph, model=MODEL, remove_bridge=(0, 5), add_bridge=bridge
    )


@pytest.fixture(scope="module")
def clock_reports():
    return _relocate(_graph()), _relocate(_graph(capacity_scale=CLOCK_SCALE))


@pytest.mark.parametrize("endpoint", ("before", "after"))
def test_clock_conversion_scales_both_native_rows_and_work(clock_reports, endpoint):
    original, converted = (getattr(report, endpoint) for report in clock_reports)
    for name in (
        "epi",
        "phase",
        "pressure",
        "phase_source",
        "phase_metric",
        "form_gradient",
        "phase_gradient",
        "relative_resultant",
        "form_storage",
        "phase_storage",
        "storage",
        "pressure_split_residual",
    ):
        assert getattr(converted, name) == getattr(original, name)
    for name in (
        "capacity",
        "form_rate",
        "phase_rate",
        "phase_mobility",
        "phase_rate_rounding_defect",
        "nodal_rate_rounding_defect",
    ):
        assert tuple(map(Q, getattr(converted, name))) == tuple(
            Q(value) / CLOCK_SCALE for value in getattr(original, name)
        )
    for name in ("continuous_loss", "storage_rate", "balance_residual"):
        assert getattr(converted, name) == getattr(original, name) / CLOCK_SCALE
    assert converted.work.form_gradient == original.work.form_gradient
    for name in (
        "dissipation",
        "exchange",
        "form_work",
        "phase_work",
        "form_residual",
        "phase_residual",
        "balance_residual",
    ):
        assert getattr(converted.work, name) == tuple(
            value / CLOCK_SCALE for value in getattr(original.work, name)
        )
    # Both rows are active; this is not an equilibrium-only covariance check.
    assert any(original.form_rate)
    assert any(original.phase_rate)


def test_event_storage_is_clock_invariant_while_rate_changes_scale(clock_reports):
    original, converted = clock_reports
    assert converted.storage_change == original.storage_change == -(EPSILON**2) / 2
    for name in ("energy_change", "edge_energy_change", "identity_residual"):
        assert getattr(converted.transport_reset, name) == getattr(
            original.transport_reset, name
        )
    assert converted.represented_zero_supply_passive
    for name in ("form_rate_change", "phase_rate_change"):
        assert getattr(converted, name) == tuple(
            value / CLOCK_SCALE for value in getattr(original, name)
        )
    assert converted.continuous_loss_change == (
        original.continuous_loss_change / CLOCK_SCALE
    )


def test_changing_only_dt_does_not_transform_the_held_capacity_law():
    dt = 2.0**-12
    original = step_relational_exchange(_graph(), model=MODEL, dt=dt)
    converted = step_relational_exchange(
        _graph(capacity_scale=CLOCK_SCALE), model=MODEL, dt=CLOCK_SCALE * dt
    )
    duration_only = step_relational_exchange(_graph(), model=MODEL, dt=CLOCK_SCALE * dt)
    assert converted.after.epi == original.after.epi
    assert converted.after.phase == original.after.phase
    assert converted.energy_change == original.energy_change
    assert converted.t_before == CLOCK_SCALE * original.t_before
    assert converted.t_after == CLOCK_SCALE * original.t_after
    assert converted.epi_update_defect == original.epi_update_defect
    assert converted.phase_update_defect == original.phase_update_defect
    # A longer step with the same rates changes the actual two-row response.
    assert duration_only.before.form_rate == original.before.form_rate
    assert duration_only.before.phase_rate == original.before.phase_rate
    assert duration_only.after.epi != original.after.epi
    assert duration_only.after.phase != original.after.phase


def test_observed_passive_slacks_leave_event_clock_and_choice_undetermined():
    graph = _graph(second_form=True)
    reports = tuple(_relocate(graph, bridge) for bridge in ((1, 6), (2, 7)))
    # Old bridge cost is eps**2/2; the two new costs are eps**2/8 and zero.
    # Equal paired phases make these exact represented storage differences.
    assert tuple(report.storage_change for report in reports) == (
        -3 * EPSILON**2 / 8,
        -(EPSILON**2) / 2,
    )
    assert all(report.represented_zero_supply_passive for report in reports)
    beta = Q(MODEL.storage_scale)
    slacks = tuple(-report.storage_change / beta for report in reports)
    capacities = reports[0].before.capacity
    rate_scale = Q(MODEL.epi_weight) * sum(map(Q, capacities)) / len(capacities)

    # Each is an independently declared first-event intensity, set to zero
    # outside its supplied recovery-admitted candidate domain or after stopping.
    # No test here derives such a domain from rounded phases or samples a clock.
    linear = tuple(rate_scale * slack for slack in slacks)
    squared = tuple(rate_scale * slack**2 for slack in slacks)
    assert all(value > 0 for value in linear + squared)
    assert sum(linear) != sum(squared)
    assert linear[0] / linear[1] == Q(3, 4)
    assert squared[0] / squared[1] == Q(9, 16)
    assert linear[0] / sum(linear) == Q(3, 7)
    assert squared[0] / sum(squared) == Q(9, 25)

    # A common total rate still permits different conditional candidate choices.
    equal_clock_squared = tuple(sum(linear) * value / sum(squared) for value in squared)
    assert sum(equal_clock_squared) == sum(linear)
    assert equal_clock_squared[0] / sum(equal_clock_squared) == Q(9, 25)

    converted = tuple(
        _relocate(_graph(capacity_scale=CLOCK_SCALE, second_form=True), bridge)
        for bridge in ((1, 6), (2, 7))
    )
    converted_slacks = tuple(-report.storage_change / beta for report in converted)
    converted_rate_scale = (
        Q(MODEL.epi_weight)
        * sum(map(Q, converted[0].before.capacity))
        / len(capacities)
    )
    assert converted_slacks == slacks
    for power, original in ((1, linear), (2, squared)):
        assert tuple(
            converted_rate_scale * slack**power for slack in converted_slacks
        ) == tuple(value / CLOCK_SCALE for value in original)
