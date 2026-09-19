"""Instantaneous canonical source work in the inherited prism internal norm.

The prepared phase profile supplies pressure, not a derived phase law or a
maintenance trajectory. Exact represented-state budgets retain kernel and
stored-pressure defects separately from the ideal trigonometric comparison.
"""

from fractions import Fraction as Q
from math import pi

import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    PROJECTION,
    _apply,
    _graph,
    _inner,
)
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from tnfr.physics.support_transport import observe_regional_support_balance

_COEFFICIENTS = (Q(1, 16), Q(0), Q(1, 16), Q(0))
_REGIONS = tuple(tuple(node for node in NODES if node[0] == a) for a in range(2))


def _source(coefficients=_COEFFICIENTS, *, means=(Q(1, 2), Q(1, 2))):
    graph = _graph(coefficients, means=means)
    graph.graph["DNFR_WEIGHTS"] = {
        "epi": 0.5,
        "phase": 0.25,
        "vf": 0.25,
        "topo": 0.0,
    }
    return graph


def _prepare_fresh_pressure(graph):
    # Detached input construction, not an executed pressure refresh or step.
    captured = capture_non_epi_forcing(graph)
    for node, pressure in zip(NODES, captured.full_kernel_pressure, strict=True):
        graph.nodes[node]["delta_nfr"] = float(pressure)
    prepared = capture_non_epi_forcing(graph)
    assert prepared.stored_pressure_residual == (0,) * 6
    return prepared


def _regional_balances(observation):
    return tuple(
        observe_regional_support_balance(
            observation.snapshot,
            region,
            epi_weight=observation.epi_weight,
            forcing=observation.forcing,
        )
        for region in _REGIONS
    )


def _model_rate(observation):
    return tuple(
        nu * (observation.epi_weight * gradient + forcing)
        for nu, gradient, forcing in zip(
            observation.snapshot.capacity,
            observation.snapshot.epi_gradient,
            observation.forcing,
            strict=True,
        )
    )


def _unit_capacity_budget(observation):
    # Only this uniform unit-capacity slice has sum(E_region)=3*S/2.
    assert observation.snapshot.capacity == (1,) * 6
    balances = _regional_balances(observation)
    c = _apply(PROJECTION, observation.snapshot.epi)
    c_dot = _apply(PROJECTION, observation.snapshot.rate)
    norm = _inner(c[:2], c[:2]) + _inner(c[2:], c[2:])
    disagreement = tuple(a - b for a, b in zip(c[:2], c[2:], strict=True))
    loss = -2 * observation.epi_weight * norm
    loss -= Q(2, 3) * observation.epi_weight * _inner(disagreement, disagreement)
    source_work = Q(2, 3) * sum(row.variance_forcing_rate for row in balances)
    defect_work = Q(2, 3) * sum(row.variance_defect_rate for row in balances)
    rate = Q(2, 3) * sum(row.stored_variance_rate for row in balances)
    centered = tuple(value for row in balances for value in row.centered_epi)
    defects = tuple(
        a + b
        for a, b in zip(
            observation.kernel_pressure_defect,
            observation.stored_pressure_residual,
            strict=True,
        )
    )
    assert norm == Q(2, 3) * sum(row.variance for row in balances)
    assert loss == Q(2, 3) * sum(
        -row.internal_dissipation + row.variance_boundary_rate for row in balances
    )
    assert source_work == 2 * sum(
        y * f
        for y, f in zip(
            centered,
            observation.forcing,
            strict=True,
        )
    )
    assert defect_work == 2 * sum(y * d for y, d in zip(centered, defects, strict=True))
    assert rate == 2 * (_inner(c[:2], c_dot[:2]) + _inner(c[2:], c_dot[2:]))
    assert rate == loss + source_work + defect_work
    return norm, loss, source_work, defect_work, rate


def test_unforced_internal_norm_budget_recovers_the_existing_passive_loss():
    observed = _prepare_fresh_pressure(_source())
    assert observed.forcing == observed.kernel_pressure_defect == (0,) * 6
    assert _unit_capacity_budget(observed) == (
        Q(1, 64),
        Q(-1, 64),
        0,
        0,
        Q(-1, 64),
    )


@pytest.mark.parametrize("capacities", ((Q(1), Q(1)), (Q(1), Q(2))))
def test_fiber_constant_phase_and_capacity_have_no_internal_source_drive(capacities):
    graph = _source()
    for a, i in NODES:
        graph.nodes[a, i]["theta"] = a * pi / 3
        graph.nodes[a, i]["nu_f"] = capacities[a]
    observed = capture_non_epi_forcing(graph)
    assert any(observed.forcing)
    assert observed.snapshot.topology_gradient == (0,) * 6
    gap = (capacities[1] - capacities[0]) / 3
    assert observed.snapshot.capacity_gradient == (gap,) * 3 + (-gap,) * 3
    for _, channel in decompose_non_epi_forcing(observed):
        assert _apply(PROJECTION, channel) == (0,) * 4
    weighted_source = tuple(
        nu * f
        for nu, f in zip(
            observed.snapshot.capacity,
            observed.forcing,
            strict=True,
        )
    )
    assert _apply(PROJECTION, weighted_source) == (0,) * 4
    assert _apply(PROJECTION, _model_rate(observed)) == (
        -observed.epi_weight * capacities[0] / 16,
        0,
        -observed.epi_weight * capacities[1] / 16,
        0,
    )
    for capacity, row in zip(capacities, _regional_balances(observed), strict=True):
        assert row.variance_forcing_rate == 0
        assert (
            row.variance
            == Q(3, 2) * _inner(_COEFFICIENTS[:2], _COEFFICIENTS[:2]) / capacity
        )
    # Multiplication by capacity preserves the zero internal drive here only
    # because capacity itself is fiberwise constant. No nu' law is inferred.


def test_prepared_internal_phase_source_can_exceed_passive_loss_instantaneously():
    graph = _source()
    phases = (0.0, pi / 3, pi / 6)
    assert all(0 <= phase < 2 * pi for phase in phases)
    for a, i in NODES:
        graph.nodes[a, i]["theta"] = phases[i]
    assert all(
        abs(graph.nodes[u]["theta"] - graph.nodes[v]["theta"]) < pi / 2
        for u, v in graph.edges
    )
    observed = _prepare_fresh_pressure(graph)
    coefficient = Q(1501199875790165, 2**55)
    assert _apply(PROJECTION, observed.forcing) == (coefficient, 0, coefficient, 0)
    channels = dict(decompose_non_epi_forcing(observed))
    assert channels["phase"] == observed.forcing
    assert channels["vf"] == channels["topo"] == (0,) * 6
    assert sum(observed.forcing) == 0
    assert observed.kernel_pressure_defect == (0,) * 6
    norm, loss, source, defect, rate = _unit_capacity_budget(observed)
    assert norm == Q(1, 64)
    assert loss == Q(-1, 64)
    assert source == Q(1501199875790165, 2**56) > -loss
    assert defect == 0
    assert rate == Q(375299968947541, 2**56) > 0
    # Ideal phases give g=(1/6,-1/6,0) per fiber, hence S'=1/192.
    # Represented phase arithmetic is retained even though assembly is exact.
    assert rate - Q(1, 192) == Q(-1, 3 * 2**56)
    # The primitive phase profile is prepared, not a derived maintained state.
    # Positive work at this snapshot is neither a trajectory nor persistence.


def test_current_stage_budget_keeps_kernel_and_stored_pressure_work_separate():
    graph = _source((Q(1, 4), 0, Q(1, 4), 0), means=(Q(11, 16), Q(5, 16)))
    fresh = _prepare_fresh_pressure(graph)
    kernel = (Q(1, 2**55), 0, 0, 0, Q(-1, 2**55), 0)
    assert fresh.kernel_pressure_defect == kernel
    added = (Q(1, 64), Q(-1, 64), Q(0)) * 2
    for node, value in zip(NODES, added, strict=True):
        graph.nodes[node]["delta_nfr"] += float(value)
    current = capture_non_epi_forcing(graph)
    assert current.forcing == (0,) * 6
    assert current.full_kernel_pressure == fresh.full_kernel_pressure
    assert current.kernel_pressure_defect == kernel
    assert current.stored_pressure_residual == added
    before = _unit_capacity_budget(fresh)
    after = _unit_capacity_budget(current)
    assert before == (Q(1, 4), Q(-1, 4), 0, Q(1, 2**55), Q(-1, 4) + Q(1, 2**55))
    assert after == (
        Q(1, 4),
        Q(-1, 4),
        0,
        Q(1, 32) + Q(1, 2**55),
        Q(-7, 32) + Q(1, 2**55),
    )
    assert after[-1] - before[-1] == Q(1, 32)
    # The stored-pressure term could encode a prior event or stale input.
    # It is measured, not silently reclassified as canonical source forcing.


def test_positive_nonuniform_capacity_breaks_four_internal_coordinate_closure():
    delta = Q(3, 16)
    first_graph = _source()
    second_graph = _source(means=(Q(1, 2) + delta, Q(1, 2)))
    capacities = (1, 2, 1, 1, 1, 1)
    for graph in (first_graph, second_graph):
        for node, capacity in zip(NODES, capacities, strict=True):
            graph.nodes[node]["nu_f"] = capacity
    first, second = map(capture_non_epi_forcing, (first_graph, second_graph))
    assert _apply(PROJECTION, first.snapshot.epi) == _apply(
        PROJECTION, second.snapshot.epi
    )
    assert first.snapshot.capacity == second.snapshot.capacity == capacities
    assert first.forcing == second.forcing
    first_rate = _apply(PROJECTION, _model_rate(first))
    second_rate = _apply(PROJECTION, _model_rate(second))
    difference = tuple(b - a for a, b in zip(first_rate, second_rate, strict=True))
    assert difference == (
        first.epi_weight * delta / 6,
        -first.epi_weight * delta / 18,
        0,
        0,
    )
    assert difference == (Q(1, 64), Q(-1, 192), 0, 0)
    # These are exact declared-model rates, not captured binary64 updates.
    # The omitted mean couples through diag(nu), even at positive capacities.
