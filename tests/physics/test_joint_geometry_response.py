"""Joint nodal derivatives retain independently supplied conductance motion.

The local jets below are differentiation controls, not selected capacity,
phase or support laws and not integrated trajectories. Fixed support and
channel coefficients are retained; only existing positive edges vary smoothly.
"""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.physics.test_joint_nodal_response import _capture, _joint


@pytest.fixture(scope="module")
def weighted_path():
    graph = nx.path_graph(3)
    graph[0][1]["weight"] = 1.0
    graph[1][2]["weight"] = 3.0
    source = _capture(graph, (0.0, 0.5, 1.0), (1.0, 2.0, 4.0))
    rates = tuple(
        Q(1, 2) if {i, j} == {0, 1} else Q(-1, 4) for i, j, _ in source.conductance
    )
    return graph, source, rates


def test_complete_derivative_matches_an_independent_weighted_nodal_jet(weighted_path):
    s = pytest.importorskip("sympy")
    graph, source, rates = weighted_path
    before = deepcopy(graph)
    t = s.Symbol("t", real=True)
    capacity_rate = (Q(1, 4), Q(-1, 2), Q(3, 4))
    angular_rate = (Q(0), Q(1, 2), Q(-1, 4))
    result = _joint(
        source,
        conductance_rates=rates,
        capacity_rate=capacity_rate,
        phase_rate_over_pi=angular_rate,
    )
    x = s.Matrix(source.epi) + t * s.Matrix(source.rate)
    nu = s.Matrix(source.capacity) + t * s.Matrix(capacity_rate)
    left, right = 1 + t / 2, 3 - t / 4
    epi_gradient = s.Matrix(
        (
            x[1] - x[0],
            (left * x[0] + right * x[2]) / (left + right) - x[1],
            x[1] - x[2],
        )
    )
    capacity_gradient = s.Matrix(
        (nu[1] - nu[0], (nu[0] + nu[2]) / 2 - nu[1], nu[1] - nu[2])
    )
    # Singleton means and the two-neighbor regular midpoint make this phase
    # jet exact near consensus. Conductance does not weight those means.
    angles = t * s.Matrix(angular_rate)  # Phase divided by exact pi.
    phase_gradient = s.Matrix(
        (
            angles[1] - angles[0],
            (angles[0] + angles[2]) / 2 - angles[1],
            angles[1] - angles[2],
        )
    )
    pressure = epi_gradient / 2 + phase_gradient / 4 + capacity_gradient / 4
    assert tuple(pressure.subs(t, 0)) == source.stored_pressure
    assert tuple(x.diff(t)) == source.rate
    derivative = pressure.diff(t).subs(t, 0)
    acceleration = (s.diag(*nu) * pressure).diff(t).subs(t, 0)
    assert s.Matrix(result.pressure_rate) == derivative
    assert s.Matrix(result.epi_acceleration) == acceleration
    assert s.Matrix(result.epi_pressure_rate) == (epi_gradient.diff(t) / 2).subs(t, 0)
    assert s.Matrix(result.phase_pressure_rate) == (phase_gradient.diff(t) / 4).subs(
        t, 0
    )
    assert s.Matrix(result.capacity_pressure_rate) == (
        capacity_gradient.diff(t) / 4
    ).subs(t, 0)
    assert sum(result.epi_acceleration) / 3 == sum(acceleration) / 3
    assert sum(acceleration) != 0
    assert result.transport.source == result.source == source
    assert result.conductance_rates == rates
    assert nx.utils.graphs_equal(graph, before)


def test_normalization_motion_changes_the_actual_form_mean(weighted_path):
    _, source, rates = weighted_path
    fixed = _joint(source)
    moving = _joint(source, conductance_rates=rates)
    # The center's denominator changes by 1/4. Omitting its quotient-rule
    # term gives -3/32 instead of -7/64 for the raw EPI gradient derivative.
    assert moving.transport.geometry_gradient_rate == (0, Q(-7, 64), 0)
    assert moving.epi_geometry_pressure_rate == (0, Q(-7, 128), 0)
    assert moving.epi_flow_pressure_rate == fixed.epi_pressure_rate
    assert moving.epi_pressure_rate == tuple(
        flow + geometry
        for flow, geometry in zip(
            moving.epi_flow_pressure_rate,
            moving.epi_geometry_pressure_rate,
            strict=True,
        )
    )
    assert moving.phase_pressure_rate == fixed.phase_pressure_rate
    assert moving.capacity_pressure_rate == fixed.capacity_pressure_rate
    assert tuple(
        after - before
        for after, before in zip(
            moving.epi_acceleration, fixed.epi_acceleration, strict=True
        )
    ) == (0, Q(-7, 64), 0)
    assert (sum(moving.epi_acceleration) - sum(fixed.epi_acceleration)) / 3 == Q(
        -7, 192
    )

    # A second detached point has x=2-nu/2 and W equal to unique support,
    # so e=1/2 and w_nu=1/4 compensate exactly at zero pressure. An edge
    # tangent alone can leave that instantaneous set without a phase or
    # capacity tangent. No autonomous occurrence of this tangent is claimed.
    equilibrium = _capture(nx.path_graph(3), (1.5, 1.0, 0.0), (1.0, 2.0, 4.0))
    assert equilibrium.stored_pressure == equilibrium.rate == (0, 0, 0)
    edge_rates = tuple(Q({i, j} == {0, 1}) for i, j, _ in equilibrium.conductance)
    departing = _joint(equilibrium, conductance_rates=edge_rates)
    assert _joint(equilibrium).epi_acceleration == (0, 0, 0)
    assert (
        departing.phase_pressure_rate == departing.capacity_pressure_rate == (0, 0, 0)
    )
    assert departing.pressure_rate == (0, Q(3, 16), 0)
    assert departing.epi_acceleration == (0, Q(3, 8), 0)
    assert sum(departing.epi_acceleration) / 3 == Q(1, 8)


def test_common_conductance_scaling_changes_energy_work_without_pressure_response(
    weighted_path,
):
    _, source, _ = weighted_path
    rates = tuple(3 * weight for _, _, weight in source.conductance)
    fixed = _joint(source)
    scaled = _joint(source, conductance_rates=rates)
    assert scaled.epi_geometry_pressure_rate == (0, 0, 0)
    assert scaled.pressure_rate == fixed.pressure_rate
    assert scaled.epi_acceleration == fixed.epi_acceleration
    assert scaled.transport.conductance_work == 3 * source.dirichlet_energy > 0
    assert (
        scaled.transport.energy_rate - fixed.transport.energy_rate
        == scaled.transport.conductance_work
    )
    assert scaled.transport.nodal_work == fixed.transport.nodal_work


def test_default_geometry_rate_equals_explicit_zero_without_changing_old_response(
    weighted_path,
):
    _, source, _ = weighted_path
    default = _joint(source)
    explicit = _joint(source, conductance_rates=(Q(0),) * len(source.conductance))
    assert default == explicit
    assert default.conductance_rates == (0,) * len(source.conductance)
    assert default.epi_geometry_pressure_rate == (0, 0, 0)
    assert default.epi_flow_pressure_rate == default.epi_pressure_rate
    assert default.transport.conductance_work == 0
    assert default.transport.energy_rate == source.energy_rate


def test_loop_motion_changes_normalization_without_energy_or_support_channel_work():
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_weighted_edges_from(((0, 1, 2.0), (1, 2, 0.0), (0, 2, 1.0), (0, 0, 3.0)))
    source = _capture(graph, (0.0, 1.0, 1.0), (1.0,) * 3)
    rates = tuple(Q(i == j) for i, j, _ in source.conductance)
    fixed = _joint(source)
    moving = _joint(source, conductance_rates=rates)
    assert source.support_neighbors == ((0, 1, 2), (0, 2), (0, 1))
    assert (1, 2) not in {(i, j) for i, j, _ in source.conductance}
    assert moving.epi_geometry_pressure_rate == (Q(-1, 24), 0, 0)
    assert moving.transport.conductance_work == 0
    assert moving.phase_pressure_rate == fixed.phase_pressure_rate == (0, 0, 0)
    assert moving.capacity_pressure_rate == fixed.capacity_pressure_rate == (0, 0, 0)
    assert moving.source.support_neighbors == fixed.source.support_neighbors


def test_joint_entry_reuses_rate_alignment_and_symmetry_validation(weighted_path):
    _, source, _ = weighted_path
    count = len(source.conductance)
    with pytest.raises(ValueError, match="align with every conductance entry"):
        _joint(source, conductance_rates=(0,) * (count - 1))
    with pytest.raises(ValueError, match="preserve symmetry"):
        _joint(source, conductance_rates=(1,) + (0,) * (count - 1))


def test_edge_rate_pairs_keep_their_identity_after_snapshot_reordering(weighted_path):
    _, source, rates = weighted_path
    reference = _joint(source, conductance_rates=rates)
    reordered = replace(source, conductance=tuple(reversed(source.conductance)))
    actual = _joint(reordered, conductance_rates=tuple(reversed(rates)))
    assert actual == reference
    assert actual.conductance_rates == rates
    assert actual.epi_geometry_pressure_rate == (0, Q(-7, 128), 0)
