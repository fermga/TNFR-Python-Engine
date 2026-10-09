"""Static full-row controls for the conditional saddle sensitivity metric.

No trajectory or retained research producer is evaluated by these tests.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import exact_matrix_product
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_sensitivity import assess_sine_saddle_sensitivity


def _source(*, order=tuple(range(10)), labels=tuple(range(10)), extra_edge=None):
    graph = nx.Graph()
    graph.add_nodes_from(labels[i] for i in order)
    graph.add_edges_from((labels[i], labels[(i + 1) % 5]) for i in range(5))
    graph.add_edges_from((labels[i], labels[i + 5]) for i in range(5))
    if extra_edge is not None:
        graph.add_edge(*(labels[i] for i in extra_edge))
    for i, label in enumerate(labels):
        graph.nodes[label].update(EPI=i / 8, theta=i / 16, nu_f=1)
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    return graph, source, labels[:5]


@pytest.fixture(scope="module")
def fixture():
    graph, source, cycle = _source()
    return graph, source, assess_sine_saddle_sensitivity(source, cycle=cycle)


def _matvec(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _quadratic(matrix, vector):
    return sum(a * b for a, b in zip(vector, _matvec(matrix, vector)))


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _full_graph_tangent(graph):
    # Directly differentiate the ten normalized neighbor rows. The exact
    # saddle has four +1/2 ring curvatures, one -1/2 and five unit contacts.
    result = [[Q(0) for _ in range(20)] for _ in range(20)]
    for i in graph:
        degree = graph.degree[i]
        for j in graph[i]:
            curvature = (
                Q(1) if max(i, j) >= 5 else Q(-1, 2) if {i, j} == {0, 4} else Q(1, 2)
            )
            result[i][10 + j] += curvature / degree
            result[i][10 + i] -= curvature / degree
            result[10 + i][i] += Q(1, degree)
            result[10 + i][j] -= Q(1, degree)
    return tuple(tuple(row) for row in result)


def test_metric_and_both_slacks_use_the_actual_full_nodal_tangent(fixture):
    graph, _, report = fixture
    tangent = _full_graph_tangent(graph)
    assert report.saddle.full_tangent_generator == tangent
    weighted = exact_matrix_product(report.full_metric, tangent)
    for i in range(20):
        for j in range(20):
            symmetric = weighted[i][j] + weighted[j][i]
            assert report.forward_tangent_slack[i][j] == (
                Q(2, 3) * report.full_metric[i][j] - symmetric
            )
            assert report.reverse_tangent_slack[i][j] == (
                Q(2, 3) * report.full_metric[i][j] + symmetric
            )
    assert report.metric_positive_definite
    assert report.both_tangent_directions_certified
    assert report.nonlinear_tube_implication_certified
    assert report.status == "certified"


def test_relative_chart_reconstructs_all_modes_and_both_common_origins(fixture):
    graph, _, report = fixture
    basis, inverse = report.coordinate_basis, report.inverse_coordinate_basis
    assert exact_matrix_product(basis, inverse) == tuple(
        tuple(Q(i == j) for j in range(10)) for i in range(10)
    )
    for column in zip(*report.relative_basis):
        assert sum(graph.degree[i] * value for i, value in enumerate(column)) == 0
    assert inverse[0] == tuple(Q(graph.degree[i], 20) for i in graph)
    metric, tangent = report.full_metric, _full_graph_tangent(graph)
    for form_origin in (True, False):
        origin = tuple(Q((i < 10) == form_origin) for i in range(20))
        assert _quadratic(metric, origin) == 1
        assert _matvec(tangent, origin) == (0,) * 20
    # This even, nonsymmetric generic state is not in the odd four-mode family.
    full = tuple(Q((i * i + 3) % 11 - 5, 7) for i in range(20))
    form, phase = _matvec(inverse, full[:10]), _matvec(inverse, full[10:])
    expected = (
        form[0] ** 2
        + phase[0] ** 2
        + _quadratic(report.relative_form_metric, form[1:])
        + _quadratic(report.relative_phase_metric, phase[1:])
    )
    assert _quadratic(metric, full) == expected > 0


def test_dual_norm_conversions_bound_coordinate_extremizers(fixture):
    report = fixture[2]
    metric, inverse = report.full_metric, report.inverse_full_metric
    assert exact_matrix_product(metric, inverse) == tuple(
        tuple(Q(i == j) for j in range(20)) for i in range(20)
    )
    absolute_sum = sum(abs(value) for row in metric for value in row)
    assert report.infinity_to_metric_upper_bound**2 >= absolute_sum
    for i in range(20):
        # W^-1 e_i attains the coordinate dual norm, so origins or transverse
        # components cannot have been discarded by the reported conversion.
        vector = tuple(inverse[j][i] for j in range(20))
        norm_squared = _quadratic(metric, vector)
        assert norm_squared == inverse[i][i] > 0
        assert (
            vector[i] ** 2
            <= report.metric_to_coordinate_upper_bounds[i] ** 2 * norm_squared
        )
    assert report.metric_to_infinity_upper_bound**2 >= max(
        inverse[i][i] for i in range(20)
    )


def test_nonlinear_neighbor_derivative_obeys_both_metric_bounds(fixture):
    graph, _, report = fixture
    error = tuple(Q((7 * i) % 11 - 5, 5000) for i in range(10))
    vector = tuple(Q((5 * i * i + i) % 13 - 6, 7) for i in range(20))
    assert max(map(abs, error)) <= report.phase_radius
    with mp.workdps(85):
        phase = tuple(
            2 * mp.pi * _mp(turn) + _mp(delta)
            for turn, delta in zip(report.saddle.target_phase_turns, error)
        )
        direction = tuple(map(_mp, vector))

        def actual_full_rows(h):
            # Large form amplitudes do not affect this phase-curvature bound.
            form = tuple(mp.mpf(1000 * i) + h * direction[i] for i in range(10))
            angles = tuple(phase[i] + h * direction[10 + i] for i in range(10))
            return tuple(
                sum(mp.sin(angles[j] - angles[i]) for j in graph[i]) / graph.degree[i]
                for i in graph
            ) + tuple(
                sum(form[i] - form[j] for j in graph[i]) / graph.degree[i]
                for i in graph
            )

        differentiated = tuple(
            mp.diff(lambda h: actual_full_rows(h)[i], 0) for i in range(20)
        )
        dual = tuple(map(_mp, _matvec(report.full_metric, vector)))
        derivative = 2 * sum(a * b for a, b in zip(dual, differentiated))
        maximum = (
            2
            * _mp(report.nonlinear_growth_rate_upper_bound)
            * _mp(_quadratic(report.full_metric, vector))
        )
        assert abs(derivative) < maximum


def test_reordering_and_relabeling_preserve_the_full_metric(fixture):
    original = fixture[2]
    labels = tuple(("node", i) for i in range(10))
    # Anchor the relative chart at a degree-three receiver instead of a leaf.
    _, source, cycle = _source(order=(8, 2, 6, 4, 0, 9, 3, 5, 7, 1), labels=labels)
    report = assess_sine_saddle_sensitivity(source, cycle=cycle)
    permutation = tuple(label[1] for label in source.nodes)
    indices = permutation + tuple(i + 10 for i in permutation)
    for i, old_i in enumerate(indices):
        for j, old_j in enumerate(indices):
            assert report.full_metric[i][j] == original.full_metric[old_i][old_j]
    assert (
        report.infinity_to_metric_upper_bound == original.infinity_to_metric_upper_bound
    )
    assert (
        report.metric_to_infinity_upper_bound == original.metric_to_infinity_upper_bound
    )


def test_cached_state_verdicts_do_not_supply_a_trajectory_or_change_the_metric(fixture):
    _, source, original = fixture
    altered = replace(source, storage=I(-123), form_rates=(I(999),) * 10)
    report = assess_sine_saddle_sensitivity(altered, cycle=range(5))
    assert report.full_metric == original.full_metric
    assert report.nonlinear_tube_implication_certified
    assert not report.captured_source_flow_bound_certified
    assert source.phase != report.saddle.target_phase_turns
    assert report.conditional_premises


@pytest.mark.parametrize("radius", [Q(0), Q(1, 2**160), Q(5)])
def test_radius_is_a_conditional_premise_not_an_observed_tube(fixture, radius):
    report = assess_sine_saddle_sensitivity(
        fixture[1], cycle=range(5), phase_radius=radius
    )
    assert report.phase_radius == radius
    assert report.nonlinear_growth_rate_upper_bound == Q(1, 3) + 28 * radius
    assert not report.captured_source_flow_bound_certified


@pytest.mark.parametrize(
    "radius", [True, False, -1, float("nan"), float("inf"), "0.001"]
)
def test_invalid_phase_radius_cannot_certify(fixture, radius):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_saddle_sensitivity(fixture[1], cycle=range(5), phase_radius=radius)


@pytest.mark.parametrize("capacity", [True, Q(0), Q(1) + Q(1, 2**150)])
def test_consumed_law_is_readmitted_before_metric_use(fixture, capacity):
    source = fixture[1]
    with pytest.raises((TypeError, ValueError)):
        assess_sine_saddle_sensitivity(
            replace(source, capacity=(capacity,) + source.capacity[1:]), cycle=range(5)
        )


@pytest.mark.parametrize("edge", [(0, 2), (5, 6)])
def test_other_support_does_not_inherit_this_metric(edge):
    _, source, cycle = _source(extra_edge=edge)
    with pytest.raises(ValueError):
        assess_sine_saddle_sensitivity(source, cycle=cycle)


def test_export_retains_conditional_scope_and_checks_nested_labels(fixture):
    from tnfr.sdk import relational_report_to_dict

    report = fixture[2]
    projection = relational_report_to_dict(report)
    assert projection["report_type"] == "SineSaddleSensitivity"
    assert projection["report"] == report.to_dict()["report"]
    assert projection["report"]["captured_source_flow_bound_certified"] is False

    @dataclass(frozen=True)
    class Opaque:
        index: int

    geometry = report.saddle.target_geometry.geometry
    bad_geometry = replace(geometry, nodes=(Opaque(0),) + geometry.nodes[1:])
    bad_target = replace(report.saddle.target_geometry, geometry=bad_geometry)
    altered = replace(report, saddle=replace(report.saddle, target_geometry=bad_target))
    with pytest.raises(TypeError):
        altered.to_dict()
