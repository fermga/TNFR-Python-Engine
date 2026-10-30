"""Independent full-row and exact algebra controls for the declared C5 saddle.

These checks differentiate instantaneous nodal equations and bound a local
remainder; no research trajectory or retained producer is evaluated.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from itertools import permutations

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_resonance import assess_sine_c5_leaf_saddle


def _source(*, order=tuple(range(10)), labels=None, extra_edge=None):
    labels = tuple(range(10)) if labels is None else labels
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
    return graph, source, assess_sine_c5_leaf_saddle(source, cycle=cycle)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _flow(graph, form, phase):
    return tuple(
        sum(mp.sin(phase[j] - phase[i]) for j in graph[i]) / graph.degree[i]
        for i in graph
    ) + tuple(sum(form[i] - form[j] for j in graph[i]) / graph.degree[i] for i in graph)


def _matvec(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _determinant(matrix):
    result = Q(0)
    for order in permutations(range(len(matrix))):
        inversions = sum(
            order[j] > order[i] for i in range(len(order)) for j in range(i)
        )
        term = Q((-1) ** inversions)
        for i, j in enumerate(order):
            term *= matrix[i][j]
        result += term
    return result


def test_target_is_reconstructed_without_asserting_the_observed_state(fixture):
    _, source, report = fixture
    assert source.epi != report.target_epi
    assert source.phase != report.target_phase_turns
    assert (
        report.target_geometry.sine_balance_status
        == "proved_by_period_reflection_cancellation"
    )
    assert report.target_storage == Q(7, 2)
    assert report.positive_root_count == report.relative_hyperbolic_pairs == 1
    assert report.relative_oscillatory_pairs == 8
    assert report.neutral_origin_modes == 2
    poisoned = replace(source, storage=I(-123), form_rates=(I(999),) * 10)
    rebuilt = assess_sine_c5_leaf_saddle(poisoned, cycle=range(5))
    assert rebuilt.full_tangent_generator == report.full_tangent_generator
    assert rebuilt.positive_root_bracket == report.positive_root_bracket


def test_tangent_is_the_derivative_of_full_nonlinear_nodal_rows(fixture):
    graph, _, report = fixture
    with mp.workdps(95):
        phase = tuple(2 * mp.pi * _mp(turn) for turn in report.target_phase_turns)
        zero = (mp.mpf(0),) * 10
        assert max(map(abs, _flow(graph, zero, phase))) < mp.mpf("1e-90")
        for column in range(20):

            def perturbed(t):
                form = list(zero)
                angles = list(phase)
                (form if column < 10 else angles)[column % 10] += t
                return _flow(graph, form, angles)

            for row in range(20):
                derivative = mp.diff(lambda t: perturbed(t)[row], 0)
                assert abs(
                    derivative - _mp(report.full_tangent_generator[row][column])
                ) < mp.mpf("1e-85")
        metric = report.full_energy_metric
        weighted = tuple(
            tuple(
                sum(
                    metric[i][k] * report.full_tangent_generator[k][j]
                    for k in range(20)
                )
                for j in range(20)
            )
            for i in range(20)
        )
        assert all(
            weighted[i][j] == -weighted[j][i] for i in range(20) for j in range(20)
        )


def test_characteristic_polynomial_and_root_are_exact_full_row_consequences(fixture):
    report = fixture[2]
    coefficients = report.odd_characteristic_coefficients
    matrix = report.odd_phase_acceleration
    for value in (Q(-3), Q(-1), Q(0), Q(1, 10), Q(1)):
        pencil = tuple(
            tuple(value * (i == j) - matrix[i][j] for j in range(4)) for i in range(4)
        )
        polynomial = sum(
            coefficient * value ** (4 - i) for i, coefficient in enumerate(coefficients)
        )
        assert polynomial == _determinant(pencil)
    assert tuple(324 * value for value in coefficients) == (324, 1404, 1527, 61, -15)
    lower, upper = report.positive_root_bracket
    assert 0 < lower < upper
    assert (
        report.positive_root_endpoint_values[0]
        < 0
        < report.positive_root_endpoint_values[1]
    )
    assert (
        all(coefficient > 0 for coefficient in coefficients[:-1])
        and coefficients[-1] < 0
    )
    schur = report.odd_leaf_schur_complement
    assert _determinant(schur) < 0
    assert report.odd_phase_hessian_inertia == (3, 1, 0)
    assert report.even_phase_hessian_inertia == (5, 0, 1)
    assert report.relative_phase_hessian_inertia == (8, 1, 0)


def test_correlated_growing_direction_satisfies_actual_nodal_eigen_equation(fixture):
    report = fixture[2]
    with mp.workdps(95):
        coefficients = tuple(map(_mp, report.odd_characteristic_coefficients))
        lam = mp.findroot(
            lambda x: mp.polyval(coefficients, x), (mp.mpf(".07"), mp.mpf(".09"))
        )
        rate = mp.sqrt(lam)
        second = (4 * lam + 1) / (18 * lam**2 + 43 * lam + 5)
        phase = (
            mp.mpf(1),
            second,
            (7 - second) / (6 * lam + 8),
            (10 * second - 1) / (6 * lam + 8),
        )
        form = tuple(
            -sum(_mp(a) * b for a, b in zip(row, phase)) / rate
            for row in report.odd_negative_form_from_phase
        )
        direction = tuple(
            sum(_mp(a) * b for a, b in zip(row, form))
            for row in report.odd_reconstruction
        ) + tuple(
            sum(_mp(a) * b for a, b in zip(row, phase))
            for row in report.odd_reconstruction
        )
        for value, bound in zip(direction, report.unstable_direction_bounds):
            assert _mp(bound.lo) <= value <= _mp(bound.hi)
        residual = tuple(
            sum(_mp(a) * b for a, b in zip(row, direction)) - rate * value
            for row, value in zip(report.full_tangent_generator, direction)
        )
        assert max(map(abs, residual)) < mp.mpf("1e-85")
    assert report.unstable_cycle_gap_direction_bounds[-1] == I(2)
    assert all(
        value.hi < 0 for value in report.unstable_cycle_gap_direction_bounds[:-1]
    )


def test_eigenvector_rational_functions_solve_the_exact_quartic(fixture):
    report = fixture[2]

    def multiply(left, right):
        result = [Q(0)] * (len(left) + len(right) - 1)
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                result[i + j] += a * b
        return tuple(result)

    # Build the common numerator from the independent rational formula:
    # b=(1+4λ)/(5+43λ+18λ²), c=(7-b)/(8+6λ), d=(10b-1)/(8+6λ).
    denominator_b, numerator_b, factor = (
        tuple(map(Q, (5, 43, 18))),
        tuple(map(Q, (1, 4))),
        tuple(map(Q, (8, 6))),
    )
    denominator = multiply(denominator_b, factor)
    padded_b = numerator_b + (Q(0),)
    rows = (
        denominator,
        multiply(numerator_b, factor) + (Q(0),),
        tuple(7 * a - b for a, b in zip(denominator_b, padded_b)) + (Q(0),),
        tuple(10 * b - a for a, b in zip(denominator_b, padded_b)) + (Q(0),),
    )
    assert rows == report.unstable_phase_numerator_coefficients
    assert denominator == report.unstable_phase_denominator_coefficients
    polynomial = tuple(reversed(report.odd_characteristic_coefficients))
    for i, matrix_row in enumerate(report.odd_phase_acceleration):
        residual = tuple(
            sum(matrix_row[j] * (rows[j][k] if k < 4 else Q(0)) for j in range(4))
            - (rows[i][k - 1] if k else Q(0))
            for k in range(5)
        )
        assert residual == tuple(
            residual[4] * coefficient for coefficient in polynomial
        )


def test_relabeling_and_source_order_preserve_the_declared_cycle_coordinates(fixture):
    original = fixture[2]
    labels = tuple(("node", i) for i in range(10))
    _, source, cycle = _source(order=(8, 2, 6, 4, 0, 9, 3, 5, 1, 7), labels=labels)
    report = assess_sine_c5_leaf_saddle(source, cycle=cycle)
    assert report.odd_phase_from_form == original.odd_phase_from_form
    assert report.odd_negative_form_from_phase == original.odd_negative_form_from_phase
    assert (
        report.odd_characteristic_coefficients
        == original.odd_characteristic_coefficients
    )
    for i, label_i in enumerate(source.nodes):
        for j, label_j in enumerate(source.nodes):
            assert (
                report.form_laplacian[i][j]
                == original.form_laplacian[label_i[1]][label_j[1]]
            )
            assert (
                report.phase_hessian[i][j]
                == original.phase_hessian[label_i[1]][label_j[1]]
            )
    assert report.to_dict()["schema"] == "tnfr.sine-c5-leaf-saddle.v1"


def test_sdk_projection_keeps_exact_evidence_and_checks_nested_geometry_labels(fixture):
    from tnfr.sdk import relational_report_to_dict

    report = fixture[2]
    projection = relational_report_to_dict(report)
    assert projection["report_type"] == "SineC5LeafSaddle"
    assert projection["report"] == report.to_dict()["report"]

    @dataclass(frozen=True)
    class Opaque:
        index: int

    geometry = report.target_geometry.geometry
    altered = replace(
        report,
        target_geometry=replace(
            report.target_geometry,
            geometry=replace(geometry, nodes=(Opaque(0),) + geometry.nodes[1:]),
        ),
    )
    # Generic dataclass projection would otherwise serialize an opaque label as
    # ordinary evidence, silently losing its graph identity inside the target.
    for export in (
        lambda: altered.to_dict(),
        lambda: relational_report_to_dict(altered),
    ):
        with pytest.raises(TypeError):
            export()


@pytest.mark.parametrize("capacity", [True, Q(0), Q(1) + Q(1, 2**120)])
def test_actual_capacity_domain_is_required(fixture, capacity):
    source = fixture[1]
    with pytest.raises((TypeError, ValueError)):
        assess_sine_c5_leaf_saddle(
            replace(source, capacity=(capacity,) + source.capacity[1:]), cycle=range(5)
        )


@pytest.mark.parametrize("edge", [(0, 2), (5, 6)])
def test_other_support_cannot_inherit_the_saddle(edge):
    _, source, cycle = _source(extra_edge=edge)
    with pytest.raises(ValueError):
        assess_sine_c5_leaf_saddle(source, cycle=cycle)


def test_conservative_law_and_cycle_admission(fixture):
    source = fixture[1]
    for field in ("epi_weight", "phase_weight", "storage_scale"):
        model = replace(source.reference_model)
        # Public construction normalizes weights; the consumed declared fields
        # still require admission before this report can use them as premises.
        object.__setattr__(model, field, Q(1, 2))
        with pytest.raises(ValueError):
            assess_sine_c5_leaf_saddle(
                replace(source, reference_model=model), cycle=range(5)
            )
    with pytest.raises(ValueError):
        assess_sine_c5_leaf_saddle(source, cycle=(0, 1, 2, 3, 3))
    with pytest.raises(ValueError):
        assess_sine_c5_leaf_saddle(source, cycle=(0, 2, 1, 3, 4))


def test_instantaneous_nonlinear_remainder_is_quadratic_and_phase_row_exact(fixture):
    graph, _, report = fixture
    perturbation = tuple(Q((3 * i) % 7 - 3, 1024) for i in range(20))
    linear = _matvec(report.full_tangent_generator, perturbation)
    radius = max(map(abs, perturbation[10:]))
    with mp.workdps(95):
        form = tuple(map(_mp, perturbation[:10]))
        phase = tuple(
            2 * mp.pi * _mp(turn) + _mp(error)
            for turn, error in zip(report.target_phase_turns, perturbation[10:])
        )
        full = _flow(graph, form, phase)
        assert max(abs(full[i] - _mp(linear[i])) for i in range(10)) < _mp(
            2 * radius**2
        )
        assert max(abs(full[i] - _mp(linear[i])) for i in range(10, 20)) < mp.mpf(
            "1e-90"
        )


def test_local_passage_constants_and_sector_direction_without_a_trajectory(fixture):
    report = fixture[2]
    phase, form = (
        report.unstable_direction_bounds[10:],
        report.unstable_direction_bounds[:10],
    )
    assert max(value.abs_max for value in phase) == 1
    assert max(value.abs_max for value in form) < 1
    assert report.growth_rate_bounds.lo > Q(7, 25)
    epsilon = Q(1, 2**24)
    # e^(3*sigma)>1+.84+.84^2/2>2 implies e^(3*sigma)-2e^(-3*sigma)>1.
    assert 1 + Q(21, 25) + Q(21, 25) ** 2 / 2 > 2
    remainder = 9 * epsilon**2 * 729 * 728
    assert remainder < epsilon
    assert 2187 * epsilon < Q(3, 12)  # pi>3, without a rounded pi comparison.
    # The declared h0=epsilon*(3X,-v) starts on the inside of the 2pi/3 face.
    # At tau=3 its linear phase exceeds epsilon*v; each endpoint error is
    # smaller than epsilon, so the closing gap has crossed outwards.
    assert report.unstable_cycle_gap_direction_bounds[-1] == I(2)
    assert 2 * epsilon - 2 * remainder > 0
