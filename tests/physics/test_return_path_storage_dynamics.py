"""Full nodal checks of the fixed return-path storage-family equilibrium."""

from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.phase_cycle_geometry import assess_return_path_storage_geometry
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

COEFFICIENTS = (Q(0), Q(1), Q(4))
CYCLES = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (0, 10, 5, 6, 1))


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _fraction(value):
    return Q(mp.nstr(value, 85))


def _graph():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 10), (10, 5), (1, 6)))
    return graph


@pytest.fixture(scope="module")
def reports():
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    return {
        epsilon: assess_return_path_storage_geometry(
            source,
            left_cycle=CYCLES[0],
            right_cycle=CYCLES[1],
            mediator=10,
            epsilon=epsilon,
        )
        for epsilon in COEFFICIENTS
    }


def _current(angle, epsilon):
    sine = mp.sin(angle)
    return sine + epsilon * sine**3


def _residual(angle, epsilon):
    return (
        _current(mp.pi / 2 - angle / 4, epsilon)
        - _current(angle, epsilon)
        - _current(2 * angle / 3, epsilon)
    )


def _angles(angle):
    bulk, joining = mp.pi / 2 - angle / 4, 2 * angle / 3
    return (
        0,
        angle,
        angle + bulk,
        angle + 2 * bulk,
        angle + 3 * bulk,
        2 * joining,
        2 * joining - angle,
        2 * joining - angle - bulk,
        2 * joining - angle - 2 * bulk,
        2 * joining - angle - 3 * bulk,
        joining,
    )


@pytest.mark.parametrize("epsilon", COEFFICIENTS)
def test_certified_geometry_contains_independent_full_nodal_equilibrium(
    reports, epsilon
):
    report, graph = reports[epsilon], _graph()
    with mp.workdps(95):
        coefficient = _mp(epsilon)
        root = mp.findroot(
            lambda angle: _residual(angle, coefficient), (mp.pi / 6, 2 * mp.pi / 5)
        )
        root_turn = root / (2 * mp.pi)
        bracket = report.root_turn_bracket
        assert _mp(bracket.lower) < root_turn < _mp(bracket.upper)
        assert bracket.lower_residual.lo > 0 > bracket.upper_residual.hi
        assert mp.pi / 6 < root < 2 * mp.pi / 5
        angles = _angles(root)
        currents = [
            sum(_current(angles[j] - angles[i], coefficient) for j in graph[i])
            for i in range(11)
        ]
        assert max(map(abs, currents)) < mp.mpf("1e-85")
        for i, value in enumerate(angles):
            intercept, slope = report.nodal_turn_affine_coefficients[i]
            assert abs(
                value / (2 * mp.pi) - _mp(intercept) - _mp(slope) * root_turn
            ) < mp.mpf("1e-85")
            assert report.nodal_turn_bounds[i].contains(_fraction(value / (2 * mp.pi)))
            assert report.nodal_current_residual_bounds[i].contains(0)

        hessian = mp.matrix(11)
        storage = mp.mpf(0)
        for index, (i, j) in enumerate(report.geometry.edges):
            gap = angles[j] - angles[i]
            turn = (gap / (2 * mp.pi) + mp.mpf("0.5")) % 1 - mp.mpf("0.5")
            assert abs(turn) < mp.mpf("0.25")
            assert report.edge_turn_bounds[index].contains(_fraction(turn))
            assert report.edge_current_bounds[index].contains(
                _fraction(_current(gap, coefficient))
            )
            curvature = mp.cos(gap) * (1 + 3 * coefficient * mp.sin(gap) ** 2)
            assert curvature > 0
            assert report.edge_curvature_bounds[index].lo > 0
            assert report.edge_curvature_bounds[index].contains(_fraction(curvature))
            hessian[i, i] += curvature
            hessian[j, j] += curvature
            hessian[i, j] -= curvature
            hessian[j, i] -= curvature
            cosine = mp.cos(gap)
            storage += (
                1 - cosine + coefficient * (mp.mpf(2) / 3 - cosine + cosine**3 / 3)
            )
        assert report.target_phase_storage_bounds.contains(_fraction(storage))
        assert report.minimum_acute_margin_turns_bounds.contains(
            _fraction(root_turn / 4)
        )
        # Removing one common phase origin gives a positive principal Hessian.
        # This finite check accompanies the connected positive-edge proof;
        # it does not supply an attraction or capture theorem.
        assert mp.eigsy(hessian[:10, :10], eigvals_only=True)[0] > 0


def test_periods_hold_for_the_entire_correlated_affine_root_family(reports):
    for report in reports.values():
        edge_map = dict(
            zip(report.geometry.edges, report.edge_turn_affine_coefficients)
        )
        for cycle, expected in zip(CYCLES, (1, -1, 0)):
            intercept = slope = Q(0)
            for i, j in zip(cycle, cycle[1:] + cycle[:1]):
                sign = 1 if i < j else -1
                constant, linear = edge_map[tuple(sorted((i, j)))]
                intercept += sign * constant
                slope += sign * linear
            assert (intercept, slope) == (expected, 0)
        assert report.named_cycle_periods == (1, -1, 0)
        # A midpoint is one rational phase reconstruction, not the exact root.
        with mp.workdps(95):
            bracket = report.root_turn_bracket
            midpoint = _mp((bracket.lower + bracket.upper) / 2)
            midpoint_residual = _residual(2 * mp.pi * midpoint, _mp(report.epsilon))
            assert abs(midpoint_residual) > mp.mpf("1e-70")


def test_complete_nodal_current_reduction_is_exact_before_solving_the_root():
    s = pytest.importorskip("sympy")
    t = s.Symbol("t", real=True)
    epsilon = s.Symbol("epsilon", nonnegative=True)
    bulk, joining = s.pi / 2 - t / 4, 2 * t / 3
    angles = (
        0,
        t,
        t + bulk,
        t + 2 * bulk,
        t + 3 * bulk,
        2 * joining,
        2 * joining - t,
        2 * joining - t - bulk,
        2 * joining - t - 2 * bulk,
        2 * joining - t - 3 * bulk,
        joining,
    )

    def current(value):
        return s.sin(value) + epsilon * s.sin(value) ** 3

    residual = current(bulk) - current(t) - current(joining)
    graph = _graph()
    expected_signs = (-1, 1, 0, 0, 0, 1, -1, 0, 0, 0, 0)
    for i, sign in enumerate(expected_signs):
        nodal = sum(current(angles[j] - angles[i]) for j in graph[i])
        assert s.trigsimp(nodal - sign * residual) == 0
    # This derives the scalar reduction from every nodal equation; a period
    # match or an imposed reflected ansatz alone would not establish this.


def test_branch_deforms_monotonically_without_reusing_the_old_sine_bracket(reports):
    with mp.workdps(95):
        roots = []
        for epsilon in COEFFICIENTS:
            coefficient = _mp(epsilon)
            root = mp.findroot(
                lambda t: _residual(t, coefficient), (mp.pi / 6, 2 * mp.pi / 5)
            )
            a, b, c = mp.cos(root / 4), mp.sin(root), mp.sin(2 * root / 3)
            numerator = a**3 - b**3 - c**3

            def curvature(angle):
                return mp.cos(angle) * (1 + 3 * coefficient * mp.sin(angle) ** 2)

            denominator = (
                curvature(mp.pi / 2 - root / 4) / 4
                + curvature(root)
                + 2 * curvature(2 * root / 3) / 3
            )
            assert numerator > 0
            assert denominator > 0
            assert numerator / denominator > 0  # dt_epsilon/d epsilon.
            if epsilon == 0:
                assert abs(numerator - 3 * b * c * (b + c)) < mp.mpf("1e-85")
                assert float(root) == pytest.approx(0.6236341875541443, abs=1e-12)
            roots.append(root)
        assert roots[0] < mp.pi / 4 < roots[1] < roots[2] < 2 * mp.pi / 5
        assert reports[1].root_turn_bracket.lower > Q(1, 8)
