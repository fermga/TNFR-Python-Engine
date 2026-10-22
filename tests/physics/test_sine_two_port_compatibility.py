"""Independent full-support controls for two-port C9 target compatibility.

Static controls require no root assessment. Reserved report fixtures are
current-source regressions only after the first archived evaluation.
"""

from fractions import Fraction as Q
from inspect import signature
from pathlib import Path
from typing import get_type_hints

import mpmath
import networkx as nx
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_two_port_compatibility as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 85
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _graph():
    graph = nx.disjoint_union(nx.cycle_graph(9), nx.cycle_graph(9))
    graph.add_edges_from(((0, 9), (1, 10)))
    return graph


def _affine(row, a, c):
    return row[0] + row[1] * a + row[2] * c


def _ideal_turns(classes, a, c):
    delta = (a - c) / 2
    result = []
    zero = a * 0
    for k, short, origin in zip(classes, (a, c), (zero, delta)):
        bulk = (k - short) / 8
        result.extend([origin] + [origin + short + (j - 1) * bulk for j in range(1, 9)])
    graph = _graph()
    mean = sum((graph.degree[i] * result[i] for i in graph), zero) / 40
    return tuple(value - mean for value in result)


class TestStaticGeometry:
    @pytest.mark.parametrize("classes", [(1, 1), (2, 2), (1, 2), (2, 1)])
    @pytest.mark.parametrize("a,c", [(Q(1, 6), Q(1, 7)), (Q(1, 5), Q(3, 20))])
    def test_full_lift_mean_and_integer_periods(self, classes, a, c):
        geometry = owner._derive(owner._NODES, owner._EDGES)
        degrees, mean, nodes, edges, offsets = owner._affine_geometry(classes, geometry)
        graph = _graph()
        assert geometry.edges == tuple(
            sorted(tuple(sorted(edge)) for edge in graph.edges)
        )
        assert degrees == tuple(graph.degree[i] for i in graph)
        assert sum(degrees) == 40
        assert mean == (Q(7 * sum(classes), 40), Q(1, 2), Q(0))
        assert tuple(_affine(row, a, c) for row in nodes) == _ideal_turns(classes, a, c)
        for column in range(3):
            assert (
                sum((Q(degrees[i]) * row[column] for i, row in enumerate(nodes)), Q(0))
                == 0
            )
        for (i, j), edge, offset in zip(geometry.edges, edges, offsets):
            assert (
                tuple(
                    nodes[j][k] - nodes[i][k] - (offset if k == 0 else 0)
                    for k in range(3)
                )
                == edge
            )
        lookup = {edge: row for edge, row in zip(geometry.edges, edges)}
        cycles = (tuple(range(9)), tuple(range(9, 18)), (0, 9, 10, 1))
        for cycle, period in zip(cycles, (*classes, 0)):
            total = [Q(0)] * 3
            for i, j in zip(cycle, cycle[1:] + cycle[:1]):
                row = lookup[tuple(sorted((i, j)))]
                for column in range(3):
                    total[column] += (1 if i < j else -1) * row[column]
            assert total == [Q(period), Q(0), Q(0)]

    def test_named_cycles_are_an_integer_basis_not_only_real_independent(self):
        graph = _graph()
        chords = ((0, 8), (9, 17), (1, 10))
        graph.remove_edges_from(chords)
        assert nx.is_tree(graph)
        # The named cycle/chord minor is diagonal with unit signed entries.
        cycles = (tuple(range(9)), tuple(range(9, 18)), (0, 9, 10, 1))
        minor = []
        for cycle in cycles:
            directed = tuple(zip(cycle, cycle[1:] + cycle[:1]))
            minor.append(
                tuple(
                    int(edge in directed) - int(edge[::-1] in directed)
                    for edge in chords
                )
            )
        assert minor == [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]

    @pytest.mark.parametrize("classes", [(1, 1), (1, 2), (2, 1), (2, 2)])
    def test_full_fine_current_factorization_at_noncritical_states(self, classes, mp):
        geometry = owner._derive(owner._NODES, owner._EDGES)
        edge_symbols, nodal, equations, factors = owner._current_factorization(geometry)
        a, c = Q(1, 6), Q(3, 20)
        theta = tuple(
            2 * mp.pi * _mp(mp, value) for value in _ideal_turns(classes, a, c)
        )
        graph = _graph()
        fine = tuple(sum(mp.sin(theta[j] - theta[i]) for j in graph[i]) for i in graph)
        symbols = tuple(
            mp.sin(2 * mp.pi * _mp(mp, value))
            for value in (a, (classes[0] - a) / 8, c, (classes[1] - c) / 8, (a - c) / 2)
        )
        residuals = tuple(
            sum(_mp(mp, q) * value for q, value in zip(row, symbols))
            for row in equations
        )
        assert max(abs(value) for value in fine) > mp.mpf("0.01")
        for index, actual in enumerate(fine):
            symbolic = sum(
                _mp(mp, q) * value for q, value in zip(nodal[index], symbols)
            )
            factored = sum(
                _mp(mp, q) * value for q, value in zip(factors[index], residuals)
            )
            assert abs(actual - symbolic) < mp.mpf("1e-78")
            assert abs(actual - factored) < mp.mpf("1e-78")
        for (i, j), row in zip(geometry.edges, edge_symbols):
            symbolic = sum(_mp(mp, q) * value for q, value in zip(row, symbols))
            assert abs(symbolic - mp.sin(theta[j] - theta[i])) < mp.mpf("1e-78")

    def test_gap_uses_complete_laplacian_and_true_port_degrees(self, mp):
        graph = _graph()
        geometry = owner._derive(owner._NODES, owner._EDGES)
        degrees = tuple(graph.degree[i] for i in graph)
        laplacian, diameter, gap, shifted = owner._laplacian_gap(geometry, degrees)
        assert diameter == nx.diameter(graph)
        assert gap == Q(4, 18 * diameter)
        assert {i for i, degree in enumerate(degrees) if degree == 3} == {0, 1, 9, 10}
        for i in graph:
            for j in graph:
                assert laplacian[i][j] == (
                    degrees[i] if i == j else -int(graph.has_edge(i, j))
                )
                assert shifted[i][j] == laplacian[i][j] - gap * (int(i == j) - Q(1, 18))
        spectrum = mp.eigsy(
            mp.matrix([[int(value) for value in row] for row in laplacian]),
            eigvals_only=True,
        )
        assert abs(spectrum[0]) < mp.mpf("1e-78")
        assert spectrum[1] > _mp(mp, gap)

    def test_affine_outer_and_inner_equations_have_the_required_monotonicity(self, mp):
        # Synthetic interior values, not an equilibrium solve or reserved response.
        for k, turn in ((1, Q(1, 7)), (2, Q(1, 6))):
            angle = _mp(mp, turn)
            derivative = (
                -2
                * mp.pi
                * (mp.cos(2 * mp.pi * (k - angle) / 8) / 8 + mp.cos(2 * mp.pi * angle))
            )
            assert derivative < 0
        # h2(1/9) < -h1(2/9) admits the entire inner root domain.
        difference = (
            2 * mp.sin(mp.pi / 3) * (mp.cos(mp.pi / 9) - mp.cos(5 * mp.pi / 36))
        )
        assert difference > 0


def _assess(classes=(2, 1), **changes):
    return owner.assess_sine_two_port_compatibility(
        **(dict(classes=classes, outer_refinements=32, inner_refinements=64) | changes)
    )


@pytest.fixture(scope="module")
def primary():
    return _assess()


@pytest.fixture(scope="module")
def matched():
    return _assess((1, 1))


@pytest.fixture(scope="module")
def reversed_pair():
    return _assess((1, 2))


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_formation_assessment():
    from tnfr.physics import relational_sine_forecast, relational_sine_formed_classes

    def forbidden(*args, **kwargs):
        pytest.fail(
            "target compatibility cannot assess a preparation or evolve a state"
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational_sine_forecast, "bound_sine_flow", forbidden)
        patch.setattr(
            relational_sine_formed_classes, "assess_sine_formed_class_pair", forbidden
        )
        yield


def test_root_enclosures_against_independent_coupled_high_precision_equations(
    primary, mp
):
    def equations(a, c):
        delta = (a - c) / 2
        return (
            mp.sin((4 * mp.pi - a) / 8) - mp.sin(a) - mp.sin(delta),
            mp.sin((2 * mp.pi - c) / 8) - mp.sin(c) + mp.sin(delta),
        )

    a, c = mp.findroot(equations, (mp.mpf("1.1"), mp.mpf("0.9")))
    for bounds, angle in zip(primary.short_arc_turn_bounds, (a, c)):
        _contains(mp, bounds, angle / (2 * mp.pi))
    assert primary.canonical_root_turn_bracket.lower_residual.lo > 0
    assert primary.canonical_root_turn_bracket.upper_residual.hi < 0
    assert (
        primary.canonical_root_turn_bracket.upper
        - primary.canonical_root_turn_bracket.lower
        == Q(1, 9 * 2**32)
    )
    assert primary.bridge_turn_bounds[0].lo > 0
    assert primary.uniform_pair_excluded and not primary.uniform_pair_compatible
    assert primary.status == "certified_compatible"
    assert primary.unavailable_reasons == ()


def test_every_full_nodal_row_and_hessian_enclose_the_implicit_target(primary, mp):
    def equations(a, c):
        return (
            mp.sin((4 * mp.pi - a) / 8) - mp.sin(a) - mp.sin((a - c) / 2),
            mp.sin((2 * mp.pi - c) / 8) - mp.sin(c) + mp.sin((a - c) / 2),
        )

    a, c = mp.findroot(equations, (mp.mpf("1.1"), mp.mpf("0.9")))
    turns = _ideal_turns((2, 1), a / (2 * mp.pi), c / (2 * mp.pi))
    theta = tuple(2 * mp.pi * value for value in turns)
    graph = _graph()
    hessian = mp.zeros(18)
    storage = mp.mpf(0)
    edge_turns = []
    for edge, current, cosine in zip(
        primary.geometry.edges, primary.edge_current_bounds, primary.edge_cosine_bounds
    ):
        i, j = edge
        sine, weight = mp.sin(theta[j] - theta[i]), mp.cos(theta[j] - theta[i])
        storage += 1 - weight
        edge_turns.append(
            (theta[j] - theta[i]) / (2 * mp.pi)
            - primary.edge_integer_offsets[primary.geometry.edges.index(edge)]
        )
        _contains(mp, current, sine)
        _contains(mp, cosine, weight)
        hessian[i, i] += weight
        hessian[j, j] += weight
        hessian[i, j] -= weight
        hessian[j, i] -= weight
    for i in graph:
        _contains(mp, primary.target_phase_bounds[i], theta[i])
        value = sum(mp.sin(theta[j] - theta[i]) for j in graph[i])
        _contains(mp, primary.nodal_current_residual_bounds[i], value)
        _contains(
            mp,
            primary.target_form_rate_bounds[i],
            value / (1023 * mp.pi * graph.degree[i]),
        )
        assert primary.target_phase_rate_bounds[i] == I(0)
        for j in graph:
            _contains(mp, primary.phase_hessian_bounds[i][j], hessian[i, j])
    spectrum = mp.eigsy(hessian, eigvals_only=True)
    assert spectrum[1] > _mp(mp, primary.phase_hessian_gap_lower_bound) > 0
    _contains(mp, primary.target_phase_storage_bounds, storage)
    _contains(
        mp, primary.acute_margin_turns_bounds, mp.mpf(1) / 4 - max(map(abs, edge_turns))
    )
    assert abs(sum(graph.degree[i] * theta[i] for i in graph)) < mp.mpf("1e-78")
    assert primary.local_attraction_certified


def test_matched_classes_retain_exact_zero_contact_and_rational_turn_correlations(
    matched,
):
    assert (
        matched.short_arc_turn_bounds
        == matched.bulk_arc_turn_bounds
        == (I(Q(1, 9)),) * 2
    )
    assert matched.bridge_turn_bounds == (I(0), I(0))
    for edge in ((0, 9), (1, 10)):
        index = matched.geometry.edges.index(edge)
        assert (
            matched.edge_turn_bounds[index]
            == matched.edge_current_bounds[index]
            == I(0)
        )
    assert matched.canonical_root_turn_bracket is None
    assert matched.inner_root_brackets_at_outer_endpoints is None
    assert matched.uniform_pair_compatible and not matched.uniform_pair_excluded
    assert matched.uniform_pair_nodal_current_bounds == (I(0),) * 18
    assert matched.implicit_equilibrium_certified and matched.local_attraction_certified


def test_component_exchange_preserves_complete_target_and_reverses_contact(
    primary, reversed_pair
):
    assert reversed_pair.short_arc_turn_bounds == primary.short_arc_turn_bounds[::-1]
    assert reversed_pair.bridge_turn_bounds == primary.bridge_turn_bounds[::-1]
    assert (
        reversed_pair.canonical_root_turn_bracket == primary.canonical_root_turn_bracket
    )
    assert reversed_pair.status == primary.status
    # Affine means are imposed globally; component swap is exact before interval evaluation.
    a, c = Q(1, 6), Q(1, 7)
    original = _ideal_turns((2, 1), a, c)
    swapped = _ideal_turns((1, 2), c, a)
    assert swapped == original[9:] + original[:9]


def test_matched_second_class_needs_no_numerical_root(monkeypatch):
    def forbidden(*args):
        pytest.fail("matched classes have exact rational-turn geometry")

    monkeypatch.setattr(owner, "_root_enclosures", forbidden)
    report = _assess((2, 2), outer_refinements=1, inner_refinements=1)
    assert report.status == "certified_compatible"
    assert report.short_arc_turn_bounds == (I(Q(2, 9)),) * 2


@pytest.mark.parametrize(
    "classes",
    [
        (),
        (1,),
        (1, 2, 1),
        (True, 1),
        (1.0, 2),
        (Q(1), 2),
        (0, 1),
        (1, 3),
        ("1", 2),
        None,
    ],
)
def test_invalid_classes_reject_before_geometry_or_root(classes, monkeypatch):
    monkeypatch.setattr(
        owner, "_derive", lambda *args: pytest.fail("invalid input reached geometry")
    )
    with pytest.raises((TypeError, ValueError)):
        _assess(classes)


@pytest.mark.parametrize("name", ["outer_refinements", "inner_refinements"])
@pytest.mark.parametrize("value", [True, 0, -1, 65, 1.0, Q(1), "32", None])
def test_invalid_budgets_reject_before_geometry_or_root(name, value, monkeypatch):
    monkeypatch.setattr(
        owner, "_derive", lambda *args: pytest.fail("invalid input reached geometry")
    )
    with pytest.raises((TypeError, ValueError)):
        _assess(**{name: value})


def test_unresolved_root_does_not_promote_partial_or_midpoint_target(monkeypatch):
    def unresolved(*args):
        raise ArithmeticError("synthetic interval-sign uncertainty")

    monkeypatch.setattr(owner, "_root_enclosures", unresolved)
    report = _assess()
    assert report.status == "unavailable"
    assert report.unavailable_reasons == (
        "strict_root_sign_unresolved_within_declared_budget",
    )
    assert not report.implicit_equilibrium_certified
    assert not report.local_attraction_certified
    for name in (
        "short_arc_turn_bounds",
        "nodal_turn_bounds",
        "target_phase_bounds",
        "edge_current_bounds",
        "phase_hessian_bounds",
        "phase_hessian_gap_lower_bound",
    ):
        assert getattr(report, name) is None
    assert report.uniform_pair_excluded


def test_unresolved_cosine_lower_bound_cannot_be_clipped_to_a_psd_claim(monkeypatch):
    monkeypatch.setattr(owner, "cos", lambda value: I(-1, 1))
    report = _assess((1, 1))
    assert report.implicit_equilibrium_certified
    assert report.minimum_cosine_lower_bound == -1
    assert report.phase_hessian_gap_lower_bound is None
    assert not report.local_attraction_certified
    assert report.status == "unavailable"
    assert report.unavailable_reasons == ("strict_acute_geometry_not_certified",)


def test_exact_json_and_generic_sdk_projection(primary, tmp_path):
    direct = primary.to_dict()
    generic = relational_report_to_dict(primary)
    assert direct["schema"] == "tnfr.sine-two-port-compatibility.v1"
    assert generic["report"] == direct["report"]
    assert generic["report_type"] == "SineTwoPortCompatibility"
    path = tmp_path / "two-port.json"
    export_to_json(primary, path)
    assert json_loads(path.read_text(encoding="utf-8")) == direct
    assert get_type_hints(owner.SineTwoPortCompatibility)
    assert all(
        parameter.default is parameter.empty
        for parameter in signature(
            owner.assess_sine_two_port_compatibility
        ).parameters.values()
    )


def test_first_reserved_report_body_is_preserved(primary, matched):
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/sine_formed_classes/two-port-compatibility-v1.json"
    )
    saved = json_loads(path.read_text(encoding="utf-8"))
    # The producer owns its outer envelope; compare retained scientific reports only.
    assert {"schema": saved["schema"], "report": saved["report"]} == primary.to_dict()
    assert saved["matched_control"] == matched.to_dict()
