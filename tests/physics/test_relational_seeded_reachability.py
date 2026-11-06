"""Static native-rate and support controls, not formation trajectories."""

import math
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    evaluate_relational_uniform_tangent,
)
from tnfr.mathematics._rational_interval import I, atan, cos, pi_interval, sin


def _graph(phases=None, forms=None, capacities=None):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    if phases is None:
        phases = tuple(math.tau * j / 5 for j in range(5)) + (6 * math.pi / 5,) * 5
    for i in graph:
        graph.nodes[i].update(
            theta=phases[i],
            EPI=0.0 if forms is None else forms[i],
            nu_f=1.0 if capacities is None else capacities[i],
        )
    return graph


@pytest.fixture(scope="module")
def initial():
    pi = pi_interval()
    c, d, s = cos(2 * pi / 5), cos(pi / 5), sin(pi / 5)
    r = atan(s / (2 - d)) / pi
    source = (I(Q(-3, 5)), I(Q(3, 5)), I(0), I(0), I(0), r, -r, I(0), I(0), I(0))
    metric = (5 * s / 3,) * 2 + (2 * pi * c,) * 3 + (s / r,) * 2 + (2 * pi,) * 3
    qdot = (
        -Q(6, 5) - r / 2,
        Q(6, 5) + r / 2,
        I(Q(-3, 10)),
        I(0),
        I(Q(3, 10)),
        2 * r + Q(3, 10),
        -2 * r - Q(3, 10),
        r / 2,
        I(0),
        -r / 2,
    )
    acceleration = tuple(q / (2 * h) for q, h in zip(qdot, metric))
    graph = _graph()
    tangent = evaluate_relational_uniform_tangent(
        graph, model=RelationalExchangeModel(1, phase_domain="regular")
    )
    return {
        "pi": pi,
        "c": c,
        "d": d,
        "s": s,
        "r": r,
        "source": source,
        "metric": metric,
        "qdot": qdot,
        "acceleration": acceleration,
        "graph": graph,
        "tangent": tangent,
    }


def _midpoints(values):
    return np.array([float(value.midpoint) for value in values])


def test_frozen_initial_rates_have_independent_exact_trigonometric_bounds(initial):
    field = initial["tangent"].field
    r = initial["r"]
    assert Q(1459, 10000) < r.lo < r.hi < Q(1460, 10000)
    assert all(value.lo > 0 for value in initial["metric"])
    assert (2 * initial["c"] - initial["d"]).hi < 0
    np.testing.assert_allclose(
        field.phase_source, _midpoints(initial["source"]), atol=2e-15
    )
    np.testing.assert_allclose(
        field.phase_metric, _midpoints(initial["metric"]), rtol=2e-14
    )
    np.testing.assert_allclose(
        field.form_rate, _midpoints(initial["source"]) / 2, atol=2e-15
    )
    assert field.phase_rate == (0.0,) * 10
    assert field.form_storage == field.continuous_loss == 0


def test_native_tangent_times_field_recovers_the_first_nonzero_phase_change(initial):
    tangent = initial["tangent"]
    field = tangent.field
    degrees = np.array([initial["graph"].degree(i) for i in range(10)])
    acceleration = np.asarray(tangent.generator) @ np.array(
        field.form_rate + field.phase_rate
    )
    expected_form = -_midpoints(initial["qdot"]) / (2 * degrees)
    np.testing.assert_allclose(acceleration[:10], expected_form, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(
        acceleration[10:], _midpoints(initial["acceleration"]), rtol=2e-14, atol=2e-15
    )
    assert initial["acceleration"][0].hi < 0
    assert initial["acceleration"][2].hi < 0
    assert initial["acceleration"][5].lo > 0
    assert initial["acceleration"][7].lo > 0


def test_source_argument_initially_accelerates_toward_the_excluded_branch(initial):
    c, d, s, pi = (initial[name] for name in ("c", "d", "s", "pi"))
    t = sin(2 * pi / 5)
    a2, b2, u2 = (initial["acceleration"][j] for j in (0, 2, 5))
    real, imaginary = 2 * c - d, -s
    real2 = (t - s) * a2 - t * b2 + s * u2
    imaginary2 = (d - 3 * c) * a2 - c * b2 - d * u2
    argument2 = (real * imaginary2 - imaginary * real2) / (real**2 + imaginary**2)
    assert Q(-120, 1000) < real2.lo < real2.hi < Q(-119, 1000)
    assert Q(41, 1000) < imaginary2.lo < imaginary2.hi < Q(42, 1000)
    assert Q(-205, 1000) < argument2.lo < argument2.hi < Q(-204, 1000)
    # The existing Arg Jacobian supplies a second independent expression.
    tangent = initial["tangent"]
    measured = math.pi * np.dot(
        tangent.phase_source_jacobian[0], _midpoints(initial["acceleration"])
    )
    assert measured == pytest.approx(float(argument2.midpoint), rel=2e-14)
    # This local sign cannot establish that a trajectory reaches the branch.


def test_storage_loss_starts_quadratically_with_a_strictly_positive_coefficient(
    initial,
):
    r = initial["r"]
    coefficient = Q(111, 200) + Q(4, 5) * r + Q(37, 24) * r**2
    degrees = [initial["graph"].degree(i) for i in range(10)]
    direct = sum(
        (q**2 / (2 * degree) for q, degree in zip(initial["qdot"], degrees)), I(0)
    )
    assert (direct - coefficient).contains(0)
    assert Q(704, 1000) < coefficient.lo < coefficient.hi < Q(705, 1000)
    # A static form ray checks the homogeneous loss formula. This is not a
    # sampled trajectory, nor an enclosure of a Taylor remainder in time.
    tangent = initial["tangent"]
    epsilon = Q(1, 16)
    probe = _graph(
        forms=tuple(float(epsilon) * value for value in tangent.field.form_rate)
    )
    field = evaluate_relational_exchange(probe, model=tangent.field.model)
    assert float(field.continuous_loss / epsilon**2) == pytest.approx(
        float(coefficient.midpoint), rel=2e-14
    )


def test_joint_reflection_is_a_native_covariance_for_nonuniform_form_and_capacity():
    reflection = (1, 0, 4, 3, 2, 6, 5, 9, 8, 7)
    graph = _graph()
    for i in graph:
        graph.nodes[i]["theta"] += (i - 4) / 200
        graph.nodes[i]["EPI"] = (i * i - 7) / 64
        graph.nodes[i]["nu_f"] = (i + 2) / 8
    reflected = _graph()
    for i, j in enumerate(reflection):
        reflected.nodes[i].update(
            EPI=-graph.nodes[j]["EPI"],
            theta=math.tau / 5 - graph.nodes[j]["theta"],
            nu_f=graph.nodes[j]["nu_f"],
        )
    model = RelationalExchangeModel(1, phase_domain="regular")
    original = evaluate_relational_exchange(graph, model=model)
    mirrored = evaluate_relational_exchange(reflected, model=model)
    for attribute in ("phase_source", "form_rate", "phase_rate"):
        expected = [-getattr(original, attribute)[j] for j in reflection]
        np.testing.assert_allclose(
            getattr(mirrored, attribute), expected, rtol=2e-14, atol=2e-15
        )
    np.testing.assert_allclose(
        mirrored.phase_metric,
        [original.phase_metric[j] for j in reflection],
        rtol=2e-14,
    )
    assert float(mirrored.storage) == pytest.approx(float(original.storage), rel=2e-14)
    assert float(mirrored.continuous_loss) == pytest.approx(
        float(original.continuous_loss), rel=2e-14
    )


def test_exact_represented_reflection_lift_matches_the_eight_coordinate_flow():
    # This exactly mirrored represented chart is a separate static control.
    # It does not project the separately rounded j*kappa preparation onto it.
    a, b, capital_a, capital_b = 2.5, 1.25, 0.25, 0.125
    u, v, capital_u, capital_v = 0.125, -0.0625, 0.25, 0.03125
    phases = (a, -a, -b, 0, b, capital_a, -capital_a, -capital_b, 0, capital_b)
    forms = (u, -u, -v, 0, v, capital_u, -capital_u, -capital_v, 0, capital_v)
    graph = _graph(phases, forms)
    field = evaluate_relational_exchange(
        graph, model=RelationalExchangeModel(1, phase_domain="regular")
    )
    assert field.form_rate[3] == field.form_rate[8] == 0
    assert field.phase_rate[3] == field.phase_rate[8] == 0
    for offset, first, second, other, x, y, other_x in (
        (0, a, b, capital_a, u, v, capital_u),
        (5, capital_a, capital_b, a, capital_u, capital_v, u),
    ):
        port_gaps = (-2 * first, second - first, other - first)
        interior_gaps = (first - second, -second)
        for index, gaps, gradient, degree in (
            (offset, port_gaps, 4 * x - y - other_x, 3),
            (offset + 4, interior_gaps, 2 * y - x, 2),
        ):
            real = sum(math.cos(gap) for gap in gaps)
            imaginary = sum(math.sin(gap) for gap in gaps)
            angle = math.atan2(imaginary, real)
            metric = math.pi * imaginary / angle if angle else math.pi * real
            assert field.form_gradient[index] == pytest.approx(gradient, abs=1e-15)
            assert field.form_rate[index] == pytest.approx(
                -gradient / (2 * degree) + angle / (2 * math.pi), abs=2e-15
            )
            assert field.phase_rate[index] == pytest.approx(
                gradient / (2 * metric), abs=2e-15
            )
        assert field.form_rate[offset] == -field.form_rate[offset + 1]
        assert field.form_rate[offset + 2] == -field.form_rate[offset + 4]
        assert field.phase_rate[offset] == -field.phase_rate[offset + 1]
        assert field.phase_rate[offset + 2] == -field.phase_rate[offset + 4]


def test_supplied_ports_lift_a_pure_receiver_skip_boundary_below_initial_storage(
    initial,
):
    pi = initial["pi"]
    source = (Q(4, 5), Q(-4, 5), Q(-2, 5), Q(0), Q(2, 5))
    receiver = (Q(3, 4), Q(-3, 4), Q(-1, 4), Q(0), Q(1, 4))
    phases = tuple(value * pi for value in source + receiver)
    graph = _graph(tuple(float(value.midpoint) for value in phases))
    field = evaluate_relational_exchange(
        graph, model=RelationalExchangeModel(1, phase_domain="positive_resultant")
    )
    # At each receiver port its two internal phasors cancel exactly; the
    # supplied bridge contributes exp(+/- i*pi/20), restoring regularity.
    internal_real = cos(-3 * pi / 2) + cos(-pi / 2)
    internal_imaginary = sin(-3 * pi / 2) + sin(-pi / 2)
    assert internal_real.contains(0) and internal_imaginary.contains(0)
    assert cos(pi / 20).lo > 0
    for j in (5, 6):
        assert field.relative_resultant[j][0] == pytest.approx(
            float(cos(pi / 20).midpoint), abs=2e-15
        )
    initial_storage = 5 * (1 - initial["c"]) + 2 + 2 * initial["d"]
    witness_storage = (
        5 * (1 - initial["c"]) + 5 - 2 * cos(pi / 4) + 2 * (1 - cos(pi / 20))
    )
    assert (initial_storage - witness_storage).lo > Q(3, 400)
    assert float(field.storage) == pytest.approx(
        float(witness_storage.midpoint), rel=2e-14
    )
    # A static sublevel witness does not prove a path from the initial state,
    # an ODE-selected crossing, an acute target, or protected maintenance.


def test_geometric_comparison_path_has_exact_storage_derivative_factorizations():
    symbolic = pytest.importorskip("sympy")
    v = symbolic.symbols("v", real=True)
    h = symbolic.cos(v)
    first = 7 - symbolic.cos(4 * v) - 4 * h - 2 * symbolic.cos(2 * v)
    ring = 5 - symbolic.cos(4 * v) - 4 * h
    expected_first = 2 * symbolic.sin(v) * (8 * h**3 - 2 * h + 1)
    expected_ring = 2 * symbolic.sin(v) * (2 * h - 1) * (4 * h**2 + 2 * h - 1)
    # v=u/2, so d/du=(1/2)d/dv.
    assert (
        symbolic.trigsimp(
            symbolic.expand_trig(symbolic.diff(first, v) / 2 - expected_first)
        )
        == 0
    )
    assert (
        symbolic.trigsimp(
            symbolic.expand_trig(symbolic.diff(ring, v) / 2 - expected_ring)
        )
        == 0
    )
    assert symbolic.simplify(2 * ring.subs(v, symbolic.pi / 3)) == 7
    c = (symbolic.sqrt(5) - 1) / 4
    assert symbolic.simplify(4 * c**2 + 2 * c - 1) == 0
    assert bool(c > symbolic.Rational(3, 10))
    # The cubic is increasing for h>=3/10 and positive at that endpoint.
    polynomial_h = symbolic.symbols("h", real=True)
    cubic = 8 * polynomial_h**3 - 2 * polynomial_h + 1
    assert symbolic.diff(cubic, polynomial_h) == 24 * polynomial_h**2 - 2
    assert cubic.subs(polynomial_h, symbolic.Rational(3, 10)) > 0
    assert 24 * symbolic.Rational(3, 10) ** 2 - 2 > 0


@pytest.mark.parametrize("fraction", (Q(0), Q(2, 3), Q(4, 5)))
@pytest.mark.parametrize("both_rings", (False, True))
def test_geometric_path_keypoints_use_existing_domain_and_storage_owners(
    initial, fraction, both_rings
):
    u = fraction * initial["pi"]
    ring = (u, -u, -u / 2, I(0), u / 2)
    phases = ring + (ring if both_rings else (I(0),) * 5)
    graph = _graph(tuple(float(value.midpoint) for value in phases))
    expected = (
        2 * (5 - cos(2 * u) - 4 * cos(u / 2))
        if both_rings
        else 7 - cos(2 * u) - 4 * cos(u / 2) - 2 * cos(u)
    )
    initial_storage = 5 * (1 - initial["c"]) + 2 + 2 * initial["d"]
    assert initial_storage.lo > 7
    if both_rings:
        assert expected.hi <= 7 + Q(1, 10**30)
    for domain in ("regular", "positive_resultant"):
        if domain == "positive_resultant" and not both_rings and fraction:
            with pytest.raises(ValueError, match="positive real part"):
                evaluate_relational_exchange(
                    graph, model=RelationalExchangeModel(1, phase_domain=domain)
                )
        else:
            field = evaluate_relational_exchange(
                graph, model=RelationalExchangeModel(1, phase_domain=domain)
            )
            assert float(field.storage) == pytest.approx(
                float(expected.midpoint), abs=3e-15
            )
    # Keypoints supplement the analytic all-u argument. They are not a grid
    # certificate, simulated response, or storage-monotone ODE path.
