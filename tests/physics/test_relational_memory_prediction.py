"""Independent static controls for the prepared amplitude prediction.

These checks use ideal-lock derivative formulas and fresh native snapshots.
They do not evaluate the reserved full-engine temporal response or certify
irrational constants, nonlinear remainder bounds, or continuous-time error.
"""

import math
from decimal import Decimal

import networkx as nx
import numpy as np
import pytest

from benchmarks import relational_memory_prediction as prediction_module
from benchmarks.relational_memory_prediction import (
    EPSILON,
    HORIZON,
    STEP_COUNTS,
    evaluate_polynomials,
    predict,
    prepare_geometry,
    quadratic_cross,
)
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)


@pytest.fixture(scope="module")
def geometry():
    return prepare_geometry()


def _mv(matrix, vector):
    return np.asarray(matrix, dtype=float) @ np.asarray(vector, dtype=float)


def _graph(form, deviation):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edge(0, 5)
    for node in graph:
        graph.nodes[node].update(
            EPI=form[node],
            theta=math.tau * (node % 5) / 5 + deviation[node],
            nu_f=1.0,
        )
    return graph


def _native_rates(form, deviation):
    field = evaluate_relational_exchange(
        _graph(form, deviation), model=RelationalExchangeModel(1)
    )
    return np.array(field.form_rate + field.phase_rate)


def _linear_rates(form, phase):
    """Independent graph Hessian/gradient formula at the ideal lock."""
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edge(0, 5)
    c = math.cos(math.tau / 5)
    form_rate, phase_rate = [], []
    for node in graph:
        degree = graph.degree(node)
        strength = 1 + 2 * c if node in (0, 5) else 2 * c
        q = sum(form[node] - form[j] for j in graph[node])
        phase_gradient = sum(
            (c if node // 5 == j // 5 else 1) * (phase[node] - phase[j])
            for j in graph[node]
        )
        form_rate.append(
            -0.5 * q / degree - 0.5 * phase_gradient / (math.pi * strength)
        )
        phase_rate.append(0.5 * q / (math.pi * strength))
    return np.array(form_rate + phase_rate)


def test_prepared_coefficients_match_independent_resultant_expansions(geometry):
    terms = evaluate_polynomials(geometry, geometry.initial_direction)
    c, s, k, w = math.cos(math.tau / 5), math.sin(math.tau / 5), 0.5, 0.5
    strength = 1 + 2 * c
    quadratic, cubic = np.zeros(20), np.zeros(20)
    quadratic[11] = -3 * k * s / (4 * math.pi * c * c)
    quadratic[14] = -quadratic[11]
    cubic[10] = -k / (3 * math.pi * strength)
    cubic[11] = cubic[14] = k / (4 * math.pi * c) + 3 * k * s * s / (8 * math.pi * c**3)
    # The recipient port reads Arg(2*c + exp(i*sigma)); its cubic source is
    # independent of the donor's exact linear Arg(h_* exp(-i*sigma)).
    cubic[5] = (
        w
        / math.pi
        * (-1 / (6 * strength) + 1 / (2 * strength**2) - 1 / (3 * strength**3))
    )
    np.testing.assert_allclose(terms.quadratic, quadratic, atol=3e-15, rtol=2e-15)
    np.testing.assert_allclose(terms.cubic, cubic, atol=3e-15, rtol=2e-15)
    np.testing.assert_allclose(
        _mv(geometry.visible_projection, quadratic), 0, atol=1e-16
    )
    np.testing.assert_allclose(_mv(geometry.hidden_projection, cubic), 0, atol=1e-16)


def test_generated_hidden_source_has_nonzero_mixed_feedback(geometry):
    direction = geometry.initial_direction
    hidden_phase = (0,) * 10 + (0, 1, 0, 0, -1, 0, 0, 0, 0, 0)
    cross = quadratic_cross(geometry, direction, hidden_phase)
    c, s, k = math.cos(math.tau / 5), math.sin(math.tau / 5), 0.5
    expected = -4 * k * s / (math.pi * (1 + 2 * c) ** 2)
    assert cross[10] == pytest.approx(expected, rel=2e-15)
    # The coarse observable is theta0 - mean(theta_right). At initial time
    # only the left near-phase hidden pair is generated. Its integrated
    # feedback starts at order t**2, unlike the direct cubic rate term.
    generated = -3 * k * s / (4 * math.pi * c**2)
    projected = _mv(geometry.visible_projection, cross)
    assert projected[5] + projected[6] == pytest.approx(expected, rel=2e-15)
    assert generated * expected == pytest.approx(
        3 * k**2 * s**2 / (math.pi**2 * c**2 * (1 + 2 * c) ** 2), rel=2e-15
    )
    assert generated * expected > 0


def test_polynomials_preserve_homogeneity_parity_and_offset_symmetry(geometry):
    direction = tuple((i * i % 17 - 8) / 16 for i in range(20))
    original = evaluate_polynomials(geometry, direction)
    scaled = evaluate_polynomials(geometry, tuple(-2 * x for x in direction))
    shifted = evaluate_polynomials(
        geometry, tuple(x + (3 if i < 10 else -2) for i, x in enumerate(direction))
    )
    np.testing.assert_allclose(scaled.quadratic, 4 * np.array(original.quadratic))
    np.testing.assert_allclose(scaled.cubic, -8 * np.array(original.cubic))
    np.testing.assert_allclose(shifted.quadratic, original.quadratic, atol=1e-15)
    np.testing.assert_allclose(shifted.cubic, original.cubic, atol=1e-15)
    reflection = (0, 4, 3, 2, 1, 5, 9, 8, 7, 6)
    indices = reflection + tuple(10 + i for i in reflection)
    transformed = evaluate_polynomials(geometry, tuple(-direction[i] for i in indices))
    np.testing.assert_allclose(
        transformed.quadratic, [-original.quadratic[i] for i in indices], atol=2e-15
    )
    np.testing.assert_allclose(
        transformed.cubic, [-original.cubic[i] for i in indices], atol=2e-15
    )


def test_cross_is_the_symmetric_mixed_coefficient_without_an_extra_half(geometry):
    left = tuple((i % 4 - 2) / 8 for i in range(20))
    right = tuple((i * i % 7 - 3) / 16 for i in range(20))
    added = tuple(a + b for a, b in zip(left, right, strict=True))
    expected = (
        np.array(evaluate_polynomials(geometry, added).quadratic)
        - evaluate_polynomials(geometry, left).quadratic
        - evaluate_polynomials(geometry, right).quadratic
    )
    actual = quadratic_cross(geometry, left, right)
    np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=3e-14)
    np.testing.assert_allclose(
        actual, quadratic_cross(geometry, right, left), atol=1e-16
    )
    np.testing.assert_allclose(
        quadratic_cross(geometry, left, left),
        2 * np.array(evaluate_polynomials(geometry, left).quadratic),
        atol=1e-16,
    )


def test_static_native_field_matches_the_independent_third_order_expansion(geometry):
    # This is one snapshot, not a time step or an amplitude trajectory study.
    # Subtract the captured rounded-lock field: binary64 theta_* is not
    # asserted to be an exact equilibrium or to have exact analytic parity.
    epsilon = 2**-11
    direction = np.array(geometry.initial_direction)
    baseline = _native_rates((0,) * 10, (0,) * 10)
    native = _native_rates(epsilon * direction[:10], epsilon * direction[10:])
    terms = evaluate_polynomials(geometry, direction)
    linear = epsilon * _linear_rates(direction[:10], direction[10:])
    quadratic = epsilon**2 * np.array(terms.quadratic)
    cubic = epsilon**3 * np.array(terms.cubic)
    remainder = native - baseline - linear - quadratic - cubic
    assert np.linalg.norm(remainder, np.inf) < 2e-12
    assert np.linalg.norm(cubic, np.inf) > 50 * np.linalg.norm(remainder, np.inf)
    assert np.linalg.norm(baseline, np.inf) < 2e-15


def test_frozen_predictions_separate_direct_and_generated_memory_terms():
    prediction = predict(64)
    precise = predict(64, decimal_precision=50)
    epsilon = float(EPSILON)
    assert STEP_COUNTS == (64, 128, 256)
    assert HORIZON == 1 / 4
    assert prediction.timestep == float(HORIZON) / prediction.steps
    np.testing.assert_array_equal(
        prediction.linear_visible, epsilon * np.array(prediction.y1)
    )
    np.testing.assert_array_equal(
        prediction.memory_visible,
        np.array(prediction.linear_visible) + epsilon**3 * np.array(prediction.y3),
    )
    np.testing.assert_array_equal(
        prediction.direct_visible,
        np.array(prediction.linear_visible)
        + epsilon**3 * np.array(prediction.y3_direct),
    )
    np.testing.assert_array_equal(
        prediction.hidden_second_order, epsilon**2 * np.array(prediction.h2)
    )
    assert np.linalg.norm(np.array(prediction.y3) - prediction.y3_direct) > 1e-6
    for name in ("linear", "memory", "direct"):
        visible = getattr(prediction, name + "_visible")
        output = getattr(prediction, name + "_output")
        np.testing.assert_array_equal(output, visible[:3] + visible[5:8])
        precise_values = getattr(precise, name + "_visible")
        assert all(isinstance(value, Decimal) for value in precise_values)
        np.testing.assert_allclose(
            visible, tuple(map(float, precise_values)), atol=1e-16
        )
    assert prediction.coefficient_constants == precise.coefficient_constants
    assert "identical exact represented constants" in precise.arithmetic_scope


def test_simultaneous_coefficient_step_does_not_consume_new_hidden_source(
    geometry, monkeypatch
):
    # Admit one auxiliary coefficient step only within this test. The actual
    # frozen engine protocol and its retained response are not executed here.
    monkeypatch.setattr(prediction_module, "STEP_COUNTS", (1,))
    prediction = predict(1)
    initial_even = _mv(
        geometry.visible_lift,
        _mv(geometry.visible_projection, geometry.initial_direction),
    )
    source = evaluate_polynomials(geometry, initial_even).quadratic
    expected_hidden = prediction.timestep * _mv(geometry.hidden_projection, source)
    np.testing.assert_allclose(prediction.h2, expected_hidden, atol=1e-16, rtol=2e-15)
    assert np.linalg.norm(prediction.h2) > 0
    assert prediction.y3 == prediction.y3_direct
    assert prediction.memory_visible == prediction.direct_visible


@pytest.mark.parametrize("steps", (0, 63, 65, 64.0, True))
def test_prediction_rejects_unregistered_grids(steps):
    with pytest.raises(ValueError, match="fixed grids"):
        predict(steps)
