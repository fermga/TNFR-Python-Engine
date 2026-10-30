"""Exact joint-boundary algebra for the complete ten-node sine law.

The critical amplitude remains a symbolic trigonometric preparation. No
rounded graph, trajectory, horizon search or numerical tie tolerance is used.
"""

from fractions import Fraction
from itertools import combinations

import pytest


@pytest.fixture(scope="module")
def symbolic():
    return pytest.importorskip("sympy")


def _neighbors():
    blocks = tuple((2 * index, 2 * index + 1) for index in range(5))
    neighbors = [set() for _ in range(10)]
    for index, block in enumerate(blocks):
        for first in block:
            for second in blocks[(index + 1) % len(blocks)]:
                neighbors[first].add(second)
                neighbors[second].add(first)
    return neighbors


def test_exact_critical_ties_and_complete_fine_row_coefficients(symbolic):
    s = symbolic
    d, amplitude = s.Rational(1, 8), s.Symbol("u", positive=True)
    phases = (0, d, 2 * d, 3 * d, 1, 1, 2, 2, 3, 3)
    forms = (amplitude, -amplitude, amplitude, -amplitude) + (0,) * 6
    neighbors = _neighbors()
    sine_sums = tuple(
        sum(s.sin(phases[j] - phases[i]) for j in row)
        for i, row in enumerate(neighbors)
    )
    form_rates = tuple(value / (4 * s.pi) for value in sine_sums)
    phase_rates = tuple(
        sum(forms[i] - forms[j] for j in row) / (4 * s.pi)
        for i, row in enumerate(neighbors)
    )
    assert all(len(row) == 4 for row in neighbors)
    assert phase_rates == tuple(value / s.pi for value in forms)

    def chord(angle):
        return 2 * (1 - s.cos(angle))

    critical_square = (chord(2 * d) - chord(d)) / 4
    first_form, second_form, first_phase, second_phase = s.symbols("x y p q")
    joint = (second_form - first_form) ** 2 + chord(second_phase - first_phase)
    coordinates = (first_form, second_form, first_phase, second_phase)
    rates = {}
    tied = ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3))
    for first, second in tied:
        source = dict(
            zip(
                coordinates,
                (forms[first], forms[second], phases[first], phases[second]),
            )
        )
        cost = joint.subs(source)
        assert s.expand(cost.subs(amplitude**2, critical_square) - chord(2 * d)) == 0
        full_velocity = (
            form_rates[first],
            form_rates[second],
            phase_rates[first],
            phase_rates[second],
        )
        rates[first, second] = s.expand(
            sum(
                s.diff(joint, coordinate).subs(source) * velocity
                for coordinate, velocity in zip(coordinates, full_velocity)
            )
            * s.pi
            / amplitude
        )

    coefficient_a = 5 * s.sin(d) - s.sin(3 * d) + 2 * (s.sin(3 - d) - s.sin(3))
    coefficient_b = (
        2 * s.sin(d) - 2 * s.sin(2 * d) + 2 * (s.sin(1 - 2 * d) - s.sin(3 - d))
    )
    coefficient_c = (
        5 * s.sin(d) - s.sin(3 * d) + 2 * (s.sin(1 - 3 * d) - s.sin(1 - 2 * d))
    )
    expected = (-coefficient_a, coefficient_b, -coefficient_c, 0, 0)
    for pair, coefficient in zip(tied, expected):
        assert s.expand(rates[pair] - coefficient) == 0


def test_coefficient_sign_proofs_keep_the_form_phase_cancellation(symbolic):
    s = symbolic
    d = s.Symbol("d", real=True)
    common = 5 * s.sin(d) - s.sin(3 * d)
    positive_common = 2 * s.sin(d) + 4 * s.sin(d) ** 3
    assert s.expand(s.expand_trig(common - positive_common)) == 0
    coefficient_c = common + 2 * (s.sin(1 - 3 * d) - s.sin(1 - 2 * d))
    positive_c = (
        4 * s.sin(d / 2) * (s.cos(d / 2) - s.cos(1 - 5 * d / 2)) + 4 * s.sin(d) ** 3
    )
    assert s.trigsimp(coefficient_c - positive_c, method="fu") == 0

    # Independently verify the rational domains needed by the analytic
    # monotonicity and Taylor bounds, rather than sample the coefficients.
    value = Fraction(1, 8)
    pi_lower, pi_upper = Fraction(3), Fraction(22, 7)
    assert pi_upper / 2 < 3 - value < 3 <= pi_lower
    assert 0 < value / 2 < 1 - 5 * value / 2 < pi_lower
    angle = 1 - 2 * value
    sine_lower = angle - angle**3 / 6
    reflected_upper = pi_upper - (3 - value)
    assert sine_lower > Fraction(2, 3)
    assert 0 < pi_lower - (3 - value) < reflected_upper < Fraction(1, 3)
    assert 2 * (sine_lower - reflected_upper - value) > Fraction(5, 12)


def test_all_tied_choices_and_structural_support_after_the_crossing(symbolic):
    s = symbolic
    a, b, c = s.symbols("A B C", positive=True)
    edge_rates = {
        (0, 1): -a,
        (1, 2): b,
        (2, 3): -c,
        (0, 2): s.Integer(0),
        (1, 3): s.Integer(0),
    }
    tied_sets = ((1, 2), (0, 2, 3), (0, 1, 3), (1, 2))
    positive_map, negative_map = (1, 0, 3, 2), (2, 2, 1, 1)
    for orientation, choices in ((1, positive_map), (-1, negative_map)):
        for node, candidates in enumerate(tied_sets):
            desired = choices[node]
            desired_rate = orientation * edge_rates[tuple(sorted((node, desired)))]
            for competitor in candidates:
                if competitor == desired:
                    continue
                competing_rate = (
                    orientation * edge_rates[tuple(sorted((node, competitor)))]
                )
                assert s.expand(competing_rate - desired_rate).is_positive
    assert all(
        positive_map[partner] == node for node, partner in enumerate(positive_map)
    )
    assert any(
        negative_map[partner] != node for node, partner in enumerate(negative_map)
    )

    # Recognition selects the already valid support quotient: all five
    # independent swaps preserve the actual graph, without deleting state.
    neighbors = _neighbors()
    edges = {frozenset((i, j)) for i, row in enumerate(neighbors) for j in row}
    blocks = tuple((2 * index, 2 * index + 1) for index in range(5))
    for first, second in blocks:
        assert second not in neighbors[first]
        permutation = list(range(10))
        permutation[first], permutation[second] = second, first
        assert {frozenset(permutation[i] for i in edge) for edge in edges} == edges
    for left, right in combinations(range(5), 2):
        present = sum(j in neighbors[i] for i in blocks[left] for j in blocks[right])
        assert present == (4 if (right - left) % 5 in (1, 4) else 0)
