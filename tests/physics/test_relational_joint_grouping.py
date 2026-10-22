"""Same-source joint grouping: exact rates and whole-box rational controls.

The source, independent preparation errors and horizon are the existing phase
forecast's. No trajectory, source scan or reselected observation time is used.
"""

from fractions import Fraction as Q
from itertools import combinations

import pytest


@pytest.fixture(scope="module")
def symbolic():
    return pytest.importorskip("sympy")


def _support():
    neighbors = [set() for _ in range(10)]
    for block in range(5):
        following = (block + 1) % 5
        for first in (2 * block, 2 * block + 1):
            for second in (2 * following, 2 * following + 1):
                neighbors[first].add(second)
                neighbors[second].add(first)
    return tuple(tuple(sorted(row)) for row in neighbors)


def _source_values():
    amplitude = Q(1, 8)
    forms = (amplitude, -amplitude, amplitude, -amplitude) + (Q(0),) * 6
    phases = tuple(
        map(Q, (0, amplitude, 2 * amplitude, 3 * amplitude, 1, 1, 2, 2, 3, 3))
    )
    return forms, phases


def test_joint_initial_costs_and_both_row_rates_on_actual_doubled_support(symbolic):
    s = symbolic
    forms, phases = (tuple(map(s.Rational, values)) for values in _source_values())
    neighbors = _support()
    form_rates = tuple(
        sum(s.sin(phases[j] - phases[i]) for j in row) / (len(row) * s.pi)
        for i, row in enumerate(neighbors)
    )
    form_gradient = tuple(
        sum(forms[i] - forms[j] for j in row) for i, row in enumerate(neighbors)
    )
    phase_rates = tuple(value / (4 * s.pi) for value in form_gradient)
    assert all(len(row) == 4 for row in neighbors)
    assert form_gradient == tuple(4 * value for value in forms)

    def joint_cost(i, j, form_values):
        return (form_values[j] - form_values[i]) ** 2 + 2 * (
            1 - s.cos(phases[j] - phases[i])
        )

    def joint_rate(i, j, form_values, phase_velocity):
        return 2 * (form_values[j] - form_values[i]) * (
            form_rates[j] - form_rates[i]
        ) + 2 * s.sin(phases[j] - phases[i]) * (phase_velocity[j] - phase_velocity[i])

    reversed_forms = tuple(-value for value in forms)
    reversed_phase_rates = tuple(-value for value in phase_rates)
    for first, second in combinations(range(10), 2):
        assert (
            s.expand(
                joint_cost(first, second, forms)
                - joint_cost(first, second, reversed_forms)
            )
            == 0
        )
        assert (
            s.expand(
                joint_rate(first, second, forms, phase_rates)
                + joint_rate(first, second, reversed_forms, reversed_phase_rates)
            )
            == 0
        )

    def chord(gap):
        return 2 * (1 - s.cos(gap))

    assert joint_cost(0, 2, forms) == joint_cost(1, 3, forms) == chord(s.Rational(1, 4))
    for pair in ((0, 1), (1, 2), (2, 3)):
        assert joint_cost(*pair, forms) == s.Rational(1, 16) + chord(s.Rational(1, 8))
    assert joint_rate(0, 2, forms, phase_rates) == 0
    assert joint_rate(1, 3, forms, phase_rates) == 0


def test_frozen_independent_boxes_keep_all_joint_nearest_choices_on_full_interval():
    forms, phases = _source_values()
    radius, end = Q(1, 2**20), Q(1, 64)
    # The full node speed is <=1/pi<1/3. The phase remainder includes both
    # independent initial errors and the entire nonlinear acceleration.
    form_error = radius + end / 3
    phase_error = radius + 2 * radius * end / 3 + end**2 / 9
    assert 3 + 2 * Q(1, 8) * end / 3 + 2 * phase_error < Q(25, 8)

    def whole_interval_cost_bounds(first, second):
        initial_form_gap = abs(forms[first] - forms[second])
        initial_phase_gap = abs(phases[first] - phases[second])
        phase_motion = initial_form_gap * end / 3
        lower_form_gap = max(Q(0), initial_form_gap - 2 * form_error)
        upper_form_gap = initial_form_gap + 2 * form_error
        lower_phase_gap = max(Q(0), initial_phase_gap - phase_motion - 2 * phase_error)
        upper_phase_gap = initial_phase_gap + phase_motion + 2 * phase_error
        # Circular gaps remain below pi. On [0,pi], .4*s^2 <= F(s) <= s^2.
        return (
            lower_form_gap**2 + Q(2, 5) * lower_phase_gap**2,
            upper_form_gap**2 + upper_phase_gap**2,
        )

    nearest = (2, 3, 0, 1, 5, 4, 7, 6, 9, 8)
    margins = []
    for node, desired in enumerate(nearest):
        _, desired_upper = whole_interval_cost_bounds(node, desired)
        for competitor in range(10):
            if competitor in (node, desired):
                continue
            competitor_lower, _ = whole_interval_cost_bounds(node, competitor)
            margins.append(competitor_lower - desired_upper)
        assert nearest[desired] == node
    assert len(margins) == 80
    assert min(margins) > Q(1, 2048)
    # Only absolute source form gaps entered the uniform bounds, so the
    # exact same argument applies to the original form-reversed box.
    assert all(
        abs(forms[i] - forms[j]) == abs(-forms[i] + forms[j])
        for i, j in combinations(range(10), 2)
    )


def test_recognized_joint_pairs_are_not_the_independent_support_symmetry():
    neighbors = _support()
    # A within-pair edge alone does not disprove the broader orbit theorem;
    # unequal external neighborhoods do. Compare both sides explicitly.
    assert 2 in neighbors[0]
    assert set(neighbors[0]) - {2} == {3, 8, 9}
    assert set(neighbors[2]) - {0} == {1, 4, 5}
    permutation = list(range(10))
    permutation[0], permutation[2] = permutation[2], permutation[0]
    edges = {frozenset((i, j)) for i, row in enumerate(neighbors) for j in row}
    swapped_edges = {frozenset(permutation[i] for i in edge) for edge in edges}
    assert swapped_edges != edges
    # The old structural pair retains its independently valid swap; the new
    # observation has not modified or invalidated that supplied support.
    permutation = list(range(10))
    permutation[0], permutation[1] = permutation[1], permutation[0]
    swapped_edges = {frozenset(permutation[i] for i in edge) for edge in edges}
    assert swapped_edges == edges
