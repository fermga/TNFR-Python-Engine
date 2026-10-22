"""Exact controls for the selected averaged DPLL law and mapping obstruction.

These algebraic IVP histories are not experimentally prepared trajectories.
No circuit response, TNFR producer, or parameter inference is executed.
"""

from fractions import Fraction as F

import pytest


def _square(turn):
    return turn % 1 < F(1, 2)


def _averaged_xor(first_phase, second_phase):
    """Integrate two ideal square waves exactly over one carrier period."""
    breaks = {F(0), F(1)}
    for phase in (first_phase, second_phase):
        breaks.update(((-phase) % 1, (F(1, 2) - phase) % 1))
    breaks = sorted(breaks)
    duty = sum(
        right - left
        for left, right in zip(breaks, breaks[1:])
        if _square((left + right) / 2 + first_phase)
        != _square((left + right) / 2 + second_phase)
    )
    return 2 * duty - 1


def _history(time, perturbed=False):
    """Turn-valued C1 source histories on [-1, 0]."""
    return (
        time / 8,
        F(1, 4) + time / 8 + (time**2 / 32 if perturbed else 0),
    )


def _history_derivative(time, perturbed=False):
    return (F(1, 8), F(1, 8) + (time / 16 if perturbed else 0))


def _source_vector(current_phase, delayed_phase, filter_state):
    """Two reciprocal loops, b=delay=1 and omega/(2pi)=K/(2pi)=1/8."""
    phase_rate = tuple(F(1, 8) + F(value) / 8 for value in filter_state)
    filter_rate = tuple(
        -filter_state[node]
        + _averaged_xor(delayed_phase[1 - node], current_phase[node])
        for node in range(2)
    )
    return phase_rate + filter_rate


def _minor(first, second, row, other_row):
    return first[row] * second[other_row] - second[row] * first[other_row]


def _matvec(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


@pytest.mark.parametrize(
    "difference", [F(-3, 8), F(1, 8), F(5, 32), F(0), F(1, 2), F(9, 8)]
)
def test_triangle_detector_follows_from_exact_square_wave_overlap(difference):
    wrapped = (difference + F(1, 2)) % 1 - F(1, 2)
    expected_triangle = 4 * abs(wrapped) - 1
    # A common nonzero carrier phase must not change the integrated duty cycle.
    for carrier in (F(0), F(2, 7), F(-5, 3)):
        assert _averaged_xor(carrier + difference, carrier) == expected_triangle
        assert _averaged_xor(carrier - difference, carrier) == expected_triangle


def test_c1_histories_share_current_state_and_compatible_current_derivative():
    for perturbed in (False, True):
        assert _history(F(0), perturbed) == (0, F(1, 4))
        assert _history_derivative(F(0), perturbed) == (F(1, 8),) * 2
        # The derivative is affine: its two endpoint values certify positivity
        # throughout the entire history interval, without a sampling argument.
        for time in (F(-1), F(0)):
            assert min(_history_derivative(time, perturbed)) >= F(1, 16)
        vector = _source_vector(_history(F(0)), _history(F(-1), perturbed), (0, 0))
        assert vector[:2] == _history_derivative(F(0), perturbed)
    assert _history(F(-1), False) != _history(F(-1), True)


def test_delayed_xor_changes_filter_rate_but_not_current_phase_rate():
    current = _history(F(0))
    vectors = []
    arguments = []
    for perturbed in (False, True):
        delayed = _history(F(-1), perturbed)
        arguments.append((delayed[1] - current[0], delayed[0] - current[1]))
        vectors.append(_source_vector(current, delayed, (0, 0)))
    assert arguments == [(F(1, 8), F(-3, 8)), (F(5, 32), F(-3, 8))]
    assert all(0 < abs(value) < F(1, 2) for pair in arguments for value in pair)
    assert vectors == [
        (F(1, 8), F(1, 8), F(-1, 2), F(1, 2)),
        (F(1, 8), F(1, 8), F(-3, 8), F(1, 2)),
    ]
    assert all(isinstance(value, F) for vector in vectors for value in vector)
    assert _minor(*vectors, 0, 2) == F(1, 64)
    # A single current-only filter-rate prediction cannot fit both exactly.
    midpoint_prediction = (vectors[0][2] + vectors[1][2]) / 2
    assert abs(vectors[0][2] - midpoint_prediction) == F(1, 16)
    assert abs(vectors[1][2] - midpoint_prediction) == F(1, 16)


@pytest.mark.parametrize("first_clock,second_clock", [(F(1), F(1)), (F(1, 3), F(7, 5))])
def test_injective_mixed_chart_and_positive_clocks_preserve_noncollinearity(
    first_clock, second_clock
):
    # The first four rows are upper triangular with unit diagonal, proving
    # injectivity. The remaining rows add current-state coordinates, not history.
    chart = (
        (1, 1, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, -1, 0),
        (0, 1, 0, -1),
    )
    vectors = [
        _source_vector(_history(F(0)), _history(F(-1), flag), (0, 0))
        for flag in (False, True)
    ]
    images = [
        tuple(value / clock for value in _matvec(chart, vector))
        for vector, clock in zip(vectors, (first_clock, second_clock))
    ]
    assert _minor(*images, 0, 1) == F(1, 32) / (first_clock * second_clock)
    # Losing the filter block removes this instantaneous distinction; this is
    # why the obstruction does not assert a theorem about every lossy readout.
    assert vectors[0][:2] == vectors[1][:2]


def test_fixed_carrier_subtraction_retains_delayed_argument_offset():
    carrier = F(1, 16)
    delayed_time, current_time = F(-1), F(0)
    delayed_phi, current_phi = _history(delayed_time)[1], _history(current_time)[0]
    original = _averaged_xor(delayed_phi, current_phi)
    delayed_theta = delayed_phi - carrier * delayed_time
    current_theta = current_phi - carrier * current_time
    corrected = _averaged_xor(delayed_theta - carrier, current_theta)
    uncorrected = _averaged_xor(delayed_theta, current_theta)
    assert corrected == original == F(-1, 2)
    assert uncorrected == F(-1, 4)


@pytest.mark.parametrize("carrier", [F(0), F(1, 8), F(-2, 5)])
def test_fixed_carrier_cannot_fix_phase_mean_for_independent_filter_states(carrier):
    current, delayed = _history(F(0)), _history(F(-1))
    first = _source_vector(current, delayed, (F(0), F(0)))
    second = _source_vector(current, delayed, (F(1, 4), F(1, 4)))
    # Both nodes have degree one. Any fixed carrier cancels in the difference.
    first_weighted_rate = sum(value - carrier for value in first[:2])
    second_weighted_rate = sum(value - carrier for value in second[:2])
    assert second_weighted_rate - first_weighted_rate == F(1, 16)
