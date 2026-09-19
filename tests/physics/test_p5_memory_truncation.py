"""Independent scalar references for finite-memory P5 diffusion bounds.

High-precision elementary solutions check the rational enclosures. A separate
block exponential checks later delay windows to numerical tolerance; that
comparison does not turn its floating-point output into an exact certificate.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from tnfr.physics.p5_memory_truncation import bound_p5_memory_truncation

F = Fraction
INITIAL = (F(0), F(3, 2), F(-3), F(3, 2), F(0))


def _mp(value):
    value = F(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, expected):
    lower, upper = interval
    assert isinstance(lower, F) and isinstance(upper, F)
    assert lower <= upper
    assert _mp(lower) <= expected <= _mp(upper)


def _coordinates(initial):
    x = tuple(F(value) for value in initial)
    a, b = (x[0] + x[4]) / 2, (x[1] + x[2] + x[3]) / 3
    u = x[2] - (x[1] + x[3]) / 2
    d = a - b
    return (
        a,
        b,
        u,
        (a + 3 * b) / 4,
        2 * d / 3 - 4 * u / 9,
        d / 3 + 4 * u / 9,
    )


def _reference(initial, time, capacity=1):
    *_, c1, c2 = _coordinates(initial)
    r = _mp(capacity) * _mp(time)
    return _mp(c1) * mp.exp(-r) + _mp(c2) * mp.exp(-2 * r)


def _zero_window(initial, time, capacity=1):
    a, b, u, *_ = _coordinates(initial)
    r = _mp(capacity) * _mp(time)
    return (_mp(a - b) - 4 * _mp(u) / 3) * mp.exp(-4 * r / 3) + 4 * _mp(u) * mp.exp(
        -5 * r / 3
    ) / 3


def _first_correction(initial, time, window, capacity=1):
    *_, c1, c2 = _coordinates(initial)
    nu = _mp(capacity)
    r = nu * (_mp(time) - _mp(window))
    decay = mp.exp(-5 * nu * _mp(window) / 3)
    return (
        -2
        * decay
        / 9
        * (
            _mp(c1) * ((r - 1) * mp.exp(-r) + mp.exp(-2 * r))
            + _mp(c2) * (mp.exp(-r) - (1 + r) * mp.exp(-2 * r))
        )
    )


def _tail(initial, time, window, capacity=1):
    if time <= window:
        return mp.mpf(0)
    *_, c1, c2 = _coordinates(initial)
    nu = _mp(capacity)
    q = nu * (_mp(time) - _mp(window))
    decay = mp.exp(-5 * nu * _mp(window) / 3)
    return (
        nu
        * decay
        / 3
        * (
            _mp(c1) * (mp.exp(-q) - mp.exp(-5 * q / 3))
            + 2 * _mp(c2) * (mp.exp(-5 * q / 3) - mp.exp(-2 * q))
        )
    )


def _check_macro(intervals, mean, contrast):
    _contains(intervals[0], _mp(mean) + 3 * contrast / 4)
    _contains(intervals[1], _mp(mean) - contrast / 4)


@pytest.mark.parametrize("capacity", [F(1), F(3, 2)])
def test_full_memory_until_cutoff_matches_both_exact_decay_modes(capacity):
    window = F(2, 3)
    result = bound_p5_memory_truncation(
        INITIAL, memory_window=window, times=(0, F(1, 3), window), capacity=capacity
    )
    with mp.workdps(180):
        for sample in result.samples:
            expected = _reference(INITIAL, sample.time, capacity)
            _contains(sample.reference_contrast, expected)
            _contains(sample.truncated_contrast, expected)
            _check_macro(sample.reference_macro, result.conserved_mean, expected)
            _check_macro(sample.truncated_macro, result.conserved_mean, expected)
            assert sample.coincides_before_cutoff
            assert sample.delayed_corrections == 0
            assert sample.contrast_error == (F(0), F(0))
            assert sample.macro_error_bound == 0
            assert sample.macro_error_enclosure == (F(0), F(0))
            assert sample.dropped_reference_forcing == (F(0), F(0))


def test_zero_window_retains_initial_hidden_state_source():
    result = bound_p5_memory_truncation(
        INITIAL, memory_window=0, times=(0, F(1, 4), 1, 3)
    )
    assert result.initial_macro == (F(0), F(0))
    assert result.initial_hidden_contrast == F(-9, 2)
    with mp.workdps(180):
        for sample in result.samples:
            reference = _reference(INITIAL, sample.time)
            truncated = _zero_window(INITIAL, sample.time)
            _contains(sample.reference_contrast, reference)
            _contains(sample.truncated_contrast, truncated)
            _contains(sample.contrast_error, truncated - reference)
            _check_macro(sample.truncated_macro, result.conserved_mean, truncated)
            _contains(sample.macro_error_enclosure, 3 * abs(truncated - reference) / 4)
            assert sample.delayed_corrections == 0
        # A naive Markov model with these zero macro coordinates stays at zero.
        assert result.samples[1].truncated_contrast[0] > 0


@pytest.mark.parametrize("initial", [INITIAL, (1, 0, 0, 0, 1), (-1, 0, 0, 0, -1)])
def test_first_delay_window_has_analytic_correction_and_signed_tail(initial):
    window, capacity = F(3, 4), F(5, 4)
    result = bound_p5_memory_truncation(
        initial,
        memory_window=window,
        times=(window, F(1), 2 * window),
        capacity=capacity,
    )
    with mp.workdps(180):
        for sample in result.samples:
            reference = _reference(initial, sample.time, capacity)
            correction = _first_correction(initial, sample.time, window, capacity)
            _contains(sample.truncated_contrast, reference + correction)
            _contains(sample.contrast_error, correction)
            _contains(
                sample.dropped_reference_forcing,
                _tail(initial, sample.time, window, capacity),
            )
            source = (
                -4
                * _mp(capacity)
                * _mp(result.initial_hidden_contrast)
                / 9
                * mp.exp(-5 * _mp(capacity) * _mp(sample.time) / 3)
            )
            _contains(sample.initial_source_contrast, source)
            assert sample.delayed_corrections == int(sample.time > window)


def _delayed_block_reference(initial, time, window, capacity):
    """Solve a cascade of forced second-order ODEs using a separate backend."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    a, b, u, *_ = _coordinates(initial)
    nu, t, cut = float(capacity), float(time), float(window)
    quotient = time / window
    count = (quotient.numerator - 1) // quotient.denominator
    generator = np.array([[0.0, 1.0], [-2 * nu**2, -3 * nu]])
    coupling = np.array([[0.0, 0.0], [-2 * nu**2 * np.exp(-5 * nu * cut / 3) / 9, 0.0]])
    total = 0.0
    for index in range(int(count) + 1):
        matrix = np.kron(np.eye(index + 1), generator)
        if index:
            matrix += np.kron(np.eye(index + 1, k=-1), coupling)
        state = np.zeros(2 * (index + 1))
        state[:2] = [float(a - b), nu * float(-4 * (a - b) / 3 - 4 * u / 9)]
        evolved = scipy_linalg.expm((t - index * cut) * matrix) @ state
        total += evolved[2 * index]
    return total


@pytest.mark.parametrize("initial", [INITIAL, (1, 0, 0, 0, 1), (2, -1, 3, 4, 0)])
def test_later_windows_agree_with_independent_second_order_delay_cascade(initial):
    window, capacity = F(1, 2), F(3, 2)
    result = bound_p5_memory_truncation(
        initial,
        memory_window=window,
        times=(F(5, 4), F(7, 4), 2),
        capacity=capacity,
    )
    for sample in result.samples:
        expected = _delayed_block_reference(initial, sample.time, window, capacity)
        midpoint = sum(sample.truncated_contrast) / 2
        assert float(midpoint) == pytest.approx(expected, abs=2e-12, rel=2e-12)
        assert sample.macro_error_enclosure[1] <= sample.macro_error_bound
        assert sample.evaluation_width >= 0


def test_reference_contrast_maximum_includes_interior_extremum():
    result = bound_p5_memory_truncation(INITIAL, memory_window=1, times=(0,))
    assert result.contrast_coefficients == (F(2), F(-2))
    assert result.initial_macro == (F(0), F(0))
    assert result.max_reference_contrast == F(1, 2)
    endpoint = bound_p5_memory_truncation((1, 0, 0, 0, 1), memory_window=1, times=(0,))
    assert endpoint.max_reference_contrast == F(1)


def test_exponential_tail_and_both_global_and_causal_error_bounds():
    window, capacity = F(1, 4), F(3, 2)
    result = bound_p5_memory_truncation(
        INITIAL,
        memory_window=window,
        times=(window, window + F(1, 1000), 1, 4),
        capacity=capacity,
    )
    with mp.workdps(180):
        delta = mp.exp(-5 * _mp(capacity) * _mp(window) / 3)
        for sample in result.samples:
            q = _mp(capacity) * max(mp.mpf(0), _mp(sample.time - window))
            exact_bound = (
                3
                * _mp(result.max_reference_contrast)
                * delta
                / 4
                * min(1 / (9 + delta), (1 - mp.exp(-q)) ** 2 / 9)
            )
            assert _mp(sample.macro_error_bound) >= exact_bound
            assert abs(_mp(sample.macro_error_bound) - exact_bound) < mp.mpf("1e-20")
            forcing = _tail(INITIAL, sample.time, window, capacity)
            assert abs(forcing) <= _mp(sample.dropped_reference_forcing_bound)
            assert sample.macro_error_enclosure[1] <= sample.macro_error_bound


def test_capacity_time_rescaling_preserves_trajectory_and_error_bound():
    first = bound_p5_memory_truncation(
        INITIAL, memory_window=F(1, 2), times=(F(3, 4), 1, F(3, 2))
    )
    second = bound_p5_memory_truncation(
        INITIAL, memory_window=F(1, 4), times=(F(3, 8), F(1, 2), F(3, 4)), capacity=2
    )
    for slow, fast in zip(first.samples, second.samples, strict=True):
        assert slow.reference_contrast == fast.reference_contrast
        assert slow.truncated_contrast == fast.truncated_contrast
        assert slow.macro_error_bound == fast.macro_error_bound
        assert slow.delayed_corrections == fast.delayed_corrections
        assert tuple(2 * value for value in slow.initial_source_contrast) == (
            fast.initial_source_contrast
        )


def test_consensus_has_exact_zero_memory_truncation_error():
    value = F(7, 3)
    result = bound_p5_memory_truncation(
        (value,) * 5, memory_window=F(1, 2), times=(0, F(1, 2), 1, 3)
    )
    assert result.max_reference_contrast == 0
    for sample in result.samples:
        assert sample.reference_contrast == (F(0), F(0))
        assert sample.truncated_contrast == (F(0), F(0))
        assert sample.reference_macro == ((value, value), (value, value))
        assert sample.truncated_macro == ((value, value), (value, value))
        assert sample.macro_error_bound == 0


def test_hidden_antisymmetric_modes_do_not_change_visible_memory_solution():
    first = bound_p5_memory_truncation(
        INITIAL, memory_window=F(1, 2), times=(0, F(3, 4), 2)
    )
    second = bound_p5_memory_truncation(
        (3, F(5, 2), -3, F(1, 2), -3),
        memory_window=F(1, 2),
        times=(0, F(3, 4), 2),
    )
    assert first.initial_epi != second.initial_epi
    assert first.initial_hidden_contrast == second.initial_hidden_contrast
    assert first.samples == second.samples


def test_inputs_are_exact_represented_rationals_detached_and_immutable():
    initial = [0.1, 1, 2, 3, 4]
    times = [0.2, F(1, 3), 0.2]
    result = bound_p5_memory_truncation(initial, memory_window=0.5, times=times)
    initial[0], times[0] = 99, 99
    assert result.initial_epi[0] == F.from_float(0.1)
    assert result.memory_window == F(1, 2)
    assert result.samples[0].time == F.from_float(0.2)
    assert result.samples[0] == result.samples[2]
    assert isinstance(result.samples, tuple)
    assert isinstance(result.initial_epi, tuple)
    with pytest.raises(FrozenInstanceError):
        result.capacity = F(2)
    with pytest.raises(FrozenInstanceError):
        result.samples[0].time = F(2)


@pytest.mark.parametrize(
    "initial",
    [
        (),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5, 6),
        (0, 0, 0, 0, True),
        (0, 0, 0, 0, "1"),
        (0, 0, 0, 0, 1j),
        (0, 0, 0, 0, float("nan")),
        (0, 0, 0, 0, float("inf")),
        [[0], [1], [2], [3], [4]],
        bytearray(range(5)),
    ],
)
def test_initial_epi_requires_five_finite_real_scalar_coordinates(initial):
    with pytest.raises((TypeError, ValueError)):
        bound_p5_memory_truncation(initial, memory_window=1, times=(0,))


@pytest.mark.parametrize("capacity", [0, -1, True, "1", 1j, float("inf")])
def test_capacity_requires_a_positive_finite_rationalizable_scalar(capacity):
    with pytest.raises((TypeError, ValueError)):
        bound_p5_memory_truncation(
            INITIAL, memory_window=1, times=(0,), capacity=capacity
        )


@pytest.mark.parametrize("window", [-1, True, "1", 1j, float("nan")])
def test_window_requires_a_nonnegative_finite_rationalizable_scalar(window):
    with pytest.raises((TypeError, ValueError)):
        bound_p5_memory_truncation(INITIAL, memory_window=window, times=(0,))


@pytest.mark.parametrize(
    "times", [(), (-1,), (True,), ("1",), (1j,), (float("inf"),), bytearray([0, 1])]
)
def test_samples_require_nonempty_nonnegative_finite_times(times):
    with pytest.raises((TypeError, ValueError)):
        bound_p5_memory_truncation(INITIAL, memory_window=1, times=times)


def test_resource_domain_rejects_excessive_exponential_or_delay_work():
    with pytest.raises(ValueError):
        bound_p5_memory_truncation(INITIAL, memory_window=2048, times=(2049,))
    with pytest.raises(ValueError):
        bound_p5_memory_truncation(
            INITIAL, memory_window=1024, times=(1025,), capacity=2
        )
    with pytest.raises(ValueError):
        bound_p5_memory_truncation(INITIAL, memory_window=1, times=(F(33001, 1000),))


def test_delay_resource_boundary_excludes_the_zero_correction_at_its_endpoint():
    window, time = F(1, 4), F(33, 4)
    result = bound_p5_memory_truncation(INITIAL, memory_window=window, times=(time,))
    assert result.samples[0].delayed_corrections == 32
    with pytest.raises(ValueError):
        bound_p5_memory_truncation(
            INITIAL, memory_window=window, times=(time + F(1, 10000),)
        )
