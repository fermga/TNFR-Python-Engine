"""Static independent contracts for one carried winding-observation history.

Algebraic records here witness necessary response envelopes, not integrated
trajectories. No copied initial state or measured unprobed branch is supplied
to the sequential inverse.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_outer_observation as previous
from tnfr.physics import _organized_sequential_observation as owner


_EPSILON = Q(1, 10)
_DELAY = Q(1, 100000)
_DURATION = 20 * _DELAY
_FORM = Q(1, 100)
_FINE = Q(1, 10**6)
_NOISE = Q(1, 10**15)
_ODD = (Q(1, 3), -Q(4, 3), Q(2), -Q(4, 3), Q(1, 3))
_EVEN = (Q(1), -Q(2), Q(0), Q(2), -Q(1))
_INTERNAL = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9))
_OUTER = tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
_EDGES = _INTERNAL + _OUTER


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220
    return context


@pytest.fixture(scope="module")
def protocol():
    return owner._assess_organized_sequential_observation_protocol()


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _q(value):
    return Q(str(value))


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def _combine(coefficients, values):
    return sum(a * b for a, b in zip(coefficients, values))


def _fields(mp, form, phase):
    """Complete fine rows and storage from the independent 90-edge list."""
    gradient, current = [mp.mpf(0)] * 45, [mp.mpf(0)] * 45
    storage = mp.mpf(0)
    for left, right in _EDGES:
        difference = form[left] - form[right]
        angle = phase[right] - phase[left]
        gradient[left] += difference
        gradient[right] -= difference
        current[left] += mp.sin(angle)
        current[right] -= mp.sin(angle)
        storage += difference**2 / 2 + 1 - mp.cos(angle)
    gamma = 1 / (1023 * mp.pi)
    return (
        tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current)),
        tuple(gamma * a / 4 for a in gradient),
        storage,
    )


def _third_form_derivative(mp, form, phase):
    """Differentiate both complete rows twice at the supplied fine state."""
    gamma = 1 / (1023 * mp.pi)
    velocity, phase_velocity, _ = _fields(mp, form, phase)

    def laplacian(values):
        result = [mp.mpf(0)] * 45
        for left, right in _EDGES:
            difference = values[left] - values[right]
            result[left] += difference
            result[right] -= difference
        return result

    phase_acceleration = tuple(gamma * value / 4 for value in laplacian(velocity))
    first_current, second_current = [mp.mpf(0)] * 45, [mp.mpf(0)] * 45
    for left, right in _EDGES:
        angle = phase[right] - phase[left]
        speed = phase_velocity[right] - phase_velocity[left]
        acceleration = phase_acceleration[right] - phase_acceleration[left]
        first = mp.cos(angle) * speed
        second = -mp.sin(angle) * speed**2 + mp.cos(angle) * acceleration
        first_current[left] += first
        first_current[right] -= first
        second_current[left] += second
        second_current[right] -= second
    acceleration = tuple(
        (-a + gamma * b) / 4 for a, b in zip(laplacian(velocity), first_current)
    )
    return tuple(
        (-a + gamma * b) / 4 for a, b in zip(laplacian(acceleration), second_current)
    )


def _state(mp, winding):
    if winding:
        gaps = tuple(
            winding * (2 * mp.pi / 5 + _mp(mp, offset))
            for offset in (Q(1, 100), -Q(3, 200), Q(1, 200), Q(1, 50), -Q(1, 50))
        )
    else:
        gaps = tuple(
            _mp(mp, value) for value in (Q(2, 5), -Q(1, 5), Q(1, 10), -Q(1, 2), Q(1, 5))
        )
    origins = [mp.mpf(0)]
    for gap in gaps[:-1]:
        origins.append(origins[-1] + gap)
    origins = tuple(value - sum(origins) / 5 + mp.mpf(13) / 7 for value in origins)
    residual = tuple(
        _mp(mp, _FINE) * (c + 1) * (3 * (j - 4) ** 2 - 20) / 500
        for c in range(5)
        for j in range(9)
    )
    phase = tuple(
        2 * mp.pi * (j - 4) / 9 + origins[c] + residual[9 * c + j]
        for c in range(5)
        for j in range(9)
    )
    form = tuple(_mp(mp, _FORM) * ((7 * i) % 17 - 8) / 10 for i in range(45))
    return form, phase, gaps


def _shift(mp, phase, marked, amplitude):
    return tuple(
        value + amplitude * (int(i // 9 == marked) - mp.mpf(1) / 5)
        for i, value in enumerate(phase)
    )


def _mean(values, child):
    return sum(values[9 * child : 9 * child + 9]) / 9


def _windings(mp, phase):
    def gap(left, right):
        angle = phase[right] - phase[left]
        return mp.atan2(mp.sin(angle), mp.cos(angle))

    internal = tuple(
        sum(gap(9 * c + j, 9 * c + (j + 1) % 9) for j in range(9)) / (2 * mp.pi)
        for c in range(5)
    )
    outer = tuple(
        sum(gap(9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5)) / (2 * mp.pi)
        for j in range(9)
    )
    return internal, outer


def _readings_from_moments(mp, cosine, sine):
    """Integrate only prescribed algebraic slopes, not a dynamical model."""
    gamma, h, epsilon = 1 / (1023 * mp.pi), _mp(mp, _DELAY), _mp(mp, _EPSILON)
    rows = []
    for c, (cv, sv) in enumerate(zip(cosine, sine)):
        plus = gamma * (mp.sin(epsilon) * cv + (1 - mp.cos(epsilon)) * sv) / 4
        minus = gamma * (-mp.sin(epsilon) * cv + (1 - mp.cos(epsilon)) * sv) / 4
        cumulative = (
            mp.mpf(0),
            mp.mpf(0),
            h * plus,
            h * (plus + minus),
            h * (plus + minus),
        )
        # Each block observes a different receiver. These nuisance terms
        # represent unknown baseline value and affine slope, not five resets.
        rows.append(
            tuple(
                _q(mp.mpf(c - 2) / 1000 + h * k * (c + 1) / 100 + (h * k) ** 2 / 7 + v)
                for k, v in enumerate(cumulative)
            )
        )
    return tuple(rows)


def _necessary_records(mp, winding, protocol, *, adversarial=False):
    """Necessary finite-envelope witnesses; no claim of joint realizability."""
    _, phase, gaps = _state(mp, winding)
    cosine, sine = [], []
    for c in range(5):
        angles = tuple(
            phase[9 * ((c + 1) % 5) + j] - phase[9 * c + j] for j in range(9)
        )
        cosine.append(sum(mp.cos(value) for value in angles) / 9)
        sine.append(sum(mp.sin(value) for value in angles) / 9)
    records = _readings_from_moments(mp, cosine, sine)
    if adversarial:
        allowance = min(
            protocol.odd_finite_remainder_upper_bound / sum(map(abs, _ODD)),
            protocol.even_finite_remainder_upper_bound / sum(map(abs, _EVEN)),
        )
        records = tuple(
            tuple(
                value
                + (allowance if k else Q(0)) * (-1) ** (c + k)
                + _NOISE * (-1) ** (c + k)
                for k, value in enumerate(row)
            )
            for c, row in enumerate(records)
        )
    return records, gaps, tuple(cosine), tuple(sine)


@pytest.mark.parametrize("marked", range(5))
def test_three_events_restore_only_the_command_and_retain_carried_state(marked):
    form = tuple(Q((7 * i) % 19 - 9, 64) for i in range(45))
    initial = tuple(Q((11 * i) % 23 - 11, 256) for i in range(45))
    phase = initial
    cumulative = Q(0)
    carried = (Q(0),) * 45
    for event, multiplier in enumerate((1, -2, 1)):
        # A retained intervening change is supplied algebraically; it is not
        # asserted to have been generated by an integration step.
        drift = tuple(Q((i + event) % 7 - 3, 10**12) for i in range(45))
        phase = tuple(v + d for v, d in zip(phase, drift))
        carried = tuple(v + d for v, d in zip(carried, drift))
        before = phase
        after_x, phase = owner._apply_sequential_observation_event(
            form=form, phase_offsets=phase, marked_child=marked, event_index=event
        )
        cumulative += multiplier * _EPSILON
        assert after_x == form
        assert sum(phase) == sum(before)
        for left, right in _INTERNAL:
            assert phase[right] - phase[left] == before[right] - before[left]
        assert phase == tuple(
            original + d + cumulative * (int(i // 9 == marked) - Q(1, 5))
            for i, (original, d) in enumerate(zip(initial, carried))
        )
    assert cumulative == 0
    assert phase == tuple(v + d for v, d in zip(initial, carried))
    assert phase != initial


@pytest.mark.parametrize("winding", (-1, 0, 1))
def test_full_fine_rows_signed_events_work_and_all_fourteen_cycles(mp, winding):
    form, phase, _ = _state(mp, winding)
    baseline, phase_row, _ = _fields(mp, form, phase)
    assert max(map(abs, phase_row)) > 0
    gamma, epsilon = 1 / (1023 * mp.pi), _mp(mp, _EPSILON)
    total_absolute_work = mp.mpf(0)
    for marked in range(5):
        receiver = (marked + 1) % 5
        angles = tuple(
            phase[9 * receiver + j] - phase[9 * marked + j] for j in range(9)
        )
        cv = sum(mp.cos(value) for value in angles) / 9
        sv = sum(mp.sin(value) for value in angles) / 9
        current = phase
        for multiplier, offset in zip((1, -2, 1), (1, -1, 0)):
            before_storage = _fields(mp, form, current)[2]
            current = _shift(mp, current, marked, multiplier * epsilon)
            dx, dt, storage = _fields(mp, form, current)
            work = storage - before_storage
            total_absolute_work += abs(work)
            assert abs(work) <= 18 * abs(multiplier) * epsilon < 4
            assert max(abs(a - b) for a, b in zip(dt, phase_row)) == 0
            actual = _mean(tuple(a - b for a, b in zip(dx, baseline)), receiver)
            expected = (
                gamma
                * (mp.sin(offset * epsilon) * cv + (1 - mp.cos(offset * epsilon)) * sv)
                / 4
            )
            assert abs(actual - expected) < mp.mpf("1e-200")
            internal, outer = _windings(mp, current)
            assert max(abs(value - 1) for value in internal) < mp.mpf("1e-200")
            assert max(abs(value - winding) for value in outer) < mp.mpf("1e-200")
    assert total_absolute_work < 36 < 40


def test_endpoint_combinations_cancel_constant_and_affine_background_slopes():
    for coefficients in (_ODD, _EVEN):
        for degree in (0, 1, 2):
            assert _combine(coefficients, tuple(Q(k) ** degree for k in range(5))) == 0
    # A background with curved slope is not discarded by the even channel.
    assert _combine(_EVEN, tuple(Q(k) ** 3 for k in range(5))) == -12
    assert _combine(_ODD, tuple(Q(k) ** 3 for k in range(5))) == 0
    positive, negative = Q(7, 11), -Q(2, 13)
    rows = (Q(0), Q(0), positive, positive + negative, positive + negative)
    assert _combine(_ODD, rows) == positive - negative
    assert _combine(_EVEN, rows) == positive + negative
    quadratic_background = tuple(Q(k) ** 2 for k in range(5))
    increments = tuple(
        b - a for a, b in zip(quadratic_background, quadratic_background[1:])
    )
    assert increments[1] - increments[2] == -2
    assert _combine(_ODD, quadratic_background) == 0


@pytest.mark.parametrize("winding", (-1, 0, 1))
def test_baseline_third_derivative_bound_contains_independently_differentiated_rows(
    mp, protocol, winding
):
    form, phase, _ = _state(mp, winding)
    derivative = _third_form_derivative(mp, form, phase)
    maximum = max(map(abs, derivative))
    assert 0 < maximum <= _mp(mp, protocol.baseline_third_derivative_upper_bound)
    assert max(abs(_mean(derivative, c)) for c in range(5)) <= maximum
    # The bound refers to an unforced smooth continuation of the actual
    # state. It does not differentiate through a supplied phase event.
    assert max(map(abs, _fields(mp, form, phase)[1])) > 0


def test_piecewise_backaction_budget_integrates_each_actual_window_weight(mp, protocol):
    h = _DELAY
    odd_weights, even_weights = (Q(1, 3), Q(1), Q(1), Q(1, 3)), (Q(1),) * 4
    moments = []
    for weights in (odd_weights, even_weights):
        moments.append(
            tuple(
                sum(
                    weight
                    * (((k + 1) * h) ** (power + 1) - (k * h) ** (power + 1))
                    / (power + 1)
                    for k, weight in enumerate(weights)
                )
                for power in (1, 2)
            )
        )
    assert moments == [(Q(16, 3) * h**2, Q(116, 9) * h**3), (8 * h**2, Q(64, 3) * h**3)]
    gamma, elapsed, epsilon = 1 / (1023 * mp.pi), _mp(mp, _DURATION), _mp(mp, _EPSILON)
    form_bound = _mp(mp, _FORM) + gamma * elapsed
    m3 = (
        8 * form_bound
        + 4 * gamma
        + 16 * gamma**2 * form_bound
        + 4 * gamma**3
        + 16 * gamma**3 * form_bound**2
    )
    assert form_bound <= _mp(mp, protocol.observation_form_upper_bound)
    assert m3 <= _mp(mp, protocol.baseline_third_derivative_upper_bound)
    for moments_for_channel, baseline_factor, actual in zip(
        moments,
        (Q(2, 3), Q(2)),
        (
            protocol.odd_finite_remainder_upper_bound,
            protocol.even_finite_remainder_upper_bound,
        ),
    ):
        first, second = map(lambda value: _mp(mp, value), moments_for_channel)
        independent = (
            4 * gamma * epsilon * (1 + gamma * form_bound) * first
            + 4 * gamma**3 * epsilon * second
            + _mp(mp, baseline_factor) * m3 * _mp(mp, h) ** 3
        )
        assert independent <= _mp(mp, actual)
        assert abs(_mp(mp, actual) - independent) < mp.mpf("1e-40")


def test_uniform_history_certificate_retains_initial_transport_noise_work_and_clock(
    mp, protocol
):
    gamma, h, duration = 1 / (1023 * mp.pi), _mp(mp, _DELAY), _mp(mp, _DURATION)
    epsilon = _mp(mp, _EPSILON)
    drift = 4 * gamma * (_mp(mp, _FORM) + gamma * duration) * duration
    assert 0 < drift <= _mp(mp, protocol.initial_edge_drift_upper_bound)
    assert (
        protocol.continuous_edge_drift_upper_bound
        == protocol.initial_edge_drift_upper_bound
    )
    assert (
        0
        < gamma * h * mp.sin(epsilon) * drift / 2
        <= _mp(mp, protocol.odd_initial_transport_error_upper_bound)
    )
    assert (
        0
        < gamma * h * (1 - mp.cos(epsilon)) * drift / 2
        <= _mp(mp, protocol.even_initial_transport_error_upper_bound)
    )
    assert (
        protocol.odd_total_error_upper_bound
        == protocol.odd_finite_remainder_upper_bound
        + Q(16, 3) * _NOISE
        + protocol.odd_initial_transport_error_upper_bound
    )
    assert (
        protocol.even_total_error_upper_bound
        == protocol.even_finite_remainder_upper_bound
        + 6 * _NOISE
        + protocol.even_initial_transport_error_upper_bound
    )
    _contains(
        mp, protocol.cosine_inverse_factor_bounds, 2 / (gamma * h * mp.sin(epsilon))
    )
    _contains(
        mp, protocol.sine_inverse_factor_bounds, 2 / (gamma * h * (1 - mp.cos(epsilon)))
    )
    assert 0 < protocol.cycle_sum_width_upper_bound < Q(97, 100)
    assert protocol.winding_resolution_margin > Q(53, 10)
    assert protocol.total_elapsed_scaled_time == _DURATION
    assert protocol.total_elapsed_original_time == _DURATION * Q(1024, 1023)
    assert protocol.elapsed_block_scaled_time == 4 * _DELAY
    assert protocol.marked_child_order == tuple(range(5))
    assert protocol.receiver_child_order == (1, 2, 3, 4, 0)
    assert protocol.window_command_multipliers == (0, 1, -1, 0)
    assert protocol.event_multipliers == (1, -2, 1)
    assert protocol.per_block_absolute_event_work_upper_bounds == (
        18 * _EPSILON,
        36 * _EPSILON,
        18 * _EPSILON,
    )
    assert (
        protocol.total_absolute_event_work_upper_bound
        == 5 * sum(protocol.per_block_absolute_event_work_upper_bounds)
        == 36
    )
    assert protocol.total_work_margin == 4
    assert protocol.event_work_margin == Q(2, 5)
    for box, amount in zip(
        protocol.per_block_signed_event_work_bounds,
        protocol.per_block_absolute_event_work_upper_bounds,
    ):
        assert box.lo <= -amount < 0 < amount <= box.hi
    outer_margin = mp.pi / 2 - mp.mpf(7) / 5 - 2 * _mp(mp, _FINE) - epsilon - drift
    assert 0 < _mp(mp, protocol.outer_acute_gap_margin) <= outer_margin
    assert (
        protocol.uniform_identification_certified
        and protocol.finite_identity_certified
        and protocol.work_certified
        and protocol.admitted
    )
    assert not protocol.unresolved_conditions


@pytest.mark.parametrize("coefficients,expected", ((_ODD, Q(16, 3)), (_EVEN, Q(6))))
def test_shared_raw_endpoint_errors_attain_exact_correlation_budget(
    coefficients, expected
):
    errors = tuple(
        _NOISE if coefficient >= 0 else -_NOISE for coefficient in coefficients
    )
    assert _combine(coefficients, errors) == expected * _NOISE
    increments = tuple(right - left for left, right in zip(errors, errors[1:]))
    increment_weights = (
        (-Q(1, 3), Q(1), -Q(1), Q(1, 3))
        if coefficients == _ODD
        else (-Q(1), Q(1), Q(1), -Q(1))
    )
    assert _combine(increment_weights, increments) == expected * _NOISE


@pytest.mark.parametrize("winding", (-1, 0, 1))
@pytest.mark.parametrize("adversarial", (False, True))
def test_necessary_envelope_records_keep_all_three_initial_sectors(
    mp, protocol, winding, adversarial
):
    readings, gaps, cosine, sine = _necessary_records(
        mp, winding, protocol, adversarial=adversarial
    )
    result = owner._infer_sequential_outer_winding(readings=readings)
    assert result.candidate_windings == (winding,)
    assert result.identified_winding == winding
    assert result.status == "conditionally_identified"
    assert result.readings == readings
    assert result.odd_reading_differences == tuple(
        _combine(_ODD, row) for row in readings
    )
    assert result.even_reading_combinations == tuple(
        _combine(_EVEN, row) for row in readings
    )
    for c in range(5):
        _contains(mp, result.cosine_bounds[c], cosine[c])
        _contains(mp, result.sine_bounds[c], sine[c])
        _contains(mp, result.child_origin_angle_bounds[c], gaps[c])
    _contains(mp, result.cycle_sum_bounds, 2 * mp.pi * winding)
    assert result.cycle_sum_bounds.width <= protocol.cycle_sum_width_upper_bound


def test_chart_edge_sector_zero_retains_all_five_cycle_terms(mp, protocol):
    gaps = tuple(_mp(mp, a) for a in (Q(7, 5), Q(7, 5), -Q(7, 5), -Q(7, 5), Q(0)))
    records = _readings_from_moments(
        mp, tuple(mp.cos(a) for a in gaps), tuple(mp.sin(a) for a in gaps)
    )
    result = owner._infer_sequential_outer_winding(readings=records)
    assert result.identified_winding == 0
    for box, actual in zip(result.child_origin_angle_bounds, gaps):
        _contains(mp, box, actual)
        assert box.width <= protocol.child_origin_angle_width_upper_bound


@pytest.mark.parametrize("kind", ("zero_signal", "short_resultant", "nonclosing_cycle"))
def test_necessary_local_and_global_contradictions_are_not_winding_zero(mp, kind):
    if kind == "zero_signal":
        records = ((Q(0),) * 5,) * 5
    else:
        angle = mp.mpf(1) / 5 if kind == "nonclosing_cycle" else mp.mpf(0)
        amplitude = mp.mpf(1) / 2 if kind == "short_resultant" else mp.mpf(1)
        records = _readings_from_moments(
            mp, (amplitude * mp.cos(angle),) * 5, (amplitude * mp.sin(angle),) * 5
        )
    result = owner._infer_sequential_outer_winding(readings=records)
    assert result.identified_winding is None
    assert not result.candidate_windings
    assert result.status == "incompatible_records_or_premises"
    assert result.candidate_meaning == "necessary_compatibility_not_joint_realizability"
    if kind == "nonclosing_cycle":
        assert all(result.resultant_compatible)
        _contains(mp, result.cycle_sum_bounds, mp.mpf(1))
    elif kind == "short_resultant":
        assert not any(result.resultant_compatible)


def test_exact_large_common_offsets_cancel_but_fail_primitive_state_premise(
    mp, protocol
):
    readings, _, _, _ = _necessary_records(mp, 1, protocol)
    original = owner._infer_sequential_outer_winding(readings=readings)
    huge = Q(10**500) + Q(1, 10**500)
    translated = tuple(tuple(value + huge for value in row) for row in readings)
    changed = owner._infer_sequential_outer_winding(readings=translated)
    assert changed.odd_reading_differences == original.odd_reading_differences
    assert changed.even_reading_combinations == original.even_reading_combinations
    assert not any(changed.raw_readings_compatible)
    assert changed.identified_winding is None
    assert changed.status == "incompatible_records_or_premises"


def test_unavailable_certificate_never_promotes_compatible_records(
    mp, protocol, monkeypatch
):
    records, _, _, _ = _necessary_records(mp, 1, protocol)
    unavailable = replace(
        protocol,
        admitted=False,
        uniform_identification_certified=False,
        unresolved_conditions=("unavailable_history_premise",),
    )
    monkeypatch.setattr(
        owner, "_assess_organized_sequential_observation_protocol", lambda: unavailable
    )
    result = owner._infer_sequential_outer_winding(readings=records)
    assert result.candidate_windings == (1,)
    assert result.identified_winding is None
    assert result.status == "unavailable_protocol_certificate"
    assert "unavailable_history_premise" in result.unresolved_conditions


def test_public_inverse_never_recovers_hidden_acquisition_or_a_matched_branch(
    mp, protocol, monkeypatch
):
    records, _, _, _ = _necessary_records(mp, 1, protocol)

    def forbidden(*args, **kwargs):
        pytest.fail("public inverse recovered hidden source or a matched branch")

    monkeypatch.setattr(
        previous, "_assess_outer_observation_source_inclusion", forbidden
    )
    monkeypatch.setattr(previous, "_infer_organized_outer_winding", forbidden)
    result = owner._infer_sequential_outer_winding(readings=records)
    assert result.identified_winding == 1


@pytest.mark.parametrize("row", range(5))
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0))
)
def test_each_block_admits_its_final_primitive_before_assessment(row, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached protocol assessment")

    monkeypatch.setattr(
        owner, "_assess_organized_sequential_observation_protocol", forbidden
    )
    readings = [(Q(0),) * 5 for _ in range(5)]
    readings[row] = (Q(0),) * 4 + (bad,)
    with pytest.raises((TypeError, ValueError)):
        owner._infer_sequential_outer_winding(readings=readings)


@pytest.mark.parametrize("bad", ((), ((),) * 4, ((),) * 6, "00000", {0}, None))
def test_outer_record_shape_is_not_broadcast_or_reordered(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._infer_sequential_outer_winding(readings=bad)


@pytest.mark.parametrize("bad", ((), (Q(0),) * 4, (Q(0),) * 6, "00000", {0}, None))
def test_each_record_row_has_five_ordered_endpoints(bad):
    rows = [(Q(0),) * 5 for _ in range(5)]
    rows[-1] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._infer_sequential_outer_winding(readings=rows)


@pytest.mark.parametrize(
    "extra",
    (
        "source",
        "phase",
        "phase_origins",
        "winding",
        "expected_sector",
        "unprobed_means",
        "baseline",
        "certified",
        "horizon",
        "amplitude",
        "raw_error",
        "trajectory",
    ),
)
def test_hidden_state_matched_branches_and_retuning_are_not_inverse_inputs(extra):
    with pytest.raises(TypeError):
        owner._infer_sequential_outer_winding(readings=((Q(0),) * 5,) * 5, **{extra: 0})


@pytest.mark.parametrize("field", ("marked_child", "event_index"))
@pytest.mark.parametrize("bad", (True, "1", Q(1), 1.0, None, 5, -1))
def test_event_labels_admit_only_declared_exact_discrete_values(field, bad):
    arguments = dict(
        form=(Q(0),) * 45, phase_offsets=(Q(0),) * 45, marked_child=0, event_index=0
    )
    arguments[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._apply_sequential_observation_event(**arguments)


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, float("nan"), complex(0)))
def test_event_rows_reject_invalid_fine_primitives(field, bad):
    arguments = dict(
        form=(Q(0),) * 45, phase_offsets=(Q(0),) * 45, marked_child=0, event_index=0
    )
    arguments[field] = (Q(0),) * 44 + (bad,)
    with pytest.raises((TypeError, ValueError)):
        owner._apply_sequential_observation_event(**arguments)


def test_reports_keep_normalized_immutable_primitive_records(mp, protocol):
    records, _, _, _ = _necessary_records(mp, 0, protocol)
    lists = [list(row) for row in records]
    result = owner._infer_sequential_outer_winding(readings=lists)
    lists[-1][-1] = Q(999)
    assert result.readings == records
    with pytest.raises(FrozenInstanceError):
        result.identified_winding = 1
    with pytest.raises(FrozenInstanceError):
        protocol.status = "forged"
