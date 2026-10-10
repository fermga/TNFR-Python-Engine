"""Independent static contracts for the frozen public-only winding inverse.

Synthetic records below are witnesses in necessary finite-response envelopes.
They are not integrated trajectories or experimentally acquired observations.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_outer_observation as owner
from tnfr.physics._sine_preparation import _prepared_degree_four_probe_remainder


_EPSILON = Q(1, 10)
_DELAY = Q(1, 100000)
_FORM = Q(1, 100)
_FINE = Q(1, 10**6)
_NOISE = Q(1, 10**15)
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
    return owner._assess_organized_outer_observation_protocol()


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _q(value):
    return Q(str(value))


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def _mean(values, child):
    return sum(values[9 * child : 9 * child + 9]) / 9


def _fields(mp, form, phase):
    """Complete fine edge sums, independent of the observation assessor."""
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
    dx = tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current))
    dt = tuple(gamma * a / 4 for a in gradient)
    return dx, dt, storage


def _state(mp, winding, *, fine=True):
    """A nonuniform admitted state, without claiming a reached trajectory."""
    if winding:
        gaps = tuple(
            winding * (2 * mp.pi / 5 + _mp(mp, offset))
            for offset in (Q(1, 100), -Q(3, 200), Q(1, 200), Q(1, 50), -Q(1, 50))
        )
    else:
        gaps = tuple(
            map(lambda x: _mp(mp, x), (Q(2, 5), -Q(1, 5), Q(1, 10), -Q(1, 2), Q(1, 5)))
        )
    origins = [mp.mpf(0)]
    for gap in gaps[:-1]:
        origins.append(origins[-1] + gap)
    origins = tuple(value - sum(origins) / 5 + mp.mpf(13) / 7 for value in origins)
    residual = tuple(
        _mp(mp, _FINE) * (c + 1) * (3 * (j - 4) ** 2 - 20) / 500 if fine else mp.mpf(0)
        for c in range(5)
        for j in range(9)
    )
    phase = tuple(
        2 * mp.pi * (j - 4) / 9 + origins[c] + residual[9 * c + j]
        for c in range(5)
        for j in range(9)
    )
    form = tuple(_mp(mp, _FORM) * ((7 * i) % 17 - 8) / 10 for i in range(45))
    return form, phase, gaps, residual


def _probed(mp, phase, marked, sign):
    return tuple(
        value + sign * _mp(mp, _EPSILON) * (int(i // 9 == marked) - mp.mpf(1) / 5)
        for i, value in enumerate(phase)
    )


def _wrapped(mp, angle):
    return mp.atan2(mp.sin(angle), mp.cos(angle))


def _windings(mp, phase):
    internal = tuple(
        sum(
            _wrapped(mp, phase[9 * c + (j + 1) % 9] - phase[9 * c + j])
            for j in range(9)
        )
        / (2 * mp.pi)
        for c in range(5)
    )
    outer = tuple(
        sum(
            _wrapped(mp, phase[9 * ((c + 1) % 5) + j] - phase[9 * c + j])
            for c in range(5)
        )
        / (2 * mp.pi)
        for j in range(9)
    )
    return internal, outer


def _necessary_records(mp, winding, *, adversarial=False):
    """Construct finite-envelope witnesses, never asserted actual responses."""
    form, phase, gaps, _ = _state(mp, winding)
    gamma, h, epsilon = 1 / (1023 * mp.pi), _mp(mp, _DELAY), _mp(mp, _EPSILON)
    x, noise = _mp(mp, _FORM), _mp(mp, _NOISE)
    remainder = (
        2 * gamma * epsilon * h**2
        + 2 * gamma**2 * epsilon * (x + gamma * h) * h**2
        + mp.mpf(4) * gamma**3 * epsilon * h**3 / 3
    )
    baseline = tuple(mp.mpf(c - 2) / 1000 for c in range(5))
    records = [list(baseline), [], []]
    cosine, sine = [], []
    for c in range(5):
        edge_angles = tuple(
            phase[9 * ((c + 1) % 5) + j] - phase[9 * c + j] for j in range(9)
        )
        cosine.append(sum(mp.cos(value) for value in edge_angles) / 9)
        sine.append(sum(mp.sin(value) for value in edge_angles) / 9)
        for row, sign in ((1, 1), (2, -1)):
            onset = (
                gamma
                / 4
                * (sign * mp.sin(epsilon) * cosine[c] + (1 - mp.cos(epsilon)) * sine[c])
            )
            error = sign * (-1) ** c * remainder if adversarial else 0
            reading_noise = (-1) ** (row + c) * noise if adversarial else 0
            records[row].append(baseline[c] + h * onset + error + reading_noise)
        if adversarial:
            records[0][c] += (-1) ** c * noise
    return (
        dict(
            unprobed_means=tuple(_q(value) for value in records[0]),
            positive_probe_means=tuple(_q(value) for value in records[1]),
            negative_probe_means=tuple(_q(value) for value in records[2]),
        ),
        gaps,
        tuple(cosine),
        tuple(sine),
    )


@pytest.mark.parametrize("marked", range(5))
@pytest.mark.parametrize("sign", (-1, 1))
def test_each_supplied_jump_preserves_fine_forms_and_internal_differences(marked, sign):
    form = tuple(Q((7 * i) % 19 - 9, 64) for i in range(45))
    phase = tuple(Q((11 * i) % 23 - 11, 256) for i in range(45))
    after_x, after_t = owner._apply_outer_observation_probe(
        form=form, phase_offsets=phase, marked_child=marked, sign=sign
    )
    assert after_x == form
    assert sum(after_t) == sum(phase)
    assert tuple(_mean(after_t, c) - _mean(phase, c) for c in range(5)) == tuple(
        sign * _EPSILON * (int(c == marked) - Q(1, 5)) for c in range(5)
    )
    for left, right in _INTERNAL:
        assert after_t[right] - after_t[left] == phase[right] - phase[left]
    changed = tuple(
        after_t[right] - after_t[left] - phase[right] + phase[left]
        for left, right in _OUTER
    )
    assert changed.count(_EPSILON) == changed.count(-_EPSILON) == 9
    assert changed.count(Q(0)) == 27


@pytest.mark.parametrize("winding", (-1, 0, 1))
def test_all_ten_probes_match_complete_rows_work_and_each_fine_cycle(mp, winding):
    form, phase, gaps, residual = _state(mp, winding)
    assert max(abs(value) for value in form) < _mp(mp, _FORM)
    assert max(abs(value) for value in residual) < _mp(mp, _FINE)
    assert max(abs(value) for value in gaps) < mp.mpf(7) / 5
    for child in range(5):
        assert abs(sum(residual[9 * child : 9 * child + 9])) < mp.mpf("1e-200")
    dx, dt, storage = _fields(mp, form, phase)
    assert max(abs(value) for value in dt) > 0
    gamma, epsilon = 1 / (1023 * mp.pi), _mp(mp, _EPSILON)
    for marked in range(5):
        receiver = (marked + 1) % 5
        angles = tuple(
            phase[9 * receiver + j] - phase[9 * marked + j] for j in range(9)
        )
        cosine = sum(mp.cos(value) for value in angles) / 9
        sine = sum(mp.sin(value) for value in angles) / 9
        for sign in (-1, 1):
            probed = _probed(mp, phase, marked, sign)
            changed_dx, changed_dt, changed_storage = _fields(mp, form, probed)
            actual = _mean(tuple(a - b for a, b in zip(changed_dx, dx)), receiver)
            expected = (
                gamma
                / 4
                * (sign * mp.sin(epsilon) * cosine + (1 - mp.cos(epsilon)) * sine)
            )
            assert abs(actual - expected) < mp.mpf("1e-200")
            assert changed_dt == dt
            assert abs(changed_storage - storage) < 18 * epsilon < 2
            internal, outer = _windings(mp, probed)
            assert max(abs(value - 1) for value in internal) < mp.mpf("1e-200")
            assert max(abs(value - winding) for value in outer) < mp.mpf("1e-200")


@pytest.mark.parametrize("winding", (-1, 0, 1))
@pytest.mark.parametrize("adversarial", (False, True))
def test_public_inverse_keeps_true_sector_for_independent_finite_envelope_witnesses(
    mp, winding, adversarial
):
    readings, gaps, cosine, sine = _necessary_records(
        mp, winding, adversarial=adversarial
    )
    result = owner._infer_organized_outer_winding(**readings)
    assert result.candidate_windings == (winding,)
    assert result.identified_winding == winding
    for c in range(5):
        _contains(mp, result.cosine_bounds[c], cosine[c])
        _contains(mp, result.sine_bounds[c], sine[c])
        _contains(mp, result.child_origin_angle_bounds[c], gaps[c])
    _contains(mp, result.cycle_sum_bounds, 2 * mp.pi * winding)
    assert result.odd_reading_differences == tuple(
        p - m
        for p, m in zip(
            readings["positive_probe_means"], readings["negative_probe_means"]
        )
    )
    assert result.even_reading_combinations == tuple(
        p + m - 2 * b
        for b, p, m in zip(
            readings["unprobed_means"],
            readings["positive_probe_means"],
            readings["negative_probe_means"],
        )
    )


def test_common_baseline_cancels_exactly_in_odd_channel_and_is_twice_in_even(mp):
    readings, _, _, _ = _necessary_records(mp, 1)
    original = owner._infer_organized_outer_winding(**readings)
    shifted = readings | {
        "unprobed_means": tuple(value + _NOISE for value in readings["unprobed_means"])
    }
    changed = owner._infer_organized_outer_winding(**shifted)
    assert changed.odd_reading_differences == original.odd_reading_differences
    assert changed.cosine_bounds == original.cosine_bounds
    assert changed.even_reading_combinations == tuple(
        value - 2 * _NOISE for value in original.even_reading_combinations
    )
    # Arbitrary extremal raw errors attain both declared allowances; no
    # independence or averaging premise reduces them.
    assert _NOISE - (-_NOISE) == 2 * _NOISE
    assert _NOISE + _NOISE - 2 * (-_NOISE) == 4 * _NOISE


def test_uniform_inverse_budget_keeps_finite_motion_noise_fine_state_and_clock(
    mp, protocol
):
    gamma, h = 1 / (1023 * mp.pi), _mp(mp, _DELAY)
    epsilon, x = _mp(mp, _EPSILON), _mp(mp, _FORM)
    remainder = (
        2 * gamma * epsilon * h**2
        + 2 * gamma**2 * epsilon * (x + gamma * h) * h**2
        + mp.mpf(4) * gamma**3 * epsilon * h**3 / 3
    )
    assert remainder <= _mp(mp, protocol.finite_remainder_upper_bound)
    assert protocol.finite_remainder_upper_bound > 6 * _NOISE
    assert (
        protocol.odd_total_error_upper_bound
        == 2 * protocol.finite_remainder_upper_bound + 2 * _NOISE
    )
    assert (
        protocol.even_total_error_upper_bound
        == 2 * protocol.finite_remainder_upper_bound + 4 * _NOISE
    )
    _contains(
        mp, protocol.cosine_inverse_factor_bounds, 2 / (gamma * h * mp.sin(epsilon))
    )
    _contains(
        mp, protocol.sine_inverse_factor_bounds, 2 / (gamma * h * (1 - mp.cos(epsilon)))
    )
    assert (
        0
        < _mp(mp, protocol.cosine_prior_lower_bound)
        <= mp.cos(mp.mpf(7) / 5 + 2 * _mp(mp, _FINE))
    )
    assert 0 < protocol.cycle_sum_width_upper_bound < Q(1, 5)
    assert 0 < _mp(mp, protocol.winding_resolution_margin) < 2 * mp.pi
    assert protocol.elapsed_probe_original_time == _DELAY * Q(1024, 1023)
    assert protocol.marked_child_order == tuple(range(5))
    assert protocol.receiver_child_order == (1, 2, 3, 4, 0)
    assert protocol.uniform_identification_certified and protocol.admitted
    assert not protocol.unresolved_conditions


def test_work_and_identity_use_actual_broad_class_only_for_the_fixed_horizon(
    mp, protocol
):
    gamma, h = 1 / (1023 * mp.pi), _mp(mp, _DELAY)
    drift = 4 * gamma * (_mp(mp, _FORM) + gamma * h) * h
    assert 0 < drift <= _mp(mp, protocol.continuous_edge_drift_upper_bound)
    assert protocol.absolute_event_work_upper_bound == 18 * _EPSILON
    assert protocol.work_margin == Q(2) - 18 * _EPSILON > 0
    _contains(mp, protocol.signed_event_work_bounds, mp.mpf(0))
    outer_limit = (
        mp.pi / 2 - mp.mpf(7) / 5 - 2 * _mp(mp, _FINE) - _mp(mp, _EPSILON) - drift
    )
    internal_limit = 2 * mp.pi / 9 - 2 * _mp(mp, _FINE) - drift
    assert 0 < _mp(mp, protocol.outer_acute_gap_margin) <= outer_limit
    assert 0 < _mp(mp, protocol.internal_positive_gap_margin) <= internal_limit
    assert protocol.finite_identity_certified and protocol.work_certified
    assert (
        "identity_retained_only_through_declared_finite_observation_horizon"
        in protocol.scope
    )


def _readings_from_moments(mp, cosine, sine):
    """Algebraic records to test necessary compatibility, not state existence."""
    gamma, h, epsilon = 1 / (1023 * mp.pi), _mp(mp, _DELAY), _mp(mp, _EPSILON)
    positive = tuple(
        _q(h * gamma * (mp.sin(epsilon) * c + (1 - mp.cos(epsilon)) * s) / 4)
        for c, s in zip(cosine, sine)
    )
    negative = tuple(
        _q(h * gamma * (-mp.sin(epsilon) * c + (1 - mp.cos(epsilon)) * s) / 4)
        for c, s in zip(cosine, sine)
    )
    return dict(
        unprobed_means=(Q(0),) * 5,
        positive_probe_means=positive,
        negative_probe_means=negative,
    )


def test_chart_edge_states_and_all_five_cycle_terms_retain_uniform_resolution(
    mp, protocol
):
    # Exact joint origins exist for this sector-zero boundary profile.
    gaps = tuple(
        _mp(mp, value) for value in (Q(7, 5), Q(7, 5), -Q(7, 5), -Q(7, 5), Q(0))
    )
    readings = _readings_from_moments(
        mp, tuple(mp.cos(a) for a in gaps), tuple(mp.sin(a) for a in gaps)
    )
    result = owner._infer_organized_outer_winding(**readings)
    assert result.identified_winding == 0
    assert result.candidate_windings == (0,)
    for angle, actual in zip(result.child_origin_angle_bounds, gaps):
        _contains(mp, angle, actual)
        assert angle.width <= protocol.child_origin_angle_width_upper_bound
    assert result.cycle_sum_bounds.width <= protocol.cycle_sum_width_upper_bound


def test_locally_valid_interface_data_can_fail_the_complete_cycle_constraint(mp):
    # Five equal angles of 1/5 rad pass every local trigonometric range,
    # but their sum is one radian and cannot close the actual cycle.
    angle = mp.mpf(1) / 5
    readings = _readings_from_moments(mp, (mp.cos(angle),) * 5, (mp.sin(angle),) * 5)
    result = owner._infer_organized_outer_winding(**readings)
    assert all(box is not None for box in result.child_origin_angle_bounds)
    _contains(mp, result.cycle_sum_bounds, mp.mpf(1))
    assert not result.candidate_windings
    assert result.identified_winding is None
    assert result.status == "incompatible_records_or_premises"


def test_failed_local_moment_admission_does_not_become_winding_zero():
    readings = dict(
        unprobed_means=(Q(0),) * 5,
        positive_probe_means=(Q(0),) * 5,
        negative_probe_means=(Q(0),) * 5,
    )
    result = owner._infer_organized_outer_winding(**readings)
    assert any(box is None for box in result.cosine_bounds)
    assert result.cycle_sum_bounds is None
    assert result.identified_winding is None
    assert not result.candidate_windings
    assert result.status == "incompatible_records_or_premises"


def test_known_fine_phase_resultant_contradiction_is_rejected(mp):
    # The actual fine class has boundary resultant norm >= cos(2*zeta).
    # These artificial moments lie near norm 1/2, so no member realizes
    # them. Marginal angle/cycle conditions alone would accept winding 0,
    # so the inverse must also retain the known necessary annulus.
    readings = _readings_from_moments(mp, (mp.mpf(1) / 2,) * 5, (mp.mpf(0),) * 5)
    result = owner._infer_organized_outer_winding(**readings)
    for c, s in zip(result.cosine_bounds, result.sine_bounds):
        assert (
            _mp(mp, c.abs_max) ** 2 + _mp(mp, s.abs_max) ** 2
            < mp.cos(2 * _mp(mp, _FINE)) ** 2
        )
    assert not any(result.resultant_compatible)
    assert not result.candidate_windings
    assert result.identified_winding is None
    assert result.status == "incompatible_records_or_premises"
    assert result.candidate_meaning == "necessary_compatibility_not_joint_realizability"
    assert (
        result.identification_scope
        == "conditional_on_the_declared_same_source_state_and_law"
    )


def test_impossible_raw_mean_offsets_preserve_exact_channels_but_reject_the_state_class(
    mp,
):
    readings, _, _, _ = _necessary_records(mp, 1)
    original = owner._infer_organized_outer_winding(**readings)
    huge = Q(10**500) + Q(1, 10**500)
    translated = {
        name: tuple(value + huge for value in values)
        for name, values in readings.items()
    }
    result = owner._infer_organized_outer_winding(**translated)
    assert result.odd_reading_differences == original.odd_reading_differences
    assert result.even_reading_combinations == original.even_reading_combinations
    assert not any(result.raw_readings_compatible)
    assert not result.candidate_windings
    assert result.identified_winding is None
    assert result.status == "incompatible_records_or_premises"


def test_unavailable_protocol_keeps_candidates_but_never_assigns_an_identity(
    mp, protocol, monkeypatch
):
    readings, _, _, _ = _necessary_records(mp, 1)
    failed = replace(
        protocol,
        admitted=False,
        uniform_identification_certified=False,
        unresolved_conditions=("missing_protocol_premise",),
    )
    monkeypatch.setattr(
        owner, "_assess_organized_outer_observation_protocol", lambda: failed
    )
    result = owner._infer_organized_outer_winding(**readings)
    assert result.candidate_windings == (1,)
    assert result.identified_winding is None
    assert result.status == "unavailable_protocol_certificate"
    assert "missing_protocol_premise" in result.unresolved_conditions


def test_actual_acquired_source_inclusion_is_separate_from_public_inference(
    monkeypatch,
):
    allowance = Q(1, 10**10)
    inclusion = owner._assess_outer_observation_source_inclusion(
        form_error_bounds=(allowance,) * 5, phase_error_bounds=(allowance,) * 5
    )
    assert inclusion.admitted and not inclusion.unresolved_conditions
    assert inclusion.outer_one_form_upper_bound < _FORM
    assert inclusion.outer_zero_form_upper_bound < _FORM
    assert 0 < inclusion.reprojected_fine_phase_upper_bound < _FINE
    assert inclusion.outer_one_origin_gap_upper_bound < Q(7, 5)
    assert inclusion.outer_zero_origin_gap_upper_bound < Q(7, 5)

    def forbidden(*args, **kwargs):
        pytest.fail("public inverse tried to recover hidden acquired source")

    monkeypatch.setattr(owner, "_assess_organized_outer_winding", forbidden)
    result = owner._infer_organized_outer_winding(
        unprobed_means=(Q(0),) * 5,
        positive_probe_means=(Q(0),) * 5,
        negative_probe_means=(Q(0),) * 5,
    )
    assert result.identified_winding is None


def test_one_interface_onsets_do_not_determine_global_winding(mp):
    # Both signs see exactly the same marked opening angle in two complete
    # states. This is a local-onset witness, not equal finite histories.
    states = []
    for winding in (0, 1):
        gaps = (mp.mpf(1),) + ((2 * mp.pi * winding - 1) / 4,) * 4
        assert max(abs(value) for value in gaps) < mp.mpf(7) / 5
        origins = [mp.mpf(0)]
        for gap in gaps[:-1]:
            origins.append(origins[-1] + gap)
        phase = tuple(
            2 * mp.pi * (j - 4) / 9 + origins[c] for c in range(5) for j in range(9)
        )
        zero = (mp.mpf(0),) * 45
        baseline = _fields(mp, zero, phase)[0]
        responses = []
        for sign in (-1, 1):
            dx = _fields(mp, zero, _probed(mp, phase, 0, sign))[0]
            responses.append(_mean(tuple(a - b for a, b in zip(dx, baseline)), 1))
        states.append(tuple(responses))
        internal, outer = _windings(mp, phase)
        assert max(abs(value - 1) for value in internal) < mp.mpf("1e-200")
        assert max(abs(value - winding) for value in outer) < mp.mpf("1e-200")
    assert max(abs(a - b) for a, b in zip(*states)) < mp.mpf("1e-200")


def test_positive_onset_alone_has_a_reflection_ambiguity(mp):
    epsilon, angle = _mp(mp, _EPSILON), mp.mpf(2) / 5
    reflected = epsilon - angle
    left = mp.sin(epsilon - angle) + mp.sin(angle)
    right = mp.sin(epsilon - reflected) + mp.sin(reflected)
    assert angle != reflected
    assert abs(left - right) < mp.mpf("1e-200")
    opposite_left = mp.sin(-epsilon - angle) + mp.sin(angle)
    opposite_right = mp.sin(-epsilon - reflected) + mp.sin(reflected)
    assert abs(opposite_left - opposite_right) > 0


@pytest.mark.parametrize(
    "field", ("unprobed_means", "positive_probe_means", "negative_probe_means")
)
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0))
)
def test_all_raw_primitives_are_admitted_before_assessment(field, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached protocol assessment")

    monkeypatch.setattr(
        owner, "_assess_organized_outer_observation_protocol", forbidden
    )
    readings = dict(
        unprobed_means=(Q(0),) * 5,
        positive_probe_means=(Q(0),) * 5,
        negative_probe_means=(Q(0),) * 5,
    )
    readings[field] = (Q(0),) * 4 + (bad,)
    with pytest.raises((TypeError, ValueError)):
        owner._infer_organized_outer_winding(**readings)


@pytest.mark.parametrize(
    "field", ("unprobed_means", "positive_probe_means", "negative_probe_means")
)
@pytest.mark.parametrize("bad", ((), (Q(0),) * 4, (Q(0),) * 6, "00000", {0}))
def test_raw_reading_shapes_are_never_broadcast_truncated_or_reordered(field, bad):
    readings = dict(
        unprobed_means=(Q(0),) * 5,
        positive_probe_means=(Q(0),) * 5,
        negative_probe_means=(Q(0),) * 5,
    )
    readings[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._infer_organized_outer_winding(**readings)


@pytest.mark.parametrize(
    "extra",
    (
        "source",
        "acquisition",
        "phase_origins",
        "phase",
        "winding",
        "expected_sector",
        "certified",
        "horizon",
        "amplitude",
        "raw_error",
        "trajectory",
    ),
)
def test_hidden_state_cached_answers_and_retuned_protocol_are_not_inverse_inputs(extra):
    readings = dict(
        unprobed_means=(Q(0),) * 5,
        positive_probe_means=(Q(0),) * 5,
        negative_probe_means=(Q(0),) * 5,
    )
    with pytest.raises(TypeError):
        owner._infer_organized_outer_winding(**readings, **{extra: 0})


@pytest.mark.parametrize("field", ("marked_child", "sign"))
@pytest.mark.parametrize("bad", (True, "1", Q(1), 1.0, None, 5, -2))
def test_probe_labels_require_exact_declared_discrete_values(field, bad):
    inputs = dict(form=(Q(0),) * 45, phase_offsets=(Q(0),) * 45, marked_child=0, sign=1)
    inputs[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._apply_outer_observation_probe(**inputs)


@pytest.mark.parametrize("field", ("phase_jump_upper_bound", "elapsed_time"))
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0), -Q(1, 10**1000))
)
def test_shared_finite_bound_admits_new_protocol_parameters_before_arithmetic(
    field, bad
):
    arguments = dict(
        gamma_upper=Q(1),
        initial_form_upper_bound=Q(0),
        phase_jump_upper_bound=Q(1),
        elapsed_time=Q(1, 2),
    )
    arguments[field] = bad
    with pytest.raises((TypeError, ValueError)):
        _prepared_degree_four_probe_remainder(**arguments)


def test_shared_finite_bound_checks_exact_contraction_boundary_and_zero_time():
    arguments = dict(
        gamma_upper=Q(1),
        initial_form_upper_bound=Q(0),
        phase_jump_upper_bound=Q(1),
        elapsed_time=Q(1, 2),
    )
    # At 2*gamma**2*h**2=1/2 the heat comparison denominator is 1/2.
    # The three independently integrated error terms are 1/2, 1/4, 1/6.
    assert _prepared_degree_four_probe_remainder(**arguments) == Q(11, 12)
    with pytest.raises(ValueError):
        _prepared_degree_four_probe_remainder(
            **(arguments | {"elapsed_time": Q(1, 2) + Q(1, 10**1000)})
        )
    assert (
        _prepared_degree_four_probe_remainder(**(arguments | {"elapsed_time": 0})) == 0
    )
    assert (
        _prepared_degree_four_probe_remainder(
            **(arguments | {"phase_jump_upper_bound": 0})
        )
        == 0
    )


def test_shared_finite_bound_retains_tiny_exact_parameters_and_represented_time():
    tiny = Q(1, 10**1000)
    arguments = dict(
        gamma_upper=Q(1),
        initial_form_upper_bound=Q(0),
        phase_jump_upper_bound=Q(1),
        elapsed_time=Q(1, 2),
    )
    small_jump = _prepared_degree_four_probe_remainder(
        **(arguments | {"phase_jump_upper_bound": tiny})
    )
    assert small_jump == Q(11, 12) * tiny > 0
    small_time = _prepared_degree_four_probe_remainder(
        **(arguments | {"elapsed_time": tiny})
    )
    assert small_time == 2 * tiny**2 + Q(10, 3) * tiny**3 > 0
    assert _prepared_degree_four_probe_remainder(
        **(arguments | {"elapsed_time": 0.125})
    ) == _prepared_degree_four_probe_remainder(
        **(arguments | {"elapsed_time": Q(1, 8)})
    )


def test_reports_are_immutable(mp, protocol):
    readings, _, _, _ = _necessary_records(mp, 0)
    result = owner._infer_organized_outer_winding(**readings)
    with pytest.raises(FrozenInstanceError):
        result.identified_winding = 1
    with pytest.raises(FrozenInstanceError):
        protocol.status = "forged"
