"""Independent static laws and source controls for the frozen external contact.

Static test points establish implemented identities and information obstructions,
not acquisition of those points. The finite certificate uses the original sources.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_external_interaction as owner


_ERROR = Q(1, 10**10)
_ZERO = (Q(0),) * 54
_INTERNAL = tuple(
    (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(6) for j in range(9)
)
_OUTER = tuple(
    (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for j in range(9) for c in range(5)
)
_CONTACT = tuple((9 * c + j, 45 + j, Q(1)) for c in range(5) for j in range(9))
_EDGES = _INTERNAL + _OUTER + _CONTACT
_CARRIER_EDGES = tuple((c, (c + 1) % 5, Q(1)) for c in range(5)) + tuple(
    (c, 5, Q(1)) for c in range(5)
)
_FORMS = tuple(Q((7 * i) % 19 - 9, 64) for i in range(54))
_OFFSETS = tuple(Q((11 * i) % 23 - 11, 256) for i in range(54))


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def _phase(mp, offsets):
    return tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value) + (mp.pi / 2 if i >= 45 else 0)
        for i, value in enumerate(offsets)
    )


def _fields(mp, form, phase, edges=_EDGES):
    """Independently sum every unit edge on the supplied static state."""
    gradient, current = [mp.mpf(0)] * len(form), [mp.mpf(0)] * len(form)
    storage = mp.mpf(0)
    for left, right, weight in edges:
        weight = _mp(mp, weight)
        difference, angle = form[left] - form[right], phase[right] - phase[left]
        gradient[left] += weight * difference
        gradient[right] -= weight * difference
        current[left] += weight * mp.sin(angle)
        current[right] -= weight * mp.sin(angle)
        storage += weight * (difference**2 / 2 + 1 - mp.cos(angle))
    gamma = 1 / (1023 * mp.pi)
    return (
        tuple(gradient),
        tuple(current),
        tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current)),
        tuple(gamma * a / 4 for a in gradient),
        storage,
    )


def _observe(**changes):
    return owner._observe_organized_external_interaction(
        **(dict(form=_FORMS, phase_offsets=_OFFSETS) | changes)
    )


def _assess(**changes):
    return owner._assess_organized_external_interaction(
        **(
            dict(form_error_bounds=(_ERROR,) * 6, phase_error_bounds=(_ERROR,) * 6)
            | changes
        )
    )


@pytest.fixture(scope="module")
def observation():
    return _observe()


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_support_is_all_144_edges_with_uniform_quarter_mobility(observation):
    geometry = observation.geometry
    assert geometry.node_order == tuple(range(54))
    assert geometry.internal_edges == _INTERNAL
    assert geometry.outer_edges == _OUTER
    assert geometry.contact_edges == _CONTACT
    assert set(geometry.weighted_edges) == set(_EDGES)
    assert len(geometry.weighted_edges) == len(set(_EDGES)) == 144
    degrees = tuple(sum(i in edge[:2] for edge in _EDGES) for i in range(54))
    assert geometry.support_degrees == degrees == (5,) * 45 + (7,) * 9
    assert geometry.held_capacities == (Q(1),) * 54
    assert geometry.clock_factor == Q(1, 4)


def test_complete_static_rows_and_symbolic_donor_origin(observation, mp):
    form = tuple(_mp(mp, v) for v in _FORMS)
    phase = _phase(mp, _OFFSETS)
    gradient, current, dx, dtheta, storage = _fields(mp, form, phase)
    for i in range(54):
        _contains(mp, observation.phase_lift_bounds[i], phase[i])
        assert abs(_mp(mp, observation.form_gradient[i]) - gradient[i]) < mp.mpf(
            "1e-160"
        )
        _contains(mp, observation.phase_current_bounds[i], current[i])
        for name, factor in (("tau", 1), ("original", mp.mpf(1023) / 1024)):
            _contains(
                mp, getattr(observation, f"form_{name}_rate_bounds")[i], factor * dx[i]
            )
            _contains(
                mp,
                getattr(observation, f"phase_{name}_rate_bounds")[i],
                factor * dtheta[i],
            )
    _contains(mp, observation.storage_bounds, storage)
    assert abs(sum(dx)) < mp.mpf("1e-160")
    assert abs(sum(dtheta)) < mp.mpf("1e-160")
    # Unequal degrees must not silently replace the declared common factor.
    assert abs(dx[45] - 4 * dx[45] / 7) > mp.mpf("1e-5")


def test_both_contact_reactions_and_actual_attachment_storage(observation, mp):
    form, phase = tuple(_mp(mp, v) for v in _FORMS), _phase(mp, _OFFSETS)
    gradient, current, dx, dtheta, work = _fields(mp, form, phase, _CONTACT)
    for i in range(54):
        assert abs(
            _mp(mp, observation.contact_form_gradient[i]) - gradient[i]
        ) < mp.mpf("1e-160")
        _contains(mp, observation.contact_phase_current_bounds[i], current[i])
        _contains(mp, observation.contact_form_tau_rate_bounds[i], dx[i])
        _contains(mp, observation.contact_phase_tau_rate_bounds[i], dtheta[i])
    for j in range(9):
        assert abs(sum(dx[9 * c + j] for c in range(5)) + dx[45 + j]) < mp.mpf("1e-160")
        assert abs(sum(dtheta[9 * c + j] for c in range(5)) + dtheta[45 + j]) < mp.mpf(
            "1e-160"
        )
    for c in range(6):
        _contains(
            mp,
            observation.contact_child_form_tau_rate_bounds[c],
            sum(dx[9 * c : 9 * c + 9]) / 9,
        )
        _contains(
            mp,
            observation.contact_child_phase_tau_rate_bounds[c],
            sum(dtheta[9 * c : 9 * c + 9]) / 9,
        )
    _contains(mp, observation.attachment_work_bounds, work)
    assert work > 0


def test_complete_and_contact_fields_share_one_admitted_edge_evaluation(
    monkeypatch, observation
):
    evaluate = owner._bounded_sine_edge_fields
    calls = []

    def fields(**kwargs):
        calls.append(kwargs["weighted_edges"])
        return evaluate(**kwargs)

    monkeypatch.setattr(owner, "_bounded_sine_edge_fields", fields)
    actual = _observe()
    assert calls == [_EDGES]
    assert actual == observation


def test_complete_storage_balance_and_mean_reactions(observation, mp):
    form, phase = tuple(_mp(mp, v) for v in _FORMS), _phase(mp, _OFFSETS)
    gradient, current, dx, dtheta, _ = _fields(mp, form, phase)
    derivative = sum(a * b - c * d for a, b, c, d in zip(gradient, dx, current, dtheta))
    loss = -sum(v * v for v in gradient) / 4
    assert abs(derivative - loss) < mp.mpf("1e-160")
    assert abs(_mp(mp, observation.storage_tau_rate) - loss) < mp.mpf("1e-160")
    _contains(mp, observation.storage_rate_from_rows_bounds, derivative)
    _contains(mp, observation.balance_residual_bounds, 0)
    for c in range(6):
        assert observation.child_form_means[c] == sum(_FORMS[9 * c : 9 * c + 9]) / 9
        assert (
            observation.child_phase_offset_means[c]
            == sum(_OFFSETS[9 * c : 9 * c + 9]) / 9
        )
        _contains(
            mp,
            observation.child_form_tau_rate_bounds[c],
            sum(dx[9 * c : 9 * c + 9]) / 9,
        )
        _contains(
            mp,
            observation.child_phase_tau_rate_bounds[c],
            sum(dtheta[9 * c : 9 * c + 9]) / 9,
        )


def test_twelve_coordinate_carrier_closes_both_rows_and_storage(mp):
    form = tuple(Q(v, 17) for v in (-5, 2, 8, -3, 4, -6))
    offsets = tuple(Q(v, 23) for v in (-2, 5, 3, -7, 1, 4))
    observed = _observe(
        form=tuple(v for v in form for _ in range(9)),
        phase_offsets=tuple(v for v in offsets for _ in range(9)),
    )
    phase = tuple(
        _mp(mp, v) + (mp.pi / 2 if c == 5 else 0) for c, v in enumerate(offsets)
    )
    gradient, _, dx, dtheta, energy = _fields(
        mp, tuple(_mp(mp, v) for v in form), phase, _CARRIER_EDGES
    )
    for i in range(54):
        _contains(mp, observed.form_tau_rate_bounds[i], dx[i // 9])
        _contains(mp, observed.phase_tau_rate_bounds[i], dtheta[i // 9])
    _contains(
        mp, observed.storage_bounds, 54 * (1 - mp.cos(2 * mp.pi / 9)) + 9 * energy
    )
    assert abs(
        _mp(mp, observed.storage_tau_rate) + 9 * sum(v * v for v in gradient) / 4
    ) < mp.mpf("1e-160")


def test_equal_composite_means_and_winding_do_not_close_external_response(mp):
    amplitude = Q(1, 5)
    origins = (amplitude, -amplitude, Q(0), Q(0), Q(0), Q(0))
    flat = _observe(form=_ZERO, phase_offsets=_ZERO)
    rearranged = _observe(
        form=_ZERO, phase_offsets=tuple(v for v in origins for _ in range(9))
    )
    assert (
        sum(flat.child_phase_offset_means[:5])
        == sum(rearranged.child_phase_offset_means[:5])
        == 0
    )
    gamma = 1 / (1023 * mp.pi)
    flat_rate = -5 * gamma / 4
    rearranged_rate = -gamma * (3 + 2 * mp.cos(_mp(mp, amplitude))) / 4
    _contains(mp, flat.child_form_tau_rate_bounds[5], flat_rate)
    _contains(mp, rearranged.child_form_tau_rate_bounds[5], rearranged_rate)
    assert (
        flat.child_form_tau_rate_bounds[5].hi
        < rearranged.child_form_tau_rate_bounds[5].lo
    )
    # All outer gaps stay on their raw branch and telescope to zero.
    assert all(abs(origins[(c + 1) % 5] - origins[c]) < 1 for c in range(5))


def test_equal_child_means_still_discard_fine_contact_correlations(mp):
    offsets = list(_ZERO)
    offsets[0], offsets[1] = Q(1, 10), -Q(1, 10)
    flat = _observe(form=_ZERO, phase_offsets=_ZERO)
    changed = _observe(form=_ZERO, phase_offsets=offsets)
    assert flat.child_phase_offset_means == changed.child_phase_offset_means
    expected = -(43 + 2 * mp.cos(mp.mpf(1) / 10)) / (36 * 1023 * mp.pi)
    _contains(mp, changed.child_form_tau_rate_bounds[5], expected)
    assert (
        flat.child_form_tau_rate_bounds[5].hi < changed.child_form_tau_rate_bounds[5].lo
    )


@pytest.mark.parametrize("channel", ("form", "phase"))
@pytest.mark.parametrize("child", range(6))
@pytest.mark.parametrize("slot", range(8))
def test_every_relative_preparation_direction_affects_the_fine_law(
    mp, channel, child, slot
):
    form, offsets = list(_ZERO), list(_ZERO)
    target = form if channel == "form" else offsets
    target[9 * child + slot], target[9 * child + 8] = Q(1, 64), -Q(1, 64)
    observed = _observe(form=form, phase_offsets=offsets)
    _, _, dx, dtheta, _ = _fields(
        mp, tuple(_mp(mp, v) for v in form), _phase(mp, offsets)
    )
    index = 9 * child + slot
    _contains(mp, observed.form_tau_rate_bounds[index], dx[index])
    _contains(mp, observed.phase_tau_rate_bounds[index], dtheta[index])
    if channel == "form":
        assert abs(dtheta[index]) > mp.mpf("1e-7")
    else:
        _, _, baseline, _, _ = _fields(mp, (mp.mpf(0),) * 54, _phase(mp, _ZERO))
        assert abs(dx[index] - baseline[index]) > mp.mpf("1e-8")


def _forbidden_rebuild(*args, **kwargs):
    pytest.fail("invalid primitive source data reached scientific source rebuilding")


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "invalid", (True, "1e-10", None, float("nan"), float("inf"), -_ERROR, 2 * _ERROR)
)
def test_all_twelve_allowances_are_admitted_before_any_source_rebuild(
    monkeypatch, field, invalid
):
    monkeypatch.setattr(owner, "_assess_organized_outer_winding", _forbidden_rebuild)
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", _forbidden_rebuild)
    values = [_ERROR] * 6
    values[5] = invalid
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: values})


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "invalid", ((), (_ERROR,) * 5, (_ERROR,) * 7, {_ERROR}, {0: _ERROR})
)
def test_source_arrays_require_six_ordered_allowances(monkeypatch, field, invalid):
    monkeypatch.setattr(owner, "_assess_organized_outer_winding", _forbidden_rebuild)
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", _forbidden_rebuild)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: invalid})


def test_excess_source_generator_is_bounded_before_rebuild(monkeypatch):
    monkeypatch.setattr(owner, "_assess_organized_outer_winding", _forbidden_rebuild)
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", _forbidden_rebuild)
    consumed = []

    def values():
        for index in range(100):
            consumed.append(index)
            yield _ERROR

    with pytest.raises(ValueError):
        _assess(phase_error_bounds=values())
    assert len(consumed) == 7


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("invalid", (True, "0", float("nan"), float("inf"), 1j))
def test_static_observer_rejects_invalid_authoritative_coordinates(field, invalid):
    values = list(_ZERO)
    values[-1] = invalid
    with pytest.raises((TypeError, ValueError)):
        _observe(**{field: values})


def test_static_admission_preserves_subfloat_signed_forms():
    values = list(_ZERO)
    values[0], values[1] = Q(1, 10**400), -Q(1, 10**400)
    observed = _observe(form=values, phase_offsets=_ZERO)
    assert observed.form_gradient[0] == 6 * values[0]
    assert observed.form_gradient[0] > 0


@pytest.mark.parametrize(
    "keyword",
    ("source_report", "host_acquisition", "donor_recovery", "observation_protocol"),
)
def test_assessor_accepts_primitives_not_incoming_report_verdicts(monkeypatch, keyword):
    monkeypatch.setattr(owner, "_assess_organized_outer_winding", _forbidden_rebuild)
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", _forbidden_rebuild)
    with pytest.raises(TypeError):
        _assess(**{keyword: {"admitted": True, "source_certified": True}})


def test_same_resultant_and_pair_means_do_not_close_its_future(mp):
    # Conditional carrier witnesses, not claimed to be reached source states.
    a, b = Q(1, 10), Q(1, 100)
    offsets = tuple(v for v in (a, -a, Q(0), Q(0), Q(0), Q(0)) for _ in range(9))
    shifted = tuple(v for v in (b, -b, Q(0), Q(0), Q(0), Q(0)) for _ in range(9))
    phase = _phase(mp, offsets)
    gamma = 1 / (1023 * mp.pi)
    phase_rates, form_rates = [], []
    for forms in (_ZERO, shifted):
        observed = _observe(form=forms, phase_offsets=offsets)
        _, _, dx, dtheta, _ = _fields(mp, tuple(_mp(mp, v) for v in forms), phase)
        for c in range(6):
            _contains(mp, observed.child_form_tau_rate_bounds[c], dx[9 * c])
            _contains(mp, observed.child_phase_tau_rate_bounds[c], dtheta[9 * c])
        assert sum(observed.child_form_means[:5]) == observed.child_form_means[5] == 0
        phase_rates.append(dtheta)
        form_rates.append(dx)
    resultant = sum(mp.exp(1j * (phase[45] - phase[9 * c])) for c in range(5))
    assert abs(resultant - 1j * (3 + 2 * mp.cos(_mp(mp, a)))) < mp.mpf("1e-160")
    derivatives = tuple(
        sum(
            1j * mp.exp(1j * (phase[45] - phase[9 * c])) * (rates[45] - rates[9 * c])
            for c in range(5)
        )
        for rates in phase_rates
    )
    assert abs(derivatives[0]) < mp.mpf("1e-160")
    assert abs(derivatives[1] + 2j * gamma * _mp(mp, b) * mp.sin(_mp(mp, a))) < mp.mpf(
        "1e-160"
    )
    assert abs(form_rates[0][45] - form_rates[1][45]) < mp.mpf("1e-160")
    acceleration_difference = -gamma * (derivatives[1].imag - derivatives[0].imag) / 4
    expected = gamma**2 * _mp(mp, b) * mp.sin(_mp(mp, a)) / 2
    assert abs(acceleration_difference - expected) < mp.mpf("1e-160")
    assert expected > 0


def test_source_handoff_is_carried_and_all_six_original_histories_retained(report):
    assert report.protocol_revision == "7edac2ab"
    assert report.relative_preparation_dimension == 96
    assert len(report.host_acquisition.child_sources) == 5
    sources = report.host_acquisition.child_sources + (report.donor_recovery,)
    assert len({id(source) for source in sources}) == 6
    assert all(source.recovery_certified for source in sources)
    assert all(
        source.form_error_bound == source.phase_error_bound == _ERROR
        for source in sources
    )
    assert report.form_error_bounds == report.phase_error_bounds == (_ERROR,) * 6
    assert (
        report.donor_formation_time == 2 * report.donor_recovery.formation_time == 200
    )
    assert (
        report.donor_recovery_duration
        == 2 * report.donor_recovery.relaxation_duration
        == 2 * 10**13
    )
    assert report.observation_protocol.total_elapsed_scaled_time == Q(1, 5000)
    assert report.elapsed_scaled_time == Q(1, 1000)
    assert report.elapsed_original_time == Q(1024, 1023 * 1000)
    assert report.carried_phase_drift_upper_bound > 0
    assert all(margin > 0 for _, margin in report.source_inclusion_margins)
    with pytest.raises(FrozenInstanceError):
        report.admitted = False


def test_finite_remainder_accounts_for_donor_diffusion_and_phase_motion(report, mp):
    g, h = report.geometry.gamma_bounds.hi, Q(1, 1000)
    observation_time = Q(1, 5000)
    b_obs = Q(1, 100) + g * observation_time
    b = b_obs + Q(7, 4) * g * h
    assert report.observation_form_upper_bound == b_obs
    assert report.carried_phase_drift_upper_bound == 2 * g * b_obs * observation_time
    assert report.joined_form_upper_bound == b
    expected = Q(15, 16) * g * h**2 + Q(15, 4) * g**2 * b * h**2
    assert report.finite_response_remainder_upper_bound == expected
    # Independent integration of the two derivative-error terms: R is not
    # frozen and the full phase-current derivative is not discarded.
    gamma = 1 / (1023 * mp.pi)
    hh, bb = _mp(mp, h), _mp(mp, b)
    integrated = mp.quad(
        lambda t: mp.mpf(15) * gamma * t / 8 + mp.mpf(15) * gamma**2 * bb * t / 2,
        [0, hh],
    )
    assert 0 < integrated <= _mp(mp, expected)
    assert expected < Q(1, 10**9)


def test_finite_response_and_all_four_reading_errors_are_enclosed(report, mp):
    gamma, h = 1 / (1023 * mp.pi), mp.mpf(1) / 1000
    nominal_moments = (sum(mp.cos(4 * (c - 2) / mp.pi) for c in range(5)), mp.mpf(5))
    moment_boxes = (
        report.outer_one_boundary_moment_bounds,
        report.outer_zero_boundary_moment_bounds,
    )
    remainder = _mp(mp, report.finite_response_remainder_upper_bound)
    for moment, moment_box, response, observed in zip(
        nominal_moments,
        moment_boxes,
        report.finite_increment_bounds,
        report.observed_increment_bounds,
    ):
        _contains(mp, moment_box, moment)
        for side in (-1, 1):
            _contains(mp, response, -gamma * h * moment / 4 + side * remainder)
        assert response.lo - observed.lo >= 2 * Q(1, 10**10)
        assert observed.hi - response.hi >= 2 * Q(1, 10**10)
    assert report.contrast_bounds.lo - report.observed_contrast_bounds.lo >= 4 * Q(
        1, 10**10
    )
    assert report.observed_contrast_bounds.hi - report.contrast_bounds.hi >= 4 * Q(
        1, 10**10
    )
    assert (
        report.observed_increment_bounds[0].lo > report.observed_increment_bounds[1].hi
    )
    assert report.observed_contrast_bounds.lo > Q(1, 10**7)
    assert report.resolution_margin == report.observed_contrast_bounds.lo - Q(1, 10**7)
    assert (
        report.source_certified and report.identity_certified and report.work_certified
    )
    assert report.comparison_certified and report.admitted
    assert report.unresolved_conditions == ()


def test_attachment_and_joined_storage_keep_every_paid_history(report, mp):
    q, b = Q(1, 10**70), report.observation_form_upper_bound
    contact = 90 + Q(45, 2) * (b + q) ** 2
    assert report.contact_work_upper_bound == contact < 100
    internal = 9 * (1 - mp.cos(2 * mp.pi / 9))
    outer = 5 * (1 - mp.cos(2 * mp.pi / 5))
    excess = _mp(mp, report.host_acquisition.endpoint_excess_upper_bound)
    for bound, expected in zip(
        report.joined_initial_storage_upper_bounds,
        (
            6 * internal
            + 9 * outer
            + excess
            + 36
            + 4 * _mp(mp, q) ** 2
            + _mp(mp, contact),
            6 * internal + 36 + 44 * _mp(mp, q) ** 2 + _mp(mp, contact),
        ),
    ):
        assert expected <= _mp(mp, bound) < 200
    assert report.observation_protocol.total_absolute_event_work_upper_bound == 36


def test_all_fifteen_marked_identities_keep_a_strict_finite_margin(report, mp):
    margins = dict(report.identity_margins)
    assert len(margins) == 21  # Two faces for each C9, one acute bound per C5.
    drift = report.contact_edge_drift_upper_bound
    assert (
        drift
        == 7
        * report.geometry.gamma_bounds.hi
        * report.joined_form_upper_bound
        * Q(1, 1000)
    )
    observed_drift = report.observation_protocol.continuous_edge_drift_upper_bound
    q, zeta = Q(1, 10**70), Q(1, 10**6)
    for c in range(6):
        error = 2 * q if c == 5 else 2 * zeta + observed_drift
        positive = 2 * mp.pi / 9 - _mp(mp, error + drift)
        acute = mp.pi / 2 - 2 * mp.pi / 9 - _mp(mp, error + drift)
        assert 0 < _mp(mp, margins[f"internal_positive_gap[{c}]"]) <= positive
        assert 0 < _mp(mp, margins[f"internal_acute_gap[{c}]"]) <= acute
    outer_margin = (
        mp.pi / 2 - mp.mpf(7) / 5 - _mp(mp, 2 * zeta + observed_drift + drift)
    )
    for j in range(9):
        assert 0 < _mp(mp, margins[f"outer_acute_gap[{j}]"]) <= outer_margin


@pytest.mark.parametrize(
    "failure,condition",
    (
        ("host", "host_acquisition_not_certified"),
        ("control", "paired_host_control_not_certified"),
        ("donor", "external_recovery_not_certified"),
        ("missing_donor_bound", "external_recovery_not_certified"),
        ("large_donor_bound", "external_recovery_not_certified"),
        ("observation", "preceding_sequential_observation_not_certified"),
        ("source_class", "actual_hosts_outside_observation_class"),
    ),
)
def test_missing_source_premises_never_certify_an_actual_interaction(
    monkeypatch, report, failure, condition
):
    host, donor, observation = (
        report.host_acquisition,
        report.donor_recovery,
        report.observation_protocol,
    )
    if failure == "host":
        host = replace(host, admitted=False)
    elif failure == "control":
        host = replace(host, control=replace(host.control, capture_certified=False))
    elif failure == "donor":
        donor = replace(donor, recovery_certified=False)
    elif failure == "missing_donor_bound":
        donor = replace(donor, endpoint_form_norm_squared_upper_bound=None)
    elif failure == "large_donor_bound":
        donor = replace(donor, endpoint_phase_norm_squared_upper_bound=Q(1, 10**140))
    elif failure == "source_class":
        host = replace(host, full_form_error_upper_bound=Q(1))
    else:
        observation = replace(observation, admitted=False)
    monkeypatch.setattr(owner, "_assess_organized_outer_winding", lambda **kwargs: host)
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", lambda **kwargs: donor)
    monkeypatch.setattr(
        owner, "_assess_organized_sequential_observation_protocol", lambda: observation
    )
    unavailable = _assess()
    assert not unavailable.source_certified
    assert not unavailable.work_certified
    assert not unavailable.identity_certified
    assert not unavailable.comparison_certified
    assert not unavailable.admitted
    assert unavailable.status == "unavailable"
    assert condition in unavailable.unresolved_conditions
