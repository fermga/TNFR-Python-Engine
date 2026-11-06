"""Independent C9 heat, complete-row and work controls for a common probe."""

from fractions import Fraction as Q
from typing import get_type_hints

import mpmath
import pytest

from tnfr.physics import relational_sine_formed_classes as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

ERROR, DELTA, RADIUS = Q(1, 10**10), Q(1, 100), Q(1, 12)
SLOPE = Q(2046, 9) * Q(355, 113) ** 2
FIELDS = (
    "formation_time",
    "probe_time",
    "probe_duration",
    "phase_increment",
    "form_error_bound",
    "phase_error_bound",
    "readout_error_bound",
    "radius",
)


def _assess(**changes):
    inputs = dict(
        formation_time=Q(100),
        probe_time=Q(200),
        probe_duration=Q(1),
        phase_increment=DELTA,
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        readout_error_bound=ERROR,
        radius=RADIUS,
    )
    return owner.assess_sine_formed_class_response(**(inputs | changes))


@pytest.fixture(scope="module")
def frozen():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _upper_close(mp, bound, value):
    difference = _mp(mp, bound) - value
    assert 0 <= difference < mp.mpf("1e-25") * max(1, abs(value))


def _laplace(values):
    return [2 * values[i] - values[(i - 1) % 9] - values[(i + 1) % 9] for i in range(9)]


def _forcing(mp, phases):
    return [
        sum(mp.sin(phases[j] - phases[i]) for j in ((i - 1) % 9, (i + 1) % 9)) / 2
        for i in range(9)
    ]


def _potential(mp, phases):
    return sum(1 - mp.cos(phases[(i + 1) % 9] - phases[i]) for i in range(9))


def _integrated_heat_entry(mp, row, column, time):
    # The full circulant spectrum, including the constant mode, independently
    # integrates the comparison's fixed forcing. No trajectory solver is used.
    value = time
    for mode in range(1, 9):
        angle = 2 * mp.pi * mode / 9
        eigenvalue = 1 - mp.cos(angle)
        value += (
            mp.cos(angle * (row - column))
            * (-mp.expm1(-eigenvalue * time))
            / eigenvalue
        )
    return value / 9


def test_complete_fine_rows_and_reflection_give_the_signed_receiver(frozen, mp):
    delta, h = _mp(mp, DELTA), mp.mpf(1)
    gamma = 1 / (1023 * mp.pi)
    q = [mp.mpf(8) / 9] + [-mp.mpf(1) / 9] * 8
    aq = [value / 2 for value in _laplace(q)]
    assert abs(sum(q)) < mp.mpf("1e-90")
    assert mp.almosteq(sum(value**2 for value in q), mp.mpf(8) / 9)
    assert mp.almosteq(sum(q[i] * aq[i] for i in range(9)), 1)
    heat = (
        sum(-mp.expm1(-(1 - mp.cos(2 * mp.pi * mode / 9)) * h) for mode in range(1, 9))
        / 9
    )
    _contains(mp, frozen.heat_response_factor_bounds, heat)
    ideal_values = []
    for index, k in enumerate((1, 2)):
        angle = 2 * mp.pi * k / 9
        phases = [(i - 4) * angle + delta * q[i] for i in range(9)]
        forcing = _forcing(mp, phases)
        odd_coefficient = mp.sin(angle) * (1 - mp.cos(delta)) / 2
        for i in range(9):
            expected = -mp.cos(angle) * mp.sin(delta) * aq[i]
            expected += odd_coefficient * ((i == 1) - (i == 8))
            assert abs(forcing[i] - expected) < mp.mpf("1e-90")
        assert abs(sum(forcing)) < mp.mpf("1e-90")
        assert abs(
            _integrated_heat_entry(mp, 0, 1, h) - _integrated_heat_entry(mp, 0, 8, h)
        ) < mp.mpf("1e-90")
        receiver = gamma * sum(
            _integrated_heat_entry(mp, 0, j, h) * forcing[j] for j in range(9)
        )
        expected = -gamma * mp.cos(angle) * mp.sin(delta) * heat
        assert abs(receiver - expected) < mp.mpf("1e-90")
        _contains(mp, frozen.ideal_readout_bounds_by_class[index], receiver)
        ideal_values.append(receiver)
    _contains(mp, frozen.recorded_contrast_bounds, ideal_values[1] - ideal_values[0])


@pytest.mark.parametrize("time", [Q(0), Q(1, 4), Q(1), Q(4)])
def test_full_spectrum_heat_factor_obeys_declared_bounds(time, mp):
    h = _mp(mp, time)
    eigenvalues = [1 - mp.cos(2 * mp.pi * j / 9) for j in range(9)]
    assert all(0 <= value <= 2 for value in eigenvalues)
    g = sum(-mp.expm1(-value * h) for value in eigenvalues) / 9
    assert (1 - mp.exp(-2 * h)) / 2 <= g + mp.mpf("1e-90")
    assert g <= min(h, mp.mpf(8) / 9) + mp.mpf("1e-90")


def test_warmup_probe_and_recovery_bounds_have_independent_analytic_oracles(frozen, mp):
    gamma = 1 / (1023 * mp.pi)
    eta, forcing, gap = gamma**2, mp.sqrt(18), mp.mpf(1) / 5
    time, h, delta, radius = mp.mpf(200), mp.mpf(1), _mp(mp, DELTA), _mp(mp, RADIUS)
    error, slope = _mp(mp, ERROR), _mp(mp, SLOPE)
    decay, ev, et = mp.exp(-gap * time), gamma * forcing * error, forcing * error
    _contains(mp, frozen.warmup_decay_bounds, decay)
    all_errors = []
    for i, k in enumerate((1, 2)):
        v = gamma * k * slope * mp.sqrt(120)
        distance = mp.sqrt(60) * k * abs(gamma * slope - 2 * mp.pi / 9)
        # Expand the decaying transient and its convolution, retaining the
        # initial residuals and nonlinear feedback history independently.
        transient = decay * (v + ev + eta * forcing / gap)
        convolution = (
            v * time * decay + (ev + et) / gap + eta * forcing * (time + 1 / gap) / gap
        )
        remainder = transient + 2 * eta * convolution
        phase_error = decay * v + ev + et + eta * forcing * (time + 1 / gap)
        x = (eta * 2 * mp.sqrt(2) * distance / gap + remainder) / (gamma * mp.sqrt(2))
        theta = distance + phase_error / mp.sqrt(2)
        response_error = (
            x + 2 * gamma * h * theta + 2 * gamma**2 * h**2 * x + 2 * gamma**3 * h**3
        )
        theta_plus = theta + delta * mp.sqrt(mp.mpf(8) / 9)
        norm_squared = x**2 + theta_plus**2
        energy = (
            2 * x**2
            + 2
            * mp.cos(max(0, 2 * k * mp.pi / 9 - mp.sqrt(2) * theta_plus))
            * theta_plus**2
        )
        for name, value in (
            ("warmup_scaled_form_remainder_upper_bounds", remainder),
            ("warmup_phase_error_upper_bounds", phase_error),
            ("warmup_form_norm_upper_bounds", x),
            ("warmup_target_phase_radius_upper_bounds", theta),
            ("response_error_upper_bounds", response_error),
            ("post_probe_phase_radius_upper_bounds", theta_plus),
            ("post_probe_relative_norm_squared_upper_bounds", norm_squared),
            ("post_probe_excess_storage_upper_bounds", energy),
        ):
            _upper_close(mp, getattr(frozen, name)[i], value)
        angle = 2 * k * mp.pi / 9 + mp.sqrt(2) * radius
        _contains(mp, frozen.acute_radius_margin_bounds_by_class[i], mp.pi / 2 - angle)
        assert (
            0
            < _mp(mp, frozen.barrier_lower_bounds_by_class[i])
            <= radius**2 * mp.cos(angle) / 5
        )
        all_errors.append(response_error)
    coefficient = (
        gamma * (mp.cos(2 * mp.pi / 9) - mp.cos(4 * mp.pi / 9)) * mp.sin(delta)
    )
    exact_lower = coefficient * (1 - mp.exp(-2 * h)) / 2 - sum(all_errors) - 2 * error
    exact_upper = coefficient * min(h, mp.mpf(8) / 9) + sum(all_errors) + 2 * error
    assert _mp(mp, frozen.recorded_contrast_bounds.lo) <= exact_lower
    assert exact_upper <= _mp(mp, frozen.recorded_contrast_bounds.hi)
    assert exact_lower - _mp(mp, frozen.recorded_contrast_bounds.lo) < mp.mpf("1e-25")


def test_supplied_probe_work_and_capture_use_actual_state_errors(frozen, mp):
    delta, radius = _mp(mp, DELTA), _mp(mp, RADIUS)
    q = [mp.mpf(8) / 9] + [-mp.mpf(1) / 9] * 8
    direction = [mp.mpf(i - 4) / mp.sqrt(60) for i in range(9)]
    for i, k in enumerate((1, 2)):
        angle = 2 * k * mp.pi / 9
        target = [(j - 4) * angle for j in range(9)]
        theta_bound = _mp(mp, frozen.warmup_target_phase_radius_upper_bounds[i])
        x_bound = _mp(mp, frozen.warmup_form_norm_upper_bounds[i])
        nominal_work = 2 * mp.cos(angle) * (1 - mp.cos(delta))
        _contains(mp, frozen.nominal_probe_work_bounds_by_class[i], nominal_work)
        for sign in (-1, 1):
            phases = [
                value + sign * theta_bound * direction[j]
                for j, value in enumerate(target)
            ]
            post = [value + delta * q[j] for j, value in enumerate(phases)]
            work = _potential(mp, post) - _potential(mp, phases)
            _contains(mp, frozen.probe_work_bounds_by_class[i], work)
            assert abs(work - nominal_work) <= _mp(
                mp, frozen.probe_work_error_upper_bounds[i]
            )
            form = [x_bound * value for value in direction]
            form_storage = sum((form[(j + 1) % 9] - form[j]) ** 2 for j in range(9)) / 2
            excess = form_storage + _potential(mp, post) - _potential(mp, target)
            assert excess <= _mp(mp, frozen.post_probe_excess_storage_upper_bounds[i])
            assert (
                sum(form[j] ** 2 + (post[j] - target[j]) ** 2 for j in range(9))
                < radius**2
            )
    assert all(bounds.lo > 0 for bounds in frozen.probe_work_bounds_by_class)


def test_frozen_response_retains_correlation_and_separate_recovery(frozen):
    assert frozen.formation_certificate.status == "certified_two_formed_classes"
    assert frozen.status == "certified_formed_class_response"
    assert frozen.unavailable_reasons == ()
    assert frozen.recovery_certified_by_class == (True, True)
    assert frozen.response_certified
    assert frozen.recorded_contrast_bounds.lo > Q(1, 10**7)
    marginal = (
        frozen.recorded_readout_bounds_by_class[1]
        - frozen.recorded_readout_bounds_by_class[0]
    )
    assert 0 < marginal.lo < Q(1, 10**7) < frozen.recorded_contrast_bounds.lo
    assert frozen.probe_original_time == Q(204800, 1023)
    assert frozen.readout_original_time == Q(205824, 1023)
    assert frozen.probe_vector == (Q(8, 9),) + (Q(-1, 9),) * 8
    assert sum(frozen.probe_vector) == 0
    assert frozen.receiver_node == 0


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"formation_time": 0}, "formation_prerequisite_unavailable"),
        ({"probe_time": 100}, "recorded_contrast_not_strictly_positive"),
        ({"probe_duration": 0}, "recorded_contrast_not_strictly_positive"),
        ({"phase_increment": 0}, "recorded_contrast_not_strictly_positive"),
        ({"readout_error_bound": Q(1, 100)}, "recorded_contrast_not_strictly_positive"),
        ({"phase_increment": 1}, "strict_post_probe_radius_not_certified"),
        ({"radius": 1}, "strict_acute_radius_not_certified"),
        ({"radius": Q(1, 2**200)}, "strict_post_probe_storage_barrier_not_certified"),
    ],
)
def test_insufficient_complete_obligations_abstain(changes, reason):
    result = _assess(**changes)
    assert result.status == "unavailable"
    assert any(reason in item for item in result.unavailable_reasons)
    for name, value in changes.items():
        assert getattr(result, name) == value
    if changes.get("phase_increment") == 0:
        assert all(
            bounds.lo == bounds.hi == 0 for bounds in result.probe_work_bounds_by_class
        )
    if changes.get("probe_duration") == 0:
        assert (
            result.heat_response_factor_bounds.lo
            == result.heat_response_factor_bounds.hi
            == 0
        )


def test_strict_outward_touch_is_not_a_response_certificate(frozen):
    # A dyadic translation of the already rounded lower endpoint moves it
    # exactly to zero without estimating or fitting a scientific parameter.
    result = _assess(readout_error_bound=ERROR + frozen.recorded_contrast_bounds.lo / 2)
    assert result.recorded_contrast_bounds.lo == 0
    assert not result.response_certified
    assert result.recovery_certified_by_class == (True, True)
    assert result.unavailable_reasons == ("recorded_contrast_not_strictly_positive",)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("bad", [True, "1", 1j, float("inf"), float("nan")])
def test_original_primitive_admission_precedes_any_certificate(monkeypatch, field, bad):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached a formation calculation")

    monkeypatch.setattr(owner, "assess_sine_formed_class_pair", forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"formation_time": -1},
        {"probe_time": 99},
        {"probe_time": 20481},
        {"probe_duration": -1},
        {"probe_duration": 2049},
        {"phase_increment": -1},
        {"form_error_bound": -1},
        {"phase_error_bound": -1},
        {"readout_error_bound": -1},
        {"radius": 0},
    ],
)
def test_domain_constraints_reject_before_source_work(monkeypatch, changes):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid domain reached a formation calculation")

    monkeypatch.setattr(owner, "assess_sine_formed_class_pair", forbidden)
    with pytest.raises(ValueError):
        _assess(**changes)


def test_tiny_exact_primitives_and_zero_budget_subfamilies_are_retained():
    tiny = Q(1, 2**300)
    result = _assess(
        phase_increment=tiny,
        form_error_bound=tiny,
        phase_error_bound=0,
        readout_error_bound=0,
    )
    assert result.phase_increment == result.form_error_bound == tiny
    assert result.phase_error_bound == result.readout_error_bound == 0
    assert result.formation_certificate.preparation_dimension == 8
    assert not result.response_certified


def test_every_call_rebuilds_formation_and_warmup_from_original_primitives(monkeypatch):
    domains, rows, pair_inputs = [], [], []
    original_domain, original_rows, original_pair = (
        owner._sine_domain,
        owner._sine_preparation_from_rows,
        owner.assess_sine_formed_class_pair,
    )

    def domain(*args, **kwargs):
        value = original_domain(*args, **kwargs)
        domains.append(value)
        return value

    def row(domain, **kwargs):
        rows.append(domain)
        return original_rows(domain, **kwargs)

    def pair(**kwargs):
        pair_inputs.append(kwargs)
        return original_pair(**kwargs)

    monkeypatch.setattr(owner, "_sine_domain", domain)
    monkeypatch.setattr(owner, "_sine_preparation_from_rows", row)
    monkeypatch.setattr(owner, "assess_sine_formed_class_pair", pair)
    first, second = _assess(form_error_bound=0), _assess(form_error_bound=ERROR)
    assert len(domains) == 4 and len({id(value) for value in domains}) == 4
    assert len(rows) == 8
    assert all(rows[2 * i] is rows[2 * i + 1] is domains[i] for i in range(4))
    assert [values["form_error_bound"] for values in pair_inputs] == [0, ERROR]
    assert (
        second.warmup_form_norm_upper_bounds[0] > first.warmup_form_norm_upper_bounds[0]
    )
    with pytest.raises(TypeError):
        owner.assess_sine_formed_class_response(
            formation_certificate=first.formation_certificate
        )


def test_direct_sdk_json_and_annotations_retain_exact_inputs(frozen, tmp_path):
    direct = frozen.to_dict()
    assert direct["schema"] == "tnfr.sine-formed-class-response.v1"
    assert relational_report_to_dict(frozen)["report"] == direct["report"]
    path = tmp_path / "formed-class-response.json"
    export_to_json(frozen, path)
    assert json_loads(path.read_text(encoding="utf8"))["report"] == direct["report"]
    for name in FIELDS:
        value = direct["report"][name]
        assert Q(value["numerator"], value["denominator"]) == getattr(frozen, name)
    assert (
        direct["report"]["formation_certificate"]["status"]
        == "certified_two_formed_classes"
    )
    get_type_hints(owner.SineFormedClassResponse)
    get_type_hints(owner.assess_sine_formed_class_response)
