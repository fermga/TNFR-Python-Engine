"""Independent static checks for the autonomous joint reservoir law."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_parent_reservoir_exchange as owner


_ERROR = Q(1, 10**10)
_EDGES = tuple(
    sorted(
        {
            tuple(sorted((18 * g + 9 * a + j, 18 * g + 9 * a + (j + 1) % 9)))
            for g in range(2)
            for a in range(2)
            for j in range(9)
        }
        | {(18 * g + j, 18 * g + 9 + j) for g in range(2) for j in range(9)}
        | {(9 * a + j, 18 + 9 * a + j) for a in range(2) for j in range(9)}
    )
)
_FORMS = tuple(
    Q(1, 7)
    + Q(2 * g - 1, 10000)
    + Q(2 * a - 1, 9000)
    + Q((g + a + 1) * (j - 4), 100000)
    for g in range(2)
    for a in range(2)
    for j in range(9)
)
_PHASES = tuple(
    Q(1, 9)
    + Q(2 * g - 1, 60)
    + Q(2 * a - 1, 20)
    + Q((g + a + 1) * (3 * j * j - 68), 300000)
    for g in range(2)
    for a in range(2)
    for j in range(9)
)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _laplacian(values):
    result = [0] * 36
    for left, right in _EDGES:
        value = values[left] - values[right]
        result[left] += value
        result[right] -= value
    return tuple(result)


def _current(mp, theta):
    result = [mp.mpf(0)] * 36
    for left, right in _EDGES:
        value = mp.sin(theta[right] - theta[left])
        result[left] += value
        result[right] -= value
    return tuple(result)


def _observe(**changes):
    return owner._observe_parent_reservoir_exchange(
        **(
            dict(
                form=_FORMS,
                phase_offsets=_PHASES,
                reservoir_state=Q(12, 5),
                port="connected",
            )
            | changes
        )
    )


@pytest.mark.parametrize("port", ("connected", "disconnected"))
@pytest.mark.parametrize("reservoir", (-Q(12, 5), Q(0), Q(12, 5)))
def test_complete_joint_rows_and_storage_balance_in_all_clocks(port, reservoir, mp):
    report = _observe(port=port, reservoir_state=reservoir)
    gamma, e = 1 / (1023 * mp.pi), mp.mpf(1023) / 1024
    decay, slow = gamma**2 / 2, 2 / gamma**2
    r = _mp(mp, reservoir)
    x = tuple(_mp(mp, value) for value in _FORMS)
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value) for i, value in enumerate(_PHASES)
    )
    q, current = _laplacian(x), _current(mp, theta)
    b = tuple(
        gamma * (2 * (i // 9 % 2) - 1) / 12 if port == "connected" else mp.mpf(0)
        for i in range(36)
    )
    forcing = tuple(value * r for value in b)
    dx = tuple(
        (-lap + gamma * sine) / 4 + added
        for lap, sine, added in zip(q, current, forcing)
    )
    dt = tuple(gamma * lap / 4 for lap in q)
    output = sum(value * lap for value, lap in zip(b, q))
    dr = -output - decay * r
    power = output * r
    network_loss = sum(value**2 for value in q) / 4
    reservoir_loss = decay * r**2
    joint_rate = -network_loss - reservoir_loss
    _contains(mp, report.reservoir_decay_bounds, decay)
    _contains(mp, report.port_output_tau_bounds, output)
    assert report.reservoir_storage == reservoir**2 / 2
    assert report.reservoir_capacity == 1
    assert (
        report.network_dissipation_tau
        == sum(value**2 for value in _laplacian(_FORMS)) / 4
    )
    for clock, factor in (("tau", 1), ("slow", slow), ("original", e)):
        for i in range(36):
            _contains(
                mp, getattr(report, f"form_{clock}_rate_bounds")[i], factor * dx[i]
            )
            _contains(
                mp, getattr(report, f"phase_{clock}_rate_bounds")[i], factor * dt[i]
            )
            _contains(
                mp,
                getattr(report, f"network_input_{clock}_bounds")[i],
                factor * forcing[i],
            )
        _contains(mp, getattr(report, f"reservoir_{clock}_rate_bounds"), factor * dr)
        _contains(
            mp, getattr(report, f"network_input_power_{clock}_bounds"), factor * power
        )
        _contains(
            mp,
            getattr(report, f"joint_storage_{clock}_rate_bounds"),
            factor * joint_rate,
        )
    _contains(mp, report.reservoir_pressure_tau_bounds, dr)
    _contains(mp, report.network_storage_tau_rate_bounds, -network_loss + power)
    _contains(mp, report.reservoir_storage_tau_rate_bounds, -power - reservoir_loss)
    _contains(mp, report.reservoir_dissipation_tau_bounds, reservoir_loss)
    _contains(mp, report.joint_dissipation_tau_bounds, network_loss + reservoir_loss)
    assert report.joint_storage_tau_rate_bounds.hi < 0
    assert report.balance_residual_bounds.lo <= 0 <= report.balance_residual_bounds.hi
    # Differentiate H_network+r²/2 independently at one static state.
    step = mp.mpf("1e-65")
    complex_x = tuple(value + mp.j * step * rate for value, rate in zip(x, dx))
    complex_theta = tuple(value + mp.j * step * rate for value, rate in zip(theta, dt))
    complex_r = r + mp.j * step * dr
    energy = (
        sum(
            (complex_x[j] - complex_x[i]) ** 2 / 2
            + 1
            - mp.cos(complex_theta[j] - complex_theta[i])
            for i, j in _EDGES
        )
        + complex_r**2 / 2
    )
    derivative = mp.im(energy) / step
    assert abs(derivative - joint_rate) < mp.mpf("1e-120")
    _contains(mp, report.joint_storage_rate_from_rows_bounds, derivative)
    if port == "disconnected":
        assert all(
            value.lo == value.hi == 0 for value in report.network_input_tau_bounds
        )
        assert (
            report.network_input_power_tau_bounds.lo
            == report.network_input_power_tau_bounds.hi
            == 0
        )
    elif reservoir:
        assert (power > 0) is (reservoir > 0)


def test_zero_reservoir_is_not_a_disconnected_port():
    connected = _observe(reservoir_state=Q(0))
    disconnected = _observe(reservoir_state=Q(0), port="disconnected")
    assert connected.reservoir_tau_rate_bounds.hi < 0
    assert (
        disconnected.reservoir_tau_rate_bounds.lo
        == disconnected.reservoir_tau_rate_bounds.hi
        == 0
    )
    assert all(
        value.lo == value.hi == 0 for value in connected.network_input_tau_bounds
    )
    assert connected.reservoir_storage == 0


def test_exact_port_observation_retains_factor_nine_and_unresolved_fine_state(mp):
    report = _observe()
    means = tuple(sum(_FORMS[9 * child : 9 * child + 9]) / 9 for child in range(4))
    square_lap = tuple(2 * means[i] - means[i ^ 1] - means[i ^ 2] for i in range(4))
    gamma = 1 / (1023 * mp.pi)
    coarse = sum(
        gamma * (2 * (child % 2) - 1) * _mp(mp, value) / 12
        for child, value in enumerate(square_lap)
    )
    _contains(mp, report.port_output_tau_bounds, 9 * coarse)
    assert abs(9 * coarse - coarse) > mp.mpf("1e-9")
    projected = tuple(value for value in means for _ in range(9))
    projected_report = _observe(form=projected)
    _contains(mp, projected_report.port_output_tau_bounds, 9 * coarse)
    assert projected != _FORMS
    assert sum(value for value in _laplacian(_FORMS)) == 0


def test_reference_slow_modes_follow_the_declared_reservoir_law(mp):
    u, v_form = Q(1, 10000), -Q(1, 9000)
    outer, inner, reservoir = Q(1, 30), Q(1, 10), Q(12, 5)
    forms = tuple(
        (2 * g - 1) * u / 2 + (2 * a - 1) * v_form / 2
        for g in range(2)
        for a in range(2)
        for _ in range(9)
    )
    phases = tuple(
        (2 * g - 1) * outer / 2 + (2 * a - 1) * inner / 2
        for g in range(2)
        for a in range(2)
        for _ in range(9)
    )
    report = _observe(form=forms, phase_offsets=phases, reservoir_state=reservoir)
    gamma = 1 / (1023 * mp.pi)
    epsilon = gamma**2
    v = _mp(mp, v_form) / gamma
    b_rate = report.phase_slow_rate_bounds[9] - report.phase_slow_rate_bounds[0]
    v_rate = (
        report.form_slow_rate_bounds[9] - report.form_slow_rate_bounds[0]
    ) / report.network.geometry.gamma_bounds
    _contains(mp, b_rate, v)
    _contains(
        mp, v_rate, (-v - mp.sin(_mp(mp, inner)) + _mp(mp, reservoir) / 3) / epsilon
    )
    _contains(mp, report.reservoir_slow_rate_bounds, -6 * v - _mp(mp, reservoir))
    d_rate = report.phase_slow_rate_bounds[18] - report.phase_slow_rate_bounds[0]
    _contains(mp, d_rate, _mp(mp, u) / gamma)


def test_joint_error_energy_cancels_the_responsive_port_without_equal_inputs(mp):
    gamma, decay = 1 / (1023 * mp.pi), 1 / (2 * (1023 * mp.pi) ** 2)
    xi = tuple(
        _mp(mp, Q(2 * (i // 9 % 2) - 1, 1000) + Q(i % 9 - 4, 10000)) for i in range(36)
    )
    zeta = tuple(
        _mp(mp, Q(3 * (i % 9) ** 2 - 68, 30000) + Q(2 * (i // 18) - 1, 1000))
        for i in range(36)
    )
    rho = mp.mpf(1) / 30
    theta0 = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value) for i, value in enumerate(_PHASES)
    )
    current0 = _current(mp, theta0)
    current1 = _current(mp, tuple(value + error for value, error in zip(theta0, zeta)))
    q = _laplacian(xi)
    b = tuple(gamma * (2 * (i // 9 % 2) - 1) / 12 for i in range(36))
    dx = tuple(
        -lap / 4 + gamma * (after - before) / 4 + coupling * rho
        for lap, after, before, coupling in zip(q, current1, current0, b)
    )
    dz = tuple(gamma * lap / 4 for lap in q)
    dr = -sum(coupling * lap for coupling, lap in zip(b, q)) - decay * rho
    energy = (
        sum(value * lap for value, lap in zip(xi, q)) / 2
        + rho**2 / 2
        + 4 * sum(value**2 for value in zeta)
    )
    rate = (
        sum(lap * value for lap, value in zip(q, dx))
        + rho * dr
        + 8 * sum(value * change for value, change in zip(zeta, dz))
    )
    cancelled = (
        -sum(value**2 for value in q) / 4
        - decay * rho**2
        + gamma
        / 4
        * sum(
            lap * (after - before) for lap, after, before in zip(q, current1, current0)
        )
        + 2 * gamma * sum(value * lap for value, lap in zip(zeta, q))
    )
    assert abs(rate - cancelled) < mp.mpf("1e-165")
    assert rate <= 4 * gamma**2 * energy
    assert (2 / gamma**2) * rate <= 8 * energy
    assert abs(sum(xi)) < mp.mpf("1e-170")
    assert any(coupling * rho != 0 for coupling in b)


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf")))
def test_all_network_primitives_are_admitted_before_static_observation(
    field, bad, monkeypatch
):
    def forbidden(**kwargs):
        pytest.fail("malformed network primitive reached the observer")

    monkeypatch.setattr(owner, "_observe_repeated_composition", forbidden)
    values = list(_FORMS)
    values[-1] = bad
    with pytest.raises((TypeError, ValueError)):
        _observe(**{field: values})


@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf")))
def test_reservoir_admission_precedes_static_observation(bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed reservoir primitive reached the observer")

    monkeypatch.setattr(owner, "_observe_repeated_composition", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _observe(reservoir_state=bad)


@pytest.mark.parametrize(
    "bad", (True, 0, None, "", "Connected", "connected ", "zero_reservoir")
)
def test_only_the_two_fixed_port_models_are_admitted(bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid model token reached the observer")

    monkeypatch.setattr(owner, "_observe_repeated_composition", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _observe(port=bad)


def _assess(**changes):
    return owner._assess_organized_parent_reservoir_exchange(
        **(
            dict(
                form_error_bounds=(_ERROR,) * 4,
                phase_error_bounds=(_ERROR,) * 4,
                reservoir_error_bound=_ERROR,
            )
            | changes
        )
    )


@pytest.fixture(scope="module")
def obstruction():
    return _assess()


def test_original_sources_and_independent_reservoir_interval_are_retained(obstruction):
    report, source = obstruction, obstruction.acquisition
    center, radius = Q(12, 5), _ERROR
    assert (
        report.reservoir_initial_center == center
        and report.reservoir_error_bound == radius
    )
    assert report.reservoir_initial_bounds.lo <= center - radius
    assert report.reservoir_initial_bounds.hi >= center + radius
    assert report.reservoir_initial_bounds.width >= 2 * radius
    assert report.reservoir_initial_storage_upper_bound == (center + radius) ** 2 / 2
    assert (
        report.network_input_work_upper_bound
        == report.reservoir_initial_storage_upper_bound
        < 3
    )
    assert report.storage_budget_margin == 3 - report.network_input_work_upper_bound > 0
    assert source.relative_preparation_dimension == 64
    for child in source.child_sources:
        assert child.form_error_bound == child.phase_error_bound == _ERROR
    r, z = source.full_form_error_upper_bound, source.full_phase_error_upper_bound
    assert report.initial_joint_error_energy_upper_bound == 4 * (r**2 + z**2)
    for prefix, power in (("readout", 2), ("whole", 32)):
        phase_bound = (r + z) * 3**power
        assert getattr(report, f"{prefix}_joint_phase_error_upper_bound") == phase_bound
        assert (
            getattr(report, f"{prefix}_joint_form_error_upper_bound") == 6 * phase_bound
        )
        assert (
            getattr(report, f"{prefix}_joint_reservoir_error_upper_bound")
            == 4 * phase_bound
        )
    with pytest.raises(FrozenInstanceError):
        report.obstruction_certified = False


@pytest.mark.parametrize("horizon", (Q(0), Q(1, 2), Q(8)))
def test_comparison_supersolutions_satisfy_the_finite_epsilon_rows(
    horizon, obstruction, mp
):
    epsilon = 1 / (1023 * mp.pi) ** 2
    h, d, c = _mp(mp, horizon), mp.mpf(1) / 64, mp.mpf(9) / 10
    r_upper = _mp(mp, Q(12, 5) + _ERROR)
    b = d + (1 - mp.exp(-4 * h)) / 4
    y = 1 + 4 * d
    p = 6 * d + mp.mpf(3) / 2 + (r_upper - 2) * mp.exp(-h) + mp.exp(-4 * h) / 2
    db, dp = mp.exp(-4 * h), -(r_upper - 2) * mp.exp(-h) - 2 * mp.exp(-4 * h)
    assert abs(db - (y - 4 * b)) < mp.mpf("1e-170")
    assert abs(dp - (6 * b - p)) < mp.mpf("1e-170")
    residual = (1 - 4 * epsilon) * y - (2 - 16 * epsilon) * b + mp.sin(b) - p / 3
    lower = (
        c * d
        + c / 4
        + (mp.mpf(1) / 3 - c / 4 - 4 * epsilon) * mp.exp(-4 * h)
        - (r_upper - 2) * mp.exp(-h) / 3
    )
    assert residual >= lower > 0
    b2 = mp.exp(-h / 5) / 3
    y2, p2 = (4 - mp.mpf(1) / 5) * b2, mp.mpf(5) / 2 * mp.exp(-h / 5)
    assert abs(-b2 / 5 - (y2 - 4 * b2)) < mp.mpf("1e-170")
    assert abs(-p2 / 5 - (6 * b2 - p2)) < mp.mpf("1e-170")
    residual2 = (
        -epsilon * y2 / 5
        + (1 - 4 * epsilon) * y2
        - (2 - 16 * epsilon - c) * b2
        - p2 / 3
    )
    assert abs(residual2 - (epsilon / 25 + mp.mpf(1) / 5) * b2) < mp.mpf("1e-170")
    assert residual2 > 0
    # These coefficients apply for every h>=0, not only these static points.
    assert 1 - (mp.mpf(17) / 64) ** 2 / 6 > c
    assert mp.mpf(1) / 3 - c / 4 - 4 * epsilon > 0
    assert c * d + c / 4 - (r_upper - 2) / 3 > 0
    assert 1 - 16 * epsilon > 0 and 3 - 2 * epsilon > 0
    assert r_upper + 6 * d < mp.mpf(5) / 2


def test_fixed_negative_readout_and_registered_deadline_are_distinct(obstruction, mp):
    report = obstruction
    _contains(
        mp,
        report.nominal_readout_envelope_bounds,
        mp.mpf(1) / 64 + (1 - mp.exp(-2)) / 4,
    )
    _contains(mp, report.nominal_deadline_envelope_bounds, mp.exp(-mp.mpf(8) / 5) / 3)
    assert (
        report.connected_parent_gap_at_readout_upper_bound
        == report.nominal_readout_envelope_bounds.hi
        + 2 * report.readout_joint_phase_error_upper_bound
    )
    assert (
        report.connected_parent_gap_at_deadline_upper_bound
        == report.nominal_deadline_envelope_bounds.hi
        + 2 * report.whole_joint_phase_error_upper_bound
    )
    assert report.connected_parent_gap_at_readout_upper_bound < Q(1, 4)
    assert report.connected_parent_gap_at_deadline_upper_bound < Q(1, 12)
    assert report.disconnected_parent_gap_upper_bound < Q(1, 4)
    assert report.source_certified and report.response_domain_certified
    assert report.finite_work_certified and report.child_identities_retained
    assert report.disconnected_registration_retained and report.registered_at_deadline
    assert (
        report.disruption_ruled_out
        and report.conditional_obstruction_certified
        and report.obstruction_certified
    )
    assert not report.disruption_certified and not report.protocol_satisfied
    assert report.status == "certified_reservoir_disruption_obstruction"
    assert not report.unresolved_conditions
    assert report.source_slow_time == 4 and report.readout_elapsed_slow_time == Q(1, 2)
    assert (
        report.readout_slow_time == Q(9, 2) and report.deadline_elapsed_slow_time == 8
    )
    assert report.deadline_slow_time == 12
    gamma = 1 / (1023 * mp.pi)
    _contains(mp, report.elapsed_scaled_readout_time_bounds, 1 / gamma**2)
    _contains(mp, report.elapsed_scaled_deadline_time_bounds, 16 / gamma**2)
    _contains(mp, report.elapsed_original_readout_time_bounds, 1024 / (1023 * gamma**2))
    _contains(
        mp, report.elapsed_original_deadline_time_bounds, 16 * 1024 / (1023 * gamma**2)
    )


@pytest.mark.parametrize(
    "field", ("form_error_bounds", "phase_error_bounds", "reservoir_error_bound")
)
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf"), -Q(1, 10**100)))
def test_all_nine_allowances_are_admitted_before_acquisition(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed allowance reached network acquisition")

    monkeypatch.setattr(owner, "_assess_organized_parent_acquisition", forbidden)
    if field == "reservoir_error_bound":
        values = bad
    else:
        values = [_ERROR] * 4
        values[-1] = bad
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: values})


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("bad", ((), (_ERROR,) * 3, (_ERROR,) * 5, "0000", {_ERROR}))
def test_network_source_shapes_reject_before_acquisition(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed allowance shape reached acquisition")

    monkeypatch.setattr(owner, "_assess_organized_parent_acquisition", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "dependency", ("sources_certified", "entry_certified", "child_identities_retained")
)
def test_source_unavailability_is_not_a_negative_response(
    dependency, obstruction, monkeypatch
):
    source = replace(obstruction.acquisition, **{dependency: False})
    monkeypatch.setattr(
        owner, "_assess_organized_parent_acquisition", lambda **kwargs: source
    )
    report = _assess()
    assert report.conditional_obstruction_certified
    assert (
        not report.source_certified
        and not report.disruption_ruled_out
        and not report.obstruction_certified
    )
    assert report.status == "unavailable" and report.unresolved_conditions
    assert not report.protocol_satisfied


def test_unrelated_old_forecasts_do_not_decide_the_joint_response(
    obstruction, monkeypatch
):
    source = replace(
        obstruction.acquisition,
        prediction_certified=False,
        parent_identity_retained=False,
        admitted=False,
        status="unavailable",
    )
    monkeypatch.setattr(
        owner, "_assess_organized_parent_acquisition", lambda **kwargs: source
    )
    report = _assess()
    assert report.source_certified and report.obstruction_certified


def test_wider_admitted_reservoir_interval_can_leave_response_domain_unavailable():
    report = _assess(reservoir_error_bound=Q(1, 100))
    assert report.source_certified and report.finite_work_certified
    assert not report.response_domain_certified and not report.obstruction_certified
    assert report.status == "unavailable"
    assert "connected_response_domain_not_certified" in report.unresolved_conditions
    assert not report.disruption_certified and not report.protocol_satisfied


def test_reservoir_storage_failure_is_separate_from_network_source_admission():
    report = _assess(reservoir_error_bound=Q(1, 10))
    assert report.network_source_certified
    assert not report.reservoir_preparation_certified and not report.source_certified
    assert report.storage_budget_margin < 0
    assert not report.obstruction_certified and report.status == "unavailable"


@pytest.mark.parametrize(
    "margin,allowed", ((-Q(1, 10**100), False), (Q(0), False), (Q(1, 10**100), True))
)
def test_negative_readout_requires_strict_registration_margin(
    margin, allowed, obstruction, monkeypatch
):
    # Arithmetic boundary only; no altered identity is a scientific protocol.
    source = replace(
        obstruction.acquisition,
        parent_identity_gap=obstruction.connected_parent_gap_at_readout_upper_bound
        + margin,
    )
    monkeypatch.setattr(
        owner, "_assess_organized_parent_acquisition", lambda **kwargs: source
    )
    report = _assess()
    assert report.registration_retention_margin == margin
    assert (
        report.disruption_ruled_out is allowed
        and report.obstruction_certified is allowed
    )
    if not allowed:
        assert report.status == "unavailable" and not report.protocol_satisfied


@pytest.mark.parametrize(
    "extra",
    (
        "source_report",
        "acquisition",
        "b",
        "coupling",
        "lambda",
        "duration",
        "readout",
        "reservoir_initial_center",
        "trajectory",
    ),
)
def test_frozen_law_preparation_and_readout_cannot_be_replaced(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})


def test_tiny_signed_reservoir_is_not_rounded_to_a_disconnection():
    tiny = -Q(1, 10**1000)
    report = _observe(reservoir_state=tiny)
    assert report.reservoir_state == tiny != 0
    assert report.reservoir_storage == tiny**2 / 2 > 0
    assert any(
        value.lo < 0 or value.hi > 0 for value in report.network_input_tau_bounds
    )
