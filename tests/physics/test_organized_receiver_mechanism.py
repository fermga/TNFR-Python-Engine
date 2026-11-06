"""Independent source and complete-law controls for the unprepared receiver."""

import importlib
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_receiver_mechanism as owner

_ERROR = Q(1, 10**10)
_RAMP = Q(2046, 9) * Q(355, 113) ** 2
_CONTACTS = ((0, 1), (1, 2), (2, 0))


@pytest.fixture(scope="module", autouse=True)
def no_response_or_other_preparation_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("receiver preparation consumed another source's verdict")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        for module_name, function in (
            ("_organized_joint_acquisition", "_assess_organized_joint_acquisition"),
            ("_sine_constitutive_recovery", "_assess_constitutive_recovery"),
            ("_sine_class_constitutive_acquisition", "_assess_constitutive_class_pair"),
            (
                "_acquired_organized_network_capture",
                "_assess_acquired_organized_triangle",
            ),
        ):
            module = importlib.import_module("tnfr.physics." + module_name)
            patch.setattr(module, function, forbidden)
            if hasattr(owner, function):
                patch.setattr(owner, function, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _assess(**changes):
    inputs = dict(form_error_bound=_ERROR, phase_error_bound=_ERROR)
    return owner._assess_organized_receiver_mechanism(**(inputs | changes))


@pytest.fixture(scope="module")
def report():
    return _assess()


def _edges():
    internal = tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in range(3) for j in range(9)
    )
    cross = tuple(
        (9 * a + j, 9 * b + (j + offset) % 9, Q(1, 2))
        for a, b in _CONTACTS
        for j in range(9)
        for offset in (0, 1)
    )
    return internal, cross


def _gradients(mp, edges, form, phase):
    gradient, current = [mp.mpf(0)] * 27, [mp.mpf(0)] * 27
    for a, b, weight in edges:
        weight = _mp(mp, weight)
        g = weight * (form[a] - form[b])
        s = weight * mp.sin(phase[b] - phase[a])
        gradient[a] += g
        gradient[b] -= g
        current[a] += s
        current[b] -= s
    return gradient, current


def test_new_source_has_one_unprepared_receiver_and_no_capture_verdict(report):
    internal, cross = _edges()
    assert report.capture_geometry.geometry.weighted_edges == internal + cross
    assert report.capture_geometry.geometry.weighted_degrees == (4,) * 27
    assert report.nominal_initial_forms == tuple(
        _RAMP * (node % 9 - 4) if node < 18 else Q(0) for node in range(27)
    )
    assert report.nominal_initial_phases == (0,) * 27
    assert report.nominal_initial_windings == (0, 0, 0)
    assert report.relative_preparation_dimension == 48
    assert report.layer_windings == (1, 1, 0)
    assert report.receiver_exclusion_start == 0
    assert report.layer_time == 200
    assert report.original_layer_time == Q(204800, 1023)
    assert report.held_capacities == (1,) * 27
    assert report.initial_zero_winding_certified
    assert report.early_receiver_zero_winding_certified
    assert report.receiver_delay_certified
    assert report.receiver_oriented_current_certified
    assert report.status == "certified_receiver_delay_and_current"
    assert not report.unresolved_conditions
    assert report.eventual_receiver_acquisition == "unresolved"
    assert report.eventual_donor_retention == "unresolved"
    assert report.joined_capture == "unresolved"


def test_complete_source_storage_and_initial_contact_response(report, mp):
    internal, cross = _edges()
    forms = report.nominal_initial_forms
    gradient = [Q(0)] * 27
    costs = []
    for edges in (internal, cross):
        energy = Q(0)
        for a, b, weight in edges:
            difference = forms[a] - forms[b]
            gradient[a] += weight * difference
            gradient[b] -= weight * difference
            energy += weight * difference**2 / 2
        costs.append(energy)
    assert costs == [72 * _RAMP**2, 78 * _RAMP**2]
    assert report.nominal_initial_internal_form_storage == costs[0]
    assert report.nominal_initial_contact_form_storage == costs[1]
    assert report.nominal_initial_form_storage == sum(costs) == 150 * _RAMP**2
    assert report.nominal_initial_form_gradient == tuple(gradient)
    assert report.initial_form_gradient_l1 == sum(map(abs, gradient)) == 117 * _RAMP
    correction = 117 * _RAMP * _ERROR + 216 * _ERROR**2
    assert report.initial_storage_correction_upper_bound == correction
    assert report.initial_storage_upper_bound == sum(costs) + correction
    assert 755_000_000 < report.initial_storage_upper_bound < 756_000_000
    expected_receiver = (Q(7, 2), 6, 4, 2, 0, -2, -4, -6, -Q(7, 2))
    assert tuple(value / _RAMP for value in gradient[18:]) == expected_receiver
    gamma = 1 / (1023 * mp.pi)
    for bounds, value in zip(report.nominal_initial_phase_rate_bounds, gradient):
        _contains(mp, bounds, gamma * _mp(mp, value) / 4)
    assert report.nominal_initial_phase_rate_bounds[18].lo > 0
    assert report.nominal_initial_phase_rate_bounds[26].hi < 0


def test_original_asymmetric_errors_retain_all_contact_costs(report, mp):
    error = _mp(mp, _ERROR)
    forms = [
        _mp(mp, value) + error * (node % 9 - 4) / (5 + node // 9)
        for node, value in enumerate(report.nominal_initial_forms)
    ]
    phases = [
        error * ((node % 9 - 4) ** 2 - mp.mpf(20) / 3) / (10 + node // 9)
        for node in range(27)
    ]
    energy = sum(
        _mp(mp, w)
        * ((forms[a] - forms[b]) ** 2 / 2 + 1 - mp.cos(phases[a] - phases[b]))
        for a, b, w in sum(_edges(), ())
    )
    assert _mp(mp, report.nominal_initial_form_storage) < energy
    assert energy < _mp(mp, report.initial_storage_upper_bound)
    for start in (0, 9, 18):
        errors = [
            forms[j] - _mp(mp, report.nominal_initial_forms[j])
            for j in range(start, start + 9)
        ]
        assert abs(sum(errors)) < mp.mpf("1e-90")
        assert abs(sum(phases[start : start + 9])) < mp.mpf("1e-90")


def test_outward_bounds_retain_full_source_and_nonlinear_history(report, mp):
    gamma, alpha = 1 / (1023 * mp.pi), 2 * mp.pi / 9
    epsilon, error, ramp = gamma**2, _mp(mp, _ERROR), _mp(mp, _RAMP)
    initial = gamma * (ramp * mp.sqrt(120) + error * mp.sqrt(27))
    scaled = mp.exp(-20) * initial + 10 * epsilon * mp.sqrt(27)
    proxy = mp.sqrt(120) * abs(gamma * ramp - alpha)
    phase = (
        proxy + scaled + 200 * epsilon * mp.sqrt(27) + mp.sqrt(27) * error * (1 + gamma)
    )
    _contains(mp, report.exponential_decay_bounds, mp.exp(-20))
    _contains(mp, report.slow_clock_factor_bounds, epsilon)
    _contains(mp, report.proxy_reference_distance_bounds, proxy)
    assert (
        scaled
        <= _mp(mp, report.layer_scaled_form_norm_upper_bound)
        < scaled + mp.mpf("1e-25")
    )
    assert (
        phase
        <= _mp(mp, report.layer_phase_error_norm_upper_bound)
        < phase + mp.mpf("1e-25")
    )
    early = 4 * gamma * ramp + 2 * error + 4 * gamma * error + 800 * epsilon
    assert (
        early
        <= _mp(mp, report.early_receiver_edge_upper_bound)
        < early + mp.mpf("1e-25")
    )
    assert _mp(mp, report.early_receiver_edge_upper_bound) < mp.pi
    offset = 2 * error * (1 + gamma) + mp.sqrt(2) * scaled
    exact_delay = (mp.pi - offset) / (2 * epsilon)
    actual = _mp(mp, report.receiver_exclusion_until)
    assert exact_delay - mp.mpf("1e-20") < actual <= exact_delay
    assert 16_000_000 < report.receiver_exclusion_until < 17_000_000
    assert report.receiver_exclusion_until == report.receiver_exclusion_time_lower_bound
    assert report.late_receiver_edge_offset_upper_bound > 0
    assert report.late_receiver_edge_growth_upper_bound > 0
    assert report.layer_donor_acute_margin > Q(4, 5)


def test_commuting_heat_generators_and_region_marginal_support_early_bound(report, mp):
    matrices = []
    for edges in _edges():
        matrix = np.zeros((27, 27))
        for a, b, w in edges:
            matrix[a, a] += float(w) / 4
            matrix[b, b] += float(w) / 4
            matrix[a, b] -= float(w) / 4
            matrix[b, a] -= float(w) / 4
        matrices.append(matrix)
    internal, cross = matrices
    assert np.array_equal(internal @ cross, cross @ internal)
    region_lifts = np.repeat(np.eye(3), 9, axis=0)
    region_generator = (3 * np.eye(3) - np.ones((3, 3))) / 4
    assert np.array_equal(cross @ region_lifts, region_lifts @ region_generator)
    assert np.array_equal(internal @ region_lifts, np.zeros((27, 3)))
    gap = mp.mpf("0.1")
    assert (2 - 2 * mp.cos(2 * mp.pi / 9)) / 4 > gap
    full_spectrum = np.linalg.eigvalsh(internal + cross)
    assert full_spectrum[1] > float(report.semigroup_gap_lower_bound)
    maximum = mp.mpf(15) / 17 * (mp.mpf(2) / 17) ** (mp.mpf(2) / 15)
    assert maximum < _mp(mp, report.heat_shape_upper_bound)
    assert 4 * 20**15 < 17**17
    assert mp.mpf(16) / 3 * maximum < _mp(
        mp, report.linear_receiver_heat_edge_upper_bound
    )
    # A ramp edge is constant except its seam; the zero-mean heat bound
    # controls either sign without assuming conditional offset independence.
    ramp = np.arange(9) - 4
    assert np.array_equal(np.roll(ramp, -1) - ramp, np.array([1] * 8 + [-8]))


def test_reference_currents_and_actual_layer_orientation(report, mp):
    alpha = 2 * mp.pi / 9
    phase = [alpha * (node % 9 - 4) if node < 18 else mp.mpf(0) for node in range(27)]
    zero = [mp.mpf(0)] * 27
    _, currents = _gradients(mp, sum(_edges(), ()), zero, phase)
    for bounds, current in zip(report.reference_current_bounds, currents):
        _contains(mp, bounds, current)
    for j in range(9):
        value = (1 + mp.cos(alpha)) * mp.sin(alpha * (j - 4))
        assert abs(currents[18 + j] - value) < mp.mpf("1e-90")
    assert abs(sum(currents[18:])) < mp.mpf("1e-90")
    assert abs(sum(currents[:9])) > 2  # Individual regional means are not fixed.
    perturbation = [mp.mpf("0.00001") * mp.sin(node + 1) for node in range(27)]
    perturbation = [value - sum(perturbation) / 27 for value in perturbation]
    assert mp.sqrt(sum(value**2 for value in perturbation)) < _mp(
        mp, report.layer_phase_error_norm_upper_bound
    )
    _, actual = _gradients(
        mp, sum(_edges(), ()), zero, [a + b for a, b in zip(phase, perturbation)]
    )
    for nominal, value in zip(currents, actual):
        assert abs(value - nominal) <= _mp(mp, report.layer_current_error_upper_bound)
    for j in range(9):
        if j != 4:
            sign = -1 if j < 4 else 1
            assert (
                sign * actual[18 + j]
                >= _mp(mp, report.layer_receiver_oriented_current_lower_bound)
                > 0
            )


def test_complete_mixed_identity_and_regional_work_keep_contact_storage(report, mp):
    gamma = 1 / (1023 * mp.pi)
    form = [mp.mpf(node % 9 - 4) / 7 + mp.mpf(node // 9 - 1) / 5 for node in range(27)]
    phase = [mp.sin(mp.mpf(node + 1) / 3) for node in range(27)]
    (a, u), (b, v) = [_gradients(mp, edges, form, phase) for edges in _edges()]
    q, s = [x + y for x, y in zip(a, b)], [x + y for x, y in zip(u, v)]
    dx = [(-x + gamma * y) / 4 for x, y in zip(q, s)]
    dt = [gamma * x / 4 for x in q]
    regional_rates = []
    for start in (0, 9, 18):
        indices = range(start, start + 9)
        measured = sum(a[j] * dx[j] - u[j] * dt[j] for j in indices)
        internal_loss = -sum(a[j] ** 2 for j in indices) / 4
        form_work = -sum(a[j] * b[j] for j in indices) / 4
        phase_work = gamma * sum(a[j] * v[j] - u[j] * b[j] for j in indices) / 4
        assert abs(measured - internal_loss - form_work - phase_work) < mp.mpf("1e-90")
        regional_rates.append(measured)
    contact_rate = sum(b[j] * dx[j] - v[j] * dt[j] for j in range(27))
    assert abs(contact_rate) > mp.mpf("0.1")
    assert abs(
        sum(regional_rates) + contact_rate + sum(value**2 for value in q) / 4
    ) < mp.mpf("1e-90")
    for j in range(27):
        assert abs(dt[j] + gamma * dx[j] - gamma**2 * s[j] / 4) < mp.mpf("1e-90")


def test_memory_comparison_resolvent_retains_the_initial_layer(report, mp):
    gamma, epsilon = 1 / (1023 * mp.pi), 1 / (1023 * mp.pi) ** 2
    error, ramp = _mp(mp, _ERROR), _mp(mp, _RAMP)
    initial = gamma * (ramp * mp.sqrt(120) + error * mp.sqrt(27))
    mismatch = mp.sqrt(120) * abs(gamma * ramp - 2 * mp.pi / 9) + mp.sqrt(
        27
    ) * error * (1 + gamma)
    lipschitz, forcing = mp.mpf(2), mp.sqrt(27)
    decay = _mp(mp, report.semigroup_gap_lower_bound) / epsilon

    def source(s):
        transient = mp.exp(-decay * s)
        return mismatch + initial * transient + forcing / decay * (1 - transient)

    def bound(s):
        transient, growth = mp.exp(-decay * s), mp.exp(lipschitz * s)
        return (
            mismatch * growth
            + initial * transient
            + (lipschitz * initial + forcing)
            / (decay + lipschitz)
            * (growth - transient)
        )

    assert abs(bound(0) - mismatch - initial) < mp.mpf("1e-90")
    assert initial > 7  # The form preparation cannot be dropped at slow time zero.
    for s in (1 / decay, 20 / decay, mp.mpf("0.25"), mp.mpf(1)):
        split = min(s, 100 / decay)
        integral = mp.quad(
            lambda v: lipschitz * mp.exp(lipschitz * (s - v)) * source(v),
            sorted(set((mp.mpf(0), split, s))),
        )
        assert abs(bound(s) - source(s) - integral) < mp.mpf("1e-85")
        residual = mp.diff(bound, s) - mp.diff(source, s) - lipschitz * bound(s)
        assert abs(residual) < mp.mpf("1e-85")


def test_slow_current_jacobian_bound_allows_signed_cosines(mp):
    phase = [mp.mpf((7 * node) % 11) / 2 for node in range(27)]
    direction = [mp.sin(node + 1) for node in range(27)]
    jacobian = [[mp.mpf(0)] * 27 for _ in range(27)]
    cosines = []
    for a, b, weight in sum(_edges(), ()):
        cosine = mp.cos(phase[b] - phase[a])
        cosines.append(cosine)
        value = _mp(mp, weight) * cosine / 4
        jacobian[a][a] -= value
        jacobian[b][b] -= value
        jacobian[a][b] += value
        jacobian[b][a] += value
    assert min(cosines) < 0 < max(cosines)
    array = np.array(jacobian, dtype=float)
    spectrum = np.linalg.eigvalsh(array)
    assert spectrum[0] < 0 < spectrum[-1]
    assert max(abs(spectrum)) < 2
    assert np.array_equal(array, array.T)
    assert max(abs(sum(row)) for row in jacobian) < mp.mpf("1e-90")
    step = mp.mpf("1e-30")
    zero = [mp.mpf(0)] * 27
    currents = [
        _gradients(
            mp,
            sum(_edges(), ()),
            zero,
            [p + sign * step * v for p, v in zip(phase, direction)],
        )[1]
        for sign in (-1, 1)
    ]
    for j, row in enumerate(jacobian):
        observed = (currents[1][j] - currents[0][j]) / (8 * step)
        expected = sum(value * v for value, v in zip(row, direction))
        assert abs(observed - expected) < mp.mpf("1e-55")


@pytest.mark.parametrize(
    "changes",
    (
        {"form_error_bound": True},
        {"form_error_bound": -Q(1, 10**80)},
        {"phase_error_bound": False},
        {"phase_error_bound": float("nan")},
        {"phase_error_bound": float("inf")},
        {"phase_error_bound": "0.0"},
    ),
)
def test_invalid_primitives_reject_before_geometry_or_transport(changes, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached geometry or transport")

    monkeypatch.setattr(owner, "_derive_organized_triangle_capture", forbidden)
    monkeypatch.setattr(owner, "_prepared_transit_radii", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**changes)


def test_failed_margins_give_no_delay_interval_or_formation_verdict():
    report = _assess(form_error_bound=1000, phase_error_bound=2)
    assert report.status == "unavailable"
    assert not report.initial_zero_winding_certified
    assert not report.early_receiver_zero_winding_certified
    assert not report.receiver_delay_certified
    assert not report.receiver_oriented_current_certified
    assert report.receiver_exclusion_until is None
    assert report.layer_windings is None
    assert report.eventual_receiver_acquisition == "unresolved"
    assert report.eventual_donor_retention == "unresolved"
    assert report.joined_capture == "unresolved"
    assert "receiver_delay_not_certified" in report.unresolved_conditions


def test_late_bound_cannot_fill_a_missing_initial_layer_certificate():
    report = _assess(form_error_bound=0, phase_error_bound=1)
    assert report.receiver_exclusion_time_lower_bound > report.layer_time
    assert not report.early_receiver_zero_winding_certified
    assert not report.receiver_delay_certified
    assert report.receiver_exclusion_until is None
    assert "early_receiver_zero_winding_not_certified" in report.unresolved_conditions


@pytest.mark.parametrize(
    "errors, dimension", (((0, 0), 0), ((0, _ERROR), 24), ((Q(1, 10**100), 0), 24))
)
def test_zero_and_subgrid_errors_retain_exact_preparation_dimension(errors, dimension):
    rx, rt = errors
    report = _assess(form_error_bound=rx, phase_error_bound=rt)
    assert report.form_error_bound == rx and report.phase_error_bound == rt
    assert report.relative_preparation_dimension == dimension
    assert report.receiver_delay_certified
    assert (
        report.initial_storage_correction_upper_bound > 0
        if rx or rt
        else report.initial_storage_correction_upper_bound == 0
    )


@pytest.mark.parametrize(
    "extra",
    (
        {"mixing": (Q(1, 2),) * 3},
        {"source": object()},
        {"capture_geometry": object()},
        {"horizon": 200},
    ),
)
def test_fixed_preparation_cannot_be_replaced_by_report_or_deadline(extra):
    with pytest.raises(TypeError):
        _assess(**extra)
