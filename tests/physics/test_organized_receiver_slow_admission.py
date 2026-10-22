"""Independent complete-law controls for the retained receiver phase inertia."""

import importlib
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_receiver_slow_admission as owner
from tnfr.physics._sine_preparation import _prepared_velocity_defect

_ERROR = Q(1, 10**10)
_RAMP = Q(2046, 9) * Q(355, 113) ** 2


@pytest.fixture(scope="module", autouse=True)
def no_response_or_other_preparation_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("slow admission executed a trajectory or another preparation")

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
    return owner._assess_organized_receiver_slow_admission(**(inputs | changes))


@pytest.fixture(scope="module")
def report():
    return _assess()


def _edges():
    return tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in range(3) for j in range(9)
    ) + tuple(
        (9 * a + j, 9 * b + (j + offset) % 9, Q(1, 2))
        for a, b in ((0, 1), (1, 2), (2, 0))
        for j in range(9)
        for offset in (0, 1)
    )


def _phase_geometry(mp, phases):
    laplacian = mp.zeros(27)
    current, jacobian = mp.zeros(27, 1), mp.zeros(27)
    potential = mp.mpf(0)
    for a, b, weight in _edges():
        weight = _mp(mp, weight)
        difference = phases[b] - phases[a]
        sine = weight * mp.sin(difference)
        cosine = weight * mp.cos(difference) / 4
        laplacian[a, a] += weight
        laplacian[b, b] += weight
        laplacian[a, b] -= weight
        laplacian[b, a] -= weight
        current[a] += sine / 4
        current[b] -= sine / 4
        jacobian[a, a] -= cosine
        jacobian[b, b] -= cosine
        jacobian[a, b] += cosine
        jacobian[b, a] += cosine
        potential += weight * (1 - mp.cos(difference))
    return laplacian / 4, current, jacobian, potential


@pytest.fixture(scope="module")
def complete_state(mp):
    gamma = 1 / (1023 * mp.pi)
    raw_form = mp.matrix(
        [mp.sin(node + 1) + mp.mpf(node // 9 - 1) / 3 for node in range(27)]
    )
    raw_phase = mp.matrix([mp.mpf((7 * node) % 11) / 2 for node in range(27)])
    form = gamma * (raw_form - mp.ones(27, 1) * sum(raw_form) / 27)
    phase = raw_phase - mp.ones(27, 1) * sum(raw_phase) / 27
    a, current, jacobian, potential = _phase_geometry(mp, phase)
    common = mp.ones(27) / 27
    inverse = (a + common) ** -1 - common
    return gamma, form, phase, a, inverse, current, jacobian, potential


def test_complete_phase_inertia_reconstructs_form_and_keeps_regional_means(
    complete_state, mp
):
    gamma, form, _, a, inverse, current, _, _ = complete_state
    epsilon = gamma**2
    form_rate = -a * form + gamma * current
    phase_rate = gamma * a * form
    velocity = phase_rate / epsilon
    acceleration = a * form_rate / (gamma * epsilon)
    assert mp.norm(gamma * inverse * velocity - form) < mp.mpf("1e-90")
    assert mp.norm(epsilon * inverse * acceleration + velocity - current) < mp.mpf(
        "1e-85"
    )
    assert abs(sum(form)) < mp.mpf("1e-95")
    assert max(abs(sum(form[9 * r : 9 * r + 9])) for r in range(3)) > gamma
    assert abs(sum(velocity)) < mp.mpf("1e-90")
    # The same phase state with zero form has a different actual phase rate.
    assert mp.norm(velocity) > 1
    assert mp.norm(velocity - current) > 1


def test_inertial_storage_is_quarter_of_complete_storage_and_has_exact_loss(
    complete_state, mp
):
    gamma, form, _, a, inverse, current, _, potential = complete_state
    epsilon = gamma**2
    form_rate = -a * form + gamma * current
    velocity = a * form / gamma
    acceleration = a * form_rate / (gamma * epsilon)
    complete_energy = 2 * (form.T * a * form)[0] + potential
    inertial_energy = epsilon * (velocity.T * inverse * velocity)[0] / 2 + potential / 4
    assert abs(inertial_energy - complete_energy / 4) < mp.mpf("1e-90")
    derivative = (
        epsilon * (velocity.T * inverse * acceleration)[0] - (current.T * velocity)[0]
    )
    assert abs(derivative + (velocity.T * velocity)[0]) < mp.mpf("1e-85")


def test_defect_row_uses_signed_jacobian_and_actual_form_rate(complete_state, mp):
    gamma, form, _, a, _, current, jacobian, _ = complete_state
    epsilon = gamma**2
    form_rate = -a * form + gamma * current
    phase_rate = gamma * a * form
    velocity = phase_rate / epsilon
    defect = velocity - current
    assert mp.norm(defect + form_rate / gamma) < mp.mpf("1e-90")
    derivative = (a * form_rate / gamma - jacobian * phase_rate) / epsilon
    predicted = -(a / epsilon + jacobian) * defect - jacobian * current
    assert mp.norm(derivative - predicted) < mp.mpf("1e-80")
    assert abs(sum(defect)) < mp.mpf("1e-90")
    assert any(jacobian[i, j] < 0 for i in range(27) for j in range(27) if i != j)
    assert any(jacobian[i, j] > 0 for i in range(27) for j in range(27) if i != j)
    assert max(sum(abs(jacobian[i, j]) for j in range(27)) for i in range(27)) <= 2
    contraction = (defect.T * (a / epsilon + jacobian) * defect)[0]
    assert contraction > (mp.mpf("0.1") / epsilon - 2) * (defect.T * defect)[0]


def test_initial_velocity_norm_keeps_both_ramps_and_all_contact_rows(mp):
    forms = tuple(_RAMP * (node % 9 - 4) if node < 18 else Q(0) for node in range(27))
    gradient = [Q(0)] * 27
    for a, b, weight in _edges():
        difference = weight * (forms[a] - forms[b])
        gradient[a] += difference
        gradient[b] -= difference
    assert sum(value**2 for value in gradient) == _RAMP**2 * Q(2277, 2)
    assert any(gradient[18:])
    gamma = 1 / (1023 * mp.pi)
    nominal = mp.sqrt(sum(_mp(mp, value) ** 2 for value in gradient)) / (4 * gamma)
    assert nominal > 60_000_000
    assert abs(
        nominal - _mp(mp, _RAMP) * mp.sqrt(mp.mpf(2277) / 2) / (4 * gamma)
    ) < mp.mpf("1e-85")


def test_fixed_source_admission_retains_mean_leaf_and_unresolved_capture(report):
    source = report.preparation
    assert source.form_error_bound == source.phase_error_bound == _ERROR
    assert source.relative_preparation_dimension == 48
    assert source.capture_geometry.geometry.weighted_edges == _edges()
    assert source.nominal_initial_form_storage == 150 * _RAMP**2
    assert report.phase_mean_leaf_dimension == 26
    assert report.structural_energy_ratio == Q(1, 4)
    assert report.release_time == 400
    assert report.original_release_time == Q(409600, 1023)
    assert report.release_windings == (1, 1, 0)
    assert report.donor_release_bridge_certified
    assert report.uniform_defect_certified
    assert report.receiver_phase_motion_certified
    assert report.status == "certified_uniform_slow_admission"
    assert not report.unresolved_conditions
    assert not report.first_order_exact_closure
    assert report.eventual_receiver_acquisition == "unresolved"
    assert report.eventual_donor_retention == "unresolved"
    assert report.joined_capture == "unresolved"


def test_all_future_defect_enclosure_matches_independent_full_source_bound(report, mp):
    gamma, ramp, error = 1 / (1023 * mp.pi), _mp(mp, _RAMP), _mp(mp, _ERROR)
    epsilon, norm = gamma**2, mp.sqrt(27)
    gradient = ramp * mp.sqrt(mp.mpf(2277) / 2)
    initial = gradient / (4 * gamma) + 2 * norm * error / gamma + 2 * norm * error
    damping = mp.mpf("0.1") - 2 * epsilon
    floor = 2 * epsilon * norm / damping
    uniform = initial * mp.exp(-400 * damping) + floor
    _contains(mp, report.nominal_form_gradient_norm_bounds, gradient)
    _contains(
        mp,
        report.defect_decay_bounds,
        mp.exp(-400 * _mp(mp, report.damping_lower_bound)),
    )
    for actual, expected in (
        (report.initial_defect_norm_upper_bound, initial),
        (report.asymptotic_defect_upper_bound, floor),
        (report.uniform_velocity_defect_upper_bound, uniform),
        (report.uniform_form_rate_norm_upper_bound, gamma * uniform),
    ):
        assert expected <= _mp(mp, actual) < expected + mp.mpf("1e-20")
    assert 0 < _mp(mp, report.damping_lower_bound) <= damping
    assert 0 < floor < uniform < mp.mpf("0.000011")
    assert (
        report.velocity_defect_margin
        == report.velocity_defect_budget - report.uniform_velocity_defect_upper_bound
    )
    for time in (400, 800, 10000):
        exact_comparison = initial * mp.exp(-damping * time) + floor * (
            1 - mp.exp(-damping * time)
        )
        assert exact_comparison < _mp(mp, report.uniform_velocity_defect_upper_bound)
    for bounds, gradient in zip(
        report.nominal_initial_slow_velocity_bounds,
        report.preparation.nominal_initial_form_gradient,
    ):
        _contains(mp, bounds, _mp(mp, gradient) / (4 * gamma))


def test_release_phase_bounds_keep_initial_mismatch_and_both_error_channels(report, mp):
    gamma, ramp, error = 1 / (1023 * mp.pi), _mp(mp, _RAMP), _mp(mp, _ERROR)
    epsilon, norm = gamma**2, mp.sqrt(27)
    initial = gamma * (ramp * mp.sqrt(120) + error * norm)
    scaled = initial * mp.exp(-40) + 10 * epsilon * norm
    phase = (
        mp.sqrt(120) * abs(gamma * ramp - 2 * mp.pi / 9)
        + norm * error * (1 + gamma)
        + scaled
        + 400 * epsilon * norm
    )
    _contains(mp, report.release_form_decay_bounds, mp.exp(-40))
    for actual, expected in (
        (report.release_scaled_form_norm_upper_bound, scaled),
        (report.release_form_norm_upper_bound, scaled / gamma),
        (report.release_phase_error_norm_upper_bound, phase),
    ):
        assert expected <= _mp(mp, actual) < expected + mp.mpf("1e-25")
    assert mp.mpf("0.0002") < phase < mp.mpf("0.00021")
    assert (
        report.release_phase_error_norm_upper_bound
        > report.preparation.layer_phase_error_norm_upper_bound
    )
    assert report.release_donor_acute_margin > Q(4, 5)
    assert report.release_receiver_zero_winding_margin > 3
    bridge = phase + initial * (mp.exp(-20) - mp.exp(-40))
    assert (
        bridge
        <= _mp(mp, report.bridge_phase_error_norm_upper_bound)
        < bridge + mp.mpf("1e-25")
    )
    assert (
        report.bridge_phase_error_norm_upper_bound
        > report.release_phase_error_norm_upper_bound
    )
    assert report.bridge_donor_acute_margin < report.release_donor_acute_margin
    assert report.bridge_donor_acute_margin > Q(4, 5)
    assert report.donor_release_bridge_certified


def test_actual_phase_rate_bounds_include_asymmetric_phase_and_velocity_errors(
    report, mp
):
    nominal_phase = mp.matrix(
        [
            2 * mp.pi * (node % 9 - 4) / 9 if node < 18 else mp.mpf(0)
            for node in range(27)
        ]
    )
    phase_error = mp.matrix([mp.sin(node + 1) for node in range(27)])
    phase_error -= mp.ones(27, 1) * sum(phase_error) / 27
    phase_error *= _mp(mp, report.release_phase_error_norm_upper_bound) / (
        2 * mp.norm(phase_error)
    )
    defect = mp.matrix([mp.cos(2 * node + 1) for node in range(27)])
    defect -= mp.ones(27, 1) * sum(defect) / 27
    defect *= _mp(mp, report.uniform_velocity_defect_upper_bound) / (
        2 * mp.norm(defect)
    )
    _, current, _, _ = _phase_geometry(mp, nominal_phase + phase_error)
    actual_rate = current + defect
    for bounds, actual in zip(report.release_phase_rate_bounds, actual_rate):
        _contains(mp, bounds, actual)
    for j in range(9):
        if j != 4:
            signed = (-1 if j < 4 else 1) * actual_rate[18 + j]
            assert (
                signed
                > _mp(mp, report.release_receiver_signed_rate_lower_bound)
                > mp.mpf("0.15")
            )
    nominal_floor = (1 + mp.cos(2 * mp.pi / 9)) * mp.sin(mp.pi / 9) / 4
    certified_floor = (
        nominal_floor
        - mp.sqrt(2) * _mp(mp, report.release_phase_error_norm_upper_bound)
        - _mp(mp, report.uniform_velocity_defect_upper_bound)
    )
    assert abs(
        _mp(mp, report.release_receiver_signed_rate_lower_bound) - certified_floor
    ) < mp.mpf("1e-25")
    assert (
        report.receiver_rate_margin
        == report.release_receiver_signed_rate_lower_bound - Q(3, 20)
    )


def test_terminal_storage_bound_needs_form_and_noncritical_linear_remainder(report, mp):
    target = mp.matrix([2 * mp.pi * (node % 9 - 4) / 9 for node in range(27)])
    direction = mp.matrix([mp.sin(node + 1) for node in range(27)])
    direction -= mp.ones(27, 1) * sum(direction) / 27
    reference = target + direction / 100
    a, current, _, reference_potential = _phase_geometry(mp, reference)
    _, target_current, _, target_potential = _phase_geometry(mp, target)
    assert mp.norm(target_current) < mp.mpf("1e-90")
    gradient = -4 * current
    assert mp.norm(gradient) > mp.mpf("0.01")
    phase_radius = mp.mpf("0.0001")
    phase = reference + phase_radius * gradient / mp.norm(gradient)
    _, _, _, actual_potential = _phase_geometry(mp, phase)
    form_radius = _mp(mp, report.release_form_norm_upper_bound)
    form = direction * form_radius / mp.norm(direction)
    excess = 2 * (form.T * a * form)[0] + actual_potential - target_potential
    upper = (
        reference_potential
        - target_potential
        + mp.norm(gradient) * phase_radius
        + 4 * phase_radius**2
        + 4 * form_radius**2
    )
    assert 0 < excess < upper < mp.mpf(196) / 625
    assert actual_potential - reference_potential > 4 * phase_radius**2
    # This is an independent sufficient-endpoint control, not a reached state.
    for i, j, _ in _edges():
        registered = (target[j] - target[i] + mp.pi) % (2 * mp.pi) - mp.pi
        reference_gap = (
            registered + (reference[j] - target[j]) - (reference[i] - target[i])
        )
        assert abs(reference_gap) + mp.sqrt(2) * phase_radius < 2


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
def test_invalid_primitives_reject_before_source_admission(changes, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached the source or slow transport")

    monkeypatch.setattr(owner, "_assess_organized_receiver_mechanism", forbidden)
    monkeypatch.setattr(owner, "_prepared_velocity_defect", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**changes)


def test_uniform_defect_does_not_supply_missing_release_identity_or_capture():
    report = _assess(phase_error_bound=2)
    assert report.uniform_defect_certified
    assert not report.receiver_phase_motion_certified
    assert report.release_windings is None
    assert not report.donor_release_bridge_certified
    assert report.status == "unavailable"
    assert (
        "release_donor_one_receiver_zero_not_certified" in report.unresolved_conditions
    )
    assert "receiver_phase_motion_floor_not_certified" in report.unresolved_conditions
    assert report.eventual_receiver_acquisition == "unresolved"
    assert report.eventual_donor_retention == "unresolved"
    assert report.joined_capture == "unresolved"


def test_large_form_error_exhausts_release_defect_budget():
    report = _assess(form_error_bound=10**10)
    assert report.velocity_defect_margin < 0
    assert not report.uniform_defect_certified
    assert report.status == "unavailable"
    assert (
        "uniform_velocity_defect_budget_not_certified" in report.unresolved_conditions
    )
    assert report.joined_capture == "unresolved"


@pytest.mark.parametrize(
    "extra",
    ({"source": object()}, {"horizon": 400}, {"mixing": (Q(1, 2),) * 3}),
)
def test_fixed_release_cannot_be_replaced_by_cached_source_or_deadline(extra):
    with pytest.raises(TypeError):
        _assess(**extra)


@pytest.mark.parametrize("gap", (0, Q(1, 5), -1))
def test_shared_velocity_defect_rejects_nonpositive_effective_damping(gap):
    with pytest.raises(ValueError, match="gap greater than feedback stiffness"):
        _prepared_velocity_defect(
            time=1,
            gap_lower=gap,
            feedback_upper=Q(1, 10),
            lipschitz_upper=2,
            forcing_upper=1,
            initial_defect_upper=1,
        )


def test_shared_velocity_defect_retains_floor_even_with_zero_initial_defect(mp):
    damping, decay, floor, uniform = _prepared_velocity_defect(
        time=2,
        gap_lower=1,
        feedback_upper=Q(1, 10),
        lipschitz_upper=2,
        forcing_upper=3,
        initial_defect_upper=0,
    )
    assert damping == Q(4, 5)
    assert floor == uniform == Q(3, 4)
    _contains(mp, decay, mp.exp(-mp.mpf(8) / 5))
