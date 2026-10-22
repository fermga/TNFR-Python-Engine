"""Static geometry controls; no nonlinear flow or reserved response evaluation."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from math import isqrt

import numpy as np
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics import _localized_mobile_interaction as owner
from tnfr.physics.relational_sine_comparison import _sine_rates


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _model(**changes):
    return RelationalExchangeModel(
        **(
            dict(
                storage_scale=1,
                epi_weight=Q(1, 2**20),
                phase_weight=1 - Q(1, 2**20),
                phase_domain="regular",
            )
            | changes
        )
    )


def _bound(**changes):
    return owner._bound_localized_mobile_geometry(
        **(
            dict(
                reference_model=_model(),
                node_count=1024,
                fejer_order=16,
                amplitude=Q(1, 2**12),
                initial_state_error_bound=Q(1, 1000 * 2**12),
                localization_radius=48,
                proximity_distance=96,
                horizon=256,
            )
            | changes
        )
    )


def _simplify(value):
    return sp.simplify(sp.expand_complex(value))


def test_fejer_first_moment_follows_from_adjacent_coefficients_and_actual_dft():
    order = sp.Symbol("Q", integer=True, positive=True)
    m = sp.Symbol("m", integer=True)
    squares = 1 + 2 * sp.summation((1 - m / order) ** 2, (m, 1, order - 1))
    adjacent = 2 * sp.summation(
        (1 - m / order) * (1 - (m + 1) / order), (m, 0, order - 1)
    )
    assert sp.factor(squares) == (2 * order**2 + 1) / (3 * order)
    assert sp.factor(adjacent / squares - (2 * order**2 - 2) / (2 * order**2 + 1)) == 0

    size, carrier, center = 8, 2, 1
    root = (1 + sp.I) / sp.sqrt(2)
    coefficients = {
        carrier + m: (1 - Q(abs(m), 2)) * root ** (-(carrier + m) * center)
        for m in (-1, 0, 1)
    }
    values = [
        _simplify(
            sum(c * root ** (k * j) for k, c in coefficients.items()) / sp.sqrt(size)
        )
        for j in range(size)
    ]
    mass = [_simplify(value * sp.conjugate(value)) for value in values]
    moment = _simplify(sum(root**j * value for j, value in enumerate(mass)))
    spectral_moment = _simplify(
        sum(
            c * sp.conjugate(coefficients.get(k + 1, 0))
            for k, c in coefficients.items()
        )
    )
    assert _simplify(moment - spectral_moment) == 0
    assert _simplify(moment / sum(mass) - Q(2, 3) * root**center) == 0
    assert _simplify(moment / sum(mass) - Q(2, 3) * root ** (-center)) != 0
    translated = sum(root**j * mass[(j - 3) % size] for j in range(size))
    reflected = sum(root**j * mass[(-j) % size] for j in range(size))
    assert _simplify(translated - root**3 * moment) == 0
    assert _simplify(reflected - sp.conjugate(moment)) == 0
    assert _bound(fejer_order=2).reference_first_moment_ratio == Q(
        adjacent.subs(order, 2) / squares.subs(order, 2)
    )


def test_moment_disk_bound_is_quadratic_and_sharp_for_norm_information():
    h = sp.Symbol("h", nonnegative=True)
    # A one-vertex vector saturates |<y,Sy>-<v,Sv>| <= (2h+h^2)||v||^2.
    shift = sp.diag(1, sp.I, -1, -sp.I)
    reference = sp.Matrix([1, 0, 0, 0])
    perturbation = h * reference
    observed = reference + perturbation
    difference = (observed.conjugate().T * shift * observed)[0] - (
        reference.conjugate().T * shift * reference
    )[0]
    assert sp.expand(difference - h * (2 + h)) == 0
    assert sp.expand(difference - 2 * h) == h**2  # Linear-only error is insufficient.

    # Tangency of the disk centered at one, radius 3/5: the maximal angle
    # has sine 3/5. This checks the asin geometry, not an atan(q) shortcut.
    x, y, q = Q(16, 25), Q(12, 25), Q(3, 5)
    assert (x - 1) ** 2 + y**2 == q**2
    assert y**2 / (x**2 + y**2) == q**2
    assert y / x > q


def test_nonzero_band_state_can_have_zero_first_moment():
    root = (1 + sp.I) / sp.sqrt(2)
    carrier = [root ** (2 * j) for j in range(8)]
    intensities = [_simplify(value * sp.conjugate(value)) for value in carrier]
    assert sum(intensities) == 8
    assert _simplify(sum(root**j * mass for j, mass in enumerate(intensities))) == 0
    # Adding a common global center to a constant intensity does not restore
    # a center. The budget must explicitly require a nonzero moment margin.
    report = _bound(initial_state_error_bound=Q(1, 5 * 2**12))
    assert report.common_family_bootstrap_certified
    assert report.band_norm_lower_bound > 0
    assert report.moment_nonvanishing_margin < 0
    assert not report.moment_nonvanishing_certified
    assert report.center_position_error_upper_bound is None
    assert report.relative_lifetime_additive_error_upper_bound is None


def _proximity_mass(left, right, size, distance):
    return sum(
        a * b
        for i, a in left.items()
        for j, b in right.items()
        if min((i - j) % size, (j - i) % size) <= distance
    )


def test_pair_mass_bounds_follow_from_product_mass_and_cyclic_triangle_inequality():
    size, radius, distance, eta = 32, 2, 4, Q(1, 5)
    left = {radius: 1 - eta, 16: eta}
    coincident = {size - radius: 1 - eta, 15: eta}
    separated = {16 - radius: 1 - eta, radius: eta}
    assert (
        sum(left.values()) == sum(coincident.values()) == sum(separated.values()) == 1
    )
    near = _proximity_mass(left, coincident, size, distance)
    far = _proximity_mass(left, separated, size, distance)
    assert near >= (1 - eta) ** 2
    assert far <= 1 - (1 - eta) ** 2
    # D=2R+d is excluded: boundary window vertices may still count as near.
    boundary_center = 2 * radius + distance
    assert (
        _proximity_mass(
            {radius: Q(1)}, {boundary_center - radius: Q(1)}, size, distance
        )
        == 1
    )
    assert _proximity_mass(left, coincident, size, distance - 1) < (1 - eta) ** 2


def test_full_channel_rotation_preserves_band_geometry_but_not_complete_sine_rows():
    size = 8
    root = (1 + sp.I) / sp.sqrt(2)
    state = [
        sp.Rational(j - 3, 17) + sp.I * sp.Rational(j * j, 23) for j in range(size)
    ]
    for band in ((1, 2, 3), (5, 6, 7)):
        coefficients = {
            k: _simplify(sum(state[j] * root ** (-j * k) for j in range(size)) / size)
            for k in band
        }
        projected = [
            _simplify(sum(value * root ** (j * k) for k, value in coefficients.items()))
            for j in range(size)
        ]
        assert all(
            _simplify(
                value * sp.conjugate(value)
                - (sp.I * value) * sp.conjugate(sp.I * value)
            )
            == 0
            for value in projected
        )

    model = _model()
    form = (Q(1, 7),) + (Q(0),) * (size - 1)
    gradient = tuple(
        2 * form[j] - form[j - 1] - form[(j + 1) % size] for j in range(size)
    )
    original = _sine_rates(model, (2,) * size, gradient, (1,) * size, (Q(0),) * size)
    # Multiplication by i sends (x,theta) to (-theta,x). Evaluate the actual
    # rotated state; its phase row is zero because its form gradient is zero.
    rotated_currents = tuple(
        sin(I(form[j - 1] - form[j])) + sin(I(form[(j + 1) % size] - form[j]))
        for j in range(size)
    )
    rotated = _sine_rates(
        model, (2,) * size, (Q(0),) * size, (1,) * size, rotated_currents
    )
    assert original["form_rates"][0].contains(-Q(model.epi_weight) / 7)
    assert original["form_rates"][0].hi < 0
    assert rotated["phase_rates"][0].contains(0)
    assert rotated["phase_rates"][0].lo == rotated["phase_rates"][0].hi == 0
    assert not (rotated["phase_rates"][0] - original["form_rates"][0]).contains(0)


def test_frozen_family_geometry_uses_common_error_and_separate_availability():
    report = _bound()
    budget, pi = report.budget, pi_interval()
    a, eps, horizon = Q(1, 2**12), Q(1, 1000 * 2**12), 256
    linear = 2 * pi.hi * horizon / (2**20 - 1)
    linear += Q(horizon, 6) * (2 * pi.hi * Q(15, 1024)) ** 3
    expected_h = 2 * eps / a + sqrt(I(2)).hi * linear + Q(64, 3) * horizon * a**2
    assert report.common_relative_error_majorant == expected_h < Q(13, 250)
    assert (
        report.common_relative_error_majorant
        > budget.joint_identity_error_upper_bound / a
    )
    assert report.common_family_bootstrap_certified
    assert report.band_norm_lower_bound == a * (1 - expected_h)
    assert report.common_tail_fraction_upper_bound < Q(1, 4)
    assert (
        not budget.prepared_identity_certified
    )  # R=48 fails the old strict tail target.
    assert report.mass_tail_bound_certified and report.moment_nonvanishing_certified
    assert report.center_position_error_upper_bound < 19
    assert report.relative_branch_certified
    assert report.initial_endpoint_distance_lower_bound > 218
    assert report.relative_lifetime_additive_error_upper_bound < 38
    assert report.coincident_reference_mass_lower_bound > Q(9, 16)
    assert report.initial_endpoint_mass_upper_bound < Q(7, 16)
    assert report.unavailable_reasons == ()
    with pytest.raises(FrozenInstanceError):
        report.proximity_distance = Q(1)


def test_residence_error_uses_both_endpoints_and_does_not_imply_monotone_speed():
    width, delta = sp.symbols("W delta", positive=True)
    duration = width / 2 + 2 * delta
    nominal_left, nominal_right = width / 2 + 2 * delta, -width / 2 - 2 * delta
    assert sp.expand(nominal_left - nominal_right - 2 * duration) == 0
    assert sp.expand(nominal_left - 2 * delta) == width / 2
    assert sp.expand(nominal_right + 2 * delta) == -width / 2
    # An arbitrarily fast bounded error can reverse the instantaneous slope.
    time = sp.Symbol("t", real=True)
    oscillating_error = 2 * delta * sp.sin(2 * time / delta)
    assert sp.diff(-2 * time + oscillating_error, time).subs(time, 0) == 2


def test_original_common_ball_criterion_excludes_endpoints_but_admits_joint_midpoint():
    report = _bound(localization_radius=96)
    pi, radius, tail = pi_interval(), 96, Q(1, 25)
    # The core moment in a radius-R ball has magnitude at least
    # (1-tail)*cos(2*pi*R/N). Adding the tail rotates it by at most
    # asin(tail/core). These bounds use the original R=96, tail=1/25.
    assert cos(2 * pi * Q(radius, 1024)).lo > Q(4, 5)
    tail_to_core = tail / ((1 - tail) * Q(4, 5))
    assert tail_to_core == Q(5, 96) < sin(I(Q(1, 18))).lo
    centroid_extra = Q(1024, 36) / pi.lo
    assert centroid_extra < 10
    common_ball_distance_upper = 2 * (radius + centroid_extra)
    assert common_ball_distance_upper < 212 < 218
    assert report.initial_endpoint_distance_lower_bound > 218
    # The old joint/tangent template certifies midpoint entry. The larger
    # common-family budget does not certify entry of the separate control.
    assert report.budget.band_tail_fraction_upper_bound < tail
    assert report.budget.prepared_identity_certified
    assert report.common_tail_fraction_upper_bound > tail


def test_lift_branch_and_mass_windows_have_independent_strict_boundaries():
    branch_failure = _bound(horizon=512)
    assert branch_failure.moment_nonvanishing_certified
    assert branch_failure.relative_branch_margin < 0
    assert not branch_failure.relative_branch_certified
    assert branch_failure.relative_lifetime_additive_error_upper_bound is None
    assert branch_failure.initial_endpoint_mass_upper_bound is not None
    assert branch_failure.center_position_error_upper_bound is not None

    boundary = _bound(proximity_distance=160)  # T=2R+d.
    assert boundary.relative_branch_certified
    assert boundary.coincident_reference_mass_lower_bound is not None
    assert boundary.initial_endpoint_mass_upper_bound is None
    below = _bound(proximity_distance=Q(96) - Q(1, 10**100))
    assert below.coincident_reference_mass_lower_bound is None
    assert below.initial_endpoint_mass_upper_bound is not None


def test_coarse_tail_does_not_disable_a_separately_certified_center():
    report = _bound(localization_radius=2)
    assert report.common_tail_fraction_upper_bound > 1
    assert not report.mass_tail_bound_certified
    assert report.coincident_reference_mass_lower_bound is None
    assert report.initial_endpoint_mass_upper_bound is None
    assert report.moment_nonvanishing_certified and report.relative_branch_certified


def test_failed_family_bootstrap_cannot_publish_geometric_conclusions():
    report = _bound(amplitude=1)
    assert not report.common_family_bootstrap_certified
    for name in (
        "band_norm_lower_bound",
        "common_tail_fraction_upper_bound",
        "center_position_error_upper_bound",
        "relative_lifetime_additive_error_upper_bound",
        "initial_endpoint_distance_lower_bound",
        "coincident_reference_mass_lower_bound",
        "initial_endpoint_mass_upper_bound",
    ):
        assert getattr(report, name) is None
    assert not report.moment_nonvanishing_certified
    assert "joint_template_bootstrap_not_closed" in report.unavailable_reasons


def test_joint_bootstrap_equality_is_not_a_positive_margin():
    a, horizon, pi = Q(1, 2**12), 256, pi_interval()
    root_two = sqrt(I(2)).hi
    linear = 2 * pi.hi * horizon / (2**20 - 1)
    linear += Q(horizon, 6) * (2 * pi.hi * Q(15, 1024)) ** 3
    # The independently reconstructed majorant exactly meets the 2a ball.
    error = ((2 - root_two) * a - root_two * a * linear - Q(32, 3) * horizon * a**3) / 2
    assert error > 0
    report = _bound(initial_state_error_bound=error)
    assert report.budget.identity_bootstrap_margin == 0
    assert not report.common_family_bootstrap_certified
    assert report.band_norm_lower_bound is None
    assert report.center_position_error_upper_bound is None


def test_exact_tiny_preparation_and_numpy_fraction_components_are_preserved():
    tiny = Q(1, 10**500)
    distance = Q(np.int64(96), np.int64(1))
    report = _bound(
        amplitude=tiny,
        initial_state_error_bound=tiny / 1000,
        proximity_distance=distance,
    )
    assert report.budget.amplitude == tiny
    assert report.budget.initial_state_error_bound == tiny / 1000
    assert type(report.proximity_distance.numerator) is int
    assert type(report.proximity_distance.denominator) is int
    assert report.band_norm_lower_bound > 0
    assert report.relative_branch_certified
    assert report.coincident_reference_mass_lower_bound > Q(9, 16)


def test_nearly_tangent_moment_disk_uses_conservative_angle_when_grid_is_unresolved():
    a, horizon, pi = Q(1, 2**12), 256, pi_interval()
    kappa = Q(2 * 16**2 - 2, 2 * 16**2 + 1)
    square = 1 + kappa
    scale = 2**512
    h = Q(isqrt(square.numerator * scale**2 // square.denominator), scale) - 1
    assert 0 < kappa - h * (2 + h) < Q(1, 2**128)
    linear = 2 * pi.hi * horizon / (2**20 - 1)
    linear += Q(horizon, 6) * (2 * pi.hi * Q(15, 1024)) ** 3
    error = a * (h - sqrt(I(2)).hi * linear - Q(64, 3) * horizon * a**2) / 2
    report = _bound(initial_state_error_bound=error)
    assert report.common_relative_error_majorant == h
    assert report.moment_nonvanishing_certified
    assert report.center_angle_error_upper_bound == pi.hi / 2
    assert not report.relative_branch_certified
    assert report.relative_lifetime_additive_error_upper_bound is None


@pytest.mark.parametrize("distance", [0, 512])
def test_admitted_distance_endpoints_remain_distinct_from_sufficient_mass_bounds(
    distance,
):
    report = _bound(proximity_distance=distance)
    assert report.proximity_distance == distance
    assert report.relative_branch_certified
    if distance == 0:
        assert report.coincident_reference_mass_lower_bound is None
    else:
        assert report.initial_endpoint_mass_upper_bound is None


@pytest.mark.parametrize(
    "bad",
    [True, np.bool_(False), float("nan"), float("inf"), complex(1, 0), {}, [], {1}],
)
def test_proximity_scalar_admission_rejects_nonphysical_inputs(bad):
    with pytest.raises((TypeError, ValueError)):
        _bound(proximity_distance=bad)


@pytest.mark.parametrize("distance", [-Q(1, 10**500), Q(512) + Q(1, 10**500)])
def test_exact_distance_domain_is_checked_before_any_float_rounding(distance):
    with pytest.raises(ValueError, match="proximity_distance"):
        _bound(proximity_distance=distance)


def test_geometry_rebuilds_actual_law_and_accepts_no_cached_budget():
    report = _bound()
    with pytest.raises(TypeError):
        _bound(budget=report.budget)
    model = _model()
    object.__setattr__(model, "storage_scale", Q(2))
    with pytest.raises(ValueError):
        _bound(reference_model=model)
    with pytest.raises(TypeError):
        _bound(horizon=np.int64(256))
