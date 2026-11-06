"""Independent instantaneous-rate controls; no trajectory or reserved response."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import numpy as np
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics import _localized_mobile_interaction as owner
from tnfr.physics.relational_sine_comparison import _sine_rates


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _model():
    return RelationalExchangeModel(
        storage_scale=1,
        epi_weight=Q(1, 2**20),
        phase_weight=1 - Q(1, 2**20),
        phase_domain="regular",
    )


def _bound(**changes):
    return owner._bound_localized_mobile_relative_rate(
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


def test_moment_derivative_keeps_complex_weight_outside_the_real_bilinear_term():
    time = sp.Symbol("t", real=True)
    state = [1 + sp.I, 2 - sp.I, -1 + 2 * sp.I, 3]
    rate = [2 - sp.I, 1, 2 + sp.I, -2 * sp.I]
    weights = [1, sp.I, -1, -sp.I]
    moment = sum(w * f * sp.conjugate(f) for w, f in zip(weights, state))
    moving = [f + time * g for f, g in zip(state, rate)]
    differentiated = sp.diff(
        sum(w * f * sp.conjugate(f) for w, f in zip(weights, moving)), time
    ).subs(time, 0)
    bilinear = sum(
        w * (g * sp.conjugate(f) + f * sp.conjugate(g))
        for w, f, g in zip(weights, state, rate)
    )
    assert _simplify(differentiated - bilinear) == 0
    wrong = 2 * sp.re(
        sum(w * g * sp.conjugate(f) for w, f, g in zip(weights, state, rate))
    )
    assert _simplify(differentiated - wrong) != 0
    # Arg derivative, without an independently chosen phase or a factor two.
    assert _simplify(sp.im(differentiated / moment)) == _simplify(
        sp.im(differentiated * sp.conjugate(moment)) / (moment * sp.conjugate(moment))
    )


def test_real_two_row_tangent_retains_opposite_band_conjugation_and_damping():
    zr, zi, wr, wi = sp.symbols("zr zi wr wi", real=True)
    eigenvalue, sigma = sp.symbols("lambda sigma", positive=True)
    z, opposite = zr + sp.I * zi, wr + sp.I * wi
    form = (z + sp.conjugate(opposite)) / 2
    phase = (z - sp.conjugate(opposite)) / (2 * sp.I)
    from_rows = (
        -sigma * eigenvalue * form - eigenvalue * phase + sp.I * eigenvalue * form
    )
    projected = (sp.I - sigma / 2) * eigenvalue * z - sigma * eigenvalue * sp.conjugate(
        opposite
    ) / 2
    assert sp.expand(from_rows - projected) == 0
    discarded_opposite = (sp.I - sigma / 2) * eigenvalue * z
    assert sp.expand(from_rows - discarded_opposite) != 0
    assert (
        sp.expand(from_rows.subs({zr: 0, zi: 0, wr: 1, wi: 0}))
        == -sigma * eigenvalue / 2
    )


def test_adjacent_frequency_difference_and_polynomial_velocity_error():
    xi, offset = sp.symbols("xi offset", real=True)
    left = sp.pi / 2 + 2 * offset * xi
    difference = sp.cos(left) - sp.cos(left + 2 * xi)
    assert (
        sp.trigsimp(
            sp.expand_trig(difference - 2 * sp.sin(xi) * sp.cos((2 * offset + 1) * xi))
        )
        == 0
    )
    # 1-sinc(xi)*cos(alpha)=(1-sinc)+sinc*(1-cos), with sinc in [0,1].
    sinc, cosine = sp.symbols("sinc cosine", real=True)
    assert sp.expand((1 - sinc) + sinc * (1 - cosine) - (1 - sinc * cosine)) == 0
    pi, size, order = pi_interval(), 1024, 16
    delta = pi / size
    ratio = I(sin(delta).lo / delta.hi, sin(delta).hi / delta.lo)
    minimum_velocity = ratio * cos((2 * order - 3) * delta)
    polynomial = (Q(1, 6) + Q((2 * order - 3) ** 2, 2)) * (pi.hi / size) ** 2
    assert 0 < 1 - minimum_velocity.lo < polynomial
    assert polynomial < Q(1, 250)


def _c12_dft(values):
    root = (sp.sqrt(3) + sp.I) / 2
    return {
        k: _simplify(
            sum(value * root ** (-k * j) for j, value in enumerate(values)) / 12
        )
        for k in (2, 3, 4, 8, 9, 10)
    }


def _c12_moment_and_rate(coefficients, rates, band):
    moment = rate = 0
    for k in band:
        next_k = (k + 1) % 12
        z, next_z = coefficients[k], coefficients.get(next_k, 0)
        g, next_g = rates[k], rates.get(next_k, 0)
        moment += 12 * z * sp.conjugate(next_z)
        rate += 12 * (g * sp.conjugate(next_z) + z * sp.conjugate(next_g))
    return _simplify(moment), _simplify(rate)


def test_c12_same_intensities_have_distinct_actual_relative_rates_at_arbitrarily_small_scale():
    q, sigma = sp.symbols("q sigma", positive=True)
    profile = [q * (sp.cos(sp.pi * j / 3) + sp.cos(sp.pi * j / 2)) for j in range(12)]
    profile = list(map(sp.expand, profile))
    half_laplacian = [
        profile[j] - (profile[j - 1] + profile[(j + 1) % 12]) / 2 for j in range(12)
    ]
    phase_current = [
        (
            sp.sin(profile[j - 1] - profile[j])
            + sp.sin(profile[(j + 1) % 12] - profile[j])
        )
        / 2
        for j in range(12)
    ]
    form_coefficients = _c12_dft(profile)
    phase_coefficients = {k: sp.I * value for k, value in form_coefficients.items()}
    form_rates = _c12_dft([(-sigma + sp.I) * value for value in half_laplacian])
    phase_rates = _c12_dft(phase_current)
    tangent_phase_rates = _c12_dft([-value for value in half_laplacian])
    assert sum(profile) == 0
    assert sp.expand(sum(value**2 for value in profile)) == 12 * q**2
    assert phase_rates[4] == phase_rates[8] == 0
    assert (
        sp.trigsimp(
            phase_rates[2] - (sp.sin(q / 2) - sp.sin(2 * q) - sp.sin(3 * q / 2)) / 12
        )
        == 0
    )
    assert (
        sp.trigsimp(
            phase_rates[3]
            + (2 * sp.sin(q / 2) + sp.sin(2 * q) + 2 * sp.sin(3 * q / 2)) / 12
        )
        == 0
    )
    form_center_rates, phase_center_rates = [], []
    for band, sign in (((2, 3, 4), -1), ((8, 9, 10), 1)):
        moment_f, derivative_f = _c12_moment_and_rate(
            form_coefficients, form_rates, band
        )
        moment_p, derivative_p = _c12_moment_and_rate(
            phase_coefficients, phase_rates, band
        )
        _, tangent_derivative_p = _c12_moment_and_rate(
            phase_coefficients, tangent_phase_rates, band
        )
        assert moment_f == moment_p == 3 * q**2
        assert sp.expand(derivative_f - 3 * q**2 * (-3 * sigma + sign * sp.I) / 2) == 0
        expected_p = sign * sp.I * q * (3 * sp.sin(q / 2) + sp.sin(3 * q / 2)) / 2
        assert sp.trigsimp(derivative_p - expected_p) == 0
        # The two tangents have exactly the same center rate for every q>0,
        # evaluated at their respective full states, not just as q tends to zero.
        assert (
            sp.simplify(
                sp.im(tangent_derivative_p / moment_p) - sp.im(derivative_f / moment_f)
            )
            == 0
        )
        form_center_rates.append(sp.im(derivative_f / moment_f))
        phase_center_rates.append(sp.im(derivative_p / moment_p))
    # A common channel rotation multiplies each projected band by i, so
    # every band intensity, center and pair proximity mass stays identical.
    assert all(
        sp.simplify(phase_coefficients[k] - sp.I * form_coefficients[k]) == 0
        for k in form_coefficients
    )
    form_relative = sp.simplify(
        6 * (form_center_rates[0] - form_center_rates[1]) / sp.pi
    )
    phase_relative = sp.simplify(
        6 * (phase_center_rates[0] - phase_center_rates[1]) / sp.pi
    )
    assert form_relative == -6 / sp.pi
    assert (
        sp.trigsimp(
            phase_relative + (6 * sp.sin(q / 2) + 2 * sp.sin(3 * q / 2)) / (sp.pi * q)
        )
        == 0
    )
    # sin(t)<t gives strict difference for 0<q<pi/4. The cubic coefficient
    # provides an independent nonzero small-amplitude control.
    difference = sp.series(phase_relative - form_relative, q, 0, 4).removeO()
    assert sp.expand(difference) == 5 * q**2 / (4 * sp.pi)
    assert sp.limit(phase_relative, q, 0) == form_relative


def test_c12_actual_shared_rows_use_both_clock_factors_and_no_invented_phase_rate():
    model, pi, q = _model(), pi_interval(), Q(1, 16)
    profile = tuple(
        q * Q(sp.cos(sp.pi * j / 3) + sp.cos(sp.pi * j / 2)) for j in range(12)
    )
    laplacian = tuple(
        2 * profile[j] - profile[j - 1] - profile[(j + 1) % 12] for j in range(12)
    )
    current = [I(0) for _ in profile]
    for j in range(12):
        next_j = (j + 1) % 12
        edge_current = sin(I(profile[next_j] - profile[j]))
        current[j] += edge_current
        current[next_j] -= edge_current
    form_state = _sine_rates(model, (2,) * 12, laplacian, (1,) * 12, (Q(0),) * 12)
    phase_state = _sine_rates(model, (2,) * 12, (Q(0),) * 12, (1,) * 12, current)
    clock = pi / Q(model.phase_weight)
    sigma = pi * Q(model.epi_weight) / Q(model.phase_weight)
    for j in range(12):
        for residual in (
            clock * form_state["form_rates"][j] + sigma * laplacian[j] / 2,
            clock * form_state["phase_rates"][j] - laplacian[j] / 2,
            clock * phase_state["form_rates"][j] - current[j] / 2,
        ):
            assert residual.contains(0) and residual.abs_max < Q(1, 10**32)
        assert phase_state["phase_rates"][j] == I(0)


def test_same_state_nonlinear_split_does_not_compare_evolved_tangent_histories():
    # A nonlinear scalar model is enough to expose the distinction exactly.
    time = sp.Symbol("t", real=True)

    def flow(z):
        return -z + z**3

    def tangent(z):
        return -z

    supplied_path = 2 + time
    assert (
        sp.expand(flow(supplied_path) - tangent(supplied_path) - supplied_path**3) == 0
    )
    tangent_path = 2 - 2 * time
    assert (
        sp.expand(flow(supplied_path) - tangent(tangent_path) - supplied_path**3)
        == -3 * time
    )
    x, y = sp.symbols("x y", real=True)
    assert sp.expand(flow(x + y) - flow(x) - flow(y) - 3 * x * y * (x + y)) == 0


def test_frozen_whole_family_has_negative_relative_rate_without_refined_radius_premises():
    report = _bound()
    h, kappa = Q(13, 250), Q(170, 171)
    margin, norm = kappa - h * (2 + h), 1 + h
    assert norm**2 / margin < Q(5, 4)
    assert norm / margin < Q(6, 5)
    pi, size, a = pi_interval(), 1024, Q(1, 2**12)
    conservative = (Q(1, 6) + Q(29**2, 2)) * (pi.hi / size) ** 2 * Q(5, 4)
    damping = 2 * size * Q(1, 2**20 - 1) * Q(5, 4)
    nonlinear = size / pi.lo * Q(32, 3) * a**2 * Q(6, 5)
    assert report.conservative_error_upper_bound < conservative < Q(1, 200)
    assert report.damping_error_upper_bound < damping < Q(1, 400)
    assert report.same_state_nonlinear_correction_upper_bound < nonlinear < Q(1, 4000)
    assert 2 * (conservative + damping + nonlinear) < Q(1, 50)
    assert report.relative_center_rate_bounds.lo > -Q(101, 50)
    assert report.relative_center_rate_bounds.hi < -Q(99, 50)
    assert report.positive_band_center_rate_bounds.hi < 0
    assert report.negative_band_center_rate_bounds.lo > 0
    assert report.strictly_decreasing_relative_center_certified
    assert report.unavailable_reasons == ()
    with pytest.raises(FrozenInstanceError):
        report.negative_relative_rate_margin = Q(0)


def test_missing_moment_and_missing_branch_leave_different_fields_unavailable():
    no_moment = _bound(initial_state_error_bound=Q(1, 5 * 2**12))
    assert no_moment.geometry.common_family_bootstrap_certified
    assert not no_moment.band_rate_bounds_available
    for name in (
        "conservative_error_upper_bound",
        "damping_error_upper_bound",
        "same_state_nonlinear_correction_upper_bound",
        "per_center_rate_error_upper_bound",
        "positive_band_center_rate_bounds",
        "negative_band_center_rate_bounds",
        "relative_center_rate_bounds",
        "negative_relative_rate_margin",
    ):
        assert getattr(no_moment, name) is None
    no_branch = _bound(horizon=512)
    assert no_branch.band_rate_bounds_available
    assert no_branch.positive_band_center_rate_bounds is not None
    assert no_branch.negative_band_center_rate_bounds is not None
    assert no_branch.relative_center_rate_bounds is None
    assert not no_branch.strictly_decreasing_relative_center_certified


def test_failed_sign_bound_is_inconclusive_and_does_not_assert_reversal():
    report = _bound(node_count=2**20)
    assert report.band_rate_bounds_available and report.relative_rate_bounds_available
    assert report.relative_center_rate_bounds.contains(0)
    assert report.negative_relative_rate_margin < 0
    assert not report.strictly_decreasing_relative_center_certified
    assert report.unavailable_reasons == (
        "strictly_negative_relative_rate_not_certified",
    )


def test_tail_mass_and_old_identity_targets_are_not_rate_prerequisites():
    report = _bound(localization_radius=2)
    assert not report.geometry.mass_tail_bound_certified
    assert not report.geometry.budget.prepared_identity_certified
    assert report.band_rate_bounds_available and report.relative_rate_bounds_available
    assert report.strictly_decreasing_relative_center_certified


def test_exact_tiny_sources_and_actual_clock_coefficients_survive_admission():
    tiny = Q(1, 10**500)
    model = _model()
    object.__setattr__(model, "phase_weight", tiny)
    object.__setattr__(model, "epi_weight", tiny / (2**20 - 1))
    report = _bound(
        reference_model=model,
        amplitude=tiny,
        initial_state_error_bound=tiny / 1000,
        proximity_distance=Q(np.int64(96), np.int64(1)),
    )
    assert report.geometry.budget.amplitude == tiny
    assert report.geometry.budget.exchange_coefficient == tiny
    assert report.same_state_nonlinear_correction_upper_bound > 0
    assert report.same_state_nonlinear_correction_upper_bound < tiny
    assert report.sigma_bounds.lo > 0
    assert report.strictly_decreasing_relative_center_certified
    assert type(report.geometry.proximity_distance.numerator) is int


@pytest.mark.parametrize(
    "changes",
    [
        {"amplitude": True},
        {"initial_state_error_bound": np.bool_(False)},
        {"proximity_distance": float("nan")},
        {"localization_radius": float("inf")},
        {"horizon": np.int64(256)},
        {"fejer_order": Q(16)},
        {"node_count": 1024.0},
        {"amplitude": 0},
        {"initial_state_error_bound": -Q(1, 10**500)},
        {"proximity_distance": Q(512) + Q(1, 10**500)},
    ],
)
def test_fresh_primitive_admission_is_not_bypassed(changes):
    with pytest.raises((TypeError, ValueError)):
        _bound(**changes)


def test_cached_reports_and_forged_actual_law_are_not_accepted():
    report = _bound()
    with pytest.raises(TypeError):
        _bound(geometry=report.geometry)
    model = _model()
    object.__setattr__(model, "storage_scale", Q(2))
    with pytest.raises(ValueError):
        _bound(reference_model=model)
