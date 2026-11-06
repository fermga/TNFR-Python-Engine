"""Static controls for shared preparation and a prospective cubic predictor."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import numpy as np
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import pi_interval
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
    return owner._bound_localized_mobile_interaction(
        **(
            dict(
                reference_model=_model(),
                node_count=1024,
                fejer_order=16,
                amplitude=Q(1, 2**12),
                initial_state_error_bound=Q(1, 1000 * 2**12),
                localization_radius=96,
                horizon=256,
            )
            | changes
        )
    )


def _incidence(size):
    matrix = sp.zeros(size)
    for edge in range(size):
        matrix[edge, edge] = 1
        matrix[edge, (edge + 1) % size] = -1
    return matrix


def test_actual_complete_rows_have_the_stated_mixed_cubic_source_and_sign():
    size = 8
    incidence = _incidence(size)
    phase_a = sp.Matrix([Q(j * j - 3, 17) for j in range(size)])
    phase_b = sp.Matrix([Q((-1) ** j * (j + 2), 19) for j in range(size)])
    gap_a, gap_b = incidence * phase_a, incidence * phase_b
    scale = sp.Symbol("scale", real=True)
    cubic_currents = []
    for j in range(size):
        current = sum(
            sp.sin(scale * (phase_a[k] + phase_b[k] - phase_a[j] - phase_b[j]))
            - sp.sin(scale * (phase_a[k] - phase_a[j]))
            - sp.sin(scale * (phase_b[k] - phase_b[j]))
            for k in ((j - 1) % size, (j + 1) % size)
        )
        assert sp.diff(current, scale).subs(scale, 0) == 0
        cubic_currents.append(Q(sp.diff(current, scale, 3).subs(scale, 0) / 6))
    expected = incidence.T * sp.Matrix(
        [a * b * (a + b) / 4 for a, b in zip(gap_a, gap_b)]
    )
    assert any(expected)
    assert sum(expected) == 0  # The cubic source retains the conserved form mean.
    model = _model()
    rates = _sine_rates(
        model, (2,) * size, (Q(0),) * size, (Q(1),) * size, cubic_currents
    )
    clock = pi_interval() / Q(model.phase_weight)
    for j in range(size):
        difference = rates["form_rates"][j] * clock - Q(expected[j])
        assert difference.contains(0)
        assert difference.abs_max < Q(1, 10**32)
        assert rates["phase_rates"][j].contains(0)
    wrong_sign = -expected
    assert wrong_sign != expected
    assert incidence.T * sp.zeros(size, 1) == sp.zeros(size, 1)


def test_actual_damped_tangent_is_contractive_without_removing_the_loss():
    incidence = _incidence(4)
    half_laplacian = incidence.T * incidence / 2
    damping = sp.Symbol("sigma", positive=True)
    tangent = (
        (-damping * half_laplacian)
        .row_join(-half_laplacian)
        .col_join(half_laplacian.row_join(sp.zeros(4)))
    )
    form = sp.Matrix(sp.symbols("x:4", real=True))
    phase = sp.Matrix(sp.symbols("p:4", real=True))
    state = form.col_join(phase)
    loss = sp.expand((state.T * tangent * state)[0])
    assert sp.expand(loss + damping * (incidence * form).dot(incidence * form) / 2) == 0
    assert tangent != tangent.subs(damping, 0)
    assert (
        sp.expand(loss.subs({form[0]: 1, form[1]: 0, form[2]: 0, form[3]: 0}))
        == -damping
    )


def test_shared_source_cancels_linear_error_but_three_independent_sources_do_not():
    operator = sp.Matrix([[2, -3], [5, 7]])
    a, b = sp.Matrix([1, 2]), sp.Matrix([-4, 3])
    error_a, error_b = sp.Matrix([Q(1, 17), Q(2, 19)]), sp.Matrix([Q(-3, 23), 0])
    shared = a + error_a + b + error_b
    assert operator * shared - operator * (a + error_a) - operator * (
        b + error_b
    ) == sp.zeros(2, 1)
    independent = a + b
    assert operator * independent - operator * (a + error_a) - operator * (
        b + error_b
    ) != sp.zeros(2, 1)
    # Squared localization weights include linear interference even with I=0.
    linear_a, linear_b = operator * a, operator * b
    intensity_defect = (
        (linear_a + linear_b).dot(linear_a + linear_b)
        - linear_a.dot(linear_a)
        - linear_b.dot(linear_b)
    )
    assert intensity_defect == 2 * linear_a.dot(linear_b) != 0


def test_disjoint_carrier_bands_give_sqrt_two_joint_norm_despite_spatial_overlap():
    size = 16
    left_modes = (3, 4, 5)
    right_modes = (-5, -4, -3)
    root = sp.exp(sp.I * sp.pi / 8)
    for left in left_modes:
        for right in right_modes:
            character_sum = sum(root ** ((left - right) * j) for j in range(size))
            assert sp.simplify(sp.expand_complex(character_sum)) == 0
    # At the common center both real envelopes are nonzero; localization alone
    # cannot replace full-state Fourier orthogonality.
    weights = (Q(1, 2), Q(1), Q(1, 2))
    assert sum(weights) ** 2 > 0
    individual_norm_squared = sum(weight**2 for weight in weights)
    joint_norm_squared = sum(weight**2 for weight in weights + weights)
    assert joint_norm_squared == 2 * individual_norm_squared


def test_cubic_and_fifth_remainders_follow_from_sine_and_two_incidence_norms():
    argument = sp.Symbol("q", real=True)
    scalar_defect = (argument - sp.sin(argument)) / 2
    assert sp.limit(scalar_defect / argument**3, argument, 0) == sp.Rational(1, 12)
    assert sp.limit(
        (scalar_defect - argument**3 / 12) / argument**5, argument, 0
    ) == -sp.Rational(1, 240)
    incidence_norm = 2
    cubic_constant = Q(1, 12) * incidence_norm**4
    fifth_constant = Q(1, 240) * incidence_norm**6
    derivative_constant = 3 * cubic_constant
    assert cubic_constant == Q(4, 3)
    assert fifth_constant == Q(4, 15)
    time, horizon, radius, peak = sp.symbols("s T r S", nonnegative=True)
    first_error = cubic_constant * time * peak**2 * radius
    integrand = (
        derivative_constant * peak**2 * first_error + fifth_constant * peak**4 * radius
    )
    integrated = sp.integrate(integrand, (time, 0, horizon))
    report = _bound()
    independently_derived = sum(
        integrated.subs({horizon: report.horizon, radius: r, peak: bound})
        for r, bound in (
            (report.paired_state_radius, report.paired_state_sup_norm_upper_bound),
            (report.single_state_radius, report.single_state_sup_norm_upper_bound),
            (report.single_state_radius, report.single_state_sup_norm_upper_bound),
        )
    )
    assert report.nominal_cubic_remainder_upper_bound == independently_derived


def test_mixed_cubic_has_null_sources_and_is_not_the_sum_of_the_two_cubics():
    left, right = sp.symbols("left right", real=True)
    mixed = sp.expand(((left + right) ** 3 - left**3 - right**3) / 12)
    assert sp.factor(mixed) == left * right * (left + right) / 4
    assert mixed.subs(left, 0) == mixed.subs(right, 0) == 0
    assert mixed.subs(right, -left) == 0
    assert mixed.subs({left: 1, right: 2}) == sp.Rational(3, 2)


def test_shared_preparation_error_is_cubic_sensitivity_not_a_raw_linear_allowance():
    report = _bound()
    eps, horizon = report.initial_state_error_bound, report.horizon
    # Variation of constants gives eta*(exp(L*T)-1) for the nonlinear part.
    # Each exponential coefficient is bounded by its geometric counterpart.
    for degree in range(8):
        assert Q(1, int(sp.factorial(degree))) <= 1
    exponents = [
        4 * peak**2 * horizon
        for peak in (
            report.paired_state_sup_norm_upper_bound,
            report.single_state_sup_norm_upper_bound,
        )
    ]
    assert all(0 < exponent < 1 for exponent in exponents)
    independent_bound = (
        2 * eps * sum(exponent / (1 - exponent) for exponent in exponents)
    )
    assert report.shared_preparation_sensitivity_upper_bound == independent_bound
    assert independent_bound < eps / 1000
    exact = _bound(initial_state_error_bound=0)
    assert exact.shared_preparation_sensitivity_upper_bound == 0
    assert (
        exact.interaction_proxy_error_upper_bound
        == exact.nominal_cubic_remainder_upper_bound
        > 0
    )


def test_frozen_budget_closes_identity_and_remainder_without_evaluating_a_signal():
    report = _bound()
    assert report.identity_bootstrap_certified
    assert report.prepared_identity_certified
    assert report.cubic_remainder_certified
    assert report.status == "certified"
    assert report.joint_relative_identity_error_upper_bound < Q(1, 25)
    assert report.band_tail_fraction_upper_bound < Q(1, 25)
    assert report.joint_state_norm_lower_bound > report.amplitude
    assert report.band_state_norm_lower_bound > Q(9, 10) * report.amplitude
    assert report.single_state_bootstrap_margin > 0
    assert report.paired_state_bootstrap_margin > 0
    assert report.interaction_proxy_error_upper_bound < Q(1, 10**9)
    assert (
        report.interaction_proxy_error_upper_bound
        < report.initial_state_error_bound / 1000
    )
    assert report.single_state_sup_norm_upper_bound < Q(19, 100) * report.amplitude
    assert report.paired_state_sup_norm_upper_bound < Q(36, 100) * report.amplitude
    assert report.shared_preparation_sensitivity_upper_bound < Q(4939, 10**15)
    assert report.nominal_cubic_remainder_upper_bound < Q(4266, 10**18)
    assert report.interaction_norm_bounds is None
    assert report.initial_centers == (report.horizon // 2, -report.horizon // 2)
    assert report.endpoint_centers == tuple(
        -center for center in report.initial_centers
    )
    with pytest.raises(FrozenInstanceError):
        report.status = "interaction_detected"


def test_identity_uses_the_joint_norm_but_each_band_uses_its_own_signal_denominator():
    report = _bound()
    a, error, horizon = (
        report.amplitude,
        report.initial_state_error_bound,
        report.horizon,
    )
    pi_upper = pi_interval().hi
    dispersion = (
        horizon * (2 * pi_upper * Q(report.fejer_order - 1, report.node_count)) ** 3 / 6
    )
    damping = (
        2
        * pi_upper
        * report.form_loss_coefficient
        * horizon
        / report.exchange_coefficient
    )
    nonlinear = Q(1, 12) * 2**4 * horizon * (2 * a) ** 3
    expected = (
        2 * error + report.sqrt_two_bounds.hi * a * (damping + dispersion) + nonlinear
    )
    assert report.joint_identity_error_upper_bound == expected
    assert report.joint_relative_identity_error_upper_bound < expected / a
    assert report.band_state_norm_lower_bound == a - expected
    assert (
        report.reference_tail_fraction_upper_bound
        == Q(report.node_count**3, 16 * report.fejer_order**3)
        / (report.localization_radius - 1) ** 3
    )


def test_failed_bootstraps_leave_actual_bounds_unavailable():
    report = _bound(amplitude=1)
    assert not report.identity_bootstrap_certified
    assert not report.cubic_remainder_certified
    assert report.fifth_order_remainder_majorant > 0
    for field in (
        "band_tail_fraction_upper_bound",
        "joint_state_norm_lower_bound",
        "joint_state_norm_upper_bound",
        "band_state_norm_lower_bound",
        "nominal_cubic_remainder_upper_bound",
        "shared_preparation_sensitivity_upper_bound",
        "interaction_proxy_error_upper_bound",
        "interaction_norm_bounds",
        "single_state_sup_norm_upper_bound",
        "paired_state_sup_norm_upper_bound",
    ):
        assert getattr(report, field) is None
    assert report.status == "not_certified"


def test_single_family_bootstrap_requires_strict_margin_and_does_not_hide_pair_failure():
    a, horizon = Q(1, 2**12), 256
    radius = Q(9, 8) * a
    boundary_error = radius - a - Q(4, 3) * horizon * radius**3
    report = _bound(initial_state_error_bound=boundary_error)
    assert report.single_state_bootstrap_margin == 0
    assert not report.cubic_remainder_certified
    assert report.interaction_proxy_error_upper_bound is None


def test_nonseparated_endpoint_windows_fail_identity_without_invalidating_cubic_budget():
    report = _bound(localization_radius=128)
    assert report.endpoint_window_separation_margin == 0
    assert not report.prepared_identity_certified
    assert report.cubic_remainder_certified
    assert report.interaction_proxy_error_upper_bound is not None


def test_exact_tiny_scalars_and_authoritative_clock_are_not_lost_to_float_materialization():
    tiny = Q(1, 10**500)
    model = _model()
    object.__setattr__(model, "phase_weight", tiny)
    object.__setattr__(model, "epi_weight", tiny / (2**20 - 1))
    report = _bound(
        reference_model=model, amplitude=tiny, initial_state_error_bound=tiny / 1000
    )
    assert report.amplitude == report.exchange_coefficient == tiny
    assert report.initial_state_error_bound == tiny / 1000
    assert report.prepared_identity_certified
    assert report.cubic_remainder_certified
    assert report.interaction_proxy_error_upper_bound > 0
    assert report.original_horizon_bounds.lo > 0
    assert report.clock_rate_bounds.lo == 0 < report.clock_rate_bounds.hi


def test_tiny_exchange_with_ordinary_loss_is_failed_sufficiency_not_zero_clock():
    model = _model()
    object.__setattr__(model, "phase_weight", Q(1, 10**500))
    report = _bound(reference_model=model)
    assert not report.prepared_identity_certified
    assert not report.cubic_remainder_certified
    assert report.interaction_proxy_error_upper_bound is None
    assert report.original_horizon_bounds.lo > 0
    assert report.exchange_coefficient > 0


@pytest.mark.parametrize(
    "name", ["amplitude", "initial_state_error_bound", "localization_radius"]
)
@pytest.mark.parametrize(
    "bad", [True, np.bool_(False), float("nan"), float("inf"), complex(1, 0), {}, []]
)
def test_physical_scalar_admission_precedes_arithmetic(name, bad):
    with pytest.raises((TypeError, ValueError)):
        _bound(**{name: bad})


@pytest.mark.parametrize("name", ["node_count", "fejer_order", "horizon"])
@pytest.mark.parametrize("bad", [True, np.int64(16), Q(16), 16.0])
def test_integer_geometry_requires_ordinary_python_integers(name, bad):
    with pytest.raises(TypeError):
        _bound(**{name: bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"node_count": 10},
        {"fejer_order": 1},
        {"fejer_order": 257},
        {"horizon": 255},
        {"horizon": 0},
        {"horizon": 514},
        {"amplitude": 0},
        {"amplitude": -1},
        {"initial_state_error_bound": -1},
        {"localization_radius": 1},
        {"localization_radius": 512},
    ],
)
def test_outside_theorem_domain_is_rejected(changes):
    with pytest.raises(ValueError):
        _bound(**changes)


def test_complete_law_is_readmitted_and_alias_models_are_not_silently_substituted():
    for model in (
        _model(storage_scale=2),
        _model(phase_domain="acute"),
        _model(epi_weight=0),
    ):
        with pytest.raises(ValueError):
            _bound(reference_model=model)
    model = _model()
    object.__setattr__(model, "epi_weight", True)
    with pytest.raises((TypeError, ValueError)):
        _bound(reference_model=model)
