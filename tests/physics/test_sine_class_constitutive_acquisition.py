"""Independent whole-source controls for the alternative C9 capture law."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_class_constitutive_acquisition as owner
from tnfr.physics import relational_sine_formed_classes as sine_owner

TIME, ERROR, RADIUS = Q(100), Q(1, 10**10), Q(1, 12)
SLOPE = Q(2046, 9) * Q(355, 113) ** 2
EDGES = tuple((i, (i + 1) % 9) for i in range(9))


@pytest.fixture(scope="module", autouse=True)
def no_sine_transfer_or_scientific_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("alternative acquisition borrowed a sine-law certificate")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        for name in (
            "assess_sine_formed_class_pair",
            "_fresh_class_preparations",
            "_sine_domain",
            "_sine_preparation_from_rows",
        ):
            patch.setattr(sine_owner, name, forbidden)
        yield


def _assess(**changes):
    inputs = dict(
        deformation=Q(1),
        scaled_time=TIME,
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        radius=RADIUS,
    )
    return owner._assess_constitutive_class_pair(**(inputs | changes))


@pytest.fixture(scope="module")
def endpoints():
    return tuple(_assess(deformation=coefficient) for coefficient in (0, 1))


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _upper_close(mp, bound, expected):
    difference = _mp(mp, bound) - expected
    assert 0 <= difference < mp.mpf("1e-25") * max(1, abs(expected))


def _laplace(values):
    return tuple(
        2 * values[i] - values[(i - 1) % 9] - values[(i + 1) % 9] for i in range(9)
    )


def test_fixed_source_geometry_cost_and_quartic_correction(endpoints, mp):
    for report in endpoints:
        coefficient = report.deformation
        assert report.nodes == tuple(range(9))
        assert set(report.edges) == {(min(i, j), max(i, j)) for i, j in EDGES}
        assert report.held_capacities == (Q(1),) * 9
        assert report.metric_weights == (Q(2),) * 9
        assert report.initial_phases == (Q(0),) * 9
        assert report.preparation_dimension == 16
        assert report.full_relative_preparation
        assert (
            report.initial_potential_deformation_upper_bound
            == 6 * coefficient * ERROR**4
        )
        for index, k in enumerate((1, 2)):
            form = tuple(k * SLOPE * (j - 4) for j in range(9))
            assert report.initial_forms_by_class[index] == form
            assert sum(form) == 0
            cost = sum(((form[j] - form[i]) ** 2 / 2 for i, j in EDGES), Q(0))
            assert report.initial_form_storage_by_class[index] == cost
            assert cost == 36 * k**2 * SLOPE**2
            # Each edge carries its own cross term, including the closing edge.
            box_cost = (
                sum(
                    ((abs(form[j] - form[i]) + 2 * ERROR) ** 2 / 2 for i, j in EDGES),
                    Q(0),
                )
                + 18 * ERROR**2
                + 6 * coefficient * ERROR**4
            )
            assert report.initial_storage_upper_bounds[index] == box_cost
            assert box_cost < 10**9

    # Check the source potential estimate independently, far above roundoff.
    report = _assess(phase_error_bound=Q(1, 100))
    phases = tuple(mp.mpf(j - 4) / 400 for j in range(9))
    extra = sum(
        (1 - mp.cos(9 * (phases[j] - phases[i]))) ** 2 / 39366 for i, j in EDGES
    )
    assert 0 < extra < _mp(mp, report.initial_potential_deformation_upper_bound)
    for i, j in EDGES:
        gap = phases[j] - phases[i]
        assert (1 - mp.cos(9 * gap)) ** 2 / 39366 <= gap**4 / 24


def test_complete_c9_rows_have_own_storage_balance_and_unchanged_gap(endpoints, mp):
    spectrum = sorted(1 - mp.cos(2 * mp.pi * j / 9) for j in range(9))
    assert spectrum[1] > mp.mpf(1) / 5
    assert spectrum[-1] < 2
    form = tuple(mp.mpf(2 * j - 8) / 13 for j in range(9))
    phase = tuple(mp.mpf((5 * j) % 11 - 5) / 17 for j in range(9))
    laplace = _laplace(form)
    gamma = 1 / (1023 * mp.pi)
    for report in endpoints:
        coefficient = _mp(mp, report.deformation)
        current = tuple(
            sum(
                mp.sin(phase[j] - phase[i])
                + coefficient
                * (
                    mp.sin(9 * (phase[j] - phase[i]))
                    - mp.sin(18 * (phase[j] - phase[i])) / 2
                )
                / 2187
                for j in ((i - 1) % 9, (i + 1) % 9)
            )
            for i in range(9)
        )
        dx = tuple((-q + gamma * s) / 2 for q, s in zip(laplace, current))
        dt = tuple(gamma * q / 2 for q in laplace)
        derivative = sum(
            q * x - s * theta for q, x, s, theta in zip(laplace, dx, current, dt)
        )
        assert abs(derivative + sum(q**2 for q in laplace) / 2) < mp.mpf("1e-90")
        assert abs(sum(dx)) < mp.mpf("1e-90")
        assert abs(sum(dt)) < mp.mpf("1e-90")
        forcing = mp.sqrt(sum(2 * (s / 2) ** 2 for s in current))
        assert forcing < _mp(mp, report.forcing_norm_bounds.hi)


def test_independent_high_precision_transit_and_own_law_barrier(endpoints, mp):
    time, rx, rt, radius = map(lambda q: _mp(mp, q), (TIME, ERROR, ERROR, RADIUS))
    gamma, gap = 1 / (1023 * mp.pi), mp.mpf(1) / 5
    metric_norm = mp.sqrt(18)
    error_x, error_theta = gamma * metric_norm * rx, metric_norm * rt
    decay, slope = mp.exp(-gap * time), _mp(mp, SLOPE)
    for report in endpoints:
        coefficient = _mp(mp, report.deformation)
        forcing = metric_norm * (1 + coefficient / 1458)
        upper_hessian = 1 + 2 * coefficient / 243
        _contains(mp, report.gamma_bounds, gamma)
        _contains(mp, report.eta_bounds, gamma**2)
        _contains(mp, report.metric_unit_norm_bounds, metric_norm)
        _contains(mp, report.forcing_norm_bounds, forcing)
        _contains(mp, report.exponential_decay_bounds, decay)
        _upper_close(mp, report.initial_phase_error_norm_upper_bound, error_theta)
        for index, k in enumerate((1, 2)):
            v = gamma * k * slope * mp.sqrt(120)
            rz = decay * (v + error_x) + gamma**2 * forcing / gap
            form = rz / (gamma * mp.sqrt(2))
            phase_error = (
                rz + gamma**2 * time * forcing + error_x + error_theta
            ) / mp.sqrt(2)
            distance = mp.sqrt(60) * k * abs(gamma * slope - 2 * mp.pi / 9)
            phase = distance + phase_error
            _upper_close(mp, report.nominal_scaled_initial_norm_upper_bounds[index], v)
            _upper_close(
                mp, report.scaled_initial_error_norm_upper_bounds[index], error_x
            )
            _upper_close(mp, report.scaled_form_radius_upper_bounds[index], rz)
            _upper_close(mp, report.endpoint_form_norm_upper_bounds[index], form)
            _upper_close(
                mp, report.endpoint_phase_error_norm_upper_bounds[index], phase_error
            )
            _contains(mp, report.proxy_target_distance_bounds[index], distance)
            _upper_close(
                mp, report.endpoint_target_phase_norm_upper_bounds[index], phase
            )
            _upper_close(
                mp,
                report.endpoint_relative_norm_squared_upper_bounds[index],
                form**2 + phase**2,
            )
            _upper_close(
                mp, report.endpoint_form_storage_upper_bounds[index], 2 * form**2
            )
            _upper_close(
                mp,
                report.endpoint_phase_excess_storage_upper_bounds[index],
                2 * upper_hessian * phase**2,
            )
            _upper_close(
                mp,
                report.endpoint_excess_storage_upper_bounds[index],
                2 * form**2 + 2 * upper_hessian * phase**2,
            )
            _contains(
                mp,
                report.target_phase_storage_bounds_by_class[index],
                9 * (1 - mp.cos(2 * k * mp.pi / 9)),
            )
        angle = 4 * mp.pi / 9 + mp.sqrt(2) * radius
        stiffness = mp.cos(angle) - 2 * coefficient / 243
        _contains(mp, report.radius_angle_bounds, angle)
        assert 0 < _mp(mp, report.phase_hessian_lower_bound) <= stiffness
        assert 0 < _mp(mp, report.barrier_lower_bound) <= stiffness * radius**2 / 5
        assert stiffness * radius**2 / 5 - _mp(
            mp, report.barrier_lower_bound
        ) < mp.mpf("1e-25")


def test_worst_endpoint_certifies_entire_parameter_interval(endpoints):
    sine, alternative = endpoints
    # Positive coefficients in these analytic bounds, not a numerical sweep,
    # make lambda=1 the worst case throughout the declared interval [0, 1].
    assert sine.current_absolute_upper_bound == 1
    assert alternative.current_absolute_upper_bound == 1 + Q(1, 1458)
    assert sine.phase_hessian_upper_bound == 1
    assert alternative.phase_hessian_upper_bound == 1 + Q(2, 243)
    assert alternative.phase_hessian_lower_bound < sine.phase_hessian_lower_bound
    assert alternative.barrier_lower_bound < sine.barrier_lower_bound
    assert alternative.barrier_lower_bound > Q(672, 10**7)
    for report in endpoints:
        assert report.status == "certified_two_formed_classes"
        assert report.unavailable_reasons == ()
        assert report.source_budget_certified_by_class == (True, True)
        assert report.formation_certified_by_class == (True, True)
        assert report.symmetry_inequivalent
        assert report.positive_phase_hessian_certified
        assert max(report.endpoint_relative_norm_squared_upper_bounds) < Q(23, 10**6)
        assert max(report.endpoint_excess_storage_upper_bounds) < Q(45, 10**6)
        assert min(item.lo for item in report.storage_barrier_margin_bounds) > Q(
            222, 10**7
        )
    assert (
        sine.scaled_initial_error_norm_upper_bounds
        == alternative.scaled_initial_error_norm_upper_bounds
    )
    assert (
        sine.initial_phase_error_norm_upper_bound
        == alternative.initial_phase_error_norm_upper_bound
    )
    for index in range(2):
        assert (
            sine.endpoint_excess_storage_upper_bounds[index]
            < alternative.endpoint_excess_storage_upper_bounds[index]
        )
        assert (
            alternative.initial_storage_upper_bounds[index]
            - sine.initial_storage_upper_bounds[index]
            == 6 * ERROR**4
        )


def test_short_horizon_retains_the_forcing_integral_time_bound(mp):
    report = _assess(scaled_time=1)
    gamma = 1 / (1023 * mp.pi)
    forcing = mp.sqrt(18) * (1 + mp.mpf(1) / 1458)
    for index, k in enumerate((1, 2)):
        initial = gamma * (
            mp.sqrt(120) * k * _mp(mp, SLOPE) + mp.sqrt(18) * _mp(mp, ERROR)
        )
        expected = mp.exp(-mp.mpf(1) / 5) * initial + gamma**2 * forcing
        _upper_close(mp, report.scaled_form_radius_upper_bounds[index], expected)
    assert report.status == "unavailable"


def test_equal_source_budget_abstains_separately_from_formation(monkeypatch, endpoints):
    monkeypatch.setattr(
        owner, "_SOURCE_BUDGET", endpoints[1].initial_storage_upper_bounds[0]
    )
    result = _assess()
    assert (
        result.source_budget_margin_bounds[0].lo
        == result.source_budget_margin_bounds[0].hi
        == 0
    )
    assert result.source_budget_certified_by_class == (False, False)
    assert result.formation_certified_by_class == (True, True)
    assert result.status == "unavailable"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"scaled_time": 0}, "strict_endpoint_radius_not_certified"),
        ({"radius": 1}, "strict_acute_radius_not_certified"),
        ({"radius": Q(1, 10**6)}, "strict_excess_storage_barrier_not_certified"),
        ({"phase_error_bound": 1}, "whole_initial_set_zero_winding_not_certified"),
        ({"form_error_bound": 10000}, "strict_source_storage_budget_not_certified"),
        ({"deformation": 10}, "strict_own_law_phase_hessian_not_certified"),
    ],
)
def test_insufficient_own_law_bounds_abstain(changes, reason):
    result = _assess(**changes)
    assert result.status == "unavailable"
    assert any(reason in item for item in result.unavailable_reasons)
    for name, value in changes.items():
        assert getattr(result, name) == value
    if changes.get("deformation") == 10:
        assert result.acute_radius_certified
        assert result.phase_hessian_lower_bound_candidate < 0
        assert result.phase_hessian_lower_bound is None
        assert result.barrier_lower_bound is None


@pytest.mark.parametrize(
    "form,phase,dimension", [(0, 0, 0), (0, ERROR, 8), (ERROR, 0, 8)]
)
def test_zero_errors_retain_lower_dimensional_source(form, phase, dimension):
    result = _assess(form_error_bound=form, phase_error_bound=phase)
    assert result.preparation_dimension == dimension
    assert not result.full_relative_preparation


def test_tiny_exact_source_errors_and_barrier_remain_distinct():
    tiny = Q(1, 2**300)
    result = _assess(form_error_bound=tiny, phase_error_bound=tiny)
    assert result.form_error_bound == result.phase_error_bound == tiny
    assert result.preparation_dimension == 16
    assert all(value > 0 for value in result.scaled_initial_error_norm_upper_bounds)
    result = _assess(radius=tiny, form_error_bound=0, phase_error_bound=0)
    assert result.positive_phase_hessian_certified
    assert result.barrier_lower_bound == 0
    assert result.status == "unavailable"
    assert result.storage_barrier_certified_by_class == (False, False)


@pytest.mark.parametrize(
    "field",
    ["deformation", "scaled_time", "form_error_bound", "phase_error_bound", "radius"],
)
@pytest.mark.parametrize("bad", [True, "1", 1j, float("inf"), float("nan")])
def test_scalar_admission_precedes_spectral_or_exponential_work(
    monkeypatch, field, bad
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached proof arithmetic")

    monkeypatch.setattr(owner, "_verify_cycle_gap", forbidden)
    monkeypatch.setattr(owner, "_negative_exp_bounds", forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"deformation": -1},
        {"scaled_time": -1},
        {"scaled_time": 20481},
        {"form_error_bound": -1},
        {"phase_error_bound": -1},
        {"radius": 0},
    ],
)
def test_exact_domain_boundaries_reject(changes):
    with pytest.raises(ValueError):
        _assess(**changes)
