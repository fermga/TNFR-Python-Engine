"""Independent own-law controls for uniform source-path capture."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_constitutive_path_capture as owner
from tnfr.physics import _sine_preparation_path_capture as sine_capture
from tnfr.physics import _sine_preparation_path_selection as source_owner

ERROR = Q(1, 10**10)


@pytest.fixture(scope="module", autouse=True)
def no_sine_verdict_or_trajectory_transfer():
    def forbidden(*args, **kwargs):
        pytest.fail("own-law capture consumed a sine classification")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_owner, "_assess_preparation_path_selection", forbidden)
        patch.setattr(sine_capture, "_assess_preparation_path_selection", forbidden)
        patch.setattr(sine_capture, "_assess_preparation_path_capture", forbidden)
        yield


def _assess(**changes):
    inputs = dict(
        deformation_interval=(0, 1),
        parameter_interval=(Q(3, 4), Q(4, 5)),
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
    )
    return owner._assess_constitutive_preparation_path_capture(**(inputs | changes))


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def endpoints():
    return tuple(_assess(deformation_interval=(value, value)) for value in (0, 1))


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _upper_close(mp, bound, value):
    assert 0 <= _mp(mp, bound) - value < mp.mpf("1e-25") * max(1, abs(value))


def _lower_close(mp, bound, value):
    assert 0 <= value - _mp(mp, bound) < mp.mpf("1e-25") * max(1, abs(value))


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _current(mp, angle, deformation):
    return (
        mp.sin(angle)
        + deformation * (mp.sin(9 * angle) - mp.sin(18 * angle) / 2) / 2187
    )


def test_closing_current_breaks_the_sine_reflection_identity(mp):
    q = mp.mpf(1) / 4
    closing = _current(mp, mp.pi - q, 1)
    actual_formula = mp.sin(q) + (mp.sin(9 * q) + mp.sin(18 * q) / 2) / 2187
    assert abs(closing - actual_formula) < mp.mpf("1e-95")
    defect = closing - _current(mp, q, 1)
    assert abs(defect - mp.sin(18 * q) / 2187) < mp.mpf("1e-95")
    assert abs(defect) > mp.mpf("0.0004")


def test_actual_closing_current_cancels_at_every_possible_minimum(report, mp):
    epsilon = _mp(mp, report.mixed_field_error_upper_bound)
    for deformation in (0, 1):
        for q in (mp.mpf(-7) / 4, mp.mpf(-1), mp.mpf(0), mp.mpf(63) / 100):
            minimum = mp.mpf(9) / 10 + q / 2
            remainder = (3 * mp.pi + q - minimum) / 7
            assert minimum <= remainder < mp.mpf(13) / 10
            for index in range(8):
                gaps = [remainder] * 8
                gaps[index] = minimum
                currents = [_current(mp, gap, deformation) for gap in gaps]
                currents.append(_current(mp, mp.pi - q, deformation))
                derivative = (
                    currents[index - 1] - 2 * currents[index] + currents[index + 1]
                ) / 2
                q_derivative = currents[8] - (currents[0] + currents[7]) / 2
                # Independent extreme residuals enlarge the actual correlated set.
                lower = derivative - epsilon - (q_derivative + epsilon) / 2
                assert lower >= -3 * epsilon / 2 - mp.mpf("1e-95")


def test_law_independent_source_norms_do_not_consume_the_force_bound(endpoints, mp):
    sine, deformed = (item.source for item in endpoints)
    assert sine == deformed
    gamma = 1 / (1023 * mp.pi)
    transport = mp.mpf(1025) / 1023
    error = _mp(mp, ERROR)
    _upper_close(
        mp,
        sine.scaled_initial_error_norm_upper_bound,
        gamma * mp.sqrt(18) * transport * error,
    )
    _upper_close(mp, sine.initial_phase_error_norm_upper_bound, mp.sqrt(18) * error)
    assert sine.parameter_interval == (Q(3, 4), Q(4, 5))
    assert sine.path.isometry_certified
    assert not hasattr(sine, "forcing_norm_bounds")
    assert not hasattr(sine, "capture_certified")


def test_entire_rectangle_is_captured_without_a_sine_verdict(report):
    assert report.deformation_interval == (Q(0), Q(1))
    assert report.source.parameter_interval == (Q(3, 4), Q(4, 5))
    assert report.source_policy_certified
    assert report.passage_certified and report.capture_certified
    assert report.captured_winding == 1
    assert all(margin > 0 for _, margin in report.sufficient_margins)
    assert report.unresolved_conditions == ()
    assert report.slow_time_deadline == 136


def test_own_law_source_cost_retains_the_quartic_correction(endpoints, mp):
    sine, deformed = endpoints
    source = deformed.source
    first = source.path.source.initial_forms_by_class[0]
    edges = tuple((j, (j + 1) % 9) for j in range(9))
    independent_cost = (
        sum(((abs(first[j] - first[i]) + 2 * ERROR) ** 2 / 2 for i, j in edges), Q(0))
        + 18 * ERROR**2
    )
    assert sine.common_initial_storage_upper_bound == independent_cost
    assert deformed.initial_potential_deformation_upper_bound == 6 * ERROR**4
    assert (
        deformed.common_initial_storage_upper_bound - independent_cost == 6 * ERROR**4
    )
    assert deformed.common_initial_storage_upper_bound < 10**15
    phases = [mp.mpf(j - 4) / 400 for j in range(9)]
    for i, j in edges:
        gap = phases[j] - phases[i]
        assert (1 - mp.cos(9 * gap)) ** 2 / 39366 <= gap**4 / 24


def test_own_force_transit_and_mixed_initialization_are_independent(report, mp):
    source = report.source
    gamma = 1 / (1023 * mp.pi)
    forcing = mp.sqrt(18) * (1 + mp.mpf(1) / 1458)
    upper_hessian = 1 + mp.mpf(2) / 243
    _contains(mp, report.forcing_norm_bounds, forcing)
    assert report.current_absolute_upper_bound == 1 + Q(1, 1458)
    assert report.phase_hessian_upper_bound == 1 + Q(2, 243)
    initial = _mp(mp, source.scaled_initial_norm_upper_bound)
    radius = mp.exp(-40) * initial + 5 * gamma**2 * forcing
    _upper_close(mp, report.scaled_form_radius_upper_bound, radius)
    z = max(radius, 5 * gamma**2 * forcing)
    _upper_close(mp, report.uniform_scaled_form_norm_upper_bound, z)
    epsilon = 2 * upper_hessian * z
    _upper_close(mp, report.mixed_field_error_upper_bound, epsilon)
    error = (
        _mp(mp, source.scaled_initial_error_norm_upper_bound)
        + _mp(mp, source.initial_phase_error_norm_upper_bound)
        + 200 * gamma**2 * forcing
    )
    _upper_close(mp, report.initial_mixed_phase_error_norm_upper_bound, error)
    _upper_close(mp, report.endpoint_phase_error_norm_upper_bound, error + radius)
    assert report.mixed_field_error_upper_bound < Q(1, 200000)
    assert (
        report.endpoint_phase_error_norm_upper_bound
        > report.initial_mixed_phase_error_norm_upper_bound
    )


def test_two_current_errors_and_the_new_contraction_are_charged(report, mp):
    original_speeds = (
        mp.sin(mp.mpf(607) / 500) - mp.sin(mp.mpf(63) / 100),
        mp.sin(mp.mpf(649) / 1000),
        mp.sin(mp.mpf(1) / 2),
    )
    for bound, value in zip(
        report.stage_one_current_speed_lower_bounds, original_speeds
    ):
        _lower_close(mp, bound, value - mp.mpf(1) / 729)
    speed = (
        min(original_speeds)
        - mp.mpf(1) / 729
        - _mp(mp, report.mixed_field_error_upper_bound)
    )
    _lower_close(mp, report.stage_one_passage_speed_lower_bound, speed)
    assert report.stage_one_passage_speed_lower_bound > Q(3, 10)
    assert report.stage_one_passage_time_upper_bound == Q(119, 15) < 8
    curvature = mp.cos(mp.mpf(7) / 5) - mp.mpf(2) / 243
    _lower_close(mp, report.stage_two_curvature_lower_bound, curvature)
    assert mp.mpf(5) / 32 < curvature < mp.mpf(1) / 6
    assert report.contraction_rate == Q(1, 32)
    _contains(mp, report.stage_two_decay_bounds, mp.exp(-4))
    radius = 15 * mp.exp(-4) + 32 * _mp(mp, report.mixed_field_error_upper_bound)
    _upper_close(mp, report.stage_two_phase_distance_upper_bound, radius)


def test_mixed_source_edges_keep_the_original_cayley_lift(report, mp):
    gamma = 1 / (1023 * mp.pi)
    scale = mp.mpf(2046) / 9 * (mp.mpf(355) / 113) ** 2
    kappa = mp.mpf(1) / (27 * 512**2)
    error = _mp(mp, report.initial_mixed_phase_error_norm_upper_bound)
    for parameter in (Q(3, 4), Q(4, 5)):
        s = _mp(mp, parameter)
        denominator = 1 + kappa * s**2
        p = (
            9 * 512 * (1 - kappa * s * s + 2 * kappa * s)
            - (2 * s - 1 + kappa * s * s) / 512
        ) / denominator
        q = (1 + s + kappa * (3 * s - s * s)) / denominator
        lift = [
            gamma * scale * (p * (int(j == 3) - int(j == 4)) + q * (j - 4))
            - 2 * mp.pi * 512 * (int(j == 3) - int(j == 4))
            for j in range(9)
        ]
        for sign in (-1, 1):
            for i, gap in enumerate(report.initial_ordinary_gap_bounds):
                _contains(mp, gap, lift[i + 1] - lift[i] + sign * error)
            _contains(
                mp,
                report.initial_closing_coordinate_bounds,
                lift[8] - lift[0] - 3 * mp.pi + sign * error,
            )


def test_acute_tube_full_cell_curvature_and_storage_are_separately_admitted(report, mp):
    error = _mp(mp, report.mixed_field_error_upper_bound)
    z = _mp(mp, report.uniform_scaled_form_norm_upper_bound)
    cap = max(mp.pi - mp.mpf(7) / 4, mp.mpf(13) / 10 + 8 * error) + 128 * error + z
    _upper_close(mp, report.stage_two_actual_gap_absolute_upper_bound, cap)
    assert cap < mp.mpf(7) / 5
    _upper_close(
        mp,
        report.stage_two_initial_distance_squared_upper_bound,
        45 * (mp.mpf(7) / 5 + 2 * mp.pi / 9) ** 2,
    )
    inner = mp.cos(mp.mpf(3) / 2) - mp.mpf(2) / 243
    _lower_close(mp, report.inner_acute_curvature_lower_bound, inner)
    assert inner > 0
    # These angular intervals prove the harmonic signs on the whole outer strip.
    assert 4 * mp.pi < mp.mpf(27) / 2 < 9 * mp.pi / 2
    assert 17 * mp.pi / 2 < 27 < 9 * mp.pi
    assert report.uniform_law_bounds_certified
    radius = 15 * mp.exp(-4) + 32 * error + z
    _upper_close(mp, report.actual_phase_distance_at_capture_upper_bound, radius)
    gamma = 1 / (1023 * mp.pi)
    storage = (
        9 * (1 - mp.cos(2 * mp.pi / 9))
        + (1 + mp.mpf(2) / 243) * radius**2
        + z**2 / gamma**2
    )
    _upper_close(mp, report.total_storage_upper_bound, storage)
    barrier = 9 - 8 * mp.cos(3 * mp.pi / 16)
    _lower_close(mp, report.sector_boundary_storage_lower_bound, barrier)
    assert len(report.sector_boundary_face_lower_bounds) == 18
    assert (
        min(report.sector_boundary_face_lower_bounds)
        == report.sector_boundary_storage_lower_bound
    )
    assert report.total_storage_upper_bound < Q(2182, 1000)
    assert report.capture_margin > Q(166, 1000)
    tau = 200 + 136 / gamma**2
    _contains(mp, report.scaled_time_deadline_bounds, tau)
    _contains(mp, report.original_time_deadline_bounds, tau * 1024 / 1023)


def test_upper_deformation_supplies_uniform_bounds_without_interval_sampling(
    report, endpoints
):
    sine, deformed = endpoints
    for name in (
        "current_absolute_upper_bound",
        "phase_hessian_upper_bound",
        "mixed_field_error_upper_bound",
        "initial_mixed_phase_error_norm_upper_bound",
        "total_storage_upper_bound",
        "common_initial_storage_upper_bound",
    ):
        assert getattr(sine, name) < getattr(deformed, name) == getattr(report, name)
    assert report.capture_margin == deformed.capture_margin < sine.capture_margin
    assert (
        report.stage_one_passage_speed_lower_bound
        < sine.stage_one_passage_speed_lower_bound
    )


def test_subgrid_law_and_source_primitives_remain_exact():
    tiny = Q(1, 2**500)
    result = _assess(deformation_interval=(tiny, tiny), phase_error_bound=tiny)
    assert result.deformation_interval == (tiny, tiny)
    assert result.source.phase_error_bound == tiny
    assert result.current_deviation_upper_bound == tiny / 1458
    assert result.phase_hessian_upper_bound - 1 == 2 * tiny / 243
    correction = 6 * tiny**5
    assert result.initial_potential_deformation_upper_bound == correction > 0
    assert (
        result.common_initial_storage_upper_bound
        - result.source.initial_form_storage_upper_bound
        - result.source.quadratic_initial_phase_storage_upper_bound
    ) == correction


def test_strict_zero_margin_does_not_certify_capture(monkeypatch, report):
    bound = (
        min(edge.lo for edge in report.initial_ordinary_gap_bounds)
        - report.initial_closing_coordinate_bounds.hi / 2
    )
    monkeypatch.setattr(owner, "_GAP_LOWER", bound)
    result = _assess()
    assert result.initial_coupled_lower_margin == 0
    assert not result.capture_certified
    assert "initial_coupled_gap_lower_bound" in result.unresolved_conditions


@pytest.mark.parametrize("errors", [(0, 0), (0, ERROR), (ERROR, 0)])
def test_nominal_and_lower_dimensional_sources_keep_the_uniform_conclusion(errors):
    result = _assess(form_error_bound=errors[0], phase_error_bound=errors[1])
    assert result.capture_certified and result.captured_winding == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"parameter_interval": (0, Q(1, 8))},
        {"parameter_interval": (Q(63, 64), 1)},
        {"parameter_interval": (0, 1)},
        {"form_error_bound": 100},
        {"phase_error_bound": Q(1, 10)},
    ],
)
def test_failed_sufficient_conditions_do_not_claim_nonformation(changes):
    result = _assess(**changes)
    assert not result.capture_certified
    assert result.captured_winding is None
    assert result.status == "unavailable"
    assert result.unresolved_conditions


@pytest.mark.parametrize(
    "changes",
    [
        {"deformation_interval": (True, 1)},
        {"deformation_interval": (0, 1 + Q(1, 2**300))},
        {"deformation_interval": (-Q(1, 2**300), 1)},
        {"deformation_interval": (1, 0)},
        {"deformation_interval": (0, float("nan"))},
        {"deformation_interval": (0, "1")},
        {"deformation_interval": (0, 1, 1)},
        {"parameter_interval": (True, 1)},
        {"form_error_bound": True},
        {"phase_error_bound": float("inf")},
        {"phase_error_bound": -1},
    ],
)
def test_raw_admission_precedes_source_geometry(monkeypatch, changes):
    def forbidden():
        pytest.fail("invalid raw input reached source construction")

    monkeypatch.setattr(source_owner, "_path_construction", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**changes)


def test_deformation_iterator_is_bounded():
    def endless():
        for _ in range(4):
            yield 0
        pytest.fail("invalid interval was drained")

    with pytest.raises(ValueError):
        _assess(deformation_interval=endless())


def test_no_incoming_certificate_or_supplied_horizon_is_accepted(report):
    with pytest.raises(TypeError):
        _assess(source=report.source)
    with pytest.raises(TypeError):
        _assess(scaled_time=300)
