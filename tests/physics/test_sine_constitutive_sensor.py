"""Independent calibration, observation-domain and sensor-nuisance controls."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_constitutive_sensor as owner


@pytest.fixture(scope="module", autouse=True)
def no_scientific_response_execution():
    with forbid_sine_regeneration():
        yield


def _assess(**changes):
    inputs = dict(
        form_error_bound=Q(1, 10**10),
        phase_error_bound=Q(1, 10**10),
        calibration_radius=Q(1, 10**8),
        gain_lower_bound=Q(1, 2),
        curvature_upper_bound=Q(1, 10**35),
        per_reading_error_bound=Q(1, 10**54),
    )
    return owner._assess_calibrated_constitutive_observation(**(inputs | changes))


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_paired_transport_and_absolute_sensor_range_have_independent_bounds(report):
    amplitude, time, residual = Q(7, 10000), Q(1, 10**6), Q(1, 10**70)
    growth = 1 / (1 - 3 * time)
    increment = 4 * amplitude * time
    assert amplitude * (growth - 1) < increment
    assert residual * growth < 2 * residual
    assert report.branch_increment_upper_bound == increment
    assert report.baseline_absolute_upper_bound == 2 * residual
    assert report.raw_reading_radius_upper_bound == 2 * residual + increment
    assert report.mixed_argument_radius_upper_bound == 2 * residual + 2 * increment
    assert report.mixed_argument_radius_upper_bound < Q(1, 10**8)
    assert report.raw_comparison.per_reading_error_bound == 0
    assert all(
        row.handoff.source.form_error_bound == Q(1, 10**10)
        and row.handoff.source.phase_error_bound == Q(1, 10**10)
        for row in report.raw_comparison.observations
    )


@pytest.mark.parametrize(
    "u,v,mixed", [(Q(2), Q(3), Q(-1)), (Q(-2), Q(3), Q(1)), (Q(2), Q(-3), Q(-1))]
)
def test_oriented_mixed_identity_against_exact_cubic_sensor(u, v, mixed):
    # A polynomial gives independent exact integrals, including signed limits.
    q, offset, a, b, c = Q(1, 3), Q(17, 5), Q(2), Q(3, 7), Q(-1, 13)

    def h(z):
        return offset + a * z + b * z**2 + c * z**3

    s = q + u + v
    recorded = h(s + mixed) - h(q + u) - h(q + v) + h(q)
    averaged_gain = a + b * (2 * s + mixed) + c * (3 * s**2 + 3 * s * mixed + mixed**2)
    rectangle = 2 * b * u * v + 6 * c * (q * u * v + u**2 * v / 2 + u * v**2 / 2)
    assert recorded == averaged_gain * mixed + rectangle


def test_calibration_separates_all_admitted_sources_and_sensor_errors(report):
    assert report.status == "certified_calibrated_mixed_separation"
    assert report.uniform_separation_certified and report.opposite_signs_certified
    assert report.fixed_common_map_certified
    assert not report.unresolved_conditions
    independent_curvature_error = Q(1, 10**35) * (4 * Q(7, 10000) * Q(1, 10**6)) ** 2
    assert report.curvature_error_upper_bound == independent_curvature_error
    for raw, sensed in zip(report.raw_comparison.observations, report.observations):
        assert raw.handoff.status == "certified_constitutive_handoff"
        raw_margin = (
            -raw.actual_response_bounds[1]
            if raw.deformation == 0
            else raw.actual_response_bounds[0]
        )
        expected = raw_margin / 2 - independent_curvature_error - Q(4, 10**54)
        assert sensed.recorded_signed_margin_lower_bound == expected
        assert expected > Q(3, 10**52)
    left, right = (
        row.recorded_signed_margin_lower_bound for row in report.observations
    )
    assert -left < report.classification_threshold < right
    assert report.uniform_mixed_separation_lower_bound == left + right
    assert report.uniform_record_distance_lower_bound == (left + right) / 4
    assert report.fixed_common_map_record_distance_lower_bound == (
        report.raw_comparison.record_separation_lower_bound / 8 - Q(2, 10**54)
    )
    assert report.fixed_common_map_record_distance_lower_bound > Q(1, 10**52)


def test_a_nonzero_threshold_can_separate_without_opposite_sign_guarantees():
    result = _assess(curvature_upper_bound=Q(1, 10**33))
    left, right = result.observations
    assert left.recorded_signed_margin_lower_bound > 0
    assert right.recorded_signed_margin_lower_bound < 0
    assert not result.opposite_signs_certified
    assert result.uniform_separation_certified
    assert -left.recorded_signed_margin_lower_bound < result.classification_threshold
    assert (
        result.classification_threshold < right.recorded_signed_margin_lower_bound < 0
    )


def test_exact_zero_mixed_gap_is_not_strict_separation(report):
    curvature = (
        report.gain_lower_bound * report.raw_comparison.record_separation_lower_bound
        - 8 * report.per_reading_error_bound
    ) / (2 * report.branch_increment_upper_bound**2)
    result = _assess(curvature_upper_bound=curvature)
    assert result.uniform_mixed_separation_lower_bound == 0
    assert not result.uniform_separation_certified
    assert result.classification_threshold is None
    assert result.fixed_common_map_certified
    assert result.status == "fixed_common_sensor_separation_only"


def test_exact_zero_sign_margin_keeps_the_nonzero_threshold_available(report):
    positive_raw_margin = report.observations[1].raw_signed_margin
    curvature = (
        report.gain_lower_bound * positive_raw_margin
        - 4 * report.per_reading_error_bound
    ) / report.branch_increment_upper_bound**2
    result = _assess(curvature_upper_bound=curvature)
    assert result.observations[1].recorded_signed_margin_lower_bound == 0
    assert not result.observations[1].opposite_sign_certified
    assert not result.opposite_signs_certified
    assert result.uniform_separation_certified
    assert result.classification_threshold < 0


def test_exact_zero_fixed_sensor_distance_is_not_strict_separation(report):
    reading_error = (
        report.gain_lower_bound * report.raw_comparison.record_separation_lower_bound
    ) / 8
    result = _assess(curvature_upper_bound=0, per_reading_error_bound=reading_error)
    assert result.fixed_common_map_record_distance_lower_bound == 0
    assert result.uniform_mixed_separation_lower_bound == 0
    assert not result.fixed_common_map_certified
    assert not result.uniform_separation_certified
    assert result.classification_threshold is None
    assert result.status == "unavailable"
    assert (
        "fixed_common_sensor_record_distance_not_certified"
        in result.unresolved_conditions
    )


def test_full_record_calibration_needs_only_the_raw_domain(report):
    result = _assess(calibration_radius=report.raw_reading_radius_upper_bound)
    assert result.raw_domain_margin == 0
    assert result.raw_domain_admitted
    assert result.mixed_domain_margin < 0
    assert not result.mixed_domain_admitted
    assert result.fixed_common_map_certified
    assert not result.uniform_separation_certified
    assert (
        "mixed_argument_envelope_not_enclosed_by_calibrated_domain"
        in result.unresolved_conditions
    )
    boundary = _assess(calibration_radius=report.mixed_argument_radius_upper_bound)
    assert boundary.mixed_domain_margin == 0
    assert boundary.uniform_separation_certified
    insufficient = _assess(calibration_radius=report.raw_reading_radius_upper_bound / 2)
    assert insufficient.status == "unavailable"
    assert (
        "raw_reading_envelope_not_enclosed_by_calibrated_domain"
        in insufficient.unresolved_conditions
    )


def test_quadratic_common_sensor_changes_actual_mixed_signs_but_not_full_records():
    result = _assess(curvature_upper_bound=2)
    assert result.status == "fixed_common_sensor_separation_only"
    assert not result.uniform_separation_certified
    assert result.fixed_common_map_certified
    # Independently rebuild the proof's actual single-input and quadratic
    # bounds. These are inequalities for the full admitted source families,
    # not a sampled or reset source disguised as an experimental response.
    m, time, eps = Q(7, 10000), Q(1, 10**6), Q(1, 10**70)
    minimum_increment = m * time / 4 - 6 * (m + 2 * eps) * time**2
    assert minimum_increment > m * time / 8
    maximum_gain = 1 + 2 * Q(1, 10**8)
    assert 1 - 2 * Q(1, 10**8) > Q(1, 2)
    for row in result.raw_comparison.observations:
        raw_size = max(map(abs, row.actual_response_bounds))
        assert raw_size < Q(1, 10**48)
        measured_lower = (
            2 * minimum_increment**2 - maximum_gain * raw_size - Q(4, 10**54)
        )
        assert measured_lower > Q(1, 10**20)
    # Finite differences of this single fixed map retain a lower gain.
    z0, z1 = Q(-1, 10**9), Q(2, 10**9)
    assert (z1 + z1**2) - (z0 + z0**2) > (z1 - z0) / 2


@pytest.mark.parametrize("deformation", [0, 1])
def test_initial_mediator_response_uses_the_actual_degree_and_unchanged_phase(
    deformation,
):
    mp = mpmath.mp.clone()
    mp.dps = 110
    m = mp.mpf(7) / 10000
    gamma = 1 / (1023 * mp.pi)
    x = [mp.mpf(i - 13) / 10**72 for i in range(27)]
    phase = [
        (2 * mp.pi / 9) * (i % 9 - 4) + mp.mpf((i % 9) - 4) / 10**72 for i in range(27)
    ]
    neighbors = (4, 12, 14, 22)

    def mediator_rate(form):
        total = mp.mpf(0)
        for j in neighbors:
            angle = phase[j] - phase[13]
            current = (
                mp.sin(angle)
                + deformation * (mp.sin(9 * angle) - mp.sin(18 * angle) / 2) / 2187
            )
            total += form[j] - form[13] + gamma * current
        return total / len(neighbors)

    for port in (4, 22):
        changed = list(x)
        changed[port] += m
        assert abs(mediator_rate(changed) - mediator_rate(x) - m / 4) < mp.mpf("1e-100")


def test_sensor_unit_change_scales_sensor_bounds_but_not_structural_inputs(report):
    scale = Q(7, 3)
    result = _assess(
        gain_lower_bound=scale * report.gain_lower_bound,
        curvature_upper_bound=scale * report.curvature_upper_bound,
        per_reading_error_bound=scale * report.per_reading_error_bound,
    )
    assert (
        result.mixed_argument_radius_upper_bound
        == report.mixed_argument_radius_upper_bound
    )
    assert (
        result.uniform_mixed_separation_lower_bound
        == scale * report.uniform_mixed_separation_lower_bound
    )
    assert result.classification_threshold == scale * report.classification_threshold
    assert (
        result.fixed_common_map_record_distance_lower_bound
        == scale * report.fixed_common_map_record_distance_lower_bound
    )


def test_exact_sensor_errors_are_not_added_again_in_raw_form_units(report):
    tiny = Q(1, 2**600)
    result = _assess(per_reading_error_bound=tiny)
    assert result.per_reading_error_bound == tiny
    assert result.mixed_reading_error_upper_bound == 4 * tiny
    assert result.raw_comparison == report.raw_comparison


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"phase_error_bound": Q(1, 10)}, "actual_source_and_response_not_certified"),
        ({"gain_lower_bound": 0}, "strictly_positive_sensor_gain_not_supplied"),
        (
            {"per_reading_error_bound": Q(1, 10**40)},
            "fixed_common_sensor_record_distance_not_certified",
        ),
    ],
)
def test_valid_but_insufficient_premises_are_unavailable(changes, reason):
    result = _assess(**changes)
    assert result.status == "unavailable"
    assert not result.uniform_separation_certified
    assert not result.fixed_common_map_certified
    assert not result.opposite_signs_certified
    assert reason in result.unresolved_conditions


@pytest.mark.parametrize(
    "field,bad",
    [
        ("form_error_bound", True),
        ("phase_error_bound", "0"),
        ("calibration_radius", False),
        ("calibration_radius", 0),
        ("calibration_radius", -1),
        ("gain_lower_bound", float("nan")),
        ("gain_lower_bound", -1),
        ("curvature_upper_bound", True),
        ("curvature_upper_bound", -Q(1, 2**600)),
        ("per_reading_error_bound", float("inf")),
        ("per_reading_error_bound", -1),
    ],
)
def test_raw_calibration_admission_precedes_source_reconstruction(
    monkeypatch, field, bad
):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached source reconstruction")

    monkeypatch.setattr(owner, "_assess_carried_constitutive_discrimination", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


def test_report_or_response_fitted_sensor_cannot_replace_raw_calibration(report):
    for extra in (
        {"source": report.raw_comparison},
        {"sensor": lambda value: value},
        {"horizon": Q(1, 8)},
    ):
        with pytest.raises(TypeError):
            _assess(**extra)
