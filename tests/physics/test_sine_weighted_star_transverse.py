"""Independent fine-row reduction and scalar bounds, without a flow solver."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.physics.relational_sine_weighted_transverse import (
    assess_sine_weighted_star_transverse_band,
)
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.types import Glyph
from tnfr.utils.io import json_loads

INTERNAL_EDGES = ((0, 1), (0, 2), (0, 3))
CROSS_EDGES = tuple((i, 4) for i in range(4))
SPATIAL_MODES = (
    (1, (0, 1, -1, 0, 0)),
    (1, (0, 1, 1, -2, 0)),
    (4, (3, -1, -1, -1, 0)),
)


def _laplacian(edges, weights):
    matrix = [[Q(0) for _ in range(5)] for _ in range(5)]
    for (i, j), weight in zip(edges, weights):
        matrix[i][i] += weight
        matrix[j][j] += weight
        matrix[i][j] -= weight
        matrix[j][i] -= weight
    return tuple(map(tuple, matrix))


def _apply(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _fine_jacobian_times_pi(scale, contact_cosine):
    """Differentiate both original node rows, using each actual edge channel."""
    phase_laplacian = _laplacian(
        INTERNAL_EDGES + CROSS_EDGES, (1,) * 3 + (contact_cosine,) * 4
    )
    form_laplacian = _laplacian(INTERNAL_EDGES + CROSS_EDGES, (3,) * 3 + (2,) * 4)
    mobility = (scale,) * 4 + (Q(2),)
    top = tuple(
        (Q(0),) * 5 + tuple(-mobility[i] * value for value in row)
        for i, row in enumerate(phase_laplacian)
    )
    bottom = tuple(
        tuple(mobility[i] * value for value in row) + (Q(0),) * 5
        for i, row in enumerate(form_laplacian)
    )
    return top + bottom


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, value):
    exact_decimal = Q(mp.nstr(value, 150))
    assert interval.lo <= exact_decimal <= interval.hi


@pytest.fixture(scope="module")
def report():
    return assess_sine_weighted_star_transverse_band()


def test_declared_law_dimensions_and_exact_spatial_modes(report):
    assert report.capacity_scale_bounds == (Q(23, 25), 1)
    assert report.amplitude_bound == Q(1, 4)
    assert report.edges == INTERNAL_EDGES + CROSS_EDGES
    assert report.conductances == (3,) * 3 + (2,) * 4
    assert report.bare_degrees == (4, 2, 2, 2, 4)
    assert report.transverse_dimension == 6
    assert report.collective_dimension == 4
    assert report.original_clock == "t"
    assert tuple(
        (mode.laplacian_eigenvalue, mode.multiplicity) for mode in report.modes
    ) == ((1, 2), (4, 1))
    assert 2 * sum(mode.multiplicity for mode in report.modes) == 6
    assert report.linear_transverse_stability_certified
    assert report.unavailable_reasons == ()
    assert report.equilibrium_transverse_stability_certified
    internal = _laplacian(INTERNAL_EDGES, (1,) * 3)
    # Three mutually orthogonal nonzero vectors span the full sum-zero A
    # space. Its orthogonal complement is the two collective spatial rows.
    basis = [vector for _, vector in SPATIAL_MODES] + [(1, 1, 1, 1, 0), (0, 0, 0, 0, 1)]
    for i, left in enumerate(basis):
        assert sum(value**2 for value in left) > 0
        for right in basis[i + 1 :]:
            assert sum(a * b for a, b in zip(left, right)) == 0
    for eigenvalue, vector in SPATIAL_MODES:
        assert _apply(internal, vector) == tuple(eigenvalue * value for value in vector)
        assert vector[4] == sum(vector[:4]) == 0
    with pytest.raises(FrozenInstanceError):
        report.status = "unavailable"


@pytest.mark.parametrize("scale", (Q(23, 25), Q(1), Q(5, 4)))
@pytest.mark.parametrize("contact_cosine", (Q(-1), Q(0), Q(7, 8), Q(1)))
def test_complete_fine_jacobian_splits_without_losing_modes(
    report, scale, contact_cosine
):
    matrix = _fine_jacobian_times_pi(scale, contact_cosine)
    for eigenvalue, vector in SPATIAL_MODES:
        assert eigenvalue in tuple(mode.laplacian_eigenvalue for mode in report.modes)
        x_only, theta_only = vector + (0,) * 5, (0,) * 5 + vector
        assert _apply(matrix, x_only) == (0,) * 5 + tuple(
            scale * (3 * eigenvalue + 2) * value for value in vector
        )
        assert (
            _apply(matrix, theta_only)
            == tuple(-scale * (eigenvalue + contact_cosine) * value for value in vector)
            + (0,) * 5
        )
    # Independent collective perturbations stay collective. Uniform form and
    # phase shifts give the two exact neutral directions, outside this proof.
    for x_a, x_b, theta_a, theta_b in ((2, -1, 3, 4), (1, 1, 0, 0), (0, 0, 1, 1)):
        vector = (x_a,) * 4 + (x_b,) + (theta_a,) * 4 + (theta_b,)
        expected = (
            (scale * contact_cosine * (theta_b - theta_a),) * 4
            + (8 * contact_cosine * (theta_a - theta_b),)
            + (2 * scale * (x_a - x_b),) * 4
            + (16 * (x_b - x_a),)
        )
        assert _apply(matrix, vector) == expected


def test_normalized_modal_coefficient_follows_both_original_clock_rows(report):
    for scale in report.capacity_scale_bounds:
        point = assess_sine_weighted_star_transverse_band(
            capacity_scale_bounds=(scale, scale), amplitude_bound=Q(1, 4)
        )
        for mode in point.modes:
            eigenvalue = mode.laplacian_eigenvalue
            for cosine in (Q(7, 8), Q(15, 16), Q(1)):
                # pi^2 times the original oscillator coefficient divided by
                # pi^2 times the squared pendulum clock rate.
                fine_coefficient = (
                    scale**2 * (3 * eigenvalue + 2) * (eigenvalue + cosine)
                )
                clock_squared = 2 * (scale + 8) ** 2
                exact_alpha = scale**2 * (3 * eigenvalue + 2) / clock_squared
                assert mode.alpha_bounds.contains(exact_alpha)
                assert (mode.alpha_bounds * (eigenvalue + cosine)).contains(
                    fine_coefficient / clock_squared
                )


def test_scalar_period_frequency_rotation_and_error_bounds_independently(report):
    # Scalar special-function values check the inequalities. They do not
    # solve a trajectory or construct an approximate monodromy matrix.
    factor = resolve_runtime_operator_factors({}, Glyph.SHA, {})["SHA_vf_factor"]
    scales = (Q(23, 25), Q.from_float(factor), Q(1))
    amplitudes = (Q(0), Q(1, 8), Q(1, 4))
    with mp.workdps(180):
        for scale in scales:
            s = _mp(scale)
            clock_rate = mp.sqrt(2) * (s + 8) / mp.pi
            _contains(report.normalized_clock_rate_bounds, clock_rate)
            for amplitude in amplitudes:
                k = _mp(amplitude)
                period = 4 * mp.ellipk(k**2)
                _contains(report.normalized_period_bounds, period)
                _contains(report.original_period_bounds, period / clock_rate)
                for mode in report.modes:
                    eigenvalue = mode.laplacian_eigenvalue
                    alpha = s**2 * (3 * eigenvalue + 2) / (2 * (s + 8) ** 2)
                    frequency = mp.sqrt(alpha * (eigenvalue + 1))
                    _contains(mode.alpha_bounds, alpha)
                    _contains(mode.frequency_bounds, frequency)
                    _contains(mode.rotation_angle_bounds, frequency * period)
                    _contains(
                        mode.reference_half_trace_bounds, mp.cos(frequency * period)
                    )
                    bound = 2 * alpha * k**2 * period / frequency
                    assert bound <= _mp(mode.perturbation_integral_upper_bound)
        for mode in report.modes:
            budget = _mp(mode.perturbation_integral_upper_bound)
            assert 0 < budget <= 1
            assert mp.expm1(budget) <= _mp(mode.rotation_error_upper_bound)
            assert mode.monodromy_determinant == 1
            assert mode.trace_certified and mode.linear_stability_certified
            assert mode.strict_trace_margin > 0
            assert -1 < mode.half_trace_bounds.lo <= mode.half_trace_bounds.hi < 1
            assert mode.strict_trace_margin == 1 - mode.half_trace_bounds.abs_max


def test_capacity_and_amplitude_correlations_are_bounded_before_materialization(report):
    for mode in report.modes:
        eigenvalue = mode.laplacian_eigenvalue
        exact_endpoints = tuple(
            Q(3 * eigenvalue + 2, 2) * (scale / (scale + 8)) ** 2
            for scale in report.capacity_scale_bounds
        )
        assert mode.alpha_bounds == I(*exact_endpoints)
        assert mode.perturbation_integral_upper_bound == (
            2
            * mode.frequency_bounds.hi
            * report.amplitude_bound**2
            * report.normalized_period_bounds.hi
            / (eigenvalue + 1)
        )
        assert mode.half_trace_bounds == (
            mode.reference_half_trace_bounds
            + I(-mode.rotation_error_upper_bound, mode.rotation_error_upper_bound)
        )


def test_default_sha_is_inside_the_fixed_band_not_an_executed_event(report):
    factor = resolve_runtime_operator_factors({}, Glyph.SHA, {})["SHA_vf_factor"]
    represented = Q.from_float(factor)
    assert (
        report.capacity_scale_bounds[0] < represented < report.capacity_scale_bounds[1]
    )
    assert (
        tuple(Q.from_float(value * factor) / value for value in (4, 2, 2, 2))
        == (represented,) * 4
    )
    after = assess_sine_weighted_star_transverse_band(
        capacity_scale_bounds=(represented, represented), amplitude_bound=Q(1, 4)
    )
    assert after.linear_transverse_stability_certified
    assert all(mode.strict_trace_margin > 0 for mode in after.modes)
    # A capacity reset changes the clock and every modal coefficient. The
    # point certificate is freshly evaluated under those changed parameters.
    at_one = assess_sine_weighted_star_transverse_band(
        capacity_scale_bounds=(1, 1), amplitude_bound=Q(1, 4)
    )
    assert (
        after.normalized_clock_rate_bounds.hi < at_one.normalized_clock_rate_bounds.lo
    )
    assert all(
        new.alpha_bounds.hi < old.alpha_bounds.lo
        for new, old in zip(after.modes, at_one.modes)
    )


def test_wide_valid_band_abstains_without_claiming_instability():
    wide = assess_sine_weighted_star_transverse_band(
        capacity_scale_bounds=(Q(23, 25), 1), amplitude_bound=Q(9, 10)
    )
    assert not wide.linear_transverse_stability_certified
    assert wide.status == "unavailable"
    assert wide.unavailable_reasons
    assert any(
        mode.perturbation_integral_upper_bound > 1
        and mode.rotation_error_upper_bound is None
        for mode in wide.modes
    )
    assert wide.equilibrium_transverse_stability_certified


def test_positive_equilibrium_oscillators_do_not_need_a_resolved_return_trace():
    equilibrium = assess_sine_weighted_star_transverse_band(
        capacity_scale_bounds=(Q(1, 100), 100), amplitude_bound=0
    )
    assert equilibrium.linear_transverse_stability_certified
    assert equilibrium.equilibrium_transverse_stability_certified
    assert equilibrium.unavailable_reasons == ()
    assert any(not mode.trace_certified for mode in equilibrium.modes)
    assert all(mode.linear_stability_certified for mode in equilibrium.modes)
    assert (
        equilibrium.period_scope == "nonzero_libration_period_and_zero_amplitude_limit"
    )
    assert all(mode.rotation_error_upper_bound == 0 for mode in equilibrium.modes)


@pytest.mark.parametrize("scale", (Q(1, 2**150), Q(1, 2**1000)))
def test_exact_positive_but_unresolved_frequency_is_not_rounded_into_a_certificate(
    scale,
):
    result = assess_sine_weighted_star_transverse_band(
        capacity_scale_bounds=(scale, scale), amplitude_bound=Q(1, 4)
    )
    assert result.capacity_scale_bounds == (scale, scale)
    assert result.status == "unavailable"
    assert all(mode.frequency_bounds.lo == 0 for mode in result.modes)
    assert all(mode.half_trace_bounds is None for mode in result.modes)
    assert all(
        "positive_rotation_frequency_unresolved_at_interval_precision"
        in mode.unavailable_reasons
        for mode in result.modes
    )


def test_exact_near_separatrix_amplitude_retains_unavailable_denominator():
    amplitude = 1 - Q(1, 2**200)
    result = assess_sine_weighted_star_transverse_band(amplitude_bound=amplitude)
    assert result.amplitude_bound == amplitude
    assert result.status == "unavailable"
    assert result.normalized_period_bounds is None
    assert result.original_period_bounds is None
    assert all(mode.rotation_error_upper_bound is None for mode in result.modes)


@pytest.mark.parametrize(
    "bounds",
    (
        (),
        (1,),
        (1, 1, 1),
        (1, 0),
        (0, 1),
        (-1, 1),
        (True, 1),
        (1, True),
        (float("nan"), 1),
        (1, float("inf")),
        (complex(1, 0), 1),
        {Q(23, 25), 1},
        "1,1",
    ),
)
def test_capacity_band_admits_ordered_finite_positive_physical_scalars(bounds):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_transverse_band(capacity_scale_bounds=bounds)


@pytest.mark.parametrize(
    "amplitude", (True, False, -1, 1, 2, float("nan"), float("inf"), complex(0, 1))
)
def test_amplitude_admission_does_not_coerce_invalid_values(amplitude):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_transverse_band(amplitude_bound=amplitude)


def test_represented_float_inputs_and_exact_rationals_keep_their_distinct_values():
    represented = assess_sine_weighted_star_transverse_band(
        capacity_scale_bounds=(0.92, 1.0), amplitude_bound=0.1
    )
    assert represented.capacity_scale_bounds == (Q.from_float(0.92), 1)
    assert represented.amplitude_bound == Q.from_float(0.1)
    assert represented.amplitude_bound != Q(1, 10)
    exact = assess_sine_weighted_star_transverse_band(amplitude_bound=Q(1, 10))
    assert exact.amplitude_bound == Q(1, 10)


def test_export_rebuilds_derived_fields_and_readmits_primitive_parameters(report):
    poisoned = replace(
        report,
        modes=(),
        edges=(),
        conductances=(),
        normalized_period_bounds=I(-1),
        original_period_bounds=I(-2),
        normalized_clock_rate_bounds=I(999),
        linear_transverse_stability_certified=False,
        equilibrium_transverse_stability_certified=False,
        status="excluded",
    )
    assert poisoned.to_dict() == report.to_dict()
    with pytest.raises((TypeError, ValueError)):
        replace(report, amplitude_bound=True).to_dict()
    with pytest.raises((TypeError, ValueError)):
        replace(report, capacity_scale_bounds=(0, 1)).to_dict()
    wide = replace(report, amplitude_bound=Q(9, 10)).to_dict()
    assert wide["report"]["status"] == "unavailable"
    assert wide["report"]["linear_transverse_stability_certified"] is False


def test_sdk_adapter_and_atomic_json_export_rebuild_then_preserve_exact_parameters(
    report, tmp_path
):
    poisoned = replace(
        report,
        modes=(),
        normalized_period_bounds=I(-1),
        linear_transverse_stability_certified=False,
        status="excluded",
    )
    exported = relational_report_to_dict(poisoned)
    assert exported == {
        "schema": "tnfr.relational-report.v1",
        "report_type": type(report).__name__,
        "report": report.to_dict()["report"],
    }
    destination = tmp_path / "weighted-star-transverse.json"
    export_to_json(exported, destination)
    restored = json_loads(destination.read_text(encoding="utf-8"))
    assert restored == exported
    assert restored["report"]["capacity_scale_bounds"] == [
        {"numerator": 23, "denominator": 25},
        {"numerator": 1, "denominator": 1},
    ]
    assert restored["report"]["amplitude_bound"] == {"numerator": 1, "denominator": 4}
    assert restored["report"]["status"] == "certified_linear_transverse_stability"
    retained_bytes = destination.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        export_to_json(
            relational_report_to_dict(replace(report, amplitude_bound=True)),
            destination,
        )
    assert destination.read_bytes() == retained_bytes
