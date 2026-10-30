"""Static necessary-condition controls; no scientific record is evaluated."""

from fractions import Fraction as Q
from math import cos, sin

import pytest

from tnfr.mathematics._exact_linear_algebra import exact_matrix_product
from tnfr.physics import _organized_law_joint_bounds as joint


@pytest.fixture(scope="module")
def report():
    return joint._bound_organized_law_joint_phases()


def test_actual_schedule_source_errors_and_strict_margins(report):
    assert report.plateau_duration == Q(1, 10**5)
    assert 2 * Q(1, 10**14) < report.sample_difference_upper < Q(201, 10**16)
    assert max(report.carrier_form_bounds) < Q(13, 1000)
    assert max(report.carrier_phase_drift_bounds) < Q(17, 10**6)
    assert max(report.plateau_second_derivative_bounds) < Q(3, 200)
    assert report.initial_reference_current_error < Q(1, 7500)
    assert report.sine_quadrature_difference_upper < Q(3, 50)
    assert report.alternative_sine_quadrature_error < Q(3, 26)
    assert report.initial_cosine_lower > Q(1, 6)
    assert report.same_winding_margin > 0
    assert report.initial_current_divergence_upper < Q(41, 10**10)
    assert report.fundamental_sine_radius_ratio < Q(13, 4)
    assert report.initial_fundamental_radius_ratio < Q(39, 2)
    assert report.reference_fundamental_radius_ratio < 20
    assert report.admitted
    assert "not_a_separation_verdict" in report.scope


def test_c5_norms_and_corrected_endpoint_coefficients(report):
    incidence = tuple(
        tuple(Q(i == (j + 1) % 5) - Q(i == j) for j in range(5)) for i in range(5)
    )
    laplacian = exact_matrix_product(incidence, tuple(zip(*incidence)))
    generator = tuple(tuple(value / 4 for value in row) for row in laplacian)
    assert all(sum(abs(v) for v in row) == 1 for row in generator)
    edge_rows = exact_matrix_product(tuple(zip(*incidence)), generator)
    assert all(sum(abs(v) for v in row) == 2 for row in edge_rows)
    h = report.plateau_duration
    for i, row in enumerate(generator):
        coefficients = tuple(-Q(i == j) + h * value / 2 for j, value in enumerate(row))
        coefficients += tuple(Q(i == j) + h * value / 2 for j, value in enumerate(row))
        assert sum(abs(v) for v in coefficients) == 2 + h / 2


def test_diffusion_trapezoid_has_a_cubic_error_before_dividing_by_h():
    # For y(s)=s**2 the trapezoid error is h**3/6. Its second derivative
    # is 2, so the h**3/12 formula is exact, independently of any flow.
    for h in (Q(1), Q(1, 10**5), Q(3, 7)):
        integral = h**3 / 3
        trapezoid = h * h**2 / 2
        assert trapezoid - integral == 2 * h**3 / 12
        assert (trapezoid - integral) / h == h**2 / 6


def test_receiver_three_level_contrast_removes_its_arbitrary_baseline():
    # Independent trigonometric identity, with a synthetic scalar gap.
    angle, epsilon = 0.37, 0.1
    for baseline in (-7.0, 0.0, 11.0):
        currents = tuple(baseline - sin(angle - q) for q in (0, epsilon, -epsilon))
        contrast = (currents[1] + currents[2] - 2 * currents[0]) / (
            2 * (1 - cos(epsilon))
        )
        assert contrast == pytest.approx(sin(angle), abs=1e-12)
    # Four absolute unit coefficients, divided by 2*(1-cos(epsilon)).
    assert sum(abs(value) for value in (-2, 1, 1)) / 2 == 2


def test_c5_current_range_requires_equal_winding_for_an_absolute_bound():
    values = (Q(-2), Q(-1), Q(0), Q(-1), Q(-2))
    adjacent = tuple(values[(i + 1) % 5] - values[i] for i in range(5))
    assert max(abs(value) for value in adjacent) == 1
    assert max(values) - min(values) == 2
    # A constant edge current is divergence-free; without a sign anchor
    # its amplitude cannot be bounded by node current differences.
    shifted = tuple(value + 100 for value in values)
    assert max(shifted) - min(shifted) == 2
    assert min(shifted) > 0
    assert max(abs(value) for value in shifted) > 2


@pytest.mark.parametrize("a,b", ((-1.4, -1.39), (-0.2, 0.3), (1.3, 1.4)))
def test_acute_phasor_identity_controls_both_fundamental_components(a, b, report):
    phasor_squared = (sin(a) - sin(b)) ** 2 + (cos(a) - cos(b)) ** 2
    sine_squared = (sin(a) - sin(b)) ** 2
    assert phasor_squared * cos((a + b) / 2) ** 2 == pytest.approx(sine_squared)
    assert phasor_squared <= sine_squared / float(report.initial_cosine_lower) ** 2


def test_a_missing_initial_plateau_rejects_instead_of_reusing_the_bound(monkeypatch):
    cells = joint._historical_cells()
    monkeypatch.setattr(joint, "_historical_cells", lambda: cells[1:])
    with pytest.raises(ArithmeticError, match="fixed fifteen paired plateaus"):
        joint._bound_organized_law_joint_phases()
