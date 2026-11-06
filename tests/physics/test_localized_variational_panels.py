"""Independent finite-panel algebra, never a canonical response evaluation."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from time import perf_counter

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I, sqrt
from tnfr.physics import _localized_variational_panels as panels


@pytest.fixture(scope="module", autouse=True)
def no_prepared_response():
    with forbid_organized_response_evaluation():
        yield


def _add(*rows):
    size = max(map(len, rows))
    return tuple(
        sum((row[k] if k < len(row) else Q(0) for row in rows), Q(0))
        for k in range(size)
    )


def _scale(row, factor):
    return tuple(factor * value for value in row)


def _multiply(left, right):
    result = [Q(0)] * (len(left) + len(right) - 1)
    for j, a in enumerate(left):
        for k, b in enumerate(right):
            result[j + k] += a * b / 2
            result[abs(j - k)] += a * b / 2
    return tuple(result)


def _derivative(row, duration):
    # dT_n/ds=n*U_(n-1); U_m=2(T_m+T_(m-2)+...)-1 for even m.
    result = [Q(0)] * max(1, len(row) - 1)
    for n, value in enumerate(row[1:], 1):
        for k in range(n - 1, -1, -2):
            result[k] += 2 * n * value * (1 if k == 0 else 2) / duration
    return tuple(result)


def _spatial_a(rows):
    return tuple(
        _add(
            row,
            _scale(rows[j - 1], Q(-1, 2)),
            _scale(rows[(j + 1) % len(rows)], Q(-1, 2)),
        )
        for j, row in enumerate(rows)
    )


def _manufactured_source(x, theta, edge, sigma, duration):
    ax, atheta = _spatial_a(x), _spatial_a(theta)
    gradients = tuple(
        _add(row, _scale(theta[(j + 1) % len(theta)], -1))
        for j, row in enumerate(theta)
    )
    flux = tuple(_multiply(c, g) for c, g in zip(edge, gradients))
    feedback = tuple(
        _scale(_add(row, _scale(flux[j - 1], -1)), Q(1, 2))
        for j, row in enumerate(flux)
    )
    rx = tuple(
        _add(_scale(u, -sigma), _scale(v, -1), f, _scale(_derivative(p, duration), -1))
        for u, v, f, p in zip(ax, atheta, feedback, x)
    )
    rt = tuple(_add(u, _scale(_derivative(p, duration), -1)) for u, p in zip(ax, theta))
    return rx, rt


def test_signed_polynomial_cancellation_and_affine_clock_are_exact():
    x = tuple((Q(j - 2, 8), Q(j % 2, 16), Q(1, 32)) for j in range(5))
    theta = tuple((Q(j * j, 32), Q(2 - j, 16)) for j in range(5))
    edge = tuple((Q(j + 1, 64), Q(-1, 128)) for j in range(5))
    sigma, duration = Q(1, 64), Q(8)
    rx, rt = _manufactured_source(x, theta, edge, sigma, duration)
    result = panels._bound_cycle_variational_panel(
        x, theta, rx, rt, edge, sigma=sigma, duration=duration
    )
    assert result.polynomial_defect_norm_upper == 0
    assert result.conditional_integrated_defect_upper == 0
    assert result.candidate_norm_upper > 0
    for supplied, left, right in (
        (x, result.initial_form, result.final_form),
        (theta, result.initial_phase, result.final_phase),
    ):
        for polynomial, initial, final in zip(supplied, left, right):
            assert initial.contains(
                sum((-1) ** k * value for k, value in enumerate(polynomial))
            )
            assert final.contains(sum(polynomial))
    # Changing only the physical panel duration changes the derivative row.
    wrong_clock = panels._bound_cycle_variational_panel(
        x, theta, rx, rt, edge, sigma=sigma, duration=duration / 2
    )
    assert wrong_clock.polynomial_defect_norm_upper > 0
    assert "caller must independently establish" in result.required_input_premise
    assert not hasattr(result, "certified")
    with pytest.raises(FrozenInstanceError):
        result.candidate_norm_upper = Q(0)


def test_source_and_generator_uncertainty_remain_separate_premises():
    x, theta, edge = ((Q(1),),) * 4, ((Q(0),),) * 4, ((Q(0),),) * 4
    result = panels._bound_cycle_variational_panel(
        x,
        theta,
        theta,
        theta,
        edge,
        sigma=0,
        duration=2,
        source_error=Q(1, 128),
        generator_error=Q(1, 256),
    )
    assert result.polynomial_defect_norm_upper == 0
    assert result.candidate_norm_upper == 2
    assert result.conditional_integrated_defect_upper == Q(1, 32)
    assert result.initial_form == result.final_form


def test_constant_mode_polynomial_detects_defect_between_zero_samples():
    # s^2*(s^2-1)^2 = (2-T2-2*T4+T6)/32 on the panel coordinate s.
    polynomial = (Q(2, 32), Q(0), Q(-1, 32), Q(0), Q(-2, 32), Q(0), Q(1, 32))
    # Check this counterexample through its independently known power coefficients.
    import sympy as sp

    s = sp.Symbol("s")
    expansion = sum(
        sp.Rational(value.numerator, value.denominator) * sp.chebyshevt(k, s)
        for k, value in enumerate(polynomial)
    )
    assert sp.expand(expansion - s**2 * (s**2 - 1) ** 2) == 0
    for value in (-1, 0, 1):
        assert expansion.subs(s, value) == sp.diff(expansion, s).subs(s, value) == 0
    zero = ((Q(0),),) * 4
    result = panels._bound_cycle_variational_panel(
        (polynomial,) * 4, zero, zero, zero, zero, sigma=Q(1, 64), duration=16
    )
    assert result.polynomial_defect_norm_upper > 0
    assert all(value.contains(0) for value in result.initial_form + result.final_form)


@pytest.mark.parametrize(
    "parameter,bad",
    [
        ("duration", True),
        ("duration", 0),
        ("duration", 17),
        ("duration", I(-1, 1)),
        ("sigma", False),
        ("sigma", -1),
        ("sigma", 2),
        ("source_error", -1),
        ("generator_error", float("nan")),
    ],
)
def test_invalid_domains_reject_before_polynomial_work(monkeypatch, parameter, bad):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached polynomial multiplication")

    monkeypatch.setattr(panels, "multiply", forbidden)
    monkeypatch.setattr(panels, "derivative", forbidden)
    zero = ((Q(0),),) * 3
    kwargs = {"duration": 1, "sigma": 0, parameter: bad}
    with pytest.raises((ValueError, TypeError)):
        panels._bound_cycle_variational_panel(zero, zero, zero, zero, zero, **kwargs)


@pytest.mark.parametrize(
    "bad",
    [((Q(0),),) * 2, {0: (0,), 1: (0,), 2: (0,)}, ((Q(0),) * 98,) * 3, ((True,),) * 3],
)
def test_node_and_coefficient_admission_rejects_invalid_rows(bad):
    zero = ((Q(0),),) * 3
    with pytest.raises((ValueError, TypeError)):
        panels._bound_cycle_variational_panel(
            bad, zero, zero, zero, zero, sigma=0, duration=1
        )


def test_interval_coefficients_include_arithmetic_and_retain_constant_mode():
    zero = ((Q(0),),) * 4
    phase = tuple((I(-1, 1), Q(1, 16)) for _ in range(4))
    result = panels._bound_cycle_variational_panel(
        zero, phase, zero, zero, zero, sigma=0, duration=2
    )
    assert result.candidate_norm_upper >= 2
    assert result.polynomial_defect_norm_upper >= Q(1, 8)
    assert all(value.contains(Q(1, 16)) for value in result.final_phase)


def test_scaled_defect_norm_retains_a_tiny_nonzero_source():
    zero = ((Q(0),),) * 3
    source = ((Q(1, 2**100),), zero[0], zero[0])
    result = panels._bound_cycle_variational_panel(
        zero, zero, source, zero, zero, sigma=0, duration=2
    )
    assert result.polynomial_defect_norm_upper == Q(1, 2**100)
    assert result.conditional_integrated_defect_upper == Q(1, 2**99)


@pytest.fixture(scope="module")
def full_size_countercontrol():
    degree, size, sigma, duration = 96, 1024, Q(1, 64), Q(16)
    x = tuple(Q((-1) ** k * (k + 1), 2**14) for k in range(degree + 1))
    theta = tuple(
        Q((-1) ** (k // 3) * (degree + 1 - k), 2**15) for k in range(degree + 1)
    )
    edge = tuple(Q((-1) ** k, 2**12) for k in range(degree + 1))
    product = _multiply(edge, theta)
    # At spatial wavenumber pi/2, A has eigenvalue one. Independently
    # construct a degree96 source, deliberately omitting the high product tail.
    form_rows, phase_rows, source_x, source_theta = [], [], [], []
    for a, b in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        form_rows.append(_scale(x, a))
        phase_rows.append(_scale(theta, b))
        rx = _add(
            _scale(x, -sigma * a),
            _scale(theta, -b),
            _scale(product, b),
            _scale(_derivative(x, duration), -a),
        )
        rt = _add(_scale(x, a), _scale(_derivative(theta, duration), -b))
        source_x.append(rx[: degree + 1])
        source_theta.append(rt)
    started = perf_counter()
    result = panels._bound_cycle_variational_panel(
        tuple(form_rows) * (size // 4),
        tuple(phase_rows) * (size // 4),
        tuple(source_x) * (size // 4),
        tuple(source_theta) * (size // 4),
        (edge,) * size,
        sigma=sigma,
        duration=duration,
    )
    return (
        result,
        sum(abs(value) for value in product[degree + 1 :]),
        perf_counter() - started,
    )


def test_full_size_validator_retains_generated_modes(
    full_size_countercontrol, record_property
):
    result, tail_sum, elapsed = full_size_countercontrol
    spatial_norm = sqrt(I(512))
    assert result.node_count == 1024
    assert result.candidate_degrees == (96, 96)
    assert result.residual_degree == 192
    assert tail_sum > 0
    assert (
        spatial_norm.lo * tail_sum
        <= result.polynomial_defect_norm_upper
        <= spatial_norm.hi * tail_sum
    )
    assert (
        result.conditional_integrated_defect_upper
        == 16 * result.polynomial_defect_norm_upper
    )
    record_property("full_size_panel_seconds", elapsed)
    record_property(
        "scope", "Synthetic polynomial tail countercontrol; no canonical response"
    )


def test_panel_input_tail_and_global_numerical_budget_are_distinct():
    assert Q(707, 500) ** 2 < 2
    assert Q(1207, 500) ** 96 > 2**122
    source, generator, candidate, residual, horizon = (
        Q(1, 2**109),
        Q(1, 2**114),
        Q(2**35),
        Q(1, 2**54),
        Q(2**19),
    )
    total = horizon * (residual + source + generator * candidate)
    assert total < Q(1, 2**34)
    assert Q(1, 2**71) + Q(1, 2**49) + 512 * total < Q(1, 2**24)
    assert 4 * 32 / Q(707, 500) / 2**122 < generator
    assert 4 * 1024 / Q(707, 500) / 2**122 < source
    assert Q(81, 2048) ** 2 + Q(1, 2**109) < Q(1, 625)
    assert Q(16, 625) < Q(1, 32)
    assert 2**14 * Q(1, 32) ** 21 / (1 - Q(1, 32)) < Q(1, 2**90)


def test_primitive_source_budget_sharpens_without_evaluating_the_response():
    # Substitute upper bounds into the positive expression (42), independently
    # of the finite source implementation or any variational correction.
    size = 1024
    b, zeta, dispersion = Q(81, 2 * size), Q(9, 2 * size), Q(49, 64)
    sigma, horizon = Q(16, 5 * size**2), Q(8 * size**2, 27)
    b1, b2, b3 = zeta * b / 4, zeta**2 * b / 2, 9 * zeta**3 * b / 8
    c0 = dispersion * b + b1
    gap = c0 + b**3 / 8
    s0 = b2 / 2 + b**3 / 72
    bound = (
        b3 / 6
        + 3 * b**2 * b1 / 8
        + b1 * c0**2 / 2
        + (3 * b**2 * s0 + 3 * b * b1**2 + Q(3, 2) * b**2 * b2) / 16
        + b**3 * gap**2 / 16
        + gap**5 / 120
        + 2 * sigma * (b + b**3 / 16)
    )
    assert bound < Q(1, 2**21)
    assert 32 * bound * horizon < 4
    assert 512 * (4 + horizon * Q(1, 2**71)) < 2**12
