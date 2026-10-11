"""Synthetic arithmetic and wiring controls; no declared kernel is evaluated."""

from fractions import Fraction as Q
import importlib.util
from pathlib import Path

import pytest

from tnfr.mathematics._rational_interval import I, cos, sin
from tnfr.physics._organized_law_contrast_kernels import (
    _MomentKernel,
    _across_mode_contrast,
    _enclose_moment,
    _ramp_monomial_moment,
    _series_tails,
    _three_ramp_contrast,
)


@pytest.mark.parametrize(
    "power,moment,expected",
    (
        (0, 0, Q(1, 2)),
        (1, 0, Q(1, 8)),
        (0, 1, Q(1, 64)),
        (1, 1, Q(1, 200)),
        (0, 2, Q(17, 28000)),
    ),
)
def test_polynomial_integrals_against_independent_antiderivatives(
    power, moment, expected
):
    assert _ramp_monomial_moment(power, moment) == expected


@pytest.mark.parametrize("bad", (True, False, 1.0, Q(1), "1", None))
def test_polynomial_admission_precedes_the_exact_cache(bad):
    _ramp_monomial_moment(1, 1)
    with pytest.raises(TypeError):
        _ramp_monomial_moment(bad, 1)
    with pytest.raises(TypeError):
        _ramp_monomial_moment(1, bad)


@pytest.mark.parametrize("bad", (-1, 40))
def test_polynomial_work_bound_rejects_outside_powers(bad):
    with pytest.raises(ValueError):
        _ramp_monomial_moment(bad, 1)


def test_fixed_tail_allowances_are_positive_and_below_declared_kernel_budget():
    assert all(0 < tail < Q(1, 10**28) for tail in _series_tails())


@pytest.mark.parametrize(
    "mode,mu,harmonic",
    (
        (True, I(0), 1),
        (0, I(0), 1),
        (1, I(-1), 1),
        (1, I(0, 2), 1),
        (1, I(0), True),
        (1, I(0), 3),
    ),
)
def test_moment_domain_rejection_evaluates_no_kernel(mode, mu, harmonic):
    with pytest.raises(ValueError):
        _enclose_moment(mode, mu, harmonic)


def _synthetic_moments(mode, multiplier=1):
    return {
        harmonic: _MomentKernel(
            mode,
            harmonic,
            I(0),
            I(multiplier * harmonic),
            I(multiplier * harmonic**2),
            Q(0),
            Q(0),
        )
        for harmonic in (1, 2, 9, 18, 36)
    }


def test_three_ramp_contrast_preserves_signed_denominators_and_double_noise_weight():
    moments = _synthetic_moments(2)
    result = _three_ramp_contrast(2, I(0), moments)
    assert result is not None
    assert result.alpha_even.hi < 0 < result.alpha_odd.lo
    s, c = sin(Q(1, 20)), cos(Q(1, 20))
    assert (result.alpha_even * (c - 4) - s).contains(0)
    assert (result.alpha_odd * (c + 2) - s).contains(0)
    for alpha, noise in (
        (result.alpha_even, result.even_noise_multiplier),
        (result.alpha_odd, result.odd_noise_multiplier),
    ):
        normalized = noise / (1 + 2 * abs(alpha))
        assert (normalized**2 - 10).contains(0)
    n = 9
    expected = sin(Q(n, 20)) * n - result.alpha_even * (
        cos(Q(n, 20)) * n**2 - (2 * n) ** 2
    )
    assert (result.even_coefficients[0] - expected).contains(0)


def test_unresolved_three_ramp_denominator_is_unavailable():
    moments = _synthetic_moments(2)
    moments[1] = _MomentKernel(2, 1, I(0), I(-1, 1), I(-1, 1), Q(0), Q(0))
    moments[2] = _MomentKernel(2, 2, I(0), I(-1, 1), I(-1, 1), Q(0), Q(0))
    assert _three_ramp_contrast(2, I(0), moments) is None


def test_across_modes_exactly_cancels_proportional_synthetic_moments():
    result = _across_mode_contrast(_synthetic_moments(1), _synthetic_moments(2, 3))
    assert result is not None
    assert result.alpha_even.contains(Q(1, 3))
    assert result.alpha_odd.contains(Q(1, 3))
    assert all(
        value.contains(0)
        for value in (*result.even_coefficients, *result.odd_coefficients)
    )


def test_benchmark_uses_exclusive_runner_without_evaluating_assessor(
    monkeypatch, tmp_path
):
    repository = Path(__file__).resolve().parents[2]
    path = repository / "benchmarks" / "organized_law_contrast_kernels.py"
    spec = importlib.util.spec_from_file_location("kernel_benchmark_wiring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorded = {}

    def runner(**arguments):
        recorded.update(arguments)
        return "synthetic_wiring_verdict"

    monkeypatch.setattr(module, "_run_single_analytic_assessment", runner)
    assert module.main(["--output", str(tmp_path / "evidence")]) == 0
    assert recorded["declaration_path"] == "theory/nodal/ORGANIZED_LAW_OBSERVATION.md"
    assert recorded["schema_name"] == "organized_law_contrast_kernels"
    assert recorded["assess"] is module._assess
    assert not (tmp_path / "evidence").exists()
