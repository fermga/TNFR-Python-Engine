"""Supplied-law calculus and public exports without bifurcation claims."""

import importlib
import math
from fractions import Fraction

import pytest

sp = pytest.importorskip("sympy")

from tnfr.math import symbolic


@pytest.mark.parametrize("rate", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize("horizon", [0.0, 3.0])
def test_improper_convergence_is_separate_from_finite_integral(rate, horizon):
    converges, explanation, value = symbolic.check_convergence_exponential(
        rate, horizon
    )
    expected = math.expm1(rate * horizon) / rate if rate else horizon
    assert value == pytest.approx(expected)
    assert converges is (rate < 0)
    assert "improper integral" in explanation
    assert math.isfinite(value)


def test_zero_rate_has_constant_pressure_and_unbounded_linear_form():
    solution = symbolic.solve_nodal_equation_constant_params(1, 1, 2)
    assert sp.diff(solution, symbolic.t) == 1
    assert sp.limit(solution, symbolic.t, sp.oo) == sp.oo
    converges, explanation, value = symbolic.check_convergence_exponential(0, 7)
    assert converges is False
    assert value == float(solution.subs(symbolic.t, 7) - 2) == 7.0
    assert "linear form growth" in explanation


@pytest.mark.parametrize(
    "capacity, pressure, initial_form, initial_time",
    [
        (sp.Rational(2, 3), sp.Rational(-7, 5), sp.Rational(11, 13), sp.Rational(3, 2)),
        (0, sp.Rational(7, 5), sp.Rational(-11, 13), sp.Rational(3, 2)),
        (sp.Rational(2, 3), 0, sp.Rational(11, 13), 0),
    ],
)
def test_constant_solution_satisfies_the_row_and_initial_condition(
    capacity, pressure, initial_form, initial_time
):
    solution = symbolic.solve_nodal_equation_constant_params(
        capacity, pressure, initial_form, initial_time
    )
    assert sp.simplify(sp.diff(solution, symbolic.t) - capacity * pressure) == 0
    assert sp.simplify(solution.subs(symbolic.t, initial_time) - initial_form) == 0


def test_constant_real_pressure_preserves_a_symbol_named_like_an_integration_constant():
    pressure = sp.re(sp.Symbol("C1"))
    solution = symbolic.solve_nodal_equation_constant_params(2, pressure, 3, 4)
    assert sp.simplify(sp.diff(solution, symbolic.t) - 2 * pressure) == 0
    assert sp.simplify(solution.subs(symbolic.t, 4) - 3) == 0


def test_constant_capacity_preserves_symbols_also_used_by_the_nodal_equation():
    capacity = symbolic.DELTA_NFR**2
    solution = symbolic.solve_nodal_equation_constant_params(capacity, 2, 3, 4)
    assert sp.simplify(sp.diff(solution, symbolic.t) - 2 * capacity) == 0
    assert sp.simplify(solution.subs(symbolic.t, 4) - 3) == 0


@pytest.mark.parametrize(
    "capacity, pressure, initial_form, initial_time",
    [
        (1e308, 2.0, 1.0, 0.0),
        (1e-200, 1e-200, 0.0, 1.0),
        (1e308, 2.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1e20),
    ],
    ids=[
        "binary64-product-overflow",
        "binary64-product-underflow",
        "large-rate-initial-form-retention",
        "large-origin-initial-form-retention",
    ],
)
def test_constant_solution_retains_float_rate_and_initial_condition(
    capacity, pressure, initial_form, initial_time
):
    solution = symbolic.solve_nodal_equation_constant_params(
        capacity, pressure, initial_form, initial_time
    )
    rate = sp.diff(solution, symbolic.t)
    assert rate.is_finite is True
    assert rate.is_positive is True
    # Compare exact represented inputs to the retained symbolic coefficient.
    # Float substitutions remain approximations, not exact decimal promises.
    exact_rate = sp.Rational(capacity) * sp.Rational(pressure)
    relative_error = abs(sp.Rational(rate) / exact_rate - 1)
    assert relative_error < sp.Rational(1, 10**14)
    assert sp.simplify(solution.subs(symbolic.t, initial_time) - initial_form) == 0


@pytest.mark.parametrize("rate", [-1e-250, 1e-250])
def test_small_nonzero_rate_does_not_lose_the_finite_integral(rate):
    converges, _, value = symbolic.check_convergence_exponential(rate, 1)
    assert converges is (rate < 0)
    assert value == pytest.approx(math.expm1(rate) / rate)


def test_finite_integral_float_overflow_is_explicitly_unavailable():
    converges, _, value = symbolic.check_convergence_exponential(1000, 1)
    assert converges is False
    assert value is None


@pytest.mark.parametrize("argument", ["growth_rate", "time_horizon"])
@pytest.mark.parametrize(
    "value, error",
    [
        (True, TypeError),
        ("1", TypeError),
        (1j, TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (Fraction(1, 10**1000), ValueError),
    ],
)
def test_supplied_exponential_inputs_use_shared_scalar_admission(
    argument, value, error
):
    arguments = {"growth_rate": -1, "time_horizon": 1}
    arguments[argument] = value
    with pytest.raises(error, match=argument):
        symbolic.check_convergence_exponential(**arguments)


def test_negative_horizon_is_rejected():
    with pytest.raises(ValueError, match="time_horizon.*nonnegative"):
        symbolic.check_convergence_exponential(-1, -1)


def test_product_rule_agrees_with_an_independently_integrated_form():
    t = symbolic.t
    capacity = t**2 + 2
    pressure = t - 3
    form = t**4 / 4 - t**3 + t**2 - 6 * t
    assert sp.expand(sp.diff(form, t) - capacity * pressure) == 0
    acceleration = (
        symbolic.compute_second_derivative_symbolic()
        .subs({sp.Function("nu_f")(t): capacity, sp.Function("DELTA_NFR")(t): pressure})
        .doit()
    )
    assert sp.expand(acceleration - sp.diff(form, t, 2)) == 0


@pytest.mark.parametrize("module_name", ["tnfr", "tnfr.math", "tnfr.mathematics"])
def test_public_symbolic_exports_keep_calculus_and_retire_risk_policy(module_name):
    module = importlib.import_module(module_name)
    for name in (
        "get_nodal_equation",
        "solve_nodal_equation_constant_params",
        "integrated_evolution_symbolic",
        "check_convergence_exponential",
        "compute_second_derivative_symbolic",
        "latex_export",
        "pretty_print",
    ):
        assert name in module.__all__
        assert getattr(module, name) is getattr(symbolic, name)
    assert "evaluate_bifurcation_risk" not in module.__all__
    with pytest.raises(AttributeError):
        getattr(module, "evaluate_bifurcation_risk")
    with pytest.raises(AttributeError):
        getattr(symbolic, "evaluate_bifurcation_risk")
    if module_name == "tnfr":
        assert "evaluate_bifurcation_risk" not in module.EXPORT_DEPENDENCIES
