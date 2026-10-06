"""Symbolic calculus for the unforced form row ``dEPI/dt = nu_f * DELTA_NFR``.

The helpers express its integral, product rule and solution with supplied
constant parameters. The exponential example supplies its own pressure law.
These calculations do not establish grammar U2, bifurcation, full-state
equilibrium or stability, and do not select operators or a complete model.
"""

import math

import sympy as sp
from sympy import Derivative, Eq, Function, Integral, integrate, simplify, symbols
from sympy.core.evalf import PrecisionExhausted

from .._exact_time import finite_represented_real, nonnegative_represented_time

# ============================================================================
# SYMBOLIC VARIABLES (TNFR canonical)
# ============================================================================

# Time variable
t = symbols("t", real=True, positive=True)

# Positive-capacity symbol for the calculations below.
nu_f = symbols("nu_f", real=True, positive=True)

# Nodal gradient (reorganization pressure) - can be positive or negative
DELTA_NFR = symbols("DELTA_NFR", real=True)

# EPI as function of time
EPI = Function("EPI")

# Coherence
C = Function("C")

# Phase
phi = symbols("phi", real=True)

# ============================================================================
# NODAL EQUATION
# ============================================================================


def get_nodal_equation() -> Eq:
    """
    Return the unforced scalar form row.

    ∂EPI/∂t = νf · ΔNFR

    Returns:
        Sympy equation with symbolic capacity and signed pressure.

    A zero product freezes this form row only. Other state rows and the
    pressure law must be specified separately; full equilibrium does not follow.
    """
    return Eq(Derivative(EPI(t), t), nu_f * DELTA_NFR)


def solve_nodal_equation_constant_params(
    nu_f_val: float, delta_nfr_val: float, EPI_0: float, t0: float = 0
) -> sp.Expr:
    """
    Solve nodal equation analytically for constant νf and ΔNFR.

    Solution: EPI(t) = EPI_0 + νf · ΔNFR · (t - t0)

    Args:
        nu_f_val: Held reorganization capacity
        delta_nfr_val: Held signed pressure
        EPI_0: Initial EPI value
        t0: Initial time

    Returns:
        Symbolic expression for EPI(t)

    This is the linear solution under the stated held-parameter premise.
    Inputs are symbolic substitutions, not execution-admission certificates.
    """
    eq = get_nodal_equation()
    # Substitute constant values
    eq_with_vals = eq.subs([(nu_f, nu_f_val), (DELTA_NFR, delta_nfr_val)])

    # Solve ODE
    solution = sp.dsolve(eq_with_vals, EPI(t))

    # Apply initial condition
    C1 = symbols("C1")
    solution_with_ic = solution.subs(C1, EPI_0 - nu_f_val * delta_nfr_val * t0)

    return solution_with_ic.rhs


# ============================================================================
# INTEGRATION UNDER SUPPLIED LAWS
# ============================================================================


def integrated_evolution_symbolic() -> sp.Integral:
    """
    Return the symbolic form increment integrated over a supplied interval.

    EPI(t_f) = EPI(t_0) + ∫[t_0 to t_f] νf(τ) · ΔNFR(τ) dτ

    Returns:
        Integral of capacity times pressure, without the initial form value.

    No capacity or pressure evolution law is selected here. Convergence,
    coherence and grammar admission require their own hypotheses.
    """
    tau = symbols("tau", real=True, positive=True)
    t_0, t_f = symbols("t_0 t_f", real=True, positive=True)

    nu_f_func = Function("nu_f")
    delta_nfr_func = Function("DELTA_NFR")

    integrand = nu_f_func(tau) * delta_nfr_func(tau)

    return Integral(integrand, (tau, t_0, t_f))


def check_convergence_exponential(
    growth_rate: float, time_horizon: float
) -> tuple[bool, str, float | None]:
    """
    Integrate a supplied exponential pressure with unit capacity and amplitude.

    The prescribed law is ``DELTA_NFR(t) = exp(growth_rate * t)``.

    Args:
        growth_rate: Finite represented λ (exponential rate).
        time_horizon: Finite nonnegative represented integration limit.

    Returns:
        ``(converges, explanation, integral_value)``. The Boolean concerns
        the improper integral over ``[0, infinity)`` under this supplied law.
        The value instead concerns the supplied finite horizon, or is ``None``
        when it cannot be materialized as a finite nonzero-preserving float.

    For real finite rates and a finite nonnegative horizon the mathematical
    integral is finite for every rate. Over an infinite horizon it converges
    only for a negative rate; zero rate gives linear form growth. This does not
    establish grammar U2, equilibrium or an operator-selection rule.
    """
    growth_rate, exact_rate = finite_represented_real(growth_rate, "growth_rate")
    _, exact_horizon = nonnegative_represented_time(time_horizon, "time_horizon")
    lambda_sym = symbols("lambda", real=True)
    tau = symbols("tau", real=True, positive=True)
    DELTA_NFR_0 = symbols("DELTA_NFR_0", real=True, positive=True)
    T = symbols("T", real=True, positive=True)

    # Exponential growth model
    delta_nfr_exp = DELTA_NFR_0 * sp.exp(lambda_sym * tau)

    # Assume constant νf for simplicity
    integrand = nu_f * delta_nfr_exp

    # Integrate
    integral = integrate(integrand, (tau, 0, T))
    integral_simplified = simplify(integral)

    # Substitute actual values
    integral_value = integral_simplified.subs(
        [
            (lambda_sym, sp.Rational(exact_rate.numerator, exact_rate.denominator)),
            (T, sp.Rational(exact_horizon.numerator, exact_horizon.denominator)),
            (nu_f, 1),  # Normalized
            (DELTA_NFR_0, 1),
        ]
    )

    converges = growth_rate < 0

    if growth_rate < 0:
        explanation = f"Convergent improper integral: λ={growth_rate} < 0"
    elif growth_rate == 0:
        explanation = (
            f"Divergent improper integral: λ={growth_rate} = 0 (linear form growth)"
        )
    else:
        explanation = f"Divergent improper integral: λ={growth_rate} > 0"

    try:
        # Small represented rates require extra precision for exp(lambda*T)-1.
        val = float(integral_value.evalf(17, maxn=1000, strict=True))
    except (OverflowError, TypeError, ValueError, PrecisionExhausted):
        val = None
    if val is not None and (
        not math.isfinite(val) or (val == 0.0 and exact_horizon != 0)
    ):
        val = None

    return converges, explanation, val


# ============================================================================
# PRODUCT RULE FOR THE UNFORCED FORM ROW
# ============================================================================


def compute_second_derivative_symbolic() -> sp.Expr:
    """
    Return the product-rule expression for the second form derivative.

    ∂²EPI/∂t² = ∂(νf · ΔNFR)/∂t = (∂νf/∂t)·ΔNFR + νf·(∂ΔNFR/∂t)

    Returns:
        Symbolic second derivative expression

    The identity assumes differentiable capacity and pressure in the same
    clock, with no additive source in the form row. Its magnitude alone does
    not establish stability, bifurcation or admission of a named operator.
    """
    # Second derivative (product rule)
    nu_f_func = Function("nu_f")
    delta_nfr_func = Function("DELTA_NFR")

    # ∂²EPI/∂t² = d/dt(νf · ΔNFR)
    second_deriv = Derivative(nu_f_func(t), t) * delta_nfr_func(t) + nu_f_func(
        t
    ) * Derivative(delta_nfr_func(t), t)

    return second_deriv


# ============================================================================
# UTILITIES
# ============================================================================


def latex_export(expr: sp.Expr) -> str:
    """
    Export symbolic expression to LaTeX format.

    Args:
        expr: Sympy expression

    Returns:
        LaTeX string for documentation/papers
    """
    return sp.latex(expr)


def pretty_print(expr: sp.Expr) -> str:
    """
    Pretty-print symbolic expression.

    Args:
        expr: Sympy expression

    Returns:
        Human-readable string representation
    """
    return sp.pretty(expr)
