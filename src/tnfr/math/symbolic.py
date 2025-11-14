"""
Symbolic Mathematics for TNFR.

Provides symbolic calculus tools for analyzing the nodal equation:
    ∂EPI/∂t = νf · ΔNFR

Key capabilities:
- Symbolic differentiation and integration
- Convergence analysis for U2 grammar rule
- Bifurcation threshold analysis (∂²EPI/∂t²)
- Analytical solutions for simple cases

Physics basis: AGENTS.md § Foundational Physics, TNFR.pdf § 2.1
"""

from typing import Optional, Tuple
import sympy as sp
from sympy import symbols, Function, Eq, Derivative, Integral, simplify
from sympy import integrate


# ============================================================================
# SYMBOLIC VARIABLES (TNFR canonical)
# ============================================================================

# Time variable
t = symbols('t', real=True, positive=True)

# Structural frequency (Hz_str) - always positive
nu_f = symbols('nu_f', real=True, positive=True)

# Nodal gradient (reorganization pressure) - can be positive or negative
DELTA_NFR = symbols('DELTA_NFR', real=True)

# EPI as function of time
EPI = Function('EPI')

# Coherence
C = Function('C')

# Phase
phi = symbols('phi', real=True)


# ============================================================================
# NODAL EQUATION
# ============================================================================

def get_nodal_equation() -> Eq:
    """
    Return the canonical TNFR nodal equation.
    
    ∂EPI/∂t = νf · ΔNFR
    
    Returns:
        Sympy equation representing the fundamental TNFR dynamics
        
    Physics:
        - Rate of structural change = Reorganization capacity ×
          Structural pressure
        - νf = 0: Node frozen, cannot reorganize
        - ΔNFR = 0: Equilibrium, no drive to change
        - Both > 0: Active reorganization
        
    See: AGENTS.md § The Nodal Equation
    """
    return Eq(Derivative(EPI(t), t), nu_f * DELTA_NFR)


def solve_nodal_equation_constant_params(
    nu_f_val: float,
    delta_nfr_val: float,
    EPI_0: float,
    t0: float = 0
) -> sp.Expr:
    """
    Solve nodal equation analytically for constant νf and ΔNFR.
    
    Solution: EPI(t) = EPI_0 + νf · ΔNFR · (t - t0)
    
    Args:
        nu_f_val: Structural frequency (Hz_str)
        delta_nfr_val: Reorganization gradient
        EPI_0: Initial EPI value
        t0: Initial time
        
    Returns:
        Symbolic expression for EPI(t)
        
    Physics:
        Linear evolution when both parameters constant.
        Real systems have time-varying νf and ΔNFR.
    """
    eq = get_nodal_equation()
    # Substitute constant values
    eq_with_vals = eq.subs([(nu_f, nu_f_val), (DELTA_NFR, delta_nfr_val)])
    
    # Solve ODE
    solution = sp.dsolve(eq_with_vals, EPI(t))
    
    # Apply initial condition
    C1 = symbols('C1')
    solution_with_ic = solution.subs(C1, EPI_0 - nu_f_val * delta_nfr_val * t0)
    
    return solution_with_ic.rhs


# ============================================================================
# INTEGRATION AND CONVERGENCE (U2 Grammar Rule)
# ============================================================================

def integrated_evolution_symbolic() -> sp.Integral:
    """
    Return symbolic form of integrated nodal equation.
    
    EPI(t_f) = EPI(t_0) + ∫[t_0 to t_f] νf(τ) · ΔNFR(τ) dτ
    
    Returns:
        Symbolic integral expression
        
    Physics:
        For bounded evolution (coherence preservation):
            ∫ νf·ΔNFR dt < ∞  (convergence requirement)
        
        Without stabilizers (IL, THOL):
            - ΔNFR grows unbounded (positive feedback)
            - Integral diverges → system fragments
            
    See: AGENTS.md § U2: CONVERGENCE & BOUNDEDNESS
    """
    tau = symbols('tau', real=True, positive=True)
    t_0, t_f = symbols('t_0 t_f', real=True, positive=True)
    
    nu_f_func = Function('nu_f')
    delta_nfr_func = Function('DELTA_NFR')
    
    integrand = nu_f_func(tau) * delta_nfr_func(tau)
    
    return Integral(integrand, (tau, t_0, t_f))


def check_convergence_exponential(
    growth_rate: float,
    time_horizon: float
) -> Tuple[bool, str, Optional[float]]:
    """
    Check convergence for exponential ΔNFR growth.
    
    Models destabilizers without stabilizers:
        ΔNFR(t) = ΔNFR_0 · e^(λt)
        
    Args:
        growth_rate: λ (exponential rate)
        time_horizon: Integration limit
        
    Returns:
        (converges, explanation, integral_value)
        
    Physics:
        - λ < 0: Decaying → converges (stabilized)
        - λ = 0: Constant → converges (equilibrium)
        - λ > 0: Growing → diverges (needs stabilizers!)
        
    Grammar: Validates U2 requirement for stabilizers
    """
    lambda_sym = symbols('lambda', real=True)
    tau = symbols('tau', real=True, positive=True)
    DELTA_NFR_0 = symbols('DELTA_NFR_0', real=True, positive=True)
    T = symbols('T', real=True, positive=True)
    
    # Exponential growth model
    delta_nfr_exp = DELTA_NFR_0 * sp.exp(lambda_sym * tau)
    
    # Assume constant νf for simplicity
    integrand = nu_f * delta_nfr_exp
    
    # Integrate
    integral = integrate(integrand, (tau, 0, T))
    integral_simplified = simplify(integral)
    
    # Substitute actual values
    integral_value = integral_simplified.subs([
        (lambda_sym, growth_rate),
        (T, time_horizon),
        (nu_f, 1.0),  # Normalized
        (DELTA_NFR_0, 1.0)
    ])
    
    converges = growth_rate <= 0
    
    if growth_rate < 0:
        explanation = f"Converges: λ={growth_rate} < 0 (decaying, stabilized)"
    elif growth_rate == 0:
        explanation = f"Converges: λ={growth_rate} = 0 (constant, equilibrium)"
    else:
        explanation = f"DIVERGES: λ={growth_rate} > 0 (growing, NEEDS STABILIZERS!)"
    
    try:
        val = float(integral_value)
    except:
        val = None
    
    return converges, explanation, val


# ============================================================================
# BIFURCATION ANALYSIS (U4 Grammar Rule)
# ============================================================================

def compute_second_derivative_symbolic() -> Derivative:
    """
    Return second derivative of EPI for bifurcation analysis.
    
    ∂²EPI/∂t² = ∂(νf · ΔNFR)/∂t = (∂νf/∂t)·ΔNFR + νf·(∂ΔNFR/∂t)
    
    Returns:
        Symbolic second derivative expression
        
    Physics:
        Bifurcation trigger: ∂²EPI/∂t² > τ (threshold)
        
        High second derivative indicates:
        - Rapid acceleration of reorganization
        - Potential phase transition
        - Need for handlers (THOL, IL) per U4a
        
    See: AGENTS.md § U4: BIFURCATION DYNAMICS
    """
    # First derivative (nodal equation)
    nu_f * DELTA_NFR
    
    # Second derivative (product rule)
    nu_f_func = Function('nu_f')
    delta_nfr_func = Function('DELTA_NFR')
    
    # ∂²EPI/∂t² = d/dt(νf · ΔNFR)
    second_deriv = (
        Derivative(nu_f_func(t), t) * delta_nfr_func(t) +
        nu_f_func(t) * Derivative(delta_nfr_func(t), t)
    )
    
    return second_deriv


def evaluate_bifurcation_risk(
    nu_f_val: float,
    delta_nfr_val: float,
    d_nu_f_dt: float,
    d_delta_nfr_dt: float,
    threshold: float = 1.0
) -> Tuple[bool, float, str]:
    """
    Evaluate if system is near bifurcation threshold.
    
    Args:
        nu_f_val: Current structural frequency
        delta_nfr_val: Current reorganization gradient
        d_nu_f_dt: Rate of change of νf
        d_delta_nfr_dt: Rate of change of ΔNFR
        threshold: Bifurcation threshold τ
        
    Returns:
        (at_risk, second_derivative_value, recommendation)
        
    Physics:
        ∂²EPI/∂t² = (∂νf/∂t)·ΔNFR + νf·(∂ΔNFR/∂t)
        
        If > τ: Apply handlers (THOL, IL) per U4a
        
    Grammar: Validates U4a requirement
    """
    # Compute second derivative
    second_deriv_val = d_nu_f_dt * delta_nfr_val + nu_f_val * d_delta_nfr_dt
    
    at_risk = abs(second_deriv_val) > threshold
    
    if at_risk:
        recommendation = (
            f"⚠️ BIFURCATION RISK: |∂²EPI/∂t²| = {abs(second_deriv_val):.4f} > τ = {threshold}\n"
            f"ACTION REQUIRED: Apply handlers {{THOL, IL}} per U4a grammar rule"
        )
    else:
        recommendation = (
            f"✓ Stable: |∂²EPI/∂t²| = {abs(second_deriv_val):.4f} ≤ τ = {threshold}\n"
            f"System within normal reorganization regime"
        )
    
    return at_risk, second_deriv_val, recommendation


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


# ============================================================================
# EXAMPLE USAGE AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TNFR Symbolic Mathematics Module")
    print("=" * 70)
    
    # 1. Display nodal equation
    print("\n1. CANONICAL NODAL EQUATION:")
    nodal_eq = get_nodal_equation()
    print(pretty_print(nodal_eq))
    print(f"LaTeX: {latex_export(nodal_eq)}")
    
    # 2. Solve for constant parameters
    print("\n2. ANALYTICAL SOLUTION (constant νf, ΔNFR):")
    solution = solve_nodal_equation_constant_params(
        nu_f_val=2.0,  # 2 Hz_str
        delta_nfr_val=0.5,
        EPI_0=1.0,
        t0=0
    )
    print(f"EPI(t) = {solution}")
    
    # 3. Convergence analysis
    print("\n3. CONVERGENCE ANALYSIS (U2 Grammar):")
    print("\nCase A: Stabilized (λ = -0.1)")
    conv_a, exp_a, val_a = check_convergence_exponential(-0.1, 10.0)
    print(f"  {exp_a}")
    print(f"  Integral value: {val_a:.4f}")
    
    print("\nCase B: Divergent (λ = +0.1) - NEEDS STABILIZERS!")
    conv_b, exp_b, val_b = check_convergence_exponential(0.1, 10.0)
    print(f"  {exp_b}")
    if val_b:
        print(f"  Integral value: {val_b:.4f}")
    
    # 4. Bifurcation analysis
    print("\n4. BIFURCATION ANALYSIS (U4 Grammar):")
    print("\nCase A: Normal operation")
    risk_a, deriv_a, rec_a = evaluate_bifurcation_risk(
        nu_f_val=1.0,
        delta_nfr_val=0.3,
        d_nu_f_dt=0.1,
        d_delta_nfr_dt=0.2,
        threshold=1.0
    )
    print(rec_a)
    
    print("\nCase B: High acceleration - bifurcation risk")
    risk_b, deriv_b, rec_b = evaluate_bifurcation_risk(
        nu_f_val=2.0,
        delta_nfr_val=1.5,
        d_nu_f_dt=0.5,
        d_delta_nfr_dt=1.0,
        threshold=1.0
    )
    print(rec_b)
    
    # 5. Second derivative formula
    print("\n5. SECOND DERIVATIVE (Bifurcation Indicator):")
    second_deriv = compute_second_derivative_symbolic()
    print(pretty_print(second_deriv))
    
    print("\n" + "=" * 70)
    print("✓ Symbolic module operational - Ready for TNFR analysis")
    print("=" * 70)
