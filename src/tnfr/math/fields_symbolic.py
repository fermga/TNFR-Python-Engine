"""
Symbolic Derivations for TNFR Structural Fields.

Provides symbolic representations of the structural fields that emerge from
TNFR dynamics, complementing the numerical implementations in
`tnfr.physics.fields`.

Key Fields:
- Φ_s (Structural Potential): Canonical field for U6 validation.
- |∇φ| (Phase Gradient): Research-phase field (EM-like analogy).
- K_φ (Phase Curvature): Research-phase field (strong-like analogy).

This module allows for formal analysis and derivation of field properties.

Physics basis: AGENTS.md § Structural Fields, U6: STRUCTURAL POTENTIAL
"""

from typing import Tuple
import sympy as sp
from sympy import symbols, Function, Sum, IndexedBase, Eq, Derivative
from .symbolic import latex_export, pretty_print


# ============================================================================
# SYMBOLIC VARIABLES
# ============================================================================

# Indices for nodes
i, j, n = symbols('i j n', integer=True)

# Nodal gradient and distance
DELTA_NFR = IndexedBase('DELTA_NFR')
d = Function('d')

# Phase
phi = IndexedBase('phi')
x, y = symbols('x y', real=True) # For spatial gradients


# ============================================================================
# U6: STRUCTURAL POTENTIAL (Φ_s) - CANONICAL
# ============================================================================

def get_structural_potential_field_symbolic() -> Tuple[Eq, str]:
    """
    Return the purely symbolic equation for the Structural Potential Field (Φ_s).

    Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α

    The equation is returned with a symbolic 'alpha'. The caller is responsible
    for substituting a concrete value for alpha if needed.
    e.g., `eq.subs(alpha_sym, 2.0)`

    Returns:
        (sympy_equation, physics_interpretation)

    Physics:
        - Canonical field for U6 grammar validation.
        - Φ_s minima represent passive equilibrium states (potential wells).
        - Displacement from minima (ΔΦ_s > 0) correlates with coherence loss.
        - The grammar (U1-U5) acts as a confinement mechanism.

    See: AGENTS.md § U6: STRUCTURAL POTENTIAL CONFINEMENT
    """
    alpha_sym = symbols('alpha', real=True, positive=True)
    Phi_s = Function('Phi_s')

    # Define the summation purely symbolically
    summation = Sum(
        DELTA_NFR[j] / (d(i, j)**alpha_sym),
        (j, 1, n)  # Sum over all nodes j from 1 to n
    )
    
    # Create the equation
    equation = Eq(Phi_s(i), summation)

    interpretation = (
        "Represents the potential at node 'i' as the sum of influences from "
        "all other nodes 'j'. Each node 'j' contributes its structural "
        "pressure (ΔNFR_j), weighted by the inverse of the distance d(i,j) "
        "to the power of a symbolic α."
    )
    
    return equation, interpretation


# ============================================================================
# RESEARCH-PHASE FIELDS
# ============================================================================

def get_phase_gradient_symbolic() -> Tuple[Eq, str]:
    """
    Return the symbolic equation for the Phase Gradient magnitude (|∇φ|).

    |∇φ| = sqrt((∂φ/∂x)² + (∂φ/∂y)²)

    Returns:
        (sympy_equation, physics_interpretation)
    """
    Nabla_phi = symbols('Nabla_phi')  # Use a symbol for the LHS
    phi_func = Function('phi')(x, y)

    # Partial derivatives
    dphi_dx = Derivative(phi_func, x)
    dphi_dy = Derivative(phi_func, y)

    # Magnitude of the gradient
    gradient_magnitude = sp.sqrt(dphi_dx**2 + dphi_dy**2)
    
    equation = Eq(Nabla_phi, gradient_magnitude)

    interpretation = (
        "Represents the magnitude of the phase gradient, which measures the "
        "rate of phase change across spatial dimensions (x, y). High values "
        "suggest rapid phase shifts, analogous to field strength in EM."
    )
    
    return equation, interpretation


def get_phase_curvature_symbolic() -> Tuple[Eq, str]:
    """
    Return the symbolic equation for the Phase Curvature (K_φ).

    K_φ = (∂²φ/∂x² + ∂²φ/∂y²) / (1 + (∂φ/∂x)² + (∂φ/∂y)²)^(3/2)

    Returns:
        (sympy_equation, physics_interpretation)

    Physics (Analogical):
        - Analogous to curvature of spacetime or field lines.
        - High |K_φ| suggests strong "bending" of the phase field, which
          may act as a confinement mechanism.
        - Very weakly correlated with coherence loss (corr ≈ -0.07).

    See: AGENTS.md § RESEARCH-PHASE Fields
    """
    phi_func = Function('phi')(x, y)
    
    # First derivatives
    dphi_dx = Derivative(phi_func, x)
    dphi_dy = Derivative(phi_func, y)
    
    # Second derivatives
    d2phi_dx2 = Derivative(dphi_dx, x)
    d2phi_dy2 = Derivative(dphi_dy, y)
    
    # Mean curvature formula for a surface z = φ(x,y)
    numerator = d2phi_dx2 * (1 + dphi_dy**2) - \
                2 * dphi_dx * dphi_dy * Derivative(dphi_dx, y) + \
                d2phi_dy2 * (1 + dphi_dx**2)
    
    denominator = (1 + dphi_dx**2 + dphi_dy**2)**(sp.S(3)/2)
    
    K_phi = symbols('K_phi')
    equation = Eq(K_phi, numerator / denominator)

    interpretation = (
        "Represents the mean curvature of the phase surface z=φ(x,y). High "
        "curvature indicates sharp 'bends' or 'folds' in the phase field, "
        "potentially acting as a local confinement force."
    )
    
    return equation, interpretation


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TNFR Symbolic Structural Fields")
    print("=" * 70)

    # 1. Structural Potential (Canonical)
    print("\n1. U6 Structural Potential (Φ_s) - CANONICAL")
    print("-" * 50)
    phi_s_eq, phi_s_interp = get_structural_potential_field_symbolic()
    print(pretty_print(phi_s_eq))
    print(f"\n   Physics: {phi_s_interp}")
    print(f"\n   LaTeX: {latex_export(phi_s_eq)}")

    # 2. Phase Gradient (Research)
    print("\n2. Phase Gradient (|∇φ|) - RESEARCH")
    print("-" * 50)
    grad_phi_eq, grad_phi_interp = get_phase_gradient_symbolic()
    print(pretty_print(grad_phi_eq))
    print(f"\n   Physics: {grad_phi_interp}")
    print(f"\n   LaTeX: {latex_export(grad_phi_eq)}")

    # 3. Phase Curvature (Research)
    print("\n3. Phase Curvature (K_φ) - RESEARCH")
    print("-" * 50)
    k_phi_eq, k_phi_interp = get_phase_curvature_symbolic()
    print(pretty_print(k_phi_eq))
    print(f"\n   Physics: {k_phi_interp}")
    print(f"\n   LaTeX: {latex_export(k_phi_eq)}")

    print("\n" + "=" * 70)
    print("✓ Symbolic fields module operational.")
    print("=" * 70)
