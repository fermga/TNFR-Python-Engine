"""
Tests for TNFR Symbolic Structural Fields Module.

Validates:
- Symbolic representation of the Structural Potential Field (Φ_s).
- Symbolic representation of research-phase fields (|∇φ|, K_φ).
"""

import sympy as sp
from sympy import Sum, Eq, Derivative, symbols

from tnfr.math import fields_symbolic


class TestStructuralPotentialField:
    """Tests for the canonical U6 Structural Potential Field (Φ_s)."""

    def test_get_phi_s_symbolic_is_purely_symbolic(self):
        """Verify the symbolic form of Φ_s is purely symbolic."""
        eq, interp = fields_symbolic.get_structural_potential_field_symbolic()

        assert isinstance(eq, Eq)
        assert isinstance(interp, str)

        # Check LHS
        assert isinstance(eq.lhs, sp.Function)
        assert 'Phi_s' in str(eq.lhs)

        # Check RHS
        assert isinstance(eq.rhs, Sum)

        # Check the function inside the sum
        integrand = eq.rhs.function
        assert isinstance(integrand, sp.Mul)

        # Find the power argument
        pow_arg = next((arg for arg in integrand.args if isinstance(arg, sp.Pow)), None)
        assert pow_arg is not None, "Could not find Pow part in integrand"

        # The exponent should be the symbolic '-alpha'
        assert str(pow_arg.exp) == '-alpha'

    def test_alpha_parameter_substitution(self):
        """Verify the symbolic alpha can be correctly substituted."""
        eq, _ = fields_symbolic.get_structural_potential_field_symbolic()
        alpha_sym = symbols('alpha', real=True, positive=True)

        # Substitute alpha=2.0
        eq_2 = eq.subs(alpha_sym, 2.0)
        integrand_2 = eq_2.rhs.function
        pow_arg_2 = next(
            (arg for arg in integrand_2.args if isinstance(arg, sp.Pow)), None
        )
        assert pow_arg_2.exp == -2.0

        # Substitute alpha=1.0
        eq_1 = eq.subs(alpha_sym, 1.0)
        integrand_1 = eq_1.rhs.function
        pow_arg_1 = next(
            (arg for arg in integrand_1.args if isinstance(arg, sp.Pow)), None
        )
        assert pow_arg_1.exp == -1.0


class TestResearchPhaseFields:
    """Tests for the research-phase fields (|∇φ|, K_φ)."""

    def test_get_phase_gradient_symbolic(self):
        """Verify the symbolic form of |∇φ|."""
        eq, interp = fields_symbolic.get_phase_gradient_symbolic()
        assert isinstance(eq, Eq)
        assert isinstance(interp, str)
        assert 'Nabla_phi' in str(eq.lhs)
        assert isinstance(eq.rhs, sp.Pow)
        assert eq.rhs.exp == sp.Rational(1, 2)

        # Check content of the sqrt
        content = eq.rhs.args[0]
        assert isinstance(content, sp.Add)
        assert len(content.args) == 2  # Two squared derivative terms

        # Check for squared derivatives of phi
        expr_str = str(content)
        assert "Derivative(phi(x, y), x)**2" in expr_str
        assert "Derivative(phi(x, y), y)**2" in expr_str

    def test_get_phase_curvature_symbolic(self):
        """Verify the symbolic form of K_φ."""
        eq, interp = fields_symbolic.get_phase_curvature_symbolic()
        assert isinstance(eq, Eq)
        assert isinstance(interp, str)
        assert 'K_phi' in str(eq.lhs)

        # Check for second derivatives
        expr_str = str(eq.rhs)
        assert "Derivative(phi(x, y), (x, 2))" in expr_str  # d²φ/dx²
        assert "Derivative(phi(x, y), (y, 2))" in expr_str  # d²φ/dy²
